"""Parametric feature preprocessor for the BERT engine's stock
domain: turns an OHLCV panel into the standardised per-ticker
feature tensor the encoder consumes, with the feature menu
itself a config value (the spec).

Panel protocol: an object with .open / .high / .low / .close /
.volume float arrays (A, T), .tickers, and optionally
.fundamentals, a dict name -> (A, T) array of point-in-time
fundamental values. Point-in-time as-of alignment of the
fundamentals is the data layer's obligation; this module passes
the values through untouched.

Spec: a list of dicts, one per feature block, e.g.
    [{"kind": "fourier", "k": 4, "window": 60},
     {"kind": "rsi", "window": 14},
     {"kind": "fundamental", "name": "earnings_yield"}]
feature_width(spec) is the total width F; build_features(panel,
spec, stats_end=None) returns (feats float32 (A, T, F), stats).

Causality: every feature at day t is computed from bars at days
<= t only (the recursive Wilder / EMA features from bars
0..t). A feature is NaN at t if any input it needs is NaN;
insufficient history (t before the first full window) is NaN
too. NaNs survive until standardisation, where they become 0.

Standardisation: per feature, mean and std (ddof=0) over all
tickers and days [0, stats_end) using finite values only;
a feature with no finite values gets mean 0 / std 1, and a
constant feature (std 0, or at float-dust level <= 1e-12) gets
std 1.0. stats_end defaults to T // 2 when None -- that default
is a placeholder for smoke use only; real runs must set the
training cutoff explicitly (domain_config "stats_end"), or the
statistics leak future data into the features.

Legacy semantics: the fourier, volatility and wavelet blocks
reproduce genetic_lab/rerun_v2.py exactly (relative series
x_k = close_{t-k}/close_t - 1 over the trailing window, rfft,
first k real parts then first k imaginary parts; volatility =
std ddof=1 of the within-window daily returns; Haar db1
approximation coefficients with periodization wrap on odd
lengths). The remaining blocks are the standard textbook
definitions, pinned in each helper's docstring.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

_SQRT2 = np.sqrt(2.0)
_CHUNK = 16          # tickers per window-op chunk (memory bound)

_SINGLE = ("volatility", "rsi", "atr", "chaikin", "stochastic",
           "ema", "sma", "obv", "fundamental")


# ------------------------------------------------------------------ widths
def _wavelet_width(window, levels):
    n = int(window)
    for _ in range(int(levels)):
        if n % 2 == 1:
            n += 1
        n //= 2
    return n


def _block_width(block):
    kind = block.get("kind")
    if kind == "fourier":
        return 2 * int(block.get("k", 4))
    if kind == "wavelet":
        return _wavelet_width(block.get("window", 60),
                              block.get("levels", 3))
    if kind == "candlestick":
        return 6
    if kind in _SINGLE:
        return 1
    raise ValueError(f"unknown feature kind {kind!r}")


def feature_width(spec):
    """Total feature width F of a spec (validates kinds)."""
    return int(sum(_block_width(b) for b in spec))


# ------------------------------------------------------------------ helpers
def _arr(panel, name, kind):
    a = getattr(panel, name, None)
    if a is None:
        raise ValueError(
            f"panel lacks .{name}, required by feature kind "
            f"{kind!r}")
    return np.asarray(a, dtype=float)


def _windows(x, w):
    """Yield (a0, a1, win) with win (a, T-w+1, w) chronological
    sliding windows of x (a, T). Yields nothing if T < w."""
    A, T = x.shape
    if T < w:
        return
    for a0 in range(0, A, _CHUNK):
        a1 = min(A, a0 + _CHUNK)
        win = np.lib.stride_tricks.sliding_window_view(
            x[a0:a1], w, axis=1)
        yield a0, a1, win


def _roll(x, w):
    """Trailing-window sums and finite counts. Returns (s, c),
    each (A, T-w+1); element j covers days [j, j+w-1] (the
    window ending at t = j + w - 1). NaNs count as 0 in s and
    are excluded from c."""
    xf = np.where(np.isfinite(x), x, 0.0)
    xc = np.isfinite(x).astype(float)
    z = np.zeros((x.shape[0], 1))
    sf = np.concatenate([z, np.cumsum(xf, axis=1)], axis=1)
    sc = np.concatenate([z, np.cumsum(xc, axis=1)], axis=1)
    return sf[:, w:] - sf[:, :-w], sc[:, w:] - sc[:, :-w]


def _seed_smooth(x, alpha, seed, n0):
    """First-order recursion s[n] = alpha*x[n] + (1-alpha)*s[n-1]
    for n > n0 with s[n0] = seed, via lfilter plus the exact
    homogeneous correction. x must be finite (NaN-guarded by the
    caller); result is NaN before n0. seed: (A,)."""
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=1)
    s = np.full(x.shape, np.nan)
    expo = np.maximum(np.arange(x.shape[1]) - n0, 0)
    corr = (seed - y[:, n0])[:, None] \
        * ((1.0 - alpha) ** expo)[None, :]
    s[:, n0:] = (y + corr)[:, n0:]
    return s


def _all_finite_prefix(*arrs):
    """(A, T) bool: True at t iff every input is finite on the
    whole prefix [0, t] (the validity mask of the recursive
    features, whose state depends on all history)."""
    ok = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        ok &= np.isfinite(a)
    cnt = np.cumsum(ok, axis=1)
    return cnt == (np.arange(ok.shape[1]) + 1)[None, :]


# ------------------------------------------------------------------ blocks
def _fourier(close, k, w):
    """Legacy Fourier block: rfft of the relative series
    (newest first), first k real parts then first k imaginary
    parts. Width 2k."""
    A, T = close.shape
    out = np.full((A, T, 2 * k), np.nan)
    for a0, a1, win in _windows(close, w):
        rel = win[..., ::-1] / close[a0:a1, w - 1:][..., None] - 1.0
        z = np.fft.rfft(rel, axis=-1)
        out[a0:a1, w - 1:, :] = np.concatenate(
            [z[..., :k].real, z[..., :k].imag], axis=-1)
    return out


def _volatility(close, w):
    """Legacy volatility: std (ddof=1) of the w-1 within-window
    daily simple returns of the trailing window."""
    A, T = close.shape
    out = np.full((A, T, 1), np.nan)
    for a0, a1, win in _windows(close, w):
        leg = win[..., ::-1]                    # newest first
        daily = leg[..., :-1] / leg[..., 1:] - 1.0
        out[a0:a1, w - 1:, 0] = daily.std(axis=-1, ddof=1)
    return out


def _wavelet(close, w, levels):
    """Legacy Haar block: db1 approximation coefficients of the
    relative series, periodization wrap (append the first
    element) on odd lengths, `levels` deep."""
    A, T = close.shape
    out = np.full((A, T, _wavelet_width(w, levels)), np.nan)
    for a0, a1, win in _windows(close, w):
        a = win[..., ::-1] / close[a0:a1, w - 1:][..., None] - 1.0
        for _ in range(int(levels)):
            if a.shape[-1] % 2 == 1:
                a = np.concatenate([a, a[..., :1]], axis=-1)
            a = (a[..., 0::2] + a[..., 1::2]) / _SQRT2
        out[a0:a1, w - 1:, :] = a
    return out


def _rsi(close, w):
    """Wilder RSI, seeded by the simple mean of the first w
    gains / losses, smoothed with alpha 1/w. Output
    (RSI - 50) / 50 in [-1, 1]; a window with no losses reads
    RSI 100, no gains 0, neither (flat) 50."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    if T <= w:
        return out[:, :, None]
    cg = np.where(np.isfinite(close), close, 0.0)
    d = np.diff(cg, axis=1)
    g = np.maximum(d, 0.0)
    lo = np.maximum(-d, 0.0)
    sg = _seed_smooth(g, 1.0 / w, g[:, :w].mean(axis=1), w - 1)
    sl = _seed_smooth(lo, 1.0 / w, lo[:, :w].mean(axis=1), w - 1)
    ag, al = sg[:, w - 1:], sl[:, w - 1:]        # day t = index + w
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al > 0, ag / np.where(al > 0, al, 1.0),
                      np.inf)
    rsi = np.where(al == 0, np.where(ag > 0, 100.0, 50.0),
                   100.0 - 100.0 / (1.0 + rs))
    out[:, w:] = (rsi - 50.0) / 50.0
    out = np.where(_all_finite_prefix(close), out, np.nan)
    return out[:, :, None]


def _atr(high, low, close, w):
    """Wilder ATR (seeded by the simple mean of the first w
    true ranges, alpha 1/w smoothing), divided by close."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    if T <= w:
        return out[:, :, None]
    hg = np.where(np.isfinite(high), high, 0.0)
    lg = np.where(np.isfinite(low), low, 0.0)
    cg = np.where(np.isfinite(close), close, 0.0)
    pc = cg[:, :-1]
    tr = np.maximum(hg[:, 1:] - lg[:, 1:],
                    np.maximum(np.abs(hg[:, 1:] - pc),
                               np.abs(lg[:, 1:] - pc)))
    st = _seed_smooth(tr, 1.0 / w, tr[:, :w].mean(axis=1), w - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, w:] = st[:, w - 1:] / cg[:, w:]
    out = np.where(_all_finite_prefix(high, low, close), out,
                   np.nan)
    return out[:, :, None]


def _chaikin(high, low, close, volume, w):
    """Chaikin Money Flow over the trailing window:
    sum(MFV, w) / sum(V, w) with MF multiplier
    ((C-L)-(H-C))/(H-L), 0 when H == L, MFV = mult * V.
    Output in [-1, 1]; NaN when the volume sum is 0."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    rng = high - low
    safe = np.where(rng > 0, rng, 1.0)
    mult = np.where(rng > 0, (2.0 * close - high - low) / safe,
                    0.0)
    mfv = mult * volume
    s_mfv, _ = _roll(mfv, w)
    s_v, _ = _roll(volume, w)
    # joint validity: the where() above reads 0 where H - L is
    # NaN, so validity comes from the inputs' own counts
    joint = np.where(np.isfinite(high) & np.isfinite(low)
                     & np.isfinite(close) & np.isfinite(volume),
                     0.0, np.nan)
    _, cj = _roll(joint, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmf = np.where((cj == w) & (s_v > 0), s_mfv / s_v,
                       np.nan)
    out[:, w - 1:] = cmf
    return out[:, :, None]


def _stochastic(high, low, close, w, smooth):
    """%K = (C - LL) / (HH - LL) over the trailing window
    (0.5 when HH == LL), smoothed by a simple moving average
    of `smooth` days. Output (K - 0.5) * 2 in [-1, 1]."""
    A, T = close.shape
    K = np.full((A, T), np.nan)
    # joint validity, aligned j <-> t = j + w - 1 (a NaN high
    # or low makes the span NaN, which the where() below
    # would otherwise read as the flat-window case)
    joint = np.where(np.isfinite(high) & np.isfinite(low),
                     0.0, np.nan)
    _, cj = _roll(joint, w)
    for a0, a1, win_h in _windows(high, w):
        win_l = np.lib.stride_tricks.sliding_window_view(
            low[a0:a1], w, axis=1)
        hh = win_h.max(axis=-1)
        ll = win_l.min(axis=-1)
        ct = close[a0:a1, w - 1:]
        span = hh - ll
        safe = np.where(span > 0, span, 1.0)
        kvals = np.where(span > 0, (ct - ll) / safe, 0.5)
        K[a0:a1, w - 1:] = np.where(
            (cj[a0:a1] == w) & np.isfinite(ct), kvals, np.nan)
    # _roll over the full-length K: element j covers K-days
    # [j, j+smooth-1] and lands on day t = j + smooth - 1; the
    # finite count keeps it NaN until `smooth` defined %K
    # values exist (first at t = w - 1 + smooth - 1).
    s, c = _roll(K, int(smooth))
    ks = np.where(c == smooth, s / smooth, np.nan)
    out = np.full((A, T), np.nan)
    out[:, int(smooth) - 1:] = ks
    return ((out - 0.5) * 2.0)[:, :, None]


def _ema(close, w):
    """close / EMA - 1; EMA alpha = 2/(w+1), seeded by the SMA
    of the first w closes (EMA defined from day w-1)."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    if T < w:
        return out[:, :, None]
    cg = np.where(np.isfinite(close), close, 0.0)
    alpha = 2.0 / (w + 1)
    ema = _seed_smooth(cg, alpha, cg[:, :w].mean(axis=1), w - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, w - 1:] = cg[:, w - 1:] / ema[:, w - 1:] - 1.0
    out = np.where(_all_finite_prefix(close), out, np.nan)
    return out[:, :, None]


def _sma(close, w):
    """close / SMA - 1 over the trailing window."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    s, c = _roll(close, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(c == w, close[:, w - 1:] / (s / w) - 1.0,
                         np.nan)
    out[:, w - 1:] = ratio
    return out[:, :, None]


def _obv(close, volume, w):
    """(OBV_t - OBV_{t-w}) / sum(V, trailing w days), OBV the
    cumulative signed volume (sign of the close change).
    Computed on NaN-guarded series; a day is NaN when its
    trailing window (t-w, t] contains a NaN close or volume,
    or the volume sum is 0. Output in [-1, 1]."""
    A, T = close.shape
    out = np.full((A, T), np.nan)
    if T <= w:
        return out[:, :, None]
    cg = np.where(np.isfinite(close), close, 0.0)
    vg = np.where(np.isfinite(volume), volume, 0.0)
    sgn = np.zeros((A, T))
    sgn[:, 1:] = np.sign(np.diff(cg, axis=1))
    obv = np.cumsum(sgn * vg, axis=1)
    # element j of the outputs below is day t = j + w; the
    # numerator spans signed volume over (t-w, t] and the
    # denominator / counts span the same days [t-w+1, t].
    num = obv[:, w:] - obv[:, :-w]
    n_out = T - w

    def _span(x):
        cs = np.concatenate([np.zeros((A, 1)),
                             np.cumsum(x, axis=1)], axis=1)
        return cs[:, w + 1:] - cs[:, 1:1 + n_out]

    den = _span(vg)
    cnt = _span(np.isfinite(close).astype(float)) \
        + _span(np.isfinite(volume).astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, w:] = np.where((cnt == 2 * w) & (den > 0),
                              num / den, np.nan)
    return out[:, :, None]


def _candlestick(open_, high, low, close, w):
    """Six per-bar shape statistics over the trailing window:
    mean body ratio (C-O)/(H-L), mean upper-shadow ratio
    (H-max(C,O))/(H-L), mean lower-shadow ratio
    (min(C,O)-L)/(H-L), mean (H-L)/C, fraction of bars with
    C > O, and the last bar's body ratio. Ratios are 0 when
    H == L. The first five need the whole window finite; the
    last needs only day t."""
    A, T = close.shape
    rng = high - low
    safe = np.where(rng > 0, rng, 1.0)
    body = np.where(rng > 0, (close - open_) / safe, 0.0)
    upper = np.where(rng > 0,
                     (high - np.maximum(close, open_)) / safe, 0.0)
    lower = np.where(rng > 0,
                     (np.minimum(close, open_) - low) / safe, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_range = np.where(close > 0,
                             rng / np.where(close > 0, close, 1.0),
                             np.nan)
    up = (close > open_).astype(float)
    joint = np.where(np.isfinite(open_) & np.isfinite(high)
                     & np.isfinite(low) & np.isfinite(close),
                     0.0, np.nan)
    out = np.full((A, T, 6), np.nan)
    _, cj = _roll(joint, w)
    for j, series in enumerate((body, upper, lower, rel_range,
                                up)):
        s, _ = _roll(series, w)
        out[:, w - 1:, j] = np.where(cj == w, s / w, np.nan)
    out[:, :, 5] = np.where(np.isfinite(joint), body, np.nan)
    return out


def _fundamental(panel, name, shape):
    funds = getattr(panel, "fundamentals", None)
    if not funds or name not in funds:
        raise ValueError(
            f"fundamental {name!r} requested but the panel has "
            f"no .fundamentals entry of that name")
    a = np.asarray(funds[name], dtype=float)
    if a.shape != shape:
        raise ValueError(
            f"fundamental {name!r} has shape {a.shape}, "
            f"expected {shape}")
    return a[:, :, None]


# ------------------------------------------------------------------ build
def _compute_block(panel, block, shape):
    kind = block.get("kind")
    if kind == "fourier":
        return _fourier(_arr(panel, "close", kind),
                        int(block.get("k", 4)),
                        int(block.get("window", 60)))
    if kind == "volatility":
        return _volatility(_arr(panel, "close", kind),
                           int(block.get("window", 60)))
    if kind == "wavelet":
        return _wavelet(_arr(panel, "close", kind),
                        int(block.get("window", 60)),
                        int(block.get("levels", 3)))
    if kind == "rsi":
        return _rsi(_arr(panel, "close", kind),
                    int(block.get("window", 14)))
    if kind == "atr":
        return _atr(_arr(panel, "high", kind),
                    _arr(panel, "low", kind),
                    _arr(panel, "close", kind),
                    int(block.get("window", 14)))
    if kind == "chaikin":
        return _chaikin(_arr(panel, "high", kind),
                        _arr(panel, "low", kind),
                        _arr(panel, "close", kind),
                        _arr(panel, "volume", kind),
                        int(block.get("window", 20)))
    if kind == "stochastic":
        return _stochastic(_arr(panel, "high", kind),
                           _arr(panel, "low", kind),
                           _arr(panel, "close", kind),
                           int(block.get("window", 14)),
                           int(block.get("smooth", 3)))
    if kind == "ema":
        return _ema(_arr(panel, "close", kind),
                    int(block.get("window", 20)))
    if kind == "sma":
        return _sma(_arr(panel, "close", kind),
                    int(block.get("window", 50)))
    if kind == "obv":
        return _obv(_arr(panel, "close", kind),
                    _arr(panel, "volume", kind),
                    int(block.get("window", 20)))
    if kind == "candlestick":
        return _candlestick(_arr(panel, "open", kind),
                            _arr(panel, "high", kind),
                            _arr(panel, "low", kind),
                            _arr(panel, "close", kind),
                            int(block.get("window", 10)))
    if kind == "fundamental":
        name = block.get("name")
        if not name:
            raise ValueError(
                "fundamental feature block lacks a 'name'")
        return _fundamental(panel, name, shape)
    raise ValueError(f"unknown feature kind {kind!r}")


def raw_features(panel, spec):
    """The unstandardised feature tensor (A, T, F) float64,
    NaN where a feature is undefined. build_features is this
    plus standardisation; exposed for tests and audits."""
    close = _arr(panel, "close", "raw_features")
    shape = close.shape
    blocks = [_compute_block(panel, b, shape) for b in spec]
    if not blocks:
        return np.zeros((shape[0], shape[1], 0))
    return np.concatenate(blocks, axis=2)


def build_features(panel, spec, stats_end=None):
    """Standardised features: (feats float32 (A, T, F), stats).

    stats = {"mean", "std" (F,), "stats_end", "width",
    "blocks": [(kind, start, stop), ...]}. Statistics use
    finite values on days [0, stats_end) across all tickers;
    stats_end defaults to T // 2 (placeholder, see module
    docstring). NaN features become 0 after standardisation.
    """
    raw = raw_features(panel, spec)
    A, T, F = raw.shape
    if stats_end is None:
        stats_end = T // 2
    stats_end = int(stats_end)
    seg = raw[:, :stats_end, :]
    finite = np.isfinite(seg)
    cnt = finite.sum(axis=(0, 1)).astype(float)
    sums = np.where(finite, seg, 0.0).sum(axis=(0, 1))
    mean = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), 0.0)
    dev = np.where(finite, (seg - mean) ** 2, 0.0).sum(axis=(0, 1))
    std = np.sqrt(np.where(cnt > 0, dev / np.maximum(cnt, 1.0),
                           0.0))
    # std at or below 1e-12 is float dust for these
    # dimensionless features (a recursion on a constant series
    # leaves ~1e-16 residue whose z-scores would otherwise
    # blow up to O(1)); treat it as a constant feature.
    std = np.where(std > 1e-12, std, 1.0)
    z = (raw - mean) / std
    z = np.where(np.isfinite(z), z, 0.0)
    blocks = []
    pos = 0
    for b in spec:
        w = _block_width(b)
        blocks.append((b.get("kind"), pos, pos + w))
        pos += w
    stats = {"mean": mean, "std": std, "stats_end": stats_end,
             "width": F, "blocks": blocks}
    return z.astype(np.float32), stats
