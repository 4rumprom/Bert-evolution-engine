"""Live-run monitor for the parametric BERT evolution engine
(stock domain only) — plan 2026-11-22, Addendum 3.

The runner keeps this page fresh as it goes; JAX only computes,
the page reads files the runner writes atomically (the same
discipline as the checkpoint), so watching never touches the
run. Files, in the run's output directory:

  - ``monitor_status.json``, rewritten every generation
    (everything cheap);
  - ``monitor_snapshots.jsonl``, one appended record per
    detailed snapshot;
  - ``monitor.html``, a self-contained page regenerated at
    every snapshot, no server needed; it is the thing Roman
    opens.

This module never imports the runner (``run_bert_engine``) —
the runner calls these functions; the import would be
circular. It also never imports the archive module: the
population density reading is passed in by the runner as a
plain dict/float (see ``build_snapshot``).

Conventions the module follows (all from the plan / the
trading program's own code, not invented here):

  - Sharpe: daily returns from the recorded value path,
    zero risk-free rate, annualised by sqrt(252). Labelled
    as a 61-mark snapshot Sharpe on the page: real, but noisy
    at 60 days. The recorded path is [V0] + end-of-day values
    + the final mark value = days+2 points, days+1 returns
    (the plan's "61-mark" shorthand counts V0 + 60
    decision-day marks; the build records one extra point,
    the final mark, so the mark-day move is inside Sharpe
    and drawdown, and the page states the counts).
  - SQN: the trading program's own convention
    (``trading_agent/metrics.py`` ``compute_sqn``), adapted
    to the engine's detailed trace. A trade is a completed
    per-name holding episode inside a validation sample (a
    maximal run of days with held weight above zero, closed
    before the sample ends; positions still open at the
    sample end are excluded). R is the name's compounded
    price return over the held days divided by twice its
    ATR(14) percent at entry. SQN pools the completed trades
    of ALL validation episodes. Buy-and-hold SQN is normally
    null on 60-day samples because its positions never close
    inside the sample; this module returns null there, never
    a fake zero.
  - Random ladder: the ``alloc_v3_env.random_baseline``
    convention — iid uniform scores 0..5, random execution
    order, unit calibrated to the champion's mean invested
    share — executed through the domain's own
    ``_execute_day`` (the same cash/cost rules as the
    fitness path), never a re-implementation.
"""

from __future__ import annotations

import html as _html
import json
import math
import os
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_domain_stock as stock

STATUS_FILE = "monitor_status.json"
SNAPSHOTS_FILE = "monitor_snapshots.jsonl"
HTML_FILE = "monitor.html"
LADDER_FILE = "monitor_ladder.json"

ATR_WINDOW = 14
TRADING_DAYS = 252
LADDER_N_DRAWS = 200  # plan's proposed draw count for the
# exposure-matched random ladder (recomputed at stage end
# and whenever the architecture changes)


# ================================================================
# JSON helpers
# ================================================================

def _jsonable(x):
    """Recursively convert numpy scalars/arrays to plain
    JSON-able Python values (floats/ints/lists/None)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, float)):
        return float(x) if isinstance(x, float) else x
    if isinstance(x, np.floating):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _f(x):
    """float(x) or None when x is None / non-finite."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _median_or_none(values):
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, 0
    return float(np.median(vals)), len(vals)


# ================================================================
# 1. Per-episode statistics from a detailed simulation
# ================================================================

def sharpe_from_path(value_path):
    """(sharpe_or_None, n_returns): daily returns from
    consecutive path values, zero risk-free rate,
    mean / sd(ddof=1) * sqrt(252). None when fewer than 2
    returns or sd is 0 / non-finite."""
    p = np.asarray(value_path, dtype=float)
    if p.size < 3:  # need at least 2 returns
        return None, int(max(p.size - 1, 0))
    rets = p[1:] / p[:-1] - 1.0
    n = int(len(rets))
    if n < 2 or not np.all(np.isfinite(rets)):
        return None, n
    sd = float(np.std(rets, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return None, n
    return float(np.mean(rets) / sd * math.sqrt(TRADING_DAYS)), n


def max_drawdown_from_path(value_path):
    """Worst peak-to-trough return of the path (negative,
    <= 0). 0.0 for a path with fewer than 2 points."""
    p = np.asarray(value_path, dtype=float)
    if p.size < 2:
        return 0.0
    peak = np.maximum.accumulate(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, p / peak - 1.0, 0.0)
    if not np.all(np.isfinite(dd)):
        dd = dd[np.isfinite(dd)]
        if dd.size == 0:
            return 0.0
    return float(np.min(dd))


# ------------------------------------------------ ATR(14) percent
# Replicates trading_agent/metrics.py _atr_pct in numpy (no
# pandas dependency in this module), over the panel's full
# history from row 0 up to the episode end, so the value at
# an entry day uses the same data compute_sqn would see on
# a price panel spanning the same rows.

def _atr_pct_series(prices, highs, lows):
    """ATR(14)% per name, rows 0..T-1 of the given arrays
    (prices/highs/lows: (N, T), prices in the same row
    order). Returns (N, T) of ATR percent (fraction, not
    percent points). With highs/lows: Wilder-style EMA
    (alpha=1/14, adjust=False) of true range, divided by
    close. Without them: the documented close-based proxy —
    trailing 14-day mean of absolute daily percent changes
    (partial windows at the sample start use the
    observations available)."""
    prices = np.asarray(prices, dtype=float)
    n, t = prices.shape
    if highs is not None and lows is not None:
        h = np.asarray(highs, dtype=float)
        lo = np.asarray(lows, dtype=float)
        prev_close = np.empty_like(prices)
        prev_close[:, 0] = np.nan
        prev_close[:, 1:] = prices[:, :-1]
        with np.errstate(invalid="ignore"):
            legs = np.stack([h - lo,
                             np.abs(h - prev_close),
                             np.abs(lo - prev_close)], axis=0)
        # nanmax over the three legs (first day falls back
        # to high - low, exactly as _atr_pct)
        with np.errstate(invalid="ignore"):
            tr = np.nanmax(legs, axis=0)
        # Wilder EMA, alpha=1/14, adjust=False: first valid
        # value seeds; NaN observations carry the last value.
        atr = np.full_like(tr, np.nan)
        alpha = 1.0 / ATR_WINDOW
        for i in range(n):
            cur = np.nan
            for j in range(t):
                v = tr[i, j]
                if math.isfinite(v):
                    cur = v if math.isnan(cur) \
                        else alpha * v + (1.0 - alpha) * cur
                atr[i, j] = cur
        with np.errstate(invalid="ignore", divide="ignore"):
            return atr / prices
    # close-based proxy
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.abs(prices[:, 1:] / prices[:, :-1] - 1.0)
    abs_pct = np.full_like(prices, np.nan)
    abs_pct[:, 1:] = pct
    out = np.full_like(prices, np.nan)
    for i in range(n):
        # trailing 14-row mean of the available (non-NaN)
        # values, min_periods=1
        s = 0.0
        c = 0
        for j in range(t):
            # value entering the window at row j
            if math.isfinite(abs_pct[i, j]):
                s += abs_pct[i, j]
                c += 1
            if j - ATR_WINDOW >= 0:
                old = abs_pct[i, j - ATR_WINDOW]
                if math.isfinite(old):
                    s -= old
                    c -= 1
            out[i, j] = s / c if c > 0 else np.nan
    return out


def _r_multiples(shares_end, prices, v_pre, v_final, basket,
                 start, panel):
    """Completed per-name holding episodes -> list of
    R-multiples, following compute_sqn exactly as adapted
    to the engine's detailed trace.

    shares_end: (days, N) end-of-day shares; prices:
    (N, days+1) decision-day prices plus the mark day;
    v_pre: (days,) pre-trade portfolio values; v_final:
    mark-day portfolio value. Held weight on day t
    (t = 1..days, the compute_sqn shift(1) convention) =
    shares_end[t-1][i] * prices[t][i] / V[t], where V[t] is
    v_pre[t] for t < days and v_final for t = days. A run
    reaching the last day (the mark day) is still open at
    the sample end and is EXCLUDED. Episode return =
    prices[end][i] / prices[start-1][i] - 1. R = episode
    return / (2 * ATR(14)% at entry), entry reference =
    the day before the first held day, falling back to the
    first held day's value when the entry value is missing.
    Episodes whose risk fraction is undefined or
    non-positive are excluded from R and from N."""
    shares_end = np.asarray(shares_end, dtype=float)
    prices = np.asarray(prices, dtype=float)
    v_pre = np.asarray(v_pre, dtype=float)
    days, n = shares_end.shape
    # V[t] for held days t = 1..days
    v = np.empty(days + 1, dtype=float)  # index by held day t
    v[1:days] = v_pre[1:days]
    v[days] = float(v_final)
    # held weight per (held day t, name): shares_end[t-1] *
    # prices[t] / V[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        held = shares_end * prices[:, 1:].T / v[1:, None]
    held = np.where(np.isfinite(held), held, 0.0)
    # ATR(14)% over the panel history up to the episode end
    # (rows 0..start+days); entry refs are <= start+days-1.
    end_row = start + days + 1  # exclusive
    panel_prices = np.asarray(
        getattr(panel, "prices", None)
        if getattr(panel, "prices", None) is not None
        else panel.close, dtype=float)
    highs = getattr(panel, "high", None)
    lows = getattr(panel, "low", None)
    basket = np.asarray(basket)
    atr = _atr_pct_series(
        panel_prices[basket][:, :end_row],
        None if highs is None else
        np.asarray(highs, dtype=float)[basket][:, :end_row],
        None if lows is None else
        np.asarray(lows, dtype=float)[basket][:, :end_row])
    r_multiples = []
    for i in range(n):
        is_held = held[:, i] > 0
        pos = 0
        while pos < days:
            if not is_held[pos]:
                pos += 1
                continue
            run_start = pos  # 0-based index into held days;
            # held day t = pos + 1
            while pos < days and is_held[pos]:
                pos += 1
            run_end = pos - 1  # last held day, 0-based
            if run_end == days - 1:
                continue  # still open at sample end: excluded
            # held days t = run_start+1 .. run_end+1;
            # episode return = prices[end]/prices[start-1]-1
            # with prices indexed by held day t
            ep_ret = float(
                prices[i, run_end + 1] / prices[i, run_start]
                - 1.0)
            # entry reference: the day before the first held
            # day, in panel rows: start + run_start
            entry_row = start + run_start
            atr_val = atr[i, entry_row]
            if not math.isfinite(atr_val):
                # fall back to the first held day's value
                atr_val = atr[i, entry_row + 1]
            risk = 2.0 * float(atr_val) \
                if atr_val is not None and math.isfinite(atr_val) \
                else float("nan")
            if not math.isfinite(risk) or risk <= 0 \
                    or not math.isfinite(ep_ret):
                continue
            r_multiples.append(ep_ret / risk)
    return r_multiples


def _sqn_from_r(r_multiples):
    """SQN dict from a list of R-multiples, exactly the
    compute_sqn tail: SQN = sqrt(min(N, 100)) * mean(R) /
    sd(R, ddof=1); None with a reason when N < 2 or sd is
    0/undefined; small_sample True when N < 30."""
    n = len(r_multiples)
    if n == 0:
        return {"sqn": None, "n_trades": 0, "mean_r": None,
                "sd_r": None, "small_sample": True,
                "reason": "no completed trades: N=0 "
                          "(< 2 required for SQN)"}
    mean_r = float(np.mean(r_multiples))
    if n < 2:
        return {"sqn": None, "n_trades": n, "mean_r": mean_r,
                "sd_r": None, "small_sample": n < 30,
                "reason": f"insufficient completed trades: "
                          f"N={n} (< 2 required for SQN)"}
    sd_r = float(np.std(r_multiples, ddof=1))
    if not math.isfinite(sd_r) or sd_r == 0.0:
        return {"sqn": None, "n_trades": n, "mean_r": mean_r,
                "sd_r": sd_r if math.isfinite(sd_r) else None,
                "small_sample": n < 30,
                "reason": "sd(R) is zero or undefined; "
                          "SQN undefined"}
    return {"sqn": float(np.sqrt(min(n, 100)) * mean_r / sd_r),
            "n_trades": n, "mean_r": mean_r, "sd_r": sd_r,
            "small_sample": n < 30, "reason": None}


def sqn_from_trace(det, episode, panel):
    """SQN dict for one detailed episode trace (see
    ``_r_multiples``). Keys: sqn, n_trades, mean_r, sd_r,
    small_sample, reason."""
    r = _r_multiples(det["shares_end"], det["prices"],
                     det["v_pre"], det["v_final"],
                     episode["basket"], episode["start"], panel)
    return _sqn_from_r(r)


def episode_stats(det, episode, panel):
    """Per-episode dict for one genome from its detailed
    simulation: cagr (recomputed from the value path's
    final value, as a cross-check against the domain's
    CAGR), sharpe, max_drawdown, sqn (the dict above),
    n_buys, n_sells, turnover, mean_invested_pct,
    n_completed_trades."""
    path = np.asarray(det["value_path"], dtype=float)
    days = int(len(path) - 2)
    cagr = float((path[-1] / stock.V0)
                 ** (TRADING_DAYS / (days + 1)) - 1.0) \
        if path[-1] > 0 else -1.0
    sharpe, n_rets = sharpe_from_path(path)
    sqn = sqn_from_trace(det, episode, panel)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sharpe_n_returns": n_rets,
        "max_drawdown": max_drawdown_from_path(path),
        "sqn": sqn,
        "n_buys": int(det["n_buys"]),
        "n_sells": int(det["n_sells"]),
        "turnover": float(det["turnover"]),
        "mean_invested_pct": float(
            np.mean(det["invested_pct"])),
        "n_completed_trades": int(sqn["n_trades"]),
    }


def buyhold_stats(episode, panel):
    """The same shape as ``episode_stats`` for uniform
    buy-and-hold of the episode's basket: cagr (the
    domain's buy-and-hold leg: V0 * mean(p_end/p0) /
    (1+COST_RATE), annualised 252/(days+1)), sharpe and
    max_drawdown from the B&H daily value path
    V[t] = V0 * mean(p[t]/p[start]) / (1+COST_RATE) for
    t = start..start+days, sqn from the B&H trace
    (normally null — its positions never close inside the
    sample; never a fake zero), turnover 0 (no fills after
    the initial purchase)."""
    basket = np.asarray(episode["basket"])
    start = int(episode["start"])
    prices = np.asarray(panel.prices, dtype=float)
    # episode length: the domain's episode_days when the
    # panel's owner is a StockDomain; otherwise infer from
    # the panel shape is impossible, so the caller passes
    # it via episode["days"] when known. Default 60 (the
    # env's EPISODE_DAYS).
    days = int(episode.get("days", 60))
    p = prices[basket][:, start:start + days + 1]  # (N, days+1)
    p0 = p[:, 0]
    mean_ratio = np.mean(p / p0[:, None], axis=0)  # (days+1,)
    path = stock.V0 * mean_ratio / (1.0 + stock.COST_RATE)
    cagr = float((path[-1] / stock.V0)
                 ** (TRADING_DAYS / (days + 1)) - 1.0) \
        if path[-1] > 0 else -1.0
    sharpe, n_rets = sharpe_from_path(path)
    # B&H trace: constant shares from the equal-dollar
    # purchase at the start day's prices, charged the 6bp
    # cost as V0/(1+COST_RATE) invested.
    n = len(basket)
    shares = (stock.V0 / n) / (1.0 + stock.COST_RATE) / p0
    shares_end = np.tile(shares, (days, 1))
    # v_pre[t] for t = 0..days-1 is the path value at
    # start+t; v_final is the path value at start+days.
    # (The B&H path has days+1 points, t = start..start+days.)
    r = _r_multiples(shares_end, p, path[:-1], float(path[-1]),
                     basket, start, panel)
    sqn = _sqn_from_r(r)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sharpe_n_returns": n_rets,
        "max_drawdown": max_drawdown_from_path(path),
        "sqn": sqn,
        "n_buys": 0,
        "n_sells": 0,
        "turnover": 0.0,
        "mean_invested_pct": float(np.mean(
            shares @ p / path * 100.0)),
        "n_completed_trades": int(sqn["n_trades"]),
    }


# ================================================================
# 2. Snapshot builder (stock domain only)
# ================================================================

def _structure(layout):
    arch = layout.arch
    stacks = []
    for c, (n_blocks, heads) in enumerate(arch.couples):
        stacks.append({"stack": int(c), "blocks": int(n_blocks),
                       "heads": int(heads), "d": int(arch.d),
                       "ffn_hidden": int(arch.ffn_hidden)})
    return {"stacks": stacks, "total_genes": int(layout.n_params),
            "arch_signature": arch.signature(),
            "d": int(arch.d),
            "ffn_hidden": int(arch.ffn_hidden),
            "n_stacks": len(stacks)}


def _density_share(density):
    """The 'useful' share out of a density reading: a float
    is already the share; a dict is read from its known
    keys (archive ``density_shares`` uses "useful"; the
    silencing table's masking density is recorded under
    "real" / "masking_density"). Returns None when the
    reading carries no usable share."""
    if density is None:
        return None
    if isinstance(density, (int, float, np.floating)):
        return _f(density)
    if isinstance(density, dict):
        for k in ("useful", "useful_share", "share_useful",
                  "real", "real_share", "masking_density",
                  "density"):
            if k in density:
                return _f(density[k])
        return None
    return None


def _n_tested_units(layout):
    """Number of silencing-testable units in this layout
    (feed-forward hidden units + whole heads, per block)."""
    try:
        return len(be.silence_units(layout))
    except Exception:
        arch = layout.arch
        return sum(b * arch.ffn_hidden + b * h
                   for (b, h) in arch.couples)


def build_snapshot(domain, layout, genome, *, total_gens, stage,
                   phase, gen_in_phase, label, genome_role,
                   density_population=None, density_masking=None,
                   ladder=None, arch_signature=None):
    """A JSON-able detailed snapshot of one genome on the
    domain's frozen validation episodes (``detailed=True``,
    one genome). Stock domain ONLY: raises ValueError with
    a clear message for any other domain — the runner only
    wires the monitor for stock.

    The function does NOT import the runner or the archive
    module. ``ladder`` is a dict {"medians": [...], "unit":
    float, "arch_signature": str, "n_draws": int} or None.
    ``density_population`` / ``density_masking`` are
    dicts/floats supplied by the runner (from the archive
    tracker and the silencing table) or None. ``label`` is
    "snapshot" | "phase_end" | "stage_end"; ``genome_role``
    is "leader" | "stage_champion"."""
    if getattr(domain, "name", None) != "stock":
        raise ValueError(
            "bert_monitor.build_snapshot is stock-domain only "
            f"(got domain name "
            f"{getattr(domain, 'name', None)!r}); the live-run "
            "monitor is not wired for other domains")
    G = np.asarray(genome, dtype=float)[None, :]
    panel = domain.panel
    episodes = domain.val_episodes()
    days = int(domain.episode_days)
    champ_stats, bh_stats = [], []
    pooled_r, pooled_bh_r = [], []
    mean_invested = []
    for ep in episodes:
        ep = dict(ep)
        ep["days"] = days
        _c, _bh, det = domain.episode_cagrs(
            layout, G, ep, detailed=True)
        st = episode_stats(det, ep, panel)
        champ_stats.append(st)
        mean_invested.append(st["mean_invested_pct"])
        pooled_r.extend(_r_multiples(
            det["shares_end"], det["prices"], det["v_pre"],
            det["v_final"], ep["basket"], ep["start"], panel))
        bh = buyhold_stats(ep, panel)
        bh_stats.append(bh)
        # B&H pooled R: rebuild its trace the same way
        # (buyhold_stats already ran the same function on
        # the same trace; recompute the R list here by
        # re-running on the B&H trace is wasteful, so
        # instead pool from a re-derived trace is avoided:
        # the B&H pooled SQN is the pooled R over episodes,
        # computed in _pooled_bh below via buyhold traces.)
        pooled_bh_r.extend(_bh_r_multiples(ep, panel, days))
    champ_cagrs = [s["cagr"] for s in champ_stats]
    bh_cagrs = [s["cagr"] for s in bh_stats]
    excess = [c - b for c, b in zip(champ_cagrs, bh_cagrs)]
    champ_sharpe_med, champ_sharpe_n = _median_or_none(
        [s["sharpe"] for s in champ_stats])
    bh_sharpe_med, bh_sharpe_n = _median_or_none(
        [s["sharpe"] for s in bh_stats])
    champ_dd = [s["max_drawdown"] for s in champ_stats]
    bh_dd = [s["max_drawdown"] for s in bh_stats]
    struct = _structure(layout)
    pop_share = _density_share(density_population)
    mask_share = _density_share(density_masking)
    n_tested = _n_tested_units(layout)
    if ladder is not None:
        medians = np.asarray(ladder["medians"], dtype=float)
        pct = champion_percentile(float(np.median(champ_cagrs)),
                                  medians)
        percentile = {"value": pct,
                      "ladder_arch_signature":
                          ladder.get("arch_signature"),
                      "ladder_n_draws": int(
                          ladder.get("n_draws", len(medians))),
                      "ladder_unit": _f(ladder.get("unit")),
                      "reason": None}
    else:
        percentile = {"value": None,
                      "ladder_arch_signature": None,
                      "ladder_n_draws": 0,
                      "ladder_unit": None,
                      "reason": "no ladder computed yet "
                                "(ladder is built at stage end "
                                "and whenever the architecture "
                                "changes)"}
    snap = {
        "total_gens": int(total_gens),
        "stage": int(stage),
        "phase": str(phase),
        "gen_in_phase": int(gen_in_phase),
        "label": str(label),
        "genome_role": str(genome_role),
        "n_episodes": len(episodes),
        "episode_days": days,
        "cagr": {
            "champion_median": float(np.median(champ_cagrs)),
            "buyhold_median": float(np.median(bh_cagrs)),
            "median_excess": float(np.median(excess)),
            "champion_per_episode": [float(v)
                                     for v in champ_cagrs],
            "buyhold_per_episode": [float(v)
                                    for v in bh_cagrs],
        },
        "sharpe": {
            "champion_median": champ_sharpe_med,
            "champion_n_defined": champ_sharpe_n,
            "buyhold_median": bh_sharpe_med,
            "buyhold_n_defined": bh_sharpe_n,
            "difference": (None if champ_sharpe_med is None
                           or bh_sharpe_med is None else
                           float(champ_sharpe_med
                                 - bh_sharpe_med)),
            "n_returns_per_episode": int(days + 1),
            "note": (f"{days + 1}-mark snapshot Sharpe "
                     f"(value path V0 + {days} decision-day "
                     f"marks + final mark = {days + 2} points, "
                     f"{days + 1} returns at {days} days); "
                     f"real, but noisy at {days} days"),
        },
        "max_drawdown": {
            "champion_worst": float(np.min(champ_dd)),
            "champion_median": float(np.median(champ_dd)),
            "buyhold_worst": float(np.min(bh_dd)),
            "buyhold_median": float(np.median(bh_dd)),
        },
        "sqn": {
            "champion": _jsonable(_sqn_from_r(pooled_r)),
            "buyhold": _jsonable(_sqn_from_r(pooled_bh_r)),
        },
        "trades": {
            "buys_mean": float(np.mean(
                [s["n_buys"] for s in champ_stats])),
            "buys_median": float(np.median(
                [s["n_buys"] for s in champ_stats])),
            "sells_mean": float(np.mean(
                [s["n_sells"] for s in champ_stats])),
            "sells_median": float(np.median(
                [s["n_sells"] for s in champ_stats])),
            "turnover_mean": float(np.mean(
                [s["turnover"] for s in champ_stats])),
            "turnover_median": float(np.median(
                [s["turnover"] for s in champ_stats])),
            "completed_mean": float(np.mean(
                [s["n_completed_trades"] for s in champ_stats])),
            "completed_median": float(np.median(
                [s["n_completed_trades"] for s in champ_stats])),
            "completed_pooled": int(len(pooled_r)),
            "mean_invested_pct": float(np.mean(mean_invested)),
        },
        "percentile": percentile,
        "density": {
            "population": _jsonable(density_population),
            "population_useful_share": pop_share,
            "masking": _jsonable(density_masking),
            "masking_useful_share": mask_share,
            "n_params": int(struct["total_genes"]),
            "n_tested_units": int(n_tested),
            "population_effective_genes":
                (None if pop_share is None else
                 float(pop_share * struct["total_genes"])),
            "masking_effective_units":
                (None if mask_share is None else
                 float(mask_share * n_tested)),
        },
        "structure": struct,
        "arch_signature": arch_signature or struct["arch_signature"],
    }
    return _jsonable(snap)


def _bh_r_multiples(episode, panel, days):
    """R-multiples for the buy-and-hold trace of one
    episode (same construction as in ``buyhold_stats``)."""
    basket = np.asarray(episode["basket"])
    start = int(episode["start"])
    prices = np.asarray(panel.prices, dtype=float)
    p = prices[basket][:, start:start + days + 1]
    p0 = p[:, 0]
    n = len(basket)
    shares = (stock.V0 / n) / (1.0 + stock.COST_RATE) / p0
    shares_end = np.tile(shares, (days, 1))
    mean_ratio = np.mean(p / p0[:, None], axis=0)
    path = stock.V0 * mean_ratio / (1.0 + stock.COST_RATE)
    return _r_multiples(shares_end, p, path[:-1],
                        float(path[-1]), basket, start, panel)


# ================================================================
# 3. Random ladder (alloc_v3_env convention)
# ================================================================

def random_ladder(domain, layout, episodes, target_invested_pct,
                 rng, n_draws=200):
    """Exposure-matched random ladder on the given episodes
    (the ``alloc_v3_env.random_baseline`` convention):
    per draw, iid uniform scores 0..5 per ticker per day
    (``rng.integers(0, 6)``), RANDOM execution order per day
    (``rng.permutation``), fixed unit
    ``min(6.0, target_invested_pct / (n * 2.5))`` floored at
    0.05, executed through the domain's own ``_execute_day``
    with the same cash/cost rules as the fitness path (V0
    start, 6bp, one overdraft penalty on unaffordable buys,
    mark at start+days, CAGR annualised 252/(days+1));
    per-draw metric = median episode CAGR across the given
    episodes. Returns ``(medians (n_draws,), unit)`` where
    ``unit`` is the EFFECTIVE unit actually applied (the
    floored value). ``layout`` is accepted for API symmetry
    with the snapshot builder and is not needed by the
    random policy (no genome is involved).
    ``target_invested_pct`` is the champion's mean invested
    % — the caller computes it from the detailed snapshot's
    mean invested % (``snapshot["trades"]
    ["mean_invested_pct"]``)."""
    if getattr(domain, "name", None) != "stock":
        raise ValueError(
            "bert_monitor.random_ladder is stock-domain only "
            f"(got domain name "
            f"{getattr(domain, 'name', None)!r})")
    panel = domain.panel
    prices_all = np.asarray(panel.prices, dtype=float)
    days = int(domain.episode_days)
    n = len(episodes[0]["basket"])
    unit = max(0.05, min(6.0, float(target_invested_pct)
                         / (n * 2.5)))
    medians = []
    for _ in range(int(n_draws)):
        cs = []
        for ep in episodes:
            basket = np.asarray(ep["basket"])
            start = int(ep["start"])
            p_ep = prices_all[basket]  # (N, T)
            cash = np.array([stock.V0])
            shares = np.zeros((1, n))
            accrued = np.zeros(1)
            for day in range(days):
                t = start + day
                prices = p_ep[:, t]
                levels = rng.integers(0, 6, size=(1, n)) \
                    .astype(float)
                order = rng.permutation(n)[None, :]
                unit_arr = np.array([unit])
                cash, shares, accrued, _ = stock._execute_day(
                    cash, shares, accrued, prices, levels,
                    unit_arr, order, record=False)
            p_end = p_ep[:, start + days]
            v_final = float(cash[0] + shares[0] @ p_end
                            - accrued[0])
            cagr = float((v_final / stock.V0)
                         ** (TRADING_DAYS / (days + 1)) - 1.0) \
                if v_final > 0 else -1.0
            cs.append(cagr)
        medians.append(float(np.median(cs)))
    return np.array(medians, dtype=float), float(unit)


def champion_percentile(champion_median_cagr, medians):
    """The champion's percentile in the ladder: the share
    (0..100) of ladder draws whose median CAGR the
    champion's median is AT OR ABOVE, i.e.
    100 * mean(medians <= champion_median_cagr)."""
    meds = np.asarray(medians, dtype=float)
    if meds.size == 0:
        return 0.0
    return float(100.0 * np.mean(
        meds <= float(champion_median_cagr)))


def attach_ladder(snapshot, ladder):
    """Fill a snapshot's ``percentile`` field from a ladder
    dict ({"medians": [...], "unit": float,
    "arch_signature": str, "n_draws": int}) — the stage-end
    flow: the runner builds the snapshot first (the ladder
    needs the snapshot's mean invested %), recomputes the
    ladder, then attaches it before appending. Returns the
    same snapshot dict. With ladder None, leaves the
    existing percentile (with its reason) untouched."""
    if ladder is None:
        return snapshot
    medians = np.asarray(ladder["medians"], dtype=float)
    snapshot["percentile"] = {
        "value": champion_percentile(
            float(snapshot["cagr"]["champion_median"]), medians),
        "ladder_arch_signature": ladder.get("arch_signature"),
        "ladder_n_draws": int(
            ladder.get("n_draws", len(medians))),
        "ladder_unit": _f(ladder.get("unit")),
        "reason": None}
    return snapshot


def save_ladder_cache(path, arch_signature, medians, unit):
    """Atomic JSON cache of a ladder: {"arch_signature",
    "medians": [...], "unit": float, "n_draws": int}."""
    obj = {"arch_signature": str(arch_signature),
           "medians": [float(v) for v in
                       np.asarray(medians, dtype=float)],
           "unit": float(unit),
           "n_draws": int(len(np.asarray(medians)))}
    be.atomic_json(Path(path), obj)
    return obj


def load_ladder_cache(path):
    """Load a ladder cache; returns the dict or None when
    the file is missing or unparseable. The CALLER compares
    the stored arch_signature with the current one and
    recomputes on mismatch (see BERT_MONITOR_HOOKS.md)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(d, dict) or "medians" not in d:
        return None
    return d


# ================================================================
# 4. Files (all atomic)
# ================================================================

def write_status(out_dir, status):
    """Rewrite monitor_status.json atomically."""
    be.atomic_json(Path(out_dir) / STATUS_FILE,
                   _jsonable(status))


def append_snapshot(out_dir, snapshot):
    """Append one JSON record to monitor_snapshots.jsonl
    (single write + fsync; the file stays valid JSONL if
    the process is killed mid-generation — a torn last
    line is tolerated by ``read_snapshots``)."""
    path = Path(out_dir) / SNAPSHOTS_FILE
    line = json.dumps(_jsonable(snapshot)) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_snapshots(out_dir):
    """List of snapshot dicts, skipping a torn trailing
    line if present (never crashes on it)."""
    path = Path(out_dir) / SNAPSHOTS_FILE
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line: skip, keep the good ones
    return out


def write_html(out_dir, html):
    """Write monitor.html via a temp file + os.replace."""
    out_dir = Path(out_dir)
    tmp = out_dir / (HTML_FILE + ".tmp")
    tmp.write_text(html)
    os.replace(tmp, out_dir / HTML_FILE)


# ------------------------------------------------ HTML rendering

def _fmt(x, digits=4, pct=False):
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def _fmt_pct_points(x):
    """A CAGR-style fraction rendered as percent points."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    return f"{v * 100:.2f}%"


def _esc(s):
    return _html.escape(str(s))


def _svg_chart(title, series, ylabel="", height=180):
    """Inline SVG line chart. series: list of
    (label, [(x, y), ...], colour). Points with y None are
    skipped (a gap in the line). No external references;
    no xmlns attribute (inline SVG in HTML needs none)."""
    w, h = 640, height
    pad_l, pad_r, pad_t, pad_b = 56, 12, 26, 28
    xs, ys = [], []
    for _lab, pts, _c in series:
        for x, y in pts:
            if y is None:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return (f"<div class='chart'><h3>{_esc(title)}</h3>"
                f"<p class='muted'>no data yet</p></div>")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    pad_y = (ymax - ymin) * 0.08 or 1.0
    ymin -= pad_y
    ymax += pad_y

    def X(x):
        return pad_l + (x - xmin) / (xmax - xmin) * (w - pad_l - pad_r)

    def Y(y):
        return pad_t + (1 - (y - ymin) / (ymax - ymin)) \
            * (h - pad_t - pad_b)
    parts = [f"<div class='chart'><h3>{_esc(title)}</h3>",
             f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
    # axes
    parts.append(
        f"<line x1='{pad_l}' y1='{h - pad_b}' x2='{w - pad_r}' "
        f"y2='{h - pad_b}' stroke='#888' stroke-width='1'/>")
    parts.append(
        f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' "
        f"y2='{h - pad_b}' stroke='#888' stroke-width='1'/>")
    parts.append(
        f"<text x='8' y='{pad_t + 4}' font-size='11' "
        f"fill='#888'>{_esc(ylabel)}</text>")
    parts.append(
        f"<text x='{pad_l}' y='{h - 6}' font-size='11' "
        f"fill='#888'>gen {xmin:g}</text>")
    parts.append(
        f"<text x='{w - pad_r}' y='{h - 6}' font-size='11' "
        f"fill='#888' text-anchor='end'>gen {xmax:g}</text>")
    parts.append(
        f"<text x='{pad_l}' y='{pad_t - 8}' font-size='11' "
        f"fill='#888'>{_fmt(ymax, 3)}</text>")
    parts.append(
        f"<text x='{pad_l}' y='{h - pad_b + 14}' font-size='11' "
        f"fill='#888'>{_fmt(ymin, 3)}</text>")
    for label, pts, colour in series:
        segs, cur = [], []
        for x, y in pts:
            if y is None:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            cur.append((X(x), Y(y)))
        if cur:
            segs.append(cur)
        for seg in segs:
            d = " ".join(
                f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}"
                for k, (px, py) in enumerate(seg))
            parts.append(
                f"<path d='{d}' fill='none' stroke='{colour}' "
                f"stroke-width='2'/>")
        for x, y in pts:
            if y is None:
                continue
            parts.append(
                f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3' "
                f"fill='{colour}'><title>{_esc(label)} gen {x:g}: "
                f"{_fmt(y, 4)}</title></circle>")
    parts.append("</svg>")
    legend = " &nbsp; ".join(
        f"<span style='color:{c}'>&#9632; {_esc(lab)}</span>"
        for lab, _pts, c in series)
    parts.append(f"<p class='legend'>{legend}</p></div>")
    return "".join(parts)


def render_monitor_html(status, snapshots):
    """A SELF-CONTAINED HTML page (no server, no external
    deps, no CDN): backend line, run position, network
    structure, what is being added now, champion vs
    same-basket buy-and-hold table, and evolution curves as
    inline SVG line charts. Every number on the page comes
    from ``status`` / ``snapshots`` (no recomputation in the
    template — products such as density x network size are
    precomputed in the snapshot). The page renders from the
    snapshots file alone if status is missing (``status``
    may be None)."""
    status = status or {}
    snaps = list(snapshots or [])
    latest = snaps[-1] if snaps else None
    backend = status.get("backend") or {}
    growth = status.get("growth") or {}
    struct = status.get("structure") or (
        latest["structure"] if latest else None) or {}
    stacks = struct.get("stacks") or []

    def st(key, default="n/a"):
        v = status.get(key)
        return default if v is None else v

    # ---- backend line
    be_line = (
        f"Backend: {_esc(backend.get('backend', 'n/a'))} on "
        f"{_esc(backend.get('device', 'n/a'))}, precision "
        f"{_esc(backend.get('precision', 'n/a'))}")
    if backend.get("fell_back"):
        be_line += (f" — FALLBACK: {_esc(backend.get('note', ''))}")
    elif backend.get("note"):
        be_line += f" ({_esc(backend.get('note'))})"

    # ---- run position
    pos_rows = [
        ("Stage (epoch)", _esc(st("stage", "n/a"))),
        ("Phase", _esc(st("phase", "n/a"))),
        ("Generation in phase",
         _esc(st("gen_in_phase", "n/a"))),
        ("Total generations", _esc(st("total_gens", "n/a"))),
        ("Species count", _esc(st("n_species", "n/a"))),
        ("Population", _esc(st("population", "n/a"))),
        ("Best training score",
         _fmt(st("best_score", None))),
        ("Median training score",
         _fmt(st("median_score", None))),
        ("Elimination median",
         _fmt(st("elim_median", None))),
        ("Seconds per generation",
         _fmt(st("seconds_per_generation", None), 2)),
        ("ETA, current phase",
         _esc(st("eta_phase", "n/a"))),
        ("Snapshots", str(len(snaps))),
    ]
    pos_html = "".join(
        f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>"
        for k, v in pos_rows)

    # ---- structure
    if stacks:
        stack_rows = "".join(
            f"<tr><td>{s.get('stack')}</td>"
            f"<td>{s.get('blocks')}</td>"
            f"<td>{s.get('heads')}</td>"
            f"<td>{s.get('d')}</td>"
            f"<td>{s.get('ffn_hidden')}</td></tr>"
            for s in stacks)
        struct_html = (
            "<table><tr><th>stack</th><th>blocks</th>"
            "<th>heads</th><th>d</th><th>ffn_hidden</th></tr>"
            + stack_rows + "</table>"
            + f"<p>Total genes: "
            f"<b>{struct.get('total_genes', 'n/a')}</b> &nbsp; "
            f"Arch: <code>{_esc(struct.get('arch_signature', ''))}"
            f"</code></p>")
    else:
        struct_html = ("<p class='muted'>No structure recorded "
                       "yet.</p>")

    # ---- growth now
    tried = growth.get("tried_rules") or []
    verdicts = growth.get("verdicts") or []
    verdict_html = ""
    if verdicts:
        verdict_html = "<ul>" + "".join(
            f"<li>{_esc(v.get('rule', ''))} "
            f"({_esc(v.get('variant', ''))}): "
            f"<b>{_esc(v.get('verdict', ''))}</b>"
            + (f" — {_esc(v.get('detail', ''))}"
               if v.get("detail") else "")
            + "</li>" for v in verdicts) + "</ul>"
    growth_html = (
        f"<p>Growth rule now: <b>"
        f"{_esc(growth.get('rule') or 'none (first stage)')}"
        f"</b> &nbsp; Growth round: "
        f"{_esc(growth.get('growth_round', 'n/a'))}</p>"
        f"<p>Rules already tried this round: "
        f"{_esc(', '.join(tried) if tried else 'none')}</p>"
        + verdict_html)

    # ---- champion vs buy-and-hold (latest snapshot)
    if latest is not None:
        c = latest["cagr"]
        sh = latest["sharpe"]
        dd = latest["max_drawdown"]
        sq = latest["sqn"]
        tr = latest["trades"]
        pc = latest["percentile"]
        sq_c = sq["champion"]
        sq_b = sq["buyhold"]
        comp_rows = [
            ("CAGR, median", _fmt_pct_points(c["champion_median"]),
             _fmt_pct_points(c["buyhold_median"]),
             _fmt_pct_points(c["median_excess"])),
            ("Sharpe, median "
             f"(n={sh['champion_n_defined']} / "
             f"{sh['buyhold_n_defined']} defined)",
             _fmt(sh["champion_median"]),
             _fmt(sh["buyhold_median"]),
             _fmt(sh["difference"])),
            ("Max drawdown, worst episode",
             _fmt_pct_points(dd["champion_worst"]),
             _fmt_pct_points(dd["buyhold_worst"]),
             _fmt_pct_points(
                 None if dd["champion_worst"] is None
                 or dd["buyhold_worst"] is None else
                 dd["champion_worst"] - dd["buyhold_worst"])),
            ("Max drawdown, median episode",
             _fmt_pct_points(dd["champion_median"]),
             _fmt_pct_points(dd["buyhold_median"]),
             _fmt_pct_points(
                 None if dd["champion_median"] is None
                 or dd["buyhold_median"] is None else
                 dd["champion_median"] - dd["buyhold_median"])),
            ("SQN (pooled, N="
             f"{sq_c['n_trades']} / {sq_b['n_trades']})",
             _fmt(sq_c["sqn"]) + (
                 " *" if sq_c["small_sample"] else ""),
             _fmt(sq_b["sqn"]) + (
                 " *" if sq_b["small_sample"] else ""),
             "n/a"),
            ("Trades per sample, buys mean / median",
             f"{_fmt(tr['buys_mean'], 1)} / "
             f"{_fmt(tr['buys_median'], 1)}",
             "0 / 0 (no fills after the initial purchase)",
             "n/a"),
            ("Trades per sample, sells mean / median",
             f"{_fmt(tr['sells_mean'], 1)} / "
             f"{_fmt(tr['sells_median'], 1)}",
             "0 / 0", "n/a"),
            ("Turnover per sample, mean / median",
             f"{_fmt(tr['turnover_mean'], 0)} / "
             f"{_fmt(tr['turnover_median'], 0)}",
             "0 / 0", "n/a"),
            ("Completed holding episodes, mean / pooled N",
             f"{_fmt(tr['completed_mean'], 1)} / "
             f"{tr['completed_pooled']}",
             f"{_fmt(0, 1)} / {sq_b['n_trades']}", "n/a"),
            ("Percentile vs random ladder",
             _fmt(pc["value"], 1) if pc["value"] is not None
             else "n/a",
             "n/a (ladder is the reference)",
             _esc(pc["reason"] or "")),
        ]
        comp_html = (
            "<table><tr><th></th><th>champion</th>"
            "<th>same-basket buy-and-hold</th>"
            "<th>difference</th></tr>"
            + "".join(
                f"<tr><td>{k}</td><td>{a}</td><td>{b}</td>"
                f"<td>{d}</td></tr>"
                for k, a, b, d in comp_rows)
            + "</table>"
            + f"<p class='muted'>Snapshot at total gen "
            f"{latest['total_gens']} (stage {latest['stage']}, "
            f"phase {latest['phase']}, label "
            f"{latest['label']}, role {latest['genome_role']}); "
            f"{latest['n_episodes']} episodes of "
            f"{latest['episode_days']} days; Sharpe from "
            f"{latest['sharpe']['n_returns_per_episode']} daily "
            f"returns per episode (value path V0 + "
            f"{latest['episode_days']} decision-day marks + "
            f"final mark = {latest['episode_days'] + 2} points). "
            f"* SQN small-sample flag: N &lt; 30. Buy-and-hold "
            f"SQN is normally n/a on 60-day samples because "
            f"its positions never close inside the sample. "
            f"Executions and completed trades are different "
            f"numbers and are never merged into one label.</p>")
    else:
        comp_html = ("<p class='muted'>No detailed snapshot yet."
                     "</p>")

    # ---- evolution curves
    def pts(key_fn):
        return [(s["total_gens"], key_fn(s)) for s in snaps]

    curves = []
    if snaps:
        curves.append(_svg_chart(
            "Median CAGR — champion vs buy-and-hold (+ excess)",
            [("champion",
              pts(lambda s: s["cagr"]["champion_median"]), "#1f77b4"),
             ("buy-and-hold",
              pts(lambda s: s["cagr"]["buyhold_median"]), "#888888"),
             ("excess (median)",
              pts(lambda s: s["cagr"]["median_excess"]), "#2ca02c")],
            ylabel="CAGR"))
        curves.append(_svg_chart(
            "Median Sharpe — champion vs buy-and-hold",
            [("champion",
              pts(lambda s: s["sharpe"]["champion_median"]),
              "#1f77b4"),
             ("buy-and-hold",
              pts(lambda s: s["sharpe"]["buyhold_median"]),
              "#888888")],
            ylabel="Sharpe"))
        curves.append(_svg_chart(
            "SQN — champion (pooled; N shown in the table)",
            [("champion SQN",
              pts(lambda s: s["sqn"]["champion"]["sqn"]),
              "#9467bd")],
            ylabel="SQN"))
        curves.append(_svg_chart(
            "Max drawdown — worst and median episode",
            [("champion worst",
              pts(lambda s: s["max_drawdown"]["champion_worst"]),
              "#d62728"),
             ("champion median",
              pts(lambda s: s["max_drawdown"]["champion_median"]),
              "#ff7f0e"),
             ("buy-and-hold worst",
              pts(lambda s: s["max_drawdown"]["buyhold_worst"]),
              "#7f7f7f"),
             ("buy-and-hold median",
              pts(lambda s: s["max_drawdown"]["buyhold_median"]),
              "#bcbd22")],
            ylabel="drawdown"))
        curves.append(_svg_chart(
            "Information density — population reading "
            "(share of weights reading useful)",
            [("population density",
              pts(lambda s: s["density"]
                  ["population_useful_share"]), "#1f77b4")],
            ylabel="share"))
        curves.append(_svg_chart(
            "Information density x network size — population "
            "effective genes",
            [("density x total genes",
              pts(lambda s: s["density"]
                  ["population_effective_genes"]), "#1f77b4")],
            ylabel="effective genes"))
        curves.append(_svg_chart(
            "Information density — masking reading "
            "(share of tested units with a real silencing drop)",
            [("masking density",
              pts(lambda s: s["density"]
                  ["masking_useful_share"]), "#2ca02c")],
            ylabel="share"))
        curves.append(_svg_chart(
            "Information density x network size — masking "
            "effective units",
            [("density x tested units",
              pts(lambda s: s["density"]
                  ["masking_effective_units"]), "#2ca02c")],
            ylabel="effective units"))
        size_pts = pts(lambda s: s["structure"]["total_genes"])
        curves.append(_svg_chart(
            "Network size — total genes (steps mark growth)",
            [("total genes", size_pts, "#d62728")],
            ylabel="genes"))
        # growth-step labels under the size chart
        steps = []
        prev = None
        for s in snaps:
            g = s["structure"]["total_genes"]
            if prev is not None and g != prev:
                steps.append(
                    f"gen {s['total_gens']}: {prev} -&gt; {g} genes")
            prev = g
        if steps:
            curves.append(
                "<p class='muted'>Growth steps: "
                + "; ".join(steps)
                + ". The rule that caused each step is recorded "
                "in the run's event log (see 'what is being "
                "added now' above).</p>")
    curves_html = "".join(curves) if curves else \
        "<p class='muted'>No snapshots yet — curves appear after " \
        "the first detailed snapshot.</p>"

    status_note = ""
    if not status:
        status_note = (
            "<p class='muted'>Note: monitor_status.json was "
            "missing when this page was rendered; run position "
            "and growth state come from the snapshots file "
            "alone.</p>")
    updated = _esc(status.get("updated_at", "n/a"))
    run_name = _esc(status.get("run_name", "n/a"))
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>BERT engine live monitor — {_esc(run_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;max-width:900px;
color:#222;background:#fff}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}
table{{border-collapse:collapse;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
th{{background:#f0f0f0}}
.muted{{color:#666;font-size:13px}}
.legend{{font-size:13px}}
.chart{{margin:16px 0}}
code{{font-size:12px;word-break:break-all}}
</style></head><body>
<h1>BERT evolution engine — live run monitor</h1>
<p class='muted'>Run: {run_name} &nbsp;|&nbsp; Updated: {updated}</p>
{status_note}
<h2>Backend</h2><p>{be_line}</p>
<h2>Run position</h2><table>{pos_html}</table>
<h2>Network structure</h2>{struct_html}
<h2>What is being added now</h2>{growth_html}
<h2>Champion vs same-basket buy-and-hold (latest snapshot)</h2>
{comp_html}
<h2>Evolution curves</h2>{curves_html}
</body></html>"""
    return html_doc
