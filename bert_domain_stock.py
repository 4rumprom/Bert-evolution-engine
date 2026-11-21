"""Stock allocation domain adapter for the parametric BERT
growth engine, against the genetic_lab conventions
(~/workspace/trading-agent/genetic_lab/alloc_v3_env.py):

  - tokens = the tickers of a basket; per-token input =
    [9 point-in-time Fourier K4 features (standardised),
    current position as a fraction of portfolio value]
    (input_dim = 10, the V3 encoder input convention);
  - allocation head (the domain decode head): a shared
    per-token score read (raw merit -> level 0..5 by the V3
    quantisation clip(round((tanh(raw)+1)*2.5), 0, 5)) and
    one unit size (percent of portfolio, capped at 6% by a
    sigmoid) read from [mean encoder token, planned invested
    %, planned cash %] -- the parallel-head analogue of the
    V3 decoder's sequential feedback (planned invested uses
    the previous day's unit, the env's FIRST_UNIT_PCT = 3.0
    seed);
  - execution: instructions run in ranked order (raw score
    descending, ties by alphabetical ticker rank -- the V3
    determinism convention), cash-checked: an unaffordable
    buy is ignored and charged ONE overdraft penalty of 6bp
    on the shortfall; executed changes pay 6bp once; costs
    beyond cash accrue as a fees-payable balance netted in
    portfolio value (the env's accounting identity
    V = cash + positions - accrued holds throughout);
  - fitness: basket-relative -- the median over episodes of
    (episode CAGR - uniform buy-and-hold CAGR of the same
    basket), the sponsor's validity-bar yardstick, with the
    env's buy-and-hold convention (one 6bp charge on the
    initial purchase, mark-to-market final value, CAGR
    annualised with exponent 252/(days+1));
  - fixed sets: the elimination set is a seeded sample of
    training episodes; the validation set is the frozen
    common episode set (validation_episodes_v1.json, 64
    episodes, basket 10, 60 days, 2020-2022) when the real
    panel is supplied, or seeded synthetic episodes on a
    synthetic panel (smoke scale);
  - episode sampling is validity-aware (the env's
    convention): a sampled (basket, start) counts only if
    every basket ticker's marking price is finite on every
    day of [start, start + episode_days] -- the decision
    days plus the final mark day -- so neither the genome
    leg nor the buy-and-hold leg of the fitness can go NaN
    on missing data. The frozen validation episodes are
    fixed and are never resampled or filtered; validity
    applies to the sampled training / elimination draws
    only. On a NaN-free panel the sampler consumes the RNG
    exactly as the plain uniform sampler did and returns
    identical episodes for identical seeds.

The panel is any object with .prices (A, T), .feats
(A, T, 9) standardised features and .tickers [str]. The
real genetic_lab Panel satisfies this unchanged; the smoke
test uses make_synthetic_panel (random-walk prices and
synthetic standardised features -- a stand-in for the
Fourier feature computation, which stays the env's job).

Alternatively, if the panel carries OHLCV (.open / .high /
.low / .close / .volume) and domain_config contains a
"feature_spec", the features are built from the bars by
bert_preprocessor and the feature count F is whatever the
spec produces: input_dim = F + 1. Prices for marking are
then .prices if present, else .close. The standardisation
cutoff is domain_config "stats_end" when set; when it is
not set and the panel carries .train_end_index (a bundle
panel from bert_stock_data), that index -- the end of the
2015-2019 training window -- is the default, so a bundle
run cannot leak post-training statistics by omission. An
explicit "stats_end" always wins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be
import bert_preprocessor as bp

COST_RATE = 0.0006          # 6bp, the env standard
V0 = 100_000.0
FIRST_UNIT_PCT = 3.0
MAX_UNIT_PCT = 6.0
N_FEAT = 9
# redraw bound for validity-aware episode sampling; mirrors
# the GA env's 200-attempt discipline in
# alloc_v3_env.Panel.sample_episode
MAX_EPISODE_REDRAWS = 200


def head_spec(d):
    return [("w_score", (d,)), ("b_score", (1,)),
            ("W_unit", (d + 2,)), ("b_unit", (1,))]


def widen_head(head_old, d0, d1):
    """Score read: new coordinates contribute zero at
    transplant. Unit read: new state-coordinate rows zero;
    the invested/cash rows are untouched."""
    out = {}
    w = np.zeros(d1); w[:d0] = head_old["w_score"]
    out["w_score"] = w
    out["b_score"] = head_old["b_score"].copy()
    W = np.zeros(d1 + 2)
    W[:d0] = head_old["W_unit"][:d0]
    W[d1:] = head_old["W_unit"][d0:]
    out["W_unit"] = W
    out["b_unit"] = head_old["b_unit"].copy()
    return out


def make_synthetic_panel(n_tickers=30, n_days=500, seed=20261120):
    """Smoke-scale panel: geometric random-walk prices and
    synthetic standardised per-ticker features with the same
    shapes as the real panel."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, (n_tickers, n_days))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=1))
    feats = rng.normal(0, 1, (n_tickers, n_days, N_FEAT))
    tickers = [f"S{i:03d}" for i in range(n_tickers)]

    class _P:
        pass
    p = _P()
    p.prices, p.feats, p.tickers = prices, feats, tickers
    return p


def make_synthetic_ohlcv_panel(n_tickers=30, n_days=500,
                               seed=20261120):
    """Smoke-scale OHLCV panel for the bert_preprocessor
    path: geometric random-walk closes, opens near the
    previous close, highs / lows bracketing open and close,
    lognormal volume, and two smooth synthetic fundamentals
    (earnings_yield, book_to_market). .prices is the close;
    there is deliberately no .feats -- features are the
    preprocessor's job when a feature_spec is configured."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.02, (n_tickers, n_days))
    close = 100.0 * np.exp(np.cumsum(rets, axis=1))
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0] * (1.0 + rng.normal(
        0, 0.003, n_tickers))
    open_[:, 1:] = close[:, :-1] * (1.0 + rng.normal(
        0, 0.003, (n_tickers, n_days - 1)))
    spread = np.abs(rng.normal(0.008, 0.005,
                               (n_tickers, n_days)))
    high = np.maximum(open_, close) * (1.0 + spread * rng.random(
        (n_tickers, n_days)))
    low = np.minimum(open_, close) * (1.0 - spread * rng.random(
        (n_tickers, n_days)))
    volume = np.exp(rng.normal(np.log(1e6), 0.5,
                               (n_tickers, n_days))) \
        * (1.0 + 3.0 * np.abs(rets))
    ey = np.clip(rng.uniform(-0.02, 0.12, n_tickers)[:, None]
                 + np.cumsum(rng.normal(
                     0, 0.0015, (n_tickers, n_days)), axis=1),
                 -0.10, 0.20)
    btm = np.clip(rng.uniform(0.2, 1.2, n_tickers)[:, None]
                  + np.cumsum(rng.normal(
                      0, 0.004, (n_tickers, n_days)), axis=1),
                  0.05, 2.0)
    tickers = [f"S{i:03d}" for i in range(n_tickers)]

    class _P:
        pass
    p = _P()
    p.open, p.high, p.low, p.close = open_, high, low, close
    p.volume = volume
    p.prices = close
    p.tickers = tickers
    p.fundamentals = {"earnings_yield": ey,
                      "book_to_market": btm}
    return p


def load_frozen_episodes(path):
    """Parse the frozen common episode set
    (validation_episodes_v1.json) into adapter episodes.
    Structure only; evaluating them needs the real panel."""
    d = json.loads(Path(path).read_text())
    eps = d["episodes"] if isinstance(d, dict) else d
    out = []
    for e in eps:
        out.append({"basket": np.array(e["basket_panel_indices"]),
                    "tickers": list(e["tickers"]),
                    "start": int(e["start_index"])})
    return out


class StockDomain:
    name = "stock"
    input_dim = N_FEAT + 1

    def __init__(self, cfg: be.EngineConfig, backend, panel=None,
                 val_episodes=None):
        self.cfg = cfg
        self.backend = backend
        dc = cfg.domain_config
        self.episode_days = int(dc.get("episode_days", 60))
        self.basket_n = int(dc.get("basket_n", 10))
        self.n_train_episodes = int(dc.get("n_train_episodes", 3))
        self.n_elim_episodes = int(dc.get("n_elim_episodes", 4))
        self.panel = panel
        self._val_episodes = val_episodes
        self._elim_episodes = None
        self._phase_a_episodes = None
        # feature source: the panel's own .feats (legacy
        # contract, N_FEAT wide), or -- when the panel has
        # OHLCV and the config carries a feature_spec --
        # bert_preprocessor output of whatever width the
        # spec produces. input_dim = n_feat + 1 either way.
        self.n_feat = N_FEAT
        self._prices = None
        self._feats = None
        self.feat_stats = None
        self._valid_start = None
        self._n_valid_at_start = None
        self._valid_any = False
        if panel is not None:
            self._prices = panel.prices \
                if hasattr(panel, "prices") else panel.close
            self._build_validity()
            spec = dc.get("feature_spec")
            if spec is not None and hasattr(panel, "close"):
                stats_end = dc.get("stats_end")
                if stats_end is None and hasattr(
                        panel, "train_end_index"):
                    stats_end = panel.train_end_index
                feats, stats = bp.build_features(
                    panel, spec, stats_end=stats_end)
                self._feats = feats
                self.feat_stats = stats
                self.n_feat = int(feats.shape[2])
            else:
                self._feats = panel.feats
        self.input_dim = self.n_feat + 1

    # ---- episode providers --------------------------------------
    def _build_validity(self):
        """Precompute per-ticker valid starts on the marking
        prices: _valid_start[a, s] is True iff the price is
        finite on every day of [s, s + episode_days] -- the
        episode's decision days plus its final mark day
        (episode_cagrs marks the final portfolio at
        start + episode_days, and the buy-and-hold leg reads
        the same day). Mirrors the price leg of
        alloc_v3_env.Panel._window_validity: cumulative
        bad-day counts, O(A * T), no window scans. The env's
        feature leg is deliberately NOT mirrored: features
        here are preprocessor output (NaN neutralised to 0)
        or the panel's own standardised feats, and only
        prices can produce NaN fitness."""
        prices = np.asarray(self._prices, dtype=float)
        A, T = prices.shape
        days = self.episode_days
        n_starts = T - days
        if n_starts <= 0:
            self._valid_start = np.zeros((A, 0), dtype=bool)
            self._n_valid_at_start = np.zeros(0, dtype=int)
            self._valid_any = False
            return
        bad = (~np.isfinite(prices)).astype(np.int32)
        cum = np.concatenate(
            [np.zeros((A, 1), np.int32),
             np.cumsum(bad, axis=1)], axis=1)
        starts = np.arange(n_starts)
        ok = (cum[:, starts + days + 1] - cum[:, starts]) == 0
        self._valid_start = ok
        self._n_valid_at_start = ok.sum(axis=0)
        self._valid_any = bool(
            (self._n_valid_at_start >= self.basket_n).any())

    def _episode_valid(self, basket, start):
        """O(basket) lookup against the precomputed
        structure; no window scan."""
        return bool(self._valid_start[basket, start].all())

    def _draw_episode(self, rng):
        """One validity-aware episode draw. The FIRST attempt
        is byte-for-byte the plain uniform draw (start via
        rng.integers, then basket via rng.choice, sorted), so
        on a NaN-free panel the RNG stream and the returned
        episodes are identical to the pre-validity sampler.
        Only an invalid first attempt consumes extra draws:
        redraw up to MAX_EPISODE_REDRAWS times; then scan
        starts deterministically (0, 1, 2, ...) for the last
        basket; if that basket has no valid start anywhere,
        redraw baskets (bounded), scanning starts for each;
        as a final guarantee, take the start with the most
        valid tickers and its first basket_n valid tickers.
        Raises RuntimeError when the panel admits no valid
        episode at all (checked up front, so a starved panel
        fails immediately, never loops)."""
        panel = self.panel
        A, T = panel.prices.shape
        days = self.episode_days
        if not self._valid_any:
            raise RuntimeError(
                "stock domain: no valid episode exists on "
                "this panel (no start has "
                f"{self.basket_n} tickers with finite prices "
                f"across a {days}-day episode window plus "
                "its mark day); refusing to sample episodes "
                "whose fitness would be NaN")
        basket = None
        for _ in range(MAX_EPISODE_REDRAWS):
            start = int(rng.integers(0, T - days))
            basket = np.sort(rng.choice(
                A, size=self.basket_n, replace=False))
            if self._episode_valid(basket, start):
                return basket, start
        for s in range(T - days):
            if self._episode_valid(basket, s):
                return basket, s
        for _ in range(MAX_EPISODE_REDRAWS):
            basket = np.sort(rng.choice(
                A, size=self.basket_n, replace=False))
            for s in range(T - days):
                if self._episode_valid(basket, s):
                    return basket, s
        s = int(np.argmax(self._n_valid_at_start))
        basket = np.sort(
            np.where(self._valid_start[:, s])[0][:self.basket_n])
        return basket, s

    def _sample_episodes(self, rng, k):
        panel = self.panel
        eps = []
        for _ in range(k):
            basket, start = self._draw_episode(rng)
            eps.append({"basket": basket,
                        "tickers": [panel.tickers[i] for i in basket],
                        "start": start})
        return eps

    def _fixed(self, seed, k):
        rng = np.random.default_rng(seed)
        return self._sample_episodes(rng, k)

    def elim_episodes(self):
        if self._elim_episodes is None:
            self._elim_episodes = self._fixed(
                self.cfg.elim_seed, self.n_elim_episodes)
        return self._elim_episodes

    def phase_a_episodes(self):
        if self._phase_a_episodes is None:
            self._phase_a_episodes = self._fixed(
                self.cfg.phase_a_seed, self.n_train_episodes)
        return self._phase_a_episodes

    def val_episodes(self):
        if self._val_episodes is not None:
            return self._val_episodes
        return self._fixed(self.cfg.val_seed + 777,
                           self.n_elim_episodes)

    # ---- batches --------------------------------------------------
    def train_batch(self, phase, stage, gen):
        if phase == "A":
            return {"episodes": self.phase_a_episodes(),
                    "tag": "phaseA_train"}
        seed = be.gen_instance_seed(self.cfg, stage, gen)
        rng = np.random.default_rng(seed)
        return {"episodes": self._sample_episodes(
                    rng, self.n_train_episodes),
                "tag": f"phaseB_s{stage}_g{gen}"}

    def elim_batch(self):
        return {"episodes": self.elim_episodes(), "tag": "elim"}

    def val_batch(self):
        return {"episodes": self.val_episodes(), "tag": "val"}

    # ---- one decision day, batched over genomes ------------------
    def _decide_day(self, layout, G, Y):
        """Allocation head. G: (M, n_params); Y: (M, N, d)
        encoder outputs for one day -> (levels (M, N) float,
        raw (M, N), unit_pct (M,), prev-dependent planned
        invested is computed by the caller)."""
        couple = len(layout.arch.couples) - 1
        w = layout.get(np.asarray(G), f"c{couple}.head.w_score")
        b = layout.get(np.asarray(G), f"c{couple}.head.b_score")
        raw = np.einsum("mnd,md->mn", Y, w) + b[:, 0][:, None]
        levels = np.clip(np.round((np.tanh(raw) + 1.0) * 2.5),
                         0, 5)
        return levels, raw

    def _unit(self, layout, G, Y, levels, prev_unit):
        couple = len(layout.arch.couples) - 1
        W = layout.get(np.asarray(G), f"c{couple}.head.W_unit")
        b = layout.get(np.asarray(G), f"c{couple}.head.b_unit")
        state = Y.mean(axis=1)                          # (M, d)
        planned_invested = (levels * prev_unit[:, None]).sum(axis=1)
        u_in = np.concatenate(
            [state, (planned_invested / 100.0)[:, None],
             ((100.0 - planned_invested) / 100.0)[:, None]], axis=1)
        raw_u = np.einsum("mf,mf->m", u_in, W) + b[:, 0]
        unit = MAX_UNIT_PCT / (1.0 + np.exp(-raw_u))
        return unit

    # ---- episode simulation, batched over genomes ----------------
    def episode_cagrs(self, layout, G, episode, cache=None,
                      tag="", collect=False):
        """Run one episode for every genome row. Returns
        (cagrs (M,), buyhold_cagr float); with collect=True
        also the per-day decision trace (levels, ranked
        order, unit) -- the stock analogue of a TSP tour for
        preservation checks. Arithmetic mirrors
        alloc_v3_env.run_episode / execute_day / buyhold_cagr,
        vectorised over genomes (ticker loop kept, as in the
        env, because execution is cash-checked in ranked
        order)."""
        panel = self.panel
        basket = episode["basket"]
        start = episode["start"]
        days = self.episode_days
        N = len(basket)
        M = len(G)
        prices_all = self._prices[basket]              # (N, T)
        feats_all = self._feats[basket]           # (N, T, F)
        cash = np.full(M, V0)
        shares = np.zeros((M, N))
        accrued = np.zeros(M)
        prev_unit = np.full(M, FIRST_UNIT_PCT)
        alpha_rank = np.empty(N, dtype=int)
        alpha_rank[np.argsort(episode["tickers"],
                              kind="stable")] = np.arange(N)
        tol = 1e-7
        trace = []
        for day in range(days):
            t = start + day
            prices = prices_all[:, t]                  # (N,)
            v_pre = cash + shares @ prices - accrued
            pos_pct = shares * prices[None, :] \
                / v_pre[:, None] * 100.0
            X = np.concatenate(
                [np.broadcast_to(feats_all[:, t, :],
                                 (M, N, self.n_feat)),
                 (pos_pct / 100.0)[:, :, None]], axis=2)
            # token inputs differ per genome (positions), so
            # encoding runs per genome on its own day tensor
            Y = self._encode_day(layout, G, X, cache,
                                 f"{tag}:{episode['start']}:{day}")
            levels, raw = self._decide_day(layout, G, Y)
            unit = self._unit(layout, G, Y, levels, prev_unit)
            # ranked order: raw descending, ties by alpha rank
            pre = np.argsort(alpha_rank, kind="stable")
            order = pre[np.argsort(-raw[:, pre], axis=1,
                                   kind="stable")]
            if collect:
                trace.append((levels.copy(), order.copy(),
                              unit.copy()))
            for k in range(N):
                i = order[:, k]
                value_now = cash + shares @ prices - accrued
                target = levels[np.arange(M), i] * unit \
                    / 100.0 * value_now
                cur = shares[np.arange(M), i] * prices[i]
                delta = target - cur
                buy = delta > tol
                afford = buy & (cash >= delta)
                if afford.any():
                    idx = np.where(afford)[0]
                    ii = i[idx]
                    shares[idx, ii] += delta[idx] / prices[ii]
                    cash[idx] -= delta[idx]
                    cost = COST_RATE * delta[idx]
                    pay = np.minimum(cash[idx], cost)
                    cash[idx] -= pay
                    accrued[idx] += cost - pay
                over = buy & ~afford
                if over.any():
                    idx = np.where(over)[0]
                    accrued[idx] += COST_RATE * (delta[idx]
                                                 - cash[idx])
                sell = delta < -tol
                if sell.any():
                    idx = np.where(sell)[0]
                    ii = i[idx]
                    proceeds = -delta[idx]
                    shares[idx, ii] -= proceeds / prices[ii]
                    cost = COST_RATE * proceeds
                    cash[idx] += proceeds - cost
                    pay = np.minimum(cash[idx], accrued[idx])
                    cash[idx] -= pay
                    accrued[idx] -= pay
            prev_unit = unit
        p_end = prices_all[:, start + days]
        v_final = cash + shares @ p_end - accrued
        cagrs = np.where(v_final > 0,
                         (v_final / V0) ** (252.0 / (days + 1)) - 1.0,
                         -1.0)
        p0 = prices_all[:, start]
        mean_ratio = float(np.mean(p_end / p0))
        bh_final = V0 * mean_ratio / (1.0 + COST_RATE)
        bh_cagr = float((bh_final / V0) ** (252.0 / (days + 1)) - 1.0)
        if collect:
            return cagrs, bh_cagr, trace
        return cagrs, bh_cagr

    def _encode_day(self, layout, G, X, cache, tag):
        """Encode one decision day. X: (M, N, input_dim) with
        per-genome token inputs. The engine encoder consumes
        one shared instance tensor per call, so each genome's
        day tensor is encoded as its own 'instance batch' of
        one: encode_population is called per genome with
        X = X[m][None] -- at smoke scale this is exact and
        simple; the segment cache still applies within the
        call (positions make later days genome-specific, so
        cross-genome sharing on day >= 1 correctly misses)."""
        outs = []
        for m in range(len(G)):
            Y = be.encode_population(
                self.backend, layout, G[m][None, :],
                X[m][None, :, :], groups=None, cache=cache,
                x_tag=tag, chunk=self.cfg.chunk)
            outs.append(Y[0, 0])
        return np.stack(outs, axis=0)

    # ---- evaluation ----------------------------------------------
    def evaluate(self, layout, G, batch, groups=None, cache=None):
        """Fitness: median over episodes of (episode CAGR -
        uniform buy-and-hold CAGR of the same basket)."""
        G = np.atleast_2d(np.asarray(G, dtype=float))
        excess = []
        for ep in batch["episodes"]:
            cagrs, bh = self.episode_cagrs(
                layout, G, ep, cache=cache, tag=batch["tag"])
            excess.append(cagrs - bh)
        return np.median(np.stack(excess, axis=1), axis=1)

    def evaluate_single(self, layout, genome, batch):
        return float(self.evaluate(
            layout, np.asarray(genome)[None, :], batch)[0])

    def validation_score(self, layout, genome):
        return self.evaluate_single(layout, genome,
                                    self.val_batch())

    def validation_detail(self, layout, genome):
        batch = self.val_batch()
        G = np.asarray(genome)[None, :]
        per = []
        for ep in batch["episodes"]:
            cagrs, bh = self.episode_cagrs(layout, G, ep)
            per.append(float(cagrs[0] - bh))
        return {"median_excess_cagr": float(np.median(per)),
                "mean_excess_cagr": float(np.mean(per)),
                "per_episode_excess": per,
                "n_episodes": len(per)}

    def preservation(self, layout_old, genome_old, layout_new,
                     genome_new):
        """The stock form of the exact preservation check:
        the DISCRETE decisions -- score levels and ranked
        execution order -- must be identical on every
        validation episode day (the analogue of identical
        TSP tours), and per-episode CAGRs must agree to
        float tolerance. (The unit size is a continuous
        read of the encoder state, so for the widen rule --
        function-preserving in exact arithmetic -- CAGRs
        agree to ~1e-12 rather than bit-exactly; for the
        zero-addition rules they are bit-exact. The
        decision-identity gate is what binds.)"""
        batch = self.val_batch()
        decisions_same = True
        max_cagr_diff = 0.0
        max_unit_rel = 0.0
        for ep in batch["episodes"]:
            co, _, tro = self.episode_cagrs(
                layout_old, np.asarray(genome_old)[None, :], ep,
                collect=True)
            cn, _, trn = self.episode_cagrs(
                layout_new, np.asarray(genome_new)[None, :], ep,
                collect=True)
            max_cagr_diff = max(max_cagr_diff,
                                float(np.max(np.abs(co - cn))))
            for (lo, oo, uo), (ln, on, un) in zip(tro, trn):
                if not (np.array_equal(lo, ln)
                        and np.array_equal(oo, on)):
                    decisions_same = False
                max_unit_rel = max(
                    max_unit_rel,
                    float(np.max(np.abs(uo - un)
                                 / np.maximum(uo, 1e-9))))
        return {"pass": bool(decisions_same
                             and max_cagr_diff < 1e-9),
                "decisions_identical": bool(decisions_same),
                "max_abs_cagr_difference": max_cagr_diff,
                "max_unit_rel_difference": max_unit_rel,
                "episodes": int(len(batch["episodes"]))}

    def _all_cagrs(self, layout, genome, batch):
        G = np.asarray(genome)[None, :]
        cs, bhs = [], []
        for ep in batch["episodes"]:
            c, bh = self.episode_cagrs(layout, G, ep)
            cs.append(c[0]); bhs.append(bh)
        return np.array(cs), np.array(bhs)

    def random_baseline(self, layout, rng, n_draws=15):
        scores = []
        for _ in range(n_draws):
            g = rng.normal(0, self.cfg.draw_sigma,
                           layout.n_params)
            scores.append(self.validation_score(layout, g))
        return float(np.median(scores))
