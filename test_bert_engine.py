"""Smoke verification for the parametric BERT growth engine.

Tiny configs only -- NO full-budget run. Exercises, end to
end: layout/default-config shape, batched vs per-member
exactness, chunk invariance, the frozen-prefix cache
exactness, all four growth transplants' preservation, a
tiny stage that beats its random baseline, the growth
sequencer's keep/revert logic, a forced cataclysm with
founder_count = 3, the stock adapter on a synthetic basket,
the parametric stock preprocessor (width, known values,
legacy-exact fourier / wavelet, causality, end-to-end
feature_spec wiring, error handling), the stock data bundle
(builder alignment / assertion reporting, loader validation,
fundamentals round-trip, preprocessor mapping, stats_end
defaulting, episode remapping by name), validity-aware
episode sampling (NaN-hole avoidance, clean-panel identity
with the old uniform sampler, starved-panel error),
the frozen-episode file structure, checkpoint resume
refusal on a mismatched config, and (when jax is
importable) the JAX backend against the numpy reference.

Run: python3 test_bert_engine.py [--jax-python PATH]
Prints one PASS/FAIL line per leg and per-phase timings.
Exit code 0 iff every leg passes (jax leg may SKIP).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be
import bert_domain_stock as stock
import bert_domain_tsp as tsp
import bert_preprocessor as bp
from bert_backend import get_backend
import run_bert_engine as runner

RESULTS = []


def leg(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}",
          flush=True)
    return ok


def tiny_cfg(domain="tsp", **over):
    cfg = runner.smoke_config(domain)
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jax-python", default=None,
                    help="python executable with jax installed, "
                         "for the cross-backend leg")
    args = ap.parse_args()
    backend = get_backend("numpy", "cpu", "x64")
    rng = np.random.default_rng(4242)

    # ------------------------------------------------ layout / defaults
    cfg_def = be.EngineConfig.load(HERE / "bert_engine_config.json")
    arch_def = be.initial_arch(cfg_def, tsp.TSPDomain.input_dim)
    layout_def = be.Layout(arch_def, tsp.head_spec(arch_def.d))
    leg("default config reproduces stage-3 shape (1880 genes)",
        layout_def.n_params == 1880,
        f"(n_params={layout_def.n_params}, arch="
        f"{arch_def.signature()})")

    # ------------------------------------------------ TSP tiny fixtures
    cfg = tiny_cfg("tsp")
    dom = tsp.TSPDomain(cfg, backend)
    arch0 = be.initial_arch(cfg, dom.input_dim)
    layout0 = be.Layout(arch0, tsp.head_spec(arch0.d))
    X = dom.X_val[:16]

    # --------------------------------------- batched vs per-member loop
    G = rng.normal(0, 0.5, (11, layout0.n_params))
    lens_b, _ = dom.evaluate_lengths(layout0, G, X)
    lens_loop = np.stack([dom.evaluate_lengths(
        layout0, G[i][None, :], X)[0][0] for i in range(len(G))])
    leg("batched eval == per-member loop exactly (float64)",
        float(np.max(np.abs(lens_b - lens_loop))) == 0.0,
        f"(max diff {np.max(np.abs(lens_b - lens_loop)):.3e})")

    # ------------------------------------------------ chunk invariance
    s_big = dom.evaluate(layout0, G, {"X": X, "tag": "t"},
                         groups=None, cache=None)
    cfg_chunk = tiny_cfg("tsp", chunk=3)
    dom_c = tsp.TSPDomain(cfg_chunk, backend)
    s_chunk = dom_c.evaluate(layout0, G, {"X": X, "tag": "t"},
                             groups=None, cache=None)
    leg("population chunking does not change results",
        float(np.max(np.abs(s_big - s_chunk))) == 0.0,
        f"(max diff {np.max(np.abs(s_big - s_chunk)):.3e})")

    # ------------------------------------------ frozen-prefix cache
    # Build an add_block-grown architecture: blocks 0..0 are a
    # genuine frozen prefix for the new block's species.
    # (sigma 0.25 for the preservation champ: healthier
    # decoder margins than a saturated random genome)
    champ = rng.normal(0, 0.25, layout0.n_params)
    a1, l1, tmpl1, cl1 = be.make_transplant(
        "add_block", arch0, layout0, champ, tsp.head_spec, cfg,
        tsp.widen_head)
    classes1 = cl1
    pvA = be.prob_vector(classes1, "A", cfg)
    draw_rng = np.random.default_rng(cfg.draw_seed)
    run_rng = np.random.default_rng(cfg.run_seed)
    species = be.create_species(tmpl1, classes1, 3, 5, draw_rng,
                                run_rng, pvA, cfg, lineage=None,
                                init_mode="transplant")
    Gall = np.concatenate([sp["pop"] for sp in species])
    groups = []
    pos = 0
    for sp in species:
        groups.append((pos, pos + len(sp["pop"])))
        pos += len(sp["pop"])
    cache = {}
    s_cached = dom.evaluate(l1, Gall, {"X": X, "tag": "cachetest"},
                            groups=groups, cache=cache)
    s_plain = dom.evaluate(l1, Gall, {"X": X, "tag": "cachetest"},
                           groups=None, cache=None)
    s_cached2 = dom.evaluate(l1, Gall,
                             {"X": X, "tag": "cachetest"},
                             groups=groups, cache=cache)
    leg("frozen-prefix cache == full recompute exactly",
        float(np.max(np.abs(s_cached - s_plain))) == 0.0
        and float(np.max(np.abs(s_cached2 - s_plain))) == 0.0,
        f"(max diff {np.max(np.abs(s_cached - s_plain)):.3e}, "
        f"cache entries {len(cache)})")
    leg("cache actually caches (prefix segments stored)",
        len(cache) > 0, f"({len(cache)} entries)")

    # ------------------------------------------ transplant preservation
    for rule in ("widen", "add_head", "add_block", "stack"):
        a2, l2, tmpl2, cl2 = be.make_transplant(
            rule, arch0, layout0, champ, tsp.head_spec, cfg,
            tsp.widen_head)
        # representative = template + a species draw
        dm = cl2 == be.CLS_DRAW
        rep = tmpl2.copy()
        rep[dm] = rng.normal(0, cfg.draw_sigma, int(dm.sum()))
        pres = dom.preservation(layout0, champ, l2, rep)
        leg(f"transplant preservation exact: {rule}",
            pres["pass"], f"({pres})")

    # ------------------------------------------ permutation invariance
    # the stage program's gate: ENCODER EQUIVARIANCE under
    # relisting (the implementation property; greedy-argmax
    # near-tie flips on a random genome are a separate,
    # documented float phenomenon, not an order leak)
    vec = rng.normal(0, 0.5, layout0.n_params)
    Xi = X[:8]
    Y0 = be.encode_population(backend, layout0, vec[None, :], Xi)
    perm = rng.permutation(Xi.shape[1])
    Yp = be.encode_population(backend, layout0, vec[None, :],
                              Xi[:, perm, :])
    equiv = float(np.max(np.abs(Yp - Y0[:, :, perm, :])))
    leg("permutation invariance audit (encoder equivariance "
        "under relisting)", equiv <= 1e-9,
        f"(max equivariance diff {equiv:.3e})")

    # ------------------------------------------ tiny end-to-end run
    # stage trains + growth trigger + keep/revert, via the
    # runner in-process on a temp dir
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run1"
        cfg_run = tiny_cfg("tsp")
        eng = runner.Engine(cfg_run, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        t_run = time.time()
        res = eng.run()
        wall = time.time() - t_run
        ok_done = (out / "COMPLETE.txt").exists() \
            and (out / "results.json").exists()
        ev_kinds = [e["event"] for e in eng.meta["events"]]
        leg("tiny TSP run completes end to end (sentinel + "
            "results on disk)", ok_done,
            f"({wall:.1f}s, events: {ev_kinds})")
        champ_score = eng.meta["champion"]["val_score"]
        base = dom.random_baseline(layout0, rng)
        leg("tiny stage champion beats its random baseline",
            champ_score > base,
            f"(champion val score {champ_score:.4f} vs random "
            f"baseline median {base:.4f}; note: score is "
            f"negative mean gap)")
        leg("growth trigger fired and a transplant was "
            "attempted", "growth_transplant" in ev_kinds
            or "growth_rejected_preservation" in ev_kinds,
            f"(events: {ev_kinds})")
        leg("keep/revert logic executed",
            ("growth_kept" in ev_kinds)
            or ("growth_reverted" in ev_kinds)
            or ("growth_exhausted" in ev_kinds),
            f"(events: {ev_kinds})")
        timings = eng.meta["phase_timings"]
        print(f"       phase timings: {timings}", flush=True)
        leg("phase A and phase B both ran (timings recorded)",
            len(timings) >= 2
            and all(t["seconds"] >= 0 for t in timings),
            f"({len(timings)} phase records)")

    # ------------------------------------------ checkpoint refusal
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run2"
        cfg_a = tiny_cfg("tsp")
        eng = runner.Engine(cfg_a, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        cfg_bad = tiny_cfg("tsp", mut_sigma=0.99)
        eng2 = runner.Engine(cfg_bad, out)
        refused = False
        try:
            eng2.setup_resume()
        except SystemExit:
            refused = True
        leg("resume refuses a mismatched config", refused)

    # ------------------------------------------ stop-after + resume
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run3"
        cfg_r = tiny_cfg("tsp", phase_a_gens=6, phase_b_gens=6,
                         max_growth_rounds=0)
        eng = runner.Engine(cfg_r, out, stop_after=3)
        eng.setup_fresh()
        eng.save_checkpoint()
        out1 = eng.run()
        stopped = bool(out1.get("stopped_early"))
        eng2 = runner.Engine(cfg_r, out)
        eng2.setup_resume()
        res2 = eng2.run()
        leg("stop-after + resume completes the run",
            stopped and (out / "COMPLETE.txt").exists(),
            f"(stopped_early={stopped})")

    # ------------------------------------------ cataclysm, founders = 3
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run4"
        cfg_c = tiny_cfg("tsp", founder_count=3,
                         cataclysm_every_n_generations=4,
                         phase_a_gens=9, phase_b_gens=9,
                         max_growth_rounds=0)
        eng = runner.Engine(cfg_c, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        # run generations until the cataclysm fires (phase A)
        eng.phase_cache = {}
        eng.elim_series = []
        fired = None
        for _ in range(9):
            eng.gen_cache = {}
            eng.generation("A")
            cats = [e for e in eng.meta["events"]
                    if e["event"] == "cataclysm"]
            if cats:
                fired = cats[-1]
                break
        ok_fire = fired is not None
        leg("forced cataclysm fires on the generation interval",
            ok_fire, f"({fired})")
        if ok_fire:
            fnd = fired["founders"]
            leg("cataclysm founders come from 3 distinct species",
                len({f["species_id"] for f in fnd}) == 3,
                f"(founders: {fnd})")
            lineages = {sp["lineage"] for sp in eng.species}
            want = {f["lineage"] for f in fnd}
            leg("regenerated population carries exactly the 3 "
                "founder lineage ids",
                lineages == want and len(eng.species)
                == cfg_c.n_species,
                f"(lineages {sorted(lineages)}, species "
                f"{len(eng.species)})")
            # founders' validation tours unchanged by the event
            unchanged = True
            vault_files = list((out / "vault").glob(
                "cataclysm_founder_*.npz"))
            for vf in vault_files:
                z = np.load(vf)
                g = z["genome"]
                now = eng.domain.validation_score(eng.layout, g)
                if abs(now - float(z["val_score"])) > 0.0:
                    unchanged = False
            leg("founders vaulted before the event; their "
                "validation scores/tours unchanged by it",
                unchanged and len(vault_files) == 3,
                f"({len(vault_files)} vault files)")

    # ------------------------------------------ stock adapter (synthetic)
    cfg_s = tiny_cfg("stock")
    panel = stock.make_synthetic_panel(30, 500, 20261120)
    dom_s = stock.StockDomain(cfg_s, backend, panel=panel)
    arch_s = be.initial_arch(cfg_s, dom_s.input_dim)
    layout_s = be.Layout(arch_s, stock.head_spec(arch_s.d))
    g0 = rng.normal(0, 0.5, layout_s.n_params)
    batch_s = dom_s.train_batch("B", 0, 0)
    sc = dom_s.evaluate(layout_s, np.tile(g0, (4, 1)), batch_s)
    leg("stock adapter evaluates a population on synthetic "
        "episodes (finite basket-relative fitness)",
        bool(np.all(np.isfinite(sc))), f"(scores {sc})")
    det1 = dom_s.validation_detail(layout_s, g0)
    leg("stock validation detail: median excess over "
        "buy-and-hold computed",
        np.isfinite(det1["median_excess_cagr"])
        and det1["n_episodes"] == len(dom_s.val_episodes()),
        f"({det1})")
    # stock cache/recompute exactness on the encoder path:
    # same genome, fixed episode, cached vs plain episode sim
    ep = dom_s.val_episodes()[0]
    c_plain, bh = dom_s.episode_cagrs(layout_s, g0[None, :], ep,
                                      cache=None)
    cache_s = {}
    c_cached, _ = dom_s.episode_cagrs(layout_s, g0[None, :], ep,
                                      cache=cache_s)
    leg("stock episode sim: cached path == plain path exactly",
        float(np.max(np.abs(c_plain - c_cached))) == 0.0,
        f"(max diff {np.max(np.abs(c_plain - c_cached)):.3e})")
    # stock widen preservation
    a2, l2, tmpl2, cl2 = be.make_transplant(
        "widen", arch_s, layout_s, g0, stock.head_spec, cfg_s,
        stock.widen_head)
    dm = cl2 == be.CLS_DRAW
    rep = tmpl2.copy()
    rep[dm] = rng.normal(0, cfg_s.draw_sigma, int(dm.sum()))
    pres_s = dom_s.preservation(layout_s, g0, l2, rep)
    leg("stock transplant preservation exact: widen",
        pres_s["pass"], f"({pres_s})")
    # frozen episode file structure (no real panel needed)
    frozen_path = (HERE.parent.parent / "trading-agent"
                   / "genetic_lab" / "validation_episodes_v1.json")
    if frozen_path.exists():
        eps = stock.load_frozen_episodes(frozen_path)
        leg("frozen common episode set parses (64 episodes, "
            "basket 10)", len(eps) == 64
            and all(len(e["basket"]) == 10 for e in eps),
            f"({len(eps)} episodes)")

    # ------------------------------------------ stock preprocessor
    FULL_SPEC = [
        {"kind": "fourier", "k": 4, "window": 60},
        {"kind": "volatility", "window": 60},
        {"kind": "wavelet", "window": 60, "levels": 3},
        {"kind": "rsi", "window": 14},
        {"kind": "atr", "window": 14},
        {"kind": "chaikin", "window": 20},
        {"kind": "stochastic", "window": 14, "smooth": 3},
        {"kind": "ema", "window": 20},
        {"kind": "sma", "window": 50},
        {"kind": "obv", "window": 20},
        {"kind": "candlestick", "window": 10},
    ]

    def _ohlcv(closes, volume=1000.0):
        closes = np.asarray(closes, dtype=float)

        class _P:
            pass
        p = _P()
        p.open = closes.copy()
        p.high = closes.copy()
        p.low = closes.copy()
        p.close = closes.copy()
        p.volume = np.full(closes.shape, volume)
        p.prices = closes.copy()
        p.tickers = [f"T{i}" for i in range(closes.shape[0])]
        return p

    leg("preprocessor width: full menu spec == 30, from "
        "per-kind widths",
        bp.feature_width(FULL_SPEC) == 30
        and [bp.feature_width([b]) for b in FULL_SPEC]
        == [8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 6],
        f"(width {bp.feature_width(FULL_SPEC)})")

    # constant OHLCV: every feature's raw value is a constant
    # (0 for the whole menu), so standardised output is the
    # neutral 0 wherever defined -- and NaN -> 0 elsewhere.
    # (The EMA recursion leaves ~1e-16 float dust on a
    # constant series; the check is at 1e-12.)
    p_const = _ohlcv(np.full((3, 150), 50.0))
    f_const, _ = bp.build_features(p_const, FULL_SPEC)
    leg("preprocessor constant series: all features 0 after "
        "standardisation",
        f_const.shape == (3, 150, 30)
        and float(np.abs(f_const).max()) < 1e-12,
        f"(max abs {np.abs(f_const).max():.3e})")

    # monotonic closes: Wilder RSI pins at 100 (rising) / 0
    # (falling); pooled standardisation of {-1, +1} leaves
    # the rising ticker at +1 on the last day
    tt = np.arange(200)
    p_mono = _ohlcv(np.stack([100 * 1.01 ** tt,
                              100 * 0.99 ** tt]), volume=1e6)
    f_rsi, _ = bp.build_features(p_mono,
                                 [{"kind": "rsi", "window": 14}])
    leg("preprocessor RSI: monotonic rise reads > 0.99 on "
        "the last day",
        float(f_rsi[0, -1, 0]) > 0.99
        and float(f_rsi[1, -1, 0]) < -0.99,
        f"(rising {f_rsi[0, -1, 0]:.4f}, falling "
        f"{f_rsi[1, -1, 0]:.4f})")

    # fourier + volatility against an independent in-test
    # computation of the legacy definitions
    rng_pp = np.random.default_rng(777)
    cl_pp = 100.0 * np.exp(np.cumsum(
        rng_pp.normal(0, 0.02, (2, 120)), axis=1))
    p_pp = _ohlcv(cl_pp)
    raw_pp = bp.raw_features(p_pp, [
        {"kind": "fourier", "k": 4, "window": 60},
        {"kind": "volatility", "window": 60}])
    a_i, t_i, w_i, k_i = 1, 100, 60, 4
    win_i = cl_pp[a_i, t_i - w_i + 1: t_i + 1]
    rel_i = win_i[::-1] / cl_pp[a_i, t_i] - 1.0
    z_i = np.fft.rfft(rel_i)
    exp_f = np.concatenate([z_i[:k_i].real, z_i[:k_i].imag])
    d_f = float(np.max(np.abs(raw_pp[a_i, t_i, :8] - exp_f)))
    leg("preprocessor fourier matches an independent rfft "
        "computation", d_f < 1e-12, f"(max diff {d_f:.3e})")
    daily_i = win_i[::-1][:-1] / win_i[::-1][1:] - 1.0
    d_v = float(abs(raw_pp[a_i, t_i, 8]
                    - daily_i.std(ddof=1)))
    leg("preprocessor volatility matches an independent "
        "std(ddof=1) computation", d_v < 1e-12,
        f"(max diff {d_v:.3e})")

    # wavelet against a hand-computed Haar recursion on a
    # crafted series (closes 1..16; at t=7 the relative
    # series is exactly [0, -1/8, ..., -7/8])
    p_wv = _ohlcv(np.arange(1, 17, dtype=float)[None, :],
                  volume=1.0)
    raw_wv = bp.raw_features(p_wv, [{"kind": "wavelet",
                                     "window": 8, "levels": 2}])

    def _haar_hand(x, levels):
        a = np.array(x, dtype=float)
        for _ in range(levels):
            if len(a) % 2 == 1:
                a = np.concatenate([a, a[:1]])
            a = (a[0::2] + a[1::2]) / np.sqrt(2.0)
        return a

    exp_w = _haar_hand([0, -1 / 8, -2 / 8, -3 / 8, -4 / 8,
                        -5 / 8, -6 / 8, -7 / 8], 2)
    d_w = float(np.max(np.abs(raw_wv[0, 7] - exp_w)))
    leg("preprocessor wavelet matches a hand-computed Haar "
        "recursion", raw_wv.shape == (1, 16, 2) and d_w < 1e-12,
        f"(max diff {d_w:.3e}, expected {exp_w})")

    # causality: scaling every bar after t0 must not move
    # any feature at days <= t0 (stats window ends at t0)
    panel_c = stock.make_synthetic_ohlcv_panel(6, 200, 99)
    t0 = 100
    f_c1, _ = bp.build_features(panel_c, FULL_SPEC,
                                stats_end=t0 + 1)
    for nm in ("open", "high", "low", "close", "volume"):
        getattr(panel_c, nm)[:, t0 + 1:] *= 3.0
    f_c2, _ = bp.build_features(panel_c, FULL_SPEC,
                                stats_end=t0 + 1)
    leg("preprocessor causality: bars after t0 do not move "
        "features at days <= t0",
        bool(np.array_equal(f_c1[:, :t0 + 1],
                            f_c2[:, :t0 + 1]))
        and not np.array_equal(f_c1[:, t0 + 1:],
                               f_c2[:, t0 + 1:]))

    # end to end through the stock domain: full spec gives
    # F = 30 features, input_dim 31, finite fitness
    cfg_fs = tiny_cfg("stock")
    cfg_fs.domain_config = {**cfg_fs.domain_config,
                            "feature_spec": FULL_SPEC}
    panel_o = stock.make_synthetic_ohlcv_panel(30, 500,
                                               20261120)
    dom_o = stock.StockDomain(cfg_fs, backend, panel=panel_o)
    arch_o = be.initial_arch(cfg_fs, dom_o.input_dim)
    layout_o = be.Layout(arch_o, stock.head_spec(arch_o.d))
    g_o = rng.normal(0, 0.5, layout_o.n_params)
    sc_o = dom_o.evaluate(layout_o, np.tile(g_o, (4, 1)),
                          dom_o.train_batch("B", 0, 0))
    leg("stock domain with the full feature_spec: "
        "input_dim == 31 and finite fitness",
        dom_o.input_dim == 31
        and bool(np.all(np.isfinite(sc_o))),
        f"(input_dim {dom_o.input_dim}, scores {sc_o})")

    # plus two fundamentals: F = 32, input_dim 33
    cfg_ff = tiny_cfg("stock")
    cfg_ff.domain_config = {
        **cfg_ff.domain_config,
        "feature_spec": FULL_SPEC + [
            {"kind": "fundamental", "name": "earnings_yield"},
            {"kind": "fundamental", "name": "book_to_market"}]}
    dom_ff = stock.StockDomain(cfg_ff, backend, panel=panel_o)
    arch_ff = be.initial_arch(cfg_ff, dom_ff.input_dim)
    layout_ff = be.Layout(arch_ff, stock.head_spec(arch_ff.d))
    g_ff = rng.normal(0, 0.5, layout_ff.n_params)
    sc_ff = dom_ff.evaluate(layout_ff, np.tile(g_ff, (4, 1)),
                            dom_ff.train_batch("B", 0, 0))
    leg("stock domain with feature_spec + 2 fundamentals: "
        "input_dim == 33 and finite fitness",
        dom_ff.input_dim == 33
        and bool(np.all(np.isfinite(sc_ff))),
        f"(input_dim {dom_ff.input_dim}, scores {sc_ff})")

    # error handling
    def _raises_valueerror(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    leg("preprocessor rejects an unknown feature kind",
        _raises_valueerror(
            lambda: bp.build_features(p_const,
                                      [{"kind": "bogus"}]))
        and _raises_valueerror(
            lambda: bp.feature_width([{"kind": "bogus"}])))
    leg("preprocessor rejects a missing fundamental",
        _raises_valueerror(
            lambda: bp.build_features(
                p_const, [{"kind": "fundamental",
                           "name": "earnings_yield"}])))

    # NaN policy: a feature is NaN at t when an input it
    # needs is NaN inside its window (or anywhere in the
    # prefix, for the recursive features); features that do
    # not need the NaN input stay finite
    p_nan = stock.make_synthetic_ohlcv_panel(2, 160, 5)
    p_nan.high[0, 80] = np.nan
    raw_nan = bp.raw_features(p_nan, FULL_SPEC)
    blk = {k: (s0, s1) for k, s0, s1 in
           bp.build_features(p_nan, FULL_SPEC)[1]["blocks"]}
    ch_nan = raw_nan[0, 85, blk["chaikin"][0]]
    st_nan = raw_nan[0, 85, blk["stochastic"][0]]
    sm_ok = raw_nan[0, 85, blk["sma"][0]]
    p_nan.close[0, 90] = np.nan
    raw_nan2 = bp.raw_features(p_nan, FULL_SPEC)
    fo_nan = raw_nan2[0, 100, blk["fourier"][0]]
    rs_nan = raw_nan2[0, 100, blk["rsi"][0]]
    leg("preprocessor NaN policy: NaN input poisons only "
        "the features that need it",
        np.isnan(ch_nan) and np.isnan(st_nan)
        and np.isfinite(sm_ok) and np.isnan(fo_nan)
        and np.isnan(rs_nan),
        f"(chaikin {ch_nan}, stochastic {st_nan}, sma "
        f"{sm_ok:.3f}, fourier {fo_nan}, rsi {rs_nan})")

    # ------------------------------------------ stock data bundle
    import datetime as _dt

    import bert_stock_data as bsd
    import build_stock_bundle as bsb

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pdir, vdir = td / "prices", td / "vol"
        pdir.mkdir()
        vdir.mkdir()
        n_mini = 150
        dates_mini = [(_dt.date(2019, 1, 2)
                       + _dt.timedelta(days=i)).isoformat()
                      for i in range(n_mini)]
        rng_mini = np.random.default_rng(11)

        def _write_prices(tk, idxs):
            closes = 100.0 * np.exp(np.cumsum(
                rng_mini.normal(0, 0.01, len(idxs))))
            lines = ["date,open,high,low,close"]
            for j, i in enumerate(idxs):
                c = closes[j]
                lines.append(
                    f"{dates_mini[i]},{c * 0.999:.6f},"
                    f"{c * 1.001:.6f},{c * 0.998:.6f},{c:.6f}")
            (pdir / f"{tk}.csv").write_text(
                "\n".join(lines) + "\n")

        def _write_volume(tk, idxs):
            lines = ["date,volume"]
            for i in idxs:
                lines.append(f"{dates_mini[i]},1000000")
            (vdir / f"{tk}.csv").write_text(
                "\n".join(lines) + "\n")

        all_idx = list(range(n_mini))
        _write_prices("SPY", all_idx)
        _write_volume("SPY", all_idx)
        # AAA: day 40 missing from prices AND volume
        aaa_idx = [i for i in all_idx if i != 40]
        _write_prices("AAA", aaa_idx)
        _write_volume("AAA", aaa_idx)
        # BBB: prices start day 10, volume starts day 5 ->
        # 5 volume dates with no price date (reported
        # violation, build continues)
        _write_prices("BBB", all_idx[10:])
        _write_volume("BBB", all_idx[5:])
        # OLD: stale, ends at day 100
        _write_prices("OLD", all_idx[:101])
        _write_volume("OLD", all_idx[:101])
        # ^VIX: full prices, no volume file at all
        _write_prices("^VIX", all_idx)

        out_npz = td / "mini_bundle.npz"
        rep_mini = bsb.build_bundle(
            pdir, vdir, out_npz, include_fundamentals=False)
        panel_mini = bsd.load_bundle_panel(out_npz)
        rows = {t: i for i, t in enumerate(panel_mini.tickers)}
        leg("mini bundle round trip: shapes, calendar, "
            "ticker order",
            panel_mini.close.shape == (5, n_mini)
            and panel_mini.tickers == ["AAA", "BBB", "OLD",
                                        "SPY", "^VIX"]
            and panel_mini.dates == dates_mini
            and panel_mini.train_end_index == n_mini
            and rep_mini["size_bytes"] > 0,
            f"(shape {panel_mini.close.shape}, "
            f"train_end {panel_mini.train_end_index})")
        leg("mini bundle alignment: missing day -> NaN, "
            "stale tail -> NaN, ^VIX volume all NaN",
            np.isnan(panel_mini.close[rows["AAA"], 40])
            and np.isfinite(panel_mini.close[rows["AAA"], 41])
            and np.isfinite(panel_mini.close[rows["OLD"], 100])
            and bool(np.isnan(
                panel_mini.close[rows["OLD"], 101:]).all())
            and bool(np.isnan(
                panel_mini.volume[rows["^VIX"]]).all())
            and bool(np.isfinite(
                panel_mini.volume[rows["SPY"]]).all()),
            f"(stale {[s['ticker'] for s in rep_mini['stale_price']]}, "
            f"no-volume {rep_mini['no_volume_file']})")
        leg("mini bundle report: volume-date violation "
            "reported, not silently passed",
            rep_mini["volume_date_violations"] == [
                {"ticker": "BBB", "n_bad": 5,
                 "first_bad": dates_mini[5]}]
            and [s["ticker"] for s in rep_mini["stale_price"]]
            == ["OLD"],
            f"({rep_mini['volume_date_violations']})")

        # fundamentals round-trip through the loader: add a
        # fund_ array to a copy of the bundle and read it back
        z = np.load(out_npz)
        payload = {k: z[k] for k in z.files}
        ey = np.full((5, n_mini), 0.05, dtype=np.float32)
        payload["fund_earnings_yield"] = ey
        fund_npz = td / "mini_bundle_fund.npz"
        np.savez_compressed(fund_npz, **payload)
        panel_f = bsd.load_bundle_panel(fund_npz)
        leg("bundle fundamentals dict round-trips",
            set(panel_f.fundamentals) == {"earnings_yield"}
            and bool(np.array_equal(
                panel_f.fundamentals["earnings_yield"], ey))
            and panel_mini.fundamentals == {})

        # loader validation: duplicate tickers and
        # non-increasing dates are clear errors
        def _bad_npz(**over):
            base = {"open": np.ones((2, 3), np.float32),
                    "high": np.ones((2, 3), np.float32),
                    "low": np.ones((2, 3), np.float32),
                    "close": np.ones((2, 3), np.float32),
                    "volume": np.ones((2, 3), np.float32),
                    "tickers": np.array(["A", "B"]),
                    "dates": np.array(["2020-01-01",
                                       "2020-01-02",
                                       "2020-01-03"]),
                    "train_end_index": np.array(3)}
            base.update(over)
            p = td / "bad.npz"
            np.savez(p, **base)
            return p

        leg("bundle loader rejects duplicate tickers and "
            "non-increasing dates",
            _raises_valueerror(lambda: bsd.load_bundle_panel(
                _bad_npz(tickers=np.array(["A", "A"]))))
            and _raises_valueerror(
                lambda: bsd.load_bundle_panel(_bad_npz(
                    dates=np.array(["2020-01-01",
                                    "2020-01-01",
                                    "2020-01-03"])))))

        # mapping leg: the preprocessor consumes the loaded
        # panel; width follows the spec and the raw features
        # are defined (finite) once the warmup has passed
        spec_mini = [{"kind": "rsi", "window": 5},
                     {"kind": "obv", "window": 3},
                     {"kind": "fourier", "k": 2, "window": 10}]
        feats_mini, stats_mini = bp.build_features(
            panel_mini, spec_mini,
            stats_end=panel_mini.train_end_index)
        raw_mini = bp.raw_features(panel_mini, spec_mini)
        leg("bundle panel -> preprocessor: width == "
            "feature_width(spec), finite after warmup",
            feats_mini.shape == (5, n_mini,
                                 bp.feature_width(spec_mini))
            and bool(np.isfinite(feats_mini).all())
            and bool(np.isfinite(
                raw_mini[rows["SPY"], -1]).all()),
            f"(width {feats_mini.shape[2]}, stats_end "
            f"{stats_mini['stats_end']})")

        # domain wiring: a bundle panel's train_end_index is
        # the default stats_end; an explicit one wins
        cfg_b = tiny_cfg("stock")
        cfg_b.domain_config = {**cfg_b.domain_config,
                               "feature_spec": spec_mini}
        dom_b = stock.StockDomain(cfg_b, backend,
                                  panel=panel_mini)
        cfg_b2 = tiny_cfg("stock")
        cfg_b2.domain_config = {**cfg_b.domain_config,
                                "stats_end": 50}
        dom_b2 = stock.StockDomain(cfg_b2, backend,
                                   panel=panel_mini)
        leg("bundle panel: stats_end defaults to "
            "train_end_index, explicit stats_end wins",
            dom_b.feat_stats["stats_end"] == n_mini
            and dom_b2.feat_stats["stats_end"] == 50
            and dom_b.input_dim
            == bp.feature_width(spec_mini) + 1,
            f"(default {dom_b.feat_stats['stats_end']}, "
            f"explicit {dom_b2.feat_stats['stats_end']})")

        # frozen-episode mapping: baskets resolve by NAME
        eps_mini = [{"basket": np.array([0, 1]),
                     "tickers": ["SPY", "AAA"], "start": 100}]
        remapped = bsd.remap_episodes(eps_mini, panel_mini,
                                      episode_days=50)
        rep_eps = bsd.check_episodes(eps_mini, panel_mini,
                                     episode_days=50)
        leg("episode remap resolves baskets by ticker name",
            list(remapped[0]["basket"])
            == [rows["SPY"], rows["AAA"]] and rep_eps["ok"]
            and not bsd.check_episodes(
                [{"basket": np.array([0]),
                  "tickers": ["NOPE"], "start": 0}],
                panel_mini, episode_days=50)["ok"])

    # --------------------------- validity-aware episode sampling
    # NaN-hole panel: sampled train / elimination episodes
    # must never touch a NaN price inside [start,
    # start + episode_days] (decision days + mark day)
    panel_h = stock.make_synthetic_panel(30, 800, 20261121)
    panel_h.prices[3, 100:201] = np.nan      # a NaN block
    panel_h.prices[7, 500:] = np.nan         # a stale tail
    panel_h.prices[11, ::37] = np.nan        # scattered holes
    cfg_h = tiny_cfg("stock")
    cfg_h.domain_config = {**cfg_h.domain_config,
                           "episode_days": 60, "basket_n": 10,
                           "n_train_episodes": 3,
                           "n_elim_episodes": 4}
    dom_h = stock.StockDomain(cfg_h, backend, panel=panel_h)

    def _ep_valid(ep, prices, days):
        s = ep["start"]
        return bool(np.isfinite(
            prices[ep["basket"], s:s + days + 1]).all())

    hole_ok = True
    n_drawn = 0
    for seed_h in (1, 2, 3):
        for ep in dom_h._sample_episodes(
                np.random.default_rng(seed_h), 30):
            hole_ok &= _ep_valid(ep, panel_h.prices, 60)
            n_drawn += 1
    for ep in (dom_h.elim_episodes() + dom_h.phase_a_episodes()
               + dom_h.val_episodes()
               + dom_h.train_batch("B", 0, 0)["episodes"]):
        hole_ok &= _ep_valid(ep, panel_h.prices, 60)
        n_drawn += 1
    # the precomputed structure itself: ticker 3's block is
    # days 100..200, so start 39 (window 39..99) is its last
    # clean start before the block and start 201 its first
    # after; ticker 7's tail starts at day 500, so start 440
    # (window 440..500) is the first start it poisons
    struct_ok = bool(
        dom_h._valid_start[3, 39]
        and not dom_h._valid_start[3, 40]
        and not dom_h._valid_start[3, 200]
        and dom_h._valid_start[3, 201]
        and dom_h._valid_start[7, 439]
        and not dom_h._valid_start[7, 440])
    leg("validity-aware sampling: no sampled episode "
        "touches a NaN price (block, tail, scattered holes)",
        hole_ok and struct_ok and n_drawn > 100,
        f"({n_drawn} episodes checked, structure pins "
        "mark-day-inclusive windows)")
    # and the point of it: fitness on sampled episodes of a
    # NaN panel is finite (no NaN buy-and-hold leg)
    arch_h = be.initial_arch(cfg_h, dom_h.input_dim)
    layout_h = be.Layout(arch_h, stock.head_spec(arch_h.d))
    g_h = rng.normal(0, 0.5, layout_h.n_params)
    sc_h = dom_h.evaluate(layout_h, np.tile(g_h, (2, 1)),
                          dom_h.train_batch("A", 0, 0))
    leg("validity-aware sampling: fitness finite on a "
        "NaN panel's sampled episodes",
        bool(np.all(np.isfinite(sc_h))), f"(scores {sc_h})")

    # clean-panel identity: on a NaN-free panel the sampler
    # is exactly the old uniform sampler (same RNG stream,
    # same episodes, same order)
    def _old_uniform_sample(rng_o, panel_o, days_o, bn_o, k_o):
        A_o, T_o = panel_o.prices.shape
        out_o = []
        for _ in range(k_o):
            s_o = int(rng_o.integers(0, T_o - days_o))
            b_o = np.sort(rng_o.choice(A_o, size=bn_o,
                                       replace=False))
            out_o.append({"basket": b_o,
                          "tickers": [panel_o.tickers[i]
                                      for i in b_o],
                          "start": s_o})
        return out_o

    def _eps_equal(ea, eb):
        return len(ea) == len(eb) and all(
            np.array_equal(x["basket"], y["basket"])
            and x["start"] == y["start"]
            and x["tickers"] == y["tickers"]
            for x, y in zip(ea, eb))

    panel_id = stock.make_synthetic_panel(30, 500, 20261120)
    cfg_id = tiny_cfg("stock")
    dom_id = stock.StockDomain(cfg_id, backend, panel=panel_id)
    days_id = dom_id.episode_days
    bn_id = dom_id.basket_n
    ident_ok = True
    for seed_id in (777, 20261120):
        ident_ok &= _eps_equal(
            dom_id._sample_episodes(
                np.random.default_rng(seed_id), 12),
            _old_uniform_sample(np.random.default_rng(seed_id),
                                panel_id, days_id, bn_id, 12))
    ident_ok &= _eps_equal(
        dom_id.elim_episodes(),
        _old_uniform_sample(
            np.random.default_rng(cfg_id.elim_seed), panel_id,
            days_id, bn_id, dom_id.n_elim_episodes))
    ident_ok &= _eps_equal(
        dom_id.phase_a_episodes(),
        _old_uniform_sample(
            np.random.default_rng(cfg_id.phase_a_seed),
            panel_id, days_id, bn_id, dom_id.n_train_episodes))
    leg("validity-aware sampling: clean panel reproduces "
        "the old uniform sampler exactly (seeds, elim, "
        "phase A)", ident_ok)

    # starved panel: fewer clean tickers than the basket
    # size at every start -> a clear error, immediately
    panel_st = stock.make_synthetic_panel(12, 200, 7)
    panel_st.prices[0:3, :] = np.nan
    cfg_st = tiny_cfg("stock")
    cfg_st.domain_config = {**cfg_st.domain_config,
                            "episode_days": 60, "basket_n": 10}
    dom_st = stock.StockDomain(cfg_st, backend, panel=panel_st)
    starved_msg = None
    try:
        dom_st._sample_episodes(np.random.default_rng(0), 1)
    except RuntimeError as e:
        starved_msg = str(e)
    leg("validity-aware sampling: starved panel raises a "
        "clear error (no valid episode exists)",
        starved_msg is not None
        and "no valid episode" in starved_msg,
        f"({starved_msg})" if starved_msg else "(no error)")

    # ------------------------------------------ jax backend leg
    if args.jax_python:
        code = (
            "import sys; sys.path.insert(0, r'%s');"
            "import numpy as np, bert_engine as be,"
            " bert_domain_tsp as tsp;"
            "from bert_backend import get_backend;"
            "b = get_backend('jax', 'cpu', 'x64');"
            "print('fell_back', b.fell_back, b.describe());"
            "cfg = be.EngineConfig.load(r'%s');"
            "cfg.backend='jax';"
            "dom = tsp.TSPDomain(cfg, b);"
            "arch = be.initial_arch(cfg, dom.input_dim);"
            "lay = be.Layout(arch, tsp.head_spec(arch.d));"
            "rng = np.random.default_rng(3);"
            "G = rng.normal(0, 0.5, (5, lay.n_params));"
            "lens, _ = dom.evaluate_lengths(lay, G, dom.X_val[:8]);"
            "np.save(r'/tmp/jax_lens.npy', lens);"
            "print('jax ok', lens.shape)"
            % (str(HERE), str(HERE / "bert_engine_config.json")))
        r = subprocess.run([args.jax_python, "-c", code],
                           capture_output=True, text=True,
                           timeout=600)
        if r.returncode != 0:
            leg("jax backend runs the same code on CPU", False,
                f"(stderr tail: {r.stderr[-400:]})")
        else:
            domn = tsp.TSPDomain(cfg_def, backend)
            archn = be.initial_arch(cfg_def, domn.input_dim)
            layn = be.Layout(archn, tsp.head_spec(archn.d))
            rng3 = np.random.default_rng(3)
            G3 = rng3.normal(0, 0.5, (5, layn.n_params))
            lens_n, _ = domn.evaluate_lengths(layn, G3,
                                              domn.X_val[:8])
            lens_j = np.load("/tmp/jax_lens.npy")
            d = float(np.max(np.abs(lens_n - lens_j)))
            leg("jax backend runs the same code on CPU; x64 "
                "results match numpy reference",
                d < 1e-9, f"(max diff {d:.3e}; {r.stdout.strip()})")
    else:
        print("[SKIP] jax backend leg (no --jax-python given)",
              flush=True)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_pass}/{len(RESULTS)} legs passed ====",
          flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
