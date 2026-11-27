"""Tests for bert_monitor.py (Addendum 3 live-run monitor).

Runnable as: python3 test_bert_monitor.py
Exit 0 iff every leg passes. Every leg prints PASS/FAIL.

Leg 7 of the task brief (the detailed episode simulation
reproduces today's final CAGRs exactly) already lives in
the core suite (test_bert_engine.py, "detailed episode sim
reproduces today's final CAGRs exactly") and is NOT
duplicated here — it is noted as covered there.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/hatch/workspace/trading-agent")

import bert_engine as be
import bert_domain_stock as stock
import bert_monitor as bm
from bert_backend import get_backend
from trading_agent.metrics import compute_sqn

RESULTS = []


def leg(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}",
          flush=True)


def _smoke_domain():
    """Stock smoke domain + layout + fixed genome, as the
    core suite builds them (runner.smoke_config('stock'))."""
    import run_bert_engine as runner
    cfg = runner.smoke_config("stock")
    backend = get_backend("numpy", "cpu", "x64")
    panel = stock.make_synthetic_panel(30, 500, 20261120)
    dom = stock.StockDomain(cfg, backend, panel=panel)
    arch = be.initial_arch(cfg, dom.input_dim)
    layout = be.Layout(arch, stock.head_spec(arch.d))
    rng = np.random.default_rng(20261122)
    genome = rng.normal(0, 0.5, layout.n_params)
    return cfg, dom, layout, genome, panel


def _detailed(dom, layout, genome, ep):
    _c, _bh, det = dom.episode_cagrs(
        layout, np.asarray(genome)[None, :], ep, detailed=True)
    return det


def _weights_df(det, ep, panel):
    """Weights + prices DataFrames for the REAL compute_sqn,
    over the panel history 0..start+days (so compute_sqn's
    ATR sees the same pre-episode history as sqn_from_trace).
    weights row r holds the weight that is HELD on day r+1
    (the shift(1) convention): w[start+t-1] = held weight
    on day start+t."""
    basket = np.asarray(ep["basket"])
    start = int(ep["start"])
    days = det["prices"].shape[1] - 1
    end_row = start + days + 1
    prices = panel.prices[basket][:, :end_row].T  # (rows, N)
    idx = list(range(end_row))
    cols = list(ep["tickers"])
    prices_df = pd.DataFrame(prices, index=idx, columns=cols)
    w = np.zeros((end_row, len(basket)))
    for t in range(1, days + 1):
        v = float(det["v_pre"][t]) if t < days \
            else float(det["v_final"])
        w[start + t - 1] = det["shares_end"][t - 1] \
            * det["prices"][:, t] / v
    weights_df = pd.DataFrame(w, index=idx, columns=cols)
    return weights_df, prices_df


def main():
    cfg, dom, layout, genome, panel = _smoke_domain()
    days = int(dom.episode_days)
    ep = dom.val_episodes()[0]
    det = _detailed(dom, layout, genome, ep)
    path = np.asarray(det["value_path"], dtype=float)

    # ---- leg 1: Sharpe / drawdown hand-check (independent
    # plain-numpy calculation, not the module functions)
    rets = path[1:] / path[:-1] - 1.0
    sharpe_ind = float(rets.mean() / rets.std(ddof=1)
                       * np.sqrt(252))
    peak = np.maximum.accumulate(path)
    dd_ind = float(np.min(path / peak - 1.0))
    sharpe_mod, n_rets = bm.sharpe_from_path(path)
    dd_mod = bm.max_drawdown_from_path(path)
    leg("1. Sharpe and max drawdown match an independent "
        "calculation from the recorded value path",
        sharpe_mod is not None
        and abs(sharpe_mod - sharpe_ind) < 1e-12
        and abs(dd_mod - dd_ind) < 1e-12
        and n_rets == len(rets),
        f"(Sharpe module {sharpe_mod:.6f} vs independent "
        f"{sharpe_ind:.6f}; drawdown {dd_mod:.6f} vs "
        f"{dd_ind:.6f}; {n_rets} returns)")

    # ---- leg 2: trade counter hand-check
    # (a) fills reconcile with holdings and turnover: replay
    # the fills day by day and compare with shares_end;
    # fill value == shares_delta * that day's price.
    shares = np.zeros(len(ep["basket"]))
    ok_fills = True
    for d in range(days):
        for (j, side, dshares, value) in det["fills"][d]:
            price = det["prices"][j, d]
            # sells record a NEGATIVE shares_delta; value is
            # the positive dollar amount either way
            if abs(abs(value) - abs(dshares) * price) > 1e-9:
                ok_fills = False
            shares[j] += dshares
        if not np.allclose(shares, det["shares_end"][d],
                           atol=1e-9):
            ok_fills = False
    turnover_ind = sum(f[3] for day_f in det["fills"]
                       for f in day_f)
    ok_fills = ok_fills and abs(
        turnover_ind - det["turnover"]) < 1e-9
    # (b) hungry genome: levels all 5, unit ~max. Most
    # ticker-days' buys must be unaffordable and NOT counted.
    hungry = np.zeros(layout.n_params)
    hungry[layout.sl("c0.head.b_score")] = 10.0
    hungry[layout.sl("c0.head.b_unit")] = 10.0
    det_h = _detailed(dom, layout, hungry, ep)
    all5 = bool(np.all(det_h["levels"] == 5))
    unit_max = bool(np.all(det_h["unit"] > 5.9))
    ticker_days = days * len(ep["basket"])
    hungry_ok = all5 and unit_max and det_h["n_buys"] < 0.5 \
        * ticker_days
    leg("2. trade counter: fills reconcile with holdings "
        "and turnover; hungry genome (levels all 5, max "
        "unit) fills far fewer buys than ticker-days",
        ok_fills and hungry_ok,
        f"(fills reconcile={ok_fills}; hungry levels all 5="
        f"{all5}, unit>5.9={unit_max}, buys filled "
        f"{det_h['n_buys']} of {ticker_days} ticker-days)")

    # ---- leg 3: SQN vs the REAL compute_sqn, proxy path
    # and true-range (high/low) path
    weights_df, prices_df = _weights_df(det, ep, panel)
    ref = compute_sqn(weights_df, prices_df)
    mine = bm.sqn_from_trace(det, ep, panel)

    def _close(a, b, tol=1e-9):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(a - b) < tol

    ok3 = _close(mine["sqn"], ref["sqn"]) \
        and mine["n_trades"] == ref["n_trades"] \
        and _close(mine["mean_r"], ref["mean_r"]) \
        and _close(mine["sd_r"], ref["sd_r"])
    # true-range path: same panel with fabricated highs/lows
    rng_hl = np.random.default_rng(5)
    panel_hl = stock.make_synthetic_panel(30, 500, 20261120)
    spread = rng_hl.uniform(0.002, 0.01, panel_hl.prices.shape)
    panel_hl.high = panel_hl.prices * (1.0 + spread)
    panel_hl.low = panel_hl.prices * (1.0 - spread)
    panel_hl.close = panel_hl.prices
    dom_hl = stock.StockDomain(cfg, get_backend("numpy", "cpu",
                                                "x64"),
                               panel=panel_hl)
    det_hl = _detailed(dom_hl, layout, genome, ep)
    w_hl, p_hl = _weights_df(det_hl, ep, panel_hl)
    h_df = pd.DataFrame(panel_hl.high[
        np.asarray(ep["basket"])][:, :ep["start"] + days + 1].T,
        index=list(range(ep["start"] + days + 1)),
        columns=list(ep["tickers"]))
    l_df = pd.DataFrame(panel_hl.low[
        np.asarray(ep["basket"])][:, :ep["start"] + days + 1].T,
        index=list(range(ep["start"] + days + 1)),
        columns=list(ep["tickers"]))
    ref_hl = compute_sqn(w_hl, p_hl, h_df, l_df)
    mine_hl = bm.sqn_from_trace(det_hl, ep, panel_hl)
    ok3_hl = _close(mine_hl["sqn"], ref_hl["sqn"]) \
        and mine_hl["n_trades"] == ref_hl["n_trades"] \
        and _close(mine_hl["mean_r"], ref_hl["mean_r"]) \
        and _close(mine_hl["sd_r"], ref_hl["sd_r"])
    # B&H SQN is null, never a fake zero
    ep_days = dict(ep)
    ep_days["days"] = days
    bh = bm.buyhold_stats(ep_days, panel)
    ok_bh = bh["sqn"]["sqn"] is None \
        and bh["sqn"]["n_trades"] == 0
    leg("3. SQN matches the real compute_sqn on a fixed "
        "trace (close proxy and true-range paths); B&H SQN "
        "is null with N=0",
        ok3 and ok3_hl and ok_bh,
        f"(proxy: mine sqn={mine['sqn']} N={mine['n_trades']} "
        f"vs ref sqn={ref['sqn']} N={ref['n_trades']}; "
        f"true-range: mine sqn={mine_hl['sqn']} "
        f"N={mine_hl['n_trades']} vs ref sqn={ref_hl['sqn']} "
        f"N={ref_hl['n_trades']}; B&H sqn={bh['sqn']['sqn']} "
        f"N={bh['sqn']['n_trades']})")

    # ---- leg 4: random ladder
    eps = dom.val_episodes()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    meds_a, unit_a = bm.random_ladder(
        dom, layout, eps, 40.0, rng_a, n_draws=5)
    meds_b, unit_b = bm.random_ladder(
        dom, layout, eps, 40.0, rng_b, n_draws=5)
    n = len(eps[0]["basket"])
    unit_exp = min(6.0, 40.0 / (n * 2.5))
    _m, unit_floor = bm.random_ladder(
        dom, layout, eps, 0.1, np.random.default_rng(1),
        n_draws=2)
    _m, unit_cap = bm.random_ladder(
        dom, layout, eps, 200.0, np.random.default_rng(1),
        n_draws=2)
    champ_med = float(np.median([
        float(dom.episode_cagrs(layout, genome[None, :], e)[0][0])
        for e in eps]))
    pct = bm.champion_percentile(champ_med, meds_a)
    leg("4. random ladder: same seed -> bit-identical "
        "medians; unit follows min(6, target/(n*2.5)) "
        "floored at 0.05; finite medians; percentile in "
        "[0, 100]",
        np.array_equal(meds_a, meds_b)
        and abs(unit_a - unit_exp) < 1e-12
        and abs(unit_floor - 0.05) < 1e-12
        and abs(unit_cap - 6.0) < 1e-12
        and np.all(np.isfinite(meds_a))
        and 0.0 <= pct <= 100.0,
        f"(unit={unit_a:.4f} expected {unit_exp:.4f}; "
        f"floored unit={unit_floor}; capped unit={unit_cap}; "
        f"medians={np.round(meds_a, 4).tolist()}; champion "
        f"median {champ_med:.4f} -> percentile {pct:.1f})")

    # ---- leg 6 (built before leg 5 so leg 5 has a real
    # snapshot to write): snapshot builder on a stock smoke
    # run, with a small ladder (n_draws=5)
    snap0 = bm.build_snapshot(
        dom, layout, genome, total_gens=10, stage=0, phase="B",
        gen_in_phase=10, label="snapshot",
        genome_role="leader")
    target = snap0["trades"]["mean_invested_pct"]
    meds, unit = bm.random_ladder(
        dom, layout, eps, target, np.random.default_rng(3),
        n_draws=5)
    ladder = {"medians": meds.tolist(), "unit": unit,
              "arch_signature": layout.arch.signature(),
              "n_draws": 5}
    snap = bm.build_snapshot(
        dom, layout, genome, total_gens=10, stage=0, phase="B",
        gen_in_phase=10, label="snapshot",
        genome_role="leader", ladder=ladder)
    ok6 = np.isfinite(snap["cagr"]["champion_median"]) \
        and np.isfinite(snap["cagr"]["buyhold_median"]) \
        and len(snap["cagr"]["champion_per_episode"]) \
        == snap["n_episodes"] \
        and len(snap["cagr"]["buyhold_per_episode"]) \
        == snap["n_episodes"] \
        and isinstance(snap["sqn"]["champion"]["n_trades"], int) \
        and snap["percentile"]["value"] is not None \
        and 0.0 <= snap["percentile"]["value"] <= 100.0
    leg("6. snapshot builder on a stock smoke run: finite "
        "medians, per-episode arrays of the right length, "
        "SQN with its N, percentile with a small ladder",
        ok6,
        f"(champion median CAGR "
        f"{snap['cagr']['champion_median']:.4f}, B&H median "
        f"{snap['cagr']['buyhold_median']:.4f}, "
        f"{snap['n_episodes']} episodes, SQN "
        f"{snap['sqn']['champion']['sqn']} "
        f"N={snap['sqn']['champion']['n_trades']}, "
        f"percentile {snap['percentile']['value']:.1f})")

    # ---- leg 5: monitor files valid after a kill
    # mid-generation (torn trailing line tolerated)
    with tempfile.TemporaryDirectory() as td:
        status = {
            "run_name": "test", "stage": 0, "phase": "B",
            "gen_in_phase": 10, "total_gens": 10,
            "n_species": 6, "population": 30,
            "best_score": 0.01, "median_score": 0.0,
            "elim_median": 0.0, "seconds_per_generation": 1.5,
            "eta_phase": "1m 0s",
            "backend": {"backend": "numpy", "device": "cpu",
                        "precision": "x64", "fell_back": False,
                        "note": "numpy reference backend"},
            "structure": snap["structure"],
            "growth": {"rule": None, "growth_round": 0,
                       "tried_rules": [], "verdicts": []},
            "updated_at": "2026-11-22T20:00:00",
        }
        bm.write_status(td, status)
        bm.append_snapshot(td, snap0)
        bm.append_snapshot(td, snap)
        html_doc = bm.render_monitor_html(status, [snap0, snap])
        bm.write_html(td, html_doc)
        # simulate a kill mid-generation: torn trailing line
        with open(Path(td) / "monitor_snapshots.jsonl",
                  "a") as f:
            f.write('{"total_gens": 99, "cag')
        read_back = bm.read_snapshots(td)
        status_ok = json.loads(
            (Path(td) / "monitor_status.json").read_text()) \
            .get("run_name") == "test"
        html_back = (Path(td) / "monitor.html").read_text()
        html_ok = all(s in html_back for s in
                      ["Backend", "Run position",
                       "Network structure",
                       "What is being added now",
                       "Champion vs same-basket",
                       "Evolution curves"]) \
            and "http://" not in html_back \
            and "https://" not in html_back
        leg("5. monitor files valid after a kill "
            "mid-generation: torn trailing line tolerated, "
            "status parses, html has the sections and no "
            "external references",
            len(read_back) == 2 and status_ok and html_ok,
            f"(read {len(read_back)} good snapshots of 2; "
            f"status parses={status_ok}; html sections ok="
            f"{html_ok})")

    # ---- leg 8: end-to-end -- a real stock smoke run through
    # the runner produces monitor_status.json, snapshots,
    # monitor.html and the ladder, and the stage-end
    # snapshots' numbers match results.json exactly
    import run_bert_engine as runner
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run8"
        cfg8 = runner.smoke_config("stock")
        eng = runner.Engine(cfg8, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        res8 = eng.run()
        done8 = (out / "COMPLETE.txt").exists()
        snaps8 = bm.read_snapshots(out)
        keys8 = [(s["total_gens"], s["label"]) for s in snaps8]
        st8 = json.loads(
            (out / "monitor_status.json").read_text()) \
            if (out / "monitor_status.json").exists() else {}
        lad8 = json.loads(
            (out / "monitor_ladder.json").read_text()) \
            if (out / "monitor_ladder.json").exists() else {}
        html8 = (out / "monitor.html").read_text() \
            if (out / "monitor.html").exists() else ""
        # stage-end snapshots vs results.json stage champions
        match8 = True
        for sr in res8["stage_results"]:
            se = [s for s in snaps8
                  if s["label"] == "stage_end"
                  and s["stage"] == sr["stage"]]
            if len(se) != 1 or abs(
                    se[0]["cagr"]["median_excess"]
                    - sr["champion_val_score"]) >= 1e-12:
                match8 = False
        leg("8. end-to-end stock smoke run: monitor files "
            "produced, stage-end snapshots match results.json "
            "exactly, no duplicate snapshot keys",
            done8 and len(snaps8) >= 2
            and len(keys8) == len(set(keys8))
            and st8.get("snapshots_count") == len(snaps8)
            and st8.get("latest_snapshot", {}).get("label") \
            == "stage_end"
            and lad8.get("n_draws") == bm.LADDER_N_DRAWS
            and match8
            and all(s in html8 for s in
                    ["Backend", "Run position",
                     "Network structure",
                     "What is being added now",
                     "Champion vs same-basket",
                     "Evolution curves"])
            and "http://" not in html8
            and "https://" not in html8,
            f"(snapshots {len(snaps8)}, stages "
            f"{len(res8['stage_results'])}, stage-end excess "
            f"matches={match8}, ladder draws "
            f"{lad8.get('n_draws')})")

    # ---- leg 7: covered in the core suite, not duplicated
    print("[NOTE] 7. detailed episode simulation reproduces "
          "today's final CAGRs exactly — covered in the core "
          "suite test_bert_engine.py (leg 'detailed episode "
          "sim reproduces today\\'s final CAGRs exactly'); "
          "not duplicated here.", flush=True)

    # ---- bert_monitor must not import the runner at module
    # import time (circular import)
    src = Path(HERE / "bert_monitor.py").read_text()
    leg("module hygiene: bert_monitor does not import "
        "run_bert_engine",
        "import run_bert_engine" not in src
        and "from run_bert_engine" not in src,
        "")

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} legs "
          f"passed", flush=True)
    if failed:
        print("FAILED: " + "; ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
