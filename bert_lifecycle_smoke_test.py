#!/usr/bin/env python3
"""SMOKE TEST for bert_lifecycle.py - a CODE CHECK, not a
science result. It runs the lifecycle driver through ONE
line at tiny budgets (founding 6 generations, every
evolution stage 6 generations, population 20) and verifies
the machinery: the state file advances through every
stage of the line, champions are vaulted and registered,
stage sanity gates pass, and the smallest-at-best
bookkeeping (step 12) is consistent. Scores at these
budgets mean nothing and must never be reported as
results.

Stages exercised (one line, steps 1-6 of the frozen
lifecycle): founding -> founders measured -> parents
matured -> stitch -> seam evolution -> stack (deepen)
-> stack evolution -> backend stitch -> backend seam
evolution -> widen (Net2WiderNet split, no-noise
preservation gate, noise 0.003 on the new copies)
-> open evolution.

Run: python3 bert_lifecycle_smoke_test.py
Exit code 0 and a final PASS line on success.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_lifecycle as bl

RUN_DIR = HERE / "engine_runs" / "lifecycle_smoke_test"


def main():
    print("=" * 60)
    print("BERT LIFECYCLE SMOKE TEST - CODE CHECK ONLY, "
          "NOT A SCIENCE RESULT")
    print("=" * 60)
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    cfg = json.loads(
        (HERE / "bert_lifecycle_config.json").read_text())
    # tiny budgets: machinery only
    cfg["run_name"] = "lifecycle_smoke_test"
    cfg["founding"]["gens"] = 6
    for k in ("parents_mature_gens", "seam_gens",
              "stack_gens", "backend_seam_gens",
              "open_gens", "merge_seam_gens",
              "appendix_gens", "appendix_seam_gens",
              "free_gens"):
        cfg["budgets"][k] = 6
    for k in ("parents_pop", "seam_pop", "stack_pop",
              "open_pop", "appendix_pop", "free_pop"):
        cfg["budgets"][k] = 20
    cfg["evolution"]["train_pool"] = 200
    cfg["evolution"]["train_batch"] = 40
    cfg["evolution"]["val_every"] = 2

    driver = bl.LifecycleDriver(cfg, RUN_DIR)
    champ = driver.build_line("lineA", cycle=0, line_index=0)

    fails = []

    def check(cond, what):
        print(f"[{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            fails.append(what)

    state = json.loads(
        (RUN_DIR / "lifecycle_state.json").read_text())
    stages = state["lines"]["lineA"]["stages"]
    for stage in ("founding", "mature_host", "mature_donor",
                  "stitch", "seam_evolution", "stack",
                  "stack_evolution", "backend_stitch",
                  "backend_seam_evolution", "widen",
                  "open_evolution"):
        check(stage in stages, f"state advanced: {stage}")
    champs = state["champions"]
    check(len(champs) >= 11,
          f"champions registered ({len(champs)})")
    # the widen stage ran and its no-noise preservation
    # gate passed inside the run
    wres = json.loads((RUN_DIR / "lines" / "lineA" / "widen"
                       / "widen_result.json").read_text())
    check(wres["parity_gate"]["passed"]
          and wres["parity_gate"]["tours_identical_all_200"],
          f"widen preservation gate passed (no-noise gap "
          f"{wres['parity_gate']['no_noise_gap_pct']}% vs "
          f"subject {wres['subject_gap_pct']}%)")
    check(wres["genes"] > wres["genes_before"]
          and wres["d_after"] == 2 * wres["d_before"],
          f"widen doubled the stream ({wres['d_before']} "
          f"-> {wres['d_after']}) and grew the genome "
          f"({wres['genes_before']} -> {wres['genes']})")
    wchamp = [c for c in champs if c["stage"] == "widen_birth"]
    check(len(wchamp) == 1
          and abs(wchamp[0]["gap_pct"]
                  - wres["birth_gap_pct"]) < 1e-9,
          f"widened genome registered as a champion "
          f"candidate ({wres['birth_gap_pct']}%)")
    for c in champs:
        check(Path(c["file"]).exists(),
              f"champion file exists: {c['stage']} "
              f"({Path(c['file']).name})")
    check(champ is not None and
          champ["stage"] == "open_evolution",
          "line champion is the open-evolution champion")
    # step 13 bookkeeping, recomputed independently
    tol = float(cfg["termination"]["best_tolerance_points"])
    best = min(c["gap_pct"] for c in champs)
    check(abs(state["standing_best_gap_pct"] - best) < 1e-9,
          f"standing best consistent ({best:.4f}%)")
    at_best = [c for c in champs if c["gap_pct"] <= best + tol]
    smallest = min(at_best, key=lambda c: c["genes"])
    check(state["smallest_at_best"]["file"] == smallest["file"]
          and state["smallest_at_best"]["genes"]
          == smallest["genes"],
          f"smallest-at-best consistent "
          f"({smallest['genes']} genes, {smallest['stage']})")
    # the line champion's recorded score reproduces
    net = bl.load_net(Path(champ["file"]))
    g, _, _ = bl.gap_net(net, driver.X_val, driver.OPT)
    check(abs(g * 100 - champ["gap_pct"]) < 0.01,
          f"line champion score reproduces "
          f"({g*100:.4f}% vs recorded {champ['gap_pct']}%)")
    # every evolution stage vaulted a COMPLETE marker and
    # kept its backbone bit-identical where a mask applied
    line_dir = RUN_DIR / "lines" / "lineA"
    for stage in ("mature_host", "mature_donor",
                  "seam_evolution", "stack_evolution",
                  "backend_seam_evolution", "open_evolution"):
        res = json.loads((line_dir / stage /
                          "result.json").read_text())
        check(res["backbone_bit_identical"],
              f"{stage}: backbone bit-identical "
              f"({res['genes_frozen']} frozen genes)")
    print("=" * 60)
    if fails:
        print(f"SMOKE TEST FAILED: {len(fails)} checks")
        return 1
    print("SMOKE TEST PASS - the driver machinery works "
          "end to end for one line.")
    print("This was a code check at tiny budgets. It is "
          "NOT a science result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
