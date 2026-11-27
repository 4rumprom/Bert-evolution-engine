#!/usr/bin/env python3
"""Diagnostic (not a test): attribution fidelity by
temperature, on two subjects with very different
silencing structure:
  - the wide founding champion (36 units, all
    band-contributing, drops dominated by global
    disruption);
  - the d2 seed champion (9 units, differentiated:
    head +111% of gap, ffn u5 +27%, u0 harmful).
Run under the venv python."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_attribution as ba
import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend


def run(tag, cfg_file, mods, genome):
    cfg = be.EngineConfig.load(HERE / cfg_file)
    cfg = dataclasses.replace(cfg, **mods)
    backend = get_backend("numpy", "cpu", "x64")
    domain = tsp.TSPDomain(cfg, backend)
    arch = be.initial_arch(cfg, 2)
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    table = be.silence_table(
        layout, genome,
        lambda G: domain.validation_scores(layout, G),
        zero_tol=domain.silence_zero_tol)
    drops = np.array([u["drop"] for u in table["units"]])
    print(f"== {tag}: gap {-table['champion_score'] * 100:.2f}%, "
          f"drops {np.round(drops / -table['champion_score'] * 100, 1)}")
    for tau in (1.0, 0.5, 0.25):
        units, A = ba.attribution_profiles(domain, layout,
                                           genome, tau=tau)
        tot = A.sum(axis=1)
        print(f"  tau {tau}: pearson "
              f"{pearsonr(tot, drops)[0]:+.3f}, spearman "
              f"{spearmanr(tot, drops)[0]:+.3f}")


z = np.load(HERE / "branch_pg_scale_test_2026-11-24_"
            "founding_champion.npz")
run("wide founding", "bert_engine_config_d2_l1.json",
    dict(d=8, ffn_hidden=16, initial_blocks=2,
         initial_heads=2, run_seed=20262400,
         draw_seed=20262401), z["genome"])

z2 = np.load(HERE / "engine_runs" / "fresh_d2_l1"
             / "champion.npz")
g2 = z2["genome"] if "genome" in z2 else z2[list(z2.keys())[0]]
run("d2 seed", "bert_engine_config_d2_l1.json",
    dict(run_seed=20262000, draw_seed=20262001), g2)
