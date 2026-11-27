#!/usr/bin/env python3
"""Random-partition control for the BACK-PROPAGATION
isolation signal (companion to
cluster_control_check_2026-11-24.py, which ran the
same control on silencing profiles and found them
indistinguishable from random).

Subject: the wide founding champion (11.39% gap, 36
units). Method under test: spectral clusters of
per-problem attribution profiles, centered per
problem (bert_branch.find_branches, signal=
"attribution"). Reads:
  (a) profile coherence of the clustered branches
      vs random partitions of the same sizes, both
      measured on the centered attribution profiles
      the clustering itself sees;
  (b) group-silencing drops of the clustered
      branches vs the random groups measured in the
      silencing control (total mean 829% of gap).
Run under the venv python."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_attribution as ba
import bert_branch as bb
import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

cfg = be.EngineConfig.load(HERE / "bert_engine_config_d2_l1.json")
cfg = dataclasses.replace(cfg, d=8, ffn_hidden=16,
                          initial_blocks=2, initial_heads=2,
                          run_seed=20262400, draw_seed=20262401)
backend = get_backend("numpy", "cpu", "x64")
domain = tsp.TSPDomain(cfg, backend)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "branch_pg_scale_test_2026-11-24_"
            "founding_champion.npz")
genome = z["genome"]

units, A = ba.attribution_profiles(domain, layout, genome,
                                   tau=0.5)
Ac = A - A.mean(axis=0, keepdims=True)
names = [u["name"] for u in units]
idx = {n: i for i, n in enumerate(names)}

branches = bb.find_branches(domain, layout, genome)
sizes = [len(b["units"]) for b in branches]
clus_groups = [[u["name"] for u in b["units"]] for b in branches]
clus_drops = [b["group_drop_rel_pct"] for b in branches]


def coherence(groups, P):
    vals = []
    for g in groups:
        ii = [idx[n] for n in g]
        for a in range(len(ii)):
            for b_ in range(a + 1, len(ii)):
                x, y = P[ii[a]], P[ii[b_]]
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                if nx > 0 and ny > 0:
                    vals.append(float(x @ y / (nx * ny)))
    return float(np.mean(vals)) if vals else 0.0


clus_coh = coherence(clus_groups, Ac)
rng = np.random.default_rng(20262499)
rand_cohs = []
for _ in range(24):
    perm = rng.permutation(names)
    groups, pos = [], 0
    for s in sizes:
        groups.append(list(perm[pos:pos + s]))
        pos += s
    rand_cohs.append(coherence(groups, Ac))

out = {
    "signal": "attribution (centered, tau 0.5)",
    "branch_sizes": sizes,
    "clustered_group_drops_pct_of_gap":
        [round(d, 1) for d in clus_drops],
    "clustered_total": round(sum(clus_drops), 1),
    "random_total_reference_from_silencing_control": 829.4,
    "clustered_coherence": round(clus_coh, 3),
    "random_coherence_mean": round(float(np.mean(rand_cohs)), 3),
    "random_coherence_max": round(float(np.max(rand_cohs)), 3),
}
print(json.dumps(out, indent=1))
(HERE / "cluster_control_check_attr_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
