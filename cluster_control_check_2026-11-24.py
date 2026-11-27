#!/usr/bin/env python3
"""Does the branch clustering find anything a random
partition would not? (Sponsor suspicion, 2026-11-24.)

Subject: the wide founding champion of the branch
scale test (11.39% gap, 36 units). The clustered
branches are compared against random partitions with
the SAME group sizes on two reads:
  (a) group-silencing drop per group and in total --
      if silencing any same-sized set of units wrecks
      the score just as much, the group proof cannot
      tell a real branch from an arbitrary slice;
  (b) profile coherence -- mean pairwise cosine of
      per-problem drop profiles inside a group; a real
      branch's units should rise and fall together on
      the same problems more than a random set's do.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np

import bert_branch as bb
import bert_domain_tsp as tsp
import bert_engine as be
from run_bert_engine import get_backend

HERE = Path(__file__).resolve().parent
cfg = be.EngineConfig.load(HERE / "bert_engine_config_d2_l1.json")
cfg = dataclasses.replace(cfg, d=8, ffn_hidden=16,
                          initial_blocks=2, initial_heads=2,
                          run_seed=20262400, draw_seed=20262401)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
domain = tsp.TSPDomain(cfg, backend)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "branch_pg_scale_test_2026-11-24_"
            "founding_champion.npz")
genome = z["genome"]

units, P = bb.unit_drop_profiles(domain, layout, genome)
base = domain.validation_score(layout, genome)
gap = -base
names = [u["name"] for u in units]
branches = bb.find_branches(domain, layout, genome, k=3)
sizes = [len(b["units"]) for b in branches]
clus_drops = [b["group_drop_rel_pct"] for b in branches]
clus_total = sum(clus_drops)

idx = {n: i for i, n in enumerate(names)}


def coherence(groups):
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


clus_groups = [[u["name"] for u in b["units"]] for b in branches]
clus_coh = coherence(clus_groups)

rng = np.random.default_rng(20262499)
rand_totals, rand_drops_by_size, rand_cohs = [], {}, []
for _ in range(24):
    perm = rng.permutation(names)
    groups, pos = [], 0
    for s in sizes:
        groups.append(list(perm[pos:pos + s]))
        pos += s
    drops = []
    for g in groups:
        d = bb.group_drop(domain, layout, genome, g)
        drops.append(d["drop_rel_pct"])
        rand_drops_by_size.setdefault(len(g), []).append(
            d["drop_rel_pct"])
    rand_totals.append(sum(drops))
    rand_cohs.append(coherence(groups))

out = {
    "subject_gap_pct": gap * 100.0,
    "branch_sizes": sizes,
    "clustered_group_drops_pct_of_gap":
        [round(d, 1) for d in clus_drops],
    "clustered_total": round(clus_total, 1),
    "random_total_mean": round(float(np.mean(rand_totals)), 1),
    "random_total_max": round(float(np.max(rand_totals)), 1),
    "random_per_size": {
        str(s): {"mean": round(float(np.mean(v)), 1),
                 "max": round(float(np.max(v)), 1)}
        for s, v in sorted(rand_drops_by_size.items())},
    "clustered_coherence": round(clus_coh, 3),
    "random_coherence_mean": round(float(np.mean(rand_cohs)), 3),
    "random_coherence_max": round(float(np.max(rand_cohs)), 3),
}
print(json.dumps(out, indent=1))
(HERE / "cluster_control_check_2026-11-24.json").write_text(
    json.dumps(out, indent=1))
