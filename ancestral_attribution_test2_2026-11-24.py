#!/usr/bin/env python3
"""Ancestral attribution, part 2: do the attribution
CLUSTERS (the branches find_branches returns) respect
ancestry? Same subject and sister pairs as part 1.
Read: how many sister pairs land in the same branch,
against random partitions of the same sizes.
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
cfg = dataclasses.replace(cfg, d=4, ffn_hidden=16,
                          initial_blocks=1, initial_heads=2,
                          run_seed=20262200, draw_seed=20262201)
backend = get_backend("numpy", "cpu", "x64")
domain = tsp.TSPDomain(cfg, backend)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "double_widen_d2_bandmarks_"
            "treatment_champion.npz")
genome = z["genome"]

table = be.silence_table(
    layout, genome,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
rows = table["units"]
units, A = ba.attribution_profiles(domain, layout, genome,
                                   tau=0.5)
Ac = A - A.mean(axis=0, keepdims=True)
lab_arr = bb.spectral_clusters(Ac, k=3)
label = {i: int(lab_arr[i]) for i in range(len(rows))}
groups = [[i for i in range(len(rows)) if label[i] == g]
          for g in range(3)]
pos = {}
for i, r in enumerate(rows):
    pos[(r["kind"], r["index"])] = i
pairs = [(pos[("ffn", i)], pos[("ffn", i + 8)])
         for i in range(8)]
pairs.append((pos[("head", 0)], pos[("head", 1)]))
free = [pairs[k] for k in (0, 1, 2)]

sizes = [len(g) for g in groups]
same = sum(1 for i, j in pairs if label[i] == label[j])
same_free = sum(1 for i, j in free if label[i] == label[j])
rng = np.random.default_rng(11)
null_same, null_free = [], []
for _ in range(2000):
    perm = rng.permutation(len(rows))
    lab = np.empty(len(rows), dtype=int)
    p = 0
    for gi, s in enumerate(sizes):
        lab[perm[p:p + s]] = gi
        p += s
    null_same.append(sum(1 for i, j in pairs
                         if lab[i] == lab[j]))
    null_free.append(sum(1 for i, j in free
                         if lab[i] == lab[j]))
out = {
    "branch_sizes": sizes,
    "sister_pairs_same_branch": f"{same}/9",
    "random_mean_same": round(float(np.mean(null_same)), 2),
    "random_max_same": int(np.max(null_same)),
    "free_sister_pairs_same_branch": f"{same_free}/3",
    "random_mean_same_free": round(float(np.mean(null_free)), 2),
}
print(json.dumps(out, indent=1))
