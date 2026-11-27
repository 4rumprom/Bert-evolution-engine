#!/usr/bin/env python3
"""Connections, not just neurons (sponsor direction,
2026-11-24). On the champion:

  1. NOISE: silence every single connection (gene)
     one at a time. A connection whose removal costs
     nothing is noise. Classed by the sponsor's cloud
     rule on the drops.
  2. PATHS: silence every pair of units together.
     If the pair costs much more than the two singles
     added together, the two units sit on one path:
     they need each other.

No back propagation. Silencing only.
Subject: the double-widen treatment champion."""
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

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
genome = np.asarray(z["genome"], dtype=float)

champ = domain.validation_score(layout, genome)
gap = -champ


def drop_of(g):
    return 100.0 * (champ - domain.validation_score(
        layout, g)) / gap


# 1. every connection, one at a time
gene_drops = np.zeros(len(genome))
for i in range(len(genome)):
    g = genome.copy()
    g[i] = 0.0
    gene_drops[i] = drop_of(g)
med = np.median(gene_drops)
mad = np.median(np.abs(gene_drops - med))
sigma = 1.4826 * mad
cloud = 2.0 * sigma
noise = int(np.sum(np.abs(gene_drops) <= cloud))
harm = int(np.sum(gene_drops < -cloud))
matter = int(np.sum(gene_drops > cloud))

# name the strongest connections in plain terms
names = []
for name, sl, shape in layout.entries:
    names.extend([(name, tuple(int(x) for x in np.unravel_index(
        k, shape))) for k in range(int(np.prod(shape)))])
top = np.argsort(-gene_drops)[:10]
top_conns = [{"gene": f"{names[i][0]}{names[i][1]}",
              "drop_pct_of_gap": round(float(gene_drops[i]), 2)}
             for i in top]

# 2. pairs of units: joint cost vs the sum of singles
table = be.silence_table(
    layout, genome,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
single = {u["name"]: u["drop"] / gap * 100.0
          for u in table["units"]}
unit_names = [u["name"] for u in table["units"]]
pairs = []
for a, b in itertools.combinations(unit_names, 2):
    r = bb.group_drop(domain, layout, genome, [a, b])
    joint = r["drop_rel_pct"]
    inter = joint - (single[a] + single[b])
    pairs.append({"pair": [a, b],
                  "joint_pct": round(joint, 1),
                  "interaction_pct": round(inter, 1)})
pairs.sort(key=lambda p: -p["interaction_pct"])

out = {
    "champion_gap_pct": round(gap * 100, 2),
    "n_connections": int(len(genome)),
    "cloud_limit_pct_of_gap": round(float(cloud), 3),
    "noise_connections": noise,
    "mattering_connections": matter,
    "harmful_connections": harm,
    "top_connections": top_conns,
    "top_paths_by_interaction": pairs[:8],
    "top_pairs_by_joint_cost":
        sorted(pairs, key=lambda p: -p["joint_pct"])[:5],
}
print(json.dumps(out, indent=1))
(HERE / "connection_test_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
