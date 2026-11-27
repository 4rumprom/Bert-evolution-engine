#!/usr/bin/env python3
"""Find a branch that matters, using ancestry to
propose and group silencing to prove. No back
propagation anywhere in this script.

Subject: the double-widen treatment champion. Its
18 units form 9 families of 2. Each family is one
unit from before the widen plus its split copy.

Test: silence each whole family. Measure the cost.
Compare with random pairs of units, same size.
A family whose cost stands clear of the random
pairs is a branch that matters."""
import dataclasses
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
genome = z["genome"]

rows = be.silence_units(layout)
pos = {}
for i, r in enumerate(rows):
    pos[(r["kind"], r["index"])] = r["name"]
families = {}
for i in range(8):
    families[f"ffn mother {i}"] = [pos[("ffn", i)],
                                   pos[("ffn", i + 8)]]
families["head mother 0"] = [pos[("head", 0)],
                             pos[("head", 1)]]

fam_drops = {}
for tag, names in families.items():
    r = bb.group_drop(domain, layout, genome, names)
    fam_drops[tag] = round(r["drop_rel_pct"], 1)

names_all = [r["name"] for r in rows]
rng = np.random.default_rng(3)
rand = []
for _ in range(60):
    pair = list(rng.choice(names_all, size=2,
                           replace=False))
    r = bb.group_drop(domain, layout, genome,
                      [str(x) for x in pair])
    rand.append(r["drop_rel_pct"])
rand = np.array(rand)

out = {
    "family_drops_pct_of_gap": fam_drops,
    "random_pair_mean": round(float(rand.mean()), 1),
    "random_pair_max": round(float(rand.max()), 1),
    "random_pair_p95": round(float(np.percentile(rand, 95)), 1),
}
print(json.dumps(out, indent=1))
(HERE / "family_group_test_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
