#!/usr/bin/env python3
"""Read the champion from its GENOME ONLY (sponsor
rule, 2026-11-24): no silencing, no scoring, no test
of any kind after the run. The only evidence is the
weights as recorded.

Logic of the read:
  - A unit can only move the answer through its
    OUTGOING weights. If they are near zero, the
    unit is dead weight, whatever it receives.
  - A unit matters when signal can flow through it:
    incoming strength times outgoing strength.
  - A connection matters when its weight is large.
    A weight near zero connects nothing.

Subject: the from-scratch 50-generation founding run
(engine_runs/fresh50), champion of species 6, gap
28.45%, 88 genes, 9 units (8 feed-forward + 1 head).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be

cfg = be.EngineConfig.load(HERE / "bert_engine_config_fresh50.json")
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "engine_runs" / "fresh50" / "champion.npz")
genome = z["genome"] if "genome" in z else z[list(z.keys())[0]]
d = layout.unpack_dict(genome)

W1 = d["c0.b0.W_1"]; b1 = d["c0.b0.b_1"]
W2 = d["c0.b0.W_2"]
units = []
for u in range(W1.shape[1]):
    in_s = float(np.linalg.norm(W1[:, u]) + abs(b1[u]))
    out_s = float(np.linalg.norm(W2[u, :]))
    units.append({"unit": f"feed-forward {u}",
                  "in": round(in_s, 3), "out": round(out_s, 3),
                  "flow": round(in_s * out_s, 3)})
Wo = d["c0.b0.h0.W_o"]; bo = d["c0.b0.h0.b_o"]
head_out = float(np.linalg.norm(Wo) + np.linalg.norm(bo))
head_in = float(np.mean([np.linalg.norm(d[f"c0.b0.h0.W_{p}"])
                         for p in "qkv"]))
units.append({"unit": "head 0", "in": round(head_in, 3),
              "out": round(head_out, 3),
              "flow": round(head_in * head_out, 3)})
units.sort(key=lambda r: -r["flow"])
strongest = units[0]["flow"]
for r in units:
    r["share_of_strongest_pct"] = round(
        100.0 * r["flow"] / strongest, 1)

allw = np.abs(genome)
med = float(np.median(allw))
mad = float(np.median(np.abs(allw - med)))
cloud = med + 2.0 * 1.4826 * mad
top_idx = np.argsort(-allw)[:10]
names = []
for name, sl, shape in layout.entries:
    names.extend([(name, tuple(int(x) for x in
                  np.unravel_index(k, shape)))
                  for k in range(int(np.prod(shape)))])
out = {
    "units_by_flow": units,
    "weights_total": int(len(genome)),
    "weights_inside_near_zero_cloud": int(np.sum(allw <= cloud)),
    "cloud_limit_abs_weight": round(cloud, 4),
    "largest_weights": [
        {"gene": f"{names[i][0]}{names[i][1]}",
         "weight": round(float(genome[i]), 3)}
        for i in top_idx],
}
print(json.dumps(out, indent=1))
(HERE / "genome_read_fresh50_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
