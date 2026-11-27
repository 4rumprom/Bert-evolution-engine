#!/usr/bin/env python3
"""Donor read for the fresh50_v2 champion (2026-11-24, night).

Roman's direction, 23:29 EST: keep the important piece of the
historical skeleton, merge it with another important piece from
another species. This script reads the OTHER species: rank every
surviving species' best genome on the fixed validation set, take
the best non-champion species as the donor, and silence its units
on the same fixed set with the same cloud rule as the champion's
read (silencing_fresh50_v2_2026-11-24.json).

Measurement only. No evolution, no graft, no run.
"""
import importlib.util
import json
import sys

import numpy as np

import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

spec = importlib.util.spec_from_file_location(
    "dw", "double_widen_d2_2026-11-23.py")
dw = importlib.util.module_from_spec(spec)
sys.modules["dw"] = dw
spec.loader.exec_module(dw)

CFG = "bert_engine_config_fresh50.json"
RUN = "engine_runs/fresh50_v2"
OUT = "donor_silencing_fresh50_v2_2026-11-24.json"

cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
domain = tsp.TSPDomain(cfg, backend)

cp = np.load(f"{RUN}/checkpoint.npz", allow_pickle=True)
species_ids = [int(s) for s in cp["species_ids"]]
best = np.asarray(cp["best_genomes"], dtype=float)
champ = np.load(f"{RUN}/champion.npz")["genome"]

# rank every surviving species best genome on the fixed validation set
cand = np.vstack([champ, best])
scores = domain.validation_scores(layout, cand)
names = ["champion"] + [f"species {s}" for s in species_ids]
print("validation gaps on the fixed 200-problem set:")
for n, s in zip(names, scores):
    print(f"  {n:12s} gap {-s * 100:.2f}%")
donor_i = None
for i in range(1, len(cand)):
    if np.array_equal(cand[i], champ):
        continue  # the champion's own species is not a donor
    if donor_i is None or scores[i] > scores[donor_i]:
        donor_i = i
donor_name = names[donor_i]
donor_genome = cand[donor_i]
print(f"donor: {donor_name} (gap {-scores[donor_i] * 100:.2f}%)")

table = be.silence_table(
    layout, donor_genome,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
read = {"champion_score": float(table["champion_score"]),
        "units": [{"name": u["name"], "drop": float(u["drop"])}
                  for u in table["units"]]}
out = dw.cloud_classify(read)
print(f"donor cloud limit: +/-{out['cloud']['limit_pct']:.1f}% of gap")
for u in out["units"]:
    print(f"  {u['name']:20s} drop {u['rel_drop_pct']:6.1f}% of gap -> "
          f"{u['cloud_cat']}")

# merge inventory: champion neutral slots vs donor units at the
# same position (same couple/block/kind/index)
champ_tab = json.load(open("silencing_fresh50_v2_2026-11-24.json"))
champ_neutral = {u["name"] for u in champ_tab["units"]
                 if u["cloud_cat"] == "neutral"}
donor_by_name = {u["name"]: u for u in out["units"]}
merge = []
for name in sorted(champ_neutral):
    d = donor_by_name[name]
    merge.append({"slot": name,
                  "donor_rel_drop_pct": round(d["rel_drop_pct"], 2),
                  "donor_cloud_cat": d["cloud_cat"],
                  "fits": d["cloud_cat"] == "contributing"})
print("merge inventory (champion neutral slot <- donor unit):")
for m in merge:
    print(f"  {m['slot']:20s} donor {m['donor_rel_drop_pct']:6.1f}% "
          f"{m['donor_cloud_cat']:12s} fits={m['fits']}")

json.dump({
    "donor": donor_name,
    "donor_gap_pct": round(float(-scores[donor_i]) * 100, 2),
    "champion_gap_pct": round(float(-scores[0]) * 100, 2),
    "all_validation_gaps_pct": {
        n: round(float(-s) * 100, 2) for n, s in zip(names, scores)},
    "cloud_limit_pct": round(out["cloud"]["limit_pct"], 2),
    "counts": out["cloud"]["counts"],
    "units": [{"name": u["name"], "rel_drop_pct": round(u["rel_drop_pct"], 2),
               "cloud_cat": u["cloud_cat"]} for u in out["units"]],
    "merge_inventory": merge,
}, open(OUT, "w"), indent=1)
print("saved", OUT)
