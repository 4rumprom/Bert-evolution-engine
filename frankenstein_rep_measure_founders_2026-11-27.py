#!/usr/bin/env python3
"""Replication stage A1 measurement: score every
surviving species' best genome of the rep founding
run on the fixed validation set, rank them, and
record host (#1) / donor (#2) for stage A2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

REP = HERE / "engine_runs" / "frankenstein_replication"
RUN = REP / "founding"
cfg = be.EngineConfig.load(HERE / "bert_engine_config_rep_deep50.json")
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
domain = tsp.TSPDomain(cfg, backend)

ck = np.load(RUN / "checkpoint.npz", allow_pickle=True)
species_ids = [int(s) for s in ck["species_ids"]]
best_genomes = np.asarray(ck["best_genomes"], dtype=float)
zc = np.load(RUN / "champion.npz")
champ = np.asarray(zc["genome"], dtype=float)

rows = []
for sid, g in zip(species_ids, best_genomes):
    gap = float(domain.validation_detail(layout, g)["mean_gap"])
    rows.append({"species": sid, "gap_pct": round(gap * 100, 4),
                 "is_run_champion": bool(np.array_equal(g, champ))})
rows.sort(key=lambda r: r["gap_pct"])
out = {"founding_run": "engine_runs/frankenstein_replication/founding",
       "config": "bert_engine_config_rep_deep50.json",
       "species_champions_ranked": rows,
       "host_species": rows[0]["species"],
       "host_gap_pct": rows[0]["gap_pct"],
       "donor_species": rows[1]["species"],
       "donor_gap_pct": rows[1]["gap_pct"]}
assert rows[0]["is_run_champion"], "best species is not the run champion"
with open(REP / "founders_measured.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
