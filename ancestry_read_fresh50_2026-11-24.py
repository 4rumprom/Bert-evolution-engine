#!/usr/bin/env python3
"""ANCESTRY READ of the champion (sponsor method,
2026-11-24). No other method. No silencing, no
gradients, no weight-size guessing.

The record of the from-scratch 50-generation run
keeps 1,293 genomes, each with its recorded score,
its species, and its generation. The champion came
from species 6, and the archive kept that species'
whole line.

The family line: in each generation, the best
scoring genome of the species. The champion descends
from that line. (Parent links are not stored, so the
best of each generation is the line's marker.)

For each neuron, the ancestry question is one thing:
when did the champion's form of this neuron first
appear in the line, and did the line then keep it,
unchanged, while the score climbed? A neuron kept
unchanged for many generations under selection is
inherited core: the branch that led to the champion.
A neuron whose form was still being replaced in the
last generations was never locked in by selection.
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
z = np.load(HERE / "engine_runs" / "fresh50" / "lineage"
            / "stage000.npz")
G, S, GEN, SP = z["genomes"], z["scores"], \
    z["generations"], z["species_ids"]

champ = np.load(HERE / "engine_runs" / "fresh50"
                / "champion.npz")
champion = champ["genome"] if "genome" in champ \
    else champ[list(champ.keys())[0]]

gens = sorted(set(GEN.tolist()))
line = []
for g in gens:
    sel = (GEN == g) & (SP == 6)
    i = np.argmax(S[sel])
    line.append(G[sel][i])
line = np.array(line)
best_scores = [float(np.max(S[(GEN == g) & (SP == 6)]))
               for g in gens]

units = be.silence_units(layout)
rows = []
for u in units:
    idx = be.unit_gene_indices(layout, u)
    target = champion[idx]
    locked = None
    for k, g in enumerate(gens):
        if np.max(np.abs(line[k][idx] - target)) < 1e-9:
            locked = int(g)
            break
    rows.append({
        "unit": u["name"], "kind": u["kind"],
        "locked_from_generation": locked,
        "kept_for_generations":
            (int(gens[-1]) - locked + 1) if locked is not None
            else 0})

out = {
    "recorded_genomes": int(len(G)),
    "score_first_generation": round(best_scores[0], 4),
    "score_last_generation": round(best_scores[-1], 4),
    "units": rows,
}
print(json.dumps(out, indent=1))
(HERE / "ancestry_read_fresh50_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
