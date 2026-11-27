#!/usr/bin/env python3
"""Branch Skeleton from Ancestry (sponsor method,
2026-11-24). Built ONLY from the recorded run: the
archive of genomes with their scores, and the
champion. No silencing. No gradients. No reading
importance off weight sizes.

For each unit of the champion, the skeleton records
its ancestry along the winning line (the best genome
of the champion's species in each generation):

  origin: where the unit's genes came from. A unit
    whose genes match the first generation's best
    genome was born in the founding draw of the
    species and inherited.
  first_seen: the first generation whose best genome
    already carries the unit in its champion form.
  last_change: the last generation where the unit's
    genes still differed from the champion's.
  kept_generations: how many generations the line
    carried the champion form.

Units that share an origin form a FAMILY. The
families are the skeleton. A unit that was never
separately born, replaced, or selected has no
separate ancestry, and the skeleton says so instead
of inventing one.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be


def build_skeleton(run_dir, layout, champion_genome,
                   species_id):
    run_dir = Path(run_dir)
    z = np.load(run_dir / "lineage" / "stage000.npz")
    G, S, GEN, SP = z["genomes"], z["scores"], \
        z["generations"], z["species_ids"]
    gens = sorted(set(GEN.tolist()))
    line = []
    for g in gens:
        sel = (GEN == g) & (SP == species_id)
        line.append(G[sel][int(np.argmax(S[sel]))])
    line = np.array(line)

    def trace(idx, name, kind):
        target = champion_genome[idx]
        first_seen = None
        last_change = None
        for k, g in enumerate(gens):
            same = np.max(np.abs(line[k][idx] - target)) < 1e-9
            if same and first_seen is None:
                first_seen = int(g)
            if not same:
                last_change = int(g)
        origin = ("founding draw of species "
                  f"{species_id}") if first_seen == gens[0] \
            else f"appeared in generation {first_seen}"
        return {
            "unit": name, "kind": kind,
            "origin": origin,
            "first_seen_generation": first_seen,
            "last_change_generation": last_change,
            "kept_generations":
                int(gens[-1]) - (last_change if last_change
                                  is not None else gens[0] - 1),
        }

    units = []
    for u in be.silence_units(layout):
        idx = be.unit_gene_indices(layout, u)
        units.append(trace(idx, u["name"], u["kind"]))
    # The output path (the decoder head) is part of the
    # network like any other path (sponsor, 2026-11-24):
    # its synapses in, its weights, its picks out. It is
    # traced as its own piece, not left over.
    head_idx = []
    for c in range(len(layout.arch.couples)):
        for name in layout.head_names(c):
            head_idx.append(layout.sl(name))
    if head_idx:
        idx = np.concatenate(
            [np.arange(s.start, s.stop) for s in head_idx])
        units.append(trace(idx, "output path (decoder head)",
                           "output_path"))
    fams = {}
    for r in units:
        fams.setdefault(r["origin"], []).append(r["unit"])
    return {
        "species": int(species_id),
        "generations": [int(gens[0]), int(gens[-1])],
        "recorded_genomes": int(len(G)),
        "families": [{"origin": o, "units": us}
                     for o, us in fams.items()],
        "units": units,
    }


if __name__ == "__main__":
    import bert_domain_tsp as tsp
    run = HERE / "engine_runs" / "fresh50"
    cfg = be.EngineConfig.load(
        HERE / "bert_engine_config_fresh50.json")
    arch = be.initial_arch(cfg, 2)
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    ch = np.load(run / "champion.npz")
    genome = ch["genome"]
    res = json.load(open(run / "results.json"))
    species = int(res["champion"]["species_id"]) \
        if "species_id" in res.get("champion", {}) else 6
    skel = build_skeleton(run, layout, genome, species)
    # genes outside every unit (the decoder head):
    covered = np.zeros(len(genome), dtype=bool)
    for u in be.silence_units(layout):
        covered[be.unit_gene_indices(layout, u)] = True
    z = np.load(run / "lineage" / "stage000.npz")
    sel = (z["generations"] == 0) & (z["species_ids"] == species)
    first = z["genomes"][sel][int(np.argmax(z["scores"][sel]))]
    rest = np.where(~covered)[0]
    skel["outside_units"] = {
        "genes": int(len(rest)),
        "genes_changed_since_first_generation":
            int(np.sum(np.abs(first[rest] - genome[rest]) > 1e-9)),
    }
    print(json.dumps(skel, indent=1))
    (HERE / "branch_skeleton_fresh50_2026-11-24.json"
     ).write_text(json.dumps(skel, indent=1))
