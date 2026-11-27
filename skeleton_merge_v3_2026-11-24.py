"""Skeleton merge test v3 (sponsor method, corrected
2026-11-24 late evening).

Roman's definition: the skeleton is built by kept
beneficial mutations. A small mutation lands, the score
improves, selection keeps the child, and that changed gene
is now part of the skeleton. Change does not break the
skeleton; winning changes BUILD it. Pieces that never
changed are its oldest parts. Pieces that kept changing
without ever winning are churn, not skeleton.

Operational rule used here:
  - Every genome of the founding run is recorded with its
    parents, then re-scored on ONE fixed set (the engine's
    elimination set, 64 problems) so parents and children
    are compared on the same yardstick.
  - Walk the champion's ancestral line from the founding
    generation to the champion. At each step, a child is a
    WINNING child when its score beats its parent's.
  - A piece's fate is set by its LAST change on the line:
    if that change arrived in a winning child, the piece's
    final form is SKELETON (with the generation it won).
    If the piece never changed on the line, it is founding
    SKELETON. Otherwise it is churn.
  - Merge: host = best species' champion; every donor
    skeleton piece whose slot is churn (or absent) in the
    host is copied in. Placebo: the same number of donor
    churn pieces copied into host churn slots instead.
  - Final scores on the fixed validation set.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bert_engine as be
import run_bert_engine as rbe

RESULTS = Path("skeleton_merge_v3_2026-11-24.json")
TOL = 1e-9


def piece_index_sets(layout):
    pieces = {}
    covered = np.zeros(layout.n_params, dtype=bool)
    for u in be.silence_units(layout):
        idx = be.unit_gene_indices(layout, u)
        pieces[u["name"]] = idx
        covered[idx] = True
    head_idx = []
    for c in range(len(layout.arch.couples)):
        for name in layout.head_names(c):
            s = layout.sl(name)
            head_idx.append(np.arange(s.start, s.stop))
    if head_idx:
        idx = np.concatenate(head_idx)
        pieces["output_path"] = idx
        covered[idx] = True
    pieces["trunk"] = np.where(~covered)[0]
    return pieces


class Genealogy:
    def __init__(self):
        self.genomes = {}
        self.gen_of = {}
        self.parents = {}
        self.next_id = 0

    def add(self, genome, gen, parents=(-1, -1)):
        gid = self.next_id
        self.next_id += 1
        self.genomes[gid] = np.asarray(genome, dtype=float).copy()
        self.gen_of[gid] = int(gen)
        self.parents[gid] = parents
        return gid


def breed_recorded(pop, ids, scores, draw_vec, classes,
                   prob_vec, rng, cfg, target, genea, gen):
    size = len(pop)
    order = np.argsort(-scores)
    fits = scores - scores.min() + 1e-9
    runs = be.full_rate_runs(classes)
    new_pop = [pop[i].copy() for i in order[:cfg.elite]]
    new_ids = [ids[i] for i in order[:cfg.elite]]

    def tourney():
        idx = rng.choice(size, cfg.tournament, replace=False)
        return idx[np.argmax(fits[idx])]

    while len(new_pop) < target:
        ia, ib = tourney(), tourney()
        a, b = pop[ia], pop[ib]
        child = a.copy()
        for sl in runs:
            take_b = rng.random(sl.stop - sl.start) < 0.5
            child[sl] = np.where(take_b, b[sl], a[sl])
        child = be.mutate(child, prob_vec, rng, cfg.mut_sigma)
        child[classes == be.CLS_DRAW] = \
            draw_vec[classes == be.CLS_DRAW]
        cid = genea.add(child, gen, (int(ids[ia]), int(ids[ib])))
        new_pop.append(child)
        new_ids.append(cid)
    return np.array(new_pop[:target]), new_ids[:target]


def line_of(genea, champ_id):
    """Champion's ancestral line via first parents, from
    the founding generation to the champion."""
    line = [champ_id]
    cur = champ_id
    while genea.parents[cur][0] >= 0:
        cur = genea.parents[cur][0]
        line.append(cur)
    return list(reversed(line))


def skeleton_read(genea, scores, champ_id, pieces):
    """Per piece: status (skeleton / churn), the generation
    its final form was set, and whether that setting step
    was a winning step."""
    line = line_of(genea, champ_id)
    status = {}
    for name, idx in pieces.items():
        last_change = None  # (gen, winning, via_crossover)
        for k in range(1, len(line)):
            child, parent = line[k], line[k - 1]
            if np.max(np.abs(genea.genomes[child][idx]
                             - genea.genomes[parent][idx])) < TOL:
                continue
            via_x = False
            b = genea.parents[child][1]
            if b >= 0 and np.max(np.abs(
                    genea.genomes[child][idx]
                    - genea.genomes[b][idx])) < TOL:
                via_x = True
            last_change = (genea.gen_of[child],
                           bool(scores[child] > scores[parent]),
                           via_x)
        if last_change is None:
            status[name] = {"status": "skeleton",
                            "set_gen": genea.gen_of[line[0]],
                            "how": "founding, never changed"}
        else:
            gen, winning, via_x = last_change
            status[name] = {
                "status": "skeleton" if winning else "churn",
                "set_gen": gen,
                "how": ("won its place"
                        + (" (arrived by crossover)"
                           if via_x else "")
                        if winning else
                        "changed in a child that did not "
                        "beat its parent")}
    return status


def main():
    cfg = be.EngineConfig.load("bert_engine_config_fresh50.json")
    cfg.run_name = "skeleton_merge_v3"
    cfg.phase_a_gens = 50
    cfg.phase_b_gens = 50
    eng = rbe.Engine(cfg, Path("engine_runs/skeleton_merge_v3"))
    eng.setup_fresh()
    layout, domain = eng.layout, eng.domain
    classes = eng.classes
    pieces = piece_index_sets(layout)
    genea = Genealogy()
    rng = eng.rng

    species = eng.species
    for sp in species:
        sp["ids"] = [genea.add(g, 0) for g in sp["pop"]]

    total_gen = 0
    t0 = time.time()
    for phase in ("A", "B"):
        pv = be.prob_vector(classes, phase, cfg, marks=None,
                            frozen=None, new_only=False,
                            flat_nominal=True)
        elim_series = []
        gen_in_phase = 0
        budget = cfg.phase_a_gens if phase == "A" \
            else cfg.phase_b_gens
        while gen_in_phase < budget:
            batch = domain.train_batch(phase, 0, gen_in_phase)
            groups, G = [], []
            pos = 0
            for sp in species:
                groups.append((pos, pos + len(sp["pop"])))
                G.append(sp["pop"])
                pos += len(sp["pop"])
            G = np.concatenate(G)
            scores = domain.evaluate(layout, G, batch,
                                     groups=groups)
            off = 0
            for sp in species:
                s = scores[off:off + len(sp["pop"])]
                off += len(sp["pop"])
                sp["lens"] = s
                bi = int(np.argmax(s))
                if s[bi] > sp["best"]["score"]:
                    sp["best"] = {
                        "score": float(s[bi]),
                        "genome": sp["pop"][bi].copy(),
                        "id": int(sp["ids"][bi]),
                        "generation": total_gen}
            elim_batch = domain.elim_batch()
            elim_bests = []
            for sp in species:
                bi = int(np.argmax(sp["lens"]))
                es = domain.evaluate(
                    layout, sp["pop"][bi][None, :], elim_batch)
                elim_bests.append(float(es[0]))
            elim_series.append(float(np.median(elim_bests)))
            if (gen_in_phase + 1) % cfg.cull_interval == 0 \
                    and len(species) > cfg.species_floor:
                champ_scores = []
                for sp in species:
                    es = domain.evaluate(
                        layout, sp["pop"], elim_batch,
                        groups=[(0, len(sp["pop"]))])
                    champ_scores.append(float(np.max(es)))
                species, entry = be.cull_species(
                    species, champ_scores, cfg,
                    cfg.n_species * cfg.per_species)
                if entry:
                    print(f"[test] gen {total_gen}: culled "
                          f"{entry['eliminated_species_ids']}")
            gen_in_phase += 1
            total_gen += 1
            pr = be.plateau_read(elim_series, cfg.plateau_window,
                                 cfg.plateau_threshold)
            if pr["plateaued"] or gen_in_phase >= budget:
                if pr["plateaued"]:
                    print(f"[test] phase {phase} plateau at "
                          f"total gen {total_gen}")
                break
            for sp in species:
                sp["pop"], sp["ids"] = breed_recorded(
                    sp["pop"], sp["ids"], sp["lens"], sp["draw"],
                    classes, pv, rng, cfg, sp["target"],
                    genea, total_gen)
    print(f"[test] run done: {total_gen} generations in "
          f"{time.time() - t0:.1f}s; {genea.next_id} genomes "
          f"recorded; species left: {[sp['id'] for sp in species]}")

    # Re-score every recorded genome on the one fixed set.
    elim_batch = domain.elim_batch()
    all_ids = sorted(genea.genomes)
    G = np.array([genea.genomes[i] for i in all_ids])
    fixed = domain.evaluate(layout, G, elim_batch)
    scores = {int(gid): float(fixed[k])
              for k, gid in enumerate(all_ids)}
    print(f"[test] re-scored {len(all_ids)} genomes on the "
          f"fixed elimination set")

    champ = {}
    for sp in species:
        sid = int(sp["id"])
        g = sp["best"]["genome"]
        champ[sid] = {
            "genome": g, "id": sp["best"]["id"],
            "gap": domain.validation_detail(layout, g)["mean_gap"]}
    ranked = sorted(champ, key=lambda s: champ[s]["gap"])
    print("[test] species by validation gap:",
          {s: round(champ[s]["gap"] * 100, 2) for s in ranked})
    a_id, b_id = ranked[0], ranked[1]

    read_a = skeleton_read(genea, scores, champ[a_id]["id"],
                           pieces)
    read_b = skeleton_read(genea, scores, champ[b_id]["id"],
                           pieces)
    skel_a = {n for n, r in read_a.items()
              if r["status"] == "skeleton"}
    skel_b = {n for n, r in read_b.items()
              if r["status"] == "skeleton"}
    for sid, read in ((a_id, read_a), (b_id, read_b)):
        print(f"[test] species {sid} read:")
        for n, r in read.items():
            print(f"    {n}: {r['status']} (set gen "
                  f"{r['set_gen']}, {r['how']})")

    merged = champ[a_id]["genome"].copy()
    moved = sorted(n for n in skel_b
                   if read_a[n]["status"] != "skeleton")
    for name in moved:
        merged[pieces[name]] = champ[b_id]["genome"][pieces[name]]
    placebo = champ[a_id]["genome"].copy()
    pool = sorted(n for n in pieces
                  if read_b[n]["status"] != "skeleton"
                  and read_a[n]["status"] != "skeleton")
    placebo_moved = pool[:len(moved)]
    for name in placebo_moved:
        placebo[pieces[name]] = champ[b_id]["genome"][pieces[name]]

    def gap(g):
        return domain.validation_detail(layout, g)["mean_gap"] * 100

    result = {
        "total_generations": int(total_gen),
        "species_a": a_id, "species_b": b_id,
        "gap_a_pct": gap(champ[a_id]["genome"]),
        "gap_b_pct": gap(champ[b_id]["genome"]),
        "read_a": read_a, "read_b": read_b,
        "skeleton_a": sorted(skel_a),
        "skeleton_b": sorted(skel_b),
        "merged_pieces": moved, "gap_merged_pct": gap(merged),
        "placebo_pieces": placebo_moved,
        "gap_placebo_pct": gap(placebo),
    }
    np.savez("skeleton_merge_v3_2026-11-24_genomes.npz",
             a=champ[a_id]["genome"], b=champ[b_id]["genome"],
             merged=merged, placebo=placebo)
    RESULTS.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in (
        "gap_a_pct", "gap_b_pct", "skeleton_a", "skeleton_b",
        "merged_pieces", "gap_merged_pct", "placebo_pieces",
        "gap_placebo_pct")}, indent=1))


if __name__ == "__main__":
    main()
