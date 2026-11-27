"""Skeleton merge test v2 (sponsor method, 2026-11-24).

Roman's method, in his words: the historical skeleton that
gets preserved over generations and leads to the champion is
the important piece; the rest is not. Keep the important
piece, merge it with another important piece from another
species.

V1 of this test traced best-of-generation lines. That was
wrong: the elite copy makes the line collapse onto the
champion genome itself, so the trace measured elitism, not
preservation. V2 records TRUE ancestry: every genome gets
an id, every child records both parents, and each piece of
the champion is followed back through whichever parent
carried its exact genes. The generation where neither
parent carried the piece is where its final form was
created. A piece is in the species' SKELETON when its
final form was created at or before the run's midpoint:
it was then preserved, unchanged, for the whole second
half of the run on the champion's own ancestry.

Breeding replicates bert_engine.breed_species exactly
(elites copied, tournament selection, uniform crossover
on the full-rate runs only, mutation, draw re-imposed),
with parent ids recorded alongside.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bert_engine as be
import run_bert_engine as rbe

RESULTS = Path("skeleton_merge_test_2026-11-24.json")
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
    """Every genome ever created: id -> genome, generation,
    species, parents (a, b). Gen-0 genomes have parents
    (-1, -1). Elites keep their id (same genome carried)."""

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

    def piece_created_gen(self, champ_id, idx):
        """Follow the piece's exact genes back through the
        ancestry. Returns the generation where the final
        form was created (neither parent carried it), or
        the generation of the oldest ancestor reached."""
        target = self.genomes[champ_id][idx]
        cur = champ_id
        while True:
            a, b = self.parents[cur]
            if a < 0:
                return self.gen_of[cur]
            if np.max(np.abs(self.genomes[a][idx] - target)) < TOL:
                cur = a
                continue
            if b >= 0 and np.max(
                    np.abs(self.genomes[b][idx] - target)) < TOL:
                cur = b
                continue
            return self.gen_of[cur]


def breed_recorded(pop, ids, scores, draw_vec, classes,
                   prob_vec, rng, cfg, target, genea, gen):
    """breed_species with parent recording. Elites keep
    their ids; children get fresh ids and (a, b) parents."""
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


def main():
    cfg = be.EngineConfig.load("bert_engine_config_fresh50.json")
    cfg.run_name = "skeleton_merge"
    cfg.phase_a_gens = 50
    cfg.phase_b_gens = 50
    eng = rbe.Engine(cfg, Path("engine_runs/skeleton_merge"))
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
        while gen_in_phase < (cfg.phase_a_gens if phase == "A"
                              else cfg.phase_b_gens):
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
            # culling on the engine's schedule
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
            last = gen_in_phase >= (cfg.phase_a_gens
                                    if phase == "A"
                                    else cfg.phase_b_gens)
            if pr["plateaued"] or last:
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
    mid = total_gen // 2

    def skeleton_of(sid):
        created = {}
        for name, idx in pieces.items():
            created[name] = genea.piece_created_gen(
                champ[sid]["id"], idx)
        skel = {n for n, g in created.items() if g <= mid}
        return created, skel

    created_a, skel_a = skeleton_of(a_id)
    created_b, skel_b = skeleton_of(b_id)
    print(f"[test] midpoint generation: {mid}")
    print(f"[test] species {a_id} piece creation gens: {created_a}")
    print(f"[test] species {a_id} skeleton: {sorted(skel_a)}")
    print(f"[test] species {b_id} piece creation gens: {created_b}")
    print(f"[test] species {b_id} skeleton: {sorted(skel_b)}")

    merged = champ[a_id]["genome"].copy()
    moved = sorted(skel_b - skel_a)
    for name in moved:
        merged[pieces[name]] = champ[b_id]["genome"][pieces[name]]
    placebo = champ[a_id]["genome"].copy()
    pool = sorted((set(pieces) - skel_b) - skel_a)
    placebo_moved = pool[:len(moved)]
    for name in placebo_moved:
        placebo[pieces[name]] = champ[b_id]["genome"][pieces[name]]

    def gap(g):
        return domain.validation_detail(layout, g)["mean_gap"] * 100

    result = {
        "total_generations": int(total_gen),
        "midpoint_generation": int(mid),
        "species_a": a_id, "species_b": b_id,
        "gap_a_pct": gap(champ[a_id]["genome"]),
        "gap_b_pct": gap(champ[b_id]["genome"]),
        "created_a": created_a, "created_b": created_b,
        "skeleton_a": sorted(skel_a), "skeleton_b": sorted(skel_b),
        "merged_pieces": moved, "gap_merged_pct": gap(merged),
        "placebo_pieces": placebo_moved,
        "gap_placebo_pct": gap(placebo),
        "never_skeleton_either": sorted(set(pieces) - skel_a - skel_b),
    }
    np.savez("skeleton_merge_test_2026-11-24_genomes.npz",
             a=champ[a_id]["genome"], b=champ[b_id]["genome"],
             merged=merged, placebo=placebo)
    RESULTS.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in (
        "gap_a_pct", "gap_b_pct", "merged_pieces", "gap_merged_pct",
        "placebo_pieces", "gap_placebo_pct",
        "never_skeleton_either")}, indent=1))


if __name__ == "__main__":
    main()
