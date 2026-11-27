#!/usr/bin/env python3
"""ITERATED prune-and-graft rounds (2026-11-23), continuing
prune_graft_test_2026-11-23.py (round 1). Standalone experiment,
NOT an engine integration. Same domain/config as round 1
(bert_engine_config_test.json, fixed 200-problem TSP validation
set, numpy backend, CPU x64), same frozen rules per round:

  PRUNE: silence the current prune-graft lineage champion
  (be.silence_table, exact-zero criterion); zero the outgoing
  weights of every unit with drop <= 0; measure the pruned
  base's validation score alone.
  DONORS: the species best genomes from the PREVIOUS round's
  prune-graft (treatment) arm evolution (genomes identical to
  the current champion excluded, as in round 1). A donor unit
  is graft-eligible only if its drop in its OWN donor genome
  is > 0 (silencing table per donor).
  GRAFT: position-preserving only (same couple/block/kind/
  index slot), copying ALL of the unit's genes
  (be.unit_gene_indices). Slots with no eligible donor stay
  zeroed.
  TREATMENT ARM: 12 species x 10, species variants of the
  grafted base (random graft subsets, fractions 25%..100% of
  eligible slots across species) + standard mutation; evolve
  50 generations (phase A 25 + phase B 25) with the identical
  primitives loop as round 1.
  CONTROL ARM: identical budget/operators, seeded from the
  same starting champion, unpruned and ungrafted.

ROUND 2 DONOR PROVENANCE: round 1 saved only its arm
champions, not arm B's species bests. Step 0 of this script
therefore re-runs round 1's arm B with the identical seeds
(run 20261300, draw 20261301, graft 20261304) and identical
construction order, captures the species bests, and VERIFIES
the reproduced champion scores exactly -0.12810164364860518
(round 1's recorded value). If it does not reproduce, the
script stops and reports that instead of proceeding.

SEEDS per round r >= 2 (same family, advanced per round):
  run_seed   = 20261300 + (r-1)*100
  draw_seed  = 20261301 + (r-1)*100
  graft_seed = 20261304 + (r-1)*100
Within a round, each arm gets freshly initialised run/draw
RNGs from that round's seeds (round 1's convention), and the
graft RNG is treatment-arm only.

STOPPING RULE (sponsor's): keep running rounds while the
treatment margin over control (treatment final champion val
score minus control final champion val score) is >= +0.01.
Stop after the first round whose margin falls below +0.01,
or after round 5, whichever comes first. Also stop if a
round's graft inventory is empty. The treatment champion of
each round becomes the next round's starting champion.

Evolution loop: the same primitives loop as round 1
(be.create_species, be.breed_species, be.cull_species,
be.prob_vector; Schedule A culling; species best-ever on
train scores; stage champion = validation argmax over
species bests), extended only to also RETURN the final
species' best genomes (needed as the next round's donors).
Classes: be._base_classes(layout, 0) (engine founder-
continuation rebase), marks None -- round 1's conventions.
"""
import json
import time
import html as htmlmod
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "engine_runs" / "test_minimal_a"
CFG_PATH = HERE / "bert_engine_config_test.json"
R1_JSON = HERE / "prune_graft_test_2026-11-23.json"
OUT_JSON = HERE / "prune_graft_rounds_2026-11-23.json"
OUT_HTML = HERE / "prune_graft_rounds_2026-11-23.html"
GENS_PER_PHASE = 25
MARGIN_FLOOR = 0.01
MAX_ROUND = 5

R1_SUBJECT_SCORE = -0.41201394818066317
R1_CONTROL_SCORE = -0.20350921193596969
R1_TREATMENT_SCORE = -0.12810164364860518
R1_MARGIN = R1_TREATMENT_SCORE - R1_CONTROL_SCORE


def log(msg):
    print(f"[pg-rounds] {msg}", flush=True)


def round_seeds(r):
    if r == 1:
        return {"run_seed": 20261300, "draw_seed": 20261301,
                "graft_seed": 20261304}
    return {"run_seed": 20261300 + (r - 1) * 100,
            "draw_seed": 20261301 + (r - 1) * 100,
            "graft_seed": 20261304 + (r - 1) * 100}


# ---------------------------------------------------------------
# density read: copied from prune_graft_test_2026-11-23.py
# (silencing table + paired-noise band categories)
# ---------------------------------------------------------------
def density_read(domain, layout, genome):
    table = be.silence_table(
        layout, genome,
        lambda G: domain.validation_scores(layout, G),
        zero_tol=domain.silence_zero_tol)
    units = [dict(u) for u in table["units"]]
    unit_list = be.silence_units(layout)
    assert [u["name"] for u in units] == \
        [u["name"] for u in unit_list]
    G = np.tile(np.asarray(genome, dtype=float),
                (len(unit_list) + 1, 1))
    for k, u in enumerate(unit_list, start=1):
        G[k, be._silence_indices(layout, u)] = 0.0
    lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
    gaps = lens / domain.optima - 1.0
    champ = float(table["champion_score"])
    assert abs(-float(gaps[0].mean()) - champ) < 1e-9, \
        (gaps[0].mean(), champ)
    for k, u in enumerate(units):
        diffs = gaps[k + 1] - gaps[0]
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        u["se"] = se
        u["band"] = 2.0 * se
        u["cat"] = ("contributing" if u["drop"] > u["band"]
                    else "harmful" if u["drop"] < -u["band"]
                    else "neutral")
        u["drop"] = float(u["drop"])
    cats = {"contributing": 0, "neutral": 0, "harmful": 0}
    for u in units:
        cats[u["cat"]] += 1
    return {
        "champion_score": champ,
        "n_units": len(units),
        "exact_zero_positive": int(sum(
            1 for u in units if u["drop"] > 0.0)),
        "band_counts": cats,
        "units": [{k: u[k] for k in
                   ("name", "kind", "couple", "block", "index",
                    "drop", "se", "band", "cat")} for u in units],
    }


def unit_key(u):
    return (u["couple"], u["block"], u["kind"], u["index"])


# ---------------------------------------------------------------
# evolution: round 1's primitives loop, extended to return
# the final species' best genomes alongside the champion
# ---------------------------------------------------------------
def evolve(domain, layout, cfg, classes, species, rng,
           gens_per_phase=GENS_PER_PHASE):
    history = []
    total_gens = 0
    for phase in ("A", "B"):
        gen_in_phase = 0
        while gen_in_phase < gens_per_phase:
            batch = domain.train_batch(phase, 0, gen_in_phase)
            groups, G = [], []
            pos = 0
            for sp in species:
                groups.append((pos, pos + len(sp["pop"])))
                G.append(sp["pop"])
                pos += len(sp["pop"])
            G = np.concatenate(G)
            scores = domain.evaluate(layout, G, batch,
                                     groups=groups, cache={})
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
                        "generation": total_gens}
            elim_batch = domain.elim_batch()
            elim_best = []
            for sp in species:
                bi = int(np.argmax(sp["lens"]))
                es = domain.evaluate(
                    layout, sp["pop"][bi][None, :], elim_batch,
                    cache={})
                elim_best.append(float(es[0]))
            culled = False
            if (gen_in_phase + 1) % cfg.cull_interval == 0 \
                    and len(species) > cfg.species_floor:
                champ_scores = []
                for sp in species:
                    es = domain.evaluate(
                        layout, sp["pop"], elim_batch,
                        groups=[(0, len(sp["pop"]))], cache={})
                    champ_scores.append(float(np.max(es)))
                species, entry = be.cull_species(
                    species, champ_scores, cfg,
                    cfg.n_species * cfg.per_species)
                culled = entry is not None
            history.append({
                "generation": total_gens, "phase": phase,
                "gen_in_phase": gen_in_phase,
                "n_species": len(species),
                "pop_total": int(sum(len(sp["pop"])
                                     for sp in species)),
                "best_score": float(np.max(scores)),
                "median_score": float(np.median(scores)),
                "elim_median": float(np.median(elim_best)),
                "cull": culled})
            gen_in_phase += 1
            total_gens += 1
            if gen_in_phase < gens_per_phase:
                pv = be.prob_vector(classes, phase, cfg,
                                    marks=None)
                for sp in species:
                    sp["pop"] = be.breed_species(
                        sp["pop"], sp["lens"], sp["draw"],
                        classes, pv, rng, cfg, sp["target"])
    val_scores = [domain.validation_score(layout,
                                          sp["best"]["genome"])
                  for sp in species]
    bi = int(np.argmax(val_scores))
    species_bests = [{
        "species_id": int(sp["id"]),
        "train_best_score": float(sp["best"]["score"]),
        "val_score": float(vs),
        "genome": species[i]["best"]["genome"]}
        for i, (sp, vs) in enumerate(zip(species, val_scores))]
    return {
        "champion_genome": species[bi]["best"]["genome"],
        "champion_val_score": float(val_scores[bi]),
        "champion_species_id": int(species[bi]["id"]),
        "n_species_final": len(species),
        "history": history,
        "species_bests": species_bests,
    }


# ---------------------------------------------------------------
# round-1 arm B construction (prune + donors + graft variants),
# parameterised so step 0 reproduces round 1 exactly and later
# rounds reuse the same construction on their own inputs
# ---------------------------------------------------------------
def build_treatment_species(domain, layout, cfg, classes, pv_a,
                            pruned_base, unit_by_key, eligible,
                            donor_genome_by_id, seeds):
    graft_rng = np.random.default_rng(seeds["graft_seed"])
    fractions = [float(x) for x in
                 np.linspace(0.25, 1.0, cfg.n_species)]
    eligible_keys = sorted(eligible, key=lambda k: str(k))
    variant_records = []
    species = []
    rng = np.random.default_rng(seeds["run_seed"])
    draw_rng = np.random.default_rng(seeds["draw_seed"])
    for s in range(cfg.n_species):
        frac = fractions[s]
        n_take = int(round(frac * len(eligible_keys)))
        chosen = []
        if n_take:
            idx = graft_rng.choice(len(eligible_keys),
                                   size=n_take, replace=False)
            chosen = [eligible_keys[int(i)] for i in idx]
        base_s = pruned_base.copy()
        grafts = []
        for k in chosen:
            sid = int(graft_rng.choice(eligible[k]))
            idxs = be.unit_gene_indices(layout, unit_by_key[k])
            base_s[idxs] = donor_genome_by_id[sid][idxs]
            grafts.append({"unit": unit_by_key[k]["name"],
                           "donor_species_id": sid})
        base_score = domain.validation_score(layout, base_s)
        variant_records.append({
            "species_index": s, "fraction": frac,
            "n_grafts": len(grafts), "grafts": grafts,
            "variant_base_val_score": float(base_score)})
        sp = be.create_species(
            base_s, classes, 1, cfg.per_species, draw_rng,
            rng, pv_a, cfg, lineage=None,
            init_mode="transplant", next_id=s)
        species.extend(sp)
    return species, rng, variant_records


def prune_and_inventory(domain, layout, champion, champion_read,
                        donor_bests):
    """donor_bests: list of {species_id, genome}. Returns
    (pruned_base, pruned_score, inventory dict,
     donor_genome_by_id, unit_by_key)."""
    prune_units = [u for u in champion_read["units"]
                   if u["drop"] <= 0.0]
    pruned_base = np.asarray(champion, dtype=float).copy()
    for u in prune_units:
        pruned_base[be._silence_indices(layout, u)] = 0.0
    pruned_score = domain.validation_score(layout, pruned_base)

    donors = []
    seen = []
    for e in donor_bests:
        g = np.asarray(e["genome"], dtype=float)
        if np.array_equal(g, np.asarray(champion, dtype=float)):
            continue
        if any(np.array_equal(g, s) for s in seen):
            continue
        seen.append(g)
        donors.append({"species_id": int(e["species_id"]),
                       "genome": g})
    donor_tables = {}
    donor_eligible = {}
    for e in donors:
        tab = be.silence_table(
            layout, e["genome"],
            lambda G: domain.validation_scores(layout, G),
            zero_tol=domain.silence_zero_tol)
        donor_eligible[e["species_id"]] = {
            unit_key(u): float(u["drop"]) for u in tab["units"]}
        donor_tables[str(e["species_id"])] = {
            "donor_val_score": float(tab["champion_score"]),
            "positive_drop_units": [
                u["name"] for u in tab["units"]
                if u["drop"] > 0.0]}
    unit_by_key = {unit_key(u): u
                   for u in champion_read["units"]}
    eligible = {}
    for u in prune_units:
        k = unit_key(u)
        ds = [sid for sid, drops in donor_eligible.items()
              if drops.get(k, 0.0) > 0.0]
        if ds:
            eligible[k] = ds
    inventory = {
        "n_pruned_slots": len(prune_units),
        "pruned_units": [u["name"] for u in prune_units],
        "pruned_base_score": float(pruned_score),
        "donor_species_ids": [e["species_id"] for e in donors],
        "donor_tables": donor_tables,
        "n_eligible_slots": len(eligible),
        "eligible_slots": [
            {"unit": unit_by_key[k]["name"],
             "donor_species_ids": eligible[k]}
            for k in sorted(eligible, key=lambda k: str(k))],
        "slots_without_eligible_donor": [
            u["name"] for u in prune_units
            if unit_key(u) not in eligible],
    }
    donor_genome_by_id = {e["species_id"]: e["genome"]
                          for e in donors}
    return pruned_base, inventory, donor_genome_by_id, \
        unit_by_key, eligible


def save_species_bests(path, species_bests):
    np.savez(
        path,
        genomes=np.array([b["genome"]
                          for b in species_bests]),
        species_ids=np.array([b["species_id"]
                              for b in species_bests]),
        val_scores=np.array([b["val_score"]
                             for b in species_bests]),
        train_best_scores=np.array(
            [b["train_best_score"] for b in species_bests]))


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    ck = np.load(RUN_DIR / "checkpoint.npz", allow_pickle=True)
    arch = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    classes = be._base_classes(layout, 0)
    pv_a = be.prob_vector(classes, "A", cfg, marks=None)

    results = {
        "experiment": "iterated prune-and-graft rounds "
                      "(standalone, not an engine integration)",
        "date": "2026-11-23",
        "continues": str(R1_JSON),
        "settings": {
            "config_file": str(CFG_PATH),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "margin_floor": MARGIN_FLOOR,
            "max_round": MAX_ROUND,
            "seed_rule": "round r>=2: run 20261300+(r-1)*100, "
                         "draw 20261301+(r-1)*100, graft "
                         "20261304+(r-1)*100; round 1 seeds "
                         "are the original test's",
            "classes_rule": "be._base_classes(layout, 0) "
                            "(round 1 convention)",
            "marks": None,
        },
        "round_1_reference": {
            "subject_score": R1_SUBJECT_SCORE,
            "control_final_score": R1_CONTROL_SCORE,
            "treatment_final_score": R1_TREATMENT_SCORE,
            "margin": R1_MARGIN,
        },
        "rounds": [],
    }

    # ============ STEP 0: reproduce round 1 arm B ============
    log("STEP 0: reproducing round 1 arm B for its species "
        "bests (donor source for round 2)")
    r1 = json.loads(R1_JSON.read_text())
    orig_champion = np.asarray(
        np.load(RUN_DIR / "champion.npz")["genome"],
        dtype=float)
    orig_read = density_read(domain, layout, orig_champion)
    assert abs(orig_read["champion_score"]
               - R1_SUBJECT_SCORE) < 1e-12
    # round 1 donors: original run's checkpoint best genomes
    best_genomes = np.asarray(ck["best_genomes"], dtype=float)
    species_ids = [int(x) for x in ck["species_ids"]]
    vault_z = np.load(
        RUN_DIR / "vault" /
        "stage_champion_stage0_sp8_86970669.npz",
        allow_pickle=True)
    r1_donor_bests = [
        {"species_id": gid, "genome": genome}
        for gid, genome in zip(species_ids, best_genomes)]
    r1_donor_bests.append({
        "species_id": int(vault_z["species_id"]),
        "genome": np.asarray(vault_z["genome"], dtype=float)})
    pruned_base, inventory, donor_genome_by_id, unit_by_key, \
        eligible = prune_and_inventory(
            domain, layout, orig_champion, orig_read,
            r1_donor_bests)
    assert inventory["n_pruned_slots"] == \
        r1["prune"]["n_units_pruned"], inventory["n_pruned_slots"]
    assert inventory["n_eligible_slots"] == \
        r1["graft_inventory"]["n_eligible_slots"]
    assert abs(inventory["pruned_base_score"]
               - r1["prune"]["pruned_base_score"]) < 1e-12
    log(f"round 1 prune reproduced: "
        f"{inventory['n_pruned_slots']} pruned, "
        f"{inventory['n_eligible_slots']} eligible, pruned "
        f"base {inventory['pruned_base_score']:.6f}")
    seeds1 = round_seeds(1)
    species_b, rng_b, variants = build_treatment_species(
        domain, layout, cfg, classes, pv_a, pruned_base,
        unit_by_key, eligible, donor_genome_by_id, seeds1)
    for got, want in zip(
            variants,
            r1["graft_inventory"]["species_variants"]):
        assert got["n_grafts"] == want["n_grafts"]
        assert abs(got["variant_base_val_score"]
                   - want["variant_base_val_score"]) < 1e-12
    t0 = time.time()
    out_b = evolve(domain, layout, cfg, classes, species_b,
                   rng_b)
    wall_b = time.time() - t0
    repro_score = out_b["champion_val_score"]
    log(f"round 1 arm B reproduced: champion "
        f"{repro_score:.15f} (recorded "
        f"{R1_TREATMENT_SCORE:.15f}), species "
        f"{out_b['champion_species_id']}, {wall_b:.1f}s")
    if abs(repro_score - R1_TREATMENT_SCORE) > 1e-12:
        results["step0_reproduction"] = {
            "status": "FAILED",
            "reproduced_champion_score": float(repro_score),
            "recorded_champion_score": R1_TREATMENT_SCORE,
            "abs_difference": float(abs(
                repro_score - R1_TREATMENT_SCORE)),
        }
        results["stopped_reason"] = (
            "round 1 arm B did NOT reproduce exactly; "
            "rounds 2+ not run (per the task's instruction)")
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log("STOPPED: reproduction mismatch")
        print("PRUNE_GRAFT_ROUNDS_DONE")
        return
    results["step0_reproduction"] = {
        "status": "exact",
        "reproduced_champion_score": float(repro_score),
        "recorded_champion_score": R1_TREATMENT_SCORE,
        "champion_species_id": out_b["champion_species_id"],
        "wall_seconds": wall_b,
        "n_species_bests_captured":
            len(out_b["species_bests"]),
    }
    save_species_bests(
        HERE / "prune_graft_rounds_2026-11-23_r1_"
               "treatment_species_bests.npz",
        out_b["species_bests"])

    # ============ ROUNDS 2..5 ============
    current_champion = np.asarray(
        np.load(HERE / "prune_graft_test_2026-11-23_"
                       "armB_champion.npz")["genome"],
        dtype=float)
    assert np.array_equal(current_champion,
                          out_b["champion_genome"])
    prev_species_bests = out_b["species_bests"]
    stop_reason = "round 5 reached (cap)"
    for r in range(2, MAX_ROUND + 1):
        seeds = round_seeds(r)
        log(f"===== ROUND {r} (seeds {seeds}) =====")
        start_score = domain.validation_score(
            layout, current_champion)
        read = density_read(domain, layout, current_champion)
        assert abs(read["champion_score"] - start_score) < 1e-9
        log(f"start champion {start_score:.6f}; density "
            f"exact-zero {read['exact_zero_positive']}/"
            f"{read['n_units']}, band {read['band_counts']}")
        donor_bests = [{"species_id": b["species_id"],
                        "genome": b["genome"]}
                       for b in prev_species_bests]
        pruned_base, inventory, donor_genome_by_id, \
            unit_by_key, eligible = prune_and_inventory(
                domain, layout, current_champion, read,
                donor_bests)
        log(f"prune: {inventory['n_pruned_slots']} slots, "
            f"pruned base {inventory['pruned_base_score']:.6f}; "
            f"donors {inventory['donor_species_ids']}; "
            f"eligible {inventory['n_eligible_slots']}")
        rec = {
            "round": r,
            "seeds": seeds,
            "start_champion_score": float(start_score),
            "start_density_exact_zero_positive":
                read["exact_zero_positive"],
            "start_density_band_counts": read["band_counts"],
            "graft_inventory": inventory,
        }
        if not eligible:
            rec["status"] = "graft inventory empty; arms " \
                            "not run"
            results["rounds"].append(rec)
            stop_reason = (f"round {r}: graft inventory "
                           f"empty")
            log(f"ROUND {r}: graft inventory EMPTY; stopping")
            break

        # treatment arm
        species_t, rng_t, variants = build_treatment_species(
            domain, layout, cfg, classes, pv_a, pruned_base,
            unit_by_key, eligible, donor_genome_by_id, seeds)
        rec["species_variants"] = variants
        t0 = time.time()
        out_t = evolve(domain, layout, cfg, classes,
                       species_t, rng_t)
        rec["treatment_wall_seconds"] = time.time() - t0
        # control arm
        rng_c = np.random.default_rng(seeds["run_seed"])
        draw_rng_c = np.random.default_rng(seeds["draw_seed"])
        species_c = be.create_species(
            current_champion, classes, cfg.n_species,
            cfg.per_species, draw_rng_c, rng_c, pv_a, cfg,
            lineage=None, init_mode="transplant", next_id=0)
        t0 = time.time()
        out_c = evolve(domain, layout, cfg, classes,
                       species_c, rng_c)
        rec["control_wall_seconds"] = time.time() - t0

        margin = out_t["champion_val_score"] \
            - out_c["champion_val_score"]
        t_read = density_read(domain, layout,
                              out_t["champion_genome"])
        rec.update({
            "status": "completed",
            "control_final_score":
                float(out_c["champion_val_score"]),
            "control_champion_species_id":
                out_c["champion_species_id"],
            "treatment_final_score":
                float(out_t["champion_val_score"]),
            "treatment_champion_species_id":
                out_t["champion_species_id"],
            "treatment_margin_over_control": float(margin),
            "treatment_delta_vs_start":
                float(out_t["champion_val_score"]
                      - start_score),
            "control_delta_vs_start":
                float(out_c["champion_val_score"]
                      - start_score),
            "treatment_density_exact_zero_positive":
                t_read["exact_zero_positive"],
            "treatment_density_band_counts":
                t_read["band_counts"],
            "treatment_density_units": t_read["units"],
        })
        np.savez(
            HERE / f"prune_graft_rounds_2026-11-23_r{r}_"
                   f"treatment_champion.npz",
            genome=out_t["champion_genome"],
            val_score=out_t["champion_val_score"])
        np.savez(
            HERE / f"prune_graft_rounds_2026-11-23_r{r}_"
                   f"control_champion.npz",
            genome=out_c["champion_genome"],
            val_score=out_c["champion_val_score"])
        save_species_bests(
            HERE / f"prune_graft_rounds_2026-11-23_r{r}_"
                   f"treatment_species_bests.npz",
            out_t["species_bests"])
        results["rounds"].append(rec)
        log(f"ROUND {r}: control "
            f"{out_c['champion_val_score']:.6f}, treatment "
            f"{out_t['champion_val_score']:.6f}, margin "
            f"{margin:+.6f}; treatment density exact-zero "
            f"{t_read['exact_zero_positive']}/36, band "
            f"{t_read['band_counts']}")
        current_champion = out_t["champion_genome"]
        prev_species_bests = out_t["species_bests"]
        if margin < MARGIN_FLOOR:
            stop_reason = (f"round {r}: margin {margin:+.4f} "
                           f"below the +0.01 floor")
            log(f"stopping: {stop_reason}")
            break
    results["stopped_reason"] = stop_reason
    final_read = density_read(domain, layout, current_champion)
    results["final_champion"] = {
        "score": float(domain.validation_score(
            layout, current_champion)),
        "mean_gap_percent": float(-100.0 *
            domain.validation_score(layout, current_champion)),
        "density_exact_zero_positive":
            final_read["exact_zero_positive"],
        "density_band_counts": final_read["band_counts"],
        "n_units": final_read["n_units"],
    }
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    log(f"DONE in {results['wall_seconds']:.1f}s; "
        f"stopped: {stop_reason}")
    print("PRUNE_GRAFT_ROUNDS_DONE")


# ---------------------------------------------------------------
# summary page: rounds table + gap trajectory + final density
# (bars/counts only, no per-neuron tiles)
# ---------------------------------------------------------------
def _gap(score):
    return f"{-100.0 * score:.2f}%"


def _bar(pct, colour):
    return (f"<div class='bar'><div style='height:100%;"
            f"width:{max(0.0, min(100.0, pct)):.1f}%;"
            f"background:{colour}'></div></div>")


def write_html(results):
    ref = results["round_1_reference"]
    rows = [f"<tr><td>1</td><td>{ref['subject_score']:+.4f} "
            f"({_gap(ref['subject_score'])})</td><td>-0.4106 "
            f"(41.06%)</td><td>{ref['control_final_score']:+.4f} "
            f"({_gap(ref['control_final_score'])})</td>"
            f"<td>{ref['treatment_final_score']:+.4f} "
            f"({_gap(ref['treatment_final_score'])})</td>"
            f"<td><b>{ref['margin']:+.4f}</b></td></tr>"]
    traj = [("subject (before round 1)", ref["subject_score"]),
            ("after round 1 (prune-graft)",
             ref["treatment_final_score"])]
    for rec in results.get("rounds", []):
        if rec.get("status") == "completed":
            inv = rec["graft_inventory"]
            rows.append(
                f"<tr><td>{rec['round']}</td>"
                f"<td>{rec['start_champion_score']:+.4f} "
                f"({_gap(rec['start_champion_score'])})</td>"
                f"<td>{inv['pruned_base_score']:+.4f} "
                f"({_gap(inv['pruned_base_score'])})</td>"
                f"<td>{rec['control_final_score']:+.4f} "
                f"({_gap(rec['control_final_score'])})</td>"
                f"<td>{rec['treatment_final_score']:+.4f} "
                f"({_gap(rec['treatment_final_score'])})</td>"
                f"<td><b>{rec['treatment_margin_over_control']:+.4f}"
                f"</b></td></tr>")
            traj.append((f"after round {rec['round']} "
                         f"(prune-graft)",
                         rec["treatment_final_score"]))
        else:
            rows.append(
                f"<tr><td>{rec['round']}</td>"
                f"<td>{rec['start_champion_score']:+.4f}</td>"
                f"<td colspan='4'>{htmlmod.escape(rec.get('status', ''))}"
                f"</td></tr>")
    max_gap = max(-s for _, s in traj) or 1.0
    traj_rows = "".join(
        f"<tr><td>{htmlmod.escape(label)}</td>"
        f"<td>{score:+.4f}</td><td>{_gap(score)}</td>"
        f"<td>{_bar(100.0 * (-score) / max_gap, '#c0392b')}</td>"
        f"</tr>" for label, score in traj)
    fin = results.get("final_champion")
    fin_html = ""
    if fin:
        total = fin["n_units"]
        cat_rows = ""
        for cat, colour in (("contributing", "#c0392b"),
                            ("neutral", "#7f9db3"),
                            ("harmful", "#5da85d")):
            n = fin["density_band_counts"].get(cat, 0)
            cat_rows += (
                f"<tr><td>{cat}</td><td>{n}</td>"
                f"<td>{_bar(100.0 * n / total, colour)}</td>"
                f"</tr>")
        fin_html = f"""
<h2>Final treatment champion</h2>
<p class="small">Score {fin['score']:+.4f}, mean gap
{fin['mean_gap_percent']:.2f}%. Exact-zero density:
{fin['density_exact_zero_positive']} of {total} units with
silencing drop &gt; 0. Band read (paired-noise, 2 SE):</p>
<table><tr><th>category</th><th>units</th><th></th></tr>
{cat_rows}</table>"""
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Iterated prune-and-graft rounds 2026-11-23</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px;
       color: #1a1a1a; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px;
       margin-top: 32px; }}
.small {{ font-size: 12px; color: #444; max-width: 920px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #ccc; padding: 3px 8px; }}
.bar {{ width: 160px; height: 10px; background: #eee;
       border: 1px solid #bbb; }}
</style></head><body>
<h1>Prune-and-graft, iterated: rounds 1-5</h1>
<p class="small">Same champion lineage, same fixed
200-problem TSP validation set throughout. Scores are
validation scores (higher is better); gap is the mean tour
gap in percent (lower is better). Each round: prune the
lineage champion, graft proven donor units from the previous
round's treatment species into the pruned slots, evolve
50 generations; control arm evolves the same starting
champion unpruned for the same budget. Margin = treatment
final minus control final. Rounds continue while the margin
is at least +0.01. Stopped: {htmlmod.escape(results.get('stopped_reason', ''))}</p>
<h2>Rounds</h2>
<table><tr><th>round</th><th>start champion</th>
<th>pruned base</th><th>control final</th>
<th>treatment final</th><th>margin</th></tr>
{''.join(rows)}</table>
<h2>Champion gap trajectory (prune-graft lineage)</h2>
<table><tr><th>point</th><th>score</th><th>mean gap</th>
<th></th></tr>{traj_rows}</table>
{fin_html}
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
