#!/usr/bin/env python3
"""PRUNE-GRAFT THEN WIDEN chain test (2026-11-23), continuing
widen_test_2026-11-23.py and prune_graft_rounds_2026-11-23.py.
Standalone experiment, NOT an engine integration. Same
domain/config (bert_engine_config_test.json, fixed
200-problem TSP validation set, numpy backend, CPU x64),
same primitives evolution loop (be.create_species /
breed_species / cull_species; Schedule A; species best-ever
on train scores; stage champion = validation argmax over
species bests), same graft construction as the rounds
script, same transplant/gate protocol as the widen script.

SUBJECT: the widen test's plain-widen (arm A) champion
(widen_test_2026-11-23_armA_champion.npz, val
-0.09816553598475075 = 9.82% mean gap; d 12, FFN 16,
2 blocks, 2 heads, 3,620 genes). Its table from the widen
test: exact-zero 35/36 positive-drop units; band read
27 contributing / 9 neutral / 0 harmful.

FREEZE RULE (sponsor, 2026-11-23), active throughout:
units marked useful (band-contributing on the current
champion's table) have their genes at mutation rate 0 and
are exempt from re-silencing while inherited verbatim.
Marks are recomputed from each new champion's table as
the test proceeds (the widen test's prep pattern).

STEP 1, prune-graft round on the subject (rounds 1-2
rules): silence only the non-exempt units (the 9 the
subject's table reads as non-contributing; the 27 marked
useful are exempt, their drops carried from the widen
test's table for this identical genome); prune = zero
the outgoing weights of units with drop <= 0; donors =
the plain-widen arm's species best genomes. The widen
test saved only its champions, so the arm is first
RE-RUN with its recorded seeds (run 20261500, draw
20261501) to capture the species bests, and the rerun
must reproduce the saved champion bit-identically or
the test stops and reports. Donor eligibility: unit's
drop in its own donor > 0; grafts position-preserving,
all of the unit's genes copied. Treatment: 12 species
x 10 grafted variants (random graft subsets, fractions
25%..100% of eligible slots) + standard mutation, 50
generations (phase A 25 + phase B 25, Schedule A),
freeze marks from the subject's table applied. Control:
same budget from the unpruned subject, same marks.

STEP 2, widen the step-1 treatment winner (plain widen
only): be.make_transplant widen variant A, widen_step
= cfg.widen_step = 4 (d 12 -> 16). PRESERVATION GATE
FIRST (domain.preservation, exact; representative draw
from step-2 draw seed + 999, the widen test's protocol);
gate rejection ends step 2 with step 1 as the final
result. If it passes: 25 generations new genes only,
then 25 generations whole network, old genes at 1/10
rate, useful-marked old genes at 0 (marks from the
step-1 winner's table, carried through the transplant
gene map). Control: the step-1 winner evolved plainly
50 generations with the same marks.

SEEDS: the widen test used run 20261500 / draw 20261501.
Step 1 advances the family one step (run 20261600, draw
20261601, graft 20261604); step 2 one more (run 20261700,
draw 20261701). Within a step, each arm freshly
initialises its run/draw RNGs from the step's seeds.

Approximations, plainly: evolution is the primitives
loop, not the runner (the runner cannot be reused whole
for one-off arm comparisons); checkpoint / monitor /
cataclysm omitted (none applied in the earlier arms
either); the archive usefulness tracker is omitted in
the donor-capture rerun -- it consumes no RNG and does
not modify populations in the widen loop, so the rerun
trajectory is unaffected.
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
WIDEN_JSON = HERE / "widen_test_2026-11-23.json"
ROUNDS_JSON = HERE / "prune_graft_rounds_2026-11-23.json"
SUBJECT_PATH = HERE / "widen_test_2026-11-23_armA_champion.npz"
R2_CONTROL_PATH = HERE / ("prune_graft_rounds_2026-11-23_r2_"
                          "control_champion.npz")
OUT_JSON = HERE / "prune_widen_chain_2026-11-23.json"
OUT_HTML = HERE / "prune_widen_chain_2026-11-23.html"
GENS_PER_PHASE = 25

SUBJECT_STORED_VAL = -0.09816553598475075
R2_CONTROL_VAL = -0.10783267537761247
# widen test's own seeds (donor-capture rerun) and the
# per-step seed family (one +100 step per chain step)
WIDEN_RUN_SEED, WIDEN_DRAW_SEED = 20261500, 20261501
STEP1_SEEDS = {"run_seed": 20261600, "draw_seed": 20261601,
               "graft_seed": 20261604}
STEP2_SEEDS = {"run_seed": 20261700, "draw_seed": 20261701,
               "graft_seed": 20261704}


def log(msg):
    print(f"[pw-chain] {msg}", flush=True)


# ---------------------------------------------------------------
# density read: copied from widen_test_2026-11-23.py
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


def density_read_partial(domain, layout, genome, measure_names):
    """Silencing drops (with paired-noise SEs) for ONLY the
    named units; the freeze rule exempts the rest from
    re-measurement. Returns {name: unit-dict}."""
    unit_list = [u for u in be.silence_units(layout)
                 if u["name"] in measure_names]
    assert len(unit_list) == len(measure_names), \
        (len(unit_list), len(measure_names))
    G = np.tile(np.asarray(genome, dtype=float),
                (len(unit_list) + 1, 1))
    for k, u in enumerate(unit_list, start=1):
        G[k, be._silence_indices(layout, u)] = 0.0
    scores = domain.validation_scores(layout, G)
    lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
    gaps = lens / domain.optima - 1.0
    champ = float(scores[0])
    assert abs(-float(gaps[0].mean()) - champ) < 1e-9
    out = {}
    for k, u in enumerate(unit_list):
        diffs = gaps[k + 1] - gaps[0]
        drop = float(diffs.mean())
        assert abs(drop - float(scores[0] - scores[k + 1])) \
            < 1e-9
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        band = 2.0 * se
        out[u["name"]] = {
            "name": u["name"], "kind": u["kind"],
            "couple": u["couple"], "block": u["block"],
            "index": u["index"], "drop": drop, "se": se,
            "band": band,
            "cat": ("contributing" if drop > band
                    else "harmful" if drop < -band
                    else "neutral")}
    return out


def unit_key(u):
    return (u["couple"], u["block"], u["kind"], u["index"])


# ---------------------------------------------------------------
# evolution: the widen test's primitives loop (caller-supplied
# per-phase prob vectors), extended to also return the final
# species' best genomes (the rounds script's extension)
# ---------------------------------------------------------------
def evolve(domain, layout, cfg, classes, species, rng,
           pv_by_phase, gens_per_phase=GENS_PER_PHASE):
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
                pv = pv_by_phase[phase]
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


def prob_vectors(classes, cfg, frozen_mask):
    """Copied from widen_test_2026-11-23.py: class prob
    vectors with the freeze rule on top (useful-marked
    genes at rate 0 in both phases)."""
    out = {}
    for phase in ("A", "B"):
        pv = be.prob_vector(classes, phase, cfg, marks=None)
        pv = pv.copy()
        pv[frozen_mask] = 0.0
        out[phase] = pv
    return out


def entries_of(layout):
    return [(n, sh) for n, _, sh in layout.entries]


def frozen_mask_from_names(layout, names):
    mask = np.zeros(layout.n_params, dtype=bool)
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for n in names:
        mask[be.unit_gene_indices(layout, by_name[n])] = True
    return mask


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


# ---------------------------------------------------------------
# graft construction: copied from prune_graft_rounds_
# 2026-11-23.py (build_treatment_species, prune_and_inventory)
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
    (pruned_base, inventory dict, donor_genome_by_id,
     unit_by_key, eligible)."""
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


def arm_block(out, after, start_score, wall):
    return {
        "status": "evolved",
        "final_champion_val_score": out["champion_val_score"],
        "mean_gap_percent":
            float(-100.0 * out["champion_val_score"]),
        "delta_vs_start":
            float(out["champion_val_score"] - start_score),
        "champion_species_id": out["champion_species_id"],
        "n_species_final": out["n_species_final"],
        "wall_seconds": wall,
        "density_exact_zero_positive":
            after["exact_zero_positive"],
        "density_band_counts": after["band_counts"],
        "density_units": after["units"],
        "history": out["history"],
    }


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)
    widen = json.loads(WIDEN_JSON.read_text())
    rounds = json.loads(ROUNDS_JSON.read_text())

    ck = np.load(RUN_DIR / "checkpoint.npz", allow_pickle=True)
    arch8 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout8 = be.Layout(arch8, tsp.head_spec(arch8.d))

    results = {
        "experiment": "prune-graft then widen chain test "
                      "(standalone, not an engine integration)",
        "date": "2026-11-23",
        "continues": [str(WIDEN_JSON), str(ROUNDS_JSON)],
        "settings": {
            "config_file": str(CFG_PATH),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "seeds": {
                "donor_capture_rerun": {
                    "run_seed": WIDEN_RUN_SEED,
                    "draw_seed": WIDEN_DRAW_SEED},
                "step1": STEP1_SEEDS,
                "step2": STEP2_SEEDS,
                "step2_gate_draw_seed":
                    STEP2_SEEDS["draw_seed"] + 999,
                "val_seed": cfg.val_seed},
            "seed_rule": "donor-capture rerun uses the widen "
                         "test's own recorded seeds; step 1 "
                         "advances the family one +100 step "
                         "past the widen test (20261500/"
                         "20261501), step 2 one more; within "
                         "a step each arm freshly initialises "
                         "its run/draw RNGs; the step-2 gate "
                         "representative draw uses step-2 "
                         "draw seed + 999 (the widen test's "
                         "protocol)",
            "widen_step": cfg.widen_step,
            "rates": {"mut_p": cfg.mut_p,
                      "old_rate_factor": cfg.old_rate_factor,
                      "phase_B_old_rate":
                          cfg.mut_p * cfg.old_rate_factor},
            "freeze_rule": "units whose silencing drop "
                           "clears the paired-noise band "
                           "upward (contributing) on the "
                           "current champion's table are "
                           "marked useful; their genes mutate "
                           "at rate 0 and they are exempt "
                           "from re-silencing while "
                           "inherited verbatim; marks "
                           "recomputed from each new "
                           "champion's table",
            "classes_rule": "step 1 arms: be._base_classes"
                            "(layout, 0) (prune-graft "
                            "continuation classes) with "
                            "freeze marks. Step 2 widened "
                            "arm: the transplant's own "
                            "classes from be.make_transplant; "
                            "step 2 control: _base_classes "
                            "with the same marks",
            "operators": {
                "tournament": cfg.tournament,
                "elite": cfg.elite,
                "mut_sigma": cfg.mut_sigma,
                "cull_interval": cfg.cull_interval,
                "cull_fraction": cfg.cull_fraction,
                "species_floor": cfg.species_floor,
                "chunk": cfg.chunk,
                "backend": cfg.backend,
                "precision": cfg.precision,
            },
        },
    }

    # ============ DONOR CAPTURE: re-run widen arm A ============
    log("DONOR CAPTURE: re-running the widen test's plain-"
        "widen arm with its recorded seeds to capture its "
        "species bests")
    r2_z = np.load(R2_CONTROL_PATH)
    r2_genome = np.asarray(r2_z["genome"], dtype=float)
    assert abs(float(r2_z["val_score"]) - R2_CONTROL_VAL) < 1e-12
    repro8 = domain.validation_score(layout8, r2_genome)
    assert abs(repro8 - R2_CONTROL_VAL) < 1e-9
    read8 = density_read(domain, layout8, r2_genome)
    useful8 = [u["name"] for u in read8["units"]
               if u["cat"] == "contributing"]
    frozen8 = frozen_mask_from_names(layout8, useful8)
    assert len(useful8) == widen["subject"]["marks"][
        "n_useful_units"]
    assert int(frozen8.sum()) == widen["subject"]["marks"][
        "n_frozen_genes"], int(frozen8.sum())
    log(f"d=8 subject prep reproduced: {len(useful8)} useful "
        f"units, {int(frozen8.sum())} frozen genes")

    arch_a, layout_a, template_a, classes_a = \
        be.make_transplant("widen", arch8, layout8, r2_genome,
                           tsp.head_spec, cfg, tsp.widen_head,
                           variant="A")
    frozen_a = be.remap_vector_by_entries(
        frozen8.astype(float), entries_of(layout8),
        entries_of(layout_a)) > 0.5
    assert int(frozen_a.sum()) == widen["arms"][
        "A_plain_widen"]["n_frozen_genes_carried"]
    pv_widen = prob_vectors(classes_a, cfg, frozen_a)
    rng = np.random.default_rng(WIDEN_RUN_SEED)
    draw_rng = np.random.default_rng(WIDEN_DRAW_SEED)
    species = be.create_species(
        template_a, classes_a, cfg.n_species, cfg.per_species,
        draw_rng, rng, pv_widen["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_w = evolve(domain, layout_a, cfg, classes_a, species,
                   rng, pv_widen)
    wall_w = time.time() - t0
    subj_z = np.load(SUBJECT_PATH)
    subject = np.asarray(subj_z["genome"], dtype=float)
    assert abs(float(subj_z["val_score"])
               - SUBJECT_STORED_VAL) < 1e-12
    bit_identical = bool(np.array_equal(
        out_w["champion_genome"], subject))
    score_exact = bool(out_w["champion_val_score"]
                       == SUBJECT_STORED_VAL)
    results["donor_capture"] = {
        "arm": "widen test arm A (plain widen), re-run with "
               "its recorded seeds to capture species bests "
               "(the widen test saved only its champions)",
        "seeds": {"run_seed": WIDEN_RUN_SEED,
                  "draw_seed": WIDEN_DRAW_SEED},
        "reproduced_champion_val_score":
            out_w["champion_val_score"],
        "recorded_champion_val_score": SUBJECT_STORED_VAL,
        "champion_genome_bit_identical": bit_identical,
        "champion_score_exact": score_exact,
        "champion_species_id": out_w["champion_species_id"],
        "n_species_bests_captured":
            len(out_w["species_bests"]),
        "wall_seconds": wall_w,
    }
    log(f"widen arm A rerun: champion "
        f"{out_w['champion_val_score']:.15f} (recorded "
        f"{SUBJECT_STORED_VAL:.15f}); bit-identical "
        f"{bit_identical}; {len(out_w['species_bests'])} "
        f"species bests captured; {wall_w:.1f}s")
    if not (bit_identical and score_exact):
        results["stopped_reason"] = (
            "donor-capture rerun did NOT reproduce the "
            "widen arm A champion bit-identically; steps "
            "1-2 not run (per the task's instruction)")
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log("STOPPED: reproduction mismatch")
        print("PRUNE_WIDEN_CHAIN_DONE")
        return
    save_species_bests(
        HERE / "prune_widen_chain_2026-11-23_widen_armA_"
               "species_bests.npz",
        out_w["species_bests"])
    donor_bests = [{"species_id": b["species_id"],
                    "genome": b["genome"]}
                   for b in out_w["species_bests"]]

    # ============ SUBJECT (d=12 layout from here on) ============
    subj_val = domain.validation_score(layout_a, subject)
    assert abs(subj_val - SUBJECT_STORED_VAL) < 1e-9
    arm_a_rec = widen["arms"]["A_plain_widen"]
    results["subject"] = {
        "file": str(SUBJECT_PATH),
        "description": "widen test plain-widen (arm A) "
                       "champion",
        "arch_signature": arch_a.signature(),
        "n_genes": int(layout_a.n_params),
        "stored_val_score": float(subj_z["val_score"]),
        "reproduced_val_score": float(subj_val),
        "mean_gap_percent": float(-100.0 * subj_val),
        "density_exact_zero_positive":
            arm_a_rec["density_exact_zero_positive"],
        "density_band_counts": arm_a_rec["density_band_counts"],
        "table_source": "widen_test_2026-11-23.json arms."
                        "A_plain_widen.density_units "
                        "(measured on this identical genome "
                        "in the widen test)",
    }
    log(f"subject val reproduced: {subj_val:.6f} "
        f"({-100.0 * subj_val:.2f}% mean gap), "
        f"{layout_a.n_params} genes, arch {arch_a.signature()}")

    # ============ STEP 1 PREP: partial silencing + marks ====
    table_units = arm_a_rec["density_units"]
    useful1 = [u["name"] for u in table_units
               if u["cat"] == "contributing"]
    measure_names = [u["name"] for u in table_units
                     if u["cat"] != "contributing"]
    assert len(useful1) + len(measure_names) == len(table_units)
    measured = density_read_partial(domain, layout_a, subject,
                                    set(measure_names))
    matches = all(
        abs(measured[u["name"]]["drop"] - u["drop"]) < 1e-9
        for u in table_units if u["name"] in measured)
    synth_units = []
    for u in table_units:
        if u["name"] in measured:
            synth_units.append(measured[u["name"]])
        else:
            synth_units.append(u)
    synth_read = {"units": synth_units}
    frozen1 = frozen_mask_from_names(layout_a, useful1)
    results["step1_prep"] = {
        "n_units_total": len(table_units),
        "n_units_exempt_freeze_rule": len(useful1),
        "exempt_units": useful1,
        "n_units_measured": len(measure_names),
        "measured_units": [
            {k: measured[n][k] for k in
             ("name", "drop", "se", "band", "cat")}
            for n in measure_names],
        "partial_read_matches_widen_table": bool(matches),
        "measurement_note": "under the freeze rule only "
            "the non-exempt units were re-silenced, on "
            "this identical genome; exempt units' drops "
            "are carried from the widen test's table "
            "(same genome, same fixed validation set)",
        "marks": {
            "useful_units": useful1,
            "n_useful_units": len(useful1),
            "n_frozen_genes": int(frozen1.sum())},
    }
    log(f"STEP 1 prep: {len(useful1)} exempt (useful-marked), "
        f"{len(measure_names)} measured; partial read matches "
        f"the widen table: {matches}; frozen genes "
        f"{int(frozen1.sum())}")

    # ============ STEP 1: prune-graft round ============
    log("STEP 1: prune-graft round on the widened champion")
    pruned_base, inventory, donor_genome_by_id, unit_by_key, \
        eligible = prune_and_inventory(
            domain, layout_a, subject, synth_read, donor_bests)
    log(f"prune: {inventory['n_pruned_slots']} slots "
        f"{inventory['pruned_units']}, pruned base "
        f"{inventory['pruned_base_score']:.6f}; donors "
        f"{inventory['donor_species_ids']}; eligible "
        f"{inventory['n_eligible_slots']}")
    classes1 = be._base_classes(layout_a, 0)
    pv1 = prob_vectors(classes1, cfg, frozen1)
    step1 = {"seeds": STEP1_SEEDS, "graft_inventory": inventory}

    species_t, rng_t, variants = build_treatment_species(
        domain, layout_a, cfg, classes1, pv1["A"], pruned_base,
        unit_by_key, eligible, donor_genome_by_id, STEP1_SEEDS)
    step1["species_variants"] = variants
    t0 = time.time()
    out_t1 = evolve(domain, layout_a, cfg, classes1, species_t,
                    rng_t, pv1)
    wall_t1 = time.time() - t0
    after_t1 = density_read(domain, layout_a,
                            out_t1["champion_genome"])
    step1["treatment"] = arm_block(out_t1, after_t1, subj_val,
                                   wall_t1)
    np.savez(HERE / "prune_widen_chain_2026-11-23_step1_"
                    "treatment_champion.npz",
             genome=out_t1["champion_genome"],
             val_score=out_t1["champion_val_score"])
    save_species_bests(
        HERE / "prune_widen_chain_2026-11-23_step1_"
               "treatment_species_bests.npz",
        out_t1["species_bests"])

    rng_c = np.random.default_rng(STEP1_SEEDS["run_seed"])
    draw_rng_c = np.random.default_rng(STEP1_SEEDS["draw_seed"])
    species_c = be.create_species(
        subject, classes1, cfg.n_species, cfg.per_species,
        draw_rng_c, rng_c, pv1["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c1 = evolve(domain, layout_a, cfg, classes1, species_c,
                    rng_c, pv1)
    wall_c1 = time.time() - t0
    after_c1 = density_read(domain, layout_a,
                            out_c1["champion_genome"])
    step1["control"] = arm_block(out_c1, after_c1, subj_val,
                                 wall_c1)
    np.savez(HERE / "prune_widen_chain_2026-11-23_step1_"
                    "control_champion.npz",
             genome=out_c1["champion_genome"],
             val_score=out_c1["champion_val_score"])
    step1["margin_treatment_over_control"] = float(
        out_t1["champion_val_score"] - out_c1["champion_val_score"])
    results["step1"] = step1
    log(f"STEP 1: control {out_c1['champion_val_score']:.6f}, "
        f"treatment {out_t1['champion_val_score']:.6f}, margin "
        f"{step1['margin_treatment_over_control']:+.6f}; "
        f"treatment density exact-zero "
        f"{after_t1['exact_zero_positive']}/{after_t1['n_units']}, "
        f"band {after_t1['band_counts']}")

    # ============ STEP 2: widen the step-1 winner ============
    log("STEP 2: plain widen of the step-1 treatment winner "
        "(d 12 -> 16)")
    winner = out_t1["champion_genome"]
    winner_val = out_t1["champion_val_score"]
    useful2 = [u["name"] for u in after_t1["units"]
               if u["cat"] == "contributing"]
    frozen2 = frozen_mask_from_names(layout_a, useful2)
    step2 = {
        "seeds": STEP2_SEEDS,
        "start_champion_val_score": float(winner_val),
        "start_mean_gap_percent": float(-100.0 * winner_val),
        "marks_from_step1_winner_table": {
            "useful_units": useful2,
            "n_useful_units": len(useful2),
            "n_frozen_genes": int(frozen2.sum())},
    }
    arch_16, layout_16, template_16, classes_16 = \
        be.make_transplant("widen", arch_a, layout_a, winner,
                           tsp.head_spec, cfg, tsp.widen_head,
                           variant="A")
    rng0 = np.random.default_rng(STEP2_SEEDS["draw_seed"] + 999)
    draw_mask_16 = classes_16 == be.CLS_DRAW
    rep_16 = template_16.copy()
    rep_16[draw_mask_16] = rng0.normal(
        0, cfg.draw_sigma, int(draw_mask_16.sum()))
    pres = domain.preservation(layout_a, winner, layout_16,
                               rep_16)
    step2["arch_signature_before"] = arch_a.signature()
    step2["arch_signature_after"] = arch_16.signature()
    step2["genome_size_after"] = int(layout_16.n_params)
    step2["n_draw_genes"] = int(draw_mask_16.sum())
    step2["preservation_gate"] = pres
    log(f"STEP 2 transplant: d {arch_a.d} -> {arch_16.d}, "
        f"{layout_16.n_params} genes; gate "
        f"{'PASS' if pres['pass'] else 'REJECT'} "
        f"(max abs length diff "
        f"{pres['max_abs_length_difference']:.3g}, tours "
        f"identical {pres['tours_identical']})")
    if not pres["pass"]:
        step2["status"] = "gate-rejected; step 2 not evolved, "\
                          "step 1 treatment winner is the "\
                          "final champion"
        results["step2"] = step2
        results["final_champion"] = {
            "stage": "step 1 treatment (prune-graft on the "
                     "widened champion); step 2 gate-rejected",
            "val_score": float(winner_val),
            "mean_gap_percent": float(-100.0 * winner_val),
            "density_exact_zero_positive":
                after_t1["exact_zero_positive"],
            "density_band_counts": after_t1["band_counts"],
            "n_units": after_t1["n_units"],
            "genome_size": int(layout_a.n_params),
        }
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log(f"DONE in {results['wall_seconds']:.1f}s; step 2 "
            f"gate-rejected")
        print("PRUNE_WIDEN_CHAIN_DONE")
        return

    frozen_16 = be.remap_vector_by_entries(
        frozen2.astype(float), entries_of(layout_a),
        entries_of(layout_16)) > 0.5
    step2["n_frozen_genes_carried"] = int(frozen_16.sum())
    pv16 = prob_vectors(classes_16, cfg, frozen_16)
    rng2 = np.random.default_rng(STEP2_SEEDS["run_seed"])
    draw_rng2 = np.random.default_rng(STEP2_SEEDS["draw_seed"])
    species_t2 = be.create_species(
        template_16, classes_16, cfg.n_species, cfg.per_species,
        draw_rng2, rng2, pv16["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_t2 = evolve(domain, layout_16, cfg, classes_16,
                    species_t2, rng2, pv16)
    wall_t2 = time.time() - t0
    after_t2 = density_read(domain, layout_16,
                            out_t2["champion_genome"])
    step2["treatment"] = arm_block(out_t2, after_t2, winner_val,
                                   wall_t2)
    np.savez(HERE / "prune_widen_chain_2026-11-23_step2_"
                    "treatment_champion.npz",
             genome=out_t2["champion_genome"],
             val_score=out_t2["champion_val_score"])

    classes_c2 = be._base_classes(layout_a, 0)
    pv_c2 = prob_vectors(classes_c2, cfg, frozen2)
    rng_c2 = np.random.default_rng(STEP2_SEEDS["run_seed"])
    draw_rng_c2 = np.random.default_rng(STEP2_SEEDS["draw_seed"])
    species_c2 = be.create_species(
        winner, classes_c2, cfg.n_species, cfg.per_species,
        draw_rng_c2, rng_c2, pv_c2["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c2 = evolve(domain, layout_a, cfg, classes_c2,
                    species_c2, rng_c2, pv_c2)
    wall_c2 = time.time() - t0
    after_c2 = density_read(domain, layout_a,
                            out_c2["champion_genome"])
    step2["control"] = arm_block(out_c2, after_c2, winner_val,
                                 wall_c2)
    np.savez(HERE / "prune_widen_chain_2026-11-23_step2_"
                    "control_champion.npz",
             genome=out_c2["champion_genome"],
             val_score=out_c2["champion_val_score"])
    step2["margin_treatment_over_control"] = float(
        out_t2["champion_val_score"] - out_c2["champion_val_score"])
    step2["status"] = "evolved"
    results["step2"] = step2
    log(f"STEP 2: control {out_c2['champion_val_score']:.6f}, "
        f"treatment {out_t2['champion_val_score']:.6f}, margin "
        f"{step2['margin_treatment_over_control']:+.6f}; "
        f"treatment density exact-zero "
        f"{after_t2['exact_zero_positive']}/{after_t2['n_units']}, "
        f"band {after_t2['band_counts']}")

    results["final_champion"] = {
        "stage": "step 2 treatment (prune-graft, then plain "
                 "widen, on the widened champion)",
        "val_score": float(out_t2["champion_val_score"]),
        "mean_gap_percent":
            float(-100.0 * out_t2["champion_val_score"]),
        "density_exact_zero_positive":
            after_t2["exact_zero_positive"],
        "density_band_counts": after_t2["band_counts"],
        "n_units": after_t2["n_units"],
        "genome_size": int(layout_16.n_params),
    }
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("PRUNE_WIDEN_CHAIN_DONE")


# ---------------------------------------------------------------
# summary page: full lineage trajectory + per-step margins
# (counts and bars only, no per-neuron tiles)
# ---------------------------------------------------------------
def _gap(score):
    return f"{-100.0 * score:.2f}%"


def _bar(pct, colour):
    return (f"<div class='bar'><div style='height:100%;"
            f"width:{max(0.0, min(100.0, pct)):.1f}%;"
            f"background:{colour}'></div></div>")


def write_html(results):
    widen = json.loads(WIDEN_JSON.read_text())
    rounds = json.loads(ROUNDS_JSON.read_text())
    ref = rounds["round_1_reference"]
    r2 = rounds["rounds"][0]
    arm_a = widen["arms"]["A_plain_widen"]
    arm_c = widen["arms"]["C_control"]
    traj = [
        ("original mapped champion (minimal test, d 8)",
         ref["subject_score"]),
        ("prune-graft round 1 winner (d 8)",
         ref["treatment_final_score"]),
        ("round 2: lineage continued via the control arm "
         "(d 8)", r2["control_final_score"]),
        ("widen test: plain widen winner (d 12)",
         arm_a["final_champion_val_score"]),
    ]
    margins = [
        ("prune-graft round 1", ref["control_final_score"],
         ref["treatment_final_score"], ref["margin"]),
        ("prune-graft round 2 (lineage continued via "
         "control)", r2["control_final_score"],
         r2["treatment_final_score"],
         r2["treatment_margin_over_control"]),
        ("widen test, plain widen vs its control",
         arm_c["final_champion_val_score"],
         arm_a["final_champion_val_score"],
         arm_a["delta_vs_control"]),
    ]
    s1 = results.get("step1")
    if s1:
        traj.append(("chain step 1: prune-graft on the "
                     "widened champion (d 12)",
                     s1["treatment"]["final_champion_val_score"]))
        margins.append((
            "chain step 1: prune-graft",
            s1["control"]["final_champion_val_score"],
            s1["treatment"]["final_champion_val_score"],
            s1["margin_treatment_over_control"]))
    s2 = results.get("step2")
    if s2 and s2.get("status") == "evolved":
        traj.append(("chain step 2: plain widen winner "
                     "(d 16)",
                     s2["treatment"]["final_champion_val_score"]))
        margins.append((
            "chain step 2: plain widen",
            s2["control"]["final_champion_val_score"],
            s2["treatment"]["final_champion_val_score"],
            s2["margin_treatment_over_control"]))
    max_gap = max(-s for _, s in traj) or 1.0
    traj_rows = "".join(
        f"<tr><td>{htmlmod.escape(label)}</td>"
        f"<td>{score:+.4f}</td><td>{_gap(score)}</td>"
        f"<td>{_bar(100.0 * (-score) / max_gap, '#c0392b')}</td>"
        f"</tr>" for label, score in traj)
    margin_rows = "".join(
        f"<tr><td>{htmlmod.escape(label)}</td>"
        f"<td>{ctrl:+.4f} ({_gap(ctrl)})</td>"
        f"<td>{treat:+.4f} ({_gap(treat)})</td>"
        f"<td><b>{margin:+.4f}</b></td></tr>"
        for label, ctrl, treat, margin in margins)

    detail = ""
    if s1:
        inv = s1["graft_inventory"]
        prep = results["step1_prep"]
        detail += f"""
<h2>Chain step 1: prune-graft on the widened champion</h2>
<p class="small">Freeze rule: {prep['n_units_exempt_freeze_rule']}
of {prep['n_units_total']} units exempt from re-silencing
(useful-marked); {prep['n_units_measured']} measured.
Pruned slots: {inv['n_pruned_slots']}
({htmlmod.escape(', '.join(inv['pruned_units']))}); pruned
base score {inv['pruned_base_score']:+.4f}
({_gap(inv['pruned_base_score'])}). Eligible graft slots:
{inv['n_eligible_slots']}; donors (widen arm species):
{htmlmod.escape(', '.join(str(d) for d in inv['donor_species_ids']))}.
Treatment density after: exact-zero
{s1['treatment']['density_exact_zero_positive']} positive,
band {htmlmod.escape(str(s1['treatment']['density_band_counts']))};
control density: exact-zero
{s1['control']['density_exact_zero_positive']}, band
{htmlmod.escape(str(s1['control']['density_band_counts']))}.</p>"""
    if s2:
        g = s2["preservation_gate"]
        detail += f"""
<h2>Chain step 2: plain widen (d 12 to 16)</h2>
<p class="small">Preservation gate:
<b>{'PASS' if g['pass'] else 'REJECT'}</b> (max abs length
difference {g['max_abs_length_difference']:.3g}, tours
identical {g['tours_identical']}). Genome size after
widening: {s2['genome_size_after']} genes."""
        if s2.get("status") == "evolved":
            detail += f""" Treatment density after:
exact-zero {s2['treatment']['density_exact_zero_positive']}
positive, band
{htmlmod.escape(str(s2['treatment']['density_band_counts']))};
control density: exact-zero
{s2['control']['density_exact_zero_positive']}, band
{htmlmod.escape(str(s2['control']['density_band_counts']))}."""
        else:
            detail += " Gate rejected: step 2 not evolved; " \
                      "the step 1 winner is the final champion."
        detail += "</p>"
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
<h2>Final champion</h2>
<p class="small">{htmlmod.escape(fin['stage'])}. Score
{fin['val_score']:+.4f}, mean gap
{fin['mean_gap_percent']:.2f}%, genome
{fin['genome_size']} genes. Exact-zero density:
{fin['density_exact_zero_positive']} of {total} units with
silencing drop &gt; 0. Band read (paired-noise, 2 SE):</p>
<table><tr><th>category</th><th>units</th><th></th></tr>
{cat_rows}</table>"""
    stopped = results.get("stopped_reason", "")
    stopped_html = (f"<p class='small'><b>Stopped:</b> "
                    f"{htmlmod.escape(stopped)}</p>"
                    if stopped else "")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prune-graft then widen chain 2026-11-23</title>
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
<h1>Prune-graft, then widen: the full chain</h1>
<p class="small">One champion lineage, one fixed
200-problem TSP validation set throughout. Scores are
validation scores (higher is better); gap is the mean
tour gap in percent (lower is better). Chain step 1 runs
a prune-graft round on the widen test's plain-widen
winner; chain step 2 plain-widens the step-1 winner.
Every step is measured against a same-budget control arm
evolving the same starting champion without the step's
operation. The freeze rule (useful-marked units do not
mutate and are exempt from re-silencing) is active in
every arm.</p>
{stopped_html}
<h2>Champion trajectory</h2>
<table><tr><th>stage</th><th>score</th><th>mean gap</th>
<th></th></tr>{traj_rows}</table>
<h2>Per-step margins (treatment minus control)</h2>
<table><tr><th>step</th><th>control final</th>
<th>treatment final</th><th>margin</th></tr>
{margin_rows}</table>
{detail}
{fin_html}
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
