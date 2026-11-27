#!/usr/bin/env python3
"""PRUNE-AND-GRAFT round on the NEW single-layer lineage
(2026-11-23). Standalone experiment, NOT an engine
integration; follows prune_graft_test_2026-11-23.py and
prune_widen_chain_2026-11-23.py exactly (same primitives
evolution loop: be.create_species / breed_species /
cull_species, Schedule A culling, species best-ever on
train scores, stage champion = validation argmax over
species bests; same prune / donor / graft construction;
same freeze-rule marks as the chain test).

SUBJECT: the fresh single-layer champion
(engine_runs/fresh_d2_l1/champion.npz, val
-0.22026697212210763 = 22.03% mean gap; d 2, FFN 8,
1 head, 1 layer, 88 genes, 9 silence units: 8 FFN + 1
head in the single block). Config:
bert_engine_config_d2_l1.json (fixed 200-problem TSP
validation set, val_seed 20261116, numpy backend, CPU
x64). The runner's vault read said 5/9 units real; the
full silencing table and paired-noise band read are
recomputed here and the measured numbers are used.

FREEZE RULE (sponsor, 2026-11-23), as in the chain test:
units whose silencing drop clears the paired-noise band
upward (contributing) on the subject's table are marked
useful; their genes mutate at rate 0 in BOTH arms
(prob_vectors: class prob vectors with the frozen mask
zeroed on top) and are exempt from re-silencing while
inherited verbatim.

PRUNE: zero the outgoing weights (be._silence_indices)
of every unit with drop <= 0 (exact-zero criterion);
the pruned base is scored on its own.
DONORS: the other species' best genomes from the
fresh_d2_l1 checkpoint (4 species survived Schedule A:
2, 11, 0, 8; the champion is species 0's best, excluded
both by species and by genome identity). A donor unit
is graft-eligible only if its drop in its OWN donor
genome is > 0. GRAFT: position-preserving only (same
block, kind, index), copying ALL of the unit's genes
(be.unit_gene_indices). Slots with no eligible donor
stay zeroed.
TREATMENT ARM: 12 species x 10 grafted-base variants
(random graft subsets, fractions 25%..100% of eligible
slots across species) + standard mutation, 50
generations (phase A 25 + phase B 25), Schedule A,
freeze marks applied.
CONTROL ARM: identical budget/operators seeded from
the unpruned subject, freeze marks applied.

SEEDS: the fresh run's family is run 20262000 / draw
20262001; this round advances one +100 step (the chain
family's convention): run 20262100, draw 20262101,
graft 20262104. Within the step, each arm freshly
initialises its run/draw RNGs from the step seeds.

Approximations, plainly: evolution is the primitives
loop, not the runner (the runner cannot be reused whole
for one-off arm comparisons); checkpoint / monitor /
cataclysm / archive tracker omitted, as in all the
earlier arms. Classes for seeding/evolution are
be._base_classes(layout, 0), the engine's
founder-continuation rebase (see
prune_graft_test_2026-11-23.py's docstring).
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
RUN_DIR = HERE / "engine_runs" / "fresh_d2_l1"
CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
OUT_JSON = HERE / "prune_graft_d2_l1_2026-11-23.json"
OUT_HTML = HERE / "prune_graft_d2_l1_2026-11-23.html"
GENS_PER_PHASE = 25

SUBJECT_STORED_VAL = -0.22026697212210763
CHAMPION_SPECIES_ID = 0
STEP_SEEDS = {"run_seed": 20262100, "draw_seed": 20262101,
              "graft_seed": 20262104}


def log(msg):
    print(f"[pg-d2l1] {msg}", flush=True)


# ---------------------------------------------------------------
# density read: copied from prune_widen_chain_2026-11-23.py
# (silencing table + paired-noise band categories;
# reference implementation: make_usefulness_map.py)
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
# evolution: the chain test's primitives loop (caller-supplied
# per-phase prob vectors carrying the freeze marks)
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
    return {
        "champion_genome": species[bi]["best"]["genome"],
        "champion_val_score": float(val_scores[bi]),
        "champion_species_id": int(species[bi]["id"]),
        "n_species_final": len(species),
        "history": history,
    }


def prob_vectors(classes, cfg, frozen_mask):
    """Chain test's helper: class prob vectors with the
    freeze rule on top (useful-marked genes at rate 0 in
    both phases)."""
    out = {}
    for phase in ("A", "B"):
        pv = be.prob_vector(classes, phase, cfg, marks=None)
        pv = pv.copy()
        pv[frozen_mask] = 0.0
        out[phase] = pv
    return out


def frozen_mask_from_names(layout, names):
    mask = np.zeros(layout.n_params, dtype=bool)
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for n in names:
        mask[be.unit_gene_indices(layout, by_name[n])] = True
    return mask


# ---------------------------------------------------------------
# graft construction: copied from prune_widen_chain_2026-11-23.py
# (prune_and_inventory, build_treatment_species)
# ---------------------------------------------------------------
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


def arm_block(out, after, start_score, wall):
    return {
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


def margins(treatment_score, control_score):
    """Margin of treatment over control: score points, and
    as a relative percent of the control's gap (positive =
    treatment's gap is that much smaller, relatively)."""
    pts = float(treatment_score - control_score)
    gap_c = float(-control_score)
    rel = float(100.0 * pts / gap_c) if gap_c > 0 else None
    return {"points": pts,
            "relative_percent_of_control_gap": rel}


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    ck = np.load(RUN_DIR / "checkpoint.npz", allow_pickle=True)
    arch = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    champ_z = np.load(RUN_DIR / "champion.npz")
    champion = np.asarray(champ_z["genome"], dtype=float)
    stored_val = float(champ_z["val_score"])
    classes = be._base_classes(layout, 0)

    results = {
        "experiment": "prune-and-graft round on the fresh "
                      "single-layer lineage (standalone, "
                      "not an engine integration)",
        "date": "2026-11-23",
        "continues": ["prune_graft_test_2026-11-23.py",
                      "prune_widen_chain_2026-11-23.py"],
        "subject": {
            "run_dir": str(RUN_DIR),
            "arch_signature": arch.signature(),
            "n_genes": int(layout.n_params),
            "stored_val_score": stored_val,
            "champion_species_id": CHAMPION_SPECIES_ID,
        },
        "settings": {
            "config_file": str(CFG_PATH),
            "config_canonical": json.loads(cfg.canonical()),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
            "seed_rule": "the fresh run's family is run "
                         "20262000 / draw 20262001; this "
                         "round advances one +100 step "
                         "(run 20262100, draw 20262101, "
                         "graft 20262104); within the "
                         "step each arm freshly initialises "
                         "its run/draw RNGs from the step "
                         "seeds",
            "classes_rule": "be._base_classes(layout, 0): "
                            "all genes CLS_OLD except "
                            "couple 0 head CLS_DECODER; no "
                            "CLS_DRAW genes (engine "
                            "founder-continuation rebase)",
            "freeze_rule": "units whose silencing drop "
                           "clears the paired-noise band "
                           "upward (contributing) on the "
                           "subject's table are marked "
                           "useful; their genes mutate at "
                           "rate 0 in both arms and they "
                           "are exempt from re-silencing "
                           "while inherited verbatim",
            "operators": {
                "tournament": cfg.tournament,
                "elite": cfg.elite,
                "mut_p": cfg.mut_p,
                "mut_sigma": cfg.mut_sigma,
                "old_rate_factor": cfg.old_rate_factor,
                "cull_interval": cfg.cull_interval,
                "cull_fraction": cfg.cull_fraction,
                "species_floor": cfg.species_floor,
                "chunk": cfg.chunk,
                "backend": cfg.backend,
                "precision": cfg.precision,
            },
        },
    }

    # ---------------- BEFORE ----------------
    repro_val = domain.validation_score(layout, champion)
    results["subject"]["reproduced_val_score"] = float(repro_val)
    results["subject"]["mean_gap_percent"] = \
        float(-100.0 * repro_val)
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-12
    log(f"champion val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% mean gap), "
        f"{layout.n_params} genes")
    before = density_read(domain, layout, champion)
    results["before"] = before
    log(f"BEFORE density: exact-zero positive "
        f"{before['exact_zero_positive']}/{before['n_units']}; "
        f"band {before['band_counts']}")
    for u in before["units"]:
        log(f"  {u['name']}: drop {u['drop']:+.4f} "
            f"(band +/-{u['band']:.4f}) {u['cat']}")

    # ---------------- FREEZE PREP ----------------
    useful = [u["name"] for u in before["units"]
              if u["cat"] == "contributing"]
    frozen = frozen_mask_from_names(layout, useful)
    results["freeze_prep"] = {
        "useful_units": useful,
        "n_useful_units": len(useful),
        "n_frozen_genes": int(frozen.sum()),
        "n_genes_total": int(layout.n_params),
    }
    log(f"FREEZE prep: {len(useful)} useful units "
        f"{useful}; {int(frozen.sum())} of "
        f"{layout.n_params} genes at mutation rate 0")
    pv = prob_vectors(classes, cfg, frozen)

    # ---------------- DONOR POOL ----------------
    best_genomes = np.asarray(ck["best_genomes"], dtype=float)
    species_ids = [int(x) for x in ck["species_ids"]]
    donor_bests = []
    pool_record = []
    for gid, genome in zip(species_ids, best_genomes):
        is_champ_species = (gid == CHAMPION_SPECIES_ID)
        pool_record.append({
            "species_id": gid,
            "identical_to_subject": bool(
                np.array_equal(genome, champion)),
            "excluded_champion_species": is_champ_species})
        if not is_champ_species:
            donor_bests.append({"species_id": gid,
                                "genome": genome})
    results["donor_pool"] = {
        "checkpoint_species": pool_record,
        "donor_species_ids": [e["species_id"]
                              for e in donor_bests],
        "note": "4 species survived Schedule A in the "
                "fresh run (2, 11, 0, 8); the champion "
                "is species 0's best genome and is "
                "excluded as a donor both by species "
                "(per the task) and by genome identity "
                "(the protocol's own filter).",
    }
    log(f"donor pool: checkpoint species {species_ids}; "
        f"donors {[e['species_id'] for e in donor_bests]}")

    # ---------------- PRUNE + INVENTORY ----------------
    pruned_base, inventory, donor_genome_by_id, unit_by_key, \
        eligible = prune_and_inventory(
            domain, layout, champion, before, donor_bests)
    inventory["pruned_base_delta_vs_champion"] = float(
        inventory["pruned_base_score"] - repro_val)
    results["graft_inventory"] = inventory
    log(f"PRUNE: {inventory['n_pruned_slots']} slots "
        f"{inventory['pruned_units']}; pruned base "
        f"{inventory['pruned_base_score']:.6f} (delta "
        f"{inventory['pruned_base_delta_vs_champion']:+.6f}); "
        f"eligible {inventory['n_eligible_slots']}; "
        f"without donor "
        f"{inventory['slots_without_eligible_donor']}")
    for sid, tab in inventory["donor_tables"].items():
        log(f"donor sp{sid}: own val "
            f"{tab['donor_val_score']:.6f}, positive-drop "
            f"units {tab['positive_drop_units']}")

    if not eligible:
        results["verdict"] = "graft inventory EMPTY: no " \
            "donor unit with positive own-drop exists " \
            "for any pruned slot; arms not run, rules " \
            "not loosened"
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log("DONE (empty inventory)")
        print("PRUNE_GRAFT_D2_L1_DONE")
        return

    # ---------------- TREATMENT ARM ----------------
    species_t, rng_t, variants = build_treatment_species(
        domain, layout, cfg, classes, pv["A"], pruned_base,
        unit_by_key, eligible, donor_genome_by_id,
        STEP_SEEDS)
    results["species_variants"] = variants
    log("treatment variants built: grafts per species "
        f"{[v['n_grafts'] for v in variants]}")
    t0 = time.time()
    out_t = evolve(domain, layout, cfg, classes, species_t,
                   rng_t, pv)
    wall_t = time.time() - t0
    after_t = density_read(domain, layout,
                           out_t["champion_genome"])
    results["treatment"] = arm_block(out_t, after_t,
                                     repro_val, wall_t)
    np.savez(HERE / "prune_graft_d2_l1_2026-11-23_"
                    "treatment_champion.npz",
             genome=out_t["champion_genome"],
             val_score=out_t["champion_val_score"])
    log(f"TREATMENT done: final champion val "
        f"{out_t['champion_val_score']:.6f} "
        f"({-100.0 * out_t['champion_val_score']:.2f}% gap, "
        f"species {out_t['champion_species_id']}, "
        f"{out_t['n_species_final']} species left, "
        f"{wall_t:.1f}s); density exact-zero "
        f"{after_t['exact_zero_positive']}/{after_t['n_units']}, "
        f"band {after_t['band_counts']}")

    # ---------------- CONTROL ARM ----------------
    rng_c = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_c = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_c = be.create_species(
        champion, classes, cfg.n_species, cfg.per_species,
        draw_rng_c, rng_c, pv["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c = evolve(domain, layout, cfg, classes, species_c,
                   rng_c, pv)
    wall_c = time.time() - t0
    after_c = density_read(domain, layout,
                           out_c["champion_genome"])
    results["control"] = arm_block(out_c, after_c,
                                   repro_val, wall_c)
    np.savez(HERE / "prune_graft_d2_l1_2026-11-23_"
                    "control_champion.npz",
             genome=out_c["champion_genome"],
             val_score=out_c["champion_val_score"])
    log(f"CONTROL done: final champion val "
        f"{out_c['champion_val_score']:.6f} "
        f"({-100.0 * out_c['champion_val_score']:.2f}% gap, "
        f"species {out_c['champion_species_id']}, "
        f"{out_c['n_species_final']} species left, "
        f"{wall_c:.1f}s); density exact-zero "
        f"{after_c['exact_zero_positive']}/{after_c['n_units']}, "
        f"band {after_c['band_counts']}")

    results["margin_treatment_over_control"] = margins(
        out_t["champion_val_score"],
        out_c["champion_val_score"])
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    m = results["margin_treatment_over_control"]
    log(f"MARGIN treatment over control: {m['points']:+.6f} "
        f"points, {m['relative_percent_of_control_gap']:+.2f}% "
        f"relative to the control's gap")
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("PRUNE_GRAFT_D2_L1_DONE")


# ---------------------------------------------------------------
# summary page: scores, margins, density counts + bars
# (no per-neuron tiles)
# ---------------------------------------------------------------
def _gap(score):
    return f"{-100.0 * score:.2f}%"


def _bar(pct, colour):
    return (f"<div class='bar'><div style='height:100%;"
            f"width:{max(0.0, min(100.0, pct)):.1f}%;"
            f"background:{colour}'></div></div>")


def _cat_rows(label, counts, total):
    out = ""
    for cat, colour in (("contributing", "#c0392b"),
                        ("neutral", "#7f9db3"),
                        ("harmful", "#5da85d")):
        n = counts.get(cat, 0)
        out += (f"<tr><td>{label}</td><td>{cat}</td>"
                f"<td>{n}</td><td>"
                f"{_bar(100.0 * n / total if total else 0, colour)}"
                f"</td></tr>")
    return out


def write_html(results):
    before = results["before"]
    total = before["n_units"]
    subj = results["subject"]["reproduced_val_score"]
    inv = results["graft_inventory"]
    score_rows = (
        f"<tr><td>subject champion (before)</td>"
        f"<td>{subj:+.4f}</td><td>{_gap(subj)}</td></tr>"
        f"<tr><td>pruned base (no evolution)</td>"
        f"<td>{inv['pruned_base_score']:+.4f}</td>"
        f"<td>{_gap(inv['pruned_base_score'])}</td></tr>")
    rows = _cat_rows("before (subject champion)",
                     before["band_counts"], total)
    unit_rows = "".join(
        f"<tr><td>{htmlmod.escape(u['name'])}</td>"
        f"<td>{u['kind']}</td><td>{u['drop']:+.4f}</td>"
        f"<td>+/-{u['band']:.4f}</td><td>{u['cat']}</td></tr>"
        for u in before["units"])
    margin_html = ""
    for key, label in (("control", "Control (continued "
                        "evolution)"),
                       ("treatment", "Treatment "
                        "(prune-and-graft)")):
        if key in results:
            arm = results[key]
            rows += _cat_rows(
                label, arm["density_band_counts"], total)
            score_rows += (
                f"<tr><td>{label} final champion</td>"
                f"<td>{arm['final_champion_val_score']:+.4f}"
                f"</td><td>{arm['mean_gap_percent']:.2f}%"
                f"</td></tr>")
    if "margin_treatment_over_control" in results:
        m = results["margin_treatment_over_control"]
        margin_html = f"""
<h2>Margin (treatment over control)</h2>
<p class="small"><b>{m['points']:+.4f} points</b>;
{m['relative_percent_of_control_gap']:+.2f}% relative to
the control's final gap (positive means the treatment's
gap is relatively smaller).</p>"""
    donors_used = sorted({g["donor_species_id"]
                          for v in results.get(
                              "species_variants", [])
                          for g in v["grafts"]})
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prune-and-graft on the single-layer lineage
2026-11-23</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px;
       color: #1a1a1a; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px;
       margin-top: 32px; }}
.small {{ font-size: 12px; color: #444; max-width: 920px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #ccc; padding: 3px 8px; }}
.bar {{ width: 140px; height: 10px; background: #eee;
       border: 1px solid #bbb; }}
</style></head><body>
<h1>Prune-and-graft: the single-layer lineage</h1>
<p class="small">Subject: the fresh_d2_l1 champion (TSP,
d 2, FFN 8, 1 head, 1 layer, 88 genes, 9 testable units).
Scores are validation scores on the fixed 200-problem
set (higher is better); gap is the mean tour gap in
percent (lower is better). Both arms evolved 50
generations at identical budget from the same subject;
the treatment arm's species were seeded from the
pruned-and-grafted base, the control's from the
unpruned subject. The freeze rule was active in both
arms: {results['freeze_prep']['n_useful_units']} units
marked useful on the subject's table
({results['freeze_prep']['n_frozen_genes']} of
{results['freeze_prep']['n_genes_total']} genes at
mutation rate 0). Density categories come from the
paired-noise band read: contributing when the silencing
drop clears +2 paired SE, harmful below -2 SE, neutral
in between.</p>
<h2>Scores</h2>
<table><tr><th>genome</th><th>validation score</th>
<th>mean gap</th></tr>{score_rows}</table>
{margin_html}
<h2>Graft inventory</h2>
<p class="small">Pruned slots:
{inv['n_pruned_slots']} of {total}
({htmlmod.escape(', '.join(inv['pruned_units']))}).
Slots with an eligible donor:
{inv['n_eligible_slots']}
({htmlmod.escape(', '.join(s['unit'] for s in inv['eligible_slots']))});
without an eligible donor:
{htmlmod.escape(', '.join(inv['slots_without_eligible_donor']) or 'none')}.
Donor species used in the seeded variants:
{htmlmod.escape(', '.join(str(d) for d in donors_used) or 'none')}.</p>
<h2>Density by category (band read)</h2>
<table><tr><th>genome</th><th>category</th><th>units</th>
<th></th></tr>{rows}</table>
<h2>Subject silencing table (all {total} units)</h2>
<table><tr><th>unit</th><th>kind</th><th>drop</th>
<th>band</th><th>category</th></tr>{unit_rows}</table>
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
