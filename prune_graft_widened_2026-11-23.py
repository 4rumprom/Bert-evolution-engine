#!/usr/bin/env python3
"""PRUNE-AND-GRAFT round on the WIDENED champion of the
new single-layer lineage (2026-11-23). Follows
prune_graft_d2_l1_2026-11-23.py exactly (same prune /
donor / graft construction, same primitives evolution
loop, 25 + 25 generations, same freeze-rule marks),
with density measured BEFORE and AFTER in both reads:
the paired-noise band and the sponsor's near-zero
cloud classification (drops as relative % of the
champion's gap; cloud = |drop| <= 2 robust sigmas).

SUBJECT: the double-widen treatment champion
(double_widen_d2_2026-11-23_treatment_champion.npz,
14.00% mean gap; d 4, FFN 16, 2 heads, 1 layer, 360
genes, 18 silence units).

DONORS: the double-widen treatment run's species
bests. That run saved only its champion, so the donor
pool is rebuilt by re-running the treatment evolution
with the identical seeds, classes and freeze mask;
the reproduction is asserted bit-identical to the
saved champion before any donor is used. The champion
species is excluded as a donor by genome identity.

FREEZE: the subject's band-contributing units are
marked useful (rate 0 in both arms), the basis kept
for comparability with the day's arms.

SEEDS: the widen step family was run 20262200; this
round advances one +100 step: run 20262300, draw
20262301, graft 20262304.

    cd ~/workspace/bert-engine && \
    setsid nohup python3 prune_graft_widened_2026-11-23.py \
        >> prune_graft_widened_2026-11-23.log 2>&1 < /dev/null &
"""
import dataclasses
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cw = _load("cw", "coupled_widen_d2_2026-11-23.py")
dw = _load("dw", "double_widen_d2_2026-11-23.py")
pg = _load("pg", "prune_graft_d2_l1_2026-11-23.py")

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
PG_SUBJECT_PATH = HERE / "prune_graft_d2_l1_2026-11-23_" \
    "treatment_champion.npz"
SUBJECT_PATH = HERE / "double_widen_d2_2026-11-23_" \
    "treatment_champion.npz"
OUT_JSON = HERE / "prune_graft_widened_2026-11-23.json"
STEP_SEEDS = {"run_seed": 20262300, "draw_seed": 20262301,
              "graft_seed": 20262304}
WIDEN_SEEDS = {"run_seed": 20262200, "draw_seed": 20262201}
SUBJECT_STORED_VAL = -0.140031


def log(msg):
    print(f"[pg-w] {msg}", flush=True)


def evolve_collect(domain, layout, cfg, classes, species,
                   rng, pv_by_phase, gens_per_phase=50):
    """cw.evolve's loop, additionally returning every
    species' best genome (the donor pool)."""
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
        "species_bests": [
            {"species_id": int(sp["id"]),
             "genome": sp["best"]["genome"],
             "train_score": float(sp["best"]["score"])}
            for sp in species],
    }


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    cfg_w = dataclasses.replace(cfg, widen_step=2)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    ck = np.load(HERE / "engine_runs" / "fresh_d2_l1" /
                 "checkpoint.npz", allow_pickle=True)
    arch0 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout0 = be.Layout(arch0, tsp.head_spec(arch0.d))
    pg_subject = np.asarray(
        np.load(PG_SUBJECT_PATH)["genome"], dtype=float)
    champ_z = np.load(SUBJECT_PATH)
    subject = np.asarray(champ_z["genome"], dtype=float)
    stored_val = float(champ_z["val_score"])

    # rebuild the widened layout exactly as the widen run did
    _, _, _, _, arch2, layout2, template2, classes2 = \
        dw.build_double_transplant(arch0, layout0, pg_subject,
                                   cfg_w)
    classes = be._base_classes(layout2, 0)

    results = {
        "experiment": "prune-and-graft round on the "
                      "widened champion (new lineage)",
        "date": "2026-11-23",
        "subject": {"file": str(SUBJECT_PATH),
                    "stored_val_score": stored_val,
                    "arch_signature": arch2.signature(),
                    "n_genes": int(layout2.n_params)},
        "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
    }

    repro_val = domain.validation_score(layout2, subject)
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-6
    log(f"subject val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% gap), "
        f"{layout2.n_params} genes, "
        f"{len(be.silence_units(layout2))} units")

    # ---------------- DENSITY BEFORE ----------------
    before = dw.cloud_classify(cw.density_read(
        domain, layout2, subject))
    results["before"] = before
    log(f"BEFORE density: band {before['band_counts']}; "
        f"cloud {before['cloud']['counts']} (limit "
        f"+/-{before['cloud']['limit_pct']:.1f}% of gap); "
        f"exact-zero positive "
        f"{before['exact_zero_positive']}/{before['n_units']}")

    # ---------------- FREEZE PREP ----------------
    useful = [u["name"] for u in before["units"]
              if u["cat"] == "contributing"]
    frozen = pg.frozen_mask_from_names(layout2, useful)
    results["freeze_prep"] = {
        "basis": "band-contributing units of the subject",
        "useful_units": useful,
        "n_frozen_genes": int(frozen.sum())}
    pv = pg.prob_vectors(classes, cfg, frozen)
    log(f"FREEZE prep: {useful}; {int(frozen.sum())} genes")

    # ---------------- DONOR POOL (reproduction) ----------------
    frozen_names2 = [f"c0.b0.ffn.u{j}" for j in dw.MARKED_FFN] \
        + [f"c0.b0.ffn.u{j + 8}" for j in dw.MARKED_FFN] \
        + ["c0.b0.head.h0", "c0.b0.head.h1"]
    frozen2 = cw.frozen_mask_from_names(layout2, frozen_names2)
    pv_w = cw.prob_vectors(classes2, cfg_w, frozen2)
    rng_w = np.random.default_rng(WIDEN_SEEDS["run_seed"])
    draw_w = np.random.default_rng(WIDEN_SEEDS["draw_seed"])
    species_w = be.create_species(
        template2, classes2, cfg_w.n_species,
        cfg_w.per_species, draw_w, rng_w, pv_w["A"], cfg_w,
        lineage=None, init_mode="transplant", next_id=0)
    out_w = evolve_collect(domain, layout2, cfg_w, classes2,
                           species_w, rng_w, pv_w)
    identical = bool(np.array_equal(
        out_w["champion_genome"], subject))
    results["donor_pool_reproduction"] = {
        "champion_val_rebuilt": out_w["champion_val_score"],
        "champion_genome_bit_identical": identical,
        "n_species_final": out_w["n_species_final"],
        "species_ids": [e["species_id"]
                        for e in out_w["species_bests"]]}
    log(f"donor pool rebuilt: champion bit-identical "
        f"{identical} (val {out_w['champion_val_score']:.6f}); "
        f"species {[e['species_id'] for e in out_w['species_bests']]}")
    assert identical, "donor pool reproduction failed"

    donor_bests = [e for e in out_w["species_bests"]
                   if not np.array_equal(e["genome"], subject)]

    # ---------------- PRUNE + INVENTORY ----------------
    pruned_base, inventory, donor_genome_by_id, unit_by_key, \
        eligible = pg.prune_and_inventory(
            domain, layout2, subject, before, donor_bests)
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
    if not eligible:
        results["verdict"] = "graft inventory EMPTY; arms " \
            "not run, rules not loosened"
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        log("DONE (empty inventory)")
        print("PRUNE_GRAFT_WIDENED_DONE")
        return

    # ---------------- TREATMENT ARM ----------------
    species_t, rng_t, variants = pg.build_treatment_species(
        domain, layout2, cfg, classes, pv["A"], pruned_base,
        unit_by_key, eligible, donor_genome_by_id, STEP_SEEDS)
    results["species_variants"] = variants
    log("treatment variants built: grafts per species "
        f"{[v['n_grafts'] for v in variants]}")
    t0 = time.time()
    out_t = pg.evolve(domain, layout2, cfg, classes,
                      species_t, rng_t, pv)
    wall_t = time.time() - t0
    after_t = dw.cloud_classify(cw.density_read(
        domain, layout2, out_t["champion_genome"]))
    results["treatment"] = pg.arm_block(out_t, after_t,
                                        repro_val, wall_t)
    results["treatment"]["density_cloud_counts"] = \
        after_t["cloud"]["counts"]
    results["treatment"]["density_cloud_limit_pct"] = \
        after_t["cloud"]["limit_pct"]
    np.savez(HERE / "prune_graft_widened_2026-11-23_"
                    "treatment_champion.npz",
             genome=out_t["champion_genome"],
             val_score=out_t["champion_val_score"])
    log(f"TREATMENT done: {out_t['champion_val_score']:.6f} "
        f"({-100.0 * out_t['champion_val_score']:.2f}% gap, "
        f"{wall_t:.1f}s); AFTER density: band "
        f"{after_t['band_counts']}; cloud "
        f"{after_t['cloud']['counts']}")

    # ---------------- CONTROL ARM ----------------
    rng_c = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_c = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_c = be.create_species(
        subject, classes, cfg.n_species, cfg.per_species,
        draw_rng_c, rng_c, pv["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c = pg.evolve(domain, layout2, cfg, classes,
                      species_c, rng_c, pv)
    wall_c = time.time() - t0
    after_c = dw.cloud_classify(cw.density_read(
        domain, layout2, out_c["champion_genome"]))
    results["control"] = pg.arm_block(out_c, after_c,
                                      repro_val, wall_c)
    results["control"]["density_cloud_counts"] = \
        after_c["cloud"]["counts"]
    results["control"]["density_cloud_limit_pct"] = \
        after_c["cloud"]["limit_pct"]
    np.savez(HERE / "prune_graft_widened_2026-11-23_"
                    "control_champion.npz",
             genome=out_c["champion_genome"],
             val_score=out_c["champion_val_score"])
    log(f"CONTROL done: {out_c['champion_val_score']:.6f} "
        f"({-100.0 * out_c['champion_val_score']:.2f}% gap, "
        f"{wall_c:.1f}s); AFTER density: band "
        f"{after_c['band_counts']}; cloud "
        f"{after_c['cloud']['counts']}")

    results["margin_treatment_over_control"] = pg.margins(
        out_t["champion_val_score"],
        out_c["champion_val_score"])
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    m = results["margin_treatment_over_control"]
    log(f"MARGIN: {m['points']:+.6f} points, "
        f"{m['relative_percent_of_control_gap']:+.2f}% "
        f"relative to the control's gap")
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("PRUNE_GRAFT_WIDENED_DONE")


if __name__ == "__main__":
    main()
