#!/usr/bin/env python3
"""Prune-and-graft TEST for the parametric BERT evolution engine
(2026-11-23). Standalone experiment, NOT an engine integration:
the chimera design stays unfrozen; this produces before/after
evidence for one prune-and-graft cycle on one existing champion.

Subject: engine_runs/test_minimal_a/champion.npz (TSP, d 8,
FFN 16, 2 blocks, 2 heads, 1,880 genes, val -0.4120 on the
fixed 200-problem validation set, val_seed 20261116).

Design (frozen for this test):
  PRUNE (safe level): zero the outgoing weights
  (be._silence_indices) of every unit whose silencing drop in
  the champion is <= 0 (exact-zero criterion).
  DONORS: the other species' best genomes from the same run
  (checkpoint.npz best_genomes) plus the vault stage champion
  (which is the subject itself; it is inventoried but, being
  the subject, contributes no grafts). A donor unit is
  graft-eligible only if its drop in its OWN donor genome
  is > 0.
  GRAFT: position-preserving only. Donor unit (couple, block,
  kind, index) fills only the pruned slot at the same
  position, copying ALL of the unit's genes
  (be.unit_gene_indices). Slots with no eligible donor stay
  zeroed.
  ARM B: 12 species x 10 from species variants of the grafted
  base (each species a different random subset of the
  eligible grafts, fractions 25%..100% across species), then
  standard engine mutation on top; evolve 50 generations.
  ARM A (control): identical, seeded from the ORIGINAL
  champion, no pruning, no grafts.

Evolution loop: a small loop over engine primitives
(be.create_species, be.breed_species, be.cull_species,
be.prob_vector) mirroring Engine.generation / run_stage in
run_bert_engine.py, minus archive/monitor/checkpoint/
cataclysm (all off or absent in the source run's config too):
  - 50 generations as one stage: phase A 25 + phase B 25
    (the engine's stage shape, budgets halved evenly);
  - train batches exactly as the runner draws them
    (domain.train_batch(phase, stage 0, gen_in_phase));
  - species best-ever tracked on train scores; Schedule A
    culling on the runner's schedule ((gen+1) % 3 == 0,
    floor 4, targets redistributed to 120);
  - final champion = argmax over species bests on the fixed
    validation set, exactly run_stage's champion rule.

ONE INTERPRETATION, stated plainly: the classes used for
seeding and evolution are be._base_classes(layout, 0) --
every gene CLS_OLD except couple 0's head (CLS_DECODER), no
CLS_DRAW genes. Reason: with the source run's stage-0
classes (1,800 of 1,880 genes CLS_DRAW), be.create_species
would REPLACE the encoder with a fresh random draw in every
species, destroying the subject champion in both arms and
overwriting every graft. The engine's own continuation path
(a founder carried into a new stage at a growth step)
rebases the classes exactly this way before regenerating
species from the founder, so this test follows the engine's
founder-continuation precedent. Mutation marks are None:
the source run evolved without them (its silencing state
was only built at stage end), and marking is a different
plan step than prune-and-graft.

Seeds: run_rng default_rng(cfg.run_seed = 20261300) and
draw_rng default_rng(cfg.draw_seed = 20261301), freshly
initialised per arm so both arms consume identical streams
for shared operations; graft subset/donor choice uses its
own graft_rng default_rng(20261304), Arm B only.
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
OUT_JSON = HERE / "prune_graft_test_2026-11-23.json"
OUT_HTML = HERE / "prune_graft_test_2026-11-23.html"
GENS_PER_PHASE = 25
GRAFT_SEED = 20261304


def log(msg):
    print(f"[pg-test] {msg}", flush=True)


# ---------------------------------------------------------------
# density read: silencing table + paired-noise band categories
# (reference implementation: make_usefulness_map.py)
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
# evolution: Engine.generation / run_stage mirrored on
# engine primitives (no archive / monitor / checkpoint /
# cataclysm; none apply to this test's config)
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
    # stage champion: run_stage's rule (validation argmax
    # over species best-ever genomes)
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
    # classes for continuation: the engine's founder rebase
    classes = be._base_classes(layout, 0)
    pv_a = be.prob_vector(classes, "A", cfg, marks=None)

    results = {
        "experiment": "prune-and-graft test (standalone, "
                      "not an engine integration)",
        "date": "2026-11-23",
        "subject": {
            "run_dir": str(RUN_DIR),
            "arch_signature": arch.signature(),
            "n_genes": int(layout.n_params),
            "stored_val_score": stored_val,
        },
        "settings": {
            "config_file": str(CFG_PATH),
            "config_canonical": json.loads(cfg.canonical()),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "classes_rule": "be._base_classes(layout, 0): all "
                            "genes CLS_OLD except couple 0 "
                            "head CLS_DECODER; no CLS_DRAW "
                            "genes (engine founder-continuation "
                            "rebase; see script docstring)",
            "marks": None,
            "seeds": {"run_seed": cfg.run_seed,
                      "draw_seed": cfg.draw_seed,
                      "graft_seed": GRAFT_SEED,
                      "val_seed": cfg.val_seed},
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
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    log(f"champion val reproduced: {repro_val:.6f}")
    before = density_read(domain, layout, champion)
    results["before"] = before
    log(f"BEFORE density: exact-zero positive "
        f"{before['exact_zero_positive']}/{before['n_units']}; "
        f"band {before['band_counts']}")
    if before["exact_zero_positive"] != 10:
        log("WARNING: exact-zero positive count differs from "
            "the expected 10; recorded as measured")

    # ---------------- PRUNE ----------------
    prune_units = [u for u in before["units"] if u["drop"] <= 0.0]
    pruned_base = champion.copy()
    for u in prune_units:
        pruned_base[be._silence_indices(layout, u)] = 0.0
    pruned_score = domain.validation_score(layout, pruned_base)
    results["prune"] = {
        "criterion": "silencing drop <= 0 (exact-zero), "
                     "outgoing weights zeroed",
        "n_units_pruned": len(prune_units),
        "pruned_units": [u["name"] for u in prune_units],
        "pruned_base_score": float(pruned_score),
        "delta_vs_champion": float(pruned_score - repro_val),
    }
    log(f"PRUNE: {len(prune_units)} units zeroed; pruned base "
        f"score {pruned_score:.6f} (delta "
        f"{pruned_score - repro_val:+.6f})")
    if len(prune_units) != 26:
        log("WARNING: pruned count differs from the expected "
            "26; recorded as measured")

    # ---------------- DONORS ----------------
    best_genomes = np.asarray(ck["best_genomes"], dtype=float)
    species_ids = [int(x) for x in ck["species_ids"]]
    vault_z = np.load(
        RUN_DIR / "vault" / "stage_champion_stage0_sp8_86970669.npz",
        allow_pickle=True)
    vault_genome = np.asarray(vault_z["genome"], dtype=float)
    pool = []
    for gid, genome in zip(species_ids, best_genomes):
        pool.append({
            "species_id": gid,
            "source": "checkpoint_best_genomes",
            "genome": genome,
            "identical_to_subject": bool(
                np.array_equal(genome, champion))})
    pool.append({
        "species_id": int(vault_z["species_id"]),
        "source": "vault_stage_champion",
        "genome": vault_genome,
        "identical_to_subject": bool(
            np.array_equal(vault_genome, champion))})
    graft_donors = []
    seen = []
    for entry in pool:
        if entry["identical_to_subject"]:
            continue
        if any(np.array_equal(entry["genome"], s)
               for s in seen):
            continue
        seen.append(entry["genome"])
        graft_donors.append(entry)
    results["donors"] = {
        "pool": [{"species_id": e["species_id"],
                  "source": e["source"],
                  "identical_to_subject":
                      e["identical_to_subject"]}
                 for e in pool],
        "grafting_donor_species_ids":
            [e["species_id"] for e in graft_donors],
        "note": "the vault stage champion is the subject "
                "itself, and the subject's own checkpoint best "
                "row is the same genome; both are inventoried "
                "but excluded from grafting (a donor identical "
                "to the subject could only re-fill slots with "
                "the subject's own genes, and at pruned slots "
                "its own drops are <= 0 by construction).",
    }
    log(f"donor pool: {[(e['species_id'], e['source']) for e in pool]}; "
        f"grafting donors: "
        f"{[e['species_id'] for e in graft_donors]}")

    donor_tables = {}
    donor_eligible = {}   # species_id -> {unit_key: drop}
    for e in graft_donors:
        tab = be.silence_table(
            layout, e["genome"],
            lambda G: domain.validation_scores(layout, G),
            zero_tol=domain.silence_zero_tol)
        drops = {unit_key(u): float(u["drop"])
                 for u in tab["units"]}
        donor_eligible[e["species_id"]] = drops
        donor_tables[str(e["species_id"])] = {
            "donor_val_score": float(tab["champion_score"]),
            "positive_drop_units": [
                u["name"] for u in tab["units"]
                if u["drop"] > 0.0],
        }
        log(f"donor sp{e['species_id']}: own val "
            f"{tab['champion_score']:.6f}, "
            f"{len(donor_tables[str(e['species_id'])]['positive_drop_units'])}"
            f" units with drop > 0")
    results["donors"]["donor_tables"] = donor_tables

    # ---------------- GRAFT INVENTORY ----------------
    unit_by_key = {unit_key(u): u for u in before["units"]}
    eligible = {}   # unit_key -> [donor species ids]
    for u in prune_units:
        k = unit_key(u)
        ds = [sid for sid, drops in donor_eligible.items()
              if drops.get(k, 0.0) > 0.0]
        if ds:
            eligible[k] = ds
    results["graft_inventory"] = {
        "n_pruned_slots": len(prune_units),
        "n_eligible_slots": len(eligible),
        "eligible_slots": [
            {"unit": unit_by_key[k]["name"],
             "donor_species_ids": eligible[k]}
            for k in sorted(eligible, key=lambda k: str(k))],
        "slots_without_eligible_donor": [
            u["name"] for u in prune_units
            if unit_key(u) not in eligible],
    }
    log(f"GRAFT inventory: {len(eligible)} of "
        f"{len(prune_units)} pruned slots have an eligible "
        f"donor")

    if not eligible:
        results["verdict"] = "graft inventory EMPTY: no donor " \
            "unit with positive own-drop exists for any " \
            "pruned slot; arms not run, rules not loosened"
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log("DONE (empty inventory)")
        print("PRUNE_GRAFT_TEST_DONE")
        return

    # ---------------- ARM B species variants ----------------
    graft_rng = np.random.default_rng(GRAFT_SEED)
    fractions = [float(x) for x in
                 np.linspace(0.25, 1.0, cfg.n_species)]
    eligible_keys = sorted(eligible, key=lambda k: str(k))
    donor_genome_by_id = {e["species_id"]: e["genome"]
                          for e in graft_donors}
    variant_records = []
    arm_b_species = []
    rng_b = np.random.default_rng(cfg.run_seed)
    draw_rng_b = np.random.default_rng(cfg.draw_seed)
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
            base_s, classes, 1, cfg.per_species, draw_rng_b,
            rng_b, pv_a, cfg, lineage=None,
            init_mode="transplant", next_id=s)
        arm_b_species.extend(sp)
    results["graft_inventory"]["species_variants"] = \
        variant_records
    log("ARM B species variants built: grafts per species "
        f"{[v['n_grafts'] for v in variant_records]}")

    # ---------------- ARM A species ----------------
    rng_a = np.random.default_rng(cfg.run_seed)
    draw_rng_a = np.random.default_rng(cfg.draw_seed)
    arm_a_species = be.create_species(
        champion, classes, cfg.n_species, cfg.per_species,
        draw_rng_a, rng_a, pv_a, cfg, lineage=None,
        init_mode="transplant", next_id=0)

    # ---------------- EVOLVE ----------------
    t0 = time.time()
    out_a = evolve(domain, layout, cfg, classes,
                   arm_a_species, rng_a)
    wall_a = time.time() - t0
    log(f"ARM A done: final champion val "
        f"{out_a['champion_val_score']:.6f} "
        f"(species {out_a['champion_species_id']}, "
        f"{out_a['n_species_final']} species left, "
        f"{wall_a:.1f}s)")
    t0 = time.time()
    out_b = evolve(domain, layout, cfg, classes,
                   arm_b_species, rng_b)
    wall_b = time.time() - t0
    log(f"ARM B done: final champion val "
        f"{out_b['champion_val_score']:.6f} "
        f"(species {out_b['champion_species_id']}, "
        f"{out_b['n_species_final']} species left, "
        f"{wall_b:.1f}s)")

    # ---------------- AFTER ----------------
    after_a = density_read(domain, layout,
                           out_a["champion_genome"])
    after_b = density_read(domain, layout,
                           out_b["champion_genome"])
    np.savez(HERE / "prune_graft_test_2026-11-23_armA_champion.npz",
             genome=out_a["champion_genome"],
             val_score=out_a["champion_val_score"])
    np.savez(HERE / "prune_graft_test_2026-11-23_armB_champion.npz",
             genome=out_b["champion_genome"],
             val_score=out_b["champion_val_score"])

    def arm_block(out, after, wall):
        return {
            "final_champion_val_score":
                out["champion_val_score"],
            "delta_vs_before":
                out["champion_val_score"] - repro_val,
            "champion_species_id": out["champion_species_id"],
            "n_species_final": out["n_species_final"],
            "wall_seconds": wall,
            "density_exact_zero_positive":
                after["exact_zero_positive"],
            "density_band_counts": after["band_counts"],
            "density_units": after["units"],
            "history": out["history"],
        }

    results["arm_A_control"] = arm_block(out_a, after_a, wall_a)
    results["arm_B_prune_graft"] = arm_block(out_b, after_b,
                                             wall_b)
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    log(f"AFTER A: density exact-zero "
        f"{after_a['exact_zero_positive']}/36, band "
        f"{after_a['band_counts']}")
    log(f"AFTER B: density exact-zero "
        f"{after_b['exact_zero_positive']}/36, band "
        f"{after_b['band_counts']}")
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("PRUNE_GRAFT_TEST_DONE")


# ---------------------------------------------------------------
# summary page: counts + bars + strips (scalable layout idea
# from make_usefulness_map.py; no per-neuron tiles)
# ---------------------------------------------------------------
def _dcolour(u, dmax):
    if u["cat"] == "contributing":
        t = min(1.0, u["drop"] / dmax) if dmax > 0 else 0.0
        return (f"rgb(255,{int(255 - 205 * t)},"
                f"{int(255 - 205 * t)})")
    if u["cat"] == "harmful":
        return "rgb(170,220,170)"
    return "rgb(196,214,228)"


def _strips(units):
    zones = {}
    for u in units:
        zones.setdefault((u["couple"], u["block"], u["kind"]),
                         []).append(u)
    dmax = max((u["drop"] for u in units), default=0.0) or 1.0
    out = []
    for (c, b, kind) in sorted(zones):
        us = sorted(zones[(c, b, kind)],
                    key=lambda u: u["index"])
        cell = 12
        cells = "".join(
            f"<rect x='{u['index'] * cell}' y='0' "
            f"width='{cell - 1}' height='16' "
            f"fill='{_dcolour(u, dmax)}' stroke='#ccc' "
            f"stroke-width='0.4'><title>"
            f"{htmlmod.escape(u['name'])}: drop "
            f"{u['drop']:+.4f}, {u['cat']}</title></rect>"
            for u in us)
        out.append(
            f"<div class='strip'><div class='slabel'>couple "
            f"{c} block {b} {kind} ({len(us)} units)</div>"
            f"<svg width='{len(us) * cell}' height='16'>"
            f"{cells}</svg></div>")
    return "".join(out)


def _cat_bars(label, counts, total):
    parts = []
    for cat, colour in (("contributing", "#c0392b"),
                        ("neutral", "#7f9db3"),
                        ("harmful", "#5da85d")):
        n = counts.get(cat, 0)
        parts.append(
            f"<tr><td>{label}</td><td>{cat}</td><td>{n}</td>"
            f"<td><div class='bar'><div style='height:100%;"
            f"width:{(n / total if total else 0):.0%};"
            f"background:{colour}'></div></div></td></tr>")
    return "".join(parts)


def write_html(results):
    before = results["before"]
    total = before["n_units"]
    rows = _cat_bars("before (original champion)",
                     before["band_counts"], total)
    score_rows = (
        f"<tr><td>original champion</td><td>"
        f"{results['subject']['reproduced_val_score']:+.4f}"
        f"</td></tr>"
        f"<tr><td>pruned base (no evolution)</td><td>"
        f"{results['prune']['pruned_base_score']:+.4f}</td></tr>")
    sections = [f"<h2>Unit strips: before</h2>"
                f"{_strips(before['units'])}"]
    for key, label in (("arm_A_control", "Arm A: control "
                        "(continued evolution)"),
                       ("arm_B_prune_graft", "Arm B: "
                        "prune-and-graft")):
        if key in results:
            arm = results[key]
            rows += _cat_bars(label, arm["density_band_counts"],
                              total)
            score_rows += (
                f"<tr><td>{label} final champion</td><td>"
                f"{arm['final_champion_val_score']:+.4f} "
                f"(delta {arm['delta_vs_before']:+.4f})</td>"
                f"</tr>")
            sections.append(
                f"<h2>Unit strips: {label}</h2>"
                f"{_strips(arm['density_units'])}")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prune-and-graft test 2026-11-23</title>
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
.strip {{ margin: 8px 0; }}
.slabel {{ font: 11px ui-monospace, monospace;
         margin-bottom: 2px; }}
</style></head><body>
<h1>Prune-and-graft test: before / after</h1>
<p class="small">Subject: the test_minimal_a champion (TSP,
1,880 genes, 36 testable units). Scores are validation scores
on the fixed 200-problem set (higher is better). Density
categories come from the paired-noise band read: a unit is
contributing when its silencing drop clears +2 paired SE,
harmful below -2 SE, neutral in between. Strips: red =
contributing, blue-grey = neutral, green = harmful; one cell
per unit, hover for the exact drop.</p>
<h2>Scores</h2>
<table><tr><th>genome</th><th>validation score</th></tr>
{score_rows}</table>
<h2>Density by category (band read)</h2>
<table><tr><th>genome</th><th>category</th><th>units</th>
<th></th></tr>{rows}</table>
{''.join(sections)}
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
