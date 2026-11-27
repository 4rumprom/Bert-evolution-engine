#!/usr/bin/env python3
"""BRANCH-LEVEL graft test on the new single-layer
lineage (2026-11-23). Roman's branch definition: a
branch is a connected chain of useful units carrying
information entry to exit; isolate it by what is
connected / what lights up together; graft the whole
branch into space cleared by removing harmful units
first, then neutral ones; the champion's proven branch
stays frozen beside it; prefer donor branches that
are genetically diverse.

OPERATIONAL DEFINITION used here (stated so the
sponsor can judge it): a branch is a spectral cluster
of units whose per-problem silencing-drop profiles
track together (units that matter on the same
problems), proved by GROUP silencing: zeroing the
whole cluster's outgoing weights together must cost
real score. With one block, branches live inside the
block (heads + the FFN units they feed); once the
lineage grows blocks, the whole block is the natural
branch. Graft mapping: donor branch units are paired
into cleared slots by kind (FFN->FFN, head->head),
best donor unit into the worst cleared slot.

SUBJECT: the prune-graft round's control champion
(prune_graft_widened_2026-11-23_control_champion.npz,
11.85% mean gap; d 4, FFN 16, 2 heads, 360 genes, 18
units). DONORS: the double-widen treatment run's
species bests, rebuilt by deterministic re-run and
asserted bit-identical (as in the widened
prune-graft round). Arms: treatment 25+25 gens from
branch-grafted variants, control 25+25 gens plain,
same freeze marks (subject's band-contributing
units), same budget. Density before/after in both
reads (band + cloud).

    cd ~/workspace/bert-engine && \
    setsid nohup python3 branch_graft_test_2026-11-23.py \
        >> branch_graft_test_2026-11-23.log 2>&1 < /dev/null &
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
pgw = _load("pgw", "prune_graft_widened_2026-11-23.py")

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
PG_SUBJECT_PATH = HERE / "prune_graft_d2_l1_2026-11-23_" \
    "treatment_champion.npz"
SUBJECT_PATH = HERE / "prune_graft_widened_2026-11-23_" \
    "control_champion.npz"
OUT_JSON = HERE / "branch_graft_test_2026-11-23.json"
STEP_SEEDS = {"run_seed": 20262400, "draw_seed": 20262401,
              "graft_seed": 20262404}
WIDEN_SEEDS = {"run_seed": 20262200, "draw_seed": 20262201}
SUBJECT_STORED_VAL = -0.118467
N_BRANCHES = 3


def log(msg):
    print(f"[bg-t] {msg}", flush=True)


def profiles_and_read(domain, layout, genome):
    """Per-unit per-problem drop profiles + the standard
    density read, in one batched evaluation."""
    read = cw.density_read(domain, layout, genome)
    unit_list = be.silence_units(layout)
    G = np.tile(np.asarray(genome, dtype=float),
                (len(unit_list) + 1, 1))
    for k, u in enumerate(unit_list, start=1):
        G[k, be._silence_indices(layout, u)] = 0.0
    lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
    gaps = lens / domain.optima - 1.0
    profiles = gaps[1:] - gaps[0]
    return read, profiles


def kmeans(X, k, rng, restarts=60):
    best = None
    for _ in range(restarts):
        idx = [rng.integers(len(X))]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None, :] - X[idx][None]) ** 2)
                        .sum(-1), axis=1)
            tot = d2.sum()
            idx.append(rng.choice(len(X), p=d2 / tot)
                       if tot > 0 else rng.integers(len(X)))
        C = X[idx].copy()
        for _ in range(100):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2)
                            .sum(-1), axis=1)
            new = np.array([X[lab == j].mean(0)
                            if (lab == j).any() else C[j]
                            for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        sse = float(((X - C[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lab, C)
    return best[1]


def spectral_clusters(profiles, k=N_BRANCHES, knn=3, seed=7):
    n = len(profiles)
    P = profiles / (np.linalg.norm(profiles, axis=1,
                                   keepdims=True) + 1e-12)
    S = np.clip(P @ P.T, 0.0, None)
    np.fill_diagonal(S, 0.0)
    W = np.zeros_like(S)
    for i in range(n):
        nn = np.argsort(-S[i])[:knn]
        W[i, nn] = S[i, nn]
    W = np.maximum(W, W.T)
    d = W.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d + 1e-12))
    L = np.eye(n) - Dm @ W @ Dm
    vals, vecs = np.linalg.eigh(L)
    U = vecs[:, 1:k + 1]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    return kmeans(U, k, rng)


def group_drop(domain, layout, genome, unit_names):
    """Score cost of silencing a whole cluster together,
    as a relative % of the champion's gap."""
    champ = domain.validation_score(layout, genome)
    g = np.asarray(genome, dtype=float).copy()
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for n in unit_names:
        g[be._silence_indices(layout, by_name[n])] = 0.0
    sil = domain.validation_score(layout, g)
    gap = -champ
    return {"drop_points": float(champ - sil),
            "drop_rel_pct": float(100.0 * (champ - sil) / gap)
            if gap > 0 else None}


def find_branches(domain, layout, genome, read, profiles):
    units = read["units"]
    lab = spectral_clusters(profiles)
    branches = []
    for b in range(N_BRANCHES):
        members = [units[i] for i in range(len(units))
                   if lab[i] == b]
        if not members:
            continue
        gd = group_drop(domain, layout, genome,
                        [u["name"] for u in members])
        branches.append({
            "branch_id": int(b),
            "units": [u["name"] for u in members],
            "n_units": len(members),
            "sum_unit_rel_drop_pct": float(sum(
                100.0 * u["drop"] / -read["champion_score"]
                for u in members)),
            "group_drop_rel_pct": gd["drop_rel_pct"],
            "group_drop_points": gd["drop_points"]})
    branches.sort(key=lambda x: (x["group_drop_rel_pct"]
                                 is None,
                                 -(x["group_drop_rel_pct"] or 0)))
    return branches


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

    _, _, _, _, arch2, layout2, template2, classes2 = \
        dw.build_double_transplant(arch0, layout0, pg_subject,
                                   cfg_w)
    classes = be._base_classes(layout2, 0)

    results = {
        "experiment": "branch-level graft test (Roman's "
                      "branch definition: connected chain, "
                      "entry to exit)",
        "date": "2026-11-23",
        "subject": {"file": str(SUBJECT_PATH),
                    "stored_val_score": stored_val,
                    "arch_signature": arch2.signature(),
                    "n_genes": int(layout2.n_params)},
        "branch_definition": "spectral clusters (k=3) of "
                             "units by per-problem "
                             "silencing-drop profile "
                             "similarity; proved by group "
                             "silencing of the whole "
                             "cluster",
        "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
    }

    repro_val = domain.validation_score(layout2, subject)
    assert abs(repro_val - stored_val) < 1e-9
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-6
    log(f"subject val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% gap)")

    before = dw.cloud_classify(cw.density_read(
        domain, layout2, subject))
    results["before"] = before
    log(f"BEFORE density: band {before['band_counts']}; "
        f"cloud {before['cloud']['counts']}")

    # ---- branches in the subject ----
    _, profiles_s = profiles_and_read(domain, layout2, subject)
    branches_s = find_branches(domain, layout2, subject,
                               before, profiles_s)
    results["subject_branches"] = branches_s
    for b in branches_s:
        log(f"subject branch {b['branch_id']}: "
            f"{b['n_units']} units, group drop "
            f"{b['group_drop_rel_pct']:+.1f}% of gap, "
            f"{b['units']}")

    # ---- freeze prep ----
    useful = [u["name"] for u in before["units"]
              if u["cat"] == "contributing"]
    frozen = pg.frozen_mask_from_names(layout2, useful)
    results["freeze_prep"] = {
        "basis": "band-contributing units of the subject",
        "useful_units": useful,
        "n_frozen_genes": int(frozen.sum())}
    pv = pg.prob_vectors(classes, cfg, frozen)
    log(f"FREEZE prep: {useful}; {int(frozen.sum())} genes")

    # ---- donor pool (reproduction) ----
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
    out_w = pgw.evolve_collect(domain, layout2, cfg_w,
                               classes2, species_w, rng_w, pv_w)
    band_champ = np.asarray(
        np.load(HERE / "double_widen_d2_bandmarks_"
                       "treatment_champion.npz")["genome"],
        dtype=float)
    identical = bool(np.array_equal(
        out_w["champion_genome"], band_champ))
    results["donor_pool_reproduction"] = {
        "champion_genome_bit_identical": identical,
        "species_ids": [e["species_id"]
                        for e in out_w["species_bests"]]}
    log(f"donor pool rebuilt: champion bit-identical "
        f"{identical}; species "
        f"{[e['species_id'] for e in out_w['species_bests']]}")
    assert identical

    donor_bests = [e for e in out_w["species_bests"]
                   if not np.array_equal(e["genome"],
                                         band_champ)]

    # ---- donor branches ----
    donor_branches = []
    for e in donor_bests:
        g = np.asarray(e["genome"], dtype=float)
        r, prof = profiles_and_read(domain, layout2, g)
        brs = find_branches(domain, layout2, g, r, prof)
        for b in brs:
            b["donor_species_id"] = int(e["species_id"])
            b["donor_val_score"] = float(r["champion_score"])
            b["donor_units"] = [
                {"name": u["name"], "kind": u["kind"],
                 "drop": u["drop"]}
                for u in r["units"]
                if u["name"] in b["units"]]
        donor_branches.extend(brs)
        log(f"donor sp{e['species_id']}: branches "
            f"{[(b['branch_id'], b['n_units'], round(b['group_drop_rel_pct'], 1)) for b in brs]}")
    donor_branches.sort(key=lambda b: -(b["group_drop_rel_pct"]
                                       or -1e9))
    results["donor_branches"] = donor_branches

    # ---- cleared space + graft mapping ----
    cleared = sorted(
        [u for u in before["units"] if u["drop"] <= 0.0],
        key=lambda u: u["drop"])
    pruned_base = np.asarray(subject, dtype=float).copy()
    by_name = {u["name"]: u for u in be.silence_units(layout2)}
    for u in cleared:
        pruned_base[be._silence_indices(layout2, u)] = 0.0
    pruned_score = domain.validation_score(layout2, pruned_base)
    # graft material: walk donor branches strongest first
    graft_units = []
    for b in donor_branches:
        for u in sorted(b["donor_units"],
                        key=lambda x: -x["drop"]):
            graft_units.append(
                {"donor_species_id": b["donor_species_id"],
                 "donor_branch_id": b["branch_id"],
                 "name": u["name"], "kind": u["kind"],
                 "drop": u["drop"]})
    pairs = []
    used = set()
    for slot in cleared:
        for i, gu in enumerate(graft_units):
            if i in used or gu["kind"] != slot["kind"]:
                continue
            used.add(i)
            pairs.append({"slot_unit": slot["name"],
                          "slot_kind": slot["kind"],
                          "slot_drop": slot["drop"],
                          "donor_species_id":
                              gu["donor_species_id"],
                          "donor_branch_id":
                              gu["donor_branch_id"],
                          "donor_unit": gu["name"],
                          "donor_unit_drop": gu["drop"]})
            break
    inventory = {
        "n_cleared_slots": len(cleared),
        "cleared_units": [u["name"] for u in cleared],
        "pruned_base_score": float(pruned_score),
        "pruned_base_delta": float(pruned_score - repro_val),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "slots_without_graft": [
            u["name"] for u in cleared
            if u["name"] not in {p["slot_unit"]
                                 for p in pairs}],
    }
    results["graft_inventory"] = inventory
    log(f"CLEARED: {len(cleared)} slots; pruned base "
        f"{pruned_score:.6f} (delta "
        f"{inventory['pruned_base_delta']:+.6f}); pairs "
        f"{len(pairs)}; without graft "
        f"{inventory['slots_without_graft']}")
    if not pairs:
        results["verdict"] = "no graft pairs; arms not run"
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        log("DONE (no pairs)")
        print("BRANCH_GRAFT_TEST_DONE")
        return

    # ---- treatment variants (fractions of the branch) ----
    graft_rng = np.random.default_rng(STEP_SEEDS["graft_seed"])
    fractions = [float(x) for x in np.linspace(
        0.25, 1.0, cfg.n_species)]
    donor_genome = {int(e["species_id"]): np.asarray(
        e["genome"], dtype=float) for e in donor_bests}
    species_t = []
    rng = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng = np.random.default_rng(STEP_SEEDS["draw_seed"])
    variant_records = []
    for s in range(cfg.n_species):
        frac = fractions[s]
        n_take = int(round(frac * len(pairs)))
        chosen = []
        if n_take:
            idx = graft_rng.choice(len(pairs), size=n_take,
                                   replace=False)
            chosen = [pairs[int(i)] for i in idx]
        base_s = pruned_base.copy()
        for p in chosen:
            slot = by_name[p["slot_unit"]]
            donor_u = by_name[p["donor_unit"]]
            idxs = be.unit_gene_indices(layout2, slot)
            base_s[idxs] = donor_genome[
                p["donor_species_id"]][
                    be.unit_gene_indices(layout2, donor_u)]
        variant_records.append({
            "species_index": s, "fraction": frac,
            "n_grafts": len(chosen),
            "grafts": [{"slot": p["slot_unit"],
                        "donor_unit": p["donor_unit"],
                        "donor_species_id":
                            p["donor_species_id"]}
                       for p in chosen],
            "variant_base_val_score": float(
                domain.validation_score(layout2, base_s))})
        sp = be.create_species(
            base_s, classes, 1, cfg.per_species, draw_rng,
            rng, pv["A"], cfg, lineage=None,
            init_mode="transplant", next_id=s)
        species_t.extend(sp)
    results["species_variants"] = variant_records
    log("treatment variants built: grafts per species "
        f"{[v['n_grafts'] for v in variant_records]}")

    t0 = time.time()
    out_t = pg.evolve(domain, layout2, cfg, classes,
                      species_t, rng, pv)
    wall_t = time.time() - t0
    after_t = dw.cloud_classify(cw.density_read(
        domain, layout2, out_t["champion_genome"]))
    results["treatment"] = pg.arm_block(out_t, after_t,
                                        repro_val, wall_t)
    results["treatment"]["density_cloud_counts"] = \
        after_t["cloud"]["counts"]
    np.savez(HERE / "branch_graft_test_2026-11-23_"
                    "treatment_champion.npz",
             genome=out_t["champion_genome"],
             val_score=out_t["champion_val_score"])
    log(f"TREATMENT done: {out_t['champion_val_score']:.6f} "
        f"({-100.0 * out_t['champion_val_score']:.2f}% gap, "
        f"{wall_t:.1f}s); AFTER density: band "
        f"{after_t['band_counts']}; cloud "
        f"{after_t['cloud']['counts']}")

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
    np.savez(HERE / "branch_graft_test_2026-11-23_"
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
    print("BRANCH_GRAFT_TEST_DONE")


if __name__ == "__main__":
    main()
