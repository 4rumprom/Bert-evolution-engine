#!/usr/bin/env python3
"""BRANCH PRUNE-AND-GRAFT SCALE TEST, from zero
(2026-11-24). Sponsor direction: verify that
branch-level prune-and-graft scales, starting from a
fresh founding run, deep enough that branches have
somewhere to live.

DESIGN (stated so the sponsor can judge it):
- Founding run from zero: TSP, 12 species x 10, 60
  generations (30 + 30), stage-0 precedent (random
  members, nothing proven, nothing frozen). The
  champion and every species' best are saved; the
  species bests are the donor pool.
- Two branch prune-and-graft rounds on the lineage
  champion. Each round: isolate branches in the
  champion and in each donor species champion
  (bert_branch.find_branches: spectral clusters of
  per-problem silencing profiles, proved by group
  silencing); clear harmful units first, then neutral
  ones (bert_branch.cleared_slots); graft the best
  diverse donor branch into the cleared slots by kind
  AND layer position (bert_branch.graft_branch);
  evolve 50 generations at the NOMINAL rate
  throughout (prob_vector flat_nominal, sponsor rule
  2026-11-24), proven units frozen at rate 0. The
  control arm prunes and evolves the same way with no
  graft, same budget and seeds. Round 2 continues
  from the better of round 1's treatment/control and
  draws its donors from that arm's species bests.
- SUCCESS CRITERIA (written before the run): (a) at
  least one branch graft actually executes in each
  round (pairs > 0); (b) the grafted branch, re-
  isolated in the round's champion, still proves
  itself by group silencing; (c) treatment vs
  control margins read relatively, as usual.

SHAPES (sponsor's choice; --shape picks it):
  deep   (default): 3 blocks, d 4, FFN 16, 2 heads
  wide:   2 blocks, d 8, FFN 16, 2 heads (the
          1,880-gene shape of the earlier measurements;
          FFN corrected 2026-11-24: FFN 32 would be
          2,424 genes, not the measured shape)

SEEDS: family 20262400 (run), 20262401 (draw),
20262402 (graft). Validation: the fixed 200-problem
TSP set of the lineage (val_seed 20261116).

    cd ~/workspace/bert-engine && \\
    setsid nohup python3 branch_pg_scale_test_2026-11-24.py \\
        [--shape deep|wide] [--smoke] \\
        >> branch_pg_scale_test_2026-11-24.log 2>&1 < /dev/null &
"""
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_branch as bb
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
pgw = _load("pgw", "prune_graft_widened_2026-11-23.py")

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
OUT_JSON = HERE / "branch_pg_scale_test_2026-11-24.json"
SHAPES = {
    "deep": {"d": 4, "ffn_hidden": 16, "initial_blocks": 3,
             "initial_heads": 2},
    "wide": {"d": 8, "ffn_hidden": 16, "initial_blocks": 2,
             "initial_heads": 2},
}
SEEDS = {"run_seed": 20262400, "draw_seed": 20262401,
         "graft_seed": 20262402}


def log(msg):
    print(f"[bp-scale] {msg}", flush=True)


def band_proven(read):
    """Units proven useful under the paired-noise band:
    silencing drop > 2 standard errors of the per-problem
    score differences (the harness band rule)."""
    return [u["name"] for u in read["units"]
            if u["drop"] > 2.0 * u.get("se", 0.0)]


def frozen_mask_from_names(layout, names):
    mask = np.zeros(layout.n_params, dtype=bool)
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for n in names:
        mask[be.unit_gene_indices(layout, by_name[n])] = True
    return mask


def branch_gene_distance(layout, branch_units, genome_a,
                         genome_b):
    """Euclidean distance between the branch's genes in
    two genomes (the diversity read for donor choice)."""
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    idx = np.concatenate([
        be.unit_gene_indices(layout, by_name[u["name"]])
        for u in branch_units]) if branch_units else \
        np.array([], dtype=int)
    if len(idx) == 0:
        return 0.0
    return float(np.linalg.norm(
        np.asarray(genome_a)[idx] - np.asarray(genome_b)[idx]))


def pick_donor_branch(layout, subject, donor_bests):
    """Best donor branch: among the top 3 branches by
    group drop across all donor species, take the one
    most genetically distant from the subject on its
    branch genes (diverse donors add information)."""
    cands = []
    for sp in donor_bests:
        if np.array_equal(sp["genome"], subject):
            continue  # champion species is never a donor
        for b in sp.get("branches", []):
            if b["group_drop_rel_pct"] is None:
                continue
            d = branch_gene_distance(
                layout, b["units"], sp["genome"], subject)
            cands.append({"species_id": sp["species_id"],
                          "genome": sp["genome"],
                          "branch": b, "distance": d})
    cands.sort(key=lambda c: -c["branch"]["group_drop_rel_pct"])
    top = cands[:3]
    top.sort(key=lambda c: -c["distance"])
    return top[0] if top else None


def evolve_arm(domain, layout, cfg, classes, base, frozen,
               rng, draw_rng, gens_per_phase):
    """One prune-graft arm: founding population of
    near-copies of `base`, evolved at the nominal rate
    throughout (flat_nominal), proven genes frozen."""
    pv_flat = be.prob_vector(
        classes, "A", cfg, frozen=frozen, flat_nominal=True)
    species = be.create_species(
        base, classes, cfg.n_species, cfg.per_species,
        draw_rng, rng, pv_flat, cfg, lineage=None,
        init_mode="transplant", next_id=0)
    return pgw.evolve_collect(
        domain, layout, cfg, classes, species, rng,
        {"A": pv_flat, "B": pv_flat},
        gens_per_phase=gens_per_phase)


def run_round(domain, layout, cfg, classes, subject,
              donor_bests, seeds, gens_per_phase, tag):
    """One branch prune-graft round with a grafted
    treatment arm and a prune-only control arm."""
    read = cw.density_read(domain, layout, subject)
    proven = band_proven(read)
    frozen = frozen_mask_from_names(layout, proven)
    branches = bb.find_branches(domain, layout, subject)
    slots = bb.cleared_slots(read["units"])
    donor = pick_donor_branch(layout, subject, donor_bests)
    out = {"tag": tag, "subject_gap_pct":
           float(-100.0 * domain.validation_score(
               layout, subject)),
           "subject_branches": branches,
           "proven_units": proven,
           "n_frozen_genes": int(frozen.sum()),
           "n_slots": len(slots)}
    log(f"{tag}: density band "
        f"{sum(1 for u in read['units'] if u['drop'] > 0)}"
        f"/{len(read['units'])}; proven {len(proven)}; "
        f"slots {len(slots)}; branches "
        f"{[round(b['group_drop_rel_pct'] or 0, 1) for b in branches]}")
    if donor is None or not slots:
        out["graft"] = None
        out["reason_no_graft"] = (
            "no donor branch" if donor is None
            else "no cleared slots")
        log(f"{tag}: NO GRAFT ({out['reason_no_graft']})")
        return out
    grafted, pairs, ungrafted = bb.graft_branch(
        layout, subject, donor["genome"], donor["branch"],
        slots)
    out["graft"] = {"donor_species_id": donor["species_id"],
                    "donor_branch_id":
                        donor["branch"]["branch_id"],
                    "donor_group_drop_rel_pct":
                        donor["branch"]["group_drop_rel_pct"],
                    "donor_distance": donor["distance"],
                    "pairs": pairs, "ungrafted": ungrafted}
    log(f"{tag}: graft {len(pairs)} pairs from donor sp"
        f"{donor['species_id']} branch "
        f"{donor['branch']['branch_id']} "
        f"(drop {donor['branch']['group_drop_rel_pct']:.1f}% "
        f"of gap, distance {donor['distance']:.3f})")
    # pruned-only base for the control: same cleared
    # units zeroed, no graft
    pruned = np.asarray(subject, dtype=float).copy()
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for s in slots:
        pruned[be._silence_indices(layout, by_name[s["name"]])] = 0.0
    rng_t = np.random.default_rng(seeds["run_seed"])
    draw_t = np.random.default_rng(seeds["draw_seed"])
    res_t = evolve_arm(domain, layout, cfg, classes,
                       grafted, frozen, rng_t, draw_t,
                       gens_per_phase)
    rng_c = np.random.default_rng(seeds["run_seed"] + 50)
    draw_c = np.random.default_rng(seeds["draw_seed"] + 50)
    res_c = evolve_arm(domain, layout, cfg, classes,
                       pruned, frozen, rng_c, draw_c,
                       gens_per_phase)
    out["treatment"] = {
        "champion_gap_pct":
            float(-100.0 * res_t["champion_val_score"]),
        "champion_species_id": res_t["champion_species_id"],
        "species_bests": res_t["species_bests"]}
    out["control"] = {
        "champion_gap_pct":
            float(-100.0 * res_c["champion_val_score"]),
        "champion_species_id": res_c["champion_species_id"],
        "species_bests": res_c["species_bests"]}
    # champion genomes ride in memory only (saved as
    # .npz by the caller, stripped before the JSON is
    # written); they are what round 2 continues from
    out["treatment_champion_genome"] = \
        res_t["champion_genome"]
    out["control_champion_genome"] = res_c["champion_genome"]
    t_gap = out["treatment"]["champion_gap_pct"]
    c_gap = out["control"]["champion_gap_pct"]
    out["margin_rel_pct"] = float(100.0 * (c_gap - t_gap) / c_gap)
    # re-prove the grafted branch in the treatment
    # champion: does the branch still carry weight?
    champ_t = res_t["champion_genome"]
    branches_t = bb.find_branches(domain, layout, champ_t)
    out["treatment_branches"] = branches_t
    log(f"{tag}: treatment {t_gap:.2f}% vs control "
        f"{c_gap:.2f}% (margin {out['margin_rel_pct']:+.1f}% "
        f"relative); post branches "
        f"{[round(b['group_drop_rel_pct'] or 0, 1) for b in branches_t]}")
    return out


def main():
    t_start = time.time()
    shape_name = "deep"
    smoke = False
    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--smoke":
            smoke = True
        elif a == "--shape" and i + 1 < len(args):
            shape_name = args[i + 1]
            i += 1
        elif a.startswith("--shape="):
            shape_name = a.split("=", 1)[1]
        else:
            raise SystemExit(f"unknown argument: {a}")
        i += 1
    if shape_name not in SHAPES:
        raise SystemExit(
            f"unknown shape: {shape_name} (choose: "
            + ", ".join(SHAPES) + ")")
    shape = SHAPES[shape_name]
    cfg = be.EngineConfig.load(CFG_PATH)
    cfg = dataclasses.replace(
        cfg, d=shape["d"], ffn_hidden=shape["ffn_hidden"],
        initial_blocks=shape["initial_blocks"],
        initial_heads=shape["initial_heads"],
        run_seed=SEEDS["run_seed"],
        draw_seed=SEEDS["draw_seed"])
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)
    arch = be.initial_arch(cfg, 2)
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    classes = be._base_classes(layout, 0)
    gens_found = 2 if smoke else 30
    gens_round = 2 if smoke else 25
    log(f"shape {shape_name}: {arch.signature()}, "
        f"{layout.n_params} genes, "
        f"{len(be.silence_units(layout))} units; "
        f"founding {gens_found}x2 gens, rounds "
        f"{gens_round}x2 gens")

    # ---- founding run from zero (stage-0 precedent) ----
    rng = np.random.default_rng(cfg.run_seed)
    draw_rng = np.random.default_rng(cfg.draw_seed)
    pv = {ph: be.prob_vector(classes, ph, cfg)
          for ph in ("A", "B")}
    species = be.create_species(
        np.zeros(layout.n_params), classes,
        cfg.n_species, cfg.per_species, draw_rng, rng,
        pv["A"], cfg, lineage=None, init_mode="random",
        next_id=0)
    found = pgw.evolve_collect(
        domain, layout, cfg, classes, species, rng, pv,
        gens_per_phase=gens_found)
    champ = found["champion_genome"]
    champ_gap = float(-100.0 * found["champion_val_score"])
    log(f"founding champion: {champ_gap:.2f}% gap, "
        f"species {found['champion_species_id']}, "
        f"{found['n_species_final']} species survive")
    results = {
        "experiment": "branch prune-and-graft scale test "
                      "from zero",
        "date": "2026-11-24", "shape": shape_name,
        "arch_signature": arch.signature(),
        "n_genes": int(layout.n_params),
        "seeds": dict(SEEDS, val_seed=cfg.val_seed),
        "founding": {"champion_gap_pct": champ_gap,
                     "champion_species_id":
                         found["champion_species_id"],
                     "n_species_final":
                         found["n_species_final"]},
        "rounds": []}
    be.atomic_npz(HERE / "branch_pg_scale_test_2026-11-24_"
                  "founding_champion.npz",
                  genome=np.asarray(champ, dtype=float),
                  val_score=float(found["champion_val_score"]))

    # donor branches for every surviving species best
    donor_bests = []
    for sp in found["species_bests"]:
        donor_bests.append({
            "species_id": sp["species_id"],
            "genome": sp["genome"],
            "branches": bb.find_branches(
                domain, layout, sp["genome"])})
    # ---- round 1 ----
    r1 = run_round(domain, layout, cfg, classes, champ,
                   donor_bests, SEEDS, gens_round, "round1")
    results["rounds"].append(r1)
    # ---- round 2: continue from the better champion ----
    if r1.get("treatment") is not None:
        t_gap = r1["treatment"]["champion_gap_pct"]
        c_gap = r1["control"]["champion_gap_pct"]
        if t_gap <= c_gap:
            r2_subject = r1["treatment_champion_genome"]
            r2_bests = r1["treatment"]["species_bests"]
        else:
            r2_subject = r1["control_champion_genome"]
            r2_bests = r1["control"]["species_bests"]
        r2_donors = []
        for sp in r2_bests:
            r2_donors.append({
                "species_id": sp["species_id"],
                "genome": sp["genome"],
                "branches": bb.find_branches(
                    domain, layout, sp["genome"])})
        r2 = run_round(domain, layout, cfg, classes,
                       r2_subject, r2_donors,
                       {"run_seed": SEEDS["run_seed"] + 100,
                        "draw_seed": SEEDS["draw_seed"] + 100},
                       gens_round, "round2")
        results["rounds"].append(r2)
    # save champion genomes as .npz and strip them
    # (and the species-bests genomes) from the JSON
    for ri, rnd in enumerate(results["rounds"]):
        for key in ("treatment_champion_genome",
                    "control_champion_genome"):
            if key in rnd:
                be.atomic_npz(
                    HERE / f"branch_pg_scale_test_2026-11-24_"
                    f"round{ri + 1}_{key}.npz",
                    genome=np.asarray(rnd[key], dtype=float))
                del rnd[key]
        for arm in ("treatment", "control"):
            for sp in rnd.get(arm, {}).get(
                    "species_bests", []):
                sp.pop("genome", None)
    results["elapsed_seconds"] = round(time.time() - t_start, 1)
    be.atomic_json(OUT_JSON, results)
    log(f"DONE in {results['elapsed_seconds']}s -> {OUT_JSON}")
    print("BRANCH_PG_SCALE_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
