#!/usr/bin/env python3
"""WIDEN TEST for the parametric BERT evolution engine
(2026-11-23). Standalone experiment, NOT an engine
integration. Continues prune_graft_test_2026-11-23.py and
prune_graft_rounds_2026-11-23.py: same domain/config
(bert_engine_config_test.json, fixed 200-problem TSP
validation set, numpy backend, CPU x64), same primitives
evolution loop (be.create_species / breed_species /
cull_species; Schedule A; species best-ever on train
scores; stage champion = validation argmax over species
bests), extended only with (a) caller-supplied per-phase
prob vectors and (b) the archive usefulness tracker hook
placed exactly where the runner places it (after the
evaluation fold, before culling/breeding).

SUBJECT: the best champion currently held, round 2's
control champion (prune_graft_rounds_2026-11-23_r2_
control_champion.npz, val -0.10783267537761247 = 10.78%
mean gap; d 8, FFN 16, 2 blocks, 2 heads, 1,880 genes).

FREEZE RULE (sponsor, 2026-11-23): units whose silencing
drop clears the paired-noise band upward (contributing)
are marked useful; their genes mutate at rate 0. Marks
are a per-gene boolean mask built from the subject's
table (be.unit_gene_indices over contributing units),
applied on top of the class prob vector
(pv = prob_vector(classes, phase, cfg); pv[frozen] = 0).
Through a transplant the mask is carried by the engine's
by-name prefix remap (be.remap_vector_by_entries, the
remap_marks implementation).

ARMS (identical budget: 50 generations, 12 species x 10,
phase A 25 + phase B 25, Schedule A, seeds run 20261500 /
draw 20261501 -- the rounds seed family advanced one step
past round 2's 20261400/20261401; each arm freshly
initialises its run/draw RNGs from these seeds):
  ARM C control: no widening. Classes are the prune-graft
  continuation classes be._base_classes(layout, 0); freeze
  marks applied in both phases.
  ARM A plain widen: be.make_transplant("widen", variant
  "A", widen_step = cfg.widen_step = 4, d 8 -> 12).
  PRESERVATION GATE FIRST, the runner's protocol:
  representative = template + a fresh draw on the
  CLS_DRAW positions from default_rng(arm draw seed + 999)
  (the runner uses cfg.draw_seed + 999), checked with
  domain.preservation(layout_old, subject, layout_new,
  rep) -- exact. Gate rejection ends the arm unevolved.
  If it passes: species are created on the template;
  phase A evolves only the new genes (transplant classes:
  old genes are CLS_OLD, prob 0 in phase A); phase B runs
  the whole network with old genes at mut_p *
  old_rate_factor and useful-marked old genes at 0.
  ARM B split-widen: be.make_transplant("widen", variant
  "B", split_units = be.split_units_from_table(subject
  table, cfg.widen_step) -- top 4 FFN units per block by
  drop; FFN 16 -> 20). Gate on the PRE-NOISE template
  (same representative protocol; the split transplant has
  no CLS_DRAW positions, so the representative is the
  template itself). If it passes, be.apply_split_noise
  (copy genes only, w * (1+u), u = sign x
  Uniform[cfg.split_noise_lo, cfg.split_noise_hi]) is
  applied with the arm's run RNG, mirroring the runner's
  consumption order (noise before species creation).
  Split units' marks lapse (their W_2 row is halved, not
  verbatim); every other old unit keeps its marks. Same
  two-phase evolution and rates as arm A.

SCALING MEASUREMENT (winning widened arm's champion):
exhaustive silencing table (the AFTER density read);
total unit count; the freeze-rule EXEMPT count = subject
units marked useful whose original genes all survive the
transplant intact (verified per gene through the
identity gene map: remap of arange+1 by entry name, plus
template value equality) -- for arm B additionally
requiring the unit was not split; and the archive
shortlist agreement: bert_archive.UsefulnessTracker
(the archive's per-species usefulness tracker; no class
literally named ArchiveTracker exists in bert_archive)
is driven from the loop for arms A and B, and for the
champion's species the per-unit mean log-score ranks the
units; the bottom 50% form the shortlist prune-candidate
set, compared against the exhaustive table's drop <= 0
set (overlap counts).

Approximations, plainly: evolution is the prune-graft
primitives loop, not the runner (the runner cannot be
reused whole for a one-off arm comparison); checkpoint /
monitor / cataclysm omitted (none applied in the earlier
arms either); the archive record npz writing is omitted,
only the usefulness tracker is driven.
"""
import json
import time
import html as htmlmod
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_archive as ba
import bert_domain_tsp as tsp
from bert_backend import get_backend

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "engine_runs" / "test_minimal_a"
CFG_PATH = HERE / "bert_engine_config_test.json"
SUBJECT_PATH = HERE / ("prune_graft_rounds_2026-11-23_r2_"
                       "control_champion.npz")
OUT_JSON = HERE / "widen_test_2026-11-23.json"
OUT_HTML = HERE / "widen_test_2026-11-23.html"
GENS_PER_PHASE = 25
RUN_SEED = 20261500          # rounds family, one step past
DRAW_SEED = 20261501         # round 2's 20261400 / 20261401
SUBJECT_STORED_VAL = -0.10783267537761247


def log(msg):
    print(f"[widen-test] {msg}", flush=True)


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
# evolution: the prune-graft primitives loop, with caller-
# supplied per-phase prob vectors and the archive tracker
# hook (runner placement: after the evaluation fold,
# before culling / breeding; culled species are removed
# from the tracker)
# ---------------------------------------------------------------
def evolve(domain, layout, cfg, classes, species, rng,
           pv_by_phase, tracker=None,
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
            if tracker is not None:
                for sp in species:
                    tracker.update(sp["id"], sp["pop"],
                                   sp["lens"])
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
                if culled and tracker is not None:
                    for did in entry[
                            "eliminated_species_ids"]:
                        tracker.remove(int(did))
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
    """Class prob vectors with the freeze rule on top:
    useful-marked genes at rate 0 in both phases."""
    out = {}
    for phase in ("A", "B"):
        pv = be.prob_vector(classes, phase, cfg, marks=None)
        pv = pv.copy()
        pv[frozen_mask] = 0.0
        out[phase] = pv
    return out


def entries_of(layout):
    return [(n, sh) for n, _, sh in layout.entries]


def gene_map_positions(layout_old, layout_new):
    """Identity gene map: new_index_of_old[i] = position in
    the new flat genome where old gene i sits, or -1 if it
    does not survive. Built by remapping arange+1 through
    be.remap_vector_by_entries (the by-name prefix rule)."""
    n_old = layout_old.n_params
    ident = np.arange(1, n_old + 1, dtype=float)
    mapped = be.remap_vector_by_entries(
        ident, entries_of(layout_old), entries_of(layout_new))
    pos = np.full(n_old, -1, dtype=int)
    hits = mapped > 0
    pos[mapped[hits].astype(int) - 1] = np.nonzero(hits)[0]
    return pos


def intact_units(layout_old, layout_new, subject, template,
                 units):
    """For each unit dict in `units` (old layout): does
    every one of its original genes survive in the
    transplanted template, at its gene-map position, with
    its exact original value? Returns {unit_key: bool}."""
    pos = gene_map_positions(layout_old, layout_new)
    out = {}
    for u in units:
        idxs = be.unit_gene_indices(layout_old, u)
        ok = True
        for g in idxs:
            p = pos[g]
            if p < 0 or template[p] != subject[g]:
                ok = False
                break
        out[unit_key(u)] = bool(ok)
    return out


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    ck = np.load(RUN_DIR / "checkpoint.npz", allow_pickle=True)
    arch = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    subj_z = np.load(SUBJECT_PATH)
    subject = np.asarray(subj_z["genome"], dtype=float)
    stored_val = float(subj_z["val_score"])
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-12
    repro_val = domain.validation_score(layout, subject)
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    log(f"subject val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% mean gap), "
        f"{layout.n_params} genes, arch {arch.signature()}")

    results = {
        "experiment": "widen test (standalone, not an "
                      "engine integration)",
        "date": "2026-11-23",
        "subject": {
            "file": str(SUBJECT_PATH),
            "arch_signature": arch.signature(),
            "n_genes": int(layout.n_params),
            "stored_val_score": stored_val,
            "reproduced_val_score": float(repro_val),
            "mean_gap_percent": float(-100.0 * repro_val),
        },
        "settings": {
            "config_file": str(CFG_PATH),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "seeds": {"run_seed": RUN_SEED,
                      "draw_seed": DRAW_SEED,
                      "gate_draw_seed": DRAW_SEED + 999,
                      "val_seed": cfg.val_seed},
            "seed_rule": "rounds seed family advanced one "
                         "step past round 2 (20261400/"
                         "20261401); each arm freshly "
                         "initialises run/draw RNGs from "
                         "these seeds; the preservation "
                         "gate's representative draw uses "
                         "arm draw seed + 999 (the runner's "
                         "cfg.draw_seed + 999 protocol)",
            "widen_step": cfg.widen_step,
            "split_noise": {"lo": cfg.split_noise_lo,
                            "hi": cfg.split_noise_hi},
            "rates": {"mut_p": cfg.mut_p,
                      "old_rate_factor": cfg.old_rate_factor,
                      "phase_B_old_rate":
                          cfg.mut_p * cfg.old_rate_factor},
            "freeze_rule": "units whose silencing drop "
                           "clears the paired-noise band "
                           "upward (contributing) are marked "
                           "useful; their genes mutate at "
                           "rate 0 in both phases (mask "
                           "applied over the class prob "
                           "vector)",
            "classes_rule": "arm C: be._base_classes"
                            "(layout, 0) (prune-graft "
                            "continuation classes). Arms "
                            "A/B: the transplant's own "
                            "classes from be.make_transplant",
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
        "arms": {},
    }

    # ---------------- PREP: table + freeze marks ----------------
    subj_read = density_read(domain, layout, subject)
    results["subject"]["density_exact_zero_positive"] = \
        subj_read["exact_zero_positive"]
    results["subject"]["density_band_counts"] = \
        subj_read["band_counts"]
    results["subject"]["table_units"] = subj_read["units"]
    useful_units = [u for u in subj_read["units"]
                    if u["cat"] == "contributing"]
    frozen_mask_old = np.zeros(layout.n_params, dtype=bool)
    for u in useful_units:
        frozen_mask_old[be.unit_gene_indices(layout, u)] = True
    results["subject"]["marks"] = {
        "rule": "contributing (drop > +2 paired SE) -> "
                "useful, genes frozen at mutation rate 0",
        "useful_units": [u["name"] for u in useful_units],
        "n_useful_units": len(useful_units),
        "n_frozen_genes": int(frozen_mask_old.sum()),
        "frozen_gene_indices": [int(i) for i in
                                np.nonzero(frozen_mask_old)[0]],
    }
    log(f"PREP density: exact-zero positive "
        f"{subj_read['exact_zero_positive']}/"
        f"{subj_read['n_units']}; band "
        f"{subj_read['band_counts']}; useful-marked "
        f"{len(useful_units)} units, "
        f"{int(frozen_mask_old.sum())} genes frozen")
    raw_table = be.silence_table(
        layout, subject,
        lambda G: domain.validation_scores(layout, G),
        zero_tol=domain.silence_zero_tol)

    def run_arm(arm_key, arm_layout, classes, base_genome,
                frozen_mask, tracker=None):
        pv = prob_vectors(classes, cfg, frozen_mask)
        rng = np.random.default_rng(RUN_SEED)
        draw_rng = np.random.default_rng(DRAW_SEED)
        species = be.create_species(
            base_genome, classes, cfg.n_species,
            cfg.per_species, draw_rng, rng, pv["A"], cfg,
            lineage=None, init_mode="transplant", next_id=0)
        t0 = time.time()
        out = evolve(domain, arm_layout, cfg, classes,
                     species, rng, pv, tracker=tracker)
        wall = time.time() - t0
        after = density_read(domain, arm_layout,
                             out["champion_genome"])
        np.savez(
            HERE / f"widen_test_2026-11-23_{arm_key}_"
                   f"champion.npz",
            genome=out["champion_genome"],
            val_score=out["champion_val_score"])
        block = {
            "status": "evolved",
            "final_champion_val_score":
                out["champion_val_score"],
            "mean_gap_percent":
                float(-100.0 * out["champion_val_score"]),
            "delta_vs_subject":
                float(out["champion_val_score"] - repro_val),
            "champion_species_id": out["champion_species_id"],
            "n_species_final": out["n_species_final"],
            "genome_size": int(arm_layout.n_params),
            "wall_seconds": wall,
            "density_exact_zero_positive":
                after["exact_zero_positive"],
            "density_band_counts": after["band_counts"],
            "density_units": after["units"],
            "history": out["history"],
        }
        log(f"{arm_key}: final champion val "
            f"{out['champion_val_score']:.6f} "
            f"({-100.0 * out['champion_val_score']:.2f}% gap; "
            f"delta vs subject "
            f"{out['champion_val_score'] - repro_val:+.6f}); "
            f"density exact-zero "
            f"{after['exact_zero_positive']}/"
            f"{after['n_units']}, band {after['band_counts']}; "
            f"{wall:.1f}s")
        return block, out, after

    # ---------------- ARM A: plain widen ----------------
    arch_a, layout_a, template_a, classes_a = \
        be.make_transplant("widen", arch, layout, subject,
                           tsp.head_spec, cfg, tsp.widen_head,
                           variant="A")
    rng0 = np.random.default_rng(DRAW_SEED + 999)
    draw_mask_a = classes_a == be.CLS_DRAW
    rep_a = template_a.copy()
    rep_a[draw_mask_a] = rng0.normal(
        0, cfg.draw_sigma, int(draw_mask_a.sum()))
    pres_a = domain.preservation(layout, subject, layout_a,
                                 rep_a)
    frozen_a = be.remap_vector_by_entries(
        frozen_mask_old.astype(float), entries_of(layout),
        entries_of(layout_a)) > 0.5
    intact_a = intact_units(layout, layout_a, subject,
                            template_a, useful_units)
    arm_a = {
        "kind": "plain widen (variant A)",
        "arch_signature": arch_a.signature(),
        "genome_size": int(layout_a.n_params),
        "n_draw_genes": int(draw_mask_a.sum()),
        "preservation_gate": pres_a,
        "n_frozen_genes_carried": int(frozen_a.sum()),
        "useful_units_intact": {
            next(u["name"] for u in useful_units
                 if unit_key(u) == k): v
            for k, v in intact_a.items()},
        "n_useful_units_intact": int(sum(intact_a.values())),
    }
    log(f"ARM A transplant: d {arch.d} -> {arch_a.d}, "
        f"{layout_a.n_params} genes; gate "
        f"{'PASS' if pres_a['pass'] else 'REJECT'} "
        f"(max abs length diff "
        f"{pres_a['max_abs_length_difference']:.3g}, tours "
        f"identical {pres_a['tours_identical']})")
    tracker_a = None
    out_a = after_a = None
    if pres_a["pass"]:
        tracker_a = ba.UsefulnessTracker(
            layout_a.n_params,
            usefulness_step=cfg.usefulness_step)
        block_a, out_a, after_a = run_arm(
            "armA", layout_a, classes_a, template_a,
            frozen_a, tracker=tracker_a)
        arm_a.update(block_a)
    else:
        arm_a["status"] = "gate-rejected"
        log("ARM A gate-rejected; arm not evolved "
            "(recorded as the result)")
    results["arms"]["A_plain_widen"] = arm_a

    # ---------------- ARM B: split-widen ----------------
    split_units = be.split_units_from_table(raw_table,
                                            cfg.widen_step)
    arch_b, layout_b, template_b, classes_b = \
        be.make_transplant("widen", arch, layout, subject,
                           tsp.head_spec, cfg, tsp.widen_head,
                           variant="B",
                           split_units=split_units)
    draw_mask_b = classes_b == be.CLS_DRAW
    rep_b = template_b.copy()
    if draw_mask_b.any():
        rng0b = np.random.default_rng(DRAW_SEED + 999)
        rep_b[draw_mask_b] = rng0b.normal(
            0, cfg.draw_sigma, int(draw_mask_b.sum()))
    pres_b = domain.preservation(layout, subject, layout_b,
                                 rep_b)
    frozen_b = be.remap_vector_by_entries(
        frozen_mask_old.astype(float), entries_of(layout),
        entries_of(layout_b)) > 0.5
    split_keys = {(c, b, "ffn", u) for (c, b, u)
                  in split_units}
    for (c, b, u) in split_units:
        frozen_b[be.unit_gene_indices(
            layout_b, {"kind": "ffn", "couple": c,
                       "block": b, "index": u})] = False
    intact_b_all = intact_units(layout, layout_b, subject,
                                template_b, useful_units)
    intact_b = {k: (v and k not in split_keys)
                for k, v in intact_b_all.items()}
    arm_b = {
        "kind": "split-widen (variant B, Net2WiderNet)",
        "arch_signature": arch_b.signature(),
        "genome_size": int(layout_b.n_params),
        "split_units": [list(s) for s in split_units],
        "n_draw_genes": int(draw_mask_b.sum()),
        "preservation_gate_pre_noise": pres_b,
        "n_frozen_genes_carried": int(frozen_b.sum()),
        "useful_units_intact_unsplit": {
            next(u["name"] for u in useful_units
                 if unit_key(u) == k): v
            for k, v in intact_b.items()},
        "n_useful_units_intact_unsplit":
            int(sum(intact_b.values())),
    }
    log(f"ARM B transplant: FFN {arch.ffn_hidden} -> "
        f"{arch_b.ffn_hidden}, {layout_b.n_params} genes; "
        f"split units {split_units}; gate "
        f"{'PASS' if pres_b['pass'] else 'REJECT'} "
        f"(max abs length diff "
        f"{pres_b['max_abs_length_difference']:.3g}, tours "
        f"identical {pres_b['tours_identical']})")
    tracker_b = None
    out_b = after_b = None
    if pres_b["pass"]:
        rng_noise = np.random.default_rng(RUN_SEED)
        noisy_b = be.apply_split_noise(
            template_b, layout_b, split_units, rng_noise,
            cfg.split_noise_lo, cfg.split_noise_hi)
        rep_noisy = noisy_b.copy()
        if draw_mask_b.any():
            rep_noisy[draw_mask_b] = rep_b[draw_mask_b]
        arm_b["split_noise_val_delta"] = float(
            domain.validation_score(layout_b, rep_noisy)
            - domain.validation_score(layout_b, rep_b))
        log(f"ARM B split noise applied; representative "
            f"val delta {arm_b['split_noise_val_delta']:+.6g}")
        tracker_b = ba.UsefulnessTracker(
            layout_b.n_params,
            usefulness_step=cfg.usefulness_step)
        # NOTE: run_arm re-initialises its own run RNG from
        # RUN_SEED; the noise draw above consumed a separate
        # fresh stream, stated here for reproducibility.
        block_b, out_b, after_b = run_arm(
            "armB", layout_b, classes_b, noisy_b, frozen_b,
            tracker=tracker_b)
        arm_b.update(block_b)
    else:
        arm_b["status"] = "gate-rejected"
        log("ARM B gate-rejected (pre-noise); arm not "
            "evolved (recorded as the result)")
    results["arms"]["B_split_widen"] = arm_b

    # ---------------- ARM C: control ----------------
    classes_c = be._base_classes(layout, 0)
    block_c, out_c, after_c = run_arm(
        "armC", layout, classes_c, subject, frozen_mask_old)
    block_c["kind"] = "control (no widening, freeze marks)"
    block_c["arch_signature"] = arch.signature()
    results["arms"]["C_control"] = block_c

    ctrl = block_c["final_champion_val_score"]
    for key in ("A_plain_widen", "B_split_widen"):
        if results["arms"][key].get("status") == "evolved":
            results["arms"][key]["delta_vs_control"] = float(
                results["arms"][key][
                    "final_champion_val_score"] - ctrl)

    # ---------------- SCALING MEASUREMENT ----------------
    evolved_widened = []
    if out_a is not None:
        evolved_widened.append(
            ("A_plain_widen", out_a, after_a, layout_a,
             tracker_a, intact_a, arm_a))
    if out_b is not None:
        evolved_widened.append(
            ("B_split_widen", out_b, after_b, layout_b,
             tracker_b, intact_b, arm_b))
    scaling = {"status": "no widened arm evolved"}
    if evolved_widened:
        win = max(evolved_widened,
                  key=lambda t: t[1]["champion_val_score"])
        (wkey, wout, wafter, wlayout, wtracker, wintact,
         wblock) = win
        n_units = wafter["n_units"]
        exempt = int(sum(wintact.values()))
        scaling = {
            "status": "measured",
            "winning_arm": wkey,
            "winning_champion_val_score":
                wout["champion_val_score"],
            "total_units": n_units,
            "exempt_units_freeze_rule": exempt,
            "units_needing_silencing": n_units - exempt,
            "exempt_definition": "subject units marked "
                "useful whose original genes all survive "
                "the transplant intact at their gene-map "
                "positions with exact original values"
                + (" and which were not split"
                   if wkey == "B_split_widen" else ""),
        }
        # shortlist agreement from the winner's tracker
        try:
            vec = wtracker.vector(
                wout["champion_species_id"])
            per_unit = []
            for u in be.silence_units(wlayout):
                idxs = be.unit_gene_indices(wlayout, u)
                per_unit.append({
                    "name": u["name"],
                    "mean_log_usefulness":
                        float(vec[idxs].mean())})
            ranked = sorted(
                per_unit,
                key=lambda r: r["mean_log_usefulness"])
            n_short = n_units // 2
            shortlist = [r["name"] for r in ranked[:n_short]]
            actual = [u["name"] for u in wafter["units"]
                      if u["drop"] <= 0.0]
            overlap = sorted(set(shortlist) & set(actual))
            scaling["shortlist_agreement"] = {
                "tracker": "bert_archive.UsefulnessTracker "
                           "driven from the evolution loop "
                           "(runner hook placement)",
                "champion_species_id":
                    wout["champion_species_id"],
                "unit_aggregation": "mean of per-weight "
                    "log-scores over the unit's genes "
                    "(new layout)",
                "shortlist_rule": "bottom 50% of units by "
                    "aggregated usefulness",
                "n_shortlist": len(shortlist),
                "n_actual_drop_le_0": len(actual),
                "n_overlap": len(overlap),
                "shortlist_units": shortlist,
                "actual_units": actual,
                "overlap_units": overlap,
                "tracker_density_shares_champion_species":
                    wtracker.density(
                        wout["champion_species_id"]),
            }
            log(f"SCALING ({wkey}): {n_units} units, "
                f"{exempt} exempt under the freeze rule, "
                f"{n_units - exempt} would need silencing; "
                f"shortlist {len(shortlist)} vs actual "
                f"{len(actual)}, overlap {len(overlap)}")
        except Exception as exc:  # noqa: BLE001
            scaling["shortlist_agreement"] = {
                "status": f"tracker read failed: {exc!r}"}
            log(f"SCALING: tracker read failed: {exc!r}")
    results["scaling"] = scaling

    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("WIDEN_TEST_DONE")


# ---------------------------------------------------------------
# summary page: arms table + gates + density bars + scaling
# (counts and bars only, no per-neuron tiles)
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
                f"<td>{n}</td>"
                f"<td>{_bar(100.0 * n / total if total else 0, colour)}</td></tr>")
    return out


def write_html(results):
    subj = results["subject"]
    arms = results["arms"]
    ctrl = arms["C_control"]
    rows = ""
    labels = {"C_control": "C: control (no widening)",
              "A_plain_widen": "A: plain widen (d 8 to 12)",
              "B_split_widen": "B: split-widen (FFN 16 to 20)"}
    for key in ("C_control", "A_plain_widen", "B_split_widen"):
        a = arms[key]
        if a.get("status") == "evolved":
            rows += (
                f"<tr><td>{labels[key]}</td>"
                f"<td>{a['final_champion_val_score']:+.4f}</td>"
                f"<td>{a['mean_gap_percent']:.2f}%</td>"
                f"<td>{a['delta_vs_subject']:+.4f}</td>"
                f"<td>{a.get('delta_vs_control', 0.0):+.4f}</td>"
                f"<td>{a['genome_size']}</td>"
                f"<td>{a['density_exact_zero_positive']}</td>"
                f"</tr>")
        else:
            rows += (
                f"<tr><td>{labels[key]}</td>"
                f"<td colspan='6'>gate-rejected at the "
                f"preservation gate; not evolved</td></tr>")
    gate_rows = ""
    for key, gname in (("A_plain_widen",
                        "preservation_gate"),
                       ("B_split_widen",
                        "preservation_gate_pre_noise")):
        g = arms[key][gname]
        gate_rows += (
            f"<tr><td>{labels[key]}</td>"
            f"<td>{'PASS' if g['pass'] else 'REJECT'}</td>"
            f"<td>{g['max_abs_length_difference']:.3g}</td>"
            f"<td>{g['tours_identical']}</td></tr>")
    total_subj = len(subj["table_units"])
    dens = _cat_rows("subject (before)",
                     subj["density_band_counts"], total_subj)
    for key in ("C_control", "A_plain_widen", "B_split_widen"):
        a = arms[key]
        if a.get("status") == "evolved":
            tot = len(a["density_units"])
            dens += _cat_rows(labels[key],
                              a["density_band_counts"], tot)
    sc = results["scaling"]
    if sc.get("status") == "measured":
        sa = sc.get("shortlist_agreement", {})
        sc_html = f"""
<h2>Scaling measurement (winning widened arm: {sc['winning_arm']})</h2>
<table>
<tr><td>total units in the widened champion</td>
<td>{sc['total_units']}</td></tr>
<tr><td>exempt from silencing in a follow-on prune-graft
round (freeze rule)</td><td>{sc['exempt_units_freeze_rule']}</td></tr>
<tr><td>units that would still need silencing</td>
<td>{sc['units_needing_silencing']}</td></tr>
<tr><td>archive shortlist size (bottom 50% by usefulness)</td>
<td>{sa.get('n_shortlist', 'n/a')}</td></tr>
<tr><td>exhaustive table's actual drop &lt;= 0 set</td>
<td>{sa.get('n_actual_drop_le_0', 'n/a')}</td></tr>
<tr><td>overlap between shortlist and actual</td>
<td>{sa.get('n_overlap', 'n/a')}</td></tr>
</table>"""
    else:
        sc_html = ("<h2>Scaling measurement</h2><p class="
                   "'small'>No widened arm evolved, so no "
                   "scaling measurement was taken.</p>")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Widen test 2026-11-23</title>
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
<h1>Widen test: plain widen vs split-widen vs control</h1>
<p class="small">Subject: round 2's control champion,
validation score {subj['reproduced_val_score']:+.4f}
({_gap(subj['reproduced_val_score'])} mean gap),
{subj['n_genes']} genes. Useful-marked under the freeze
rule: {subj['marks']['n_useful_units']} units
({subj['marks']['n_frozen_genes']} genes at mutation rate
0). Every arm: 50 generations, 12 species x 10, same fixed
200-problem TSP validation set. Scores are validation
scores (higher is better); gap is the mean tour gap
(lower is better).</p>
<h2>Preservation gates</h2>
<table><tr><th>arm</th><th>gate</th>
<th>max abs length difference</th><th>tours identical</th>
</tr>{gate_rows}</table>
<h2>Arms</h2>
<table><tr><th>arm</th><th>final score</th><th>mean gap</th>
<th>delta vs subject</th><th>delta vs control</th>
<th>genome size</th><th>exact-zero positive units</th></tr>
{rows}</table>
<p class="small">Control final score:
{ctrl['final_champion_val_score']:+.4f}
({_gap(ctrl['final_champion_val_score'])}).</p>
<h2>Density by category (paired-noise band read)</h2>
<table><tr><th>genome</th><th>category</th><th>units</th>
<th></th></tr>{dens}</table>
{sc_html}
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
