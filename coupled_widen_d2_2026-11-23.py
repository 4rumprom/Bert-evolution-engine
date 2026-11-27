#!/usr/bin/env python3
"""COUPLED-WIDEN test on the NEW single-layer lineage
(2026-11-23). Standalone experiment, NOT an engine
integration; follows prune_graft_d2_l1_2026-11-23.py and
widen_test_2026-11-23.py exactly (same primitives
evolution loop: be.create_species / breed_species /
cull_species, Schedule A culling, species best-ever on
train scores, stage champion = validation argmax over
species bests; same freeze-rule marks).

LAUNCH (after the sponsor's yes):
    cd ~/workspace/bert-engine && \
    setsid nohup python3 coupled_widen_d2_2026-11-23.py \
        >> coupled_widen_d2_2026-11-23.log 2>&1 < /dev/null &

SUBJECT: the prune-graft treatment champion of the new
single-layer lineage
(prune_graft_d2_l1_2026-11-23_treatment_champion.npz, val
-0.18797565272540126 = 18.80% mean gap; d 2, FFN 8,
1 head, 1 layer, 88 genes, 9 silence units). Its band
density read: 6 contributing / 3 neutral / 0 harmful.
Config: bert_engine_config_d2_l1.json (fixed 200-problem
TSP validation set, val_seed 20261116, numpy backend, CPU
x64).

THE TRANSPLANT (new; the engine has no combined widen):
  - stream d 2 -> 6 with the engine's transplant_widen
    (cfg.widen_step = 4; embed old weights, new dimensions
    zero/mean-filled per the preserving construction);
  - in the SAME transplant, FFN 8 -> 24 (BERT ratio 4x)
    with FRESH units: new FFN units get random incoming
    weights (W_1 columns and b_1 drawn N(0, draw_sigma),
    same scale as species draws) and ZERO outgoing weights
    (W_2 rows 0), so they are silent at birth. Old 8 FFN
    units keep their weights exactly.
  - the coupled transplant must pass domain.preservation
    EXACTLY (max abs length difference 0.0, tours
    identical on all 200 validation problems). If it does
    not pass, the script records the rejection and stops;
    it does not work around the gate.
  - gene map: every old gene survives at its remapped
    position (widen_test's gene_map_positions via
    be.remap_vector_by_entries). The only old genes whose
    values change are the 4 LN gains, intentionally
    rescaled by sqrt(d0/d1) inside transplant_widen (the
    preserving LN rescale); anything else mismatching is
    a verification failure and stops the run.

FREEZE RULE (sponsor, 2026-11-23), as in the chain test:
units whose silencing drop clears the paired-noise band
upward (contributing) on the subject's table are marked
useful; their genes mutate at rate 0 in BOTH arms and
are exempt from re-silencing while inherited verbatim.
The subject's useful units are c0.b0.ffn.u5 and
c0.b0.head.h0 (29 genes); their marks carry through the
transplant's gene map, verified per gene. The 16 fresh
FFN units are unmarked (unproven).

ARM T (coupled widen): after the gate passes, phase A 50
generations evolving ONLY the new genes (CLS_IFACE fresh
FFN genes at full rate; old genes frozen, freeze-marked
units at rate 0), then phase B 50 generations whole
network, old genes at 1/10 standard mutation, useful-
marked old genes at 0.
ARM C (control): the subject evolved plainly for the
same 100 generations (50 + 50) with the same freeze
marks. Both arms: 12 species x 10, Schedule A.

SEEDS: the d2_l1 round advanced one +100 step from the
prune-graft round (run 20262100 / draw 20262101 /
graft 20262104): run 20262200, draw 20262201, gate
20262204 (the transplant's representative build: fresh
FFN incoming draws in layout order, then the d-widen
CLS_DRAW fill). Within the step, each arm freshly
initialises its run/draw RNGs from the step seeds.

Approximations, plainly: evolution is the primitives
loop, not the runner (the runner cannot be reused whole
for one-off arm comparisons); checkpoint / monitor /
cataclysm / archive tracker omitted, as in all the
earlier arms. Classes for the control arm are
be._base_classes(layout, 0), the engine's
founder-continuation rebase; classes for the widened
arm are the composed transplant classes (old genes keep
their transplant_widen classes, fresh FFN genes are
CLS_IFACE so they evolve in phase A).
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
CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
SUBJECT_PATH = HERE / "prune_graft_d2_l1_2026-11-23_" \
    "treatment_champion.npz"
OUT_JSON = HERE / "coupled_widen_d2_2026-11-23.json"
OUT_HTML = HERE / "coupled_widen_d2_2026-11-23.html"
GENS_PER_PHASE = 50

SUBJECT_STORED_VAL = -0.18797565272540126
FFN_NEW = 24
D_STEP = 4
STEP_SEEDS = {"run_seed": 20262200, "draw_seed": 20262201,
              "gate_seed": 20262204}


def log(msg):
    print(f"[cw-d2] {msg}", flush=True)


# ---------------------------------------------------------------
# density read: copied from prune_graft_d2_l1_2026-11-23.py
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
    """Class prob vectors with the freeze rule on top:
    useful-marked genes at rate 0 in both phases."""
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


def entries_of(layout):
    return [(n, sh) for n, _, sh in layout.entries]


def gene_map_positions(layout_old, layout_new):
    """Identity gene map (widen_test's helper): position in
    the new flat genome where old gene i sits, or -1."""
    n_old = layout_old.n_params
    ident = np.arange(1, n_old + 1, dtype=float)
    mapped = be.remap_vector_by_entries(
        ident, entries_of(layout_old), entries_of(layout_new))
    pos = np.full(n_old, -1, dtype=int)
    hits = mapped > 0
    pos[mapped[hits].astype(int) - 1] = np.nonzero(hits)[0]
    return pos


# ---------------------------------------------------------------
# coupled transplant: engine d-widen + fresh-unit FFN widen
# ---------------------------------------------------------------
def build_coupled_transplant(arch0, layout0, subject, cfg,
                             gate_seed):
    """d 2 -> 2 + D_STEP via transplant_widen, then FFN
    8 -> FFN_NEW with fresh units (random incoming, zero
    outgoing). Returns (arch2, layout2, template2, classes2,
    gate_rng). The representative template is built with ONE
    RNG stream (gate_seed): fresh FFN incoming draws in
    layout entry order, then the d-widen CLS_DRAW fill."""
    arch1, layout1, template1, classes1 = be.make_transplant(
        "widen", arch0, layout0, subject, tsp.head_spec, cfg,
        tsp.widen_head, variant="A")
    assert arch1.d == arch0.d + D_STEP
    arch2 = be.ArchSpec(input_dim=arch1.input_dim, d=arch1.d,
                        ffn_hidden=FFN_NEW,
                        couples=arch1.couples,
                        ln_eps=arch1.ln_eps,
                        attn_scale=arch1.attn_scale)
    layout2 = be.Layout(arch2, tsp.head_spec(arch2.d))
    d1 = layout1.unpack_dict(template1)
    d2 = {n: np.zeros(sh) for n, _, sh in layout2.entries}
    classes2 = np.zeros(layout2.n_params, dtype=np.int8)
    n_old_ffn = arch0.ffn_hidden
    for name, _, shape in layout2.entries:
        s2 = layout2.sl(name)
        if name in d1 and d1[name].shape == shape:
            d2[name] = d1[name].copy()
            classes2[s2] = classes1[layout1.sl(name)]
        elif name.endswith(".W_1"):
            old = d1[name]                    # (d, ffn_old)
            new = np.zeros(shape)             # (d, FFN_NEW)
            new[:, :old.shape[1]] = old
            d2[name] = new
            c = np.zeros(shape, dtype=np.int8)
            c[:, :old.shape[1]] = classes1[
                layout1.sl(name)].reshape(shape[0],
                                          old.shape[1])
            c[:, old.shape[1]:] = be.CLS_IFACE
            classes2[s2] = c.reshape(-1)
        elif name.endswith(".W_2"):
            old = d1[name]                    # (ffn_old, d)
            new = np.zeros(shape)             # (FFN_NEW, d)
            new[:old.shape[0], :] = old      # fresh rows stay 0
            d2[name] = new
            c = np.zeros(shape, dtype=np.int8)
            c[:old.shape[0], :] = classes1[
                layout1.sl(name)].reshape(old.shape[0],
                                          shape[1])
            c[old.shape[0]:, :] = be.CLS_IFACE
            classes2[s2] = c.reshape(-1)
        elif name.endswith(".b_1"):
            old = d1[name]                    # (ffn_old,)
            new = np.zeros(shape)
            new[:len(old)] = old
            d2[name] = new
            c = np.zeros(shape, dtype=np.int8)
            c[:len(old)] = classes1[layout1.sl(name)]
            c[len(old):] = be.CLS_IFACE
            classes2[s2] = c.reshape(-1)
        else:
            raise ValueError(
                f"unhandled grown entry {name} {shape}")
    rng = np.random.default_rng(gate_seed)
    template2 = layout2.pack_dict(d2)
    n_fresh_in = 0
    for name, _, shape in layout2.entries:
        if name.endswith(".W_1"):
            s = layout2.sl(name)
            cols = np.arange(shape[1])[n_old_ffn:]
            idx = s.start + np.array(
                [r * shape[1] + c for r in range(shape[0])
                 for c in cols])
            template2[idx] = rng.normal(0, cfg.draw_sigma,
                                        len(idx))
            n_fresh_in += len(idx)
        if name.endswith(".b_1"):
            s = layout2.sl(name)
            template2[s.start + n_old_ffn:
                      s.start + FFN_NEW] = rng.normal(
                0, cfg.draw_sigma, FFN_NEW - n_old_ffn)
            n_fresh_in += FFN_NEW - n_old_ffn
    draw_mask = classes2 == be.CLS_DRAW
    template2[draw_mask] = rng.normal(
        0, cfg.draw_sigma, int(draw_mask.sum()))
    return {
        "arch1": arch1, "layout1": layout1,
        "template1": template1, "classes1": classes1,
        "arch2": arch2, "layout2": layout2,
        "template2": template2, "classes2": classes2,
        "n_fresh_incoming_genes": int(n_fresh_in),
        "n_draw_genes_filled": int(draw_mask.sum()),
        "n_iface_genes": int((classes2 == be.CLS_IFACE).sum()),
    }


def verify_gene_map(layout0, layout2, subject, template2,
                    useful_units):
    """Every old gene survives at its remapped position.
    The only old genes allowed to change value are the LN
    gains, rescaled by sqrt(d0/d1) inside transplant_widen
    (the preserving LN rescale). Returns the verification
    record; `ok` is False if anything else mismatches."""
    pos = gene_map_positions(layout0, layout2)
    d0, d1 = layout0.arch.d, layout2.arch.d
    gain_fix = float(np.sqrt(d0 / d1))
    names = {}
    for n, s, sh in layout0.entries:
        for i in range(s.stop - s.start):
            names[s.start + i] = f"{n}[{i}]"
    n_survive = int((pos >= 0).sum())
    mismatches = []
    for g in range(layout0.n_params):
        if pos[g] < 0 or template2[pos[g]] == subject[g]:
            continue
        nm = names[g]
        ratio = (template2[pos[g]] / subject[g]
                 if subject[g] != 0 else float("nan"))
        is_ln_gain = ("lnA_gain" in nm or "lnF_gain" in nm) \
            and abs(ratio - gain_fix) < 1e-12
        mismatches.append({
            "gene": int(g), "name": nm,
            "old_value": float(subject[g]),
            "new_value": float(template2[pos[g]]),
            "ratio": float(ratio),
            "expected_ln_gain_rescale": bool(is_ln_gain)})
    ok = n_survive == layout0.n_params and all(
        m["expected_ln_gain_rescale"] for m in mismatches)
    # useful units intact, gene by gene (widen test's check)
    by_name = {u["name"]: u for u in be.silence_units(layout0)}
    intact = {}
    for u in useful_units:
        idxs = be.unit_gene_indices(layout0, by_name[u["name"]])
        intact[u["name"]] = bool(all(
            pos[g] >= 0 and template2[pos[g]] == subject[g]
            for g in idxs))
    return {
        "n_old_genes": int(layout0.n_params),
        "n_surviving_genes": n_survive,
        "n_exact_genes": int(layout0.n_params
                             - len(mismatches)),
        "ln_gain_rescale_factor": gain_fix,
        "mismatches": mismatches,
        "useful_units_intact": intact,
        "all_useful_units_intact": bool(
            all(intact.values())) if intact else False,
        "ok": bool(ok),
    }


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

    ck = np.load(HERE / "engine_runs" / "fresh_d2_l1" /
                 "checkpoint.npz", allow_pickle=True)
    arch0 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout0 = be.Layout(arch0, tsp.head_spec(arch0.d))
    champ_z = np.load(SUBJECT_PATH)
    subject = np.asarray(champ_z["genome"], dtype=float)
    stored_val = float(champ_z["val_score"])
    classes0 = be._base_classes(layout0, 0)

    results = {
        "experiment": "coupled-widen test on the new "
                      "single-layer lineage (standalone, "
                      "not an engine integration)",
        "date": "2026-11-23",
        "continues": ["prune_graft_d2_l1_2026-11-23.py",
                      "widen_test_2026-11-23.py"],
        "subject": {
            "file": str(SUBJECT_PATH),
            "arch_signature": arch0.signature(),
            "n_genes": int(layout0.n_params),
            "stored_val_score": stored_val,
            "mean_gap_percent": float(-100.0 * stored_val),
        },
        "settings": {
            "config_file": str(CFG_PATH),
            "config_canonical": json.loads(cfg.canonical()),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
            "seed_rule": "the prune-graft round's family is "
                         "run 20262100 / draw 20262101; this "
                         "test advances one +100 step (run "
                         "20262200, draw 20262201, gate "
                         "20262204). The gate seed builds the "
                         "transplant representative (fresh "
                         "FFN incoming draws in layout order, "
                         "then the d-widen CLS_DRAW fill). "
                         "Within the step, each arm freshly "
                         "initialises its run/draw RNGs from "
                         "the step seeds.",
            "classes_rule": "control arm: "
                            "be._base_classes(layout, 0) "
                            "(founder-continuation rebase). "
                            "Widened arm: composed transplant "
                            "classes (old genes keep their "
                            "transplant_widen classes; fresh "
                            "FFN genes are CLS_IFACE so they "
                            "evolve in phase A).",
            "freeze_rule": "units whose silencing drop "
                           "clears the paired-noise band "
                           "upward (contributing) on the "
                           "subject's table are marked "
                           "useful; their genes mutate at "
                           "rate 0 in both arms and are "
                           "exempt from re-silencing while "
                           "inherited verbatim; marks carry "
                           "through the transplant gene map.",
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
    repro_val = domain.validation_score(layout0, subject)
    results["subject"]["reproduced_val_score"] = float(repro_val)
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-12
    log(f"champion val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% mean gap), "
        f"{layout0.n_params} genes")
    before = density_read(domain, layout0, subject)
    results["before"] = before
    log(f"BEFORE density: exact-zero positive "
        f"{before['exact_zero_positive']}/{before['n_units']}; "
        f"band {before['band_counts']}")
    for u in before["units"]:
        log(f"  {u['name']}: drop {u['drop']:+.4f} "
            f"(band +/-{u['band']:.4f}) {u['cat']}")

    # ---------------- FREEZE PREP ----------------
    useful = [u for u in before["units"]
              if u["cat"] == "contributing"]
    frozen0 = frozen_mask_from_names(
        layout0, [u["name"] for u in useful])
    results["freeze_prep"] = {
        "useful_units": [u["name"] for u in useful],
        "n_useful_units": len(useful),
        "n_frozen_genes": int(frozen0.sum()),
        "n_genes_total": int(layout0.n_params),
    }
    log(f"FREEZE prep: {len(useful)} useful units "
        f"{[u['name'] for u in useful]}; "
        f"{int(frozen0.sum())} of {layout0.n_params} genes "
        f"at mutation rate 0")

    # ---------------- COUPLED TRANSPLANT + GATE ----------------
    built = build_coupled_transplant(
        arch0, layout0, subject, cfg, STEP_SEEDS["gate_seed"])
    layout2 = built["layout2"]
    template2 = built["template2"]
    classes2 = built["classes2"]
    pres = domain.preservation(layout0, subject, layout2,
                               template2)
    gmap = verify_gene_map(layout0, layout2, subject,
                           template2, useful)
    frozen2 = be.remap_vector_by_entries(
        frozen0.astype(float), entries_of(layout0),
        entries_of(layout2)) > 0.5
    results["transplant"] = {
        "d_old": int(arch0.d), "d_new": int(layout2.arch.d),
        "ffn_old": int(arch0.ffn_hidden),
        "ffn_new": int(layout2.arch.ffn_hidden),
        "n_genes_old": int(layout0.n_params),
        "n_genes_new": int(layout2.n_params),
        "n_units_old": int(before["n_units"]),
        "n_units_new": int(len(be.silence_units(layout2))),
        "n_fresh_ffn_units": int(FFN_NEW - arch0.ffn_hidden),
        "n_fresh_incoming_genes":
            built["n_fresh_incoming_genes"],
        "n_draw_genes_filled": built["n_draw_genes_filled"],
        "n_iface_genes": built["n_iface_genes"],
        "arch_signature_new": layout2.arch.signature(),
        "preservation_gate": pres,
        "gene_map_verification": gmap,
        "n_frozen_genes_carried": int(frozen2.sum()),
    }
    log(f"TRANSPLANT: d {arch0.d} -> {layout2.arch.d}, FFN "
        f"{arch0.ffn_hidden} -> {layout2.arch.ffn_hidden}, "
        f"{layout0.n_params} -> {layout2.n_params} genes, "
        f"{before['n_units']} -> "
        f"{len(be.silence_units(layout2))} units")
    log(f"GATE: {'PASS' if pres['pass'] else 'REJECT'} "
        f"(max abs length diff "
        f"{pres['max_abs_length_difference']:.3g}, tours "
        f"identical {pres['tours_identical']}, "
        f"{pres['instances']} problems)")
    log(f"GENE MAP: {gmap['n_surviving_genes']}/"
        f"{gmap['n_old_genes']} old genes survive, "
        f"{gmap['n_exact_genes']} exact; "
        f"{len(gmap['mismatches'])} mismatches, all the "
        f"LN-gain rescale: "
        f"{all(m['expected_ln_gain_rescale'] for m in gmap['mismatches'])}; "
        f"useful units intact: {gmap['all_useful_units_intact']}; "
        f"frozen genes carried: {int(frozen2.sum())}")
    if not pres["pass"] or not gmap["ok"] \
            or not gmap["all_useful_units_intact"]:
        results["verdict"] = ("gate-rejected: preservation "
                              "gate failed or gene-map "
                              "verification failed; arms not "
                              "run, gate not worked around")
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        write_html(results)
        log("DONE (gate-rejected)")
        print("COUPLED_WIDEN_D2_DONE")
        return

    pv_t = prob_vectors(classes2, cfg, frozen2)
    pv_c = prob_vectors(classes0, cfg, frozen0)

    # ---------------- ARM T: coupled widen ----------------
    rng_t = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_t = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_t = be.create_species(
        template2, classes2, cfg.n_species, cfg.per_species,
        draw_rng_t, rng_t, pv_t["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_t = evolve(domain, layout2, cfg, classes2, species_t,
                   rng_t, pv_t)
    wall_t = time.time() - t0
    after_t = density_read(domain, layout2,
                           out_t["champion_genome"])
    results["treatment"] = arm_block(out_t, after_t,
                                     repro_val, wall_t)
    np.savez(HERE / "coupled_widen_d2_2026-11-23_"
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

    # ---------------- ARM C: control ----------------
    rng_c = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_c = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_c = be.create_species(
        subject, classes0, cfg.n_species, cfg.per_species,
        draw_rng_c, rng_c, pv_c["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c = evolve(domain, layout0, cfg, classes0, species_c,
                   rng_c, pv_c)
    wall_c = time.time() - t0
    after_c = density_read(domain, layout0,
                           out_c["champion_genome"])
    results["control"] = arm_block(out_c, after_c,
                                   repro_val, wall_c)
    np.savez(HERE / "coupled_widen_d2_2026-11-23_"
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
    print("COUPLED_WIDEN_D2_DONE")


# ---------------------------------------------------------------
# HTML summary
# ---------------------------------------------------------------
def _gap(score):
    return -100.0 * score


def _bar(pct, colour):
    w = max(0.0, min(100.0, pct))
    return (f"<div style='background:#eee;border:1px solid #bbb;"
            f"width:220px;height:12px;display:inline-block'>"
            f"<div style='background:{colour};width:{w:.1f}%;"
            f"height:100%'></div></div>")


def _cat_rows(label, counts, total):
    rows = ""
    for cat, colour in (("contributing", "#c0392b"),
                        ("neutral", "#7f8c8d"),
                        ("harmful", "#27ae60")):
        n = counts.get(cat, 0)
        rows += (f"<tr><td>{label}</td><td>{cat}</td>"
                 f"<td>{n} / {total}</td>"
                 f"<td>{_bar(100.0 * n / total, colour)}</td>"
                 f"<td>{100.0 * n / total:.0f}%</td></tr>")
    return rows


def write_html(results):
    sub = results["subject"]
    tr = results.get("transplant", {})
    gate = tr.get("preservation_gate", {})
    gm = tr.get("gene_map_verification", {})
    before = results.get("before", {})
    arms = []
    for key, label in (("treatment", "T: coupled widen"),
                       ("control", "C: control")):
        a = results.get(key)
        if a is None:
            continue
        arms.append(
            f"<tr><td>{label}</td>"
            f"<td>{_gap(a['final_champion_val_score']):.2f}%</td>"
            f"<td>{a['delta_vs_start']:+.4f}</td>"
            f"<td>{a['density_exact_zero_positive']}/"
            f"{a['density_band_counts'] and sum(a['density_band_counts'].values())}</td>"
            f"<td>{a['density_band_counts']}</td>"
            f"<td>sp{a['champion_species_id']}, "
            f"{a['n_species_final']} species</td></tr>")
    m = results.get("margin_treatment_over_control", {})
    verdict = results.get("verdict", "arms completed")
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Coupled widen, d2 lineage</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}
.small{{font-size:12px;color:#444;max-width:920px}}
table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:3px 8px}}</style></head><body>
<h1>Coupled widen: d 2 -&gt; 6, FFN 8 -&gt; 24, single-layer lineage</h1>
<p class="small">Subject: the prune-graft treatment champion,
{_gap(sub['stored_val_score']):.2f}% mean gap, {sub['n_genes']} genes,
9 units. Trajectory of the lineage: seed 22.03% -&gt; prune-graft
18.80% -&gt; this test. The transplant widens the stream and the
feed-forward together at BERT's 4x ratio, with 16 fresh FFN units
born silent (random incoming, zero outgoing).</p>
<h2>Gate and gene map</h2>
<table><tr><th>check</th><th>result</th></tr>
<tr><td>preservation gate</td><td>
{'PASS' if gate.get('pass') else 'REJECT'}; max abs length
difference {gate.get('max_abs_length_difference')}; tours identical
{gate.get('tours_identical')}; {gate.get('instances')} problems</td></tr>
<tr><td>gene map</td><td>{gm.get('n_surviving_genes')}/
{gm.get('n_old_genes')} old genes survive; {gm.get('n_exact_genes')}
exact; mismatches {len(gm.get('mismatches', []))}, all the LN-gain
rescale (sqrt(d0/d1) = {gm.get('ln_gain_rescale_factor')});
useful units intact: {gm.get('all_useful_units_intact')}</td></tr>
<tr><td>size</td><td>{tr.get('n_genes_old')} -&gt; {tr.get('n_genes_new')}
genes; {tr.get('n_units_old')} -&gt; {tr.get('n_units_new')} units;
{tr.get('n_fresh_ffn_units')} fresh FFN units;
{tr.get('n_frozen_genes_carried')} frozen genes carried</td></tr>
</table>
<h2>Arms (100 generations each, 50 + 50)</h2>
<table><tr><th>arm</th><th>final gap</th><th>delta vs start</th>
<th>exact-zero positive</th><th>band counts</th><th>champion</th></tr>
{''.join(arms) if arms else '<tr><td colspan="6">no arms run</td></tr>'}
</table>
<p class="small">Margin treatment over control:
{m.get('points', 'n/a')} points,
{m.get('relative_percent_of_control_gap', 'n/a')}% relative to the
control's gap. Verdict: {verdict}</p>
<h2>Density, before and after</h2>
<table><tr><th>champion</th><th>category</th><th>count</th><th></th>
<th>share</th></tr>
{_cat_rows('subject', before.get('band_counts', {}), before.get('n_units', 1))}
</table>
<p class="small">Density after each arm is in
coupled_widen_d2_2026-11-23.json. Method: silencing table on the
fixed 200-problem validation set, categories by the paired-noise
band (reference: make_usefulness_map.py). The transplant and the
gate were verified before any evolution ran; if the gate had
rejected, the arms would not have run.</p>
</body></html>"""
    OUT_HTML.write_text(body)


if __name__ == "__main__":
    main()
