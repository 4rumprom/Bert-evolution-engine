#!/usr/bin/env python3
"""OPTIMIZATION RUN (2026-11-23): plain continued evolution
of the best champion currently held. NOT an experiment:
no pruning, no grafting, no widening, no control arm.
Continues prune_widen_chain_2026-11-23.py and
widen_test_2026-11-23.py: same domain/config
(bert_engine_config_test.json, fixed 200-problem TSP
validation set, numpy backend, CPU x64), same primitives
evolution loop (be.create_species / breed_species /
cull_species; Schedule A; species best-ever on train
scores; champion = validation argmax over species
bests), copied from the chain script with two changes:
the phase budgets are 50 + 50 = 100 generations, and the
loop snapshots the champion at the phase boundary
(generation 50, same selection rule) so the per-50 gain
can be read.

SUBJECT: the chain test's step-1 control champion
(prune_widen_chain_2026-11-23_step1_control_champion.npz,
stored val -0.08179468538032875 = 8.18% mean gap; d 12,
FFN 16, 2 blocks, 2 heads, 3,620 genes). Its silencing
table already exists in prune_widen_chain_2026-11-23.json
(step1.control.density_units: exact-zero 35/36, band
29 contributing / 7 neutral / 0 harmful); it is reused
for the freeze marks, not recomputed.

FREEZE RULE (sponsor, 2026-11-23), active throughout:
units marked useful (band-contributing on the subject's
table) have their genes at mutation rate 0 in both
phases (mask over be._base_classes(layout, 0) prob
vectors, the chain/widen continuation pattern).

SEEDS: the chain used step 1 = 20261600 / 20261601 and
step 2 = 20261700 / 20261701; this run advances the
family one more step: run 20261800, draw 20261801.

Approximations, plainly: evolution is the primitives
loop, not the runner (the runner cannot be reused whole
for one-off continuations); checkpoint / monitor /
cataclysm / archive tracker omitted (none applied in
the earlier arms either).
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
CHAIN_JSON = HERE / "prune_widen_chain_2026-11-23.json"
WIDEN_JSON = HERE / "widen_test_2026-11-23.json"
ROUNDS_JSON = HERE / "prune_graft_rounds_2026-11-23.json"
R2_CONTROL_PATH = HERE / ("prune_graft_rounds_2026-11-23_r2_"
                          "control_champion.npz")
SUBJECT_PATH = HERE / ("prune_widen_chain_2026-11-23_step1_"
                       "control_champion.npz")
OUT_JSON = HERE / "optimize_control_2026-11-23.json"
OUT_HTML = HERE / "optimize_control_2026-11-23.html"
GENS_PER_PHASE = 50
RUN_SEED, DRAW_SEED = 20261800, 20261801
SUBJECT_STORED_VAL = -0.08179468538032875


def log(msg):
    print(f"[optimize] {msg}", flush=True)


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


def prob_vectors(classes, cfg, frozen_mask):
    """Copied from the chain script: class prob vectors
    with the freeze rule on top (useful-marked genes at
    rate 0 in both phases)."""
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


def pick_champion(domain, layout, species):
    """The harness selection rule: validation argmax over
    species best-ever genomes."""
    val_scores = [domain.validation_score(layout,
                                          sp["best"]["genome"])
                  for sp in species]
    bi = int(np.argmax(val_scores))
    return {
        "genome": species[bi]["best"]["genome"],
        "val_score": float(val_scores[bi]),
        "species_id": int(species[bi]["id"]),
    }


# ---------------------------------------------------------------
# evolution: the chain script's primitives loop, phase
# budgets 50 + 50, with a champion snapshot at the phase
# A -> B boundary (generation 50)
# ---------------------------------------------------------------
def evolve(domain, layout, cfg, classes, species, rng,
           pv_by_phase, gens_per_phase=GENS_PER_PHASE):
    history = []
    total_gens = 0
    snapshot50 = None
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
        if phase == "A":
            snapshot50 = pick_champion(domain, layout, species)
            snapshot50["generation"] = total_gens
            log(f"generation {total_gens} snapshot champion: "
                f"{snapshot50['val_score']:.6f} "
                f"({-100.0 * snapshot50['val_score']:.2f}% gap)")
    final = pick_champion(domain, layout, species)
    return {
        "champion_genome": final["genome"],
        "champion_val_score": final["val_score"],
        "champion_species_id": final["species_id"],
        "snapshot50": snapshot50,
        "n_species_final": len(species),
        "history": history,
    }


def rel_gain(gap_before, gap_after):
    """Improvement in gap points and as a relative
    percentage of the starting gap (sponsor's relative
    reading, 2026-11-23)."""
    return {
        "points": float(gap_before - gap_after),
        "relative_percent":
            float(100.0 * (gap_before - gap_after) / gap_before),
    }


def main():
    t_start = time.time()
    cfg = be.EngineConfig.load(CFG_PATH)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)
    chain = json.loads(CHAIN_JSON.read_text())

    # ---- layout: rebuild the d=12 layout exactly as the
    # chain script built layout_a (transplant off the d=8
    # round-2 control champion) ----
    ck = np.load(RUN_DIR / "checkpoint.npz", allow_pickle=True)
    arch8 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout8 = be.Layout(arch8, tsp.head_spec(arch8.d))
    r2_genome = np.asarray(
        np.load(R2_CONTROL_PATH)["genome"], dtype=float)
    arch, layout, _template, _classes = be.make_transplant(
        "widen", arch8, layout8, r2_genome, tsp.head_spec,
        cfg, tsp.widen_head, variant="A")
    assert layout.n_params == 3620, layout.n_params

    # ---- subject ----
    subj_z = np.load(SUBJECT_PATH)
    subject = np.asarray(subj_z["genome"], dtype=float)
    stored_val = float(subj_z["val_score"])
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-12
    repro_val = domain.validation_score(layout, subject)
    assert abs(repro_val - stored_val) < 1e-9, \
        (repro_val, stored_val)
    gap0 = float(-100.0 * repro_val)
    log(f"subject val reproduced: {repro_val:.6f} "
        f"({gap0:.2f}% mean gap), {layout.n_params} genes, "
        f"arch {arch.signature()}")

    # ---- freeze marks from the subject's EXISTING table
    # (chain results, step1.control.density_units) ----
    subj_units = chain["step1"]["control"]["density_units"]
    assert chain["step1"]["control"][
        "density_exact_zero_positive"] == 35
    assert chain["step1"]["control"]["density_band_counts"] == \
        {"contributing": 29, "neutral": 7, "harmful": 0}
    useful = [u["name"] for u in subj_units
              if u["cat"] == "contributing"]
    assert len(useful) == 29
    frozen = frozen_mask_from_names(layout, useful)
    log(f"freeze marks from the chain table: {len(useful)} "
        f"useful units, {int(frozen.sum())} genes at rate 0")

    results = {
        "run": "plain optimization run (standalone, not an "
               "experiment: no pruning, grafting, widening "
               "or control arm)",
        "date": "2026-11-23",
        "continues": [str(CHAIN_JSON), str(WIDEN_JSON),
                      str(ROUNDS_JSON)],
        "settings": {
            "config_file": str(CFG_PATH),
            "phase_budgets": {"A": GENS_PER_PHASE,
                              "B": GENS_PER_PHASE},
            "total_generations": 2 * GENS_PER_PHASE,
            "seeds": {"run_seed": RUN_SEED,
                      "draw_seed": DRAW_SEED,
                      "val_seed": cfg.val_seed},
            "seed_rule": "chain seed family advanced one "
                         "+100 step past chain step 2 "
                         "(20261700/20261701) -> run "
                         "20261800, draw 20261801",
            "rates": {"mut_p": cfg.mut_p,
                      "old_rate_factor": cfg.old_rate_factor,
                      "phase_B_old_rate":
                          cfg.mut_p * cfg.old_rate_factor},
            "freeze_rule": "units band-contributing on "
                           "the subject's existing "
                           "silencing table (chain results) "
                           "are marked useful; their genes "
                           "mutate at rate 0 in both phases",
            "classes_rule": "be._base_classes(layout, 0) "
                            "(continuation classes, as in "
                            "the chain/widen control arms) "
                            "with freeze marks",
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
        "subject": {
            "file": str(SUBJECT_PATH),
            "description": "chain test step-1 control "
                           "champion (plain evolution of "
                           "the plain-widen winner)",
            "arch_signature": arch.signature(),
            "n_genes": int(layout.n_params),
            "stored_val_score": stored_val,
            "reproduced_val_score": float(repro_val),
            "mean_gap_percent": gap0,
            "density_exact_zero_positive": 35,
            "density_band_counts": {
                "contributing": 29, "neutral": 7,
                "harmful": 0},
            "table_source": "prune_widen_chain_2026-11-23"
                            ".json step1.control."
                            "density_units (measured on "
                            "this identical genome in the "
                            "chain test; reused, not "
                            "recomputed)",
            "marks": {
                "useful_units": useful,
                "n_useful_units": len(useful),
                "n_frozen_genes": int(frozen.sum())},
        },
    }

    # ---- the run ----
    classes = be._base_classes(layout, 0)
    pv = prob_vectors(classes, cfg, frozen)
    rng = np.random.default_rng(RUN_SEED)
    draw_rng = np.random.default_rng(DRAW_SEED)
    species = be.create_species(
        subject, classes, cfg.n_species, cfg.per_species,
        draw_rng, rng, pv["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out = evolve(domain, layout, cfg, classes, species, rng,
                 pv)
    wall = time.time() - t0

    snap = out["snapshot50"]
    gap50 = float(-100.0 * snap["val_score"])
    gap100 = float(-100.0 * out["champion_val_score"])
    np.savez(HERE / "optimize_control_2026-11-23_gen50_"
                    "champion.npz",
             genome=snap["genome"],
             val_score=snap["val_score"])
    np.savez(HERE / "optimize_control_2026-11-23_champion.npz",
             genome=out["champion_genome"],
             val_score=out["champion_val_score"])

    after = density_read(domain, layout,
                         out["champion_genome"])
    results["run_results"] = {
        "start": {"val_score": float(repro_val),
                  "mean_gap_percent": gap0},
        "generation_50": {
            "val_score": snap["val_score"],
            "mean_gap_percent": gap50,
            "champion_species_id": snap["species_id"]},
        "generation_100": {
            "val_score": out["champion_val_score"],
            "mean_gap_percent": gap100,
            "champion_species_id":
                out["champion_species_id"]},
        "gain_start_to_50": rel_gain(gap0, gap50),
        "gain_50_to_100": rel_gain(gap50, gap100),
        "gain_start_to_100": rel_gain(gap0, gap100),
        "n_species_final": out["n_species_final"],
        "wall_seconds": wall,
        "final_density_exact_zero_positive":
            after["exact_zero_positive"],
        "final_density_band_counts": after["band_counts"],
        "final_density_n_units": after["n_units"],
        "final_density_units": after["units"],
        "history": out["history"],
    }
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    write_html(results)
    g = results["run_results"]
    log(f"DONE in {results['wall_seconds']:.1f}s: start "
        f"{gap0:.2f}% -> gen50 {gap50:.2f}% "
        f"({g['gain_start_to_50']['relative_percent']:+.1f}% "
        f"relative) -> gen100 {gap100:.2f}% "
        f"({g['gain_50_to_100']['relative_percent']:+.1f}% "
        f"relative); total "
        f"{g['gain_start_to_100']['relative_percent']:+.1f}% "
        f"relative; final density exact-zero "
        f"{after['exact_zero_positive']}/{after['n_units']}, "
        f"band {after['band_counts']}")
    print("OPTIMIZE_CONTROL_DONE")


# ---------------------------------------------------------------
# summary page: lineage trajectory extended to this run,
# margins in relative terms (counts and bars only)
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
    chain = json.loads(CHAIN_JSON.read_text())
    ref = rounds["round_1_reference"]
    r2 = rounds["rounds"][0]
    arm_a = widen["arms"]["A_plain_widen"]
    step1c = chain["step1"]["control"]
    rr = results["run_results"]
    traj = [
        ("original mapped champion (minimal test, d 8)",
         ref["subject_score"]),
        ("prune-graft round 1 winner (d 8)",
         ref["treatment_final_score"]),
        ("round 2 control (d 8)", r2["control_final_score"]),
        ("plain-widen winner (d 12)",
         arm_a["final_champion_val_score"]),
        ("chain step-1 control (d 12)",
         step1c["final_champion_val_score"]),
        ("optimization run, generation 50 (d 12)",
         rr["generation_50"]["val_score"]),
        ("optimization run, generation 100 (d 12)",
         rr["generation_100"]["val_score"]),
    ]
    max_gap = max(-s for _, s in traj) or 1.0
    traj_rows = "".join(
        f"<tr><td>{htmlmod.escape(label)}</td>"
        f"<td>{score:+.4f}</td><td>{_gap(score)}</td>"
        f"<td>{_bar(100.0 * (-score) / max_gap, '#c0392b')}</td>"
        f"</tr>" for label, score in traj)
    gain_rows = ""
    for label, key in (
            ("start to generation 50", "gain_start_to_50"),
            ("generation 50 to 100", "gain_50_to_100"),
            ("start to generation 100", "gain_start_to_100")):
        gn = rr[key]
        gain_rows += (
            f"<tr><td>{label}</td>"
            f"<td>{gn['points']:+.2f} points</td>"
            f"<td><b>{gn['relative_percent']:+.1f}%</b> "
            f"relative</td></tr>")
    total = rr["final_density_n_units"]
    cat_rows = ""
    for cat, colour in (("contributing", "#c0392b"),
                        ("neutral", "#7f9db3"),
                        ("harmful", "#5da85d")):
        n = rr["final_density_band_counts"].get(cat, 0)
        cat_rows += (
            f"<tr><td>{cat}</td><td>{n}</td>"
            f"<td>{_bar(100.0 * n / total, colour)}</td></tr>")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Optimization run 2026-11-23</title>
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
<h1>Optimization run: 100 more generations of plain
evolution</h1>
<p class="small">Subject: the chain test's step-1 control
champion ({results['subject']['reproduced_val_score']:+.4f},
{_gap(results['subject']['reproduced_val_score'])} mean
gap, d 12, {results['subject']['n_genes']} genes), the
best champion of the lineage. This is an optimization
run, not an experiment: plain continued evolution, 100
generations (50 + 50), 12 species x 10, no pruning, no
grafting, no widening, no control arm. The freeze rule
is active: the subject's {results['subject']['marks']
['n_useful_units']} useful-marked units
({results['subject']['marks']['n_frozen_genes']} genes)
mutate at rate 0 throughout. All scores on the same
fixed 200-problem TSP validation set; gap is the mean
tour gap (lower is better). Gains are stated in points
and as a relative percentage of the gap they started
from, because the gap is converging toward 0.</p>
<h2>Lineage trajectory</h2>
<table><tr><th>stage</th><th>score</th><th>mean gap</th>
<th></th></tr>{traj_rows}</table>
<h2>This run's gains</h2>
<table><tr><th>span</th><th>points</th><th>relative</th>
</tr>{gain_rows}</table>
<h2>Final champion density</h2>
<p class="small">Exact-zero density:
{rr['final_density_exact_zero_positive']} of {total}
units with silencing drop &gt; 0. Band read
(paired-noise, 2 SE):</p>
<table><tr><th>category</th><th>units</th><th></th></tr>
{cat_rows}</table>
</body></html>"""
    OUT_HTML.write_text(page)


if __name__ == "__main__":
    main()
