#!/usr/bin/env python3
"""Gates-only sparsification test, BIG champion replication
(sponsor-approved, Roman, 2026-11-25 08:38 EST:
"Run gates on the bigger champion").

REPLICATION of gates_test_2026-11-25.py on a new
subject. The first test (fresh50_v2 champion, 9 units)
found a JOINT NEUTRAL SET: 7 of 9 units removed
together cost +0.5% of the champion's gap on the real
validation set (+10.0% held out) against +111.8% for
random sets of the same size, although silencing any
one of those units alone cost about a third of the
gap. This run asks whether that finding replicates
on a bigger champion.

SUBJECT: the 11.85% champion, lineage best of the new
single-layer lineage: the CONTROL champion of the
prune-and-graft round on the double-widened champion
(prune_graft_widened_2026-11-23). Shape d=4, FFN 16,
2 heads, 1 layer, 360 genes, 18 silence units. The
genome is IDENTIFIED BY MEASUREMENT, not by filename
trust (there was a filename-overwrite incident on
2026-11-23): every 360-gene lineage candidate is
scored on the real validation set, and only a genome
reproducing the 11.85% gap is used. If none does,
the run stops and reports the candidates and gaps.

PROTOCOL: identical to the first gates test. Freeze
every weight; train ONLY the 18 per-unit gates
(sigmoid, in (0,1)) against the anchored soft
surrogate of bert_attribution.py (the champion's own
greedy picks held constant), loss = mean soft gap +
lambda * sum(gates), lambda swept on the same log
grid with continuation warm starts. At each lambda,
units with gate < 0.5 are silenced for real with the
engine's own _silence_indices and the pruned genome
is scored on the REAL 200-problem validation set and
the 64-problem elimination set (held out).

CONTROLS: (a) single-unit ablation drops for all 18
units on the real validation set, Spearman agreement
with gate values and with closing order; (b) random
control: 20 random subsets of the same size as each
gate-chosen set, silenced and scored for real.

DENSITY before and after by the cloud rule (drops as
relative % of the champion's gap; cloud half-width =
2 * 1.4826 * MAD of the rel drops), with the paired-
noise band read alongside as the measurement-
confidence read. All costs as relative % of the
champion's gap.

HARD RULES honoured: no evolved weight is trained;
only the 18 gate numbers move, on a copy. The
champion file is read, never written. New dated
outputs only: gates_test_big_2026-11-25.json.

Run under the project venv (jax):
  ~/workspace/bert-engine/.venv/bin/python \
      gates_test_big_2026-11-25.py
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_attribution as ba
import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend
from tsp_genome2 import held_karp

import jax
import jax.numpy as jnp


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cw = _load("cw", "coupled_widen_d2_2026-11-23.py")
dw = _load("dw", "double_widen_d2_2026-11-23.py")

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
PG_SUBJECT_PATH = HERE / "prune_graft_d2_l1_2026-11-23_" \
    "treatment_champion.npz"
OUT = HERE / "gates_test_big_2026-11-25.json"
TARGET_GAP = 0.1185            # the 11.85% champion
TARGET_TOL = 5e-4              # +/- 0.05 points
CANDIDATES = [
    "prune_graft_widened_2026-11-23_control_champion.npz",
    "prune_graft_widened_2026-11-23_treatment_champion.npz",
    "double_widen_d2_2026-11-23_treatment_champion.npz",
    "double_widen_d2_bandmarks_treatment_champion.npz",
    "double_widen_d2_bandmarks_control_champion.npz",
    "double_widen_d2_cloudmarks_treatment_champion.npz",
    "double_widen_d2_cloudmarks_control_champion.npz",
    "double_widen_d2_2026-11-23_control_champion.npz",
]
GATE_THRESHOLD = 0.5
LAMBDAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
ADAM_STEPS = 800
ADAM_LR = 0.05
THETA0 = 4.6                        # sigmoid = 0.990
N_RANDOM_DRAWS = 20
RANDOM_SEED = 20261125

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------
# Setup: config, domain, and the widened layout rebuilt
# exactly as prune_graft_widened_2026-11-23.py built it.
# Same val_seed / elim_seed as the first gates test, so the
# validation and elimination sets are the same problems.
# ----------------------------------------------------------------
cfg = be.EngineConfig.load(CFG_PATH)
cfg_w = dataclasses.replace(cfg, widen_step=2)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
domain = tsp.TSPDomain(cfg, backend)

ck = np.load(HERE / "engine_runs" / "fresh_d2_l1" /
             "checkpoint.npz", allow_pickle=True)
arch0 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
layout0 = be.Layout(arch0, tsp.head_spec(arch0.d))
pg_subject = np.asarray(
    np.load(PG_SUBJECT_PATH)["genome"], dtype=float)
_, _, _, _, arch, layout, _template, _classes = \
    dw.build_double_transplant(arch0, layout0, pg_subject,
                               cfg_w)
units = be.silence_units(layout)
unit_names = [u["name"] for u in units]
n_units = len(units)
log(f"layout rebuilt: {layout.n_params} genes, "
    f"{n_units} units, arch {arch.signature()}")
assert layout.n_params == 360 and n_units == 18, \
    (layout.n_params, n_units)

# ----------------------------------------------------------------
# SUBJECT IDENTIFICATION (the sanity gate that decides):
# score every lineage candidate on the REAL validation set;
# only a genome reproducing the 11.85% gap may be used.
# ----------------------------------------------------------------
scan = []
subject_file, genome, stored_val = None, None, None
for fname in CANDIDATES:
    p = HERE / fname
    if not p.exists():
        scan.append({"file": fname, "exists": False})
        continue
    z = np.load(p)
    g = np.asarray(z["genome"], dtype=float)
    entry = {"file": fname, "exists": True,
             "n_genes": int(g.shape[0]),
             "stored_val_score": (float(z["val_score"])
                                  if "val_score" in z.files
                                  else None)}
    if g.shape == (layout.n_params,):
        gap = float(domain.validation_detail(
            layout, g)["mean_gap"])
        entry["real_val_gap_pct"] = round(gap * 100, 4)
        entry["matches_target"] = bool(
            abs(gap - TARGET_GAP) <= TARGET_TOL)
        if entry["matches_target"] and genome is None:
            subject_file, genome = fname, g
            stored_val = entry["stored_val_score"]
    scan.append(entry)
    log(f"candidate {fname}: {entry}")

if genome is None:
    out = {"test": "gates-only sparsification, big champion",
           "date": "2026-11-25",
           "status": "STOPPED at subject identification: "
                     "no candidate reproduced the 11.85% gap",
           "target_gap_pct": TARGET_GAP * 100,
           "candidate_scan": scan}
    json.dump(out, open(OUT, "w"), indent=1)
    log("SANITY GATE FAILED: no candidate reproduces "
        "11.85%. STOP. Candidates and gaps are in the "
        "output JSON.")
    sys.exit(1)

log(f"subject identified: {subject_file} "
    f"(stored val_score {stored_val})")

# ----------------------------------------------------------------
# SANITY GATE 1 (real score): the untouched champion must
# reproduce its stored score exactly. Gates fully open ==
# the champion itself (a gate of 1 multiplies a unit's
# contribution by 1), so this IS the gates-open real score.
# ----------------------------------------------------------------
detail = domain.validation_detail(layout, genome)
real_gap = float(detail["mean_gap"])
log(f"sanity 1 (real): champion validation gap "
    f"{real_gap * 100:.4f}% vs stored score "
    f"{-stored_val * 100:.4f}%")
if abs(real_gap + stored_val) > 1e-9:
    log("SANITY GATE 1 FAILED: real gap != stored "
        "val_score. STOP.")
    sys.exit(1)

# elimination set gaps (held out from gate training)
optima_elim = np.array([held_karp(domain.X_elim[i])
                        for i in range(len(domain.X_elim))])


def elim_gap(g):
    lens, _ = domain.evaluate_lengths(
        layout, np.asarray(g, dtype=float)[None, :],
        domain.X_elim)
    return float((lens[0] / optima_elim - 1.0).mean())


champ_elim_gap = elim_gap(genome)
log(f"champion elimination-set gap (held out): "
    f"{champ_elim_gap * 100:.2f}%")

# ----------------------------------------------------------------
# SANITY GATE 2 (mirror): the jax mirror's encoder must match
# the engine's encoder, and the anchored surrogate at gates=1
# must track the true greedy lengths. Pick tau by tracking.
# ----------------------------------------------------------------
Y_ref = be.encode_population(backend, layout, genome[None, :],
                             domain.X_val[:16])
Y_jax = ba.encode_jax(layout, genome, domain.X_val[:16])
parity = float(np.max(np.abs(Y_ref[0] - Y_jax)))
log(f"sanity 2 (mirror): encoder parity max abs diff "
    f"{parity:.2e}")
if parity > 1e-9:
    log("SANITY GATE 2 FAILED: jax mirror != engine. STOP.")
    sys.exit(1)

lens_true, picks_all = domain.evaluate_lengths(
    layout, genome[None, :], domain.X_val)
greedy_lengths = lens_true[0]
tau_scan = {}
best_tau, best_r = None, -2.0
for tau in (1.0, 0.5, 0.25, 0.1):
    soft = ba.soft_lengths(domain, layout, genome, tau=tau)
    r = float(pearsonr(soft, greedy_lengths)[0])
    soft_gap = float((soft / domain.optima - 1.0).mean())
    tau_scan[str(tau)] = {"pearson": r, "soft_gap": soft_gap}
    log(f"  tau {tau}: pearson {r:.3f}, mean soft gap "
        f"{soft_gap * 100:.2f}% (real {real_gap * 100:.2f}%)")
    if r > best_r:
        best_tau, best_r = tau, r
TAU = best_tau
log(f"chosen tau {TAU} (best tracking, pearson {best_r:.3f})")

# ----------------------------------------------------------------
# CONTROL (a) base + DENSITY BEFORE: the lineage's own
# density read (silencing table + paired-noise band),
# classified by the cloud rule. The ablation drops from
# this read are the control-(a) numbers. (No runner vault
# silence table exists for this harness champion, so there
# is no vault cross-check; drops are computed here.)
# ----------------------------------------------------------------
before = dw.cloud_classify(cw.density_read(domain, layout,
                                           genome))
ablation_drops = np.array([u["drop"] for u in before["units"]])
assert [u["name"] for u in before["units"]] == unit_names
band_before = float(np.mean(ablation_drops > 0.0))
log(f"density BEFORE: cloud counts "
    f"{before['cloud']['counts']} (limit "
    f"+/-{before['cloud']['limit_pct']:.1f}% of gap); "
    f"band counts {before['band_counts']}; "
    f"band read {band_before:.2f} of units with a real drop")

# ----------------------------------------------------------------
# GATE TRAINING. Only theta (18 numbers) moves. The genome
# is a constant inside soft_forward. Picks are the
# champion's own greedy picks (constants of the surrogate).
# ----------------------------------------------------------------
genome_j = jnp.asarray(genome, dtype=jnp.float64)
X_j = jnp.asarray(domain.X_val, dtype=jnp.float64)
opt_j = jnp.asarray(domain.optima, dtype=jnp.float64)
picks_j = jnp.asarray(picks_all[0])


def data_term(theta):
    gates = jax.nn.sigmoid(theta)
    L = jax.vmap(lambda x, pk: ba.soft_forward(
        layout, genome_j, x, gates, pk, TAU))(X_j, picks_j)
    return jnp.mean(L / opt_j - 1.0)


def loss_fn(theta, lam):
    gates = jax.nn.sigmoid(theta)
    return data_term(theta) + lam * jnp.sum(gates)


vg = jax.jit(lambda th, lam: jax.value_and_grad(loss_fn)(th, lam))


def adam(theta0, lam, steps=ADAM_STEPS, lr=ADAM_LR):
    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss = None
    for t in range(1, steps + 1):
        loss, grad = vg(theta, lam)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        theta = theta - lr * mh / (jnp.sqrt(vh) + eps)
    return theta, float(loss)


theta = jnp.full((n_units,), THETA0, dtype=jnp.float64)
levels = []
for lam in LAMBDAS:
    theta, final_loss = adam(theta, lam)
    gates = np.asarray(jax.nn.sigmoid(theta))
    dt = float(data_term(theta))
    closed = [unit_names[i] for i in range(n_units)
              if gates[i] < GATE_THRESHOLD]
    levels.append({"lambda": lam, "gates": gates,
                   "final_loss": final_loss,
                   "final_data_term_soft_gap": dt,
                   "closed": closed})
    log(f"lambda {lam:g}: gates "
        f"{[round(float(g), 3) for g in gates]} "
        f"closed {closed} (soft gap {dt * 100:.2f}%)")

# ----------------------------------------------------------------
# REAL VERIFICATION per level + RANDOM CONTROL + density after
# ----------------------------------------------------------------
def pruned_genome(closed_names):
    g = genome.copy()
    for u in units:
        if u["name"] in closed_names:
            g[be._silence_indices(layout, u)] = 0.0
    return g


rng = np.random.default_rng(RANDOM_SEED)
results_levels = []
for lev in levels:
    lam = lev["lambda"]
    closed = lev["closed"]
    k = len(closed)
    g_pruned = pruned_genome(set(closed))
    vg_real = float(domain.validation_detail(
        layout, g_pruned)["mean_gap"])
    eg_real = elim_gap(g_pruned)
    entry = {
        "lambda": lam,
        "gates": {unit_names[i]: round(float(lev["gates"][i]), 4)
                  for i in range(n_units)},
        "closed_units": closed,
        "n_closed": k,
        "surrogate_soft_gap_pct": round(
            lev["final_data_term_soft_gap"] * 100, 2),
        "real_val_gap_pct": round(vg_real * 100, 2),
        "real_val_cost_rel_pct": round(
            100.0 * (vg_real - real_gap) / real_gap, 2),
        "real_elim_gap_pct": round(eg_real * 100, 2),
        "real_elim_cost_rel_pct": round(
            100.0 * (eg_real - champ_elim_gap) / champ_elim_gap, 2),
    }
    # random control: same-size random subsets, real val score
    if 1 <= k <= n_units - 1:
        draws = []
        for _ in range(N_RANDOM_DRAWS):
            sub = set(rng.choice(unit_names, size=k,
                                 replace=False).tolist())
            gp = pruned_genome(sub)
            draws.append(float(domain.validation_detail(
                layout, gp)["mean_gap"]))
        draws = np.array(draws)
        entry["random_control"] = {
            "n_draws": N_RANDOM_DRAWS,
            "mean_gap_pct": round(float(draws.mean()) * 100, 2),
            "min_gap_pct": round(float(draws.min()) * 100, 2),
            "max_gap_pct": round(float(draws.max()) * 100, 2),
            "mean_cost_rel_pct": round(
                100.0 * (float(draws.mean()) - real_gap)
                / real_gap, 2),
            "best_cost_rel_pct": round(
                100.0 * (float(draws.min()) - real_gap)
                / real_gap, 2),
            "gate_set_cost_rel_pct": entry["real_val_cost_rel_pct"],
            "gate_set_beats_random_mean": bool(
                vg_real < float(draws.mean())),
            "gate_set_beats_random_min": bool(
                vg_real <= float(draws.min())),
        }
    # density after: the lineage density read + cloud rule
    # on the pruned net
    after = dw.cloud_classify(cw.density_read(domain, layout,
                                              g_pruned))
    remaining = [u for u in after["units"]
                 if u["name"] not in closed]
    entry["density_after"] = {
        "cloud_counts_all_units": after["cloud"]["counts"],
        "cloud_limit_pct": round(
            after["cloud"]["limit_pct"], 2),
        "band_counts_all_units": after["band_counts"],
        "remaining_units": len(remaining),
        "remaining_contributing_cloud": int(sum(
            1 for u in remaining
            if after["units"][unit_names.index(u["name"])]
            ["cloud_cat"] == "contributing")),
        "band_fraction_real_drop": round(float(np.mean(
            np.array([u["drop"] for u in after["units"]])
            > 0.0)), 3),
    }
    results_levels.append(entry)
    log(f"lambda {lam:g}: REAL val gap "
        f"{entry['real_val_gap_pct']:.2f}% (cost "
        f"{entry['real_val_cost_rel_pct']:+.1f}% of champion gap), "
        f"elim gap {entry['real_elim_gap_pct']:.2f}% (cost "
        f"{entry['real_elim_cost_rel_pct']:+.1f}%)"
        + (f", random mean cost "
           f"{entry['random_control']['mean_cost_rel_pct']:+.1f}%"
           if "random_control" in entry else ""))

# ----------------------------------------------------------------
# RANK AGREEMENT: gates vs ablation drops
# ----------------------------------------------------------------
rank_rows = []
for lev in levels:
    rs = float(spearmanr(lev["gates"], ablation_drops)[0])
    rank_rows.append({"lambda": lev["lambda"],
                      "spearman_gate_vs_ablation": round(rs, 3)})
closing_lambda = {}
for i, name in enumerate(unit_names):
    cl = None
    for lev in levels:
        if lev["gates"][i] < GATE_THRESHOLD:
            cl = lev["lambda"]
            break
    closing_lambda[name] = cl
cl_vec = np.array([closing_lambda[n] if closing_lambda[n] is not None
                   else 10.0 for n in unit_names])
rs_close = float(spearmanr(cl_vec, ablation_drops)[0])
log(f"rank agreement: closing-lambda vs ablation spearman "
    f"{rs_close:.3f}; per-level "
    f"{[(r['lambda'], r['spearman_gate_vs_ablation']) for r in rank_rows]}")

# ----------------------------------------------------------------
# REPLICATION READ: the largest gate-chosen set whose real
# validation cost is near zero (<= 5% of the champion's
# gap), with its held-out cost and random-control numbers.
# ----------------------------------------------------------------
joint = None
for e in results_levels:
    if 1 <= e["n_closed"] <= n_units - 1 \
            and e["real_val_cost_rel_pct"] <= 5.0:
        if joint is None or e["n_closed"] > joint["n_closed"]:
            joint = e
replication = {
    "criterion": "largest gate-chosen closed set with real "
                 "validation cost <= 5% of the champion gap",
    "found": joint is not None,
}
if joint is not None:
    replication.update({
        "lambda": joint["lambda"],
        "n_closed": joint["n_closed"],
        "closed_units": joint["closed_units"],
        "real_val_cost_rel_pct": joint["real_val_cost_rel_pct"],
        "real_elim_cost_rel_pct": joint["real_elim_cost_rel_pct"],
        "random_mean_cost_rel_pct":
            joint["random_control"]["mean_cost_rel_pct"],
        "random_best_cost_rel_pct":
            joint["random_control"]["best_cost_rel_pct"],
        "gate_set_beats_random_mean":
            joint["random_control"]["gate_set_beats_random_mean"],
        "gate_set_beats_random_min":
            joint["random_control"]["gate_set_beats_random_min"],
    })
min_single_rel = float(np.min(
    100.0 * ablation_drops / real_gap))
replication["smallest_single_unit_rel_drop_pct"] = round(
    min_single_rel, 2)
replication["every_single_unit_costly"] = bool(
    min_single_rel > 5.0)
log(f"replication read: {replication}")

out = {
    "test": "gates-only sparsification on a copy of the "
            "11.85% champion (big-champion replication of "
            "gates_test_2026-11-25)",
    "date": "2026-11-25",
    "sponsor_approval": "Roman, 2026-11-25 08:38 EST",
    "subject": {
        "file": subject_file,
        "lineage": "control champion of the prune-and-graft "
                   "round on the double-widened champion "
                   "(prune_graft_widened_2026-11-23)",
        "arch": "d=4, FFN 16, 2 heads, 1 layer, 360 genes",
        "arch_signature": arch.signature(),
        "stored_val_score": stored_val,
        "champion_val_gap_pct": round(real_gap * 100, 4),
        "champion_elim_gap_pct": round(champ_elim_gap * 100, 2),
        "n_units": n_units,
        "units": unit_names,
    },
    "candidate_scan": scan,
    "method": {
        "weights_trained": "none: every evolved weight frozen; "
                           "only 18 gate parameters trained, "
                           "on a copy",
        "surrogate": "bert_attribution.soft_forward (anchored, "
                     "champion's own greedy picks held constant)",
        "tau": TAU,
        "tau_scan": tau_scan,
        "loss": "mean soft gap + lambda * sum(sigmoid gates)",
        "optimizer": f"hand-rolled Adam, lr {ADAM_LR}, "
                     f"{ADAM_STEPS} steps per level, continuation "
                     "warm start across ascending lambda",
        "gate_threshold_for_real_pruning": GATE_THRESHOLD,
        "lambdas": LAMBDAS,
        "encoder_parity_max_abs_diff": parity,
    },
    "ablation_control": {
        "drops_by_unit": {
            unit_names[i]: round(float(ablation_drops[i]), 5)
            for i in range(n_units)},
        "rel_drops_pct_of_gap": {
            unit_names[i]: round(
                100.0 * float(ablation_drops[i]) / real_gap, 2)
            for i in range(n_units)},
        "note": "no runner vault silence table exists for "
                "this harness champion; drops computed "
                "in-script by the lineage density read",
    },
    "density_before": {
        "cloud_counts": before["cloud"]["counts"],
        "cloud_limit_pct": round(
            before["cloud"]["limit_pct"], 2),
        "band_counts": before["band_counts"],
        "band_fraction_units_with_real_drop": band_before,
    },
    "levels": results_levels,
    "rank_agreement": {
        "per_level_spearman_gate_vs_ablation": rank_rows,
        "closing_lambda_by_unit": closing_lambda,
        "spearman_closing_lambda_vs_ablation": round(rs_close, 3),
    },
    "replication_read": replication,
    "wall_seconds": round(time.time() - t_start, 1),
}
json.dump(out, open(OUT, "w"), indent=1)
log(f"saved {OUT} in {out['wall_seconds']}s")
print("GATES_TEST_BIG_DONE")
