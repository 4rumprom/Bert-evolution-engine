#!/usr/bin/env python3
"""Gates-only sparsification test (sponsor-approved,
Roman, 2026-11-25 00:08 EST: "Yes, run the gates test").

THE IDEA (Roman's, 2026-11-24 23:57 EST): use back
propagation to push certain neurons to zero while
keeping the same value, then take them out. The
research report (~/workspace/research_notes/
network-importance-and-merging-research-20261125-0456/
report.md, section 1h) found this is a real, named
family: Optimal Brain Damage / Surgeon, Molchanov
Taylor pruning (importance DEFINED as the loss change
on removal), L0 hard-concrete gates (Louizos et al.),
Network Slimming, Movement Pruning. The forcing cost
is an accepted importance measure. The cleanest
variant for an evolved network is GATES-ONLY: freeze
every evolved weight, train only per-unit on/off
gates on a copy.

WHAT THIS SCRIPT DOES. Subject: the fresh50_v2
champion (d=2, FFN 8, 1 head, 1 layer, 88 genes,
validation gap 15.13% on the fixed 200-problem set).
One gate per silence unit (9 units), gates in (0,1)
via sigmoid, multiplying the unit's contribution in
the anchored soft surrogate of bert_attribution.py
(the picks are the champion's own greedy picks, held
constant). Loss = mean soft gap over the validation
set + lambda * sum(gates). Lambda is swept over a log
grid with continuation (each level warm-starts from
the previous level's gates) to trace the frontier:
which gates close, at what penalty, at what cost.

HARD RULES honoured here:
  - NO weight of the evolved network is trained.
    Only the 9 gate parameters move. The champion
    genome is read, never written.
  - Training happens against the SOFT SURROGATE.
    Every claim is then verified on the REAL score:
    at each lambda, the units whose gate fell below
    0.5 are silenced for real with the engine's own
    _silence_indices, and the pruned genome is scored
    by the engine's greedy decode on the fixed
    200-problem validation set AND on the 64-problem
    elimination set (held out from gate training).
  - Controls: (a) gate ranking vs plain single-unit
    ablation (silencing) drops on the same champion,
    Spearman agreement; (b) random control: random
    subsets of the same size as each gate-chosen set,
    silenced and scored for real; the gate set must
    cost clearly less than random sets of its size.
  - Density (cloud rule + band read) before and
    after; all margins as relative % of the
    champion's gap.

Run under the project venv (jax):
  ~/workspace/bert-engine/.venv/bin/python \
      gates_test_2026-11-25.py
Outputs gates_test_2026-11-25.json next to this file.
"""
from __future__ import annotations

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

spec = importlib.util.spec_from_file_location(
    "dw", HERE / "double_widen_d2_2026-11-23.py")
dw = importlib.util.module_from_spec(spec)
sys.modules["dw"] = dw
spec.loader.exec_module(dw)

CFG = HERE / "bert_engine_config_fresh50.json"
RUN = HERE / "engine_runs" / "fresh50_v2"
OUT = HERE / "gates_test_2026-11-25.json"
EXPECTED_GAP = 0.1513148917731675   # results.json champion
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
# Setup: domain, layout, champion (read-only)
# ----------------------------------------------------------------
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
domain = tsp.TSPDomain(cfg, backend)

z = np.load(RUN / "champion.npz")
genome = np.asarray(z["genome"], dtype=float)
assert genome.shape == (layout.n_params,), genome.shape
units = be.silence_units(layout)
unit_names = [u["name"] for u in units]
n_units = len(units)
log(f"subject: fresh50_v2 champion, {layout.n_params} genes, "
    f"{n_units} units: {unit_names}")

# ----------------------------------------------------------------
# SANITY GATE 1 (real score): the untouched champion through
# the engine's real greedy path must reproduce 15.13%.
# Gates fully open == the champion itself: a gate of 1
# multiplies a unit's contribution by 1 (identity), and the
# "pruned" genome with no closed units is the champion
# verbatim, so this IS the gates-open real score.
# ----------------------------------------------------------------
detail = domain.validation_detail(layout, genome)
real_gap = float(detail["mean_gap"])
log(f"sanity 1 (real): champion validation gap "
    f"{real_gap * 100:.4f}% vs expected "
    f"{EXPECTED_GAP * 100:.4f}%")
if abs(real_gap - EXPECTED_GAP) > 1e-9:
    log("SANITY GATE 1 FAILED: real gap mismatch. STOP.")
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
# CONTROL (a) base: plain single-unit ablation on the real
# validation set (engine silencing), recomputed in-script and
# checked against the runner's vault table.
# ----------------------------------------------------------------
table = be.silence_table(
    layout, genome,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
ablation_drops = np.array([u["drop"] for u in table["units"]])
vault = json.load(open(
    RUN / "vault" / "stage_champion_stage0_sp8_79231594_silence.json"))
vault_drops = np.array([u["drop"] for u in vault["units"]])
vault_diff = float(np.max(np.abs(ablation_drops - vault_drops)))
log(f"ablation recomputed; max diff vs vault table "
    f"{vault_diff:.2e}")
assert [u["name"] for u in table["units"]] == unit_names

cloud_before = dw.cloud_classify({
    "champion_score": float(table["champion_score"]),
    "units": [{"name": u["name"], "drop": float(u["drop"])}
              for u in table["units"]]})
band_before = float(np.mean(ablation_drops > 0.0))
log(f"density BEFORE: cloud counts "
    f"{cloud_before['cloud']['counts']} (limit "
    f"+/-{cloud_before['cloud']['limit_pct']:.1f}% of gap); "
    f"band read {band_before:.2f} of units with a real drop")

# ----------------------------------------------------------------
# GATE TRAINING. Only theta (9 numbers) moves. The genome is a
# constant inside soft_forward. Picks are the champion's own
# greedy picks (constants of the anchored surrogate).
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
            "gate_set_cost_rel_pct": entry["real_val_cost_rel_pct"],
            "gate_set_beats_random_mean": bool(
                vg_real < float(draws.mean())),
            "gate_set_beats_random_min": bool(
                vg_real <= float(draws.min())),
        }
    # density after: silencing table + cloud on the pruned net
    tab_p = be.silence_table(
        layout, g_pruned,
        lambda G: domain.validation_scores(layout, G),
        zero_tol=domain.silence_zero_tol)
    cloud_after = dw.cloud_classify({
        "champion_score": float(tab_p["champion_score"]),
        "units": [{"name": u["name"], "drop": float(u["drop"])}
                  for u in tab_p["units"]]})
    remaining = [u for u in tab_p["units"]
                 if u["name"] not in closed]
    entry["density_after"] = {
        "cloud_counts_all_units": cloud_after["cloud"]["counts"],
        "cloud_limit_pct": round(
            cloud_after["cloud"]["limit_pct"], 2),
        "remaining_units": len(remaining),
        "remaining_contributing": int(sum(
            1 for u in remaining
            if cloud_after["units"][unit_names.index(u["name"])]
            ["cloud_cat"] == "contributing")),
        "band_fraction_real_drop": round(float(np.mean(
            np.array([u["drop"] for u in tab_p["units"]]) > 0.0)), 3),
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

out = {
    "test": "gates-only sparsification on a copy of the "
            "fresh50_v2 champion",
    "date": "2026-11-25",
    "sponsor_approval": "Roman, 2026-11-25 00:08 EST",
    "subject": {
        "run": "engine_runs/fresh50_v2",
        "arch": "d=2, FFN 8, 1 head, 1 layer, 88 genes",
        "champion_val_gap_pct": round(real_gap * 100, 4),
        "champion_elim_gap_pct": round(champ_elim_gap * 100, 2),
        "n_units": n_units,
        "units": unit_names,
    },
    "method": {
        "weights_trained": "none: every evolved weight frozen; "
                           "only 9 gate parameters trained, on a copy",
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
        "max_diff_vs_runner_vault_table": vault_diff,
    },
    "density_before": {
        "cloud_counts": cloud_before["cloud"]["counts"],
        "cloud_limit_pct": round(
            cloud_before["cloud"]["limit_pct"], 2),
        "band_fraction_units_with_real_drop": band_before,
    },
    "levels": results_levels,
    "rank_agreement": {
        "per_level_spearman_gate_vs_ablation": rank_rows,
        "closing_lambda_by_unit": closing_lambda,
        "spearman_closing_lambda_vs_ablation": round(rs_close, 3),
    },
    "wall_seconds": round(time.time() - t_start, 1),
}
json.dump(out, open(OUT, "w"), indent=1)
log(f"saved {OUT} in {out['wall_seconds']}s")
