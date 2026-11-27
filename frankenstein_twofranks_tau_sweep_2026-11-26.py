#!/usr/bin/env python3
"""TAU SWEEP — diagnostic probe (proactive, 2026-11-26):
at what temperature, if any, does the soft surrogate
actually track the true score on the two-Franks
stitch?

Background: two backprop runs on this subject both
failed to improve it. Run 1 (tau 0.1, Adam lr 0.01)
saturated the network in 100 steps. Run 2 (tau 0.01,
lr 1e-4, revert-to-best guard) never damaged the
network but never found an improving direction in
3,000 steps. Both runs measured the surrogate
tracking the true score poorly: tau 0.1 Pearson
0.5845, on-pick mass ~0.38; tau 0.01 Pearson 0.5686,
on-pick mass 0.4066. The question this probe
settles: does ANY tau make the surrogate track the
true score, or does it stay flat as tau drops?

Method: one fixed batch of 200 freshly sampled
training problems (n = 10 cities, uniform, one
recorded seed; NEVER the validation problems). Same
batch for every tau. For each tau: (a) Pearson
correlation between the soft expected tour length
and the true greedy tour length across the 200
problems; (b) mean probability mass the soft
distribution puts on the greedy pick; (c) mean soft
length minus mean true length. Plus a diagnostic
the runs did not measure: the scale of the decoder
logit gaps (top-1 minus top-2 logit at each
decoding step) — if the gaps are far smaller than
the smallest usable tau, no tau can both track the
true score and keep a usable gradient.

Machinery reused verbatim from
frankenstein_twofranks_backprop2_2026-11-26.py
(JAX mirror verified to 3.55e-15; soft surrogate =
anchored expected tour length, greedy picks held
constant), with tau passed as a parameter instead
of a module global.

SUBJECT: the extended Two-Franks champion
(engine_runs/frankenstein_twofranks_evolution/
frankenstein_twofranks_extended_champion.npz),
true gap 7.3066% on the fixed 200-problem
validation set (val_seed 20261116).

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_twofranks_tau_sweep_2026-11-26.py
Outputs: frankenstein_twofranks_tau_sweep_2026-11-26.json
         frankenstein_twofranks_tau_sweep_2026-11-26.log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

T0 = time.time()
LOGF = open(HERE / "frankenstein_twofranks_tau_sweep_2026-11-26.log",
            "w", buffering=1)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")


# ================================================================
# Constants
# ================================================================
CFG = HERE / "bert_engine_config_fresh50.json"
SUBJECT = (HERE / "engine_runs" / "frankenstein_twofranks_evolution"
           / "frankenstein_twofranks_extended_champion.npz")
STITCH_JSON = HERE / "frankenstein_twofranks_stitch_2026-11-26.json"
BATCH_SEED = 20261210
BATCH_N = 200
TAUS = [0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4]
BLOCKS = ("b0", "b1", "b2")

# ================================================================
# Setup: domain (validation set), scale, subject
# (verbatim from backprop run 2)
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps
N_CITIES = int(cfg.domain_config.get("n_cities", 10))
log(f"attn_scale = {SCALE!r}, ln_eps = {EPS!r}, "
    f"n_cities = {N_CITIES}")
assert abs(SCALE - 1.0 / np.sqrt(2.0)) < 1e-15

ST = json.load(open(STITCH_JSON))
KEYS = list(ST["stitching_mask"]["gene_order"])
zs = np.load(SUBJECT)
SUBJ_META = json.loads(str(zs["meta"]))
assert [k for k in zs.files if k != "meta"] == KEYS
SUBJ_W = {k: np.asarray(zs[k], dtype=float) for k in KEYS}
SHAPES = {k: SUBJ_W[k].shape for k in KEYS}
SLICES = {}
_off = 0
for k in KEYS:
    SLICES[k] = (slice(_off, _off + SUBJ_W[k].size), SHAPES[k])
    _off += SUBJ_W[k].size
N_GENES = _off
assert N_GENES == 1462, N_GENES
w_subj = np.concatenate([SUBJ_W[k].ravel() for k in KEYS])
RECORDED_GAP_PCT = float(SUBJ_META["best_validation_gap_pct"])
log(f"subject loaded: {N_GENES} genes; recorded gap "
    f"{RECORDED_GAP_PCT:.4f}%")


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k]) for k in KEYS}


# ================================================================
# Numpy forward + greedy decode (verbatim from run 2)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS:
        Hb = ln_np(A, W[f"{blk}.lnA_gain"], W[f"{blk}.lnA_bias"],
                   EPS)
        Q = Hb @ W[f"{blk}.W_q"] + W[f"{blk}.b_q"]
        K = Hb @ W[f"{blk}.W_k"] + W[f"{blk}.b_k"]
        V = Hb @ W[f"{blk}.W_v"] + W[f"{blk}.b_v"]
        S = np.einsum("btj,bsj->bts", Q, K) * SCALE
        S = S - S.max(axis=-1, keepdims=True)
        E = np.exp(S)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        A = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        Hf = ln_np(A, W[f"{blk}.lnF_gain"], W[f"{blk}.lnF_bias"],
                   EPS)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"]) \
            @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
    return A


def decode_general(W, Y):
    B, n, d = Y.shape
    gmean = Y.mean(axis=1)
    last = np.broadcast_to(W["head.v_start"], (B, d)).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    picks = np.zeros((B, n), dtype=int)
    for t in range(n):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * np.tanh(
            np.einsum("bi,bni->bn", q, Y) * SCALE)
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last = Y[arB, pick]
    return picks


def true_lengths(vec, X):
    W = unpack(vec)
    Y = forward_general(W, X)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return lens, picks


def mean_gap(vec, X, opt):
    lens, _ = true_lengths(vec, X)
    return float((lens / opt - 1.0).mean())


# ================================================================
# JAX mirror: encoder + anchored soft surrogate, tau as a
# parameter (structure verbatim from run 2)
# ================================================================
def _ln_j(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * gain + bias


def encode_jax(W, x):
    A = x @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS:
        Hb = _ln_j(A, W[f"{blk}.lnA_gain"], W[f"{blk}.lnA_bias"],
                   EPS)
        Q = Hb @ W[f"{blk}.W_q"] + W[f"{blk}.b_q"]
        K = Hb @ W[f"{blk}.W_k"] + W[f"{blk}.b_k"]
        V = Hb @ W[f"{blk}.W_v"] + W[f"{blk}.b_v"]
        scores = Q @ K.T * SCALE
        attn = jax.nn.softmax(scores, axis=-1)
        Ah = attn @ V
        A = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        Hf = _ln_j(A, W[f"{blk}.lnF_gain"], W[f"{blk}.lnF_bias"],
                   EPS)
        A = A + jnp.tanh(Hf @ W[f"{blk}.W_1"]
                         + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
    return A


def soft_len_jax(W, x, picks, tau):
    """Anchored soft tour length for ONE problem at
    temperature tau: greedy picks held constant; at
    each visited state the pointer distribution at
    temperature tau prices the expected next leg."""
    Y = encode_jax(W, x)
    n = x.shape[0]
    D = jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                 .sum(axis=-1))
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    first_p = None
    L = 0.0
    for t in range(n):
        last = W["head.v_start"] if t == 0 else Y[picks[t - 1]]
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        p = jax.nn.softmax(logits / tau)
        if t == 0:
            first_p = p
        else:
            L = L + p @ D[picks[t - 1]]
        visited = visited.at[picks[t]].set(1.0)
    L = L + first_p @ D[picks[n - 1]]
    return L


def onpick_mass_one(W, x, picks, tau):
    """Mean share of the pointer distribution's mass
    on the network's own greedy pick, averaged over
    the visited states of ONE problem, at tau."""
    Y = encode_jax(W, x)
    n = x.shape[0]
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    tot = 0.0
    for t in range(n):
        last = W["head.v_start"] if t == 0 else Y[picks[t - 1]]
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        p = jax.nn.softmax(logits / tau)
        tot = tot + p[picks[t]]
        visited = visited.at[picks[t]].set(1.0)
    return tot / n


def logit_gaps_one(W, x, picks):
    """Top-1 minus top-2 decoder logit at each visited
    state of ONE problem (unvisited cities only).
    The final step has only one unvisited city, so it
    carries no gap and is skipped. Diagnostic: the
    scale of the logit gaps decides what tau would be
    needed to concentrate the distribution on the
    greedy pick."""
    Y = encode_jax(W, x)
    n = x.shape[0]
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    gaps = []
    for t in range(n):
        last = W["head.v_start"] if t == 0 else Y[picks[t - 1]]
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        masked = jnp.where(visited > 0.5, -jnp.inf, logits)
        srt = jnp.sort(masked, descending=True)
        if t < n - 1:
            gaps.append(srt[0] - srt[1])
        visited = visited.at[picks[t]].set(1.0)
    return jnp.stack(gaps)


# ================================================================
# SANITY: the subject's true gap must reproduce 7.3066%
# ================================================================
subj_gap = mean_gap(w_subj, X_val, OPT_VAL)
log(f"sanity: subject true validation gap "
    f"{subj_gap*100:.6f}% vs recorded "
    f"{RECORDED_GAP_PCT:.4f}%")
if abs(subj_gap * 100 - RECORDED_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY FAILED. STOPPING.")
    sys.exit(3)
log("sanity passed")

# ================================================================
# MIRROR CHECK: JAX encoder vs numpy forward
# ================================================================
W_subj = unpack(w_subj)
Y_np = forward_general(W_subj, X_val[:8])
W_j = {k: jnp.asarray(W_subj[k], dtype=jnp.float64)
       for k in KEYS}
Y_jx = np.asarray(jax.vmap(
    lambda x: encode_jax(W_j, x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
parity = float(np.max(np.abs(Y_np - Y_jx)))
log(f"mirror check: JAX encoder vs numpy forward, max "
    f"abs diff {parity:.2e} on 8 validation problems")
if parity > 1e-9:
    log("MIRROR CHECK FAILED. STOPPING.")
    sys.exit(3)
log("mirror check passed")

# ================================================================
# THE BATCH: one fixed set of 200 fresh training
# problems, never the validation problems
# ================================================================
rng = np.random.default_rng(BATCH_SEED)
X_batch = rng.random((BATCH_N, N_CITIES, 2))
true_lens, picks_batch = true_lengths(w_subj, X_batch)
log(f"batch: {BATCH_N} fresh problems, seed "
    f"{BATCH_SEED}; mean true length "
    f"{true_lens.mean():.4f}")

# Logit-gap scale diagnostic (tau-independent)
gaps = np.asarray(jax.vmap(
    lambda x, pk: logit_gaps_one(W_j, x, pk))(
    jnp.asarray(X_batch, dtype=jnp.float64),
    jnp.asarray(picks_batch)))
gap_stats = {
    "mean": float(gaps.mean()),
    "median": float(np.median(gaps)),
    "p10": float(np.percentile(gaps, 10)),
    "p90": float(np.percentile(gaps, 90)),
    "min": float(gaps.min()),
    "max": float(gaps.max()),
    "fraction_exactly_zero": float((gaps == 0.0).mean()),
    "fraction_below_1e-3": float((gaps < 1e-3).mean()),
    "fraction_below_1e-2": float((gaps < 1e-2).mean()),
    "fraction_below_0.1": float((gaps < 0.1).mean()),
    "n_steps_measured": int(gaps.size),
}
log(f"logit gaps (top1-top2, per decoding step, "
    f"final step excluded): mean "
    f"{gap_stats['mean']:.5f}, median "
    f"{gap_stats['median']:.5f}, p10 "
    f"{gap_stats['p10']:.5f}, p90 "
    f"{gap_stats['p90']:.5f}; fraction exactly zero: "
    f"{gap_stats['fraction_exactly_zero']:.3f}, below "
    f"1e-3: {gap_stats['fraction_below_1e-3']:.3f}, below "
    f"1e-2: {gap_stats['fraction_below_1e-2']:.3f}, "
    f"below 0.1: {gap_stats['fraction_below_0.1']:.3f}")

# ================================================================
# THE SWEEP
# ================================================================
rows = []
Xb_j = jnp.asarray(X_batch, dtype=jnp.float64)
Pk_j = jnp.asarray(picks_batch)
for tau in TAUS:
    soft_lens = np.asarray(jax.vmap(
        lambda x, pk: soft_len_jax(W_j, x, pk, tau))(Xb_j, Pk_j))
    onpick = float(np.mean(np.asarray(jax.vmap(
        lambda x, pk: onpick_mass_one(W_j, x, pk, tau))(
        Xb_j, Pk_j))))
    pearson = float(pearsonr(soft_lens, true_lens)[0])
    row = {
        "tau": tau,
        "pearson_vs_true_length": round(pearson, 4),
        "onpick_mass": round(onpick, 4),
        "mean_soft_length": round(float(soft_lens.mean()), 4),
        "mean_true_length": round(float(true_lens.mean()), 4),
        "soft_minus_true_length": round(
            float(soft_lens.mean() - true_lens.mean()), 4),
    }
    rows.append(row)
    log(f"tau {tau:g}: Pearson {pearson:.4f}, on-pick "
        f"mass {onpick:.4f}, mean soft "
        f"{soft_lens.mean():.4f} vs true "
        f"{true_lens.mean():.4f} (soft-true "
        f"{soft_lens.mean() - true_lens.mean():+.4f})")

# ================================================================
# Record
# ================================================================
record = {
    "probe": "tau sweep: at what temperature, if any, "
             "does the soft surrogate track the true "
             "score on the two-Franks stitch?",
    "date": "2026-11-26",
    "subject": {
        "file": "engine_runs/frankenstein_twofranks_"
                "evolution/frankenstein_twofranks_"
                "extended_champion.npz",
        "recorded_gap_pct": RECORDED_GAP_PCT,
        "recomputed_gap_pct": round(subj_gap * 100, 6),
        "genes_total": N_GENES,
    },
    "sanity": {
        "passed": True,
        "jax_mirror_encoder_parity_max_abs_diff": parity,
    },
    "batch": {
        "n_problems": BATCH_N,
        "n_cities": N_CITIES,
        "seed": BATCH_SEED,
        "source": "fresh uniform training problems; "
                  "never the validation problems",
        "mean_true_length": round(float(true_lens.mean()), 4),
    },
    "logit_gap_scale": gap_stats,
    "prior_measurements": {
        "tau_0.1_validation_set": {
            "pearson": 0.5845, "onpick_mass": 0.38},
        "tau_0.01_validation_set": {
            "pearson": 0.5686, "onpick_mass": 0.4066},
        "note": "measured on the fixed validation set "
                "in backprop runs 1 and 2; this sweep "
                "uses the fresh 200-problem batch",
    },
    "sweep": rows,
    "wall_seconds": round(time.time() - T0, 1),
}
with open(HERE / "frankenstein_twofranks_tau_sweep_"
                 "2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
log(f"saved frankenstein_twofranks_tau_sweep_"
    f"2026-11-26.json in {record['wall_seconds']}s")
log("DONE sweep")
