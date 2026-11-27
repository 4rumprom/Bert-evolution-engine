#!/usr/bin/env python3
"""REINFORCE probe on the Two-Franks stitch (sponsor
direction, 2026-11-26: the soft-surrogate route to
back propagation is CLOSED for this network; the one
unprobed gradient route is REINFORCE, which the
sponsor described himself in 10 steps and confirmed
is the standard method).

WHY THIS EXISTS. Two back-propagation runs on this
subject failed in two different ways. Run 1's setup
was wrong (Adam lr 0.01 drove the network into a
saturated configuration, true gap 7.3066% ->
85.0839%, stuck). Run 2 fixed the setup (lr 1e-4,
tau 0.01, guard) and stalled cleanly: no read after
step 0 ever improved. The temperature sweep then
closed the soft-surrogate route at every setting
(frankenstein_twofranks_tau_sweep_2026-11-26.json:
tau 0.1 -> 0.0001, Pearson vs true length stays
~0.57; the decoder's logit gaps between cities are
microscopic, median 2.7e-7, so the soft distribution
never concentrates on the greedy tour, and a
temperature cold enough to concentrate would leave
no gradient). REINFORCE has no surrogate at all:
tours are sampled from the network's own pointer
distribution, and the gradient flows through the
log-probabilities of the actions actually taken,
weighted by an advantage signal. The reward
(-tour length) and the sampling need no gradient.

THE METHOD (the sponsor's 10 steps, as built):
  1. Present the problem to the network.
  2. The network produces a probability distribution
     over the next city (softmax over the decoder
     logits exactly as the decoder produces them;
     NO temperature scaling, NO soft surrogate).
  3. Sample an action from that distribution (an
     external stochastic operation; sampled actions
     are stop-gradient).
  4. Continue until a complete tour is sampled.
  5. Score the tour with the external function:
     reward = -tour length.
  6. Compare against the other sampled tours for
     the same problem: advantage = reward - mean
     reward of the samples for that problem.
  7. Loss = -mean over samples of advantage x
     (sum of log-probabilities of the sampled
     actions), averaged over a batch of fresh
     training problems. Gradients flow only
     through the log-probabilities.
  8. Only the 1,106 stitching genes move; the
     356 backbone genes are frozen (gradient
     masked to zero, exactly as in the backprop
     runs).
A greedy-tour baseline variant (advantage vs the
network's own greedy tour length for the same
problem) is ALSO recorded for comparison, but the
gradient uses the sample-mean baseline only.

MODES:
  --check : read-only verifications ONLY — subject
    sanity (true gap reproduces 7.3066%), JAX
    encoder mirror parity, ONE gradient computation
    with its norm reported. Applies NO weight
    update and writes NO champion.
  --probe : the bounded probe (needs the sponsor's
    yes; do NOT run without it):
      (a) gradient consistency — cosine similarity
          between REINFORCE gradients from three
          independent problem batches, plus norms;
      (b) single-step test — from the subject, ONE
          Adam step (lr 1e-4) along a batch's
          gradient, true greedy gap measured on
          the fixed validation set before and
          after; repeated from 3 independent
          batches; reports whether improvements
          repeat or are noise. Single steps only,
          never a training run. If no step
          improves, it reports that and stops.
    No champion is written by the probe either:
    a single Adam step is a measurement, not a
    champion.

SUBJECT (unchanged): the extended Two-Franks
champion (engine_runs/frankenstein_twofranks_
evolution/frankenstein_twofranks_extended_
champion.npz), true gap 7.3066% on the fixed
200-problem validation set (val_seed 20261116).
1,462 genes = 356 backbone + 1,106 stitching; the
mask comes from the stitch record
(frankenstein_twofranks_stitch_2026-11-26.json).

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_reinforce_probe_2026-11-26.py \
    --probe > engine_runs/frankenstein_twofranks_\
reinforce_probe/launcher.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


# ================================================================
# Constants
# ================================================================
CFG = HERE / "bert_engine_config_fresh50.json"
SUBJECT = (HERE / "engine_runs" / "frankenstein_twofranks_evolution"
           / "frankenstein_twofranks_extended_champion.npz")
STITCH_JSON = HERE / "frankenstein_twofranks_stitch_2026-11-26.json"

BATCH = 64                    # fresh training problems per batch
K_SAMPLES = 8                 # sampled tours per problem
N_CITIES = 10
TRAIN_SEED = 20261204         # fresh seed stream for this probe
LR_STEP = 1e-4                # single Adam step size (probe)
CLIP_NORM = 1.0               # global-norm gradient clip, as in
                              # backprop run 2
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
BLOCKS = ("b0", "b1", "b2")

# ================================================================
# Setup: domain (validation set), scale, subject, mask
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
log(f"attn_scale = {SCALE!r}, ln_eps = {EPS!r}, "
    f"n_cities = {N_CITIES}")
assert abs(SCALE - 1.0 / np.sqrt(2.0)) < 1e-15

ST = json.load(open(STITCH_JSON))
KEYS = list(ST["stitching_mask"]["gene_order"])
flat_mask = np.array(ST["stitching_mask"]["flat_mask_stitching"],
                     dtype=bool)     # True = stitching
STITCH_IDX = np.flatnonzero(flat_mask)
BACKBONE_IDX = np.flatnonzero(~flat_mask)
assert len(STITCH_IDX) == ST["stitching_mask"]["stitching"] == 1106
assert len(BACKBONE_IDX) == ST["stitching_mask"]["backbone"] == 356

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
log(f"subject loaded: {N_GENES} genes "
    f"({len(BACKBONE_IDX)} backbone + {len(STITCH_IDX)} "
    f"stitching); recorded gap {RECORDED_GAP_PCT:.4f}%")


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k]) for k in KEYS}


# ================================================================
# Numpy forward + greedy decode (verbatim from backprop
# run 2) — the TRUE score, reporting only
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
# JAX mirror: encoder (verbatim from backprop run 2)
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


# ================================================================
# REINFORCE: sample a tour from the network's OWN
# pointer distribution. No temperature, no surrogate.
# The sampled picks are integers (no gradient flows
# through them); gradients flow only through the
# log-probabilities of the actions actually taken.
# ================================================================
def sample_tour_jax(W, x, key):
    """One sampled tour for ONE problem.
    Returns (picks, sum_logp, tour_length)."""
    Y = encode_jax(W, x)
    n = x.shape[0]
    D = jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                 .sum(axis=-1))
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    picks = jnp.zeros(n, dtype=jnp.int32)
    sum_logp = 0.0
    last = W["head.v_start"]
    for t in range(n):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        logp = jax.nn.log_softmax(logits)
        key, sub = jax.random.split(key)
        pick = jax.random.categorical(sub, logits).astype(
            jnp.int32)   # picks array is int32; cast so the
        # scatter stays int32 under x64 (avoids a
        # FutureWarning that later JAX versions turn
        # into an error)
        sum_logp = sum_logp + logp[pick]
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return picks, sum_logp, length


def reinforce_loss(w_flat, Xb, keys):
    """REINFORCE loss over a batch of problems.
    Xb: (B, n, 2) fresh training problems.
    keys: (B, K, 2) PRNG keys, K sampled tours per
    problem. Returns (loss, sampled_lengths (B, K)).
    Per problem: advantage = reward - mean reward of
    that problem's K samples; loss contribution =
    -mean_k(advantage_k * sum_logp_k). The advantage
    is stop-gradient (it is built from sampled
    integer picks, which carry no gradient anyway;
    the stop is explicit)."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x, ks):
        def per_sample(key):
            _, sum_logp, length = sample_tour_jax(W, x, key)
            return sum_logp, length
        sum_logps, lengths = jax.vmap(per_sample)(ks)
        rewards = -lengths
        adv = jax.lax.stop_gradient(rewards - rewards.mean())
        return -jnp.mean(adv * sum_logps), lengths

    losses, all_lengths = jax.vmap(per_problem)(Xb, keys)
    return jnp.mean(losses), all_lengths


vg = jax.jit(jax.value_and_grad(reinforce_loss, has_aux=True))


def make_batch(batch_seed):
    """One batch of fresh training problems + sample
    keys, deterministically from a seed. Problems are
    uniform on the unit square (never the validation
    problems); keys derive from the same seed."""
    rng = np.random.default_rng(batch_seed)
    Xb = rng.random((BATCH, N_CITIES, 2))
    key = jax.random.PRNGKey(batch_seed)
    keys = jax.random.split(key, BATCH * K_SAMPLES).reshape(
        BATCH, K_SAMPLES, 2)
    return Xb, keys


def masked_grad_norm(grad):
    """Gradient norms: stitching genes only (the
    moving set) and all genes."""
    g_stitch = np.where(flat_mask, grad, 0.0)
    return (float(np.linalg.norm(g_stitch)),
            float(np.linalg.norm(grad)))


def compute_grad(w, batch_seed):
    """REINFORCE gradient for one batch. Returns
    (loss, grad [backbone entries zeroed], lengths
    (B, K), Xb)."""
    Xb, keys = make_batch(batch_seed)
    (loss, lengths), grad = vg(
        jnp.asarray(w, dtype=jnp.float64),
        jnp.asarray(Xb, dtype=jnp.float64), keys)
    grad = np.asarray(grad, dtype=float)
    grad = np.where(flat_mask, grad, 0.0)   # backbone frozen
    return float(loss), grad, np.asarray(lengths), Xb


def adam_one_step(w, grad, lr):
    """ONE Adam step from a zero state (t = 1), with
    global-norm gradient clipping, exactly the
    single-step rule the probe uses. Returns
    (w_new, clip_factor_applied)."""
    g = grad.copy()
    gn = float(np.linalg.norm(g))
    clip = 1.0
    if gn > CLIP_NORM:
        clip = CLIP_NORM / gn
        g = g * clip
    m = (1 - ADAM_B1) * g
    v = (1 - ADAM_B2) * g * g
    mhat = m / (1 - ADAM_B1)
    vhat = v / (1 - ADAM_B2)
    upd = mhat / (np.sqrt(vhat) + ADAM_EPS)
    return w - lr * upd, clip


def greedy_baseline_stats(w, Xb, lengths):
    """The greedy-tour baseline variant (recorded for
    comparison; never used for a gradient): per
    problem, the network's own greedy tour length on
    the same batch problems; advantage_greedy =
    reward - (-greedy length). Returns summary
    stats and the correlation between the two
    advantage vectors."""
    greedy_lens, _ = true_lengths(w, Xb)
    rewards = -lengths                       # (B, K)
    adv_mean = rewards - rewards.mean(axis=1, keepdims=True)
    adv_greedy = rewards - (-greedy_lens)[:, None]
    a = adv_mean.ravel()
    b = adv_greedy.ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 \
        and b.std() > 0 else float("nan")
    return {
        "mean_sampled_length": round(float(lengths.mean()), 4),
        "mean_greedy_length": round(float(greedy_lens.mean()), 4),
        "mean_advantage_sample_mean_baseline": round(
            float(adv_mean.mean()), 6),
        "std_advantage_sample_mean_baseline": round(
            float(adv_mean.std()), 4),
        "mean_advantage_greedy_baseline": round(
            float(adv_greedy.mean()), 4),
        "std_advantage_greedy_baseline": round(
            float(adv_greedy.std()), 4),
        "corr_sample_mean_vs_greedy_advantages": round(corr, 4),
    }


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ================================================================
# Shared read-only verifications (both modes)
# ================================================================
def verify():
    # (a) subject sanity: true gap reproduces 7.3066%
    subj_gap = mean_gap(w_subj, X_val, OPT_VAL)
    log(f"sanity (a): subject true validation gap "
        f"{subj_gap*100:.6f}% vs recorded "
        f"{RECORDED_GAP_PCT:.4f}%")
    if abs(subj_gap * 100 - RECORDED_GAP_PCT) > 5e-5 + 1e-9:
        log("SANITY (a) FAILED. STOPPING.")
        sys.exit(3)
    log("sanity (a) passed")
    # (b) JAX encoder mirror parity
    W_subj = unpack(w_subj)
    Y_np = forward_general(W_subj, X_val[:8])
    W_j = {k: jnp.asarray(W_subj[k], dtype=jnp.float64)
           for k in KEYS}
    Y_jx = np.asarray(jax.vmap(
        lambda x: encode_jax(W_j, x))(
        jnp.asarray(X_val[:8], dtype=jnp.float64)))
    parity = float(np.max(np.abs(Y_np - Y_jx)))
    log(f"mirror check: JAX encoder vs numpy forward, "
        f"max abs diff {parity:.2e} on 8 validation "
        f"problems")
    if parity > 1e-9:
        log("MIRROR CHECK FAILED. STOPPING.")
        sys.exit(3)
    log("mirror check passed")
    return subj_gap, parity


# ================================================================
# --check: read-only. One gradient computation, its
# norm reported. NO weight update, NO champion.
# ================================================================
def run_check():
    log("CHECK MODE: read-only verifications; no weight "
        "update, no champion written")
    subj_gap, parity = verify()
    loss, grad, lengths, Xb = compute_grad(w_subj, TRAIN_SEED)
    gn_stitch, gn_full = masked_grad_norm(grad)
    stats = greedy_baseline_stats(w_subj, Xb, lengths)
    log(f"check gradient: batch seed {TRAIN_SEED} "
        f"({BATCH} problems x {K_SAMPLES} samples); "
        f"REINFORCE loss {loss:.4f}; gradient norm "
        f"(stitching genes) {gn_stitch:.4f}, (all genes) "
        f"{gn_full:.4f}")
    log(f"check batch: mean sampled length "
        f"{stats['mean_sampled_length']:.4f} vs mean "
        f"greedy length {stats['mean_greedy_length']:.4f}; "
        f"sample-mean advantage std "
        f"{stats['std_advantage_sample_mean_baseline']:.4f}; "
        f"greedy-baseline advantage mean "
        f"{stats['mean_advantage_greedy_baseline']:.4f}; "
        f"advantage correlation "
        f"{stats['corr_sample_mean_vs_greedy_advantages']:.4f}")
    record = {
        "mode": "check",
        "date": "2026-11-26",
        "subject": {
            "file": "engine_runs/frankenstein_twofranks_"
                    "evolution/frankenstein_twofranks_"
                    "extended_champion.npz",
            "recorded_gap_pct": RECORDED_GAP_PCT,
            "recomputed_gap_pct": round(subj_gap * 100, 6),
            "genes_total": N_GENES,
            "genes_backbone": int(len(BACKBONE_IDX)),
            "genes_stitching": int(len(STITCH_IDX)),
        },
        "verifications": {
            "sanity_gap_reproduced": True,
            "jax_mirror_encoder_parity_max_abs_diff": parity,
        },
        "check_gradient": {
            "batch_seed": TRAIN_SEED,
            "batch_problems": BATCH,
            "samples_per_problem": K_SAMPLES,
            "reinforce_loss": round(loss, 4),
            "gradient_norm_stitching_genes": round(gn_stitch, 4),
            "gradient_norm_all_genes": round(gn_full, 4),
            "backbone_gradient_zeroed": True,
            "weight_update_applied": False,
            "champion_written": False,
        },
        "batch_diagnostics": stats,
        "wall_seconds": round(time.time() - T0, 1),
    }
    with open(HERE / "frankenstein_twofranks_reinforce_probe_"
                     "check_2026-11-26.json", "w") as f:
        json.dump(record, f, indent=1)
    log("saved frankenstein_twofranks_reinforce_probe_"
        "check_2026-11-26.json")
    log(f"CHECK PASSED: sanity {subj_gap*100:.4f}%, "
        f"gradient norm (stitching) {gn_stitch:.4f}; no "
        f"weights changed, no champion written")


# ================================================================
# --probe: bounded probe. (a) gradient consistency
# across 3 independent batches; (b) single-step test
# from 3 independent batches. Single steps only,
# never a training run. No champion written.
# ================================================================
def run_probe():
    log("PROBE MODE: bounded probe — gradient "
        "consistency + single-step test; single steps "
        "only, never a training run; no champion "
        "written")
    subj_gap, _parity = verify()
    gap_before = subj_gap * 100

    # ---- (a) gradient consistency: 3 independent batches
    log("probe (a): gradient consistency across 3 "
        "independent batches")
    grads = []
    grad_info = []
    for i in range(3):
        seed = TRAIN_SEED + 1 + i
        loss, grad, lengths, Xb = compute_grad(w_subj, seed)
        gn_stitch, gn_full = masked_grad_norm(grad)
        grads.append(grad)
        stats = greedy_baseline_stats(w_subj, Xb, lengths)
        grad_info.append({
            "batch": i + 1,
            "batch_seed": seed,
            "reinforce_loss": round(loss, 4),
            "gradient_norm_stitching_genes": round(gn_stitch, 4),
            "gradient_norm_all_genes": round(gn_full, 4),
            "batch_diagnostics": stats,
        })
        log(f"  batch {i + 1} (seed {seed}): loss "
            f"{loss:.4f}, gradient norm (stitching) "
            f"{gn_stitch:.4f}")
    cosines = {
        "batch1_vs_batch2": round(cosine(grads[0], grads[1]), 4),
        "batch1_vs_batch3": round(cosine(grads[0], grads[2]), 4),
        "batch2_vs_batch3": round(cosine(grads[1], grads[2]), 4),
    }
    log(f"probe (a): cosine similarities between the "
        f"three batch gradients: {cosines}")

    # ---- (b) single-step test: 3 independent batches
    log("probe (b): single-step test — one Adam step "
        f"(lr {LR_STEP:.0e}) from the subject, true "
        f"gap measured before and after, 3 "
        f"independent batches")
    step_tests = []
    n_improved = 0
    for i in range(3):
        seed = TRAIN_SEED + 11 + i
        loss, grad, lengths, Xb = compute_grad(w_subj, seed)
        w_after, clip = adam_one_step(w_subj, grad, LR_STEP)
        gap_after = mean_gap(w_after, X_val, OPT_VAL) * 100
        improved = bool(gap_after < gap_before)
        n_improved += int(improved)
        # backbone must be untouched by the step
        backbone_untouched = bool(
            (w_after[BACKBONE_IDX]
             == w_subj[BACKBONE_IDX]).all())
        n_stitch_changed = int((w_after[STITCH_IDX]
                                != w_subj[STITCH_IDX]).sum())
        step_tests.append({
            "batch": i + 1,
            "batch_seed": seed,
            "reinforce_loss": round(loss, 4),
            "clip_factor_applied": round(clip, 4),
            "gap_before_pct": round(gap_before, 4),
            "gap_after_pct": round(gap_after, 4),
            "change_pct_points": round(gap_after - gap_before, 4),
            "improved": improved,
            "backbone_untouched": backbone_untouched,
            "stitching_genes_changed": n_stitch_changed,
        })
        log(f"  batch {i + 1} (seed {seed}): gap "
            f"{gap_before:.4f}% -> {gap_after:.4f}% "
            f"({'IMPROVED' if improved else 'worse'}; "
            f"backbone untouched: {backbone_untouched})")
    verdict = ("improvements repeat" if n_improved == 3 else
               "mixed" if n_improved > 0 else
               "no step improved")
    log(f"probe (b): {n_improved} of 3 single steps "
        f"improved the true gap — {verdict}")

    OUT = HERE / "engine_runs" / "frankenstein_twofranks_reinforce_probe"
    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": "probe",
        "date": "2026-11-26",
        "subject": {
            "file": "engine_runs/frankenstein_twofranks_"
                    "evolution/frankenstein_twofranks_"
                    "extended_champion.npz",
            "recorded_gap_pct": RECORDED_GAP_PCT,
            "recomputed_gap_pct": round(gap_before, 4),
            "genes_total": N_GENES,
            "genes_backbone": int(len(BACKBONE_IDX)),
            "genes_stitching": int(len(STITCH_IDX)),
        },
        "method": {
            "name": "REINFORCE (policy gradient): sample "
                    "tours from the network's own "
                    "pointer distribution (softmax over "
                    "the decoder logits as produced; "
                    "no temperature, no surrogate); "
                    "advantage = reward - mean reward "
                    "of the problem's samples; loss = "
                    "-mean(advantage x sum log-prob "
                    "of sampled actions)",
            "samples_per_problem": K_SAMPLES,
            "batch_problems": BATCH,
            "baseline": "sample-mean reward per problem "
                        "(gradient); greedy-tour length "
                        "also recorded for comparison",
            "moving_genes": "stitching only "
                            f"({len(STITCH_IDX)}); "
                            "backbone frozen",
            "single_step": f"one Adam step from a zero "
                           f"state, lr {LR_STEP:.0e}, "
                           f"global-norm clip {CLIP_NORM}",
            "seed_stream": f"batches from numpy "
                           f"default_rng seed {TRAIN_SEED} "
                           f"onwards; JAX sample keys "
                           f"from PRNGKey(seed)",
        },
        "gradient_consistency": {
            "batches": grad_info,
            "cosine_similarities": cosines,
        },
        "single_step_tests": step_tests,
        "single_steps_improved": n_improved,
        "verdict": verdict,
        "comparisons_pct": {
            "subject": RECORDED_GAP_PCT,
            "frank1_parent": 6.8959,
            "frank2_parent": 7.017,
        },
        "champion_written": False,
        "wall_seconds": round(time.time() - T0, 1),
    }
    with open(OUT / "frankenstein_twofranks_reinforce_probe_"
                     "2026-11-26.json", "w") as f:
        json.dump(record, f, indent=1)
    log("saved engine_runs/frankenstein_twofranks_"
        "reinforce_probe/frankenstein_twofranks_"
        "reinforce_probe_2026-11-26.json")
    log(f"PROBE DONE: {verdict} ({n_improved}/3 steps "
        f"improved); no champion written")


# ================================================================
# Entry
# ================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="REINFORCE probe on the Two-Franks "
                    "stitch: --check verifies (read-only); "
                    "--probe runs the bounded probe "
                    "(needs the sponsor's yes).")
    ap.add_argument("--check", action="store_true",
                    help="read-only verifications only")
    ap.add_argument("--probe", action="store_true",
                    help="bounded probe: gradient "
                         "consistency + single-step test")
    args = ap.parse_args()
    if args.check and args.probe:
        print("choose exactly one of --check or --probe",
              file=sys.stderr)
        sys.exit(2)
    if args.check:
        run_check()
    elif args.probe:
        run_probe()
    else:
        ap.print_help()
        sys.exit(2)
