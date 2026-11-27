#!/usr/bin/env python3
"""BACKPROPAGATION on the Two-Franks stitch, CORRECTED
setup, under the sponsor's full schedule (Roman,
2026-11-26 09:07 EST: "Your full schedule: seams
first, backbone in once below 6.90%"; correction
directed 2026-11-26 09:16 EST after the first run's
setup failed).

This is run 2 of back propagation on this subject.
Run 1 (frankenstein_twofranks_backprop_2026-11-26.py)
verified its machinery three ways (JAX mirror matches
numpy to 3.55e-15; finite-difference gradient check
passed; the tau->0 limit of the surrogate reproduces
the true greedy length exactly) but its SETUP was
wrong, in the sponsor's words, not the schedule:
Adam lr 0.01 moved all 1,106 stitching genes at once
into a saturated configuration within 100 steps
(true gap 7.3066% -> 85.0839%, stuck for 1,500
steps); tau 0.1 made the surrogate price a tour with
~62% of its mass off the greedy pick (Pearson
0.5845); and there was no guard, so the run never
reverted to good weights. This run REUSES run 1's
machinery verbatim and changes only the setup:

CORRECTED SETUP:
  - tau = 0.01 (far colder: the surrogate prices
    close to the tour the network actually takes).
  - Adam learning rate 1e-4 (was 0.01).
  - Gradient clipping: global norm 1.0.
  - THE GUARD: true-gap reads on the fixed
    validation set every 25 steps (was 100). The
    best-on-validation weights are the WORKING
    point, not just a vault copy: whenever a read
    is worse than the best read so far, the working
    weights revert to the best-so-far weights, the
    Adam state resets, and the learning rate halves
    (floor 1e-6). Whenever a read improves,
    training continues from the new best. Every
    revert and halving is logged. The guard stays
    active in all phases.

SUBJECT (unchanged): the extended Two-Franks
champion (engine_runs/frankenstein_twofranks_
evolution/frankenstein_twofranks_extended_
champion.npz), true gap 7.3066% on the fixed
200-problem validation set (val_seed 20261116).
1,462 genes = 356 backbone + 1,106 stitching; the
mask comes from the stitch record
(frankenstein_twofranks_stitch_2026-11-26.json).

NETWORK / OBJECTIVE (unchanged from run 1): the
Two-Franks network is an explicit weight set
(stream width 8, blocks b0/b1/b2 with fused
attention, one merged decoder), NOT an engine
ArchSpec shape. The objective is the anchored
expected tour length of the gates work: the
network's own greedy picks are held constant, and
at each visited state the pointer distribution at
temperature tau prices the expected next leg. Loss
= mean soft length over a batch of 64 freshly
sampled training problems per step (n = 10 cities,
uniform, recorded seed stream; NEVER the
validation problems).

THE SCHEDULE (the sponsor's design, unchanged):
  PHASE 1: only the 1,106 stitching genes train;
  the 356 backbone genes are frozen (multiplier 0
  and gradient masked to zero, so frozen genes are
  bit-identical, exactly as in run 1). Run until
  the TRUE gap first reads below 6.8959% (Frank
  #1's score, the better parent) — the trigger.
  After the trigger fires, run 200 more steps of
  phase 1, then phase 2. If the trigger has not
  fired after 3,000 steps (was 1,500), STOP the
  whole run and report that phase 2 never started.
  PHASE 2: the backbone phases in progressively
  but steeply: its learning-rate multiplier ramps
  linearly from 0 to 1.0 over 200 steps
  (multiplier on ramp step j, 0-indexed, is
  (j+1)/200); the stitching genes stay at full
  rate throughout.
  PHASE 3: after the ramp, 500 more steps at full
  rate, then STOP.
  Hard cap: 3,900 steps total.

BEFORE TRAINING: (a) sanity — the subject's true
greedy gap must reproduce 7.3066%; (b) the JAX
mirror's encoder must match the numpy forward;
(c) the surrogate's Pearson correlation against
true greedy tour lengths is measured on the fixed
validation set for THIS subject at tau 0.01 and
recorded, together with the mean share of
probability mass on the greedy pick at tau 0.01
(run 1's tau 0.1 figures, for comparison: Pearson
0.5845, on-pick mass ~38%).

OUTPUTS (new files only; run 1's outputs are not
touched):
  engine_runs/frankenstein_twofranks_backprop2/
    run.log, mask.json, checkpoint_best.npz,
    checkpoint_phase1_end.npz (when phase 1 ends),
    frankenstein_twofranks_backprop2_champion.npz,
    COMPLETE
  frankenstein_twofranks_backprop2_2026-11-26.json

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_backprop2_2026-11-26.py \
    > engine_runs/frankenstein_twofranks_backprop2/launcher.log \
    2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import os
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
OUT = HERE / "engine_runs" / "frankenstein_twofranks_backprop2"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a", buffering=1)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")


# ================================================================
# Constants: the schedule and the CORRECTED setup
# ================================================================
CFG = HERE / "bert_engine_config_fresh50.json"
SUBJECT = (HERE / "engine_runs" / "frankenstein_twofranks_evolution"
           / "frankenstein_twofranks_extended_champion.npz")
FOUNDER_STITCH = HERE / "frankenstein_twofranks_genome.npz"
STITCH_JSON = HERE / "frankenstein_twofranks_stitch_2026-11-26.json"

TAU = 0.01                    # corrected: was 0.1 in run 1
BATCH = 64
LR0 = 1e-4                    # corrected: was 0.01 in run 1
LR_FLOOR = 1e-6
CLIP_NORM = 1.0               # corrected: new in run 2
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
TRAIN_SEED = 20261203
READ_EVERY = 25               # corrected: was 100 in run 1

TRIGGER_PCT = 6.8959          # Frank #1, the better parent
PHASE1_MAX_STEPS = 3000       # corrected: was 1500 in run 1
PHASE1_EXTRA_STEPS = 200      # "a little more" after trigger
RAMP_STEPS = 200              # phase 2 backbone ramp
PHASE3_STEPS = 500            # fixed cycles, then stop
HARD_CAP = 3900               # corrected: was 2200 in run 1

# Run 1's tau-0.1 diagnostics, for comparison in the record
RUN1_TAU = 0.1
RUN1_PEARSON = 0.5845
RUN1_ONPICK_MASS = 0.38       # ~62% of mass off the greedy pick

BLOCKS = ("b0", "b1", "b2")

# ================================================================
# Setup: domain (validation set), scale, subject, mask
# (verbatim from run 1)
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
    f"stitching); recorded gap {RECORDED_GAP_PCT:.4f}% "
    f"(meta: best at generation "
    f"{SUBJ_META.get('best_validation_generation')})")

zf = np.load(FOUNDER_STITCH)
assert [k for k in zf.files if k != "meta"] == KEYS
w_founder = np.concatenate(
    [np.asarray(zf[k], dtype=float).ravel() for k in KEYS])


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k]) for k in KEYS}


# ================================================================
# Numpy forward + greedy decode (verbatim from run 1)
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
# JAX mirror: encoder + anchored soft surrogate
# (verbatim from run 1; TAU is now 0.01)
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


def soft_len_jax(W, x, picks):
    """Anchored soft tour length for ONE problem: the
    greedy picks are constants; at each visited state
    the pointer distribution at temperature TAU prices
    the expected next leg (bert_attribution's
    surrogate, verbatim in structure)."""
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
        p = jax.nn.softmax(logits / TAU)
        if t == 0:
            first_p = p
        else:
            L = L + p @ D[picks[t - 1]]
        visited = visited.at[picks[t]].set(1.0)
    L = L + first_p @ D[picks[n - 1]]
    return L


def onpick_mass_one(W, x, picks, tau):
    """Mean share of the pointer distribution's mass
    sitting on the network's own greedy pick, averaged
    over the visited states of ONE problem, at the
    given temperature. Diagnostic only."""
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


def loss_fn(w_flat, Xb, Pk):
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}
    L = jax.vmap(lambda x, pk: soft_len_jax(W, x, pk))(Xb, Pk)
    return jnp.mean(L)


vg = jax.jit(jax.value_and_grad(loss_fn))

# ================================================================
# SANITY (a): the subject's true gap must reproduce
# the recorded 7.3066%
# ================================================================
subj_gap = mean_gap(w_subj, X_val, OPT_VAL)
log(f"sanity (a): subject true validation gap "
    f"{subj_gap*100:.6f}% vs recorded "
    f"{RECORDED_GAP_PCT:.4f}%")
if abs(subj_gap * 100 - RECORDED_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY (a) FAILED. STOPPING; no training.")
    sys.exit(3)
log("sanity (a) passed")

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
    log("MIRROR CHECK FAILED. STOPPING; no training.")
    sys.exit(3)
log("mirror check passed")

# ================================================================
# SANITY (b): surrogate diagnostics for THIS subject
# at tau 0.01, on the fixed validation set: Pearson
# vs true lengths, and the mean on-pick mass (also
# measured at tau 0.1, for a like-for-like beside
# run 1's ~38%)
# ================================================================
true_lens_val, picks_val = true_lengths(w_subj, X_val)
soft_lens_val = np.asarray(jax.vmap(
    lambda x, pk: soft_len_jax(W_j, x, pk))(
    jnp.asarray(X_val, dtype=jnp.float64),
    jnp.asarray(picks_val)))
pearson = float(pearsonr(soft_lens_val, true_lens_val)[0])
onpick_001 = float(np.mean(np.asarray(jax.vmap(
    lambda x, pk: onpick_mass_one(W_j, x, pk, TAU))(
    jnp.asarray(X_val, dtype=jnp.float64),
    jnp.asarray(picks_val)))))
onpick_01_measured = float(np.mean(np.asarray(jax.vmap(
    lambda x, pk: onpick_mass_one(W_j, x, pk, RUN1_TAU))(
    jnp.asarray(X_val, dtype=jnp.float64),
    jnp.asarray(picks_val)))))
log(f"sanity (b): surrogate Pearson vs true lengths on "
    f"the validation set (tau {TAU}): {pearson:.4f} "
    f"(run 1 at tau {RUN1_TAU}: {RUN1_PEARSON}); mean "
    f"soft length {soft_lens_val.mean():.4f} vs mean "
    f"true length {true_lens_val.mean():.4f}")
log(f"sanity (b): mean on-pick mass at tau {TAU}: "
    f"{onpick_001:.4f}; measured at tau {RUN1_TAU} on "
    f"this subject: {onpick_01_measured:.4f} (run 1 "
    f"reported ~{RUN1_ONPICK_MASS})")

# ================================================================
# Training state, with THE GUARD
# ================================================================
def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_stitching": flat_mask.astype(int).tolist(),
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "subject": str(SUBJECT.relative_to(HERE)),
    "rule": "exact-from-a-Frank = backbone (frozen in "
            "phase 1); drawn-as-filler = stitching",
}, indent=1))

w = w_subj.copy()
m_adam = np.zeros(N_GENES)
v_adam = np.zeros(N_GENES)
adam_t = 0
cur_lr = LR0
rng = np.random.default_rng(TRAIN_SEED)

reads = []
guard_events = []
n_reverts = 0
n_halvings = 0
best_gap = np.inf
best_w = w.copy()
best_step = -1


def do_read(step, segment):
    """True-gap read + THE GUARD. On improvement the
    working weights become the new best and training
    continues from them. On a worse read the working
    weights revert to the best-so-far weights, the
    Adam state resets, and the learning rate halves
    (floor LR_FLOOR)."""
    global best_gap, best_w, best_step, cur_lr, adam_t
    global n_reverts, n_halvings
    g = mean_gap(w, X_val, OPT_VAL)
    if g < best_gap:
        best_gap = g
        best_w = w.copy()
        best_step = step
        atomic_savez(OUT / "checkpoint_best.npz",
                     w=best_w, step=step, gap=g)
        reads.append({"step": step, "segment": segment,
                      "gap_pct": round(g * 100, 4),
                      "is_best": True,
                      "guard_action": "none_new_best",
                      "lr_after": cur_lr})
        log(f"READ step {step} ({segment}): true "
            f"validation gap {g*100:.4f}%  <-- new best")
        return g
    # worse (or equal): revert to the best working point
    w[:] = best_w
    m_adam[:] = 0.0
    v_adam[:] = 0.0
    adam_t = 0
    new_lr = max(cur_lr / 2.0, LR_FLOOR)
    halved = bool(new_lr < cur_lr)
    cur_lr = new_lr
    n_reverts += 1
    if halved:
        n_halvings += 1
    ev = {"step": step, "segment": segment,
          "gap_pct": round(g * 100, 4),
          "action": "revert_to_best"
                    + ("_and_halve_lr" if halved
                       else "_lr_at_floor"),
          "lr_after": cur_lr,
          "best_gap_pct": round(best_gap * 100, 4),
          "best_step": best_step}
    guard_events.append(ev)
    reads.append({"step": step, "segment": segment,
                  "gap_pct": round(g * 100, 4),
                  "is_best": False,
                  "guard_action": ev["action"],
                  "lr_after": cur_lr})
    log(f"READ step {step} ({segment}): true validation "
        f"gap {g*100:.4f}% — worse than best "
        f"{best_gap*100:.4f}% (step {best_step}); GUARD: "
        f"reverted to best weights, Adam reset, lr -> "
        f"{cur_lr:.2e} (revert #{n_reverts})")
    return g


def adam_step(backbone_mult, backbone_trainable):
    """One Adam step on a fresh batch, at the current
    guarded learning rate, with global-norm gradient
    clipping. Returns the batch soft loss. Frozen
    genes (multiplier 0 and not trainable) receive
    zero gradient and zero update: bit-identical,
    moments stay zero."""
    global adam_t
    Xb = rng.random((BATCH, N_CITIES, 2))
    W_now = unpack(w)
    Yb = forward_general(W_now, Xb)
    Pk = decode_general(W_now, Yb)
    loss, grad = vg(jnp.asarray(w, dtype=jnp.float64),
                    jnp.asarray(Xb, dtype=jnp.float64),
                    jnp.asarray(Pk))
    grad = np.asarray(grad, dtype=float)
    trainable = flat_mask.copy()          # stitching: True
    if backbone_trainable:
        trainable[:] = True
    g = np.where(trainable, grad, 0.0)
    gn = float(np.linalg.norm(g))
    if gn > CLIP_NORM:
        g = g * (CLIP_NORM / gn)
    adam_t += 1
    m_adam[:] = ADAM_B1 * m_adam + (1 - ADAM_B1) * g
    v_adam[:] = ADAM_B2 * v_adam + (1 - ADAM_B2) * g * g
    mh = m_adam / (1 - ADAM_B1 ** adam_t)
    vh = v_adam / (1 - ADAM_B2 ** adam_t)
    upd = mh / (np.sqrt(vh) + ADAM_EPS)
    mult = np.where(flat_mask, 1.0, backbone_mult)
    w[:] = w - cur_lr * mult * upd
    return float(loss)


# ================================================================
# PHASE 1: seams only, backbone frozen, until the
# trigger (true gap < 6.8959%) + 200 more steps;
# stop entirely at 3,000 steps if no trigger.
# The guard is active throughout.
# ================================================================
log(f"PHASE 1: stitching genes only "
    f"({len(STITCH_IDX)}), backbone frozen "
    f"({len(BACKBONE_IDX)}); trigger = true gap < "
    f"{TRIGGER_PCT:.4f}% at a read; reads every "
    f"{READ_EVERY} steps; guard active; lr starts "
    f"{LR0:.0e}, clip norm {CLIP_NORM}")
g0 = do_read(0, "phase1")
trigger_step = None
if g0 * 100 < TRIGGER_PCT:
    trigger_step = 0
    log(f"TRIGGER already met at step 0 (gap "
        f"{g0*100:.4f}% < {TRIGGER_PCT:.4f}%)")
step = 0
phase1_end = None
while True:
    step += 1
    loss = adam_step(backbone_mult=0.0,
                     backbone_trainable=False)
    seg = "phase1" if trigger_step is None else "phase1_extra"
    log(f"step {step} ({seg}): soft batch loss "
        f"{loss:.4f}, backbone mult 0.0, lr {cur_lr:.2e}")
    if step % READ_EVERY == 0:
        g = do_read(step, seg)
        if trigger_step is None and g * 100 < TRIGGER_PCT:
            trigger_step = step
            log(f"TRIGGER FIRED at step {step}: true "
                f"gap {g*100:.4f}% < {TRIGGER_PCT:.4f}%. "
                f"Running {PHASE1_EXTRA_STEPS} more "
                f"phase-1 steps, then phase 2.")
    if trigger_step is not None \
            and step >= trigger_step + PHASE1_EXTRA_STEPS:
        phase1_end = step
        break
    if trigger_step is None and step >= PHASE1_MAX_STEPS:
        phase1_end = step
        break
log(f"phase 1 ended at step {phase1_end}; trigger "
    f"step: {trigger_step}; guard so far: {n_reverts} "
    f"reverts, {n_halvings} halvings, lr {cur_lr:.2e}")

# phase-1 checkpoint + backbone integrity verification
atomic_savez(OUT / "checkpoint_phase1_end.npz",
             w=w.copy(), step=phase1_end,
             trigger_step=(trigger_step
                           if trigger_step is not None else -1))
p1 = np.load(OUT / "checkpoint_phase1_end.npz")["w"]
phase1_backbone_identical = bool(
    (p1[BACKBONE_IDX] == w_subj[BACKBONE_IDX]).all())
log(f"phase-1 verification: backbone genes bit-"
    f"identical to the subject's after phase 1: "
    f"{phase1_backbone_identical}")
assert phase1_backbone_identical

phase2_end = None
phase3_end = None
phase3_truncated = False
ramp_record = None

if trigger_step is None:
    log(f"NO TRIGGER after {PHASE1_MAX_STEPS} steps: "
        f"phase 2 never started. Stopping the whole "
        f"run, per the schedule.")
else:
    # ============================================================
    # PHASE 2: backbone phases in, multiplier ramping
    # linearly 0 -> 1.0 over 200 steps. Guard active.
    # ============================================================
    ramp_record = {
        "steps": RAMP_STEPS,
        "form": "linear: multiplier on ramp step j "
                "(0-indexed) = (j+1)/200, so the first "
                "ramp step uses 0.005 and the last 1.0",
        "from": 0.0, "to": 1.0,
        "stitching_multiplier_throughout": 1.0,
        "note": "the multiplier scales the current "
                "guarded learning rate (cur_lr), which "
                "the guard may have halved in phase 1",
    }
    log(f"PHASE 2: backbone learning-rate multiplier "
        f"ramps linearly 0 -> 1.0 over {RAMP_STEPS} "
        f"steps; stitching stays at 1.0; guard active")
    for j in range(RAMP_STEPS):
        step += 1
        mult = (j + 1) / RAMP_STEPS
        loss = adam_step(backbone_mult=mult,
                         backbone_trainable=True)
        log(f"step {step} (phase2): soft batch loss "
            f"{loss:.4f}, backbone mult {mult:.3f}, "
            f"lr {cur_lr:.2e}")
        if step % READ_EVERY == 0:
            do_read(step, "phase2")
    phase2_end = step
    log(f"phase 2 ended at step {phase2_end} (ramp "
        f"complete, backbone at full rate)")

    # ============================================================
    # PHASE 3: fixed 500 steps at full rate, then STOP
    # (hard cap 3,900 total steps). Guard active.
    # ============================================================
    log(f"PHASE 3: {PHASE3_STEPS} steps at full rate "
        f"(hard cap {HARD_CAP} total steps); guard "
        f"active")
    for j in range(PHASE3_STEPS):
        if step >= HARD_CAP:
            phase3_truncated = True
            log(f"hard cap {HARD_CAP} reached: phase 3 "
                f"truncated after {j} of {PHASE3_STEPS} "
                f"steps")
            break
        step += 1
        loss = adam_step(backbone_mult=1.0,
                         backbone_trainable=True)
        log(f"step {step} (phase3): soft batch loss "
            f"{loss:.4f}, backbone mult 1.0, "
            f"lr {cur_lr:.2e}")
        if step % READ_EVERY == 0:
            do_read(step, "phase3")
    phase3_end = step
    log(f"phase 3 ended at step {phase3_end}")

if step % READ_EVERY != 0:
    seg = ("phase1" if phase2_end is None
           else "phase3" if phase3_end is not None
           else "phase2")
    do_read(step, seg + "_final")

total_steps = step

# ================================================================
# Final verification: which genes did the best
# champion change vs the extended champion (subject)?
# ================================================================
BW = unpack(best_w)
final_check_gap = mean_gap(best_w, X_val, OPT_VAL)
assert abs(final_check_gap - best_gap) < 1e-12
n_stitch_changed = int((best_w[STITCH_IDX]
                        != w_subj[STITCH_IDX]).sum())
n_backbone_changed = int((best_w[BACKBONE_IDX]
                          != w_subj[BACKBONE_IDX]).sum())
founder_backbone_diff = int((best_w[BACKBONE_IDX]
                             != w_founder[BACKBONE_IDX]).sum())
log(f"final verification: best champion (step "
    f"{best_step}) changed {n_stitch_changed} of "
    f"{len(STITCH_IDX)} stitching genes and "
    f"{n_backbone_changed} of {len(BACKBONE_IDX)} "
    f"backbone genes vs the extended champion; its "
    f"backbone differs from the founder stitch's "
    f"backbone at {founder_backbone_diff} genes "
    f"(nonzero only possible if phase 2 ran: "
    f"{phase2_end is not None})")
if phase2_end is None:
    assert n_backbone_changed == 0
    assert founder_backbone_diff == 0

# ================================================================
# Vault the best-on-validation weights + COMPLETE
# ================================================================
meta = {
    "kind": "frankenstein_twofranks_backprop2_champion",
    "date": "2026-11-26",
    "base": "frankenstein_twofranks_extended_champion.npz "
            f"(true gap {RECORDED_GAP_PCT:.4f}%)",
    "method": "back propagation run 2 (corrected "
              "setup) under the sponsor's full "
              "schedule (2026-11-26): Adam on the "
              f"anchored soft surrogate (tau {TAU}), "
              f"lr {LR0} with guard halving (floor "
              f"{LR_FLOOR}), gradient clip norm "
              f"{CLIP_NORM}, batch {BATCH} fresh "
              f"problems per step, true reads every "
              f"{READ_EVERY} steps with revert-to-"
              "best guard; phase 1 seams only, "
              "backbone frozen; trigger = true gap < "
              f"{TRIGGER_PCT:.4f}%; +"
              f"{PHASE1_EXTRA_STEPS} steps; phase 2 "
              f"backbone multiplier ramp 0->1 over "
              f"{RAMP_STEPS} steps; phase 3 "
              f"{PHASE3_STEPS} steps at full rate; "
              f"hard cap {HARD_CAP}",
    "trigger_step": trigger_step,
    "phase1_end_step": phase1_end,
    "phase2_end_step": phase2_end,
    "phase3_end_step": phase3_end,
    "total_steps": total_steps,
    "guard_reverts": n_reverts,
    "guard_halvings": n_halvings,
    "final_lr": cur_lr,
    "best_validation_gap_pct": round(best_gap * 100, 4),
    "best_validation_step": best_step,
    "stitching_genes_changed_vs_subject": n_stitch_changed,
    "backbone_genes_changed_vs_subject": n_backbone_changed,
    "surrogate_pearson_validation_tau001": round(pearson, 4),
    "onpick_mass_tau001": round(onpick_001, 4),
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_twofranks_backprop2_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_twofranks_backprop2",
    "subject_gap_pct": RECORDED_GAP_PCT,
    "trigger_step": trigger_step,
    "phase2_started": phase2_end is not None,
    "total_steps": total_steps,
    "guard_reverts": n_reverts,
    "guard_halvings": n_halvings,
    "final_lr": cur_lr,
    "best_validation_gap_pct": round(best_gap * 100, 4),
    "best_validation_step": best_step,
    "champion": "frankenstein_twofranks_backprop2_champion.npz",
    "reads": reads,
    "guard_events": guard_events,
}, indent=1))
log(f"vaulted best champion: gap {best_gap*100:.4f}% "
    f"at step {best_step}; COMPLETE marker written")

# ================================================================
# Full record
# ================================================================
record = {
    "run": "backpropagation on the Two-Franks stitch, "
           "CORRECTED setup (run 2), sponsor's full "
           "schedule",
    "date": "2026-11-26",
    "sponsor_direction": "Roman, 2026-11-26 09:07 EST: "
                         "'Your full schedule: seams "
                         "first, backbone in once below "
                         "6.90%'; correction directed "
                         "2026-11-26 09:16 EST: rerun "
                         "with the corrected setup "
                         "(the first run's setup was "
                         "wrong, not the schedule)",
    "subject": {
        "file": "engine_runs/frankenstein_twofranks_"
                "evolution/frankenstein_twofranks_"
                "extended_champion.npz",
        "recorded_gap_pct": RECORDED_GAP_PCT,
        "genes_total": N_GENES,
        "genes_backbone": int(len(BACKBONE_IDX)),
        "genes_stitching": int(len(STITCH_IDX)),
    },
    "sanity": {
        "recomputed_gap_pct": round(subj_gap * 100, 6),
        "recorded_gap_pct": RECORDED_GAP_PCT,
        "abs_diff_pct_points": round(
            abs(subj_gap * 100 - RECORDED_GAP_PCT), 8),
        "passed": True,
        "jax_mirror_encoder_parity_max_abs_diff": parity,
    },
    "surrogate": {
        "kind": "anchored expected tour length "
                "(bert_attribution.soft_forward "
                "structure): the network's own greedy "
                "picks held constant; at each visited "
                "state the pointer distribution at "
                "temperature tau prices the expected "
                "next leg",
        "tau": TAU,
        "pearson_vs_true_lengths_validation": round(
            pearson, 4),
        "mean_soft_length_validation": round(
            float(soft_lens_val.mean()), 4),
        "mean_true_length_validation": round(
            float(true_lens_val.mean()), 4),
        "onpick_mass_tau001": round(onpick_001, 4),
        "onpick_mass_tau01_measured_this_subject": round(
            onpick_01_measured, 4),
        "run1_comparison": {
            "tau": RUN1_TAU,
            "pearson_vs_true_lengths_validation":
                RUN1_PEARSON,
            "onpick_mass_reported": RUN1_ONPICK_MASS,
            "note": "run 1 reported ~62% of the mass "
                    "off the greedy pick at tau 0.1, "
                    "i.e. ~38% on-pick",
        },
        "note": "measured on the fixed validation set "
                "for this subject before training; "
                "the run proceeded regardless of the "
                "values",
    },
    "training": {
        "optimizer": "hand-rolled Adam, bias-corrected, "
                     "one global step counter (reset "
                     "by the guard on every revert)",
        "learning_rate_initial": LR0,
        "learning_rate_floor": LR_FLOOR,
        "learning_rate_final": cur_lr,
        "gradient_clip_global_norm": CLIP_NORM,
        "adam_beta1": ADAM_B1, "adam_beta2": ADAM_B2,
        "adam_eps": ADAM_EPS,
        "batch": BATCH,
        "batch_stream": f"{BATCH} fresh problems per "
                        f"step, n = {N_CITIES} cities, "
                        f"uniform on the unit square, "
                        f"numpy default_rng seed "
                        f"{TRAIN_SEED}; never the "
                        f"validation problems",
        "picks": "each step's batch picks are the "
                 "current weights' own greedy picks "
                 "(numpy decode), held constant "
                 "inside that step's gradient",
        "gradient_masking": "a gene with multiplier 0 "
                            "also receives zero "
                            "gradient, so its Adam "
                            "moments stay at zero and "
                            "its value is bit-"
                            "identical while frozen",
        "objective": "mean anchored soft tour length "
                     "over the batch (raw lengths, "
                     "not gaps: fresh problems have "
                     "no optima computed)",
    },
    "guard": {
        "description": "true-gap reads on the fixed "
                       "validation set every "
                       f"{READ_EVERY} steps; the best-"
                       "on-validation weights are the "
                       "working point: a worse read "
                       "reverts the working weights "
                       "to the best-so-far weights, "
                       "resets the Adam state, and "
                       "halves the learning rate "
                       f"(floor {LR_FLOOR}); an "
                       "improving read continues from "
                       "the new best. Active in all "
                       "phases.",
        "reads_every_steps": READ_EVERY,
        "reverts": n_reverts,
        "halvings": n_halvings,
        "final_lr": cur_lr,
        "events": guard_events,
    },
    "schedule": {
        "trigger": f"true validation gap < "
                   f"{TRIGGER_PCT:.4f}% at a read "
                   f"(reads every {READ_EVERY} steps)",
        "phase1": f"stitching genes only; stop the "
                  f"whole run if no trigger after "
                  f"{PHASE1_MAX_STEPS} steps",
        "phase1_extra_steps_after_trigger":
            PHASE1_EXTRA_STEPS,
        "phase2_ramp": ramp_record,
        "phase3_steps": PHASE3_STEPS,
        "hard_cap_steps": HARD_CAP,
    },
    "outcome": {
        "trigger_fired": trigger_step is not None,
        "trigger_step": trigger_step,
        "phase1_end_step": phase1_end,
        "phase2_end_step": phase2_end,
        "phase3_end_step": phase3_end,
        "phase3_truncated_by_hard_cap": phase3_truncated,
        "total_steps": total_steps,
        "phase1_backbone_bit_identical":
            phase1_backbone_identical,
        "best_validation_gap_pct": round(
            best_gap * 100, 4),
        "best_validation_step": best_step,
        "genes_changed_vs_subject": {
            "stitching": n_stitch_changed,
            "backbone": n_backbone_changed},
        "backbone_diff_vs_founder_stitch_genes":
            founder_backbone_diff,
    },
    "reads": reads,
    "comparisons_pct": {
        "subject_evolution_result": RECORDED_GAP_PCT,
        "frank1_parent": 6.8959,
        "frank2_parent": 7.017,
        "this_run_best": round(best_gap * 100, 4),
    },
    "wall_seconds": round(time.time() - T0, 1),
}
with open(HERE / "frankenstein_twofranks_backprop2_"
                 "2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
log(f"saved frankenstein_twofranks_backprop2_"
    f"2026-11-26.json in {record['wall_seconds']}s")
log(f"DONE: best true gap {best_gap*100:.4f}% at step "
    f"{best_step} (subject {RECORDED_GAP_PCT:.4f}%, "
    f"Frank #1 6.8959%, Frank #2 7.017%); trigger "
    f"{'fired at step ' + str(trigger_step) if trigger_step is not None else 'never fired'}; "
    f"guard: {n_reverts} reverts, {n_halvings} "
    f"halvings, final lr {cur_lr:.2e}")
