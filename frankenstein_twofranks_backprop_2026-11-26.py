#!/usr/bin/env python3
"""BACKPROPAGATION on the Two-Franks stitch, under the
sponsor's full schedule (Roman, 2026-11-26 09:07 EST:
"Your full schedule: seams first, backbone in once
below 6.90%").

This run is the one place back propagation is used to
TRAIN weights in the Frankenstein line, and only
because the sponsor directed it for this subject. It
defines no branches and changes no standing rule.

SUBJECT: the extended Two-Franks champion
(engine_runs/frankenstein_twofranks_evolution/
frankenstein_twofranks_extended_champion.npz), true
gap 7.3066% on the fixed 200-problem validation set
(val_seed 20261116). 1,462 genes = 356 backbone +
1,106 stitching; the mask comes from the stitch
record (frankenstein_twofranks_stitch_2026-11-26.json,
stitching_mask.flat_mask_stitching, in gene order):
exact-from-a-Frank = backbone; drawn-as-filler =
stitching (all filler, the whole joining block, the
decoder cross blocks).

NETWORK: the Two-Franks network is an explicit weight
set (stream width 8, blocks b0/b1/b2 with fused
attention, one merged decoder), NOT an engine
ArchSpec shape. Its numpy forward + greedy decode
are copied verbatim from the stitch/extend scripts.
The JAX mirror below reproduces that forward exactly
(checked against the numpy forward before training).

OBJECTIVE: the soft surrogate of the gates work
(gates_test_2026-11-25.py / bert_attribution.py):
the ANCHORED expected tour length — the network's
own greedy picks are held constant, and at each
visited state the pointer distribution at
temperature tau = 0.1 prices the expected next leg.
Loss = mean soft length over a batch of 64 freshly
sampled training problems per step (n = 10 cities,
uniform on the unit square, from a recorded seed
stream; NEVER the validation problems). Gradients
flow through the encoder and the decoder weights;
the picks are constants within a step and are
recomputed from the current weights for each fresh
batch.

OPTIMIZER: hand-rolled Adam (beta1 0.9, beta2 0.999,
eps 1e-8, bias-corrected, one global step counter),
learning rate 0.01, applied per gene through a
learning-rate multiplier: stitching genes always
1.0; backbone genes 0 while frozen, ramping in
phase 2, 1.0 in phase 3. While a gene's multiplier
is 0 its gradient is also masked to zero, so its
Adam moments stay at zero and its value is
bit-identical — the phase-1 freeze is exact, not
approximate.

TRUE READS: the real greedy gap on the fixed
validation set at step 0 and every 100 steps. The
best-on-validation weights seen at any read are
checkpointed atomically and vaulted at the end.

THE SCHEDULE (the sponsor's design, followed
exactly):
  PHASE 1: only the 1,106 stitching genes train;
  the 356 backbone genes are frozen. Run until the
  TRUE gap first reads below 6.8959% (Frank #1's
  score, the better parent) — the trigger. After
  the trigger fires, run 200 more steps of phase 1
  ("a little more"), then phase 2. If the trigger
  has not fired after 1,500 steps, STOP the whole
  run and report that phase 2 never started.
  PHASE 2: the backbone phases in progressively but
  steeply: its learning-rate multiplier ramps
  linearly from 0 to 1.0 over 200 steps (multiplier
  on ramp step j, 0-indexed, is (j+1)/200); the
  stitching genes stay at full rate throughout.
  PHASE 3: after the ramp, 500 more steps at full
  rate, then STOP.
  Hard cap: 2,200 steps total; if the cap truncates
  phase 3, that is recorded.

BEFORE TRAINING: (a) sanity — the subject's true
greedy gap must reproduce 7.3066%; (b) the JAX
mirror's encoder must match the numpy forward;
(c) the surrogate's Pearson correlation against
true greedy tour lengths is measured on the fixed
validation set for THIS subject and recorded (a
prior subject tracked poorly at 0.547; the number
is reported, and the run proceeds regardless).

OUTPUTS (new files only):
  engine_runs/frankenstein_twofranks_backprop/
    run.log, checkpoint_best.npz,
    checkpoint_phase1_end.npz (when phase 1 ends),
    frankenstein_twofranks_backprop_champion.npz,
    COMPLETE
  frankenstein_twofranks_backprop_2026-11-26.json

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_backprop_2026-11-26.py \
    > engine_runs/frankenstein_twofranks_backprop/launcher.log \
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
OUT = HERE / "engine_runs" / "frankenstein_twofranks_backprop"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a", buffering=1)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")


# ================================================================
# Constants: the schedule and the training protocol
# ================================================================
CFG = HERE / "bert_engine_config_fresh50.json"
SUBJECT = (HERE / "engine_runs" / "frankenstein_twofranks_evolution"
           / "frankenstein_twofranks_extended_champion.npz")
FOUNDER_STITCH = HERE / "frankenstein_twofranks_genome.npz"
STITCH_JSON = HERE / "frankenstein_twofranks_stitch_2026-11-26.json"

TAU = 0.1
BATCH = 64
LR = 0.01
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
TRAIN_SEED = 20261202
READ_EVERY = 100

TRIGGER_PCT = 6.8959          # Frank #1, the better parent
PHASE1_MAX_STEPS = 1500       # stop if no trigger by here
PHASE1_EXTRA_STEPS = 200      # "a little more" after trigger
RAMP_STEPS = 200              # phase 2 backbone ramp
PHASE3_STEPS = 500            # fixed cycles, then stop
HARD_CAP = 2200

BLOCKS = ("b0", "b1", "b2")

# ================================================================
# Setup: domain (validation set), scale, subject, mask
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
# Numpy forward + greedy decode (verbatim conventions
# from the stitch / extend scripts)
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
# (bert_attribution.soft_forward, generalised to this
# fused-attention explicit weight set)
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
# SANITY (b): surrogate correlation for THIS subject,
# on the fixed validation set (soft anchored lengths
# at tau = 0.1 vs the subject's true greedy lengths)
# ================================================================
true_lens_val, picks_val = true_lengths(w_subj, X_val)
soft_lens_val = np.asarray(jax.vmap(
    lambda x, pk: soft_len_jax(W_j, x, pk))(
    jnp.asarray(X_val, dtype=jnp.float64),
    jnp.asarray(picks_val)))
pearson = float(pearsonr(soft_lens_val, true_lens_val)[0])
log(f"sanity (b): surrogate Pearson vs true lengths on "
    f"the validation set (tau {TAU}): {pearson:.4f} "
    f"(mean soft length {soft_lens_val.mean():.4f} vs "
    f"mean true length {true_lens_val.mean():.4f})")

# ================================================================
# Training state
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
rng = np.random.default_rng(TRAIN_SEED)

reads = []
best_gap = np.inf
best_w = w.copy()
best_step = -1


def do_read(step, segment):
    global best_gap, best_w, best_step
    g = mean_gap(w, X_val, OPT_VAL)
    is_best = bool(g < best_gap)
    if is_best:
        best_gap = g
        best_w = w.copy()
        best_step = step
        atomic_savez(OUT / "checkpoint_best.npz",
                     w=best_w, step=step, gap=g)
    rec = {"step": step, "segment": segment,
           "gap_pct": round(g * 100, 4), "is_best": is_best}
    reads.append(rec)
    log(f"READ step {step} ({segment}): true validation "
        f"gap {g*100:.4f}%"
        + ("  <-- new best" if is_best else
           f"  (best so far {best_gap*100:.4f}% at step "
           f"{best_step})"))
    return g


def adam_step(backbone_mult, backbone_trainable):
    """One Adam step on a fresh batch. Returns the
    batch soft loss. Frozen genes (multiplier 0 and
    not trainable) receive zero gradient and zero
    update: bit-identical, moments stay zero."""
    global adam_t
    Xb = rng.random((BATCH, N_CITIES, 2))
    W_now = unpack(w)
    Yb = forward_general(W_now, Xb)
    Pk = decode_general(W_now, Yb)
    loss, grad = vg(jnp.asarray(w, dtype=jnp.float64),
                    jnp.asarray(Xb, dtype=jnp.float64),
                    jnp.asarray(Pk))
    grad = np.asarray(grad, dtype=float)
    adam_t += 1
    trainable = flat_mask.copy()          # stitching: True
    if backbone_trainable:
        trainable[:] = True
    g = np.where(trainable, grad, 0.0)
    m_adam[:] = ADAM_B1 * m_adam + (1 - ADAM_B1) * g
    v_adam[:] = ADAM_B2 * v_adam + (1 - ADAM_B2) * g * g
    mh = m_adam / (1 - ADAM_B1 ** adam_t)
    vh = v_adam / (1 - ADAM_B2 ** adam_t)
    upd = mh / (np.sqrt(vh) + ADAM_EPS)
    mult = np.where(flat_mask, 1.0, backbone_mult)
    w[:] = w - LR * mult * upd
    return float(loss)


# ================================================================
# PHASE 1: seams only, backbone frozen, until the
# trigger (true gap < 6.8959%) + 200 more steps;
# stop entirely at 1,500 steps if no trigger.
# ================================================================
log(f"PHASE 1: stitching genes only "
    f"({len(STITCH_IDX)}), backbone frozen "
    f"({len(BACKBONE_IDX)}); trigger = true gap < "
    f"{TRIGGER_PCT:.4f}% at a read; reads every "
    f"{READ_EVERY} steps")
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
        f"{loss:.4f}, backbone mult 0.0")
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
    f"step: {trigger_step}")

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
    # linearly 0 -> 1.0 over 200 steps
    # ============================================================
    ramp_record = {
        "steps": RAMP_STEPS,
        "form": "linear: multiplier on ramp step j "
                "(0-indexed) = (j+1)/200, so the first "
                "ramp step uses 0.005 and the last 1.0",
        "from": 0.0, "to": 1.0,
        "stitching_multiplier_throughout": 1.0,
    }
    log(f"PHASE 2: backbone learning-rate multiplier "
        f"ramps linearly 0 -> 1.0 over {RAMP_STEPS} "
        f"steps; stitching stays at 1.0")
    for j in range(RAMP_STEPS):
        step += 1
        mult = (j + 1) / RAMP_STEPS
        loss = adam_step(backbone_mult=mult,
                         backbone_trainable=True)
        log(f"step {step} (phase2): soft batch loss "
            f"{loss:.4f}, backbone mult {mult:.3f}")
        if step % READ_EVERY == 0:
            do_read(step, "phase2")
    phase2_end = step
    log(f"phase 2 ended at step {phase2_end} (ramp "
        f"complete, backbone at full rate)")

    # ============================================================
    # PHASE 3: fixed 500 steps at full rate, then STOP
    # (hard cap 2,200 total steps)
    # ============================================================
    log(f"PHASE 3: {PHASE3_STEPS} steps at full rate "
        f"(hard cap {HARD_CAP} total steps)")
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
            f"{loss:.4f}, backbone mult 1.0")
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
    "kind": "frankenstein_twofranks_backprop_champion",
    "date": "2026-11-26",
    "base": "frankenstein_twofranks_extended_champion.npz "
            f"(true gap {RECORDED_GAP_PCT:.4f}%)",
    "method": "back propagation under the sponsor's "
              "full schedule (2026-11-26): Adam "
              f"(lr {LR}) on the anchored soft "
              f"surrogate (tau {TAU}), batch {BATCH} "
              "fresh problems per step; phase 1 seams "
              "only, backbone frozen; trigger = true "
              f"gap < {TRIGGER_PCT:.4f}%; +"
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
    "best_validation_gap_pct": round(best_gap * 100, 4),
    "best_validation_step": best_step,
    "stitching_genes_changed_vs_subject": n_stitch_changed,
    "backbone_genes_changed_vs_subject": n_backbone_changed,
    "surrogate_pearson_validation": round(pearson, 4),
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_twofranks_backprop_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_twofranks_backprop",
    "subject_gap_pct": RECORDED_GAP_PCT,
    "trigger_step": trigger_step,
    "phase2_started": phase2_end is not None,
    "total_steps": total_steps,
    "best_validation_gap_pct": round(best_gap * 100, 4),
    "best_validation_step": best_step,
    "champion": "frankenstein_twofranks_backprop_champion.npz",
    "reads": reads,
}, indent=1))
log(f"vaulted best champion: gap {best_gap*100:.4f}% "
    f"at step {best_step}; COMPLETE marker written")

# ================================================================
# Full record
# ================================================================
record = {
    "run": "backpropagation on the Two-Franks stitch, "
           "sponsor's full schedule",
    "date": "2026-11-26",
    "sponsor_direction": "Roman, 2026-11-26 09:07 EST: "
                         "'Your full schedule: seams "
                         "first, backbone in once below "
                         "6.90%'",
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
        "note": "measured on the fixed validation set "
                "for this subject before training; "
                "the run proceeded regardless of the "
                "value",
    },
    "training": {
        "optimizer": "hand-rolled Adam, bias-corrected, "
                     "one global step counter",
        "learning_rate": LR,
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
with open(HERE / "frankenstein_twofranks_backprop_"
                 "2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
log(f"saved frankenstein_twofranks_backprop_"
    f"2026-11-26.json in {record['wall_seconds']}s")
log(f"DONE: best true gap {best_gap*100:.4f}% at step "
    f"{best_step} (subject {RECORDED_GAP_PCT:.4f}%, "
    f"Frank #1 6.8959%, Frank #2 7.017%); trigger "
    f"{'fired at step ' + str(trigger_step) if trigger_step is not None else 'never fired'}")
