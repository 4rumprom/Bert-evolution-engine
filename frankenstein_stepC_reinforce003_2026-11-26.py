#!/usr/bin/env python3
"""STAGE 2 of the sponsor's 2026-11-26 direction:
REINFORCE training on the Net2WiderNet-widened
Step C Frankenstein (stage 1:
frankenstein_stepC_widen_2026-11-26.py).

SUBJECT: frankenstein_stepC_widened003_genome.npz —
the Step C champion (8.1018% on the fixed
200-problem validation set, val_seed 20261116,
10 cities) widened across the whole network
(stream 4 -> 8, FFN x2, heads x2 per block,
decoder widened with the stream; 8,480 genes =
1,440 frozen original copies + 7,040 trainable
new copies), with Gaussian noise std 0.003 on
the new copies only (RERUN of the 0.03 run,
whose birth gap drifted to 20.312084%; the
sponsor directed the noise be tiny so the
birth score is nearly identical). Widened
birth gap WITH noise is read from the
stage-1 widen003 JSON at runtime (the
no-noise split was verified exact first:
identical tours, gap 8.101755%). attn_scale of the widened network
= 0.35355339059327373 (halved), ln_eps 1e-5.

THE METHOD (REINFORCE, the sponsor's spec of
2026-11-26, with his 22:01 amendment for this
run: one sampled path per starting city,
20 cities):
  - The network is deterministic. It produces a
    probability distribution over the next city
    at each step (softmax over the decoder
    logits exactly as the decoder produces
    them; NO temperature, NO soft surrogate
    anywhere).
  - Training problems: 20 cities, drawn fresh
    uniform on the unit square per problem
    (seed stream 20261241; update u uses batch
    seed 20261241 + u; never the validation
    problems).
  - For each problem, ONE sampled path per
    starting city: 20 rollouts, the first pick
    FORCED to each city in turn, the remaining
    picks sampled from the network's own
    probabilities. The forced first pick is not
    a network choice, so its log-probability is
    EXCLUDED from the loss (the rollout still
    conditions on it: the start city is visited
    and its embedding is the 'last' state for
    step 2).
  - Baseline: the network's own GREEDY tour
    length on the same problem (computed once
    per problem). Advantage per rollout =
    greedy length - sampled length.
  - Loss per rollout = -advantage x (sum of
    log-probabilities of the sampled actions,
    forced first pick excluded). Losses are
    averaged over the whole batch (4 problems x
    20 rollouts = 80 sampled tours); the weights
    are frozen while all rollouts of the batch
    are generated; ONE Adam update per batch.
  - Optimizer: Adam, lr 1e-4, global-norm
    gradient clip 1.0 (the corrected-backprop
    conditions).
  - ONLY the trainable (new-copy) genes move:
    the gradient is masked to zero on the 1,440
    frozen genes; frozen genes are verified
    bit-identical at the end.
  - GUARD (from the corrected backprop runs):
    true-score reads on the fixed 10-city
    validation set every 25 updates; the
    best-on-validation weights are the working
    point; on a worse read, revert to the best
    weights, reset Adam, halve lr (floor 1e-6).
    Every revert and halving is logged.
  - 400 updates.

SANITY FIRST: the widened founder must
reproduce the stage-1 recorded birth gap
(20.312084%) under the numpy forward; the JAX
mirror must match the numpy forward (encoder
parity and greedy-tour parity at n = 10 on
validation problems, and greedy parity at
n = 20 on fresh problems) before training.

MACHINERY: follows
frankenstein_twofranks_reinforce_train_
2026-11-26.py (sampling, guard, vaulting) and
frankenstein_twofranks_reinforce_sizes_
2026-11-26.py (n = 20 training), extended to
the widened multi-head forward and to the
one-path-per-starting-city rollout (the
encoder runs once per problem and is shared
by its greedy rollout and its 20 sampled
rollouts).

OUTPUTS (new files only):
  engine_runs/frankenstein_stepC_reinforce003/
    run.log, mask.json,
    frankenstein_stepC_reinforce003_champion.npz,
    COMPLETE
  frankenstein_stepC_reinforce003_2026-11-26.json

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_stepC_reinforce003_2026-11-26.py \
    > engine_runs/frankenstein_stepC_reinforce003/launcher.log \
    2>&1 < /dev/null &
"""
from __future__ import annotations

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
OUT = HERE / "engine_runs" / "frankenstein_stepC_reinforce003"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a", buffering=1)


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")


# ================================================================
# Constants
# ================================================================
CFG = HERE / "bert_engine_config_deep50_stepA.json"
WIDENED = HERE / "frankenstein_stepC_widened003_genome.npz"
WIDEN_JSON = HERE / "frankenstein_stepC_widen003_2026-11-26.json"

VAL_CITIES = 10
N_CITIES = 20                 # training problems
PROBLEMS_PER_UPDATE = 4
ROLLOUTS_PER_PROBLEM = N_CITIES   # one per starting city
UPDATES = 400
SEED_BASE = 20261241
LR0 = 1e-4
LR_FLOOR = 1e-6
CLIP_NORM = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
READ_EVERY = 25
BLOCKS = ("b0", "b1", "b2", "b3", "b4", "b5w")
HEADS = (0, 1, 2, 3)
SUBJECT_GAP_PCT = 8.1018

# ================================================================
# Setup: domain (validation set), widened genome,
# mask
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
EPS = float(arch.ln_eps)

zw = np.load(WIDENED)
WMETA = json.loads(str(zw["meta"]))
KEYS = list(WMETA["gene_order"])
assert [k for k in zw.files if k not in
        ("meta", "trainable_mask", "frozen_idx",
         "trainable_idx")] == KEYS or \
       set(k for k in zw.files if k not in
           ("meta", "trainable_mask", "frozen_idx",
            "trainable_idx")) == set(KEYS)
WID_W = {k: np.asarray(zw[k], dtype=float) for k in KEYS}
SHAPES = {k: WID_W[k].shape for k in KEYS}
SLICES = {}
_off = 0
for k in KEYS:
    SLICES[k] = (slice(_off, _off + WID_W[k].size), SHAPES[k])
    _off += WID_W[k].size
N_GENES = _off
train_mask = np.asarray(zw["trainable_mask"]).astype(bool)
FROZEN_IDX = np.asarray(zw["frozen_idx"])
TRAIN_IDX = np.asarray(zw["trainable_idx"])
assert N_GENES == WMETA["genes_total"] == 8480
assert len(FROZEN_IDX) == WMETA["genes_frozen"] == 1440
assert len(TRAIN_IDX) == WMETA["genes_trainable"] == 7040
assert (train_mask[TRAIN_IDX].all()
        and not train_mask[FROZEN_IDX].any())
SCALE = float(WMETA["attn_scale"])     # halved: 0.35355...
assert abs(SCALE - 0.5 / np.sqrt(2.0)) < 1e-15
assert abs(float(WMETA["ln_eps"]) - EPS) < 1e-15
w_widened = np.concatenate([WID_W[k].ravel() for k in KEYS])

WJ = json.load(open(WIDEN_JSON))
BIRTH_GAP_PCT = float(WJ["birth_gap_with_noise"]["gap_pct"])
log(f"widened subject loaded: {N_GENES} genes "
    f"({len(FROZEN_IDX)} frozen + {len(TRAIN_IDX)} "
    f"trainable); attn_scale = {SCALE!r}, "
    f"ln_eps = {EPS!r}; recorded birth gap (with "
    f"noise) {BIRTH_GAP_PCT:.4f}%; pre-widen "
    f"subject gap {SUBJECT_GAP_PCT:.4f}%")


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k])
            for k in KEYS}


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ================================================================
# Numpy forward (widened, multi-head) + greedy
# decode — the TRUE score, reporting only
# (verbatim conventions from
# frankenstein_stepC_widen_2026-11-26.py)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_wide(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS:
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        out = 0.0
        for h in HEADS:
            p = f"{blk}.h{h}"
            Q = Hb @ W[f"{p}.W_q"] + W[f"{p}.b_q"]
            K = Hb @ W[f"{p}.W_k"] + W[f"{p}.b_k"]
            V = Hb @ W[f"{p}.W_v"] + W[f"{p}.b_v"]
            S = np.einsum("btj,bsj->bts", Q, K) * SCALE
            S = S - S.max(axis=-1, keepdims=True)
            E = np.exp(S)
            Atn = E / E.sum(axis=-1, keepdims=True)
            Ah = np.einsum("bts,bsi->bti", Atn, V)
            out = out + Ah @ W[f"{p}.W_o"] + W[f"{p}.b_o"]
        A = A + out
        Hf = ln_np(A, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"]
                        + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
    return A


def decode_greedy(W, Y):
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
    Y = forward_wide(W, X)
    picks = decode_greedy(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return lens, picks


def mean_gap(vec, X, opt):
    lens, _ = true_lengths(vec, X)
    return float((lens / opt - 1.0).mean())


# ================================================================
# JAX mirror: encoder (widened, multi-head)
# ================================================================
def _ln_j(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * gain + bias


def encode_jax(W, x):
    A = x @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS:
        Hb = _ln_j(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        out = 0.0
        for h in HEADS:
            p = f"{blk}.h{h}"
            Q = Hb @ W[f"{p}.W_q"] + W[f"{p}.b_q"]
            K = Hb @ W[f"{p}.W_k"] + W[f"{p}.b_k"]
            V = Hb @ W[f"{p}.W_v"] + W[f"{p}.b_v"]
            scores = Q @ K.T * SCALE
            attn = jax.nn.softmax(scores, axis=-1)
            Ah = attn @ V
            out = out + Ah @ W[f"{p}.W_o"] + W[f"{p}.b_o"]
        A = A + out
        Hf = _ln_j(A, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A = A + jnp.tanh(Hf @ W[f"{blk}.W_1"]
                         + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
    return A


# ================================================================
# Rollouts on a precomputed encoding Y of ONE
# problem: the greedy rollout (baseline) and the
# sampled rollout with a FORCED first pick (one
# per starting city)
# ================================================================
def _dist(x):
    return jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                    .sum(axis=-1))


def greedy_from_Y(W, Y, x):
    """Greedy tour length for ONE problem from its
    encoding: always the most probable city. The
    picks are integers, so no gradient flows; the
    length is the BASELINE constant."""
    n = x.shape[0]
    D = _dist(x)
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    picks = jnp.zeros(n, dtype=jnp.int32)
    last = W["head.v_start"]
    for t in range(n):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        pick = jnp.argmax(logits).astype(jnp.int32)
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return length


def sample_from_Y(W, Y, x, start, key):
    """One sampled tour for ONE problem whose
    FIRST pick is FORCED to `start` (one rollout
    per starting city). The remaining picks are
    sampled from the network's OWN distribution
    (no temperature). The forced first pick is
    not a network choice: its log-probability is
    EXCLUDED from sum_logp (the rollout still
    conditions on it — the start city is visited
    and its embedding is the 'last' state for
    the next step). Returns (sum_logp, length)."""
    n = x.shape[0]
    D = _dist(x)
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n).at[start].set(1.0)
    picks = jnp.zeros(n, dtype=jnp.int32).at[0].set(start)
    sum_logp = 0.0
    last = Y[start]
    for t in range(1, n):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        logp = jax.nn.log_softmax(logits)
        key, sub = jax.random.split(key)
        pick = jax.random.categorical(sub, logits).astype(
            jnp.int32)
        sum_logp = sum_logp + logp[pick]
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return sum_logp, length


def reinforce_loss(w_flat, Xb, keys):
    """REINFORCE loss with the GREEDY baseline and
    one sampled path per starting city. Xb:
    (B, 20, 2) fresh problems; keys: (B, 20, 2),
    one per (problem, starting city), all
    generated under the SAME weights (w_flat).
    Per problem: greedy length = baseline;
    advantage_k = baseline - sampled_length_k
    (stop-gradient); loss contribution =
    -mean_k(advantage_k x sum_logp_k) with the
    forced first pick's log-prob excluded.
    Returns (loss, (sampled_lengths (B, 20),
    greedy_lengths (B,), advantages (B, 20)))."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x, ks):
        Y = encode_jax(W, x)
        greedy_len = greedy_from_Y(W, Y, x)
        starts = jnp.arange(x.shape[0], dtype=jnp.int32)

        def per_rollout(start, key):
            return sample_from_Y(W, Y, x, start, key)
        sum_logps, lengths = jax.vmap(per_rollout)(
            starts, ks)
        adv = jax.lax.stop_gradient(greedy_len - lengths)
        return (-jnp.mean(adv * sum_logps), lengths,
                greedy_len, adv)

    losses, lengths, greedy_lens, advs = jax.vmap(
        per_problem)(Xb, keys)
    return jnp.mean(losses), (lengths, greedy_lens, advs)


vg = jax.jit(jax.value_and_grad(reinforce_loss, has_aux=True))


def make_batch(batch_seed):
    """Fresh training problems + rollout keys for
    one update, deterministically from a seed.
    Problems are uniform on the unit square
    (never the validation problems); keys derive
    from the same seed."""
    rng = np.random.default_rng(batch_seed)
    Xb = rng.random((PROBLEMS_PER_UPDATE, N_CITIES, 2))
    key = jax.random.PRNGKey(batch_seed)
    keys = jax.random.split(
        key, PROBLEMS_PER_UPDATE
        * ROLLOUTS_PER_PROBLEM).reshape(
        PROBLEMS_PER_UPDATE, ROLLOUTS_PER_PROBLEM, 2)
    return Xb, keys


# ================================================================
# SANITY + MACHINERY CHECKS (before training)
# ================================================================
birth_gap = mean_gap(w_widened, X_val, OPT_VAL)
log(f"sanity: widened founder true validation gap "
    f"{birth_gap*100:.6f}% vs stage-1 recorded "
    f"birth gap {BIRTH_GAP_PCT:.4f}%")
if abs(birth_gap * 100 - BIRTH_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY FAILED. STOPPING; no training.")
    sys.exit(3)
log("sanity passed")

W_subj = unpack(w_widened)
Y_np = forward_wide(W_subj, X_val[:8])
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
    log("MIRROR CHECK FAILED. STOPPING; no training.")
    sys.exit(3)
log("mirror check passed")

np_greedy_lens, _ = true_lengths(w_widened, X_val[:8])
jx_greedy_lens = np.asarray(jax.vmap(
    lambda x: greedy_from_Y(W_j, encode_jax(W_j, x), x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
greedy_parity = float(np.max(np.abs(np_greedy_lens
                                    - jx_greedy_lens)))
log(f"greedy-baseline check (n=10): JAX greedy "
    f"rollout vs numpy greedy lengths, max abs "
    f"diff {greedy_parity:.2e} on 8 validation "
    f"problems")
if greedy_parity > 1e-9:
    log("GREEDY-BASELINE CHECK FAILED (n=10). "
        "STOPPING; no training.")
    sys.exit(3)
log("greedy-baseline check (n=10) passed")

# greedy parity at the TRAINING size, n = 20, on
# fresh problems (generation under the widened
# weights must be clean at n = 20)
rng_chk = np.random.default_rng(SEED_BASE)
X_chk = rng_chk.random((4, N_CITIES, 2))
np_greedy20, _ = true_lengths(w_widened, X_chk)
jx_greedy20 = np.asarray(jax.vmap(
    lambda x: greedy_from_Y(W_j, encode_jax(W_j, x), x))(
    jnp.asarray(X_chk, dtype=jnp.float64)))
greedy20_parity = float(np.max(np.abs(np_greedy20
                                      - jx_greedy20)))
log(f"greedy check (n=20): JAX greedy rollout vs "
    f"numpy greedy lengths, max abs diff "
    f"{greedy20_parity:.2e} on 4 fresh 20-city "
    f"problems")
if greedy20_parity > 1e-9:
    log("GREEDY CHECK FAILED (n=20). STOPPING; "
        "no training.")
    sys.exit(3)
log("greedy check (n=20) passed")

atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_trainable": train_mask.astype(int).tolist(),
    "genes_total": N_GENES,
    "genes_frozen": int(len(FROZEN_IDX)),
    "genes_trainable": int(len(TRAIN_IDX)),
    "subject": "frankenstein_stepC_widened003_genome.npz",
    "rule": "trainable = the new copies' genes "
            "(stage-1 widen mask); frozen = the "
            "original copies' genes, bit-identical "
            "throughout",
}, indent=1))


# ================================================================
# Training, with THE GUARD (pattern from the
# corrected backprop run / twofranks REINFORCE)
# ================================================================
log(f"TRAINING: {PROBLEMS_PER_UPDATE} problems x "
    f"{ROLLOUTS_PER_PROBLEM} rollouts (one sampled "
    f"path per starting city) = "
    f"{PROBLEMS_PER_UPDATE * ROLLOUTS_PER_PROBLEM} "
    f"sampled tours + {PROBLEMS_PER_UPDATE} greedy "
    f"baselines per update, {UPDATES} updates, "
    f"n = {N_CITIES} cities, seed base {SEED_BASE}, "
    f"Adam lr {LR0:.0e}, guard reads every "
    f"{READ_EVERY} updates on the fixed "
    f"{VAL_CITIES}-city validation set")
t_train = time.time()

w = w_widened.copy()
adam = {"m": np.zeros(N_GENES), "v": np.zeros(N_GENES),
        "t": 0}
state = {"lr": LR0, "best_gap": np.inf,
         "best_w": w.copy(), "best_update": -1,
         "reverts": 0, "halvings": 0}
reads = []
guard_events = []
stat_lengths = []
stat_advantages = []
stat_greedy = []


def do_read(update):
    g = mean_gap(w, X_val, OPT_VAL)
    if g < state["best_gap"]:
        state["best_gap"] = g
        state["best_w"] = w.copy()
        state["best_update"] = update
        reads.append({"update": update,
                      "gap_pct": round(g * 100, 4),
                      "is_best": True,
                      "guard_action": "none_new_best",
                      "lr_after": state["lr"]})
        log(f"READ update {update}: true validation "
            f"gap {g*100:.4f}%  <-- new best")
        return g
    w[:] = state["best_w"]
    adam["m"][:] = 0.0
    adam["v"][:] = 0.0
    adam["t"] = 0
    new_lr = max(state["lr"] / 2.0, LR_FLOOR)
    halved = bool(new_lr < state["lr"])
    state["lr"] = new_lr
    state["reverts"] += 1
    if halved:
        state["halvings"] += 1
    ev = {"update": update,
          "gap_pct": round(g * 100, 4),
          "action": "revert_to_best"
                    + ("_and_halve_lr" if halved
                       else "_lr_at_floor"),
          "lr_after": state["lr"],
          "best_gap_pct": round(state["best_gap"]
                                * 100, 4),
          "best_update": state["best_update"]}
    guard_events.append(ev)
    reads.append({"update": update,
                  "gap_pct": round(g * 100, 4),
                  "is_best": False,
                  "guard_action": ev["action"],
                  "lr_after": state["lr"]})
    log(f"READ update {update}: true validation gap "
        f"{g*100:.4f}% — worse than best "
        f"{state['best_gap']*100:.4f}% (update "
        f"{state['best_update']}); GUARD: reverted "
        f"to best weights, Adam reset, lr -> "
        f"{state['lr']:.2e} (revert "
        f"#{state['reverts']})")
    return g


do_read(0)
losses = []
for u in range(1, UPDATES + 1):
    Xb, keys = make_batch(SEED_BASE + u)
    (loss, (lengths, greedy_lens, advs)), grad = vg(
        jnp.asarray(w, dtype=jnp.float64),
        jnp.asarray(Xb, dtype=jnp.float64), keys)
    grad = np.asarray(grad, dtype=float)
    grad = np.where(train_mask, grad, 0.0)  # frozen stay frozen
    gn = float(np.linalg.norm(grad))
    if gn > CLIP_NORM:
        grad = grad * (CLIP_NORM / gn)
    adam["t"] += 1
    adam["m"][:] = ADAM_B1 * adam["m"] + (1 - ADAM_B1) * grad
    adam["v"][:] = ADAM_B2 * adam["v"] \
        + (1 - ADAM_B2) * grad * grad
    mh = adam["m"] / (1 - ADAM_B1 ** adam["t"])
    vh = adam["v"] / (1 - ADAM_B2 ** adam["t"])
    w[:] = w - state["lr"] * mh / (np.sqrt(vh) + ADAM_EPS)
    losses.append(float(loss))
    stat_lengths.extend(
        np.asarray(lengths, dtype=float).ravel().tolist())
    stat_advantages.extend(
        np.asarray(advs, dtype=float).ravel().tolist())
    stat_greedy.extend(
        np.asarray(greedy_lens, dtype=float).ravel().tolist())
    if u % READ_EVERY == 0:
        do_read(u)
    elif u % 100 == 0:
        log(f"update {u}: loss {float(loss):.4f}, "
            f"best so far {state['best_gap']*100:.4f}% "
            f"(update {state['best_update']}), lr "
            f"{state['lr']:.2e}")

wall = time.time() - t_train
best_w = state["best_w"]
frozen_ok = bool(
    (w[FROZEN_IDX] == w_widened[FROZEN_IDX]).all()
    and (best_w[FROZEN_IDX]
         == w_widened[FROZEN_IDX]).all())
n_train_changed = int((best_w[TRAIN_IDX]
                       != w_widened[TRAIN_IDX]).sum())
assert frozen_ok
la = np.asarray(stat_lengths)
aa = np.asarray(stat_advantages)
ga = np.asarray(stat_greedy)
stats = {
    "n_sampled_tours": int(la.size),
    "n_greedy_tours": int(ga.size),
    "mean_sampled_length": round(float(la.mean()), 4),
    "std_sampled_length": round(float(la.std()), 4),
    "mean_greedy_length": round(float(ga.mean()), 4),
    "mean_advantage": round(float(aa.mean()), 4),
    "advantage_spread_std": round(float(aa.std()), 4),
    "advantage_min": round(float(aa.min()), 4),
    "advantage_max": round(float(aa.max()), 4),
    "frac_samples_beating_greedy": round(
        float((aa > 0).mean()), 4),
    "mean_loss_first_25_updates": round(
        float(np.mean(losses[:25])), 4),
    "mean_loss_last_25_updates": round(
        float(np.mean(losses[-25:])), 4),
}
log(f"DONE in {wall:.1f}s: best true gap "
    f"{state['best_gap']*100:.4f}% at update "
    f"{state['best_update']} (widened birth "
    f"{BIRTH_GAP_PCT:.4f}%, pre-widen subject "
    f"{SUBJECT_GAP_PCT:.4f}%); guard: "
    f"{state['reverts']} reverts, "
    f"{state['halvings']} halvings, final lr "
    f"{state['lr']:.2e}; signal (n=20): mean "
    f"sampled length {stats['mean_sampled_length']:.4f}, "
    f"mean greedy length "
    f"{stats['mean_greedy_length']:.4f}, mean "
    f"advantage {stats['mean_advantage']:.4f}, "
    f"advantage spread (std) "
    f"{stats['advantage_spread_std']:.4f}, "
    f"{stats['frac_samples_beating_greedy']*100:.2f}% "
    f"of samples beat their greedy baseline; "
    f"frozen genes bit-identical: {frozen_ok}; "
    f"trainable genes changed in best: "
    f"{n_train_changed}")

# ================================================================
# Vault the best-on-validation weights + record
# ================================================================
champ = OUT / "frankenstein_stepC_reinforce003_champion.npz"
meta = {
    "kind": "frankenstein_stepC_reinforce003_champion",
    "date": "2026-11-26",
    "base": "frankenstein_stepC_widened003_genome.npz "
            f"(birth gap with noise "
            f"{BIRTH_GAP_PCT:.4f}%; pre-widen Step C "
            f"subject {SUBJECT_GAP_PCT:.4f}%)",
    "method": "REINFORCE (sponsor's spec, "
              "2026-11-26): tours sampled from the "
              "network's own pointer distribution "
              "(no temperature, no surrogate); ONE "
              "sampled path per starting city on "
              "20-city problems (first pick forced "
              "to each city in turn, its "
              "log-probability excluded from the "
              "loss); greedy tour length as the "
              "per-problem baseline; advantage = "
              "baseline - sampled length; loss = "
              "-mean(advantage x sum log-prob of "
              "sampled actions); weights frozen "
              "while a batch is generated, losses "
              "averaged over the batch, one Adam "
              f"update per batch; Adam lr {LR0} "
              f"with guard halving (floor "
              f"{LR_FLOOR}), gradient clip norm "
              f"{CLIP_NORM}; guard reads every "
              f"{READ_EVERY} updates with "
              "revert-to-best; trainable (new-copy) "
              "genes only, frozen genes "
              "bit-identical",
    "problems_per_update": PROBLEMS_PER_UPDATE,
    "rollouts_per_problem": ROLLOUTS_PER_PROBLEM,
    "sampled_tours_per_update":
        PROBLEMS_PER_UPDATE * ROLLOUTS_PER_PROBLEM,
    "training_n_cities": N_CITIES,
    "updates": UPDATES,
    "seed_base": SEED_BASE,
    "best_validation_gap_pct": round(
        state["best_gap"] * 100, 4),
    "best_validation_update": state["best_update"],
    "guard_reverts": state["reverts"],
    "guard_halvings": state["halvings"],
    "final_lr": state["lr"],
    "signal_stats_n20": stats,
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_frozen": int(len(FROZEN_IDX)),
    "genes_trainable": int(len(TRAIN_IDX)),
    "trainable_genes_changed_vs_widened":
        n_train_changed,
    "frozen_bit_identical_to_widened": frozen_ok,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(champ, **{k: unpack(best_w)[k] for k in KEYS},
         trainable_mask=train_mask.astype(np.int8),
         meta=np.array(json.dumps(meta)))
log(f"vaulted {champ.name} (best gap "
    f"{state['best_gap']*100:.4f}% at update "
    f"{state['best_update']})")

record = {
    "run": "stage 2 RERUN: REINFORCE on the widened "
           "(noise std 0.003) Step C Frankenstein — "
           "one sampled path per starting city, "
           "20 cities",
    "date": "2026-11-26",
    "sponsor_direction": "Roman, 2026-11-26 22:01 "
                         "EST: widen with Net2Net "
                         "split + noise on the new "
                         "synapses, then back "
                         "propagation with "
                         "reinforcement learning — "
                         "one path per city given "
                         "with 20 cities. RERUN, "
                         "2026-11-26 22:12 EST: "
                         "noise at 0.003 (birth "
                         "score nearly identical)",
    "method": {
        "name": "REINFORCE (policy gradient) with a "
                "greedy baseline, exactly as the "
                "sponsor specified",
        "sampling": "external sampling from the "
                    "network's own pointer "
                    "distribution (softmax over the "
                    "decoder logits as produced; no "
                    "temperature, no soft surrogate)",
        "rollouts": "one sampled path per starting "
                    "city: 20 rollouts per problem, "
                    "the first pick forced to each "
                    "city in turn",
        "first_pick_logprob": "EXCLUDED from the "
                              "loss — the forced "
                              "first pick is not a "
                              "network choice. The "
                              "rollout still "
                              "conditions on it: the "
                              "start city is marked "
                              "visited and its "
                              "embedding is the "
                              "'last' state for the "
                              "next step",
        "baseline": "the problem's own greedy tour "
                    "length (always the most "
                    "probable city), computed once "
                    "per problem",
        "advantage": "baseline length - sampled "
                     "tour length (positive when "
                     "the sample beats the greedy "
                     "tour)",
        "loss": "-mean over the update batch "
                "(4 problems x 20 rollouts = 80 "
                "sampled tours) of advantage x (sum "
                "of log-probabilities of the "
                "sampled actions, forced first pick "
                "excluded); one optimizer update "
                "per batch; weights unchanged while "
                "all tours of the batch are "
                "generated",
        "optimizer": f"Adam, lr {LR0}, global-norm "
                     f"gradient clip {CLIP_NORM}, "
                     f"beta1 {ADAM_B1}, beta2 "
                     f"{ADAM_B2}, eps {ADAM_EPS}",
        "moving_genes": f"trainable (new copies) "
                        f"only ({len(TRAIN_IDX)}); "
                        f"frozen original copies "
                        f"({len(FROZEN_IDX)}) "
                        "bit-identical throughout",
        "guard": f"true-gap reads on the fixed "
                 f"validation set (10 cities) every "
                 f"{READ_EVERY} updates; "
                 f"best-on-validation weights are "
                 f"the working point; a worse read "
                 f"reverts to the best, resets "
                 f"Adam, and halves the lr (floor "
                 f"{LR_FLOOR})",
        "problems": f"fresh problems, n = "
                    f"{N_CITIES} cities, uniform on "
                    f"the unit square, seed stream "
                    f"{SEED_BASE} (update u uses "
                    f"batch seed {SEED_BASE} + u); "
                    "never the validation problems",
    },
    "stage1_widen": {
        "genome_file": "frankenstein_stepC_"
                       "widened003_genome.npz",
        "genes_total": N_GENES,
        "genes_frozen": int(len(FROZEN_IDX)),
        "genes_trainable": int(len(TRAIN_IDX)),
        "parity_gate_no_noise": WJ["parity_gate_no_noise"],
        "birth_gap_with_noise_pct": BIRTH_GAP_PCT,
        "pre_widen_subject_gap_pct": SUBJECT_GAP_PCT,
    },
    "sanity": {
        "widened_birth_gap_recomputed_pct": round(
            birth_gap * 100, 6),
        "stage1_recorded_birth_gap_pct": BIRTH_GAP_PCT,
        "passed": True,
        "jax_mirror_encoder_parity_max_abs_diff":
            parity,
        "jax_greedy_parity_n10_max_abs_diff":
            greedy_parity,
        "jax_greedy_parity_n20_max_abs_diff":
            greedy20_parity,
    },
    "training": {
        "updates": UPDATES,
        "problems_per_update": PROBLEMS_PER_UPDATE,
        "rollouts_per_problem": ROLLOUTS_PER_PROBLEM,
        "sampled_tours_per_update":
            PROBLEMS_PER_UPDATE * ROLLOUTS_PER_PROBLEM,
        "greedy_tours_per_update": PROBLEMS_PER_UPDATE,
        "seed_base": SEED_BASE,
        "reads": reads,
        "guard": {
            "reverts": state["reverts"],
            "halvings": state["halvings"],
            "final_lr": state["lr"],
            "events": guard_events,
        },
        "signal_stats_n20": stats,
        "best_validation_gap_pct": round(
            state["best_gap"] * 100, 4),
        "best_validation_update": state["best_update"],
        "improvement_vs_widened_birth_pct_points":
            round(BIRTH_GAP_PCT
                  - state["best_gap"] * 100, 4),
        "improvement_vs_subject_pct_points": round(
            SUBJECT_GAP_PCT
            - state["best_gap"] * 100, 4),
        "frozen_bit_identical_to_widened": frozen_ok,
        "trainable_genes_changed_vs_widened":
            n_train_changed,
        "champion_file": "engine_runs/"
                         "frankenstein_stepC_reinforce003/"
                         "frankenstein_stepC_reinforce003"
                         "_champion.npz",
        "wall_seconds": round(wall, 1),
    },
    "wall_seconds_total": round(time.time() - T0, 1),
}
with open(HERE / "frankenstein_stepC_reinforce003_"
                 "2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_stepC_reinforce003",
    "widened_birth_gap_pct": BIRTH_GAP_PCT,
    "subject_gap_pct": SUBJECT_GAP_PCT,
    "best_validation_gap_pct": round(
        state["best_gap"] * 100, 4),
    "best_validation_update": state["best_update"],
    "guard_reverts": state["reverts"],
    "guard_halvings": state["halvings"],
}, indent=1))
log("saved frankenstein_stepC_reinforce003_"
    "2026-11-26.json; COMPLETE marker written")
log(f"DONE stage 2 REINFORCE: best true gap "
    f"{state['best_gap']*100:.4f}% at update "
    f"{state['best_update']} (widened birth "
    f"{BIRTH_GAP_PCT:.4f}%, pre-widen subject "
    f"{SUBJECT_GAP_PCT:.4f}%)")
