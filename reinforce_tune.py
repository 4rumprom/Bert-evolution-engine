#!/usr/bin/env python3
"""Parameterized REINFORCE trainer for the tuning mission
(2026-11-26, sponsor: "keep playing with parameters until it
gives us something; look for tips and tricks in the literature").

Base machinery is the reference run
frankenstein_stepC_reinforce003_lr20_2026-11-26.py, unchanged:
the widened Step C Frankenstein
(frankenstein_stepC_widened003_genome.npz, 8,480 genes = 1,440
frozen original copies + 7,040 trainable new copies, born
8.7828% on the fixed 200-problem 10-city validation set),
REINFORCE with one sampled path per starting city, first pick
forced with its log-prob excluded, weights frozen during batch
generation, one Adam update per batch, gradient clip 1.0,
validation reads every 25 updates.

Knobs (one family at a time, staged in
reinforce_tuning_log_2026-11-26.md):
  --baseline greedy|pomo|loo   greedy = the problem's own greedy
      tour length (reference). pomo = POMO shared baseline, the
      mean rollout length on the same problem (Kwon et al.
      2020). loo = leave-one-out mean of the other rollouts
      (Kool et al. 2019, "Buy 4 REINFORCE Samples").
  --adv-norm 0|1   divide each problem's advantages by the
      std of its rollout lengths (group standardization).
  --temp FLOAT     multiplier on the decoder logits for
      SAMPLING and for the loss's log-probs (1.0 = reference).
  --entropy FLOAT  entropy bonus coefficient: loss -=
      coef * mean step entropy of the sampling distribution.
  --lr FLOAT, --updates INT
  --guard strict|soft3   strict = reference guard (revert +
      Adam reset + halve lr on every worse read). soft3 = never
      revert; halve lr only after 3 consecutive worse reads;
      best-on-validation weights always snapshotted.
  --curriculum fixed20|step|mix   step = 10 cities for the
      first half of updates, 20 after; mix = each update's
      problem size drawn from {10, 15, 20}.
  --problems INT   problems per update (rollouts per problem
      = the problem's city count, one per starting city).
  --seed-base INT  batch seed stream (reference: 20261241).

Outputs: engine_runs/reinforce_tuning/<name>.json and
<name>_champion.npz.
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

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--baseline", default="greedy",
                choices=["greedy", "pomo", "loo"])
ap.add_argument("--adv-norm", type=int, default=0)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--entropy", type=float, default=0.0)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--updates", type=int, default=400)
ap.add_argument("--guard", default="strict",
                choices=["strict", "soft3", "restart"])
ap.add_argument("--curriculum", default="fixed20",
                choices=["fixed20", "fixed10", "step", "mix"])
ap.add_argument("--read-every", type=int, default=25)
ap.add_argument("--problems", type=int, default=4)
ap.add_argument("--seed-base", type=int, default=20261241)
ARGS = ap.parse_args()

BASELINE = ARGS.baseline
ADV_NORM = bool(ARGS.adv_norm)
TEMP = float(ARGS.temp)
ENT = float(ARGS.entropy)
LR0 = float(ARGS.lr)
UPDATES = int(ARGS.updates)
GUARD = ARGS.guard
CURRICULUM = ARGS.curriculum
PROBLEMS_PER_UPDATE = int(ARGS.problems)
SEED_BASE = int(ARGS.seed_base)
LR_FLOOR = LR0 * 0.01
CLIP_NORM = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
READ_EVERY = int(ARGS.read_every)
VAL_CITIES = 10
BLOCKS = ("b0", "b1", "b2", "b3", "b4", "b5w")
HEADS = (0, 1, 2, 3)
SUBJECT_GAP_PCT = 8.1018
PRIOR_BEST_PCT = 8.5963

T0 = time.time()
OUT = HERE / "engine_runs" / "reinforce_tuning"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


# ================================================================
# Setup: domain, widened genome, mask (verbatim from reference)
# ================================================================
CFG = HERE / "bert_engine_config_deep50_stepA.json"
WIDENED = HERE / "frankenstein_stepC_widened003_genome.npz"
WIDEN_JSON = HERE / "frankenstein_stepC_widen003_2026-11-26.json"

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
assert N_GENES == 8480 and len(FROZEN_IDX) == 1440 \
    and len(TRAIN_IDX) == 7040
SCALE = float(WMETA["attn_scale"])
w_widened = np.concatenate([WID_W[k].ravel() for k in KEYS])

WJ = json.load(open(WIDEN_JSON))
BIRTH_GAP_PCT = float(WJ["birth_gap_with_noise"]["gap_pct"])
log(f"run {ARGS.name}: baseline={BASELINE} adv_norm={ADV_NORM} "
    f"temp={TEMP} entropy={ENT} lr={LR0:.1e} updates={UPDATES} "
    f"guard={GUARD} curriculum={CURRICULUM} "
    f"problems/update={PROBLEMS_PER_UPDATE} seed={SEED_BASE}")
log(f"subject: {N_GENES} genes ({len(FROZEN_IDX)} frozen + "
    f"{len(TRAIN_IDX)} trainable); birth gap "
    f"{BIRTH_GAP_PCT:.4f}%")


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k])
            for k in KEYS}


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


# ================================================================
# Numpy forward + greedy decode (verbatim from reference)
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
# JAX mirror (verbatim from reference)
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


def _dist(x):
    return jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                    .sum(axis=-1))


def greedy_from_Y(W, Y, x):
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
    """One sampled tour, first pick FORCED to `start`
    (excluded from sum_logp and from the entropy mean).
    Sampling and log-probs use the decoder logits
    multiplied by TEMP (TEMP = 1.0 is the reference).
    Returns (sum_logp, length, mean_step_entropy)."""
    n = x.shape[0]
    D = _dist(x)
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n).at[start].set(1.0)
    picks = jnp.zeros(n, dtype=jnp.int32).at[0].set(start)
    sum_logp = 0.0
    sum_ent = 0.0
    last = Y[start]
    for t in range(1, n):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
        logits = logits - 1e9 * visited
        logits = logits * TEMP
        logp = jax.nn.log_softmax(logits)
        probs = jnp.exp(logp)
        sum_ent = sum_ent - jnp.sum(probs * logp)
        key, sub = jax.random.split(key)
        pick = jax.random.categorical(sub, logits).astype(
            jnp.int32)
        sum_logp = sum_logp + logp[pick]
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return sum_logp, length, sum_ent / max(n - 1, 1)


def reinforce_loss(w_flat, Xb, keys):
    """Loss with the configured baseline. Per problem:
    rollouts = one per starting city. Advantage_k =
    baseline - length_k (positive = better than
    baseline), baseline per BASELINE; optional group
    standardization (ADV_NORM). Loss per problem =
    -mean_k(adv_k * sum_logp_k) - ENT * mean_k(ent_k);
    batch loss = mean over problems. Aux: (lengths,
    greedy_lengths, advantages, mean_entropy)."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x, ks):
        Y = encode_jax(W, x)
        greedy_len = greedy_from_Y(W, Y, x)
        starts = jnp.arange(x.shape[0], dtype=jnp.int32)

        def per_rollout(start, key):
            return sample_from_Y(W, Y, x, start, key)
        sum_logps, lengths, ents = jax.vmap(per_rollout)(
            starts, ks)
        k = lengths.shape[0]
        if BASELINE == "greedy":
            base = jnp.full_like(lengths, greedy_len)
        elif BASELINE == "pomo":
            base = jnp.full_like(lengths, jnp.mean(lengths))
        else:  # loo
            base = (jnp.sum(lengths) - lengths) / (k - 1)
        adv = jax.lax.stop_gradient(base - lengths)
        if ADV_NORM:
            adv = adv / (jnp.std(lengths) + 1e-8)
        loss = -jnp.mean(adv * sum_logps) \
            - ENT * jnp.mean(ents)
        return loss, (lengths, greedy_len, adv,
                      jnp.mean(ents))

    losses, aux = jax.vmap(per_problem)(Xb, keys)
    lengths, greedy_lens, advs, ents = aux
    return jnp.mean(losses), (lengths, greedy_lens, advs,
                              ents)


vg = jax.jit(jax.value_and_grad(reinforce_loss, has_aux=True))


def size_for_update(u):
    if CURRICULUM == "fixed20":
        return 20
    if CURRICULUM == "fixed10":
        return 10
    if CURRICULUM == "step":
        return 10 if u <= UPDATES // 2 else 20
    raise ValueError(CURRICULUM)


def make_batch(batch_seed, u):
    """Fresh problems + rollout keys for one update.
    mix curriculum: the problem size is drawn from
    {10, 15, 20} by the batch rng (homogeneous within
    the update). Rollouts per problem = city count."""
    rng = np.random.default_rng(batch_seed)
    if CURRICULUM == "mix":
        n = int(rng.choice([10, 15, 20]))
    else:
        n = size_for_update(u)
    Xb = rng.random((PROBLEMS_PER_UPDATE, n, 2))
    key = jax.random.PRNGKey(batch_seed)
    keys = jax.random.split(
        key, PROBLEMS_PER_UPDATE * n).reshape(
        PROBLEMS_PER_UPDATE, n, 2)
    return Xb, keys, n


# ================================================================
# SANITY + MACHINERY CHECKS (from the reference)
# ================================================================
birth_gap = mean_gap(w_widened, X_val, OPT_VAL)
log(f"sanity: widened founder true validation gap "
    f"{birth_gap*100:.6f}% vs recorded {BIRTH_GAP_PCT:.4f}%")
if abs(birth_gap * 100 - BIRTH_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY FAILED. STOPPING; no training.")
    sys.exit(3)

W_subj = unpack(w_widened)
Y_np = forward_wide(W_subj, X_val[:8])
W_j = {k: jnp.asarray(W_subj[k], dtype=jnp.float64)
       for k in KEYS}
Y_jx = np.asarray(jax.vmap(
    lambda x: encode_jax(W_j, x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
parity = float(np.max(np.abs(Y_np - Y_jx)))
if parity > 1e-9:
    log(f"MIRROR CHECK FAILED ({parity:.2e}). STOPPING.")
    sys.exit(3)
np_greedy_lens, _ = true_lengths(w_widened, X_val[:8])
jx_greedy_lens = np.asarray(jax.vmap(
    lambda x: greedy_from_Y(W_j, encode_jax(W_j, x), x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
greedy_parity = float(np.max(np.abs(np_greedy_lens
                                    - jx_greedy_lens)))
if greedy_parity > 1e-9:
    log(f"GREEDY CHECK FAILED ({greedy_parity:.2e}). "
        "STOPPING.")
    sys.exit(3)
log(f"sanity + mirror + greedy parity passed "
    f"(encoder {parity:.2e}, greedy {greedy_parity:.2e})")

# ================================================================
# Training with the configured guard
# ================================================================
log(f"TRAINING {ARGS.name}: {UPDATES} updates, "
    f"{PROBLEMS_PER_UPDATE} problems/update, baseline "
    f"{BASELINE}, lr {LR0:.1e} (floor {LR_FLOOR:.0e}), "
    f"guard {GUARD}")
t_train = time.time()

w = w_widened.copy()
adam = {"m": np.zeros(N_GENES), "v": np.zeros(N_GENES),
        "t": 0}
state = {"lr": LR0, "best_gap": np.inf,
         "best_w": w.copy(), "best_update": -1,
         "reverts": 0, "halvings": 0, "consec_worse": 0}
reads = []
guard_events = []
stat_lengths = []
stat_advantages = []
stat_greedy = []
stat_entropy = []
losses = []


def do_read(update):
    g = mean_gap(w, X_val, OPT_VAL)
    if g < state["best_gap"]:
        state["best_gap"] = g
        state["best_w"] = w.copy()
        state["best_update"] = update
        state["consec_worse"] = 0
        action_best = "none_new_best"
        if GUARD == "restart" and update > 0:
            # fresh excursion budget from the new peak:
            # restore the full step size
            state["lr"] = LR0
            action_best = "new_best_lr_restored"
        reads.append({"update": update,
                      "gap_pct": round(g * 100, 4),
                      "is_best": True,
                      "guard_action": action_best,
                      "lr_after": state["lr"]})
        log(f"READ {update}: gap {g*100:.4f}% <-- new best")
        return g
    if GUARD in ("strict", "restart"):
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
        action = "revert_to_best" + (
            "_and_halve_lr" if halved else "_lr_at_floor")
    else:  # soft3: no revert; halve only on 3rd consecutive
        state["consec_worse"] += 1
        state["reverts"] += 0  # soft guard never reverts
        if state["consec_worse"] >= 3:
            new_lr = max(state["lr"] / 2.0, LR_FLOOR)
            halved = bool(new_lr < state["lr"])
            state["lr"] = new_lr
            if halved:
                state["halvings"] += 1
            state["consec_worse"] = 0
            action = "soft3_halve_lr" if halved \
                else "soft3_lr_at_floor"
        else:
            action = f"soft3_tolerate_" \
                     f"{state['consec_worse']}"
    ev = {"update": update, "gap_pct": round(g * 100, 4),
          "action": action, "lr_after": state["lr"],
          "best_gap_pct": round(state["best_gap"] * 100, 4),
          "best_update": state["best_update"]}
    guard_events.append(ev)
    reads.append({"update": update,
                  "gap_pct": round(g * 100, 4),
                  "is_best": False,
                  "guard_action": action,
                  "lr_after": state["lr"]})
    log(f"READ {update}: gap {g*100:.4f}% (best "
        f"{state['best_gap']*100:.4f}% @ "
        f"{state['best_update']}); guard: {action}, "
        f"lr {state['lr']:.2e}")
    return g


do_read(0)
for u in range(1, UPDATES + 1):
    Xb, keys, n_b = make_batch(SEED_BASE + u, u)
    (loss, (lengths, greedy_lens, advs, ents)), grad = vg(
        jnp.asarray(w, dtype=jnp.float64),
        jnp.asarray(Xb, dtype=jnp.float64), keys)
    grad = np.asarray(grad, dtype=float)
    grad = np.where(train_mask, grad, 0.0)
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
    stat_entropy.extend(
        np.asarray(ents, dtype=float).ravel().tolist())
    if u % READ_EVERY == 0:
        do_read(u)

wall = time.time() - t_train
best_w = state["best_w"]
frozen_ok = bool(
    (w[FROZEN_IDX] == w_widened[FROZEN_IDX]).all()
    and (best_w[FROZEN_IDX]
         == w_widened[FROZEN_IDX]).all())
n_train_changed = int((best_w[TRAIN_IDX]
                       != w_widened[TRAIN_IDX]).sum())
assert frozen_ok, "FROZEN GENES CHANGED"
la = np.asarray(stat_lengths)
aa = np.asarray(stat_advantages)
ga = np.asarray(stat_greedy)
ea = np.asarray(stat_entropy)
# fraction of rollouts beating greedy needs length<greedy
# per problem; recompute from paired records: stats were
# appended rollout-major per problem, greedy repeated per
# problem in order; lengths and greedy arrays are aligned
# per rollout only within fixed-size runs, so compute the
# paired fraction directly from lengths vs greedy is not
# well-defined here; use advantage-based diagnostics
# plus the paired fraction reconstructed per update is
# skipped. Report advantage sign fraction instead.
stats = {
    "n_sampled_tours": int(la.size),
    "mean_sampled_length": round(float(la.mean()), 4),
    "std_sampled_length": round(float(la.std()), 4),
    "mean_greedy_length": round(float(ga.mean()), 4),
    "mean_advantage": round(float(aa.mean()), 4),
    "advantage_std": round(float(aa.std()), 4),
    "advantage_min": round(float(aa.min()), 4),
    "advantage_max": round(float(aa.max()), 4),
    "frac_rollouts_positive_advantage": round(
        float((aa > 0).mean()), 4),
    "mean_step_entropy": round(float(ea.mean()), 4),
    "mean_loss_first_25": round(
        float(np.mean(losses[:25])), 4),
    "mean_loss_last_25": round(
        float(np.mean(losses[-25:])), 4),
}
best_pct = round(state["best_gap"] * 100, 4)
log(f"DONE {ARGS.name} in {wall:.1f}s: best {best_pct:.4f}% "
    f"@ update {state['best_update']} (birth "
    f"{BIRTH_GAP_PCT:.4f}%, win bar {SUBJECT_GAP_PCT}%, "
    f"prior best {PRIOR_BEST_PCT}%); guard: "
    f"{state['reverts']} reverts, {state['halvings']} "
    f"halvings, final lr {state['lr']:.2e}; frac "
    f"positive advantage "
    f"{stats['frac_rollouts_positive_advantage']:.2f}; "
    f"frozen ok {frozen_ok}")

# ================================================================
# Vault champion + record
# ================================================================
champ = OUT / f"{ARGS.name}_champion.npz"
meta = {
    "kind": "reinforce_tuning_champion",
    "run": ARGS.name,
    "date": "2026-11-26",
    "base": "frankenstein_stepC_widened003_genome.npz",
    "config": {
        "baseline": BASELINE, "adv_norm": ADV_NORM,
        "temp": TEMP, "entropy": ENT, "lr": LR0,
        "lr_floor": LR_FLOOR, "updates": UPDATES,
        "guard": GUARD, "curriculum": CURRICULUM,
        "problems_per_update": PROBLEMS_PER_UPDATE,
        "read_every": READ_EVERY,
        "seed_base": SEED_BASE,
    },
    "best_validation_gap_pct": best_pct,
    "best_validation_update": state["best_update"],
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_frozen": int(len(FROZEN_IDX)),
    "genes_trainable": int(len(TRAIN_IDX)),
    "trainable_genes_changed_vs_widened": n_train_changed,
    "frozen_bit_identical_to_widened": frozen_ok,
    "attn_scale": SCALE, "ln_eps": EPS,
}
atomic_savez(champ, **{k: unpack(best_w)[k] for k in KEYS},
             trainable_mask=train_mask.astype(np.int8),
             meta=np.array(json.dumps(meta)))

record = {
    "run": ARGS.name,
    "date": "2026-11-26",
    "config": meta["config"],
    "birth_gap_pct": BIRTH_GAP_PCT,
    "win_bar_pct": SUBJECT_GAP_PCT,
    "prior_best_pct": PRIOR_BEST_PCT,
    "sanity": {"parity_encoder": parity,
               "parity_greedy_n10": greedy_parity},
    "reads": reads,
    "guard": {"reverts": state["reverts"],
              "halvings": state["halvings"],
              "final_lr": state["lr"],
              "events": guard_events},
    "signal_stats": stats,
    "best_validation_gap_pct": best_pct,
    "best_validation_update": state["best_update"],
    "beats_prior_best": bool(best_pct < PRIOR_BEST_PCT),
    "beats_win_bar": bool(best_pct < SUBJECT_GAP_PCT),
    "frozen_bit_identical_to_widened": frozen_ok,
    "trainable_genes_changed_vs_widened": n_train_changed,
    "champion_file": f"engine_runs/reinforce_tuning/"
                     f"{ARGS.name}_champion.npz",
    "wall_seconds": round(wall, 1),
}
with open(OUT / f"{ARGS.name}.json", "w") as f:
    json.dump(record, f, indent=1)
log(f"saved {ARGS.name}.json + champion; best "
    f"{best_pct:.4f}%")
