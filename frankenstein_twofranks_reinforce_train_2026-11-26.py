#!/usr/bin/env python3
"""REINFORCE training on the Two-Franks stitch, exactly
as the sponsor specified (Roman, 2026-11-26), and a
test of his open question: at equal compute, are 4
stochastic draws on the SAME city set better than 1
draw on each of 4 DIFFERENT city sets?

THE METHOD (the sponsor's spec, followed exactly):
  - The network is deterministic. It produces a
    probability distribution over the next city at
    each step (softmax over the decoder logits
    exactly as the decoder produces them; NO
    temperature, NO soft surrogate anywhere).
  - Sampling is external, from the network's own
    distribution. Identical inputs can therefore
    produce different sampled tours without any
    weight change.
  - For each problem, the GREEDY tour (always the
    most probable city) is built first; its length
    is the BASELINE.
  - K stochastic tours are drawn for the problem
    under the SAME weights (weights do not change
    while any tour of the update batch is being
    generated). Each sampled tour's ADVANTAGE =
    baseline length - sampled length (positive
    when the sample beats the greedy tour).
  - Loss for one sampled tour = -advantage x (sum
    of the log-probabilities of the actions
    actually taken). The losses are averaged
    across the update batch, and ONE optimizer
    update is performed per batch.
  - The reward (-tour length, via the advantage)
    and the sampling need no gradient; gradients
    flow only through the log-probabilities.
  - Optimizer: Adam, learning rate 1e-4, global-
    norm gradient clip 1.0 (the same optimizer
    conditions as the corrected backprop run).
  - Trainable genes: ONLY the 1,106 stitching
    genes; the 356 backbone genes are frozen
    (gradient masked to zero), the same condition
    as both prior refinement runs on this subject,
    so all three runs stay comparable.
  - GUARD (from the corrected backprop run): TRUE
    gap read on the fixed validation set every 25
    updates; the best-on-validation weights are
    the working point — on a worse read, revert
    the weights to the best, reset Adam, halve the
    learning rate (floor 1e-6); every revert and
    halving is logged.

THE TEST — two arms, identical starting weights,
identical optimizer and guard, 400 updates each,
4 SAMPLED tours per update in both arms:
  - ARM A (draws): each update uses ONE fresh
    problem with K=4 draws on that same city set,
    plus its 1 greedy baseline. Loss = mean over
    the 4 tours.
  - ARM B (problems): each update uses FOUR fresh
    problems with 1 draw each, each problem with
    its own greedy baseline. Loss = mean over the
    4 tours.

EQUALIZATION DECISION (recorded plainly, as the
sponsor's question requires a stated rule): the
arms are equalized on SAMPLED tours per update
(= 4) and on updates (= 400), because the
sponsor's question is about the training signal
per sampled tour. Greedy baseline tours are
counted as overhead, not as part of the equalized
budget: arm A computes 1 greedy tour per update
(400 total), arm B computes 4 per update (1,600
total). Greedy tours are cheap deterministic
rollouts; both arms' wall-times are reported so
the overhead difference is visible.

Fresh problems: n = 10 cities, uniform on the
unit square, from recorded seed streams,
DIFFERENT streams per arm (arm A base seed
20261210, arm B base seed 20261220; update u uses
seed base + u), the same problem distribution;
never the validation problems.

SUBJECT (unchanged): the extended Two-Franks
champion (engine_runs/frankenstein_twofranks_
evolution/frankenstein_twofranks_extended_
champion.npz), true gap 7.3066% on the fixed
200-problem validation set (val_seed 20261116).
1,462 genes = 356 backbone + 1,106 stitching; the
mask comes from the stitch record
(frankenstein_twofranks_stitch_2026-11-26.json).
SANITY FIRST: the subject must reproduce 7.3066%
before either arm starts; the JAX greedy rollout
(the new baseline machinery) must reproduce the
numpy greedy lengths.

MACHINERY: the sampling, log-probability and
encoder code is verbatim from the verified probe
(frankenstein_twofranks_reinforce_probe_
2026-11-26.py); the guard and vaulting follow
frankenstein_twofranks_backprop2_2026-11-26.py.
The ONE new piece is the JAX greedy rollout used
for the baseline; it is parity-checked against
the numpy greedy decoder before training.

OUTPUTS (new files only):
  engine_runs/frankenstein_twofranks_reinforce/
    run.log, mask.json,
    frankenstein_twofranks_reinforceA_champion.npz,
    frankenstein_twofranks_reinforceB_champion.npz,
    COMPLETE
  frankenstein_twofranks_reinforce_2026-11-26.json

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_reinforce_train_2026-11-26.py \
    > engine_runs/frankenstein_twofranks_reinforce/launcher.log \
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
OUT = HERE / "engine_runs" / "frankenstein_twofranks_reinforce"
OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a", buffering=1)


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

N_CITIES = 10
UPDATES = 400                 # per arm
SAMPLED_PER_UPDATE = 4        # the equalized budget
LR0 = 1e-4
LR_FLOOR = 1e-6
CLIP_NORM = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
READ_EVERY = 25
BLOCKS = ("b0", "b1", "b2")

ARMS = {
    "A": {"problems_per_update": 1, "draws_per_problem": 4,
          "seed_base": 20261210,
          "label": "4 draws on the same city set"},
    "B": {"problems_per_update": 4, "draws_per_problem": 1,
          "seed_base": 20261220,
          "label": "1 draw on 4 different city sets"},
}

# ================================================================
# Setup: domain (validation set), scale, subject, mask
# (verbatim from the probe / backprop run 2)
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


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# ================================================================
# Numpy forward + greedy decode (verbatim from the
# probe) — the TRUE score, reporting only
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
# JAX mirror: encoder (verbatim from the probe)
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
# REINFORCE rollouts (sampling verbatim from the
# probe; the greedy rollout is the one new piece —
# the same loop with argmax in place of the
# categorical draw, used ONLY for the baseline
# length, which is a stop-gradient constant)
# ================================================================
def sample_tour_jax(W, x, key):
    """One sampled tour for ONE problem, from the
    network's OWN distribution (no temperature).
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
            jnp.int32)
        sum_logp = sum_logp + logp[pick]
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return picks, sum_logp, length


def greedy_tour_jax(W, x):
    """The greedy tour for ONE problem: always the
    most probable city. Returns (picks, length).
    The picks are integers, so no gradient flows
    through this rollout; its length is the
    BASELINE constant of the sponsor's spec."""
    Y = encode_jax(W, x)
    n = x.shape[0]
    D = jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                 .sum(axis=-1))
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
    return picks, length


def reinforce_loss(w_flat, Xb, keys):
    """REINFORCE loss with the GREEDY baseline (the
    sponsor's spec). Xb: (B, n, 2) fresh problems;
    keys: (B, K, 2), K sampled tours per problem,
    all generated under the SAME weights (w_flat).
    Per problem: greedy length = baseline;
    advantage_k = baseline - sampled_length_k
    (stop-gradient); loss contribution =
    -mean_k(advantage_k x sum_logp_k). Returns
    (loss, (sampled_lengths (B, K), greedy_lengths
    (B,), advantages (B, K)))."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x, ks):
        _, greedy_len = greedy_tour_jax(W, x)

        def per_sample(key):
            _, sum_logp, length = sample_tour_jax(W, x, key)
            return sum_logp, length
        sum_logps, lengths = jax.vmap(per_sample)(ks)
        adv = jax.lax.stop_gradient(greedy_len - lengths)
        return (-jnp.mean(adv * sum_logps), lengths,
                greedy_len, adv)

    losses, lengths, greedy_lens, advs = jax.vmap(
        per_problem)(Xb, keys)
    return jnp.mean(losses), (lengths, greedy_lens, advs)


vg = jax.jit(jax.value_and_grad(reinforce_loss, has_aux=True))


def make_batch(batch_seed, B, K):
    """Fresh training problems + sample keys for one
    update, deterministically from a seed. Problems
    are uniform on the unit square (never the
    validation problems); keys derive from the
    same seed."""
    rng = np.random.default_rng(batch_seed)
    Xb = rng.random((B, N_CITIES, 2))
    key = jax.random.PRNGKey(batch_seed)
    keys = jax.random.split(key, B * K).reshape(B, K, 2)
    return Xb, keys


# ================================================================
# SANITY + MACHINERY CHECKS (before either arm)
# ================================================================
subj_gap = mean_gap(w_subj, X_val, OPT_VAL)
log(f"sanity: subject true validation gap "
    f"{subj_gap*100:.6f}% vs recorded "
    f"{RECORDED_GAP_PCT:.4f}%")
if abs(subj_gap * 100 - RECORDED_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY FAILED. STOPPING; no training.")
    sys.exit(3)
log("sanity passed")

# JAX encoder mirror parity (verbatim check from
# the probe)
W_subj = unpack(w_subj)
Y_np = forward_general(W_subj, X_val[:8])
W_j = {k: jnp.asarray(W_subj[k], dtype=jnp.float64)
       for k in KEYS}
Y_jx = np.asarray(jax.vmap(
    lambda x: encode_jax(W_j, x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
parity = float(np.max(np.abs(Y_np - Y_jx)))
log(f"mirror check: JAX encoder vs numpy forward, "
    f"max abs diff {parity:.2e} on 8 validation problems")
if parity > 1e-9:
    log("MIRROR CHECK FAILED. STOPPING; no training.")
    sys.exit(3)
log("mirror check passed")

# NEW machinery check: the JAX greedy rollout (the
# baseline) must reproduce the numpy greedy lengths
np_greedy_lens, _ = true_lengths(w_subj, X_val[:8])
jx_greedy_lens = np.asarray(jax.vmap(
    lambda x: greedy_tour_jax(W_j, x)[1])(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
greedy_parity = float(np.max(np.abs(np_greedy_lens
                                    - jx_greedy_lens)))
log(f"greedy-baseline check: JAX greedy rollout vs "
    f"numpy greedy lengths, max abs diff "
    f"{greedy_parity:.2e} on 8 validation problems")
if greedy_parity > 1e-9:
    log("GREEDY-BASELINE CHECK FAILED. STOPPING; "
        "no training.")
    sys.exit(3)
log("greedy-baseline check passed")

atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_stitching": flat_mask.astype(int).tolist(),
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "subject": str(SUBJECT.relative_to(HERE)),
    "rule": "exact-from-a-Frank = backbone (frozen); "
            "drawn-as-filler = stitching (trainable)",
}, indent=1))


# ================================================================
# One arm of the test, with THE GUARD (pattern from
# backprop run 2)
# ================================================================
def run_arm(name, arm_cfg):
    B = arm_cfg["problems_per_update"]
    K = arm_cfg["draws_per_problem"]
    seed_base = arm_cfg["seed_base"]
    assert B * K == SAMPLED_PER_UPDATE
    log(f"ARM {name}: {arm_cfg['label']} — {B} "
        f"problem(s) x {K} draw(s) per update, "
        f"{SAMPLED_PER_UPDATE} sampled tours + {B} "
        f"greedy baseline tour(s) per update, "
        f"{UPDATES} updates, seed base {seed_base}, "
        f"Adam lr {LR0:.0e}, guard reads every "
        f"{READ_EVERY} updates")
    t_arm = time.time()

    w = w_subj.copy()
    adam = {"m": np.zeros(N_GENES), "v": np.zeros(N_GENES),
            "t": 0}
    state = {"lr": LR0, "best_gap": np.inf,
             "best_w": w.copy(), "best_update": -1,
             "reverts": 0, "halvings": 0}
    reads = []
    guard_events = []
    stat_lengths = []      # every sampled tour length
    stat_advantages = []   # every sampled tour advantage
    stat_greedy = []       # every greedy baseline length

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
            log(f"ARM {name} READ update {update}: "
                f"true validation gap {g*100:.4f}%  "
                f"<-- new best")
            return g
        # worse (or equal): revert to the best
        # working point, reset Adam, halve the lr
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
        log(f"ARM {name} READ update {update}: true "
            f"validation gap {g*100:.4f}% — worse "
            f"than best {state['best_gap']*100:.4f}% "
            f"(update {state['best_update']}); GUARD: "
            f"reverted to best weights, Adam reset, "
            f"lr -> {state['lr']:.2e} (revert "
            f"#{state['reverts']})")
        return g

    do_read(0)
    losses = []
    for u in range(1, UPDATES + 1):
        Xb, keys = make_batch(seed_base + u, B, K)
        (loss, (lengths, greedy_lens, advs)), grad = vg(
            jnp.asarray(w, dtype=jnp.float64),
            jnp.asarray(Xb, dtype=jnp.float64), keys)
        grad = np.asarray(grad, dtype=float)
        grad = np.where(flat_mask, grad, 0.0)  # backbone frozen
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
            log(f"ARM {name} update {u}: loss "
                f"{float(loss):.4f}, best so far "
                f"{state['best_gap']*100:.4f}% (update "
                f"{state['best_update']}), lr "
                f"{state['lr']:.2e}")

    wall = time.time() - t_arm
    best_w = state["best_w"]
    # frozen-gene integrity: the backbone must be
    # bit-identical to the subject's everywhere,
    # in the working weights and in the best
    backbone_ok = bool(
        (w[BACKBONE_IDX] == w_subj[BACKBONE_IDX]).all()
        and (best_w[BACKBONE_IDX]
             == w_subj[BACKBONE_IDX]).all())
    n_stitch_changed = int((best_w[STITCH_IDX]
                            != w_subj[STITCH_IDX]).sum())
    assert backbone_ok
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
    log(f"ARM {name} DONE in {wall:.1f}s: best true "
        f"gap {state['best_gap']*100:.4f}% at update "
        f"{state['best_update']} (start "
        f"{RECORDED_GAP_PCT:.4f}%); guard: "
        f"{state['reverts']} reverts, "
        f"{state['halvings']} halvings, final lr "
        f"{state['lr']:.2e}; signal: mean sampled "
        f"length {stats['mean_sampled_length']:.4f}, "
        f"mean advantage {stats['mean_advantage']:.4f}, "
        f"advantage spread (std) "
        f"{stats['advantage_spread_std']:.4f}, "
        f"{stats['frac_samples_beating_greedy']*100:.1f}% "
        f"of samples beat their greedy baseline; "
        f"backbone bit-identical: {backbone_ok}; "
        f"stitching genes changed in best: "
        f"{n_stitch_changed}")

    # vault this arm's best-on-validation weights
    champ = (OUT / f"frankenstein_twofranks_reinforce"
                   f"{name}_champion.npz")
    meta = {
        "kind": f"frankenstein_twofranks_reinforce{name}"
                "_champion",
        "date": "2026-11-26",
        "base": "frankenstein_twofranks_extended_"
                f"champion.npz (true gap "
                f"{RECORDED_GAP_PCT:.4f}%)",
        "method": "REINFORCE (sponsor's spec, "
                  "2026-11-26): tours sampled from "
                  "the network's own pointer "
                  "distribution (no temperature, no "
                  "surrogate); greedy tour length as "
                  "the per-problem baseline; "
                  "advantage = baseline - sampled "
                  "length; loss = -mean(advantage x "
                  "sum log-prob of sampled actions); "
                  f"Adam lr {LR0} with guard halving "
                  f"(floor {LR_FLOOR}), gradient clip "
                  f"norm {CLIP_NORM}; guard reads "
                  f"every {READ_EVERY} updates with "
                  "revert-to-best; stitching genes "
                  "only, backbone frozen",
        "arm": name,
        "arm_label": arm_cfg["label"],
        "problems_per_update": B,
        "draws_per_problem": K,
        "sampled_tours_per_update": SAMPLED_PER_UPDATE,
        "updates": UPDATES,
        "seed_base": seed_base,
        "best_validation_gap_pct": round(
            state["best_gap"] * 100, 4),
        "best_validation_update": state["best_update"],
        "guard_reverts": state["reverts"],
        "guard_halvings": state["halvings"],
        "final_lr": state["lr"],
        "signal_stats": stats,
        "gene_order": KEYS,
        "genes_total": N_GENES,
        "genes_backbone": int(len(BACKBONE_IDX)),
        "genes_stitching": int(len(STITCH_IDX)),
        "stitching_genes_changed_vs_subject":
            n_stitch_changed,
        "backbone_bit_identical_to_subject": backbone_ok,
        "attn_scale": SCALE, "ln_eps": EPS,
    }
    np.savez(champ, **{k: unpack(best_w)[k] for k in KEYS},
             meta=np.array(json.dumps(meta)))
    log(f"ARM {name}: vaulted {champ.name} (best gap "
        f"{state['best_gap']*100:.4f}% at update "
        f"{state['best_update']})")

    return {
        "arm": name,
        "label": arm_cfg["label"],
        "problems_per_update": B,
        "draws_per_problem": K,
        "sampled_tours_per_update": SAMPLED_PER_UPDATE,
        "greedy_tours_per_update": B,
        "greedy_tours_total": B * UPDATES,
        "updates": UPDATES,
        "seed_base": seed_base,
        "reads": reads,
        "guard": {
            "reverts": state["reverts"],
            "halvings": state["halvings"],
            "final_lr": state["lr"],
            "events": guard_events,
        },
        "signal_stats": stats,
        "best_validation_gap_pct": round(
            state["best_gap"] * 100, 4),
        "best_validation_update": state["best_update"],
        "improvement_vs_start_pct_points": round(
            RECORDED_GAP_PCT - state["best_gap"] * 100, 4),
        "backbone_bit_identical_to_subject": backbone_ok,
        "stitching_genes_changed_vs_subject":
            n_stitch_changed,
        "champion_file": f"engine_runs/"
                         f"frankenstein_twofranks_reinforce/"
                         f"frankenstein_twofranks_reinforce"
                         f"{name}_champion.npz",
        "wall_seconds": round(wall, 1),
    }


arm_results = {}
for _name in ("A", "B"):
    arm_results[_name] = run_arm(_name, ARMS[_name])

best_a = arm_results["A"]["best_validation_gap_pct"]
best_b = arm_results["B"]["best_validation_gap_pct"]
if best_a < best_b:
    winner, margin = "A", round(best_b - best_a, 4)
elif best_b < best_a:
    winner, margin = "B", round(best_a - best_b, 4)
else:
    winner, margin = "tie", 0.0
if winner == "A":
    answer = (f"4 draws on one city set (arm A) worked "
              f"better: its best true gap was "
              f"{best_a:.4f}% against {best_b:.4f}% for "
              f"1 draw on four different city sets "
              f"(arm B) — a margin of {margin:.4f} "
              f"percentage points at the same sampled-"
              f"tour budget (4 per update, 400 updates).")
elif winner == "B":
    answer = (f"1 draw on four different city sets "
              f"(arm B) worked better: its best true "
              f"gap was {best_b:.4f}% against "
              f"{best_a:.4f}% for 4 draws on one city "
              f"set (arm A) — a margin of {margin:.4f} "
              f"percentage points at the same sampled-"
              f"tour budget (4 per update, 400 updates).")
else:
    answer = (f"Neither arrangement won: both arms "
              f"reached the same best true gap of "
              f"{best_a:.4f}% at the same sampled-tour "
              f"budget (4 per update, 400 updates).")

record = {
    "run": "REINFORCE training on the Two-Franks "
           "stitch (sponsor's spec) — arm A (4 draws "
           "on one city set) vs arm B (1 draw on four "
           "different city sets)",
    "date": "2026-11-26",
    "sponsor_direction": "Roman, 2026-11-26: greedy "
                         "baseline with a configurable "
                         "number of stochastic draws; "
                         "weights unchanged while "
                         "generating all the tours, "
                         "losses averaged, one optimizer "
                         "update; and his open question — "
                         "four draws on the same city set "
                         "are not necessarily better than "
                         "one draw on four different city "
                         "sets for the same budget; test "
                         "and see what works better",
    "method": {
        "name": "REINFORCE (policy gradient) with a "
                "greedy baseline, exactly as the "
                "sponsor specified",
        "sampling": "external sampling from the "
                    "network's own pointer "
                    "distribution (softmax over the "
                    "decoder logits as produced; no "
                    "temperature, no soft surrogate)",
        "baseline": "the problem's own greedy tour "
                    "length (always the most probable "
                    "city)",
        "advantage": "baseline length - sampled tour "
                     "length (positive when the sample "
                     "beats the greedy tour)",
        "loss": "-mean over the update batch of "
                "advantage x (sum of log-probabilities "
                "of the actions actually taken); one "
                "optimizer update per batch; weights "
                "unchanged while all tours of the "
                "batch are generated",
        "optimizer": f"Adam, lr {LR0}, global-norm "
                     f"gradient clip {CLIP_NORM}, "
                     f"beta1 {ADAM_B1}, beta2 {ADAM_B2}, "
                     f"eps {ADAM_EPS}",
        "moving_genes": f"stitching only "
                        f"({len(STITCH_IDX)}); backbone "
                        f"frozen ({len(BACKBONE_IDX)}), "
                        "the same condition as both "
                        "prior refinement runs on this "
                        "subject",
        "guard": f"true-gap reads on the fixed "
                 f"validation set every {READ_EVERY} "
                 f"updates; best-on-validation weights "
                 f"are the working point; a worse read "
                 f"reverts to the best, resets Adam, "
                 f"and halves the lr (floor {LR_FLOOR})",
        "problems": f"fresh problems, n = {N_CITIES} "
                    "cities, uniform on the unit "
                    "square, recorded seed streams, "
                    "different per arm, same "
                    "distribution; never the validation "
                    "problems",
    },
    "equalization_decision": "The arms are equalized "
        "on SAMPLED tours per update (= 4) and on "
        "updates (= 400), because the sponsor's "
        "question is about the training signal per "
        "sampled tour. Greedy baseline tours are "
        "counted as overhead, not as part of the "
        "equalized budget: arm A computes 1 greedy "
        "tour per update (400 total); arm B computes "
        "4 per update (1,600 total). Greedy tours are "
        "cheap deterministic rollouts; both arms' "
        "wall-times are reported so the overhead "
        "difference is visible.",
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
        "passed": True,
        "jax_mirror_encoder_parity_max_abs_diff": parity,
        "jax_greedy_rollout_parity_max_abs_diff":
            greedy_parity,
    },
    "arms": arm_results,
    "comparisons_pct": {
        "start_subject": RECORDED_GAP_PCT,
        "frank1_parent": 6.8959,
        "frank2_parent": 7.017,
        "arm_A_best": best_a,
        "arm_B_best": best_b,
    },
    "answer_to_sponsor_question": answer,
    "wall_seconds_total": round(time.time() - T0, 1),
}
with open(HERE / "frankenstein_twofranks_reinforce_"
                 "2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_twofranks_reinforce_train",
    "subject_gap_pct": RECORDED_GAP_PCT,
    "arm_A_best_gap_pct": best_a,
    "arm_B_best_gap_pct": best_b,
    "answer": answer,
}, indent=1))
log(f"saved frankenstein_twofranks_reinforce_"
    f"2026-11-26.json; COMPLETE marker written")
log(f"DONE: arm A best {best_a:.4f}%, arm B best "
    f"{best_b:.4f}% (start {RECORDED_GAP_PCT:.4f}%, "
    f"Frank #1 6.8959%, Frank #2 7.017%)")
log(f"ANSWER: {answer}")
