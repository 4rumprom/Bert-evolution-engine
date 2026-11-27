#!/usr/bin/env python3
"""REINFORCE training on the Two-Franks stitch —
TRAINING PROBLEM SIZE test (sponsor direction,
2026-11-26): does training on SMALLER or LARGER
problems than the 10-city target help or hurt?

The winning recipe of the previous test (arm B:
1 stochastic draw on each of 4 different city
sets per update, greedy baseline) is kept EXACTLY.
Only the size of the TRAINING problems changes:
  - ARM B5:  training problems have 5 cities.
  - ARM B20: training problems have 20 cities.
Both arms are judged where it counts: the TRUE
gap on the fixed 10-city validation set (200
problems, val_seed 20261116), read by the guard
every 25 updates.

Reference points (already measured in
frankenstein_twofranks_reinforce_2026-11-26.json;
NOT rerun here): arm B at 10 cities reached a
best true gap of 7.2830% (update 175); the start
(the subject itself) is 7.3066%; the parents are
Frank #1 6.8959% and Frank #2 7.017%.

THE METHOD (identical to arm B of the prior test):
  - The network is deterministic; it produces a
    probability distribution over the next city
    at each step (softmax over the decoder logits
    exactly as produced; NO temperature, NO soft
    surrogate anywhere). Sampling is external,
    from the network's own distribution.
  - The network is size-agnostic: attention runs
    over however many tokens the problem has, and
    the pointer decoder runs until every city is
    visited. Training on 5- or 20-city problems
    is therefore well-defined; the SANITY step
    below verifies tours generate cleanly at both
    sizes before any training.
  - For each problem, the GREEDY tour (always the
    most probable city) is built first; its length
    is the BASELINE. 1 stochastic tour is drawn
    per problem under the SAME weights (weights
    do not change while any tour of the update
    batch is generated). ADVANTAGE = baseline
    length - sampled length.
  - Loss for one sampled tour = -advantage x
    (sum of the log-probabilities of the actions
    actually taken). Losses are averaged across
    the 4 problems of the update; ONE Adam update
    per batch (lr 1e-4, global-norm clip 1.0).
  - Trainable genes: ONLY the 1,106 stitching
    genes; the 356 backbone genes are frozen
    (gradient masked to zero) — the same condition
    as every prior refinement run on this subject.
  - GUARD (from the corrected backprop run):
    TRUE gap read on the fixed 10-city validation
    set every 25 updates; the best-on-validation
    weights are the working point — on a worse
    read, revert to the best, reset Adam, halve
    the learning rate (floor 1e-6); every revert
    and halving is logged.

Fresh problems: uniform on the unit square, from
recorded seed streams, different per arm (arm B5
base seed 20261230, arm B20 base seed 20261240;
update u uses seed base + u); never the
validation problems.

SUBJECT (unchanged): the extended Two-Franks
champion (engine_runs/frankenstein_twofranks_
evolution/frankenstein_twofranks_extended_
champion.npz), true gap 7.3066% on the fixed
200-problem validation set. 1,462 genes = 356
backbone + 1,106 stitching; the mask comes from
the stitch record (frankenstein_twofranks_
stitch_2026-11-26.json).

MACHINERY: verbatim from the verified REINFORCE
trainer (frankenstein_twofranks_reinforce_train_
2026-11-26.py); the only change is that the
training problem size is a per-arm parameter
(the rollouts already take n from the input).

OUTPUTS (new files only):
  engine_runs/frankenstein_twofranks_reinforce_sizes/
    run.log, mask.json,
    frankenstein_twofranks_reinforceB5_champion.npz,
    frankenstein_twofranks_reinforceB20_champion.npz,
    COMPLETE
  frankenstein_twofranks_reinforce_sizes_2026-11-26.json

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_reinforce_sizes_2026-11-26.py \
    > engine_runs/frankenstein_twofranks_reinforce_sizes/launcher.log \
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
OUT = HERE / "engine_runs" / "frankenstein_twofranks_reinforce_sizes"
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

VAL_CITIES = 10               # the fixed validation set
UPDATES = 400                 # per arm
SAMPLED_PER_UPDATE = 4        # arm B recipe: 4 problems x 1 draw
LR0 = 1e-4
LR_FLOOR = 1e-6
CLIP_NORM = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
READ_EVERY = 25
BLOCKS = ("b0", "b1", "b2")

ARMS = {
    "B5": {"problems_per_update": 4, "draws_per_problem": 1,
           "n_cities": 5, "seed_base": 20261230,
           "label": "arm B recipe, 5-city training problems"},
    "B20": {"problems_per_update": 4, "draws_per_problem": 1,
            "n_cities": 20, "seed_base": 20261240,
            "label": "arm B recipe, 20-city training problems"},
}

# Reference points from the prior test (NOT rerun)
REFERENCE = {
    "arm_B10_best_gap_pct": 7.283,
    "arm_B10_best_update": 175,
    "arm_B10_frac_samples_beating_greedy": 0.0006,
    "start_gap_pct": 7.3066,
    "frank1_parent_gap_pct": 6.8959,
    "frank2_parent_gap_pct": 7.017,
}

# ================================================================
# Setup: domain (validation set), scale, subject, mask
# (verbatim from the verified trainer)
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
    f"validation n_cities = {VAL_CITIES}")
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
# verified trainer) — the TRUE score, reporting only
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
# JAX mirror: encoder (verbatim from the verified
# trainer)
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
# REINFORCE rollouts (verbatim from the verified
# trainer; n comes from the input, so both arms'
# sizes run through the same code)
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
    BASELINE constant."""
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
    """REINFORCE loss with the GREEDY baseline.
    Xb: (B, n, 2) fresh problems (n = the arm's
    training size); keys: (B, K, 2). Per problem:
    greedy length = baseline; advantage_k =
    baseline - sampled_length_k (stop-gradient);
    loss contribution = -mean_k(advantage_k x
    sum_logp_k). Returns (loss, (sampled_lengths
    (B, K), greedy_lengths (B,), advantages
    (B, K)))."""
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


def make_batch(batch_seed, B, K, n_cities):
    """Fresh training problems + sample keys for one
    update, deterministically from a seed. Problems
    are uniform on the unit square (never the
    validation problems); keys derive from the
    same seed."""
    rng = np.random.default_rng(batch_seed)
    Xb = rng.random((B, n_cities, 2))
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

W_subj = unpack(w_subj)
W_j = {k: jnp.asarray(W_subj[k], dtype=jnp.float64)
       for k in KEYS}

# JAX encoder mirror parity on the validation size
Y_np = forward_general(W_subj, X_val[:8])
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

# SIZE SANITY: the network is size-agnostic — tours
# (greedy AND sampled) must generate cleanly at
# n = 5 and n = 20 under the subject weights, and
# the JAX greedy rollout must reproduce the numpy
# greedy lengths at both sizes.
size_sanity = {}
for n_sz in (5, 20):
    rng_sz = np.random.default_rng(20261250 + n_sz)
    X_sz = rng_sz.random((8, n_sz, 2))
    np_lens, _ = true_lengths(w_subj, X_sz)
    jx_greedy = np.asarray(jax.vmap(
        lambda x: greedy_tour_jax(W_j, x)[1])(
        jnp.asarray(X_sz, dtype=jnp.float64)))
    keys_sz = jax.random.split(
        jax.random.PRNGKey(20261260 + n_sz), 8)
    jx_sampled = np.asarray(jax.vmap(
        lambda x, k: sample_tour_jax(W_j, x, k)[2])(
        jnp.asarray(X_sz, dtype=jnp.float64), keys_sz))
    finite_ok = bool(np.isfinite(np_lens).all()
                     and np.isfinite(jx_greedy).all()
                     and np.isfinite(jx_sampled).all())
    gpar = float(np.max(np.abs(np_lens - jx_greedy)))
    size_sanity[f"n{n_sz}"] = {
        "numpy_greedy_lengths_mean": round(
            float(np_lens.mean()), 4),
        "jax_greedy_lengths_mean": round(
            float(jx_greedy.mean()), 4),
        "jax_sampled_lengths_mean": round(
            float(jx_sampled.mean()), 4),
        "greedy_parity_max_abs_diff": gpar,
        "all_lengths_finite": finite_ok,
    }
    log(f"size sanity n={n_sz}: numpy greedy mean "
        f"{np_lens.mean():.4f}, JAX greedy mean "
        f"{jx_greedy.mean():.4f} (parity {gpar:.2e}), "
        f"JAX sampled mean {jx_sampled.mean():.4f}, "
        f"all finite: {finite_ok}")
    if not finite_ok or gpar > 1e-9:
        log(f"SIZE SANITY FAILED at n={n_sz}. "
            f"STOPPING; no training.")
        sys.exit(3)
log("size sanity passed: tours generate cleanly "
    "at n=5 and n=20")

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
# One arm, with THE GUARD (pattern verbatim from
# the verified trainer)
# ================================================================
def run_arm(name, arm_cfg):
    B = arm_cfg["problems_per_update"]
    K = arm_cfg["draws_per_problem"]
    n_cities = arm_cfg["n_cities"]
    seed_base = arm_cfg["seed_base"]
    assert B * K == SAMPLED_PER_UPDATE
    log(f"ARM {name}: {arm_cfg['label']} — {B} "
        f"problem(s) of {n_cities} cities x {K} "
        f"draw(s) per update, {SAMPLED_PER_UPDATE} "
        f"sampled tours + {B} greedy baseline "
        f"tour(s) per update, {UPDATES} updates, "
        f"seed base {seed_base}, Adam lr {LR0:.0e}, "
        f"guard reads every {READ_EVERY} updates "
        f"on the fixed 10-city validation set")
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
        Xb, keys = make_batch(seed_base + u, B, K, n_cities)
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
        f"mean greedy length "
        f"{stats['mean_greedy_length']:.4f}, mean "
        f"advantage {stats['mean_advantage']:.4f}, "
        f"advantage spread (std) "
        f"{stats['advantage_spread_std']:.4f}, "
        f"{stats['frac_samples_beating_greedy']*100:.2f}% "
        f"of samples beat their greedy baseline; "
        f"backbone bit-identical: {backbone_ok}; "
        f"stitching genes changed in best: "
        f"{n_stitch_changed}")

    champ = (OUT / f"frankenstein_twofranks_reinforce"
                   f"{name}_champion.npz")
    meta = {
        "kind": f"frankenstein_twofranks_reinforce{name}"
                "_champion",
        "date": "2026-11-26",
        "base": "frankenstein_twofranks_extended_"
                f"champion.npz (true gap "
                f"{RECORDED_GAP_PCT:.4f}%)",
        "method": "REINFORCE arm B recipe (sponsor's "
                  "spec, 2026-11-26): tours sampled "
                  "from the network's own pointer "
                  "distribution (no temperature, no "
                  "surrogate); greedy tour length as "
                  "the per-problem baseline; "
                  "advantage = baseline - sampled "
                  "length; loss = -mean(advantage x "
                  "sum log-prob of sampled actions); "
                  f"Adam lr {LR0} with guard halving "
                  f"(floor {LR_FLOOR}), gradient clip "
                  f"norm {CLIP_NORM}; guard reads "
                  f"every {READ_EVERY} updates on the "
                  "fixed 10-city validation set with "
                  "revert-to-best; stitching genes "
                  "only, backbone frozen",
        "arm": name,
        "arm_label": arm_cfg["label"],
        "training_n_cities": n_cities,
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
        "training_n_cities": n_cities,
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
                         f"frankenstein_twofranks_reinforce"
                         f"_sizes/frankenstein_twofranks_"
                         f"reinforce{name}_champion.npz",
        "wall_seconds": round(wall, 1),
    }


arm_results = {}
for _name in ("B5", "B20"):
    arm_results[_name] = run_arm(_name, ARMS[_name])

best5 = arm_results["B5"]["best_validation_gap_pct"]
best20 = arm_results["B20"]["best_validation_gap_pct"]
best10 = REFERENCE["arm_B10_best_gap_pct"]
start = RECORDED_GAP_PCT
ranked = sorted(
    [("5 cities (arm B5)", best5),
     ("10 cities (arm B, prior test)", best10),
     ("20 cities (arm B20)", best20)],
    key=lambda t: t[1])
best_size_label, best_size_gap = ranked[0]
if best5 < best10 and best5 < best20:
    verdict = "down"
    answer = (f"Going DOWN in training problem size "
              f"helped: 5-city training (arm B5) "
              f"reached {best5:.4f}%, the best final "
              f"score of the three sizes — better "
              f"than 10-city training ({best10:.4f}%, "
              f"prior test) and 20-city training "
              f"({best20:.4f}%).")
elif best20 < best10 and best20 < best5:
    verdict = "up"
    answer = (f"Going UP in training problem size "
              f"helped: 20-city training (arm B20) "
              f"reached {best20:.4f}%, the best final "
              f"score of the three sizes — better "
              f"than 10-city training ({best10:.4f}%, "
              f"prior test) and 5-city training "
              f"({best5:.4f}%).")
elif best10 <= best5 and best10 <= best20:
    verdict = "neither"
    answer = (f"Neither direction helped: training "
              f"at the target size (10 cities, "
              f"{best10:.4f}% in the prior test) "
              f"stayed best; 5-city training reached "
              f"{best5:.4f}% and 20-city training "
              f"{best20:.4f}%, so going down or up "
              f"both hurt (or at best matched) "
              f"relative to training at 10 cities.")
else:
    verdict = "mixed"
    answer = (f"The size ranking was: "
              + "; ".join(f"{lab} {g:.4f}%"
                          for lab, g in ranked)
              + f". The best final score came from "
                f"{best_size_label} at "
                f"{best_size_gap:.4f}%.")

record = {
    "run": "REINFORCE training on the Two-Franks "
           "stitch — TRAINING PROBLEM SIZE test: "
           "arm B recipe trained on 5-city problems "
           "(B5) and on 20-city problems (B20), "
           "judged on the fixed 10-city validation "
           "set",
    "date": "2026-11-26",
    "sponsor_direction": "Roman, 2026-11-26: the "
                         "training runs pull 10-city "
                         "problems — 'Try to go down "
                         "and to go up. Could go "
                         "either way.' Test whether "
                         "the training problem size "
                         "helps or hurts.",
    "method": {
        "name": "REINFORCE (policy gradient) with a "
                "greedy baseline, arm B recipe exactly "
                "as in the prior test; only the "
                "training problem size varies",
        "sampling": "external sampling from the "
                    "network's own pointer "
                    "distribution (softmax over the "
                    "decoder logits as produced; no "
                    "temperature, no soft surrogate)",
        "baseline": "each problem's own greedy tour "
                    "length (at that problem's size)",
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
                        f"frozen ({len(BACKBONE_IDX)})",
        "guard": f"true-gap reads on the fixed "
                 f"10-city validation set every "
                 f"{READ_EVERY} updates; "
                 f"best-on-validation weights are the "
                 f"working point; a worse read reverts "
                 f"to the best, resets Adam, and "
                 f"halves the lr (floor {LR_FLOOR})",
        "problems": "fresh problems, uniform on the "
                    "unit square, recorded seed "
                    "streams, different per arm; "
                    "never the validation problems",
    },
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
        "size_sanity": size_sanity,
        "size_sanity_passed": True,
    },
    "arms": arm_results,
    "reference_10_city_arm_B": {
        "source": "frankenstein_twofranks_reinforce_"
                  "2026-11-26.json (prior test, not "
                  "rerun)",
        "best_validation_gap_pct": best10,
        "best_validation_update":
            REFERENCE["arm_B10_best_update"],
        "frac_samples_beating_greedy":
            REFERENCE["arm_B10_frac_samples_beating_greedy"],
    },
    "comparisons_pct": {
        "start_subject": start,
        "frank1_parent": REFERENCE["frank1_parent_gap_pct"],
        "frank2_parent": REFERENCE["frank2_parent_gap_pct"],
        "arm_B5_best": best5,
        "arm_B10_best_reference": best10,
        "arm_B20_best": best20,
    },
    "verdict_direction": verdict,
    "answer": answer,
    "wall_seconds_total": round(time.time() - T0, 1),
}
with open(HERE / "frankenstein_twofranks_reinforce_"
                 "sizes_2026-11-26.json", "w") as f:
    json.dump(record, f, indent=1)
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_twofranks_reinforce_sizes",
    "subject_gap_pct": RECORDED_GAP_PCT,
    "arm_B5_best_gap_pct": best5,
    "arm_B20_best_gap_pct": best20,
    "arm_B10_best_gap_pct_reference": best10,
    "verdict_direction": verdict,
    "answer": answer,
}, indent=1))
log(f"saved frankenstein_twofranks_reinforce_sizes_"
    f"2026-11-26.json; COMPLETE marker written")
log(f"DONE: arm B5 best {best5:.4f}%, arm B20 best "
    f"{best20:.4f}% (10-city arm B reference "
    f"{best10:.4f}%, start {start:.4f}%, Frank #1 "
    f"6.8959%, Frank #2 7.017%)")
log(f"ANSWER: {answer}")
