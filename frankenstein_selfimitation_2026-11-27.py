#!/usr/bin/env python3
"""Self-imitation probe (2026-11-27).

Tests the remedy proposed for the REINFORCE limitation:
the champion's decoder distribution is too flat (median score
gap between cities 0.00000027, sampled tours never beat
greedy), so teach the distribution to match the network's own
greedy tours before any further REINFORCE.

SUBJECT: the scaled-evolution champion
(engine_runs/frankenstein_widened_evolution_scaled/
frankenstein_widened_scaled_champion.npz), gap 5.2474% on the
fixed validation set (TSPDomain, val_seed 20261116, 200
problems, 10 cities).

METHOD (expert iteration, one round): per update, draw 4
fresh 10-city problems; for each, compute the network's
GREEDY tour (weights frozen during generation); loss =
cross-entropy of the decoder's action distribution against
the greedy action at each step (one-hot over unvisited
cities, visited masked; the first action is the forced start
city and carries no loss). Teacher-forced: the CE pass walks
the greedy trajectory, so states come from the greedy tour.
Train ALL genes (nothing frozen) with Adam lr 5e-6, gradient
clip 1.0, batch losses averaged, ONE update per batch, 400
updates. Generator seed 20261280. Guard: true reads on the
fixed set every 25 updates; best-on-validation is the working
point; on a worse read revert to best, reset Adam, halve lr
(floor 5e-8).

MEASUREMENTS before and after (same code, same seeds):
 1. validation gap on the fixed set;
 2. sampled-tour quality on 100 fresh 10-city and 100 fresh
    20-city problems, one sampled tour per starting city:
    share of sampled tours beating their greedy tour, and
    mean (sampled length - greedy length);
 3. distribution sharpness: median absolute gap between the
    best and second-best city scores (decoder logits,
    post-tanh, over unvisited cities) at each decoding step,
    across 20 fresh 10-city problems.

Forward/JAX machinery is verbatim from reinforce_tune.py
(the widened Step C Frankenstein mirror).
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

CHAMPION = (HERE / "engine_runs"
            / "frankenstein_widened_evolution_scaled"
            / "frankenstein_widened_scaled_champion.npz")
CFG = HERE / "bert_engine_config_deep50_stepA.json"
OUT = HERE / "engine_runs" / "frankenstein_selfimitation"
OUT.mkdir(parents=True, exist_ok=True)

LR0 = 5e-6
LR_FLOOR = 5e-8
UPDATES = 400
PROBLEMS_PER_UPDATE = 4
SEED_BASE = 20261280
READ_EVERY = 25
CLIP_NORM = 1.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
BLOCKS = ("b0", "b1", "b2", "b3", "b4", "b5w")
HEADS = (0, 1, 2, 3)
START_GAP_PCT = 5.2474
MEASURE_SEED = 20261281

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


# ================================================================
# Setup: domain, champion genome
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
EPS = float(arch.ln_eps)

zc = np.load(CHAMPION)
CMETA = json.loads(str(zc["meta"]))
KEYS = list(CMETA["gene_order"])
W_SUBJ = {k: np.asarray(zc[k], dtype=float) for k in KEYS}
SHAPES = {k: W_SUBJ[k].shape for k in KEYS}
SLICES = {}
_off = 0
for k in KEYS:
    SLICES[k] = (slice(_off, _off + W_SUBJ[k].size), SHAPES[k])
    _off += W_SUBJ[k].size
N_GENES = _off
SCALE = float(CMETA["attn_scale"])
w0 = np.concatenate([W_SUBJ[k].ravel() for k in KEYS])
assert N_GENES == 8480, N_GENES
log(f"subject: scaled-evolution champion, {N_GENES} genes; "
    f"starting gap {START_GAP_PCT:.4f}%")


def unpack(vec):
    return {k: vec[SLICES[k][0]].reshape(SHAPES[k])
            for k in KEYS}


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


# ================================================================
# Numpy forward + greedy decode (verbatim from reinforce_tune.py)
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
# JAX mirror (verbatim from reinforce_tune.py, plus the
# self-imitation loss and the measurement decoders)
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
    return length, picks


def _decoder_logits(W, Y, gmean, last, visited):
    state = gmean + last
    q = state @ W["head.W_qd"] + W["head.b_qd"]
    logits = be.CLIP * jnp.tanh((Y @ q) * SCALE)
    return logits - 1e9 * visited


def selfimitation_loss(w_flat, Xb):
    """Per problem: greedy tour from the CURRENT weights
    (stop-gradient targets); teacher-forced cross-entropy
    of the decoder distribution against the greedy action
    at every step after the forced first pick. Returns
    (mean loss, (greedy lengths, mean CE per problem))."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x):
        Y = encode_jax(W, x)
        n = x.shape[0]
        # Targets: greedy tour under detached weights.
        Wd = jax.tree_util.tree_map(
            jax.lax.stop_gradient, W)
        _, greedy_picks = greedy_from_Y(Wd, Y, x)
        gmean = Y.mean(axis=0)
        visited = jnp.zeros(n).at[greedy_picks[0]].set(1.0)
        last = Y[greedy_picks[0]]
        ce_sum = 0.0
        for t in range(1, n):
            logits = _decoder_logits(W, Y, gmean, last,
                                     visited)
            logp = jax.nn.log_softmax(logits)
            ce_sum = ce_sum - logp[greedy_picks[t]]
            visited = visited.at[greedy_picks[t]].set(1.0)
            last = Y[greedy_picks[t]]
        greedy_len, _ = greedy_from_Y(W, Y, x)
        return ce_sum / (n - 1), greedy_len

    ces, greedy_lens = jax.vmap(per_problem)(Xb)
    return jnp.mean(ces), (greedy_lens, ces)


vg = jax.jit(jax.value_and_grad(selfimitation_loss,
                                has_aux=True))


def sample_from_Y(W, Y, x, start, key):
    """One sampled tour, first pick FORCED to `start`
    (excluded from sum_logp). Verbatim from
    reinforce_tune.py with TEMP = 1.0."""
    n = x.shape[0]
    D = _dist(x)
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n).at[start].set(1.0)
    picks = jnp.zeros(n, dtype=jnp.int32).at[0].set(start)
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
        picks = picks.at[t].set(pick)
        visited = visited.at[pick].set(1.0)
        last = Y[pick]
    length = (D[picks[:-1], picks[1:]].sum()
              + D[picks[-1], picks[0]])
    return length


def sampled_quality_jax(w_flat, Xb, key):
    """One sampled tour per starting city for each problem
    in Xb. Returns (share of sampled tours strictly
    beating their greedy tour, mean sampled-minus-greedy
    length, n tours)."""
    W = {k: w_flat[sl].reshape(sh)
         for k, (sl, sh) in SLICES.items()}

    def per_problem(x, k0):
        Y = encode_jax(W, x)
        greedy_len, _ = greedy_from_Y(W, Y, x)
        n = x.shape[0]
        keys = jax.random.split(k0, n)
        starts = jnp.arange(n, dtype=jnp.int32)
        lens = jax.vmap(
            lambda s, kk: sample_from_Y(W, Y, x, s, kk))(
            starts, keys)
        return lens, greedy_len

    lens, greedy_lens = jax.vmap(per_problem)(
        Xb, jax.random.split(key, Xb.shape[0]))
    lens = np.asarray(lens)
    greedy_lens = np.asarray(greedy_lens)
    diffs = lens - greedy_lens[:, None]
    return {
        "n_tours": int(diffs.size),
        "share_sampled_beats_greedy": round(
            float((diffs < 0).mean()), 4),
        "mean_sampled_minus_greedy": round(
            float(diffs.mean()), 4),
        "std_sampled_minus_greedy": round(
            float(diffs.std()), 4),
    }


def sharpness_np(vec, Xb):
    """Median absolute gap between the best and second-best
    city scores (decoder logits, post-tanh, over unvisited
    cities) at each decoding step, across the given
    problems. Numpy forward, greedy decode."""
    W = unpack(vec)
    Y = forward_wide(W, Xb)
    B, n, d = Y.shape
    gmean = Y.mean(axis=1)
    last = np.broadcast_to(W["head.v_start"], (B, d)).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    gaps = []
    for t in range(n - 1):
        state = gmean + last
        q = state @ W["head.W_qd"] + W["head.b_qd"]
        logits = be.CLIP * np.tanh(
            np.einsum("bi,bni->bn", q, Y) * SCALE)
        logits = np.where(visited, -np.inf, logits)
        part = np.partition(logits, -2, axis=1)
        gaps.extend((part[:, -1] - part[:, -2]).tolist())
        pick = np.argmax(np.where(visited, -1e18, logits),
                         axis=1)
        visited[arB, pick] = True
        last = Y[arB, pick]
    gaps = np.asarray(gaps)
    return {
        "n_steps": int(gaps.size),
        "median_top1_minus_top2": float(np.median(gaps)),
        "mean_top1_minus_top2": float(gaps.mean()),
        "p10_top1_minus_top2": float(np.percentile(gaps, 10)),
        "p90_top1_minus_top2": float(np.percentile(gaps, 90)),
    }


def measure_all(vec, tag):
    """The three before/after measurements with fixed
    seeds, so before and after are directly comparable."""
    gap = mean_gap(vec, X_val, OPT_VAL) * 100
    rng = np.random.default_rng(MEASURE_SEED)
    out = {"validation_gap_pct": round(float(gap), 4)}
    for n in (10, 20):
        Xb = rng.random((100, n, 2))
        key = jax.random.PRNGKey(MEASURE_SEED + n)
        q = sampled_quality_jax(
            jnp.asarray(vec, dtype=jnp.float64),
            jnp.asarray(Xb, dtype=jnp.float64), key)
        out[f"sampled_n{n}"] = q
    Xb20 = rng.random((20, 10, 2))
    out["sharpness"] = sharpness_np(vec, Xb20)
    log(f"MEASURE {tag}: gap {gap:.4f}%; "
        f"n10 share-beats {out['sampled_n10']['share_sampled_beats_greedy']:.4f} "
        f"mean-diff {out['sampled_n10']['mean_sampled_minus_greedy']:.4f}; "
        f"n20 share-beats {out['sampled_n20']['share_sampled_beats_greedy']:.4f} "
        f"mean-diff {out['sampled_n20']['mean_sampled_minus_greedy']:.4f}; "
        f"sharpness median {out['sharpness']['median_top1_minus_top2']:.3e}")
    return out


# ================================================================
# SANITY + MACHINERY CHECKS
# ================================================================
birth_gap = mean_gap(w0, X_val, OPT_VAL)
log(f"sanity: champion true validation gap "
    f"{birth_gap*100:.6f}% vs recorded {START_GAP_PCT:.4f}%")
if abs(birth_gap * 100 - START_GAP_PCT) > 5e-5 + 1e-9:
    log("SANITY FAILED. STOPPING; no training.")
    sys.exit(3)

Y_np = forward_wide(W_SUBJ, X_val[:8])
W_j = {k: jnp.asarray(W_SUBJ[k], dtype=jnp.float64)
       for k in KEYS}
Y_jx = np.asarray(jax.vmap(
    lambda x: encode_jax(W_j, x))(
    jnp.asarray(X_val[:8], dtype=jnp.float64)))
parity = float(np.max(np.abs(Y_np - Y_jx)))
if parity > 1e-9:
    log(f"MIRROR CHECK FAILED ({parity:.2e}). STOPPING.")
    sys.exit(3)
np_greedy_lens, _ = true_lengths(w0, X_val[:8])
jx_greedy_lens = np.asarray(jax.vmap(
    lambda x: greedy_from_Y(W_j, encode_jax(W_j, x), x)[0])(
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
# MEASURE BEFORE
# ================================================================
measure_before = measure_all(w0, "BEFORE")

# ================================================================
# Training with the strict guard
# ================================================================
log(f"TRAINING self-imitation: {UPDATES} updates, "
    f"{PROBLEMS_PER_UPDATE} fresh 10-city problems/update, "
    f"Adam lr {LR0:.1e} (floor {LR_FLOOR:.0e}), all "
    f"{N_GENES} genes, seed {SEED_BASE}")
t_train = time.time()

w = w0.copy()
adam = {"m": np.zeros(N_GENES), "v": np.zeros(N_GENES),
        "t": 0}
state = {"lr": LR0, "best_gap": np.inf,
         "best_w": w.copy(), "best_update": -1,
         "reverts": 0, "halvings": 0}
reads = []
guard_events = []
losses = []


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
        log(f"READ {update}: gap {g*100:.4f}% <-- new best")
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
    action = "revert_to_best" + (
        "_and_halve_lr" if halved else "_lr_at_floor")
    guard_events.append(
        {"update": update, "gap_pct": round(g * 100, 4),
         "action": action, "lr_after": state["lr"],
         "best_gap_pct": round(state["best_gap"] * 100, 4),
         "best_update": state["best_update"]})
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
    rng = np.random.default_rng(SEED_BASE + u)
    Xb = rng.random((PROBLEMS_PER_UPDATE, 10, 2))
    (loss, (greedy_lens, ces)), grad = vg(
        jnp.asarray(w, dtype=jnp.float64),
        jnp.asarray(Xb, dtype=jnp.float64))
    grad = np.asarray(grad, dtype=float)
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
    if u % READ_EVERY == 0:
        do_read(u)

wall = time.time() - t_train
best_w = state["best_w"]
best_pct = round(state["best_gap"] * 100, 4)
n_changed = int((best_w != w0).sum())
log(f"DONE in {wall:.1f}s: best {best_pct:.4f}% "
    f"@ update {state['best_update']} (start "
    f"{START_GAP_PCT:.4f}%); guard: {state['reverts']} "
    f"reverts, {state['halvings']} halvings, final lr "
    f"{state['lr']:.2e}; {n_changed} genes changed")

# ================================================================
# MEASURE AFTER (on the best weights, the vaulted champion)
# ================================================================
measure_after = measure_all(best_w, "AFTER")

# ================================================================
# Vault champion + record
# ================================================================
champ = OUT / "frankenstein_selfimitation_champion.npz"
meta = {
    "kind": "frankenstein_selfimitation_champion",
    "run": "selfimitation",
    "date": "2026-11-27",
    "base": ("engine_runs/frankenstein_widened_evolution_scaled/"
             "frankenstein_widened_scaled_champion.npz"),
    "method": ("expert iteration, one round: cross-entropy "
               "of the decoder distribution against the "
               "network's own greedy actions, teacher-forced, "
               "first pick forced, no loss on it; all 8,480 "
               "genes train, Adam lr 5e-6, clip 1.0, 400 "
               "updates x 4 fresh 10-city problems, seed "
               "20261280, strict guard"),
    "best_validation_gap_pct": best_pct,
    "best_validation_update": state["best_update"],
    "start_gap_pct": START_GAP_PCT,
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_changed_vs_champion": n_changed,
    "attn_scale": SCALE, "ln_eps": EPS,
}
atomic_savez(champ, **{k: unpack(best_w)[k] for k in KEYS},
             meta=np.array(json.dumps(meta)))

record = {
    "run": "frankenstein_selfimitation",
    "date": "2026-11-27",
    "method": meta["method"],
    "subject": meta["base"],
    "start_gap_pct": START_GAP_PCT,
    "sanity": {"parity_encoder": parity,
               "parity_greedy_n10": greedy_parity},
    "config": {"lr": LR0, "lr_floor": LR_FLOOR,
               "updates": UPDATES,
               "problems_per_update": PROBLEMS_PER_UPDATE,
               "seed_base": SEED_BASE,
               "measure_seed": MEASURE_SEED,
               "read_every": READ_EVERY,
               "clip_norm": CLIP_NORM,
               "train_all_genes": True},
    "reads": reads,
    "guard": {"reverts": state["reverts"],
              "halvings": state["halvings"],
              "final_lr": state["lr"],
              "events": guard_events},
    "losses": {"mean_first_25": round(
                   float(np.mean(losses[:25])), 4),
               "mean_last_25": round(
                   float(np.mean(losses[-25:])), 4),
               "first": round(float(losses[0]), 4),
               "last": round(float(losses[-1]), 4)},
    "measure_before": measure_before,
    "measure_after": measure_after,
    "best_validation_gap_pct": best_pct,
    "best_validation_update": state["best_update"],
    "genes_changed_vs_champion": n_changed,
    "champion_file": ("engine_runs/frankenstein_selfimitation/"
                      "frankenstein_selfimitation_champion.npz"),
    "wall_seconds": round(wall, 1),
}
with open(HERE / "frankenstein_selfimitation_2026-11-27.json",
          "w") as f:
    json.dump(record, f, indent=1)
with open(OUT / "COMPLETE", "w") as f:
    f.write(f"self-imitation probe complete; best "
            f"{best_pct:.4f}% @ update "
            f"{state['best_update']}\n")
log(f"saved frankenstein_selfimitation_2026-11-27.json + "
    f"champion; best {best_pct:.4f}%")
