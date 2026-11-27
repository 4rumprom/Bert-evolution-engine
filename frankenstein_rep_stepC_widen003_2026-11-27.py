#!/usr/bin/env python3
# REPLICATION (2026-11-27): adapted copy of
# frankenstein_stepC_widen003_2026-11-26.py - subject is the
# replication step C champion (4.1580%), noise seed 20261340,
# FFN widths derived from the subject genome, outputs under
# engine_runs/frankenstein_replication/. See
# frankenstein_replication_log_2026-11-27.md.
"""STAGE 1 of the sponsor's 2026-11-26 direction
(widen the Step C Frankenstein by Net2WiderNet
split, with noise on the new copies; REINFORCE
follows as stage 2 in
frankenstein_stepC_reinforce_2026-11-26.py).

SUBJECT: engine_runs/frankenstein_stepC_evolution/
frankenstein_stepC_evolved_champion.npz — the
Step C evolved champion, validation gap 8.1018%
on the fixed 200-problem set (val_seed 20261116,
10 cities). Shape, verified from the genome
itself: entry projection (2 -> 4); fused blocks
b0..b4 (stream 4; FFN widths 7, 5, 6, 5, 5; each
stores ONE fused attention set (4,4)); merged
pointer decoder over width 4; backend block b5w
with TWO explicit heads (h0, h1) and FFN 8.
Total 1,040 genes (788 trunk + 252 b5w).
attn_scale = 1/sqrt(2), ln_eps = 1e-5 (recorded
in the champion's meta; the Step C forward uses
the same values).

THE WIDEN (the sponsor's frozen default widen
rule — d x2, FFN x2, heads x2, Net2WiderNet
splits — following the verified construction of
frankenstein_widen4_2026-11-25.py exactly):
  - Stream d 4 -> 8 by duplication: new stream
    coords 4..7 are copies of coords 0..3. Every
    array follows one uniform rule: on an OUTPUT
    side of the widened stream the axis is
    duplicated; on an INPUT side the axis is
    split evenly (each copy carries half),
    because the duplicated input coordinates sum
    into every consumer.
  - attn_scale: HALVED (SCALE/2). Both dot
    products of the forward pass — attention
    scores and decoder logits — run over the
    duplicated stream and would otherwise double
    exactly; halving the one shared scale
    restores them.
  - ln_eps: UNCHANGED. Duplication preserves the
    layer norm mean and variance over the stream
    exactly. LN gains/biases are duplicated.
  - Heads x2 in every block, Net2WiderNet split:
    * Fused blocks (b0..b4): the stored block
      holds one fused attention; it is first
      decomposed into 2 heads (both heads carry
      the full W_q/W_k/W_v; W_o and b_o halved
      between them — head outputs are summed, so
      the decomposition is exact), then each of
      the 2 heads is split again: Q/K/V copied
      exactly, W_o and b_o halved in both
      daughters. Net: 4 identical heads per
      block, W_o/4, b_o/4 of the widened
      originals. (Verbatim from widen4.)
    * b5w (2 explicit heads h0, h1): each head
      is split: Q/K/V copied exactly into both
      daughters, W_o and b_o halved in both.
      Daughters of h0 are h0 (base) and h2
      (appended copy); daughters of h1 are h1
      (base) and h3 (appended copy).
  - FFN x2 per block, Net2WiderNet split
    (transplant_split_widen convention): each
    unit's W_1 column and b_1 copied exactly
    into the appended copy unit; the unit's
    outgoing W_2 row is halved in both copies.
    (The stream input-split also applies to
    W_1's rows and W_2's columns per the
    uniform rule.)
  - Decoder widened with the stream: W_qd (8,8)
    by the uniform rule (input split, output
    duplicated); b_qd and v_start duplicated.
  - Entry: W_in columns duplicated; b_in
    duplicated.

NOISE AMENDMENT (sponsor, 2026-11-26): after
splitting, small random noise is added to the
NEW copies' synapse weights — every weight
entry that belongs to a new copy (its incoming
and outgoing weights), NOT the halved outgoing
values of the original copies. Noise: Gaussian
std 0.003, one rng stream, seed 20261240 (the
SAME seed as the 0.03 run, so the noise pattern
is identical and only the scale changes),
drawn per key in gene order. RERUN NOTE: this
is the rerun of frankenstein_stepC_widen_
2026-11-26.py with NOISE_STD 0.003 instead of
0.03 (sponsor, 2026-11-26 22:12 EST: the birth
score should have been nearly identical; the
0.03 run drifted +12.2103 points to 20.3121%). The original copies stay exact
(with their outgoing weights halved per the
split rule).

MASK (saved with the genome): FROZEN = every
gene of the original copies — an entry is
frozen iff every widened axis it touches is an
original index (stream coords 0..3, FFN units
0..U0-1, heads 0..1), including their halved
outgoing weights; TRAINABLE = every gene
belonging to a new copy (the union rule of
widen4: any entry touching a copy index, or
belonging to a copy unit/head). The noise is
applied to exactly the trainable entries.
Hand count (asserted): total 8,480, frozen
1,440, trainable 7,040.

PARITY GATE: a no-noise widened variant is
built first and must reproduce the subject
EXACTLY — identical greedy tours on all 200
validation problems (hence the same gap,
8.1018%). Only then is the noise applied for
the real widened genome, whose birth gap is
measured and reported plainly (exact
preservation does NOT hold for the noisy
genome, by design).

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_stepC_widen003_2026-11-26.py
Outputs: frankenstein_stepC_widened003_genome.npz
(with the frozen/trainable mask), and
frankenstein_stepC_widen003_2026-11-26.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CFG = HERE / "bert_engine_config_rep_deep50.json"
CHAMP = (HERE / "engine_runs" / "frankenstein_replication"
         / "stepC_evolution"
         / "frankenstein_rep_stepC_evolved_champion.npz")
CHAMP_GAP_PCT = 4.158
D0, D1 = 4, 8
NOISE_SEED = 20261340
NOISE_STD = 0.003
FUSED_BLOCKS = ("b0", "b1", "b2", "b3", "b4")
ALL_BLOCKS = FUSED_BLOCKS + ("b5w",)
FFN0 = None  # derived from the subject genome below (replication widths differ from the original line)

# ================================================================
# Setup: domain (validation set), subject
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = float(arch.attn_scale)
EPS = float(arch.ln_eps)
SCALE_W = SCALE / 2.0
log(f"attn_scale {SCALE!r} -> widened {SCALE_W!r} "
    f"(halved: dot products run over the duplicated "
    f"stream); ln_eps {EPS!r} unchanged")

zc = np.load(CHAMP)
META_C = json.loads(str(zc["meta"]))
KEYS0 = [k for k in zc.files if k != "meta"]
W0 = {k: np.asarray(zc[k], dtype=float) for k in KEYS0}
FFN0 = {blk: int(W0[f"{blk}.W_1"].shape[1]) for blk in ALL_BLOCKS}
assert abs(float(META_C["best_validation_gap_pct"])
           - CHAMP_GAP_PCT) < 5e-5
assert abs(float(META_C["attn_scale"]) - SCALE) < 1e-15
assert abs(float(META_C["ln_eps"]) - EPS) < 1e-15
for blk in ALL_BLOCKS:
    assert W0[f"{blk}.W_1"].shape == (D0, FFN0[blk]), \
        (blk, W0[f"{blk}.W_1"].shape)
log(f"subject loaded: {sum(v.size for v in W0.values())} "
    f"genes, recorded gap {CHAMP_GAP_PCT}%; FFN widths "
    f"{[FFN0[b] for b in ALL_BLOCKS]} (b0..b4 fused, "
    f"b5w 2 heads)")


# ================================================================
# Subject forward (verbatim from
# frankenstein_stepC_evolve_2026-11-26.py)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_subject(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in FUSED_BLOCKS:
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        Q = Hb @ W[f"{blk}.W_q"] + W[f"{blk}.b_q"]
        K = Hb @ W[f"{blk}.W_k"] + W[f"{blk}.b_k"]
        V = Hb @ W[f"{blk}.W_v"] + W[f"{blk}.b_v"]
        S = np.einsum("btj,bsj->bts", Q, K) * SCALE
        S = S - S.max(axis=-1, keepdims=True)
        E = np.exp(S)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        A = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        Hf = ln_np(A, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"]
                        + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
    blk = "b5w"
    Hb = ln_np(A, W[f"{blk}.lnA_gain"], W[f"{blk}.lnA_bias"],
               EPS)
    for h in (0, 1):
        Q = Hb @ W[f"{blk}.h{h}.W_q"] + W[f"{blk}.h{h}.b_q"]
        K = Hb @ W[f"{blk}.h{h}.W_k"] + W[f"{blk}.h{h}.b_k"]
        V = Hb @ W[f"{blk}.h{h}.W_v"] + W[f"{blk}.h{h}.b_v"]
        S = np.einsum("btj,bsj->bts", Q, K) * SCALE
        S = S - S.max(axis=-1, keepdims=True)
        E = np.exp(S)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        A = A + Ah @ W[f"{blk}.h{h}.W_o"] \
            + W[f"{blk}.h{h}.b_o"]
    Hf = ln_np(A, W[f"{blk}.lnF_gain"], W[f"{blk}.lnF_bias"],
               EPS)
    A = A + np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"]) \
        @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
    return A


def decode(W, Y, w_qd, b_qd, v_start, scale):
    B, n, d = Y.shape
    gmean = Y.mean(axis=1)
    last = np.broadcast_to(v_start, (B, d)).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    picks = np.zeros((B, n), dtype=int)
    for t in range(n):
        state = gmean + last
        q = state @ w_qd + b_qd
        logits = be.CLIP * np.tanh(
            np.einsum("bi,bni->bn", q, Y) * scale)
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last = Y[arB, pick]
    return picks


def gap_subject(W, X):
    Y = forward_subject(W, X)
    picks = decode(W, Y, W["head.W_qd"], W["head.b_qd"],
                   W["head.v_start"], SCALE)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / OPT_VAL - 1.0).mean()), picks


subj_gap, subj_picks = gap_subject(W0, X_val)
log(f"subject re-scored under the stepC forward: "
    f"{subj_gap*100:.6f}% (recorded {CHAMP_GAP_PCT}%)")
assert abs(subj_gap * 100 - CHAMP_GAP_PCT) < 5e-5 + 1e-9

# ================================================================
# Widened forward (multi-head in every block,
# summed head outputs; halved scale)
# ================================================================


def forward_wide(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ALL_BLOCKS:
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        out = 0.0
        for h in range(4):
            p = f"{blk}.h{h}"
            Q = Hb @ W[f"{p}.W_q"] + W[f"{p}.b_q"]
            K = Hb @ W[f"{p}.W_k"] + W[f"{p}.b_k"]
            V = Hb @ W[f"{p}.W_v"] + W[f"{p}.b_v"]
            S = np.einsum("btj,bsj->bts", Q, K) * SCALE_W
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


def gap_wide(W, X):
    Y = forward_wide(W, X)
    picks = decode(W, Y, W["head.W_qd"], W["head.b_qd"],
                   W["head.v_start"], SCALE_W)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / OPT_VAL - 1.0).mean()), picks


# ================================================================
# The widen construction (no noise yet)
# ================================================================
def dup_vec(v):
    """(d0,) -> (2*d0,): originals first, copies appended."""
    return np.concatenate([v, v])


def split_mat(M):
    """(d0, d0) -> (d1, d1) under the uniform rule:
    input axis split evenly (each copy block carries
    M/2), output axis duplicated. Every 4x4 copy block
    of the result equals M/2."""
    out = np.zeros((D1, D1))
    for ci in (0, D0):
        for cj in (0, D0):
            out[ci:ci + D0, cj:cj + D0] = M / 2.0
    return out


W8 = {}
# ---- embed: output side is the stream (duplicated);
# input side is the raw input (not widened)
W8["embed.W_in"] = np.concatenate(
    [W0["embed.W_in"], W0["embed.W_in"]], axis=1)
W8["embed.b_in"] = dup_vec(W0["embed.b_in"])

for blk in ALL_BLOCKS:
    U0 = FFN0[blk]
    U1 = 2 * U0
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
        W8[f"{blk}.{nm}"] = dup_vec(W0[f"{blk}.{nm}"])
    if blk in FUSED_BLOCKS:
        # heads: 4 identical heads (decompose the
        # fused attention into 2, then split each
        # into 2 — verbatim from widen4). W_o and
        # b_o carry /4 of the widened originals.
        for h in range(4):
            p = f"{blk}.h{h}"
            for nm in ("W_q", "W_k", "W_v"):
                W8[f"{p}.{nm}"] = split_mat(W0[f"{blk}.{nm}"])
            W8[f"{p}.W_o"] = split_mat(W0[f"{blk}.W_o"]) / 4.0
            for nm in ("b_q", "b_k", "b_v"):
                W8[f"{p}.{nm}"] = dup_vec(W0[f"{blk}.{nm}"])
            W8[f"{p}.b_o"] = dup_vec(W0[f"{blk}.b_o"]) / 4.0
    else:
        # b5w: split each of the 2 explicit heads.
        # Daughters of h0: h0 (base), h2 (copy);
        # daughters of h1: h1 (base), h3 (copy).
        for src, daughters in ((0, (0, 2)), (1, (1, 3))):
            s = f"{blk}.h{src}"
            for h in daughters:
                p = f"{blk}.h{h}"
                for nm in ("W_q", "W_k", "W_v"):
                    W8[f"{p}.{nm}"] = split_mat(W0[f"{s}.{nm}"])
                W8[f"{p}.W_o"] = split_mat(W0[f"{s}.W_o"]) / 2.0
                for nm in ("b_q", "b_k", "b_v"):
                    W8[f"{p}.{nm}"] = dup_vec(W0[f"{s}.{nm}"])
                W8[f"{p}.b_o"] = dup_vec(W0[f"{s}.b_o"]) / 2.0
    # FFN split: W_1 (d, U): input rows split, unit
    # columns duplicated -> every copy block W_1/2.
    W1 = np.zeros((D1, U1))
    for ci in (0, D0):
        for cu in (0, U0):
            W1[ci:ci + D0, cu:cu + U0] = W0[f"{blk}.W_1"] / 2.0
    W8[f"{blk}.W_1"] = W1
    W8[f"{blk}.b_1"] = dup_vec(W0[f"{blk}.b_1"])
    # W_2 (U, d): unit rows halved in both copies
    # (the split), stream columns duplicated under
    # the split rule -> every copy block W_2/2.
    W2 = np.zeros((U1, D1))
    for cu in (0, U0):
        for cj in (0, D0):
            W2[cu:cu + U0, cj:cj + D0] = W0[f"{blk}.W_2"] / 2.0
    W8[f"{blk}.W_2"] = W2
    W8[f"{blk}.b_2"] = dup_vec(W0[f"{blk}.b_2"])

# ---- decoder over the widened stream
W8["head.W_qd"] = split_mat(W0["head.W_qd"])
W8["head.b_qd"] = dup_vec(W0["head.b_qd"])
W8["head.v_start"] = dup_vec(W0["head.v_start"])

# gene order of the widened genome
KEYS8 = ["embed.W_in", "embed.b_in"]
for blk in ALL_BLOCKS:
    KEYS8 += [f"{blk}.lnA_gain", f"{blk}.lnA_bias"]
    for h in range(4):
        p = f"{blk}.h{h}"
        KEYS8 += [f"{p}.W_q", f"{p}.W_k", f"{p}.W_v",
                  f"{p}.W_o", f"{p}.b_q", f"{p}.b_k",
                  f"{p}.b_v", f"{p}.b_o"]
    KEYS8 += [f"{blk}.lnF_gain", f"{blk}.lnF_bias",
              f"{blk}.W_1", f"{blk}.b_1", f"{blk}.W_2",
              f"{blk}.b_2"]
KEYS8 += ["head.W_qd", "head.b_qd", "head.v_start"]
assert set(KEYS8) == set(W8.keys()) and len(KEYS8) == len(W8)

# ================================================================
# Frozen / trainable mask (union rule; see
# docstring). True = trainable (a new copy's gene).
# ================================================================


def trainable_for(key, shape):
    m = np.zeros(shape, dtype=bool)
    if key == "embed.W_in":
        m[:, D0:] = True
    elif key in ("embed.b_in", "head.b_qd", "head.v_start"):
        m[D0:] = True
    elif key == "head.W_qd":
        m[D0:, :] = True
        m[:, D0:] = True
    else:
        blk, rest = key.split(".", 1)
        U0 = FFN0[blk]
        if rest in ("lnA_gain", "lnA_bias", "lnF_gain",
                    "lnF_bias", "b_2"):
            m[D0:] = True
        elif rest == "W_1":
            m[D0:, :] = True
            m[:, U0:] = True
        elif rest == "b_1":
            m[U0:] = True
        elif rest == "W_2":
            m[U0:, :] = True
            m[:, D0:] = True
        elif rest.startswith("h"):
            h = int(rest.split(".")[0][1:])
            nm = rest.split(".")[1]
            if h >= 2:
                m[:] = True
            elif nm in ("W_q", "W_k", "W_v", "W_o"):
                m[D0:, :] = True
                m[:, D0:] = True
            else:  # base-head biases
                m[D0:] = True
        else:
            raise ValueError(f"unhandled key {key}")
    return m


TRAIN8 = {k: trainable_for(k, W8[k].shape) for k in KEYS8}
flat_train = np.concatenate(
    [TRAIN8[k].ravel() for k in KEYS8])
n_total = int(flat_train.size)
n_train = int(flat_train.sum())
n_frozen = n_total - n_train
log(f"gene counts: total {n_total}, frozen (original "
    f"copies) {n_frozen}, trainable (new copies) "
    f"{n_train}")
log(f"replication gene counts recorded: total {n_total}, frozen {n_frozen}, trainable {n_train} (original line: 8480 / 1440 / 7040)")

# ================================================================
# PARITY GATE: the no-noise widened network must
# reproduce the subject exactly (identical tours)
# ================================================================
wide_gap, wide_picks = gap_wide(W8, X_val)
tours_identical = bool((wide_picks == subj_picks).all())
log(f"PARITY GATE (no noise): widened gap "
    f"{wide_gap*100:.6f}% vs subject "
    f"{subj_gap*100:.6f}%; tours identical on all "
    f"200 problems: {tours_identical}")
if not tours_identical or abs(wide_gap - subj_gap) > 1e-9:
    log("GATE FAILED — the split construction does "
        "not reproduce the subject. STOPPING; no "
        "genome saved.")
    sys.exit(3)
log("GATE PASSED")

# ================================================================
# NOISE AMENDMENT: Gaussian std 0.003 on the new
# copies' entries only (the trainable set), one
# rng stream, seed 20261240, drawn per key in
# gene order. Original copies stay exact.
# ================================================================
rng_noise = np.random.default_rng(NOISE_SEED)
W8N = {}
for k in KEYS8:
    noise = rng_noise.normal(0.0, NOISE_STD,
                             size=W8[k].shape)
    W8N[k] = W8[k] + np.where(TRAIN8[k], noise, 0.0)
# the frozen entries must be bit-identical to the
# no-noise construction
for k in KEYS8:
    assert np.array_equal(W8N[k][~TRAIN8[k]],
                          W8[k][~TRAIN8[k]]), k
n_noised = int(sum(int(TRAIN8[k].sum()) for k in KEYS8))
log(f"noise applied: std {NOISE_STD}, seed "
    f"{NOISE_SEED}, {n_noised} trainable entries "
    f"noised; frozen entries bit-identical to the "
    f"no-noise construction")

noisy_gap, noisy_picks = gap_wide(W8N, X_val)
n_tours_changed = int((noisy_picks != subj_picks).any(
    axis=1).sum())
log(f"WIDENED BIRTH GAP (with noise): "
    f"{noisy_gap*100:.6f}% vs subject "
    f"{CHAMP_GAP_PCT:.4f}% — drift "
    f"{noisy_gap*100 - CHAMP_GAP_PCT:+.4f} points; "
    f"tours changed on {n_tours_changed} of 200 "
    f"problems (exact preservation does not hold "
    f"for the noisy genome, by design)")

# ================================================================
# Save genome (with mask) + results JSON
# ================================================================
frozen_idx = np.flatnonzero(~flat_train)
train_idx = np.flatnonzero(flat_train)
shapes_before = {k: list(W0[k].shape) for k in KEYS0}
shapes_after = {k: list(W8N[k].shape) for k in KEYS8}
meta = {
    "kind": "frankenstein_rep_stepC_widened_genome",
    "date": "2026-11-27",
    "base": "engine_runs/frankenstein_stepC_evolution/"
            "frankenstein_stepC_evolved_champion.npz",
    "widen": "sponsor default widen across the whole "
             "network (Net2WiderNet split): stream "
             "4 -> 8 by duplication, FFN x2 and "
             "heads x2 in every block, decoder "
             "widened with the stream",
    "noise_amendment": f"Gaussian std {NOISE_STD} "
                       f"added after splitting to "
                       f"the new copies' synapse "
                       f"weights only (the trainable "
                       f"set), one rng stream, seed "
                       f"{NOISE_SEED}; original "
                       f"copies exact (outgoing "
                       f"weights halved per the split "
                       f"rule)",
    "attn_scale": SCALE_W,
    "attn_scale_note": "halved vs the subject's "
                       "SCALE (dot products over the "
                       "duplicated stream double)",
    "ln_eps": EPS,
    "ln_eps_note": "unchanged (duplication preserves "
                   "LN mean/variance exactly)",
    "heads_per_block": 4,
    "head_note": "fused blocks b0..b4: the stored "
                 "fused attention is decomposed "
                 "into 2 heads by the split (W_o, "
                 "b_o halved) and each head split "
                 "again to 4 (all 4 identical before "
                 "noise); b5w: each of its 2 heads "
                 "split into 2 (daughters of h0 are "
                 "h0/h2, of h1 are h1/h3)",
    "copy_placement": "originals first, copies "
                      "appended (stream coords 4..7, "
                      "FFN units U0..2U0-1, heads "
                      "2..3)",
    "mask_rule": "frozen = every gene of the "
                 "original copies (every widened "
                 "axis it touches is an original "
                 "index; halved outgoing weights "
                 "included); trainable = every gene "
                 "belonging to a new copy (union "
                 "rule of widen4)",
    "gene_order": KEYS8,
    "genes_total": n_total, "genes_frozen": n_frozen,
    "genes_trainable": n_train,
    "subject_gap_pct": CHAMP_GAP_PCT,
    "parity_gate": {
        "no_noise_gap_pct": round(wide_gap * 100, 6),
        "tours_identical_all_200": tours_identical},
    "birth_gap_pct_with_noise": round(noisy_gap * 100, 6),
}
np.savez(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepC_widened003_genome.npz",
         **W8N,
         trainable_mask=flat_train.astype(np.int8),
         frozen_idx=frozen_idx, trainable_idx=train_idx,
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_rep_stepC_widened003_genome.npz "
    "(with frozen/trainable mask)")

results = {
    "run": "stage 1 RERUN: Net2WiderNet split-widen "
           "of the Step C Frankenstein, with the "
           "sponsor's noise amendment on the new "
           "copies at std 0.003 (was 0.03 in "
           "frankenstein_stepC_widen_2026-11-26.py; "
           "same noise seed 20261240)",
    "date": "2026-11-27",
    "sponsor_direction": "Roman, 2026-11-26 22:01 "
                         "EST: widen it now with "
                         "Net2Net and split neurons "
                         "with the 50/50 weight "
                         "technique; add a small "
                         "random noise to the weight "
                         "of the new synapses. "
                         "RERUN, 2026-11-26 22:12 "
                         "EST: birth score should "
                         "have been nearly "
                         "identical; noise needs to "
                         "be tiny; rerun at 0.003",
    "subject": {
        "file": "engine_runs/frankenstein_stepC_"
                "evolution/frankenstein_stepC_"
                "evolved_champion.npz",
        "validation_gap_pct": CHAMP_GAP_PCT,
        "recomputed_gap_pct": round(subj_gap * 100, 6),
        "genes": int(sum(v.size for v in W0.values())),
        "ffn_widths": {b: FFN0[b] for b in ALL_BLOCKS},
        "heads_before": {"b0..b4": "one fused "
                                   "attention set",
                         "b5w": 2},
    },
    "shapes_before": shapes_before,
    "shapes_after": shapes_after,
    "gene_counts": {"total": n_total,
                    "frozen_original_copies": n_frozen,
                    "trainable_new_copies": n_train},
    "conventions": {
        "stream": "d 4 -> 8 by duplication; copies "
                  "appended (coords 4..7 copy 0..3); "
                  "output sides of the stream are "
                  "duplicated, input sides are split "
                  "evenly (each copy carries half)",
        "attn_scale": f"{SCALE} -> {SCALE_W} "
                      "(halved; attention-score and "
                      "decoder dot products run over "
                      "the duplicated stream)",
        "ln_eps": f"{EPS} unchanged (duplication "
                  "preserves LN mean/variance)",
        "ln_gains_biases": "duplicated",
        "heads": "x2 in every block. Fused blocks "
                 "(b0..b4): the stored fused "
                 "attention is decomposed into 2 "
                 "heads (full W_q/W_k/W_v each; W_o, "
                 "b_o halved between them), then "
                 "each head is split (Q/K/V copied "
                 "exactly, W_o and b_o halved in "
                 "both daughters) -> 4 heads, all "
                 "identical before noise. b5w: each "
                 "of its 2 heads is split the same "
                 "way -> 4 heads (daughters of h0: "
                 "h0, h2; of h1: h1, h3)",
        "ffn": "x2 per block by Net2WiderNet split: "
               "W_1 column and b_1 copied exactly "
               "into the appended copy unit, W_2 row "
               "halved in both copies (the stream "
               "input-split also applies to W_1's "
               "rows and W_2's columns)",
        "decoder": "W_qd (8,8) by the uniform "
                   "duplication rule (input split, "
                   "output duplicated); b_qd and "
                   "v_start duplicated",
        "entry": "W_in columns duplicated; b_in "
                 "duplicated",
        "noise": f"Gaussian std {NOISE_STD}, seed "
                 f"{NOISE_SEED}, one rng stream "
                 f"drawn per key in gene order, "
                 f"added AFTER splitting to the "
                 f"trainable entries only (every "
                 f"weight entry belonging to a new "
                 f"copy: its incoming and outgoing "
                 f"weights); the original copies' "
                 f"entries, including their halved "
                 f"outgoing weights, stay exact",
        "mask_rule": "frozen = every gene of the "
                     "original copies (union rule: "
                     "an entry is frozen iff every "
                     "widened axis it touches is an "
                     "original index — stream 0..3, "
                     "units 0..U0-1, heads 0..1); "
                     "trainable = the rest (any "
                     "entry touching a copy index, "
                     "or belonging to a copy unit "
                     "or head). The noise set equals "
                     "the trainable set",
    },
    "parity_gate_no_noise": {
        "pass": True,
        "tours_identical_all_200": tours_identical,
        "subject_gap_pct_recomputed":
            round(subj_gap * 100, 6),
        "widened_gap_pct": round(wide_gap * 100, 6),
    },
    "birth_gap_with_noise": {
        "gap_pct": round(noisy_gap * 100, 6),
        "subject_gap_pct": CHAMP_GAP_PCT,
        "drift_points": round(
            noisy_gap * 100 - CHAMP_GAP_PCT, 4),
        "tours_changed_of_200": n_tours_changed,
        "note": "exact preservation does NOT hold "
                "for the noisy genome — the noise "
                "on the new copies moves the "
                "function, by the sponsor's design",
    },
    "genome_file": "frankenstein_stepC_widened003_"
                   "genome.npz (arrays + "
                   "trainable_mask + frozen_idx + "
                   "trainable_idx + meta)",
}
with open(HERE / "engine_runs" / "frankenstein_replication"
                 / "frankenstein_rep_stepC_widen003_"
                 "2026-11-27.json", "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_rep_stepC_widen003_2026-11-27.json")
log(f"DONE stage 1 widen: no-noise gate passed "
    f"(gap {wide_gap*100:.6f}%, tours identical); "
    f"with-noise birth gap {noisy_gap*100:.6f}% "
    f"(subject {CHAMP_GAP_PCT:.4f}%); genes "
    f"{n_total} = {n_frozen} frozen + {n_train} "
    f"trainable")
