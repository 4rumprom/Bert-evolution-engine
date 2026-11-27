#!/usr/bin/env python3
# REPLICATION (2026-11-27): adapted copy of
# frankenstein_stepB_deepen_2026-11-26.py - subject is the
# replication step A champion (824 genes, 7.5641%), fill seed
# 20261320, outputs under engine_runs/frankenstein_replication/.
# See frankenstein_replication_log_2026-11-27.md.
"""STEP B (construction) of the deep-narrow
Frankenstein cycle (2026-11-26): stack NARROW
depth on the back of the Step A2 Frankenstein.

Sponsor's cycle: depth arrives narrow; width
arrives only by stitching. Step B appends ONE
new block at the back of the Step A2 evolved
champion - after the joining block (b4),
before the merged decoder - and this script
only CONSTRUCTS the deepened network and
measures its birth gap. Breeding the new
block in is frankenstein_stepB_evolve_2026-
11-26.py.

SUBJECT: engine_runs/frankenstein_stepA2_
evolution/frankenstein_stepA2_evolved_
champion.npz (788 genes; stream width 4;
blocks b0..b3 stitched with FFN widths
[7, 5, 6, 5], b4 joining block FFN 5, merged
decoder; validation gap 8.0009% on the fixed
200-problem set, val_seed 20261116).

NEW BLOCK (b5): NARROW - 1 attention head
(the Frankenstein block form carries a single
fused attention projection set), FFN width 4,
stream width stays 4. This brings the network
to 6 blocks deep (the sponsor's TSP depth cap
is about 6). Gene count: 136 (lnA/lnF gains
and biases 4x4=16; W_q/W_k/W_v/W_o 4x16=64;
b_q/b_k/b_v/b_o/b_2 5x4=20; W_1 16 + b_1 4 +
W_2 16 = 36; total 136).

WEIGHTS: all tiny-random filler, std 0.03,
ONE rng stream, seed 20261220, drawn in the
joining block's exact order from the stitch
build (frankenstein_stepA2_stitch_2026-11-
26.py): lnA_gain = 1 + N(0, 0.01) (the
reference layer-norm gain convention),
lnA_bias, lnF_gain = 1 + N(0, 0.01), lnF_bias,
W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o, b_2,
W_1, b_1, W_2. The subject's 788 genes are
copied bit-exact and are FROZEN from here on
(the evolve script asserts it).

GENE ORDER: the step A2 order (from
frankenstein_stepA2_genome.npz) for the 788
trunk genes, then the b5 entries in the same
per-block order the trunk blocks use. Flat
indices 0..787 = trunk, 788..923 = new block.

VERIFICATION: (1) the subject champion, run
through the 5-block forward, must reproduce
its recorded 8.0009% before anything is
built; (2) every trunk array in the deepened
genome must be bit-identical to the subject's;
(3) the deepened network's birth gap is
measured on the fixed validation set (it is
expected to be much worse than 8.0009% - the
new block is filler; that is recorded, not
treated as a failure).

OUTPUTS: frankenstein_stepB_genome.npz and
frankenstein_stepB_deepen_2026-11-26.json.
Nothing existing is overwritten.
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
SUBJECT = (HERE / "engine_runs" / "frankenstein_replication"
           / "stepA_evolution"
           / "frankenstein_rep_stepA_evolved_champion.npz")
SUBJECT_GAP_PCT = 7.5641
FILL_SEED = 20261320
SIGMA_FILL = 0.03
LN_GAIN_SIGMA = 0.01
D = 4                 # stream width (unchanged)
U_NEW = 4             # new block FFN width (narrow)
NEW_BLOCK = "b5"
TRUNK_BLOCKS = tuple(f"b{k}" for k in range(5))       # b0..b4
ALL_BLOCKS = TRUNK_BLOCKS + (NEW_BLOCK,)

# ================================================================
# Setup: domain (fixed validation set)
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

# ================================================================
# Load the subject champion (by name, in the
# step A2 gene order)
# ================================================================
zg = np.load(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_genome.npz")
KEYS = [k for k in zg.files if k != "meta"]
zs = np.load(SUBJECT)
assert set(k for k in zs.files if k != "meta") == set(KEYS), \
    "subject champion keys do not match the step A2 gene order"
SUBJ_W = {k: np.asarray(zs[k], dtype=float) for k in KEYS}
subj_flat = np.concatenate([SUBJ_W[k].ravel() for k in KEYS])
N_TRUNK = subj_flat.size
assert N_TRUNK == 824, f"subject has {N_TRUNK} genes, expected 824 (replication step A total)"
log(f"subject loaded: {N_TRUNK} genes from {SUBJECT.name}")


def unpack(vec, keys, shapes):
    W = {}
    off = 0
    for k in keys:
        n = int(np.prod(shapes[k]))
        W[k] = vec[off:off + n].reshape(shapes[k])
        off += n
    return W


SHAPES = {k: SUBJ_W[k].shape for k in KEYS}


# ================================================================
# Forward + decode (the Frankenstein builds'
# conventions, block list parameterized)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X, blocks):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in blocks:
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


def mean_gap(W, X, opt, blocks):
    Y = forward_general(W, X, blocks)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean())


# ================================================================
# VERIFY (1): the subject reproduces its
# recorded gap through the 5-block forward
# ================================================================
subj_gap = mean_gap(SUBJ_W, X_val, OPT_VAL, TRUNK_BLOCKS)
log(f"subject check: measured gap {subj_gap*100:.6f}% vs "
    f"recorded {SUBJECT_GAP_PCT:.4f}%")
assert abs(subj_gap * 100 - SUBJECT_GAP_PCT) <= 5e-5 + 1e-9, \
    "subject does not reproduce its recorded gap; STOPPING"
log("subject check passed")

# ================================================================
# Build the new block (one rng stream, seed
# 20261220, joining-block draw order)
# ================================================================
rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


NB = {}
NB[f"{NEW_BLOCK}.lnA_gain"] = 1.0 + rng_fill.normal(
    0.0, LN_GAIN_SIGMA, D)
NB[f"{NEW_BLOCK}.lnA_bias"] = fill((D,))
NB[f"{NEW_BLOCK}.lnF_gain"] = 1.0 + rng_fill.normal(
    0.0, LN_GAIN_SIGMA, D)
NB[f"{NEW_BLOCK}.lnF_bias"] = fill((D,))
for nm in ("W_q", "W_k", "W_v", "W_o"):
    NB[f"{NEW_BLOCK}.{nm}"] = fill((D, D))
for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
    NB[f"{NEW_BLOCK}.{nm}"] = fill((D,))
NB[f"{NEW_BLOCK}.W_1"] = fill((D, U_NEW))
NB[f"{NEW_BLOCK}.b_1"] = fill((U_NEW,))
NB[f"{NEW_BLOCK}.W_2"] = fill((U_NEW, D))

# b5 keys in the same per-block order the
# trunk blocks use in KEYS
NEW_KEYS = [f"{NEW_BLOCK}.{s}" for s in (
    "lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
    "W_q", "W_k", "W_v", "W_o",
    "b_q", "b_k", "b_v", "b_o", "b_2",
    "W_1", "W_2", "b_1")]
assert set(NEW_KEYS) == set(NB), "new-block key set mismatch"
N_NEW = int(sum(NB[k].size for k in NEW_KEYS))
log(f"new block {NEW_BLOCK}: {N_NEW} genes "
    f"(1 head, FFN width {U_NEW}, stream width {D}), "
    f"filler std {SIGMA_FILL}, seed {FILL_SEED}")
assert N_NEW == 136, f"new block has {N_NEW} genes, expected 136"

# ================================================================
# Assemble the deepened genome + VERIFY (2):
# trunk bit-identical
# ================================================================
KEYS_B = KEYS + NEW_KEYS
W_B = dict(SUBJ_W)
for k in NEW_KEYS:
    W_B[k] = NB[k]
for k in KEYS:
    assert np.array_equal(W_B[k], SUBJ_W[k]), \
        f"trunk array {k} is not bit-identical to the subject"
flat_B = np.concatenate([W_B[k].ravel() for k in KEYS_B])
assert flat_B.size == N_TRUNK + N_NEW
assert np.array_equal(flat_B[:N_TRUNK], subj_flat), \
    "flat trunk prefix is not bit-identical to the subject"
log(f"deepened genome: {flat_B.size} genes "
    f"({N_TRUNK} trunk + {N_NEW} new block); trunk "
    f"verified bit-identical to the subject champion")

# ================================================================
# VERIFY (3): birth gap of the deepened
# network on the fixed validation set
# ================================================================
birth_gap = mean_gap(W_B, X_val, OPT_VAL, ALL_BLOCKS)
log(f"BIRTH GAP of the deepened network: "
    f"{birth_gap*100:.4f}% (subject: "
    f"{SUBJECT_GAP_PCT:.4f}%; the new block is filler, "
    f"a much worse birth gap is expected and is not "
    f"a failure)")

# ================================================================
# Save genome + construction record
# ================================================================
meta = {
    "kind": "frankenstein_rep_stepB_genome",
    "date": "2026-11-27",
    "step": "B (construction): one narrow block "
            "appended at the back of the step A2 "
            "Frankenstein",
    "subject": "engine_runs/frankenstein_replication/"
               "stepA_evolution/frankenstein_rep_stepA_"
               "evolved_champion.npz",
    "subject_gap_pct": SUBJECT_GAP_PCT,
    "subject_gap_pct_measured": round(subj_gap * 100, 6),
    "new_block": NEW_BLOCK,
    "new_block_position": "after the joining block "
                          "(b4), before the merged "
                          "decoder",
    "new_block_shape": {"heads": 1, "ffn_hidden": U_NEW,
                        "stream_width": D},
    "blocks_total": len(ALL_BLOCKS),
    "blocks": list(ALL_BLOCKS),
    "fill": {"sigma": SIGMA_FILL, "seed": FILL_SEED,
             "ln_gain_convention": "1 + N(0, 0.01), "
                                   "the joining-block "
                                   "reference convention",
             "draw_order": "lnA_gain, lnA_bias, "
                           "lnF_gain, lnF_bias, W_q, "
                           "W_k, W_v, W_o, b_q, b_k, "
                           "b_v, b_o, b_2, W_1, b_1, "
                           "W_2"},
    "birth_gap_pct": round(birth_gap * 100, 4),
    "gene_order": KEYS_B,
    "genes_total": int(flat_B.size),
    "genes_trunk": int(N_TRUNK),
    "genes_new_block": int(N_NEW),
    "trunk_flat_range": [0, int(N_TRUNK)],
    "new_block_flat_range": [int(N_TRUNK),
                             int(flat_B.size)],
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepB_genome.npz",
         **{k: W_B[k] for k in KEYS_B},
         meta=np.array(json.dumps(meta)))
record = {
    "step": "B construction (deepen)",
    "date": "2026-11-27",
    "subject": meta["subject"],
    "subject_gap_pct_recorded": SUBJECT_GAP_PCT,
    "subject_gap_pct_measured": round(subj_gap * 100, 6),
    "subject_reproduces_recorded_gap": True,
    "trunk_genes": int(N_TRUNK),
    "trunk_bit_identical_to_subject": True,
    "new_block": NEW_BLOCK,
    "new_block_genes": int(N_NEW),
    "new_block_shape": meta["new_block_shape"],
    "total_genes": int(flat_B.size),
    "blocks": list(ALL_BLOCKS),
    "fill_seed": FILL_SEED,
    "fill_sigma": SIGMA_FILL,
    "birth_gap_pct": round(birth_gap * 100, 4),
    "note": "The birth gap is expected to be much "
            "worse than the subject's 8.0009%: the "
            "appended block is tiny-random filler "
            "that has not been bred in yet. Breeding "
            "is frankenstein_stepB_evolve_2026-11-"
            "26.py (new block's genes only, trunk "
            "frozen).",
}
with open(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepB_deepen_2026-11-27.json",
          "w") as f:
    json.dump(record, f, indent=1)
log("saved frankenstein_rep_stepB_genome.npz and "
    "frankenstein_rep_stepB_deepen_2026-11-27.json")
log(f"DONE: birth gap {birth_gap*100:.4f}%")
