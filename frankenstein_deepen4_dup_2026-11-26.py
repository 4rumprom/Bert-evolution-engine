#!/usr/bin/env python3
"""Deepen Frankenstein 4 by DUPLICATING a proven
block (sponsor direction, 2026-11-26): insert ONE
new BERT block into the EVOLVED Frankenstein 4
champion, after the joining block and before the
decoder, as an EXACT COPY of the champion's b0
block (the main block).

WHY NOT IDENTITY: the sponsor rejected the
identity block ("Not the identity though, that's
dumb"). The identity deepen (frankenstein_
deepen4_2026-11-25.py) passed its preservation
gate bit-for-bit and then earned nothing in 100
generations of evolution: a block that starts as
dead weight stayed dead weight. This build does
the opposite: the new depth starts as a working
piece, a second pass of a transformation that
already earned its place.

STARTING POINT: the evolved Frankenstein 4
champion (engine_runs/frankenstein4_evolution/
frankenstein_4_evolved_champion.npz), validation
gap 6.8959% on the fixed 200-problem set. Its
network: entry projection, block b0 (main,
stream width 4, FFN 6), joining block b1 (width
4, FFN 6), merged decoder.

NEW BLOCK (key prefix b2, same per-block key
set as b0/b1):
  - every array an EXACT COPY of the champion's
    b0 array with the same suffix: weights,
    biases, layer-norm gains and biases. b0 and
    b1 have the same 16 suffixes (same shapes);
    the copy is by suffix, not by position, since
    the two blocks' key orders differ slightly.
  - 154 genes, same count and shapes as the
    identity arm's new block.
This is NOT function-preserving by design:
applying the proven transformation twice
changes the function. There is no preservation
gate. Instead the build verifies the network
runs, produces valid permutations on all 200
validation problems, and reports the birth
score once.

MASK: the 344 original genes are backbone
(frozen); the 154 new-block genes are the
evolving set (for the companion evolution
script, not run here).

Outputs (new files only):
  frankenstein_4_deepened_dup_genome.npz
  frankenstein_deepen4_dup_2026-11-26.json
  (stdout is the .log, via the caller's redirect)

Run: python3 frankenstein_deepen4_dup_2026-11-26.py
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


CFG = HERE / "bert_engine_config_fresh50.json"
CHAMP = (HERE / "engine_runs" / "frankenstein4_evolution"
         / "frankenstein_4_evolved_champion.npz")
EXPECTED_GAP_PCT = 6.8959
COPIED_BLOCK = "b0"

# ================================================================
# Setup: domain (validation set), champion
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

zc = np.load(CHAMP)
CMETA = json.loads(str(zc["meta"]))
CHAMP_KEYS = list(CMETA["gene_order"])
CHAMP_W = {k: np.asarray(zc[k], dtype=float)
           for k in CHAMP_KEYS}
assert sum(v.size for v in CHAMP_W.values()) == 344

# per-block key suffixes: b0 and b1 must carry the
# same 16 suffixes (same shapes); their key
# ORDERS differ slightly, so the copy is by suffix.
SUF_B0 = [k[len("b0."):] for k in CHAMP_KEYS
          if k.startswith("b0.")]
SUF_B1 = [k[len("b1."):] for k in CHAMP_KEYS
          if k.startswith("b1.")]
assert len(SUF_B0) == 16 and len(SUF_B1) == 16
assert sorted(SUF_B0) == sorted(SUF_B1), (SUF_B0, SUF_B1)
for s in SUF_B0:
    assert CHAMP_W[f"b0.{s}"].shape == \
        CHAMP_W[f"b1.{s}"].shape, s
# b2 keys follow b1's suffix order, exactly as in
# the identity deepen (so the two deepened genomes
# share gene order)
B2_KEYS = [f"b2.{s}" for s in SUF_B1]

# deepened gene order: champion order with the b2
# keys inserted after the last b1 key, before head.*
DEEP_KEYS = []
for k in CHAMP_KEYS:
    DEEP_KEYS.append(k)
    if k == "b1.W_2":
        DEEP_KEYS.extend(B2_KEYS)
assert len(DEEP_KEYS) == len(CHAMP_KEYS) + 16


# ================================================================
# Forward + decode (verbatim conventions from the
# Frankenstein builds / evolve4)
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


def gap_and_picks(W, X, blocks):
    Y = forward_general(W, X, blocks)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / OPT_VAL - 1.0).mean()), picks


# ================================================================
# Champion reference score (recomputed here, same
# forward pass, blocks b0+b1)
# ================================================================
champ_gap, champ_picks = gap_and_picks(
    CHAMP_W, X_val, ("b0", "b1"))
log(f"champion recomputed: validation gap "
    f"{champ_gap*100:.6f}% (recorded "
    f"{EXPECTED_GAP_PCT:.4f}%), tours on "
    f"{champ_picks.shape[0]} problems")
assert round(champ_gap * 100, 4) == EXPECTED_GAP_PCT, \
    champ_gap * 100

# ================================================================
# Build the new block b2: an EXACT COPY of b0
# ================================================================
B2_W = {}
for k in B2_KEYS:
    sfx = k[len("b2."):]
    B2_W[k] = np.array(CHAMP_W[f"{COPIED_BLOCK}.{sfx}"],
                       dtype=float, copy=True)
n_new = int(sum(v.size for v in B2_W.values()))
# every copied array bit-identical to its source
for k in B2_KEYS:
    sfx = k[len("b2."):]
    assert np.array_equal(B2_W[k],
                          CHAMP_W[f"{COPIED_BLOCK}.{sfx}"]), k
log(f"new block b2: exact copy of the champion's "
    f"{COPIED_BLOCK} block (main block, stream 4, "
    f"FFN {CHAMP_W['b0.W_1'].shape[1]}, 1 head); "
    f"{n_new} genes, all bit-identical to their "
    f"source arrays")

DEEP_W = {}
for k in DEEP_KEYS:
    DEEP_W[k] = B2_W[k] if k.startswith("b2.") \
        else CHAMP_W[k]
n_total = int(sum(v.size for v in DEEP_W.values()))
n_frozen = n_total - n_new
log(f"gene counts: total {n_total}, new block "
    f"{n_new}, frozen (champion's) {n_frozen}")
assert n_total == 498 and n_new == 154 and n_frozen == 344

# every champion array must be carried over untouched
for k in CHAMP_KEYS:
    assert np.array_equal(DEEP_W[k], CHAMP_W[k]), k

# ================================================================
# BIRTH MEASUREMENT (no preservation gate: the
# duplicated block changes the function by design)
# ================================================================
deep_gap, deep_picks = gap_and_picks(
    DEEP_W, X_val, ("b0", "b1", "b2"))
# validity: every tour must be a permutation of
# the 10 cities
n_prob, n_cities = deep_picks.shape
sorted_picks = np.sort(deep_picks, axis=1)
valid = bool((sorted_picks
              == np.arange(n_cities)[None, :]).all())
assert valid, "deepened network produced an invalid tour"
log(f"BIRTH: duplicated-block deepened network "
    f"validation gap {deep_gap*100:.6f}% vs champion "
    f"{champ_gap*100:.6f}% (delta "
    f"{(deep_gap - champ_gap)*100:+.4f} points); "
    f"valid permutations on all {n_prob} problems: "
    f"{valid}")

results = {
    "design": "deepen the evolved Frankenstein 4 "
              "champion by duplicating a proven block "
              "(sponsor direction, 2026-11-26): the "
              "identity block was rejected, so the new "
              "depth starts as a working piece",
    "base": "engine_runs/frankenstein4_evolution/"
            "frankenstein_4_evolved_champion.npz",
    "new_block": {
        "prefix": "b2",
        "position": "after the joining block (b1), "
                    "before the decoder",
        "copied_block": COPIED_BLOCK,
        "copy_rule": "every array of b2 is an exact "
                     "copy of the champion's b0 array "
                     "with the same suffix (weights, "
                     "biases, layer-norm gains and "
                     "biases), bit-identical; b0 was "
                     "chosen over b1 because b0 is "
                     "the main processing block and "
                     "b1 is the joining block, which "
                     "stays unique",
        "shape": "stream width 4, FFN width "
                 f"{CHAMP_W['b0.W_1'].shape[1]}, "
                 "1 head (b0's layout)",
        "genes": n_new,
    },
    "gene_counts": {"total": n_total,
                    "new_block": n_new,
                    "frozen_champion": n_frozen},
    "birth": {
        "champion_gap_pct_recomputed":
            champ_gap * 100,
        "champion_gap_pct_recorded": EXPECTED_GAP_PCT,
        "deepened_gap_pct": deep_gap * 100,
        "delta_points_vs_champion":
            (deep_gap - champ_gap) * 100,
        "valid_permutations_all_200": valid,
    },
    "note_vs_identity_deepen": "the identity arm "
        "(frankenstein_deepen4_2026-11-25.py) "
        "inserted the same 154-gene slot as an exact "
        "pass-through, passed its preservation gate "
        "bit-for-bit (6.8959%), and then never beat "
        "the founder in 100 generations of evolution "
        "of the new block. The duplicated block is "
        "NOT function-preserving: it changes the "
        "network at birth, by a second pass of a "
        "proven transformation.",
    "gene_order": DEEP_KEYS,
    "attn_scale": SCALE, "ln_eps": EPS,
}
with open(HERE / "frankenstein_deepen4_dup_2026-11-26.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_deepen4_dup_2026-11-26.json")

meta = {
    "kind": "frankenstein_4_deepened_dup",
    "date": "2026-11-26",
    "base": "engine_runs/frankenstein4_evolution/"
            "frankenstein_4_evolved_champion.npz",
    "deepening": "one new block b2 inserted after "
                 "the joining block, before the "
                 "decoder; every array an exact copy "
                 "of the champion's b0 block "
                 "(main block, stream 4, FFN 6); "
                 "NOT function-preserving by design",
    "blocks": ["b0", "b1", "b2"],
    "birth_gap_pct": round(deep_gap * 100, 4),
    "champion_gap_pct": EXPECTED_GAP_PCT,
    "gene_order": DEEP_KEYS,
    "genes_total": n_total,
    "genes_new_block": n_new,
    "genes_frozen": n_frozen,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_4_deepened_dup_genome.npz",
         **{k: DEEP_W[k] for k in DEEP_KEYS},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_4_deepened_dup_genome.npz")
log(f"DONE deepen-dup: birth gap {deep_gap*100:.4f}% "
    f"vs champion {EXPECTED_GAP_PCT:.4f}% (delta "
    f"{(deep_gap - champ_gap)*100:+.4f} points)")
