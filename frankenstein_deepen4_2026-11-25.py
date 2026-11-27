#!/usr/bin/env python3
"""Deepen Frankenstein 4 (sponsor direction,
2026-11-25): insert ONE new BERT block into the
EVOLVED Frankenstein 4 champion, after the joining
block and before the decoder, initialised as an
exact pass-through.

STARTING POINT: the evolved Frankenstein 4
champion (engine_runs/frankenstein4_evolution/
frankenstein_4_evolved_champion.npz), validation
gap 6.8959% on the fixed 200-problem set. Its
network: entry projection, block 1 (stream width
4, FFN 6), joining block (width 4, FFN 6), merged
decoder. (The task text said FFN 16; the genome
itself says the joining block's FFN width is 6 --
b1.W_1 is (4, 6). The new block copies the joining
block's shapes EXACTLY, which is the operative
instruction: "same shape as the joining block".)

NEW BLOCK (key prefix b2, same per-block key set
and relative order as b0/b1):
  - every array drawn tiny random, std 0.03, ONE
    draw sequence from one recorded seed
    (DEEPEN_SEED), in canonical key order;
  - EXCEPT the two output projections, which are
    EXACTLY ZERO: attention W_o and b_o, and
    feed-forward W_2 and b_2.
  In a pre-norm block the stream itself is never
  normed: A -> A + (attention out) then
  A -> A + (FFN out). With both output projections
  zero, both adds add exactly 0.0, so the block
  computes the identity on the stream and the
  deepened network computes the IDENTICAL function
  as the champion.

GATE: the deepened network, scored on the fixed
validation set, must reproduce the champion
exactly: identical tours on all 200 problems and
the same gap (6.8959%). If not, this script stops
and reports the mismatch; no genome is saved and
nothing evolves.

Outputs (new files only):
  frankenstein_4_deepened_genome.npz
  frankenstein_deepen4_2026-11-25.json
  (stdout is the .log, via the caller's redirect)

Run: python3 frankenstein_deepen4_2026-11-25.py
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
SIGMA_NEW = 0.03
DEEPEN_SEED = 20261130
EXPECTED_GAP_PCT = 6.8959
ZERO_KEYS = ("b2.W_o", "b2.b_o", "b2.W_2", "b2.b_2")

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

# per-block key suffixes, in the champion's own
# relative order (taken from its b1 keys)
BLOCK_SUFFIXES = [k[len("b1."):] for k in CHAMP_KEYS
                  if k.startswith("b1.")]
assert len(BLOCK_SUFFIXES) == 16, BLOCK_SUFFIXES
B2_KEYS = [f"b2.{s}" for s in BLOCK_SUFFIXES]

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
# Build the new block b2: one seed, draws in
# canonical key order, output projections zero
# ================================================================
rng = np.random.default_rng(DEEPEN_SEED)
B2_W = {}
for k in B2_KEYS:
    shape = CHAMP_W[f"b1.{k[len('b2.'):]}"].shape
    if k in ZERO_KEYS:
        B2_W[k] = np.zeros(shape)
    else:
        B2_W[k] = rng.normal(0.0, SIGMA_NEW, size=shape)
n_new = int(sum(v.size for v in B2_W.values()))
n_zero = int(sum(B2_W[k].size for k in ZERO_KEYS))
log(f"new block b2 drawn: seed {DEEPEN_SEED}, std "
    f"{SIGMA_NEW}; {n_new} genes, of which {n_zero} "
    f"are the zeroed output projections "
    f"(W_o, b_o, W_2, b_2); shapes copied from the "
    f"joining block (stream 4, FFN "
    f"{CHAMP_W['b1.W_1'].shape[1]})")

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
# GATE: the deepened network must reproduce the
# champion EXACTLY (identical tours, same gap)
# ================================================================
deep_gap, deep_picks = gap_and_picks(
    DEEP_W, X_val, ("b0", "b1", "b2"))
tours_identical = bool((deep_picks == champ_picks).all())
gap_diff = abs(deep_gap - champ_gap)
log(f"GATE: deepened validation gap "
    f"{deep_gap*100:.6f}% vs champion "
    f"{champ_gap*100:.6f}% (diff {gap_diff:.3e}); "
    f"tours identical on all 200 problems: "
    f"{tours_identical}")
gate_pass = tours_identical and gap_diff < 1e-12

results = {
    "design": "deepen the evolved Frankenstein 4 "
              "champion by one block (sponsor "
              "direction, 2026-11-25)",
    "base": "engine_runs/frankenstein4_evolution/"
            "frankenstein_4_evolved_champion.npz",
    "new_block": {
        "prefix": "b2",
        "position": "after the joining block (b1), "
                    "before the decoder",
        "shape_source": "joining block arrays copied "
                        "for shape: stream width 4, "
                        "FFN width "
                        f"{CHAMP_W['b1.W_1'].shape[1]}, "
                        "1 head (the joining block's "
                        "single-head layout)",
        "note_on_ffn_width": "the task text said "
                             "FFN 16; the genome's "
                             "joining block is FFN 6 "
                             "(b1.W_1 is (4, 6)). The "
                             "new block copies the "
                             "joining block's shapes "
                             "exactly.",
        "init": "all arrays tiny random std 0.03, "
                "one draw sequence, seed "
                f"{DEEPEN_SEED}, canonical key "
                "order; EXCEPT W_o, b_o, W_2, b_2 "
                "exactly zero, so the pre-norm block "
                "adds exactly nothing (exact "
                "pass-through)",
        "zeroed_arrays": list(ZERO_KEYS),
        "genes": n_new,
        "genes_zeroed": n_zero,
    },
    "gene_counts": {"total": n_total,
                    "new_block": n_new,
                    "frozen_champion": n_frozen},
    "gate": {
        "champion_gap_pct_recomputed":
            champ_gap * 100,
        "champion_gap_pct_recorded": EXPECTED_GAP_PCT,
        "deepened_gap_pct": deep_gap * 100,
        "gap_abs_diff": gap_diff,
        "tours_identical_all_200": tours_identical,
        "passed": gate_pass,
    },
    "gene_order": DEEP_KEYS,
    "attn_scale": SCALE, "ln_eps": EPS,
}
with open(HERE / "frankenstein_deepen4_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_deepen4_2026-11-25.json")

if not gate_pass:
    log("GATE FAILED: the deepened network does NOT "
        "reproduce the champion exactly. STOPPING; "
        "no genome saved, nothing evolves.")
    sys.exit(3)

meta = {
    "kind": "frankenstein_4_deepened",
    "date": "2026-11-25",
    "base": "engine_runs/frankenstein4_evolution/"
            "frankenstein_4_evolved_champion.npz",
    "deepening": "one new block b2 inserted after "
                 "the joining block, before the "
                 "decoder; shapes copied from the "
                 "joining block; all arrays tiny "
                 f"random std {SIGMA_NEW} (seed "
                 f"{DEEPEN_SEED}) except W_o, b_o, "
                 "W_2, b_2 exactly zero (exact "
                 "pass-through at birth)",
    "blocks": ["b0", "b1", "b2"],
    "start_gap_pct": EXPECTED_GAP_PCT,
    "gate": "deepened network reproduces the "
            "champion's validation tours identically "
            "and its gap to <1e-12",
    "gene_order": DEEP_KEYS,
    "genes_total": n_total,
    "genes_new_block": n_new,
    "genes_frozen": n_frozen,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_4_deepened_genome.npz",
         **{k: DEEP_W[k] for k in DEEP_KEYS},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_4_deepened_genome.npz")
log(f"DONE deepen: gate passed, deepened genome "
    f"birth gap {deep_gap*100:.4f}% == champion "
    f"{EXPECTED_GAP_PCT:.4f}%")
