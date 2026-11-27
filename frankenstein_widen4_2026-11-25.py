#!/usr/bin/env python3
"""WIDEN the evolved Frankenstein 4 champion by the
sponsor's default widen rule (d x2, FFN x2, heads x2,
Net2WiderNet splits), applied across the WHOLE
network, with a hard preservation gate.

SUBJECT: engine_runs/frankenstein4_evolution/
frankenstein_4_evolved_champion.npz (validation gap
6.8959% on the fixed 200-problem set). Shape,
confirmed from the genome itself: entry projection
(2 -> 4); block b0 (stream 4, FFN 6); joining block
b1 (stream 4, FFN 6); merged pointer decoder over
width 4. Each block stores ONE fused attention set
(W_q/W_k/W_v/W_o, each (4,4)); the forward pass of
the Frankenstein builds computes that single
attention and adds Ah @ W_o + b_o.

THE WIDEN (all of it function-preserving; the gate
below is the binding check):
  - Stream d 4 -> 8 by duplication: new stream
    coords 4..7 are copies of coords 0..3. Every
    array follows one uniform rule: on an OUTPUT
    side of the widened stream the axis is
    duplicated; on an INPUT side of the widened
    stream the axis is split evenly (each copy
    carries half), because the duplicated input
    coordinates sum into every consumer.
  - attn_scale: HALVED (SCALE/2). Both dot products
    of the forward pass -- attention scores and
    decoder logits -- run over the duplicated
    stream and would otherwise double exactly;
    halving the one shared scale restores them.
    (The Frankenstein forward uses the same SCALE
    for attention and decoder, recorded in the
    genome meta; the widened meta records SCALE/2.)
  - ln_eps: UNCHANGED. Duplication preserves the
    layer norm mean and variance over the stream
    exactly (every coordinate appears twice), so
    unlike the engine's mean-based widen no eps
    rescaling is needed. LN gains/biases are
    duplicated.
  - Heads 2 -> 4 per block, Net2WiderNet split.
    The stored block holds one fused attention;
    it is first decomposed into the sponsor's 2
    heads by the split itself (both heads carry
    the full W_q/W_k/W_v; W_o and b_o are halved
    between them -- head outputs are summed, so
    the decomposition is exact), then each of the
    2 heads is split again: Q/K/V copied exactly,
    W_o and b_o halved in both daughters. Net
    effect per block: 4 identical heads, each with
    the widened W_q/W_k/W_v and W_o/4, b_o/4 of
    the widened originals. Heads 0,1 are the base
    heads; heads 2,3 are their appended copies
    (h2 of h0, h3 of h1), per the codebase's
    split placement (copies appended after the
    originals, as in transplant_split_widen and
    double_widen_d2).
  - FFN 6 -> 12 per block, Net2WiderNet split
    (transplant_split_widen convention): each
    unit's W_1 column and b_1 copied exactly into
    the appended copy unit; the unit's outgoing
    W_2 row is halved in both copies. (The stream
    input-split also applies to W_1's rows and
    W_2's columns per the uniform rule above.)
  - Decoder: W_qd (8,8) by the uniform rule
    (input side split, output side duplicated),
    so q is the duplicated old q and, with the
    halved scale, the logits are unchanged;
    b_qd and v_start duplicated.
  - NO split noise. The codebase's engine runner
    applies split noise to copies AFTER its
    preservation gate; here the task's gate
    requires the evolution founder to be the
    exact widened champion, and evolution's own
    mutation (p = 0.10 on the added genes) breaks
    the twin symmetry from the first child.

BASE / ADDED (for the evolution mask), the union
rule mirroring the engine's class assignment
(any entry touching a new index is interface):
a gene is BASE iff every widened axis it touches
is an original index (stream coords 0..3, FFN
units 0..5, heads 0..1); otherwise it is ADDED
(evolving). Hand count: total 2896, base 504,
added 2392 -- the script asserts its derived
mask against these numbers.

PRESERVATION GATE (hard): the widened network,
scored with the widened forward pass on the fixed
validation set, must produce IDENTICAL tours to
the champion on all 200 problems (hence the same
gap, 6.8959%). If the gate fails this script
stops and no evolution is launched from it.

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_widen4_2026-11-25.py
Outputs: frankenstein_4_widened_genome.npz,
frankenstein_widen4_2026-11-25.json (the .log is
this script's stdout, redirected by the caller).
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
CHAMP_GAP_PCT = 6.8959
D0, D1 = 4, 8
U0, U1 = 6, 12
H0, H1 = 2, 4

# ================================================================
# Setup (identical objects to the evolve4 run)
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps
SCALE_W = SCALE / 2.0
log(f"attn_scale {SCALE!r} -> widened {SCALE_W!r} "
    f"(halved: dot products run over the duplicated "
    f"stream); ln_eps {EPS!r} unchanged")

zc = np.load(CHAMP)
META_C = json.loads(str(zc["meta"]))
KEYS0 = [k for k in zc.files if k != "meta"]
W0 = {k: np.asarray(zc[k], dtype=float) for k in KEYS0}
assert META_C["best_validation_gap_pct"] == CHAMP_GAP_PCT
log(f"champion loaded: {sum(v.size for v in W0.values())} "
    f"genes, recorded gap {CHAMP_GAP_PCT}%")


# ================================================================
# Champion forward (verbatim conventions from
# frankenstein_evolve4_2026-11-25.py)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_champ(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1"):
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


def gap_champ(W, X):
    Y = forward_champ(W, X)
    picks = decode(W, Y, W["head.W_qd"], W["head.b_qd"],
                   W["head.v_start"], SCALE)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / OPT_VAL - 1.0).mean()), picks


champ_gap, champ_picks = gap_champ(W0, X_val)
log(f"champion re-scored under the evolve4 forward: "
    f"{champ_gap*100:.6f}% (recorded {CHAMP_GAP_PCT}%)")
assert abs(champ_gap * 100 - CHAMP_GAP_PCT) < 5e-5 + 1e-9

# ================================================================
# Widened forward (multi-head, summed head outputs;
# engine semantics)
# ================================================================
def forward_wide(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1"):
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        out = 0.0
        for h in range(H1):
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
# The widen construction
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

for blk in ("b0", "b1"):
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
        W8[f"{blk}.{nm}"] = dup_vec(W0[f"{blk}.{nm}"])
    # heads: 4 identical heads (see module docstring).
    # W_q/W_k/W_v: split_mat of the originals (copied
    # exactly across heads). W_o: split_mat then /4 --
    # /2 for the 1->2 head decomposition, /2 for the
    # 2->4 split. b_o: duplicated then /4.
    for h in range(H1):
        p = f"{blk}.h{h}"
        for nm in ("W_q", "W_k", "W_v"):
            W8[f"{p}.{nm}"] = split_mat(W0[f"{blk}.{nm}"])
        W8[f"{p}.W_o"] = split_mat(W0[f"{blk}.W_o"]) / 4.0
        for nm in ("b_q", "b_k", "b_v"):
            W8[f"{p}.{nm}"] = dup_vec(W0[f"{blk}.{nm}"])
        W8[f"{p}.b_o"] = dup_vec(W0[f"{blk}.b_o"]) / 4.0
    # FFN: W_1 (d, U): input rows split, unit columns
    # duplicated -> every copy block is W_1/2.
    W1 = np.zeros((D1, U1))
    for ci in (0, D0):
        for cu in (0, U0):
            W1[ci:ci + D0, cu:cu + U0] = W0[f"{blk}.W_1"] / 2.0
    W8[f"{blk}.W_1"] = W1
    W8[f"{blk}.b_1"] = dup_vec(W0[f"{blk}.b_1"])
    # W_2 (U, d): unit rows halved in both copies
    # (Net2WiderNet split), stream columns duplicated
    # under the split rule -> every copy block W_2/2.
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

KEYS8 = list(W8.keys())

# ================================================================
# Base / added mask (union rule; see docstring)
# ================================================================
def added_for(key, shape):
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
        assert blk in ("b0", "b1")
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
            if h >= H0:
                m[:] = True
            elif nm in ("W_q", "W_k", "W_v", "W_o"):
                m[D0:, :] = True
                m[:, D0:] = True
            else:  # head biases
                m[D0:] = True
        else:
            raise ValueError(f"unhandled key {key}")
    return m


flat_added = np.concatenate([
    added_for(k, W8[k].shape).ravel() for k in KEYS8])
n_total = int(flat_added.size)
n_added = int(flat_added.sum())
n_base = n_total - n_added
log(f"gene counts: total {n_total}, base (frozen) "
    f"{n_base}, added (evolving) {n_added}")
assert (n_total, n_base, n_added) == (2896, 504, 2392), \
    (n_total, n_base, n_added)

# ================================================================
# PRESERVATION GATE (hard)
# ================================================================
wide_gap, wide_picks = gap_wide(W8, X_val)
tours_identical = bool((wide_picks == champ_picks).all())
log(f"PRESERVATION GATE: widened gap "
    f"{wide_gap*100:.6f}% vs champion "
    f"{champ_gap*100:.6f}%; tours identical on all "
    f"200 problems: {tours_identical}")
if not tours_identical or abs(wide_gap - champ_gap) > 1e-9:
    log("GATE FAILED -- widened network does not "
        "reproduce the champion. STOPPING; no genome "
        "saved, no evolution launched.")
    sys.exit(3)
log("GATE PASSED")

# ================================================================
# Save genome + results
# ================================================================
shapes_before = {k: list(W0[k].shape) for k in KEYS0}
shapes_after = {k: list(W8[k].shape) for k in KEYS8}
meta = {
    "kind": "frankenstein_4_widened_genome",
    "date": "2026-11-25",
    "base": "engine_runs/frankenstein4_evolution/"
            "frankenstein_4_evolved_champion.npz",
    "widen": "sponsor default widen across the whole "
             "network: stream 4 -> 8 by duplication, "
             "FFN 6 -> 12 and heads 2 -> 4 per block "
             "by Net2WiderNet split",
    "attn_scale": SCALE_W,
    "attn_scale_note": "halved vs the champion's "
                       "SCALE (dot products over the "
                       "duplicated stream double)",
    "ln_eps": EPS,
    "ln_eps_note": "unchanged (duplication preserves "
                   "LN mean/variance exactly)",
    "heads_per_block": H1,
    "head_note": "the champion stores one fused "
                 "attention set per block; it is "
                 "decomposed into 2 heads by the "
                 "split (W_o, b_o halved) and each "
                 "head split again to 4; all 4 heads "
                 "are identical at the founder",
    "split_noise": "none (see docstring)",
    "copy_placement": "originals first, copies "
                      "appended (stream 4..7, units "
                      "6..11, heads 2..3)",
    "mask_rule": "union: a gene is added iff any "
                 "widened axis it touches is a copy "
                 "index or it belongs to a copy "
                 "unit/head; else base",
    "gene_order": KEYS8,
    "genes_total": n_total, "genes_base": n_base,
    "genes_added": n_added,
    "champion_gap_pct": CHAMP_GAP_PCT,
    "gate": {"tours_identical": tours_identical,
             "widened_gap_pct": round(wide_gap * 100, 6),
             "champion_gap_pct_recomputed":
                 round(champ_gap * 100, 6)},
}
np.savez(HERE / "frankenstein_4_widened_genome.npz",
         **W8, added_mask=flat_added.astype(np.int8),
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_4_widened_genome.npz")

results = {
    "design": "sponsor direction, 2026-11-25 22:20 "
              "EST: widen the evolved Frankenstein 4 "
              "champion (Net2WiderNet), then evolve "
              "only the added material",
    "subject": {
        "file": str(CHAMP),
        "validation_gap_pct": CHAMP_GAP_PCT,
        "genes": int(sum(v.size for v in W0.values())),
    },
    "shapes_before": shapes_before,
    "shapes_after": shapes_after,
    "gene_counts": {"total": n_total, "base_frozen": n_base,
                    "added_evolving": n_added},
    "conventions": {
        "stream": "d 4 -> 8 by duplication; copies "
                  "appended (coords 4..7 copy 0..3); "
                  "output sides of the stream are "
                  "duplicated, input sides are split "
                  "evenly (each copy carries half)",
        "attn_scale": f"{SCALE} -> {SCALE_W} (halved; "
                      "attention-score and decoder "
                      "dot products run over the "
                      "duplicated stream)",
        "ln_eps": f"{EPS} unchanged (duplication "
                  "preserves LN mean/variance)",
        "ln_gains_biases": "duplicated",
        "heads": "2 -> 4 per block by split: the "
                 "stored fused attention is first "
                 "decomposed into 2 heads (full "
                 "W_q/W_k/W_v each; W_o, b_o halved "
                 "between them), then each head is "
                 "split (Q/K/V copied exactly, W_o "
                 "and b_o halved in both daughters). "
                 "All 4 heads identical at the "
                 "founder; heads 0,1 base, heads 2,3 "
                 "appended copies",
        "ffn": "6 -> 12 per block by Net2WiderNet "
               "split (transplant_split_widen "
               "convention): W_1 column and b_1 "
               "copied exactly, W_2 row halved in "
               "both copies; units 0..5 base, 6..11 "
               "appended copies",
        "decoder": "W_qd (8,8) by the uniform "
                   "duplication rule (input split, "
                   "output duplicated); b_qd and "
                   "v_start duplicated",
        "entry": "W_in columns duplicated; b_in "
                 "duplicated",
        "split_noise": "none applied (the founder "
                       "must be the exact widened "
                       "champion for the gate; "
                       "evolution's mutation breaks "
                       "twin symmetry)",
        "base_added_rule": "union rule: added iff "
                           "any widened axis touched "
                           "is a copy index or the "
                           "gene belongs to a copy "
                           "unit/head (mirrors the "
                           "engine's class assignment "
                           "for widen transplants)",
    },
    "preservation_gate": {
        "pass": True,
        "tours_identical_all_200": tours_identical,
        "champion_gap_pct_recomputed":
            round(champ_gap * 100, 6),
        "widened_gap_pct": round(wide_gap * 100, 6),
        "recorded_champion_gap_pct": CHAMP_GAP_PCT,
    },
}
with open(HERE / "frankenstein_widen4_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_widen4_2026-11-25.json")
log("DONE widen")
