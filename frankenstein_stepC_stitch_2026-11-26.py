#!/usr/bin/env python3
"""STEP C (stitch) of the deep-narrow
Frankenstein cycle (2026-11-26): widen the
backend block b5 of the Step B Frankenstein by
stitching the top two champions of the Step B
run side by side inside one block, over their
commonized trunk.

BACKGROUND: Step B appended one narrow block
b5 (1 head, FFN 4, 136 genes) to the Step A2
Frankenstein (788 trunk genes) and bred a
population of 100 for 100 generations mutating
ONLY b5. Every genome in that run shares the
identical frozen 788-gene trunk; they differ
only in b5. The B evolved champion scores
7.9739% on the fixed validation set
(TSPDomain, val_seed 20261116, 200 problems,
10 cities).

STEP 1 - find the two backend champions:
score EVERY genome of the final population
(engine_runs/frankenstein_stepB_evolution/
checkpoint.npz) on the fixed validation set
with the Step B forward machinery. #1 must
reproduce 7.9739% (and be bit-identical to
the checkpoint's best_val_genome). The second
champion is the best-scoring genome not
bit-identical to #1.

STEP 2 - the stitch (commonized trunk):
- trunk (embed, b0..b4, merged decoder; the
  788 genes) kept ONCE, bit-exact from
  champion #1 (identical in both parents).
- the widened backend block b5w replaces b5:
  stream width stays 4; attention = 2 heads
  (head 0 = #1's b5 head, head 1 = #2's b5
  head, every entry bit-exact); FFN width 8
  (units 0-3 = #1's, units 4-7 = #2's, via
  W_1 columns / W_2 rows / b_1 halves).
- the block's layer-norms (lnA, lnF) are
  shared: champion #1's genes are used and
  champion #2's LN genes are DROPPED. The
  FFN output bias b_2 is likewise a single
  block-level parameter: #1's is used and
  #2's is dropped. Both drops are recorded
  in the stitch JSON.
- NO rescaling or halving anywhere; there
  are no filler entries in this stitch.
- gene accounting: backbone = the 788 trunk
  genes; stitching genes = ALL 252 genes of
  b5w (LN 8 + 2 heads x 80 + LN 8 + FFN 76);
  total 1040.

Verification: trunk bit-identical to #1;
every placed branch entry bit-exact against
its source; plus a functional parity check -
the stitch with branch 2 masked out (head 1
output projection zeroed, FFN units 4-7
output rows zeroed) must reproduce champion
#1's tours and gap exactly (its LNs are
#1's). The stitch's BIRTH gap is measured on
the fixed set.

Outputs: frankenstein_stepC_genome.npz,
frankenstein_stepC_stitch_2026-11-26.json,
frankenstein_stepC_state.json. The seam
evolution is frankenstein_stepC_evolve_
2026-11-26.py.
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


CFG = HERE / "bert_engine_config_deep50_stepA.json"
B_OUT = HERE / "engine_runs" / "frankenstein_stepB_evolution"
B_CHAMP_GAP_PCT = 7.9739
N_TRUNK = 788
N_B5 = 136

cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

# ================================================================
# Step B layout + forward machinery (verbatim
# conventions from frankenstein_stepB_evolve)
# ================================================================
zg = np.load(HERE / "frankenstein_stepB_genome.npz")
META_B = json.loads(str(zg["meta"]))
KEYS_B = [k for k in zg.files if k != "meta"]
assert KEYS_B == META_B["gene_order"]
SHAPES_B = {k: np.asarray(zg[k]).shape for k in KEYS_B}
SIZES_B = {k: int(np.asarray(zg[k]).size) for k in KEYS_B}
BLOCKS_B = tuple(f"b{k}" for k in range(6))   # b0..b5


def unpack_B(vec):
    W = {}
    off = 0
    for k in KEYS_B:
        n = SIZES_B[k]
        W[k] = vec[off:off + n].reshape(SHAPES_B[k])
        off += n
    return W


def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_B(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS_B:
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


def decode(W, Y):
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


def gap_B(vec, X, opt):
    W = unpack_B(vec)
    Y = forward_B(W, X)
    picks = decode(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean()), picks


# ================================================================
# STEP 1 - score the whole final B population
# ================================================================
ck = np.load(B_OUT / "checkpoint.npz", allow_pickle=True)
pop = np.asarray(ck["pop"], dtype=float)
best_val_genome = np.asarray(ck["best_val_genome"], dtype=float)
best_val_gap_ck = float(ck["best_val_gap"])
assert pop.shape == (100, N_TRUNK + N_B5), pop.shape
assert int(ck["next_gen"]) == 100

# every member shares the frozen trunk
trunk0 = pop[0, :N_TRUNK]
assert bool((pop[:, :N_TRUNK] == trunk0).all()), \
    "final B population does not share one trunk"
assert np.array_equal(trunk0, best_val_genome[:N_TRUNK])

gaps = np.empty(len(pop))
for i in range(len(pop)):
    gaps[i], _ = gap_B(pop[i], X_val, OPT_VAL)
order = np.argsort(gaps)
i1 = int(order[0])
champ1 = pop[i1]
gap1 = float(gaps[i1])
log(f"population scored: best validation gap "
    f"{gap1*100:.6f}% (pop index {i1}); checkpoint "
    f"best_val_gap {best_val_gap_ck*100:.6f}%")

# SANITY: #1 must reproduce the B champion
if not np.array_equal(champ1, best_val_genome) \
        or abs(gap1 - best_val_gap_ck) > 1e-9 \
        or abs(gap1 * 100 - B_CHAMP_GAP_PCT) > 5e-5:
    log("SANITY FAILED: the best final-population genome "
        f"scores {gap1*100:.6f}%, bit-identical to the "
        f"checkpoint best_val_genome: "
        f"{np.array_equal(champ1, best_val_genome)}; "
        f"expected {B_CHAMP_GAP_PCT:.4f}%. STOPPING - "
        f"no stitch.")
    sys.exit(3)
log("sanity passed: #1 is the B evolved champion, "
    f"{gap1*100:.4f}%")

# cross-check against the vaulted champion arrays
zc = np.load(B_OUT / "frankenstein_stepB_evolved_champion.npz")
vault_flat = np.concatenate(
    [np.asarray(zc[k], dtype=float).ravel() for k in KEYS_B])
assert np.array_equal(vault_flat, champ1), \
    "vaulted B champion arrays != checkpoint best genome"

i2 = None
for j in order[1:]:
    if not np.array_equal(pop[int(j)], champ1):
        i2 = int(j)
        break
assert i2 is not None, "no distinct runner-up in the population"
champ2 = pop[i2]
gap2 = float(gaps[i2])
d_b5 = champ1[N_TRUNK:] - champ2[N_TRUNK:]
dist = {
    "genes": N_B5,
    "n_differing": int((d_b5 != 0).sum()),
    "max_abs_diff": float(np.abs(d_b5).max()),
    "rms_diff": float(np.sqrt((d_b5 ** 2).mean())),
}
log(f"second champion: pop index {i2}, validation gap "
    f"{gap2*100:.4f}%; b5 distance over {N_B5} genes: "
    f"max abs {dist['max_abs_diff']:.6f}, RMS "
    f"{dist['rms_diff']:.6f}, differing "
    f"{dist['n_differing']}/{N_B5}")

W1 = unpack_B(champ1)
W2 = unpack_B(champ2)

# ================================================================
# STEP 2 - build the C genome
# ================================================================
TRUNK_KEYS = [k for k in KEYS_B if not k.startswith("b5.")]
assert sum(SIZES_B[k] for k in TRUNK_KEYS) == N_TRUNK

WC = {}
for k in TRUNK_KEYS:
    WC[k] = W1[k].copy()
# widened backend block b5w
WC["b5w.lnA_gain"] = W1["b5.lnA_gain"].copy()
WC["b5w.lnA_bias"] = W1["b5.lnA_bias"].copy()
for h, Wsrc in ((0, W1), (1, W2)):
    for nm in ("W_q", "W_k", "W_v", "W_o",
               "b_q", "b_k", "b_v", "b_o"):
        WC[f"b5w.h{h}.{nm}"] = Wsrc[f"b5.{nm}"].copy()
WC["b5w.lnF_gain"] = W1["b5.lnF_gain"].copy()
WC["b5w.lnF_bias"] = W1["b5.lnF_bias"].copy()
W_1 = np.zeros((4, 8))
W_1[:, 0:4] = W1["b5.W_1"]
W_1[:, 4:8] = W2["b5.W_1"]
WC["b5w.W_1"] = W_1
WC["b5w.b_1"] = np.concatenate([W1["b5.b_1"], W2["b5.b_1"]])
W_2 = np.zeros((8, 4))
W_2[0:4, :] = W1["b5.W_2"]
W_2[4:8, :] = W2["b5.W_2"]
WC["b5w.W_2"] = W_2
WC["b5w.b_2"] = W1["b5.b_2"].copy()

KEYS_C = TRUNK_KEYS + [
    "b5w.lnA_gain", "b5w.lnA_bias",
    "b5w.h0.W_q", "b5w.h0.W_k", "b5w.h0.W_v", "b5w.h0.W_o",
    "b5w.h0.b_q", "b5w.h0.b_k", "b5w.h0.b_v", "b5w.h0.b_o",
    "b5w.h1.W_q", "b5w.h1.W_k", "b5w.h1.W_v", "b5w.h1.W_o",
    "b5w.h1.b_q", "b5w.h1.b_k", "b5w.h1.b_v", "b5w.h1.b_o",
    "b5w.lnF_gain", "b5w.lnF_bias",
    "b5w.W_1", "b5w.b_1", "b5w.W_2", "b5w.b_2",
]
assert set(WC) == set(KEYS_C)
N_B5W = int(sum(WC[k].size for k in KEYS_C
                if k.startswith("b5w.")))
N_C = int(sum(WC[k].size for k in KEYS_C))
log(f"gene accounting: trunk {N_TRUNK}, widened block "
    f"b5w {N_B5W}, total {N_C}")
assert N_B5W == 252 and N_C == N_TRUNK + 252

founder_C = np.concatenate([WC[k].ravel() for k in KEYS_C])

# ---- verification: trunk bit-identical ----
assert np.array_equal(founder_C[:N_TRUNK], champ1[:N_TRUNK])
# ---- verification: every placed branch entry
# bit-exact against its source ----
checks = {}
for nm in ("W_q", "W_k", "W_v", "W_o",
           "b_q", "b_k", "b_v", "b_o"):
    checks[f"h0.{nm} == champ1 b5.{nm}"] = bool(
        np.array_equal(WC[f"b5w.h0.{nm}"], W1[f"b5.{nm}"]))
    checks[f"h1.{nm} == champ2 b5.{nm}"] = bool(
        np.array_equal(WC[f"b5w.h1.{nm}"], W2[f"b5.{nm}"]))
checks["lnA/lnF == champ1"] = bool(
    np.array_equal(WC["b5w.lnA_gain"], W1["b5.lnA_gain"])
    and np.array_equal(WC["b5w.lnA_bias"], W1["b5.lnA_bias"])
    and np.array_equal(WC["b5w.lnF_gain"], W1["b5.lnF_gain"])
    and np.array_equal(WC["b5w.lnF_bias"], W1["b5.lnF_bias"]))
checks["W_1 cols 0-3 == champ1, 4-7 == champ2"] = bool(
    np.array_equal(WC["b5w.W_1"][:, 0:4], W1["b5.W_1"])
    and np.array_equal(WC["b5w.W_1"][:, 4:8], W2["b5.W_1"]))
checks["W_2 rows 0-3 == champ1, 4-7 == champ2"] = bool(
    np.array_equal(WC["b5w.W_2"][0:4, :], W1["b5.W_2"])
    and np.array_equal(WC["b5w.W_2"][4:8, :], W2["b5.W_2"]))
checks["b_1 halves == champ1/champ2"] = bool(
    np.array_equal(WC["b5w.b_1"][0:4], W1["b5.b_1"])
    and np.array_equal(WC["b5w.b_1"][4:8], W2["b5.b_1"]))
checks["b_2 == champ1"] = bool(
    np.array_equal(WC["b5w.b_2"], W1["b5.b_2"]))
assert all(checks.values()), checks
log("placement verification: all branch entries bit-exact")


# ================================================================
# C forward (trunk blocks single-head + the
# two-head widened backend block)
# ================================================================
def forward_C(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1", "b2", "b3", "b4"):
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
        A = A + Ah @ W[f"{blk}.h{h}.W_o"] + W[f"{blk}.h{h}.b_o"]
    Hf = ln_np(A, W[f"{blk}.lnF_gain"], W[f"{blk}.lnF_bias"],
               EPS)
    A = A + np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"]) \
        @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
    return A


def gap_C(W, X, opt):
    Y = forward_C(W, X)
    picks = decode(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean()), picks


# functional parity: branch 2 masked out must
# reproduce champion #1 exactly (shared LNs
# and b_2 are #1's, head 0 and FFN units 0-3
# are #1's)
WM = {k: v.copy() for k, v in WC.items()}
WM["b5w.h1.W_o"] = np.zeros_like(WM["b5w.h1.W_o"])
WM["b5w.h1.b_o"] = np.zeros_like(WM["b5w.h1.b_o"])
WM["b5w.W_2"] = WM["b5w.W_2"].copy()
WM["b5w.W_2"][4:8, :] = 0.0
g_mask, picks_mask = gap_C(WM, X_val, OPT_VAL)
g_c1, picks_c1 = gap_B(champ1, X_val, OPT_VAL)
parity = bool(abs(g_mask - g_c1) < 1e-12
              and (picks_mask == picks_c1).all())
checks["branch-2-masked stitch == champion #1 "
       "(gap + tours)"] = parity
assert parity, (g_mask, g_c1)
log(f"functional parity: branch-2-masked stitch "
    f"{g_mask*100:.6f}% == champion #1 "
    f"{g_c1*100:.6f}%, tours identical")

birth_gap, _ = gap_C(WC, X_val, OPT_VAL)
log(f"STITCH BIRTH GAP: {birth_gap*100:.4f}% "
    f"(champion #1 {gap1*100:.4f}%, champion #2 "
    f"{gap2*100:.4f}%)")

# ================================================================
# Save genome, stitch record, state
# ================================================================
meta = {
    "kind": "frankenstein_stepC_stitch",
    "date": "2026-11-26",
    "step": "C: widen the Step B backend block by "
            "stitching the top two Step B champions "
            "over their commonized trunk",
    "parents": {
        "champion_1": {"pop_index": i1,
                       "gap_pct": round(gap1 * 100, 4)},
        "champion_2": {"pop_index": i2,
                       "gap_pct": round(gap2 * 100, 4)},
    },
    "b5_distance": dist,
    "gene_order": KEYS_C,
    "genes_total": N_C,
    "genes_trunk_backbone": N_TRUNK,
    "genes_b5w_stitching": N_B5W,
    "trunk_flat_range": [0, N_TRUNK],
    "b5w_flat_range": [N_TRUNK, N_C],
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_stepC_genome.npz",
         **{k: WC[k] for k in KEYS_C},
         meta=np.array(json.dumps(meta)))

record = {
    "run": "frankenstein_stepC stitch (step C: widen "
           "the backend block b5 by stitching the top "
           "two Step B champions over the commonized "
           "trunk)",
    "date": "2026-11-26",
    "validation_set": "fixed 200 problems, 10 cities, "
                      "val_seed 20261116 (TSPDomain)",
    "step1_champion_search": {
        "population": "final population of the Step B "
                      "run (checkpoint.npz, 100 genomes, "
                      "all sharing the frozen 788-gene "
                      "trunk, differing only in b5)",
        "champion_1": {"pop_index": i1,
                       "gap_pct": round(gap1 * 100, 4),
                       "bit_identical_to_checkpoint_best":
                       True},
        "champion_2": {"pop_index": i2,
                       "gap_pct": round(gap2 * 100, 4),
                       "definition": "best validation gap "
                                     "among genomes not "
                                     "bit-identical to "
                                     "champion #1"},
        "b5_distance": dist,
        "top10_validation_gaps_pct": [
            round(float(gaps[j]) * 100, 4) for j in order[:10]],
    },
    "stitch": {
        "trunk": "embed + blocks b0..b4 + merged "
                 "decoder, 788 genes, kept ONCE, "
                 "bit-exact from champion #1 (the same "
                 "trunk in both parents - the common "
                 "ancestor, not duplicated)",
        "widened_block_b5w": {
            "position": "after the joining block (b4), "
                        "before the merged decoder - "
                        "replaces b5",
            "stream_width": 4,
            "attention": "2 heads; head 0 = champion "
                         "#1's b5 head, head 1 = "
                         "champion #2's b5 head; head "
                         "outputs summed into the "
                         "stream, each head adding its "
                         "own W_o and b_o; every entry "
                         "bit-exact",
            "ffn": "width 8; units 0-3 = champion #1's "
                   "(W_1 columns 0-3, W_2 rows 0-3, "
                   "b_1[0:4]), units 4-7 = champion "
                   "#2's (W_1 columns 4-7, W_2 rows "
                   "4-7, b_1[4:8]); every entry "
                   "bit-exact",
            "shared_layer_norms": "lnA and lnF are "
                                  "champion #1's genes",
            "dropped_from_champion_2": [
                "b5.lnA_gain", "b5.lnA_bias",
                "b5.lnF_gain", "b5.lnF_bias",
                "b5.b_2 (the FFN output bias is a "
                "single block-level parameter, like "
                "the layer-norms: champion #1's is "
                "used)"],
            "rescaling": "none - no weight is halved "
                         "or rescaled; there are no "
                         "filler entries in this stitch",
        },
        "gene_accounting": {
            "backbone_trunk": N_TRUNK,
            "stitching_b5w": N_B5W,
            "b5w_breakdown": {"lnA": 8, "heads": 160,
                              "lnF": 8, "ffn": 76},
            "total": N_C,
            "rule": "backbone = the 788 trunk genes; "
                    "stitching = ALL genes of the "
                    "widened block b5w, which must now "
                    "be refined as a unit",
        },
    },
    "verification": checks,
    "birth_gap_pct": round(birth_gap * 100, 4),
    "birth_gap_note": "two branches' outputs now sum "
                      "into the same stream with no "
                      "rescaling, so a birth gap worse "
                      "than champion #1's 7.9739% is "
                      "expected",
}
with open(HERE / "frankenstein_stepC_stitch_2026-11-26.json",
          "w") as f:
    json.dump(record, f, indent=1)

state = {
    "kind": "frankenstein_stepC_state",
    "date": "2026-11-26",
    "gene_order_C": KEYS_C,
    "trunk_flat_range": [0, N_TRUNK],
    "b5w_flat_range": [N_TRUNK, N_C],
    "champion_1": {"pop_index": i1,
                   "gap_pct": round(gap1 * 100, 4),
                   "genome_flat": champ1.tolist()},
    "champion_2": {"pop_index": i2,
                   "gap_pct": round(gap2 * 100, 4),
                   "genome_flat": champ2.tolist()},
    "b5_distance": dist,
    "placement": {
        "trunk_keys_from_champion_1": TRUNK_KEYS,
        "b5w": {
            "lnA_gain": "champ1 b5.lnA_gain",
            "lnA_bias": "champ1 b5.lnA_bias",
            "h0": "champ1 b5 head (W_q/W_k/W_v/W_o/"
                  "b_q/b_k/b_v/b_o)",
            "h1": "champ2 b5 head (W_q/W_k/W_v/W_o/"
                  "b_q/b_k/b_v/b_o)",
            "lnF_gain": "champ1 b5.lnF_gain",
            "lnF_bias": "champ1 b5.lnF_bias",
            "W_1": "columns 0-3 champ1 b5.W_1, "
                   "columns 4-7 champ2 b5.W_1",
            "b_1": "[0:4] champ1, [4:8] champ2",
            "W_2": "rows 0-3 champ1 b5.W_2, rows "
                   "4-7 champ2 b5.W_2",
            "b_2": "champ1 b5.b_2 (champ2's dropped, "
                   "with its LN genes)",
        },
    },
    "birth_gap_pct": round(birth_gap * 100, 4),
}
with open(HERE / "frankenstein_stepC_state.json", "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_stepC_genome.npz, "
    "frankenstein_stepC_stitch_2026-11-26.json, "
    "frankenstein_stepC_state.json")
