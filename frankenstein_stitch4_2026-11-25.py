#!/usr/bin/env python3
"""Frankenstein stitch #4 (sponsor decision,
2026-11-25 10:39 EST): Frankenstein 2's body with a
MERGED decoder as the ONLY change.

Frankenstein 2 (birth gap 41.47%) carried the host's
decoder stretched over the joined width-4 stream:
the host block was exact only where the host's path
ran, and the donor side of that decoder was
tiny-random filler; species 9's own decoder was
dropped entirely. Frankenstein 3 (25.76%) instead
kept two whole decoders side by side and joined
their finished logits afterwards.

Frankenstein 4 replaces the decoder with ONE
pointer decoder whose weights are BOTH parents'
decoders side by side in the same matrices:

  - W_qd (4x4) is block diagonal: species 8's
    original 2x2 W_qd on the species 8 stream
    coordinates (joined coords 0,1), species 9's
    original 2x2 W_qd on the species 9 coordinates
    (joined coords 2,3). The two off-diagonal 2x2
    blocks (host weights reading donor coordinates
    and vice versa) are tiny random, std 0.03,
    ONE draw.
  - b_qd = species 8's b_qd on the host
    coordinates, species 9's b_qd on the donor
    coordinates (the concatenation, placed by the
    territory maps). v_start the same.
  - No feed-forward layer after the decoder. No
    second decoder. No logit averaging. One
    decoder, one pick.

BODY: loaded verbatim from
frankenstein_2_genome.npz (the two corrected paths
side by side, the joining block before the decoder,
all filler exactly as drawn in stitch #2). No
random weight in the body is redrawn; the decoder
is the only part that changes, and the build
asserts every non-decoder array is bit-identical
to Frankenstein 2's.

DECODER CONVENTIONS: identical to Frankenstein 2's
forward pass (decode_general in stitch #2): state
= mean of encoder outputs + last picked stream
vector (v_start at t=0); q = state @ W_qd + b_qd;
logits = CLIP * tanh((Y . q) * attn_scale) with
CLIP = bert_engine.CLIP (10.0); visited cities
masked; greedy argmax. attn_scale = SCALE =
arch.attn_scale of the engine's initial arch,
be.initial_arch(cfg, 2): the parents' width d=2,
so attn_scale = 1/sqrt(2) = 0.7071067811865475,
the same value recorded in Frankenstein 2's
genome meta and used by its forward pass for both
attention and the decoder logits.

Scored ONCE on the fixed 200-problem validation
set. One build, one score, no retries.

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_stitch4_2026-11-25.py
Outputs: frankenstein_4_genome.npz,
frankenstein_stitch4_2026-11-25.json,
frankenstein_stitch4_state.json (diagram input).
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
RUN = HERE / "engine_runs" / "fresh50_v2"
SIGMA_CROSS = 0.03
MERGE_SEED = 20261127
EXPECTED_HOST_GAP = 0.1513148917731675
F2_GAP_PCT = 41.474        # frankenstein_stitch2 results
F3_GAP_PCT = 25.7642       # frankenstein_stitch3 results

# ================================================================
# Setup (identical to stitch #2, so SCALE / EPS /
# the validation set are the same objects)
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps
log(f"attn_scale = {SCALE!r} (arch.attn_scale of "
    f"be.initial_arch(cfg, 2), the parents' d=2 arch; "
    f"the value Frankenstein 2's forward pass used)")

# ================================================================
# Parents (only their original decoders are needed)
# ================================================================
z = np.load(RUN / "champion.npz")
host_genome = np.asarray(z["genome"], dtype=float)
host_detail = domain.validation_detail(layout, host_genome)
log(f"host (run champion): gap {host_detail['mean_gap']*100:.4f}%")
assert abs(host_detail["mean_gap"] - EXPECTED_HOST_GAP) < 1e-9

ck = np.load(RUN / "checkpoint.npz", allow_pickle=True)
sp_ids = [int(s) for s in ck["species_ids"]]
sp_best = {int(s): np.asarray(g, dtype=float)
           for s, g in zip(ck["species_ids"], ck["best_genomes"])}
gaps = {}
for s, g in sp_best.items():
    gaps[s] = float(domain.validation_detail(layout, g)["mean_gap"])
donor_id = min((s for s in sp_ids if s != 8), key=lambda s: gaps[s])
donor_genome = sp_best[donor_id]
log(f"host = species 8 (gap {gaps[8]*100:.2f}%), "
    f"donor = species {donor_id} (gap {gaps[donor_id]*100:.2f}%)")
assert donor_id == 9


def parent_weights(genome):
    """Engine genome -> generic weight dict (verbatim
    from stitch #2)."""
    d = layout.unpack_dict(genome)
    return {
        "embed.W_in": d["c0.embed.W_in"],
        "embed.b_in": d["c0.embed.b_in"],
        "b0.lnA_gain": d["c0.b0.lnA_gain"],
        "b0.lnA_bias": d["c0.b0.lnA_bias"],
        "b0.W_q": d["c0.b0.h0.W_q"], "b0.b_q": d["c0.b0.h0.b_q"],
        "b0.W_k": d["c0.b0.h0.W_k"], "b0.b_k": d["c0.b0.h0.b_k"],
        "b0.W_v": d["c0.b0.h0.W_v"], "b0.b_v": d["c0.b0.h0.b_v"],
        "b0.W_o": d["c0.b0.h0.W_o"], "b0.b_o": d["c0.b0.h0.b_o"],
        "b0.lnF_gain": d["c0.b0.lnF_gain"],
        "b0.lnF_bias": d["c0.b0.lnF_bias"],
        "b0.W_1": d["c0.b0.W_1"], "b0.b_1": d["c0.b0.b_1"],
        "b0.W_2": d["c0.b0.W_2"], "b0.b_2": d["c0.b0.b_2"],
        "head.v_start": d["c0.head.v_start"],
        "head.W_qd": d["c0.head.W_qd"],
        "head.b_qd": d["c0.head.b_qd"],
    }


HW = parent_weights(host_genome)
DW = parent_weights(donor_genome)

# ================================================================
# Forward + decode (verbatim from stitch #2)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X, blocks=("b0",)):
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


def decode_general(W, Y, X):
    B, n, d = Y.shape
    gmean = Y.mean(axis=1)                       # (B, d)
    v_start = W["head.v_start"]
    W_qd = W["head.W_qd"]
    b_qd = W["head.b_qd"]
    last = np.broadcast_to(v_start, (B, d)).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    picks = np.zeros((B, n), dtype=int)
    for t in range(n):
        state = gmean + last
        q = state @ W_qd + b_qd                  # (B, d)
        logits = be.CLIP * np.tanh(
            np.einsum("bi,bni->bn", q, Y) * SCALE)
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last = Y[arB, pick]
    return picks


def gap_of(W, X, blocks=("b0",)):
    Y = forward_general(W, X, blocks)
    picks = decode_general(W, Y, X)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    gaps = lens / OPT - 1.0
    return float(gaps.mean()), picks, lens


# ================================================================
# Load Frankenstein 2's body verbatim
# ================================================================
z2 = np.load(HERE / "frankenstein_2_genome.npz")
meta2 = json.loads(str(z2["meta"]))
FW = {k: np.asarray(z2[k], dtype=float)
      for k in z2.files if k != "meta"}
assert abs(meta2["attn_scale"] - SCALE) < 1e-15
h_map = {int(k): v for k, v in meta2["host_coord_map"].items()}
d_map = {int(k): v for k, v in meta2["donor_coord_map"].items()}
dF = meta2["d"]

# body check: the loaded genome, scored with this
# forward pass, must reproduce Frankenstein 2's
# recorded gap before anything is changed
S2 = json.load(open(HERE / "frankenstein_stitch2_state.json"))
f2_gap_stored = float(S2["frank"]["gap"])
f2_gap_mine, _, _ = gap_of(FW, X_val, blocks=("b0", "b1"))
log(f"body check: Frankenstein 2 genome re-scored "
    f"{f2_gap_mine*100:.4f}% vs recorded "
    f"{f2_gap_stored*100:.4f}%")
assert abs(f2_gap_mine - f2_gap_stored) < 1e-9

# ================================================================
# The merged decoder (the ONLY change)
# ================================================================
rng = np.random.default_rng(MERGE_SEED)
W_qd = rng.normal(0.0, SIGMA_CROSS, size=(dF, dF))  # one draw
for k_old, k_new in h_map.items():
    for j_old, j_new in h_map.items():
        W_qd[k_new, j_new] = HW["head.W_qd"][k_old, j_old]
for k_old, k_new in d_map.items():
    for j_old, j_new in d_map.items():
        W_qd[k_new, j_new] = DW["head.W_qd"][k_old, j_old]

b_qd = np.zeros(dF)
v_start = np.zeros(dF)
for k_old, k_new in h_map.items():
    b_qd[k_new] = HW["head.b_qd"][k_old]
    v_start[k_new] = HW["head.v_start"][k_old]
for k_old, k_new in d_map.items():
    b_qd[k_new] = DW["head.b_qd"][k_old]
    v_start[k_new] = DW["head.v_start"][k_old]

# ---- construction check
h_rows = sorted(h_map.values())
h_cols = sorted(h_map.values())
d_rows = sorted(d_map.values())
d_cols = sorted(d_map.values())
chk_host_block = bool(np.array_equal(
    W_qd[np.ix_(h_rows, h_cols)], HW["head.W_qd"]))
chk_donor_block = bool(np.array_equal(
    W_qd[np.ix_(d_rows, d_cols)], DW["head.W_qd"]))
exp_b = np.concatenate([HW["head.b_qd"], DW["head.b_qd"]])
exp_v = np.concatenate([HW["head.v_start"],
                        DW["head.v_start"]])
# territory maps are identity-then-shift, so the
# placed vectors must equal the plain concatenations
chk_b = bool(np.array_equal(b_qd, exp_b))
chk_v = bool(np.array_equal(v_start, exp_v))
cross = np.concatenate([
    W_qd[np.ix_(h_rows, d_cols)].ravel(),
    W_qd[np.ix_(d_rows, h_cols)].ravel()])
log(f"construction check: host 2x2 block == species 8 "
    f"W_qd exactly: {chk_host_block}; donor 2x2 block "
    f"== species 9 W_qd exactly: {chk_donor_block}; "
    f"b_qd == concat(host, donor) exactly: {chk_b}; "
    f"v_start == concat(host, donor) exactly: {chk_v}; "
    f"cross blocks: max |w| {np.abs(cross).max():.4f}, "
    f"std {cross.std():.4f} (tiny random, std "
    f"{SIGMA_CROSS}, seed {MERGE_SEED}, one draw)")
assert chk_host_block and chk_donor_block and chk_b and chk_v

FW4 = dict(FW)
FW4["head.W_qd"] = W_qd
FW4["head.b_qd"] = b_qd
FW4["head.v_start"] = v_start

# body-unchanged check: every non-decoder array must
# be bit-identical to Frankenstein 2's
body_same = all(
    np.array_equal(FW4[k], FW[k])
    for k in FW if not k.startswith("head."))
log(f"body check: every non-decoder array bit-identical "
    f"to Frankenstein 2: {body_same}")
assert body_same

# ================================================================
# Score ONCE
# ================================================================
frank_gap, frank_picks, frank_lens = gap_of(
    FW4, X_val, blocks=("b0", "b1"))
log(f"FRANKENSTEIN 4 validation gap: "
    f"{frank_gap*100:.2f}% (Frankenstein 2 "
    f"{F2_GAP_PCT:.2f}%, Frankenstein 3 "
    f"{F3_GAP_PCT:.2f}%, host {gaps[8]*100:.2f}%, "
    f"donor {gaps[donor_id]*100:.2f}%)")

# ================================================================
# Save genome + results + diagram state
# ================================================================
meta = {
    "kind": "frankenstein_stitch4",
    "date": "2026-11-25",
    "base": "frankenstein_2_genome.npz body verbatim; "
            "decoder replaced by the merged decoder",
    "decoder": "one pointer decoder; W_qd block "
               "diagonal (species 8's 2x2 on joined "
               "coords 0,1; species 9's 2x2 on joined "
               "coords 2,3); off-diagonal 2x2 blocks "
               "tiny random std 0.03, one draw, seed "
               f"{MERGE_SEED}; b_qd and v_start are "
               "the parents' concatenations by "
               "territory",
    "host_species": 8, "donor_species": donor_id,
    "d": dF,
    "ffn_hidden_block1": meta2["ffn_hidden_block1"],
    "ffn_hidden_block2_joining": meta2[
        "ffn_hidden_block2_joining"],
    "blocks": 2, "heads_per_block": 1,
    "host_coord_map": meta2["host_coord_map"],
    "donor_coord_map": meta2["donor_coord_map"],
    "host_unit_map": meta2["host_unit_map"],
    "donor_unit_map": meta2["donor_unit_map"],
    "sigma_cross": SIGMA_CROSS, "merge_seed": MERGE_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_4_genome.npz",
         **{k: v for k, v in FW4.items()},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_4_genome.npz")

results = {
    "design": "sponsor decision, 2026-11-25 10:39 "
              "EST: Frankenstein 2's body with a "
              "merged decoder as the only change",
    "parents": {
        "host": {"species": 8,
                 "gap_pct": round(gaps[8] * 100, 4)},
        "donor": {"species": donor_id,
                  "gap_pct": round(gaps[donor_id] * 100, 4)},
    },
    "merged_decoder": {
        "W_qd_host_block": HW["head.W_qd"].tolist(),
        "W_qd_donor_block": DW["head.W_qd"].tolist(),
        "W_qd_full": W_qd.tolist(),
        "b_qd": b_qd.tolist(),
        "v_start": v_start.tolist(),
        "sigma_cross": SIGMA_CROSS,
        "merge_seed": MERGE_SEED,
        "attn_scale": SCALE,
        "attn_scale_source": "arch.attn_scale of "
            "be.initial_arch(cfg, 2) (the parents' "
            "d=2 arch), the same value Frankenstein "
            "2's forward pass used for attention and "
            "decoder logits alike; recorded in "
            "Frankenstein 2's genome meta",
        "clip": float(be.CLIP),
    },
    "construction_check": {
        "host_block_equals_species8_W_qd_exactly":
            chk_host_block,
        "donor_block_equals_species9_W_qd_exactly":
            chk_donor_block,
        "b_qd_equals_concatenation_exactly": chk_b,
        "v_start_equals_concatenation_exactly": chk_v,
        "cross_block_max_abs": float(np.abs(cross).max()),
        "cross_block_std": float(cross.std()),
        "body_non_decoder_arrays_bit_identical_to_f2":
            body_same,
        "f2_genome_rescored_gap_pct":
            round(f2_gap_mine * 100, 4),
    },
    "frankenstein4": {
        "shape": {"stream_width": dF,
                  "block1_ffn_width":
                      meta2["ffn_hidden_block1"],
                  "joining_block_ffn_width":
                      meta2["ffn_hidden_block2_joining"],
                  "blocks": 2, "heads_per_block": 1,
                  "decoders": 1,
                  "total_params": int(sum(
                      v.size for v in FW4.values()))},
        "validation_gap_pct": round(frank_gap * 100, 4),
        "mean_length": float(frank_lens.mean()),
        "frankenstein2_gap_pct": F2_GAP_PCT,
        "frankenstein3_gap_pct": F3_GAP_PCT,
    },
}
with open(HERE / "frankenstein_stitch4_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_stitch4_2026-11-25.json")

state = {"parents": S2["parents"],
         "frank": {"gap": frank_gap, "dF": dF,
                   "U": meta2["ffn_hidden_block1"],
                   "U2": meta2["ffn_hidden_block2_joining"],
                   "h_map": {str(k): v for k, v in
                             h_map.items()},
                   "d_map": {str(k): v for k, v in
                             d_map.items()},
                   "hu_map": {int(k): v for k, v in
                              ((int(a), b) for a, b in
                               meta2["host_unit_map"].items())},
                   "du_map": {int(k): v for k, v in
                              ((int(a), b) for a, b in
                               meta2["donor_unit_map"].items())},
                   "h_path": S2["frank"]["h_path"],
                   "d_path": S2["frank"]["d_path"]}}
# unit maps in the stitch-2 state are keyed by the old
# unit index as strings already; copy them verbatim
state["frank"]["hu_map"] = S2["frank"]["hu_map"]
state["frank"]["du_map"] = S2["frank"]["du_map"]
state["frank"]["h_map"] = S2["frank"]["h_map"]
state["frank"]["d_map"] = S2["frank"]["d_map"]
with open(HERE / "frankenstein_stitch4_state.json",
          "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_stitch4_state.json "
    "(diagram input)")
log("DONE build")
