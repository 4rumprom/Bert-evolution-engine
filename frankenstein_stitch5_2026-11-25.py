#!/usr/bin/env python3
"""Frankenstein stitch #5 (sponsor decision,
2026-11-25 21:34 EST): Frankenstein 1's
architecture pattern with two changes: WHOLE
parents, and the MERGED decoder of Frankenstein 4.

Frankenstein 1 (birth gap 20.15%) kept only the
walker-threshold paths of the two parents and
stretched the host's decoder over the joined
width. Frankenstein 4 (23.29%) kept Frankenstein
2's thin corrected paths and swapped in the merged
decoder. Frankenstein 5 drops NOTHING from either
parent:

  - Parents: species 8 champion (gap 15.13%) and
    species 9 champion (gap 19.44%) of the
    fresh50_v2 run, the same parent genomes as
    every earlier Frankenstein build. Each parent's
    single block goes in WHOLE: every synapse exact,
    on its own territory. No path extraction, no
    dropping (the sponsor's rule: drop only
    irrelevant synapses; here nothing is dropped).
  - Body: one widened block, stream width 4
    (species 8 on coordinates 0-1, species 9 on
    coordinates 2-3), FFN width 16 (8+8). Every
    cross-territory entry is tiny random filler,
    std 0.03. Then the joining block before the
    decoder, exactly as in Frankenstein 1/2/4: a
    full pre-norm BERT block at width 4, FFN 16,
    all tiny random std 0.03 (its layer-norm gains
    are 1 + N(0, 0.01), as in the earlier builds).
  - Decoder: the MERGED decoder exactly as built
    for Frankenstein 4: W_qd 4x4 block diagonal
    (species 8's 2x2 top-left exact, species 9's
    2x2 bottom-right exact, cross blocks tiny
    random std 0.03), b_qd and v_start the exact
    concatenations. NO post-decoder feed-forward
    layer, NO logit averaging. All other decoder
    conventions (attn_scale 1/sqrt(2), CLIP 10
    tanh, state = mean of encoder outputs + last
    picked vector, greedy argmax, visited mask)
    exactly as the earlier builds.
  - ALL filler in the build comes from ONE rng
    stream, seed FILL_SEED, drawn in the fixed
    order below. One build, one draw, one score,
    no retries.

CONSTRUCTION CHECKS (all must pass; the build
asserts and reports them):
  - every in-territory entry of every block-1 and
    embedding array is bit-exact against its
    parent genome;
  - the merged decoder's diagonal blocks, b_qd
    and v_start are bit-exact;
  - the stitching mask accounts for every gene:
    total = backbone + stitching, with per-array
    counts reported.

PARITY: the generic forward pass is verified
against the engine on both parents (identical
tours, gap to 1e-9), as in every earlier build.
A decoder wiring check feeds each parent's own
encoder output, embedded in its territory, through
the merged decoder with the cross blocks zeroed
and the other territory's v_start zeroed, and
requires the parent's tours back exactly; that
verifies the diagonal blocks and biases are wired
to the right territory. END-TO-END parity with a
territory silenced is NOT claimed and does not
hold by design: block 1's LayerNorm normalises
over the joined width 4, and the decoder cross
blocks couple the territories through v_start at
the first pick. The birth score below is the
honest whole-network number.

The stitching-gene mask (for the evolution run):
stitching genes = every cross-territory entry of
the widened block, the WHOLE joining block, and
the merged decoder's cross blocks. Backbone genes
= both parents' own weights, including their
decoder blocks, b_qd and v_start. Only stitching
genes may mutate in the evolution.

ENGINE FIT: as with Frankensteins 1-4, the
current ArchSpec cannot hold this shape; the
Frankenstein is an explicit weight set with its
own forward pass (parity-verified on the parents).

Run: python3 frankenstein_stitch5_2026-11-25.py
Outputs: frankenstein_5_genome.npz,
frankenstein_stitch5_2026-11-25.json,
frankenstein_stitch5_state.json (diagram input).
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
SIGMA_FILL = 0.03
FILL_SEED = 20261128
EXPECTED_HOST_GAP = 0.1513148917731675
F1_GAP_PCT = 20.1514       # frankenstein_stitch results
F4_GAP_PCT = 23.2865       # frankenstein_stitch4 results

# canonical gene order (the order arrays are built
# and saved; the evolution packs genomes this way)
KEYS = [
    "embed.W_in", "embed.b_in",
    "b0.lnA_gain", "b0.lnA_bias", "b0.lnF_gain",
    "b0.lnF_bias",
    "b0.W_q", "b0.W_k", "b0.W_v", "b0.W_o",
    "b0.b_q", "b0.b_k", "b0.b_v", "b0.b_o", "b0.b_2",
    "b0.W_1", "b0.W_2", "b0.b_1",
    "b1.lnA_gain", "b1.lnA_bias", "b1.lnF_gain",
    "b1.lnF_bias",
    "b1.W_q", "b1.W_k", "b1.W_v", "b1.W_o",
    "b1.b_q", "b1.b_k", "b1.b_v", "b1.b_o", "b1.b_2",
    "b1.W_1", "b1.b_1", "b1.W_2",
    "head.W_qd", "head.v_start", "head.b_qd",
]

# ================================================================
# Setup (identical to stitches 1/2/4)
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
log(f"attn_scale = {SCALE!r} (parents' d=2 arch, "
    f"as in every Frankenstein build)")

# ================================================================
# Parents (same genomes as stitches 1-4)
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


# parity check: generic forward on both parents must
# reproduce the engine exactly (the builds' standard
# parity)
for tag, genome, W in (("host", host_genome, HW),
                       ("donor", donor_genome, DW)):
    g_mine, picks_mine, _ = gap_of(W, X_val)
    lens_e, picks_e = domain.evaluate_lengths(
        layout, genome[None, :], X_val)
    tours_same = bool((picks_mine == picks_e[0]).all())
    log(f"parity {tag}: my gap {g_mine*100:.6f}% vs engine "
        f"{gaps[8 if tag == 'host' else donor_id]*100:.6f}%, "
        f"tours identical: {tours_same}")
    assert abs(g_mine - gaps[8 if tag == 'host' else donor_id]) < 1e-9 \
        and tours_same

# ================================================================
# Territories: WHOLE parents. Host stream coords
# 0-1, donor 2-3; host FFN units 0-7, donor 8-15.
# ================================================================
h_map = {0: 0, 1: 1}
d_map = {0: 2, 1: 3}
hu_map = {u: u for u in range(8)}
du_map = {u: 8 + u for u in range(8)}
dF, U, U2 = 4, 16, 16


def terr_stream(i):
    return "h" if i < 2 else "d"


def terr_unit(u):
    return "h" if u < 8 else "d"


# ================================================================
# Build. ONE rng stream for every filler draw.
# ================================================================
rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


FW = {}
# ---- embed: columns are territory-owned; rows are
# the shared input coordinates. EVERY entry is a
# parent's own weight (both parents are whole).
W_in = fill((2, dF))
for c in range(2):
    for k, kn in h_map.items():
        W_in[c, kn] = HW["embed.W_in"][c, k]
    for k, kn in d_map.items():
        W_in[c, kn] = DW["embed.W_in"][c, k]
FW["embed.W_in"] = W_in
b_in = fill((dF,))
for k, kn in h_map.items():
    b_in[kn] = HW["embed.b_in"][k]
for k, kn in d_map.items():
    b_in[kn] = DW["embed.b_in"][k]
FW["embed.b_in"] = b_in


def territory_vec(name, size, stream=True):
    """Vector whose entries are territory-owned:
    every entry takes its parent's exact value."""
    v = fill((size,))
    mp_s = h_map if stream else hu_map
    mp_d = d_map if stream else du_map
    for k, kn in mp_s.items():
        v[kn] = HW[name][k]
    for k, kn in mp_d.items():
        v[kn] = DW[name][k]
    return v


def block1_matrix(name, row_terr, col_terr, n_rows, n_cols):
    """Territory blocks EXACT (the whole parent);
    cross-territory entries stay tiny random."""
    M = fill((n_rows, n_cols))
    hW, dW = HW[name], DW[name]
    rmap_h = h_map if row_terr == "stream" else hu_map
    rmap_d = d_map if row_terr == "stream" else du_map
    cmap_h = h_map if col_terr == "stream" else hu_map
    cmap_d = d_map if col_terr == "stream" else du_map
    for i_old, i_new in rmap_h.items():
        for j_old, j_new in cmap_h.items():
            M[i_new, j_new] = hW[i_old, j_old]
    for i_old, i_new in rmap_d.items():
        for j_old, j_new in cmap_d.items():
            M[i_new, j_new] = dW[i_old, j_old]
    return M


# ---- block 1: the two WHOLE parents side by side
for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
    FW[f"b0.{nm}"] = territory_vec(f"b0.{nm}", dF)
for nm in ("W_q", "W_k", "W_v", "W_o"):
    FW[f"b0.{nm}"] = block1_matrix(
        f"b0.{nm}", "stream", "stream", dF, dF)
for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
    FW[f"b0.{nm}"] = territory_vec(f"b0.{nm}", dF)
FW["b0.W_1"] = block1_matrix("b0.W_1", "stream", "unit", dF, U)
FW["b0.W_2"] = block1_matrix("b0.W_2", "unit", "stream", U, dF)
FW["b0.b_1"] = territory_vec("b0.b_1", U, stream=False)

# ---- block 2: the joining block before the decoder,
# exactly as in Frankenstein 1/2/4: a legal pre-norm
# BERT block over the full joined width, ALL tiny
# random filler.
FW["b1.lnA_gain"] = 1.0 + rng_fill.normal(0.0, 0.01, dF)
FW["b1.lnA_bias"] = fill((dF,))
FW["b1.lnF_gain"] = 1.0 + rng_fill.normal(0.0, 0.01, dF)
FW["b1.lnF_bias"] = fill((dF,))
for nm in ("W_q", "W_k", "W_v", "W_o"):
    FW[f"b1.{nm}"] = fill((dF, dF))
for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
    FW[f"b1.{nm}"] = fill((dF,))
FW["b1.W_1"] = fill((dF, U2))
FW["b1.b_1"] = fill((U2,))
FW["b1.W_2"] = fill((U2, dF))

# ---- decoder: the MERGED decoder, exactly as in
# Frankenstein 4 (one draw for the full matrix from
# the same rng stream, diagonal blocks overwritten
# with the parents' exact W_qd; b_qd and v_start
# the exact concatenations).
W_qd = fill((dF, dF))
for k_old, k_new in h_map.items():
    for j_old, j_new in h_map.items():
        W_qd[k_new, j_new] = HW["head.W_qd"][k_old, j_old]
for k_old, k_new in d_map.items():
    for j_old, j_new in d_map.items():
        W_qd[k_new, j_new] = DW["head.W_qd"][k_old, j_old]
FW["head.W_qd"] = W_qd
v_start = np.zeros(dF)
b_qd = np.zeros(dF)
for k_old, k_new in h_map.items():
    v_start[k_new] = HW["head.v_start"][k_old]
    b_qd[k_new] = HW["head.b_qd"][k_old]
for k_old, k_new in d_map.items():
    v_start[k_new] = DW["head.v_start"][k_old]
    b_qd[k_new] = DW["head.b_qd"][k_old]
FW["head.v_start"] = v_start
FW["head.b_qd"] = b_qd

assert set(FW.keys()) == set(KEYS), \
    f"key mismatch: {sorted(set(FW) ^ set(KEYS))}"

# ================================================================
# Stitching-gene mask (per array; True = stitching,
# may mutate in the evolution; False = backbone)
# ================================================================
def stream_cross(shape):
    m = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            m[i, j] = terr_stream(i) != terr_stream(j)
    return m


MASKS = {}
MASKS["embed.W_in"] = np.zeros((2, dF), dtype=bool)
MASKS["embed.b_in"] = np.zeros(dF, dtype=bool)
for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
           "b_q", "b_k", "b_v", "b_o", "b_2"):
    MASKS[f"b0.{nm}"] = np.zeros(dF, dtype=bool)
for nm in ("W_q", "W_k", "W_v", "W_o"):
    MASKS[f"b0.{nm}"] = stream_cross((dF, dF))
m = np.zeros((dF, U), dtype=bool)
for i in range(dF):
    for j in range(U):
        m[i, j] = terr_stream(i) != terr_unit(j)
MASKS["b0.W_1"] = m
m = np.zeros((U, dF), dtype=bool)
for i in range(U):
    for j in range(dF):
        m[i, j] = terr_unit(i) != terr_stream(j)
MASKS["b0.W_2"] = m
MASKS["b0.b_1"] = np.zeros(U, dtype=bool)
for k in KEYS:
    if k.startswith("b1."):
        MASKS[k] = np.ones_like(FW[k], dtype=bool)
MASKS["head.W_qd"] = stream_cross((dF, dF))
MASKS["head.v_start"] = np.zeros(dF, dtype=bool)
MASKS["head.b_qd"] = np.zeros(dF, dtype=bool)
for k in KEYS:
    assert MASKS[k].shape == FW[k].shape, k

flat_mask = np.concatenate([MASKS[k].ravel() for k in KEYS])
n_total = int(flat_mask.size)
n_stitch = int(flat_mask.sum())
n_backbone = n_total - n_stitch
per_array = {k: {"shape": list(FW[k].shape),
                 "total": int(FW[k].size),
                 "stitching": int(MASKS[k].sum()),
                 "backbone": int(FW[k].size - MASKS[k].sum())}
             for k in KEYS}
log(f"genes: total {n_total}, backbone {n_backbone}, "
    f"stitching {n_stitch}")

# ================================================================
# CONSTRUCTION CHECKS
# ================================================================
checks = {}

# (1) every in-territory entry of the embedding and
# block 1 is bit-exact against its parent
def check_territory_exact(name, row_maps, col_maps=None):
    """row_maps/col_maps: (host_map, donor_map) or
    None for a vector (rows only)."""
    M = FW[name]
    ok = True
    n_checked = 0
    pairs = [(HW[name], row_maps[0], col_maps[0] if col_maps else None),
             (DW[name], row_maps[1], col_maps[1] if col_maps else None)]
    for P, rm, cm in pairs:
        for i_old, i_new in rm.items():
            if cm is None:
                ok &= bool(M[i_new] == P[i_old])
                n_checked += 1
            else:
                for j_old, j_new in cm.items():
                    ok &= bool(M[i_new, j_new] == P[i_old, j_old])
                    n_checked += 1
    return ok, n_checked


exact_report = {}
n_exact_checked = 0
vec_maps = (h_map, d_map)
unit_maps = (hu_map, du_map)
for name, rm, cm in [
        ("embed.W_in", ( {c: c for c in range(2)}, {c: c for c in range(2)} ), vec_maps),
        ("embed.b_in", vec_maps, None),
        ("b0.lnA_gain", vec_maps, None), ("b0.lnA_bias", vec_maps, None),
        ("b0.lnF_gain", vec_maps, None), ("b0.lnF_bias", vec_maps, None),
        ("b0.W_q", vec_maps, vec_maps), ("b0.W_k", vec_maps, vec_maps),
        ("b0.W_v", vec_maps, vec_maps), ("b0.W_o", vec_maps, vec_maps),
        ("b0.b_q", vec_maps, None), ("b0.b_k", vec_maps, None),
        ("b0.b_v", vec_maps, None), ("b0.b_o", vec_maps, None),
        ("b0.b_2", vec_maps, None),
        ("b0.W_1", vec_maps, unit_maps), ("b0.W_2", unit_maps, vec_maps),
        ("b0.b_1", unit_maps, None)]:
    ok, n = check_territory_exact(name, rm, cm)
    exact_report[name] = {"exact": ok, "entries_checked": n}
    n_exact_checked += n
    assert ok, f"in-territory entries not exact: {name}"
checks["block1_and_embed_in_territory_bit_exact"] = True
checks["in_territory_entries_checked"] = n_exact_checked
log(f"construction check 1: all {n_exact_checked} "
    f"in-territory embedding/block-1 entries bit-exact "
    f"against the parents: True")

# (2) merged decoder exactness (Frankenstein 4's checks)
h_rows, d_rows = sorted(h_map.values()), sorted(d_map.values())
chk_host_block = bool(np.array_equal(
    W_qd[np.ix_(h_rows, h_rows)], HW["head.W_qd"]))
chk_donor_block = bool(np.array_equal(
    W_qd[np.ix_(d_rows, d_rows)], DW["head.W_qd"]))
chk_b = bool(np.array_equal(
    b_qd, np.concatenate([HW["head.b_qd"], DW["head.b_qd"]])))
chk_v = bool(np.array_equal(
    v_start, np.concatenate([HW["head.v_start"],
                             DW["head.v_start"]])))
cross = np.concatenate([
    W_qd[np.ix_(h_rows, d_rows)].ravel(),
    W_qd[np.ix_(d_rows, h_rows)].ravel()])
checks["decoder_host_block_bit_exact"] = chk_host_block
checks["decoder_donor_block_bit_exact"] = chk_donor_block
checks["decoder_b_qd_bit_exact"] = chk_b
checks["decoder_v_start_bit_exact"] = chk_v
checks["decoder_cross_max_abs"] = float(np.abs(cross).max())
checks["decoder_cross_std"] = float(cross.std())
log(f"construction check 2: merged decoder diagonal "
    f"blocks exact: host {chk_host_block}, donor "
    f"{chk_donor_block}; b_qd exact {chk_b}; v_start "
    f"exact {chk_v}; cross blocks max |w| "
    f"{np.abs(cross).max():.4f}, std {cross.std():.4f}")
assert chk_host_block and chk_donor_block and chk_b and chk_v

# (3) every gene accounted for by the mask
totals = {"total": sum(v["total"] for v in per_array.values()),
          "stitching": sum(v["stitching"] for v in per_array.values()),
          "backbone": sum(v["backbone"] for v in per_array.values())}
checks["gene_accounting"] = totals
assert totals == {"total": n_total, "stitching": n_stitch,
                  "backbone": n_backbone}
log(f"construction check 3: gene accounting {totals} "
    f"(mask covers every array in the canonical order)")

# (4) decoder wiring check: each parent's own encoder
# output, embedded in its territory, through the
# merged decoder with cross blocks zeroed and the
# other territory's v_start zeroed, must reproduce
# the parent's tours exactly.
def decoder_alone_check(W_parent, terr):
    Yp = forward_general(W_parent, X_val, blocks=("b0",))
    picks_parent = decode_general(W_parent, Yp, X_val)
    Yemb = np.zeros((X_val.shape[0], X_val.shape[1], dF))
    Wt = dict(FW)
    Wq = FW["head.W_qd"].copy()
    Wq[np.ix_(h_rows, d_rows)] = 0.0
    Wq[np.ix_(d_rows, h_rows)] = 0.0
    Wt["head.W_qd"] = Wq
    vs = FW["head.v_start"].copy()
    if terr == "h":
        Yemb[:, :, :2] = Yp
        vs[2:] = 0.0
    else:
        Yemb[:, :, 2:] = Yp
        vs[:2] = 0.0
    Wt["head.v_start"] = vs
    picks_test = decode_general(Wt, Yemb, X_val)
    return bool((picks_test == picks_parent).all())


chk_dec_h = decoder_alone_check(HW, "h")
chk_dec_d = decoder_alone_check(DW, "d")
checks["decoder_wiring_host_tours_reproduced"] = chk_dec_h
checks["decoder_wiring_donor_tours_reproduced"] = chk_dec_d
log(f"construction check 4: merged-decoder wiring — "
    f"host tours reproduced exactly {chk_dec_h}, "
    f"donor tours reproduced exactly {chk_dec_d} "
    f"(cross blocks and other territory's v_start "
    f"zeroed for this check only)")
assert chk_dec_h and chk_dec_d

# ================================================================
# Score ONCE on the fixed validation set
# ================================================================
frank_gap, frank_picks, frank_lens = gap_of(
    FW, X_val, blocks=("b0", "b1"))
log(f"FRANKENSTEIN 5 validation gap (birth): "
    f"{frank_gap*100:.2f}% — parents: species 8 "
    f"{gaps[8]*100:.2f}%, species 9 "
    f"{gaps[donor_id]*100:.2f}%; Frankenstein 1 "
    f"{F1_GAP_PCT:.2f}%, Frankenstein 4 {F4_GAP_PCT:.2f}%")

# ================================================================
# Save genome + results + diagram state
# ================================================================
meta = {
    "kind": "frankenstein_stitch5",
    "date": "2026-11-25",
    "design": "Frankenstein 1's architecture pattern; "
              "WHOLE parents (nothing dropped); merged "
              "decoder exactly as Frankenstein 4",
    "host_species": 8, "donor_species": donor_id,
    "d": dF, "ffn_hidden_block1": U,
    "ffn_hidden_block2_joining": U2,
    "blocks": 2, "heads_per_block": 1, "decoders": 1,
    "host_coord_map": {str(k): v for k, v in h_map.items()},
    "donor_coord_map": {str(k): v for k, v in d_map.items()},
    "host_unit_map": {str(k): v for k, v in hu_map.items()},
    "donor_unit_map": {str(k): v for k, v in du_map.items()},
    "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
    "gene_order": KEYS,
    "genes_total": n_total, "genes_backbone": n_backbone,
    "genes_stitching": n_stitch,
    "birth_gap_pct": round(frank_gap * 100, 4),
}
np.savez(HERE / "frankenstein_5_genome.npz",
         **{k: FW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_5_genome.npz")

# full in-territory (entry, i, j) lists for the diagram:
# in Frankenstein 5 these are ALL the parents' synapses
def all_territory_triples():
    h, d = [], []
    for c in range(2):
        for k in range(2):
            h.append(["embed.W_in", c, k])
            d.append(["embed.W_in", c, k])
    for nm in ("W_q", "W_k", "W_v", "W_o", "W_1", "W_2"):
        rng_i = range(2) if nm != "W_2" else range(8)
        rng_j = range(2) if nm != "W_1" else range(8)
        for i in rng_i:
            for j in rng_j:
                h.append([f"b0.{nm}", i, j])
                d.append([f"b0.{nm}", i, j])
    for k in range(2):
        for j in range(2):
            h.append(["head.W_qd", k, j])
            d.append(["head.W_qd", k, j])
    return h, d


h_all, d_all = all_territory_triples()
S2 = json.load(open(HERE / "frankenstein_stitch2_state.json"))
state = {"parents": S2["parents"],
         "frank": {"gap": frank_gap, "dF": dF, "U": U,
                   "U2": U2,
                   "h_map": {str(k): v for k, v in h_map.items()},
                   "d_map": {str(k): v for k, v in d_map.items()},
                   "hu_map": {str(k): v for k, v in hu_map.items()},
                   "du_map": {str(k): v for k, v in du_map.items()},
                   "h_path": h_all, "d_path": d_all}}
with open(HERE / "frankenstein_stitch5_state.json", "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_stitch5_state.json (diagram input)")

results = {
    "design": "sponsor decision, 2026-11-25 21:34 EST: "
              "Frankenstein 1's pattern, WHOLE parents, "
              "Frankenstein 4's merged decoder",
    "parents": {
        "host": {"species": 8,
                 "gap_pct": round(gaps[8] * 100, 4)},
        "donor": {"species": donor_id,
                  "gap_pct": round(gaps[donor_id] * 100, 4)},
    },
    "construction_check": checks,
    "parity": {
        "generic_forward_vs_engine_on_parents":
            "exact (identical tours, gap to 1e-9), "
            "asserted above as in every build",
        "decoder_wiring_check":
            "each parent's encoder output embedded in "
            "its territory, decoded through the merged "
            "decoder with cross blocks zeroed and the "
            "other territory's v_start zeroed, "
            "reproduces the parent's tours exactly "
            "(host and donor)",
        "end_to_end_territory_silencing_parity":
            "NOT claimed and does not hold by design: "
            "block 1's LayerNorm normalises over the "
            "joined width 4, and the decoder cross "
            "blocks couple the territories through "
            "v_start at the first pick. The birth gap "
            "is the honest whole-network number.",
    },
    "stitching_mask": {
        "definition": "stitching genes (may mutate in "
                      "the evolution) = every "
                      "cross-territory entry of the "
                      "widened block 1, the WHOLE "
                      "joining block, and the merged "
                      "decoder's cross blocks. Backbone "
                      "genes (never mutate) = both "
                      "parents' own weights, including "
                      "their decoder blocks, b_qd and "
                      "v_start.",
        "gene_order": KEYS,
        "total": n_total, "backbone": n_backbone,
        "stitching": n_stitch,
        "per_array": per_array,
        "flat_mask": [int(x) for x in flat_mask],
    },
    "frankenstein5": {
        "shape": {"stream_width": dF,
                  "block1_ffn_width": U,
                  "joining_block_ffn_width": U2,
                  "blocks": 2, "heads_per_block": 1,
                  "decoders": 1,
                  "total_params": n_total},
        "validation_gap_pct": round(frank_gap * 100, 4),
        "mean_length": float(frank_lens.mean()),
        "comparisons_pct": {
            "species_8_parent": round(gaps[8] * 100, 4),
            "species_9_parent": round(gaps[donor_id] * 100, 4),
            "frankenstein_1": F1_GAP_PCT,
            "frankenstein_4": F4_GAP_PCT,
        },
        "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
        "engine_fit": "DOES NOT FIT the current "
                      "ArchSpec (same finding as "
                      "Frankensteins 1-4): explicit "
                      "weight set with its own forward "
                      "pass, parity-verified on the "
                      "parents.",
    },
}
with open(HERE / "frankenstein_stitch5_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_stitch5_2026-11-25.json")
log("DONE build")
