#!/usr/bin/env python3
"""Stitch the two Frankensteins (sponsor direction,
Roman, 2026-11-26 08:21 EST): stitch the two evolved
Frankenstein 4 champions together by the Frankenstein
4 rule, then (in the companion script) evolve the
seams.

Frankenstein 4 is the rule for the engine: isolate
the main path in each parent, put the two paths side
by side sharing no node, holes filled with tiny
random synapses (std 0.03), ONE merged decoder, no
post-decoder layer, no averaging.

PARENTS (both are themselves Frankenstein networks
with two blocks and a merged decoder; neither is an
engine ArchSpec shape):
  - Frank #1: the evolved Frankenstein 4 champion
    (engine_runs/frankenstein4_evolution/
    frankenstein_4_evolved_champion.npz), validation
    gap 6.8959%. Stream width 4; block 1 FFN 6,
    block 2 (its joining block) FFN 6; merged
    decoder 4x4; 344 genes.
  - Frank #2: the second build's evolved champion
    (engine_runs/frankenstein4_second_evolution/
    frankenstein4_second_evolved_champion.npz),
    validation gap 7.017%. Stream width 4; block 1
    FFN 4, block 2 (its joining block) FFN 4; merged
    decoder 4x4; 308 genes.
The two Franks share NO common ancestor (different
founding runs, different seeds: fresh50_v2 seed
20262500 and fresh50_v3 seed 20262700). There is
nothing to commonize: every placed entry is
backbone, all connective material is stitching.

STEP 1 - MEASURE AND EXTRACT. The signal measurement
of stitch #2 (std of the flow through each synapse =
presynaptic activation times the weight, over the
fixed 200-problem validation set; attention mixing
as per-pair contribution conduits), generalised in
the Frankenstein Squared build to any number of
blocks and any width, run on BOTH Franks. Corrected
path rule (sponsor, 2026-11-25 10:04 EST): a weight
synapse is on the main path iff its flow std is at
least HALF the largest flow std among the weight
synapses feeding the same node (relative bar inside
the fan-in group; residual stream edges and
attention/logit conduits are structural connectors,
not competitors) AND it lies on a connected route
from the entry nodes to the decoder output. The
generalisation is validated in this build exactly
as in the Squared build: run on species 8 and
species 9 it must reproduce stitch #2's recorded
path sets exactly (13 and 26 synapses).

STEP 2 - BUILD. Stream width 8 (Frank #1 on
coordinates 0-3, Frank #2 on 4-7). Each Frank's
path keeps its own block structure:
  - two-Franks block 1: Frank #1's block-1 path and
    Frank #2's block-1 path side by side;
  - two-Franks block 2: Frank #1's block-2 path and
    Frank #2's block-2 path side by side;
  - two-Franks block 3: ONE joining block before
    the decoder spanning the full width 8, ALL tiny
    random std 0.03. Its FFN width = the sum of the
    two Franks' FFN path widths counted in the block
    that feeds the joining block on each side (each
    Frank's block 2) -- so the joining FFN width
    equals the two-Franks block-2 FFN width, exactly
    as in stitches #2/#4 and the Squared build,
    where the joining FFN width equalled the
    preceding block's FFN width.
Path entries keep their EXACT source weights;
territory vectors (embedding bias, layer norms,
biases, FFN-unit biases of path units) are the
source Franks' own values at their territory
positions, as in stitches #2/#4 and the Squared
build; everything else is tiny random filler,
std 0.03, ONE draw from ONE rng stream (seed
recorded), no retries.
  - Merged decoder over width 8: Frank #1's decoder
    (4x4, itself a merged decoder) EXACT in its
    diagonal block, Frank #2's decoder (4x4) EXACT
    in its diagonal block, cross blocks tiny random
    std 0.03 from the same single draw; b_qd and
    v_start are the two Franks' concatenations
    placed by the maps. No post-decoder layer, no
    logit averaging.

DECODER / ATTENTION SCALE: attn_scale = SCALE =
1/sqrt(2) = 0.7071067811865475 everywhere (every
block's attention and the decoder logits). This is
the engine's initial-arch scale at the original
parents' width d=2, the value BOTH Franks were
built and evolved under (their forward passes in
frankenstein_evolve4_2026-11-25.py and
frankenstein4_second_evolve_2026-11-26.py used this
same SCALE for all blocks and the decoder).
Keeping it means every copied weight keeps the
scale it was built/evolved under.

MASK: backbone = every entry copied exact from a
Frank (path synapses, territory vectors, the two
decoder blocks, b_qd, v_start); stitching = every
drawn entry (filler, the whole joining block, the
decoder cross blocks). The build classifies each
entry at the moment it is placed or drawn, so no
entry's origin is ambiguous; the mask is verified
by value below.

Scored ONCE on the fixed 200-problem validation
set (val_seed 20261116): the birth gap.

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_twofranks_stitch_2026-11-26.py
Outputs: frankenstein_twofranks_genome.npz,
frankenstein_twofranks_stitch_2026-11-26.json,
frankenstein_twofranks_state.json (diagram input).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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
RUN_V2 = HERE / "engine_runs" / "fresh50_v2"
F1_CHAMP = (HERE / "engine_runs" / "frankenstein4_evolution"
            / "frankenstein_4_evolved_champion.npz")
F2_CHAMP = (HERE / "engine_runs" / "frankenstein4_second_evolution"
            / "frankenstein4_second_evolved_champion.npz")
SIGMA_FILL = 0.03
FILL_SEED = 20261201
EXPECTED_F1_GAP_PCT = 6.8959
EXPECTED_F2_GAP_PCT = 7.017

# ================================================================
# Setup (identical to the earlier stitches, so SCALE /
# EPS / the validation set are the same objects)
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
log(f"attn_scale = {SCALE!r}: the engine's initial-arch "
    f"scale at d=2; both Franks were built and evolved "
    f"under this value")

# ================================================================
# Generic forward + greedy pointer decode (stitch #2's,
# valid for any block list / width)
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


def gap_of(W, X, blocks):
    Y = forward_general(W, X, blocks)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    gaps = lens / OPT - 1.0
    return float(gaps.mean()), picks, lens


# ================================================================
# Generic signal measurement + corrected path rule
# (verbatim from the Frankenstein Squared build:
# compute_flows_gen / extract_path / path_report)
# ================================================================
def compute_flows_gen(W, blocks, d, Us):
    X = X_val
    B, n, in_dim = X.shape
    edges = []

    def add(src, dst, kind, entry, i, j, samples):
        edges.append({"src": src, "dst": dst, "kind": kind,
                      "entry": entry, "i": int(i), "j": int(j),
                      "samples": np.asarray(
                          samples, dtype=float).ravel()})

    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for c in range(in_dim):
        for k in range(d):
            add(f"in{c}", f"s0_{k}", "weight", "embed.W_in",
                c, k, X[:, :, c] * W["embed.W_in"][c, k])
    for bi, blk in enumerate(blocks):
        s_in, s_mid, s_out = f"s{2*bi}", f"s{2*bi+1}", f"s{2*bi+2}"
        U = Us[bi]
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
        for nm, tgt in (("W_q", "q"), ("W_k", "k"), ("W_v", "v")):
            M = W[f"{blk}.{nm}"]
            for k in range(d):
                for j in range(d):
                    add(f"{s_in}_{k}", f"{tgt}{bi}_{j}", "weight",
                        f"{blk}.{nm}", k, j, Hb[:, :, k] * M[k, j])
        for j in range(d):
            qflow = Q[:, :, j] * np.einsum(
                "bts,bs->bt", Atn, K[:, :, j] * SCALE)
            add(f"q{bi}_{j}", f"mix{bi}_{j}", "conduit", "attnQ",
                -1, j, qflow)
            kflow = K[:, :, j] * np.einsum(
                "bts,bt->bs", Atn, Q[:, :, j] * SCALE)
            add(f"k{bi}_{j}", f"mix{bi}_{j}", "conduit", "attnK",
                -1, j, kflow)
            aflow = Atn * V[:, :, j][:, None, :]
            add(f"v{bi}_{j}", f"mix{bi}_{j}", "conduit", "attnA",
                -1, j, aflow)
        for k in range(d):
            for j in range(d):
                add(f"mix{bi}_{k}", f"{s_mid}_{j}", "weight",
                    f"{blk}.W_o", k, j,
                    Ah[:, :, k] * W[f"{blk}.W_o"][k, j])
        A1 = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        for k in range(d):
            add(f"{s_in}_{k}", f"{s_mid}_{k}", "residual",
                "residual1", k, k, A[:, :, k])
        Hf = ln_np(A1, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A1f = np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"])
        for k in range(d):
            for u in range(U):
                add(f"{s_mid}_{k}", f"ffn{bi}_{u}", "weight",
                    f"{blk}.W_1", k, u,
                    Hf[:, :, k] * W[f"{blk}.W_1"][k, u])
        for u in range(U):
            for j in range(d):
                add(f"ffn{bi}_{u}", f"{s_out}_{j}", "weight",
                    f"{blk}.W_2", u, j,
                    A1f[:, :, u] * W[f"{blk}.W_2"][u, j])
        A = A1 + A1f @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
        for k in range(d):
            add(f"{s_mid}_{k}", f"{s_out}_{k}", "residual",
                "residual2", k, k, A1[:, :, k])
    # decoder flows along the network's own greedy tour
    _, picks, _ = gap_of(W, X, blocks)
    gmean = A.mean(axis=1)
    states = np.zeros((B, n, d))
    for t in range(n):
        last = W["head.v_start"] if t == 0 else \
            A[np.arange(B), picks[:, t - 1]]
        states[:, t, :] = gmean + last
    qd = states @ W["head.W_qd"] + W["head.b_qd"]
    s_last = f"s{2*len(blocks)}"
    for k in range(d):
        for j in range(d):
            add(f"{s_last}_{k}", f"dec_{j}", "weight",
                "head.W_qd", k, j,
                states[:, :, k] * W["head.W_qd"][k, j])
    for j in range(d):
        add(f"dec_{j}", "pick", "conduit", "logit", -1, j,
            A[:, :, j][:, None, :] * qd[:, :, j][:, :, None]
            * SCALE)
    for e in edges:
        e["mean"] = float(e["samples"].mean())
        e["std"] = float(e["samples"].std())
        del e["samples"]
    fanin = defaultdict(float)
    fanmax = defaultdict(float)
    for e in edges:
        fanin[e["dst"]] += e["std"]
        fanmax[e["dst"]] = max(fanmax[e["dst"]], e["std"])
    for e in edges:
        tot = fanin[e["dst"]]
        e["share"] = e["std"] / tot if tot > 0 else 0.0
        mx = fanmax[e["dst"]]
        e["rel"] = e["std"] / mx if mx > 0 else 0.0
    return edges


def extract_path(edges):
    """Corrected path rule (stitch #2): relative fan-in
    bar over weight synapses + entry-to-exit route
    connectivity through passing synapses and the
    structural connectors."""
    wmax = defaultdict(float)
    for e in edges:
        if e["kind"] == "weight":
            wmax[e["dst"]] = max(wmax[e["dst"]], e["std"])
    for e in edges:
        e["passes"] = bool(
            e["kind"] == "weight" and e["std"] > 0
            and e["std"] >= 0.5 * wmax[e["dst"]])

    def routable(e):
        return e["kind"] in ("conduit", "residual") \
            or e["passes"]

    fwd = defaultdict(set)
    bwd = defaultdict(set)
    for e in edges:
        if routable(e):
            fwd[e["src"]].add(e["dst"])
            bwd[e["dst"]].add(e["src"])

    def reach(starts, graph):
        seen = set(starts)
        stack = list(starts)
        while stack:
            nd = stack.pop()
            for m in graph[nd]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    core = reach({"in0", "in1"}, fwd) & reach({"pick"}, bwd)
    for e in edges:
        e["on_path"] = bool(routable(e)
                            and e["src"] in core
                            and e["dst"] in core)
    return edges


def path_report(edges, blocks):
    wp = [e for e in edges
          if e["on_path"] and e["kind"] == "weight"]
    units = {}
    for blk in blocks:
        units[blk] = sorted(
            {e["j"] for e in wp if e["entry"] == f"{blk}.W_1"}
            | {e["i"] for e in wp if e["entry"] == f"{blk}.W_2"})
    return {
        "n_weight_synapses_total": int(sum(
            1 for e in edges if e["kind"] == "weight")),
        "n_passing_weight_synapses": int(sum(
            1 for e in edges
            if e["kind"] == "weight" and e["passes"])),
        "n_path_weight_synapses": len(wp),
        "path_weight_synapses": [
            {"entry": e["entry"], "i": e["i"], "j": e["j"],
             "share": round(e["share"], 4), "std": e["std"]}
            for e in wp],
        "ffn_units_on_path_by_block": units,
        "path_set": {(e["entry"], e["i"], e["j"]) for e in wp},
    }


def parent_weights(genome):
    """Engine genome -> generic weight dict (verbatim
    from stitch #2); used only for the species 8/9
    machinery validation."""
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


# ================================================================
# Parents: the two evolved Frankenstein champions,
# loaded and re-scored
# ================================================================
def load_frank(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    W = {k: np.asarray(z[k], dtype=float)
         for k in z.files if k != "meta"}
    return W, meta


F1W, F1_META = load_frank(F1_CHAMP)
F2W, F2_META = load_frank(F2_CHAMP)
FRANK_BLOCKS = ("b0", "b1")
F1_US = (6, 6)     # block FFN widths of Frank #1
F2_US = (4, 4)     # block FFN widths of Frank #2
for tag, W, want_us in (("Frank #1", F1W, F1_US),
                        ("Frank #2", F2W, F2_US)):
    assert W["embed.W_in"].shape == (2, 4), tag
    assert W["head.W_qd"].shape == (4, 4), tag
    for bi, blk in enumerate(FRANK_BLOCKS):
        assert W[f"{blk}.W_1"].shape == (4, want_us[bi]), tag
        assert W[f"{blk}.W_2"].shape == (want_us[bi], 4), tag
    log(f"{tag}: shapes confirmed from the genome: stream "
        f"4, blocks b0 (FFN {want_us[0]}) and b1 (FFN "
        f"{want_us[1]}), decoder 4x4, "
        f"{sum(v.size for v in W.values())} genes")

f1_gap_mine, _, _ = gap_of(F1W, X_val, FRANK_BLOCKS)
log(f"Frank #1 re-scored {f1_gap_mine*100:.4f}% (record "
    f"{EXPECTED_F1_GAP_PCT:.4f}%)")
assert abs(f1_gap_mine * 100 - EXPECTED_F1_GAP_PCT) < 5e-5
f2_gap_mine, _, _ = gap_of(F2W, X_val, FRANK_BLOCKS)
log(f"Frank #2 re-scored {f2_gap_mine*100:.4f}% (record "
    f"{EXPECTED_F2_GAP_PCT:.4f}%)")
assert abs(f2_gap_mine * 100 - EXPECTED_F2_GAP_PCT) < 5e-5

# ================================================================
# STEP 1 - measure + extract both paths; validate the
# generalised machinery on species 8 / 9 first (as in
# the Squared build)
# ================================================================
S2RES = json.load(open(HERE / "frankenstein_stitch2_"
                             "2026-11-25.json"))
ck = np.load(RUN_V2 / "checkpoint.npz", allow_pickle=True)
sp_best = {int(s): np.asarray(g, dtype=float)
           for s, g in zip(ck["species_ids"], ck["best_genomes"])}
for tag, sp in (("host", 8), ("donor", 9)):
    W = parent_weights(sp_best[sp])
    rep = path_report(extract_path(
        compute_flows_gen(W, ("b0",), 2, (8,))), ("b0",))
    want = {(e["entry"], e["i"], e["j"]) for e in
            S2RES["parents"][tag]["path_weight_synapses"]}
    ok = rep["path_set"] == want
    log(f"machinery validation on species {sp}: generalised "
        f"path {rep['n_path_weight_synapses']} synapses vs "
        f"stitch #2 record {len(want)}; sets identical: {ok}")
    assert ok

f1_edges = extract_path(compute_flows_gen(F1W, FRANK_BLOCKS,
                                          4, F1_US))
F1REP = path_report(f1_edges, FRANK_BLOCKS)
log(f"Frank #1 path: {F1REP['n_path_weight_synapses']} of "
    f"{F1REP['n_weight_synapses_total']} weight synapses "
    f"({F1REP['n_passing_weight_synapses']} passing); FFN "
    f"units on path by block: "
    f"{F1REP['ffn_units_on_path_by_block']}")

f2_edges = extract_path(compute_flows_gen(F2W, FRANK_BLOCKS,
                                          4, F2_US))
F2REP = path_report(f2_edges, FRANK_BLOCKS)
log(f"Frank #2 path: {F2REP['n_path_weight_synapses']} of "
    f"{F2REP['n_weight_synapses_total']} weight synapses "
    f"({F2REP['n_passing_weight_synapses']} passing); FFN "
    f"units on path by block: "
    f"{F2REP['ffn_units_on_path_by_block']}")

# ================================================================
# STEP 2 - build the Two-Franks stitch
# ================================================================
# territory maps
A_MAP = {0: 0, 1: 1, 2: 2, 3: 3}          # Frank #1 -> joined
B_MAP = {0: 4, 1: 5, 2: 6, 3: 7}          # Frank #2 -> joined
D_TF = 8
A_U0 = F1REP["ffn_units_on_path_by_block"]["b0"]
A_U1 = F1REP["ffn_units_on_path_by_block"]["b1"]
B_U0 = F2REP["ffn_units_on_path_by_block"]["b0"]
B_U1 = F2REP["ffn_units_on_path_by_block"]["b1"]
AU0_MAP = {u: i for i, u in enumerate(A_U0)}       # tf b0 units
BU0_MAP = {u: len(A_U0) + i for i, u in enumerate(B_U0)}
AU1_MAP = {u: i for i, u in enumerate(A_U1)}       # tf b1 units
BU1_MAP = {u: len(A_U1) + i for i, u in enumerate(B_U1)}
U0 = len(A_U0) + len(B_U0)
U1 = len(A_U1) + len(B_U1)
U2 = U1   # joining FFN width (see docstring reading)
log(f"two-Franks shape: stream {D_TF} (Frank #1 4 + "
    f"Frank #2 4); block FFN widths {U0} (F1 {len(A_U0)} "
    f"+ F2 {len(B_U0)}), {U1} (F1 {len(A_U1)} + F2 "
    f"{len(B_U1)}), joining {U2}")

BLOCK_KEYS = ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
              "W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v",
              "b_o", "b_2", "W_1", "W_2", "b_1")
KEYS_TF = (["embed.W_in", "embed.b_in"]
           + [f"{b}.{nm}" for b in ("b0", "b1", "b2")
              for nm in BLOCK_KEYS]
           + ["head.W_qd", "head.v_start", "head.b_qd"])

SHAPES = {"embed.W_in": (2, D_TF), "embed.b_in": (D_TF,),
          "head.W_qd": (D_TF, D_TF), "head.v_start": (D_TF,),
          "head.b_qd": (D_TF,)}
for b, U in (("b0", U0), ("b1", U1), ("b2", U2)):
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
               "b_q", "b_k", "b_v", "b_o", "b_2"):
        SHAPES[f"{b}.{nm}"] = (D_TF,)
    for nm in ("W_q", "W_k", "W_v", "W_o"):
        SHAPES[f"{b}.{nm}"] = (D_TF, D_TF)
    SHAPES[f"{b}.W_1"] = (D_TF, U)
    SHAPES[f"{b}.W_2"] = (U, D_TF)
    SHAPES[f"{b}.b_1"] = (U,)

rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


TF = {}
STITCH = {}   # per-array bool mask, True = stitching (drawn)


def new_array(key):
    TF[key] = fill(SHAPES[key])
    STITCH[key] = np.ones(SHAPES[key], dtype=bool)


def place(key, pos, value):
    """Copy a source-Frank value in exact; the entry
    becomes backbone."""
    TF[key][pos] = value
    STITCH[key][pos] = False


A_PATH = F1REP["path_set"]
B_PATH = F2REP["path_set"]
TF_PATH_A = set()   # (joined entry, i, j) placed from F1's path
TF_PATH_B = set()

# ---- embed
new_array("embed.W_in")
for (entry, i, j) in sorted(A_PATH):
    if entry == "embed.W_in":
        place("embed.W_in", (i, A_MAP[j]), F1W[entry][i, j])
        TF_PATH_A.add(("embed.W_in", i, A_MAP[j]))
for (entry, i, j) in sorted(B_PATH):
    if entry == "embed.W_in":
        place("embed.W_in", (i, B_MAP[j]), F2W[entry][i, j])
        TF_PATH_B.add(("embed.W_in", i, B_MAP[j]))
new_array("embed.b_in")
for k_old, k_new in A_MAP.items():
    place("embed.b_in", (k_new,), F1W["embed.b_in"][k_old])
for k_old, k_new in B_MAP.items():
    place("embed.b_in", (k_new,), F2W["embed.b_in"][k_old])

VEC_NAMES = ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
             "b_q", "b_k", "b_v", "b_o", "b_2")
MAT_NAMES = ("W_q", "W_k", "W_v", "W_o")


def build_block(tf_blk):
    """Both Franks have a block named tf_blk ('b0' or
    'b1'); place each side's path and territory from
    its own block of that name."""
    src_blk = tf_blk
    a_umap = AU0_MAP if tf_blk == "b0" else AU1_MAP
    b_umap = BU0_MAP if tf_blk == "b0" else BU1_MAP
    for nm in VEC_NAMES:
        key = f"{tf_blk}.{nm}"
        new_array(key)
        for k_old, k_new in A_MAP.items():
            place(key, (k_new,), F1W[f"{src_blk}.{nm}"][k_old])
        for k_old, k_new in B_MAP.items():
            place(key, (k_new,), F2W[f"{src_blk}.{nm}"][k_old])
    for nm in MAT_NAMES:
        key = f"{tf_blk}.{nm}"
        new_array(key)
        for (entry, i, j) in sorted(A_PATH):
            if entry == f"{src_blk}.{nm}":
                place(key, (A_MAP[i], A_MAP[j]),
                      F1W[entry][i, j])
                TF_PATH_A.add((key, A_MAP[i], A_MAP[j]))
        for (entry, i, j) in sorted(B_PATH):
            if entry == f"{src_blk}.{nm}":
                place(key, (B_MAP[i], B_MAP[j]),
                      F2W[entry][i, j])
                TF_PATH_B.add((key, B_MAP[i], B_MAP[j]))
    # W_1 (stream x units), W_2 (units x stream), b_1
    key = f"{tf_blk}.W_1"
    new_array(key)
    for (entry, i, j) in sorted(A_PATH):
        if entry == f"{src_blk}.W_1":
            place(key, (A_MAP[i], a_umap[j]), F1W[entry][i, j])
            TF_PATH_A.add((key, A_MAP[i], a_umap[j]))
    for (entry, i, j) in sorted(B_PATH):
        if entry == f"{src_blk}.W_1":
            place(key, (B_MAP[i], b_umap[j]), F2W[entry][i, j])
            TF_PATH_B.add((key, B_MAP[i], b_umap[j]))
    key = f"{tf_blk}.W_2"
    new_array(key)
    for (entry, i, j) in sorted(A_PATH):
        if entry == f"{src_blk}.W_2":
            place(key, (a_umap[i], A_MAP[j]), F1W[entry][i, j])
            TF_PATH_A.add((key, a_umap[i], A_MAP[j]))
    for (entry, i, j) in sorted(B_PATH):
        if entry == f"{src_blk}.W_2":
            place(key, (b_umap[i], B_MAP[j]), F2W[entry][i, j])
            TF_PATH_B.add((key, b_umap[i], B_MAP[j]))
    key = f"{tf_blk}.b_1"
    new_array(key)
    for u_old, u_new in a_umap.items():
        place(key, (u_new,), F1W[f"{src_blk}.b_1"][u_old])
    for u_old, u_new in b_umap.items():
        place(key, (u_new,), F2W[f"{src_blk}.b_1"][u_old])


build_block("b0")
build_block("b1")

# ---- joining block (two-Franks b2): ALL tiny random
# filler; LN gains 1 + N(0, 0.01) as in the earlier builds
for nm in BLOCK_KEYS:
    key = f"b2.{nm}"
    if nm in ("lnA_gain", "lnF_gain"):
        TF[key] = 1.0 + rng_fill.normal(0.0, 0.01, D_TF)
        STITCH[key] = np.ones(SHAPES[key], dtype=bool)
    else:
        new_array(key)

# ---- merged decoder over width 8: each Frank's own
# 4x4 decoder EXACT in its diagonal block
new_array("head.W_qd")
for i in range(4):
    for j in range(4):
        place("head.W_qd", (A_MAP[i], A_MAP[j]),
              F1W["head.W_qd"][i, j])
        place("head.W_qd", (B_MAP[i], B_MAP[j]),
              F2W["head.W_qd"][i, j])
TF["head.b_qd"] = np.zeros(D_TF)
STITCH["head.b_qd"] = np.ones(D_TF, dtype=bool)
TF["head.v_start"] = np.zeros(D_TF)
STITCH["head.v_start"] = np.ones(D_TF, dtype=bool)
for k_old, k_new in A_MAP.items():
    place("head.b_qd", (k_new,), F1W["head.b_qd"][k_old])
    place("head.v_start", (k_new,), F1W["head.v_start"][k_old])
for k_old, k_new in B_MAP.items():
    place("head.b_qd", (k_new,), F2W["head.b_qd"][k_old])
    place("head.v_start", (k_new,), F2W["head.v_start"][k_old])

assert set(TF.keys()) == set(KEYS_TF)
for k in KEYS_TF:
    assert TF[k].shape == SHAPES[k], k
    assert STITCH[k].shape == SHAPES[k], k


# ================================================================
# Construction checks (ALL must pass)
# ================================================================
checks = {}

# (1) every path entry bit-exact against its source
# Frank, at its placed position, and classed backbone
# there. (head.W_qd path entries are covered by check
# 2: both decoder blocks are wholly exact.)
def src_of(side, entry):
    return F1W if side == "A" else F2W


UMAPS = {("A", "b0"): AU0_MAP, ("A", "b1"): AU1_MAP,
         ("B", "b0"): BU0_MAP, ("B", "b1"): BU1_MAP}
n_path_checked = 0
for side, path, smap in (
        ("A", A_PATH, A_MAP),
        ("B", B_PATH, B_MAP)):
    for (entry, i, j) in sorted(path):
        if entry == "head.W_qd":
            continue
        blk = entry.split(".")[0] if "." in entry else None
        if entry == "embed.W_in":
            pos = ("embed.W_in", i, smap[j])
        elif entry.endswith(".W_1"):
            pos = (entry, smap[i], UMAPS[(side, blk)][j])
        elif entry.endswith(".W_2"):
            pos = (entry, UMAPS[(side, blk)][i], smap[j])
        else:
            pos = (entry, smap[i], smap[j])
        k2, i2, j2 = pos
        assert TF[k2][i2, j2] == src_of(side, entry)[entry][i, j], \
            f"path entry not exact: side {side} {entry} {(i, j)}"
        assert not STITCH[k2][i2, j2], \
            f"path entry classed stitching: side {side} {entry}"
        n_path_checked += 1
checks["path_entries_bit_exact"] = True
checks["path_entries_checked"] = n_path_checked
log(f"construction check 1: {n_path_checked} placed path "
    f"entries bit-exact against their source Franks "
    f"and classed backbone: True")

# (2) merged decoder: both 4x4 blocks, b_qd, v_start exact
chk_a_block = bool(np.array_equal(
    TF["head.W_qd"][np.ix_([0, 1, 2, 3], [0, 1, 2, 3])],
    F1W["head.W_qd"]))
chk_b_block = bool(np.array_equal(
    TF["head.W_qd"][np.ix_([4, 5, 6, 7], [4, 5, 6, 7])],
    F2W["head.W_qd"]))
chk_bqd = bool(np.array_equal(
    TF["head.b_qd"],
    np.concatenate([F1W["head.b_qd"], F2W["head.b_qd"]])))
chk_vs = bool(np.array_equal(
    TF["head.v_start"],
    np.concatenate([F1W["head.v_start"],
                    F2W["head.v_start"]])))
cross = np.concatenate([
    TF["head.W_qd"][np.ix_([0, 1, 2, 3],
                           [4, 5, 6, 7])].ravel(),
    TF["head.W_qd"][np.ix_([4, 5, 6, 7],
                           [0, 1, 2, 3])].ravel()])
checks["decoder_frank1_block_bit_exact"] = chk_a_block
checks["decoder_frank2_block_bit_exact"] = chk_b_block
checks["decoder_b_qd_bit_exact"] = chk_bqd
checks["decoder_v_start_bit_exact"] = chk_vs
checks["decoder_cross_max_abs"] = float(np.abs(cross).max())
checks["decoder_cross_std"] = float(cross.std())
log(f"construction check 2: merged decoder — Frank #1 "
    f"4x4 block exact {chk_a_block}, Frank #2 4x4 block "
    f"exact {chk_b_block}, b_qd exact {chk_bqd}, v_start "
    f"exact {chk_vs}; cross blocks max |w| "
    f"{np.abs(cross).max():.4f}, std {cross.std():.4f}")
assert chk_a_block and chk_b_block and chk_bqd and chk_vs

# (3) mask accounting: every gene classified exactly once
per_array = {k: {"shape": list(TF[k].shape),
                 "total": int(TF[k].size),
                 "stitching": int(STITCH[k].sum()),
                 "backbone": int(TF[k].size - STITCH[k].sum())}
             for k in KEYS_TF}
flat_mask = np.concatenate([STITCH[k].ravel() for k in KEYS_TF])
n_total = int(flat_mask.size)
n_stitch = int(flat_mask.sum())
n_backbone = n_total - n_stitch
totals = {"total": sum(v["total"] for v in per_array.values()),
          "stitching": sum(v["stitching"]
                           for v in per_array.values()),
          "backbone": sum(v["backbone"]
                          for v in per_array.values())}
checks["gene_accounting"] = totals
assert totals == {"total": n_total, "stitching": n_stitch,
                  "backbone": n_backbone}
checks["ambiguous_entries"] = []
log(f"construction check 3: gene accounting {totals}; "
    f"every entry was classified at placement/draw time, "
    f"so no entry's origin is ambiguous")

# ================================================================
# Score ONCE on the fixed validation set: the birth gap
# ================================================================
birth_gap, _, birth_lens = gap_of(TF, X_val, ("b0", "b1", "b2"))
log(f"TWO FRANKS validation gap (birth): "
    f"{birth_gap*100:.2f}% — Frank #1 "
    f"{f1_gap_mine*100:.2f}%, Frank #2 "
    f"{f2_gap_mine*100:.2f}%")

# ================================================================
# Save genome + results + diagram state
# ================================================================
meta = {
    "kind": "frankenstein_twofranks_stitch",
    "date": "2026-11-26",
    "design": "Frankenstein 4 rule: the two evolved "
              "Frankenstein 4 champions' extracted "
              "paths side by side, holes tiny random "
              "std 0.03, one joining block, one merged "
              "decoder over the joined width",
    "frank1": "evolved Frankenstein 4 champion "
              "(frankenstein_4_evolved_champion.npz)",
    "frank2": "second build's evolved champion "
              "(frankenstein4_second_evolved_champion.npz)",
    "frank1_gap_pct": round(f1_gap_mine * 100, 4),
    "frank2_gap_pct": round(f2_gap_mine * 100, 4),
    "common_ancestor": "none: different founding runs "
                       "and seeds (fresh50_v2 seed "
                       "20262500; fresh50_v3 seed "
                       "20262700); nothing commonized",
    "d": D_TF, "blocks": 3, "heads_per_block": 1,
    "ffn_hidden": {"b0": U0, "b1": U1, "b2_joining": U2},
    "frank1_coord_map": {str(k): v for k, v in A_MAP.items()},
    "frank2_coord_map": {str(k): v for k, v in B_MAP.items()},
    "frank1_unit_maps": {
        "b0": {str(k): v for k, v in AU0_MAP.items()},
        "b1": {str(k): v for k, v in AU1_MAP.items()}},
    "frank2_unit_maps": {
        "b0": {str(k): v for k, v in BU0_MAP.items()},
        "b1": {str(k): v for k, v in BU1_MAP.items()}},
    "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
    "gene_order": KEYS_TF,
    "genes_total": n_total, "genes_backbone": n_backbone,
    "genes_stitching": n_stitch,
    "birth_gap_pct": round(birth_gap * 100, 4),
}
np.savez(HERE / "frankenstein_twofranks_genome.npz",
         **{k: TF[k] for k in KEYS_TF},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_twofranks_genome.npz")


def rep_json(rep):
    return {
        "n_weight_synapses_total":
            rep["n_weight_synapses_total"],
        "n_passing_weight_synapses":
            rep["n_passing_weight_synapses"],
        "n_path_weight_synapses":
            rep["n_path_weight_synapses"],
        "path_weight_synapses":
            rep["path_weight_synapses"],
        "ffn_units_on_path_by_block":
            rep["ffn_units_on_path_by_block"],
    }


results = {
    "design": "sponsor direction, 2026-11-26 08:21 EST: "
              "stitch the two Frankensteins together — "
              "the two evolved Frankenstein 4 champions "
              "— by the Frankenstein 4 rule",
    "parents": {
        "frank1_evolved_frankenstein4": {
            "source": "engine_runs/frankenstein4_evolution/"
                      "frankenstein_4_evolved_champion.npz",
            "validation_gap_pct": round(f1_gap_mine * 100, 4),
            "shape": {"stream_width": 4, "blocks": 2,
                      "ffn_hidden": [6, 6],
                      "genes": 344},
            "path": rep_json(F1REP)},
        "frank2_evolved_frankenstein4_second": {
            "source": "engine_runs/frankenstein4_second_"
                      "evolution/frankenstein4_second_"
                      "evolved_champion.npz",
            "validation_gap_pct": round(f2_gap_mine * 100, 4),
            "shape": {"stream_width": 4, "blocks": 2,
                      "ffn_hidden": [4, 4],
                      "genes": 308},
            "path": rep_json(F2REP)},
    },
    "path_rule": "corrected rule (sponsor, 2026-11-25 "
                 "10:04 EST): flow = presynaptic "
                 "activation times weight over the fixed "
                 "200-problem validation set; a weight "
                 "synapse passes iff its flow std is at "
                 "least half the largest flow std among "
                 "the weight synapses feeding the same "
                 "node; the main path is the passing "
                 "synapses on a connected entry-to-exit "
                 "route (routes may step through the "
                 "residual and attention/logit conduits, "
                 "which are structural connectors, not "
                 "competitors). The generalised machinery "
                 "was validated against stitch #2: run on "
                 "species 8 and 9 it reproduces the "
                 "recorded path sets exactly (13 and 26 "
                 "synapses).",
    "attn_scale": {
        "value": SCALE,
        "why": "the engine's initial-arch scale at "
               "d=2 (arch.attn_scale of "
               "be.initial_arch(cfg, 2)). Both Franks "
               "were built and evolved under this value "
               "for every block's attention and the "
               "decoder logits; the two-Franks network "
               "keeps it so every copied weight retains "
               "the scale it was built/evolved under. "
               "It is used for all three blocks' "
               "attention and the decoder logits, as in "
               "every earlier build."},
    "layout": "stream width 8: Frank #1's territory on "
              "coordinates 0-3, Frank #2's on 4-7. "
              "Two-Franks block 1 carries Frank #1's "
              "block-1 path and Frank #2's block-1 path "
              "side by side; block 2 carries each "
              "Frank's block-2 path in the same way "
              "(both Franks have two blocks, so each "
              "Frank's extracted branch material is "
              "placed in its original block order, "
              "block 1 then block 2, inside its own "
              "territory). Block 3 is the joining "
              "block, all filler.",
    "joining_ffn_width_reading": "the joining block's "
        "FFN width = the sum of the two Franks' FFN "
        "path widths, each Frank's width counted in "
        "the block that feeds the joining block on "
        "that side (each Frank's block 2: Frank #1 "
        f"{len(A_U1)} units, Frank #2 {len(B_U1)} "
        f"units) = {U2}, which equals the two-Franks "
        "block-2 FFN width — the same relation as "
        "stitches #2/#4 and the Squared build, where "
        "the joining FFN width equalled the preceding "
        "block's FFN width.",
    "construction_check": checks,
    "stitching_mask": {
        "definition": "backbone = every entry copied "
                      "exact from a Frank (path "
                      "synapses, territory vectors, "
                      "the two decoder blocks, b_qd, "
                      "v_start); stitching = every "
                      "drawn entry (filler, the whole "
                      "joining block, the decoder cross "
                      "blocks). Each entry was "
                      "classified when placed or drawn, "
                      "so no entry's origin is "
                      "ambiguous.",
        "gene_order": KEYS_TF,
        "total": n_total, "backbone": n_backbone,
        "stitching": n_stitch,
        "per_array": per_array,
        "flat_mask_stitching": [int(x) for x in flat_mask],
    },
    "frankenstein_twofranks": {
        "shape": {"stream_width": D_TF, "blocks": 3,
                  "ffn_hidden": {"b0": U0, "b1": U1,
                                 "b2_joining": U2},
                  "heads_per_block": 1, "decoders": 1,
                  "total_params": n_total},
        "validation_gap_pct": round(birth_gap * 100, 4),
        "validation_gap": birth_gap,
        "mean_length": float(birth_lens.mean()),
        "comparisons_pct": {
            "frank1": round(f1_gap_mine * 100, 4),
            "frank2": round(f2_gap_mine * 100, 4)},
        "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
        "engine_fit": "DOES NOT FIT the current ArchSpec "
                      "(same finding as Frankensteins "
                      "1-5 and Squared): explicit weight "
                      "set with its own forward pass.",
    },
}
with open(HERE / "frankenstein_twofranks_stitch_"
                 "2026-11-26.json", "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_twofranks_stitch_2026-11-26.json")

# ---- diagram state (JSON only, no pickle)
def panel_edges(edges, path_cls):
    return [{"src": e["src"], "dst": e["dst"],
             "kind": e["kind"], "entry": e["entry"],
             "i": e["i"], "j": e["j"],
             "rel": round(e["rel"], 4),
             "cls": path_cls if e["on_path"] else "o"}
            for e in edges]


tf_edges = []
# embed edges
for c in range(2):
    for k in range(D_TF):
        cls = "f"
        if ("embed.W_in", c, k) in TF_PATH_A:
            cls = "a"
        elif ("embed.W_in", c, k) in TF_PATH_B:
            cls = "b"
        tf_edges.append({"src": f"in{c}", "dst": f"s0_{k}",
                         "kind": "weight", "cls": cls,
                         "rel": None})
U_BY_BLOCK = (U0, U1, U2)
for bi, blk in enumerate(("b0", "b1", "b2")):
    s_in, s_mid, s_out = f"s{2*bi}", f"s{2*bi+1}", f"s{2*bi+2}"
    U = U_BY_BLOCK[bi]
    for nm, tgt in (("W_q", "q"), ("W_k", "k"), ("W_v", "v")):
        for i in range(D_TF):
            for j in range(D_TF):
                key = (f"{blk}.{nm}", i, j)
                cls = "a" if key in TF_PATH_A else \
                    ("b" if key in TF_PATH_B else "f")
                tf_edges.append({"src": f"{s_in}_{i}",
                                 "dst": f"{tgt}{bi}_{j}",
                                 "kind": "weight", "cls": cls,
                                 "rel": None})
    for src_pre in ("q", "k", "v"):
        for j in range(D_TF):
            tf_edges.append({"src": f"{src_pre}{bi}_{j}",
                             "dst": f"mix{bi}_{j}",
                             "kind": "conduit", "cls": "f",
                             "rel": None})
    for i in range(D_TF):
        for j in range(D_TF):
            key = (f"{blk}.W_o", i, j)
            cls = "a" if key in TF_PATH_A else \
                ("b" if key in TF_PATH_B else "f")
            tf_edges.append({"src": f"mix{bi}_{i}",
                             "dst": f"{s_mid}_{j}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
        tf_edges.append({"src": f"{s_in}_{i}",
                         "dst": f"{s_mid}_{i}",
                         "kind": "residual", "cls": "f",
                         "rel": None})
    for i in range(D_TF):
        for u in range(U):
            key = (f"{blk}.W_1", i, u)
            cls = "a" if key in TF_PATH_A else \
                ("b" if key in TF_PATH_B else "f")
            tf_edges.append({"src": f"{s_mid}_{i}",
                             "dst": f"ffn{bi}_{u}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
    for u in range(U):
        for j in range(D_TF):
            key = (f"{blk}.W_2", u, j)
            cls = "a" if key in TF_PATH_A else \
                ("b" if key in TF_PATH_B else "f")
            tf_edges.append({"src": f"ffn{bi}_{u}",
                             "dst": f"{s_out}_{j}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
    for k in range(D_TF):
        tf_edges.append({"src": f"{s_mid}_{k}",
                         "dst": f"{s_out}_{k}",
                         "kind": "residual", "cls": "f",
                         "rel": None})
for i in range(D_TF):
    for j in range(D_TF):
        if i < 4 and j < 4:
            cls = "a"
        elif i >= 4 and j >= 4:
            cls = "b"
        else:
            cls = "f"
        tf_edges.append({"src": f"s6_{i}", "dst": f"dec_{j}",
                         "kind": "weight", "cls": cls,
                         "rel": None})
for j in range(D_TF):
    tf_edges.append({"src": f"dec_{j}", "dst": "pick",
                     "kind": "conduit",
                     "cls": "a" if j < 4 else "b",
                     "rel": None})

state = {
    "panels": {
        "frank1": {
            "title": "Frank #1: evolved Frankenstein 4 "
                     "champion (first build)",
            "gap_pct": round(f1_gap_mine * 100, 4),
            "d": 4, "blocks": [6, 6],
            "path_synapses": F1REP["n_path_weight_synapses"],
            "total_synapses": F1REP["n_weight_synapses_total"],
            "edges": panel_edges(f1_edges, "a")},
        "frank2": {
            "title": "Frank #2: evolved Frankenstein 4 "
                     "champion (second build)",
            "gap_pct": round(f2_gap_mine * 100, 4),
            "d": 4, "blocks": [4, 4],
            "path_synapses": F2REP["n_path_weight_synapses"],
            "total_synapses": F2REP["n_weight_synapses_total"],
            "edges": panel_edges(f2_edges, "b")},
        "twofranks": {
            "title": "Two Frankensteins stitched",
            "gap_pct": round(birth_gap * 100, 4),
            "d": D_TF, "blocks": [U0, U1, U2],
            "stream_split": 4,
            "unit_splits": [len(A_U0), len(A_U1),
                            len(A_U1)],
            "edges": tf_edges},
    },
}
with open(HERE / "frankenstein_twofranks_state.json", "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_twofranks_state.json (diagram input)")
log("DONE stitch")
