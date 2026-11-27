#!/usr/bin/env python3
"""Frankenstein SQUARED (sponsor direction, Roman,
2026-11-25 22:06 EST): stitch the THIRD species
champion's branch into the evolved Frankenstein 4
champion, by the Frankenstein 4 rule, then (in the
companion script) evolve its stitching genes.

Frankenstein 4 is the rule for the engine: isolate
the main path in each parent, put the two paths
side by side sharing no node, holes filled with
tiny random synapses (std 0.03), ONE merged
decoder, no post-decoder layer, no averaging.

PARENTS:
  - Side A: the EVOLVED Frankenstein 4 champion
    (engine_runs/frankenstein4_evolution/
    frankenstein_4_evolved_champion.npz), validation
    gap 6.8959%. Stream width 4; TWO blocks (block 1
    = the species 8 / species 9 paths side by side,
    FFN 6; block 2 = its joining block, FFN 6);
    merged decoder 4x4.
  - Side B: species 10's champion of the fresh50_v2
    run, identified BY MEASUREMENT from the run
    checkpoint's recorded species best genomes (the
    source the stitch and skeleton scripts used):
    validation gap 20.3364% (the 20.34% champion).
    Stream width 2, ONE block (FFN 8), its own
    decoder 2x2.

STEP 1 - MEASURE AND EXTRACT. The signal
measurement of stitch #2 (std of the flow through
each synapse = presynaptic activation times the
weight, over the fixed 200-problem validation set;
attention mixing as per-pair contribution
conduits), generalised to any number of blocks and
any width, run on BOTH champions. Corrected path
rule (sponsor, 2026-11-25 10:04 EST): a weight
synapse is on the main path iff its flow std is at
least HALF the largest flow std among the weight
synapses feeding the same node (relative bar
inside the fan-in group; residual stream edges and
attention/logit conduits are structural connectors,
not competitors) AND it lies on a connected route
from the entry nodes to the decoder output.
The generalisation is validated in this build:
run on species 8 and species 9 it must reproduce
stitch #2's recorded path sets exactly (13 and 26
synapses).

STEP 2 - BUILD. Stream width 6 (side A on
coordinates 0-3, side B on 4-5). Each side's path
keeps its own block structure:
  - squared block 1: side A's block-1 path and
    side B's block-1 path side by side;
  - squared block 2: side A's block-2 path; side B
    has no second block, so its territory there
    (stream rows/columns and a shadow set of FFN
    units, as many as its block-1 path width) is
    filler;
  - squared block 3: ONE joining block before the
    decoder spanning the full width 6, ALL tiny
    random std 0.03. Its FFN width = the sum of
    the two sides' FFN path widths, where a side's
    FFN path width is counted in the block that
    feeds the joining block on that side (side A:
    its block 2; side B: its block 1) -- so the
    joining FFN width equals the squared block-2
    FFN width, exactly as in stitches #2/#4, where
    the joining FFN width equalled the preceding
    block's FFN width.
Path entries keep their EXACT source weights;
territory vectors (embedding bias, layer norms,
biases, FFN-unit biases of path units) are the
source champions' own values at their territory
positions, as in stitches #2/#4; everything else
is tiny random filler, std 0.03, ONE draw from
ONE rng stream (seed recorded), no retries.
  - Merged decoder over width 6: side A's evolved
    decoder (4x4) EXACT in its block, species 10's
    decoder (2x2) EXACT in its block, cross blocks
    tiny random std 0.03; b_qd and v_start are the
    two sides' concatenations placed by the maps.
    No post-decoder layer, no logit averaging.

DECODER / ATTENTION SCALE: attn_scale = SCALE =
1/sqrt(2) = 0.7071067811865475 everywhere (every
block's attention and the decoder logits). This
is the engine's initial-arch scale at the parents'
width d=2, the value BOTH source champions were
built and evolved under: species 10's native arch
is d=2, and the Frankenstein 4 champion's forward
pass (frankenstein_evolve4_2026-11-25.py) used
this same SCALE for all its blocks and its
decoder. Keeping it means every copied weight
keeps the scale it was built/evolved under.

MASK: backbone = every entry copied exact from a
source champion (path synapses, territory vectors,
the two decoder blocks, b_qd, v_start);
stitching = every drawn entry (filler, the whole
joining block, the decoder cross blocks). The
build classifies each entry at the moment it is
placed or drawn, so no entry's origin is
ambiguous; the mask is verified by value below.

Scored ONCE on the fixed 200-problem validation
set (val_seed 20261116): the birth gap.

Run: python3 frankenstein_squared_stitch_2026-11-25.py
Outputs: frankenstein_squared_genome.npz,
frankenstein_squared_stitch_2026-11-25.json,
frankenstein_squared_state.json (diagram input).
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
RUN = HERE / "engine_runs" / "fresh50_v2"
F4_CHAMP = (HERE / "engine_runs" / "frankenstein4_evolution"
            / "frankenstein_4_evolved_champion.npz")
SIGMA_FILL = 0.03
FILL_SEED = 20261130
EXPECTED_F4_GAP_PCT = 6.8959
EXPECTED_SP10_GAP_PCT = 20.3364

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
    f"scale at d=2; both source champions were built and "
    f"evolved under this value (species 10's native arch "
    f"is d=2; the Frankenstein 4 evolution used it for "
    f"every block and the decoder)")

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
# (stitch #2's compute_flows / extract_path, generalised
# to any block list, width and FFN widths; node names:
# in{c}, s{2b}_k, q{b}_j, k{b}_j, v{b}_j, mix{b}_j,
# s{2b+1}_k, ffn{b}_u, s{2b+2}_k, dec_j, pick)
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


# ================================================================
# Parents, identified by measurement
# ================================================================
# ---- side B: species 10, from the fresh50_v2 run
# checkpoint's recorded species best genomes (the
# source the stitch and skeleton scripts used)
ck = np.load(RUN / "checkpoint.npz", allow_pickle=True)
sp_best = {int(s): np.asarray(g, dtype=float)
           for s, g in zip(ck["species_ids"], ck["best_genomes"])}
sp_gaps = {s: float(domain.validation_detail(layout, g)["mean_gap"])
           for s, g in sp_best.items()}
for s in sorted(sp_gaps):
    log(f"fresh50_v2 species {s} best genome: validation "
        f"gap {sp_gaps[s]*100:.4f}%")
SP10 = 10
assert abs(sp_gaps[SP10] * 100 - EXPECTED_SP10_GAP_PCT) < 5e-5, \
    sp_gaps[SP10]
sp10_genome = sp_best[SP10]
S10W = parent_weights(sp10_genome)
log(f"side B identified: species 10 champion from the "
    f"fresh50_v2 checkpoint's recorded best genomes, "
    f"validation gap {sp_gaps[SP10]*100:.4f}%")

# ---- side A: the evolved Frankenstein 4 champion
zf = np.load(F4_CHAMP)
F4_META = json.loads(str(zf["meta"]))
F4W = {k: np.asarray(zf[k], dtype=float)
       for k in zf.files if k != "meta"}
F4_BLOCKS = ("b0", "b1")
F4_D, F4_US = 4, (6, 6)
f4_gap_mine, _, _ = gap_of(F4W, X_val, F4_BLOCKS)
log(f"side A: evolved Frankenstein 4 champion re-scored "
    f"{f4_gap_mine*100:.4f}% (record "
    f"{EXPECTED_F4_GAP_PCT:.4f}%)")
assert abs(f4_gap_mine * 100 - EXPECTED_F4_GAP_PCT) < 5e-5

# ---- parity: generic forward on species 10 must
# reproduce the engine exactly
g10, picks10, _ = gap_of(S10W, X_val, ("b0",))
lens_e, picks_e = domain.evaluate_lengths(
    layout, sp10_genome[None, :], X_val)
tours_same = bool((picks10 == picks_e[0]).all())
log(f"parity species 10: generic gap {g10*100:.6f}% vs "
    f"engine {sp_gaps[SP10]*100:.6f}%, tours identical: "
    f"{tours_same}")
assert abs(g10 - sp_gaps[SP10]) < 1e-9 and tours_same

# ================================================================
# STEP 1 - measure + extract both paths; validate the
# generalised machinery on species 8 / 9 first
# ================================================================
S2RES = json.load(open(HERE / "frankenstein_stitch2_"
                             "2026-11-25.json"))
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

f4_edges = extract_path(compute_flows_gen(F4W, F4_BLOCKS,
                                          F4_D, F4_US))
F4REP = path_report(f4_edges, F4_BLOCKS)
log(f"side A path (evolved Frankenstein 4): "
    f"{F4REP['n_path_weight_synapses']} of "
    f"{F4REP['n_weight_synapses_total']} weight synapses "
    f"({F4REP['n_passing_weight_synapses']} passing); FFN "
    f"units on path by block: "
    f"{F4REP['ffn_units_on_path_by_block']}")

s10_edges = extract_path(compute_flows_gen(S10W, ("b0",), 2, (8,)))
S10REP = path_report(s10_edges, ("b0",))
log(f"side B path (species 10): "
    f"{S10REP['n_path_weight_synapses']} of "
    f"{S10REP['n_weight_synapses_total']} weight synapses "
    f"({S10REP['n_passing_weight_synapses']} passing); FFN "
    f"units on path: {S10REP['ffn_units_on_path_by_block']}")

# ================================================================
# STEP 2 - build Frankenstein Squared
# ================================================================
# territory maps
A_MAP = {0: 0, 1: 1, 2: 2, 3: 3}          # F4 stream -> squared
B_MAP = {0: 4, 1: 5}                      # sp10 stream -> squared
D_SQ = 6
A_U0 = F4REP["ffn_units_on_path_by_block"]["b0"]
A_U1 = F4REP["ffn_units_on_path_by_block"]["b1"]
B_U0 = S10REP["ffn_units_on_path_by_block"]["b0"]
AU0_MAP = {u: i for i, u in enumerate(A_U0)}       # sq b0 units
BU0_MAP = {u: len(A_U0) + i for i, u in enumerate(B_U0)}
AU1_MAP = {u: i for i, u in enumerate(A_U1)}       # sq b1 units
# side B shadow units in squared block 2: one per B_U0 unit
BU1_MAP = {u: len(A_U1) + i for i, u in enumerate(B_U0)}
U0 = len(A_U0) + len(B_U0)
U1 = len(A_U1) + len(B_U0)
U2 = U1   # joining FFN width (see docstring reading)
log(f"squared shape: stream {D_SQ} (A 4 + B 2); block FFN "
    f"widths {U0} (A {len(A_U0)} + B {len(B_U0)}), {U1} "
    f"(A {len(A_U1)} + B shadow {len(B_U0)}), joining {U2}")

BLOCK_KEYS = ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
              "W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v",
              "b_o", "b_2", "W_1", "W_2", "b_1")
KEYS_SQ = (["embed.W_in", "embed.b_in"]
           + [f"{b}.{nm}" for b in ("b0", "b1", "b2")
              for nm in BLOCK_KEYS]
           + ["head.W_qd", "head.v_start", "head.b_qd"])

SHAPES = {"embed.W_in": (2, D_SQ), "embed.b_in": (D_SQ,),
          "head.W_qd": (D_SQ, D_SQ), "head.v_start": (D_SQ,),
          "head.b_qd": (D_SQ,)}
for b, U in (("b0", U0), ("b1", U1), ("b2", U2)):
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
               "b_q", "b_k", "b_v", "b_o", "b_2"):
        SHAPES[f"{b}.{nm}"] = (D_SQ,)
    for nm in ("W_q", "W_k", "W_v", "W_o"):
        SHAPES[f"{b}.{nm}"] = (D_SQ, D_SQ)
    SHAPES[f"{b}.W_1"] = (D_SQ, U)
    SHAPES[f"{b}.W_2"] = (U, D_SQ)
    SHAPES[f"{b}.b_1"] = (U,)

rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


SQ = {}
STITCH = {}   # per-array bool mask, True = stitching (drawn)


def new_array(key):
    SQ[key] = fill(SHAPES[key])
    STITCH[key] = np.ones(SHAPES[key], dtype=bool)


def place(key, pos, value):
    """Copy a source-champion value in exact; the entry
    becomes backbone."""
    SQ[key][pos] = value
    STITCH[key][pos] = False


A_PATH = F4REP["path_set"]
B_PATH = S10REP["path_set"]
SQ_PATH_A = set()   # (squared entry, i, j) placed from A's path
SQ_PATH_B = set()

# ---- embed
new_array("embed.W_in")
for (entry, i, j) in sorted(A_PATH):
    if entry == "embed.W_in":
        place("embed.W_in", (i, A_MAP[j]), F4W[entry][i, j])
        SQ_PATH_A.add(("embed.W_in", i, A_MAP[j]))
for (entry, i, j) in sorted(B_PATH):
    if entry == "embed.W_in":
        place("embed.W_in", (i, B_MAP[j]), S10W[entry][i, j])
        SQ_PATH_B.add(("embed.W_in", i, B_MAP[j]))
new_array("embed.b_in")
for k_old, k_new in A_MAP.items():
    place("embed.b_in", (k_new,), F4W["embed.b_in"][k_old])
for k_old, k_new in B_MAP.items():
    place("embed.b_in", (k_new,), S10W["embed.b_in"][k_old])

VEC_NAMES = ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias",
             "b_q", "b_k", "b_v", "b_o", "b_2")
MAT_NAMES = ("W_q", "W_k", "W_v", "W_o")


def build_block(sq_blk, a_blk, with_b):
    """a_blk: side A's source block name; with_b: side B's
    source block is 'b0' (squared block 1 only)."""
    for nm in VEC_NAMES:
        key = f"{sq_blk}.{nm}"
        new_array(key)
        for k_old, k_new in A_MAP.items():
            place(key, (k_new,), F4W[f"{a_blk}.{nm}"][k_old])
        if with_b:
            for k_old, k_new in B_MAP.items():
                place(key, (k_new,), S10W[f"b0.{nm}"][k_old])
    for nm in MAT_NAMES:
        key = f"{sq_blk}.{nm}"
        new_array(key)
        for (entry, i, j) in sorted(A_PATH):
            if entry == f"{a_blk}.{nm}":
                place(key, (A_MAP[i], A_MAP[j]),
                      F4W[entry][i, j])
                SQ_PATH_A.add((key, A_MAP[i], A_MAP[j]))
        if with_b:
            for (entry, i, j) in sorted(B_PATH):
                if entry == f"b0.{nm}":
                    place(key, (B_MAP[i], B_MAP[j]),
                          S10W[entry][i, j])
                    SQ_PATH_B.add((key, B_MAP[i], B_MAP[j]))
    # W_1 (stream x units), W_2 (units x stream), b_1
    a_umap = AU0_MAP if sq_blk == "b0" else AU1_MAP
    b_umap = BU0_MAP if sq_blk == "b0" else BU1_MAP
    key = f"{sq_blk}.W_1"
    new_array(key)
    for (entry, i, j) in sorted(A_PATH):
        if entry == f"{a_blk}.W_1":
            place(key, (A_MAP[i], a_umap[j]), F4W[entry][i, j])
            SQ_PATH_A.add((key, A_MAP[i], a_umap[j]))
    if with_b:
        for (entry, i, j) in sorted(B_PATH):
            if entry == "b0.W_1":
                place(key, (B_MAP[i], b_umap[j]),
                      S10W[entry][i, j])
                SQ_PATH_B.add((key, B_MAP[i], b_umap[j]))
    key = f"{sq_blk}.W_2"
    new_array(key)
    for (entry, i, j) in sorted(A_PATH):
        if entry == f"{a_blk}.W_2":
            place(key, (a_umap[i], A_MAP[j]), F4W[entry][i, j])
            SQ_PATH_A.add((key, a_umap[i], A_MAP[j]))
    if with_b:
        for (entry, i, j) in sorted(B_PATH):
            if entry == "b0.W_2":
                place(key, (b_umap[i], B_MAP[j]),
                      S10W[entry][i, j])
                SQ_PATH_B.add((key, b_umap[i], B_MAP[j]))
    key = f"{sq_blk}.b_1"
    new_array(key)
    for u_old, u_new in a_umap.items():
        place(key, (u_new,), F4W[f"{a_blk}.b_1"][u_old])
    if with_b:
        for u_old, u_new in b_umap.items():
            place(key, (u_new,), S10W["b0.b_1"][u_old])


build_block("b0", "b0", with_b=True)
build_block("b1", "b1", with_b=False)

# ---- joining block (squared b2): ALL tiny random filler;
# LN gains 1 + N(0, 0.01) as in the earlier builds
for nm in BLOCK_KEYS:
    key = f"b2.{nm}"
    if nm in ("lnA_gain", "lnF_gain"):
        SQ[key] = 1.0 + rng_fill.normal(0.0, 0.01, D_SQ)
        STITCH[key] = np.ones(SHAPES[key], dtype=bool)
    else:
        new_array(key)

# ---- merged decoder over width 6
new_array("head.W_qd")
for i in range(4):
    for j in range(4):
        place("head.W_qd", (i, j), F4W["head.W_qd"][i, j])
for i in range(2):
    for j in range(2):
        place("head.W_qd", (B_MAP[i], B_MAP[j]),
              S10W["head.W_qd"][i, j])
SQ["head.b_qd"] = np.zeros(D_SQ)
STITCH["head.b_qd"] = np.ones(D_SQ, dtype=bool)
SQ["head.v_start"] = np.zeros(D_SQ)
STITCH["head.v_start"] = np.ones(D_SQ, dtype=bool)
for k_old, k_new in A_MAP.items():
    place("head.b_qd", (k_new,), F4W["head.b_qd"][k_old])
    place("head.v_start", (k_new,), F4W["head.v_start"][k_old])
for k_old, k_new in B_MAP.items():
    place("head.b_qd", (k_new,), S10W["head.b_qd"][k_old])
    place("head.v_start", (k_new,), S10W["head.v_start"][k_old])

assert set(SQ.keys()) == set(KEYS_SQ)
for k in KEYS_SQ:
    assert SQ[k].shape == SHAPES[k], k
    assert STITCH[k].shape == SHAPES[k], k


# ================================================================
# Construction checks (ALL must pass)
# ================================================================
checks = {}

# (1) every path entry bit-exact against its source
# champion, at its placed position, and classed
# backbone there. (head.W_qd path entries are covered
# by check 2: both decoder blocks are wholly exact.)
def src_of(side, entry):
    return F4W if side == "A" else S10W


n_path_checked = 0
for side, path, smap, placed in (
        ("A", A_PATH, A_MAP, SQ_PATH_A),
        ("B", B_PATH, B_MAP, SQ_PATH_B)):
    for (entry, i, j) in sorted(path):
        if entry == "head.W_qd":
            continue
        # re-derive the placed position from the maps:
        if entry == "embed.W_in":
            pos = ("embed.W_in", i, smap[j])
        elif entry.endswith(".W_1"):
            umap = (AU0_MAP if entry.startswith("b0.")
                    else AU1_MAP) if side == "A" else BU0_MAP
            pos = (f"b0.W_1" if side == "B" or
                   entry.startswith("b0.") else "b1.W_1",
                   smap[i], umap[j])
        elif entry.endswith(".W_2"):
            umap = (AU0_MAP if entry.startswith("b0.")
                    else AU1_MAP) if side == "A" else BU0_MAP
            pos = (f"b0.W_2" if side == "B" or
                   entry.startswith("b0.") else "b1.W_2",
                   umap[i], smap[j])
        else:
            sq_entry = entry if side == "A" else \
                ("b0." + entry.split(".", 1)[1])
            pos = (sq_entry, smap[i], smap[j])
        k2, i2, j2 = pos
        assert SQ[k2][i2, j2] == src_of(side, entry)[entry][i, j], \
            f"path entry not exact: side {side} {entry} {(i, j)}"
        assert not STITCH[k2][i2, j2], \
            f"path entry classed stitching: side {side} {entry}"
        n_path_checked += 1
checks["path_entries_bit_exact"] = True
checks["path_entries_checked"] = n_path_checked
log(f"construction check 1: {n_path_checked} placed path "
    f"entries bit-exact against their source champions "
    f"and classed backbone: True")

# (2) merged decoder: both blocks, b_qd, v_start exact
chk_a_block = bool(np.array_equal(
    SQ["head.W_qd"][np.ix_([0, 1, 2, 3], [0, 1, 2, 3])],
    F4W["head.W_qd"]))
chk_b_block = bool(np.array_equal(
    SQ["head.W_qd"][np.ix_([4, 5], [4, 5])],
    S10W["head.W_qd"]))
chk_bqd = bool(np.array_equal(
    SQ["head.b_qd"],
    np.concatenate([F4W["head.b_qd"], S10W["head.b_qd"]])))
chk_vs = bool(np.array_equal(
    SQ["head.v_start"],
    np.concatenate([F4W["head.v_start"],
                    S10W["head.v_start"]])))
cross = np.concatenate([
    SQ["head.W_qd"][np.ix_([0, 1, 2, 3], [4, 5])].ravel(),
    SQ["head.W_qd"][np.ix_([4, 5], [0, 1, 2, 3])].ravel()])
checks["decoder_sideA_block_bit_exact"] = chk_a_block
checks["decoder_sideB_block_bit_exact"] = chk_b_block
checks["decoder_b_qd_bit_exact"] = chk_bqd
checks["decoder_v_start_bit_exact"] = chk_vs
checks["decoder_cross_max_abs"] = float(np.abs(cross).max())
checks["decoder_cross_std"] = float(cross.std())
log(f"construction check 2: merged decoder — side A 4x4 "
    f"block exact {chk_a_block}, side B 2x2 block exact "
    f"{chk_b_block}, b_qd exact {chk_bqd}, v_start exact "
    f"{chk_vs}; cross blocks max |w| "
    f"{np.abs(cross).max():.4f}, std {cross.std():.4f}")
assert chk_a_block and chk_b_block and chk_bqd and chk_vs

# (3) mask accounting: every gene classified exactly once
per_array = {k: {"shape": list(SQ[k].shape),
                 "total": int(SQ[k].size),
                 "stitching": int(STITCH[k].sum()),
                 "backbone": int(SQ[k].size - STITCH[k].sum())}
             for k in KEYS_SQ}
flat_mask = np.concatenate([STITCH[k].ravel() for k in KEYS_SQ])
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
birth_gap, _, birth_lens = gap_of(SQ, X_val, ("b0", "b1", "b2"))
log(f"FRANKENSTEIN SQUARED validation gap (birth): "
    f"{birth_gap*100:.2f}% — side A (evolved Frankenstein "
    f"4) {f4_gap_mine*100:.2f}%, side B (species 10) "
    f"{sp_gaps[SP10]*100:.2f}%")

# ================================================================
# Save genome + results + diagram state
# ================================================================
meta = {
    "kind": "frankenstein_squared_stitch",
    "date": "2026-11-25",
    "design": "Frankenstein 4 rule: the two extracted "
              "paths side by side, holes tiny random "
              "std 0.03, one joining block, one merged "
              "decoder over the joined width",
    "sideA": "evolved Frankenstein 4 champion "
             "(frankenstein_4_evolved_champion.npz)",
    "sideB": "species 10 champion, fresh50_v2 run "
             "checkpoint best genome, identified by "
             "measurement",
    "sideA_gap_pct": round(f4_gap_mine * 100, 4),
    "sideB_species": SP10,
    "sideB_gap_pct": round(sp_gaps[SP10] * 100, 4),
    "d": D_SQ, "blocks": 3, "heads_per_block": 1,
    "ffn_hidden": {"b0": U0, "b1": U1, "b2_joining": U2},
    "sideA_coord_map": {str(k): v for k, v in A_MAP.items()},
    "sideB_coord_map": {str(k): v for k, v in B_MAP.items()},
    "sideA_unit_maps": {
        "b0": {str(k): v for k, v in AU0_MAP.items()},
        "b1": {str(k): v for k, v in AU1_MAP.items()}},
    "sideB_unit_maps": {
        "b0": {str(k): v for k, v in BU0_MAP.items()},
        "b1_shadow": {str(k): v for k, v in BU1_MAP.items()}},
    "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
    "gene_order": KEYS_SQ,
    "genes_total": n_total, "genes_backbone": n_backbone,
    "genes_stitching": n_stitch,
    "birth_gap_pct": round(birth_gap * 100, 4),
}
np.savez(HERE / "frankenstein_squared_genome.npz",
         **{k: SQ[k] for k in KEYS_SQ},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_squared_genome.npz")


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
    "design": "sponsor direction, 2026-11-25 22:06 EST: "
              "grab the 3rd champion and stitch it into "
              "the evolved Frankenstein 4 — Frankenstein "
              "squared — by the Frankenstein 4 rule",
    "parents": {
        "sideA_evolved_frankenstein4": {
            "source": "engine_runs/frankenstein4_evolution/"
                      "frankenstein_4_evolved_champion.npz",
            "validation_gap_pct": round(f4_gap_mine * 100, 4),
            "shape": {"stream_width": 4, "blocks": 2,
                      "ffn_hidden": [6, 6]},
            "path": rep_json(F4REP)},
        "sideB_species10": {
            "identification": "the species-10 entry of "
                "the fresh50_v2 run checkpoint's recorded "
                "species best genomes (checkpoint.npz "
                "best_genomes, the source the earlier "
                "stitch and skeleton scripts used), "
                "confirmed by measurement on the fixed "
                "validation set: gap 20.3364% (the 20.34% "
                "champion); the run's other species "
                "measure 15.1315% (8), 19.4412% (9), "
                "22.2554% (4)",
            "validation_gap_pct": round(sp_gaps[SP10] * 100, 4),
            "shape": {"stream_width": 2, "blocks": 1,
                      "ffn_hidden": [8]},
            "path": rep_json(S10REP)},
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
               "be.initial_arch(cfg, 2)). Species 10's "
               "native arch is d=2, and the Frankenstein "
               "4 champion's evolution forward pass used "
               "this same value for every block's "
               "attention and for the decoder logits; "
               "the squared network keeps it so every "
               "copied weight retains the scale it was "
               "built/evolved under. It is used for all "
               "three blocks' attention and the decoder "
               "logits, as in every earlier build."},
    "joining_ffn_width_reading": "the joining block's "
        "FFN width = the sum of the two sides' FFN path "
        "widths, each side's width counted in the block "
        "that feeds the joining block on that side "
        "(side A: its block 2, "
        f"{len(A_U1)} units; side B: its block 1, "
        f"{len(B_U0)} units) = {U2}, which equals the "
        "squared block-2 FFN width — the same relation "
        "as stitches #2/#4, where the joining FFN width "
        "equalled the preceding block's FFN width.",
    "construction_check": checks,
    "stitching_mask": {
        "definition": "backbone = every entry copied "
                      "exact from a source champion "
                      "(path synapses, territory "
                      "vectors, the two decoder blocks, "
                      "b_qd, v_start); stitching = every "
                      "drawn entry (filler, the whole "
                      "joining block, the decoder cross "
                      "blocks). Each entry was classified "
                      "when placed or drawn, so no "
                      "entry's origin is ambiguous.",
        "gene_order": KEYS_SQ,
        "total": n_total, "backbone": n_backbone,
        "stitching": n_stitch,
        "per_array": per_array,
        "flat_mask_stitching": [int(x) for x in flat_mask],
    },
    "frankenstein_squared": {
        "shape": {"stream_width": D_SQ, "blocks": 3,
                  "ffn_hidden": {"b0": U0, "b1": U1,
                                 "b2_joining": U2},
                  "heads_per_block": 1, "decoders": 1,
                  "total_params": n_total},
        "validation_gap_pct": round(birth_gap * 100, 4),
        "validation_gap": birth_gap,
        "mean_length": float(birth_lens.mean()),
        "comparisons_pct": {
            "sideA_evolved_frankenstein4":
                round(f4_gap_mine * 100, 4),
            "sideB_species10": round(sp_gaps[SP10] * 100, 4),
            "frankenstein4_birth": 23.2865,
            "frankenstein4_evolved": 6.8959},
        "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
        "engine_fit": "DOES NOT FIT the current ArchSpec "
                      "(same finding as Frankensteins "
                      "1-5): explicit weight set with "
                      "its own forward pass.",
    },
}
with open(HERE / "frankenstein_squared_stitch_"
                 "2026-11-25.json", "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_squared_stitch_2026-11-25.json")

# ---- diagram state (JSON only, no pickle)
def panel_edges(edges, path_cls):
    return [{"src": e["src"], "dst": e["dst"],
             "kind": e["kind"], "entry": e["entry"],
             "i": e["i"], "j": e["j"],
             "rel": round(e["rel"], 4),
             "cls": path_cls if e["on_path"] else "o"}
            for e in edges]


sq_edges = []
# embed edges
for c in range(2):
    for k in range(D_SQ):
        cls = "f"
        if ("embed.W_in", c, k) in SQ_PATH_A:
            cls = "a"
        elif ("embed.W_in", c, k) in SQ_PATH_B:
            cls = "b"
        sq_edges.append({"src": f"in{c}", "dst": f"s0_{k}",
                         "kind": "weight", "cls": cls,
                         "rel": None})
U_BY_BLOCK = (U0, U1, U2)
for bi, blk in enumerate(("b0", "b1", "b2")):
    s_in, s_mid, s_out = f"s{2*bi}", f"s{2*bi+1}", f"s{2*bi+2}"
    U = U_BY_BLOCK[bi]
    for nm, tgt in (("W_q", "q"), ("W_k", "k"), ("W_v", "v")):
        for i in range(D_SQ):
            for j in range(D_SQ):
                key = (f"{blk}.{nm}", i, j)
                cls = "a" if key in SQ_PATH_A else \
                    ("b" if key in SQ_PATH_B else "f")
                sq_edges.append({"src": f"{s_in}_{i}",
                                 "dst": f"{tgt}{bi}_{j}",
                                 "kind": "weight", "cls": cls,
                                 "rel": None})
    for src_pre in ("q", "k", "v"):
        for j in range(D_SQ):
            sq_edges.append({"src": f"{src_pre}{bi}_{j}",
                             "dst": f"mix{bi}_{j}",
                             "kind": "conduit", "cls": "f",
                             "rel": None})
    for i in range(D_SQ):
        for j in range(D_SQ):
            key = (f"{blk}.W_o", i, j)
            cls = "a" if key in SQ_PATH_A else \
                ("b" if key in SQ_PATH_B else "f")
            sq_edges.append({"src": f"mix{bi}_{i}",
                             "dst": f"{s_mid}_{j}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
        sq_edges.append({"src": f"{s_in}_{i}",
                         "dst": f"{s_mid}_{i}",
                         "kind": "residual", "cls": "f",
                         "rel": None})
    for i in range(D_SQ):
        for u in range(U):
            key = (f"{blk}.W_1", i, u)
            cls = "a" if key in SQ_PATH_A else \
                ("b" if key in SQ_PATH_B else "f")
            sq_edges.append({"src": f"{s_mid}_{i}",
                             "dst": f"ffn{bi}_{u}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
    for u in range(U):
        for j in range(D_SQ):
            key = (f"{blk}.W_2", u, j)
            cls = "a" if key in SQ_PATH_A else \
                ("b" if key in SQ_PATH_B else "f")
            sq_edges.append({"src": f"ffn{bi}_{u}",
                             "dst": f"{s_out}_{j}",
                             "kind": "weight", "cls": cls,
                             "rel": None})
    for k in range(D_SQ):
        sq_edges.append({"src": f"{s_mid}_{k}",
                         "dst": f"{s_out}_{k}",
                         "kind": "residual", "cls": "f",
                         "rel": None})
for i in range(D_SQ):
    for j in range(D_SQ):
        if i < 4 and j < 4:
            cls = "a"
        elif i >= 4 and j >= 4:
            cls = "b"
        else:
            cls = "f"
        sq_edges.append({"src": f"s6_{i}", "dst": f"dec_{j}",
                         "kind": "weight", "cls": cls,
                         "rel": None})
for j in range(D_SQ):
    sq_edges.append({"src": f"dec_{j}", "dst": "pick",
                     "kind": "conduit",
                     "cls": "a" if j < 4 else "b",
                     "rel": None})

state = {
    "panels": {
        "sideA": {
            "title": "Side A: evolved Frankenstein 4 "
                     "champion",
            "gap_pct": round(f4_gap_mine * 100, 4),
            "d": F4_D, "blocks": [6, 6],
            "path_synapses": F4REP["n_path_weight_synapses"],
            "total_synapses": F4REP["n_weight_synapses_total"],
            "edges": panel_edges(f4_edges, "a")},
        "sideB": {
            "title": "Side B: species 10 champion",
            "gap_pct": round(sp_gaps[SP10] * 100, 4),
            "d": 2, "blocks": [8],
            "path_synapses": S10REP["n_path_weight_synapses"],
            "total_synapses": S10REP["n_weight_synapses_total"],
            "edges": panel_edges(s10_edges, "b")},
        "squared": {
            "title": "Frankenstein Squared",
            "gap_pct": round(birth_gap * 100, 4),
            "d": D_SQ, "blocks": [U0, U1, U2],
            "stream_split": 4,
            "unit_splits": [len(A_U0), len(A_U1),
                            len(A_U1)],
            "edges": sq_edges},
    },
}
with open(HERE / "frankenstein_squared_state.json", "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_squared_state.json (diagram input)")
log("DONE stitch")
