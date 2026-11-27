#!/usr/bin/env python3
# REPLICATION (2026-11-27): adapted copy of
# frankenstein_stepA2_stitch_2026-11-26.py - replication parents
# (species 6 host, species 11 donor, both matured +100 gens),
# fill/walk seed 20261312, outputs under
# engine_runs/frankenstein_replication/. See
# frankenstein_replication_log_2026-11-27.md.
"""Frankenstein 4 stitch for STEP A REDO of the
deep-narrow Frankenstein line (2026-11-26).

The Frankenstein 4 rule (stitch #2 paths +
stitch #4 merged decoder, as replicated by
frankenstein4_second_stitch_2026-11-26.py)
generalized from 1-block parents to the
4-block parents of the deep50_stepA founding
run (config bert_engine_config_deep50_stepA
.json: d=2, FFN 8, 1 head, 4 blocks, 310
genes; fixed validation set = TSPDomain
default, val_seed 20261116, 200 problems,
10 cities).

PARENTS:
  - Host: species 1, the run champion
    (engine_runs/deep50_stepA/champion.npz),
    validation gap 17.1420%.
  - Donor: species 0, best genome in
    engine_runs/deep50_stepA/checkpoint.npz,
    validation gap 19.6989%.
Both gaps are re-measured here and the
generalized forward pass is parity-checked
against the engine on both parents (identical
tours, gap to 1e-9) before any flow is
trusted.

PATH ISOLATION (corrected rule, unchanged):
signal flows (activation x weight samples
over the fixed validation set) for every
weight synapse in all 4 blocks plus the
decoder; a weight synapse PASSES iff its
std >= half the max std among weight
synapses feeding the same node; the PATH is
the passing synapses on a connected
entry-to-exit route (conduit / residual
edges always routable). Node names are
per-block and the residuals chain across
blocks.

STOP RULE (as briefed): if either parent's
path holds more than 60% of its total
weight synapses (briefed total: 4 blocks x
56 + 4 decoder = 228), stop, write the JSON
report with the stop reason, do not stitch.
(The generalized graph itself carries 200
weight synapses: embed 4 + 4 blocks x 48 +
decoder 4; both totals are recorded.)

STITCH (Frankenstein 4 pattern): stream
width 4; host on coords 0-1, donor on
coords 2-3. For each block position
k = 0..3 the stitched block carries both
parents' block-k path branches side by
side (path entries bit-exact; cross-
territory and off-path entries tiny-random
filler, std 0.03); the FFN width of
stitched block k = host path FFN units in
block k + donor path FFN units in block k.
After the 4 stitched blocks, ONE joining
block (full block, all filler) copying the
LAST stitched block's shapes. Then the
MERGED DECODER: W_qd block-diagonal with
the host's 2x2 top-left exact and the
donor's 2x2 bottom-right exact, cross
blocks tiny random (std 0.03), b_qd and
v_start exact concatenations. attn_scale
= 1/sqrt(2) everywhere. ALL filler draws
come from ONE rng stream, seed 20261212.

Backbone = every entry copied exact from
a parent; stitching = every filler entry
+ the whole joining block + the decoder
cross blocks. No ambiguous entries: the
territory vectors (embed bias, per-block
layer norms and biases, FFN unit biases)
are placed from the parents at every
territory position by construction, so
they are backbone; every matrix entry not
placed as a path synapse or a decoder
diagonal entry is filler by construction.

Outputs: frankenstein_stepA2_genome.npz,
frankenstein_stepA2_stitch_2026-11-26.json,
frankenstein_stepA2_state.json.
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
RUN = HERE / "engine_runs" / "frankenstein_replication" / "founding"
SIGMA_FILL = 0.03
FILL_SEED = 20261312
WALK_SEED = 20261312
N_WALKERS = 200_000
N_BLOCKS = 4
BLOCKS = tuple(f"b{k}" for k in range(N_BLOCKS))
# briefed stop-rule denominator: 4 blocks x 56
# + 4 decoder synapses
N_WEIGHT_TOTAL_BRIEF = 228
# the generalized graph's own weight-synapse
# count: embed 4 + 4 x 48 + decoder 4
N_WEIGHT_TOTAL_GRAPH = 200

# ================================================================
# Setup
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
assert SCALE == 1.0 / np.sqrt(2)

# ================================================================
# STEP 1 - parents (host and donor fixed by
# the brief; gaps re-measured here)
# ================================================================
zh = np.load(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_parent_host_evolved.npz")
host_genome = np.asarray(zh["genome"], dtype=float)
host_meta = json.loads(str(zh["meta"]))
zd = np.load(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_parent_donor_evolved.npz")
donor_genome = np.asarray(zd["genome"], dtype=float)
donor_meta = json.loads(str(zd["meta"]))
host_id = int(host_meta["species"])
donor_id = int(donor_meta["species"])
gaps = {host_id: float(domain.validation_detail(
            layout, host_genome)["mean_gap"]),
        donor_id: float(domain.validation_detail(
            layout, donor_genome)["mean_gap"])}
log(f"host = species {host_id} EVOLVED +100 gens (gap "
    f"{gaps[host_id]*100:.4f}%; founding "
    f"{host_meta['founding_gap_pct']:.4f}%, recorded "
    f"evolved {host_meta['best_validation_gap_pct']:.4f}%), "
    f"donor = species {donor_id} EVOLVED +100 gens (gap "
    f"{gaps[donor_id]*100:.4f}%; founding "
    f"{donor_meta['founding_gap_pct']:.4f}%, recorded "
    f"evolved {donor_meta['best_validation_gap_pct']:.4f}%)")
assert abs(gaps[host_id] * 100
           - host_meta["best_validation_gap_pct"]) < 5e-5
assert abs(gaps[donor_id] * 100
           - donor_meta["best_validation_gap_pct"]) < 5e-5

PARENTS = {
    "host": {"species": host_id, "genome": host_genome,
             "gap": gaps[host_id]},
    "donor": {"species": donor_id, "genome": donor_genome,
              "gap": gaps[donor_id]},
}


def parent_weights(genome):
    """Engine genome -> generic weight dict for
    the 4-block parent shape."""
    d = layout.unpack_dict(genome)
    W = {"embed.W_in": d["c0.embed.W_in"],
         "embed.b_in": d["c0.embed.b_in"],
         "head.v_start": d["c0.head.v_start"],
         "head.W_qd": d["c0.head.W_qd"],
         "head.b_qd": d["c0.head.b_qd"]}
    for k in range(N_BLOCKS):
        p = f"c0.b{k}"
        W[f"b{k}.lnA_gain"] = d[f"{p}.lnA_gain"]
        W[f"b{k}.lnA_bias"] = d[f"{p}.lnA_bias"]
        for nm in ("W_q", "W_k", "W_v", "W_o"):
            W[f"b{k}.{nm}"] = d[f"{p}.h0.{nm}"]
        for nm in ("b_q", "b_k", "b_v", "b_o"):
            W[f"b{k}.{nm}"] = d[f"{p}.h0.{nm}"]
        W[f"b{k}.lnF_gain"] = d[f"{p}.lnF_gain"]
        W[f"b{k}.lnF_bias"] = d[f"{p}.lnF_bias"]
        W[f"b{k}.W_1"] = d[f"{p}.W_1"]
        W[f"b{k}.b_1"] = d[f"{p}.b_1"]
        W[f"b{k}.W_2"] = d[f"{p}.W_2"]
        W[f"b{k}.b_2"] = d[f"{p}.b_2"]
    return W


# ================================================================
# Generic forward + greedy pointer decode
# (conventions verbatim from stitch #2,
# generalized to a block list)
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


def decode_general(W, Y, X):
    B, n, d = Y.shape
    gmean = Y.mean(axis=1)
    v_start = W["head.v_start"]
    W_qd = W["head.W_qd"]
    b_qd = W["head.b_qd"]
    last = np.broadcast_to(v_start, (B, d)).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    picks = np.zeros((B, n), dtype=int)
    for t in range(n):
        state = gmean + last
        q = state @ W_qd + b_qd
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
    picks = decode_general(W, Y, X)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    gaps = lens / OPT - 1.0
    return float(gaps.mean()), picks, lens


for tag, P in PARENTS.items():
    W = parent_weights(P["genome"])
    g_mine, picks_mine, _ = gap_of(W, X_val, BLOCKS)
    lens_e, picks_e = domain.evaluate_lengths(
        layout, P["genome"][None, :], X_val)
    tours_same = bool((picks_mine == picks_e[0]).all())
    log(f"parity {tag}: my gap {g_mine*100:.6f}% vs engine "
        f"{P['gap']*100:.6f}%, tours identical: {tours_same}")
    assert abs(g_mine - P["gap"]) < 1e-9 and tours_same
    P["weights"] = W

# ================================================================
# STEP 2 - signal flows with per-block node
# names and residuals chained across blocks;
# fan-in shares; corrected path extraction
# (walk + extraction verbatim from stitch #2)
# ================================================================
def compute_flows(W):
    X = X_val
    B, n, _ = X.shape
    edges = []

    def add(src, dst, kind, entry, i, j, samples):
        edges.append({"src": src, "dst": dst, "kind": kind,
                      "entry": entry, "i": i, "j": j,
                      "samples": np.asarray(
                          samples, dtype=float).ravel()})

    # stream states: S[k] is the stream entering
    # block k; S[N_BLOCKS] leaves the last block
    S = [None] * (N_BLOCKS + 1)
    Emb = X @ W["embed.W_in"] + W["embed.b_in"]
    S[0] = Emb
    for c in range(2):
        for k in range(2):
            add(f"in{c}", f"s0_{k}", "weight", "embed.W_in",
                c, k, X[:, :, c] * W["embed.W_in"][c, k])
    for k in range(N_BLOCKS):
        blk = f"b{k}"
        A = S[k]
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        Q = Hb @ W[f"{blk}.W_q"] + W[f"{blk}.b_q"]
        K = Hb @ W[f"{blk}.W_k"] + W[f"{blk}.b_k"]
        V = Hb @ W[f"{blk}.W_v"] + W[f"{blk}.b_v"]
        Sc = np.einsum("btj,bsj->bts", Q, K) * SCALE
        Sc = Sc - Sc.max(axis=-1, keepdims=True)
        E = np.exp(Sc)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        for nm, tgt in (("W_q", "q"), ("W_k", "k"),
                        ("W_v", "v")):
            M = W[f"{blk}.{nm}"]
            for c in range(2):
                for j in range(2):
                    add(f"s{k}_{c}", f"{blk}{tgt}_{j}",
                        "weight", f"{blk}.{nm}", c, j,
                        Hb[:, :, c] * M[c, j])
        for j in range(2):
            qflow = Q[:, :, j] * np.einsum(
                "bts,bs->bt", Atn, K[:, :, j] * SCALE)
            add(f"{blk}q_{j}", f"{blk}x_{j}", "conduit",
                "attnQ", -1, j, qflow)
            kflow = K[:, :, j] * np.einsum(
                "bts,bt->bs", Atn, Q[:, :, j] * SCALE)
            add(f"{blk}k_{j}", f"{blk}x_{j}", "conduit",
                "attnK", -1, j, kflow)
            aflow = Atn * V[:, :, j][:, None, :]
            add(f"{blk}v_{j}", f"{blk}x_{j}", "conduit",
                "attnA", -1, j, aflow)
        for c in range(2):
            for j in range(2):
                add(f"{blk}x_{c}", f"m{k}_{j}", "weight",
                    f"{blk}.W_o", c, j,
                    Ah[:, :, c] * W[f"{blk}.W_o"][c, j])
        A1 = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        for c in range(2):
            add(f"s{k}_{c}", f"m{k}_{c}", "residual",
                "residual1", c, c, A[:, :, c])
        Hf = ln_np(A1, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A1f = np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"])
        for c in range(2):
            for u in range(8):
                add(f"m{k}_{c}", f"{blk}f_{u}", "weight",
                    f"{blk}.W_1", c, u,
                    Hf[:, :, c] * W[f"{blk}.W_1"][c, u])
        for u in range(8):
            for j in range(2):
                add(f"{blk}f_{u}", f"s{k+1}_{j}", "weight",
                    f"{blk}.W_2", u, j,
                    A1f[:, :, u] * W[f"{blk}.W_2"][u, j])
        A2 = A1 + A1f @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
        for c in range(2):
            add(f"m{k}_{c}", f"s{k+1}_{c}", "residual",
                "residual2", c, c, A1[:, :, c])
        S[k + 1] = A2
    A2 = S[N_BLOCKS]
    _, picks, _ = gap_of(W, X, BLOCKS)
    gmean = A2.mean(axis=1)
    states = np.zeros((B, n, 2))
    for t in range(n):
        last = W["head.v_start"] if t == 0 else \
            A2[np.arange(B), picks[:, t - 1]]
        states[:, t, :] = gmean + last
    qd = states @ W["head.W_qd"] + W["head.b_qd"]
    for c in range(2):
        for j in range(2):
            add(f"s{N_BLOCKS}_{c}", f"dec_{j}", "weight",
                "head.W_qd", c, j,
                states[:, :, c] * W["head.W_qd"][c, j])
    for j in range(2):
        add(f"dec_{j}", "pick", "conduit", "logit", -1, j,
            A2[:, :, j][:, None, :] * qd[:, :, j][:, :, None]
            * SCALE)
    for e in edges:
        e["mean"] = float(e["samples"].mean())
        e["std"] = float(e["samples"].std())
    from collections import defaultdict
    fanin = defaultdict(float)
    for e in edges:
        fanin[e["dst"]] += e["std"]
    for e in edges:
        tot = fanin[e["dst"]]
        e["share"] = e["std"] / tot if tot > 0 else 0.0
    return edges


def run_walk(edges):
    nodes = sorted({e["src"] for e in edges}
                   | {e["dst"] for e in edges})
    nidx = {nm: i for i, nm in enumerate(nodes)}
    out = {i: [] for i in range(len(nodes))}
    for ei, e in enumerate(edges):
        out[nidx[e["src"]]].append(ei)
    trans = {}
    for ni, eis in out.items():
        w = np.array([edges[ei]["share"] for ei in eis])
        if w.sum() <= 0:
            w = np.ones(len(eis))
        trans[ni] = (np.array(eis),
                     np.cumsum(w / w.sum()))
    rng = np.random.default_rng(WALK_SEED)
    in_std = np.array([X_val[:, :, c].std() for c in range(2)])
    p_in = in_std / in_std.sum()
    cur = rng.choice([nidx["in0"], nidx["in1"]],
                     size=N_WALKERS, p=p_in)
    counts = np.zeros(len(edges))
    pick_id = nidx["pick"]
    for _ in range(64):
        active = cur != pick_id
        if not active.any():
            break
        nxt = cur.copy()
        for ni in np.unique(cur[active]):
            m = active & (cur == ni)
            eis, cum = trans[ni]
            r = rng.random(m.sum())
            sel = np.searchsorted(cum, r)
            chosen = eis[sel]
            counts += np.bincount(
                chosen, minlength=len(edges)).astype(float)
            nxt[m] = np.array(
                [nidx[edges[ei]["dst"]] for ei in chosen])
        cur = nxt
    return counts / N_WALKERS, nidx


def extract_path(edges):
    from collections import defaultdict
    trav, _ = run_walk(edges)
    for e, t in zip(edges, trav):
        e["trav"] = float(t)
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


# ================================================================
# Path summaries per parent (per block)
# ================================================================
for tag, P in PARENTS.items():
    edges = extract_path(compute_flows(P["weights"]))
    P["edges"] = edges
    n_weight = sum(1 for e in edges if e["kind"] == "weight")
    assert n_weight == N_WEIGHT_TOTAL_GRAPH, n_weight
    wp = [e for e in edges
          if e["on_path"] and e["kind"] == "weight"]
    P["path_weight_edges"] = wp
    P["n_pass"] = int(sum(
        1 for e in edges
        if e["kind"] == "weight" and e["passes"]))
    P["path_per_block"] = {}
    P["ffn_units_per_block"] = {}
    for k in range(N_BLOCKS):
        blk = f"b{k}"
        sub = [e for e in wp
               if e["entry"].startswith(blk + ".")]
        P["path_per_block"][k] = len(sub)
        P["ffn_units_per_block"][k] = sorted(
            {e["j"] for e in sub if e["entry"] == f"{blk}.W_1"}
            | {e["i"] for e in sub if e["entry"] == f"{blk}.W_2"})
    P["path_embed"] = sum(1 for e in wp
                          if e["entry"] == "embed.W_in")
    P["path_decoder"] = sum(1 for e in wp
                            if e["entry"] == "head.W_qd")
    P["head_on_path"] = any(
        e["on_path"] and e["entry"] in
        tuple(f"b{k}.{nm}" for k in range(N_BLOCKS)
              for nm in ("W_q", "W_k", "W_v", "W_o"))
        for e in edges)
    import re
    coords = set()
    for e in edges:
        if not e["on_path"]:
            continue
        for nd in (e["src"], e["dst"]):
            m = re.fullmatch(r"[sm]\d+_(\d)", nd)
            if m:
                coords.add(int(m.group(1)))
            m = re.fullmatch(r"b\d+[qkvxf]_(\d)", nd)
            if m:
                coords.add(int(m.group(1)))
            m = re.fullmatch(r"dec_(\d)", nd)
            if m:
                coords.add(int(m.group(1)))
    P["coords_used"] = sorted(coords)
    spine = []
    node = "in0"
    seen = set()
    while node != "pick" and node not in seen:
        seen.add(node)
        spine.append(node)
        outs = [e for e in edges
                if e["src"] == node and e["on_path"]]
        if not outs:
            break
        node = max(outs, key=lambda e: e["trav"])["dst"]
    spine.append("pick")
    P["spine"] = spine
    log(f"{tag} (species {P['species']}): passing "
        f"synapses {P['n_pass']} of {N_WEIGHT_TOTAL_GRAPH}, "
        f"path weight synapses {len(wp)} (embed "
        f"{P['path_embed']}, per block "
        f"{[P['path_per_block'][k] for k in range(N_BLOCKS)]}, "
        f"decoder {P['path_decoder']}), path FFN units "
        f"per block "
        f"{[P['ffn_units_per_block'][k] for k in range(N_BLOCKS)]}, "
        f"stream coords used {P['coords_used']}")

# ================================================================
# STOP RULE (briefed denominator 228; the
# graph's own total, 200, is recorded too)
# ================================================================
_path_sizes = {tag: len(P["path_weight_edges"])
               for tag, P in PARENTS.items()}
if any(n > 0.6 * N_WEIGHT_TOTAL_BRIEF
       for n in _path_sizes.values()):
    stop_report = {
        "design": "Frankenstein step A REPLICATION stitch "
                  "(deep-narrow line), 2026-11-26",
        "stopped": True,
        "stop_reason": "a corrected path holds more "
                       "than 60% of the total weight "
                       "synapses (briefed total 228)",
        "path_sizes": _path_sizes,
        "path_sizes_per_block": {
            tag: P["path_per_block"]
            for tag, P in PARENTS.items()},
        "n_passing": {tag: P["n_pass"]
                      for tag, P in PARENTS.items()},
        "n_weight_total_brief": N_WEIGHT_TOTAL_BRIEF,
        "n_weight_total_graph": N_WEIGHT_TOTAL_GRAPH,
        "parents": {
            tag: {"species": P["species"],
                  "gap_pct": round(P["gap"] * 100, 4)}
            for tag, P in PARENTS.items()},
    }
    with open(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_stitch_2026-11-27.json", "w") as f:
        json.dump(stop_report, f, indent=1)
    log(f"STOP: corrected path sizes {_path_sizes} "
        f"exceed 60% of {N_WEIGHT_TOTAL_BRIEF}; not "
        f"stitching. Report saved.")
    sys.exit(0)
log(f"corrected path sizes {_path_sizes}: both under "
    f"the 60% stop line (briefed total "
    f"{N_WEIGHT_TOTAL_BRIEF}, graph total "
    f"{N_WEIGHT_TOTAL_GRAPH}); stitching proceeds")

# ================================================================
# STEP 3 - build the stitch. ONE rng stream
# (seed 20261212) for every filler draw.
# ================================================================
rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


H, D = PARENTS["host"], PARENTS["donor"]
h_map = {0: 0, 1: 1}
d_map = {0: 2, 1: 3}
dF = 4
hu_maps, du_maps, U_per_block = {}, {}, {}
for k in range(N_BLOCKS):
    hu = H["ffn_units_per_block"][k]
    du = D["ffn_units_per_block"][k]
    hu_maps[k] = {u: i for i, u in enumerate(hu)}
    du_maps[k] = {u: len(hu) + i for i, u in enumerate(du)}
    U_per_block[k] = len(hu) + len(du)
log(f"Frankenstein shape: stream width {dF} "
    f"(host coords 0-1 + donor coords 2-3), stitched "
    f"FFN widths per block {U_per_block}, joining "
    f"block copies block {N_BLOCKS - 1} shapes "
    f"(FFN width {U_per_block[N_BLOCKS - 1]})")

h_path = {(e["entry"], e["i"], e["j"]): True
          for e in H["path_weight_edges"]}
d_path = {(e["entry"], e["i"], e["j"]): True
          for e in D["path_weight_edges"]}
HW, DW = H["weights"], D["weights"]

FW = {}
W_in = fill((2, dF))
for c in range(2):
    for k in range(2):
        if ("embed.W_in", c, k) in h_path:
            W_in[c, h_map[k]] = HW["embed.W_in"][c, k]
        if ("embed.W_in", c, k) in d_path:
            W_in[c, d_map[k]] = DW["embed.W_in"][c, k]
FW["embed.W_in"] = W_in
b_in = fill((dF,))
for k in range(2):
    b_in[h_map[k]] = HW["embed.b_in"][k]
    b_in[d_map[k]] = DW["embed.b_in"][k]
FW["embed.b_in"] = b_in


def territory_vec(PW, nm, size):
    v = fill((size,))
    for k in range(2):
        v[h_map[k]] = HW[nm][k]
        v[d_map[k]] = DW[nm][k]
    return v


def block_matrix(k, entry, hW, dW, row_map_h, row_map_d,
                 col_map_h, col_map_d, n_rows, n_cols):
    M = fill((n_rows, n_cols))
    for i_old, i_new in row_map_h.items():
        for j_old, j_new in col_map_h.items():
            if (entry, i_old, j_old) in h_path:
                M[i_new, j_new] = hW[i_old, j_old]
    for i_old, i_new in row_map_d.items():
        for j_old, j_new in col_map_d.items():
            if (entry, i_old, j_old) in d_path:
                M[i_new, j_new] = dW[i_old, j_old]
    return M


for k in range(N_BLOCKS):
    blk = f"b{k}"
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
        FW[f"{blk}.{nm}"] = territory_vec(
            None, f"{blk}.{nm}", dF)
    for nm in ("W_q", "W_k", "W_v", "W_o"):
        FW[f"{blk}.{nm}"] = block_matrix(
            k, f"{blk}.{nm}", HW[f"{blk}.{nm}"],
            DW[f"{blk}.{nm}"], h_map, d_map, h_map, d_map,
            dF, dF)
    for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
        FW[f"{blk}.{nm}"] = territory_vec(
            None, f"{blk}.{nm}", dF)
    FW[f"{blk}.W_1"] = block_matrix(
        k, f"{blk}.W_1", HW[f"{blk}.W_1"], DW[f"{blk}.W_1"],
        h_map, d_map, hu_maps[k], du_maps[k], dF,
        U_per_block[k])
    FW[f"{blk}.W_2"] = block_matrix(
        k, f"{blk}.W_2", HW[f"{blk}.W_2"], DW[f"{blk}.W_2"],
        hu_maps[k], du_maps[k], h_map, d_map,
        U_per_block[k], dF)
    b_1 = fill((U_per_block[k],))
    for u, u_new in hu_maps[k].items():
        b_1[u_new] = HW[f"{blk}.b_1"][u]
    for u, u_new in du_maps[k].items():
        b_1[u_new] = DW[f"{blk}.b_1"][u]
    FW[f"{blk}.b_1"] = b_1

# joining block (b4): copies the LAST stitched
# block's shapes; all filler (layer-norm gains
# follow the reference convention: 1 + small
# noise from the same stream)
JB = f"b{N_BLOCKS}"
UJ = U_per_block[N_BLOCKS - 1]
FW[f"{JB}.lnA_gain"] = 1.0 + rng_fill.normal(0.0, 0.01, dF)
FW[f"{JB}.lnA_bias"] = fill((dF,))
FW[f"{JB}.lnF_gain"] = 1.0 + rng_fill.normal(0.0, 0.01, dF)
FW[f"{JB}.lnF_bias"] = fill((dF,))
for nm in ("W_q", "W_k", "W_v", "W_o"):
    FW[f"{JB}.{nm}"] = fill((dF, dF))
for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
    FW[f"{JB}.{nm}"] = fill((dF,))
FW[f"{JB}.W_1"] = fill((dF, UJ))
FW[f"{JB}.b_1"] = fill((UJ,))
FW[f"{JB}.W_2"] = fill((UJ, dF))

# ================================================================
# STEP 4 - the merged decoder (Frankenstein 4
# rule): W_qd drawn from the SAME rng stream,
# diagonal territory blocks overwritten with
# the parents' own; b_qd / v_start exact
# concatenations.
# ================================================================
W_qd_m = fill((dF, dF))  # one draw, same stream
for k_old, k_new in h_map.items():
    for j_old, j_new in h_map.items():
        W_qd_m[k_new, j_new] = HW["head.W_qd"][k_old, j_old]
for k_old, k_new in d_map.items():
    for j_old, j_new in d_map.items():
        W_qd_m[k_new, j_new] = DW["head.W_qd"][k_old, j_old]
b_qd_m = np.zeros(dF)
v_start_m = np.zeros(dF)
for k_old, k_new in h_map.items():
    b_qd_m[k_new] = HW["head.b_qd"][k_old]
    v_start_m[k_new] = HW["head.v_start"][k_old]
for k_old, k_new in d_map.items():
    b_qd_m[k_new] = DW["head.b_qd"][k_old]
    v_start_m[k_new] = DW["head.v_start"][k_old]
FW["head.W_qd"] = W_qd_m
FW["head.b_qd"] = b_qd_m
FW["head.v_start"] = v_start_m

# ---- construction checks
n_path_checked = 0
for tag, P, PW, cmap in (("host", H, HW, h_map),
                         ("donor", D, DW, d_map)):
    umaps = hu_maps if tag == "host" else du_maps
    for e in P["path_weight_edges"]:
        entry, i, j = e["entry"], e["i"], e["j"]
        if entry == "head.W_qd":
            continue  # decoder checked below
        if entry == "embed.W_in":
            pos = (i, cmap[j])
        elif entry.endswith(".W_1"):
            k = int(entry[1])
            pos = (cmap[i], umaps[k][j])
        elif entry.endswith(".W_2"):
            k = int(entry[1])
            pos = (umaps[k][i], cmap[j])
        else:
            pos = (cmap[i], cmap[j])
        assert FW[entry][pos] == PW[entry][i, j], \
            f"path synapse not exact: {tag} {entry} {(i, j)}"
        n_path_checked += 1
log(f"construction check: {n_path_checked} body path "
    f"synapses bit-exact against their parents")

chk_blocks = True
for cmap, PW in ((h_map, HW), (d_map, DW)):
    for i_old, i_new in cmap.items():
        for j_old, j_new in cmap.items():
            chk_blocks &= bool(
                W_qd_m[i_new, j_new]
                == PW["head.W_qd"][i_old, j_old])
chk_vecs = True
for nm, vec in (("head.b_qd", b_qd_m),
                ("head.v_start", v_start_m)):
    for cmap, PW in ((h_map, HW), (d_map, DW)):
        for k_old, k_new in cmap.items():
            chk_vecs &= bool(vec[k_new] == PW[nm][k_old])
h_rows = sorted(h_map.values())
d_rows = sorted(d_map.values())
cross = np.concatenate([
    W_qd_m[np.ix_(h_rows, d_rows)].ravel(),
    W_qd_m[np.ix_(d_rows, h_rows)].ravel()])
log(f"construction check: merged decoder diagonal "
    f"blocks exact: {chk_blocks}; b_qd / v_start "
    f"territory values exact: {chk_vecs}; cross "
    f"blocks: max |w| {np.abs(cross).max():.4f}, std "
    f"{cross.std():.4f} (tiny random, std "
    f"{SIGMA_FILL}, seed {FILL_SEED}, same stream)")
assert chk_blocks and chk_vecs

# ================================================================
# Score ONCE on the fixed validation set
# ================================================================
STITCH_BLOCKS = tuple(f"b{k}" for k in range(N_BLOCKS + 1))
frank_gap, frank_picks, frank_lens = gap_of(
    FW, X_val, STITCH_BLOCKS)
log(f"FRANKENSTEIN STEP A2 validation gap (birth): "
    f"{frank_gap*100:.4f}% (host "
    f"{H['gap']*100:.2f}%, donor {D['gap']*100:.2f}%)")

# ================================================================
# Save genome + results + state
# ================================================================
meta = {
    "kind": "frankenstein_rep_stepA_stitch",
    "date": "2026-11-27",
    "founding_run": "rep_deep50 (run seed 20261300); "
               "REDO: parents evolved a further 100 "
               "generations each before stitching",
    "path_rule": "corrected: fan-in relative bar "
                 "(>= half the busiest synapse into "
                 "the same node) + entry-to-exit "
                 "route connectivity",
    "decoder": "merged (Frankenstein 4 rule): one "
               "pointer decoder; W_qd diagonal "
               "territory blocks are the parents' "
               "own; cross blocks tiny random std "
               f"{SIGMA_FILL} from the single filler "
               f"stream, seed {FILL_SEED}; b_qd and "
               "v_start placed by the territory maps",
    "host_species": H["species"],
    "donor_species": D["species"],
    "d": dF, "blocks": N_BLOCKS + 1,
    "stitched_blocks": N_BLOCKS,
    "joining_block": f"b{N_BLOCKS}",
    "ffn_hidden_per_block": {str(k): U_per_block[k]
                             for k in range(N_BLOCKS)},
    "ffn_hidden_joining": UJ,
    "heads_per_block": 1,
    "host_coord_map": {str(k): v for k, v in h_map.items()},
    "donor_coord_map": {str(k): v for k, v in d_map.items()},
    "host_unit_maps": {str(k): {str(u): v for u, v in
                                hu_maps[k].items()}
                       for k in range(N_BLOCKS)},
    "donor_unit_maps": {str(k): {str(u): v for u, v in
                                 du_maps[k].items()}
                        for k in range(N_BLOCKS)},
    "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_genome.npz",
         **{k: v for k, v in FW.items()},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_rep_stepA_genome.npz")


def edge_report(P):
    return {
        "species": P["species"],
        "gap_pct": round(P["gap"] * 100, 4),
        "spine": P["spine"],
        "ffn_units_on_path_per_block": {
            str(k): P["ffn_units_per_block"][k]
            for k in range(N_BLOCKS)},
        "stream_coords_used": P["coords_used"],
        "n_weight_synapses_total_graph":
            N_WEIGHT_TOTAL_GRAPH,
        "n_weight_synapses_total_brief":
            N_WEIGHT_TOTAL_BRIEF,
        "n_passing_weight_synapses": P["n_pass"],
        "n_path_weight_synapses": len(P["path_weight_edges"]),
        "n_path_embed": P["path_embed"],
        "n_path_per_block": {str(k): P["path_per_block"][k]
                             for k in range(N_BLOCKS)},
        "n_path_decoder": P["path_decoder"],
        "path_weight_synapses": [
            {"entry": e["entry"], "i": e["i"], "j": e["j"],
             "trav": round(e["trav"], 4),
             "share": round(e["share"], 4),
             "std": e["std"]}
            for e in P["path_weight_edges"]],
        "all_edges": [
            {"src": e["src"], "dst": e["dst"],
             "kind": e["kind"], "entry": e["entry"],
             "i": e["i"], "j": e["j"],
             "mean": e["mean"], "std": e["std"],
             "share": round(e["share"], 4),
             "trav": round(e["trav"], 4),
             "passes": e["passes"],
             "on_path": e["on_path"]}
            for e in P["edges"]],
    }


results = {
    "design": "Frankenstein step A REPLICATION stitch "
              "(deep-narrow line, 2026-11-26): the "
              "Frankenstein 4 rule (corrected paths "
              "+ merged decoder) applied to the "
              "4-block parents of rep_deep50",
    "founding_run": {
        "out_dir": "engine_runs/frankenstein_replication/founding",
        "config": "bert_engine_config_rep_deep50.json",
        "run_seed": 20261300,
    },
    "parent_evolution": {
        "note": "step A redo (sponsor correction): each "
                "parent evolved a further 100 generations "
                "(all genes, founding rules) before "
                "stitching; script "
                "frankenstein_rep_parents_evolve_2026-11-27.py",
        "host": {"species": 6,
                 "founding_gap_pct":
                     host_meta["founding_gap_pct"],
                 "evolved_gap_pct":
                     host_meta["best_validation_gap_pct"],
                 "seed": 20261310,
                 "genome":
                     "frankenstein_rep_parent_host_evolved.npz"},
        "donor": {"species": 11,
                  "founding_gap_pct":
                      donor_meta["founding_gap_pct"],
                  "evolved_gap_pct":
                      donor_meta["best_validation_gap_pct"],
                  "seed": 20261311,
                  "genome":
                      "frankenstein_rep_parent_donor_evolved.npz"},
    },
    "parents": {"host": edge_report(H),
                "donor": edge_report(D)},
    "merged_decoder": {
        "W_qd_full": W_qd_m.tolist(),
        "b_qd": b_qd_m.tolist(),
        "v_start": v_start_m.tolist(),
        "sigma_cross": SIGMA_FILL,
        "fill_seed": FILL_SEED,
        "attn_scale": SCALE,
        "clip": float(be.CLIP),
    },
    "construction_check": {
        "body_path_synapses_bit_exact": n_path_checked,
        "decoder_blocks_exact": bool(chk_blocks),
        "decoder_vectors_exact": bool(chk_vecs),
        "cross_block_max_abs": float(np.abs(cross).max()),
        "cross_block_std": float(cross.std()),
    },
    "frankenstein_stepA2": {
        "shape": {"stream_width": dF,
                  "ffn_width_per_stitched_block":
                      {str(k): U_per_block[k]
                       for k in range(N_BLOCKS)},
                  "joining_block_ffn_width": UJ,
                  "blocks": N_BLOCKS + 1,
                  "heads_per_block": 1,
                  "decoders": 1,
                  "total_params": int(sum(
                      v.size for v in FW.values()))},
        "validation_gap_pct": round(frank_gap * 100, 4),
        "mean_length": float(frank_lens.mean()),
        "sigma_fill": SIGMA_FILL,
        "fill_seed": FILL_SEED,
        "exact_path_synapses": {
            "host": int(sum(1 for _ in h_path)),
            "donor": int(sum(1 for _ in d_path))},
    },
}
with open(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_stitch_2026-11-27.json", "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_rep_stepA_stitch_2026-11-27.json")

state = {"parents": {
    tag: {"species": P["species"], "gap": P["gap"],
          "edges": [{k: e[k] for k in
                     ("src", "dst", "kind", "entry",
                      "i", "j", "on_path", "passes",
                      "trav")}
                    for e in P["edges"]]}
    for tag, P in PARENTS.items()},
    "frank": {"gap": frank_gap, "dF": dF,
              "U_per_block": {str(k): U_per_block[k]
                              for k in range(N_BLOCKS)},
              "UJ": UJ, "h_map": h_map, "d_map": d_map,
              "hu_maps": {str(k): hu_maps[k]
                          for k in range(N_BLOCKS)},
              "du_maps": {str(k): du_maps[k]
                          for k in range(N_BLOCKS)},
              "h_path": [list(x) for x in
                         sorted(h_path.keys())],
              "d_path": [list(x) for x in
                         sorted(d_path.keys())]}}
with open(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_state.json",
          "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_rep_stepA_state.json")
log("DONE build")
