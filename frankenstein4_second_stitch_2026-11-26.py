#!/usr/bin/env python3
"""Second Frankenstein 4 build (replication run,
2026-11-26): the Frankenstein 4 rule applied to
the fresh50_v3 founding run (new run seed
20262700; the same fixed founding stage and the
same fixed 200-problem validation set, val_seed
20261116).

PIPELINE (mirrors stitch #2 + stitch #4 exactly):
  - Parents: host = the fresh50_v3 run champion
    (species 1, gap 16.2028%); donor = the best
    OTHER species champion of the same run, ranked
    by gap on the fixed validation set (species 5,
    gap 17.9167%). All four surviving species
    champions were ranked by measurement: species
    1: 16.2028%, species 5: 17.9167%, species 0:
    19.0105%, species 11: 19.5670%.
  - Main path per parent: the corrected rule of
    stitch #2 (a weight synapse passes iff its
    flow std is at least half the largest flow
    std among the weight synapses feeding the
    same node; the main path is the passing
    synapses on a connected entry-to-exit route;
    residual stream edges and attention/logit
    conduits are structural connectors, not
    synapses). Same STOP RULE: a path holding
    more than ~60% of the 56 weight synapses
    stops the build.
  - Body: the two paths side by side, sharing
    no node; in-territory path synapses exact;
    every other entry tiny random filler
    (std 0.03, ONE draw, seed 20261128); joining
    block before the decoder spanning the joined
    width, its feed-forward width = the combined
    path FFN width, all tiny random.
  - Decoder: stitch #4's MERGED decoder as the
    only change to the body: ONE pointer decoder
    whose W_qd carries BOTH parents' decoders in
    the diagonal territory blocks (exact), the
    off-diagonal cross blocks tiny random
    (std 0.03, ONE draw, seed 20261130); b_qd and
    v_start are the parents' values placed by
    the territory maps. No feed-forward layer
    after the decoder, no second decoder, no
    logit averaging. Decoder conventions
    identical to stitch #2/#4 (attn_scale =
    arch.attn_scale of be.initial_arch(cfg, 2)
    = 1/sqrt(2); CLIP = bert_engine.CLIP).

Construction checks (all asserted): generic
forward parity on both parents; every path
synapse in the body bit-exact against its
parent; merged decoder blocks and vectors exact
against the parents; body bit-identical except
the head arrays.

Scored ONCE on the fixed validation set.
Outputs: frankenstein4_second_genome.npz,
frankenstein4_second_stitch_2026-11-26.json,
frankenstein4_second_state.json and
frankenstein4_second_merged_state.json (diagram
inputs).
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
RUN = HERE / "engine_runs" / "fresh50_v3"
SIGMA_FILL = 0.03
FILL_SEED = 20261128
WALK_SEED = 20261128
SIGMA_CROSS = 0.03
MERGE_SEED = 20261130
N_WALKERS = 200_000

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

# ================================================================
# STEP 1 - parents: host = run champion; donor =
# best other species champion, all ranked by
# measurement on the fixed validation set
# ================================================================
z = np.load(RUN / "champion.npz")
host_genome = np.asarray(z["genome"], dtype=float)

ck = np.load(RUN / "checkpoint.npz", allow_pickle=True)
sp_ids = [int(s) for s in ck["species_ids"]]
sp_best = {int(s): np.asarray(g, dtype=float)
           for s, g in zip(ck["species_ids"], ck["best_genomes"])}
gaps = {}
for s, g in sp_best.items():
    gaps[s] = float(domain.validation_detail(layout, g)["mean_gap"])
    log(f"species {s} best genome: validation gap {gaps[s]*100:.4f}%")
host_id = next(s for s in sp_ids
               if np.array_equal(sp_best[s], host_genome))
host_detail = domain.validation_detail(layout, host_genome)
assert abs(host_detail["mean_gap"] - gaps[host_id]) < 1e-12
donor_id = min((s for s in sp_ids if s != host_id),
               key=lambda s: gaps[s])
donor_genome = sp_best[donor_id]
log(f"host = species {host_id} (gap {gaps[host_id]*100:.2f}%), "
    f"donor = species {donor_id} (gap {gaps[donor_id]*100:.2f}%)")

PARENTS = {
    "host": {"species": host_id, "genome": host_genome,
             "gap": gaps[host_id]},
    "donor": {"species": donor_id, "genome": donor_genome,
              "gap": gaps[donor_id]},
}


def parent_weights(genome):
    """Engine genome -> generic weight dict for the
    single-block parent shape (verbatim from
    stitch #2)."""
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
# Generic forward + greedy pointer decode (verbatim
# from stitch #2; parity-checked on both parents)
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


def gap_of(W, X, blocks=("b0",)):
    Y = forward_general(W, X, blocks)
    picks = decode_general(W, Y, X)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    gaps = lens / OPT - 1.0
    return float(gaps.mean()), picks, lens


for tag, P in PARENTS.items():
    W = parent_weights(P["genome"])
    g_mine, picks_mine, _ = gap_of(W, X_val)
    lens_e, picks_e = domain.evaluate_lengths(
        layout, P["genome"][None, :], X_val)
    tours_same = bool((picks_mine == picks_e[0]).all())
    log(f"parity {tag}: my gap {g_mine*100:.6f}% vs engine "
        f"{P['gap']*100:.6f}%, tours identical: {tours_same}")
    assert abs(g_mine - P["gap"]) < 1e-9 and tours_same
    P["weights"] = W

# ================================================================
# STEP 2 - signal flows, fan-in shares, corrected
# path extraction (verbatim from stitch #2)
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

    Emb = X @ W["embed.W_in"] + W["embed.b_in"]
    for c in range(2):
        for k in range(2):
            add(f"in{c}", f"s0_{k}", "weight", "embed.W_in",
                c, k, X[:, :, c] * W["embed.W_in"][c, k])
    Hb = ln_np(Emb, W["b0.lnA_gain"], W["b0.lnA_bias"], EPS)
    Q = Hb @ W["b0.W_q"] + W["b0.b_q"]
    K = Hb @ W["b0.W_k"] + W["b0.b_k"]
    V = Hb @ W["b0.W_v"] + W["b0.b_v"]
    S = np.einsum("btj,bsj->bts", Q, K) * SCALE
    S = S - S.max(axis=-1, keepdims=True)
    E = np.exp(S)
    Atn = E / E.sum(axis=-1, keepdims=True)
    Ah = np.einsum("bts,bsi->bti", Atn, V)
    for nm, M in (("W_q", W["b0.W_q"]), ("W_k", W["b0.W_k"]),
                  ("W_v", W["b0.W_v"])):
        tgt = {"W_q": "q", "W_k": "k", "W_v": "v"}[nm]
        for k in range(2):
            for j in range(2):
                add(f"s0_{k}", f"{tgt}_{j}", "weight",
                    f"b0.{nm}", k, j, Hb[:, :, k] * M[k, j])
    for j in range(2):
        qflow = Q[:, :, j] * np.einsum(
            "bts,bs->bt", Atn, K[:, :, j] * SCALE)
        add(f"q_{j}", f"mix_{j}", "conduit", "attnQ", -1, j,
            qflow)
        kflow = K[:, :, j] * np.einsum(
            "bts,bt->bs", Atn, Q[:, :, j] * SCALE)
        add(f"k_{j}", f"mix_{j}", "conduit", "attnK", -1, j,
            kflow)
        aflow = Atn * V[:, :, j][:, None, :]
        add(f"v_{j}", f"mix_{j}", "conduit", "attnA", -1, j,
            aflow)
    for k in range(2):
        for j in range(2):
            add(f"mix_{k}", f"s1_{j}", "weight", "b0.W_o",
                k, j, Ah[:, :, k] * W["b0.W_o"][k, j])
    A1 = Emb + Ah @ W["b0.W_o"] + W["b0.b_o"]
    for k in range(2):
        add(f"s0_{k}", f"s1_{k}", "residual", "residual1",
            k, k, Emb[:, :, k])
    Hf = ln_np(A1, W["b0.lnF_gain"], W["b0.lnF_bias"], EPS)
    A1f = np.tanh(Hf @ W["b0.W_1"] + W["b0.b_1"])
    for k in range(2):
        for u in range(8):
            add(f"s1_{k}", f"ffn_{u}", "weight", "b0.W_1",
                k, u, Hf[:, :, k] * W["b0.W_1"][k, u])
    for u in range(8):
        for j in range(2):
            add(f"ffn_{u}", f"s2_{j}", "weight", "b0.W_2",
                u, j, A1f[:, :, u] * W["b0.W_2"][u, j])
    A2 = A1 + A1f @ W["b0.W_2"] + W["b0.b_2"]
    for k in range(2):
        add(f"s1_{k}", f"s2_{k}", "residual", "residual2",
            k, k, A1[:, :, k])
    _, picks, _ = gap_of(W, X)
    gmean = A2.mean(axis=1)
    states = np.zeros((B, n, 2))
    for t in range(n):
        last = W["head.v_start"] if t == 0 else \
            A2[np.arange(B), picks[:, t - 1]]
        states[:, t, :] = gmean + last
    qd = states @ W["head.W_qd"] + W["head.b_qd"]
    for k in range(2):
        for j in range(2):
            add(f"s2_{k}", f"dec_{j}", "weight", "head.W_qd",
                k, j, states[:, :, k] * W["head.W_qd"][k, j])
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
    for _ in range(32):
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


for tag, P in PARENTS.items():
    edges = extract_path(compute_flows(P["weights"]))
    P["edges"] = edges
    wp = [e for e in edges
          if e["on_path"] and e["kind"] == "weight"]
    ffn_units = sorted({e["j"] for e in wp
                        if e["entry"] == "b0.W_1"}
                       | {e["i"] for e in wp
                          if e["entry"] == "b0.W_2"})
    head_on = any(e["on_path"] and e["entry"] in
                  ("b0.W_q", "b0.W_k", "b0.W_v", "b0.W_o")
                  for e in edges)
    coords = set()
    for e in edges:
        if not e["on_path"]:
            continue
        for nd in (e["src"], e["dst"]):
            base = nd.split("_")[0]
            if base in ("s0", "s1", "s2", "q", "k", "v",
                        "mix", "dec"):
                coords.add(int(nd.split("_")[1]))
    P["path_weight_edges"] = wp
    P["n_pass"] = int(sum(
        1 for e in edges
        if e["kind"] == "weight" and e["passes"]))
    P["ffn_units"] = ffn_units
    P["head_on_path"] = head_on
    P["coords_used"] = sorted(coords)
    spine = []
    best_in = max((e for e in edges if e["src"] == "in0"),
                  key=lambda e: e["trav"], default=None)
    node = "in0" if best_in is not None else "in1"
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
        f"synapses {P['n_pass']} of 56, path weight "
        f"synapses {len(wp)}, path FFN units {ffn_units}, "
        f"head on path: {head_on}, stream coords used "
        f"{P['coords_used']}, spine {' -> '.join(spine)}")

# STOP RULE (as in stitch #2)
N_WEIGHT_TOTAL = 56
_path_sizes = {tag: len(P["path_weight_edges"])
               for tag, P in PARENTS.items()}
if any(n > 0.6 * N_WEIGHT_TOTAL
       for n in _path_sizes.values()):
    stop_report = {
        "design": "second Frankenstein 4 build, "
                  "replication run, 2026-11-26",
        "stopped": True,
        "stop_reason": "a corrected path still holds "
                       "more than ~60% of the 56 "
                       "weight synapses",
        "path_sizes": _path_sizes,
        "n_passing": {tag: P["n_pass"]
                      for tag, P in PARENTS.items()},
        "parents": {
            tag: {"species": P["species"],
                  "gap_pct": round(P["gap"] * 100, 4)}
            for tag, P in PARENTS.items()},
    }
    with open(HERE / "frankenstein4_second_stitch_"
                     "2026-11-26.json", "w") as f:
        json.dump(stop_report, f, indent=1)
    log(f"STOP: corrected path sizes {_path_sizes} "
        f"exceed 60% of {N_WEIGHT_TOTAL}; not "
        f"stitching. Report saved.")
    sys.exit(0)
log(f"corrected path sizes {_path_sizes} of "
    f"{N_WEIGHT_TOTAL}: both under the 60% stop "
    f"line, stitching proceeds")

# ================================================================
# STEP 3 - build the body (verbatim from stitch #2)
# ================================================================
rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


H, D = PARENTS["host"], PARENTS["donor"]
h_coords = H["coords_used"]
d_coords = D["coords_used"]
h_map = {k: i for i, k in enumerate(h_coords)}
d_map = {k: len(h_coords) + i for i, k in enumerate(d_coords)}
dF = len(h_coords) + len(d_coords)
h_units = H["ffn_units"]
d_units = D["ffn_units"]
hu_map = {u: i for i, u in enumerate(h_units)}
du_map = {u: len(h_units) + i for i, u in enumerate(d_units)}
U = len(h_units) + len(d_units)
U2 = U
log(f"Frankenstein shape: stream width {dF} "
    f"(host {len(h_coords)} + donor {len(d_coords)}), "
    f"block-1 FFN width {U} (host {len(h_units)} + donor "
    f"{len(d_units)}), joining FFN width {U2}")

h_path = {(e["entry"], e["i"], e["j"]): True
          for e in H["path_weight_edges"]}
d_path = {(e["entry"], e["i"], e["j"]): True
          for e in D["path_weight_edges"]}
HW, DW = H["weights"], D["weights"]

FW = {}
W_in = fill((2, dF))
for c in range(2):
    for k in h_coords:
        if ("embed.W_in", c, k) in h_path:
            W_in[c, h_map[k]] = HW["embed.W_in"][c, k]
    for k in d_coords:
        if ("embed.W_in", c, k) in d_path:
            W_in[c, d_map[k]] = DW["embed.W_in"][c, k]
FW["embed.W_in"] = W_in
b_in = fill((dF,))
for k in h_coords:
    b_in[h_map[k]] = HW["embed.b_in"][k]
for k in d_coords:
    b_in[d_map[k]] = DW["embed.b_in"][k]
FW["embed.b_in"] = b_in


def territory_vec(name_h, name_d, size):
    v = fill((size,))
    for k in h_coords:
        v[h_map[k]] = HW[name_h][k]
    for k in d_coords:
        v[d_map[k]] = DW[name_d][k]
    return v


def block1_matrix(entry, hW, dW, row_map_h, row_map_d,
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


for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
    FW[f"b0.{nm}"] = territory_vec(f"b0.{nm}", f"b0.{nm}", dF)
for nm in ("W_q", "W_k", "W_v", "W_o"):
    FW[f"b0.{nm}"] = block1_matrix(
        f"b0.{nm}", HW[f"b0.{nm}"], DW[f"b0.{nm}"],
        h_map, d_map, h_map, d_map, dF, dF)
for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
    FW[f"b0.{nm}"] = territory_vec(f"b0.{nm}", f"b0.{nm}", dF)
FW["b0.W_1"] = block1_matrix(
    "b0.W_1", HW["b0.W_1"], DW["b0.W_1"],
    h_map, d_map, hu_map, du_map, dF, U)
FW["b0.W_2"] = block1_matrix(
    "b0.W_2", HW["b0.W_2"], DW["b0.W_2"],
    hu_map, du_map, h_map, d_map, U, dF)
b_1 = fill((U,))
for u in h_units:
    b_1[hu_map[u]] = HW["b0.b_1"][u]
for u in d_units:
    b_1[du_map[u]] = DW["b0.b_1"][u]
FW["b0.b_1"] = b_1

# joining block: all tiny random filler
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

# decoder placeholder (stitch #2 form: host's over
# the joined width); replaced wholesale by the
# merged decoder below, exactly as stitch #4 did
W_qd = fill((dF, dF))
for i_old, i_new in h_map.items():
    for j_old, j_new in h_map.items():
        if ("head.W_qd", i_old, j_old) in h_path:
            W_qd[i_new, j_new] = HW["head.W_qd"][i_old, j_old]
FW["head.W_qd"] = W_qd
v_start = fill((dF,))
b_qd = fill((dF,))
for k in h_coords:
    v_start[h_map[k]] = HW["head.v_start"][k]
    b_qd[h_map[k]] = HW["head.b_qd"][k]
FW["head.v_start"] = v_start
FW["head.b_qd"] = b_qd

# ---- construction check: every path synapse in
# the body is bit-exact against its parent
MAT_ENTRIES = ("embed.W_in", "b0.W_q", "b0.W_k", "b0.W_v",
               "b0.W_o", "b0.W_1", "b0.W_2")
n_path_checked = 0
for tag, P, PW, cmap, umap in (
        ("host", H, HW, h_map, hu_map),
        ("donor", D, DW, d_map, du_map)):
    for e in P["path_weight_edges"]:
        entry, i, j = e["entry"], e["i"], e["j"]
        if entry == "head.W_qd":
            continue  # decoder checked below
        assert entry in MAT_ENTRIES, entry
        if entry == "embed.W_in":
            pos = (i, cmap[j])
        elif entry == "b0.W_1":
            pos = (cmap[i], umap[j])
        elif entry == "b0.W_2":
            pos = (umap[i], cmap[j])
        else:
            pos = (cmap[i], cmap[j])
        assert FW[entry][pos] == PW[entry][i, j], \
            f"path synapse not exact: {tag} {entry} {(i, j)}"
        n_path_checked += 1
log(f"construction check: {n_path_checked} body path "
    f"synapses bit-exact against their parents")

# ================================================================
# STEP 4 - the merged decoder (verbatim method
# from stitch #4; the ONLY change to the body)
# ================================================================
rng_merge = np.random.default_rng(MERGE_SEED)
W_qd_m = rng_merge.normal(0.0, SIGMA_CROSS,
                          size=(dF, dF))  # one draw
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

# ---- construction checks on the merged decoder
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
    f"{SIGMA_CROSS}, seed {MERGE_SEED}, one draw)")
assert chk_blocks and chk_vecs

FW4 = dict(FW)
FW4["head.W_qd"] = W_qd_m
FW4["head.b_qd"] = b_qd_m
FW4["head.v_start"] = v_start_m
body_same = all(
    np.array_equal(FW4[k], FW[k])
    for k in FW if not k.startswith("head."))
log(f"body check: every non-decoder array "
    f"bit-identical to the stitch body: {body_same}")
assert body_same

# ================================================================
# Score ONCE on the fixed validation set
# ================================================================
frank_gap, frank_picks, frank_lens = gap_of(
    FW4, X_val, blocks=("b0", "b1"))
log(f"SECOND FRANKENSTEIN 4 validation gap: "
    f"{frank_gap*100:.4f}% (host "
    f"{H['gap']*100:.2f}%, donor {D['gap']*100:.2f}%)")

# ================================================================
# Save genome + results + diagram states
# ================================================================
meta = {
    "kind": "frankenstein4_second_stitch",
    "date": "2026-11-26",
    "founding_run": "fresh50_v3 (run seed 20262700)",
    "path_rule": "corrected: fan-in relative bar "
                 "(>= half the busiest synapse into "
                 "the same node) + entry-to-exit "
                 "route connectivity",
    "decoder": "merged (stitch #4 rule): one "
               "pointer decoder; W_qd diagonal "
               "territory blocks are the parents' "
               "own; cross blocks tiny random std "
               f"{SIGMA_CROSS}, one draw, seed "
               f"{MERGE_SEED}; b_qd and v_start "
               "placed by the territory maps",
    "host_species": H["species"],
    "donor_species": D["species"],
    "d": dF, "ffn_hidden_block1": U,
    "ffn_hidden_block2_joining": U2,
    "blocks": 2, "heads_per_block": 1,
    "host_coord_map": {str(k): v for k, v in h_map.items()},
    "donor_coord_map": {str(k): v for k, v in d_map.items()},
    "host_unit_map": {str(k): v for k, v in hu_map.items()},
    "donor_unit_map": {str(k): v for k, v in du_map.items()},
    "sigma_fill": SIGMA_FILL, "fill_seed": FILL_SEED,
    "sigma_cross": SIGMA_CROSS, "merge_seed": MERGE_SEED,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein4_second_genome.npz",
         **{k: v for k, v in FW4.items()},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein4_second_genome.npz")


def edge_report(P):
    return {
        "species": P["species"],
        "gap_pct": round(P["gap"] * 100, 4),
        "spine": P["spine"],
        "ffn_units_on_path": P["ffn_units"],
        "head_on_path": P["head_on_path"],
        "stream_coords_used": P["coords_used"],
        "n_weight_synapses_total": N_WEIGHT_TOTAL,
        "n_passing_weight_synapses": P["n_pass"],
        "n_path_weight_synapses": len(P["path_weight_edges"]),
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
    "design": "second Frankenstein 4 build "
              "(replication run, 2026-11-26): the "
              "Frankenstein 4 rule (stitch #2 paths "
              "+ stitch #4 merged decoder) applied "
              "to the fresh50_v3 founding run",
    "founding_run": {
        "out_dir": "engine_runs/fresh50_v3",
        "config": "bert_engine_config_fresh50_v3.json",
        "run_seed": 20262700,
        "species_champion_gaps_pct": {
            str(s): round(gaps[s] * 100, 4)
            for s in sorted(gaps)},
    },
    "parents": {"host": edge_report(H),
                "donor": edge_report(D)},
    "merged_decoder": {
        "W_qd_full": W_qd_m.tolist(),
        "b_qd": b_qd_m.tolist(),
        "v_start": v_start_m.tolist(),
        "sigma_cross": SIGMA_CROSS,
        "merge_seed": MERGE_SEED,
        "attn_scale": SCALE,
        "clip": float(be.CLIP),
    },
    "construction_check": {
        "body_path_synapses_bit_exact": n_path_checked,
        "decoder_blocks_exact": bool(chk_blocks),
        "decoder_vectors_exact": bool(chk_vecs),
        "cross_block_max_abs": float(np.abs(cross).max()),
        "cross_block_std": float(cross.std()),
        "body_non_decoder_arrays_bit_identical":
            bool(body_same),
    },
    "frankenstein4_second": {
        "shape": {"stream_width": dF,
                  "block1_ffn_width": U,
                  "joining_block_ffn_width": U2,
                  "blocks": 2, "heads_per_block": 1,
                  "decoders": 1,
                  "total_params": int(sum(
                      v.size for v in FW4.values()))},
        "validation_gap_pct": round(frank_gap * 100, 4),
        "mean_length": float(frank_lens.mean()),
        "sigma_fill": SIGMA_FILL,
        "fill_seed": FILL_SEED,
        "exact_path_synapses": {"host": len(h_path),
                                "donor": len(d_path)},
    },
}
with open(HERE / "frankenstein4_second_stitch_"
                 "2026-11-26.json", "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein4_second_stitch_2026-11-26.json")

state = {"parents": {
    tag: {"species": P["species"], "gap": P["gap"],
          "edges": [{k: e[k] for k in
                     ("src", "dst", "kind", "entry",
                      "i", "j", "on_path", "passes",
                      "trav")}
                    for e in P["edges"]]}
    for tag, P in PARENTS.items()},
    "frank": {"gap": frank_gap, "dF": dF, "U": U,
              "U2": U2, "h_map": h_map, "d_map": d_map,
              "hu_map": hu_map, "du_map": du_map,
              "h_path": [list(x) for x in
                         sorted(h_path.keys())],
              "d_path": [list(x) for x in
                         sorted(d_path.keys())]}}
with open(HERE / "frankenstein4_second_state.json",
          "w") as f:
    json.dump(state, f, indent=1)
merged_state = {"parents": state["parents"],
                "frank": state["frank"]}
with open(HERE / "frankenstein4_second_merged_state"
                 ".json", "w") as f:
    json.dump(merged_state, f, indent=1)
log("saved diagram state files")
log("DONE build")
