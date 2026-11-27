#!/usr/bin/env python3
"""Frankenstein stitch #2 (sponsor design, Roman,
2026-11-25 09:52 EST; corrected path rule,
2026-11-25 10:04 EST).

THE DESIGN (sponsor's words, operationalised):
  - Take the main information path of two species
    champions of the fresh50_v2 run. Put the two paths
    side by side in one network. They share NO node.
  - Join them with a feed-forward layer placed before
    the decoder, so the joined layer can pick and
    choose between the two paths.
  - The result is the smallest whole network that
    fits a BERT shape. Widths follow the paths and
    need NOT be multiples of 2.
  - Stitching = mapping the genomes side by side.
    Holes (cross connections between the two paths'
    territories, and any slot the wider shape needs
    that neither parent fills, and in-territory
    synapses that are NOT on the parent's path) are
    filled with synapses of TINY RANDOM weight.
    NOT zeros.

STEP 1 - PARENTS. Host = the fresh50_v2 run champion
(species 8, gap 15.13%). Donor = the best OTHER
species champion of the same run, ranked by gap on
the fixed 200-problem validation set, taken from
the run checkpoint's recorded species best genomes.

STEP 2 - MAIN PATH (sponsor's method, agreed
2026-11-25 09:35 EST). For each parent:
  - Run the fixed validation set through the network.
    For every synapse (every individual weight),
    record the signal that actually flows through
    it: presynaptic activation times the weight. For
    the attention mixing itself (which has no
    weights), record the actual per-pair contribution
    terms as conduit flows (see flow definitions in
    compute_flows).
  - Per synapse: mean and std of the flow.
  - Rank synapses relative to their neighbours:
    neighbours = synapses connecting into the same
    node. Share = std / (sum of std over the fan-in
    group at that node).
  - CORRECTED MAIN PATH RULE (sponsor, 2026-11-25
    10:04 EST; stitch #1's walker threshold put
    48 of 56 synapses on the path, which the
    sponsor rejected: the path must be a PORTION
    of the network). A weight synapse PASSES iff
    its flow std is at least HALF the largest flow
    std among the weight synapses feeding the same
    node (a relative bar inside the fan-in group;
    equivalently share >= half the busiest share,
    since the group total cancels). The fan-in
    group is over weight synapses only: the
    residual stream edges and the attention/logit
    conduits are structural connectors, not
    synapses, and stay available for routing.
    (If residuals competed in the group, their
    stream std would dwarf every weight into the
    post-attention and post-FFN stream nodes and
    the path would collapse to the bare residual
    wire, contradicting the measured ablation
    facts, e.g. FFN unit 4 alone costs 119% of
    this champion's gap when silenced.) The MAIN
    PATH = the passing synapses that lie on a
    connected route from the entry nodes to the
    decoder output, where a route may step through
    passing synapses and the structural connectors;
    passing synapses on no such route are dropped.
    The random walk of stitch #1 is still run, but
    ONLY as a descriptive statistic (traversal
    counts and the busiest single route / spine);
    it no longer decides path membership.
    STOP RULE: if either parent's corrected path
    holds more than ~60% of the 56 weight
    synapses, stop and report instead of
    stitching.

STEP 3 - BUILD. Stream width = the stream
coordinates each parent's path actually uses, side
by side (host territory first, then donor). Block 1
holds both paths: in-territory path synapses keep
their EXACT parent weights; everything else in the
block matrices is tiny random filler. Block 2 is
the joining block before the decoder: a legal
pre-norm BERT block (attention + feed-forward,
residual) spanning the full joined width, ALL tiny
random (near pass-through filler), its feed-forward
width = the combined path FFN width. Decoder = the
host's decoder over the host territory (exact where
on the host's path), all other entries tiny random.
Filler std SIGMA_FILL = 0.03, fixed seed, ONE draw,
ONE score, reported as-is.

ENGINE FIT: the current ArchSpec cannot hold this
shape. A couple in the engine carries its own embed
projection and its own pointer head, and couples
chain d -> d; a pre-decoder joining block after the
single block, with the decoder reading the joined
width, is not expressible. So the Frankenstein is
built as an explicit weight set with its own
forward pass (below), verified against the engine
on both parents first (identical tours, gap equal
to 1e-9). That the engine needs a pre-decoder
joining slot is a finding of this build.

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_stitch_2026-11-25.py
(or system python3; numpy + matplotlib only.)
Outputs: frankenstein_2_genome.npz,
frankenstein_stitch2_2026-11-25.json,
frankenstein_stitch2_state.json (diagram input,
JSON, no pickle),
diagrams/frankenstein_2.png (separate renderer).
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
FILL_SEED = 20261125
WALK_SEED = 20261125
N_WALKERS = 200_000
PATH_MIN_TRAV = 0.02
EXPECTED_HOST_GAP = 0.1513148917731675

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
# STEP 1 - parents
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
    log(f"species {s} best genome: validation gap {gaps[s]*100:.4f}%")
donor_id = min((s for s in sp_ids if s != 8), key=lambda s: gaps[s])
donor_genome = sp_best[donor_id]
log(f"host = species 8 (gap {gaps[8]*100:.2f}%), "
    f"donor = species {donor_id} (gap {gaps[donor_id]*100:.2f}%)")

PARENTS = {
    "host": {"species": 8, "genome": host_genome,
             "gap": gaps[8]},
    "donor": {"species": donor_id, "genome": donor_genome,
              "gap": gaps[donor_id]},
}


def parent_weights(genome):
    """Engine genome -> generic weight dict for the
    single-block parent shape."""
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
# Generic forward + greedy pointer decode (mirrors the
# engine exactly; parity-checked on both parents below)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X, blocks=("b0",)):
    """X: (B, n, in_dim) -> Y: (B, n, d)."""
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
    """Greedy pointer decode, identical to
    bert_domain_tsp.decode_pointer for one genome."""
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
    # tour_lengths_batched expects (M, B, n) picks
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    gaps = lens / OPT - 1.0
    return float(gaps.mean()), picks, lens


# parity check: generic forward on both parents must
# reproduce the engine exactly
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
# STEP 2 - signal flows, fan-in shares, random walk
# ================================================================
# Node naming: in{c}, s0_{k}, q_{j}, k_{j}, v_{j},
# mix_{j}, s1_{k}, ffn_{u}, s2_{k}, dec_{j}, pick.
# Edge record: (src, dst, kind, entry, i, j, samples)
# kind: 'weight' (a real synapse / gene) or 'conduit'
# (attention mixing term, carries flow but is not a
# gene) or 'residual' (identity synapse, weight 1).

def compute_flows(W):
    """Per-synapse flow samples on the validation set.
    Flow through a weight = presynaptic activation
    times the weight, the term that actually enters
    the target. Attention conduits:
      A conduit v_k -> mix_k: A_attn[t,s] * V[s,k],
        the contribution term summed into the head
        output (samples over problem, query t, key s).
      Q conduit q_j -> mix_j: Q[t,j] times its
        attention-weighted K_j partner average,
        sum_s A_attn[t,s] * scale * K[s,j]  -- the
        product term through which Q_j enters the
        scores that set the mix (samples over t).
      K conduit k_j -> mix_j: K[s,j] times its
        attention-weighted Q_j partner average,
        sum_t A_attn[t,s] * scale * Q[t,j].
    Residual conduits: the stream value itself."""
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
    Atn = E / E.sum(axis=-1, keepdims=True)      # (B,t,s)
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
        aflow = Atn * V[:, :, j][:, None, :]     # (B,t,s)
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
    # decoder flows along the genome's own greedy tour
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
        # pre-tanh logit contribution of decoder coord j
        add(f"dec_{j}", "pick", "conduit", "logit", -1, j,
            A2[:, :, j][:, None, :] * qd[:, :, j][:, :, None]
            * SCALE)
    for e in edges:
        e["mean"] = float(e["samples"].mean())
        e["std"] = float(e["samples"].std())
    # fan-in shares (neighbours = same target node)
    from collections import defaultdict
    fanin = defaultdict(float)
    for e in edges:
        fanin[e["dst"]] += e["std"]
    for e in edges:
        tot = fanin[e["dst"]]
        e["share"] = e["std"] / tot if tot > 0 else 0.0
    return edges


def run_walk(edges):
    """Monte Carlo random walk over the synapse DAG.
    Returns traversal probability per edge index."""
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
    """Corrected path rule (see module docstring).
    The walk is run for descriptive traversal counts
    only; membership comes from the relative fan-in
    bar plus entry-to-exit route connectivity."""
    from collections import defaultdict
    trav, _ = run_walk(edges)
    for e, t in zip(edges, trav):
        e["trav"] = float(t)
    # pass test: weight synapses only, grouped by
    # target node; pass iff std >= half the group's
    # largest std (and std > 0)
    wmax = defaultdict(float)
    for e in edges:
        if e["kind"] == "weight":
            wmax[e["dst"]] = max(wmax[e["dst"]],
                                 e["std"])
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

    core = reach({"in0", "in1"}, fwd) \
        & reach({"pick"}, bwd)
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
    # spine: greedy max-trav route from the likeliest entry
    spine = []
    node = "in0"
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

# ================================================================
# STOP RULE (sponsor): if either corrected path is
# still more than ~60% of all 56 weight synapses,
# the rule or the measurement needs the sponsor's
# eye. Report and stop; do not stitch.
# ================================================================
N_WEIGHT_TOTAL = 56
_path_sizes = {tag: len(P["path_weight_edges"])
               for tag, P in PARENTS.items()}
if any(n > 0.6 * N_WEIGHT_TOTAL
       for n in _path_sizes.values()):
    stop_report = {
        "design": "sponsor stitching design, "
                  "2026-11-25; corrected path rule",
        "stopped": True,
        "stop_reason": "a corrected path still "
                       "holds more than ~60% of "
                       "the 56 weight synapses",
        "path_sizes": _path_sizes,
        "n_passing": {tag: P["n_pass"]
                      for tag, P in PARENTS.items()},
        "parents": {
            tag: {"species": P["species"],
                  "gap_pct": round(P["gap"] * 100, 4)}
            for tag, P in PARENTS.items()},
    }
    with open(HERE / "frankenstein_stitch2_"
                     "2026-11-25.json", "w") as f:
        json.dump(stop_report, f, indent=1)
    log(f"STOP: corrected path sizes {_path_sizes} "
        f"exceed 60% of {N_WEIGHT_TOTAL}; not "
        f"stitching. Report saved.")
    sys.exit(0)
log(f"corrected path sizes {_path_sizes} of "
    f"{N_WEIGHT_TOTAL}: both under the 60% stop "
    f"line, stitching proceeds")

# ================================================================
# STEP 3 - build the Frankenstein
# ================================================================
rng_fill = np.random.default_rng(FILL_SEED)


def fill(shape):
    return rng_fill.normal(0.0, SIGMA_FILL, size=shape)


H, D = PARENTS["host"], PARENTS["donor"]
h_coords = H["coords_used"]            # host stream coords kept
d_coords = D["coords_used"]
h_map = {k: i for i, k in enumerate(h_coords)}
d_map = {k: len(h_coords) + i for i, k in enumerate(d_coords)}
dF = len(h_coords) + len(d_coords)
h_units = H["ffn_units"]
d_units = D["ffn_units"]
hu_map = {u: i for i, u in enumerate(h_units)}
du_map = {u: len(h_units) + i for i, u in enumerate(d_units)}
U = len(h_units) + len(d_units)
U2 = U                                  # joining FFN width
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
# ---- embed: columns are territory-owned; rows are the
# shared input coordinates. Exact iff on that parent's
# path, else tiny random (a hole in the wider shape).
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
    """Territory blocks exact where on the parent's
    path; cross-territory and non-path in-territory
    entries are tiny random holes."""
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


# ---- block 1: the two paths side by side
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

# ---- block 2: the joining block before the decoder.
# A legal pre-norm BERT block over the full joined
# width, ALL tiny random filler (near pass-through):
# attention + feed-forward spanning the joined width
# so it can learn to pick and choose between paths.
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

# ---- decoder: the host's, over the joined width.
# Host-territory block exact where on the host's
# path; donor-side and cross entries tiny random.
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

# count exact vs filler synapses (matrix weights only)
n_exact_h = sum(1 for (e, i, j) in h_path
                if e != "embed.W_in") + sum(
        1 for (e, i, j) in h_path if e == "embed.W_in")
n_exact_d = len(d_path)
n_mat = sum(int(np.prod(v.shape)) for v in FW.values()
            if v.ndim == 2)
log(f"path weight synapses kept exact: host "
    f"{len(h_path)}, donor {len(d_path)}; total matrix "
    f"weights in Frankenstein {n_mat}")

frank_gap, frank_picks, frank_lens = gap_of(
    FW, X_val, blocks=("b0", "b1"))
log(f"FRANKENSTEIN validation gap: {frank_gap*100:.2f}% "
    f"(host {H['gap']*100:.2f}%, donor {D['gap']*100:.2f}%)")

# ================================================================
# Save genome + results
# ================================================================
meta = {
    "kind": "frankenstein_stitch2",
    "date": "2026-11-25",
    "path_rule": "corrected: fan-in relative bar "
                 "(>= half the busiest synapse into "
                 "the same node) + entry-to-exit "
                 "route connectivity",
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
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_2_genome.npz",
         **{k: v for k, v in FW.items()},
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_2_genome.npz")


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
    "design": "sponsor stitching design, 2026-11-25; "
              "corrected path rule (sponsor, "
              "2026-11-25 10:04 EST)",
    "path_rule": {
        "method": "a weight synapse passes iff its "
                  "flow std is at least half the "
                  "largest flow std among the weight "
                  "synapses feeding the same node "
                  "(relative bar inside the fan-in "
                  "group; the residual stream edges "
                  "and attention/logit conduits are "
                  "structural connectors, not "
                  "synapses, and stay available for "
                  "routing). The main path is the "
                  "passing synapses lying on a "
                  "connected route from the entry "
                  "nodes to the decoder output "
                  "(routes may step through passing "
                  "synapses and the connectors); "
                  "passing synapses on no such route "
                  "are dropped.",
        "flow": "presynaptic activation times weight, "
                "over the fixed 200-problem validation "
                "set; attention mixing recorded as "
                "per-pair contribution conduits",
        "walk": "the stitch #1 random walk is kept "
                "as a descriptive statistic only "
                "(traversal counts, busiest route); "
                "it no longer decides membership",
        "stitch1_comparison": "stitch #1's walker "
                              "threshold (>=2% of "
                              "walkers) put 48 of 56 "
                              "(host) and 45 of 56 "
                              "(donor) synapses on "
                              "the path",
    },
    "parents": {"host": edge_report(H),
                "donor": edge_report(D)},
    "frankenstein": {
        "shape": {"stream_width": dF,
                  "block1_ffn_width": U,
                  "joining_block_ffn_width": U2,
                  "blocks": 2, "heads_per_block": 1,
                  "total_matrix_weights": n_mat,
                  "total_params": int(sum(
                      v.size for v in FW.values()))},
        "validation_gap_pct": round(frank_gap * 100, 4),
        "frankenstein1_gap_pct": 20.1514,
        "mean_length": float(frank_lens.mean()),
        "sigma_fill": SIGMA_FILL,
        "exact_path_synapses": {"host": len(h_path),
                                "donor": len(d_path)},
        "engine_fit": "DOES NOT FIT the current "
                      "ArchSpec: a couple carries its "
                      "own embed and pointer head and "
                      "couples chain d->d, so a "
                      "pre-decoder joining block after "
                      "the main block is not "
                      "expressible. Built as an explicit "
                      "weight set with its own forward "
                      "pass (parity-verified on both "
                      "parents). The engine needs a "
                      "pre-decoder joining slot.",
        "frictions": [
            "LayerNorm in block 1 runs over the "
            "joined width (a legal BERT block), so "
            "the normalisation statistics are the one "
            "place the two paths touch inside block "
            "1; gains and biases are the parents' own, "
            "concatenated per territory.",
            "Both parents read the same input "
            "coordinates, so the entry projection is "
            "shared at the input side only: host "
            "columns and donor columns sit side by "
            "side in one (2 x dF) matrix.",
        ],
    },
}
with open(HERE / "frankenstein_stitch2_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_stitch2_2026-11-25.json")

# expose state for the diagram step as JSON (no
# pickle anywhere in this build)
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
with open(HERE / "frankenstein_stitch2_state.json",
          "w") as f:
    json.dump(state, f, indent=1)
log("saved frankenstein_stitch2_state.json "
    "(diagram input)")
log("DONE build")
