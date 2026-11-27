#!/usr/bin/env python3
"""BERT LIFECYCLE DRIVER - the Frankenstein lifecycle,
frozen by the sponsor (Roman) on 2026-11-27 and
specified in FRANKENSTEIN_METHOD.md.

The lifecycle (his step order):
  1  deep and skinny founding evolution
  2  Frankenstein: mature the top two parents, isolate
     their paths, stitch side by side, merged decoder,
     evolve the stitching genes (seams)
  3  stack one skinny narrow block at the back, evolve
     only the new block
  4  Frankenstein the stack: stitch the stack run's top
     two champions side by side over the commonized
     trunk, evolve the widened backend block
  5  widen the whole network (Net2WiderNet split: stream
     x2, every FFN x2, every head count x2, decoder
     widened with the stream, attn_scale halved) with
     tiny noise (std 0.003, config) on the new copies
     only; the no-noise widen must first reproduce the
     subject exactly (preservation gate)
  6  open evolution (whole network, scaled rate),
     founded on the widened genome
  7  start over: a second independent line, steps 1-6
  8  merge: stitch the two lines' champions (step 2
     rule), evolve the seams
  9  growth rounds: append a skinny appendix block
     (~1/4 size), breed in it, Frankenstein it (stitch
     its top two; the appendix roughly doubles); repeat
 10  the grown stack is Frankensteined into the standing
     network (the step 9 stitch over the common trunk)
 11  free evolution (whole network, scaled rate)
 12  repeat cycles until a cap is hit or a full cycle
     brings no improvement
 13  keep the smallest network that had the best score

Width grows by Frankenstein stitching AND by the one
widen step (step 5) before open evolution. The widen
step was added on 2026-11-27 by the sponsor's ruling
that what was just done with Frankenstein is the logic
for evolution: both completed lines (original and
replication) ran exactly this widen between the backend
Frankenstein and open evolution, and the 2026-11-27
control test showed it contributed (evolution alone on
the un-widened step C champion reached 6.1116% vs the
widened line's 5.2474%). The earlier same-day freeze,
which had open evolution follow the backend Frankenstein
directly, is superseded on this point. There is NO
REINFORCE stage in this lifecycle: the replication
skipped the original line's REINFORCE tuning phase and
still won.

STAGE CODE is reused from the proven scripts, faithfully
adapted and parameterized (the originals are hardcoded
one-off runs):
  - founding: run_bert_engine.py, driven by a generated
    engine config (deep50_stepA shape)
  - founders measured: frankenstein_rep_measure_
    founders_2026-11-27.py
  - parents maturing / seam / stack / backend / open
    evolution: the single masked-evolution loop shared
    by frankenstein_stepA_parents_evolve, stepA2_evolve,
    stepB_evolve, stepC_evolve and
    frankenstein_widened_evolve_scaled_2026-11-27.py
    (population on a flat genome, mask of mutable genes,
    elites 2, tournament 3, no crossover, rotating
    200-problem batches from a 2,000-problem pool,
    validation on the fixed set, per-generation atomic
    checkpoints, best-on-validation vaulting, backbone
    bit-identity assertions in seam stages)
  - path isolation: compute_flows_gen / extract_path
    from frankenstein_twofranks_stitch_2026-11-26.py
    (generalized there from stitch #2), extended to
    multi-head blocks
  - stitch construction: frankenstein_stepA2_stitch_
    2026-11-26.py (territory placement, joining block,
    merged decoder, one filler rng stream, std 0.03)
  - deepen: frankenstein_stepB_deepen_2026-11-26.py
  - backend stitch: frankenstein_stepC_stitch_
    2026-11-26.py (commonized trunk, widened block)
  - widen: frankenstein_stepC_widen003_2026-11-26.py and
    frankenstein_rep_stepC_widen003_2026-11-27.py (the
    Net2WiderNet split construction, the new-copy noise
    amendment at std 0.003, and the no-noise parity
    gate), generalized here to the driver's Net form

DISCIPLINES kept at every stage: sanity gates (a
founder's validation gap must reproduce before a stage
counts), per-generation atomic checkpoints, fully
detached stage launches (subprocess.Popen with
start_new_session=True), best-on-fixed-validation
vaulting (val_seed 20261116, 200 problems, 10 cities),
backbone bit-identity assertions in seam stages.

STATE: <run_dir>/lifecycle_state.json records the
cycle, line and stage, every champion (file, genes,
score), the standing best, the smallest-at-best
champion (step 13) and the termination check after
each cycle. The driver is resumable: a stage whose
COMPLETE marker and result exist is not rerun.

USAGE:
  python3 bert_lifecycle.py run --config bert_lifecycle_config.json \
      --run-dir engine_runs/lifecycle_run
  python3 bert_lifecycle.py _worker evolve --job <job.json>
      (internal: one evolution stage, launched detached
      by the driver)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from tsp_genome2 import held_karp

T0 = time.time()

SIGMA_FILL = 0.03          # stitch filler / new-block filler
LN_GAIN_SIGMA = 0.01       # layer-norm gain convention
VAL_SEED = 20261116        # the fixed validation set
ELIM_SEED = 20261272       # engine config convention, unchanged


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, indent=1))


def atomic_savez(path: Path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


# ================================================================
# Domain (the fixed validation set, as in every proven script)
# ================================================================
def build_domain():
    """TSPDomain on the deep50 engine config purely as the
    carrier of the fixed validation set (val_seed 20261116,
    200 problems, 10 cities) and the reference ln_eps /
    initial attn_scale, exactly as the proven scripts set
    up. Returns (domain, X_val, OPT, eps)."""
    cfg = be.EngineConfig.load(
        HERE / "bert_engine_config_deep50_stepA.json")
    backend = get_backend(cfg.backend, cfg.device, cfg.precision)
    arch = be.initial_arch(cfg, 2)
    domain = tsp.TSPDomain(cfg, backend)
    return domain, domain.X_val, domain.optima, arch.ln_eps


# ================================================================
# Network representation (the Frankenstein form)
#
# A network is a weights dict plus a layout:
#   blocks: ordered block names ("b0", "b1", ...)
#   block_heads: block -> number of attention heads
#   block_style: block -> "fused" (one projection set, keys
#       b.W_q ...) or "multi" (per-head keys b.h0.W_q ...,
#       head outputs summed, each head adding its own b_o)
#   d: stream width; attn_scale and ln_eps carried in meta
#       (attn_scale is inherited from the founding arch and
#       NEVER recomputed from d - the proven stitches keep
#       the parents' scale when the stream doubles)
# ================================================================
FUSED_ATTN = ("W_q", "W_k", "W_v", "W_o",
              "b_q", "b_k", "b_v", "b_o")
BLOCK_TAIL = ("lnF_gain", "lnF_bias", "W_1", "b_1", "W_2", "b_2")


def attn_keys(blk, style, heads):
    if style == "fused":
        return [f"{blk}.{nm}" for nm in FUSED_ATTN]
    return [f"{blk}.h{h}.{nm}" for h in range(heads)
            for nm in FUSED_ATTN]


def block_keys(blk, style, heads):
    return ([f"{blk}.lnA_gain", f"{blk}.lnA_bias"]
            + attn_keys(blk, style, heads)
            + [f"{blk}.{nm}" for nm in BLOCK_TAIL])


def gene_order(blocks, styles, heads_map):
    keys = ["embed.W_in", "embed.b_in"]
    for blk in blocks:
        keys += block_keys(blk, styles[blk], heads_map[blk])
    keys += ["head.W_qd", "head.b_qd", "head.v_start"]
    return keys


def pack(W, order):
    return np.concatenate([np.asarray(W[k], dtype=float).ravel()
                           for k in order])


def unpack(vec, order, shapes):
    W = {}
    off = 0
    for k in order:
        n = int(np.prod(shapes[k]))
        W[k] = vec[off:off + n].reshape(shapes[k])
        off += n
    return W


class Net:
    def __init__(self, W, blocks, styles, heads_map, d,
                 attn_scale, ln_eps, meta_extra=None):
        self.W = W
        self.blocks = list(blocks)
        self.styles = dict(styles)
        self.heads = dict(heads_map)
        self.d = int(d)
        self.attn_scale = float(attn_scale)
        self.ln_eps = float(ln_eps)
        self.meta_extra = meta_extra or {}

    @property
    def order(self):
        return gene_order(self.blocks, self.styles, self.heads)

    @property
    def shapes(self):
        return {k: np.asarray(self.W[k]).shape for k in self.order}

    def flat(self):
        return pack(self.W, self.order)

    @property
    def n_genes(self):
        return int(sum(np.asarray(self.W[k]).size
                       for k in self.order))

    def ffn_width(self, blk):
        return int(np.asarray(self.W[f"{blk}.W_1"]).shape[1])

    def save(self, path: Path, kind: str, extra=None):
        meta = {
            "kind": kind,
            "gene_order": self.order,
            "blocks": self.blocks,
            "block_styles": self.styles,
            "block_heads": self.heads,
            "d": self.d,
            "attn_scale": self.attn_scale,
            "ln_eps": self.ln_eps,
            "ffn_hidden": {b: self.ffn_width(b)
                           for b in self.blocks},
            "n_genes": self.n_genes,
        }
        meta.update(self.meta_extra)
        if extra:
            meta.update(extra)
        atomic_savez(path, **{k: np.asarray(self.W[k], dtype=float)
                              for k in self.order},
                     meta=np.array(json.dumps(meta)))

    def copy(self):
        return Net({k: np.asarray(v, dtype=float).copy()
                    for k, v in self.W.items()},
                   self.blocks, self.styles, self.heads, self.d,
                   self.attn_scale, self.ln_eps,
                   dict(self.meta_extra))


def load_net(path: Path) -> Net:
    """Load a lifecycle genome, or a proven Frankenstein
    genome npz (stepA2 / stepB / stepC / twofranks forms),
    deriving the layout from its keys when the lifecycle
    meta fields are absent."""
    z = np.load(path)
    keys = [k for k in z.files if k != "meta"]
    W = {k: np.asarray(z[k], dtype=float) for k in keys}
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if "block_styles" in meta and "blocks" in meta:
        return Net(W, meta["blocks"], meta["block_styles"],
                   meta["block_heads"], meta["d"],
                   meta["attn_scale"], meta["ln_eps"],
                   {k: v for k, v in meta.items()})
    # proven-genome form: derive blocks / styles / heads
    blocks = sorted({k.split(".")[0] for k in keys
                     if k.startswith("b") and k[1:].split(".")[0].isdigit()},
                    key=lambda b: int(b[1:]))
    styles, heads_map = {}, {}
    for blk in blocks:
        hs = sorted({int(k.split(".")[1][1:]) for k in keys
                     if k.startswith(f"{blk}.h")
                     and k.split(".")[1][1:].isdigit()})
        if hs:
            styles[blk] = "multi"
            heads_map[blk] = len(hs)
        else:
            styles[blk] = "fused"
            heads_map[blk] = 1
    d = int(W["embed.b_in"].shape[0])
    scale = float(meta.get("attn_scale", 1.0 / np.sqrt(2)))
    eps = float(meta.get("ln_eps", 1e-5))
    return Net(W, blocks, styles, heads_map, d, scale, eps,
               {"proven_meta": meta})


def plain_parent_net(genome, layout, attn_scale, ln_eps):
    """Engine genome -> Net (verbatim conversion from the
    proven stitches' parent_weights: couple 0, each block's
    head 0 fused projection set). attn_scale / ln_eps are
    the founding arch's, as in the proven scripts."""
    dmap = layout.unpack_dict(genome)
    n_blocks = len([k for k in dmap
                    if k.startswith("c0.b")
                    and k.endswith(".lnA_gain")])
    blocks = [f"b{k}" for k in range(n_blocks)]
    W = {"embed.W_in": dmap["c0.embed.W_in"],
         "embed.b_in": dmap["c0.embed.b_in"],
         "head.v_start": dmap["c0.head.v_start"],
         "head.W_qd": dmap["c0.head.W_qd"],
         "head.b_qd": dmap["c0.head.b_qd"]}
    for k in range(n_blocks):
        p = f"c0.b{k}"
        W[f"b{k}.lnA_gain"] = dmap[f"{p}.lnA_gain"]
        W[f"b{k}.lnA_bias"] = dmap[f"{p}.lnA_bias"]
        for nm in FUSED_ATTN:
            W[f"b{k}.{nm}"] = dmap[f"{p}.h0.{nm}"]
        W[f"b{k}.lnF_gain"] = dmap[f"{p}.lnF_gain"]
        W[f"b{k}.lnF_bias"] = dmap[f"{p}.lnF_bias"]
        W[f"b{k}.W_1"] = dmap[f"{p}.W_1"]
        W[f"b{k}.b_1"] = dmap[f"{p}.b_1"]
        W[f"b{k}.W_2"] = dmap[f"{p}.W_2"]
        W[f"b{k}.b_2"] = dmap[f"{p}.b_2"]
    arch_d = int(W["embed.b_in"].shape[0])
    return Net(W, blocks, {b: "fused" for b in blocks},
               {b: 1 for b in blocks}, arch_d,
               attn_scale, ln_eps)


# ================================================================
# Forward + greedy pointer decode (the Frankenstein builds'
# conventions, generalized to fused and multi-head blocks;
# multi-head outputs are summed, each head adding its own
# b_o - verbatim from frankenstein_stepC_evolve)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_net(net: Net, X):
    W, scale, eps = net.W, net.attn_scale, net.ln_eps
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in net.blocks:
        Hb = ln_np(A, W[f"{blk}.lnA_gain"], W[f"{blk}.lnA_bias"],
                   eps)
        if net.styles[blk] == "fused":
            prefixes = [f"{blk}."]
        else:
            prefixes = [f"{blk}.h{h}."
                        for h in range(net.heads[blk])]
        for pfx in prefixes:
            Q = Hb @ W[f"{pfx}W_q"] + W[f"{pfx}b_q"]
            K = Hb @ W[f"{pfx}W_k"] + W[f"{pfx}b_k"]
            V = Hb @ W[f"{pfx}W_v"] + W[f"{pfx}b_v"]
            S = np.einsum("btj,bsj->bts", Q, K) * scale
            S = S - S.max(axis=-1, keepdims=True)
            E = np.exp(S)
            Atn = E / E.sum(axis=-1, keepdims=True)
            Ah = np.einsum("bts,bsi->bti", Atn, V)
            A = A + Ah @ W[f"{pfx}W_o"] + W[f"{pfx}b_o"]
        Hf = ln_np(A, W[f"{blk}.lnF_gain"], W[f"{blk}.lnF_bias"],
                   eps)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"] + W[f"{blk}.b_1"]) \
            @ W[f"{blk}.W_2"] + W[f"{blk}.b_2"]
    return A


def decode_net(net: Net, Y):
    W, scale = net.W, net.attn_scale
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
            np.einsum("bi,bni->bn", q, Y) * scale)
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last = Y[arB, pick]
    return picks


def gap_net(net: Net, X, opt):
    Y = forward_net(net, X)
    picks = decode_net(net, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean()), picks, lens


# ================================================================
# Signal flows + corrected path rule (verbatim logic from
# frankenstein_twofranks_stitch's compute_flows_gen /
# extract_path, themselves generalized from stitch #2;
# extended here to multi-head blocks: each head gets its
# own q/k/v/mix nodes and its W_o entries feed the block's
# mid-stream nodes, so a head's synapses compete in the
# same fan-in group as the proven fused form)
#
# Corrected path rule (sponsor, 2026-11-25): a weight
# synapse is on the main path iff its signal std is at
# least half the max std among weight synapses feeding
# the same node AND it lies on a connected entry-to-exit
# route (residual / conduit edges are structural
# connectors, always routable).
# ================================================================
def compute_flows(net: Net, X_val, opt):
    W, scale, eps = net.W, net.attn_scale, net.ln_eps
    d = net.d
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
    for bi, blk in enumerate(net.blocks):
        s_in, s_mid, s_out = f"s{2*bi}", f"s{2*bi+1}", f"s{2*bi+2}"
        U = net.ffn_width(blk)
        Hb = ln_np(A, W[f"{blk}.lnA_gain"], W[f"{blk}.lnA_bias"],
                   eps)
        if net.styles[blk] == "fused":
            head_list = [(f"{blk}.", "")]
        else:
            head_list = [(f"{blk}.h{h}.", f"h{h}")
                         for h in range(net.heads[blk])]
        A1 = A
        for pfx, htag in head_list:
            Q = Hb @ W[f"{pfx}W_q"] + W[f"{pfx}b_q"]
            K = Hb @ W[f"{pfx}W_k"] + W[f"{pfx}b_k"]
            V = Hb @ W[f"{pfx}W_v"] + W[f"{pfx}b_v"]
            S = np.einsum("btj,bsj->bts", Q, K) * scale
            S = S - S.max(axis=-1, keepdims=True)
            E = np.exp(S)
            Atn = E / E.sum(axis=-1, keepdims=True)
            Ah = np.einsum("bts,bsi->bti", Atn, V)
            qn, kn, vn, mx = (f"q{bi}{htag}", f"k{bi}{htag}",
                              f"v{bi}{htag}", f"mix{bi}{htag}")
            for nm, tgt in (("W_q", qn), ("W_k", kn),
                            ("W_v", vn)):
                M = W[f"{pfx}{nm}"]
                entry = f"{pfx}{nm}"
                for k in range(d):
                    for j in range(d):
                        add(f"{s_in}_{k}", f"{tgt}_{j}", "weight",
                            entry, k, j, Hb[:, :, k] * M[k, j])
            for j in range(d):
                qflow = Q[:, :, j] * np.einsum(
                    "bts,bs->bt", Atn, K[:, :, j] * scale)
                add(f"{qn}_{j}", f"{mx}_{j}", "conduit", "attnQ",
                    -1, j, qflow)
                kflow = K[:, :, j] * np.einsum(
                    "bts,bt->bs", Atn, Q[:, :, j] * scale)
                add(f"{kn}_{j}", f"{mx}_{j}", "conduit", "attnK",
                    -1, j, kflow)
                aflow = Atn * V[:, :, j][:, None, :]
                add(f"{vn}_{j}", f"{mx}_{j}", "conduit", "attnA",
                    -1, j, aflow)
            for k in range(d):
                for j in range(d):
                    add(f"{mx}_{k}", f"{s_mid}_{j}", "weight",
                        f"{pfx}W_o", k, j,
                        Ah[:, :, k] * W[f"{pfx}W_o"][k, j])
            A1 = A1 + Ah @ W[f"{pfx}W_o"] + W[f"{pfx}b_o"]
        for k in range(d):
            add(f"{s_in}_{k}", f"{s_mid}_{k}", "residual",
                "residual1", k, k, A[:, :, k])
        Hf = ln_np(A1, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], eps)
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
    _, picks, _ = gap_net(net, X, opt)
    gmean = A.mean(axis=1)
    states = np.zeros((B, n, d))
    for t in range(n):
        last = W["head.v_start"] if t == 0 else \
            A[np.arange(B), picks[:, t - 1]]
        states[:, t, :] = gmean + last
    qd = states @ W["head.W_qd"] + W["head.b_qd"]
    s_last = f"s{2*len(net.blocks)}"
    for k in range(d):
        for j in range(d):
            add(f"{s_last}_{k}", f"dec_{j}", "weight",
                "head.W_qd", k, j,
                states[:, :, k] * W["head.W_qd"][k, j])
    for j in range(d):
        add(f"dec_{j}", "pick", "conduit", "logit", -1, j,
            A[:, :, j][:, None, :] * qd[:, :, j][:, :, None]
            * scale)
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
    """Corrected path rule: relative fan-in bar over
    weight synapses + entry-to-exit route connectivity
    through passing synapses and structural connectors.
    (The proven scripts also run a random walk for the
    spine report; walk counts do not enter path
    membership, so the walk is not reproduced here.)"""
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


def path_summary(net: Net, edges):
    wp = [e for e in edges
          if e["on_path"] and e["kind"] == "weight"]
    units = {}
    for blk in net.blocks:
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
        "path_set": {(e["entry"], e["i"], e["j"]) for e in wp},
        "ffn_units_on_path_by_block": units,
        "path_weight_synapses": [
            {"entry": e["entry"], "i": e["i"], "j": e["j"],
             "share": round(e["share"], 4), "std": e["std"]}
            for e in wp],
    }


# ================================================================
# Frankenstein stitch construction (generalized from
# frankenstein_stepA2_stitch_2026-11-26.py, which is the
# Frankenstein 4 rule generalized to multi-block parents;
# the two-Franks merge uses the same construction):
#   - both parents must share block count, per-position
#     block style, stream width d and attn_scale
#   - stream doubles: host keeps coords 0..d-1, donor
#     takes d..2d-1 (identity / offset territory maps)
#   - every stitched matrix starts as tiny-random filler
#     (std 0.03, ONE rng stream); the parents' PATH
#     synapses are placed bit-exact on their territories;
#     off-path synapses are dropped (they become filler)
#   - per block, only path FFN units get territory; the
#     stitched FFN width is |host path units| + |donor
#     path units|
#   - vector entries (layer norms, biases, embed bias,
#     FFN output bias) take each parent's values on its
#     own territory at every position
#   - one joining block, all filler, copies the last
#     stitched block's FFN width; layer-norm gains are
#     1 + N(0, 0.01)
#   - merged decoder: W_qd drawn from the same filler
#     stream, its diagonal territory blocks overwritten
#     with the parents' FULL W_qd; b_qd / v_start are
#     exact territory placements on a zero base
#   - the backbone mask (exact-from-a-parent) is recorded
#     during construction; it equals the mask the proven
#     seam-evolution scripts derive by value
# Filler draw order follows the proven script exactly for
# fused blocks, so a stitch of the recorded step A
# parents with their recorded seed reproduces the proven
# genome bit-for-bit.
# ================================================================
def stitch_nets(host: Net, donor: Net, host_path, donor_path,
                fill_seed):
    assert len(host.blocks) == len(donor.blocks), \
        "stitch parents must have the same block count"
    assert host.d == donor.d, "stitch parents must share d"
    assert host.attn_scale == donor.attn_scale
    assert host.styles == donor.styles, \
        "stitch parents must share per-block styles"
    d = host.d
    dF = 2 * d
    in_dim = int(np.asarray(host.W["embed.W_in"]).shape[0])
    rng_fill = np.random.default_rng(fill_seed)

    def fill(shape):
        return rng_fill.normal(0.0, SIGMA_FILL, size=shape)

    h_map = {c: c for c in range(d)}
    d_map = {c: d + c for c in range(d)}
    maps = ((host, host_path, h_map), (donor, donor_path, d_map))

    # per-block FFN unit territory maps (path units only)
    hu_maps, du_maps, U_new = {}, {}, {}
    for pos, blk in enumerate(host.blocks):
        hu = host_path["ffn_units_on_path_by_block"][blk]
        du = donor_path["ffn_units_on_path_by_block"][blk]
        hu_maps[pos] = {u: i for i, u in enumerate(hu)}
        du_maps[pos] = {u: len(hu) + i for i, u in enumerate(du)}
        U_new[pos] = len(hu) + len(du)

    FW = {}
    MK = {}

    def placed(shape):
        return np.zeros(shape, dtype=bool)

    # ---- embed
    W_in = fill((in_dim, dF))
    mk_in = placed((in_dim, dF))
    for P, Ppath, cmap in maps:
        ps = Ppath["path_set"]
        for c in range(in_dim):
            for k in range(d):
                if ("embed.W_in", c, k) in ps:
                    W_in[c, cmap[k]] = P.W["embed.W_in"][c, k]
                    mk_in[c, cmap[k]] = True
    FW["embed.W_in"], MK["embed.W_in"] = W_in, mk_in

    def territory_vec(nm_host, nm_donor, size):
        v = fill((size,))
        mk = placed((size,))
        for k in range(d):
            v[h_map[k]] = host.W[nm_host][k]
            mk[h_map[k]] = True
            v[d_map[k]] = donor.W[nm_donor][k]
            mk[d_map[k]] = True
        return v, mk

    FW["embed.b_in"], MK["embed.b_in"] = territory_vec(
        "embed.b_in", "embed.b_in", dF)

    def block_matrix(entry_h, entry_d, PW_h, PW_d, ps_h, ps_d,
                     row_h, row_d, col_h, col_d, n_rows, n_cols):
        M = fill((n_rows, n_cols))
        mk = placed((n_rows, n_cols))
        for PW, ps, rmap, cmap in ((PW_h, ps_h, row_h, col_h),
                                   (PW_d, ps_d, row_d, col_d)):
            for i_old, i_new in rmap.items():
                for j_old, j_new in cmap.items():
                    entry = entry_h if PW is PW_h else entry_d
                    if (entry, i_old, j_old) in ps:
                        M[i_new, j_new] = PW[i_old, j_old]
                        mk[i_new, j_new] = True
        return M, mk

    new_blocks, new_styles, new_heads = [], {}, {}
    for pos, blk in enumerate(host.blocks):
        style = host.styles[blk]
        nblk = f"b{pos}"
        new_blocks.append(nblk)
        new_styles[nblk] = style
        ps_h, ps_d = host_path["path_set"], donor_path["path_set"]
        for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
            FW[f"{nblk}.{nm}"], MK[f"{nblk}.{nm}"] = \
                territory_vec(f"{blk}.{nm}", f"{blk}.{nm}", dF)
        if style == "fused":
            new_heads[nblk] = 1
            for nm in ("W_q", "W_k", "W_v", "W_o"):
                FW[f"{nblk}.{nm}"], MK[f"{nblk}.{nm}"] = \
                    block_matrix(f"{blk}.{nm}", f"{blk}.{nm}",
                                 host.W[f"{blk}.{nm}"],
                                 donor.W[f"{blk}.{nm}"],
                                 ps_h, ps_d, h_map, d_map,
                                 h_map, d_map, dF, dF)
            for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
                FW[f"{nblk}.{nm}"], MK[f"{nblk}.{nm}"] = \
                    territory_vec(f"{blk}.{nm}", f"{blk}.{nm}",
                                  dF)
        else:
            # multi-head: heads concatenate (host heads, then
            # donor heads); each head's matrices are placed
            # from that parent head's path on its territory
            heads_h, heads_d = host.heads[blk], donor.heads[blk]
            new_heads[nblk] = heads_h + heads_d
            head_src = [(host, h, h_map, ps_h)
                        for h in range(heads_h)] + \
                       [(donor, h, d_map, ps_d)
                        for h in range(heads_d)]
            for nm in ("W_q", "W_k", "W_v", "W_o"):
                for nh, (P, h, cmap, ps) in enumerate(head_src):
                    entry = f"{blk}.h{h}.{nm}"
                    key = f"{nblk}.h{nh}.{nm}"
                    M = fill((dF, dF))
                    mk = placed((dF, dF))
                    for i_old, i_new in cmap.items():
                        for j_old, j_new in cmap.items():
                            if (entry, i_old, j_old) in ps:
                                M[i_new, j_new] = \
                                    P.W[entry][i_old, j_old]
                                mk[i_new, j_new] = True
                    FW[key], MK[key] = M, mk
            for nm in ("b_q", "b_k", "b_v", "b_o"):
                for nh, (P, h, cmap, ps) in enumerate(head_src):
                    entry = f"{blk}.h{h}.{nm}"
                    key = f"{nblk}.h{nh}.{nm}"
                    v = fill((dF,))
                    mk = placed((dF,))
                    for k_old, k_new in cmap.items():
                        v[k_new] = P.W[entry][k_old]
                        mk[k_new] = True
                    FW[key], MK[key] = v, mk
            FW[f"{nblk}.b_2"], MK[f"{nblk}.b_2"] = \
                territory_vec(f"{blk}.b_2", f"{blk}.b_2", dF)
        # FFN (shared across heads, as in the proven form)
        FW[f"{nblk}.W_1"], MK[f"{nblk}.W_1"] = block_matrix(
            f"{blk}.W_1", f"{blk}.W_1", host.W[f"{blk}.W_1"],
            donor.W[f"{blk}.W_1"], ps_h, ps_d,
            h_map, d_map, hu_maps[pos], du_maps[pos],
            dF, U_new[pos])
        FW[f"{nblk}.W_2"], MK[f"{nblk}.W_2"] = block_matrix(
            f"{blk}.W_2", f"{blk}.W_2", host.W[f"{blk}.W_2"],
            donor.W[f"{blk}.W_2"], ps_h, ps_d,
            hu_maps[pos], du_maps[pos], h_map, d_map,
            U_new[pos], dF)
        b_1 = fill((U_new[pos],))
        mk_1 = placed((U_new[pos],))
        for u, u_new in hu_maps[pos].items():
            b_1[u_new] = host.W[f"{blk}.b_1"][u]
            mk_1[u_new] = True
        for u, u_new in du_maps[pos].items():
            b_1[u_new] = donor.W[f"{blk}.b_1"][u]
            mk_1[u_new] = True
        FW[f"{nblk}.b_1"], MK[f"{nblk}.b_1"] = b_1, mk_1

    # ---- joining block (all filler / all stitching)
    JB = f"b{len(host.blocks)}"
    UJ = U_new[len(host.blocks) - 1]
    new_blocks.append(JB)
    new_styles[JB] = "fused"
    new_heads[JB] = 1
    FW[f"{JB}.lnA_gain"] = 1.0 + rng_fill.normal(
        0.0, LN_GAIN_SIGMA, dF)
    FW[f"{JB}.lnA_bias"] = fill((dF,))
    FW[f"{JB}.lnF_gain"] = 1.0 + rng_fill.normal(
        0.0, LN_GAIN_SIGMA, dF)
    FW[f"{JB}.lnF_bias"] = fill((dF,))
    for nm in ("W_q", "W_k", "W_v", "W_o"):
        FW[f"{JB}.{nm}"] = fill((dF, dF))
    for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
        FW[f"{JB}.{nm}"] = fill((dF,))
    FW[f"{JB}.W_1"] = fill((dF, UJ))
    FW[f"{JB}.b_1"] = fill((UJ,))
    FW[f"{JB}.W_2"] = fill((UJ, dF))
    for k in [k for k in FW if k.startswith(f"{JB}.")]:
        MK[k] = placed(np.asarray(FW[k]).shape)

    # ---- merged decoder (Frankenstein 4 rule)
    W_qd = fill((dF, dF))
    mk_qd = placed((dF, dF))
    for P, cmap in ((host, h_map), (donor, d_map)):
        for i_old, i_new in cmap.items():
            for j_old, j_new in cmap.items():
                W_qd[i_new, j_new] = P.W["head.W_qd"][i_old, j_old]
                mk_qd[i_new, j_new] = True
    FW["head.W_qd"], MK["head.W_qd"] = W_qd, mk_qd
    for nm in ("head.b_qd", "head.v_start"):
        v = np.zeros(dF)
        mk = placed((dF,))
        for k in range(d):
            v[h_map[k]] = host.W[nm][k]
            mk[h_map[k]] = True
            v[d_map[k]] = donor.W[nm][k]
            mk[d_map[k]] = True
        FW[nm], MK[nm] = v, mk

    net = Net(FW, new_blocks, new_styles, new_heads, dF,
              host.attn_scale, host.ln_eps)
    order = net.order
    backbone = np.concatenate(
        [MK[k].ravel() for k in order])

    # ---- construction check (the proven one): every path
    # synapse of both parents placed bit-exact
    n_checked = 0
    for P, Ppath, cmap, umaps_all in (
            (host, host_path, h_map, hu_maps),
            (donor, donor_path, d_map, du_maps)):
        for (entry, i, j) in Ppath["path_set"]:
            if entry == "head.W_qd":
                continue  # full diagonal blocks, checked by mask
            if entry == "embed.W_in":
                pos = (i, cmap[j])
                tgt = FW[entry]
            elif entry.endswith(".W_1"):
                pos_i = entry.split(".")[0]
                k = host.blocks.index(pos_i)
                pos = (cmap[i], umaps_all[k][j])
                tgt = FW[f"b{k}.W_1"]
            elif entry.endswith(".W_2"):
                pos_i = entry.split(".")[0]
                k = host.blocks.index(pos_i)
                pos = (umaps_all[k][i], cmap[j])
                tgt = FW[f"b{k}.W_2"]
            elif ".h" in entry:
                parts = entry.split(".")
                k = host.blocks.index(parts[0])
                h_src = int(parts[1][1:])
                nh = h_src if P is host else host.heads[parts[0]] + h_src
                pos = (cmap[i], cmap[j])
                tgt = FW[f"b{k}.h{nh}.{parts[2]}"]
            else:
                k = host.blocks.index(entry.split(".")[0])
                pos = (cmap[i], cmap[j])
                tgt = FW[f"b{k}.{entry.split('.', 1)[1]}"]
            assert tgt[pos] == P.W[entry][i, j], \
                f"path synapse not exact: {entry} {(i, j)}"
            n_checked += 1
    info = {
        "d": dF, "blocks": len(new_blocks),
        "ffn_hidden_per_block": {str(i): U_new[i]
                                 for i in range(len(host.blocks))},
        "ffn_hidden_joining": UJ,
        "path_synapses_bit_exact": n_checked,
        "backbone_genes": int(backbone.sum()),
        "stitching_genes": int((~backbone).sum()),
        "host_coord_map": {str(k): v for k, v in h_map.items()},
        "donor_coord_map": {str(k): v for k, v in d_map.items()},
    }
    return net, backbone, info


# ================================================================
# Deepen (verbatim construction from
# frankenstein_stepB_deepen_2026-11-26.py): append ONE
# narrow fused block at the back (after the last block,
# before the decoder), all tiny-random filler from one
# rng stream, layer-norm gains 1 + N(0, 0.01), draw
# order: lnA_gain, lnA_bias, lnF_gain, lnF_bias, W_q,
# W_k, W_v, W_o, b_q, b_k, b_v, b_o, b_2, W_1, b_1, W_2.
# The trunk is copied bit-exact.
# ================================================================
def deepen_net(net: Net, ffn_width: int, fill_seed: int):
    rng_fill = np.random.default_rng(fill_seed)

    def fill(shape):
        return rng_fill.normal(0.0, SIGMA_FILL, size=shape)

    d = net.d
    NB = f"b{len(net.blocks)}"
    W = {k: np.asarray(v, dtype=float).copy()
         for k, v in net.W.items()}
    W[f"{NB}.lnA_gain"] = 1.0 + rng_fill.normal(
        0.0, LN_GAIN_SIGMA, d)
    W[f"{NB}.lnA_bias"] = fill((d,))
    W[f"{NB}.lnF_gain"] = 1.0 + rng_fill.normal(
        0.0, LN_GAIN_SIGMA, d)
    W[f"{NB}.lnF_bias"] = fill((d,))
    for nm in ("W_q", "W_k", "W_v", "W_o"):
        W[f"{NB}.{nm}"] = fill((d, d))
    for nm in ("b_q", "b_k", "b_v", "b_o", "b_2"):
        W[f"{NB}.{nm}"] = fill((d,))
    W[f"{NB}.W_1"] = fill((d, ffn_width))
    W[f"{NB}.b_1"] = fill((ffn_width,))
    W[f"{NB}.W_2"] = fill((ffn_width, d))
    styles = dict(net.styles)
    styles[NB] = "fused"
    heads = dict(net.heads)
    heads[NB] = 1
    out = Net(W, net.blocks + [NB], styles, heads, d,
              net.attn_scale, net.ln_eps)
    for k in net.order:
        assert np.array_equal(out.W[k], net.W[k]), \
            f"trunk array {k} not bit-identical after deepen"
    return out, NB


# ================================================================
# Widen (construction from frankenstein_stepC_widen003_
# 2026-11-26.py and frankenstein_rep_stepC_widen003_
# 2026-11-27.py, generalized to the driver's Net form):
# Net2WiderNet split-widen of the WHOLE network, the
# sponsor's frozen default widen rule.
#   - Stream d -> 2d by duplication: new coords are
#     copies of the originals. Uniform rule: on an
#     OUTPUT side of the widened stream the axis is
#     duplicated; on an INPUT side the axis is split
#     evenly (each copy carries half), because the
#     duplicated input coordinates sum into every
#     consumer. A (d, d) matrix therefore maps to a
#     (2d, 2d) matrix whose four copy blocks are M/2.
#   - attn_scale HALVED: both dot products (attention
#     scores and decoder logits) run over the duplicated
#     stream and would otherwise double exactly.
#   - ln_eps UNCHANGED: duplication preserves the layer
#     norm mean and variance over the stream exactly.
#     LN gains/biases are duplicated.
#   - Heads x2 in every block, Net2WiderNet split:
#     * a multi block's head is split into two daughters
#       (the base keeps the head's index, the copy is
#       appended after all base heads): Q/K/V copied
#       exactly (via the uniform matrix rule), W_o and
#       b_o halved in both daughters, because head
#       outputs are summed.
#     * a fused block's single projection set is first
#       decomposed into 2 heads (W_o, b_o halved between
#       them) and each is split again, giving 4 heads,
#       each carrying W_o / 4 and b_o / 4 - verbatim
#       from the proven scripts.
#   - FFN x2 per block (holding the stream ratio): each
#     unit's W_1 column and b_1 copied exactly into the
#     appended copy unit; the unit's outgoing W_2 row
#     halved in both copies (the stream input-split also
#     applies to W_1's rows and W_2's columns).
#   - Decoder widened with the stream (W_qd by the
#     uniform rule; b_qd and v_start duplicated). Entry
#     projection columns duplicated, b_in duplicated.
# NOISE AMENDMENT (sponsor, 2026-11-26; rerun at 0.003,
# 2026-11-26 22:12 EST): after splitting, Gaussian noise
# is added to the NEW copies' synapse weights only -
# every entry belonging to a new copy (the mask built
# below, the union rule of the proven scripts: an entry
# is a new copy's iff any widened axis it touches is a
# copy index, or it belongs to a copy unit or copy
# head). The original copies stay exact. The noise is
# drawn per key in gene order from one rng stream.
# Returns (clean Net, noisy Net, flat mask over the
# widened gene order [True = new copy], info). The
# caller runs the PARITY GATE on the clean Net: it must
# reproduce the subject exactly (identical tours, gap
# unchanged) before the noisy genome counts.
# ================================================================
def widen_net(net: Net, noise_std: float, noise_seed: int):
    d0 = net.d
    d1 = 2 * d0

    def dup(v):
        return np.concatenate([v, v])

    def split_mat(M):
        out = np.zeros((d1, d1))
        for ci in (0, d0):
            for cj in (0, d0):
                out[ci:ci + d0, cj:cj + d0] = M / 2.0
        return out

    def vec_mask(n0):
        m = np.zeros(2 * n0, dtype=bool)
        m[n0:] = True
        return m

    def mat_mask():
        m = np.zeros((d1, d1), dtype=bool)
        m[d0:, :] = True
        m[:, d0:] = True
        return m

    W, MK = {}, {}

    # ---- embed: output side is the stream (duplicated);
    # input side is the raw input (not widened)
    W["embed.W_in"] = np.concatenate(
        [net.W["embed.W_in"], net.W["embed.W_in"]], axis=1)
    m = np.zeros(W["embed.W_in"].shape, dtype=bool)
    m[:, d0:] = True
    MK["embed.W_in"] = m
    W["embed.b_in"] = dup(net.W["embed.b_in"])
    MK["embed.b_in"] = vec_mask(d0)

    styles, heads_map = {}, {}
    for blk in net.blocks:
        U0 = net.ffn_width(blk)
        U1 = 2 * U0
        for nm in ("lnA_gain", "lnA_bias", "lnF_gain",
                   "lnF_bias"):
            W[f"{blk}.{nm}"] = dup(net.W[f"{blk}.{nm}"])
            MK[f"{blk}.{nm}"] = vec_mask(d0)
        # attention sources: (per-name arrays, daughters,
        # W_o divisor, base-daughter indices)
        if net.styles[blk] == "fused":
            srcs = [{nm: net.W[f"{blk}.{nm}"]
                     for nm in FUSED_ATTN}]
            # decompose the fused set into 2 heads, then
            # split each: daughters 0..3, divisor 4; the
            # decomposition heads 0 and 1 are the base
            # heads for the mask, 2 and 3 are copies
            plans = [(0, (0, 1, 2, 3), 4.0, (0, 1))]
        else:
            H0 = net.heads[blk]
            srcs = [{nm: net.W[f"{blk}.h{h}.{nm}"]
                     for nm in FUSED_ATTN}
                    for h in range(H0)]
            plans = [(si, (si, si + H0), 2.0, (si,))
                     for si in range(H0)]
        H1 = 0
        for src, (si, daughters, div, base) in zip(srcs,
                                                  plans):
            for h in daughters:
                p = f"{blk}.h{h}"
                for nm in ("W_q", "W_k", "W_v"):
                    W[f"{p}.{nm}"] = split_mat(src[nm])
                    MK[f"{p}.{nm}"] = mat_mask()
                W[f"{p}.W_o"] = split_mat(src["W_o"]) / div
                MK[f"{p}.W_o"] = mat_mask()
                for nm in ("b_q", "b_k", "b_v"):
                    W[f"{p}.{nm}"] = dup(src[nm])
                    MK[f"{p}.{nm}"] = vec_mask(d0)
                W[f"{p}.b_o"] = dup(src["b_o"]) / div
                MK[f"{p}.b_o"] = vec_mask(d0)
                if h not in base:
                    for nm in FUSED_ATTN:
                        MK[f"{p}.{nm}"][:] = True
                H1 = max(H1, h + 1)
        styles[blk] = "multi"
        heads_map[blk] = H1
        # FFN split: W_1 (d, U): input rows split, unit
        # columns duplicated -> every copy block W_1 / 2
        W1 = np.zeros((d1, U1))
        for ci in (0, d0):
            for cu in (0, U0):
                W1[ci:ci + d0, cu:cu + U0] = \
                    net.W[f"{blk}.W_1"] / 2.0
        W[f"{blk}.W_1"] = W1
        m = np.zeros((d1, U1), dtype=bool)
        m[d0:, :] = True
        m[:, U0:] = True
        MK[f"{blk}.W_1"] = m
        W[f"{blk}.b_1"] = dup(net.W[f"{blk}.b_1"])
        MK[f"{blk}.b_1"] = vec_mask(U0)
        # W_2 (U, d): unit rows halved in both copies,
        # stream columns under the uniform rule -> every
        # copy block W_2 / 2
        W2 = np.zeros((U1, d1))
        for cu in (0, U0):
            for cj in (0, d0):
                W2[cu:cu + U0, cj:cj + d0] = \
                    net.W[f"{blk}.W_2"] / 2.0
        W[f"{blk}.W_2"] = W2
        m = np.zeros((U1, d1), dtype=bool)
        m[U0:, :] = True
        m[:, d0:] = True
        MK[f"{blk}.W_2"] = m
        W[f"{blk}.b_2"] = dup(net.W[f"{blk}.b_2"])
        MK[f"{blk}.b_2"] = vec_mask(d0)

    # ---- decoder over the widened stream
    W["head.W_qd"] = split_mat(net.W["head.W_qd"])
    MK["head.W_qd"] = mat_mask()
    W["head.b_qd"] = dup(net.W["head.b_qd"])
    MK["head.b_qd"] = vec_mask(d0)
    W["head.v_start"] = dup(net.W["head.v_start"])
    MK["head.v_start"] = vec_mask(d0)

    clean = Net(W, net.blocks, styles, heads_map, d1,
                net.attn_scale / 2.0, net.ln_eps,
                {"widened_from_d": d0})
    order = clean.order
    assert set(MK.keys()) == set(order), \
        "widen mask keys do not match the widened gene order"
    flat_mask = np.concatenate(
        [MK[k].ravel() for k in order])

    # noise on the new copies only, drawn per key in
    # gene order from one rng stream (proven convention)
    rng_noise = np.random.default_rng(noise_seed)
    WN = {}
    for k in order:
        noise = rng_noise.normal(0.0, noise_std,
                                 size=W[k].shape)
        WN[k] = W[k] + np.where(MK[k], noise, 0.0)
    for k in order:
        assert np.array_equal(WN[k][~MK[k]], W[k][~MK[k]]), \
            f"noise touched an original copy's entry in {k}"
    noisy = Net(WN, net.blocks, styles, heads_map, d1,
                net.attn_scale / 2.0, net.ln_eps,
                {"widened_from_d": d0,
                 "widen_noise_std": float(noise_std),
                 "widen_noise_seed": int(noise_seed)})
    info = {"d_before": d0, "d_after": d1,
            "heads_before": {b: (1 if net.styles[b]
                                 == "fused"
                                 else net.heads[b])
                             for b in net.blocks},
            "heads_after": dict(heads_map),
            "ffn_before": {b: net.ffn_width(b)
                           for b in net.blocks},
            "ffn_after": {b: 2 * net.ffn_width(b)
                          for b in net.blocks},
            "genes_before": net.n_genes,
            "genes_after": clean.n_genes,
            "genes_new_copies": int(flat_mask.sum()),
            "genes_original_copies":
                int(flat_mask.size - flat_mask.sum()),
            "attn_scale_before": net.attn_scale,
            "attn_scale_after": net.attn_scale / 2.0,
            "noise_std": float(noise_std),
            "noise_seed": int(noise_seed)}
    return clean, noisy, flat_mask, info


# ================================================================
# Backend stitch (verbatim construction from
# frankenstein_stepC_stitch_2026-11-26.py): the stack
# run's final population shares one frozen trunk and
# differs only in the backend block. Score every member
# on the fixed set; champion #1 must be the stage's
# vaulted best (sanity gate); champion #2 is the best
# distinct genome. The widened backend block keeps the
# trunk once (bit-exact from #1): layer norms and the
# FFN output bias from #1, both champions' heads side by
# side, both FFN unit sets side by side. No filler, no
# rescaling.
# ================================================================
def backend_stitch(template: Net, pop, best_val_genome,
                   best_val_gap, backend_blk, X_val, opt):
    order, shapes = template.order, template.shapes
    backend_keys = {k for k in order
                    if k.startswith(f"{backend_blk}.")}
    trunk_keys = [k for k in order if k not in backend_keys]
    n_trunk = int(sum(int(np.prod(shapes[k]))
                      for k in trunk_keys))

    def gap_of_vec(vec):
        W = unpack(vec, order, shapes)
        probe = Net(W, template.blocks, template.styles,
                    template.heads, template.d, template.attn_scale,
                    template.ln_eps)
        g, _, _ = gap_net(probe, X_val, opt)
        return g

    pop = np.asarray(pop, dtype=float)
    appended_best = False
    if not any(np.array_equal(row, best_val_genome)
               for row in pop):
        # the vaulted best was bred out of the final
        # population (possible when it was the founder);
        # it is a legitimate member - append it so the
        # proven sanity check below can find it
        pop = np.vstack([pop, best_val_genome[None, :]])
        appended_best = True
    trunk0 = None
    for row in pop:
        W = unpack(row, order, shapes)
        flat_trunk = pack({k: W[k] for k in trunk_keys},
                          trunk_keys)
        if trunk0 is None:
            trunk0 = flat_trunk
        assert np.array_equal(flat_trunk, trunk0), \
            "final population does not share one trunk"
    gaps = np.array([gap_of_vec(row) for row in pop])
    order_idx = np.argsort(gaps)
    i1 = int(order_idx[0])
    champ1, gap1 = pop[i1], float(gaps[i1])
    if not np.array_equal(champ1, best_val_genome) \
            or abs(gap1 - best_val_gap) > 1e-9:
        raise RuntimeError(
            f"backend stitch sanity FAILED: best population "
            f"genome scores {gap1*100:.6f}%, checkpoint best "
            f"{best_val_gap*100:.6f}%, bit-identical: "
            f"{np.array_equal(champ1, best_val_genome)}")
    i2 = None
    for j in order_idx[1:]:
        if not np.array_equal(pop[int(j)], champ1):
            i2 = int(j)
            break
    assert i2 is not None, "no distinct runner-up"
    champ2, gap2 = pop[i2], float(gaps[i2])
    W1 = unpack(champ1, order, shapes)
    W2 = unpack(champ2, order, shapes)
    d_b = champ1 - champ2
    backend_slice = np.zeros(len(champ1), dtype=bool)
    off = 0
    for k in order:
        n = int(np.prod(shapes[k]))
        if k in backend_keys:
            backend_slice[off:off + n] = True
        off += n
    dist = {
        "genes": int(backend_slice.sum()),
        "n_differing": int((d_b[backend_slice] != 0).sum()),
        "max_abs_diff": float(np.abs(d_b[backend_slice]).max()),
        "rms_diff": float(np.sqrt(
            (d_b[backend_slice] ** 2).mean())),
    }
    assert template.styles[backend_blk] == "fused" \
        and template.heads[backend_blk] == 1, \
        "backend stitch expects a fused 1-head backend block"
    blk = backend_blk
    WC = {k: W1[k].copy() for k in trunk_keys}
    WC[f"{blk}.lnA_gain"] = W1[f"{blk}.lnA_gain"].copy()
    WC[f"{blk}.lnA_bias"] = W1[f"{blk}.lnA_bias"].copy()
    for h, Wsrc in ((0, W1), (1, W2)):
        for nm in FUSED_ATTN:
            WC[f"{blk}.h{h}.{nm}"] = Wsrc[f"{blk}.{nm}"].copy()
    WC[f"{blk}.lnF_gain"] = W1[f"{blk}.lnF_gain"].copy()
    WC[f"{blk}.lnF_bias"] = W1[f"{blk}.lnF_bias"].copy()
    WC[f"{blk}.W_1"] = np.concatenate(
        [W1[f"{blk}.W_1"], W2[f"{blk}.W_1"]], axis=1)
    WC[f"{blk}.b_1"] = np.concatenate(
        [W1[f"{blk}.b_1"], W2[f"{blk}.b_1"]])
    WC[f"{blk}.W_2"] = np.concatenate(
        [W1[f"{blk}.W_2"], W2[f"{blk}.W_2"]], axis=0)
    WC[f"{blk}.b_2"] = W1[f"{blk}.b_2"].copy()
    styles = dict(template.styles)
    styles[blk] = "multi"
    heads = dict(template.heads)
    heads[blk] = 2
    net = Net(WC, template.blocks, styles, heads, template.d,
              template.attn_scale, template.ln_eps)
    for k in trunk_keys:
        assert np.array_equal(net.W[k], W1[k]), \
            f"trunk array {k} not bit-identical in backend stitch"
    info = {"champion_1_gap_pct": round(gap1 * 100, 4),
            "best_appended_to_population": appended_best,
            "champion_2_gap_pct": round(gap2 * 100, 4),
            "champion_1_pop_index": i1,
            "champion_2_pop_index": i2,
            "backend_distance": dist,
            "trunk_genes": n_trunk,
            "widened_backend_genes": int(sum(
                np.asarray(net.W[k]).size for k in net.order
                if k.startswith(f"{blk}.")))}
    return net, info


# ================================================================
# Masked evolution - THE proven loop, parameterized. This
# single loop is, verbatim in structure, the machinery of
# frankenstein_stepA_parents_evolve (mask = all genes),
# frankenstein_stepA2_evolve / stepB_evolve /
# stepC_evolve (mask = the stage's mutable set, backbone
# asserted bit-identical every generation) and
# frankenstein_widened_evolve_scaled (mask = all genes,
# scaled rate, population founded by one mutation
# application). Conventions:
#   - training pool of `train_pool` problems drawn once
#     from the stage seed; rotating batches of
#     `train_batch`; fitness = mean gap vs Held-Karp
#     optima on the batch
#   - validation of the best-of-generation on the FIXED
#     set every `val_every` generations and at the last;
#     the best-on-validation genome is vaulted
#   - elites top 2 verbatim; tournament 3; no crossover;
#     mutation: per-gene p on the mutable set, Gaussian
#     sigma; rng = default_rng(seed + 1000)
#   - per-generation atomic checkpoint (pop, best, rng
#     state, history) + status.json; resume refuses
#     nothing but continues from the checkpoint
#   - SANITY GATE: the founder's validation gap must
#     reproduce the recorded gap (job.founder_gap_pct)
#     before the run counts
# Driver-side difference from the one-off scripts,
# recorded: the best tracker is INITIALIZED with the
# founder's own validation gap (the one-off scripts
# initialized best to infinity with the founder genome,
# so a stage could in principle vault a genome worse
# than its founder; initializing at the founder makes
# "a stage never returns worse than its founder" a
# guarantee, and matches the replication run's status
# convention of best-at-generation -1 = founder).
# ================================================================
def run_evolution(job):
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    founder_path = Path(job["founder"])
    net = load_net(founder_path)
    order, shapes = net.order, net.shapes
    founder = net.flat()
    n_genes = founder.size

    domain, X_val, OPT_VAL, _eps = build_domain()
    n_cities = int(X_val.shape[1])

    def mean_gap(vec, X, opt):
        W = unpack(vec, order, shapes)
        probe = Net(W, net.blocks, net.styles, net.heads,
                    net.d, net.attn_scale, net.ln_eps)
        Y = forward_net(probe, X)
        picks = decode_net(probe, Y)
        lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
        return float((lens / opt - 1.0).mean())

    # ---- sanity gate
    founder_val = mean_gap(founder, X_val, OPT_VAL)
    expected = job.get("founder_gap_pct")
    log(f"sanity gate: founder validation gap "
        f"{founder_val*100:.6f}%"
        + (f" vs recorded {expected:.4f}%" if expected
           is not None else ""))
    if expected is not None and \
            abs(founder_val * 100 - expected) > 5e-5 + 1e-9:
        log("SANITY GATE FAILED; no evolution run.")
        atomic_json(out / "FAILED.json", {
            "stage": job["stage"], "reason": "sanity gate",
            "founder_gap_pct": founder_val * 100,
            "expected_gap_pct": expected})
        return 3

    # ---- mutable set
    if job.get("mask_file"):
        mz = np.load(job["mask_file"])
        mutable = np.asarray(mz["mutable"], dtype=bool)
        assert mutable.shape == (n_genes,)
    else:
        mutable = np.ones(n_genes, dtype=bool)
    mut_idx = np.flatnonzero(mutable)
    backbone_idx = np.flatnonzero(~mutable)
    if job.get("mut_p") is not None:
        mut_p = float(job["mut_p"])
    else:
        mut_p = float(job["target_nudges"]) / n_genes
    mut_sigma = float(job.get("mut_sigma", 0.15))
    pop_n = int(job["pop"])
    gens = int(job["gens"])
    val_every = int(job.get("val_every", 10))
    train_pool = int(job.get("train_pool", 2000))
    train_batch = int(job.get("train_batch", 200))
    seed = int(job["seed"])
    log(f"evolution '{job['stage']}': {n_genes} genes, "
        f"{len(mut_idx)} mutable, {len(backbone_idx)} frozen; "
        f"p = {mut_p:.6f} (expected "
        f"{mut_p * len(mut_idx):.1f} nudges/child on the "
        f"mutable set), sigma {mut_sigma}; pop {pop_n}, "
        f"{gens} generations, seed {seed}")

    # ---- training pool (drawn once from the stage seed)
    rng_pool = np.random.default_rng(seed)
    X_pool = rng_pool.random((train_pool, n_cities, 2))
    OPT_POOL = np.array([held_karp(X_pool[i])
                         for i in range(train_pool)])
    log(f"training pool: {train_pool} problems from seed "
        f"{seed}")

    CKPT = out / "checkpoint.npz"
    STATUS = out / "status.json"
    rng = np.random.default_rng(seed + 1000)
    history = []
    best_val_gap = founder_val
    best_val_genome = founder.copy()
    best_val_gen = -1
    start_gen = 0

    if CKPT.exists():
        ck = np.load(CKPT, allow_pickle=True)
        pop = np.asarray(ck["pop"], dtype=float)
        start_gen = int(ck["next_gen"])
        best_val_gap = float(ck["best_val_gap"])
        best_val_genome = np.asarray(ck["best_val_genome"],
                                     dtype=float)
        best_val_gen = int(ck["best_val_gen"])
        history = json.loads(str(ck["history"]))
        rng.bit_generator.state = json.loads(str(ck["rng_state"]))
        log(f"RESUMED from checkpoint at generation "
            f"{start_gen}; best validation gap so far "
            f"{best_val_gap*100:.4f}% (generation "
            f"{best_val_gen})")
    else:
        pop = np.tile(founder, (pop_n, 1))
        if job.get("pop_init") == "mutate_once":
            for i in range(1, pop_n):
                m = rng.random(len(mut_idx)) < mut_p
                if m.any():
                    pop[i, mut_idx[m]] += rng.normal(
                        0.0, mut_sigma, size=int(m.sum()))
        log(f"fresh start: population {pop_n} founded on "
            f"the stage founder (init "
            f"{job.get('pop_init', 'tile')})")

    founder_backbone = founder[backbone_idx].copy() \
        if len(backbone_idx) else None

    def checkpoint(next_gen):
        atomic_savez(
            CKPT, pop=pop, next_gen=next_gen,
            best_val_gap=best_val_gap,
            best_val_genome=best_val_genome,
            best_val_gen=best_val_gen,
            history=np.array(json.dumps(history)),
            rng_state=np.array(json.dumps(
                rng.bit_generator.state)))
        atomic_json(STATUS, {
            "stage": job["stage"], "next_gen": next_gen,
            "gens_budget": gens,
            "best_val_gap_pct": round(best_val_gap * 100, 4),
            "best_val_gen": best_val_gen,
            "last": history[-1] if history else None})

    for gen in range(start_gen, gens):
        idx = rng.choice(train_pool, size=train_batch,
                         replace=False)
        Xb, ob = X_pool[idx], OPT_POOL[idx]
        fit = np.array([mean_gap(pop[i], Xb, ob)
                        for i in range(pop_n)])
        bi = int(np.argmin(fit))
        rec = {"gen": gen,
               "train_best_gap_pct": round(
                   float(fit[bi]) * 100, 4),
               "train_mean_gap_pct": round(
                   float(fit.mean()) * 100, 4)}
        if gen % val_every == 0 or gen == gens - 1:
            vg = mean_gap(pop[bi], X_val, OPT_VAL)
            rec["val_gap_pct"] = round(vg * 100, 4)
            if vg < best_val_gap:
                best_val_gap = vg
                best_val_genome = pop[bi].copy()
                best_val_gen = gen
            log(f"gen {gen}: train best "
                f"{fit[bi]*100:.2f}%, mean "
                f"{fit.mean()*100:.2f}%; VALIDATION "
                f"best-of-gen {vg*100:.2f}% (best so far "
                f"{best_val_gap*100:.2f}% at gen "
                f"{best_val_gen})")
        else:
            log(f"gen {gen}: train best {fit[bi]*100:.2f}%, "
                f"mean {fit.mean()*100:.2f}%")
        history.append(rec)
        if founder_backbone is not None:
            assert bool((pop[:, backbone_idx]
                         == founder_backbone).all()), \
                f"backbone moved at generation {gen}"
        checkpoint(gen + 1)
        if gen == gens - 1:
            break
        order_fit = np.argsort(fit)
        new_pop = np.empty_like(pop)
        for e in range(2):
            new_pop[e] = pop[order_fit[e]]
        for slot in range(2, pop_n):
            cand = rng.choice(pop_n, size=3, replace=False)
            parent = pop[cand[int(np.argmin(fit[cand]))]]
            child = parent.copy()
            m = rng.random(len(mut_idx)) < mut_p
            if m.any():
                child[mut_idx[m]] += rng.normal(
                    0.0, mut_sigma, size=int(m.sum()))
            new_pop[slot] = child
        pop = new_pop

    # ---- vault the best-validation genome + COMPLETE
    final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
    assert abs(final_val - best_val_gap) < 1e-12
    n_mut_changed = int((best_val_genome[mut_idx]
                         != founder[mut_idx]).sum())
    backbone_ok = True
    if founder_backbone is not None:
        backbone_ok = bool((best_val_genome[backbone_idx]
                            == founder_backbone).all())
        assert backbone_ok
    champ_net = Net(unpack(best_val_genome, order, shapes),
                    net.blocks, net.styles, net.heads, net.d,
                    net.attn_scale, net.ln_eps)
    champ_name = job.get("champion_name", "champion.npz")
    champ_net.save(out / champ_name, "bert_lifecycle_champion",
                   {"stage": job["stage"],
                    "founder": str(founder_path),
                    "best_validation_gap_pct":
                        round(best_val_gap * 100, 4),
                    "best_validation_generation": best_val_gen})
    walk = [{"gen": r["gen"], "val_gap_pct": r["val_gap_pct"]}
            for r in history if "val_gap_pct" in r]
    result = {
        "stage": job["stage"],
        "founder": str(founder_path),
        "founder_gap_pct": round(founder_val * 100, 4),
        "genes_total": int(n_genes),
        "genes_mutable": int(len(mut_idx)),
        "genes_frozen": int(len(backbone_idx)),
        "mut_p": mut_p,
        "expected_nudges_per_child_mutable":
            round(mut_p * len(mut_idx), 2),
        "generations": gens, "seed": seed,
        "validation_walk": walk,
        "final_gap_pct": round(best_val_gap * 100, 4),
        "final_gap_generation": best_val_gen,
        "mutable_genes_changed_vs_founder": n_mut_changed,
        "backbone_bit_identical": backbone_ok,
        "champion": champ_name,
        "history": history,
    }
    atomic_json(out / "result.json", result)
    atomic_json(out / "COMPLETE", result)
    log(f"DONE '{job['stage']}': best validation gap "
        f"{best_val_gap*100:.4f}% at generation "
        f"{best_val_gen} (founder "
        f"{founder_val*100:.4f}%); champion vaulted")
    return 0


def launch_detached(argv, out_dir: Path, complete_marker: Path,
                    log_name="run.log", poll_seconds=2.0):
    """Launch a stage fully detached (standing sponsor
    rule: subprocess.Popen with start_new_session=True)
    and wait for its COMPLETE marker. If the process dies
    without the marker, raise with the log tail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / log_name
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(
            argv, stdout=lf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            cwd=str(HERE))
    log(f"launched detached (pid {proc.pid}): "
        f"{' '.join(str(a) for a in argv[1:])}")
    while True:
        if complete_marker.exists():
            text = complete_marker.read_text()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"marker_text": text}
        rc = proc.poll()
        if rc is not None:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"stage process exited rc={rc} without "
                f"COMPLETE marker {complete_marker}; log "
                f"tail:\n{tail}")
        time.sleep(poll_seconds)


# ================================================================
# The lifecycle driver (state machine)
# ================================================================
class LifecycleDriver:
    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "lifecycle_state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {
                "run_name": cfg.get("run_name", "lifecycle_run"),
                "method": "Frankenstein lifecycle, frozen "
                          "2026-11-27 (FRANKENSTEIN_METHOD.md)",
                "cycle": 0,
                "stage": "init",
                "champions": [],
                "standing_best_gap_pct": None,
                "smallest_at_best": None,
                "standing_champion": None,
                "lines": {},
                "termination": {"stopped": False,
                                "reason": None},
            }
            self.save_state()
        (_dom, self.X_val, self.OPT, self.eps) = build_domain()

    # ---------- state / champion registry (step 13)
    def save_state(self):
        atomic_json(self.state_path, self.state)

    def set_stage(self, stage):
        self.state["stage"] = stage
        self.save_state()
        log(f"stage: {stage}")

    def register_champion(self, stage, cycle, line, file,
                          genes, gap_pct):
        entry = {"stage": stage, "cycle": cycle, "line": line,
                 "file": str(file), "genes": int(genes),
                 "gap_pct": round(float(gap_pct), 4)}
        for old in self.state["champions"]:
            if (old["stage"], old["cycle"], old["line"],
                    old["file"]) == (stage, cycle, line,
                                     str(file)):
                return old  # resume: already registered
        self.state["champions"].append(entry)
        tol = float(self.cfg["termination"]
                    ["best_tolerance_points"])
        best = min(c["gap_pct"] for c in self.state["champions"])
        self.state["standing_best_gap_pct"] = best
        at_best = [c for c in self.state["champions"]
                   if c["gap_pct"] <= best + tol]
        self.state["smallest_at_best"] = min(
            at_best, key=lambda c: c["genes"])
        self.save_state()
        log(f"champion registered: {stage} (cycle {cycle}, "
            f"{line}) gap {entry['gap_pct']:.4f}%, "
            f"{genes} genes; standing best {best:.4f}%, "
            f"smallest at best: "
            f"{self.state['smallest_at_best']['genes']} genes "
            f"({self.state['smallest_at_best']['stage']})")
        return entry

    # ---------- seeds
    def seed_family(self, cycle, line_index):
        return int(self.cfg["seed_base"]) + cycle * 100000 \
            + line_index * 10000

    # ---------- founding (step 1): the engine runner,
    # launched detached, driven by a generated config
    def run_founding(self, line_dir: Path, seed0: int,
                     gens: int):
        fdir = line_dir / "founding"
        if (fdir / "COMPLETE.txt").exists():
            log("founding already complete; skipping")
            return fdir
        template = json.loads((HERE / "bert_engine_config_"
                               "deep50_stepA.json").read_text())
        fc = self.cfg["founding"]
        template.update({
            "run_name": f"{self.cfg.get('run_name', 'lifecycle')}"
                        f"_{line_dir.name}_founding",
            "run_seed": seed0, "draw_seed": seed0 + 1,
            "check_seed": seed0 + 2, "audit_seed": seed0 + 3,
            "phase_a_seed": seed0 + 4,
            "phase_a_gens": gens // 2,
            "phase_b_gens": gens - gens // 2,
            "d": int(fc["d"]),
            "ffn_hidden": int(fc["ffn_hidden"]),
            "initial_blocks": int(fc["initial_blocks"]),
            "initial_heads": int(fc["initial_heads"]),
            "n_species": int(fc["n_species"]),
            "per_species": int(fc["per_species"]),
            "val_seed": VAL_SEED, "elim_seed": ELIM_SEED,
            "max_growth_rounds": 0,
            "cataclysm_enabled": False,
        })
        cfg_path = line_dir / "founding_config.json"
        atomic_json(cfg_path, template)
        launch_detached(
            [sys.executable, str(HERE / "run_bert_engine.py"),
             "--config", str(cfg_path), "--out-dir", str(fdir),
             "--fresh"],
            fdir, fdir / "COMPLETE.txt", "founding_run.log")
        return fdir

    def measure_founders(self, line_dir: Path):
        """Adapted from frankenstein_rep_measure_founders:
        score every surviving species' best genome on the
        fixed set; host = #1, donor = #2."""
        out_path = line_dir / "founders_measured.json"
        if out_path.exists():
            return json.loads(out_path.read_text())
        cfg = be.EngineConfig.load(
            line_dir / "founding_config.json")
        backend = get_backend(cfg.backend, cfg.device,
                              cfg.precision)
        arch = be.initial_arch(cfg, 2)
        layout = be.Layout(arch, tsp.head_spec(arch.d))
        domain = tsp.TSPDomain(cfg, backend)
        ck = np.load(line_dir / "founding" / "checkpoint.npz",
                     allow_pickle=True)
        species_ids = [int(s) for s in ck["species_ids"]]
        best_genomes = np.asarray(ck["best_genomes"],
                                  dtype=float)
        rows = []
        for sid, g in zip(species_ids, best_genomes):
            gap = float(domain.validation_detail(
                layout, g)["mean_gap"])
            rows.append({"species": sid, "gap": gap,
                         "gap_pct": round(gap * 100, 6),
                         "genome": g})
        rows.sort(key=lambda r: r["gap_pct"])
        assert len(rows) >= 2, \
            "fewer than two species survived the founding; " \
            "cannot stitch"
        out = {"species_champions_ranked": [
                   {"species": r["species"],
                    "gap_pct": r["gap_pct"]} for r in rows],
               "host_species": rows[0]["species"],
               "host_gap_pct": rows[0]["gap_pct"],
               "donor_species": rows[1]["species"],
               "donor_gap_pct": rows[1]["gap_pct"]}
        for tag, r in (("host", rows[0]), ("donor", rows[1])):
            net = plain_parent_net(r["genome"], layout,
                                   arch.attn_scale, arch.ln_eps)
            g_mine, picks_mine, _ = gap_net(
                net, self.X_val, self.OPT)
            lens_e, picks_e = domain.evaluate_lengths(
                layout, r["genome"][None, :], self.X_val)
            assert abs(g_mine - r["gap"]) < 1e-9 \
                and bool((picks_mine == picks_e[0]).all()), \
                f"parent parity failed for {tag}"
            net.save(line_dir / f"parent_{tag}_founder.npz",
                     "bert_lifecycle_plain_parent",
                     {"species": r["species"],
                      "founding_gap_pct": r["gap_pct"]})
            out[f"{tag}_file"] = f"parent_{tag}_founder.npz"
            out[f"{tag}_genes"] = net.n_genes
        atomic_json(out_path, out)
        log(f"founders measured: host species "
            f"{out['host_species']} at {out['host_gap_pct']}%, "
            f"donor species {out['donor_species']} at "
            f"{out['donor_gap_pct']}% (parity vs engine "
            f"passed for both)")
        return out

    # ---------- one evolution stage, launched detached
    def run_evolve_stage(self, line_dir: Path, stage: str,
                         founder_file: Path, **job_kw):
        out_dir = line_dir / stage
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "COMPLETE").exists():
            log(f"evolution stage {stage} already complete; "
                f"skipping")
            return json.loads(
                (out_dir / "result.json").read_text())
        job = {"stage": stage, "founder": str(founder_file),
               "out_dir": str(out_dir),
               "mut_sigma": float(
                   self.cfg["evolution"]["mut_sigma"]),
               "train_pool": int(
                   self.cfg["evolution"]["train_pool"]),
               "train_batch": int(
                   self.cfg["evolution"]["train_batch"]),
               "val_every": int(
                   self.cfg["evolution"]["val_every"])}
        job.update(job_kw)
        job_path = out_dir / "job.json"
        atomic_json(job_path, job)
        launch_detached(
            [sys.executable, str(HERE / "bert_lifecycle.py"),
             "_worker", "evolve", "--job", str(job_path)],
            out_dir, out_dir / "COMPLETE")
        return json.loads((out_dir / "result.json").read_text())

    def mask_file_for(self, line_dir: Path, stage: str,
                      net: Net, key_pred):
        order = net.order
        shapes = net.shapes
        mutable = np.zeros(net.n_genes, dtype=bool)
        off = 0
        for k in order:
            n = int(np.prod(shapes[k]))
            if key_pred(k):
                mutable[off:off + n] = True
            off += n
        path = line_dir / stage / "mask.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_savez(path, mutable=mutable)
        return path

    # ---------- one line: steps 1-6
    def build_line(self, line_name: str, cycle: int,
                   line_index: int):
        lines = self.state["lines"]
        if line_name in lines and \
                lines[line_name].get("champion"):
            log(f"line {line_name} already built; skipping")
            return lines[line_name]["champion"]
        line_dir = self.run_dir / "lines" / line_name
        line_dir.mkdir(parents=True, exist_ok=True)
        sb = self.seed_family(cycle, line_index)
        ev = self.cfg["evolution"]
        bd = self.cfg["budgets"]
        rec = lines.setdefault(line_name, {"stages": {}})
        st = rec["stages"]

        # step 1: founding
        self.set_stage(f"{line_name}:founding")
        self.run_founding(line_dir, sb,
                          int(self.cfg["founding"]["gens"]))
        founders = self.measure_founders(line_dir)
        st["founding"] = founders
        for tag in ("host", "donor"):
            self.register_champion(
                "founding", cycle, line_name,
                line_dir / founders[f"{tag}_file"],
                founders[f"{tag}_genes"],
                founders[f"{tag}_gap_pct"])

        # step 2a: parents mature (whole network free)
        matured = {}
        for tag, seed_off in (("host", 10), ("donor", 11)):
            stage = f"mature_{tag}"
            self.set_stage(f"{line_name}:{stage}")
            res = self.run_evolve_stage(
                line_dir, stage,
                line_dir / founders[f"{tag}_file"],
                gens=int(bd["parents_mature_gens"]),
                pop=int(bd["parents_pop"]), seed=sb + seed_off,
                mut_p=float(ev["parents_mut_p"]),
                pop_init="tile",
                founder_gap_pct=founders[f"{tag}_gap_pct"],
                champion_name=f"parent_{tag}_evolved.npz")
            st[stage] = res
            matured[tag] = line_dir / stage / \
                f"parent_{tag}_evolved.npz"
            self.register_champion(
                "mature_parent", cycle, line_name, matured[tag],
                res["genes_total"], res["final_gap_pct"])

        # step 2b: paths + stitch + seam evolution
        self.set_stage(f"{line_name}:stitch")
        stitch_dir = line_dir / "stitch"
        stitch_dir.mkdir(parents=True, exist_ok=True)
        genome_path = stitch_dir / "genome.npz"
        if not (stitch_dir / "stitch_result.json").exists():
            host_net = load_net(matured["host"])
            donor_net = load_net(matured["donor"])
            paths = {}
            for tag, pnet in (("host", host_net),
                              ("donor", donor_net)):
                edges = extract_path(compute_flows(
                    pnet, self.X_val, self.OPT))
                paths[tag] = path_summary(pnet, edges)
                log(f"{line_name} {tag} path: "
                    f"{paths[tag]['n_path_weight_synapses']} "
                    f"of {paths[tag]['n_weight_synapses_total']}"
                    f" weight synapses on the main path")
            net, backbone, info = stitch_nets(
                host_net, donor_net, paths["host"],
                paths["donor"], sb + 12)
            net.save(genome_path, "bert_lifecycle_stitch",
                     {"stage": "stitch", "line": line_name})
            birth, _, _ = gap_net(net, self.X_val, self.OPT)
            atomic_savez(stitch_dir / "backbone_mask.npz",
                         mutable=~backbone)
            sres = {"birth_gap_pct": round(birth * 100, 4),
                    "genes": net.n_genes, **info,
                    "host_path": {
                        k: v for k, v in paths["host"].items()
                        if k != "path_set"},
                    "donor_path": {
                        k: v for k, v in paths["donor"].items()
                        if k != "path_set"}}
            atomic_json(stitch_dir / "stitch_result.json",
                        sres)
            st["stitch"] = sres
        else:
            sres = json.loads((stitch_dir / "stitch_result.json")
                              .read_text())
            net = load_net(genome_path)
        self.register_champion("stitch_birth", cycle, line_name,
                               genome_path, sres["genes"],
                               sres["birth_gap_pct"])

        self.set_stage(f"{line_name}:seam_evolution")
        res = self.run_evolve_stage(
            line_dir, "seam_evolution", genome_path,
            gens=int(bd["seam_gens"]), pop=int(bd["seam_pop"]),
            seed=sb + 13, mut_p=float(ev["seam_mut_p"]),
            pop_init="tile",
            mask_file=str(stitch_dir / "backbone_mask.npz"),
            founder_gap_pct=sres["birth_gap_pct"],
            champion_name="seam_champion.npz")
        st["seam_evolution"] = res
        seam_champ = line_dir / "seam_evolution" / \
            "seam_champion.npz"
        self.register_champion("seam_evolution", cycle,
                               line_name, seam_champ,
                               res["genes_total"],
                               res["final_gap_pct"])

        # step 3: stack one skinny block, evolve only it
        self.set_stage(f"{line_name}:stack")
        stack_dir = line_dir / "stack"
        stack_dir.mkdir(parents=True, exist_ok=True)
        deep_path = stack_dir / "genome.npz"
        if not (stack_dir / "stack_result.json").exists():
            subject = load_net(seam_champ)
            deep_net, new_blk = deepen_net(
                subject, subject.d, sb + 20)
            deep_net.save(deep_path, "bert_lifecycle_stack",
                          {"stage": "stack", "line": line_name,
                           "new_block": new_blk})
            birth, _, _ = gap_net(deep_net, self.X_val,
                                  self.OPT)
            dres = {"birth_gap_pct": round(birth * 100, 4),
                    "genes": deep_net.n_genes,
                    "new_block": new_blk,
                    "new_block_ffn": subject.d}
            atomic_json(stack_dir / "stack_result.json", dres)
            st["stack"] = dres
        else:
            dres = json.loads((stack_dir / "stack_result.json")
                              .read_text())
            deep_net = load_net(deep_path)
        self.register_champion("stack_birth", cycle, line_name,
                               deep_path, dres["genes"],
                               dres["birth_gap_pct"])
        new_blk = dres["new_block"]

        self.set_stage(f"{line_name}:stack_evolution")
        mask_path = self.mask_file_for(
            line_dir, "stack_evolution", deep_net,
            lambda k: k.startswith(f"{new_blk}."))
        res = self.run_evolve_stage(
            line_dir, "stack_evolution", deep_path,
            gens=int(bd["stack_gens"]),
            pop=int(bd["stack_pop"]), seed=sb + 21,
            mut_p=float(ev["seam_mut_p"]), pop_init="tile",
            mask_file=str(mask_path),
            founder_gap_pct=dres["birth_gap_pct"],
            champion_name="stack_champion.npz")
        st["stack_evolution"] = res
        stack_champ = line_dir / "stack_evolution" / \
            "stack_champion.npz"
        self.register_champion("stack_evolution", cycle,
                               line_name, stack_champ,
                               res["genes_total"],
                               res["final_gap_pct"])

        # step 4: Frankenstein the stack (backend stitch),
        # then evolve the widened backend block
        self.set_stage(f"{line_name}:backend_stitch")
        bdir = line_dir / "backend_stitch"
        bdir.mkdir(parents=True, exist_ok=True)
        bpath = bdir / "genome.npz"
        if not (bdir / "backend_result.json").exists():
            ck = np.load(line_dir / "stack_evolution" /
                         "checkpoint.npz", allow_pickle=True)
            pop = np.asarray(ck["pop"], dtype=float)
            best_genome = np.asarray(ck["best_val_genome"],
                                     dtype=float)
            best_gap = float(ck["best_val_gap"])
            bnet, binfo = backend_stitch(
                deep_net, pop, best_genome, best_gap, new_blk,
                self.X_val, self.OPT)
            bnet.save(bpath, "bert_lifecycle_backend_stitch",
                      {"stage": "backend_stitch",
                       "line": line_name})
            birth, _, _ = gap_net(bnet, self.X_val, self.OPT)
            bres = {"birth_gap_pct": round(birth * 100, 4),
                    "genes": bnet.n_genes, **binfo}
            atomic_json(bdir / "backend_result.json", bres)
            st["backend_stitch"] = bres
        else:
            bres = json.loads((bdir / "backend_result.json")
                              .read_text())
            bnet = load_net(bpath)
        self.register_champion("backend_stitch_birth", cycle,
                               line_name, bpath, bres["genes"],
                               bres["birth_gap_pct"])

        self.set_stage(f"{line_name}:backend_seam_evolution")
        mask_path = self.mask_file_for(
            line_dir, "backend_seam_evolution", bnet,
            lambda k: k.startswith(f"{new_blk}."))
        res = self.run_evolve_stage(
            line_dir, "backend_seam_evolution", bpath,
            gens=int(bd["backend_seam_gens"]),
            pop=int(bd["seam_pop"]), seed=sb + 30,
            mut_p=float(ev["seam_mut_p"]), pop_init="tile",
            mask_file=str(mask_path),
            founder_gap_pct=bres["birth_gap_pct"],
            champion_name="backend_champion.npz")
        st["backend_seam_evolution"] = res
        backend_champ = line_dir / "backend_seam_evolution" / \
            "backend_champion.npz"
        self.register_champion("backend_seam_evolution", cycle,
                               line_name, backend_champ,
                               res["genes_total"],
                               res["final_gap_pct"])

        # step 5: widen the whole network (Net2WiderNet
        # split, tiny noise on the new copies only). The
        # no-noise widen must pass the exact preservation
        # gate before the noisy genome counts.
        self.set_stage(f"{line_name}:widen")
        wdir = line_dir / "widen"
        wdir.mkdir(parents=True, exist_ok=True)
        wpath = wdir / "genome.npz"
        if not (wdir / "widen_result.json").exists():
            subject = load_net(backend_champ)
            subj_gap, subj_picks, _ = gap_net(
                subject, self.X_val, self.OPT)
            founder_gap = res_gap(st, "backend_seam_evolution")
            assert abs(subj_gap * 100 - founder_gap) < 0.01, \
                f"widen subject gap {subj_gap * 100:.4f}% " \
                f"does not reproduce the backend champion " \
                f"{founder_gap}%"
            wcfg = self.cfg.get("widen", {})
            noise_std = float(wcfg.get("noise_std", 0.003))
            clean, noisy, flat_mask, winfo = widen_net(
                subject, noise_std, sb + 40)
            clean_gap, clean_picks, _ = gap_net(
                clean, self.X_val, self.OPT)
            gate = bool((clean_picks == subj_picks).all()) \
                and abs(clean_gap - subj_gap) <= 1e-9
            if not gate:
                raise RuntimeError(
                    "widen preservation gate FAILED: the "
                    "no-noise widen does not reproduce the "
                    f"subject (gap {clean_gap * 100:.6f}% vs "
                    f"{subj_gap * 100:.6f}%, tours identical: "
                    f"{bool((clean_picks == subj_picks).all())}"
                    "); no widened genome saved")
            noisy_gap, noisy_picks, _ = gap_net(
                noisy, self.X_val, self.OPT)
            tours_changed = int(
                (noisy_picks != subj_picks).any(axis=1).sum())
            noisy.save(wpath, "bert_lifecycle_widen",
                       {"stage": "widen", "line": line_name,
                        "subject_gap_pct":
                            round(subj_gap * 100, 6),
                        "parity_gate_no_noise_gap_pct":
                            round(clean_gap * 100, 6)})
            atomic_savez(wdir / "widen_mask.npz",
                         trainable=flat_mask)
            wres = {"birth_gap_pct": round(noisy_gap * 100, 4),
                    "genes": noisy.n_genes,
                    "subject_gap_pct": round(subj_gap * 100, 6),
                    "parity_gate": {
                        "passed": True,
                        "no_noise_gap_pct":
                            round(clean_gap * 100, 6),
                        "tours_identical_all_200": True},
                    "noise_drift_points":
                        round(noisy_gap * 100
                              - subj_gap * 100, 4),
                    "tours_changed_of_200": tours_changed,
                    **winfo}
            atomic_json(wdir / "widen_result.json", wres)
            st["widen"] = wres
            log(f"{line_name} widen: preservation gate "
                f"passed (no-noise gap "
                f"{clean_gap * 100:.6f}%, tours identical); "
                f"with-noise birth gap "
                f"{noisy_gap * 100:.4f}% (subject "
                f"{subj_gap * 100:.4f}%, drift "
                f"{noisy_gap * 100 - subj_gap * 100:+.4f} "
                f"points, tours changed on {tours_changed} "
                f"of 200); genes {winfo['genes_before']} -> "
                f"{winfo['genes_after']} "
                f"({winfo['genes_new_copies']} new copies)")
        else:
            wres = json.loads((wdir / "widen_result.json")
                              .read_text())
            st["widen"] = wres
        self.register_champion("widen_birth", cycle, line_name,
                               wpath, wres["genes"],
                               wres["birth_gap_pct"])

        # step 6: open evolution (scaled rate, nothing
        # frozen), founded on the widened genome
        self.set_stage(f"{line_name}:open_evolution")
        res = self.run_evolve_stage(
            line_dir, "open_evolution", wpath,
            gens=int(bd["open_gens"]), pop=int(bd["open_pop"]),
            seed=sb + 70, mut_p=None,
            target_nudges=float(
                ev["target_nudges_per_child"]),
            pop_init="mutate_once",
            founder_gap_pct=wres["birth_gap_pct"],
            champion_name="open_champion.npz")
        st["open_evolution"] = res
        open_champ = line_dir / "open_evolution" / \
            "open_champion.npz"
        entry = self.register_champion(
            "open_evolution", cycle, line_name, open_champ,
            res["genes_total"], res["final_gap_pct"])
        rec["champion"] = entry
        self.save_state()
        return entry


def res_gap(stages, name):
    return stages[name]["final_gap_pct"]


    # ---------- step 8: merge two champions (the step 2
    # stitch rule applied to two Frankenstein champions)
    def merge_champions(self, standing, new_line, cycle):
        mdir = self.run_dir / "merges" / f"cycle{cycle}"
        mdir.mkdir(parents=True, exist_ok=True)
        sb = int(self.cfg["seed_base"]) + cycle * 100000 + 9000
        ev, bd = self.cfg["evolution"], self.cfg["budgets"]
        genome_path = mdir / "genome.npz"
        if not (mdir / "merge_result.json").exists():
            host_net = load_net(Path(standing["file"]))
            donor_net = load_net(Path(new_line["file"]))
            paths = {}
            for tag, pnet in (("host", host_net),
                              ("donor", donor_net)):
                edges = extract_path(compute_flows(
                    pnet, self.X_val, self.OPT))
                paths[tag] = path_summary(pnet, edges)
            net, backbone, info = stitch_nets(
                host_net, donor_net, paths["host"],
                paths["donor"], sb + 40)
            net.save(genome_path, "bert_lifecycle_merge",
                     {"stage": "merge", "cycle": cycle})
            birth, _, _ = gap_net(net, self.X_val, self.OPT)
            atomic_savez(mdir / "backbone_mask.npz",
                         mutable=~backbone)
            mres = {"birth_gap_pct": round(birth * 100, 4),
                    "genes": net.n_genes, **info}
            atomic_json(mdir / "merge_result.json", mres)
        else:
            mres = json.loads((mdir / "merge_result.json")
                              .read_text())
        self.register_champion("merge_birth", cycle, "merge",
                               genome_path, mres["genes"],
                               mres["birth_gap_pct"])
        res = self.run_evolve_stage(
            mdir, "merge_seam_evolution", genome_path,
            gens=int(bd["merge_seam_gens"]),
            pop=int(bd["seam_pop"]), seed=sb + 41,
            mut_p=float(ev["seam_mut_p"]), pop_init="tile",
            mask_file=str(mdir / "backbone_mask.npz"),
            founder_gap_pct=mres["birth_gap_pct"],
            champion_name="merged_champion.npz")
        entry = self.register_champion(
            "merge_seam_evolution", cycle, "merge",
            mdir / "merge_seam_evolution" /
            "merged_champion.npz",
            res["genes_total"], res["final_gap_pct"])
        return entry

    # ---------- steps 9-10: growth rounds on the standing
    # champion (append a skinny appendix block, breed in
    # it, Frankenstein it - the backend stitch over the
    # common trunk - which is also how the grown stack is
    # Frankensteined into the standing network)
    def growth_rounds(self, standing, cycle):
        caps = self.cfg["caps"]
        ev, bd = self.cfg["evolution"], self.cfg["budgets"]
        frac = float(self.cfg["growth"]
                     ["appendix_width_fraction"])
        cur = standing
        rnd = 0
        while True:
            net = load_net(Path(cur["file"]))
            if caps.get("max_layers") is not None and \
                    len(net.blocks) >= int(caps["max_layers"]):
                log(f"growth: depth cap hit "
                    f"({len(net.blocks)} blocks >= "
                    f"{caps['max_layers']}); rounds stop")
                return cur, "cap:max_layers"
            if caps.get("max_heads") is not None and \
                    int(caps["max_heads"]) < 2:
                log("growth: heads cap forbids the appendix "
                    "stitch (needs 2 heads); rounds stop")
                return cur, "cap:max_heads"
            rnd += 1
            gdir = self.run_dir / "growth" / \
                f"cycle{cycle}_round{rnd}"
            gdir.mkdir(parents=True, exist_ok=True)
            sb = int(self.cfg["seed_base"]) + cycle * 100000 \
                + 8000 + rnd * 100
            backend_blk = net.blocks[-1]
            u_app = max(1, int(round(
                frac * net.ffn_width(backend_blk))))
            log(f"growth round {rnd}: appendix block with "
                f"FFN width {u_app} (fraction {frac} of "
                f"backend FFN {net.ffn_width(backend_blk)})")
            deep_net, new_blk = deepen_net(net, u_app, sb)
            deep_path = gdir / "genome.npz"
            deep_net.save(deep_path, "bert_lifecycle_appendix",
                          {"stage": "appendix", "cycle": cycle,
                           "round": rnd, "new_block": new_blk})
            birth, _, _ = gap_net(deep_net, self.X_val,
                                  self.OPT)
            self.register_champion(
                "appendix_birth", cycle, "growth", deep_path,
                deep_net.n_genes, round(birth * 100, 4))
            mask_path = self.mask_file_for(
                gdir, "appendix_evolution", deep_net,
                lambda k: k.startswith(f"{new_blk}."))
            res = self.run_evolve_stage(
                gdir, "appendix_evolution", deep_path,
                gens=int(bd["appendix_gens"]),
                pop=int(bd["appendix_pop"]), seed=sb + 1,
                mut_p=float(ev["seam_mut_p"]), pop_init="tile",
                mask_file=str(mask_path),
                founder_gap_pct=round(birth * 100, 4),
                champion_name="appendix_champion.npz")
            self.register_champion(
                "appendix_evolution", cycle, "growth",
                gdir / "appendix_evolution" /
                "appendix_champion.npz",
                res["genes_total"], res["final_gap_pct"])
            # Frankenstein the appendix: stitch its top two
            ck = np.load(gdir / "appendix_evolution" /
                         "checkpoint.npz", allow_pickle=True)
            bnet, binfo = backend_stitch(
                deep_net, np.asarray(ck["pop"], dtype=float),
                np.asarray(ck["best_val_genome"], dtype=float),
                float(ck["best_val_gap"]), new_blk,
                self.X_val, self.OPT)
            bpath = gdir / "appendix_stitched.npz"
            bnet.save(bpath, "bert_lifecycle_appendix_stitch",
                      {"stage": "appendix_stitch",
                       "cycle": cycle, "round": rnd})
            birth2, _, _ = gap_net(bnet, self.X_val, self.OPT)
            self.register_champion(
                "appendix_stitch_birth", cycle, "growth",
                bpath, bnet.n_genes, round(birth2 * 100, 4))
            mask_path = self.mask_file_for(
                gdir, "appendix_seam_evolution", bnet,
                lambda k: k.startswith(f"{new_blk}."))
            res = self.run_evolve_stage(
                gdir, "appendix_seam_evolution", bpath,
                gens=int(bd["appendix_seam_gens"]),
                pop=int(bd["seam_pop"]), seed=sb + 3,
                mut_p=float(ev["seam_mut_p"]), pop_init="tile",
                mask_file=str(mask_path),
                founder_gap_pct=round(birth2 * 100, 4),
                champion_name="round_champion.npz")
            entry = self.register_champion(
                "appendix_seam_evolution", cycle, "growth",
                gdir / "appendix_seam_evolution" /
                "round_champion.npz",
                res["genes_total"], res["final_gap_pct"])
            if entry["gap_pct"] < cur["gap_pct"]:
                log(f"growth round {rnd} improved the "
                    f"standing champion: "
                    f"{cur['gap_pct']:.4f}% -> "
                    f"{entry['gap_pct']:.4f}%")
                cur = entry
            else:
                log(f"growth round {rnd} brought no "
                    f"improvement ({entry['gap_pct']:.4f}% "
                    f"vs standing {cur['gap_pct']:.4f}%); "
                    f"rounds stop")
                return cur, "no_improvement"

    # ---------- step 11: free evolution
    def free_evolution(self, standing, cycle):
        fdir = self.run_dir / "free_evolution" / f"cycle{cycle}"
        fdir.mkdir(parents=True, exist_ok=True)
        sb = int(self.cfg["seed_base"]) + cycle * 100000 + 8500
        ev, bd = self.cfg["evolution"], self.cfg["budgets"]
        res = self.run_evolve_stage(
            fdir, "free_evolution", Path(standing["file"]),
            gens=int(bd["free_gens"]), pop=int(bd["free_pop"]),
            seed=sb, mut_p=None,
            target_nudges=float(
                ev["target_nudges_per_child"]),
            pop_init="mutate_once",
            founder_gap_pct=standing["gap_pct"],
            champion_name="free_champion.npz")
        return self.register_champion(
            "free_evolution", cycle, "free",
            fdir / "free_evolution" / "free_champion.npz",
            res["genes_total"], res["final_gap_pct"])

    # ---------- the cycle loop (steps 6-11)
    def run(self):
        caps = self.cfg["caps"]
        max_cycles = self.cfg["termination"].get("max_cycles")
        if self.state["standing_champion"] is None:
            self.set_stage("initial_line")
            champ = self.build_line("lineA", cycle=0,
                                    line_index=0)
            self.state["standing_champion"] = champ
            self.save_state()
        standing = self.state["standing_champion"]
        cycle = int(self.state["cycle"])
        while True:
            if self.state["termination"]["stopped"]:
                break
            cycle += 1
            if max_cycles is not None and cycle > int(
                    max_cycles):
                self.stop(f"max_cycles ({max_cycles}) reached")
                break
            net = load_net(Path(standing["file"]))
            if caps.get("max_width") is not None and \
                    2 * net.d > int(caps["max_width"]):
                self.stop(f"width cap: merging would take "
                          f"the stream to {2 * net.d} > "
                          f"{caps['max_width']}")
                break
            best_at_cycle_start = \
                self.state["standing_best_gap_pct"]
            self.state["cycle"] = cycle
            self.set_stage(f"cycle{cycle}:new_line")
            line = self.build_line(f"line_cycle{cycle}",
                                   cycle=cycle, line_index=cycle)
            self.set_stage(f"cycle{cycle}:merge")
            standing = self.merge_champions(standing, line,
                                            cycle)
            self.state["standing_champion"] = standing
            self.save_state()
            self.set_stage(f"cycle{cycle}:growth")
            standing, _why = self.growth_rounds(standing,
                                                cycle)
            self.state["standing_champion"] = standing
            self.save_state()
            self.set_stage(f"cycle{cycle}:free_evolution")
            standing = self.free_evolution(standing, cycle)
            self.state["standing_champion"] = standing
            self.save_state()
            # step 12: no improvement over a full cycle?
            if standing["gap_pct"] >= best_at_cycle_start:
                self.stop(
                    f"no improvement: cycle {cycle} final "
                    f"champion {standing['gap_pct']:.4f}% did "
                    f"not beat the standing best "
                    f"{best_at_cycle_start:.4f}%")
                break
            log(f"cycle {cycle} improved the standing best: "
                f"{best_at_cycle_start:.4f}% -> "
                f"{standing['gap_pct']:.4f}%; continuing")
        log(f"lifecycle stopped: "
            f"{self.state['termination']['reason']}")
        log(f"standing best "
            f"{self.state['standing_best_gap_pct']:.4f}%; "
            f"smallest at best: "
            f"{self.state['smallest_at_best']}")
        return self.state

    def stop(self, reason):
        self.state["termination"] = {"stopped": True,
                                     "reason": reason}
        self.set_stage("stopped")
        log(f"TERMINATION: {reason}")


# ================================================================
# CLI
# ================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--config",
                    default=str(HERE /
                                "bert_lifecycle_config.json"))
    pr.add_argument("--run-dir", default=None)
    pw = sub.add_parser("_worker")
    pw.add_argument("kind")
    pw.add_argument("--job", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "_worker":
        assert args.kind == "evolve"
        job = json.loads(Path(args.job).read_text())
        return run_evolution(job)
    cfg = json.loads(Path(args.config).read_text())
    run_dir = Path(args.run_dir) if args.run_dir else \
        HERE / "engine_runs" / cfg.get("run_name",
                                       "lifecycle_run")
    driver = LifecycleDriver(cfg, run_dir)
    driver.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
