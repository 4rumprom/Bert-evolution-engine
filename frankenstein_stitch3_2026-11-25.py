#!/usr/bin/env python3
"""Frankenstein stitch #3 (sponsor addition,
2026-11-25): the join AFTER the decoders.

Same corrected paths and the same side-by-side
block 1 as Frankenstein 2 (the block-1 weights are
taken verbatim from frankenstein_2_genome.npz, so
the block, the paths and the filler are identical).
No joining block before the decoder. Instead:

  - Each parent keeps its OWN decoder, kept whole
    (every decoder weight exact from the parent).
    Species 8's decoder reads only the species 8
    stream territory (joined coords 0-1); species
    9's decoder reads only its own (coords 2-3).
  - Each decoder runs its own state (its own
    stream mean + its own last-picked embedding)
    and produces its own per-step logits over the
    cities, exactly as in its parent. Both halves
    follow the ONE shared tour: the pick is made
    from the joined logits, then both halves take
    the picked city's embedding from their own
    territory.
  - THE JOIN, after the decoders: final logits at
    each step = the even average of the two
    decoders' logits, PLUS the output of a small
    feed-forward network (inputs: both decoders'
    raw logits, 10 + 10, and both decoder query
    vectors, 2 + 2 = 24 inputs; hidden 16, tanh;
    output: a per-city correction, 10). All join
    weights and biases are tiny random (std 0.03,
    one fixed-seed draw), so at birth the join
    contributes almost nothing and the network is
    born as a plain average of the two decoders:
    born whole. The join is the part that can
    later learn to pick and choose between the
    decoders.
  - Greedy decode on the joined logits, visited
    mask as usual. ONE build, ONE draw, ONE score.

PARITY: the dual-decoder implementation is
verified half by half: run on a parent's own
encoder output with only that parent's half
active, it must reproduce the parent's engine
tours exactly. (On the stitched block-1 output
the halves cannot reproduce the parents exactly:
block 1's LayerNorm runs over the joined width
and the filler perturbs each territory. That is
the known stitch friction, not a code error.)

Run: ~/workspace/bert-engine/.venv/bin/python
    frankenstein_stitch3_2026-11-25.py
Outputs: frankenstein_3_genome.npz,
frankenstein_stitch3_2026-11-25.json,
frankenstein_stitch3_state.json (diagram input).
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
JOIN_SEED = 20261126
JOIN_HIDDEN = 16
EXPECTED_HOST_GAP = 0.1513148917731675

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
# Parents (same selection as stitches 1 and 2)
# ================================================================
z = np.load(RUN / "champion.npz")
host_genome = np.asarray(z["genome"], dtype=float)
host_detail = domain.validation_detail(layout, host_genome)
assert abs(host_detail["mean_gap"] - EXPECTED_HOST_GAP) < 1e-9
ck = np.load(RUN / "checkpoint.npz", allow_pickle=True)
sp_best = {int(s): np.asarray(g, dtype=float)
           for s, g in zip(ck["species_ids"], ck["best_genomes"])}
gaps = {s: float(domain.validation_detail(layout, g)["mean_gap"])
        for s, g in sp_best.items()}
donor_id = min((s for s in sp_best if s != 8),
               key=lambda s: gaps[s])
donor_genome = sp_best[donor_id]
log(f"host = species 8 (gap {gaps[8]*100:.2f}%), "
    f"donor = species {donor_id} (gap {gaps[donor_id]*100:.2f}%)")


def parent_weights(genome):
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
# Forward (block 1 only) + dual decoder
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_b0(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    Hb = ln_np(A, W["b0.lnA_gain"], W["b0.lnA_bias"], EPS)
    Q = Hb @ W["b0.W_q"] + W["b0.b_q"]
    K = Hb @ W["b0.W_k"] + W["b0.b_k"]
    V = Hb @ W["b0.W_v"] + W["b0.b_v"]
    S = np.einsum("btj,bsj->bts", Q, K) * SCALE
    S = S - S.max(axis=-1, keepdims=True)
    E = np.exp(S)
    Atn = E / E.sum(axis=-1, keepdims=True)
    Ah = np.einsum("bts,bsi->bti", Atn, V)
    A = A + Ah @ W["b0.W_o"] + W["b0.b_o"]
    Hf = ln_np(A, W["b0.lnF_gain"], W["b0.lnF_bias"], EPS)
    A = A + np.tanh(Hf @ W["b0.W_1"] + W["b0.b_1"]) \
        @ W["b0.W_2"] + W["b0.b_2"]
    return A


def half_logits(Y, head, state):
    """One decoder half's raw per-city logits for
    the current state (state = gmean + last)."""
    q = state @ head["W_qd"] + head["b_qd"]
    logits = be.CLIP * np.tanh(
        np.einsum("bi,bni->bn", q, Y) * SCALE)
    return logits, q


def decode_dual(Yh, head_h, Yd, head_d, join=None,
                only=None):
    """Greedy decode with two decoder halves.
    only='h'/'d': use one half alone (parity mode).
    Otherwise final logits = even average of the
    halves + join correction."""
    B, n, _ = Yh.shape
    gm_h, gm_d = Yh.mean(axis=1), Yd.mean(axis=1)
    last_h = np.broadcast_to(
        head_h["v_start"], (B, Yh.shape[2])).copy()
    last_d = np.broadcast_to(
        head_d["v_start"], (B, Yd.shape[2])).copy()
    visited = np.zeros((B, n), dtype=bool)
    arB = np.arange(B)
    picks = np.zeros((B, n), dtype=int)
    for t in range(n):
        lh, qh = half_logits(Yh, head_h, gm_h + last_h)
        ld, qd = half_logits(Yd, head_d, gm_d + last_d)
        if only == "h":
            final = lh
        elif only == "d":
            final = ld
        else:
            final = 0.5 * (lh + ld)
            if join is not None:
                jin = np.concatenate(
                    [lh, ld, qh, qd], axis=1)
                hid = np.tanh(jin @ join["W1"]
                              + join["b1"])
                final = final + hid @ join["W2"] \
                    + join["b2"]
        final = np.where(visited, -1e18, final)
        pick = np.argmax(final, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last_h = Yh[arB, pick]
        last_d = Yd[arB, pick]
    return picks


def gap_from_picks(picks):
    lens = be.tour_lengths_batched(
        X_val, picks[None, :, :])[0]
    return float((lens / OPT - 1.0).mean()), lens


# ---- parity: each half alone, on its parent's own
# encoder output, must reproduce the parent's tours
head_h = {"v_start": HW["head.v_start"],
          "W_qd": HW["head.W_qd"], "b_qd": HW["head.b_qd"]}
head_d = {"v_start": DW["head.v_start"],
          "W_qd": DW["head.W_qd"], "b_qd": DW["head.b_qd"]}
for tag, W, genome, head in (
        ("host", HW, host_genome, head_h),
        ("donor", DW, donor_genome, head_d)):
    Y = forward_b0(W, X_val)
    picks = decode_dual(Y, head, Y, head,
                        only="h" if tag == "host" else "d")
    _, picks_e = domain.evaluate_lengths(
        layout, genome[None, :], X_val)
    same = bool((picks == picks_e[0]).all())
    g, _ = gap_from_picks(picks)
    log(f"parity {tag} half-decoder: gap {g*100:.6f}% "
        f"vs parent {gaps[8 if tag == 'host' else donor_id]*100:.6f}%, "
        f"tours identical: {same}")
    assert same

# ================================================================
# Build Frankenstein 3
# ================================================================
# block 1 verbatim from Frankenstein 2's genome
f2 = np.load(HERE / "frankenstein_2_genome.npz")
F3 = {k: np.asarray(f2[k]) for k in f2.files
      if k.startswith("embed.") or k.startswith("b0.")}
state2 = json.load(open(
    HERE / "frankenstein_stitch2_state.json"))
h_map = {int(k): v for k, v in
         state2["frank"]["h_map"].items()}
d_map = {int(k): v for k, v in
         state2["frank"]["d_map"].items()}
dF = state2["frank"]["dF"]
U = state2["frank"]["U"]
log(f"block 1 taken verbatim from Frankenstein 2 "
    f"(stream {dF}, FFN {U}); host territory "
    f"{h_map}, donor territory {d_map}")

# the two decoders, kept whole from the parents
for nm in ("v_start", "W_qd", "b_qd"):
    F3[f"head_h.{nm}"] = HW[f"head.{nm}"].copy()
    F3[f"head_d.{nm}"] = DW[f"head.{nm}"].copy()

# the join after the decoders: tiny random, one draw
rng = np.random.default_rng(JOIN_SEED)
join = {
    "W1": rng.normal(0.0, SIGMA_FILL, (24, JOIN_HIDDEN)),
    "b1": rng.normal(0.0, SIGMA_FILL, (JOIN_HIDDEN,)),
    "W2": rng.normal(0.0, SIGMA_FILL, (JOIN_HIDDEN, 10)),
    "b2": rng.normal(0.0, SIGMA_FILL, (10,)),
}
for k, v in join.items():
    F3[f"join.{k}"] = v

# ---- score once
Y = forward_b0(F3, X_val)
h_cols = [h_map[k] for k in sorted(h_map)]
d_cols = [d_map[k] for k in sorted(d_map)]
Yh, Yd = Y[:, :, h_cols], Y[:, :, d_cols]
picks_avg = decode_dual(Yh, head_h, Yd, head_d,
                        join=None)
gap_avg, _ = gap_from_picks(picks_avg)
picks = decode_dual(Yh, head_h, Yd, head_d, join=join)
gap3, lens3 = gap_from_picks(picks)
log(f"FRANKENSTEIN 3 (plain average of decoders): "
    f"gap {gap_avg*100:.2f}%")
log(f"FRANKENSTEIN 3 (average + tiny join): "
    f"gap {gap3*100:.2f}% (host {gaps[8]*100:.2f}%, "
    f"donor {gaps[donor_id]*100:.2f}%, Frankenstein "
    f"2 was 41.47%)")

# ================================================================
# Save
# ================================================================
meta = {
    "kind": "frankenstein_stitch3",
    "date": "2026-11-25",
    "join": "after the decoders: final logits = "
            "even average of the two decoders' "
            "logits + small FF join (24 -> "
            f"{JOIN_HIDDEN} -> 10, tiny random std "
            f"{SIGMA_FILL}, seed {JOIN_SEED})",
    "host_coord_map": {str(k): v for k, v in h_map.items()},
    "donor_coord_map": {str(k): v for k, v in d_map.items()},
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / "frankenstein_3_genome.npz", **F3,
         meta=np.array(json.dumps(meta)))
log("saved frankenstein_3_genome.npz")

results = {
    "design": "sponsor stitching design, stitch #3: "
              "join after the decoders (sponsor "
              "addition, 2026-11-25)",
    "parents": {
        "host": {"species": 8,
                 "gap_pct": round(gaps[8] * 100, 4)},
        "donor": {"species": donor_id,
                  "gap_pct": round(
                      gaps[donor_id] * 100, 4)},
    },
    "frankenstein3": {
        "shape": {
            "stream_width": dF,
            "block1_ffn_width": U,
            "blocks": 1,
            "decoders": 2,
            "decoder_territories": {
                "host": h_cols, "donor": d_cols},
            "join": {"in": 24, "hidden": JOIN_HIDDEN,
                     "out": 10,
                     "sigma": SIGMA_FILL,
                     "seed": JOIN_SEED},
            "total_params": int(sum(
                v.size for v in F3.values())),
        },
        "block1": "verbatim from Frankenstein 2 "
                  "(same corrected paths, same "
                  "side-by-side block, same filler)",
        "decoders": "each parent's decoder kept "
                    "whole, exact, reading only "
                    "its own stream territory",
        "validation_gap_pct": round(gap3 * 100, 4),
        "validation_gap_plain_average_pct": round(
            gap_avg * 100, 4),
        "frankenstein2_gap_pct": 41.474,
        "mean_length": float(lens3.mean()),
        "parity": "each decoder half alone, on its "
                  "parent's own encoder output, "
                  "reproduces the parent's engine "
                  "tours exactly (asserted in the "
                  "build log)",
    },
}
with open(HERE / "frankenstein_stitch3_2026-11-25.json",
          "w") as f:
    json.dump(results, f, indent=1)
log("saved frankenstein_stitch3_2026-11-25.json")

state3 = {"gap": gap3, "gap_plain_average": gap_avg,
          "dF": dF, "U": U, "h_map": h_map,
          "d_map": d_map,
          "hu_map": state2["frank"]["hu_map"],
          "du_map": state2["frank"]["du_map"],
          "h_path": state2["frank"]["h_path"],
          "d_path": state2["frank"]["d_path"]}
with open(HERE / "frankenstein_stitch3_state.json",
          "w") as f:
    json.dump(state3, f, indent=1)
log("saved frankenstein_stitch3_state.json "
    "(diagram input)")
log("DONE build")
