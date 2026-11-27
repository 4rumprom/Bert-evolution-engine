#!/usr/bin/env python3
"""Evolution of Frankenstein 4, with the exact
protocol used for Frankenstein 5
(frankenstein_evolve5_2026-11-25.py), for a clean
comparison: the stitching genes mutate, the
backbone never does.

SUBJECT: frankenstein_4_genome.npz (Frankenstein
2's body with the merged decoder; birth gap
23.2865% on the fixed 200-problem validation set;
parents species 8 at 15.1315%, species 9 at
19.4412%).

MASK (derived here from the build records, rule:
exact-from-a-parent = backbone, drawn-as-filler =
stitching):
  - embed.W_in and the b0 (widened block) matrices
    W_q/W_k/W_v/W_o/W_1/W_2: the entries the stitch
    #2 build copied exact are exactly the parents'
    path synapses (host 13, donor 26, recorded in
    frankenstein_stitch2_2026-11-25.json), placed
    by the territory maps in the genome meta.
    Those positions are backbone; every other
    entry of those matrices was drawn as filler
    and is stitching.
  - embed.b_in, b0 layer norms, b0 biases
    (b_q/b_k/b_v/b_o/b_2) and b0.b_1: the build
    placed the parents' own values at every
    territory position (territory_vec / unit
    placement) -> all backbone.
  - b1 (the joining block): drawn entirely as
    filler in stitch #2 -> all stitching.
  - head (the merged decoder of stitch #4):
    W_qd diagonal 2x2 blocks are the parents'
    full W_qd (backbone); the off-diagonal cross
    blocks are the one tiny-random draw
    (stitching); b_qd and v_start are the exact
    concatenations (backbone).
  No entry's origin is ambiguous in the records:
  the stitch #2 build code assigns every position
  deterministically (fill, then overwrite exact
  positions), and stitch #4 replaced the head
  wholesale. The derivation is VERIFIED below by
  value: every backbone position must equal the
  mapped parent value exactly, and the head must
  equal the stitch #4 record; any mismatch stops
  the run before it starts.
  Derived counts: 344 total, 97 backbone,
  247 stitching.

POPULATION: 100 genomes, all founded on the
Frankenstein 4 genome. Single founder, no
crossover.

MUTATION: only the stitching genes, each with
probability 0.10 per child, the nudge Gaussian
with sigma 0.15 (the engine's nominal values).
Backbone genes are NEVER touched: mutation rate
0, inherited verbatim. The run asserts at the end
of every generation that the population's
backbone still equals the founder's exactly.

SELECTION: tournament size 3 on training fitness;
the top 2 genomes are copied verbatim each
generation (elites).

FITNESS: mean gap against Held-Karp optima on a
rotating batch of 200 training problems. The
problems come from a fixed pool of 2,000 drawn
once from the run seed (optima computed once);
each generation draws its batch of 200 from the
pool without replacement, with the evolution rng.
Every 10 generations (and at the last), the
generation's best genome is scored on the fixed
200-problem validation set (val_seed 20261116)
and logged; the best-validation genome seen is
kept. Same generation scheme and run seed as the
Frankenstein 5 evolution (EVOLVE_SEED 20261129).

SANITY GATE: before the run counts, the founder's
validation gap must reproduce the birth gap
23.2865% recorded in
frankenstein_stitch4_2026-11-25.json. If it does
not, the script stops and reports the mismatch.

BUDGET: 100 generations. Per-generation atomic
checkpoint in engine_runs/frankenstein4_evolution/
(population, rng state, history, best-validation
genome), so the run resumes after a VM
replacement: rerunning this script picks up from
the checkpoint. At the end the best-validation
genome is vaulted as
frankenstein_4_evolved_champion.npz in the run
directory and a COMPLETE marker file is written.

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup python3 frankenstein_evolve4_2026-11-25.py \
    >> engine_runs/frankenstein4_evolution/run.log \
    2>&1 < /dev/null &
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend
from tsp_genome2 import held_karp

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CFG = HERE / "bert_engine_config_fresh50.json"
RUN = HERE / "engine_runs" / "fresh50_v2"
OUT = HERE / "engine_runs" / "frankenstein4_evolution"
POP_N = 100
GENS = 100
MUT_P = 0.10
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
EVOLVE_SEED = 20261129
VAL_EVERY = 10
EXPECTED_HOST_GAP = 0.1513148917731675

# ================================================================
# Setup: domain (validation set), parents, founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

S4 = json.load(open(HERE / "frankenstein_stitch4_2026-11-25.json"))
S2 = json.load(open(HERE / "frankenstein_stitch2_2026-11-25.json"))
BIRTH_GAP_PCT = S4["frankenstein4"]["validation_gap_pct"]
assert BIRTH_GAP_PCT == 23.2865

zg = np.load(HERE / "frankenstein_4_genome.npz")
META4 = json.loads(str(zg["meta"]))
KEYS = [k for k in zg.files if k != "meta"]
FOUNDER_W = {k: np.asarray(zg[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
assert founder.size == 344, founder.size
OFFSETS = {}
_off = 0
for k in KEYS:
    OFFSETS[k] = _off
    _off += FOUNDER_W[k].size


def flat_idx(key, idx2):
    return OFFSETS[key] + int(np.ravel_multi_index(
        idx2, FOUNDER_W[key].shape))


# parents (generic weights, verbatim from stitch #4)
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
assert donor_id == 9
donor_genome = sp_best[donor_id]


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
h_map = {int(k): v for k, v in META4["host_coord_map"].items()}
d_map = {int(k): v for k, v in META4["donor_coord_map"].items()}
hu_map = {int(k): v for k, v in META4["host_unit_map"].items()}
du_map = {int(k): v for k, v in META4["donor_unit_map"].items()}

# ================================================================
# Mask derivation (rule: exact-from-a-parent =
# backbone; drawn-as-filler = stitching), verified
# by value against the parents and the stitch #4
# record.
# ================================================================
backbone = np.zeros(344, dtype=bool)


def mark(key, idx2):
    backbone[flat_idx(key, idx2)] = True


def check_eq(key, idx2, value, what):
    got = FOUNDER_W[key][idx2]
    assert got == value, \
        f"mask verification failed: {what} {key}{idx2}: " \
        f"founder {got!r} != parent {value!r}"


# --- path synapses in embed.W_in and the b0 matrices
MAT_ENTRIES = ("embed.W_in", "b0.W_q", "b0.W_k", "b0.W_v",
               "b0.W_o", "b0.W_1", "b0.W_2")
for tag, P, PW, cmap, umap in (
        ("host", S2["parents"]["host"], HW, h_map, hu_map),
        ("donor", S2["parents"]["donor"], DW, d_map, du_map)):
    for e in P["path_weight_synapses"]:
        entry, i, j = e["entry"], e["i"], e["j"]
        if entry == "head.W_qd":
            continue  # decoder handled below (stitch #4)
        assert entry in MAT_ENTRIES, entry
        if entry == "embed.W_in":
            pos = (i, cmap[j])            # rows are shared
        elif entry == "b0.W_1":
            pos = (cmap[i], umap[j])
        elif entry == "b0.W_2":
            pos = (umap[i], cmap[j])
        else:
            pos = (cmap[i], cmap[j])
        check_eq(entry, pos, PW[entry][i, j],
                 f"{tag} path synapse")
        mark(entry, pos)

# --- territory vectors: parents' own values at every
# territory position (stitch #2 territory_vec and the
# b_1 placement)
for nm in ("embed.b_in", "b0.lnA_gain", "b0.lnA_bias",
           "b0.lnF_gain", "b0.lnF_bias", "b0.b_q", "b0.b_k",
           "b0.b_v", "b0.b_o", "b0.b_2"):
    for cmap, PW in ((h_map, HW), (d_map, DW)):
        for k_old, k_new in cmap.items():
            check_eq(nm, (k_new,), PW[nm][k_old],
                     "territory vector")
            mark(nm, (k_new,))
for umap, PW in ((hu_map, HW), (du_map, DW)):
    for u_old, u_new in umap.items():
        check_eq("b0.b_1", (u_new,), PW["b0.b_1"][u_old],
                 "territory vector")
        mark("b0.b_1", (u_new,))

# --- merged decoder (stitch #4): W_qd diagonal blocks
# are the parents' full W_qd; b_qd / v_start the exact
# concatenations
for cmap, PW in ((h_map, HW), (d_map, DW)):
    for i_old, i_new in cmap.items():
        for j_old, j_new in cmap.items():
            check_eq("head.W_qd", (i_new, j_new),
                     PW["head.W_qd"][i_old, j_old],
                     "merged decoder block")
            mark("head.W_qd", (i_new, j_new))
for nm in ("head.b_qd", "head.v_start"):
    for cmap, PW in ((h_map, HW), (d_map, DW)):
        for k_old, k_new in cmap.items():
            check_eq(nm, (k_new,), PW[nm][k_old],
                     "merged decoder vector")
            mark(nm, (k_new,))
# cross-check the head against the stitch #4 record
assert np.array_equal(FOUNDER_W["head.W_qd"],
                      np.array(S4["merged_decoder"]["W_qd_full"]))
assert np.array_equal(FOUNDER_W["head.b_qd"],
                      np.array(S4["merged_decoder"]["b_qd"]))
assert np.array_equal(FOUNDER_W["head.v_start"],
                      np.array(S4["merged_decoder"]["v_start"]))

flat_mask = ~backbone          # True = stitching (mutates)
STITCH_IDX = np.flatnonzero(flat_mask)
BACKBONE_IDX = np.flatnonzero(backbone)
assert len(STITCH_IDX) + len(BACKBONE_IDX) == 344
assert len(BACKBONE_IDX) == 97, len(BACKBONE_IDX)
assert len(STITCH_IDX) == 247, len(STITCH_IDX)
log(f"mask derived and verified by value: 344 genes, "
    f"{len(STITCH_IDX)} stitching (mutate), "
    f"{len(BACKBONE_IDX)} backbone (frozen)")


def unpack(vec):
    W = {}
    off = 0
    for k in KEYS:
        n = FOUNDER_W[k].size
        W[k] = vec[off:off + n].reshape(FOUNDER_W[k].shape)
        off += n
    return W


# ================================================================
# Forward + decode (verbatim conventions from the
# Frankenstein builds / evolve5)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1"):
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


def mean_gap(vec, X, opt):
    W = unpack(vec)
    Y = forward_general(W, X)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean())


# ================================================================
# SANITY GATE: the founder, scored on the fixed
# validation set, must reproduce the recorded birth
# gap before the run counts.
# ================================================================
founder_val = mean_gap(founder, X_val, OPT_VAL)
log(f"sanity gate: founder validation gap "
    f"{founder_val*100:.6f}% vs recorded birth gap "
    f"{BIRTH_GAP_PCT:.4f}%")
if abs(founder_val * 100 - BIRTH_GAP_PCT) > 5e-5 + 1e-9:
    log(f"SANITY GATE FAILED: founder scores "
        f"{founder_val*100:.6f}%, record says "
        f"{BIRTH_GAP_PCT:.4f}%. STOPPING; no evolution "
        f"run.")
    sys.exit(3)
log("sanity gate passed")

# ================================================================
# Training pool (drawn once from the run seed)
# ================================================================
rng_pool = np.random.default_rng(EVOLVE_SEED)
X_pool = rng_pool.random((TRAIN_POOL, cfg.domain_config.get(
    "n_cities", 10), 2))
t_hk = time.time()
OPT_POOL = np.array([held_karp(X_pool[i])
                     for i in range(TRAIN_POOL)])
log(f"training pool: {TRAIN_POOL} problems from seed "
    f"{EVOLVE_SEED}, optima computed in "
    f"{time.time() - t_hk:.1f}s")

# ================================================================
# Population state (fresh, or resumed from checkpoint)
# ================================================================
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "checkpoint.npz"
STATUS = OUT / "status.json"


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_stitching": flat_mask.astype(int).tolist(),
    "genes_total": 344,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "rule": "exact-from-a-parent = backbone (False); "
            "drawn-as-filler = stitching (True)",
}, indent=1))

rng = np.random.default_rng(EVOLVE_SEED + 1000)
history = []          # per-gen dicts
best_val_gap = np.inf
best_val_genome = founder.copy()
best_val_gen = -1
start_gen = 0
pop = np.tile(founder, (POP_N, 1))

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
    log(f"fresh start: population {POP_N} founded on "
        f"the Frankenstein 4 genome")

founder_backbone = founder[BACKBONE_IDX].copy()


def checkpoint(next_gen):
    atomic_savez(
        CKPT,
        pop=pop, next_gen=next_gen,
        best_val_gap=best_val_gap,
        best_val_genome=best_val_genome,
        best_val_gen=best_val_gen,
        history=np.array(json.dumps(history)),
        rng_state=np.array(json.dumps(
            rng.bit_generator.state)))
    atomic_write(STATUS, json.dumps({
        "next_gen": next_gen, "gens_budget": GENS,
        "best_val_gap_pct": round(best_val_gap * 100, 4)
        if np.isfinite(best_val_gap) else None,
        "best_val_gen": best_val_gen,
        "last": history[-1] if history else None,
    }, indent=1))


# ================================================================
# Evolution
# ================================================================
for gen in range(start_gen, GENS):
    idx = rng.choice(TRAIN_POOL, size=TRAIN_BATCH,
                     replace=False)
    Xb, ob = X_pool[idx], OPT_POOL[idx]
    fit = np.array([mean_gap(pop[i], Xb, ob)
                    for i in range(POP_N)])
    bi = int(np.argmin(fit))
    rec = {"gen": gen,
           "train_best_gap_pct": round(float(fit[bi]) * 100, 4),
           "train_mean_gap_pct": round(float(fit.mean()) * 100, 4)}
    if gen % VAL_EVERY == 0 or gen == GENS - 1:
        vg = mean_gap(pop[bi], X_val, OPT_VAL)
        rec["val_gap_pct"] = round(vg * 100, 4)
        rec["val_genome_train_rank"] = "best-of-generation"
        if vg < best_val_gap:
            best_val_gap = vg
            best_val_genome = pop[bi].copy()
            best_val_gen = gen
        log(f"gen {gen}: train best "
            f"{fit[bi]*100:.2f}%, mean {fit.mean()*100:.2f}%; "
            f"VALIDATION best-of-gen {vg*100:.2f}% "
            f"(best so far {best_val_gap*100:.2f}% at "
            f"gen {best_val_gen})")
    else:
        log(f"gen {gen}: train best {fit[bi]*100:.2f}%, "
            f"mean {fit.mean()*100:.2f}%")
    history.append(rec)

    # backbone integrity: no backbone gene may ever move
    assert bool((pop[:, BACKBONE_IDX]
                 == founder_backbone).all()), \
        f"backbone moved at generation {gen}"
    checkpoint(gen + 1)
    if gen == GENS - 1:
        break

    # breed the next generation
    order = np.argsort(fit)
    new_pop = np.empty_like(pop)
    for e in range(ELITES):
        new_pop[e] = pop[order[e]]
    for slot in range(ELITES, POP_N):
        cand = rng.choice(POP_N, size=TOURNAMENT,
                          replace=False)
        parent = pop[cand[int(np.argmin(fit[cand]))]]
        child = parent.copy()
        m = rng.random(len(STITCH_IDX)) < MUT_P
        if m.any():
            child[STITCH_IDX[m]] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + COMPLETE marker
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_stitch_changed = int((best_val_genome[STITCH_IDX]
                        != founder[STITCH_IDX]).sum())
assert bool((best_val_genome[BACKBONE_IDX]
             == founder_backbone).all())
meta = {
    "kind": "frankenstein_4_evolved_champion",
    "date": "2026-11-25",
    "base": "frankenstein_4_genome.npz",
    "evolution": "population 100, stitching genes only "
                 "(247 of 344) mutated at p=0.10, sigma "
                 "0.15; backbone frozen; tournament 3; "
                 "elites 2; 100 generations; rotating "
                 "200-problem training batches from a "
                 "2,000-problem pool, seed "
                 f"{EVOLVE_SEED} (the Frankenstein 5 "
                 "evolution protocol, mirrored)",
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "gene_order": KEYS,
    "genes_total": 344, "genes_backbone": 97,
    "genes_stitching": 247,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_4_evolved_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein4_evolution",
    "generations": GENS,
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "champion": "frankenstein_4_evolved_champion.npz",
    "history": history,
}, indent=1))
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (birth "
    f"{BIRTH_GAP_PCT:.2f}%); champion vaulted, COMPLETE "
    f"marker written; stitching genes changed "
    f"{n_stitch_changed} of {len(STITCH_IDX)}, backbone "
    f"bit-identical")
