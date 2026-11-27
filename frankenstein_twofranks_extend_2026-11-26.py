#!/usr/bin/env python3
"""EXTENSION of the Two-Franks evolution: generations
100-199, continuing the finished 100-generation run
from its checkpoint. Same protocol, same population,
same RNG stream (the checkpoint stores the RNG state),
same stitching-only mutation, backbone still frozen.
It exists to answer the equal-dose question: the
100-generation stitch carried 1,106 stitching genes
and its best validation score came at the last
generation, so the curve had not settled.

Evolution of the Two-Franks stitch, with the exact
protocol used for Frankenstein 4
(frankenstein_evolve4_2026-11-25.py): the stitching
genes mutate, the backbone never does.

SUBJECT: frankenstein_twofranks_genome.npz (the two
evolved Frankenstein 4 champions' extracted paths
side by side, one joining block, one merged decoder
over width 8; birth gap recorded in
frankenstein_twofranks_stitch_2026-11-26.json).

MASK: loaded from the stitch record
(frankenstein_twofranks_stitch_2026-11-26.json,
stitching_mask.flat_mask_stitching, in the genome's
gene order). Rule used at build time:
exact-from-a-Frank = backbone (path synapses,
territory vectors, the two decoder blocks, b_qd,
v_start); drawn-as-filler = stitching (all filler,
the whole joining block, the decoder cross blocks).
Every entry was classified when placed or drawn, so
no entry's origin is ambiguous.

POPULATION: 100 genomes, all founded on the
Two-Franks genome. Single founder, no crossover.

MUTATION: only the stitching genes, each with
probability 0.10 per child, the nudge Gaussian with
sigma 0.15 (the engine's nominal values). Backbone
genes are NEVER touched: mutation rate 0, inherited
verbatim. The run asserts at the end of every
generation that the population's backbone still
equals the founder's exactly.

SELECTION: tournament size 3 on training fitness;
the top 2 genomes are copied verbatim each
generation (elites).

FITNESS: mean gap against Held-Karp optima on a
rotating batch of 200 training problems. The
problems come from a fixed pool of 2,000 drawn once
from the run seed (optima computed once); each
generation draws its batch of 200 from the pool
without replacement, with the evolution rng. Every
10 generations (and at the last), the generation's
best genome is scored on the fixed 200-problem
validation set (val_seed 20261116) and logged; the
best-validation genome seen is kept. Same generation
scheme and run seed as the Frankenstein 4 evolution
(EVOLVE_SEED 20261129).

SANITY GATE: before the run counts, the founder's
validation gap must reproduce the birth gap recorded
in the stitch record. If it does not, the script
stops and reports the mismatch.

BUDGET: 200 generations total (the first 100 already
run; this script runs 100-199 from the checkpoint).
Per-generation atomic
checkpoint in
engine_runs/frankenstein_twofranks_evolution/
(population, rng state, history, best-validation
genome), so the run resumes after a VM replacement:
rerunning this script picks up from the checkpoint.
At the end the best-validation genome is vaulted as
frankenstein_twofranks_extended_champion.npz in the
run directory and a COMPLETE_EXTENDED marker file is
written. The original 100-generation champion and its
COMPLETE marker are NOT touched.

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup ~/workspace/bert-engine/.venv/bin/python \
    frankenstein_twofranks_extend_2026-11-26.py \
    >> engine_runs/frankenstein_twofranks_evolution/run_extend.log \
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
OUT = HERE / "engine_runs" / "frankenstein_twofranks_evolution"
POP_N = 100
GENS = 200
MUT_P = 0.10
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
EVOLVE_SEED = 20261129
VAL_EVERY = 10
BLOCKS = ("b0", "b1", "b2")

# ================================================================
# Setup: domain (validation set) and founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

ST = json.load(open(HERE / "frankenstein_twofranks_stitch_"
                         "2026-11-26.json"))
BIRTH_GAP_PCT = ST["frankenstein_twofranks"]["validation_gap_pct"]

zg = np.load(HERE / "frankenstein_twofranks_genome.npz")
META = json.loads(str(zg["meta"]))
KEYS = [k for k in zg.files if k != "meta"]
assert KEYS == ST["stitching_mask"]["gene_order"]
FOUNDER_W = {k: np.asarray(zg[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
N_GENES = int(founder.size)
assert N_GENES == ST["stitching_mask"]["total"], N_GENES

flat_mask = np.array(ST["stitching_mask"]["flat_mask_stitching"],
                     dtype=bool)     # True = stitching (mutates)
STITCH_IDX = np.flatnonzero(flat_mask)
BACKBONE_IDX = np.flatnonzero(~flat_mask)
assert len(STITCH_IDX) == ST["stitching_mask"]["stitching"]
assert len(BACKBONE_IDX) == ST["stitching_mask"]["backbone"]
log(f"mask loaded from the stitch record: {N_GENES} genes, "
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
# Frankenstein builds)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in BLOCKS:
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
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "rule": "exact-from-a-Frank = backbone (False); "
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
        f"the Two-Franks genome")

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
    "kind": "frankenstein_twofranks_extended_champion",
    "date": "2026-11-26",
    "base": "frankenstein_twofranks_genome.npz",
    "evolution": "population 100, stitching genes only "
                 f"({len(STITCH_IDX)} of {N_GENES}) "
                 "mutated at p=0.10, sigma 0.15; backbone "
                 "frozen; tournament 3; elites 2; 200 "
                 "generations; rotating 200-problem "
                 "training batches from a 2,000-problem "
                 f"pool, seed {EVOLVE_SEED} (the "
                 "Frankenstein 4 evolution protocol, "
                 "mirrored)",
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "gene_order": KEYS,
    "genes_total": N_GENES,
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_twofranks_extended_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE_EXTENDED", json.dumps({
    "run": "frankenstein_twofranks_evolution_extended",
    "generations": GENS,
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "champion": "frankenstein_twofranks_extended_champion.npz",
    "history": history,
}, indent=1))
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (birth "
    f"{BIRTH_GAP_PCT:.2f}%); champion vaulted, COMPLETE "
    f"marker written; stitching genes changed "
    f"{n_stitch_changed} of {len(STITCH_IDX)}, backbone "
    f"bit-identical")
