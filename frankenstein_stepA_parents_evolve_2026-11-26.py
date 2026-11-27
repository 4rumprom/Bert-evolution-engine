#!/usr/bin/env python3
"""STEP A REDO, step 1 (2026-11-26): evolve each
deep50_stepA parent a further 100 generations
BEFORE stitching (sponsor correction: the
parents must be evolved further first).

The evolution machinery of
frankenstein_stepA_evolve_2026-11-26.py with
the mutation mask set to ALL genes: founding
rules, nothing frozen, every gene mutates at
the nominal rate (per-gene 0.10, Gaussian
sigma 0.15), elites top 2 verbatim,
tournament 3, no crossover, rotating
200-problem training batches from a
2,000-problem pool drawn once from the run
seed, validation on the fixed 200-problem
set (val_seed 20261116) every 10 generations,
100 generations.

SUBJECTS (310-gene, 4-block parents):
  host:  engine_runs/deep50_stepA/champion.npz
         (species 1, founding gap 17.1420%),
         seed 20261210
  donor: species 0 best genome in
         engine_runs/deep50_stepA/checkpoint.npz
         (founding gap 19.6989%), seed 20261211

Usage:
  python3 frankenstein_stepA_parents_evolve_2026-11-26.py host
  python3 frankenstein_stepA_parents_evolve_2026-11-26.py donor

Outputs per parent: per-generation atomic
checkpoint + status in
engine_runs/frankenstein_stepA_parent_<tag>_evolution/,
and at the end the evolved genome vaulted as
frankenstein_stepA_parent_<tag>_evolved.npz
(here) plus a results JSON in the run dir.

SANITY GATE: the founder's validation gap
must reproduce the recorded founding gap
before the run counts.

LAUNCH (standing sponsor rule, fully
detached):
  setsid nohup python3 \
    frankenstein_stepA_parents_evolve_2026-11-26.py host \
    >> engine_runs/frankenstein_stepA_parent_host_evolution/run.log \
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


TAG = sys.argv[1]
assert TAG in ("host", "donor"), TAG
SEEDS = {"host": 20261210, "donor": 20261211}
FOUNDING_GAP_PCT = {"host": 17.1420, "donor": 19.6989}
EVOLVE_SEED = SEEDS[TAG]

CFG = HERE / "bert_engine_config_deep50_stepA.json"
RUN = HERE / "engine_runs" / "deep50_stepA"
OUT = HERE / "engine_runs" / f"frankenstein_stepA_parent_{TAG}_evolution"
POP_N = 100
GENS = 100
MUT_P = 0.10
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
VAL_EVERY = 10
N_BLOCKS = 4
BLOCKS = tuple(f"b{k}" for k in range(N_BLOCKS))

# ================================================================
# Setup: domain (validation set) + founder
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

if TAG == "host":
    z = np.load(RUN / "champion.npz")
    founder = np.asarray(z["genome"], dtype=float)
    SPECIES = 1
else:
    ck0 = np.load(RUN / "checkpoint.npz", allow_pickle=True)
    sp_best = {int(s): np.asarray(g, dtype=float)
               for s, g in zip(ck0["species_ids"],
                               ck0["best_genomes"])}
    SPECIES = 0
    founder = sp_best[SPECIES]
N_GENES = founder.size
MUTATE_IDX = np.arange(N_GENES)   # ALL genes mutate
log(f"{TAG}: species {SPECIES} founder, {N_GENES} genes, "
    f"seed {EVOLVE_SEED}, mask = ALL genes")


def parent_weights(genome):
    """Engine genome -> generic weight dict for
    the 4-block parent shape (verbatim from the
    step A stitch script)."""
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
# Forward + decode (verbatim conventions from
# the step A stitch script)
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


def mean_gap(vec, X, opt):
    W = parent_weights(vec)
    Y = forward_general(W, X, BLOCKS)
    picks = decode_general(W, Y, X)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean())


# ================================================================
# SANITY GATE: the founder must reproduce the
# recorded founding gap before the run counts
# (parity against the engine's own read too).
# ================================================================
founder_val = mean_gap(founder, X_val, OPT_VAL)
engine_val = float(domain.validation_detail(
    layout, founder)["mean_gap"])
log(f"sanity gate: founder validation gap "
    f"{founder_val*100:.6f}% (engine "
    f"{engine_val*100:.6f}%) vs recorded founding gap "
    f"{FOUNDING_GAP_PCT[TAG]:.4f}%")
assert abs(founder_val - engine_val) < 1e-9
if abs(founder_val * 100 - FOUNDING_GAP_PCT[TAG]) > 5e-5 + 1e-9:
    log(f"SANITY GATE FAILED: founder scores "
        f"{founder_val*100:.6f}%, record says "
        f"{FOUNDING_GAP_PCT[TAG]:.4f}%. STOPPING; no "
        f"evolution run.")
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
# Population state (fresh, or resumed)
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


rng = np.random.default_rng(EVOLVE_SEED + 1000)
history = []
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
        f"the deep50_stepA {TAG} genome (species "
        f"{SPECIES})")


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
# Evolution (the step A seam-evolution loop
# with the mask set to ALL genes; nothing
# frozen)
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
    checkpoint(gen + 1)
    if gen == GENS - 1:
        break

    order = np.argsort(fit)
    new_pop = np.empty_like(pop)
    for e in range(ELITES):
        new_pop[e] = pop[order[e]]
    for slot in range(ELITES, POP_N):
        cand = rng.choice(POP_N, size=TOURNAMENT,
                          replace=False)
        parent = pop[cand[int(np.argmin(fit[cand]))]]
        child = parent.copy()
        m = rng.random(N_GENES) < MUT_P
        if m.any():
            child[m] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + results
# ================================================================
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_changed = int((best_val_genome != founder).sum())
walk = [{"gen": r["gen"], "val_gap_pct": r["val_gap_pct"]}
        for r in history if "val_gap_pct" in r]
meta = {
    "kind": f"frankenstein_stepA_parent_{TAG}_evolved",
    "date": "2026-11-26",
    "step": "step A redo, step 1: parent evolved a "
            "further 100 generations before stitching "
            "(sponsor correction)",
    "base": "engine_runs/deep50_stepA "
            f"{'champion.npz' if TAG == 'host' else 'checkpoint.npz species 0 best'}",
    "species": SPECIES,
    "evolution": "population 100, ALL genes mutated "
                 "at p=0.10, sigma 0.15; nothing frozen; "
                 "tournament 3; elites 2; no crossover; "
                 "100 generations; rotating 200-problem "
                 "training batches from a 2,000-problem "
                 f"pool, seed {EVOLVE_SEED} (the step A "
                 "evolution machinery with the mask set "
                 "to all genes)",
    "founding_gap_pct": FOUNDING_GAP_PCT[TAG],
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "genes_total": int(N_GENES),
    "genes_changed_vs_founder": n_changed,
    "validation_walk": walk,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(HERE / f"frankenstein_stepA_parent_{TAG}_evolved.npz",
         genome=best_val_genome,
         meta=np.array(json.dumps(meta)))
results = {
    "parent": TAG, "species": SPECIES,
    "seed": EVOLVE_SEED,
    "founding_gap_pct": FOUNDING_GAP_PCT[TAG],
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "genes_total": int(N_GENES),
    "genes_changed_vs_founder": n_changed,
    "validation_walk": walk,
    "history": history,
}
atomic_write(OUT / "results.json", json.dumps(results,
                                              indent=1))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": f"frankenstein_stepA_parent_{TAG}_evolution",
    "generations": GENS,
    "founding_gap_pct": FOUNDING_GAP_PCT[TAG],
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "genome": f"frankenstein_stepA_parent_{TAG}_evolved.npz",
}, indent=1))
log(f"DONE: {TAG} best validation gap "
    f"{best_val_gap*100:.4f}% at generation {best_val_gen} "
    f"(founding {FOUNDING_GAP_PCT[TAG]:.4f}%); evolved "
    f"genome vaulted; genes changed {n_changed} of "
    f"{N_GENES}")
