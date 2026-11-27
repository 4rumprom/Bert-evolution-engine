#!/usr/bin/env python3
"""Evolution of the DEEPENED Frankenstein 4, with
the exact protocol used for Frankenstein 4
(frankenstein_evolve4_2026-11-25.py): only the
new material mutates, everything inherited stays
frozen.

SUBJECT: frankenstein_4_deepened_genome.npz (the
evolved Frankenstein 4 champion, gap 6.8959%,
plus one new pass-through block b2 inserted after
the joining block; the deepen build's gate
verified the deepened network reproduces the
champion's validation tours identically).

MASK: the new block's genes (prefix b2, 154 of
498) are the ONLY mutable genes. Every other gene
-- the whole evolved Frankenstein 4 champion,
344 genes -- is frozen: mutation rate 0,
inherited verbatim. The run asserts at the end of
every generation that the population's frozen
genes still equal the founder's exactly, and at
the end verifies the vaulted champion's frozen
genes against the evolved Frankenstein 4 champion
file itself (bit-identical, through the key
mapping).

POPULATION: 100 genomes, all founded on the
deepened genome. Single founder, no crossover.

MUTATION: only the new block's genes, each with
probability 0.10 per child, the nudge Gaussian
with sigma 0.15 (the engine's nominal values).

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
Frankenstein 4 evolution (EVOLVE_SEED 20261129).

SANITY GATE: generation 0's validation (the
founder, scored before the run) must equal the
starting gap 6.8959%. If it does not, the script
stops and reports the mismatch.

BUDGET: 100 generations. Per-generation atomic
checkpoint in
engine_runs/frankenstein4_deepened_evolution/
(population, rng state, history, best-validation
genome), so the run resumes after a VM
replacement: rerunning this script picks up from
the checkpoint. At the end the best-validation
genome is vaulted as
frankenstein_4_deepened_champion.npz in the run
directory and a COMPLETE marker file is written.

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup python3 \
    frankenstein_deepen4_evolve_2026-11-25.py \
    >> engine_runs/frankenstein4_deepened_evolution/run.log \
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
CHAMP = (HERE / "engine_runs" / "frankenstein4_evolution"
         / "frankenstein_4_evolved_champion.npz")
DEEP = HERE / "frankenstein_4_deepened_genome.npz"
OUT = HERE / "engine_runs" / "frankenstein4_deepened_evolution"
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
START_GAP_PCT = 6.8959

# ================================================================
# Setup: domain (validation set), deepened founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

zg = np.load(DEEP)
DMETA = json.loads(str(zg["meta"]))
KEYS = list(DMETA["gene_order"])
FOUNDER_W = {k: np.asarray(zg[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
assert founder.size == 498, founder.size
OFFSETS = {}
_off = 0
for k in KEYS:
    OFFSETS[k] = _off
    _off += FOUNDER_W[k].size

NEW_IDX = np.concatenate([
    np.arange(OFFSETS[k], OFFSETS[k] + FOUNDER_W[k].size)
    for k in KEYS if k.startswith("b2.")])
FROZEN_IDX = np.setdiff1d(np.arange(498), NEW_IDX)
assert len(NEW_IDX) == 154, len(NEW_IDX)
assert len(FROZEN_IDX) == 344, len(FROZEN_IDX)
log(f"mask: 498 genes, {len(NEW_IDX)} new-block "
    f"(mutate), {len(FROZEN_IDX)} frozen (the whole "
    f"evolved Frankenstein 4 champion)")

# the founder's frozen genes must be the champion's,
# key for key (the deepen build's carry-over)
zc = np.load(CHAMP)
CHAMP_KEYS = list(json.loads(str(zc["meta"]))["gene_order"])
for k in CHAMP_KEYS:
    assert np.array_equal(FOUNDER_W[k],
                          np.asarray(zc[k], dtype=float)), k
log("founder's frozen arrays verified bit-identical "
    "to the evolved Frankenstein 4 champion, key for key")


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
# Frankenstein builds / evolve4; blocks b0, b1, b2)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1", "b2"):
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
# SANITY GATE: generation 0 -- the founder, scored
# on the fixed validation set, must reproduce the
# starting gap 6.8959%.
# ================================================================
founder_val = mean_gap(founder, X_val, OPT_VAL)
log(f"sanity gate (generation 0): founder validation "
    f"gap {founder_val*100:.6f}% vs starting gap "
    f"{START_GAP_PCT:.4f}%")
if abs(founder_val * 100 - START_GAP_PCT) > 5e-5 + 1e-9:
    log(f"SANITY GATE FAILED: founder scores "
        f"{founder_val*100:.6f}%, the starting gap is "
        f"{START_GAP_PCT:.4f}%. STOPPING; no evolution "
        f"run.")
    sys.exit(3)
log("sanity gate passed")

# ================================================================
# Training pool (drawn once from the run seed; the
# same scheme and seed as the Frankenstein 4 run)
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
    "flat_mask_new_block": np.isin(
        np.arange(498), NEW_IDX).astype(int).tolist(),
    "genes_total": 498,
    "genes_new_block": int(len(NEW_IDX)),
    "genes_frozen": int(len(FROZEN_IDX)),
    "rule": "the inserted block b2 = new (True, "
            "mutates); every gene of the evolved "
            "Frankenstein 4 champion = frozen (False)",
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
        f"the deepened Frankenstein 4 genome")

founder_frozen = founder[FROZEN_IDX].copy()


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

    # frozen integrity: no frozen gene may ever move
    assert bool((pop[:, FROZEN_IDX]
                 == founder_frozen).all()), \
        f"frozen gene moved at generation {gen}"
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
        m = rng.random(len(NEW_IDX)) < MUT_P
        if m.any():
            child[NEW_IDX[m]] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + COMPLETE marker
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_new_changed = int((best_val_genome[NEW_IDX]
                     != founder[NEW_IDX]).sum())
assert bool((best_val_genome[FROZEN_IDX]
             == founder_frozen).all())
# cross-check the frozen genes against the evolved
# Frankenstein 4 champion file itself, key for key
for k in CHAMP_KEYS:
    assert np.array_equal(BW[k],
                          np.asarray(zc[k], dtype=float)), k
log("post-run verification: the champion's frozen "
    "genes are bit-identical to the evolved "
    "Frankenstein 4 champion, key for key; "
    f"new-block genes changed {n_new_changed} of "
    f"{len(NEW_IDX)}")
meta = {
    "kind": "frankenstein_4_deepened_champion",
    "date": "2026-11-25",
    "base": "frankenstein_4_deepened_genome.npz",
    "evolution": "population 100, new block's genes "
                 "only (154 of 498) mutated at "
                 "p=0.10, sigma 0.15; the whole "
                 "evolved Frankenstein 4 champion "
                 "frozen; tournament 3; elites 2; "
                 "100 generations; rotating "
                 "200-problem training batches from "
                 "a 2,000-problem pool, seed "
                 f"{EVOLVE_SEED} (the Frankenstein 4 "
                 "evolution protocol, mirrored)",
    "start_gap_pct": START_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "new_block_genes_changed_vs_founder": n_new_changed,
    "gene_order": KEYS,
    "genes_total": 498, "genes_new_block": 154,
    "genes_frozen": 344,
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_4_deepened_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein4_deepened_evolution",
    "generations": GENS,
    "start_gap_pct": START_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "new_block_genes_changed_vs_founder": n_new_changed,
    "frozen_genes_bit_identical_to_evolved_f4_champion": True,
    "champion": "frankenstein_4_deepened_champion.npz",
    "history": history,
}, indent=1))
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (start "
    f"{START_GAP_PCT:.2f}%); champion vaulted, COMPLETE "
    f"marker written; new-block genes changed "
    f"{n_new_changed} of {len(NEW_IDX)}, frozen genes "
    f"bit-identical")
