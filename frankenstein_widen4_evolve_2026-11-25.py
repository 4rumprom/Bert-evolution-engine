#!/usr/bin/env python3
"""Evolution of the WIDENED Frankenstein 4
champion, with the exact protocol used for
Frankenstein 4 (frankenstein_evolve4_2026-11-25.py):
only the material the widen ADDED mutates; the
base genes (the champion's original units, heads
and stream dimensions, at their widened preserving
values) never move.

SUBJECT: frankenstein_4_widened_genome.npz (the
evolved Frankenstein 4 champion after the sponsor
default widen: stream 4 -> 8, FFN 6 -> 12, heads
2 -> 4 per block, Net2WiderNet splits; preservation
gate passed exactly -- identical tours, gap
6.8959% on the fixed 200-problem validation set).
2896 genes: 504 base (frozen), 2392 added
(evolving); the mask is stored in the widened
genome file and asserted here.

POPULATION: 100 genomes, all founded on the
widened genome. Single founder, no crossover.

MUTATION: only the added genes, each with
probability 0.10 per child, the nudge Gaussian
with sigma 0.15 (the engine's nominal values).
Base genes are NEVER touched: mutation rate 0,
inherited verbatim. The run asserts at the end of
every generation that the population's base still
equals the founder's exactly.

SELECTION: tournament size 3 on training fitness;
the top 2 genomes are copied verbatim each
generation (elites).

FITNESS: mean gap against Held-Karp optima on a
rotating batch of 200 training problems from a
fixed pool of 2,000 drawn once from the run seed
(same generation scheme and run seed as the
Frankenstein 4 evolution, EVOLVE_SEED 20261129).
Every 10 generations (and at the last), the
generation's best genome is scored on the fixed
200-problem validation set and logged; the
best-validation genome seen is kept.

SANITY GATE: before the run counts, the founder's
validation gap must reproduce 6.8959% (the
preservation gate's value). If it does not, the
script stops and reports the mismatch.

BUDGET: 100 generations. Per-generation atomic
checkpoint in
engine_runs/frankenstein4_widened_evolution/
(population, rng state, history, best-validation
genome); rerunning this script resumes from the
checkpoint. At the end the best-validation genome
is vaulted as frankenstein_4_widened_champion.npz
in the run directory and a COMPLETE marker file
is written, including the KEEP/REVERT read: the
widened line is kept only if the final best beats
the champion's 6.8959%.

LAUNCH (standing sponsor rule, fully detached):
  setsid nohup .venv/bin/python \
    frankenstein_widen4_evolve_2026-11-25.py \
    >> engine_runs/frankenstein4_widened_evolution/run.log \
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
WIDE = HERE / "frankenstein_4_widened_genome.npz"
OUT = HERE / "engine_runs" / "frankenstein4_widened_evolution"
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
CHAMP_GAP_PCT = 6.8959
HEADS = 4

# ================================================================
# Setup: domain (validation set), widened founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
EPS = arch.ln_eps

zw = np.load(WIDE)
META_W = json.loads(str(zw["meta"]))
KEYS = [k for k in zw.files if k not in ("meta", "added_mask")]
FOUNDER_W = {k: np.asarray(zw[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
flat_added = np.asarray(zw["added_mask"]).astype(bool)
assert flat_added.size == founder.size
ADDED_IDX = np.flatnonzero(flat_added)
BASE_IDX = np.flatnonzero(~flat_added)
assert founder.size == 2896, founder.size
assert len(BASE_IDX) == 504 and len(ADDED_IDX) == 2392, \
    (len(BASE_IDX), len(ADDED_IDX))
SCALE_W = float(META_W["attn_scale"])
assert abs(SCALE_W - arch.attn_scale / 2.0) < 1e-15
log(f"widened founder loaded: {founder.size} genes, "
    f"{len(ADDED_IDX)} added (mutate), "
    f"{len(BASE_IDX)} base (frozen); attn_scale "
    f"{SCALE_W!r}, ln_eps {EPS!r}")


def unpack(vec):
    W = {}
    off = 0
    for k in KEYS:
        n = FOUNDER_W[k].size
        W[k] = vec[off:off + n].reshape(FOUNDER_W[k].shape)
        off += n
    return W


# ================================================================
# Widened forward + decode (identical to the widen
# build's forward_wide / decode)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_wide(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1"):
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        out = 0.0
        for h in range(HEADS):
            p = f"{blk}.h{h}"
            Q = Hb @ W[f"{p}.W_q"] + W[f"{p}.b_q"]
            K = Hb @ W[f"{p}.W_k"] + W[f"{p}.b_k"]
            V = Hb @ W[f"{p}.W_v"] + W[f"{p}.b_v"]
            S = np.einsum("btj,bsj->bts", Q, K) * SCALE_W
            S = S - S.max(axis=-1, keepdims=True)
            E = np.exp(S)
            Atn = E / E.sum(axis=-1, keepdims=True)
            Ah = np.einsum("bts,bsi->bti", Atn, V)
            out = out + Ah @ W[f"{p}.W_o"] + W[f"{p}.b_o"]
        A = A + out
        Hf = ln_np(A, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"]
                        + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
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
            np.einsum("bi,bni->bn", q, Y) * SCALE_W)
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[arB, pick] = True
        last = Y[arB, pick]
    return picks


def mean_gap(vec, X, opt):
    W = unpack(vec)
    Y = forward_wide(W, X)
    picks = decode_general(W, Y)
    lens = be.tour_lengths_batched(X, picks[None, :, :])[0]
    return float((lens / opt - 1.0).mean())


# ================================================================
# SANITY GATE: the founder, scored on the fixed
# validation set, must reproduce the champion's
# 6.8959% (the preservation gate's value).
# ================================================================
founder_val = mean_gap(founder, X_val, OPT_VAL)
log(f"sanity gate: founder validation gap "
    f"{founder_val*100:.6f}% vs champion "
    f"{CHAMP_GAP_PCT:.4f}%")
if abs(founder_val * 100 - CHAMP_GAP_PCT) > 5e-5 + 1e-9:
    log(f"SANITY GATE FAILED: founder scores "
        f"{founder_val*100:.6f}%, champion is "
        f"{CHAMP_GAP_PCT:.4f}%. STOPPING; no evolution "
        f"run.")
    sys.exit(3)
log("sanity gate passed")

# ================================================================
# Training pool (drawn once from the run seed;
# identical scheme to the Frankenstein 4 evolution)
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
    "flat_mask_added": flat_added.astype(int).tolist(),
    "genes_total": int(founder.size),
    "genes_base": int(len(BASE_IDX)),
    "genes_added": int(len(ADDED_IDX)),
    "rule": "base = the champion's original units / "
            "heads / stream dimensions at their "
            "widened preserving values (frozen); "
            "added = the widen's duplicated copies "
            "(mutate)",
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
        f"the widened Frankenstein 4 genome")

founder_base = founder[BASE_IDX].copy()


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

    # base integrity: no base gene may ever move
    assert bool((pop[:, BASE_IDX]
                 == founder_base).all()), \
        f"base moved at generation {gen}"
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
        m = rng.random(len(ADDED_IDX)) < MUT_P
        if m.any():
            child[ADDED_IDX[m]] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + COMPLETE marker
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_added_changed = int((best_val_genome[ADDED_IDX]
                       != founder[ADDED_IDX]).sum())
assert bool((best_val_genome[BASE_IDX]
             == founder_base).all())
kept = bool(best_val_gap * 100 < CHAMP_GAP_PCT)
keep_read = (
    f"KEEP: the widened line's final best "
    f"{best_val_gap*100:.4f}% beats the champion's "
    f"{CHAMP_GAP_PCT:.4f}%"
    if kept else
    f"REVERT: the widened line's final best "
    f"{best_val_gap*100:.4f}% does not beat the "
    f"champion's {CHAMP_GAP_PCT:.4f}%")
meta = {
    "kind": "frankenstein_4_widened_champion",
    "date": "2026-11-25",
    "base": "frankenstein_4_widened_genome.npz",
    "evolution": "population 100, added genes only "
                 "(2392 of 2896) mutated at p=0.10, "
                 "sigma 0.15; base frozen; tournament "
                 "3; elites 2; 100 generations; "
                 "rotating 200-problem training "
                 "batches from a 2,000-problem pool, "
                 f"seed {EVOLVE_SEED} (the "
                 "Frankenstein 4 evolution protocol, "
                 "mirrored)",
    "champion_gap_pct": CHAMP_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "added_genes_changed_vs_founder": n_added_changed,
    "gene_order": KEYS,
    "genes_total": int(founder.size),
    "genes_base": int(len(BASE_IDX)),
    "genes_added": int(len(ADDED_IDX)),
    "attn_scale": SCALE_W, "ln_eps": EPS,
    "keep_revert": keep_read,
}
np.savez(OUT / "frankenstein_4_widened_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein4_widened_evolution",
    "generations": GENS,
    "champion_gap_pct": CHAMP_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "added_genes_changed_vs_founder": n_added_changed,
    "champion": "frankenstein_4_widened_champion.npz",
    "keep_revert": keep_read,
    "history": history,
}, indent=1))
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (champion "
    f"{CHAMP_GAP_PCT:.2f}%); champion vaulted, COMPLETE "
    f"marker written; added genes changed "
    f"{n_added_changed} of {len(ADDED_IDX)}, base "
    f"bit-identical")
log(f"KEEP/REVERT READ: {keep_read}")
