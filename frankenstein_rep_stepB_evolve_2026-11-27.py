#!/usr/bin/env python3
# REPLICATION (2026-11-27): adapted copy of
# frankenstein_stepB_evolve_2026-11-26.py - replication step B
# genome/deepen JSON, evolve seed 20261321, outputs under
# engine_runs/frankenstein_replication/stepB_evolution/. See
# frankenstein_replication_log_2026-11-27.md.
"""STEP B (breeding) of the deep-narrow
Frankenstein cycle (2026-11-26): breed in the
narrow block that frankenstein_stepB_deepen_
2026-11-26.py appended to the Step A2
Frankenstein.

SUBJECT: frankenstein_stepB_genome.npz - the
step A2 evolved champion (788 trunk genes,
validation gap 8.0009%) with ONE new narrow
block b5 appended after the joining block,
before the merged decoder (136 genes, filler
std 0.03, seed 20261220). Birth gap of the
deepened network: recorded in
frankenstein_stepB_deepen_2026-11-26.json.

PROTOCOL (the Frankenstein seam-evolution
protocol, mask = the new block's genes):
population 100 founded on the deepened
genome; ONLY the new block's 136 genes mutate
(p = 0.10 per gene per child, Gaussian sigma
0.15); the 788 trunk genes NEVER move -
asserted every generation and re-verified at
the end against the subject champion itself.
Selection: tournament 3; top 2 elites copied
verbatim; no crossover. FITNESS: mean gap
against Held-Karp optima on a rotating batch
of 200 training problems from a fixed pool of
2,000 drawn once from the run seed; every 10
generations (and at the last) the generation's
best is scored on the fixed 200-problem
validation set (val_seed 20261116) and logged;
best-validation genome kept. BUDGET: 100
generations. Seed 20261221.

SANITY GATE: the founder's validation gap
must reproduce the birth gap recorded in the
deepen JSON before the run counts.

Per-generation atomic checkpoint in
engine_runs/frankenstein_stepB_evolution/;
at the end the best-validation genome is
vaulted as frankenstein_stepB_evolved_
champion.npz, a COMPLETE marker and the run
summary (frankenstein_stepB_summary_2026-11-
26.json) are written.

LAUNCH (standing sponsor rule, fully
detached):
  setsid nohup python3 \
    frankenstein_stepB_evolve_2026-11-26.py \
    >> engine_runs/frankenstein_stepB_evolution/run.log \
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


CFG = HERE / "bert_engine_config_rep_deep50.json"
OUT = HERE / "engine_runs" / "frankenstein_replication" / "stepB_evolution"
POP_N = 100
GENS = 100
MUT_P = 0.10
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
EVOLVE_SEED = 20261321
VAL_EVERY = 10
BLOCKS = tuple(f"b{k}" for k in range(6))     # b0..b5
SUBJECT_GAP_PCT = 7.5641

# ================================================================
# Setup: domain (validation set), founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima
SCALE = arch.attn_scale
EPS = arch.ln_eps

DJ = json.load(open(
    HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepB_deepen_2026-11-27.json"))
BIRTH_GAP_PCT = DJ["birth_gap_pct"]
N_TRUNK = int(DJ["trunk_genes"])
N_NEW = int(DJ["new_block_genes"])

zg = np.load(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepB_genome.npz")
META = json.loads(str(zg["meta"]))
KEYS = [k for k in zg.files if k != "meta"]
assert KEYS == META["gene_order"]
FOUNDER_W = {k: np.asarray(zg[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
N_GENES = founder.size
assert N_GENES == N_TRUNK + N_NEW, \
    f"{N_GENES} genes != {N_TRUNK} trunk + {N_NEW} new"

# the subject champion, for the end-of-run
# trunk verification against the subject
# itself (not just the founder)
zgA = np.load(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepA_genome.npz")
KEYS_A = [k for k in zgA.files if k != "meta"]
zs = np.load(HERE / "engine_runs"
             / "frankenstein_replication"
             / "stepA_evolution"
             / "frankenstein_rep_stepA_evolved_champion.npz")
SUBJ_W = {k: np.asarray(zs[k], dtype=float) for k in KEYS_A}
subj_flat = np.concatenate([SUBJ_W[k].ravel() for k in KEYS_A])
assert np.array_equal(founder[:N_TRUNK], subj_flat), \
    "founder trunk is not the subject champion"

# mask: the new block's genes mutate; the
# trunk is frozen
NEW_IDX = np.arange(N_TRUNK, N_GENES)
TRUNK_IDX = np.arange(0, N_TRUNK)
log(f"mask: {N_GENES} genes, {len(NEW_IDX)} new-block "
    f"(mutate), {len(TRUNK_IDX)} trunk (frozen)")


def unpack(vec):
    W = {}
    off = 0
    for k in KEYS:
        n = FOUNDER_W[k].size
        W[k] = vec[off:off + n].reshape(FOUNDER_W[k].shape)
        off += n
    return W


# ================================================================
# Forward + decode (verbatim conventions from
# the Frankenstein builds, block list
# generalized to b0..b5)
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
# SANITY GATE: the founder must reproduce the
# recorded birth gap before the run counts.
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


flat_mask_new = np.zeros(N_GENES, dtype=bool)
flat_mask_new[NEW_IDX] = True
atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_new_block": flat_mask_new.astype(int).tolist(),
    "genes_total": int(N_GENES),
    "genes_trunk_frozen": int(len(TRUNK_IDX)),
    "genes_new_block_mutating": int(len(NEW_IDX)),
    "rule": "the appended block b5's genes mutate "
            "(True); the step A2 trunk is frozen "
            "(False)",
}, indent=1))

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
        f"the step B deepened genome")

founder_trunk = founder[TRUNK_IDX].copy()


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
# Evolution (the seam-evolution loop; mask =
# the new block's genes)
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

    assert bool((pop[:, TRUNK_IDX]
                 == founder_trunk).all()), \
        f"trunk moved at generation {gen}"
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
        m = rng.random(len(NEW_IDX)) < MUT_P
        if m.any():
            child[NEW_IDX[m]] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + COMPLETE
# marker + run summary
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_new_changed = int((best_val_genome[NEW_IDX]
                     != founder[NEW_IDX]).sum())
trunk_ok_founder = bool((best_val_genome[TRUNK_IDX]
                         == founder_trunk).all())
trunk_ok_subject = bool((best_val_genome[TRUNK_IDX]
                         == subj_flat).all())
assert trunk_ok_founder and trunk_ok_subject
walk = [{"gen": r["gen"], "val_gap_pct": r["val_gap_pct"]}
        for r in history if "val_gap_pct" in r]
meta = {
    "kind": "frankenstein_rep_stepB_evolved_champion",
    "date": "2026-11-27",
    "base": "frankenstein_rep_stepB_genome.npz",
    "subject": "step A2 evolved champion (gap "
               f"{SUBJECT_GAP_PCT}%) + appended narrow "
               "block b5",
    "evolution": "population 100, the new block's "
                 "genes only mutated at p=0.10, sigma "
                 "0.15; trunk frozen; tournament 3; "
                 "elites 2; 100 generations; rotating "
                 "200-problem training batches from a "
                 "2,000-problem pool, seed "
                 f"{EVOLVE_SEED} (the Frankenstein seam-"
                 "evolution protocol, mask = the new "
                 "block)",
    "birth_gap_pct": BIRTH_GAP_PCT,
    "subject_gap_pct": SUBJECT_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "new_block_genes_changed_vs_founder": n_new_changed,
    "trunk_bit_identical_to_subject": trunk_ok_subject,
    "gene_order": KEYS,
    "genes_total": int(N_GENES),
    "genes_trunk": int(len(TRUNK_IDX)),
    "genes_new_block": int(len(NEW_IDX)),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_rep_stepB_evolved_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_rep_stepB_evolution",
    "generations": GENS,
    "birth_gap_pct": BIRTH_GAP_PCT,
    "subject_gap_pct": SUBJECT_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "new_block_genes_changed_vs_founder": n_new_changed,
    "trunk_bit_identical_to_subject": trunk_ok_subject,
    "champion": "frankenstein_rep_stepB_evolved_champion.npz",
    "history": history,
}, indent=1))
summary = {
    "run": "frankenstein_stepB (step B: stack one "
           "narrow block on the step A2 Frankenstein "
           "and breed the new block in)",
    "date": "2026-11-27",
    "subject": "engine_runs/frankenstein_stepA2_"
               "evolution/frankenstein_stepA2_evolved_"
               "champion.npz",
    "subject_gap_pct": SUBJECT_GAP_PCT,
    "new_block": {"name": "b5", "position": "after the "
                  "joining block (b4), before the "
                  "merged decoder", "heads": 1,
                  "ffn_hidden": 4, "stream_width": 4,
                  "genes": int(len(NEW_IDX)),
                  "fill_seed": 20261320,
                  "fill_sigma": 0.03},
    "genes": {"total": int(N_GENES),
              "trunk": int(len(TRUNK_IDX)),
              "new_block": int(len(NEW_IDX))},
    "blocks_total": len(BLOCKS),
    "birth_gap_pct": BIRTH_GAP_PCT,
    "generations": GENS,
    "evolve_seed": EVOLVE_SEED,
    "validation_walk": walk,
    "final_gap_pct": round(best_val_gap * 100, 4),
    "final_gap_generation": best_val_gen,
    "trunk_bit_identical_to_subject": trunk_ok_subject,
    "new_block_genes_changed": n_new_changed,
    "champion": "engine_runs/frankenstein_stepB_"
                "evolution/frankenstein_stepB_evolved_"
                "champion.npz",
}
with open(HERE / "engine_runs" / "frankenstein_replication" / "frankenstein_rep_stepB_summary_2026-11-27.json",
          "w") as f:
    json.dump(summary, f, indent=1)
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (birth "
    f"{BIRTH_GAP_PCT:.2f}%, subject "
    f"{SUBJECT_GAP_PCT:.2f}%); champion vaulted, "
    f"COMPLETE marker and summary written; new-block "
    f"genes changed {n_new_changed} of {len(NEW_IDX)}, "
    f"trunk bit-identical to the subject")
