#!/usr/bin/env python3
"""CONTROL TEST (2026-11-27): does the
Net2WiderNet widen stage of the Frankenstein
pipeline actually matter?

Full-network evolution run DIRECTLY on the
original line's Step C champion, SKIPPING the
Net2WiderNet widen entirely. Same evolution
protocol as frankenstein_widened_evolve_
scaled_2026-11-27.py (the run that produced
5.2474% on the widened genome), different
founder and rate.

FOUNDER: engine_runs/frankenstein_stepC_
evolution/frankenstein_stepC_evolved_
champion.npz - the original line's Step C
champion (commonized Step B trunk + widened
backend block b5w, seam-evolved 200 gens),
1,040 genes, recorded gap 8.1018% on the
fixed 200-problem 10-city validation set
(val_seed 20261116).

POPULATION: 1,000 individuals, ONE species.
Member 0 = the founder verbatim. Members
1-999 = the founder plus exactly one mutation
application each (per-gene p = 25/1040,
Gaussian sigma 0.15), seed 20261350.

EVOLUTION, 200 generations: NOTHING frozen -
every one of the 1,040 genes can mutate in
every child. Per-gene rate = 25/1040 =
0.024038... - the SCALED dose: ~25 nudges per
child, the same dose that produced 5.2474% on
the 8,480-gene widened genome (there p =
0.003). Elites: top 2 copied verbatim.
Selection: tournament size 3. No crossover
(single-founder lineage, as in the
Frankenstein seam runs). FITNESS (the
Frankenstein evolve convention): mean gap
against Held-Karp optima on a rotating batch
of 200 training problems from a fixed pool of
2,000 drawn once from the run seed.
VALIDATION: the generation's training-best
genome is scored on the fixed validation set
every 10 generations (and at the last); the
best-on-validation genome is tracked (the
founder itself, measured at the sanity gate,
is the initial best) and vaulted at the end.

FORWARD: the Step C forward from
frankenstein_stepC_evolve_2026-11-26.py
(fused single-head trunk blocks b0-b4 +
two-head widened backend block b5w, einsum
form - the exact machinery that recorded the
founder's 8.1018%).

SANITY GATE: the founder's validation gap
must reproduce the recorded 8.1018% before
the run counts.

Per-generation atomic checkpoint in
engine_runs/frankenstein_stepC_evolution_
only/; at the end the best-validation genome
is vaulted as frankenstein_stepC_evolve_only_
champion.npz, a COMPLETE marker is written,
and the run summary goes to
frankenstein_stepC_evolve_only_2026-11-27
.json.

LAUNCH (standing sponsor rule, fully
detached via subprocess.Popen with
start_new_session=True):
  python3 \
    frankenstein_stepC_evolve_only_2026-11-27.py
  (log: engine_runs/frankenstein_stepC_
   evolution_only/run.log)
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
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


CFG = HERE / "bert_engine_config_deep50_stepA.json"
OUT = HERE / "engine_runs" / "frankenstein_stepC_evolution_only"
FOUNDER_FILE = (HERE / "engine_runs" / "frankenstein_stepC_evolution"
                / "frankenstein_stepC_evolved_champion.npz")
POP_N = 1000
GENS = 200
N_GENES_EXPECTED = 1040
MUT_P = 25 / N_GENES_EXPECTED
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
RUN_SEED = 20261350
VAL_EVERY = 10

# ================================================================
# Setup: domain (validation set), founder
# ================================================================
cfg = be.EngineConfig.load(CFG)
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
arch = be.initial_arch(cfg, 2)
domain = tsp.TSPDomain(cfg, backend)
X_val = domain.X_val
OPT_VAL = domain.optima

zf = np.load(FOUNDER_FILE)
META = json.loads(str(zf["meta"]))
KEYS = list(META["gene_order"])
assert all(k in zf.files for k in KEYS)
FOUNDER_W = {k: np.asarray(zf[k], dtype=float) for k in KEYS}
SHAPES = {k: FOUNDER_W[k].shape for k in KEYS}
SIZES = {k: FOUNDER_W[k].size for k in KEYS}
OFFS = {}
_off = 0
for k in KEYS:
    OFFS[k] = _off
    _off += SIZES[k]
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
N_GENES = founder.size
assert N_GENES == N_GENES_EXPECTED, \
    f"{N_GENES} genes != {N_GENES_EXPECTED}"
SCALE = float(META["attn_scale"])
EPS = float(META["ln_eps"])
assert abs(SCALE - float(arch.attn_scale)) < 1e-12
assert abs(EPS - float(arch.ln_eps)) < 1e-12
FOUNDER_GAP_PCT = float(META["best_validation_gap_pct"])
log(f"founder: {N_GENES} genes from "
    f"{FOUNDER_FILE.name} (kind {META['kind']}); "
    f"recorded gap {FOUNDER_GAP_PCT:.4f}%; "
    f"attn_scale {SCALE}, ln_eps {EPS}; "
    f"mut_p {MUT_P:.8f} "
    f"(~{MUT_P * N_GENES:.2f} nudges/child)")


def unpack(vec):
    return {k: vec[OFFS[k]:OFFS[k] + SIZES[k]].reshape(SHAPES[k])
            for k in KEYS}


# ================================================================
# Forward + greedy decode (the Step C network:
# fused trunk blocks b0-b4 + two-head b5w;
# the exact machinery of the Step C run)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in ("b0", "b1", "b2", "b3", "b4"):
        Hb = ln_np(A, W[f"{blk}.lnA_gain"],
                   W[f"{blk}.lnA_bias"], EPS)
        Q = Hb @ W[f"{blk}.W_q"] + W[f"{blk}.b_q"]
        K = Hb @ W[f"{blk}.W_k"] + W[f"{blk}.b_k"]
        V = Hb @ W[f"{blk}.W_v"] + W[f"{blk}.b_v"]
        S = np.einsum("btj,bsj->bts", Q, K) * SCALE
        S = S - S.max(axis=-1, keepdims=True)
        E = np.exp(S)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        A = A + Ah @ W[f"{blk}.W_o"] + W[f"{blk}.b_o"]
        Hf = ln_np(A, W[f"{blk}.lnF_gain"],
                   W[f"{blk}.lnF_bias"], EPS)
        A = A + np.tanh(Hf @ W[f"{blk}.W_1"]
                        + W[f"{blk}.b_1"]) @ W[f"{blk}.W_2"] \
            + W[f"{blk}.b_2"]
    blk = "b5w"
    Hb = ln_np(A, W[f"{blk}.lnA_gain"],
               W[f"{blk}.lnA_bias"], EPS)
    for h in (0, 1):
        Q = Hb @ W[f"{blk}.h{h}.W_q"] + W[f"{blk}.h{h}.b_q"]
        K = Hb @ W[f"{blk}.h{h}.W_k"] + W[f"{blk}.h{h}.b_k"]
        V = Hb @ W[f"{blk}.h{h}.W_v"] + W[f"{blk}.h{h}.b_v"]
        S = np.einsum("btj,bsj->bts", Q, K) * SCALE
        S = S - S.max(axis=-1, keepdims=True)
        E = np.exp(S)
        Atn = E / E.sum(axis=-1, keepdims=True)
        Ah = np.einsum("bts,bsi->bti", Atn, V)
        A = A + Ah @ W[f"{blk}.h{h}.W_o"] + W[f"{blk}.h{h}.b_o"]
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
# recorded gap before the run counts.
# ================================================================
founder_val = mean_gap(founder, X_val, OPT_VAL)
log(f"sanity gate: founder validation gap "
    f"{founder_val*100:.6f}% vs recorded "
    f"{FOUNDER_GAP_PCT:.4f}%")
if abs(founder_val * 100 - FOUNDER_GAP_PCT) > 5e-5 + 1e-9:
    log(f"SANITY GATE FAILED: founder scores "
        f"{founder_val*100:.6f}%, record says "
        f"{FOUNDER_GAP_PCT:.4f}%. STOPPING; no "
        f"evolution run.")
    sys.exit(3)
log("sanity gate passed")

# ================================================================
# Training pool (drawn once from the run seed;
# the Frankenstein evolve convention)
# ================================================================
rng_pool = np.random.default_rng(RUN_SEED)
X_pool = rng_pool.random((TRAIN_POOL, cfg.domain_config.get(
    "n_cities", 10), 2))
t_hk = time.time()
OPT_POOL = np.array([held_karp(X_pool[i])
                     for i in range(TRAIN_POOL)])
log(f"training pool: {TRAIN_POOL} problems from seed "
    f"{RUN_SEED}, optima computed in "
    f"{time.time() - t_hk:.1f}s")

# ================================================================
# Population state (fresh, or resumed)
# ================================================================
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "checkpoint.npz"
STATUS = OUT / "status.json"
CONFIG = {"pop_n": POP_N, "gens": GENS, "mut_p": MUT_P,
          "mut_sigma": MUT_SIGMA, "elites": ELITES,
          "tournament": TOURNAMENT,
          "train_batch": TRAIN_BATCH,
          "train_pool": TRAIN_POOL, "run_seed": RUN_SEED,
          "founder_file": str(FOUNDER_FILE.relative_to(HERE)),
          "founder_gap_pct": FOUNDER_GAP_PCT,
          "genes": int(N_GENES)}


def atomic_savez(path, **kw):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


rng = np.random.default_rng(RUN_SEED + 1000)
history = []
best_val_gap = founder_val
best_val_genome = founder.copy()
best_val_gen = -1  # -1 = the founder itself
start_gen = 0

if CKPT.exists():
    ck = np.load(CKPT, allow_pickle=True)
    saved_cfg = json.loads(str(ck["config"]))
    assert saved_cfg == CONFIG, \
        f"checkpoint config mismatch: {saved_cfg} vs {CONFIG}"
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
    rng_found = np.random.default_rng(RUN_SEED)
    pop = np.tile(founder, (POP_N, 1))
    for i in range(1, POP_N):
        m = rng_found.random(N_GENES) < MUT_P
        if m.any():
            pop[i, m] += rng_found.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
    assert np.array_equal(pop[0], founder)
    log(f"fresh start: population {POP_N} founded on "
        f"the champion (member 0 verbatim; members "
        f"1-{POP_N - 1} one mutation application "
        f"each, seed {RUN_SEED})")


def checkpoint(next_gen):
    atomic_savez(
        CKPT,
        pop=pop, next_gen=next_gen,
        best_val_gap=best_val_gap,
        best_val_genome=best_val_genome,
        best_val_gen=best_val_gen,
        history=np.array(json.dumps(history)),
        rng_state=np.array(json.dumps(
            rng.bit_generator.state)),
        config=np.array(json.dumps(CONFIG)))
    atomic_write(STATUS, json.dumps({
        "next_gen": next_gen, "gens_budget": GENS,
        "best_val_gap_pct": round(best_val_gap * 100, 4),
        "best_val_gen": best_val_gen,
        "last": history[-1] if history else None,
    }, indent=1))


# ================================================================
# Evolution: NOTHING frozen, one species
# ================================================================
for gen in range(start_gen, GENS):
    t_gen = time.time()
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
        rec["val_genome"] = "best-of-generation"
        if vg < best_val_gap:
            best_val_gap = vg
            best_val_genome = pop[bi].copy()
            best_val_gen = gen
        log(f"gen {gen}: train best "
            f"{fit[bi]*100:.2f}%, mean {fit.mean()*100:.2f}%; "
            f"VALIDATION best-of-gen {vg*100:.2f}% "
            f"(best so far {best_val_gap*100:.2f}% at "
            f"gen {best_val_gen}); gen time "
            f"{time.time() - t_gen:.0f}s")
    else:
        log(f"gen {gen}: train best {fit[bi]*100:.2f}%, "
            f"mean {fit.mean()*100:.2f}%; gen time "
            f"{time.time() - t_gen:.0f}s")
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

wall = time.time() - T0

# ================================================================
# Vault the best-validation genome + COMPLETE
# marker + run summary
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-9
n_changed = int((best_val_genome != founder).sum())
walk = [{"gen": r["gen"], "val_gap_pct": r["val_gap_pct"]}
        for r in history if "val_gap_pct" in r]
meta = {
    "kind": "frankenstein_stepC_evolve_only_champion",
    "date": "2026-11-27",
    "base": "engine_runs/frankenstein_stepC_evolution/"
            "frankenstein_stepC_evolved_champion.npz",
    "subject": "CONTROL TEST: the original line's "
               "Step C champion evolved directly, "
               "the Net2WiderNet widen SKIPPED",
    "evolution": "population 1000, ONE species "
                 "founded on the champion; NOTHING "
                 "frozen - all 1,040 genes mutate "
                 f"at p=25/1040={MUT_P:.8f}, sigma "
                 "0.15 in every child (~25 nudges "
                 "per child, the scaled dose that "
                 "produced 5.2474% on the widened "
                 "genome); tournament 3; elites 2; "
                 "no crossover; 200 generations; "
                 "rotating 200-problem training "
                 "batches from a 2,000-problem "
                 "pool, seed 20261350 (the "
                 "Frankenstein evolve protocol, "
                 "mask = everything)",
    "founder_gap_pct": round(founder_val * 100, 4),
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "genes_changed_vs_founder": n_changed,
    "gene_order": KEYS,
    "genes_total": int(N_GENES),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_stepC_evolve_only_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_stepC_evolution_only",
    "generations": GENS,
    "population": POP_N,
    "founder_gap_pct": round(founder_val * 100, 4),
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "genes_changed_vs_founder": n_changed,
    "champion": "frankenstein_stepC_evolve_only_champion.npz",
    "wall_seconds": round(wall, 1),
    "history": history,
}, indent=1))
summary = {
    "run": "CONTROL TEST: full-network evolution "
           "of the original line's Step C champion, "
           "Net2WiderNet widen skipped (same "
           "protocol as frankenstein_widened_"
           "evolve_scaled_2026-11-27.py; founder "
           "and rate differ: 1,040 genes, "
           "p = 25/1040, seed 20261350)",
    "date": "2026-11-27",
    "founder": "engine_runs/frankenstein_stepC_"
               "evolution/frankenstein_stepC_"
               "evolved_champion.npz (Step C "
               "champion, original line)",
    "founder_gap_pct": round(founder_val * 100, 4),
    "mut_p": MUT_P,
    "expected_nudges_per_child": round(MUT_P * N_GENES, 2),
    "config": CONFIG,
    "validation_walk": walk,
    "final_best_gap_pct": round(best_val_gap * 100, 4),
    "final_best_generation": best_val_gen,
    "genes_changed_vs_founder": n_changed,
    "wall_seconds": round(wall, 1),
    "champion": "engine_runs/frankenstein_stepC_"
                "evolution_only/frankenstein_stepC_"
                "evolve_only_champion.npz",
}
with open(HERE / "frankenstein_stepC_evolve_only_"
                "2026-11-27.json", "w") as f:
    json.dump(summary, f, indent=1)
log(f"DONE in {wall:.0f}s: best validation gap "
    f"{best_val_gap*100:.4f}% at generation "
    f"{best_val_gen} (founder "
    f"{founder_val*100:.4f}%); {n_changed} of "
    f"{N_GENES} genes changed vs founder; champion "
    f"vaulted, COMPLETE marker and summary written")
