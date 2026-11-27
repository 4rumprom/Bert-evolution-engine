#!/usr/bin/env python3
"""Evolution of the Frankenstein step A REDO stitch
(deep-narrow line, 2026-11-26), with the exact
protocol used for the Frankenstein 4 builds
(frankenstein4_second_evolve_2026-11-26.py):
the stitching genes mutate, the backbone
never does.

SUBJECT: frankenstein_stepA2_genome.npz (the
step A stitch: 4 stitched blocks + 1 joining
block + merged decoder; birth gap recorded
in frankenstein_stepA2_stitch_2026-11-26.json;
parents: deep50_stepA species 1 at 17.1420%
and species 0 at 19.6989%).

MASK (derived here from the build records,
rule: exact-from-a-parent = backbone,
drawn-as-filler = stitching), verified by
value exactly as in the earlier builds:
  - embed.W_in and the stitched-block
    matrices b0..b3: entries the stitch build
    copied exact are the parents' path
    synapses (recorded in the stitch JSON),
    placed by the territory maps in the
    genome meta -> backbone; every other
    entry was drawn as filler -> stitching.
  - embed.b_in, the stitched blocks' layer
    norms, biases and FFN unit biases: the
    parents' own values at every territory
    position -> backbone there.
  - b4 (joining block): drawn entirely as
    filler -> all stitching.
  - head (merged decoder): W_qd diagonal
    territory blocks are the parents' full
    W_qd (backbone); off-diagonal cross
    blocks are tiny-random filler
    (stitching); b_qd and v_start are the
    parents' values placed by the territory
    maps (backbone).

POPULATION: 100 genomes founded on the step
A stitch genome. Single founder, no
crossover. MUTATION: only the stitching
genes, p = 0.10 per child, Gaussian sigma
0.15. Backbone genes never touched;
asserted every generation. SELECTION:
tournament 3; top 2 elites copied verbatim.

FITNESS: mean gap against Held-Karp optima
on a rotating batch of 200 training
problems from a fixed pool of 2,000 drawn
once from the run seed; every 10
generations (and at the last) the
generation's best is scored on the fixed
200-problem validation set (val_seed
20261116) and logged; best-validation
genome kept.

BUDGET: 200 generations (sponsor change,
2026-11-26: longer seam evolution).
Per-generation atomic checkpoint in
engine_runs/frankenstein_stepA2_evolution/;
at the end the best-validation genome is
vaulted as
frankenstein_stepA2_evolved_champion.npz
and a COMPLETE marker is written.

SANITY GATE: the founder's validation gap
must reproduce the birth gap recorded in
the stitch JSON before the run counts.

LAUNCH (standing sponsor rule, fully
detached):
  setsid nohup python3 \
    frankenstein_stepA2_evolve_2026-11-26.py \
    >> engine_runs/frankenstein_stepA2_evolution/run.log \
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


CFG = HERE / "bert_engine_config_deep50_stepA.json"
RUN = HERE / "engine_runs" / "deep50_stepA"
OUT = HERE / "engine_runs" / "frankenstein_stepA2_evolution"
POP_N = 100
GENS = 200
MUT_P = 0.10
MUT_SIGMA = 0.15
ELITES = 2
TOURNAMENT = 3
TRAIN_BATCH = 200
TRAIN_POOL = 2000
EVOLVE_SEED = 20261213
VAL_EVERY = 10
N_BLOCKS = 4
STITCH_BLOCKS = tuple(f"b{k}" for k in range(N_BLOCKS + 1))
JOINING = f"b{N_BLOCKS}"

# ================================================================
# Setup: domain (validation set), parents,
# founder
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

SJ = json.load(open(
    HERE / "frankenstein_stepA2_stitch_2026-11-26.json"))
BIRTH_GAP_PCT = SJ["frankenstein_stepA2"]["validation_gap_pct"]

zg = np.load(HERE / "frankenstein_stepA2_genome.npz")
META = json.loads(str(zg["meta"]))
KEYS = [k for k in zg.files if k != "meta"]
FOUNDER_W = {k: np.asarray(zg[k], dtype=float) for k in KEYS}
founder = np.concatenate([FOUNDER_W[k].ravel() for k in KEYS])
N_GENES = founder.size
OFFSETS = {}
_off = 0
for k in KEYS:
    OFFSETS[k] = _off
    _off += FOUNDER_W[k].size


def flat_idx(key, idx2):
    return OFFSETS[key] + int(np.ravel_multi_index(
        idx2, FOUNDER_W[key].shape))


# parents (generic weights, generalized from
# the second build's evolve script)
zh = np.load(HERE / "frankenstein_stepA_parent_host_evolved.npz")
host_genome = np.asarray(zh["genome"], dtype=float)
zd = np.load(HERE / "frankenstein_stepA_parent_donor_evolved.npz")
donor_genome = np.asarray(zd["genome"], dtype=float)
host_id = int(META["host_species"])
donor_id = int(META["donor_species"])
assert host_id == 1 and donor_id == 0
gaps = {host_id: float(domain.validation_detail(
            layout, host_genome)["mean_gap"]),
        donor_id: float(domain.validation_detail(
            layout, donor_genome)["mean_gap"])}
log(f"parents (each evolved +100 gens before the "
    f"stitch): host species {host_id} gap "
    f"{gaps[host_id]*100:.4f}%, donor species "
    f"{donor_id} gap {gaps[donor_id]*100:.4f}%")


def parent_weights(genome):
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


HW = parent_weights(host_genome)
DW = parent_weights(donor_genome)
h_map = {int(k): v for k, v in META["host_coord_map"].items()}
d_map = {int(k): v for k, v in META["donor_coord_map"].items()}
hu_maps = {int(bk): {int(u): v for u, v in m.items()}
           for bk, m in META["host_unit_maps"].items()}
du_maps = {int(bk): {int(u): v for u, v in m.items()}
           for bk, m in META["donor_unit_maps"].items()}

# ================================================================
# Mask derivation (rule: exact-from-a-parent =
# backbone; drawn-as-filler = stitching),
# verified by value against the parents and
# the stitch record.
# ================================================================
backbone = np.zeros(N_GENES, dtype=bool)


def mark(key, idx2):
    backbone[flat_idx(key, idx2)] = True


def check_eq(key, idx2, value, what):
    got = FOUNDER_W[key][idx2]
    assert got == value, \
        f"mask verification failed: {what} {key}{idx2}: " \
        f"founder {got!r} != parent {value!r}"


for tag, P, PW, cmap in (
        ("host", SJ["parents"]["host"], HW, h_map),
        ("donor", SJ["parents"]["donor"], DW, d_map)):
    umaps = hu_maps if tag == "host" else du_maps
    for e in P["path_weight_synapses"]:
        entry, i, j = e["entry"], e["i"], e["j"]
        if entry == "head.W_qd":
            continue  # decoder handled below
        if entry == "embed.W_in":
            pos = (i, cmap[j])            # rows are shared
        elif entry.endswith(".W_1"):
            pos = (cmap[i], umaps[int(entry[1])][j])
        elif entry.endswith(".W_2"):
            pos = (umaps[int(entry[1])][i], cmap[j])
        else:
            pos = (cmap[i], cmap[j])
        check_eq(entry, pos, PW[entry][i, j],
                 f"{tag} path synapse")
        mark(entry, pos)

for nm in ["embed.b_in"] + [
        f"b{k}.{s}" for k in range(N_BLOCKS)
        for s in ("lnA_gain", "lnA_bias", "lnF_gain",
                  "lnF_bias", "b_q", "b_k", "b_v", "b_o",
                  "b_2")]:
    for cmap, PW in ((h_map, HW), (d_map, DW)):
        for k_old, k_new in cmap.items():
            check_eq(nm, (k_new,), PW[nm][k_old],
                     "territory vector")
            mark(nm, (k_new,))
for k in range(N_BLOCKS):
    for umaps, PW in ((hu_maps, HW), (du_maps, DW)):
        for u_old, u_new in umaps[k].items():
            check_eq(f"b{k}.b_1", (u_new,),
                     PW[f"b{k}.b_1"][u_old],
                     "territory vector")
            mark(f"b{k}.b_1", (u_new,))

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
# cross-check the head against the stitch record
assert np.array_equal(FOUNDER_W["head.W_qd"],
                      np.array(SJ["merged_decoder"]["W_qd_full"]))
assert np.array_equal(FOUNDER_W["head.b_qd"],
                      np.array(SJ["merged_decoder"]["b_qd"]))
assert np.array_equal(FOUNDER_W["head.v_start"],
                      np.array(SJ["merged_decoder"]["v_start"]))

flat_mask = ~backbone          # True = stitching (mutates)
STITCH_IDX = np.flatnonzero(flat_mask)
BACKBONE_IDX = np.flatnonzero(backbone)
assert len(STITCH_IDX) + len(BACKBONE_IDX) == N_GENES
log(f"mask derived and verified by value: {N_GENES} "
    f"genes, {len(STITCH_IDX)} stitching (mutate), "
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
# Forward + decode (verbatim conventions from
# the Frankenstein builds, block list
# generalized)
# ================================================================
def ln_np(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def forward_general(W, X):
    A = X @ W["embed.W_in"] + W["embed.b_in"]
    for blk in STITCH_BLOCKS:
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


atomic_write(OUT / "mask.json", json.dumps({
    "gene_order": KEYS,
    "flat_mask_stitching": flat_mask.astype(int).tolist(),
    "genes_total": int(N_GENES),
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "rule": "exact-from-a-parent = backbone (False); "
            "drawn-as-filler = stitching (True)",
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
        f"the Frankenstein step A genome")

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
# Evolution (verbatim loop from the earlier
# builds; budget 200 generations)
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

    assert bool((pop[:, BACKBONE_IDX]
                 == founder_backbone).all()), \
        f"backbone moved at generation {gen}"
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
        m = rng.random(len(STITCH_IDX)) < MUT_P
        if m.any():
            child[STITCH_IDX[m]] += rng.normal(
                0.0, MUT_SIGMA, size=int(m.sum()))
        new_pop[slot] = child
    pop = new_pop

# ================================================================
# Vault the best-validation genome + COMPLETE
# marker
# ================================================================
BW = unpack(best_val_genome)
final_val = mean_gap(best_val_genome, X_val, OPT_VAL)
assert abs(final_val - best_val_gap) < 1e-12
n_stitch_changed = int((best_val_genome[STITCH_IDX]
                        != founder[STITCH_IDX]).sum())
assert bool((best_val_genome[BACKBONE_IDX]
             == founder_backbone).all())
meta = {
    "kind": "frankenstein_stepA2_evolved_champion",
    "date": "2026-11-26",
    "base": "frankenstein_stepA2_genome.npz",
    "evolution": "population 100, stitching genes "
                 "only mutated at p=0.10, sigma 0.15; "
                 "backbone frozen; tournament 3; "
                 "elites 2; 200 generations; rotating "
                 "200-problem training batches from a "
                 "2,000-problem pool, seed "
                 f"{EVOLVE_SEED} (the Frankenstein 4 "
                 "evolution protocol, mirrored; budget "
                 "raised to 200 generations by the "
                 "sponsor, 2026-11-26)",
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "gene_order": KEYS,
    "genes_total": int(N_GENES),
    "genes_backbone": int(len(BACKBONE_IDX)),
    "genes_stitching": int(len(STITCH_IDX)),
    "attn_scale": SCALE, "ln_eps": EPS,
}
np.savez(OUT / "frankenstein_stepA2_evolved_champion.npz",
         **{k: BW[k] for k in KEYS},
         meta=np.array(json.dumps(meta)))
atomic_write(OUT / "COMPLETE", json.dumps({
    "run": "frankenstein_stepA2_evolution",
    "generations": GENS,
    "birth_gap_pct": BIRTH_GAP_PCT,
    "best_validation_gap_pct": round(best_val_gap * 100, 4),
    "best_validation_generation": best_val_gen,
    "stitching_genes_changed_vs_founder": n_stitch_changed,
    "champion": "frankenstein_stepA2_evolved_champion.npz",
    "history": history,
}, indent=1))
log(f"DONE: best validation gap {best_val_gap*100:.2f}% "
    f"at generation {best_val_gen} (birth "
    f"{BIRTH_GAP_PCT:.2f}%); champion vaulted, COMPLETE "
    f"marker written; stitching genes changed "
    f"{n_stitch_changed} of {len(STITCH_IDX)}, backbone "
    f"bit-identical")
