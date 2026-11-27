#!/usr/bin/env python3
"""ANCESTRAL ATTRIBUTION, quick test (sponsor idea,
2026-11-24): a branch may be recoverable from how its
units ORIGINATED, not only from what they do now.

Ground-truth ancestry on hand: the double-widen
treatment champion (14.00% gap). At the widen, every
FFN unit was split Net2WiderNet style: unit i kept its
slot, its exact copy landed at i+8 (W_1 and b_1 copied,
W_2 halved in both); the single head split the same
way into heads 0 and 1. So the 18 units form 9 known
SISTER PAIRS sharing one mother unit. Five FFN
mothers (u3-u7) and the head were freeze-marked, so
their daughters never mutated and remain identical
twins; three FFN pairs (mothers u0, u1, u2) evolved
freely for 100 generations after the split.

Question: do sisters still behave as one unit --
attribution profiles tracking together, silencing
drops concordant -- more than unrelated pairs? Read
against a permutation control (random pairings of the
same units, same pair count). Sisters reported two
ways: all 9 pairs, and the 3 freely evolved pairs
alone (the frozen twins are identical by
construction, so they prove preservation, not
persistence under evolution).

Run under the venv python."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_attribution as ba
import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

cfg = be.EngineConfig.load(HERE / "bert_engine_config_d2_l1.json")
cfg = dataclasses.replace(cfg, d=4, ffn_hidden=16,
                          initial_blocks=1, initial_heads=2,
                          run_seed=20262200, draw_seed=20262201)
backend = get_backend("numpy", "cpu", "x64")
domain = tsp.TSPDomain(cfg, backend)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "double_widen_d2_bandmarks_"
            "treatment_champion.npz")
genome = z["genome"]

table = be.silence_table(
    layout, genome,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
rows = table["units"]
units, A = ba.attribution_profiles(domain, layout, genome,
                                   tau=0.5)
assert [u["name"] for u in units] == [r["name"] for r in rows]
Ac = A - A.mean(axis=0, keepdims=True)
drops = np.array([r["drop"] for r in rows])
gap = -table["champion_score"]

# sister pairs by (kind, index): FFN (i, i+8), head (0, 1)
pos = {}
for i, r in enumerate(rows):
    pos[(r["kind"], r["index"])] = i
pairs = [(pos[("ffn", i)], pos[("ffn", i + 8)])
         for i in range(8)]
pairs.append((pos[("head", 0)], pos[("head", 1)]))
frozen_mothers = {3, 4, 5, 6, 7}
free_pairs = [p for k, p in enumerate(pairs)
              if k == 8 or k not in frozen_mothers]
# head pair is a frozen mother too; free = FFN u0,u1,u2
free_pairs = [pairs[k] for k in (0, 1, 2)]


def cos(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else 0.0


def pair_stats(pl):
    coh = float(np.mean([cos(Ac[i], Ac[j]) for i, j in pl]))
    a = np.array([drops[i] for i, _ in pl])
    b = np.array([drops[j] for _, j in pl])
    dc = float(np.corrcoef(a, b)[0, 1]) if len(pl) > 2 else None
    return coh, dc


def null(pl, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    idxs = sorted({i for p in pl for i in p})
    cohs = []
    for _ in range(n):
        perm = rng.permutation(idxs)
        rp = [(perm[k], perm[k + 1])
              for k in range(0, len(perm) - 1, 2)]
        cohs.append(np.mean([cos(Ac[i], Ac[j])
                             for i, j in rp]))
    return float(np.mean(cohs)), float(np.max(cohs))


out = {"subject_gap_pct": round(gap * 100, 2)}
for tag, pl in (("all_9_pairs", pairs),
                ("free_3_pairs", free_pairs)):
    coh, dc = pair_stats(pl)
    nm, nx = null(pl)
    out[tag] = {
        "mean_sister_profile_cosine": round(coh, 3),
        "random_pairing_mean": round(nm, 3),
        "random_pairing_max_of_2000": round(nx, 3),
        "sister_drop_correlation": (round(dc, 3)
                                    if dc is not None else None),
        "sister_drops_pct_of_gap":
            [[round(drops[i] / gap * 100, 1),
              round(drops[j] / gap * 100, 1)] for i, j in pl],
    }
print(json.dumps(out, indent=1))
(HERE / "ancestral_attribution_test_2026-11-24.json"
 ).write_text(json.dumps(out, indent=1))
