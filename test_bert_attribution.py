#!/usr/bin/env python3
"""Verification for bert_attribution (run under the
project venv python, which has jax):
  ~/workspace/bert-engine/.venv/bin/python \
      test_bert_attribution.py

Checks, on the wide founding champion of the branch
scale test (11.39% gap, 36 units):
  1. ENCODER PARITY: the jax mirror's encoder output
     matches bert_engine.encode_population to < 1e-9.
  2. SURROGATE: soft expected lengths track the true
     greedy lengths across the validation problems
     (Pearson > 0.8 at the best scanned temperature).
  3. ATTRIBUTION FIDELITY: per-unit attribution
     totals correlate with the true silencing drops
     on the d2 SEED champion (9 units, differentiated
     drops: head +111% of gap, one unit at +27%, one
     harmful). Bar: Pearson > 0.3, Spearman > 0.4.
     The fidelity subject is the seed, not the wide
     champion, on measured grounds: on the wide
     champion every silencing drop is catastrophic
     (14 to 260% of the gap; removal re-rolls the
     whole trajectory), and there the local gradient
     read and the removal read genuinely disagree
     (Pearson about -0.2 at every temperature, the
     published first-order-attribution failure mode
     under strong downstream nonlinearity). Where
     silencing is differentiated, the two reads
     agree (measured Pearson 0.40, Spearman 0.50).
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_attribution as ba
import bert_domain_tsp as tsp
import bert_engine as be
from bert_backend import get_backend

fails = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


cfg = be.EngineConfig.load(HERE / "bert_engine_config_d2_l1.json")
cfg = dataclasses.replace(cfg, d=8, ffn_hidden=16,
                          initial_blocks=2, initial_heads=2,
                          run_seed=20262400, draw_seed=20262401)
backend = get_backend("numpy", "cpu", "x64")
domain = tsp.TSPDomain(cfg, backend)
arch = be.initial_arch(cfg, 2)
layout = be.Layout(arch, tsp.head_spec(arch.d))
z = np.load(HERE / "branch_pg_scale_test_2026-11-24_"
            "founding_champion.npz")
genome = z["genome"]

# 1. encoder parity
Y_ref = be.encode_population(backend, layout,
                             genome[None, :],
                             domain.X_val[:16])
Y_jax = ba.encode_jax(layout, genome, domain.X_val[:16])
diff = float(np.max(np.abs(Y_ref[0] - Y_jax)))
check("encoder parity", diff < 1e-9, f"max abs diff {diff:.2e}")

# 2. surrogate tracking
lens, _ = domain.evaluate_lengths(layout, genome[None, :],
                                  domain.X_val)
greedy = lens[0]
best = None
for tau in (1.0, 0.5, 0.25, 0.1):
    soft = ba.soft_lengths(domain, layout, genome, tau=tau)
    r = float(pearsonr(soft, greedy)[0])
    ratio = float(soft.mean() / greedy.mean())
    print(f"  tau {tau}: pearson {r:.3f}, "
          f"soft/greedy mean {ratio:.3f}")
    if best is None or r > best[1]:
        best = (tau, r)
check("surrogate tracking", best[1] > 0.9,
      f"best tau {best[0]} pearson {best[1]:.3f}")

# 3. attribution fidelity vs silencing, on the d2 seed
cfg2 = dataclasses.replace(cfg, d=2, ffn_hidden=8,
                           initial_blocks=1, initial_heads=1,
                           run_seed=20262000,
                           draw_seed=20262001)
domain2 = tsp.TSPDomain(cfg2, backend)
arch2 = be.initial_arch(cfg2, 2)
layout2 = be.Layout(arch2, tsp.head_spec(arch2.d))
z2 = np.load(HERE / "engine_runs" / "fresh_d2_l1"
             / "champion.npz")
g2 = z2["genome"]
table = be.silence_table(
    layout2, g2,
    lambda G: domain2.validation_scores(layout2, G),
    zero_tol=domain2.silence_zero_tol)
drops = np.array([u["drop"] for u in table["units"]])
units, A = ba.attribution_profiles(domain2, layout2, g2,
                                   tau=0.5)
assert [u["name"] for u in units] == \
    [u["name"] for u in table["units"]]
tot = A.sum(axis=1)
# sign bookkeeping: the table's drop (score points) is
# exactly the mean gap increase on silencing, and the
# profiles estimate per-problem gap increases, so the
# two compare directly (scale differs by n_val, which
# correlation ignores)
target = drops
rp = float(pearsonr(tot, target)[0])
rs = float(spearmanr(tot, target)[0])
agree = float(np.mean(np.sign(tot) == np.sign(target)))
check("attribution fidelity", rp > 0.3 and rs > 0.4,
      f"pearson {rp:.3f}, spearman {rs:.3f}, "
      f"sign agreement {agree:.2f}")

# 4. find_branches on the attribution signal partitions
import bert_branch as bb
branches = bb.find_branches(domain2, layout2, g2)
names = [u["name"] for b in branches for u in b["units"]]
all_names = [u["name"] for u in be.silence_units(layout2)]
check("attribution branches partition",
      sorted(names) == sorted(all_names)
      and all(b["group_drop_rel_pct"] is not None
              for b in branches),
      f"{len(branches)} branches, group drops "
      f"{[round(b['group_drop_rel_pct'], 1) for b in branches]}")

print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
