# Engine Test Results: Minimal TSP Run, 2026-11-20

Judged against `TEST_PROTOCOL.md`, written before the run. Config:
`bert_engine_config_test.json` (12 species x 10, 40 generations, stage-3
architecture of 1,880 genes, numpy backend, growth and cataclysm off).
Two identical runs (A and B) plus a shuffle audit of the champion.

## Verdicts

| Leg | Verdict | Evidence |
|---|---|---|
| 1. Completion | PASS | Both runs finished 40 of 40 generations and wrote results.json, champion.npz, COMPLETE.txt. Wall 4.5 s (A), 4.1 s (B) |
| 2. Learning | FAIL on the gap bar; PASS on descent | Champion mean validation gap +41.20% (median +41.32%) against a bar of at most +30% and at least 10 points under the +45.98% random median; it is 4.8 points under. Elimination-set best tour length improved 4.0250 to 3.4697, a 13.8% relative improvement against a 2% bar |
| 3. Culling | PASS, with a bookkeeping note | Species 12, 10, 8, 6, 5, 4 (cull counts round to nearest), floor of 4 reached at generation 14 and held. Population targets preserve 120; the recorded population dips for exactly one generation after each cull (100, 96, 90, 100, 96) because history is written after the cull and before breeding refills survivors to their new targets |
| 4. Invariance | PASS, bit-exact | Shuffle audit, 20 instances x 20 relistings: mean and max absolute relative length change both 0.0, zero flips in 400 relistings |
| 5. Reproducibility | PASS | Run B's champion genome is bit-identical to run A's; validation scores identical |
| 6. Timing | Reported | 0.11 s per generation at population 120. Extrapolated to the standard budget (2,000 x 150 generations): about 4.7 minutes on this CPU. The old stage-3 runner took about 34 minutes for that budget, so the engine is roughly 7x faster per unit of work here |

## Reading of the learning leg

The machinery learns: the elimination best descended steadily and the
champion is a real, invariant, reproducible artifact. The endpoint is
what this scale buys. At the seed stage only the 80 decoder genes evolve;
the 1,800 encoder genes are the frozen species draws, exactly as in the
stage 0-3 design this engine generalises. The substrate lottery
therefore had 12 tickets instead of the stage runs' 100, with 120
population instead of 2,000 and 40 generations instead of 150. The
stage-0 champion reached +9.18% under the full budget; +41.20% at this
budget is a scale reading, not a machinery defect. The gap bar in the
protocol was set too optimistically for the minimal scale, and this
record says so rather than moving it after the fact.

## Bottom line

The engine runs end to end as exported, its operators behave as
specified, and its outputs are exact and reproducible. Score validation
requires a full-budget run: the default config (100 species x 20, 150
generations) extrapolates to about 5 minutes on this CPU, or a GPU run
on the sponsor's PC.
