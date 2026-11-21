# Engine Test Protocol: Minimal TSP Run

Written 2026-11-20, before the test run. The criteria below are fixed in
advance; the results record (`ENGINE_TEST_2026-11-20.md`) is written after
the run and judged against this page, not the other way round.

## Purpose

Prove the engine, as exported in this repository, runs a real evolution
end to end on the travelling salesman problem at minimal scale: few
species, few generations, CPU only. This is a machinery and learning-signal
test, not a performance run. Growth rules and the cataclysm are excluded
here on purpose; they are covered by the verification suite
(`test_bert_engine.py`) and are tested live in later runs.

## What is reused

The engine is a generalisation of the attention stage 0-3 runners it
replaces, not a new invention: same genome arithmetic (the default config
reproduces the stage-3 shape exactly, 1,880 genes; suite leg 1), same
breeding operators, same Schedule A culling, same fixed elimination and
validation sets. This test therefore also serves as a regression check
that the generalisation did not break the original design.

## Configuration

File: `bert_engine_config_test.json` (the default TSP config with only
the lines below changed).

| Knob | Value | Why |
|---|---|---|
| backend / device / precision | numpy / cpu / x64 | reference arithmetic, exact comparisons |
| d, ffn_hidden, blocks, heads | 8, 16, 2, 2 | the stage-3 architecture, 1,880 genes |
| n_species x per_species | 12 x 10 (pop 120) | minimal, but enough for culling to bite |
| species_floor | 4 | lets Schedule A run 12 down toward 4 inside 40 generations |
| phase_a_gens + phase_b_gens | 20 + 20 | 40 generations total |
| max_growth_rounds | 0 | core loop only, no growth step |
| cataclysm_enabled | false | excluded, see Purpose |
| n_train / n_elim / n_val | 64 / 64 / 200 | house standards |
| validation set | seed 20261116 | the same fixed set as stages 0-3, so gaps are comparable |
| seeds | the default seed family | continuity with the stage runs |

## Legs and pass criteria

1. **Completion.** The run finishes and writes `results.json`,
   `champion.npz` and `COMPLETE.txt`, with no exception.
2. **Learning.** The champion's validation mean gap is at most +30% and
   at least 10 points better than the random baseline median for this
   architecture on the same set (+45.98%, stage-0 record); and the
   elimination-set best improves by at least 2% relative over the run.
   (For scale: the fully budgeted stage-3 champion reads +4.36% on this
   set. A 40-generation, 120-population run is not expected to approach
   it; this leg checks for a real descent, not a score.)
3. **Culling.** Species count follows Schedule A from 12 toward the
   floor of 4 (12, 10, 8, 7, 6, 5, 4) while total population stays 120.
4. **Invariance.** Shuffle audit of the champion: 20 validation instances
   x 20 relistings. Pass if mean absolute relative tour-length change is
   at most 0.005. (Near-tie greedy flips at float tolerance are the
   known stage-3 behaviour and are reported, not hidden.)
5. **Reproducibility.** A second run with the identical config and seeds
   produces a bit-identical champion genome and an identical champion
   validation score.
6. **Timing.** Report wall time and seconds per generation, and the
   implied cost of the standard budget (2,000 population x 150
   generations) on this CPU, for planning.

## Results

See `ENGINE_TEST_2026-11-20.md`.
