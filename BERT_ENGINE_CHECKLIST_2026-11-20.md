# Parametric Engine Build Checklist (sponsor: Roman, 2026-11-20)

Rule: an item is checked only with evidence (passing test output or a file on disk).
The builder iterates until every item is checked; unchecked items are reported
as unchecked, never as done on intention.

## Parameters and config
- [ ] One config file controls: input feature count, width (neurons), FFN hidden,
      heads per block, blocks (layers), species count, members per species,
      population chunk size, training instances per generation, generation
      budgets per phase, mutation sigma and rates, culling knobs, cataclysm
      knobs, founder_count, all seeds, backend and device.
- [ ] Defaults in the config reproduce the stage-3 attention architecture shape.

## Batched evaluation
- [ ] Whole population evaluated as one stacked matrix (species x members x
      problems) in a single pass; fitness folded back per member and per species.
- [ ] Batched result matches a per-member loop reference exactly (float64 mode).
- [ ] Population axis can be chunked for memory without changing results.

## Backends and portability
- [ ] Numpy backend runs on this VM (CPU) for test and debug.
- [ ] JAX backend runs the same code on CPU here; device selectable in config
      so the same code runs on GPU on the sponsor's PC.
- [ ] Float64 reference mode available so cross-backend checks are meaningful.
- [ ] Run instructions documented for both machines (CPU here, GPU at home,
      including the JAX CUDA install line).

## Domains
- [ ] TSP adapter complete: coordinate inputs, pointer decoder, tour-length
      fitness, fixed validation set (seed 20261116).
- [ ] Stock adapter against genetic_lab conventions: tickers as tokens,
      per-ticker features, allocation decoder (scores + unit size),
      cash-checked execution, basket-relative fitness (excess over uniform
      buy-and-hold of the same basket) on the frozen common episode set;
      smoke-scale correctness at minimum, gaps flagged honestly.

## Frozen-prefix cache and seeds
- [ ] Encoder split into segments at component boundaries; a segment whose
      weights are identical across a group and whose inputs are
      seed-reproducible is computed once per group and cached.
- [ ] Cache output equals full recompute exactly (max diff 0.0), proven by test.
- [ ] Every generation's training set is recreatable from seed + generation
      number; fixed elimination/validation sets use their own seeds.
- [ ] Checkpoints save full RNG state; resume refuses a mismatched config.

## Stage protocol
- [ ] Phase A: only the newest component evolves, fed by cached frozen outputs
      on the same known training sets; base mutation rate 0; culling active.
- [ ] Phase B: full network recombined, fresh instance stream, old genes at
      the 0.1 rate factor, newest component + decoder at full rate.
- [ ] Stage champion chosen on the fixed validation set.

## Species and culling
- [ ] Species defined by frozen random draws on the preserved base; breeding
      within species only.
- [ ] Schedule A culling: worst 20% of species every 3 generations on the
      fixed elimination set, floor 10 species, slots redistributed;
      interval, fraction and floor are config knobs.

## Growth rules
- [ ] A stage ends on budget exhausted OR plateau (threshold in config).
- [ ] Rules tried least-change first: (1) widen by a few neurons, (2) add a
      head, (3) add a layer (block), (4) stack a second BERT couple.
- [ ] Every transplant passes an exact preservation check (max diff 0.0 on
      validation tours) before evolution starts.
- [ ] Keep if the stage champion improves the fixed-validation mean gap,
      else revert and advance to the next rule; stop and report when the
      rules are exhausted.

## Cataclysm and founders
- [ ] Cataclysm is a named, configurable rule (enabled, trigger at growth
      transitions and/or every N generations).
- [ ] Survivors = founder_count parameter: best champions of the top K
      DISTINCT species (K = 1, 2, 3, 5...), all vaulted before the event.
- [ ] New species split evenly among founders, bred from fresh draws on each
      founder's base; every species records its founder lineage.
- [ ] Each cataclysm logged as an event (generation, founders, validation
      before/after); founders' own validation tours unchanged by the event.
- [ ] Smoke test: forced cataclysm with founder_count = 3 yields survivors
      from 3 distinct species and exactly 3 lineage ids afterwards.

## Invariance and integrity
- [ ] No positional encodings, no scorer, no recurrence in the attention
      domain; permutation invariance holds by construction and is audited.
- [ ] Champion vaulting and completion marker only after evolution,
      validation and vaulting are all on disk.

## Verification (smoke scale only; no full-budget run)
- [ ] Tiny config runs end to end and beats its random baseline.
- [ ] Growth trigger fires; rule-1 transplant preservation is exact.
- [ ] Phase A cache test and Phase B run pass; keep/revert logic executes.
- [ ] Design record written: rules as implemented, config surface, smoke
      results and timings, run instructions for both machines.
