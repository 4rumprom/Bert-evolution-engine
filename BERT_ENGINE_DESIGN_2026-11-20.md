# Parametric Evolution Engine: Design Record (2026-11-20)

Sponsor: Roman. Built 2026-11-20 in `~/workspace/paper-adapter/tsp_lab/`.
This record states the rules exactly as implemented, the full config
surface, the smoke results with timings, run instructions for both
machines, the design joints where the brief was ambiguous, and the
governing checklist with its final state and evidence.

No existing stage file was modified (`attn_genome*.py`,
`run_tsp_attn_stage*.py` are untouched). No full-budget run was
launched; all verification is smoke scale, per orders.

## Files

| File | Role |
|---|---|
| `bert_engine.py` | Domain-independent core: config dataclass, architecture spec, name-based gene layout, gene classes, segmented batched encoder with frozen-prefix cache, breeding, Schedule A culling, plateau read, the four growth transplants, founder selection, atomic checkpointing with RNG state. |
| `bert_backend.py` | Backend selection: numpy reference or JAX (lazy import), device cpu/gpu, precision x64/x32. Engine imports cleanly with numpy alone; a JAX or GPU request that cannot be served falls back with a recorded note. |
| `bert_domain_tsp.py` | TSP adapter: coordinate inputs, attention pointer decoder, tour-length fitness, gap against Held-Karp optima, fixed validation set (seed 20261116, 200 instances at n=10), fixed elimination set (seed 20261272). |
| `bert_domain_stock.py` | Stock adapter against `trading-agent/genetic_lab` conventions: tickers as tokens, per-ticker features plus position fraction (input_dim 10), decoder emits position levels 0-5 plus a unit size, cash-checked execution at 6bp with accrued fees, fitness = median excess CAGR over uniform buy-and-hold of the same basket. |
| `run_bert_engine.py` | Runner: two-phase stage protocol, growth sequencer state machine, cataclysm handling, vaulting, completion sentinel, CLI (`--config`, `--domain`, `--smoke`, `--fresh`, `--stop-after`, `--out-dir`). |
| `bert_engine_config.json` | Default TSP config; reproduces the stage-3 architecture shape (1880 genes). |
| `bert_engine_config_stock.json` | Default stock config; validation points at the frozen common episode set `validation_episodes_v1.json` (64 episodes, basket 10, 60 days). |
| `test_bert_engine.py` | Smoke test suite: 27 legs, all passing (see Results). |

## Architecture as implemented

- Tokens are the domain's items (cities for TSP, tickers for
  stocks). A shared linear projection maps raw inputs to width d.
  There are no positional encodings, no scorer, and no recurrence
  anywhere in the encoder; every operation is permutation
  equivariant over tokens.
- The encoder is a list of couples. Each couple has its own input
  projection and one or more blocks; each block is an attention
  sub-block plus a feed-forward sub-block, both in the uniform
  residual form `out = y + Branch(Norm(y))`. Attention heads share
  the block width; the attention logit scale is a constant of the
  architecture family, fixed at the initial width (the stage
  program's SCALE = 1/sqrt(8)), and is carried unchanged through
  every transplant.
- The decode head is domain-supplied. TSP: the existing attention
  pointer decoder (state = mean of encoder outputs plus the
  last-picked token, learned start vector, tanh clip 10, visited
  mask, greedy picks). Stock: a score head producing a continuous
  score per ticker (quantised to levels 0-5 exactly as the V3/V5
  allocation genome does) plus a unit head (unit = 6% times a
  sigmoid over the mean token, planned invested fraction and cash
  fraction), executed in ranked order with cash checks and one
  overdraft penalty per the genetic_lab accounting.
- A genome is one flat vector. A name-based Layout maps names such
  as `c0.b1.h0.W_q` to slices, so transplants remap genes by name
  and old positions need not survive architecture changes. Every
  gene carries a class in a parallel int8 vector: DRAW (species
  defining frozen draw), OLD (preserved base), IFACE (new
  interface genes initialised at function-preserving values),
  DECODER (the active couple's decode head).

## Rules exactly as implemented

### Batched evaluation and the frozen-prefix cache

- The whole population is one stacked matrix (total_population,
  n_genes). Evaluation runs the member axis through every
  operation (einsum-style) in a single pass and folds results back
  to (species, member). The `chunk` config slices the population
  axis for memory; chunking does not change results (test: max
  diff 0.0).
- Encoding is split into segments at component boundaries
  (embedding, then per block: attention segment, feed-forward
  segment). When a segment's genes are identical across a group
  (a species in phase A, a lineage group after regeneration) and
  its inputs come from a seed-reproducible set, the segment is
  computed once from a representative row and cached; the cache
  key chains the input tag, the input bytes and the segment gene
  bytes. Cached output equals full recompute exactly (test: max
  diff 0.0, and the cache is non-empty in the grown-architecture
  leg).

### Species, breeding, culling

- A species is the preserved base plus one frozen random draw of
  the newest component's DRAW genes (draw_sigma from the draw
  seed stream). Species are created on transplant templates (or,
  at the initial stage, on a fresh random base) with one mutation
  applied at the phase-A probability vector, so phase-A prefix
  sharing holds within a species.
- Breeding is within species only: elite 2 kept, tournament 3,
  uniform crossover restricted to full-rate gene runs, mutation
  per the phase probability vector, and the species draw
  re-imposed afterwards so DRAW genes never drift within a
  species.
- Schedule A culling uses the named config knobs `cull_interval`
  (default 3), `cull_fraction` (default 0.2), `species_floor`
  (default 10). At each interval, species champions are scored on
  the fixed elimination set; the worst fraction of species is
  removed (never below the floor) and their member slots are
  redistributed so total population stays constant.

### Two-phase stage protocol

- Phase A (default 75 generations): only the newest component
  evolves. The probability vector gives DRAW and IFACE genes and
  the decoder the full mutation rate `mut_p`, and OLD genes rate
  0. Training replays ONE fixed instance set (seed
  `phase_a_seed`), and the frozen-prefix cache persists across
  the whole phase. Culling is active.
- Phase B (default 75 generations): the full network is
  recombined. Training sets are fresh per generation, addressed
  by seed: `gen_instance_seed = (run_seed + 1000003*(stage+1) +
  10007*gen) mod (2^31 - 1)`, so any generation's set is
  recreatable from seed plus generation number alone, independent
  of the breeding stream. OLD genes mutate at `mut_p *
  old_rate_factor` (factor default 0.1); newest component and
  decoder at full rate.
- The stage champion is the best species best-ever genome ranked
  directly on the fixed validation set.
- A phase ends on its generation budget or on plateau; a stage
  ends when phase B ends. Plateau read: slope of the elimination
  median over the final `plateau_window` generations (default
  30); plateaued when relative improvement over the window is
  below `plateau_threshold` (default 0.005).

### Growth sequencer

- On stage end, if growth rounds remain, the engine attempts the
  next untried rule from `growth_order`, least change first:
  (1) `widen` (width + `widen_step`, default +4), (2) `add_head`
  (one head per block of the last couple, stage-3 mechanism),
  (3) `add_block` (one identity block in the last couple),
  (4) `stack` (a second encoder-decoder couple mirrored from the
  last, on the frozen first).
- Widen construction (function preserving): new embedding
  coordinates are the token's own mean over old coordinates
  (new input-projection columns are column means); LayerNorm
  epsilon scales by d/d', old LayerNorm gains scale by
  sqrt(d/d'); Q/K new output columns are exactly zero; the
  output-projection and second feed-forward matrix new output
  columns are per-row means, so every new stream coordinate
  equals the mean of the old output; free input-side new genes
  are the species draw; new output sides are IFACE genes at the
  preserving values. The decode head is widened through a domain
  hook with the same discipline.
- Every transplant must pass the domain's exact preservation
  check before evolution begins: for TSP, identical tours and
  max length difference 0.0 on the validation set; for stocks,
  identical discrete decisions (levels and ranked order) on all
  validation episode days and CAGR difference below 1e-9
  (observed 0.0; the continuous unit agrees to about 1e-12
  relative because widening is exact in exact arithmetic).
  A failed check rejects the rule without evolving.
- Keep/revert: the bar is the best founder's validation score.
  If the post-stage champion beats the bar, the rule is KEPT and
  the new architecture, classes and founders are recorded. If
  not, the engine reverts to the champion architecture recorded
  at acceptance, regenerates species from the pre-attempt
  founders, and advances to the next rule. When all rules are
  exhausted (or `max_growth_rounds` is reached), the run stops
  and reports.

### Cataclysm and founders

- Knobs: `cataclysm_enabled` (default true),
  `cataclysm_every_n_generations` (default null, meaning the
  event fires at every growth-stage transition; when set, it
  also fires on that generation interval within a stage), and
  `founder_count` (default 1; 2, 3, 5 are the sponsor's planned
  values).
- Founders are the best champion of each of the top
  `founder_count` DISTINCT species, ranked by their champions'
  fixed-validation scores; never two founders from one species.
- All founders are vaulted BEFORE the event. Regeneration splits
  the new species set as evenly as possible among the founders;
  each founder seeds its share with fresh draws on its own base,
  through the same code path as growth-stage regeneration. Every
  species records its founder lineage id (`L<species_id>`), and
  cache grouping follows lineage.
- Each cataclysm is logged as a named event with generation,
  every founder (species, lineage, validation score), and the
  best validation score before and after the event. Founders' own
  validation tours are unchanged by the event (test-verified).

### Checkpointing and completion

- Per-generation atomic checkpoints (write-temp-then-rename)
  carry the full config canonical form, architecture and layout
  signatures, gene classes, all species state, and both RNG
  bit-generator states (breeding stream and draw stream) as JSON.
  Resume refuses (SystemExit) on any config, architecture or
  layout mismatch.
- The completion sentinel `COMPLETE.txt` is written only after
  evolution, validation and champion vaulting are all on disk;
  `results.json` and `champion.npz` land with it.

## Config surface

One JSON file plus the EngineConfig dataclass. Fields (defaults
from `bert_engine_config.json`):

- Architecture: `input_dim` (0 = domain default; TSP 2, stock
  10), `d` 8, `ffn_hidden` 16, `initial_blocks` 2,
  `initial_heads` 2.
- Population: `n_species` 100, `per_species` 20, `chunk` 128,
  `elite` 2, `tournament` 3.
- Training: `n_train` 64, `n_elim` 64, `n_val` 200,
  `phase_a_gens` 75, `phase_b_gens` 75.
- Mutation: `mut_p` 0.1, `mut_sigma` 0.15, `draw_sigma` 0.5,
  `old_rate_factor` 0.1.
- Culling: `cull_interval` 3, `cull_fraction` 0.2,
  `species_floor` 10.
- Growth: `growth_order` [widen, add_head, add_block, stack],
  `widen_step` 4, `max_growth_rounds` 8, `plateau_window` 30,
  `plateau_threshold` 0.005.
- Cataclysm: `cataclysm_enabled` true,
  `cataclysm_every_n_generations` null, `founder_count` 1.
- Seeds: `run_seed` 20261300, `draw_seed` 20261301,
  `elim_seed` 20261272, `val_seed` 20261116, `phase_a_seed`
  20261310, `audit_seed` 20261303, `check_seed` 20261302.
- Backend: `backend` jax|numpy, `device` cpu|gpu, `precision`
  x64|x32.
- Domain: `domain` tsp|stock plus a `domain_config` block (TSP:
  `n_cities`; stock: `basket_n`, `episode_days`,
  `frozen_episodes_path`, `n_train_episodes`,
  `n_elim_episodes`, `synthetic_panel` for smoke runs).

## Backends and portability

- Numpy is the reference path and needs nothing beyond numpy;
  the engine imports cleanly when JAX is absent.
- JAX is the primary backend for the sponsor's machine. The
  reference configuration enables float64 (`jax_enable_x64`,
  `precision: "x64"`), under which JAX-on-CPU results matched the
  numpy reference exactly in the smoke test (max diff 0.0). A
  float32 performance mode (`precision: "x32"`, the practical
  GPU mode) may differ at float tolerance; that is accepted and
  documented here, per the brief.
- Device selection is config-only (`backend`, `device`). If JAX
  or a GPU is requested but unavailable, the backend falls back
  and records `fell_back` plus a note in the run results; it
  never silently pretends.

## Run instructions

This VM (CPU, numpy only, system python3):

```
cd ~/workspace/paper-adapter/tsp_lab
python3 test_bert_engine.py                 # smoke suite
python3 run_bert_engine.py --smoke          # tiny end-to-end run
python3 run_bert_engine.py --config bert_engine_config.json --fresh
```

This VM with JAX on CPU (venv built for this project, JAX
0.12.1):

```
~/workspace/.jaxvenv/bin/python3 test_bert_engine.py \
    --jax-python ~/workspace/.jaxvenv/bin/python3
~/workspace/.jaxvenv/bin/python3 run_bert_engine.py \
    --config bert_engine_config.json --fresh
```

The sponsor's PC (GPU): install JAX with CUDA support, then use
the same commands and the same config with `"backend": "jax"`,
`"device": "gpu"` (keep `"precision": "x64"` for
reference-identical results, or `"x32"` for GPU performance).
Per the official JAX installation page
(https://docs.jax.dev/en/latest/installation.html):

```
pip install -U "jax[cuda12]"   # CUDA 12, NVIDIA driver >= 525
pip install -U "jax[cuda13]"   # CUDA 13, NVIDIA driver >= 580
```

Long runs on either machine must be launched fully detached
(`setsid nohup ... < /dev/null &`), per the standing GA launch
rule. Resume is the same command without `--fresh`; a mismatched
config is refused.

## Smoke results (2026-11-20, this VM)

`python3 test_bert_engine.py --jax-python
~/workspace/.jaxvenv/bin/python3`: **27/27 legs passed**, total
wall clock about 12.6 s. Legs, condensed:

- Default config reproduces the stage-3 shape: 1880 genes,
  couples [[2, 2]], d 8, FFN 16.
- Batched evaluation equals a per-member loop exactly (max diff
  0.000e+00); population chunking changes nothing (0.000e+00).
- Frozen-prefix cache equals full recompute exactly (0.000e+00)
  and actually caches (3 segment entries on the grown
  architecture).
- Transplant preservation exact for all four rules on the TSP
  validation set: tours identical, max length difference 0.0
  (widen, add_head, add_block, stack).
- Permutation audit: encoder equivariance under relisting, max
  diff 1.776e-15.
- Tiny TSP run end to end (1.0 s): sentinel and results on disk;
  champion validation score -0.1993 (negative mean gap) against
  a random-baseline median of -0.5173; growth trigger fired;
  widen attempted and KEPT; phase A and phase B both ran.
  Phase timings: stage 0 phase A 0.13 s / 10 generations,
  phase B 0.11 s / 9; stage 1 phase A 0.29 s / 6, phase B
  0.46 s / 9 (phases also end early on plateau).
- Resume refuses a mismatched config; stop-after 3 generations
  plus resume completes the run.
- Forced cataclysm with founder_count=3 (interval 4): founders
  from 3 distinct species (species 3, 2, 0; validation scores
  -0.3476, -0.4509, -0.5019); regenerated population carries
  exactly lineage ids L0, L2, L3 at the full species count (6);
  3 founder vault files written before the event; founders'
  validation scores unchanged by the event (val before
  -0.347579, val after -0.261896 for the regenerated best).
- Stock adapter: finite basket-relative fitness on synthetic
  episodes; validation detail reports median excess CAGR over
  buy-and-hold; episode simulation cached path equals plain path
  exactly (0.000e+00); stock widen preservation exact (decisions
  identical, max CAGR difference 0.0, max unit relative
  difference 4.5e-15); frozen common episode set parses (64
  episodes, basket 10).
- JAX leg: same code under JAX 0.12.1 on CPU with x64 enabled;
  tour lengths match the numpy reference exactly (0.000e+00),
  no fallback.

Additional evidence runs (same day):

- CLI smoke (`run_bert_engine.py --smoke --fresh`): completes in
  1.1 s; stage 1 champion -0.199318 beats the bar -0.208950,
  widen KEPT; `COMPLETE.txt`, `champion.npz`, `results.json`
  and `vault/` on disk.
- Seed recreation check: phase-B training batch for (stage 2,
  generation 7) regenerated twice is bit-identical; generation
  8 differs; the phase-A set is one fixed set.
- Forced-bar revert run (test harness pins post-growth
  validation scores to -1e9 so no champion can beat the bar;
  the engine's own decision code runs unmodified): all four
  rules attempted in order (widen, add_head, add_block, stack),
  each REVERTED, then `growth_exhausted`, run finishes cleanly
  with the stage-0 champion retained (-0.208950). A 16-seed scan
  of natural tiny runs produced no natural revert: with the bar
  set at the best founder, a short growth stage usually improves
  on validation, so the revert branch is exercised by the
  forced-bar run instead.

## Design joints (ambiguities in the brief, and what was chosen)

1. Block form: the engine uses ONE uniform block form for all
   blocks (`out = y + Branch(Norm(y))`, FFN always present).
   The bespoke stage runners used a different internal form for
   the first block at stages 0-1. Total gene count at the
   default config is still exactly the stage-3 count, 1880, and
   the growth rules in the brief (widen, head, block, stack)
   have no add-FFN rule, so the uniform form is the consistent
   reading.
2. Attention scale: fixed at the initial width as an
   architecture constant (ArchSpec `attn_scale`), matching the
   stage program, whose SCALE never changed as heads and blocks
   were added. Recomputing 1/sqrt(d) after widening rescales
   every attention score and breaks exact preservation; this
   was caught by the preservation test during the build and
   fixed. Related catch from the same test: in the `x @ W`
   convention, axis 0 is the input dim and axis 1 the output
   dim; the widen draw/interface classes for Q/K follow that
   (new input rows are the free draw, new output columns are
   zero-initialised interface genes).
3. Layout is name-based and transplants remap by name; gene
   classes are a per-gene vector rebuilt index-exactly by each
   transplant rather than inferred from positions.
4. Phase A replays one fixed training set (the sponsor's "same
   known sets"); phase B sets are addressed by (seed, stage,
   generation), which satisfies "recreatable from seed plus
   generation number" independently of checkpoint state.
5. Stage champion selection ranks species bests directly on the
   validation set (the brief's literal wording). The bespoke
   runners ranked on the elimination set first; noted as a
   deliberate deviation toward the brief.
6. Plateau uses the stage records' operational read
   (elimination-median slope over the final window) with the
   threshold as a config value (0.005 relative improvement over
   30 generations). A phase-A plateau ends phase A early and
   moves to phase B; a phase-B plateau ends the stage.
7. After a stack, add_head and add_block apply to the LAST
   couple; widen is global because d is global; stack mirrors
   the last couple, the new couple's input projection starts as
   identity (OLD class), its decode head copies the previous
   head (DECODER class) and previous heads become OLD (inert).
8. `founder_count` supersedes the earlier `cataclysm_survivors`
   knob from the first cataclysm steer; only `founder_count` is
   implemented.
9. Initial-stage species creation applies the phase-A
   probability vector for its one init mutation (OLD rate 0) so
   phase-A prefix sharing holds from generation 0. The stage-3
   runner perturbed old genes at init; noted deviation.
10. Stock preservation is judged on the discrete decisions
    (levels and ranked order identical on every validation
    episode day) plus CAGR difference below 1e-9, because the
    continuous unit size cannot be bit-exact under any
    architecture change; observed CAGR difference was 0.0 and
    unit relative difference 4.5e-15.
11. Stock adapter scope: smoke-scale correctness as briefed.
    HONEST GAP: production stock runs are not wired to the real
    genetic_lab Panel in this build; `build_domain` raises
    SystemExit for the real panel, and the smoke path uses a
    documented synthetic panel (random-walk prices, synthetic
    standardised features). The frozen episode set file parses
    and drives validation when configured. Wiring the real
    panel is the remaining stock work.
12. Backend fallback: a JAX or GPU request that cannot be
    served falls back to numpy/CPU with the fallback recorded
    in the run results, rather than failing the run.

## Checklist final state

Governed by `BERT_ENGINE_CHECKLIST_2026-11-20.md`. Every item
is checked with evidence (test output or a file on disk).

### Parameters and config
- [x] One config file controls: input feature count, width,
      FFN hidden, heads per block, blocks, species count,
      members per species, population chunk size, training
      instances per generation, generation budgets per phase,
      mutation sigma and rates, culling knobs, cataclysm knobs,
      founder_count, all seeds, backend and device.
      Evidence: `bert_engine_config.json`,
      `bert_engine_config_stock.json`, EngineConfig in
      `bert_engine.py`.
- [x] Defaults reproduce the stage-3 architecture shape.
      Evidence: test leg 1 (1880 genes, couples [[2, 2]]).

### Batched evaluation
- [x] Whole population as one stacked matrix in a single pass;
      fitness folded back per member and per species.
      Evidence: test leg 2; `encode_population` in
      `bert_engine.py`.
- [x] Batched result matches per-member loop exactly (float64).
      Evidence: test leg 2, max diff 0.000e+00.
- [x] Population axis chunked without changing results.
      Evidence: test leg 3, max diff 0.000e+00.

### Backends and portability
- [x] Numpy backend runs on this VM (CPU).
      Evidence: full suite on system python3 (numpy 1.26.4).
- [x] JAX backend runs the same code on CPU here; device
      selectable in config for GPU on the sponsor's PC.
      Evidence: JAX leg under JAX 0.12.1 CPU, max diff
      0.000e+00, fell_back False; `device` in both configs.
- [x] Float64 reference mode available.
      Evidence: `precision: "x64"` in both configs; JAX leg ran
      with x64 on.
- [x] Run instructions documented for both machines, including
      the JAX CUDA install line.
      Evidence: this record, Run instructions section.

### Domains
- [x] TSP adapter complete: coordinate inputs, pointer
      decoder, tour-length fitness, fixed validation set (seed
      20261116). Evidence: `bert_domain_tsp.py`; test legs 10
      to 15.
- [x] Stock adapter against genetic_lab conventions, smoke
      scale, gaps flagged. Evidence: `bert_domain_stock.py`;
      stock test legs (fitness, excess over buy-and-hold,
      cached equals plain exactly, widen preservation, frozen
      set parses). Gap: real Panel not wired (Design joint 11).

### Frozen-prefix cache and seeds
- [x] Segments at component boundaries; shared segments
      computed once per group and cached.
      Evidence: test legs 4 and 5 (3 cache entries).
- [x] Cache output equals full recompute exactly (0.0).
      Evidence: test leg 4.
- [x] Every generation's training set recreatable from seed
      plus generation number; elimination and validation sets
      use their own seeds. Evidence: `gen_instance_seed`;
      recreation check (bit-identical regeneration, see Smoke
      results).
- [x] Checkpoints save full RNG state; resume refuses a
      mismatched config. Evidence: test legs 16 and 17.

### Stage protocol
- [x] Phase A: only the newest component evolves, cached
      frozen outputs, same known sets, base rate 0, culling
      active. Evidence: tiny run phase timings and cull events;
      probability vector in `bert_engine.py`.
- [x] Phase B: full network, fresh stream, old genes at the
      0.1 factor, newest component and decoder at full rate.
      Evidence: tiny run; `prob_vector` phase B.
- [x] Stage champion chosen on the fixed validation set.
      Evidence: tiny run champion validation score; runner
      selection code.

### Species and culling
- [x] Species from frozen draws on the preserved base;
      breeding within species only. Evidence: creation and
      breeding code; cataclysm lineage legs.
- [x] Schedule A culling with interval, fraction and floor as
      config knobs; slots redistributed. Evidence: config
      fields `cull_interval` 3, `cull_fraction` 0.2,
      `species_floor` 10; cull events in tiny run logs with
      population restored to constant size.

### Growth rules
- [x] Stage ends on budget exhausted OR plateau (threshold in
      config). Evidence: PLATEAU fired in tiny run stage 1
      phase B and in the CLI smoke; `plateau_threshold` in
      config.
- [x] Rules least-change first: widen, head, layer, stack.
      Evidence: `growth_order` in config; forced-bar run
      attempted them in exactly that order.
- [x] Every transplant passes an exact preservation check
      (max diff 0.0 on validation tours) before evolution.
      Evidence: test legs 6 to 9, all four rules, tours
      identical, max length difference 0.0.
- [x] Keep on validation improvement, else revert and advance;
      stop and report when exhausted. Evidence: tiny run
      growth_kept (widen); forced-bar run growth_reverted for
      all four rules in order, then growth_exhausted, stage-0
      champion retained.

### Cataclysm and founders
- [x] Cataclysm named and configurable (enabled; at growth
      transitions and/or every N generations). Evidence:
      `cataclysm_enabled`, `cataclysm_every_n_generations` in
      config; transition cataclysm event in tiny run;
      within-stage event in the forced cataclysm leg.
- [x] Survivors = founder_count: best champions of the top K
      distinct species, all vaulted before the event.
      Evidence: forced cataclysm leg (3 distinct species, 3
      vault files).
- [x] New species split evenly among founders, fresh draws on
      each founder's base, lineage recorded. Evidence: leg
      asserting exactly lineage ids L0, L2, L3 after the event.
- [x] Each cataclysm logged (generation, founders, validation
      before/after); founders' validation tours unchanged.
      Evidence: event record (val before -0.347579, after
      -0.261896); leg asserting founders' scores unchanged.
- [x] Smoke test: forced cataclysm with founder_count = 3
      yields 3 distinct species and exactly 3 lineage ids.
      Evidence: cataclysm test legs.

### Invariance and integrity
- [x] No positional encodings, no scorer, no recurrence in the
      attention domain; invariance audited. Evidence:
      equivariance test leg, max diff 1.776e-15 under
      relisting.
- [x] Champion vaulting and completion marker only after
      evolution, validation and vaulting are on disk.
      Evidence: tiny run and CLI smoke outputs (`COMPLETE.txt`,
      `champion.npz`, `results.json`, `vault/`).

### Verification (smoke scale only; no full-budget run)
- [x] Tiny config runs end to end and beats its random
      baseline. Evidence: champion -0.1993 vs random baseline
      median -0.5173.
- [x] Growth trigger fires; rule-1 transplant preservation is
      exact. Evidence: test legs 6, 12, 13.
- [x] Phase A cache test and Phase B run pass; keep/revert
      logic executes. Evidence: test legs 4, 14, 15; forced-bar
      revert run.
- [x] Design record written: rules as implemented, config
      surface, smoke results and timings, run instructions for
      both machines. Evidence: this file.
