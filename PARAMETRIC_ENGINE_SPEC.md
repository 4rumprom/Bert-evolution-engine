# Parametric Growth Engine — DRAFT SPEC (not frozen, nothing built, nothing run)

Traveler (TSP) domain. Draft written 2026-11-20, from Roman's order in chat
2026-11-20T13:58:41Z (UTC): build a parametric engine so he can play with the
variables, with rules that switch to the next growth step when a stage runs
out of budget or stops learning.

Status: DRAFT for Roman to freeze or amend. No code has been written from
this spec, no run has been launched from it, and no value below is frozen
until he says so. Where this draft proposes a number Roman did not state,
it is labelled PROPOSED and listed as an open joint at the end.

## 1. Roman's order, in his words (restated, not paraphrased away)

- Build the frozen-input idea from the same conversation: champions are
  frozen, their inputs and outputs are remembered, the network is widened
  or a layer is added, only the new part is recomputed on the known same
  training sets by feeding the remembered inputs directly, then the refined
  part is recombined into the full network and finalised on a new training
  set so it also sees new data.
- First 50 or 100 generations: grunt work on the new random network —
  culling, initial conversion, deciding which mutations / gene combinations
  make sense. Then 50 or 100 generations or more: the whole network with
  refinement, where the old network also gets mutations to see if that
  helps.
- The engine must have a very simple way to adjust: network size, number
  of features as inputs, number of neurons, all the parameters, how many
  generations are in the budget.
- Rules: if a stage runs out of budget OR starts to stop learning, switch
  to the next rule. Next rule, in his order: (1) add a few neurons in
  width, (2) add a layer of depth, (3) stack another BERT on top of it.
- Do each growth in the way that changes the network the least: very few
  new neurons in combination to explore initially, so the new stage can
  always be run in parallel and get quickly to the frozen stage.

## 2. Baseline verified from the code (the engine must reproduce this)

Base: stage-3 champion, species 27, generation 146, 1,880 genes,
mean validation gap +4.36% / median +1.48% at n=10
(record `ATTN_STAGE3_2026-11-20.md`; champion
`tsp_lab/attn_stage3_champion.npz`). Growth line so far:
+9.18% -> +7.66% -> +5.53% -> +4.36% across stages 0-3, all at n=10.

Values read from `tsp_lab/run_tsp_attn_stage3.py` and
`tsp_lab/attn_genome_s3.py` on 2026-11-20 (not from memory):

| Parameter | Stage-3 value | Where it lives now |
|---|---|---|
| Width d | 8 | `attn_genome.D` |
| FFN hidden | 16 | `attn_genome_s2.H_FFN` |
| Genome size | 1,880 genes (base 1,304 + head-2 draw 432 + head-2 interface 144) | `attn_genome_s3.N_PARAMS` |
| Full-rate evolving genes | 224 (decoder 80 + head-2 interfaces 144) | `N_FULL_RATE` |
| Older genes | 1,224, mutate p=0.01, no crossover | `N_OLD`, `OLD_MUT_P` |
| Species x members | 100 x 20 = 2,000 total | `N_SPECIES0, PER_SPECIES0` |
| Generations | 150 | `GENERATIONS` |
| Cities (train/val) | 10; diagnostic n=20 | `N_CITIES`, `DIAG_N_CITIES` |
| Train / elim / val instances | 64 / 64 / 200 | `N_TRAIN, N_ELIM, N_VAL` |
| Culling | every 3 gens, worst 20%, floor 10 species | `CULL_EVERY, CULL_FRAC, MIN_SPECIES` |
| Tournament / elite | 3 / 2 | `TOURNAMENT, ELITE` |
| Mutation sigma / p | 0.15 / 0.10 | `MUT_SIGMA, MUT_P` |
| Species draw | N(0, 0.5), frozen per species | draw seed 20261301 |
| Seeds | run 20261300, draw 20261301, elim 20261272, check 20261302, audit 20261303, val 20261116, diag 20261203 | runner constants |
| Eval chunk | 128 genomes per chunk, all problems at once | `attn_genome_s3.CHUNK` |
| Preservation gate | unmutated transplant reproduces base champion exactly (max length diff 0.0, all tours identical) | runner, hard gate |
| Invariance gate | encoder equivariance <= 1e-9 (tour-length diff reported as measured; greedy near-tie flips on random genomes, per `INVARIANCE_AUDIT_2026-11-20.md`: NO BUG proven at 50/100-digit precision) | runner |
| Keep rule (stage) | champion mean validation gap improves on previous stage champion (+5.53% was the stage-3 bar) | runner |
| Plateau read | elimination-median slope over final 30 generations | stage-0 criterion, as in stage-3 record |
| Checkpointing | per-generation atomic checkpoint; resume refuses mismatched config; RNG state saved in checkpoint | runner |
| Launch rule | fully detached (`setsid`) | standing sponsor rule |

## 3. Config surface (the "play with the variables" part)

One config object (dict / small YAML or JSON file — implementation choice,
not frozen) plus CLI overrides. Every value in section 2 becomes a named
field with the stage-3 value as its default. No more hardcoded constants
scattered across a new runner per stage. Fields, grouped:

- Architecture: `width_d`, `ffn_hidden`, `n_blocks`, `n_heads_per_block`,
  `n_features` (input features per token; today 2 coordinates — the
  "number of features as inputs" knob Roman asked for), `decoder` type.
- Population: `n_species`, `per_species`, `pop_total` (derived, checked),
  `draw_sigma`, `draw_seed`.
- Evolution: `generations_budget`, `phase_a_generations`,
  `phase_b_generations`, `mut_sigma`, `mut_p_full`, `mut_p_old`,
  `tournament`, `elite`, `cull_every`, `cull_frac`, `min_species`.
- Data: `n_cities`, `n_train`, `n_elim`, `n_val`, `train_seed_mode`,
  `elim_seed`, `val_seed`, `diag_n_cities`, `diag_seed`.
- Growth rules: `growth_ladder` (ordered list, see section 5),
  `growth_step_width` (how many neurons "a few" means),
  `plateau_threshold` (see open joints), `max_stages`.
- Execution: `eval_chunk`, `backend` (`numpy` today; `jax` later, see
  section 7), `seed` (run seed).

The first engine run replays stage 3 exactly from this config (same seeds,
same values) and must reproduce the stage-3 champion's numbers before any
new growth is trusted — that replay is the engine's acceptance test.

## 4. Two phases inside each stage (Roman's 50/100 + refinement)

- Phase A — grunt work on the new part only. New block's interface genes
  and the decoder evolve at full rate; ALL older genes frozen (mutation
  p = 0, no crossover). Length: 50 or 100 generations (Roman's numbers;
  exact split is an open joint). Culling runs as now (Schedule A).
  Because the old genes do not move in Phase A, the frozen-input cache
  in section 6 is exact here.
- Phase B — whole-network refinement. Old genes rejoin at the standing
  1/10 rate (p = 0.01, no crossover), exactly as stages 0-3 already do.
  Length: 50 or 100 generations or more (Roman's numbers; exact split is
  an open joint). The cache falls back to full recompute here (section 6).
- READING TO CONFIRM: Roman's words at 13:55:52Z were "you add the 10th
  mutation to the old network". This draft reads that as the existing
  1/10-rate old-gene mutation rule. If he meant something else (e.g. a
  10th mutation event, or one old-gene mutation per genome), the Phase B
  rule changes and this draft is wrong on that point — confirm before
  freezing.

## 5. Growth-rule ladder (switch on budget-out OR stop-learning)

Trigger (either): the stage's generation budget is exhausted, OR the
plateau read says learning has stopped. Plateau definition carried over:
elimination-median slope over the final 30 generations (stage-0
criterion). PROPOSED, not frozen: the numeric slope threshold that counts
as "stopped" — no threshold has been set by Roman yet (open joint).

On trigger, take the next rule in this order, smallest change first:

1. Widen by a few neurons (PROPOSED step size: +4 neurons on the grown
   component, matching growth stages 1-2 precedent — open joint).
   Neutral addition: new interface weights start at exactly zero, new
   cross-connections into old units start at zero and stay frozen where
   a random draw would break the base (the growth-stage-1 correction).
   Preservation gate (exact 0.0) must pass before evolution.
2. Add a layer of depth (one more transformer block, neutral addition,
   same preservation gate).
3. Stack another BERT on top of the whole network (neutral addition,
   same preservation gate).

Each step grows by ONE neutral addition on the preserved champion, kept
only on measured improvement against the previous stage champion (the
standing keep rule). Species are re-drawn around the new champion for
the new stage (100 fresh species from the new component, as stages 1-3
did); the culling ladder restarts.

## 6. Frozen-input cache (the saving Roman described)

Idea: when a prefix of the network is frozen for a whole species, compute
its output once per training instance and reuse it for every member of
that species, instead of recomputing the whole encoder per member.

- What is cached: the frozen prefix's output activations for a given
  instance set, keyed by (champion genome hash, frozen-prefix definition,
  instance-array hash). The instance array itself is hashed, not just a
  seed — correction to an earlier chat claim: training instances each
  generation are drawn from the single run RNG stream
  (`Xg = rng.random(...)` in the runner; RNG state is saved in the
  checkpoint), so they are reproducible via the saved RNG state, not via
  a per-generation seed. Fixed sets (elimination seed 20261272,
  validation seed 20261116) are reproducible from their seeds.
- When it is exact: Phase A only, while old genes are frozen (p = 0).
  Then every member of a species shares the same frozen prefix for a
  given instance, so one prefix pass per species per instance replaces
  up to 20 (or 50) per-member passes. The new part alone is evaluated
  per member, fed from the cached prefix output.
- When it is NOT used: Phase B (old genes mutate, so members diverge)
  and any stage where the prefix is not shared — fall back to the full
  per-member evaluation as today. Never cache across a mutation.
- Recombine and finalise (Roman's step): after the new part is refined
  on the cached inputs, recombine it into the full network and run the
  final generations on a fresh training set (new instances from the run
  stream), so the grown network also sees new data before the stage
  champion is judged. The champion is still judged on the fixed
  elimination set and the fixed validation set, as now.
- What is stored, not recomputed forever: weights + instance-array hash
  (+ RNG state for stream-drawn sets) are enough to rebuild a cache
  entry; cache entries are keyed, so a stale entry can never be reused
  after a mutation or a new instance set.

## 7. Batched evaluation (Roman's one-big-matrix idea)

Already true one level down, verified in `attn_genome_s3.py`:
`encode_s3_chunk` carries a population-member dimension through every
matrix multiply; a species (20 members) runs in one pass over all 64
problems, returning 20 x 64 tour lengths. The engine generalises this:
stack ALL species into one block (2,000 x 1,880 genes today) and evaluate
in chunks of `eval_chunk` (128 today, now a config knob), folding results
back into species slices for selection and culling. Evolution logic does
not change; only the bookkeeping of which row belongs to which species.

- The decoder is sequential over picks (10 steps at n=10): at each step
  all members and all problems move together in one matrix operation.
  Generations still run in sequence — a future generation cannot be
  computed before the present one.
- Memory, as measured in chat: the genome block itself is ~15 MB at
  2,000 x 1,880; working space at the current size is tens of MB. A
  24 GB card could hold tens of thousands of members at this network
  size in one unchunked pass; when width, cities and population grow
  together, chunk the rows (or the problems) and run chunks back to
  back — same results, slightly less elegant. Roman's floated shape,
  640 species x 50 members (32,000 individuals), is a config choice,
  not a frozen default: it buys more substrate lottery tickets at the
  cost of more evaluation per generation. Default stays 100 x 20 until
  he changes it.
- GPU / JAX: NOT built now. Verified 2026-11-20: this machine has no
  GPU (`nvidia-smi` not found), no JAX and no PyTorch installed, and
  `BENCHMARKS.md` records NumPy-only on ~1.5 shared CPU cores. The
  engine stays NumPy with the same evaluator API, so a JAX evaluator
  can drop in later without rewriting the evolution logic. If the
  engine ever switches at a bigger size, the gradient method is policy
  gradient (REINFORCE-style), not plain backpropagation — tour length
  does not differentiate through greedy picks.

## 8. What this engine does NOT do

- No run is launched from this draft. The first run (a stage-3 replay
  as acceptance test, then stage 4) needs Roman's freeze of the open
  joints plus an explicit yes.
- No JAX / PyTorch install, no GPU, no cloud box — that is Roman's call
  when the engine outgrows this machine.
- No change to the paper draft's thesis handling: the attention line
  gets shuffle resistance by construction, not from the adapter; the
  paper call stays Roman's when he reads the draft (see
  `ATTN_STAGE3_2026-11-20.md` "Named growth units remaining" and the
  2026-11-20 chat record).

## 9. Open joints — Roman decides before freezing

1. Plateau threshold: what elimination-median slope over 30 generations
   counts as "stopped learning"? (No number set yet.)
2. Phase lengths: Phase A 50 or 100 generations? Phase B 50, 100, or
   more? (Roman gave both numbers as ranges.)
3. Growth step size: how many neurons is "a few" for a width step?
   (+4 proposed, from stages 1-2 precedent.)
4. First engine run's shape: default 100 species x 20, or his floated
   640 x 50? And at which city size — n=10 again, or start growing
   city size too?
5. Phase B reading: is "add the 10th mutation" the existing 1/10-rate
   old-gene rule (this draft's reading), or something else?
6. Budget units: generations only, or also a wall-time / evaluation
   cap? (Roman said "a budget" and "how many generations we have in
   the budget" — generations is the draft's reading.)
7. Config format: dict in code, or a small YAML/JSON file per run?
8. Which growth unit is stage 4: the remaining named units are the
   glimpse layer (decoder), FFN widening (hidden 16 -> wider), further
   attention heads (2 -> more), further whole blocks, width, city size.
   The ladder in section 5 orders the *kind* of step; the *specific*
   unit for stage 4 is still his pick.

## Sources (all read 2026-11-20)

- Order: chat 2026-11-20T13:58:41Z; cache idea 13:54:24Z and 13:55:52Z;
  batching idea 13:45:27Z–13:49:16Z; species/population discussion
  13:30:29Z–13:51:00Z.
- Baseline: `tsp_lab/run_tsp_attn_stage3.py` (constants lines 96-119,
  training draw line 680, checkpoint RNG state lines 445/477-478),
  `tsp_lab/attn_genome_s3.py` (gene layout, CHUNK=128, batched encode),
  `ATTN_STAGE3_2026-11-20.md`, `INVARIANCE_AUDIT_2026-11-20.md`,
  `BENCHMARKS.md` (NumPy-only, no GPU).
