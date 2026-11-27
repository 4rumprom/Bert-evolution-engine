# The BERT Evolution Engine

### A parametric engine that grows a small BERT by evolution, described from its code

Draft, 2026-11-23. Every claim in this paper traces to the engine's source
code (`bert_engine.py`, `run_bert_engine.py`, `bert_backend.py`,
`bert_domain_tsp.py`, `bert_domain_stock.py`, `bert_archive.py`,
`bert_monitor.py`, `bert_preprocessor.py`, `bert_stock_data.py`) or to a
dated test record in the same directory. Where an experiment was run by a
standalone test script rather than by the engine itself, the paper says so.

Two sections are placeholders. The sponsor has directed changes to how
growth stages mutate the network, to how freeze marks treat layer-norm
genes, and a new branch-level grafting operation. Those changes are being
made in the code first; the growth and grafting sections below will be
rewritten against the changed code.

---

## 1. What the engine is

The engine is a single program that evolves a small neural network. The
network it evolves is a **BERT**: a stack of identical transformer blocks
of the kind introduced with BERT (Bidirectional Encoder Representations
from Transformers), preceded by one learned input projection (the
embedding). There is no other kind of network in the genome. All
evolution, pruning, and growth operate inside those BERT blocks.

Three properties define the engine:

- **One engine, configured, not rewritten.** Every architectural and run
  variable lives in one configuration object (`EngineConfig`, a Python
  dataclass mirrored to JSON). Earlier work used a separate hand-written
  runner per growth stage; this engine replaced them. The same core runs
  both problem domains and both compute backends.
- **Evolved, never gradient-trained.** Nothing in the codebase computes
  a gradient or calls an optimizer. Populations of networks are scored,
  bred, and mutated. Even the smallest parameters, the layer-norm gains
  and biases inside each block, are ordinary genes in the genome and
  evolve by mutation like every other weight. (Layer norm is the
  per-vector rescaling inside each block: subtract the vector's mean,
  divide by its spread, then apply a learned gain and bias per
  coordinate.)
- **Two problem domains, plugged in as adapters.** A domain adapter
  supplies what the core does not know: the input features, the output
  head's shape and meaning, the fitness score, the fixed test sets, and
  the exactness check used when the architecture grows. Two adapters
  exist: a traveling salesman problem (TSP) adapter, which asks the
  network to order cities into a short tour, and a stock allocation
  adapter, which asks it to divide a portfolio among a basket of stocks.
  All scores inside the engine are higher-is-better; adapters negate
  their natural loss metrics (tour length, for TSP) to comply.

The engine's unit of work is a **stage**: a fixed budget of generations
of evolution, after which the best network (the stage champion) is
scored on a fixed validation set, vaulted to disk, and measured for
usefulness (Section 4). Between stages, the architecture may grow
(Section 5).

## 2. The genome and the architecture

A **genome** is one flat vector of real numbers: every weight of one
network, laid end to end. A **layout** (`Layout`, generated from an
`ArchSpec`) is the map from positions in that vector to named arrays, so
growth operations can remap genes by name when the architecture changes
shape, and no gene's meaning depends on its old position surviving.

The architecture is a stack of one or more **couples**. A couple is a
group of transformer blocks that share one input projection and one
output head. (Couples exist so the engine can stack a second encoder on
top of a first one later; every run so far uses a single couple.) Each
couple contains:

- **An embedding**: one learned linear projection from the raw input
  features to the network's working width, called **d**.
- **Blocks**. Each block is the standard BERT block in the modern
  pre-norm arrangement: the block's input stream is layer-normalized,
  passed through a branch, and the branch's output is added back to the
  stream. The first branch is multi-head self-attention; the second is
  a feed-forward network (FFN) that expands the stream to a wider
  hidden width and projects it back. The FFN activation is tanh.
- **A decode head**, supplied by the domain adapter (below).

Details that matter:

- **Heads carry full projections.** Each attention head has its own
  complete (d, d) query, key, value, and output projection matrices.
  Heads do not split d into slices, as textbook BERT does; each head
  reads and writes the full width, and the heads' outputs are summed.
  The head count is set **per couple**: every block in a couple has the
  same number of heads. (Whether heads should be set per individual
  block instead is an open item, Section 9.)
- **The FFN width is a separate config value.** In the configurations
  used for the current lineage it is held at 4 times d, the ratio BERT
  itself uses; the engine does not enforce the ratio, the configs do.
- **The attention scale is frozen at birth.** Attention logits are
  scaled by 1/sqrt(d) using the *initial* width, and widening the
  network later does not rescale it; rescaling would silently change
  every attention score the champion has learned.
- **No positional encodings.** The operations are deliberately
  permutation-equivariant: relabeling the input tokens relabels the
  outputs the same way and changes nothing else. For the TSP this is a
  correctness property (a tour must not depend on the order cities are
  listed in), and the test suite verifies it bit-exactly (Section 7).

The two decode heads:

- **TSP**: a pointer decoder. Starting from a learned start vector and
  the mean of the encoder outputs, it repeatedly scores every unvisited
  city by attention, picks the best (with the usual tanh clipping of
  logits at 10), and appends it to the tour. Fitness is the negative
  mean tour length; the validation score is the negative mean gap
  against exact Held-Karp optima on a fixed set of 200 problems with
  10 cities each.
- **Stock**: a parallel allocation head. A shared per-stock score is
  quantized into levels 0 to 5, and a separate read produces one unit
  size (a percentage of portfolio value, capped at 6% by a sigmoid).
  Section 6 describes the execution and fitness rules.

**Batched evaluation.** The whole population is evaluated as one
stacked matrix of shape (population, n_genes): the population axis runs
through every operation, and results are folded back per member and per
species. Two consequences are engineered carefully:

- *Chunking is free.* The population axis is processed in chunks
  (config `chunk`, default 128). Rows are independent and the batched
  operations are row-wise bit-identical, so chunk size never changes a
  result; it only bounds memory.
- *Shared prefixes are computed once.* Encoding proceeds in segments
  (embedding, then per block: attention segment, FFN segment). Within a
  group of genomes (a species, typically), if a segment's genes are
  identical across the group and its input is still group-shared, the
  segment is evaluated once from one representative row and cached.
  Because frozen champion material is shared by construction, most of
  a species' encoding is usually shared; only the newest, still-varying
  component is computed per member. Cached values are the exact floats
  a full recompute would produce.

**Backends.** One code path runs on two array backends
(`bert_backend.py`): NumPy, the float64 reference, and JAX, the
portable backend that runs the identical code on CPU here and on GPU on
the sponsor's PC (device and precision are config values). With
precision `x64`, JAX results match the NumPy reference (the suite
checks agreement below 1e-9). If JAX is requested but not importable,
the backend falls back to NumPy and records the fallback in the run's
results rather than failing. Host-side logic (breeding, culling,
checkpointing) is always NumPy, and checkpoints are
backend-independent.

## 3. Evolution

**Species.** The population is divided into species. A species is a
family of genomes that share one random draw of the newest component's
input-side genes on top of a preserved base; the draw is what makes the
species distinct, and it is re-imposed on every child after breeding,
so a species can never drift out of its own territory. Breeding happens
only within a species. Each species tracks its best-ever genome by
training score.

**Gene classes.** Every gene carries a class that sets its mutation
rate by phase:

| Class | Meaning | Phase A | Phase B |
|---|---|---|---|
| draw | the species-defining random draw | never mutates (re-imposed) | never mutates |
| old | material from earlier components | frozen | mutates at 1/10 the standard rate |
| interface | the newest component's output side | full rate | full rate |
| decoder | the active decode head | full rate | full rate |

A stage runs in two phases. **Phase A** evolves only the newest
material (interface and decoder genes); the base is untouched.
**Phase B** evolves the whole network, with old genes at the reduced
1/10 rate. Crossover (uniform, per gene) applies only to interface and
decoder genes; old genes are inherited whole from one parent. Selection
is by tournament (size 3 by default), with the top 2 genomes of each
species copied unchanged (elites). Mutation adds Gaussian noise
(sigma 0.15 by default) to each gene with its class probability
(standard rate 0.10).

**Schedule A culling.** Every 3 generations by default, species are
ranked by their champion's score on a fixed elimination set; the worst
20% of species are dropped, down to a floor (10 species by default),
and the survivors' population targets are redistributed so the total
population is preserved.

**Founders and cataclysm.** At each growth transition, the engine
selects **founders**: the best champion of each of the top-K *distinct*
species on the fixed validation set (K is config `founder_count`,
default 1). Founders are vaulted, and the next stage's species set is
regenerated split as evenly as possible among them, each new species
carrying its founder's lineage label. This regeneration path is the
**cataclysm**: a periodic mass extinction in which only the founders'
lineages continue. The same single code path also performs ordinary
growth-stage regeneration, so the two mechanisms cannot diverge.

**Plateau detection.** Each phase watches its elimination-set series;
if a fitted line over the last `plateau_window` generations (default
30) improves by less than `plateau_threshold` (default 0.5%) relative
to its level, the phase ends early and the plateau is recorded.

**Checkpointing and resume.** After every generation the runner writes
an atomic checkpoint (write to a temporary file, then rename) holding
the full population, every species' best genome, both random number
generators' complete state, the gene classes, the mutation marks, the
usefulness tracker state, and fingerprints of the config, architecture,
and layout. Resume **refuses** (exits with an explicit message) if any
fingerprint does not match the current run exactly, including a marks
vector of the wrong length. A run writes its `COMPLETE.txt` marker
only after evolution, validation, and champion vaulting are all on
disk, so a missing marker always means an unfinished run.

There is no random-immigrant or heavily-mutated-clone operator in this
engine's code. Diversity comes from the species draws, from founder
regeneration, and from culling pressure. (Immigrant and "Chernobyl"
clone operators exist in the separate V3 genetic allocation program,
not in this engine.)

## 4. Usefulness measurement

The engine measures which parts of a champion are doing work, in two
independent ways.

**Silencing.** When a stage champion is vaulted, the runner computes a
**silencing table**: for each testable unit of the champion, one copy
of the genome with that unit's outgoing weights zeroed, all scored in
one batched call on the fixed validation set. The unit's **drop** is
the champion's score minus the silenced copy's score: a bigger drop
means a more important unit. Testable units in this version are the
FFN hidden units and the whole attention heads; stream coordinates are
not silenced, because a coordinate is woven through every matrix and
every layer norm, so zeroing one is not a clean single edit. The
table's headline read is a **density**: the share of units whose drop
is real (above the domain's exact-zero tolerance; for TSP, a drop of
exactly 0.0 is achievable because scores are tour-based). The stock
domain's tolerance is 1e-9, since its scores are continuous.

**Mutation marking.** From the silencing table the runner builds a
per-gene **marks** vector that adjusts phase-B mutation rates
(`marks_from_table`): the top few units by drop (the count is the
config `widen_step`, 4 by default) are *protected*, held to the
old-gene rate; units whose drop is at or below the zero tolerance are
*useless*, raised to a high mutation rate (config `useless_mut_p`,
default 0.50) so dead capacity is recycled; the middle is unmarked. In
phase B a marked gene's rate is the maximum of its class rate and its
mark. Marks ride in the checkpoint and are remapped by gene name
across any architecture change, so a champion's evidence survives
growth.

**The freeze rule, as practiced in the harness experiments.** The
sponsor's standing rule is stronger than marking: a unit *proven*
useful (its silencing drop clears a paired-noise band: twice the
standard error of its per-problem score differences) mutates at rate
**0**, is inherited verbatim, and is exempt from re-silencing while so
inherited. In the current code this rule is implemented by the
standalone experiment scripts of Section 8, which build their own
probability vectors with the frozen genes zeroed; the runner's own
marking mechanism is the protected/useless banding above. Making the
freeze rule, including its extension to layer-norm genes, an engine
operation is one of the directed changes now in progress.

**Per-weight usefulness archive.** Separately from silencing, the
engine keeps a running per-weight usefulness score for every species
(`bert_archive.py`). Each generation, every gene of a species earns a
small log-score step (config `usefulness_step`, default 0.02) from
three signals computed from the species' own genomes and scores: the
correlation between the gene's values and members' relative standing,
whether the top quarter of members agree on the gene more tightly than
the whole species does, and whether the top quarter's mean has shifted
away from the species mean. Genes that do not vary in a generation earn
exactly 0, so a gene the population never touches reads as
*never judged*, not as neutral. Children inherit a copy of their
founder's vector at regeneration; vectors are remapped by gene name
across growth, so the evidence survives architecture changes. The
archive also records per-generation population snapshots under a
per-stage byte cap (default 16 GiB), and at the end of a run it
analyses the archive: it traces the winning champion's lineage back
through cataclysms and growth steps, keeps the archive records on that
chain (plus stage-boundary records), deletes the rest, and writes a
usefulness heatmap page.

**Live run monitor (stock domain).** For stock runs the runner also
maintains a monitor (`bert_monitor.py`): every 10 generations by
default, and at phase and stage ends, it snapshots the current leader
with a detailed episode simulation (portfolio value path, trades,
turnover, invested share, per-episode statistics including a Van Tharp
SQN computed from trade R-multiples), compares the champion against a
cached ladder of random genomes at matched invested share, and renders
a status file and an HTML page. The detailed simulation is guaranteed
to change nothing it measures: final scores are bit-identical with or
without it.

## 5. Growth

> **TO BE REWRITTEN AFTER THE CODE CHANGES.**
>
> This section will describe the growth machinery as it stands after
> the directed changes land: growth stages that evolve only the added
> material (the pre-existing base untouched for the whole growth
> stage; whole-network mutation reserved to prune-and-graft stages),
> and freeze marks extended to a block's layer-norm genes. The
> transplant operations themselves (widen, including the Net2WiderNet
> split variant; add head; add block; stack a couple), the exact
> preservation gate each transplant must pass before any evolution,
> and the runner's growth sequencer (rules attempted least-change
> first; a stage is kept only if its champion strictly beats the
> previous champion on the fixed validation set, otherwise the engine
> reverts to the champion architecture and rebuilds from the founders)
> are implemented in `bert_engine.py` (`make_transplant` and the
> `transplant_*` functions) and `run_bert_engine.py`
> (`growth_attempt`, `run`), and will be documented here from that
> code once the changes above are in.

## 6. Grafting

> **TO BE REWRITTEN AFTER THE CODE CHANGES.**
>
> Branch-level grafting, isolating a branch as a cluster of units
> whose per-problem silencing profiles track together and proving it
> by group silencing, is being made an engine operation. Unit-level
> prune-and-graft exists today only in the standalone experiment
> scripts measured in Section 8; this section will describe the
> engine operation from its code when it lands.

## 7. The stock side

**The panel.** The stock adapter consumes a panel of daily bars. The
production data artifact is a single bundle file
(`stock_data_bundle.npz`, 42.5 MB) built from the trading program's
point-in-time price store and volume sidecar, aligned on the SPY
trading calendar: **666 tickers over 2,989 trading days,
2015-01-02 to 2026-11-19**, with a training-window end index of 1258
(the last day of 2019). Alongside open/high/low/close/volume, the
bundle carries four point-in-time fundamental series derived from
SEC filings data: earnings yield, book to market, return on assets,
and debt to assets (612 of the 666 tickers covered). The loader
(`bert_stock_data.py`) validates shapes, ticker uniqueness, and
calendar order, and can remap the frozen validation episodes (whose
stored indices address an older panel ordering) into bundle rows by
ticker name.

**Features.** Each stock on each day is described by a feature vector
built by a configurable preprocessor (`bert_preprocessor.py`). The
feature menu is itself a config value: a list of blocks chosen from
Fourier coefficients of the trailing price window (the legacy
fourier/volatility/wavelet blocks reproduce the trading program's
earlier feature code exactly), volatility, RSI, average true range,
Chaikin oscillator, stochastic oscillator, EMA, SMA, on-balance
volume, candlestick geometry, and fundamental pass-throughs. Every
feature at day t uses only bars up to day t. Features are standardized
per feature over all tickers and days before a cutoff; for bundle
panels the cutoff defaults to the bundle's training end index, so a
run cannot leak post-training statistics into its features by
omission. With the stock config's spec, the preprocessor produces 30
features, and the network's input width is 31: the 30 features plus
the stock's current position as a fraction of portfolio value. (The
adapter also supports a legacy contract of 9 precomputed Fourier
features plus position, input width 10.)

**Episodes and fitness.** Evaluation runs over **episodes**: a basket
of 10 stocks over 60 trading days. On each decision day the network
scores the basket, instructions execute in ranked order (raw score
descending, ties broken alphabetically for determinism), cash-checked,
at a cost of 6 basis points per executed change, with one 6bp
overdraft penalty on the shortfall of any unaffordable buy. Fitness is
**basket-relative**: the median over episodes of the episode's
compound annual growth rate minus the uniform buy-and-hold growth
rate of the same basket. This is the sponsor's validity bar built into
the fitness function itself: a genome earns nothing for riding a good
basket, only for beating it. The validation set is a frozen common
set of 64 episodes (basket 10, 60 days, drawn from 2020-2022), never
resampled or filtered.

**Validity-aware sampling.** Training and elimination episodes are
sampled, but a sampled (basket, start) counts only if every basket
ticker's price is finite on every day of the episode window plus its
final mark day, so neither the genome leg nor the buy-and-hold leg
can produce a NaN. The sampler's first attempt is byte-for-byte the
plain uniform draw, so on a clean panel it returns exactly the
episodes the old sampler did; only an invalid draw consumes redraws
(bounded at 200, then a deterministic scan). Measured on the real
bundle, only **54.4%** of uniform (basket, start) pairs are valid. A
panel that admits no valid episode at all raises a clear error up
front instead of looping.

**Preservation, stock form.** The stock adapter's exactness check
requires the discrete decisions (score levels and ranked execution
order) to be identical on every validation episode day after a
transplant, with per-episode growth rates agreeing to below 1e-9.

**Status.** The stock side is smoke-tested end to end (synthetic
panels, the preprocessor path, validity-aware sampling, the monitor)
but has not had a production run: the shipped stock config carries
smoke-scale episode counts (3 training and 4 elimination episodes per
generation), and the adapter performs no point-in-time index
membership filtering. See Section 9.

## 8. Verification

The engine is verified by three test suites, all run and passing on
2026-11-23 (the JAX-parity leg of the core suite requires a JAX
interpreter flag and was skipped in that run):

- **Core suite** (`test_bert_engine.py`): **70 of 70** checks. These
  cover config handling, layout and transplant exactness, the
  preservation gates, checkpoint/resume refusal, breeding and culling
  rules, silencing and marking, the stock adapter's execution and
  validity-aware sampling, and cross-checks against the earlier
  stage runners' arithmetic.
- **Archive suite** (`test_bert_archive.py`): **11 of 11** checks,
  including that the online usefulness tracker and the post-hoc
  archive analysis agree exactly (max difference 0.0) across a growth
  step.
- **Monitor suite** (`test_bert_monitor.py`): **8 of 8** checks.

Two properties deserve stating separately:

- **Invariance.** In the minimal engine test (below), a shuffle audit
  of the champion over 400 relistings of the validation problems
  changed its tour lengths by exactly 0.0, mean and max: the
  permutation-equivariance the architecture promises is the one it
  delivers.
- **Reproducibility.** Two identical minimal-test runs produced
  bit-identical champion genomes and identical validation scores.
  Deterministic reproduction is also used as a working tool: the
  harness experiments of Section 8 rebuild donor pools by re-running
  an earlier arm and asserting the champion comes back bit-identical
  before trusting the pool.

**The minimal test** (record: `ENGINE_TEST_2026-11-20.md`, protocol
written before the run). A small TSP run (12 species of 10, 40
generations, the 1,880-gene architecture, growth and cataclysm off)
passed completion, culling, invariance, and reproducibility. Its
learning leg **failed** the pre-stated gap bar: the champion's mean
validation gap was +41.20% against a bar of at most +30%, though its
elimination-set best improved 13.8% relative over the run and the
champion sat 4.8 points under the random median (+45.98%). The record
reads this as a scale effect (at this budget only the 80 decoder genes
evolve; the 1,800 encoder genes are the frozen species draws), not a
machinery defect, and says so rather than moving the bar after the
fact. Timing: 0.11 seconds per generation at population 120, roughly
7 times faster per unit of work than the bespoke stage runner it
replaced.

## 9. Measured results so far

All results in this section come from **standalone experiment scripts**
in the engine directory (see the note at the end of this section), run
on the fixed 200-problem TSP validation set. Scores are mean tour gap
in percent; lower is better. Margins between two arms are stated
relative to the control arm's gap, the sponsor's convention: a margin
of +16% relative means the treatment's gap is 16% smaller than the
control's, relative to the control's gap.

**The first lineage (a wide, sparse starting champion).** Starting
from the minimal test's champion (gap 41.20%, 1,880 genes, only about
a seventh of its units measurably contributing):

- *Prune-and-graft, round 1.* Pruning the 26 units with no positive
  drop was free (41.20% to 41.06%); refilling the slots with donor
  units and evolving 50 generations reached 12.81% against a control's
  20.35%, a margin of **+37% relative**. On a sparse champion,
  prune-and-graft won clearly.
- *Round 2.* The round-1 champion was already dense; pruning its 8
  remaining non-positive units *cost* score (they help in
  combination), and the treatment lost to its control (14.05% vs
  10.78%). The stopping rule fired; later rounds were not run.
- *Widening.* Widening the round-2 control champion (width 8 to 12)
  was kept under the engine's keep rule (9.82% beats its own start)
  but lost to plain evolution at equal budget (8.77%). A split-widen
  variant (widening the FFN by duplicating working units) did worse
  than its own start (11.07%) and would have been reverted.
- *Convergence check.* 100 further plain generations on the lineage's
  best champion moved it from 8.18% to **8.15%** (a 0.35% relative
  gain, against 16-19% per 50 generations earlier in the lineage).
  The lineage is converged for plain evolution and prune-and-graft
  alike at width 12; its best champion is 8.15% at 3,620 genes, with
  every one of its 36 units showing a positive silencing drop.

**The new lineage (a minimal BERT, grown step by step).** Restarted
at the smallest BERT proportions: width 2, FFN width 8 (the 4x
ratio), one head, a single block, 88 genes.

- *Seed run.* 40 generations: champion gap **22.03%**. Of its 9 units,
  2 were proven contributing under the paired-noise band (the head,
  whose silencing more than doubles the gap, and one FFN unit);
  1 was proven harmful.
- *Prune-and-graft.* Pruning the 4 non-positive units improved the
  base to 21.43%; with all 4 slots graft-filled, the treatment reached
  **18.80%** against the control's 18.91%, a margin of only +0.6%
  relative. A 9-unit network has almost no dead space, so the
  operation saturates almost immediately.
- *Double widen.* Widening by doubling everything at once (width
  2 to 4, FFN 8 to 16, heads 1 to 2, the FFN units and the head grown
  by Net2WiderNet splitting: duplicate exactly, halve the outgoing
  weights in both copies) passed the preservation gate exactly and
  reached **14.00%** against a control's 16.75%, a margin of **+16.4%
  relative**. A companion run freezing only the 2 units proven under
  the sponsor's cloud classification (Section 9, open items) instead
  of the 6 proven under the noise band reached 15.10%: the stricter
  freeze won on both arms of that comparison.
- *Prune-and-graft on the widened champion.* Density before: 3
  contributing, 15 neutral, 0 harmful of 18 units (cloud read).
  Pruning the 5 dead units improved the base for free (to 13.66%),
  but the grafts did not pay: treatment 12.57% against a control's
  **11.85%**, a margin of -6.1% relative. The control champion is the
  lineage's current best: **22.03% -> 18.80% -> 14.00% -> 11.85%**
  across the seed, prune-graft, widen, and consolidation steps.

**Harness experiments vs engine.** The prune-graft, widen, and
freeze-rule results above were produced by standalone scripts
(`prune_graft_*.py`, `widen_test_*.py`, `double_widen_*.py`,
`coupled_widen_*.py`, `prune_widen_chain_*.py`,
`optimize_control_*.py`, `clustering_units_*.py`,
`branch_graft_test_*.py`) that drive the engine's primitives (species
creation, breeding, culling, silencing, transplants) in their own
loops. They are not the runner: they omit checkpointing, the monitor,
and the archive tracker, and the runner's own growth sequencer has
not yet run a full-budget stage sequence in either domain. What the
harness results establish is the behavior of the operations; what
remains unproven is the assembled runner at production budget.

## 10. Limits and open items

- **No production run yet.** Every measured result is at test scale.
  The default TSP config (100 species of 20, 150 generations per
  stage) has not been run end to end through the growth sequencer,
  and the stock adapter has run only on synthetic panels and in
  smoke configurations.
- **The stock run is not production-ready.** The shipped stock config
  carries smoke-scale episode counts, and the adapter has no
  point-in-time index membership filtering: episodes sample from the
  whole bundle panel as it stands, not from the index membership as
  of each episode's date.
- **The growth and grafting changes are in flight.** Growth stages
  that evolve only the added material, freeze marks extended to
  layer-norm genes, and branch-level grafting as an engine operation
  are directed and being implemented; Sections 5 and 6 will be
  rewritten against the changed code. Until then, prune-and-graft
  remains a harness procedure, and the runner's phase B still mutates
  old genes at the 1/10 rate inside growth stages.
- **Heads are defined per couple, not per block.** The sponsor's
  intent is a head cap per layer (block); the architecture currently
  sets one head count for every block in a couple. Reconciling this
  is part of the pending freeze of the growth design, along with the
  three growth caps (maximum width, maximum heads per layer, maximum
  layers), which are sponsor inputs and are not yet fields in
  `EngineConfig`.
- **Classification of units is read two ways.** The harness reports
  density under both the paired-noise band (a unit is proven when its
  drop clears twice its standard error) and the sponsor's near-zero
  cloud rule (drops expressed as a percentage of the champion's gap;
  the neutral group is the cloud within two robust standard
  deviations of zero; outside it, positive is contributing and
  negative is harmful). The two reads agree on dense champions and
  differ on sparse ones; the freeze-marks comparison of Section 9
  recommends the band for freezing and the cloud for reporting, and
  the sponsor's decision is recorded as pending in the experiment
  files.
- **Usefulness shortlists are unproven at scale.** The archive's
  per-weight tracker was proposed as a way to nominate silencing
  candidates without silencing every unit of a large champion. In the
  one direct check, the tracker's bottom half and the measured prune
  set did not overlap, on a champion that had only one prunable unit;
  the check was inconclusive by construction, and two rules from it
  stand: shortlists must be drawn only from unproven units, and a
  never-judged unit reads as unknown, not as middling. Exhaustive
  silencing remains the measurement of record, and its cost grows
  with the champion.
- **Backends.** The JAX parity leg of the core suite was skipped in
  the 2026-11-23 run for want of the interpreter flag; cross-backend
  agreement below 1e-9 is asserted by the suite when run with JAX
  available. On a GPU in float32, greedy decoding can flip on
  near-ties; that is documented in the backend module as accepted.
