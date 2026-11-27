# Plan: champion silencing, split-widening, and mutation marking

Status: LOGGED AND PREPPED, not built, not frozen. Sponsor direction by Roman,
2026-11-22 (early morning, phone). One more sponsor feature for the ending is
still to come; the last section of this plan is left open for it on purpose.

Domain: the parametric BERT evolution engine (traveler domain first, stock
domain inherits it through the same core). Code homes:
`~/workspace/paper-adapter/tsp_lab/` (dev) and `~/workspace/bert-engine/`
(export). All file and line references below are to the export tree as read on
2026-11-22.

## What Roman directed, in his words as settled in conversation

1. Widening today starts new neurons at zero effect. Add a second widening
   method, Net2WiderNet style: widen by SPLITTING neurons that already work.
   Especially for later in a run, when zero-effect neurons have had zero
   effect on the results.
2. Which neurons to split is decided by gene analysis: silence neurons and
   read which ones the champion actually needs.
3. A split copy gets a tiny random noise on its weights, plus or minus
   1/1000 to 1/100 of the weight, so the copy is not a perfect twin.
4. Silencing runs on champions generally, not only at widening time. It
   produces two markings:
   - important neurons: split candidates, and protected later (low mutation);
   - useless neurons: marked for a much higher mutation chance once the
     champion's frozen phase ends (Roman floated about 50%, against the low
     rate preserved old genes get today), so evolution recycles dead
     capacity instead of protecting it.
5. The backup-pair weakness of one-by-one silencing needs NO extra safety
   pass (Roman's correction, 2026-11-22): evolution handles it. A copy that
   mutates one of a covering pair is unharmed; a copy that badly mutates the
   remaining sole carrier dies in selection; and silencing is re-run at the
   next champion evaluation, so the surviving carrier then reads as
   important and gets marked for splitting and protection. The importance
   table is a per-champion snapshot that refreshes, never a permanent label.

## Where the code stands today (verified by reading it, 2026-11-22)

- Mutation rates are per gene CLASS only: `prob_vector(classes, phase, cfg)`
  in `bert_engine.py` (line ~477). Classes: draw (never mutates), old
  (phase A rate 0, phase B rate `mut_p * old_rate_factor` = 0.10 * 0.1 =
  0.01), interface and decoder (full rate `mut_p` = 0.10). There is no
  per-neuron or per-gene marking today.
- `transplant_widen` in `bert_engine.py` (line ~801) widens the shared
  stream width `d` by `widen_step` (default 4), function-preserving: new
  output sides at zero or at per-row means, new input sides from the species
  draw. It does not split existing units, and it widens `d`, not the
  feed-forward hidden width.
- Growth sequencing lives in `run_bert_engine.py`: `growth_attempt` (line
  ~436) with the preservation gate, and the decision point in `run()` (line
  ~558) using `growth_order` and `tried_rules`. The persistent-queue
  sequencer change Roman directed on 2026-11-21 (kept rule stays at the
  front, failed rule goes to the back) is a separate logged change, still
  awaiting his freeze; this plan is written to slot into either sequencer.
- Evaluation already stacks a whole population as one batched matrix and
  runs it in chunks of `cfg.chunk` (default 128), results bit-identical
  regardless of chunking (`encode_population`, `bert_engine.py` line ~378,
  and the chunk-invariance test leg). Silencing reuses exactly this path.

## The plan

### Step 1. Silencing table (new core function)

- New function in `bert_engine.py`, working for both domains through the
  existing domain evaluate path. Input: a champion genome, its layout, and
  the fixed validation set. Output: one importance score per testable unit,
  defined as champion validation score minus silenced-copy validation score
  (sign set per domain so that a bigger number always means more important).
- A silenced copy is a GENOME EDIT, not a change to the forward code: the
  unit's outgoing weights are set to zero in that copy, so its signal never
  reaches the rest of the network. Copies are stacked and evaluated as one
  population through the existing chunked evaluator, untouched champion
  included as copy zero.
- Testable units in version 1, in this order:
  - feed-forward hidden units, per block (16 per block at the current
    shape): silence by zeroing that unit's column in the block's output
    matrix;
  - whole attention heads, per block (2 per block at the current shape):
    silence by zeroing that head's slice of the output projection.
  Stream coordinates (`d`) are NOT silenced in version 1: a stream
  coordinate is woven through every matrix and every layernorm, so zeroing
  it is not one clean edit. Noted as a possible later extension, not
  silently dropped.
- Cost, stated up front: one batched pass of (number of units + 1) copies.
  About 37 copies at the current shape (32 feed-forward units + 4 heads +
  the untouched champion), against 2,000 evaluations in a single ordinary
  generation. At the RTX 2060 switch-zone shape (roughly 1,000 units) it is
  about half of one generation's evaluating, paid once per champion.
- When it runs: every time a champion is vaulted (stage end and growth
  keep), inside `run_bert_engine.py` next to the vault call. The table is
  saved as a JSON artifact beside the vault entry and carried in the
  checkpoint, so a resumed run keeps the current champion's table.

### Step 2. Split-widening (new transplant, widen variant B)

- New transplant beside `transplant_widen`, targeting the feed-forward
  hidden width, where splitting is one clean surgery: for each chosen unit,
  the new unit copies the original's incoming row and bias, and the
  original's outgoing column is halved between the two. Before any noise,
  the grown network computes exactly what the champion computed, and it must
  pass the existing preservation gate exactly, like every other transplant.
- Which units: the top units of the current champion's silencing table.
  How many per attempt: `widen_step` (default 4), same knob as widen.
- Noise: applied AFTER the preservation gate passes, as the first change of
  the new-part phase, on the copies' genes only. Each copied weight w gets
  w * u, with u drawn per weight from plus or minus the band
  [split_noise_lo, split_noise_hi], defaults 0.001 and 0.01 (Roman's
  1/1000 to 1/100). New config knobs, both in `EngineConfig`.
- Sequencer hookup (proposed, rides on Roman's freeze of the queue change):
  widen keeps two variants. Variant A is today's widen. Variant B is
  split-widening. When a variant A attempt is reverted, the next widen
  attempt uses variant B; the rule's place in the growth line does not
  change. Rationale in Roman's words: split-widening is the one to try
  especially later, when zero-effect neurons have had zero effect.

### Step 3. Mutation marking (per-neuron rates from the table)

- The silencing table is converted into a per-gene mutation multiplier
  vector aligned with the flat genome, stored with the champion and in the
  checkpoint. A neuron's genes are its incoming row, its bias, and its
  outgoing column.
- Three bands, cut from the table (proposed cuts, Roman to confirm):
  - important (largest drops): protected. Phase B rate stays the old-gene
    rate, `mut_p * old_rate_factor` = 0.01. These are also the split
    candidates of Step 2.
  - middle: unchanged, class rates as today.
  - useless (drop zero, or within the evaluation's exact-zero tolerance;
    the validation set is fixed and deterministic, so a true zero is
    measurable): marked rate `useless_mut_p`, proposed default 0.50 per
    Roman's figure, applying in phase B (phase A still freezes the base,
    as today).
- Implementation point: `prob_vector` gains an optional marks argument;
  marked genes take max(class rate, band rate) in phase B. Marks from a
  champion are discarded and rebuilt at the next champion vault. Nothing
  about draw genes changes: they never mutate and are re-imposed after
  breeding, marks or not.
- Deliberately NOT in this step: any group-silencing pass over the useless
  band. Superseded by Roman's correction (item 5 above).

### Step 4. Config and checkpoint changes

- `EngineConfig` gains: `split_noise_lo` (0.001), `split_noise_hi` (0.01),
  `useless_mut_p` (0.50), `silence_enabled` (True). All defaults reproduce
  today's behaviour when silencing is off.
- Checkpoint payload (`checkpoint_payload` / `load_checkpoint`,
  `bert_engine.py` lines ~1025 and ~1059) gains the current silencing table
  and the marks vector; the existing signature checks must refuse a resume
  whose marks do not match the layout, the same refusal discipline as
  architecture mismatches today.

### Step 5. Verification legs (builder checklist, evidence required)

1. Split transplant exactness: a split champion, before noise, reproduces
   the champion's validation score exactly (difference 0.0), same standard
   as the four existing transplants.
2. Noise leg: after noise, the validation score moves, and the move is
   small; the measured move is reported, not asserted.
3. Silencing batched equals silencing one by one, exactly (0.0), and
   chunk size does not change the table.
4. Silencing sanity: silencing a unit whose outgoing weights are already
   zero gives a drop of exactly 0; silencing every feed-forward unit in a
   block badly damages the score.
5. Marking: with marks applied, the measured mutation rate on useless-band
   genes in phase B matches `useless_mut_p`, and protected-band genes match
   the old-gene rate; marks survive a checkpoint save and load
   bit-identically; a new champion's table replaces the old marks.
6. Regression: the full existing suite passes unchanged with
   `silence_enabled` off, and with it on, a smoke run completes, vaults a
   champion, writes a table, and attempts a split-widen when growth fires.

### Step 6. Build order

Silencing table first (it feeds both other pieces), then marking (smaller,
rides on the table), then split-widening (rides on both). Each step lands
with its verification legs before the next starts. No full-budget run is
part of this plan; a run under the new machinery is a separate sponsor call.

## Addendum, Roman 2026-11-22: learn from the species, not only the champion

Silencing reads one genome, the champion. The population carries a second,
finer source of importance information, per individual weight, and it is
free: the genomes and their scores are already computed every generation.

- Per gene, per species, tracked as running statistics (cheap bookkeeping
  on arrays that already exist, no extra evaluations):
  - elite spread: how tightly the species' best members agree on the
    gene's value. Tight agreement under selection means the gene is being
    held in place because it matters.
  - elite shift: how far the elites' average sits from the whole species'
    average. Selection dragging a gene to a specific value is evidence it
    earns its place.
  - gene-to-score correlation across members: members who differ at this
    gene score systematically differently.
- Aggregated per neuron these statistics cross-check the silencing table;
  aggregated per zone (a block, a head, a feed-forward layer) they say
  WHERE the network is working and where it is lazy, at a grain silencing
  does not reach: every single weight, continuously through the cycle, not
  only at champion time.
- Standing between the two sources: population statistics are
  correlational and propose; silencing is causal and confirms at the
  champion. Split and protection decisions keep silencing as the deciding
  vote in version 1; the statistics rank candidates and feed the density
  reading below.
- Density (Roman's framing): the share of units and zones reading
  important. While density is low, there is dead capacity to recycle
  (mutation marks, Step 3) and lazy zones to target. As density rises
  toward everything reading important, the network is full of information
  and recycling has nothing left to give back; that is the evidence-based
  moment to grow. Which growth follows the map: a saturated zone gets the
  split or the widening, a broadly saturated network gets a new block or a
  stack, and a lazy zone gets marked, not fed. Version 1 reports density
  per zone at each champion and does not hard-code a growth threshold on
  it; the threshold, if any, is Roman's call once real readings exist.

## Addendum 2, Roman 2026-11-22: record everything, score every weight, heatmap

The champion is only known at the end, so importance cannot be computed
retrospectively unless the raw material survives the run. Roman directed:
record ALL genomes of ALL species at every generation, with their scores,
then analyse the winning species' history afterwards.

- Archive layout: per stage, one archive stream of per-generation records:
  the full population matrix (genomes), every member's score, and every
  member's species id. Written incrementally as the run goes (append per
  generation, atomic per record), never held in memory.
- Measured cost at the current shape: population 2,000 x 1,880 genes =
  30.1 MB per generation in float64 (15.0 MB in float32), 4.5 GB per
  150-generation stage, about 9 GB for a two-phase stage. This machine has
  146 GB free, so full recording is affordable now. It is NOT affordable at
  the switch-zone shape (a 100,000-gene genome x 1,000 members is 800 MB
  per generation, 120 GB per stage), so the archive carries a size guard:
  a config cap on archive bytes per stage; what happens past the cap is an
  open joint below, decided before any run that large, not improvised
  mid-run.
- Per-weight usefulness score (Roman's definition): every weight starts at
  exactly 1, neutral. Across the winning species' recorded history the
  score is multiplied up when the generation's evidence says the weight is
  useful, multiplied down when the evidence says useless, and multiplied
  by exactly 1, left untouched, when there is no evidence either way. A
  weight that stays at 1 is a weight the run never judged, and the heatmap
  must show it as such, distinct from a weight judged useless. Stored as a
  log score (additive), displayed as the score itself.
- Evidence per gene per generation, from that species' own members that
  generation (the Addendum 1 signals, now computed from the archive after
  the run, and keyed to the winning lineage): gene-to-score correlation
  across members, elite agreement, elite shift. The performance measure
  in every one of these signals is RELATIVE standing, never the raw score
  (Roman, 2026-11-22): each genome is stored with its score, and at
  analysis time each member is ranked against the other members of its
  own species in its own generation (rank, or score minus the species'
  median that generation). Raw scores are not comparable across
  generations because the training samples rotate, the same rule Roman
  set for the evolution observatory in the stock GA; a gene earns
  "useful" only by separating the winners from the losers standing next
  to it, in the same generation, on the same problems. Genes with no variation
  across the species that generation (frozen base genes in phase A, draw
  genes) produce no evidence and stay neutral for that generation, which
  is the honest reading, not a gap.
- Lineage keying: scores accumulate per species lineage. When a cataclysm
  or a growth step regenerates species from founders, the child lineage
  inherits its founder's score vector, the same way it inherits the
  founder's genes. The winning species at the end therefore carries a
  score history reaching back through its ancestors.
- Because the lineage archive holds every generation, the usefulness
  scores can also be computed as of any generation, not only at the end:
  one heatmap per generation. Rendered in sequence that is a video of the
  network's evolution (Roman's observation, 2026-11-22): density visibly
  spreading through the matrices as the species learns, beside the
  champion's improving result on a fixed problem. Optional output of the
  analysis step, cheap because every input to it is already archived.
- Deliverable per champion: a heatmap of the network, one panel per weight
  matrix (per block: attention projections, feed-forward matrices, plus
  embedding and head), coloured by usefulness score with neutral 1 as the
  midpoint colour, and a per-zone density table (share of weights reading
  useful, share useless, share never judged). This is the density reading
  of Addendum 1, now with a picture and exact numbers, and it feeds the
  same decisions: what to split, what to mark, what to prune, which zone
  is lazy, and whether the network is full enough to grow.
- Retention (DECIDED by Roman, 2026-11-22, and clarified the same
  morning): the full population archive is working data for the analysis
  only. What is kept permanently is the champion's LINEAGE, in full: every
  member of the winning species at every generation of each epoch (an
  epoch is one stage: its phase A and phase B), meaning the champion's
  species brethren who fell in battle are kept too, not only the
  per-generation champions; and where a cataclysm or growth step rebuilt
  species from founders, the ancestor species in the champion's line are
  kept the same way, so the line is unbroken back to the start. The
  per-generation champions, the final champion, the usefulness score
  vectors, and the heatmaps are kept with them. Every species outside the
  champion's lineage is disposed of once the analysis for that epoch is
  done. Kept size: the winning species is 20 members per generation, so
  about 0.3 MB per generation and roughly 90 MB per epoch of lineage
  archive, against the 9 GB working archive. Why the working archive is
  9 GB while the kept set is megabytes: the archive holds EVERY member
  (100 species x 20 members = 2,000 genomes of 1,880 numbers at 8 bytes
  each = 30.1 MB per generation, times about 300 generations), because
  the winning species is not known until the end; the champions are one
  genome per generation out of those 2,000.

## Addendum 3, Roman 2026-11-22: live run monitor (stock domain)

Roman asked for a way to watch the engine while it runs on JAX, and named
the contents himself. The monitor is a page the runner keeps fresh as it
goes; JAX only computes, the page reads files the runner writes atomically
(the same discipline as the checkpoint, which is already rewritten at the
end of every generation), so watching never touches the run.

Files, in the run's output directory:
- `monitor_status.json`, rewritten every generation: everything cheap.
- `monitor_snapshots.jsonl`, one appended record per detailed snapshot.
- `monitor.html`, a self-contained page regenerated at every snapshot,
  no server needed; it is the thing Roman opens.

Every generation (cheap, already known or nearly so):
- backend line: JAX or numpy, device (CPU or GPU), precision, and the
  fallback note if JAX or the GPU was requested but not found; seconds
  per generation and an estimated time left in the current phase.
- run position: stage (epoch), phase A or B, generation in phase and in
  total, species count, population, best and median training score,
  elimination median.
- network structure, drawn as it currently stands: number of stacks,
  and per stack its blocks (depth) and heads, stream width d,
  feed-forward hidden width, and total gene count.
- what is being added now: the growth rule whose stage is currently
  evolving (widen, add head, add block, stack, or none for the first
  stage), the growth round, the rules already tried this round, and the
  keep or revert verdicts from the event log as they land.

Detailed snapshots (every `monitor_every_gens` generations, set to 10:
Roman confirmed the proposed 10-generation cadence on 2026-11-22; also
at every phase end and stage end). One snapshot evaluates
the current leading genome (the species generation-best with the best
elimination score; at stage end this is the official stage champion) on
the frozen 64-episode validation set, with a detailed episode simulation
that also records the daily portfolio value path and the executed trades.
That extra recording is new work in the stock domain: today
`episode_cagrs` returns only the final CAGR per episode, so Sharpe,
drawdown, and trade counts cannot be had from the current output at all.
Per snapshot, champion against same-basket uniform buy-and-hold:
- CAGR: median champion CAGR, median buy-and-hold CAGR, median excess
  (the engine's fitness today), per-episode values kept in the record.
- Sharpe: per episode, from the daily value path (zero risk-free rate,
  annualised by sqrt(252)); median across episodes for the champion and
  for buy-and-hold, and the difference. Labelled as a 61-mark snapshot
  Sharpe: real, but noisy at 60 days.
- Max drawdown: per episode, from the same daily path; the headline is
  the worst episode's drawdown, with the median episode drawdown beside
  it, for champion and buy-and-hold.
- SQN: the trading program's own convention, adapted honestly. A trade
  is a completed per-name holding episode inside a validation sample (a
  maximal run of days with held weight above zero, closed before the
  sample ends; positions still open at the sample end are excluded, the
  `compute_sqn` rule in trading_agent/metrics.py). R is the name's
  compounded price return over the held days divided by twice its
  ATR(14) percent at entry (true range from the bundle's highs and lows;
  the documented close-based proxy only if highs or lows are absent).
  SQN pools the completed trades of all 64 samples:
  sqrt(min(N, 100)) * mean(R) / sd(R), with the count N and the
  small-sample flag (N below 30) always shown beside it. Buy-and-hold
  SQN is normally null on 60-day samples because its positions never
  close inside the sample; the page shows n/a there, never a fake zero.
- Trades per sample: executed ticker-day buys and sells actually filled
  in the simulation (an unaffordable buy that was ignored and charged
  the overdraft penalty is not a trade), mean and median per episode,
  buys and sells separately, plus turnover per episode and the completed
  holding-episode count N that feeds SQN. Executions and completed
  trades are different numbers and are never merged into one label.
- Percentile against random: at stage end and whenever the architecture
  changes, an exposure-matched random ladder on the same frozen episodes
  (the alloc_v3 convention: iid uniform scores 0 to 5, unit calibrated
  to the champion's mean invested share), 200 draws proposed, cached per
  architecture signature. The snapshot then reports the champion's
  percentile in that ladder. Between ladder computations the page shows
  the last computed percentile with the architecture it belongs to.

Evolution curves (the point of the page; every snapshot appends a point,
Roman added Sharpe, SQN, and max drawdown to this list explicitly):
- median CAGR: champion line and buy-and-hold line, plus the excess.
- median Sharpe: champion line and buy-and-hold line.
- SQN of the champion, with its N.
- max drawdown: worst-episode line and median-episode line, champion
  against buy-and-hold.
- information density, two readings: (a) population density, from the
  archive's cumulative per-weight usefulness scores for the leading
  species (share of weights reading useful), updated every snapshot;
  (b) masking density, from the silencing table at stage champions
  (share of tested units whose silencing drop is real). Each density is
  plotted on its own and multiplied by network size (density times total
  gene count, and density times tested-unit count): the effective
  information content of the network, so a bigger network that learned
  nothing shows a flat effective line under a rising size line.
- network size itself: total genes, and the structure numbers above, so
  every growth step is visible as a step in the size curve with the rule
  that caused it labelled.

Verification legs for this addendum: the detailed episode simulation
must reproduce today's final CAGRs exactly on the same genomes and
episodes (the new recording changes nothing it computes); hand-check
Sharpe and drawdown on one episode against an independent calculation
from its recorded daily path; hand-check the trade counter on one
episode against its recorded decisions; SQN on a fixed trace must match
a direct implementation of the stated convention; the monitor files
must be valid and complete after a kill mid-generation (atomic writes);
a smoke run must produce a monitor page whose numbers match its
results.json.

## Addendum 4, Roman 2026-11-22: chimera species, prune and graft (brainstorm, not frozen)

Roman's last idea before tonight's build: cross-species mutation, called
chimera species. Motivation: when information density is low there are
holes and gaps in a network, and the gaps should be filled with useful
material transplanted from other species rather than left to random
mutation.

Framing as stated: prune-and-graft sits alongside the existing growth
steps (widen, add depth, add a stack). Take the champion, prune it, and
keep the remaining base. Prune in two modes: a small, very safe prune,
and a more exploratory prune. Fill the gaps with grafted material from
other species, chosen on two conditions: (A) it carries useful
information, and (B) it is different from what the base already has.
Graft sizes vary, from microsurgery-sized chunks to more invasive ones,
composed at random.

Placement, as stated and not yet settled: Roman also said the rest of the
process stays as usual and only the way species are initially generated
changes. That sits against the growth-step framing above, so the plan
records both statements and does not resolve them here. Open: whether
prune-and-graft is a growth step applied to the champion, a way of
generating the initial species set, or both at different points; what
"useful" and "different" measure exactly (the silencing table and the
usefulness scores are the candidate readings, not yet chosen); prune
thresholds for the safe and exploratory modes; which units can be
grafted (feed-forward hidden units and whole heads are the version 1
silencing units, stream coordinates are excluded in version 1); and how
a grafted network is validated before it enters evolution (a
preservation-style gate is a candidate, not yet chosen).

Status: brainstormed 2026-11-22, NOT frozen, NOT built. No change to
tonight's build scope beyond the builder reading this addendum under the
existing instruction to check the plan for the ending feature.

## Addendum 4, Roman 2026-11-22 (voice brainstorm): chimera step, prune and graft

STATUS: brainstormed, NOT frozen, NOT in the Sunday-night build scope
unless Roman explicitly puts it there. Recorded so the design is not
lost.

Roman's idea, in his structure: cross-species grafting as ONE MORE
GROWTH STEP, sitting in the growth line beside widen, add head, add
block, and stack. It does not change how evolution runs day to day; like
every growth step it changes only how the species of the next stage are
initially generated, and everything after that is the usual machinery.

The step, applied to the champion:
1. Prune the champion, leaving a base. Two prune levels: a small, very
   safe prune, and a larger, more exploratory prune. (Natural readings,
   unfrozen: safe = remove only units whose measured silencing drop is
   exactly zero, the useless band of step 1; exploratory = also remove
   units up to a small positive drop, or the bottom slice by
   importance.)
2. Fill the gaps with material grafted from other species, under two
   conditions on the donor material: (A) it is known useful, per the
   donor species' own evidence (silencing and usefulness scores), and
   (B) it is different from what the base already has, so the graft adds
   information the base lacks rather than a second copy of what it has.
3. Regenerate the species set from the grafted bases, composing the
   grafts at random across the population: different chunks, different
   chunk sizes, some microsurgeries and some invasive grafts, so the
   stage explores many compositions at once.

Why Roman wants it: at low information density a network has holes that
random mutation fills slowly and blindly; other species have already
grown useful material that could fill those holes directly, raising
density by replacement (dead units out, proven units in) rather than by
growth alone.

Two mechanical notes from Clayton, checked against the current engine:
- Grafting is position-clean inside one run: every species of a stage
  shares the same architecture and layout, so a chunk (one feed-forward
  unit's incoming row, bias and outgoing column; or one attention head's
  matrices) sits at the same gene positions in donor and base.
- The exact preservation gate cannot govern this step: pruning changes
  the computed function on purpose, so a pruned and grafted transplant
  will not reproduce the champion exactly. For this rule the gate has to
  be the ordinary stage-end verdict (the stage's champion must beat the
  pre-step bar on the fixed validation set, else revert), which already
  exists. Whether Roman accepts that exception is his call at freeze
  time.

## Addendum 4b, Roman 2026-11-22 (second voice call): where the chimera step sits in the growth order

STATUS: brainstorm continued, still NOT frozen. This sketch extends
Addendum 4 and overlaps the still-unfrozen 2026-11-21 sequencer line;
the two sketches are to be reconciled when Roman freezes the sequencer.
Roman had to leave mid-discussion; this is where it stopped.

Roman's ordering for the events after a champion is kept:
1. Prune-and-graft FIRST, before any growth. It is the opening move on
   a fresh champion.
2. Then widen. Widening can leave fresh unused neurons behind, so after
   widening the cycle returns to prune-and-graft, which recycles them.
   The loop is: prune-graft, widen, prune-graft, widen, repeated until
   it stops helping.
3. Only when that loop stops helping, add a layer (add block). Then the
   whole cycle restarts on the deeper network: prune-graft, widen,
   prune-graft, widen. Per-stack depth cap in Roman's words: "it's a
   BERT network, let's not do more than three layers."
4. When depth is capped too, add a new stack. The new stack starts
   MINIMAL: small width, small depth, the fewest neurons that can work.
   The standing philosophy Roman stated: keep the number of neurons
   minimal, run a lot of parallel exploration, make every neuron earn
   its usefulness; only when a small addition does not help do you try
   a wider one.
5. Per-stack width and depth ceilings: once a stack reaches them, stop
   widening and deepening that stack and grow the next stack instead.
6. Global stop, in Roman's words: if a new stack has been added,
   widened, and brought to two layers of depth and it still does not
   help, growth is done.

Depth cap RESOLVED (Roman, voice call 2026-11-22 ~17:33 EST): the two
numbers do two different jobs. Within a stack, keep adding layers for
as long as each one helps, with a hard ceiling of 3 layers per stack.
A NEW stack is on a shorter leash: it starts minimal (one layer), goes
through the widen and prune-graft cycle, and when widening stops
helping it may add a second layer; if that second layer does not help
either, growth is DONE globally. A new stack never gets a third layer
to prove itself; the ceiling of 3 belongs to stacks that are already
earning their keep.

Per-stack growth order clarified (Roman, voice call 2026-11-22 ~17:35
EST): grow the CURRENT stack all the way before adding a new one, and
inside the stack, widening first and deepening second. In his words:
grow the current stack all the way, widening it first, deepening it
second, and once it is wide and deep, then add a new stack. Read
together with the cycle above: the widen phase is the widen and
prune-graft recycling loop run until width stops helping or reaches
its ceiling; only then does the stack deepen (to the cap of 3, or the
new-stack proving limit of 2); only when the stack is both wide and
deep does growth move to the next stack, which starts minimal and runs
the same order.

Max width as a configuration parameter (Roman, 2026-11-22 20:37 EST):
the width ceiling is NOT a number fixed in this design. It is a
parameter of the engine configuration (`max_d`, sitting beside the
starting width `d` and `widen_step`), set per run, so different runs
can carry different ceilings. Checked 2026-11-22: no such parameter
exists in EngineConfig today (only `d`, `widen_step`, and
`max_growth_rounds`); widening currently stops only when a rule stops
helping or the growth rounds run out. Under Roman's growth order,
hitting `max_d` is what ends the widen phase of a stack even if
widening is still helping, and moves growth to deepening. One
mechanical note: in the current engine the width is one global stream
width shared by every stack, so `max_d` caps that shared width;
per-stack widths do not exist today and are not assumed here. Roman
set the mechanism, not a value; the value belongs in each run's config
and no default is recorded until he sets one.

Prune-graft recurrence, density bar, and graft compatibility (Roman,
voice call 2026-11-22 ~22:32 EST, second call on the same topic):
- Prune-and-graft is also the move AFTER a layer is added, not only
  after widening. A new layer helps a little, but it arrives full of
  probably-useless neurons, and parallel species have developed
  features in the meantime, so prune-and-graft is the cleanup that
  follows a layer addition. No widening is needed right after adding a
  layer, because the new layer starts at the same width as the others.
- How many prune-and-graft rounds is not a fixed count. Prune-and-graft
  is how each species set is created, then it evolves for a capped run
  (50 or 100 generations, cap not yet set); at the end, the champion's
  density is checked again, and if there are still many empty neurons
  the cycle repeats.
- Density bar, Roman's figure, revised during the call: a champion is
  dense enough when MORE THAN 50% of its neurons are contributing
  (reading important on the silencing table, the density measure of
  Addendum 1). He first floated 75-80% and then corrected himself down:
  above 50% is good. The leftover non-contributing neurons are future
  space for grafts.
- Graft compatibility, Roman's rule: donor material must come from the
  same layer position as the gap it fills. Material developed in layers
  1 and 2 can only graft into a space in layers 1 and 2; it cannot be
  pasted into layers 2 and 3. Size and layer position must match, and
  the material must still be useful.
- End of the line: once the network is dense enough, a final prune or a
  gradient-descent pass is available as a last cleanup. Not designed,
  noted only.

Open points remaining: add_head was not mentioned anywhere in this
ordering, so where heads fit is undecided (Roman's order as stated has
no head step in it; the second call did not resolve it either); and
this loop has to be reconciled with the
2026-11-21 persistent-line sequencer sketch (front rule repeats on a
keep, failed rule goes to the back) before either is frozen.

TERMINOLOGY CORRECTION (Roman, 2026-11-23 00:01 EST, supersedes the
wording of Addendum 4b above wherever the two conflict): in the voice
calls Roman's word "layers" meant HEADS, and his word "stacks" meant
LAYERS. Read back under that mapping, and restated by Roman himself,
the growth order is:
1. Prune-and-graft first, on the kept champion.
2. Widen, and keep iterating the prune-graft and widen cycle, until
   widening is ineffective or the width cap is hit.
3. Add a head, and start over: prune-and-graft, then the widen cycle
   again. Heads repeat under the same rule until ineffective or the
   heads-per-layer cap is hit.
4. Only then add a layer. The new layer starts minimal and runs the
   same order from the top.
Everything has a use cap: width, heads per layer, and number of
layers. The width cap is the config parameter `max_d` (settled
2026-11-22 20:37 EST); the heads and layers caps sit in the config by
the same ruling, values per run, none recorded yet.
Mapped readings of the numbers settled in the second voice call
(presented to Roman 2026-11-23 for confirmation, not yet confirmed):
the "three layers" ceiling becomes a cap of 3 heads per layer, and
the new stack's "second layer" proving test becomes a new layer's
second head: a new layer starts with one head, runs prune-graft and
widen, may add a second head, and if that does not help, growth is
done. This correction also settles the heads question left open in
Addendum 4b: heads are a growth step of their own, third in the
order, after the widen cycle and before a new layer. In engine terms
Roman's "layer" is a block; one mismatch to iron out at freeze time
is that heads in the current engine are set per couple (a stack of
blocks sharing one head count), not per block.

Ceiling values (Roman, 2026-11-23 00:03 EST, supersedes the fixed 3
above): the heads and layers ceilings are USER INPUTS, config
parameters like `max_d` (so `max_heads` and `max_layers`), set per
run. `max_heads` is PER LAYER (confirmed by Roman 2026-11-23 00:04
EST): it caps the head count inside each layer, not a network total,
so total heads in the network are heads per layer times layers. No ceiling value is fixed in this design; the 3 from the voice
calls is retired as a fixed number. Roman's reason: BERT itself goes
up to 24 (BERT-large has 24 layers and 16 heads; BERT-base has 12 of
each), so a hard 3 seems small. The proving-test structure from the
mapped reading (a new layer gets a second head to prove itself) is a
shape, not a value, and still stands as presented. What a run should
actually set is a per-run choice; the recorded RTX 2060 switch zone
(evolution hands over to gradient descent around 8 to 12 total
blocks) is the practical evidence a config value has to respect,
whatever the parameter allows.

## Open joints for Roman (defaults proposed, none frozen)

1. `useless_mut_p` at 0.50: per-gene probability in phase B, as proposed
   above. (Against 0.01 protected and 0.10 full rate today.)
2. Band cuts: important = top `widen_step` units for splitting; useless =
   measured drop of exactly zero. Middle band is everything else.
3. Split-widening targets feed-forward hidden units only in version 1;
   stream-width splitting is not attempted.
4. Variant B is attempted after a variant A widen reverts, within the same
   place in the growth line.
5. Archive past the size cap (switch-zone shapes): proposed fallback is
   elites-plus-genealogy recording (every species' top members in full,
   plus parent indices and mutation seeds for the rest, enough to rebuild
   any genome exactly), instead of every genome in full.
6. RESOLVED 2026-11-22: species outside the champion's lineage are
   disposed of after analysis; the lineage is kept in full, every member
   of every species in the line, at every generation (see Retention).
7. Usefulness score step sizes: how far one generation of strong evidence
   moves a score (proposed: a bounded step, so one loud generation cannot
   swamp fifty quiet ones; exact numbers set from the first real archive
   before the first heatmap is trusted).

## Left open on purpose

- The ending: Roman's last idea, chimera species / prune-and-graft, is
  now logged as Addendum 4 above. It is a brainstorm, not a frozen
  spec, and the builder reads it under the existing check-the-plan
  instruction.
- The growth-sequencer queue change of 2026-11-21 still awaits Roman's
  freeze; this plan does not freeze it by the back door.

## Addendum 2026-11-23: useful-neuron freeze rule (sponsor)
Neurons marked useful (silencing-confirmed on a champion) receive NO mutation (per-gene rate 0, superseding the 0.01 protection rate). Rationale (sponsor): (A) never risk undoing what is proven to work; (B) marked units are exempt from re-silencing in later prune-and-graft rounds, so per-round measurement covers only the unproven remainder (at 65% density, 1,000 neurons: 350 tests, falling as density climbs). Open joints (Clayton's recommendations, unconfirmed): verbatim inheritance of marked units (no crossover mixing of their genes) as the condition that makes (B) sound; mark invalidation when an architecture change touches the unit's zone, plus a small periodic recalibration sample of marked units.
