# The Frankenstein Champion Method

Set by the sponsor, Roman, on 2026-11-25. Plain words. Short sentences.
Updated 2026-11-26 with the results so far. The lifecycle was frozen in its
final form by the sponsor on 2026-11-27 and written into the engine as
`bert_lifecycle.py` the same day; the lifecycle section below is the
standing method. Amended 2026-11-27 (afternoon): the sponsor ruled that
what was just done with Frankenstein is the logic for evolution, so the
widen both completed lines ran is now step 5 of the lifecycle; the morning
freeze had omitted it.

## What it is

A way to grow a BERT network by combining two champions, then refining the combination. Evolution builds the parts. Evolution refines the joins. The result is the Frankenstein champion.

## The lifecycle (frozen by the sponsor, 2026-11-27)

This is the standing lifecycle. It supersedes the step list set on 2026-11-25 and the deep-narrow cycle set on 2026-11-26 (both kept below, marked superseded; their results stand). The sponsor's words for the shape of it: deep and skinny evolution runs. Then Frankenstein. Then a new skinny deep stack with evolution. Then Frankenstein. Then open evolution. Then start over to build a new Frankenstein. Then merge. Then stack skinny and Frankenstein it, repeat to grow a new Frankenstein stack. Frankenstein them together. Then free evolution. Keep going until a cap is hit or there is no improvement. Keep the smallest network that had the best score. Later the same day he added the ruling that what was just done with Frankenstein is the logic for evolution; the widen both completed lines ran between the backend Frankenstein and open evolution is therefore step 5 below.

The lifecycle is written into the engine as `bert_lifecycle.py`, with every cap and budget an input in `bert_lifecycle_config.json`. One scoreboard throughout: the fixed 200-problem validation set, 10 cities, val_seed 20261116. Lower gap is better.

**Step 1. Deep and skinny evolution runs.** Found a brand-new network from scratch: deep and narrow (the proven shape: width 2, feed-forward 8, 1 head, 4 blocks), several species, nothing frozen, the whole network evolves as one BERT. The founding budget is 50 generations. The top two species champions are the parents for step 2.

**Step 2. Frankenstein.** First, the parents mature: each parent is evolved further, 100 generations, whole network free, before any stitching. Young champions are not stitch-ready. Then isolate the path in each parent. Measure the signal (activation times weight) across each synapse on the fixed set. A synapse is on the main path only if it carries at least half the flow of the busiest synapse feeding the same node, and it lies on a connected entry-to-exit route. The path work extracts the branches. Stitch the top two champions only, no testing matrix: branches side by side, sharing no node, each on its own stream coordinates, so the stream width doubles. Drop only the synapses the path read calls irrelevant. Holes are tiny random filler, std 0.03. One joining block is added, all filler. The decoder is merged: both parents' decoder weights side by side in the same matrices, each on its own coordinates, diagonal blocks exact, cross blocks tiny random, the bias and start vectors concatenated by territory. Then refine: evolve the stitching genes only (per-gene rate 0.10, Gaussian sigma 0.15, population 100, 200 generations), the backbone frozen and verified bit-identical. What comes out is the line's Frankenstein.

**Step 3. A new skinny deep stack, with evolution.** Append one narrow block at the back of the Frankenstein, after the joining block, before the decoder. One head. Its weights are tiny random filler, std 0.03, layer-norm gains 1 plus noise of 0.01. Evolve only the new block (population 100, 100 generations). The trunk never moves.

**Step 4. Frankenstein again.** The stack run breeds a population that differs only in the new block. Score the whole final population on the fixed set. Take the top two distinct champions. Stitch their back blocks side by side inside one widened block, over the shared trunk: the trunk is kept once, bit-exact, never doubled; the widened block carries both champions' heads side by side and both sets of feed-forward units side by side, so the backend roughly doubles; the shared layer norms and the feed-forward output bias come from champion number one. Then evolve the widened block's genes only, 200 generations, trunk frozen.

**Step 5. Widen the whole network.** Net2WiderNet split-widen of the whole Frankenstein: the stream doubles, every feed-forward width doubles with it, every head count doubles, the decoder is widened with the stream, and the attention scale is halved. Every neuron is split into two copies that share its work, so a widen with no noise reproduces the network exactly: same tours, same score. That preservation is a gate and is checked first. Then tiny random noise is added to the new copies' synapse weights only, std 0.003; the original copies stay exact. The noise size is a config input (`widen.noise_std`). The widened genome is registered as a champion candidate, with its birth score. This step is in the method because both completed lines ran it here: the original line's open evolution founded on the widened network and reached 5.2474 percent, and the control test (below) showed evolution alone on the un-widened champion reached only 6.1116 percent. The sponsor ruled on 2026-11-27: what was just done with Frankenstein is the logic for evolution. The morning freeze of the same day omitted this step; it is superseded on this point. REINFORCE is not a step: the original line ran a REINFORCE tuning phase before its open evolution, the replication skipped it and won anyway.

**Step 6. Open evolution.** The whole network, nothing frozen, founded on the widened genome from step 5. Population 1,000, 200 generations. The per-gene mutation rate is scaled to the genome size: about 25 nudges per child, never a flat 10 percent (the mutation scaling rule, 2026-11-27, below). The output is the line's Frankenstein champion.

**Step 7. Start over.** Build a second, independent line: steps 1 to 6 again, from a fresh founding, with new seeds.

**Step 8. Merge.** Stitch the two lines' Frankenstein champions together by the same Frankenstein rule as step 2: paths isolated in each, branches side by side over a doubled stream, merged decoder, then seam evolution with the backbone frozen. The two lines share no ancestor, so there is nothing to commonize; every placed entry is backbone, all connective material is stitching.

**Step 9. Stack skinny, Frankenstein, repeat.** On the standing champion, append a skinny appendix block at the back, about one quarter the size (the recorded reading: its feed-forward width is one quarter of the current backend block's feed-forward width, minimum 1). Breed in it with evolution, only the appendix moving. Then Frankenstein it: stitch the appendix run's top two champions side by side, which takes the appendix to almost twice its size. Repeat to grow a new Frankenstein stack, one block per round.

**Step 10. Frankenstein the grown stacks together.** The grown appendix stack is merged into the standing network by the same stitch rule: its proven pieces are placed side by side over the common trunk, the trunk kept once, and the seams are evolved.

**Step 11. Free evolution.** The whole merged network, nothing frozen, mutation rate scaled to genome size, same protocol as step 6.

**Step 12. Keep going until a cap is hit or there is no improvement.** A full cycle is one pass through steps 7 to 11 (a new line, a merge, the growth rounds, free evolution). If a full cycle's final champion does not beat the standing best score, stop: no improvement. If the next growth would pass a cap, stop: cap hit.

**Step 13. Keep the smallest network that had the best score.** Every champion of every stage is registered with its gene count and its score. The standing best is the lowest score. Among champions at the best score (within the tolerance in the config, 0.05 points), the one with the fewest genes is the kept champion.

Notes on the frozen method.

Width grows two ways in this lifecycle: by Frankenstein stitching (steps 2, 4, 8, 9) and by the one Net2WiderNet widen at step 5, before open evolution. An earlier draft of this section, frozen the morning of 2026-11-27, had no widen step and said width grows only by stitching; the sponsor's ruling the same afternoon (what was just done with Frankenstein is the logic for evolution) superseded it, because both completed lines ran the widen and the control test showed it contributed.

Caps are config inputs, never hard-coded: `max_layers` 6 and `max_heads` 6 for the traveler (the sponsor's numbers), `max_width` unset (null) until the sponsor gives one. One recorded tension: a line reaches 6 blocks at step 3 (4 founding blocks, the joining block, 1 stack block), so with the default depth cap the step 9 appendix rounds have no room. The cap is the sponsor's to raise in the config; the driver stops at the cap and records why.

Pruning is not a lifecycle step anymore. The 2026-11-25 step list had pruning later in the cycle; the frozen lifecycle does not include it. Path isolation still runs at every stitch, because the stitch needs it.

## The back propagation schedule

Back propagation is two halves. The measuring half passes blame backwards, from the output toward the entry. Every weight gets a number. The moving half nudges each weight a small step in the direction that shrinks the error. JAX does the measuring. The nudges are ours to control.

Phase one. Blame only the new neurons. The backbone stays frozen, held still by a mask. The backbone is the strong inherited connections that form the Frankenstein's spine. Left alone, back propagation would blame the backbone most, because it carries the most signal. A new synapse that carries no signal to the output gets no blame. It excludes itself.

Trigger. Phase one runs until the Frankenstein's score drops below the better parent's score. Then run a little more at the same setting.

Phase two. The backbone phases in. Progressively, but steeply. Its share of the nudges ramps from nothing to full over a short span.

Phase three. Keep running for a fixed number of cycles, set before the run. Then stop, whether or not it has settled.

## Back propagation, what the runs showed

The schedule above is the sponsor's design. It was tried twice on the two-Franks stitch, 2026-11-26.

First run: step size too big, softness borrowed from the gates test, no guard. It broke the network in the first 100 steps. True score 7.31 percent to 85.08 percent, stuck. The setup was wrong. The sponsor said so.

Second run: step size a hundred times smaller, softness far colder, true score read every 25 steps with a revert to the best weights when it worsened. Nothing broke. Nothing improved either. Best point of the run was the starting champion itself. The trigger never fired.

The reason measured: the training score does not track the true score on this network. At the colder softness its agreement with the true score is 0.57. The training distribution still puts about 59 percent of its weight on cities the network itself would not pick. The decoder's score differences between cities are so small that the soft version barely sharpens. Back propagation followed a score that prices a different tour than the network takes.

Standing rule, from the sponsor 2026-11-25, confirmed by these runs: refinement is evolution of the stitching genes. Back propagation is not the operating refinement. The schedule above stays on the record as the sponsor's design, tried and measured, not as the plan.

## Results so far

All scores are gaps on the fixed 200-problem validation set. Lower is better.

Frankenstein 4, first build. Parents species 8 at 15.13 percent, species 9 at 19.44 percent. Born at 23.29 percent. Seams evolved to 6.90 percent. It beats both parents by a wide margin. It was the best score on this set until the widened line passed it, 2026-11-27.

Frankenstein 4, second build, from a fresh founding run with new seeds. Parents species 1 at 16.20 percent, species 5 at 17.92 percent. Born at 17.45 percent. Seams evolved to 7.02 percent. The pattern repeats: two independent builds land in the same place.

Deepen arm on the first Frankenstein 4. One block added after the joining block, only the new block evolved. The founder won. The block earned nothing.

## The deep-narrow cycle (sponsor, 2026-11-26) — SUPERSEDED as an ordering

Superseded 2026-11-27 by the frozen lifecycle above: its steps A, B and C are the lifecycle's steps 1 to 4, and its step order survives there. The Net2WiderNet widen that was run after step C in the 2026-11-26 line also survives, as lifecycle step 5, by the sponsor's ruling of 2026-11-27. The text and the measurements below are kept as the record.

Depth always arrives narrow. Width always arrives by stitching proven pieces, never by plain widening. The cycle runs in three steps, then repeats.

Step A. Found narrow and deep. Stitch the top two champions side by side. The stitch is the widening. Parents must be evolved to maturity before they are stitched. That rule was set 2026-11-26, when the sponsor corrected a run that had stitched 50-generation founding champions directly. The parents were then evolved 100 more generations each, whole network free, and re-stitched. The mature-parent stitch landed at 8.00 percent, against 9.09 percent for the same stitch built from young parents. A full point. The 50-generation champions are not stitch-ready.

Step B. Stack more depth on the back of the Frankenstein. Narrow blocks only. No widening in this step. The new block is bred in with the trunk frozen.

Step C. Widen that new backend the Frankenstein way. Breed species in it, stitch the top two side by side over the shared trunk. The trunk is kept once.

## The deep line, measured 2026-11-26

Founding, deep and narrow: width 2, four blocks, 310 genes. Champion species 1 at 17.14 percent, species 0 at 19.70 percent. Slightly behind the one-block foundings at equal budget.

Step A, short parents: stitch born at 48.48 percent, seams evolved 200 generations to 9.09 percent. Step A, mature parents: host evolved to 10.15 percent, donor to 15.98 percent; stitch born at 31.79 percent, seams evolved 200 generations to 8.00 percent.

Step B: one narrow block stacked on the back of the 8.00 percent Frankenstein, one head, feed-forward width 4, 136 genes, six blocks deep. Born at 10.42 percent. Bred in for 100 generations to 7.97 percent. The block earned 0.03 points on its own.

Step C: the top two champions of the B run, 7.97 percent and 8.43 percent, their back blocks genuinely different. Stitched side by side into one wider back block, trunk kept once, 1,040 genes. Born at 25.60 percent. Seams evolved 200 generations to 8.10 percent. Widening the backend did not pay. It ended behind the single narrow block.

The appendix pattern for deepening is now tested, and it is the cycle above. The deep line's best is the step B champion at 7.97 percent.

Widen arm on the first Frankenstein 4. Net2WiderNet double widen across the whole network, 344 genes to 2,896. The gate passed exactly. The founder won again. The population drifted worse.

Two Frankensteins stitched. Parents 6.90 percent and 7.02 percent. Born at 25.08 percent. Seams evolved to 8.74 percent in 100 generations, then 7.31 percent in 100 more. Still behind both parents. Doubling the dose closed most of the gap and then stopped. The curve had flattened.

Frankenstein Squared, for the record: the first Frankenstein 4 stitched with species 10. Born at 22.07 percent. Evolved to 7.86 percent. Behind the first Frankenstein 4 alone.

Frankenstein 5, whole parents carried side by side with nothing dropped, is not the pattern. The sponsor said so. Frankenstein 4 is the rule.

## Caps for the traveler problem

About six layers deep. About six attention heads. Width cap: not yet set. Since 2026-11-27 these are config inputs in `bert_lifecycle_config.json` (`max_layers` 6, `max_heads` 6, `max_width` null), never hard-coded in the driver.

## Open points

- The width ceiling has no number yet.
- Deepening by the simple step, add a block and evolve the new neurons, was tried and earned nothing. The appendix pattern for deepening is now tested as the deep-narrow cycle above. Stacking the narrow block earned 0.03 points; widening the backend by stitching did not pay.
- Back propagation on a stitch needs a training score that tracks the true score. The soft surrogate does not, at any softness tried. A tau sweep, 2026-11-26, measured it: agreement stays flat near 0.57 from tau 0.1 down to 0.0001, because the decoder's logit gaps are microscopic, median 0.00000027. No usable temperature exists. The surrogate route is closed for this network.

## REINFORCE refinement (sponsor's method, 2026-11-26)

The sponsor laid the method out himself. Sample real tours from the network's action probabilities. Score them with the true external score. Derive an advantage against the network's own greedy tour on the same problem. Train on the log probabilities of the sampled actions. The reward and the sampling need not be differentiable. Confirmed as REINFORCE, also called policy gradient.

His configuration: greedy baseline. A configurable number of stochastic draws, 1 for initial testing, 4 for initial training, 8 for more intensive training. Keep the network weights unchanged while generating all the tours, then average their losses and perform one optimizer update.

His design question, tested the same day: four draws on the same city set, or one draw on each of four different city sets? Diversity won. One draw on each of four different sets reached 7.2830 percent against 7.3066 percent for the repeated set, on the two-Franks subject. It was the first gradient run that ever improved this network.

City size: training on larger problems helped more. Twenty-city training reached 7.2144 percent, against 7.2830 at ten cities and 7.2867 at five. No sampled tour in 1,600 beat its greedy baseline at twenty cities. What tracked the gains was the spread of the advantage, 1.18 at twenty cities against 0.72 at ten and 0.37 at five.

The soft-score back propagation route is closed. See the open points below.

## The widened network, gradient work 2026-11-26

The step C network was widened with Net2WiderNet across the whole network, stream 4 to 8, feed-forward and heads doubled everywhere, outgoing weights halved on the split copies. The no-noise split is exact: it reproduces 8.1018 percent, identical tours. Small random noise on the new copies breaks the symmetry between them, which back propagation needs, since exact copies receive identical gradients and never separate.

The noise scale is a parameter, not a detail. At std 0.03 the widen was born at 20.31 percent. At std 0.003 it was born at 8.78 percent, a drift of 0.68 points. Even at 0.003 the noise moves 7,040 weights across six blocks.

REINFORCE on the widened network, one sampled path per starting city at 20 cities, batch 4 problems. At step size 1e-4 the training spiked to 24.95 percent on the first read and never recovered. The sponsor's correction: divide the step by 20. At 5e-6 there is no jump. Training the new copies only reached 8.5963 percent, the only gradient gain on this network. Training the stitches only gained nothing. Unfreezing everything gained nothing. The pre-widen network at 8.10 percent still beats every widened form.

## Tuning search, 2026-11-26

Twenty-five runs, with the literature's tricks. The POMO shared baseline and the leave-one-out baseline fixed the signal exactly as promised: advantages centered on zero, half positive, where the greedy baseline gives none positive. Outcomes did not improve. Free-running training degrades this network under every baseline and every step size. Every gain came from the guard: step away from the best weights, keep the move only if the true score improves, otherwise revert. That is selection, not learning. A (1+1) evolution wearing a gradient's clothes.

Best found: 8.4415 percent, restart guard with sampling temperature 0.8. Still behind the pre-widen network. On a fresh seed the same configuration found nothing, so the gain is small and seed-dependent.

The search's read: the evidence points away from more REINFORCE knobs and toward the selection mechanism itself, run as a population of excursions in parallel, keeping the best. Which is close to the evolution that built this network.

## Full-network evolution on the widened network, 2026-11-27

One species founded on the tuning champion at 8.4415 percent. One thousand individuals. Nothing frozen, all 8,480 genes mutating at the nominal rate, 0.10 per gene. Two hundred generations. The founder won. No evolved genome beat it. The population mean got steadily worse, 28 percent to 50 percent.

The cause is measurable. At the nominal rate each child of an 8,480-gene network takes about 848 nudges at once. The rate was set on the first tiny networks, where a child took about 9 nudges. On a network this size the same rate is not a nudge.

## Mutation scaling, 2026-11-27

The sponsor's correction: the 10 percent is a config value, not a law. Adjust it to the state of the art. The rerun, identical except per-gene rate 0.003, about 25 nudges per child, finished at 5.2474 percent at generation 160. It beats the founder by 3.19 points, the pre-widen network by 2.85 points, and the standing Frankenstein 4 champion by 1.65 points. Same run, same founder, same everything. Only the dose changed.

Standing rule, sponsor 2026-11-27: the per-gene mutation rate scales with the size of the genome. A child takes a small fixed number of nudges, whatever the size. Never a flat 10 percent regardless of size.

## Self-imitation, 2026-11-27

The sponsor asked what limits REINFORCE and what can overcome it. The answer measured: the network's choices are nearly ties, so sampled tours are near random and never beat greedy. The gradient only ever pushes bad tours down. The distribution REINFORCE trains is unselected noise, because evolution trained the network's argmax and never its distribution.

The remedy proposed: teach the distribution to imitate the network's own greedy tours first, then REINFORCE has real wins to reinforce. Expert iteration, one round, cross-entropy against the champion's own greedy actions, all 8,480 genes, step size 5e-6, 400 updates.

It did not work as designed. The imitation loss never went down, 1.29 to 1.33. The distribution barely sharpened. Sampled tours beat greedy in 0.0 percent of cases at 20 cities before and after, 0.2 percent at 10 cities. The flat-distribution wall stands.

The score improved anyway, 5.2474 percent to 5.0476 percent, a new best. The pattern is guard selection again, not the imitation objective: fourteen reverts, two excursions kept. Gradient steps keep buying about 0.2 points through selection alone, whatever loss is attached to them.

## Status

Method defined 2026-11-25. Run end to end twice, and twice more as a pair. The deep-narrow cycle ran 2026-11-26: step A 8.00 percent, step B 7.97 percent, step C 8.10 percent. The widened step C network, after Net2WiderNet, REINFORCE tuning, and full-network evolution at the scaled mutation rate, reached 5.2474 percent. One round of self-imitation then moved it to 5.0476 percent. Standing champion: the self-imitation champion at 5.0476 percent, ahead of the two evolved Frankenstein 4 builds, 6.90 percent and 7.02 percent.

Replication and the widen question, 2026-11-27. A second full line was built from a fresh founding with new seeds, same process: founding 18.37 percent, matured parents 13.90 and 10.72 percent, stitch born at 74.93 percent, seams evolved to 7.5641 percent, the skinny stack earned real points this time (4.9699 percent), the backend stitch was born at 85.04 percent and recovered to 4.1580 percent, ahead of the original line's final champion before its open evolution finished. The process repeats. Separately, the control Roman asked for: open evolution alone on the original step C champion, no widen, same scaled dose. It finished at 6.1116 percent, from 8.1018 percent. A real gain, well short of 5.2474 percent. On the original line the widen did earn its place; the frozen lifecycle still drops it, because width there grows by stitching instead.

Frozen lifecycle and driver, 2026-11-27. The sponsor froze the lifecycle (the section at the top of this file) and ordered it written into the engine. `bert_lifecycle.py` is the driver: a state machine over the stages above, reusing the proven stage code, with caps and budgets as inputs in `bert_lifecycle_config.json`, a lifecycle state file recording every champion, the standing best, and the smallest network at the best score. A tiny-budget smoke test exercises one line end to end as a code check; it is not a science result. No full science run of the driver has been launched yet.

Widen amendment, 2026-11-27 (afternoon). The sponsor ruled: what was just done with Frankenstein is the logic for evolution, and the engine was updated accordingly. The widen is now step 5 of the lifecycle, between the backend Frankenstein and open evolution, in the driver and in the section at the top of this file. The sentence above (the frozen lifecycle still drops it) records the morning freeze and is superseded on this point. The driver's widen construction was verified against the replication's widen: run on the replication step C champion it reproduces the proven widened genome bit for bit, mask included, and the no-noise preservation gate passes exactly.

Merge of the two widened networks, 2026-11-27 (evening). The sponsor ordered a Frankenstein of the two lines' widened networks: the original line's scaled-evolution champion at 5.2474 percent (8,480 genes) as host, and the replication line's widened network at its birth, 4.1885 percent (8,616 genes; the stage D widen output that founded stage E, not the replication's later 3.7038 percent champion) as donor. Both parents were re-measured on the fixed set before building and reproduce exactly. Paths: 1,352 of 7,376 weight synapses for the host, 1,993 of 7,504 for the donor. The stitch is 58,102 genes at stream 16: 5,584 backbone genes placed bit-exact from the parents, 52,518 stitching genes. Born at 85.4354 percent. Seam evolution ran the full 200 generations with only the stitching genes moving, at the scaled dose of about 25 nudges per child across the stitching set (per-gene rate 0.000476, sigma 0.15, population 100, backbone asserted bit-identical throughout). No evolved genome ever beat the birth: the validation walk sat at 85.4354 percent through generation 90 and the best-of-generation readings then drifted slightly upward, 85.50 to 85.88 percent. Final: 85.4354 percent, the untouched stitch, beating neither parent. The scaled dose that worked on small seam sets changes only about 0.05 percent of a 52,518-gene stitching set per child; on this evidence it is too weak to move a merge of this size in 200 generations. Record: frankenstein_merge_525_419_2026-11-27.json and frankenstein_merge_525_419_log_2026-11-27.md; the vaulted champion is the stitch itself.

## Note on allocation

Back propagation does not need a known ideal answer. The error is built from a smooth score the network's choices move. For allocation the score can be built from the sponsor's own bars: return above the basket's even holding, return above SPY, minus penalties for variance and turnover. The allocation line stays parked. The worry that stands is trust in the score, not its existence: for the traveler the score never lies, for stocks it is part signal, part weather. The sponsor's ruling stands: allocation weights are found by evolution, not back propagation.
