# REINFORCE tuning log — widened Step C Frankenstein (2026-11-26)

Mission (sponsor): "Keep playing with parameters until it gives us something.
Look for tips and tricks in the literature."

## Subject and bars

- Subject: `frankenstein_stepC_widened003_genome.npz` — Net2WiderNet
  split-widen of the Step C Frankenstein with noise std 0.003 on the new
  copies. 8,480 genes = 1,440 frozen original copies + 7,040 trainable new
  copies. attn_scale 0.35355339059327373, ln_eps 1e-5.
- Born at **8.7828%** on the fixed validation set (TSPDomain, val_seed
  20261116, 200 problems, 10 cities, gap vs Held-Karp optima).
- **Win bar: below 8.1018%** (the pre-widen Step C subject) — the widen +
  gradient path pays. **Progress bar: below 8.5963%** (the best prior
  REINFORCE result on this subject: greedy baseline, new copies only,
  lr 5e-6, 400 updates, strict guard).
- Base recipe (reference run
  `frankenstein_stepC_reinforce003_lr20_2026-11-26.py`): REINFORCE, fresh
  20-city problems, ONE sampled path per starting city (20 rollouts per
  problem, first pick forced, its log-prob excluded from the loss), batch
  4 problems, weights frozen during generation, one Adam update per batch,
  gradient clip norm 1.0, guard reads on the fixed set every 25 updates.
  Trainable mask = the new copies' genes; frozen genes verified
  bit-identical after every run.

## The wall (measured, reference run)

At 20 cities the sampled tours are far worse than the greedy tour: mean
sampled length 9.4476 vs mean greedy 4.4979. Against the greedy baseline
every advantage is negative: mean −4.9497, std 1.0740, max −1.3347;
**0.00% of 32,000 sampled tours beat their greedy baseline**. The gradient
is dominated by the constant −4.95 offset — pure push-down on everything
sampled, with the useful per-rollout variation (±1.07) riding on top.
Prime suspect: the baseline. A baseline built from the rollouts
themselves is zero-centered per problem by construction, so the best
rollouts on every problem get positive advantage and the constant offset
vanishes.

## Literature scan — tricks to try, with sources

1. **POMO shared baseline** — Kwon et al., "POMO: Policy Optimization with
   Multiple Optima for Reinforcement Learning", NeurIPS 2020
   (arXiv:2010.16011). One rollout per starting node; the baseline is the
   MEAN length of all rollouts on the same instance, shared by every
   rollout. Low-variance, instance-specific, and by construction roughly
   half the rollouts beat it. POMO reports this makes training fast,
   stable, and resistant to local minima vs the greedy rollout baseline.
   → Stage A1.
2. **Leave-one-out baseline** — Kool, van Hoof & Welling, "Buy 4
   REINFORCE Samples, Get a Baseline for Free!", ICLR 2019 Deep RL meets
   Structured Prediction workshop. With k samples per input, sample i's
   baseline is the mean of the other k−1: an unbiased, critic-free
   baseline that costs nothing. (Later popularized for LLMs as RLOO,
   Ahmadian et al. 2024.) Note: LOO advantages are exactly the POMO
   advantages scaled by k/(k−1) = 20/19, so the two should behave nearly
   alike under Adam; both are run as briefed. → Stage A2.
3. **Greedy rollout baseline + training scale** — Kool, van Hoof &
   Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019
   (arXiv:1803.08475). From-scratch training of the Attention Model:
   constant lr 1e-4, batches of 512 instances, gradient clipping; the
   greedy baseline comes from a frozen copy of the policy that is only
   replaced when the current policy beats it significantly (paired
   t-test). Two lessons for us: (a) their batch is 512 instances vs our
   80 rollouts — gradient noise is a suspect (→ Stage E); (b) for
   fine-tuning an already-good network the step size must be far below
   the from-scratch 1e-4 — our established 5e-6 — and progress may need
   many more updates at a stable lr than 400 (→ Stage B).
4. **Advantage normalization (group standardization)** — standard
   REINFORCE variance practice; in group-relative form (GRPO) each
   group's advantages are divided by the group's std so every instance
   pushes comparably hard regardless of its length scale. Directly
   attacks our offset/scale problem in combination with (1)/(2).
   → variant flag `--adv-norm`, tried as Stage A3 on the better of
   A1/A2.
5. **Entropy bonus** — add β·H(π) to the objective to keep the sampling
   distribution from collapsing prematurely (standard policy-gradient
   practice). Wang et al., "Arbitrary Entropy Policy Optimization",
   arXiv:2510.08141 (2025), show in RL fine-tuning that performance is
   NON-monotone in entropy: some exploration helps, too much hurts —
   hence small coefficients. → Stage C, β = 1e-3 and 1e-2.
6. **Sampling temperature** — sample from a temperature-adjusted
   distribution. A logit multiplier below 1 flattens the distribution
   (more diverse tours, more exploration); above 1 sharpens it toward
   greedy (rollouts closer to the greedy tour, finer-grained advantage
   differences near the policy's best behavior). → Stage C, multipliers
   0.8 and 1.2, log-probs taken under the same tempered distribution
   that generated the samples.
7. **Problem-size curriculum / mixed sizes** — Kool et al. train a model
   per problem size; mixed-size batches are the standard way to get
   size generalization. Our own earlier measurement on the two-Franks
   subject: at 5 cities 11.50% of sampled tours beat greedy, at 10
   cities 0.06%, at 20 cities 0.00% — smaller problems give a cleaner,
   sometimes positive signal. → Stage D: step 10 → 20 halfway, and a
   per-update size mix over {10, 15, 20}.
8. **Longer training with a softer guard** — the reference guard reverts
   and halves the lr on EVERY worse read; in the reference run it halved
   7 times and floored the lr at 5e-8, after which the reads repeat the
   same value (frozen progress). POMO/Kool-style training runs long at
   a constant lr. → Stage B: 2,000 updates; guard still vaults the
   best-on-validation weights, but halves lr only after THREE
   consecutive worse reads and never reverts mid-run.

## Harness

`reinforce_tune.py` — the reference implementation, parameterized
(baseline, advantage normalization, sampling temperature, entropy
coefficient, lr, updates, guard mode, curriculum, batch size, seed).
At (greedy, temp 1.0, entropy 0, strict guard, lr 5e-6, 400 updates,
seed 20261241) it must reproduce the reference run exactly; that control
run (A0) is the harness check. Every run writes
`engine_runs/reinforce_tuning/<name>.json` (full config, every read,
guard events, signal stats) and `<name>_champion.npz`. The best genome
overall lives at `engine_runs/reinforce_tuning/best_champion.npz`.

Bars: founder birth 8.7828% · progress 8.5963% · **win 8.1018%**.

## Runs

### Harness control — A0 (greedy, lr 5e-6, 400 upd, strict guard)
Reproduces the reference run EXACTLY (every read identical; best
8.5963% @ 175; 13 reverts, 7 halvings). Harness verified.

### Stage A — baseline family (400 updates, strict guard, seed 20261241)

| run | config | best | frac pos adv | mean adv | adv std |
|---|---|---|---|---|---|
| A0 | greedy, lr 5e-6 | 8.5963% @175 | 0.00 | −4.950 | 1.074 |
| A1a | POMO, lr 5e-6 | 8.7828% @0 (founder) | 0.50 | 0.000 | 0.905 |
| A1b | POMO, lr 1e-5 | 8.7828% @0 (founder) | 0.50 | 0.000 | 0.906 |
| A2a | LOO, lr 5e-6 | 8.7828% @0 (founder) | 0.50 | 0.000 | 0.953 |
| A2b | LOO, lr 1e-5 | 8.7828% @0 (founder) | 0.50 | 0.000 | 0.954 |

Signal side: POMO/LOO work exactly as the literature says — advantages
zero-centered, exactly half the rollouts positive. Result side: NO
POMO/LOO run ever beat the founder under the strict guard. Walks: first
read jumps (11.26% at lr 5e-6; 15.46% at 1e-5), then descends toward
birth (A1b reached 8.7981% @125) and stalls ~8.89–8.91% once the guard
has halved the lr to its floor (by ~update 250). LOO's reads are
identical to POMO's at 4 decimals (the k/(k−1) factor is invisible
under Adam + the guard, as predicted).

Diagnosis: the strict guard is the strangler for the zero-centered
baselines — every 25-update excursion is reverted and the lr halves
until nothing can move. The greedy baseline's constant −4.95 offset,
by contrast, acts as a strong uniform push-down on sampled tours that
happened to help greedy validation reads in short excursions; it
saturates at 8.5963%. Stage B (long runs, soft guard) is the right
test for POMO. A3 adds advantage normalization to POMO at lr 1e-5.

### Stage A3 — POMO + advantage normalization (lr 1e-5, strict)
Best 8.7828% @0 (founder). Reads nearly identical to A1b (one read
differs by 0.009). Advantage std is exactly 1.0 by construction and
it changes nothing under Adam + the strict guard. Normalization is
not the missing piece.

### Stage B — POMO, soft3 guard (no reverts), 2,000 updates

| run | lr | best | final reads | mean entropy |
|---|---|---|---|---|
| B1 | 5e-6 | 8.7828% @0 | climbs to 26.68% | 1.634 |
| B2 | 1e-5 | 8.7828% @0 | climbs to 32.11% | 1.325 |
| B3 | 2e-5 | 8.7828% @0 | climbs to 34.36% | 1.167 |

CATASTROPHIC AND INFORMATIVE. With no reverts, POMO training degrades
the validation gap MONOTONICALLY (8.78% → 27–34%) while the sampling
entropy falls (2.03 → 1.17–1.63): the policy sharpens onto worse
greedy behavior. The POMO objective never references the policy's own
greedy decode — and our validation IS greedy decode (single greedy
tour from the learned start state, 10 cities). POMO's forced-start
rollouts also never exercise the policy's own first pick, which
greedy validation depends on. The greedy baseline's hidden virtue is
that it ANCHORS training to greedy behavior: advantage is measured
against the greedy tour itself. Remove the anchor (POMO/LOO) and the
policy drifts to optimize average sampled length at 20 cities while
greedy quality at 10 cities is collateral damage.

New diagnosis: two candidate mechanisms for the only gain on record
(greedy + strict guard, 8.5963%): (i) the greedy anchor in the
objective, and/or (ii) the strict guard itself — a (1+1)-style
selection over 25-update excursions, keeping only improving flips.
Stage B' separates them: greedy baseline under the soft guard, and a
"restart" guard (strict reverts, but lr restored to full on every
new best — the reference run froze because by the time it found
8.5963% the lr had already halved to ~1.6e-7).

### Stage B' — greedy baseline × guard family (2,000 updates)

| run | guard | lr | best | verdict |
|---|---|---|---|---|
| B4 | soft3 | 5e-6 | 8.7828% @0 | diverges to 18.72% |
| B5 | soft3 | 1e-5 | 8.7828% @0 | diverges to 27.74% |
| B6 | restart | 5e-6 | **8.5797% @475** | beats prior best 8.5963% |
| B7 | restart | 1e-5 | 8.6077% @200 | ≈ prior best |

DECISIVE. The greedy anchor is NOT the mechanism: greedy + soft
guard diverges just like POMO + soft guard. Free-running REINFORCE
on this network — any baseline tried — degrades greedy validation
quality, because the training objective (expected sampled length,
20 cities, forced starts) does not track greedy 10-city decode
quality. **The strict guard is the mechanism**: the reference gain
was (1+1)-style selection over 25-update excursions from the best
weights, keeping only improving flips. The restart guard (lr
restored on each new best) confirms it and improves slightly:
8.5797% (progression 8.7828 → 8.7323 @50 → 8.6305 @125 → 8.5797
@475; nothing after, in 1,525 further updates). Still 0.478 points
from the win bar. What limits it now is excursion quality: most
25-update excursions hurt, and kept improvements are 0.05–0.10 each.
Stages C/D/E therefore run on the B6 base (greedy, restart guard,
lr 5e-6, 2,000 updates) and aim at making each excursion a better
proposal: sampling temperature / entropy (C), problem-size match
to validation (D), batch size (E).

### Stage C — sampling knobs on the B6 base

| run | knob | best | verdict |
|---|---|---|---|
| C1 | temp 0.8 (flatten) | **8.4415% @275** | new overall best |
| C2 | temp 1.2 (sharpen) | 8.5456% @175 | beats B6, less than C1 |
| C3 | entropy 1e-3 | 8.5797% @475 | identical to B6 — no effect |
| C4 | entropy 1e-2 | 8.5797% @625 | same best, later — no effect |

Temperature is a real lever on excursion quality: flattening the
sampling distribution (more diverse rollouts per problem) gives the
best excursions (8.4415%, +0.138 over B6); sharpening also helps,
less. The entropy bonus at these coefficients does nothing
measurable (the bonus gradient is evidently negligible next to the
REINFORCE term; the policy's entropy stays ~2.03 regardless).
Stage D/E run on the C1 config (temp 0.8).

### Stage D — curriculum (C1 config)

| run | curriculum | best | verdict |
|---|---|---|---|
| D1 | step 10→20 @1000 | 8.6225% @425 | all bests in the n=10 phase |
| D2 | mix {10,15,20} | 8.7828% @0 | no improving excursion at all |
| D3 | fixed 10 | 8.6225% @425 | identical to D1's first half |

Size-matching to validation does NOT beat C1 (8.4415% at fixed 20).
Mixed sizes actively hurt. (D1 and D3 share their first 1,000
updates by construction — same seed stream at n=10 — hence the
identical progression; neither improved after update 425.)

### Stage E — batch 8 problems × 20 rollouts (C1 config)
Best 8.7828% @0 (founder): NO improving excursion in 2,000 updates.
Doubling the batch starves the mechanism. Consistent picture across
D2 and E1: progress here is selection over NOISY excursion
proposals; reducing proposal variance (bigger batch, mixed sizes)
removes the lucky fluctuations the guard selects for. The noise is
a feature, not a bug.

### Convergence checks on C1 (temp 0.8) + extra round

| run | change vs C1 | best | verdict |
|---|---|---|---|
| F1 | new seed (20271241) | 8.7828% @0 | NO improvement — does not replicate |
| F2 | temp 0.7 | 8.6277% @175 | improves, misses progress bar |
| F3 | temp 0.9 | 8.4775% @300 | beats progress bar |
| X1 | read every 10 (extra round) | 8.7828% @0 | 10-update excursions never help |

Mixed verdict on C1. The temperature knob has a real effect at the
original seed (0.8 → 8.4415, 0.9 → 8.4775, 1.2 → 8.5456, all beat
the prior best; 0.7 → 8.6277 does not), but the identical config on
a fresh seed finds nothing at all. Single-seed results in this
whole search — the reference 8.5963 included — carry large seed
variance: the guard selects lucky excursions, and luck is
seed-lottery. Shorter excursions (X1) are strictly worse: 10-update
proposals are too small to flip a validation tour beneficially.

## Final table — every run

Birth 8.7828% · progress bar 8.5963% · **win bar 8.1018% (NOT beaten)**

| run | config (delta from base) | best | @ upd |
|---|---|---|---|
| A0 | greedy, strict, lr 5e-6, 400 upd (= reference) | 8.5963% | 175 |
| A1a | POMO, strict, lr 5e-6 | 8.7828% | 0 |
| A1b | POMO, strict, lr 1e-5 | 8.7828% | 0 |
| A2a | LOO, strict, lr 5e-6 | 8.7828% | 0 |
| A2b | LOO, strict, lr 1e-5 | 8.7828% | 0 |
| A3 | POMO + adv-norm, strict, lr 1e-5 | 8.7828% | 0 |
| B1 | POMO, soft3, lr 5e-6, 2000 upd | 8.7828% (→26.7%) | 0 |
| B2 | POMO, soft3, lr 1e-5, 2000 upd | 8.7828% (→32.1%) | 0 |
| B3 | POMO, soft3, lr 2e-5, 2000 upd | 8.7828% (→34.4%) | 0 |
| B4 | greedy, soft3, lr 5e-6, 2000 upd | 8.7828% (→18.7%) | 0 |
| B5 | greedy, soft3, lr 1e-5, 2000 upd | 8.7828% (→27.7%) | 0 |
| B6 | greedy, restart, lr 5e-6, 2000 upd | 8.5797% | 475 |
| B7 | greedy, restart, lr 1e-5, 2000 upd | 8.6077% | 200 |
| C1 | B6 + temp 0.8 | **8.4415%** | 275 |
| C2 | B6 + temp 1.2 | 8.5456% | 175 |
| C3 | B6 + entropy 1e-3 | 8.5797% | 475 |
| C4 | B6 + entropy 1e-2 | 8.5797% | 625 |
| D1 | C1 + step curriculum | 8.6225% | 425 |
| D2 | C1 + mix curriculum | 8.7828% | 0 |
| D3 | C1 + fixed 10 cities | 8.6225% | 425 |
| E1 | C1 + batch 8 problems | 8.7828% | 0 |
| F1 | C1, new seed | 8.7828% | 0 |
| F2 | temp 0.7 | 8.6277% | 175 |
| F3 | temp 0.9 | 8.4775% | 300 |
| X1 | C1 + read every 10 | 8.7828% | 0 |

Best genome: `engine_runs/reinforce_tuning/best_champion.npz`
(= C1_temp08_champion.npz; frozen genes verified bit-identical to
the widened founder; 7,027 of 7,040 trainable genes changed).

## Honest read — what made the difference

1. **The guard, not the gradient, is the engine of every gain.**
   Free-running REINFORCE degrades this network monotonically
   under every baseline tried (greedy, POMO, LOO), at every lr,
   because the training objective — expected sampled tour length
   at 20 cities with forced starts — does not track greedy
   10-city decode quality. All recorded gains are (1+1)-style
   selection: 25-update excursions from the best weights, keep
   only improving flips. The literature's baselines fixed the
   SIGNAL exactly as advertised (POMO/LOO: advantages
   zero-centered, 50% positive) and it did not matter, because
   signal quality was never the binding constraint.
2. **The restart guard is a genuine, if small, improvement** over
   the reference guard: restoring the full lr on each new best
   unfreezes the search (8.5797% vs 8.5963%, and it is the base
   every later gain stands on).
3. **Sampling temperature is the only gradient-side knob that
   helped**: flattening to 0.8 (more diverse rollouts) produced
   the best excursions, 8.4415%; 0.9 and 1.2 also beat the prior
   best at the same seed. Entropy bonuses did nothing; advantage
   normalization did nothing; bigger batches and mixed sizes
   actively hurt (they reduce the excursion variance the guard
   feeds on); size-matching training to validation did not beat
   training at 20 cities.
4. **Seed variance is large and is the main caveat on the best
   number.** C1's config on a fresh seed scored the founder. The
   defensible claim is not "temp 0.8 gives 8.44%" but "temp
   0.8–0.9 with the restart guard gives the best excursion
   proposals we measured; expected gains are a few tenths of a
   point and seed-dependent." The win bar (8.1018%, the pre-widen
   subject) was not beaten by any of the 25 runs: the widen +
   gradient path does not pay on this subject yet. The gap from
   the best found (8.4415%) to the bar is 0.34 points.
5. If this line continues, the evidence points away from more
   REINFORCE knobs and toward the selection mechanism itself:
   e.g. population-based excursion selection (several parallel
   excursions per read, keep the best — closer to the engine's
   own evolution, which is what built this network in the first
   place), or training the greedy decode directly (a baseline
   anchored at the CURRENT best weights' greedy tours with
   excursions proposed by evolution rather than by Adam).






