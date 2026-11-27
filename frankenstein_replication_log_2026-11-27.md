# Frankenstein Replication Log — 2026-11-27

Goal: rebuild the entire deep-narrow Frankenstein line that produced the
5.2474% champion, from a fresh founding with NEW seeds, same process and
budgets at every stage. Fixed validation set throughout: TSPDomain
val_seed 20261116, 200 problems, 10 cities (mean gap vs Held-Karp, %).

## Seed family (replication; original in parentheses)

- A1 founding: run 20261300, draw 20261301, check 20261302, audit
  20261303, phase_a 20261304 (orig 20262800..20262804); val_seed 20261116
  and elim_seed 20261272 unchanged, as in the original config.
- A2 parents evolve: host 20261310, donor 20261311 (orig 20261210/11)
- A3 stitch: fill/walk 20261312 (orig 20261212)
- A4 seam evolve: 20261313 (orig 20261213)
- B deepen fill: 20261320 (orig 20261220); B evolve: 20261321 (orig 20261221)
- C stitch: deterministic from the B population; C evolve: 20261330
  (orig 20261230)
- D widen noise: 20261340 (orig 20261240)
- E full evolution: 20261370 (orig 20261270)

## Stage-by-stage numbers (replication vs original)

(to be filled after every stage)

| Stage | Replication | Original |
|---|---|---|
| A1 founding best | 18.3703% (species 6); #2 19.8899% (species 11); also sp8 20.4542%, sp10 21.7934% (4 survivors) | 17.1420% (species 1); #2 19.6989% (species 0) |
| A2 matured parents | host (sp6) 13.9011%, donor (sp11) 10.7208% — donor matured past the host; roles kept by founding rank | host 10.1484%, donor 15.9819% |
| A3 stitch birth | 74.9335% (824 genes; paths 81 host / 70 donor synapses; stitched FFN widths [7, 9, 10, 3], joining 3) — birth far worse than the original's; parity + bit-exact checks all passed | 31.7941% (788 genes: 322 backbone + 466 stitching) |
| A4 seam evolution final | 7.5641% at gen 199 (walk still descending; 824 genes = 338 backbone + 486 stitching, all 486 stitching changed, backbone bit-identical). One service restart killed the process at gen ~156; resumed from the per-generation checkpoint with no work lost | 8.0009% |
| B birth / final | 8.8571% birth (960 genes = 824 trunk + 136 new block) / 4.9699% final at gen 99 — the new block earned 2.59 points here, unlike the original line where B barely moved | 10.4234% / 7.9739% |
| C birth / final | 85.0404% birth (1,076 genes = 824 trunk + 252 widened block; champion #2 = pop index 0 at 5.2300%, b5 distance max 0.487 / RMS 0.123, 80/136 genes differ) / 4.1580% final at gen 180 — recovered from the 85% birth and beat the B champion by 0.81 points (the original line's C ended worse than its B). All 252 widened-block genes changed, trunk bit-identical. One adaptation miss caught by the stitch script's own sanity gate: it first loaded the original line's B genome layout; fixed and rerun clean | 25.6008% / 8.1018% |
| D widen birth | 4.1885% (8,616 genes = 1,476 frozen originals + 7,140 new copies; no-noise parity gate passed exactly at 4.158019%; noise drift only +0.0305 points, tours changed on 23/200) | 8.7828% (8,480 genes) |
| E evolution final | 3.7038% at gen 199 (founder 4.1885%; rate 25/8616 = 0.002902; 3,592 of 8,616 genes changed; best arrived on the LAST generation — the walk was still descending. Process killed once by a service teardown at gen ~62; resumed from checkpoint, no work lost) | 5.2474% at gen 160 |

## Verdict

The pipeline REPEATED, and this run landed better than the original:
final champion 3.7038% vs the original line's 5.2474%, on the same
fixed 200-problem validation set. Every stage's checks passed
(parity gates, bit-exact trunk checks, sanity gates).

Where it differed most from the original line:
- Births were far worse (A3 stitch 74.93% vs 31.79%; C stitch 85.04%
  vs 25.60%) and did not matter: seam evolution recovered fully both
  times. Birth gap is the noisiest, least predictive number in the
  pipeline.
- Step B paid here and did not there: the new narrow block earned
  2.59 points (7.5641% -> 4.9699%) vs the original's 0.03 points.
  This is the single biggest divergence and the main reason the
  replication finished ahead.
- Step C also paid here (4.9699% -> 4.1580%) where the original's C
  ended worse than its B (7.9739% -> 8.1018%).
- Stage D noise drift was much smaller (+0.03 vs +0.68 points).
- Stage E's founder was already at 4.1885% (the original's E founder
  was the REINFORCE-tuned 8.4415%; the tuning phase is not part of
  the replicated pipeline), so E's own gain was smaller (0.48 points)
  but from a far better start.

Final replication champion:
engine_runs/frankenstein_replication/stepE_evolution/frankenstein_rep_widened_scaled_champion.npz
at 3.7038%.
