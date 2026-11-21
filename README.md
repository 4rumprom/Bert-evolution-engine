# BERT Evolution Engine

A parametric engine that grows a tiny Transformer by evolution instead of
training one big network from scratch.

The idea: start with an extremely small attention network. Evolve it until
performance plateaus. Preserve what it learned. Add the smallest possible
piece of new capacity (a few neurons of width, then a head, then a layer,
then a whole stacked block), evolve the new parts while the old ones are
mostly preserved, and keep a growth step only when it measurably improves
the result on a fixed validation set. Repeat.

Two domains plug into the same engine core:

- **TSP** (travelling salesman): cities are tokens, no positional encoding,
  so the encoder is permutation invariant by construction.
- **Stocks**: the genetic stock-allocation adapter (smoke scale today, see
  the honesty note below).

## How it works

- **One stacked matrix.** The whole population is evaluated as a single
  batched tensor operation, all species and all members at once, then the
  results are folded back per individual and per species for selection.
  Chunking (the `chunk` config value) bounds memory without changing
  results, bit for bit.
- **Species.** A species is one frozen random draw of the newest
  component's input side on top of the preserved champion base. Evolution
  tunes the decoder and the new component's output side inside the
  species; the frozen draw itself never changes. Species are culled on a
  schedule (worst 20% every 3 generations, floor of 10), judged on one
  fixed elimination set shared by all.
- **Two-phase growth.** Each growth step first evolves only the new part
  against the frozen champion's cached outputs (the grunt-work phase,
  ~50-100 generations), then recombines the full network and refines it
  on fresh data with old genes mutating at 1/10 rate.
- **Frozen-prefix cache.** When part of the network is frozen, its outputs
  on the training problems are computed once and reused; training inputs
  are seed-reproducible, so nothing has to be stored.
- **Cataclysm.** A periodic mass extinction: the best champion of each of
  the top K distinct species (K = `founder_count`, default 1) is vaulted,
  everything else is wiped, and a fresh species set is bred from the
  founders.
- **Growth rules.** On budget exhaustion or plateau, the engine tries the
  growth moves in order, least change first: `widen`, `add_head`,
  `add_block`, `stack`. Keep on improvement, revert and try the next rule
  otherwise. Every growth move is function-preserving at transplant: the
  widened network reproduces the champion's behaviour exactly before
  evolution starts (verified to 0.0 difference in the test suite).

## Files

| File | What it is |
|---|---|
| `bert_engine.py` | The domain-independent engine core: genome layout, batched evaluation, breeding, culling, growth, cataclysm, checkpointing |
| `bert_backend.py` | Backend selection: numpy reference, or JAX (CPU/GPU). Falls back to numpy with a printed note if JAX is missing |
| `bert_domain_tsp.py` | TSP adapter: problem generation, tour decoding, fitness |
| `tsp_genome2.py` | Exact TSP optima (Held-Karp) used by the TSP adapter to score validation gaps |
| `bert_domain_stock.py` | Stock adapter (see honesty note) |
| `bert_preprocessor.py` | Parametric feature preprocessor for the stock side: turns raw OHLCV (+ optional fundamentals) into the standardised feature tensor, driven by `feature_spec` |
| `bert_stock_data.py` | The mapping: loads `stock_data_bundle.npz` into the panel shape the stock domain and preprocessor consume, and remaps the frozen episodes onto the bundle by ticker name |
| `build_stock_bundle.py` | Rebuilds the data bundle from the trading program's price panel, volume sidecar and EDGAR fact store (needs pandas; not needed to run the engine) |
| `stock_data_bundle.npz` | The real stock data bundle (42.5 MB, binary; see below) |
| `run_bert_engine.py` | The runner CLI |
| `test_bert_engine.py` | The verification suite: 27 checks, all exactness-critical |
| `bert_engine_config.json` | Default TSP run configuration: every knob in one place |
| `bert_engine_config_stock.json` | Stock run configuration |
| `bert_engine_config_test.json` | Minimal TSP test configuration (12 species, 40 generations) |
| `TEST_PROTOCOL.md` | The pre-stated test protocol and pass criteria for the minimal run |
| `BERT_ENGINE_DESIGN_2026-11-20.md` | Full design record, with the checklist and its evidence |
| `BERT_ENGINE_CHECKLIST_2026-11-20.md` | The build checklist, 35 of 35 checked |
| `PARAMETRIC_ENGINE_SPEC.md` | The frozen specification the engine was built against |

## Requirements

- Python 3.10+
- numpy and scipy (required; scipy is used by the stock preprocessor's
  recursive indicators)
- JAX (optional, for speed and GPU). Without it the engine runs on the
  numpy reference backend automatically.

```bash
pip install numpy scipy
pip install jax            # CPU JAX
pip install "jax[cuda12]"  # NVIDIA GPU JAX; match the CUDA version to your driver
```

## Quick start

Verify the install first (tiny configs, a minute or two):

```bash
python3 test_bert_engine.py
```

Expected: every leg PASS (the JAX leg may SKIP if JAX is not installed).

Smoke run:

```bash
python run_bert_engine.py --config bert_engine_config.json --smoke
```

Full run:

```bash
python run_bert_engine.py --config bert_engine_config.json --fresh
```

Useful flags: `--stop-after N` (stop after N generations, checkpointed),
`--out-dir DIR`, `--domain tsp|stock`. A run checkpoints as it goes and
refuses to resume under a changed configuration.

## Running on a GPU

The shipped config uses `"backend": "jax"` with `"device": "cpu"`. On a
machine with an NVIDIA GPU and GPU JAX installed, change one line in the
config:

```json
"device": "gpu"
```

Same code, same checkpoints; checkpoints are backend-independent.
Precision: `"x64"` matches the float64 reference exactly. `"x32"` is the
fast GPU mode; greedy tour decoders can flip on near-ties there, which is
accepted and documented in the design record.

When the network grows so large that GPU memory can no longer hold enough
species and population for evolution to make sense, that is the designed
handoff point: vault the champion and switch to gradient training (policy
gradient for the TSP decoder, since tour length does not differentiate
through greedy picks). That gradient stage is not in this repo yet.

## The knobs

Everything is in the config JSON: network width `d`, `ffn_hidden`,
`initial_blocks`, `initial_heads`, `input_dim`; population `n_species`,
`per_species`, `species_floor`, culling `cull_fraction` / `cull_interval`;
budgets `phase_a_gens`, `phase_b_gens`, `max_growth_rounds`; growth
`growth_order`, `widen_step`; cataclysm `cataclysm_enabled`,
`founder_count`; evolution `mut_p`, `mut_sigma`, `old_rate_factor`,
`tournament`, `elite`; memory `chunk`; and the seeds.

## The stock preprocessor

The stock side eats features, not raw prices. `bert_preprocessor.py`
turns a raw OHLCV panel (plus optional point-in-time fundamentals) into
the standardised feature tensor the engine consumes, and what it computes
is fully parametric, set by the `feature_spec` list in the stock config's
`domain_config`. One entry per feature block:

```json
{"kind": "fourier", "k": 4, "window": 60}
{"kind": "wavelet", "window": 60, "levels": 3}
{"kind": "rsi", "window": 14}
{"kind": "atr", "window": 14}
{"kind": "chaikin", "window": 20}
{"kind": "stochastic", "window": 14, "smooth": 3}
{"kind": "ema", "window": 20}
{"kind": "sma", "window": 50}
{"kind": "obv", "window": 20}
{"kind": "candlestick", "window": 10}
{"kind": "volatility", "window": 60}
{"kind": "fundamental", "name": "earnings_yield"}
```

Available kinds: `fourier` and `wavelet` (the genetic program's original
transforms, reproduced bit-exactly; Fourier over the trailing window's
close ratios, Haar wavelet approximation of the same series),
`volatility`, `rsi` (Wilder), `atr` (Wilder), `chaikin` (Chaikin Money
Flow), `stochastic` (smoothed %K), `ema`, `sma` (both as close/average
minus 1), `obv` (windowed signed-volume share), `candlestick` (six
body/shadow/range statistics over the window), and `fundamental`
(passthrough of a named point-in-time series; as-of alignment is the
data layer's obligation). Add, drop, or re-window entries and the
network's input size follows automatically.

Every feature is stationary by construction (ratios and bounded
oscillators, never raw levels), causal (day t uses only bars up to t;
the test suite proves rewriting the future leaves past features
bit-identical), and standardised per feature from a training slice.
That slice is set by `domain_config.stats_end` (a day index); the
default of half the panel is a placeholder, and a real run must set it
to the training cutoff so no validation-era statistics leak in.

Sampling knobs live in the same `domain_config`: `n_train_episodes`
(samples scored per generation), `n_elim_episodes`, `episode_days`
(sample length), and `basket_n`.

The shipped stock config carries the full technical menu (30 features,
input size 31 with the position input). Fundamentals are implemented
and tested on synthetic series but are not in the shipped default,
because no real fundamentals source is wired yet.

## The stock data bundle

`stock_data_bundle.npz` (42.5 MB) carries the real data the other
genetic programs ran on, packaged for this engine: 666 tickers on one
frozen calendar (the SPY calendar, 2,989 trading days, 2015-01-02 to
2026-11-19), with open/high/low/close from the point-in-time price
panel and split-adjusted volume from the volume sidecar, aligned day
for day (missing days stay NaN; the one ticker with no volume, ^VIX,
is NaN on volume throughout). It also carries four point-in-time
fundamental series derived with the fundamental strategies' own
machinery (filed-date as-of, first vintage, split-corrected market
caps): earnings yield, book-to-market, return on assets, and
debt-to-assets. Six tickers are stale (their source data ends in
August to October 2026) and stay NaN after their last day.

The stock config points at the bundle (`panel_bundle` in
`domain_config`); the runner loads it through `bert_stock_data.py`,
defaults the standardisation cutoff to the end of the 2015-2019
training window stored in the bundle, and resolves the frozen
64-episode validation baskets by ticker name (the stored row indices
address an older 622-ticker panel ordering). A smoke run on the real
bundle completes end to end.

Two honest limits. First, the bundle is a binary file: it lives with
the export locally, but the automated GitHub snapshot push used for
this project carries text only, so the bundle itself does not ride
that push; moving it through GitHub would take a release asset or a
direct git push. Second, episode sampling on real data is
validity-aware, under the same rule the genetic program's environment
uses: an episode is drawable only when every ticker in its basket has
a finite price on every day of the window plus its final mark day.
Measured on this bundle, only 54.4% of uniform (basket, start) pairs
satisfy that, so nearly half of naive draws were unusable; the sampler
now redraws invalid draws (with a bounded deterministic fallback), and
on a panel with no missing data its draws are identical to the old
uniform sampler for the same seed.

## Honesty note on the stock adapter

The stock adapter runs on the real data through the bundle described
above, verified end to end at smoke scale. It does not read a raw panel
directory directly; the bundle is the supported route, and a config
with neither a bundle nor the synthetic panel raises a clear error
rather than faking results. The TSP side is complete and verified end to end.

## Provenance

This repo is a snapshot export of the tested engine from the author's
research workspace (where it lives alongside the earlier per-stage runners
it replaces). The code here is the version that passed the 50-check
verification suite on 2026-11-20.
