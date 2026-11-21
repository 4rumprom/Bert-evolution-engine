"""Parametric BERT growth engine: domain-independent core.

Replaces the per-stage bespoke attention runners
(run_tsp_attn_stage0..3.py) with ONE parametric engine.
Design record: ../BERT_ENGINE_DESIGN_2026-11-20.md.
Governing checklist: ../BERT_ENGINE_CHECKLIST_2026-11-20.md.

Core responsibilities (domain-independent):
  - one config (EngineConfig, JSON + dataclass) for every
    architectural and run variable;
  - a parametric genome: a stack of encoder-decoder couples;
    each couple = token projection + uniform transformer
    blocks (multi-head self-attention branch and FFN branch,
    each `out = y + Branch(Norm(y))`) + a domain decode head.
    No positional encodings, no scorer, no recurrence;
    permutation-equivariant operations only.
  - whole-population batched evaluation: genomes stacked as
    one (population, n_genes) matrix, the member axis running
    through every operation, results folded back per member
    and per species; the population axis is chunked by the
    config `chunk` size without changing results.
  - frozen-prefix cache: encoding is split into segments at
    component boundaries; a segment whose genes are identical
    across a group (species / lineage / whole population) and
    whose inputs are seed-reproducible is computed ONCE per
    group and cached (exact: cached values are the very
    floats a full recompute produces, bit-identical by the
    verified row-independence of the batched einsums).
  - species mechanics: species = frozen random draws of the
    newest component's input-side genes on a preserved base;
    breeding within species only; Schedule A culling.
  - growth transplants (widen / add head / add block / stack
    couple), each function-preserving, gated by the domain's
    exact preservation check in the runner.
  - founder / cataclysm mechanics: founders are the best
    champions of the top-K DISTINCT species on the fixed
    validation set; regeneration splits a fresh species set
    among founders (one code path backs both cataclysm
    regeneration and growth-stage regeneration).
  - checkpointing with full RNG state; resume refuses a
    mismatched config.

Domain adapters (bert_domain_tsp.py, bert_domain_stock.py)
supply: input features, decode-head gene spec and forward
semantics, fitness, fixed sets, and the preservation check.
All fitness/score values inside the engine are HIGHER IS
BETTER (domains negate where their natural metric is a
loss, e.g. TSP tour length).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

LN_EPS0 = 1e-5
CLIP = 10.0

# gene classes (int8 vector aligned with the flat genome)
CLS_DRAW = 0        # species-defining frozen draw
CLS_OLD = 1         # older components (phase A rate 0,
                    # phase B rate mut_p * old_rate_factor,
                    # no crossover)
CLS_IFACE = 2       # newest component output side (full rate)
CLS_DECODER = 3     # active decode head (full rate, always)
CLASS_NAMES = {0: "draw", 1: "old", 2: "interface", 3: "decoder"}


# ================================================================
# Config
# ================================================================

@dataclass
class EngineConfig:
    # domain / backend
    domain: str = "tsp"                 # "tsp" | "stock"
    backend: str = "jax"                # "jax" | "numpy"
    device: str = "cpu"                 # "cpu" | "gpu"
    precision: str = "x64"              # reference mode
    # architecture (defaults reproduce the stage-3 shape:
    # input 2, d 8, FFN 16, 2 blocks, 2 heads -> 1880 genes
    # with the TSP pointer head)
    input_dim: int = 0                  # 0 = domain default
    d: int = 8
    ffn_hidden: int = 16
    initial_blocks: int = 2
    initial_heads: int = 2
    # population
    n_species: int = 100
    per_species: int = 20
    chunk: int = 128
    # training sets
    n_train: int = 64
    n_elim: int = 64
    n_val: int = 200
    # stage protocol
    phase_a_gens: int = 75
    phase_b_gens: int = 75
    # operators
    mut_p: float = 0.10
    mut_sigma: float = 0.15
    old_rate_factor: float = 0.1
    draw_sigma: float = 0.5
    tournament: int = 3
    elite: int = 2
    # culling (Schedule A)
    cull_interval: int = 3
    cull_fraction: float = 0.20
    species_floor: int = 10
    # cataclysm / founders
    cataclysm_enabled: bool = True
    cataclysm_every_n_generations: int | None = None
    founder_count: int = 1
    # growth
    growth_order: tuple = ("widen", "add_head", "add_block", "stack")
    widen_step: int = 4
    plateau_window: int = 30
    plateau_threshold: float = 0.005
    max_growth_rounds: int = 8
    # seeds
    run_seed: int = 20261300
    draw_seed: int = 20261301
    elim_seed: int = 20261272
    val_seed: int = 20261116
    phase_a_seed: int = 20261310
    audit_seed: int = 20261303
    check_seed: int = 20261302
    # run identity / domain extras
    run_name: str = "bert_engine_run"
    domain_config: dict = field(default_factory=dict)

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "EngineConfig":
        known = set(EngineConfig.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown config keys: {sorted(bad)}")
        cfg = EngineConfig(**d)
        cfg.growth_order = tuple(cfg.growth_order)
        return cfg

    @staticmethod
    def load(path) -> "EngineConfig":
        with open(path) as f:
            return EngineConfig.from_dict(json.load(f))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)


def gen_instance_seed(cfg: EngineConfig, stage: int, gen: int) -> int:
    """Phase-B generation instance sets are recreatable from
    seed + generation number alone (checklist item): the set
    for (stage, gen) is drawn from this derived seed, never
    from the breeding stream."""
    return int((cfg.run_seed + 1_000_003 * (stage + 1)
                + 10_007 * gen) % (2 ** 31 - 1))


# ================================================================
# Architecture spec + layout
# ================================================================

@dataclass
class ArchSpec:
    input_dim: int
    d: int
    ffn_hidden: int
    couples: tuple          # ((n_blocks, heads), ...) per couple
    ln_eps: float = LN_EPS0
    # the attention / pointer logit scale is a constant of
    # the architecture family, fixed at the INITIAL width
    # (the stage program's SCALE = 1/sqrt(8) never changed
    # as heads and blocks were added); widening must NOT
    # rescale it, or every attention score changes
    attn_scale: float = 0.0

    def signature(self) -> str:
        return json.dumps({
            "input_dim": self.input_dim, "d": self.d,
            "ffn_hidden": self.ffn_hidden,
            "couples": [list(c) for c in self.couples],
            "ln_eps": self.ln_eps,
            "attn_scale": self.attn_scale}, sort_keys=True)

    @staticmethod
    def from_signature(s: str) -> "ArchSpec":
        d = json.loads(s)
        return ArchSpec(d["input_dim"], d["d"], d["ffn_hidden"],
                        tuple(tuple(c) for c in d["couples"]),
                        d["ln_eps"],
                        d.get("attn_scale",
                              1.0 / np.sqrt(d["d"])))


def initial_arch(cfg: EngineConfig, input_dim: int) -> ArchSpec:
    return ArchSpec(input_dim=input_dim, d=cfg.d,
                    ffn_hidden=cfg.ffn_hidden,
                    couples=((cfg.initial_blocks,
                              cfg.initial_heads),),
                    attn_scale=1.0 / np.sqrt(cfg.d))


class Layout:
    """Flat-genome layout generated from an ArchSpec plus a
    domain head spec. Entries are named arrays; growth
    transplants remap by name, so old gene positions never
    need to survive an architecture change."""

    def __init__(self, arch: ArchSpec, head_spec):
        self.arch = arch
        self.entries = []            # (name, slice, shape)
        self._slices = {}
        off = 0

        def add(name, shape):
            nonlocal off
            n = int(np.prod(shape))
            self.entries.append((name, slice(off, off + n), shape))
            self._slices[name] = (slice(off, off + n), shape)
            off += n

        d = arch.d
        for c, (n_blocks, heads) in enumerate(arch.couples):
            in_dim = arch.input_dim if c == 0 else d
            add(f"c{c}.embed.W_in", (in_dim, d))
            add(f"c{c}.embed.b_in", (d,))
            for b in range(n_blocks):
                p = f"c{c}.b{b}"
                add(f"{p}.lnA_gain", (d,)); add(f"{p}.lnA_bias", (d,))
                for h in range(heads):
                    hp = f"{p}.h{h}"
                    for nm in ("W_q", "b_q", "W_k", "b_k",
                               "W_v", "b_v", "W_o", "b_o"):
                        add(f"{hp}.{nm}",
                            (d, d) if nm.startswith("W") else (d,))
                add(f"{p}.lnF_gain", (d,)); add(f"{p}.lnF_bias", (d,))
                add(f"{p}.W_1", (d, arch.ffn_hidden))
                add(f"{p}.b_1", (arch.ffn_hidden,))
                add(f"{p}.W_2", (arch.ffn_hidden, d))
                add(f"{p}.b_2", (d,))
            for nm, shape in head_spec:
                add(f"c{c}.head.{nm}", shape)
        self.n_params = off
        # segments at component boundaries, per couple: a list
        # of (segment_name, gene_index_array)
        self.segments = []
        for c, (n_blocks, heads) in enumerate(arch.couples):
            segs = []
            segs.append((f"c{c}.embed",
                         self._idx([f"c{c}.embed.W_in",
                                    f"c{c}.embed.b_in"])))
            for b in range(n_blocks):
                p = f"c{c}.b{b}"
                attn = [f"{p}.lnA_gain", f"{p}.lnA_bias"]
                for h in range(heads):
                    hp = f"{p}.h{h}"
                    attn += [f"{hp}.{nm}" for nm in
                             ("W_q", "b_q", "W_k", "b_k",
                              "W_v", "b_v", "W_o", "b_o")]
                segs.append((f"{p}.attn", self._idx(attn)))
                segs.append((f"{p}.ffn", self._idx(
                    [f"{p}.lnF_gain", f"{p}.lnF_bias",
                     f"{p}.W_1", f"{p}.b_1", f"{p}.W_2",
                     f"{p}.b_2"])))
            self.segments.append(segs)
        self.signature = hashlib.sha256(json.dumps(
            [(n, s.start, s.stop, list(sh))
             for n, s, sh in self.entries]).encode()).hexdigest()

    def _idx(self, names):
        out = []
        for n in names:
            sl, _ = self._slices[n]
            out.append(np.arange(sl.start, sl.stop))
        return np.concatenate(out)

    def sl(self, name):
        return self._slices[name][0]

    def get(self, G, name):
        """Named array from genome rows G: (..., n_params)."""
        sl, shape = self._slices[name]
        return G[..., sl].reshape(G.shape[:-1] + shape)

    def head_names(self, couple):
        return [n for n, _, _ in self.entries
                if n.startswith(f"c{couple}.head.")]

    def unpack_dict(self, vec):
        return {n: np.array(vec[sl]).reshape(sh)
                for n, sl, sh in self.entries}

    def pack_dict(self, d):
        out = np.zeros(self.n_params)
        for n, sl, sh in self.entries:
            out[sl] = np.asarray(d[n], dtype=float).reshape(-1)
        return out


# ================================================================
# Encoder math (backend-agnostic: xp is numpy or jax.numpy)
# ================================================================

def _ln(xp, x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / xp.sqrt(var + eps) \
        * gain[:, None, None, :] + bias[:, None, None, :]


def _attn_values(xp, Hb, W_q, b_q, W_k, b_k, W_v, b_v, scale):
    Q = xp.einsum("mbnd,mde->mbne", Hb, W_q) + b_q[:, None, None, :]
    K = xp.einsum("mbnd,mde->mbne", Hb, W_k) + b_k[:, None, None, :]
    V = xp.einsum("mbnd,mde->mbne", Hb, W_v) + b_v[:, None, None, :]
    scores = xp.einsum("mbnd,mbkd->mbnk", Q, K) * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = xp.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return xp.einsum("mbnk,mbkd->mbnd", attn, V)


def _segment_forward(xp, layout, c, seg_i, G, A):
    """One encoder segment. G: (M, n_params) backend array;
    A: (M, B, n, d_in) activations -> (M, B, n, d)."""
    arch = layout.arch
    scale = arch.attn_scale
    segs = layout.segments[c]
    name = segs[seg_i][0]
    if name.endswith(".embed"):
        W_in = layout.get(G, f"c{c}.embed.W_in")
        b_in = layout.get(G, f"c{c}.embed.b_in")
        if A.ndim == 3:      # raw inputs (B, n, input_dim)
            return xp.einsum("bnc,mcd->mbnd", A, W_in) \
                + b_in[:, None, None, :]
        # tokens from the previous couple (M, B, n, d)
        return xp.einsum("mbnd,mde->mbne", A, W_in) \
            + b_in[:, None, None, :]
    b = int(name.split(".b")[1].split(".")[0])
    p = f"c{c}.b{b}"
    if name.endswith(".attn"):
        n_blocks, heads = arch.couples[c]
        Hb = _ln(xp, A, layout.get(G, f"{p}.lnA_gain"),
                 layout.get(G, f"{p}.lnA_bias"), arch.ln_eps)
        O = None
        for h in range(heads):
            hp = f"{p}.h{h}"
            Ah = _attn_values(
                xp, Hb, layout.get(G, f"{hp}.W_q"),
                layout.get(G, f"{hp}.b_q"),
                layout.get(G, f"{hp}.W_k"),
                layout.get(G, f"{hp}.b_k"),
                layout.get(G, f"{hp}.W_v"),
                layout.get(G, f"{hp}.b_v"), scale)
            Oh = xp.einsum("mbnd,mde->mbne", Ah,
                           layout.get(G, f"{hp}.W_o")) \
                + layout.get(G, f"{hp}.b_o")[:, None, None, :]
            O = Oh if O is None else O + Oh
        return A + O
    # FFN branch
    Hf = _ln(xp, A, layout.get(G, f"{p}.lnF_gain"),
             layout.get(G, f"{p}.lnF_bias"), arch.ln_eps)
    A1 = xp.tanh(xp.einsum("mbnd,mde->mbne", Hf,
                           layout.get(G, f"{p}.W_1"))
                 + layout.get(G, f"{p}.b_1")[:, None, None, :])
    F = xp.einsum("mbnd,mde->mbne", A1, layout.get(G, f"{p}.W_2")) \
        + layout.get(G, f"{p}.b_2")[:, None, None, :]
    return A + F


def encode_population(backend, layout, G, X, groups=None,
                      cache=None, x_tag="", chunk=128):
    """Encode a stacked population. G: (M, n_params) numpy;
    X: (B, n, input_dim) numpy -> Y: (M, B, n, d) numpy, the
    LAST couple's encoder output.

    groups: list of (start, stop) row ranges (species). Within
    a group, a segment whose genes are identical across the
    group AND whose input activation is still group-shared is
    computed once from the group's representative row. If
    `cache` (a dict) is given, shared segment outputs are
    memoised under a prefix hash (seed-reproducible inputs +
    gene bytes), so later calls with the same prefix reuse
    the exact stored floats. Without cache/sharing every row
    is computed in full, in population chunks of `chunk` rows;
    chunking never changes results (rows are independent and
    the batched einsums are row-wise bit-identical, verified
    against the stage-3 genome code)."""
    xp = backend.xp
    G = np.asarray(G, dtype=float)
    X = np.asarray(X, dtype=float)
    M = len(G)
    if groups is None:
        groups = [(0, M)]
    Gx = backend.from_numpy(G)
    outs = [None] * M
    for (gs, ge) in groups:
        A = backend.from_numpy(X)          # shared across rows
        shared = True
        prefix = hashlib.sha256(
            x_tag.encode() + X.tobytes()).hexdigest()
        for c in range(len(layout.arch.couples)):
            # couple c > 0 reads couple c-1's output tokens
            # as its input activation A (shared or row-wise,
            # exactly as couple c-1 left it)
            for seg_i, (seg_name, idx) in enumerate(
                    layout.segments[c]):
                const = bool(np.all(
                    G[gs:ge, idx] == G[gs, idx]))
                if shared and const:
                    key = (c, seg_i, prefix) if cache is not None \
                        else None
                    if key is not None and key in cache:
                        A = backend.from_numpy(cache[key])
                    else:
                        A = _segment_forward(
                            xp, layout, c, seg_i,
                            Gx[gs:gs + 1], A)
                        if key is not None:
                            cache[key] = backend.to_numpy(A)
                    prefix = hashlib.sha256(
                        prefix.encode()
                        + G[gs, idx].tobytes()).hexdigest()
                else:
                    if shared:
                        # materialise the group-shared input
                        # once; from here the group is row-wise
                        if A.ndim == 3:
                            # raw inputs (B, n, in): add the
                            # member axis first
                            A = xp.repeat(A[None], ge - gs,
                                          axis=0)
                        else:
                            A = xp.repeat(A, ge - gs, axis=0)
                        shared = False
                    parts = []
                    for s in range(gs, ge, chunk):
                        e = min(s + chunk, ge)
                        parts.append(_segment_forward(
                            xp, layout, c, seg_i, Gx[s:e],
                            A[s - gs:e - gs]))
                    A = parts[0] if len(parts) == 1 \
                        else xp.concatenate(parts, axis=0)
                    prefix = None
            # end segments of couple c; A is the couple output
        Y = backend.to_numpy(A)
        if shared:
            Y = np.repeat(Y, ge - gs, axis=0)
        for r in range(gs, ge):
            outs[r] = Y[r - gs]
    return np.stack(outs, axis=0)


def tour_lengths_batched(X, picks):
    """X: (B, n, 2); picks: (..., B, n) -> lengths (..., B).
    Same arithmetic as attn_genome.tour_lengths_batched."""
    B = X.shape[0]
    if picks.ndim == 2:
        pts = X[np.arange(B)[:, None], picks]
    else:
        pts = X[np.arange(B)[None, :, None], picks]
    nxt = np.roll(pts, -1, axis=-2)
    return np.sqrt(((pts - nxt) ** 2).sum(axis=-1)).sum(axis=-1)


# ================================================================
# Operators: mutation classes, breeding, species creation
# ================================================================

def prob_vector(classes, phase, cfg: EngineConfig):
    """Per-gene mutation probability by class and phase.
    Phase A: only the newest component (interface) and the
    decoder evolve; base rate 0. Phase B: old genes at
    mut_p * old_rate_factor; draw genes never mutate (they
    are re-imposed after breeding regardless)."""
    p = np.zeros(len(classes))
    if phase == "A":
        p[classes == CLS_IFACE] = cfg.mut_p
        p[classes == CLS_DECODER] = cfg.mut_p
    elif phase == "B":
        p[classes == CLS_OLD] = cfg.mut_p * cfg.old_rate_factor
        p[classes == CLS_IFACE] = cfg.mut_p
        p[classes == CLS_DECODER] = cfg.mut_p
    else:
        raise ValueError(phase)
    return p


def full_rate_runs(classes):
    """Contiguous slices where uniform crossover applies
    (interface + decoder genes), as in the stage runners."""
    runs = []
    full = (classes == CLS_IFACE) | (classes == CLS_DECODER)
    i, n = 0, len(classes)
    while i < n:
        if full[i]:
            j = i
            while j < n and full[j]:
                j += 1
            runs.append(slice(i, j))
            i = j
        else:
            i += 1
    return runs


def mutate(vec, prob_vec, rng, sigma):
    mut = rng.random(len(vec)) < prob_vec
    return vec + np.where(mut, rng.normal(0, sigma, len(vec)), 0.0)


def breed_species(pop, scores, draw_vec, classes, prob_vec,
                  rng, cfg: EngineConfig, target):
    """One generation of within-species breeding. scores are
    higher-is-better. Elites copied; tournament selection;
    uniform crossover on full-rate runs only (old genes are
    inherited whole from the first parent, draw genes are
    re-imposed after mutation) -- the stage runners' rule."""
    size = len(pop)
    order = np.argsort(-scores)
    fits = scores - scores.min() + 1e-9
    runs = full_rate_runs(classes)
    new_pop = [pop[i].copy() for i in order[:cfg.elite]]

    def tourney():
        idx = rng.choice(size, cfg.tournament, replace=False)
        return pop[idx[np.argmax(fits[idx])]]

    while len(new_pop) < target:
        a, b = tourney(), tourney()
        child = a.copy()
        for sl in runs:
            take_b = rng.random(sl.stop - sl.start) < 0.5
            child[sl] = np.where(take_b, b[sl], a[sl])
        child = mutate(child, prob_vec, rng, cfg.mut_sigma)
        child[classes == CLS_DRAW] = draw_vec[classes == CLS_DRAW]
        new_pop.append(child)
    return np.array(new_pop[:target])


def create_species(base_genome, classes, n_species, per_species,
                   draw_rng, run_rng, prob_vec, cfg: EngineConfig,
                   lineage, init_mode="transplant",
                   next_id=0):
    """A fresh species set on a preserved base. Each species
    gets a fresh random draw of the draw-class genes (drawn
    from draw_rng, N(0, draw_sigma)); members are the base
    with that draw imposed, plus one mutation application
    (init_mode='transplant', the growth-program precedent),
    or fully random genomes with the draw imposed
    (init_mode='random', the stage-0 precedent)."""
    n = len(base_genome)
    draw_mask = classes == CLS_DRAW
    species = []
    for s in range(n_species):
        draw = np.zeros(n)
        draw[draw_mask] = draw_rng.normal(
            0, cfg.draw_sigma, int(draw_mask.sum()))
        pop = []
        for _ in range(per_species):
            if init_mode == "random":
                ind = run_rng.normal(0, cfg.draw_sigma, n)
            else:
                ind = mutate(base_genome, prob_vec, run_rng,
                             cfg.mut_sigma)
            ind[draw_mask] = draw[draw_mask]
            pop.append(ind)
        species.append({
            "id": next_id + s,
            "lineage": lineage if lineage is not None
                       else f"L{next_id + s}",
            "draw": draw, "pop": np.array(pop),
            "target": per_species,
            "best": {"score": -np.inf, "genome": None,
                     "generation": -1},
        })
    return species


def split_counts(total, k):
    """Split `total` as evenly as possible into k parts."""
    base = total // k
    rem = total - base * k
    return [base + (1 if i < rem else 0) for i in range(k)]


def regenerate_species(founders, classes, total_species,
                       per_species, draw_rng, run_rng, prob_vec,
                       cfg: EngineConfig, next_id=0):
    """Founder regeneration: the single code path behind
    BOTH cataclysm regeneration and growth-stage
    regeneration. founders: list of {"genome", "lineage"}.
    The new species set is split as evenly as possible among
    the founders; each founder seeds its share with fresh
    random draws on its own base; every new species records
    its founder's lineage id."""
    counts = split_counts(total_species, len(founders))
    species = []
    nid = next_id
    for f, cnt in zip(founders, counts):
        if cnt == 0:
            continue
        sp = create_species(f["genome"], classes, cnt,
                            per_species, draw_rng, run_rng,
                            prob_vec, cfg, lineage=f["lineage"],
                            init_mode="transplant", next_id=nid)
        species.extend(sp)
        nid += cnt
    return species


# ================================================================
# Culling (Schedule A) and plateau read
# ================================================================

def cull_species(species, champ_scores, cfg: EngineConfig,
                 total_pop):
    """Schedule A: drop the worst cull_fraction of species
    by champion score on the fixed elimination set, floor
    species_floor; survivors' targets redistribute so the
    total population is preserved. Returns (survivors,
    log_entry)."""
    rank = np.argsort(-np.asarray(champ_scores))   # best first
    n_remove = int(round(cfg.cull_fraction * len(species)))
    n_remove = min(n_remove, len(species) - cfg.species_floor)
    if n_remove <= 0:
        return species, None
    keep = [int(i) for i in rank[:len(species) - n_remove]]
    dropped = [int(species[i]["id"])
               for i in rank[len(species) - n_remove:]]
    survivors = [species[i] for i in keep]
    counts = split_counts(total_pop, len(survivors))
    for sp, t in zip(survivors, counts):
        sp["target"] = t
    entry = {"event": "cullA",
             "species_before": len(species),
             "species_after": len(survivors),
             "eliminated_species_ids": dropped,
             "best_champion_elim_score":
                 float(max(champ_scores)),
             "worst_survivor_champion_elim_score":
                 float(min(champ_scores[i] for i in keep))}
    return survivors, entry


def plateau_read(series, window, threshold):
    """The stage records' operational plateau read, on a
    higher-is-better per-generation series: fit the slope
    over the final `window` generations; plateaued iff the
    fitted improvement over the window is under `threshold`
    of the level. Needs at least `window` points."""
    s = np.asarray(series, dtype=float)
    if len(s) < window:
        return {"plateaued": False, "reason": "window not full",
                "window": int(len(s))}
    tail = s[-window:]
    slope, _ = np.polyfit(np.arange(window), tail, 1)
    improvement = float(slope * (window - 1))
    level = float(np.mean(tail))
    rel = improvement / abs(level) if level != 0 else 0.0
    return {"plateaued": bool(rel < threshold),
            "slope_per_generation": float(slope),
            "relative_improvement_over_window": float(rel),
            "window": int(window), "threshold": threshold}


# ================================================================
# Growth transplants. Each returns (arch_new, template,
# classes_new): the template is the preserved base genome in
# the new layout with draw positions zeroed (filled per
# species at creation) and interface positions at their
# function-preserving initial values. Every pre-existing
# gene becomes CLS_OLD except the active couple's head,
# which stays CLS_DECODER (the standing rule).
# ================================================================

def _base_classes(layout_new, active_couple):
    classes = np.full(layout_new.n_params, CLS_OLD, dtype=np.int8)
    for name in layout_new.head_names(active_couple):
        sl = layout_new.sl(name)
        classes[sl] = CLS_DECODER
    return classes


def transplant_add_head(arch, layout_old, genome, head_spec_fn):
    couples = [list(c) for c in arch.couples]
    couples[-1][1] += 1
    arch_new = ArchSpec(arch.input_dim, arch.d, arch.ffn_hidden,
                        tuple(tuple(c) for c in couples),
                        arch.ln_eps, arch.attn_scale)
    layout_new = Layout(arch_new, head_spec_fn(arch_new.d))
    old_d = layout_old.unpack_dict(genome)
    new_d = {n: np.zeros(sh) for n, _, sh in layout_new.entries}
    for n, arr in old_d.items():
        if n in new_d:
            new_d[n] = arr
    classes = _base_classes(layout_new, len(couples) - 1)
    # the new head (last head index in each block of the
    # active couple): Q/K/V are the species draw, Wo/bo the
    # evolving interface at exactly zero (stage-3 mechanism)
    c = len(couples) - 1
    n_blocks, heads = couples[-1]
    for b in range(n_blocks):
        hp = f"c{c}.b{b}.h{heads - 1}"
        for nm in ("W_q", "b_q", "W_k", "b_k", "W_v", "b_v"):
            classes[layout_new.sl(f"{hp}.{nm}")] = CLS_DRAW
        for nm in ("W_o", "b_o"):
            classes[layout_new.sl(f"{hp}.{nm}")] = CLS_IFACE
    return arch_new, layout_new, layout_new.pack_dict(new_d), classes


def transplant_add_block(arch, layout_old, genome, head_spec_fn):
    couples = [list(c) for c in arch.couples]
    couples[-1][0] += 1
    arch_new = ArchSpec(arch.input_dim, arch.d, arch.ffn_hidden,
                        tuple(tuple(c) for c in couples),
                        arch.ln_eps, arch.attn_scale)
    layout_new = Layout(arch_new, head_spec_fn(arch_new.d))
    old_d = layout_old.unpack_dict(genome)
    new_d = {n: np.zeros(sh) for n, _, sh in layout_new.entries}
    for n, arr in old_d.items():
        if n in new_d:
            new_d[n] = arr
    classes = _base_classes(layout_new, len(couples) - 1)
    # the new block (last block of the active couple) is the
    # identity at transplant: branch norms at gain 1 / bias 0
    # (interface), input-side genes drawn per species (QKV of
    # every head + FFN W_1/b_1), output sides exactly zero
    # (interface): Wo/b_o and W_2/b_2
    c = len(couples) - 1
    n_blocks, heads = couples[-1]
    p = f"c{c}.b{n_blocks - 1}"
    new_d[f"{p}.lnA_gain"] = np.ones(arch.d)
    new_d[f"{p}.lnF_gain"] = np.ones(arch.d)
    for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
        classes[layout_new.sl(f"{p}.{nm}")] = CLS_IFACE
    for h in range(heads):
        hp = f"{p}.h{h}"
        for nm in ("W_q", "b_q", "W_k", "b_k", "W_v", "b_v"):
            classes[layout_new.sl(f"{hp}.{nm}")] = CLS_DRAW
        for nm in ("W_o", "b_o"):
            classes[layout_new.sl(f"{hp}.{nm}")] = CLS_IFACE
    classes[layout_new.sl(f"{p}.W_1")] = CLS_DRAW
    classes[layout_new.sl(f"{p}.b_1")] = CLS_DRAW
    classes[layout_new.sl(f"{p}.W_2")] = CLS_IFACE
    classes[layout_new.sl(f"{p}.b_2")] = CLS_IFACE
    return arch_new, layout_new, layout_new.pack_dict(new_d), classes


def transplant_stack(arch, layout_old, genome, head_spec_fn):
    couples = [list(c) for c in arch.couples]
    couples.append(list(couples[-1]))     # mirror the last couple
    arch_new = ArchSpec(arch.input_dim, arch.d, arch.ffn_hidden,
                        tuple(tuple(c) for c in couples),
                        arch.ln_eps, arch.attn_scale)
    layout_new = Layout(arch_new, head_spec_fn(arch_new.d))
    old_d = layout_old.unpack_dict(genome)
    new_d = {n: np.zeros(sh) for n, _, sh in layout_new.entries}
    for n, arr in old_d.items():
        if n in new_d:
            new_d[n] = arr
    c = len(couples) - 1
    # new couple's token projection is the identity (old,
    # forced): tokens pass through unchanged
    new_d[f"c{c}.embed.W_in"] = np.eye(arch.d)
    # new couple's head starts as an exact copy of the
    # previous couple's head (it is the active decoder now)
    prev = c - 1
    for n in layout_new.head_names(c):
        short = n.split(".head.")[1]
        new_d[n] = old_d[f"c{prev}.head.{short}"]
    classes = _base_classes(layout_new, c)
    # new couple's blocks follow the add-block class pattern
    n_blocks, heads = couples[-1]
    for b in range(n_blocks):
        p = f"c{c}.b{b}"
        new_d[f"{p}.lnA_gain"] = np.ones(arch.d)
        new_d[f"{p}.lnF_gain"] = np.ones(arch.d)
        for nm in ("lnA_gain", "lnA_bias", "lnF_gain", "lnF_bias"):
            classes[layout_new.sl(f"{p}.{nm}")] = CLS_IFACE
        for h in range(heads):
            hp = f"{p}.h{h}"
            for nm in ("W_q", "b_q", "W_k", "b_k", "W_v", "b_v"):
                classes[layout_new.sl(f"{hp}.{nm}")] = CLS_DRAW
            for nm in ("W_o", "b_o"):
                classes[layout_new.sl(f"{hp}.{nm}")] = CLS_IFACE
        classes[layout_new.sl(f"{p}.W_1")] = CLS_DRAW
        classes[layout_new.sl(f"{p}.b_1")] = CLS_DRAW
        classes[layout_new.sl(f"{p}.W_2")] = CLS_IFACE
        classes[layout_new.sl(f"{p}.b_2")] = CLS_IFACE
    return arch_new, layout_new, layout_new.pack_dict(new_d), classes


def transplant_widen(arch, layout_old, genome, head_spec_fn,
                     step, widen_head):
    """Widen d -> d + step, function-preserving in exact
    arithmetic (Net2Net-style, adapted to this architecture's
    layer norms):
      - new embedding coordinates equal the token's own mean
        over the old coordinates (W_in new columns = column
        means of the old W_in), so LN statistics over the
        widened axis keep the same mean and a scaled variance;
      - ln_eps scales by d/d' and old LN gains by sqrt(d/d'),
        which together restore the old normalised values
        exactly;
      - Q/K new output rows are exactly zero (attention
        scores unchanged); output projections' new columns
        are per-row means of the old columns, so every new
        stream coordinate equals the mean of the old output
        (the invariant the next LN relies on);
      - free input-side new genes (V rows, Q/K input columns,
        FFN W_1 input rows) are the species draw; new output
        sides are the evolving interface at their preserving
        values; the head is widened by the domain hook.
    In floating point the widened network agrees with the
    base to ~1e-15 relative; the runner's hard gate is the
    domain preservation check (identical validation tours /
    decisions, hence exactly equal lengths)."""
    d0, d1 = arch.d, arch.d + step
    arch_new = ArchSpec(arch.input_dim, d1, arch.ffn_hidden,
                        arch.couples, arch.ln_eps * d0 / d1,
                        arch.attn_scale)
    layout_new = Layout(arch_new, head_spec_fn(arch_new.d))
    old_d = layout_old.unpack_dict(genome)
    new_d = {n: np.zeros(sh) for n, _, sh in layout_new.entries}
    classes = _base_classes(layout_new, len(arch.couples) - 1)
    gain_fix = np.sqrt(d0 / d1)

    def widen_mat(W, row_means_out=False):
        """(r, d0) -> (r, d1): old columns kept; new columns
        are the per-row mean over old columns."""
        out = np.zeros((W.shape[0], d1))
        out[:, :d0] = W
        out[:, d0:] = W.mean(axis=1, keepdims=True)
        return out

    for c, (n_blocks, heads) in enumerate(arch.couples):
        # embed: couple 0 reads raw inputs (rows unchanged);
        # later couples read tokens (both axes widen, with
        # the W_o pattern: new input rows 0, new output
        # columns per-row means)
        W_in = old_d[f"c{c}.embed.W_in"]
        if c == 0:
            new_d[f"c{c}.embed.W_in"] = widen_mat(W_in)
        else:
            NW = np.zeros((d1, d1))
            NW[:d0, :d0] = W_in
            NW[:d0, d0:] = W_in.mean(axis=1, keepdims=True)
            new_d[f"c{c}.embed.W_in"] = NW
            cls = classes[layout_new.sl(
                f"c{c}.embed.W_in")].reshape(d1, d1)
            cls[:, d0:] = CLS_IFACE
            cls[d0:, :] = CLS_IFACE
            cls[:d0, :d0] = CLS_OLD
        b_in = old_d[f"c{c}.embed.b_in"]
        nb = np.zeros(d1); nb[:d0] = b_in; nb[d0:] = b_in.mean()
        new_d[f"c{c}.embed.b_in"] = nb
        classes[layout_new.sl(f"c{c}.embed.b_in")][d0:] = CLS_IFACE
        for b in range(n_blocks):
            p = f"c{c}.b{b}"
            for nm in ("lnA_gain", "lnA_bias", "lnF_gain",
                       "lnF_bias"):
                old = old_d[f"{p}.{nm}"]
                nv = np.zeros(d1)
                nv[:d0] = old * gain_fix if nm.endswith("gain") \
                    else old
                nv[d0:] = 1.0 if nm.endswith("gain") else 0.0
                new_d[f"{p}.{nm}"] = nv
                sl = layout_new.sl(f"{p}.{nm}")
                classes[sl][d0:] = CLS_IFACE
            for h in range(heads):
                hp = f"{p}.h{h}"
                for nm in ("W_q", "W_k"):
                    # x @ W: axis 0 = input dim, axis 1 =
                    # output dim. New INPUT rows multiply the
                    # (exactly-zero) normalised new coords ->
                    # free, species draw. New OUTPUT columns
                    # produce Q_new/K_new and must start at
                    # exactly zero -> interface.
                    W = old_d[f"{hp}.{nm}"]
                    NW = np.zeros((d1, d1)); NW[:d0, :d0] = W
                    new_d[f"{hp}.{nm}"] = NW
                    sl = layout_new.sl(f"{hp}.{nm}")
                    cls = classes[sl].reshape(d1, d1)
                    cls[d0:, :] = CLS_DRAW       # input rows
                    cls[:, d0:] = CLS_IFACE      # output cols
                    bv = np.zeros(d1)
                    bv[:d0] = old_d[f"{hp}.b_{nm[-1]}"]
                    new_d[f"{hp}.b_{nm[-1]}"] = bv
                    classes[layout_new.sl(f"{hp}.b_{nm[-1]}")][d0:] \
                        = CLS_IFACE
                # W_v: every entry touching a new index is a
                # free input-side value -> species draw
                new_d[f"{hp}.W_v"] = np.zeros((d1, d1))
                new_d[f"{hp}.W_v"][:d0, :d0] = old_d[f"{hp}.W_v"]
                sl = layout_new.sl(f"{hp}.W_v")
                cls = classes[sl].reshape(d1, d1)
                cls[:, d0:] = CLS_DRAW
                cls[d0:, :] = CLS_DRAW
                bv = np.zeros(d1)
                bv[:d0] = old_d[f"{hp}.b_v"]
                new_d[f"{hp}.b_v"] = bv
                classes[layout_new.sl(f"{hp}.b_v")][d0:] = CLS_DRAW
                # W_o: new input rows forced 0; new output
                # columns are per-row means (interface)
                W = old_d[f"{hp}.W_o"]
                NW = np.zeros((d1, d1))
                NW[:d0, :d0] = W
                NW[:d0, d0:] = W.mean(axis=1, keepdims=True)
                new_d[f"{hp}.W_o"] = NW
                sl = layout_new.sl(f"{hp}.W_o")
                cls = classes[sl].reshape(d1, d1)
                cls[:, d0:] = CLS_IFACE      # new output cols
                cls[d0:, :] = CLS_IFACE      # new input rows
                cls[:d0, :d0] = CLS_OLD
                bo = np.zeros(d1)
                bo[:d0] = old_d[f"{hp}.b_o"]
                bo[d0:] = old_d[f"{hp}.b_o"].mean()
                new_d[f"{hp}.b_o"] = bo
                classes[layout_new.sl(f"{hp}.b_o")][d0:] = CLS_IFACE
            # FFN: W_1 new input rows are the draw; W_2 new
            # output columns are per-row means (interface)
            W1 = old_d[f"{p}.W_1"]
            NW1 = np.zeros((d1, arch.ffn_hidden))
            NW1[:d0, :] = W1
            new_d[f"{p}.W_1"] = NW1
            classes[layout_new.sl(f"{p}.W_1")].reshape(
                d1, arch.ffn_hidden)[d0:, :] = CLS_DRAW
            new_d[f"{p}.b_1"] = old_d[f"{p}.b_1"]
            W2 = old_d[f"{p}.W_2"]
            new_d[f"{p}.W_2"] = widen_mat(W2)
            classes[layout_new.sl(f"{p}.W_2")].reshape(
                arch.ffn_hidden, d1)[:, d0:] = CLS_IFACE
            b2 = np.zeros(d1)
            b2[:d0] = old_d[f"{p}.b_2"]
            b2[d0:] = old_d[f"{p}.b_2"].mean()
            new_d[f"{p}.b_2"] = b2
            classes[layout_new.sl(f"{p}.b_2")][d0:] = CLS_IFACE
        # head: domain hook widens the head dict entries
        head_old = {n.split(".head.")[1]: old_d[n]
                    for n in layout_old.head_names(c)}
        head_new = widen_head(head_old, d0, d1)
        for short, arr in head_new.items():
            new_d[f"c{c}.head.{short}"] = arr
    return arch_new, layout_new, layout_new.pack_dict(new_d), classes


TRANSPLANTS = ("widen", "add_head", "add_block", "stack")


def make_transplant(rule, arch, layout_old, genome, head_spec_fn,
                    cfg: EngineConfig, widen_head):
    if rule == "widen":
        return transplant_widen(arch, layout_old, genome,
                                head_spec_fn, cfg.widen_step,
                                widen_head)
    if rule == "add_head":
        return transplant_add_head(arch, layout_old, genome,
                                   head_spec_fn)
    if rule == "add_block":
        return transplant_add_block(arch, layout_old, genome,
                                    head_spec_fn)
    if rule == "stack":
        return transplant_stack(arch, layout_old, genome,
                                head_spec_fn)
    raise ValueError(rule)


# ================================================================
# Founders (cataclysm survivors / growth regeneration bases)
# ================================================================

def select_founders(species, val_scores, founder_count):
    """Founders = the best champion of each of the top
    founder_count DISTINCT species, ranked by their
    champions' fixed-validation scores (one per species,
    never two from the same species -- a slightly weaker
    champion from another species may carry better genes
    for what comes next). val_scores: per-species champion
    validation scores aligned with `species`. Returns a
    list of founder dicts {genome, lineage, source_species_id,
    val_score}; lineage roots at the founder species' id."""
    order = np.argsort(-np.asarray(val_scores))
    founders = []
    for i in order[:max(1, min(founder_count, len(species)))]:
        sp = species[int(i)]
        founders.append({
            "genome": sp["best"]["genome"].copy(),
            "lineage": f"L{sp['id']}",
            "source_species_id": int(sp["id"]),
            "val_score": float(val_scores[int(i)]),
        })
    return founders


# ================================================================
# Checkpointing (atomic; resume refuses a mismatched config)
# ================================================================

def atomic_npz(path, **arrays):
    import os
    from pathlib import Path
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path, obj):
    import os
    from pathlib import Path
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def checkpoint_payload(cfg: EngineConfig, arch: ArchSpec,
                       layout: Layout, state: dict) -> dict:
    """Assemble the checkpoint arrays. `state` carries the
    runner's dynamic state (species, rng, counters, logs);
    the config/arch/layout fingerprints travel alongside so
    resume can refuse a mismatch."""
    species = state["species"]
    payload = {
        "config_canonical": cfg.canonical(),
        "arch_signature": arch.signature(),
        "layout_signature": layout.signature,
        "classes": state["classes"],
        "pops": np.concatenate([sp["pop"] for sp in species]),
        "sizes": np.array([len(sp["pop"]) for sp in species]),
        "species_ids": np.array([sp["id"] for sp in species]),
        "lineages": np.array([sp["lineage"] for sp in species]),
        "draws": np.array([sp["draw"] for sp in species]),
        "targets": np.array([sp["target"] for sp in species]),
        "best_genomes": np.array(
            [sp["best"]["genome"] if sp["best"]["genome"]
             is not None else np.zeros(layout.n_params)
             for sp in species]),
        "best_scores": np.array([sp["best"]["score"]
                                 for sp in species]),
        "best_gens": np.array([sp["best"]["generation"]
                               for sp in species]),
        "rng_state": json.dumps(state["rng"].bit_generator.state),
        "draw_rng_state": json.dumps(
            state["draw_rng"].bit_generator.state),
        "meta": json.dumps(state["meta"]),
    }
    return payload


def load_checkpoint(path, cfg: EngineConfig, arch: ArchSpec,
                    layout: Layout):
    """Load and verify a checkpoint. Refuses (SystemExit) if
    the saved config, architecture or layout does not match
    the current run exactly."""
    z = np.load(path, allow_pickle=False)
    if str(z["config_canonical"]) != cfg.canonical():
        raise SystemExit(
            "REFUSING to resume: checkpoint config does not "
            "match the current configuration.")
    if str(z["arch_signature"]) != arch.signature():
        raise SystemExit(
            "REFUSING to resume: checkpoint architecture does "
            "not match the current architecture.")
    if str(z["layout_signature"]) != layout.signature:
        raise SystemExit(
            "REFUSING to resume: checkpoint layout does not "
            "match the current layout.")
    sizes = z["sizes"]
    pops = np.split(z["pops"], np.cumsum(sizes)[:-1])
    species = []
    for k in range(len(sizes)):
        bg = z["best_genomes"][k]
        species.append({
            "id": int(z["species_ids"][k]),
            "lineage": str(z["lineages"][k]),
            "draw": z["draws"][k],
            "pop": pops[k],
            "target": int(z["targets"][k]),
            "best": {
                "score": float(z["best_scores"][k]),
                "genome": bg if z["best_gens"][k] >= 0 else None,
                "generation": int(z["best_gens"][k])},
        })
    rng = np.random.default_rng(cfg.run_seed)
    rng.bit_generator.state = json.loads(str(z["rng_state"]))
    draw_rng = np.random.default_rng(cfg.draw_seed)
    draw_rng.bit_generator.state = json.loads(
        str(z["draw_rng_state"]))
    return {"species": species, "classes": z["classes"],
            "rng": rng, "draw_rng": draw_rng,
            "meta": json.loads(str(z["meta"]))}
