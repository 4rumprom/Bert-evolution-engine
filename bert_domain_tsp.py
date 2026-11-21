"""TSP domain adapter for the parametric BERT growth engine.

Supplies what the domain-independent core does not know:
  - inputs: raw city coordinates (input_dim = 2), instances
    drawn from seeds (phase A: one fixed known set from
    phase_a_seed, replayed; phase B: per-generation sets
    recreatable from seed + generation number; elimination
    and validation sets from their own seeds);
  - decode head: the attention pointer decoder of the stage
    program (state = mean of encoder outputs + last picked
    token, learned start vector, tanh clip 10, visited mask,
    greedy), emitting a tour;
  - fitness: negative mean tour length (engine scores are
    higher-is-better); validation metric: negative mean gap
    against Held-Karp optima on the fixed validation set
    (seed 20261116, 200 instances at n = 10, the stage
    program's held-out set);
  - preservation check: identical tours and exactly equal
    tour lengths (max diff 0.0) on the validation set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be

try:
    from tsp_genome2 import held_karp
except Exception:  # pragma: no cover
    held_karp = None


def head_spec(d):
    return [("v_start", (d,)), ("W_qd", (d, d)), ("b_qd", (d,))]


def widen_head(head_old, d0, d1):
    """Pointer-head widening at preserving values: new input
    rows and new output columns of W_qd are zero, v_start /
    b_qd new entries zero, so decoder logits are unchanged
    in exact arithmetic."""
    out = {}
    v = np.zeros(d1); v[:d0] = head_old["v_start"]
    out["v_start"] = v
    W = np.zeros((d1, d1)); W[:d0, :d0] = head_old["W_qd"]
    out["W_qd"] = W
    b = np.zeros(d1); b[:d0] = head_old["b_qd"]
    out["b_qd"] = b
    return out


def decode_pointer(backend, layout, couple, G, Y):
    """Greedy pointer decode. G: (M, n_params) numpy genome
    rows (head genes of `couple`); Y: (M, B, n, d) numpy
    encoder outputs -> picks (M, B, n) numpy int."""
    xp = backend.xp
    d = Y.shape[-1]
    scale = layout.arch.attn_scale
    Gx = backend.from_numpy(np.asarray(G, dtype=float))
    Yx = backend.from_numpy(np.asarray(Y, dtype=float))
    v_start = layout.get(Gx, f"c{couple}.head.v_start")
    W_qd = layout.get(Gx, f"c{couple}.head.W_qd")
    b_qd = layout.get(Gx, f"c{couple}.head.b_qd")
    M, B, n, _ = Y.shape
    gmean = Yx.mean(axis=2)                       # (M, B, d)
    last = xp.broadcast_to(v_start[:, None, :],
                           (M, B, d)).copy()
    visited = np.zeros((M, B, n), dtype=bool)     # host bookkeeping
    arM = np.arange(M)
    arB = np.arange(B)
    pick_list = []
    for t in range(n):
        state = gmean + last
        q = xp.einsum("mbd,mde->mbe", state, W_qd) \
            + b_qd[:, None, :]
        logits = be.CLIP * xp.tanh(
            xp.einsum("mbe,mbne->mbn", q, Yx) * scale)
        logits = xp.where(backend.from_numpy(visited),
                          -1e18, logits)
        pick = backend.to_numpy(xp.argmax(logits, axis=2))
        pick_list.append(pick)
        visited[arM[:, None], arB[None, :], pick] = True
        last = Yx[arM[:, None], arB[None, :], pick]
    return np.stack(pick_list, axis=2)


class TSPDomain:
    name = "tsp"
    input_dim = 2

    def __init__(self, cfg: be.EngineConfig, backend):
        self.cfg = cfg
        self.backend = backend
        dc = cfg.domain_config
        self.n_cities = int(dc.get("n_cities", 10))
        # fixed sets, each from its own seed
        vrng = np.random.default_rng(cfg.val_seed)
        self.X_val = vrng.random((cfg.n_val, self.n_cities, 2))
        if held_karp is None:
            raise RuntimeError("tsp_genome2.held_karp unavailable")
        self.optima = np.array(
            [held_karp(self.X_val[i]) for i in range(cfg.n_val)])
        erng = np.random.default_rng(cfg.elim_seed)
        self.X_elim = erng.random((cfg.n_elim, self.n_cities, 2))
        arng = np.random.default_rng(cfg.phase_a_seed)
        self.X_phase_a = arng.random(
            (cfg.n_train, self.n_cities, 2))

    # ---- batches ------------------------------------------------
    def train_batch(self, phase, stage, gen):
        if phase == "A":
            return {"X": self.X_phase_a, "tag": "phaseA_train"}
        seed = be.gen_instance_seed(self.cfg, stage, gen)
        rng = np.random.default_rng(seed)
        return {"X": rng.random((self.cfg.n_train, self.n_cities, 2)),
                "tag": f"phaseB_s{stage}_g{gen}"}

    def elim_batch(self):
        return {"X": self.X_elim, "tag": "elim"}

    def val_batch(self):
        return {"X": self.X_val, "tag": "val"}

    # ---- evaluation ---------------------------------------------
    def evaluate_lengths(self, layout, G, X, groups=None,
                         cache=None, x_tag=""):
        Y = be.encode_population(self.backend, layout, G, X,
                                 groups=groups, cache=cache,
                                 x_tag=x_tag,
                                 chunk=self.cfg.chunk)
        picks = decode_pointer(self.backend, layout,
                               len(layout.arch.couples) - 1, G, Y)
        assert (np.sort(picks, axis=-1)
                == np.arange(X.shape[1])).all(), \
            "invalid tour produced in population evaluation"
        return be.tour_lengths_batched(X, picks), picks

    def evaluate(self, layout, G, batch, groups=None, cache=None):
        """Engine fitness: negative mean tour length."""
        lens, _ = self.evaluate_lengths(
            layout, G, batch["X"], groups=groups, cache=cache,
            x_tag=batch["tag"])
        return -lens.mean(axis=1)

    def evaluate_single(self, layout, genome, batch):
        return float(self.evaluate(
            layout, np.asarray(genome)[None, :], batch)[0])

    # ---- validation ----------------------------------------------
    def validation_score(self, layout, genome):
        lens, _ = self.evaluate_lengths(
            layout, np.asarray(genome)[None, :], self.X_val)
        gaps = lens[0] / self.optima - 1.0
        return -float(gaps.mean())

    def validation_detail(self, layout, genome):
        lens, _ = self.evaluate_lengths(
            layout, np.asarray(genome)[None, :], self.X_val)
        gaps = lens[0] / self.optima - 1.0
        return {"mean_length": float(lens.mean()),
                "mean_optimum": float(self.optima.mean()),
                "mean_gap": float(gaps.mean()),
                "median_gap": float(np.median(gaps))}

    def preservation(self, layout_old, genome_old, layout_new,
                     genome_new):
        """Exact: identical tours and max length diff 0.0 on
        the fixed validation set."""
        lo, po = self.evaluate_lengths(
            layout_old, np.asarray(genome_old)[None, :],
            self.X_val)
        ln, pn = self.evaluate_lengths(
            layout_new, np.asarray(genome_new)[None, :],
            self.X_val)
        max_diff = float(np.max(np.abs(lo - ln)))
        tours_same = bool((po == pn).all())
        return {"pass": bool(max_diff == 0.0 and tours_same),
                "max_abs_length_difference": max_diff,
                "tours_identical": tours_same,
                "instances": int(len(self.X_val))}

    def random_baseline(self, layout, rng, n_draws=25):
        """Median validation score of fresh random genomes
        (the random baseline a stage must beat)."""
        G = rng.normal(0, self.cfg.draw_sigma,
                       (n_draws, layout.n_params))
        scores = []
        for i in range(n_draws):
            scores.append(self.validation_score(layout, G[i]))
        return float(np.median(scores))
