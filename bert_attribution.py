"""Back-propagation isolation for the BERT evolution
engine (sponsor direction, 2026-11-24: "modify to
isolate via back propagation").

WHY THIS EXISTS. Branch isolation used to cluster
per-problem SILENCING profiles. Silencing is zero
ablation, and ablation degrades as networks grow:
the rest of the network compensates for a removed
unit (self-repair, the Hydra effect, one mechanism
of which is LayerNorm rescaling, present at every
block here), so a silencing drop is a distorted,
lower-bound read; and the drop profiles are
dominated by a common problem-difficulty factor, so
the clusters came out indistinguishable from random
partitions under a random control (checked
2026-11-24: clustered group drops 301/293/253% of
gap vs random means 277/288/264; coherence 0.708 vs
random 0.685).

THE METHOD (edge attribution patching, specialised
to units). Every testable unit gets a scalar GATE
that multiplies the unit's contribution to the
stream (a feed-forward unit's contribution is its
activation times its outgoing W_2 row; a head's
contribution is its whole projected output, W_o and
b_o included, exactly the pieces silencing zeroes).
Gates at 1 are the intact network. The gradient of
a differentiable score with respect to gate u is,
by the chain rule, the inner product of the unit's
contribution with the score's gradient at the point
where the contribution enters: the first-order
estimate of what removing the unit would do, for
every unit at once, from ONE backward pass. This is
a measuring instrument only; evolution still makes
every change, no gradient ever trains a gene here.

THE SCORE. Engine fitness is a greedy tour length,
which has no gradient (the picks are discrete). The
differentiable surrogate anchors on the champion's
own greedy trajectory (the picks come from the
engine's exact decode and are held constant): at
each visited state of that trajectory it takes the
pointer distribution at temperature `tau` and
charges the EXPECTED leg length under that
distribution, closing the tour through the first
step's distribution. As tau -> 0 the distributions
collapse onto the greedy picks and the surrogate
becomes the true tour length; at finite tau it is a
smooth local relaxation of it, with gradients
flowing through the encoder outputs at every step.
(A free-running soft decode was tried first and
rejected on measurement: its lengths ran 29 to 47%
long and correlated only ~0.5 with the true lengths,
and its attributions anti-correlated with silencing.
The anchored surrogate is the one verified in
test_bert_attribution.py.)

PROFILES. attribution_profiles returns, per unit and
per validation problem, the first-order estimate of
the silencing drop in gap units (soft length change
on removing the unit, divided by the problem's
optimum): the direct analogue of
bert_branch.unit_drop_profiles, at the cost of one
backward pass per problem instead of one full
evaluation per unit.

Requires jax (x64). Run under the project venv:
~/workspace/bert-engine/.venv/bin/python.
"""

from __future__ import annotations

import numpy as np

import bert_engine as be

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except Exception as e:  # pragma: no cover - env dependent
    raise ImportError(
        "bert_attribution needs jax; run under "
        "~/workspace/bert-engine/.venv/bin/python "
        f"(import failed: {type(e).__name__}: {e})")


# ================================================================
# Differentiable mirror of the engine forward (single genome)
# ================================================================

def _ln(x, gain, bias, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * gain + bias


def _attn(Hb, W_q, b_q, W_k, b_k, W_v, b_v, scale):
    Q = Hb @ W_q + b_q
    K = Hb @ W_k + b_k
    V = Hb @ W_v + b_v
    scores = Q @ K.T * scale
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = jnp.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return attn @ V


def soft_forward(layout, genome, x, gates, picks, tau=1.0):
    """Anchored soft tour length for ONE problem.
    genome: (n_params,) jax array; x: (n, input_dim);
    gates: (n_units,) in bert_engine.silence_units
    order; picks: (n,) the genome's own greedy tour
    on this problem (constant, from the engine's
    decode); tau: temperature of the per-step pointer
    distributions. Returns a scalar (lower is
    better)."""
    arch = layout.arch
    g = {name: genome[sl].reshape(sh)
         for name, (sl, sh) in layout._slices.items()}
    units = be.silence_units(layout)
    gate_of = {u["name"]: gates[i] for i, u in enumerate(units)}
    A = x
    for c, (n_blocks, heads) in enumerate(arch.couples):
        A = A @ g[f"c{c}.embed.W_in"] + g[f"c{c}.embed.b_in"]
        for b in range(n_blocks):
            p = f"c{c}.b{b}"
            Hb = _ln(A, g[f"{p}.lnA_gain"], g[f"{p}.lnA_bias"],
                     arch.ln_eps)
            O = 0.0
            for h in range(heads):
                hp = f"{p}.h{h}"
                Ah = _attn(Hb, g[f"{hp}.W_q"], g[f"{hp}.b_q"],
                           g[f"{hp}.W_k"], g[f"{hp}.b_k"],
                           g[f"{hp}.W_v"], g[f"{hp}.b_v"],
                           arch.attn_scale)
                Oh = Ah @ g[f"{hp}.W_o"] + g[f"{hp}.b_o"]
                O = O + gate_of[f"{p}.head.h{h}"] * Oh
            A = A + O
            Hf = _ln(A, g[f"{p}.lnF_gain"], g[f"{p}.lnF_bias"],
                     arch.ln_eps)
            A1 = jnp.tanh(Hf @ g[f"{p}.W_1"] + g[f"{p}.b_1"])
            gf = jnp.array([gate_of[f"{p}.ffn.u{j}"]
                            for j in range(arch.ffn_hidden)])
            F = (A1 * gf) @ g[f"{p}.W_2"] + g[f"{p}.b_2"]
            A = A + F
    Y = A
    # ---- anchored soft decode: the greedy picks are
    # constants; at each visited state the pointer
    # distribution prices the next leg ----
    couple = len(arch.couples) - 1
    v_start = g[f"c{couple}.head.v_start"]
    W_qd = g[f"c{couple}.head.W_qd"]
    b_qd = g[f"c{couple}.head.b_qd"]
    n = x.shape[0]
    D = jnp.sqrt(((x[:, None, :] - x[None, :, :]) ** 2)
                 .sum(axis=-1))
    gmean = Y.mean(axis=0)
    visited = jnp.zeros(n)
    first_p = None
    L = 0.0
    for t in range(n):
        last = v_start if t == 0 else Y[picks[t - 1]]
        state = gmean + last
        q = state @ W_qd + b_qd
        logits = be.CLIP * jnp.tanh((Y @ q) * arch.attn_scale)
        logits = logits - 1e9 * visited
        p = jax.nn.softmax(logits / tau)
        if t == 0:
            first_p = p
        else:
            L = L + p @ D[picks[t - 1]]
        visited = visited.at[picks[t]].set(1.0)
    L = L + first_p @ D[picks[n - 1]]
    return L


def encode_jax(layout, genome, X):
    """Encoder-only forward, batched over problems
    X: (B, n, input_dim) numpy -> (B, n, d) numpy.
    Exists for the parity check against
    bert_engine.encode_population."""
    genome_j = jnp.asarray(genome, dtype=jnp.float64)
    ones = jnp.ones(len(be.silence_units(layout)))

    def enc_one(x):
        # reuse soft_forward's encoder by re-running
        # with a dummy decode is wasteful; inline copy:
        arch = layout.arch
        g = {name: genome_j[sl].reshape(sh)
             for name, (sl, sh) in layout._slices.items()}
        A = x
        for c, (n_blocks, heads) in enumerate(arch.couples):
            A = A @ g[f"c{c}.embed.W_in"] + g[f"c{c}.embed.b_in"]
            for b in range(n_blocks):
                p = f"c{c}.b{b}"
                Hb = _ln(A, g[f"{p}.lnA_gain"],
                         g[f"{p}.lnA_bias"], arch.ln_eps)
                O = 0.0
                for h in range(heads):
                    hp = f"{p}.h{h}"
                    Ah = _attn(Hb, g[f"{hp}.W_q"], g[f"{hp}.b_q"],
                               g[f"{hp}.W_k"], g[f"{hp}.b_k"],
                               g[f"{hp}.W_v"], g[f"{hp}.b_v"],
                               arch.attn_scale)
                    O = O + Ah @ g[f"{hp}.W_o"] + g[f"{hp}.b_o"]
                A = A + O
                Hf = _ln(A, g[f"{p}.lnF_gain"],
                         g[f"{p}.lnF_bias"], arch.ln_eps)
                A1 = jnp.tanh(Hf @ g[f"{p}.W_1"] + g[f"{p}.b_1"])
                A = A + A1 @ g[f"{p}.W_2"] + g[f"{p}.b_2"]
        return A

    Y = jax.vmap(enc_one)(jnp.asarray(X, dtype=jnp.float64))
    return np.asarray(Y)


# ================================================================
# Attribution profiles
# ================================================================

def attribution_profiles(domain, layout, genome, tau=1.0):
    """Per-unit, per-problem first-order silencing
    drops, estimated by back propagation. Returns
    (units, profiles): units from
    bert_engine.silence_units; profiles[i, p] is the
    estimated drop of unit i on validation problem p,
    in gap units (soft length increase on removal /
    optimum), the same units as
    bert_branch.unit_drop_profiles."""
    units = be.silence_units(layout)
    genome_j = jnp.asarray(np.asarray(genome, dtype=float),
                           dtype=jnp.float64)
    ones = jnp.ones(len(units))
    X = jnp.asarray(domain.X_val, dtype=jnp.float64)
    optima = jnp.asarray(domain.optima, dtype=jnp.float64)
    _, picks = domain.evaluate_lengths(
        layout, np.asarray(genome, dtype=float)[None, :],
        domain.X_val)
    picks_j = jnp.asarray(picks[0])

    def attr_one(x, pk, opt):
        grad = jax.grad(
            lambda gates: soft_forward(
                layout, genome_j, x, gates, pk, tau))(ones)
        return -grad / opt

    A = jax.vmap(attr_one)(X, picks_j, optima)
    return units, np.asarray(A).T


def soft_lengths(domain, layout, genome, tau=1.0):
    """Anchored soft tour lengths on the validation
    set (numpy (n_val,)), for surrogate checks."""
    genome_j = jnp.asarray(np.asarray(genome, dtype=float),
                           dtype=jnp.float64)
    ones = jnp.ones(len(be.silence_units(layout)))
    X = jnp.asarray(domain.X_val, dtype=jnp.float64)
    _, picks = domain.evaluate_lengths(
        layout, np.asarray(genome, dtype=float)[None, :],
        domain.X_val)
    picks_j = jnp.asarray(picks[0])
    L = jax.vmap(lambda x, pk: soft_forward(
        layout, genome_j, x, ones, pk, tau))(X, picks_j)
    return np.asarray(L)
