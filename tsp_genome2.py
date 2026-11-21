"""Height-parameterized TSP encoder-decoder genome for PILOT 2
(Epic P1.3 follow-up, sponsor diagnostic mandate 2026-11-15).

Identical architecture and gene layout to tsp_genome.py (pilot 1),
but with hidden width H as a parameter, so Arm C (H = 16) uses the
same code path as the H = 8 arms. Gene order is exactly pilot 1's:

  scorer   sW1 (2H), sb1 (H), sW2 (H), sb2 (1)        -> 4H + 1
  encoder  eWx (2H), eWh (H^2), eb (H)                -> H^2 + 3H
  decoder  dWxd ((H+3)H), dWhd (H^2), dbd (H)         -> 2H^2 + 4H
  pointer  pWk (2H), pbk (H), pv (H)                  -> 4H

H = 8:  33 + 88 + 160 + 32 = 313 (pilot 1, verified identical).
H = 16: 65 + 304 + 576 + 64 = 1009.

Components (for the Arm E ablation and displacement reads):
  scorer  = genes [0, 4H+1)
  encoder = next H^2 + 3H genes
  decoder = next 2H^2 + 4H genes
  pointer = last 4H genes
"""

from __future__ import annotations

import numpy as np

COMPONENTS = ("scorer", "encoder", "decoder", "pointer")


def make_layout(H):
    H = int(H)
    dec_in = 2 + H + 1
    sizes = [
        ("sW1", 2 * H), ("sb1", H), ("sW2", H), ("sb2", 1),
        ("eWx", 2 * H), ("eWh", H * H), ("eb", H),
        ("dWxd", dec_in * H), ("dWhd", H * H), ("dbd", H),
        ("pWk", 2 * H), ("pbk", H), ("pv", H),
    ]
    slices, off = {}, 0
    for name, n in sizes:
        slices[name] = slice(off, off + n)
        off += n
    comp = {
        "scorer": slice(0, 4 * H + 1),
        "encoder": slice(4 * H + 1, 4 * H + 1 + H * H + 3 * H),
        "decoder": slice(4 * H + 1 + H * H + 3 * H,
                         4 * H + 1 + H * H + 3 * H + 2 * H * H + 4 * H),
        "pointer": slice(off - 4 * H, off),
    }
    return {"H": H, "dec_in": dec_in, "slices": slices,
            "components": comp, "n_params": off}


def unpack(vec, lay):
    H = lay["H"]
    S = lay["slices"]
    v = np.asarray(vec, dtype=float)
    assert v.shape == (lay["n_params"],), \
        f"genome must be ({lay['n_params']},)"
    return {
        "sW1": v[S["sW1"]].reshape(2, H), "sb1": v[S["sb1"]],
        "sW2": v[S["sW2"]], "sb2": v[S["sb2"]],
        "eWx": v[S["eWx"]].reshape(2, H),
        "eWh": v[S["eWh"]].reshape(H, H), "eb": v[S["eb"]],
        "dWxd": v[S["dWxd"]].reshape(lay["dec_in"], H),
        "dWhd": v[S["dWhd"]].reshape(H, H), "dbd": v[S["dbd"]],
        "pWk": v[S["pWk"]].reshape(2, H), "pbk": v[S["pbk"]],
        "pv": v[S["pv"]],
    }


def scorer_scores(P, X):
    h = np.tanh(X @ P["sW1"] + P["sb1"])
    return h @ P["sW2"] + P["sb2"]


def ranked_order(P, X):
    s = scorer_scores(P, X)
    return np.argsort(-s, axis=1, kind="stable")


def encode(P, seq):
    B = seq.shape[0]
    H = P["eb"].shape[0]
    h = np.zeros((B, H))
    for t in range(seq.shape[1]):
        h = np.tanh(seq[:, t, :] @ P["eWx"] + h @ P["eWh"] + P["eb"])
    return h


def city_keys(P, X):
    return X @ P["pWk"] + P["pbk"]


def decode_greedy(P, X, ctx):
    B, n, _ = X.shape
    H = ctx.shape[1]
    k = city_keys(P, X)
    d = ctx.copy()
    prev = np.zeros((B, 2))
    visited = np.zeros((B, n), dtype=bool)
    picks = np.empty((B, n), dtype=int)
    ar = np.arange(B)
    for t in range(n):
        frac = np.full((B, 1), t / n)
        inp = np.concatenate([prev, ctx, frac], axis=1)
        d = np.tanh(inp @ P["dWxd"] + d @ P["dWhd"] + P["dbd"])
        logits = np.tanh(k + d[:, None, :]) @ P["pv"]
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        picks[:, t] = pick
        visited[ar, pick] = True
        prev = X[ar, pick]
    return picks


def decode_with_diag(P, X, ctx):
    """Greedy decode identical to decode_greedy, plus per-step
    diagnostics: for steps t >= 1, whether the picked city is the
    nearest unvisited city to the previously picked city.
    Returns picks (B,n), nearest_hit (B,n) bool (col 0 = False,
    excluded by callers)."""
    B, n, _ = X.shape
    k = city_keys(P, X)
    d = ctx.copy()
    prev = np.zeros((B, 2))
    visited = np.zeros((B, n), dtype=bool)
    picks = np.empty((B, n), dtype=int)
    nearest_hit = np.zeros((B, n), dtype=bool)
    ar = np.arange(B)
    for t in range(n):
        frac = np.full((B, 1), t / n)
        inp = np.concatenate([prev, ctx, frac], axis=1)
        d = np.tanh(inp @ P["dWxd"] + d @ P["dWhd"] + P["dbd"])
        logits = np.tanh(k + d[:, None, :]) @ P["pv"]
        logits = np.where(visited, -1e18, logits)
        pick = np.argmax(logits, axis=1)
        if t >= 1:
            dist = np.sqrt(((X - prev[:, None, :]) ** 2)
                           .sum(axis=2))
            dist = np.where(visited, np.inf, dist)
            nearest_hit[:, t] = (np.argmin(dist, axis=1) == pick)
        picks[:, t] = pick
        visited[ar, pick] = True
        prev = X[ar, pick]
    return picks, nearest_hit


def tour_lengths(X, picks):
    B = X.shape[0]
    ar = np.arange(B)
    pts = X[ar[:, None], picks]
    nxt = np.roll(pts, -1, axis=1)
    return np.sqrt(((pts - nxt) ** 2).sum(axis=2)).sum(axis=1)


def policy_tour(P, X, seq_perm=None):
    R = ranked_order(P, X)
    B = X.shape[0]
    ar = np.arange(B)
    seq = X[ar[:, None], R]
    if seq_perm is not None:
        seq = seq[:, seq_perm, :]
    ctx = encode(P, seq)
    return decode_greedy(P, X, ctx)


def canonical_cycle(tour):
    t = [int(c) for c in tour]
    i = int(np.argmin(t))
    rot = t[i:] + t[:i]
    rev = [rot[0]] + rot[1:][::-1]
    return tuple(min(rot, rev))


def count_crossings(coords, tour):
    """Number of properly intersecting edge pairs in the closed
    tour. An optimal Euclidean tour has zero."""
    n = len(tour)
    pts = coords[np.asarray(tour)]

    def seg_intersect(a, b, c, d):
        def orient(p, q, r):
            val = (q[0] - p[0]) * (r[1] - p[1]) \
                - (q[1] - p[1]) * (r[0] - p[0])
            return val

        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    cnt = 0
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent (wrap) edges share a vertex
            c, d = pts[j], pts[(j + 1) % n]
            if seg_intersect(a, b, c, d):
                cnt += 1
    return cnt


def held_karp(coords):
    n = coords.shape[0]
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2)
                .sum(axis=2))
    INF = np.inf
    dp = np.full((1 << n, n), INF)
    dp[1, 0] = 0.0
    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for j in range(n):
            cur = dp[mask, j]
            if cur == INF or not (mask >> j) & 1:
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                val = cur + D[j, k]
                if val < dp[nm, k]:
                    dp[nm, k] = val
    full = (1 << n) - 1
    return float(np.min(dp[full, 1:] + D[1:, 0])) if n > 1 else 0.0
