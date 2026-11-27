#!/usr/bin/env python3
"""Split a champion's units into 3 groups from the data, two ways.

Method 1: k-means (k=3) on each unit's relative silencing drop
(drop as a fraction of the champion's mean gap).
Method 2: spectral clustering (k=3) on each unit's per-problem
drop PROFILE (the 200-vector of per-problem gap changes when the
unit is silenced): cosine affinity on a symmetrised kNN graph,
symmetric normalised Laplacian, first eigenvectors, k-means.

Usage: python3 clustering_units_2026-11-23.py
Writes clustering_units_2026-11-23.json and prints the groups.
"""
import json, dataclasses
import numpy as np
import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend


def kmeans(X, k, rng, restarts=60):
    best = None
    for _ in range(restarts):
        idx = [rng.integers(len(X))]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None, :] - X[idx][None]) ** 2).sum(-1),
                        axis=1)
            tot = d2.sum()
            idx.append(rng.choice(len(X), p=d2 / tot) if tot > 0
                       else rng.integers(len(X)))
        C = X[idx].copy()
        for _ in range(100):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1),
                            axis=1)
            new = np.array([X[lab == j].mean(0) if (lab == j).any()
                            else C[j] for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        sse = float(((X - C[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lab, C)
    return best[1], best[2]


def spectral_groups(profiles, k=3, knn=3, seed=7):
    n = len(profiles)
    P = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True)
                    + 1e-12)
    S = np.clip(P @ P.T, 0.0, None)
    np.fill_diagonal(S, 0.0)
    W = np.zeros_like(S)
    for i in range(n):
        nn = np.argsort(-S[i])[:knn]
        W[i, nn] = S[i, nn]
    W = np.maximum(W, W.T)
    d = W.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d + 1e-12))
    L = np.eye(n) - Dm @ W @ Dm
    vals, vecs = np.linalg.eigh(L)
    U = vecs[:, 1:k + 1]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    lab, _ = kmeans(U, k, rng)
    return lab


def unit_profiles(cfg_raw, genome, arch=None):
    fields = {f.name for f in dataclasses.fields(be.EngineConfig)}
    cfg = be.EngineConfig(**{k: v for k, v in cfg_raw.items()
                             if k in fields})
    backend = get_backend(cfg.backend, cfg.device, cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)
    if arch is None:
        arch = be.initial_arch(cfg, domain.input_dim)
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    g = np.asarray(genome, dtype=float)
    units = be.silence_units(layout)
    G = np.tile(g, (len(units) + 1, 1))
    for k, u in enumerate(units, start=1):
        G[k, be._silence_indices(layout, u)] = 0.0
    lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
    gaps = lens / domain.optima - 1.0
    champ_gap = float(gaps[0].mean())
    assert abs(-champ_gap - domain.validation_score(layout, g)) < 1e-9
    profiles = gaps[1:] - gaps[0]           # per-problem drop parts
    drops = profiles.mean(1)
    rel = drops / champ_gap                 # fraction of the gap
    return units, rel, profiles, champ_gap


def analyse(tag, cfg_raw, genome, arch=None):
    units, rel, profiles, champ_gap = unit_profiles(
        cfg_raw, genome, arch)
    rng = np.random.default_rng(7)
    lab1, _ = kmeans(rel[:, None], 3, rng)
    lab2 = spectral_groups(profiles, seed=7)
    out = {"champion_gap_pct": round(100 * champ_gap, 2), "units": []}
    for i, u in enumerate(units):
        out["units"].append({
            "name": u["name"], "rel_drop_pct": round(100 * rel[i], 1),
            "kmeans_group": int(lab1[i]),
            "spectral_group": int(lab2[i])})
    print(f"\n== {tag}: champion gap {100 * champ_gap:.2f}%, "
          f"{len(units)} units")
    for method in ("kmeans_group", "spectral_group"):
        print(f"-- {method}")
        for grp in range(3):
            mem = [out["units"][i] for i in range(len(units))
                   if out["units"][i][method] == grp]
            if not mem:
                continue
            mean = np.mean([m["rel_drop_pct"] for m in mem])
            names = ", ".join(
                f"{m['name'].split('.')[-1]}({m['rel_drop_pct']:+.0f}%)"
                for m in mem)
            print(f"   group {grp}: mean {mean:+.1f}%  [{names}]")
    return out


res = {}
cfg_d2 = json.load(open("bert_engine_config_d2_l1.json"))
g_d2 = np.load("engine_runs/fresh_d2_l1/champion.npz")["genome"]
res["d2_single_layer"] = analyse("d2 single-layer seed", cfg_d2, g_d2)

cfg_t = json.load(open("bert_engine_config_test.json"))
g_big = np.load("optimize_control_2026-11-23_champion.npz")["genome"]
arch_big = be.ArchSpec(input_dim=2, d=12, ffn_hidden=16,
                       couples=((2, 2),), attn_scale=1.0 / np.sqrt(8))
res["d12_converged"] = analyse("d12 converged champion", cfg_t,
                               g_big, arch_big)

json.dump(res, open("clustering_units_2026-11-23.json", "w"), indent=1)
print("\nwrote clustering_units_2026-11-23.json")
