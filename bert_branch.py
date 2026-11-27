#!/usr/bin/env python3
"""Branch-level operations for the BERT evolution engine
(sponsor design, 2026-11-23).

A BRANCH is a connected chain of useful units carrying
information from entry to exit -- not a loose neuron.
Roman's isolation method: feed information in and see
which units light up together. Operationally (stated so
it can be judged): two units belong to the same branch
when their per-problem profiles track together (they
matter on the same problems); branches are spectral
clusters of those profiles, and a cluster only counts
as a branch when silencing the WHOLE cluster together
costs real score (the group-silencing proof). With one
block, branches live inside the block (heads + the
feed-forward units they feed); once the lineage grows
blocks, the whole block is the natural branch.

ISOLATION SIGNAL: silencing-drop profiles. They
failed a random-partition control on a dense champion
(indistinguishable from random slices), so clustered
branches are UNPROVEN until a control separates them;
silencing remains in force where it is well-posed:
the per-unit table behind freeze marks and cleared
slots, and the group-silencing proof.

BACK-PROPAGATION ATTRIBUTION: built 2026-11-24
(bert_attribution.py; it passed its random control
where silencing failed), then DROPPED the same day
by sponsor decision: "drop back propagation, it's
not going to work." Parked as a desperation fallback
only; find_branches does not use it. The standing
direction for branch investigation is ANCESTRY --
the recorded genome and lineage history -- plus
other recorded-genome methods.

A GRAFT copies a whole donor branch into space cleared
in the recipient: harmful units are removed first, then
neutral ones (the sponsor's clearing order); the
champion's own proven branch stays in place, frozen,
beside the graft. Donor and recipient descend from the
same champion, so they share the normalization baseline;
the graft's genes are copied whole (every gene of every
unit in the branch), and the post-graft evolution phase
adapts the branch to its new home.
"""
import numpy as np

import bert_engine as be

N_BRANCHES = 3


# ---------------------------------------------------------------
# profiles: what lights up together
# ---------------------------------------------------------------

def unit_drop_profiles(domain, layout, genome):
    """Per-unit, per-problem silencing drops. Row i is
    unit i's drop on each validation problem (silenced
    gap minus champion gap); units that matter on the
    same problems have tracking rows. Requires a domain
    with evaluate_lengths / X_val / optima (the TSP
    domain's interface). Returns (units, profiles) with
    units from bert_engine.silence_units."""
    units = be.silence_units(layout)
    G = np.tile(np.asarray(genome, dtype=float),
                (len(units) + 1, 1))
    for k, u in enumerate(units, start=1):
        G[k, be._silence_indices(layout, u)] = 0.0
    lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
    gaps = lens / domain.optima - 1.0
    return units, gaps[1:] - gaps[0]


def _kmeans(X, k, rng, restarts=60):
    best = None
    for _ in range(restarts):
        idx = [rng.integers(len(X))]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None, :] - X[idx][None]) ** 2)
                        .sum(-1), axis=1)
            tot = d2.sum()
            idx.append(rng.choice(len(X), p=d2 / tot)
                       if tot > 0 else rng.integers(len(X)))
        C = X[idx].copy()
        for _ in range(100):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2)
                            .sum(-1), axis=1)
            new = np.array([X[lab == j].mean(0)
                            if (lab == j).any() else C[j]
                            for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        sse = float(((X - C[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lab, C)
    return best[1]


def spectral_clusters(profiles, k=N_BRANCHES, knn=3, seed=7):
    """Cluster units by the cosine affinity of their
    drop profiles (spectral embedding + k-means)."""
    n = len(profiles)
    P = profiles / (np.linalg.norm(profiles, axis=1,
                                   keepdims=True) + 1e-12)
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
    return _kmeans(U, k, rng)


# ---------------------------------------------------------------
# branch proof: group silencing
# ---------------------------------------------------------------

def group_drop(domain, layout, genome, unit_names):
    """Cost of silencing a whole set of units together,
    in score points and as a relative % of the genome's
    gap (score is negative gap for the TSP domain)."""
    champ = domain.validation_score(layout, genome)
    g = np.asarray(genome, dtype=float).copy()
    by_name = {u["name"]: u for u in be.silence_units(layout)}
    for n in unit_names:
        g[be._silence_indices(layout, by_name[n])] = 0.0
    sil = domain.validation_score(layout, g)
    gap = -champ
    return {"drop_points": float(champ - sil),
            "drop_rel_pct": float(100.0 * (champ - sil) / gap)
            if gap > 0 else None}


def find_branches(domain, layout, genome, k=N_BRANCHES,
                  signal="silencing", tau=0.5):
    """Isolate the genome's branches: spectral clusters
    of the per-problem profiles, each proved by group
    silencing. signal="silencing" (the default)
    clusters silencing-drop profiles.
    signal="attribution" clusters back-propagation
    attribution profiles, centered per problem -- it
    exists in the code but is PARKED by sponsor
    decision (2026-11-24) and is not the standing
    method. Returns branch dicts sorted by group drop
    (largest first): {branch_id, units (table rows
    with drops), n_units, group_drop_points,
    group_drop_rel_pct}."""
    table = be.silence_table(
        layout, genome,
        lambda G: domain.validation_scores(layout, G),
        zero_tol=domain.silence_zero_tol)
    if signal == "attribution":
        import bert_attribution as ba
        units, profiles = ba.attribution_profiles(
            domain, layout, genome, tau=tau)
        profiles = profiles - profiles.mean(
            axis=0, keepdims=True)
    elif signal == "silencing":
        units, profiles = unit_drop_profiles(domain, layout,
                                             genome)
    else:
        raise ValueError(f"unknown signal {signal!r}")
    drops = {u["name"]: u["drop"] for u in table["units"]}
    lab = spectral_clusters(profiles, k=k)
    branches = []
    for b in range(k):
        members = [units[i] for i in range(len(units))
                   if lab[i] == b]
        if not members:
            continue
        rows = [{**u, "drop": drops[u["name"]]}
                for u in members]
        gd = group_drop(domain, layout, genome,
                        [u["name"] for u in members])
        branches.append({
            "branch_id": int(b), "units": rows,
            "n_units": len(members),
            "group_drop_points": gd["drop_points"],
            "group_drop_rel_pct": gd["drop_rel_pct"]})
    branches.sort(key=lambda x: (x["group_drop_rel_pct"]
                                 is None,
                                 -(x["group_drop_rel_pct"]
                                   or 0)))
    return branches


# ---------------------------------------------------------------
# grafting: whole branch into cleared space
# ---------------------------------------------------------------

def cleared_slots(table_units, zero_tol=0.0):
    """The sponsor's clearing order: harmful units first
    (negative drop, most harmful first), then neutral
    ones (drop at or below the zero tolerance). Proven
    units are never slots. Returns table rows."""
    slots = [u for u in table_units if u["drop"] <= zero_tol]
    slots.sort(key=lambda u: u["drop"])
    return slots


def graft_branch(layout, base_genome, donor_genome, branch,
                 slots):
    """Copy a whole donor branch into cleared recipient
    slots. Pairing is by kind AND layer position
    (feed-forward into feed-forward, head into head, in
    the same couple and block the unit came from -- the
    sponsor's graft rule): the donor's strongest branch
    unit (largest drop in its own network) goes into the
    worst cleared slot of its group. Every gene of each
    grafted unit is copied; all other genes of the base
    are untouched. `branch` is a donor branch dict from
    find_branches (units carry donor drops); `slots` are
    recipient table rows from cleared_slots. Returns
    (new_genome, pairs, ungrafted_unit_names)."""
    by_name = {u["name"]: u
               for u in be.silence_units(layout)}
    out = np.asarray(base_genome, dtype=float).copy()
    donor = np.asarray(donor_genome, dtype=float)
    pairs, used = [], set()
    groups = {(u["kind"], u["couple"], u["block"])
              for u in branch["units"]}
    for kind, cpl, blk in sorted(groups):
        b_units = sorted(
            (u for u in branch["units"]
             if (u["kind"], u["couple"], u["block"])
             == (kind, cpl, blk)),
            key=lambda u: -u["drop"])
        s_units = [u for u in slots
                   if (u["kind"], u["couple"], u["block"])
                   == (kind, cpl, blk)]
        for bu, su in zip(b_units, s_units):
            di = be.unit_gene_indices(layout, by_name[bu["name"]])
            si = be.unit_gene_indices(layout, by_name[su["name"]])
            out[si] = donor[di]
            pairs.append((bu["name"], su["name"]))
            used.add(bu["name"])
    ungrafted = [u["name"] for u in branch["units"]
                 if u["name"] not in used]
    return out, pairs, ungrafted
