"""Full-population genome archive + per-weight usefulness scoring
(Addendum 2 of PLAN_SILENCE_SPLIT_MARK_2026-11-22.md, plus the
Addendum 1 per-gene signals).

What this module owns:
  - incremental per-generation archive records (one .npz per
    generation, atomic via ``bert_engine.atomic_npz``, never
    held in memory), with a per-stage byte cap;
  - per-weight usefulness scores: every weight starts at
    exactly 1 (neutral), stored as a log score (additive),
    displayed as the score itself (exp of the log score);
  - post-hoc analysis of a finished run: trace the winning
    species' lineage back through parent pointers, recompute
    its per-generation usefulness vectors, keep the lineage
    in full, dispose of the rest, and render a self-contained
    usefulness heatmap (HTML/SVG, no external deps).

Performance measure (Roman's rule, 2026-11-22): RELATIVE
standing only, never the raw score. Within one species in
one generation, ``rel = score - species_median``. Raw scores
are not comparable across generations because the training
samples rotate. Rank is the allowed alternative; this module
uses score-minus-median.

Evidence per gene per generation (three signals, Addendum 1):
  - gene-to-score correlation across members (Pearson of gene
    values vs ``rel``);
  - elite agreement: std of the gene among the elites (top
    max(2, ceil(25% of members)) by ``rel``) divided by std
    among all members;
  - elite shift: |mean_elite - mean_all| / (std_all + eps).
Genes with no variation across the species that generation
(frozen base genes in phase A, draw genes) produce NO
evidence and stay neutral (log-step exactly 0) for that
generation.

Evidence -> log-step mappings (PROVISIONAL, to be calibrated
from the first real archive; see ``update_log_scores``):
  - correlation: ``clip(2*|corr| - 0.5, -0.5, 1.0)`` -- a gene
    that varies but separates nothing reads mildly useless;
    a strong |corr| reads useful;
  - elite spread: ``clip(1 - ratio, 0, 1)`` where ratio =
    std_elite/std_all -- tight agreement reads useful, loose
    (>= 1) reads NO evidence (excluded from the mean, never
    negative);
  - elite shift: ``clip(shift, 0, 1)`` -- a large shift reads
    useful, no shift reads NO evidence (excluded).
Per-generation log-step = ``usefulness_step`` (default 0.02)
times the mean evidence over the signals that HAVE evidence;
if none have evidence the log-step is exactly 0.

Reading cuts for density (PROVISIONAL, same caveat):
log-score > +0.05 useful, < -0.05 useless, == 0.0 exactly
never judged, anything else (nonzero but within +/-0.05)
"weakly read". ``density_shares`` reports all four shares.

The runner (run_bert_engine.py) imports this module; this
module therefore never imports run_bert_engine (no circular
import). It imports bert_engine (atomic_npz / atomic_json)
and, inside analyse_run only, the domain modules needed to
rebuild the final layout for the heatmap.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be

# Plan defaults. The size cap is read from the EngineConfig
# passed to the recording functions (field
# ``archive_cap_bytes_per_stage``, added by the core build);
# this constant is only the fallback when no config (or no
# field) is supplied, so a smoke config without the field
# still behaves per the plan.
DEFAULT_ARCHIVE_CAP_BYTES_PER_STAGE = 16 * 1024 ** 3  # 16 GiB
DEFAULT_USEFULNESS_STEP = 0.02
# Reading cuts (provisional; see module docstring).
DENSITY_CUT = 0.05
# Colour clip for the heatmap (log-score units).
HEATMAP_CLIP = 1.0
# Max embedded per-generation series JSON in heatmap.html.
HEATMAP_EMBED_LIMIT_BYTES = 1_000_000

EPS = 1e-12
# A gene counts as varying across the species only if its
# std exceeds this absolute tolerance. Exact-zero variance
# is the ideal, but in float64 the std of a constant column
# is ~1e-16 dust, not 0.0, so a bare `> 0` test would read
# evidence into a frozen gene. 1e-12 is far below any real
# gene variation (genes are O(0.1-1)) and far above the dust.
VAR_TOL = 1e-12


# ================================================================
# Archive writer / reader
# ================================================================

def archive_dir(out_dir):
    return Path(out_dir) / "archive"


def _record_path(out_dir, stage, total_gens):
    return archive_dir(out_dir) / (
        f"stage{int(stage):03d}_gen{int(total_gens):06d}.npz")


def archive_bytes_used(out_dir, stage):
    """Bytes of archive records already on disk for `stage`
    (measured file sizes, so a resumed run sees the true
    total)."""
    total = 0
    d = archive_dir(out_dir)
    if not d.is_dir():
        return 0
    for p in d.glob(f"stage{int(stage):03d}_gen*.npz"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _cap_bytes(cfg):
    if cfg is not None:
        v = getattr(cfg, "archive_cap_bytes_per_stage", None)
        if v is not None:
            return int(v)
    return DEFAULT_ARCHIVE_CAP_BYTES_PER_STAGE


def _usefulness_step(cfg):
    if cfg is not None:
        v = getattr(cfg, "usefulness_step", None)
        if v is not None:
            return float(v)
    return DEFAULT_USEFULNESS_STEP


def record_generation(out_dir, stage, phase, gen_in_phase,
                      total_gens, genomes, scores, species_ids,
                      lineages, parent_species_ids, cfg=None,
                      entry_shapes=None):
    """Write one per-generation archive record, atomically.

    Returns a dict: {written, capped, path (str or None),
    bytes_used_stage, record_bytes}. When the next record
    would exceed the per-stage cap, NOTHING is written and
    ``capped`` is True (the plan leaves the past-cap fallback
    undecided; stopping + logging is the specified behaviour,
    never a partial-record scheme). ``bytes_used_stage`` is
    measured from the files on disk for the stage.
    ``entry_shapes`` (optional) is the run's ordered
    [(entry name, shape), ...] for this generation's layout;
    the analysis needs it to rebase the usefulness vectors
    across a growth step (gene positions shift when the
    layout grows, entry names do not).
    """
    genomes = np.asarray(genomes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    species_ids = np.asarray(species_ids, dtype=np.int64)
    parent_species_ids = np.asarray(parent_species_ids,
                                    dtype=np.int64)
    # unicode, not object: an object array needs pickle to
    # load, and the reader loads with allow_pickle=False
    lineages = np.asarray(lineages, dtype=str)
    n = len(scores)
    if genomes.shape[0] != n or len(species_ids) != n \
            or len(lineages) != n \
            or len(parent_species_ids) != n:
        raise ValueError(
            "archive record arrays disagree on population size")
    d = archive_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    used = archive_bytes_used(out_dir, stage)
    # Estimated record size: the arrays' bytes + the lineage
    # strings' bytes + a small overhead allowance for the
    # npz container and the scalar fields.
    lineage_bytes = sum(len(str(s).encode("utf-8"))
                        for s in lineages)
    estimate = int(genomes.nbytes + scores.nbytes
                   + species_ids.nbytes
                   + parent_species_ids.nbytes
                   + lineage_bytes + 8192)
    cap = _cap_bytes(cfg)
    if used + estimate > cap:
        return {"written": False, "capped": True, "path": None,
                "bytes_used_stage": int(used),
                "record_bytes": int(estimate)}
    path = _record_path(out_dir, stage, total_gens)
    extra = {}
    if entry_shapes is not None:
        extra["entry_names"] = np.array(
            [str(n) for n, _ in entry_shapes])
        extra["entry_shapes"] = np.array(json.dumps(
            [list(sh) for _, sh in entry_shapes]))
    be.atomic_npz(
        path, genomes=genomes, scores=scores,
        species_ids=species_ids, lineages=lineages,
        parent_species_ids=parent_species_ids,
        stage=np.int64(stage), phase=np.array(str(phase)),
        gen_in_phase=np.int64(gen_in_phase),
        total_gens=np.int64(total_gens), **extra)
    used = archive_bytes_used(out_dir, stage)
    return {"written": True, "capped": False,
            "path": str(path), "bytes_used_stage": int(used),
            "record_bytes": int(path.stat().st_size)}


def record_species_generation(out_dir, stage, phase,
                              gen_in_phase, total_gens, species,
                              cfg=None, entry_shapes=None):
    """Convenience wrapper for the runner: `species` is the
    runner's species list (each with ``pop``, ``lens``,
    ``id``, ``lineage`` and ``parent_species_id``; the last
    defaults to -1 when the hook has not set it).
    ``entry_shapes`` is the layout's ordered [(name, shape),
    ...], passed through to the record."""
    genomes, scores, sids, lins, parents = [], [], [], [], []
    for sp in species:
        pop = np.asarray(sp["pop"], dtype=np.float64)
        lens = sp.get("lens")
        if lens is None:
            raise ValueError(
                f"species {sp.get('id')} has no 'lens' scores; "
                "record after the fold-back in generation()")
        lens = np.asarray(lens, dtype=np.float64)
        if len(pop) != len(lens):
            raise ValueError(
                f"species {sp.get('id')}: pop/score length "
                f"mismatch ({len(pop)} vs {len(lens)})")
        genomes.append(pop)
        scores.append(lens)
        sids.append(np.full(len(pop), int(sp["id"]),
                            dtype=np.int64))
        lins.append(np.full(len(pop), str(sp["lineage"])))
        parents.append(np.full(
            len(pop), int(sp.get("parent_species_id", -1)),
            dtype=np.int64))
    return record_generation(
        out_dir, stage, phase, gen_in_phase, total_gens,
        np.concatenate(genomes) if genomes
        else np.zeros((0, 0)), np.concatenate(scores),
        np.concatenate(sids),
        np.concatenate(lins) if lins else np.zeros(0),
        np.concatenate(parents), cfg=cfg,
        entry_shapes=entry_shapes)


def load_entry_shapes(z):
    """The record's [(entry name, shape), ...] as stored by
    record_generation, or None when the record carries none
    (records written before shapes were stored)."""
    if "entry_names" not in z.files \
            or "entry_shapes" not in z.files:
        return None
    try:
        names = [str(n) for n in np.asarray(z["entry_names"])]
        shapes = json.loads(str(z["entry_shapes"].item()))
        return [(n, tuple(int(x) for x in sh))
                for n, sh in zip(names, shapes)]
    except Exception:  # noqa: BLE001 - malformed shapes
        return None


def _parse_record_filename(name):
    # stage{stage:03d}_gen{total_gens:06d}.npz
    try:
        stem = name[:-4] if name.endswith(".npz") else name
        s_part, g_part = stem.split("_")
        return int(s_part[len("stage"):]), \
            int(g_part[len("gen"):])
    except (ValueError, IndexError):
        return None


def iter_archive_records(archive_dir_path):
    """Yield archive records in (stage, total_gens) order.

    Each yielded dict holds the stored arrays plus the scalar
    fields (stage, phase, gen_in_phase, total_gens). A kill
    mid-write leaves at most the tmp file, never a partial
    record (atomic_npz); an unreadable file is still skipped
    with a warning, never a crash.
    """
    d = Path(archive_dir_path)
    files = []
    if d.is_dir():
        for p in d.glob("stage*_gen*.npz"):
            parsed = _parse_record_filename(p.name)
            if parsed is not None:
                files.append((parsed, p))
    files.sort(key=lambda t: (t[0][0], t[0][1], t[1].name))
    for (stage, total_gens), p in files:
        try:
            z = np.load(p, allow_pickle=False)
            phase = z["phase"]
            if isinstance(phase, np.ndarray):
                phase = phase.item()
            if isinstance(phase, bytes):
                phase = phase.decode("utf-8")
            rec = {
                "genomes": np.asarray(z["genomes"],
                                      dtype=np.float64),
                "scores": np.asarray(z["scores"],
                                     dtype=np.float64),
                "species_ids": np.asarray(z["species_ids"],
                                          dtype=np.int64),
                "lineages": np.asarray(z["lineages"]),
                "parent_species_ids": np.asarray(
                    z["parent_species_ids"], dtype=np.int64)
                if "parent_species_ids" in z.files
                else np.full(len(z["scores"]), -1,
                             dtype=np.int64),
                "entry_shapes": load_entry_shapes(z),
                "stage": int(z["stage"]),
                "phase": str(phase),
                "gen_in_phase": int(z["gen_in_phase"]),
                "total_gens": int(z["total_gens"]),
                "path": str(p),
            }
        except Exception as exc:  # noqa: BLE001 - skip, warn
            print(f"[archive] WARNING: skipping unreadable "
                  f"record {p}: {exc}", file=sys.stderr,
                  flush=True)
            continue
        yield rec


# ================================================================
# Per-weight usefulness scoring
# ================================================================

def update_log_scores(log_scores, genomes, scores,
                     usefulness_step=DEFAULT_USEFULNESS_STEP):
    """One generation's log-score update for ONE species.

    Pure function (single implementation shared by the online
    tracker and the post-hoc analysis). ``log_scores`` is the
    species' current (n_params,) log-score vector; returns a
    NEW vector. Genes with no variation across the species
    this generation get a log-step of exactly 0.0.
    """
    log_scores = np.asarray(log_scores, dtype=np.float64)
    G = np.asarray(genomes, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if G.ndim != 2 or G.shape[1] != log_scores.shape[0]:
        raise ValueError("genomes/log_scores shape mismatch")
    P = G.shape[0]
    if P == 0:
        return log_scores.copy()
    rel = s - np.median(s)
    k = min(P, max(2, int(math.ceil(0.25 * P))))
    elite_idx = np.argsort(-rel, kind="stable")[:k]
    mean_all = G.mean(axis=0)
    std_all = G.std(axis=0)
    mean_elite = G[elite_idx].mean(axis=0)
    std_elite = G[elite_idx].std(axis=0)
    varies = std_all > VAR_TOL
    n_params = G.shape[1]

    # --- signal 1: gene-to-score correlation (Pearson vs rel)
    corr_ev = np.full(n_params, np.nan)
    rel_c = rel - rel.mean()
    denom_rel = math.sqrt(float(np.sum(rel_c ** 2)))
    if denom_rel > 0:
        Gc = G - mean_all
        denom = np.sqrt(np.sum(Gc ** 2, axis=0)) * denom_rel
        valid = (denom > 0) & varies
        if np.any(valid):
            corr = np.zeros(n_params)
            corr[valid] = (Gc.T @ rel_c)[valid] / denom[valid]
            corr_ev[valid] = np.clip(
                2.0 * np.abs(corr[valid]) - 0.5, -0.5, 1.0)
    # --- signal 2: elite agreement (spread ratio)
    ratio = np.full(n_params, np.nan)
    ratio[varies] = std_elite[varies] / std_all[varies]
    spread_ev = np.full(n_params, np.nan)
    tight = varies & (ratio < 1.0)
    spread_ev[tight] = np.clip(1.0 - ratio[tight], 0.0, 1.0)
    # ratio >= 1 (loose) reads NO evidence: left NaN.
    # --- signal 3: elite shift
    shift = np.full(n_params, np.nan)
    shift[varies] = np.abs(mean_elite[varies]
                           - mean_all[varies]) \
        / (std_all[varies] + EPS)
    shift_ev = np.full(n_params, np.nan)
    moved = varies & (shift > 0.0)
    shift_ev[moved] = np.clip(shift[moved], 0.0, 1.0)
    # shift == 0 reads NO evidence: left NaN.
    stack = np.vstack([corr_ev, spread_ev, shift_ev])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_ev = np.nanmean(stack, axis=0)
    step = np.where(np.isnan(mean_ev), 0.0,
                    float(usefulness_step) * mean_ev)
    return log_scores + step


def density_shares(log_scores):
    """Shares {useful, useless, never_judged, weak} over a
    log-score vector, using the provisional cuts
    |log-score| > 0.05 (useful/useless), == 0.0 exactly
    (never judged), else weak. The four shares sum to 1."""
    ls = np.asarray(log_scores, dtype=np.float64).ravel()
    n = ls.size
    if n == 0:
        return {"useful": 0.0, "useless": 0.0,
                "never_judged": 0.0, "weak": 0.0}
    useful = int(np.sum(ls > DENSITY_CUT))
    useless = int(np.sum(ls < -DENSITY_CUT))
    never = int(np.sum(ls == 0.0))
    weak = int(n - useful - useless - never)
    return {"useful": useful / n, "useless": useless / n,
            "never_judged": never / n, "weak": weak / n}


class UsefulnessTracker:
    """The runner's online per-species log-score state.

    One vector per live species id. Children inherit a COPY
    of their founder's vector at regeneration; culled
    species' vectors are dropped (bounded memory).
    """

    def __init__(self, n_params,
                 usefulness_step=DEFAULT_USEFULNESS_STEP):
        self.n_params = int(n_params)
        self.usefulness_step = float(usefulness_step)
        self._vectors = {}  # species_id (int) -> (n_params,) float64

    def species_ids(self):
        return sorted(self._vectors.keys())

    def update(self, species_id, genomes, scores):
        sid = int(species_id)
        vec = self._vectors.get(sid)
        if vec is None:
            vec = np.zeros(self.n_params, dtype=np.float64)
        self._vectors[sid] = update_log_scores(
            vec, genomes, scores, self.usefulness_step)

    def inherit(self, child_species_id, parent_species_id):
        parent = self._vectors.get(int(parent_species_id))
        if parent is None:
            self._vectors[int(child_species_id)] = np.zeros(
                self.n_params, dtype=np.float64)
        else:
            self._vectors[int(child_species_id)] = parent.copy()

    def remove(self, species_id):
        self._vectors.pop(int(species_id), None)

    def set_vector(self, species_id, vec):
        """Replace one species' log-score vector (runner use
        at a growth rebase: the old vector is remapped into
        the new layout by entry name, so the evidence
        survives the transplant and new genes start at 0)."""
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != (self.n_params,):
            raise ValueError(
                "tracker set_vector: vector has wrong length "
                f"for species {species_id}")
        self._vectors[int(species_id)] = arr.copy()

    def vector(self, species_id):
        vec = self._vectors.get(int(species_id))
        if vec is None:
            return np.zeros(self.n_params, dtype=np.float64)
        return vec.copy()

    def density(self, species_id):
        """{useful, useless, never_judged, weak} shares for
        the species' current vector; an unknown species reads
        as entirely never judged."""
        vec = self._vectors.get(int(species_id))
        if vec is None:
            return {"useful": 0.0, "useless": 0.0,
                    "never_judged": 1.0, "weak": 0.0}
        return density_shares(vec)

    def state_dict(self):
        """JSON-able: {"species_ids": [...],
        "log_scores": [[...], ...]}. Floats go through
        Python float repr, so a JSON round trip is
        bit-identical."""
        ids = sorted(self._vectors.keys())
        return {"species_ids": [int(i) for i in ids],
                "log_scores": [self._vectors[i].tolist()
                               for i in ids]}

    def load_state_dict(self, d):
        self._vectors = {}
        for sid, vec in zip(d["species_ids"],
                            d["log_scores"]):
            arr = np.asarray(vec, dtype=np.float64)
            if arr.shape != (self.n_params,):
                raise ValueError(
                    "tracker state vector has wrong length "
                    f"for species {sid}")
            self._vectors[int(sid)] = arr


# ================================================================
# Heatmap
# ================================================================

def _color_for_log(v):
    """Diverging colour: neutral 1 (log 0) is the midpoint
    (#e8e8e8); log < 0 goes to blue, log > 0 to red; log
    scores are clipped to [-1, +1] for colour."""
    v = max(-HEATMAP_CLIP, min(HEATMAP_CLIP, float(v)))
    if v == 0.0:
        return "#e8e8e8"
    if v < 0:
        t, c0, c1 = -v, (232, 232, 232), (33, 102, 172)
    else:
        t, c0, c1 = v, (232, 232, 232), (178, 24, 43)
    rgb = tuple(round(c0[i] + (c1[i] - c0[i]) * t)
                for i in range(3))
    return "#%02x%02x%02x" % rgb


def _matrix_entries(layout):
    """Weight-matrix entries of a layout, in layout order:
    every entry whose shape is 2-D (attention projections
    W_q/W_k/W_v/W_o per head, feed-forward W_1/W_2, embedding
    W_in, decode-head matrices). Biases / gains (1-D) are not
    panels."""
    return [(name, sl, shape) for name, sl, shape
            in layout.entries if len(shape) == 2]


def build_heatmap_html(layout, final_log_scores,
                       per_gen_vectors, per_gen_totals,
                       out_path, title="Usefulness heatmap"):
    """Write a self-contained heatmap page (inline SVG, no
    external deps, no server). ``per_gen_vectors`` is a list
    of (n_params,) log-score vectors, one per generation, in
    generation order; ``per_gen_totals`` the matching
    total_gens values. The per-generation series is embedded
    as JSON only if it is under 1 MB, else it goes to a
    ``heatmap_series.json`` sidecar beside the page and the
    page says so (the slider is then disabled; the final
    heatmap still renders).

    Returns {path, embedded, n_panels, series_bytes}.
    """
    final_log = np.asarray(final_log_scores, dtype=np.float64)
    matrices = _matrix_entries(layout)
    cell = 10
    parts = []
    parts.append("<!DOCTYPE html>\n<html><head><meta charset="
                 "'utf-8'><title>" + title + "</title><style>"
                 "body{font-family:sans-serif;margin:16px}"
                 "table{border-collapse:collapse}"
                 "td,th{border:1px solid #999;padding:3px 8px;"
                 "text-align:right}th{text-align:left}"
                 ".panel{display:inline-block;vertical-align:top;"
                 "margin:8px;padding:6px;border:1px solid #bbb}"
                 ".panel h3{margin:2px 0 4px 0;font-size:13px}"
                 "</style></head><body>")
    parts.append(f"<h1>{title}</h1>")
    overall = density_shares(final_log)
    parts.append(
        "<p>Final density (all weights): useful "
        f"{overall['useful']:.4f}, useless "
        f"{overall['useless']:.4f}, never judged "
        f"{overall['never_judged']:.4f}, weak "
        f"{overall['weak']:.4f}. Log-score cuts: useful &gt; "
        "+0.05, useless &lt; -0.05, never judged exactly 0. "
        "Colour: blue = judged useless, red = judged useful, "
        "pale grey = neutral 1, clipped to log-score [-1, +1]."
        "</p>")
    # --- per-zone density table
    parts.append("<h2>Density per weight matrix</h2>"
                 "<table id='density'><tr><th>matrix</th>"
                 "<th>shape</th><th>useful</th><th>useless</th>"
                 "<th>never judged</th><th>weak</th></tr>")
    for name, sl, shape in matrices:
        shares = density_shares(final_log[sl])
        parts.append(
            f"<tr class='zone-row'><td>{name}</td>"
            f"<td>{shape[0]}x{shape[1]}</td>"
            f"<td>{shares['useful']:.4f}</td>"
            f"<td>{shares['useless']:.4f}</td>"
            f"<td>{shares['never_judged']:.4f}</td>"
            f"<td>{shares['weak']:.4f}</td></tr>")
    parts.append("</table>")
    # --- generation slider
    n_gen = len(per_gen_vectors)
    series_json = json.dumps(
        [np.asarray(v, dtype=np.float64).tolist()
         for v in per_gen_vectors])
    series_bytes = len(series_json.encode("utf-8"))
    embedded = series_bytes < HEATMAP_EMBED_LIMIT_BYTES
    if not embedded:
        sidecar = Path(out_path).with_name(
            "heatmap_series.json")
        sidecar.write_text(series_json)
    if n_gen > 1:
        parts.append(
            "<h2>Evolution</h2>"
            "<p><label>Generation <input type='range' "
            f"id='genSlider' min='0' max='{n_gen - 1}' "
            f"value='{n_gen - 1}'></label> "
            "<span id='genLabel'></span></p>")
        if not embedded:
            parts.append(
                "<p>Per-generation series is NOT embedded in "
                "this page (it is "
                f"{series_bytes} bytes, over the 1 MB limit); "
                "it is in heatmap_series.json beside this page. "
                "The slider is disabled; the panels below show "
                "the final generation.</p>")
    # --- panels
    parts.append("<h2>Weight matrices</h2>")
    for name, sl, shape in matrices:
        rows, cols = int(shape[0]), int(shape[1])
        idx = np.arange(sl.start, sl.stop).reshape(rows, cols)
        rects = []
        for r in range(rows):
            for c in range(cols):
                gi = int(idx[r, c])
                rects.append(
                    f"<rect x='{c * cell}' y='{r * cell}' "
                    f"width='9' height='9' "
                    f"fill='{_color_for_log(final_log[gi])}' "
                    f"data-i='{gi}'/>")
        parts.append(
            f"<div class='panel' data-matrix='{name}'>"
            f"<h3>{name} {rows}x{cols}</h3>"
            f"<svg width='{cols * cell}' height='{rows * cell}'>"
            + "".join(rects) + "</svg></div>")
    # --- slider script
    parts.append("<script>\nfunction colorFor(v){"
                 "v=Math.max(-1,Math.min(1,v));"
                 "if(v===0)return '#e8e8e8';"
                 "var c0=[232,232,232],"
                 "c1=v<0?[33,102,172]:[178,24,43],t=Math.abs(v);"
                 "var rgb=c0.map(function(a,i){"
                 "return Math.round(a+(c1[i]-a)*t);});"
                 "return '#'+rgb.map(function(x){"
                 "return x.toString(16).padStart(2,'0');"
                 "}).join('');}\n")
    if embedded and n_gen > 1:
        parts.append(
            f"var SERIES={series_json};\n"
            f"var GEN_TOTALS={json.dumps(list(per_gen_totals))};\n"
            "var slider=document.getElementById('genSlider');"
            "var label=document.getElementById('genLabel');"
            "var rects=document.querySelectorAll('svg rect');"
            "function showGen(g){"
            "label.textContent='generation '+GEN_TOTALS[g]"
            "+' ('+(g+1)+' of '+SERIES.length+')';"
            "var vec=SERIES[g];"
            "for(var i=0;i<rects.length;i++){"
            "var r=rects[i];"
            "r.setAttribute('fill',"
            "colorFor(vec[+r.dataset.i]));}}\n"
            "slider.addEventListener('input',function(){"
            "showGen(+slider.value);});\n"
            f"showGen({n_gen - 1});\n")
    elif n_gen > 1:
        parts.append(
            "var slider=document.getElementById('genSlider');"
            "slider.disabled=true;"
            "document.getElementById('genLabel').textContent="
            "'final generation (series in heatmap_series.json)';\n")
    parts.append("</script></body></html>\n")
    html = "".join(parts)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(html)
    os.replace(tmp, out_path)
    return {"path": str(out_path), "embedded": bool(embedded),
            "n_panels": len(matrices),
            "series_bytes": int(series_bytes)}


# ================================================================
# Post-hoc analysis: lineage, retention, heatmap
# ================================================================

def _load_layout_for_run(out_dir, results):
    """Rebuild the final Layout from results.json (or the
    checkpoint) so the heatmap panels match the run's final
    architecture. Returns (layout, domain, source_note) or
    (None, domain, note)."""
    domain = None
    arch_sig = None
    source = None
    if results is not None:
        cfg = results.get("config") or {}
        domain = cfg.get("domain")
        fa = results.get("final_arch")
        if isinstance(fa, dict):
            arch_sig = json.dumps(fa, sort_keys=True)
            source = "results.json final_arch"
        elif isinstance(fa, str):
            arch_sig = fa
            source = "results.json final_arch"
    if arch_sig is None:
        ckpt = Path(out_dir) / "checkpoint.npz"
        if ckpt.exists():
            try:
                z = np.load(ckpt, allow_pickle=False)
                arch_sig = str(z["arch_signature"])
                source = "checkpoint.npz arch_signature"
                if domain is None:
                    domain = json.loads(
                        str(z["config_canonical"])).get("domain")
            except Exception as exc:  # noqa: BLE001
                return None, domain, \
                    f"checkpoint unreadable: {exc}"
    if arch_sig is None:
        return None, domain, "no arch signature found"
    try:
        if domain == "stock":
            import bert_domain_stock as stock
            head_spec_fn = stock.head_spec
        else:
            import bert_domain_tsp as tsp
            head_spec_fn = tsp.head_spec
        arch = be.ArchSpec.from_signature(arch_sig)
        layout = be.Layout(arch, head_spec_fn(arch.d))
        return layout, domain, source
    except Exception as exc:  # noqa: BLE001
        return None, domain, f"layout rebuild failed: {exc}"


def analyse_run(out_dir, usefulness_step=DEFAULT_USEFULNESS_STEP,
                cfg=None):
    """Analyse a finished run's archive and retain the winning
    lineage.

    Reads ``results.json`` for the champion species id (falls
    back to the last generation's best-scoring species),
    traces the lineage chain backward through parent
    pointers (falling back to lineage-string matching),
    recomputes the per-generation usefulness vectors for the
    chain, writes the retained lineage archive + heatmap
    under ``{out_dir}/lineage/``, then deletes the
    full-population archive records. Returns a report dict.
    The runner calls this best-effort (try/except, never
    fail the run).
    """
    if cfg is not None:
        usefulness_step = _usefulness_step(cfg)
    out_dir = Path(out_dir)
    arch_dir = archive_dir(out_dir)
    lineage_dir = out_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    results = None
    results_path = out_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # ---- pass 1: parent/lineage maps only (streaming; the
    # genomes are discarded as each record is read).
    parent_by_gen = {}
    lineage_by_gen = {}
    record_files = []
    if arch_dir.is_dir():
        for p in arch_dir.glob("stage*_gen*.npz"):
            try:
                record_files.append((p, p.stat().st_size))
            except OSError:
                pass
    n_records = 0
    last_gen = None
    last_best_species = None
    for rec in iter_archive_records(arch_dir):
        g = rec["total_gens"]
        n_records += 1
        last_gen = g
        pmap, lmap = {}, {}
        for sid, parent, lin in zip(rec["species_ids"],
                                    rec["parent_species_ids"],
                                    rec["lineages"]):
            sid = int(sid)
            pmap[sid] = int(parent)
            lmap[sid] = str(lin)
        parent_by_gen[g] = pmap
        lineage_by_gen[g] = lmap
        # last generation's best-scoring species (fallback winner)
        best_i = int(np.argmax(rec["scores"]))
        last_best_species = int(rec["species_ids"][best_i])
        del rec
    if n_records == 0:
        raise ValueError(f"no archive records in {arch_dir}")

    fallback_used = []
    winner = None
    if results is not None:
        srs = results.get("stage_results") or []
        if srs:
            winner = srs[-1].get("champion_species_id")
    if winner is None:
        winner = last_best_species
        fallback_used.append(
            "winner from last generation's best-scoring species "
            "(no results.json stage_results)")
    winner = int(winner)

    # ---- trace the chain backward.
    # A species' parent pointer is recorded at every
    # generation the species EXISTS, and it names the species
    # the child was regenerated FROM -- but the parent itself
    # only existed BEFORE the child first appeared. So the
    # switch from child to parent happens when stepping back
    # across the child's FIRST generation, not at every
    # generation: while the child still existed at g-1, the
    # chain species at g-1 is still the child.
    gens_sorted = sorted(parent_by_gen)
    first_gen = {}
    for g in gens_sorted:
        for sid in parent_by_gen.get(g, {}):
            if sid not in first_gen:
                first_gen[sid] = g
    chain_at = {}
    cur = winner
    chain_broken = None
    for gi in range(len(gens_sorted) - 1, -1, -1):
        g = gens_sorted[gi]
        chain_at[g] = cur
        if gi == 0:
            break
        prev_g = gens_sorted[gi - 1]
        if first_gen.get(cur, g) <= prev_g:
            continue  # cur already existed at prev_g
        parent = parent_by_gen.get(g, {}).get(cur, -1)
        if parent is not None and int(parent) != -1:
            cur = int(parent)
        else:
            lin = lineage_by_gen.get(g, {}).get(cur)
            cand = None
            if lin is not None:
                for sid, l in lineage_by_gen.get(
                        prev_g, {}).items():
                    if l == lin:
                        cand = sid
                        break
            if cand is not None:
                if "lineage-string matching" \
                        not in fallback_used:
                    fallback_used.append(
                        "lineage-string matching (parent "
                        "pointer missing)")
                cur = int(cand)
            else:
                chain_broken = g
                fallback_used.append(
                    f"chain broken at generation {g}: no parent "
                    "pointer and no lineage match")
                break
    # compress chain_at into [(generation_range, species_id)]
    chain = []
    for g in gens_sorted:
        if g not in chain_at:
            continue
        sid = chain_at[g]
        if chain and chain[-1][1] == sid \
                and chain[-1][0][1] == g - 1:
            chain[-1][0][1] = g
        else:
            chain.append([[g, g], int(sid)])
    generations_covered = len(chain_at)

    # ---- pass 2: recompute log-scores, collect chain members
    # A growth step changes the genome width (gene positions
    # shift, entry names do not), so the accumulated vectors
    # are rebased by entry name at each width change -- the
    # same by-name rule the online tracker uses. Without the
    # shapes a width change can only reset the vectors (the
    # runner always records shapes; the fallback covers
    # records written before shapes were stored).
    vectors = {}
    per_gen_vectors = []   # (total_gens, vector) for chain gens
    per_gen_totals = []
    stage_members = {}     # stage -> {genomes, scores, sids, gens}
    shapes_by_gen = {}
    shape_chain = []       # distinct shapes, generation order
    n_params = None
    cur_shapes = None
    for rec in iter_archive_records(arch_dir):
        g = rec["total_gens"]
        stage = rec["stage"]
        G, S = rec["genomes"], rec["scores"]
        rec_shapes = rec.get("entry_shapes")
        shapes_by_gen[g] = rec_shapes
        if rec_shapes is not None and (not shape_chain
                or shape_chain[-1] != rec_shapes):
            shape_chain.append(rec_shapes)
        if n_params is None:
            n_params = G.shape[1]
            cur_shapes = rec_shapes
        elif G.shape[1] != n_params:
            if cur_shapes is not None and rec_shapes is not None:
                for sid in list(vectors):
                    vectors[sid] = be.remap_vector_by_entries(
                        vectors[sid], cur_shapes, rec_shapes)
            else:
                for sid in list(vectors):
                    vectors[sid] = np.zeros(G.shape[1],
                                            dtype=np.float64)
                if "usefulness reset at a width change " \
                        "(no entry shapes in the records)" \
                        not in fallback_used:
                    fallback_used.append(
                        "usefulness reset at a width change "
                        "(no entry shapes in the records)")
            n_params = G.shape[1]
            cur_shapes = rec_shapes
        sids = rec["species_ids"]
        pmap = parent_by_gen.get(g, {})
        # group row indices by species, in order of appearance
        order = []
        rows_by_sid = {}
        for i, sid in enumerate(sids):
            sid = int(sid)
            if sid not in rows_by_sid:
                rows_by_sid[sid] = []
                order.append(sid)
            rows_by_sid[sid].append(i)
        for sid in order:
            if sid not in vectors:
                parent = pmap.get(sid, -1)
                if parent != -1 and parent in vectors:
                    vectors[sid] = vectors[parent].copy()
                else:
                    vectors[sid] = np.zeros(
                        n_params, dtype=np.float64)
            idx = rows_by_sid[sid]
            vectors[sid] = update_log_scores(
                vectors[sid], G[idx], S[idx], usefulness_step)
        chain_sid = chain_at.get(g)
        if chain_sid is not None and chain_sid in vectors:
            per_gen_vectors.append(vectors[chain_sid].copy())
            per_gen_totals.append(g)
            acc = stage_members.setdefault(
                stage, {"genomes": [], "scores": [],
                        "species_ids": [], "generations": []})
            idx = rows_by_sid.get(chain_sid, [])
            if idx:
                acc["genomes"].append(G[idx])
                acc["scores"].append(S[idx])
                acc["species_ids"].append(
                    np.full(len(idx), chain_sid, dtype=np.int64))
                acc["generations"].append(
                    np.full(len(idx), g, dtype=np.int64))
        del rec
    final_log = per_gen_vectors[-1] if per_gen_vectors \
        else np.zeros(n_params or 0, dtype=np.float64)

    # ---- write the retained lineage archive (one npz per stage)
    # Which stage each per-generation vector belongs to, from
    # the record filenames (computed once, before any file
    # is deleted).
    stage_of_gen = {}
    for rec_stage, rec_gens in _stage_gen_map(arch_dir):
        for gg in rec_gens:
            stage_of_gen[gg] = rec_stage
    kept_files = []
    kept_bytes = 0
    for stage in sorted(stage_members):
        acc = stage_members[stage]
        idxs = [i for i, gt in enumerate(per_gen_totals)
                if stage_of_gen.get(gt) == stage]
        lg = np.array([per_gen_vectors[i] for i in idxs],
                      dtype=np.float64) if idxs else \
            np.zeros((0, n_params), dtype=np.float64)
        gt_arr = np.array([per_gen_totals[i] for i in idxs],
                          dtype=np.int64)
        genomes_c = np.concatenate(acc["genomes"]) \
            if acc["genomes"] else np.zeros((0, n_params))
        scores_c = np.concatenate(acc["scores"]) \
            if acc["scores"] else np.zeros(0)
        sids_c = np.concatenate(acc["species_ids"]) \
            if acc["species_ids"] else np.zeros(0, dtype=np.int64)
        gens_c = np.concatenate(acc["generations"]) \
            if acc["generations"] else np.zeros(0, dtype=np.int64)
        fpath = lineage_dir / f"stage{int(stage):03d}.npz"
        be.atomic_npz(
            fpath, genomes=np.asarray(genomes_c,
                                      dtype=np.float64),
            scores=np.asarray(scores_c, dtype=np.float64),
            species_ids=sids_c, generations=gens_c,
            log_scores_per_gen=lg, gen_totals=gt_arr,
            final_log_scores=np.asarray(final_log,
                                        dtype=np.float64))
        kept_files.append(str(fpath))
        kept_bytes += fpath.stat().st_size

    # ---- heatmap series: every per-generation vector in the
    # FINAL layout's coordinates, so the slider indexes the
    # final panels. A vector is rebased forward through the
    # shape chain from its own generation's shapes; a
    # generation with no shapes (or no chain) reads as never
    # judged. The per-stage lineage npz files keep the
    # vectors in their own stage's coordinates (aligned with
    # the genomes in the same file); only the heatmap series
    # is rebased to the final layout.
    final_shapes = shape_chain[-1] if shape_chain else None
    final_n = len(per_gen_vectors[-1]) if per_gen_vectors \
        else (n_params or 0)
    heatmap_vectors = []
    heatmap_zeroed = False
    for vec, g in zip(per_gen_vectors, per_gen_totals):
        if len(vec) == final_n:
            heatmap_vectors.append(vec)
            continue
        src = shapes_by_gen.get(g)
        if src is None or final_shapes is None \
                or src not in shape_chain:
            heatmap_vectors.append(np.zeros(final_n))
            heatmap_zeroed = True
            continue
        v = vec
        cur = src
        for nxt in shape_chain[shape_chain.index(src) + 1:]:
            v = be.remap_vector_by_entries(v, cur, nxt)
            cur = nxt
        heatmap_vectors.append(v)
    if heatmap_zeroed and "heatmap series: some generations " \
            "had no entry shapes (shown as never judged)" \
            not in fallback_used:
        fallback_used.append(
            "heatmap series: some generations had no entry "
            "shapes (shown as never judged)")

    # ---- heatmap (needs the final layout)
    layout, domain, layout_source = _load_layout_for_run(
        out_dir, results)
    heatmap_info = None
    heatmap_note = None
    if layout is None:
        heatmap_note = (f"heatmap skipped: {layout_source}")
    elif layout.n_params != n_params:
        heatmap_note = (
            f"heatmap skipped: layout n_params "
            f"{layout.n_params} != archive n_params {n_params} "
            f"({layout_source})")
    else:
        heatmap_info = build_heatmap_html(
            layout, final_log, heatmap_vectors, per_gen_totals,
            lineage_dir / "heatmap.html",
            title=f"Usefulness heatmap (winner species {winner})")
    heatmap_path = heatmap_info["path"] if heatmap_info else None

    # ---- retention manifest, then dispose of the working archive
    deleted_files = [str(p) for p, _ in record_files]
    deleted_bytes = sum(s for _, s in record_files)
    retention = {
        "winner_species_id": winner,
        "chain": [{"generations": gr, "species_id": sid}
                  for gr, sid in chain],
        "generations_covered": generations_covered,
        "n_archive_records": n_records,
        "kept_files": kept_files,
        "kept_bytes": int(kept_bytes),
        "deleted_files": deleted_files,
        "deleted_bytes": int(deleted_bytes),
        "deleted_count": len(deleted_files),
        "heatmap_path": heatmap_path,
        "heatmap_note": heatmap_note,
        "fallback_used": fallback_used,
    }
    be.atomic_json(lineage_dir / "retention.json", retention)
    for p, _ in record_files:
        try:
            os.remove(p)
        except OSError:
            pass

    report = {
        "winner_species_id": winner,
        "chain": [{"generations": gr, "species_id": sid}
                  for gr, sid in chain],
        "chain_length": len(chain),
        "generations_covered": generations_covered,
        "final_density": density_shares(final_log),
        "heatmap_path": heatmap_path,
        "heatmap_note": heatmap_note,
        "heatmap_embedded_series":
            heatmap_info["embedded"] if heatmap_info else None,
        "lineage_files": kept_files,
        "retention_path": str(lineage_dir / "retention.json"),
        "bytes_kept": int(kept_bytes),
        "bytes_deleted": int(deleted_bytes),
        "n_archive_records": n_records,
        "fallback_used": fallback_used,
    }
    return report


def _stage_gen_map(arch_dir):
    """Yield (stage, [total_gens, ...]) from the record
    filenames on disk (cheap; no file reads)."""
    d = Path(arch_dir)
    by_stage = {}
    if d.is_dir():
        for p in d.glob("stage*_gen*.npz"):
            parsed = _parse_record_filename(p.name)
            if parsed is not None:
                by_stage.setdefault(parsed[0], []).append(
                    parsed[1])
    for stage in sorted(by_stage):
        yield stage, sorted(by_stage[stage])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args(argv)
    try:
        report = analyse_run(args.run_dir)
    except Exception as exc:  # noqa: BLE001 - CLI reports it
        print(json.dumps({"error": str(exc)}, indent=1))
        return 1
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
