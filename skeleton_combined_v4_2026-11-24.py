"""Combined skeleton read v4 (sponsor, 2026-11-24).

Two signals decide what a piece is:
  1. What made it win: on the champion's ancestral line,
     the piece's final form arrived in a child that beat
     its parent (the v3 kept-win read).
  2. What does not matter: disturb the piece in the
     champion (redraw its genes at random, 10 tries) and
     measure the damage. If it does not hurt, the piece
     is NEUTRAL (sponsor's word). The size of the fall
     grades the importance.

Classification:
  - skeleton: won its place AND disturbing it hurts.
  - neutral: disturbing it does not hurt, whatever its
    history.
  - load-bearing without a win: disturbing it hurts but
    it never won its place on the line.
"Does not hurt" line (Clayton's, stated openly): mean
damage under 5% of the champion's own gap, relative.

Merge by the combined rule: donor pieces that are
skeleton move into host slots that are neutral (free
space). Placebo: donor neutral pieces into the same
host neutral slots, same count. All scored on the fixed
validation set.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bert_engine as be
import bert_domain_tsp as tsp
import run_bert_engine as rbe

NEUTRAL_LINE = 0.05   # of the champion's gap, relative
DRAWS = 10
SEED = 20261124


def piece_index_sets(layout):
    pieces = {}
    covered = np.zeros(layout.n_params, dtype=bool)
    for u in be.silence_units(layout):
        idx = be.unit_gene_indices(layout, u)
        pieces[u["name"]] = idx
        covered[idx] = True
    head_idx = []
    for c in range(len(layout.arch.couples)):
        for name in layout.head_names(c):
            s = layout.sl(name)
            head_idx.append(np.arange(s.start, s.stop))
    if head_idx:
        idx = np.concatenate(head_idx)
        pieces["output_path"] = idx
        covered[idx] = True
    pieces["trunk"] = np.where(~covered)[0]
    return pieces


def main():
    cfg = be.EngineConfig.load("bert_engine_config_fresh50.json")
    backend = rbe.get_backend(cfg.backend, cfg.device,
                              cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)
    arch = be.initial_arch(cfg, 2)
    layout = be.Layout(arch, tsp.head_spec(arch.d))
    pieces = piece_index_sets(layout)

    v3 = json.loads(Path(
        "skeleton_merge_v3_2026-11-24.json").read_text())
    z = np.load("skeleton_merge_v3_2026-11-24_genomes.npz")
    champs = {"a": (v3["species_a"], z["a"], v3["read_a"]),
              "b": (v3["species_b"], z["b"], v3["read_b"])}
    rng = np.random.default_rng(SEED)

    def gap_of(genomes):
        s = domain.validation_scores(layout, genomes)
        return -s * 100.0  # scores are -mean gap

    reads = {}
    for tag, (sid, genome, win_read) in champs.items():
        base_gap = float(gap_of(genome[None, :])[0])
        per = {}
        for name, idx in pieces.items():
            draws = np.tile(genome, (DRAWS, 1))
            draws[:, idx] = rng.normal(
                0, cfg.draw_sigma, (DRAWS, len(idx)))
            gaps = gap_of(draws)
            damage = float(gaps.mean() - base_gap)
            rel = damage / base_gap if base_gap else 0.0
            won = win_read[name]["status"] == "skeleton"
            neutral = rel < NEUTRAL_LINE
            if neutral:
                cls = "neutral"
            elif won:
                cls = "skeleton"
            else:
                cls = "load-bearing, never won"
            per[name] = {
                "won": bool(won),
                "damage_points": damage,
                "damage_relative": rel,
                "class": cls,
            }
        reads[tag] = {"species": sid, "gap": base_gap,
                      "pieces": per}
        print(f"[test] species {sid} (gap {base_gap:.2f}%):")
        for name, r in per.items():
            print(f"    {name}: {r['class']} "
                  f"(won={r['won']}, damage "
                  f"{r['damage_points']:+.2f} pts, "
                  f"{r['damage_relative'] * 100:.0f}% of gap)")

    a = reads["a"]["pieces"]
    b = reads["b"]["pieces"]
    ga, gb = champs["a"][1], champs["b"][1]
    host_neutral = sorted(n for n, r in a.items()
                          if r["class"] == "neutral")
    moved = sorted(n for n in host_neutral
                   if b[n]["class"] == "skeleton")
    merged = ga.copy()
    for n in moved:
        merged[pieces[n]] = gb[pieces[n]]
    pool = sorted(n for n in host_neutral
                  if b[n]["class"] == "neutral")
    placebo_moved = pool[:len(moved)]
    placebo = ga.copy()
    for n in placebo_moved:
        placebo[pieces[n]] = gb[pieces[n]]

    def gap1(g):
        return float(gap_of(g[None, :])[0])

    result = {
        "neutral_line_relative": NEUTRAL_LINE,
        "reads": reads,
        "host_neutral_slots": host_neutral,
        "merged_pieces": moved,
        "gap_a_pct": gap1(ga), "gap_b_pct": gap1(gb),
        "gap_merged_pct": gap1(merged),
        "placebo_pieces": placebo_moved,
        "gap_placebo_pct": gap1(placebo),
    }
    np.savez("skeleton_combined_v4_2026-11-24_genomes.npz",
             merged=merged, placebo=placebo)
    Path("skeleton_combined_v4_2026-11-24.json").write_text(
        json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result[k] for k in (
        "host_neutral_slots", "merged_pieces", "gap_a_pct",
        "gap_b_pct", "gap_merged_pct", "placebo_pieces",
        "gap_placebo_pct")}, indent=1))


if __name__ == "__main__":
    main()
