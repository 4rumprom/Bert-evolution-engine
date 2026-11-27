#!/usr/bin/env python3
"""Render diagrams/frankenstein_squared_flow_tall.png.

Tall stacked layout (the flow-renderer approach of
frankenstein_render_flow_2026-11-25.py): three
panels — the evolved Frankenstein 4 champion,
species 10's champion, and Frankenstein Squared —
with every synapse GRADED by its measured signal
flow in the champion panels (line thickness = the
synapse's flow std relative to the busiest synapse
feeding the same node; path synapses keep their
side's colour, off-path synapses are dark gray
graded by the same measure). In the Squared panel
only the two extracted paths are real (orange =
Frankenstein 4 side, blue = species 10 side);
everything else is tiny-random filler, thin light
gray; the merged decoder is coloured by block
(orange 4x4 = the evolved Frankenstein 4 decoder,
blue 2x2 = species 10's, gray = cross filler).

Input: frankenstein_squared_state.json (written by
frankenstein_squared_stitch_2026-11-25.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STATE = json.load(open(HERE / "frankenstein_squared_state.json"))

C_A = "#e4572e"        # side A (Frankenstein 4) orange
C_B = "#2166c7"        # side B (species 10) blue
C_GRAY = "#c6cbce"
C_OFF_LO = "#a8b0b4"
C_OFF_HI = "#2b3033"
C_DOT_A = "#14666b"
C_DOT_B = "#6a4ba8"
C_DOT_J = "#8a939a"
C_PICK = "#444c50"
BG = "#f7f3ec"
LW0, LW1 = 0.55, 3.4


def _lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        round(a[k] + (b[k] - a[k]) * t) for k in range(3))


def column_positions(node_ids, x):
    m = len(node_ids)
    step = 0.55 if m >= 4 else (0.95 if m == 2 else 0.0)
    top = (m - 1) * step / 2.0
    return {nd: (x, top - i * step)
            for i, nd in enumerate(node_ids)}


def _style(cls, rel):
    if cls == "a":
        lw = 2.2 if rel is None else LW0 + (LW1 - LW0) * rel
        return C_A, lw, 1.0, 2
    if cls == "b":
        lw = 2.2 if rel is None else LW0 + (LW1 - LW0) * rel
        return C_B, lw, 1.0, 2
    if cls == "o":
        r = 0.0 if rel is None else rel
        return _lerp(C_OFF_LO, C_OFF_HI, r), \
            LW0 + (LW1 - LW0) * r, 1.0, 1
    return C_GRAY, 0.65, 0.75, 0


def draw_panel_edges(ax, pos, edge_list):
    styled = []
    for e in edge_list:
        src, dst = e["src"], e["dst"]
        if src not in pos or dst not in pos:
            continue
        color, lw, alpha, prio = _style(e["cls"], e["rel"])
        styled.append((prio, lw, src, dst, color, alpha))
    styled.sort(key=lambda t: (t[0], t[1]))
    for _, lw, src, dst, color, alpha in styled:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                alpha=alpha, solid_capstyle="round",
                zorder=1)


def draw_dots(ax, pos, color_of):
    for nd, (x, y) in pos.items():
        ax.plot([x], [y], marker="o", markersize=9,
                markerfacecolor=color_of(nd),
                markeredgecolor="white",
                markeredgewidth=1.1, linestyle="",
                zorder=3)


def finish_panel(ax, pos, labels, title, note=None):
    ymax = max(abs(y) for _, y in pos.values())
    for x, text in labels:
        ax.text(x, ymax + 0.85, text, ha="center",
                va="center", fontsize=11,
                color="#5a666b", fontweight="bold")
    ax.set_xlim(-0.7, max(x for x, _ in pos.values()) + 0.7)
    ax.set_ylim(-ymax - (2.3 if note else 1.1),
                ymax + 1.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, color="#333333",
                 fontweight="bold", pad=10)
    if note:
        ax.text(max(x for x, _ in pos.values()) / 2.0,
                -ymax - 1.55, note, ha="center",
                va="center", fontsize=11, color="#5a666b")


def panel_layout(panel):
    """Columns for a network with stream width d and
    per-block FFN widths panel['blocks']."""
    d = panel["d"]
    cols = [("Input", ["in0", "in1"])]
    for bi, U in enumerate(panel["blocks"]):
        if bi == 0:
            cols.append(
                ("Stream", [f"s{2*bi}_{k}" for k in range(d)]))
        # for bi > 0 the input-stream column is the
        # previous block's output-stream column (same
        # nodes); adding it again would leave an empty
        # column, so it is not repeated
        cols += [
            ("Query", [f"q{bi}_{k}" for k in range(d)]),
            ("Key", [f"k{bi}_{k}" for k in range(d)]),
            ("Value", [f"v{bi}_{k}" for k in range(d)]),
            ("Mix", [f"mix{bi}_{k}" for k in range(d)]),
            ("Stream", [f"s{2*bi+1}_{k}" for k in range(d)]),
            ("Feed-forward" if bi < len(panel["blocks"]) - 1
             or len(panel["blocks"]) == 1 else "Joining FF",
             [f"ffn{bi}_{u}" for u in range(U)]),
            ("Stream", [f"s{2*bi+2}_{k}" for k in range(d)]),
        ]
    cols += [("Decoder", [f"dec_{k}" for k in range(d)]),
             ("Pick", ["pick"])]
    pos = {}
    labels = []
    for ci, (name, nodes) in enumerate(cols):
        pos.update(column_positions(nodes, ci * 1.0))
        labels.append((ci * 1.0, name))
    return pos, labels


def champion_dots(panel, terr_color):
    def color_of(nd):
        if nd == "pick" or nd.startswith("in"):
            return C_PICK
        return terr_color
    return color_of


def squared_dots(panel):
    split = panel["stream_split"]
    usplits = panel["unit_splits"]
    nblocks = len(panel["blocks"])

    def color_of(nd):
        if nd == "pick" or nd.startswith("in"):
            return C_PICK
        base = nd.split("_")[0]
        idx = int(nd.split("_")[1])
        if base.startswith("ffn"):
            bi = int(base[3:])
            if bi == nblocks - 1:
                return C_DOT_J
            return C_DOT_A if idx < usplits[bi] else C_DOT_B
        bi = None
        for pre in ("q", "k", "v", "mix"):
            if base.startswith(pre) and base[len(pre):].isdigit():
                bi = int(base[len(pre):])
        if bi is not None and bi == nblocks - 1:
            return C_DOT_J
        return C_DOT_A if idx < split else C_DOT_B

    return color_of


P = STATE["panels"]
fig = plt.figure(figsize=(26, 42), dpi=200)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.45],
                      hspace=0.30)
axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
for ax in axes:
    ax.set_facecolor(BG)

# ---- panel A: evolved Frankenstein 4 champion
pan = P["sideA"]
pos, labels = panel_layout(pan)
draw_panel_edges(axes[0], pos, pan["edges"])
draw_dots(axes[0], pos, champion_dots(pan, C_DOT_A))
finish_panel(axes[0], pos, labels,
             f"Side A: evolved Frankenstein 4 champion\n"
             f"stream 4 wide, 2 blocks, FFN 6 + 6, "
             f"gap {pan['gap_pct']:.2f}%\ncorrected path: "
             f"{pan['path_synapses']} of "
             f"{pan['total_synapses']} synapses")

# ---- panel B: species 10 champion
pan = P["sideB"]
pos, labels = panel_layout(pan)
draw_panel_edges(axes[1], pos, pan["edges"])
draw_dots(axes[1], pos, champion_dots(pan, C_DOT_B))
finish_panel(axes[1], pos, labels,
             f"Side B: species 10 champion\nstream 2 "
             f"wide, FFN 8, 1 head, gap "
             f"{pan['gap_pct']:.2f}%\ncorrected path: "
             f"{pan['path_synapses']} of "
             f"{pan['total_synapses']} synapses")

# ---- panel C: Frankenstein Squared
pan = P["squared"]
pos, labels = panel_layout(pan)
draw_panel_edges(axes[2], pos, pan["edges"])
draw_dots(axes[2], pos, squared_dots(pan))
U = pan["blocks"]
finish_panel(axes[2], pos, labels,
             f"Frankenstein Squared: the two paths side "
             f"by side, joining block, one merged "
             f"decoder\nstream 6 wide, FFN {U[0]} + "
             f"{U[1]} + joining FFN {U[2]}, gap "
             f"{pan['gap_pct']:.2f}% (birth)",
             note="ONE merged decoder over width 6: "
                  "orange block is the evolved "
                  "Frankenstein 4 decoder, blue block "
                  "is species 10's, both kept exact.\n"
                  "Gray cross entries are tiny random "
                  "(std 0.03). No second decoder, no "
                  "averaging after.")

handles = [
    Line2D([0], [0], color=C_A, lw=3,
           label="Frankenstein 4 side: path synapses "
                 "(kept exact)"),
    Line2D([0], [0], color=C_B, lw=3,
           label="species 10 side: path synapses "
                 "(kept exact)"),
    Line2D([0], [0], color=C_OFF_HI, lw=2.5,
           label="off-path synapses in the champions: "
                 "real synapses, thinner and lighter "
                 "the less signal they carry"),
    Line2D([0], [0], color=C_GRAY, lw=2,
           label="tiny-random filler (Frankenstein "
                 "Squared; no measured flow)"),
    Line2D([0], [0], color="none", lw=0,
           label="line thickness = measured share of "
                 "signal at the node it feeds"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=12, frameon=False,
           bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Frankenstein Squared: species 10's branch "
             "stitched into the evolved Frankenstein 4",
             fontsize=19, color="#333333",
             fontweight="bold", y=0.995)
out = HERE / "diagrams" / "frankenstein_squared_flow_tall.png"
fig.savefig(out, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
print("saved", out)
plt.close(fig)
print("DONE render squared")
