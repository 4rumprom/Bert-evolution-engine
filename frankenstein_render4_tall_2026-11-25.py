#!/usr/bin/env python3
"""Render frankenstein_4_tall.png (tall stacked
format, sponsor reads on his phone).

Three panels: species 8 genome, species 9 genome,
Frankenstein 4. Every neuron a dot, every synapse
a line. Corrected-path synapses in orange
(species 8) and blue (species 9); off-path
synapses and tiny-random filler in light gray.
In Frankenstein 4's decoder column the merged
decoder is coloured by block: species 8's 2x2
block orange, species 9's 2x2 block blue, the
tiny-random cross blocks gray.

Helpers and layout are copied from
frankenstein_render_tall_2026-11-25.py; inputs
are the JSON state files (no pickle).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
S2 = json.load(open(HERE / "frankenstein_stitch2_state.json"))
S4 = json.load(open(HERE / "frankenstein_stitch4_state.json"))
R2 = json.load(open(HERE / "frankenstein_stitch2_2026-11-25.json"))
R4 = json.load(open(HERE / "frankenstein_stitch4_2026-11-25.json"))

C_HOST = "#e4572e"
C_DONOR = "#2166c7"
C_GRAY = "#c6cbce"
C_DOT = "#14666b"
C_DOT_D = "#6a4ba8"
C_DOT_J = "#8a939a"
C_PICK = "#444c50"
BG = "#f7f3ec"


def column_positions(node_ids, x):
    m = len(node_ids)
    step = 0.55 if m >= 4 else (0.95 if m == 2 else 0.0)
    top = (m - 1) * step / 2.0
    return {nd: (x, top - i * step)
            for i, nd in enumerate(node_ids)}


def draw_panel_edges(ax, pos, edge_list):
    style = {"h": (C_HOST, 2.2), "d": (C_DONOR, 2.2),
             "f": (C_GRAY, 0.65)}
    for src, dst, cls in edge_list:
        if src not in pos or dst not in pos:
            continue
        color, lw = style[cls]
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                alpha=1.0 if cls != "f" else 0.75,
                solid_capstyle="round", zorder=1)


def draw_dots(ax, pos, color_of):
    for nd, (x, y) in pos.items():
        ax.plot([x], [y], marker="o", markersize=9,
                markerfacecolor=color_of(nd),
                markeredgecolor="white",
                markeredgewidth=1.1, linestyle="",
                zorder=3)


def finish_panel(ax, pos, labels, title):
    ymax = max(abs(y) for _, y in pos.values())
    for x, text in labels:
        ax.text(x, ymax + 0.85, text, ha="center",
                va="center", fontsize=11,
                color="#5a666b", fontweight="bold")
    ax.set_xlim(-0.7, max(x for x, _ in pos.values())
                + 0.7)
    ax.set_ylim(-ymax - 1.1, ymax + 1.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, color="#333333",
                 fontweight="bold", pad=10)


def parent_panel(ax, P, path_color_cls, title):
    cols = [("Input", [f"in{c}" for c in range(2)]),
            ("Stream", [f"s0_{k}" for k in range(2)]),
            ("Query", [f"q_{k}" for k in range(2)]),
            ("Key", [f"k_{k}" for k in range(2)]),
            ("Value", [f"v_{k}" for k in range(2)]),
            ("Mix", [f"mix_{k}" for k in range(2)]),
            ("Stream", [f"s1_{k}" for k in range(2)]),
            ("Feed-forward",
             [f"ffn_{u}" for u in range(8)]),
            ("Stream", [f"s2_{k}" for k in range(2)]),
            ("Decoder", [f"dec_{k}" for k in range(2)]),
            ("Pick", ["pick"])]
    pos = {}
    labels = []
    for ci, (name, nodes) in enumerate(cols):
        pos.update(column_positions(nodes, ci * 1.0))
        labels.append((ci * 1.0, name))
    edges = [(e["src"], e["dst"],
              path_color_cls if e["on_path"] else "f")
             for e in P["edges"]]
    draw_panel_edges(ax, pos, edges)
    draw_dots(ax, pos,
              lambda nd: C_PICK if nd == "pick" else C_DOT)
    finish_panel(ax, pos, labels, title)


# ---------- shared Frankenstein block-1 layout ----------
def frank_block1(state):
    dF, U = state["dF"], state["U"]
    h_map = {int(k): v for k, v in
             state["h_map"].items()}
    d_map = {int(k): v for k, v in
             state["d_map"].items()}
    hu_map = {int(k): v for k, v in
              state["hu_map"].items()}
    du_map = {int(k): v for k, v in
              state["du_map"].items()}
    h_path = {tuple(x) for x in state["h_path"]}
    d_path = {tuple(x) for x in state["d_path"]}
    NH = len(h_map)
    inv_h = {v: k for k, v in h_map.items()}
    inv_d = {v: k for k, v in d_map.items()}
    inv_hu = {v: k for k, v in hu_map.items()}
    inv_du = {v: k for k, v in du_map.items()}

    def terr(fi):
        return "h" if fi < NH else "d"

    def old(fi):
        return inv_h[fi] if fi < NH else inv_d[fi]

    def mat_cls(entry, fi, fj):
        if terr(fi) != terr(fj):
            return "f"
        s = h_path if terr(fi) == "h" else d_path
        return terr(fi) if (entry, old(fi),
                            old(fj)) in s else "f"

    cols = [("Input", [f"in{c}" for c in range(2)]),
            ("Stream", [f"s0_{i}" for i in range(dF)]),
            ("Query", [f"q_{i}" for i in range(dF)]),
            ("Key", [f"k_{i}" for i in range(dF)]),
            ("Value", [f"v_{i}" for i in range(dF)]),
            ("Mix", [f"mix_{i}" for i in range(dF)]),
            ("Stream", [f"s1_{i}" for i in range(dF)]),
            ("Feed-forward",
             [f"ffn_{i}" for i in range(U)]),
            ("Stream", [f"s2_{i}" for i in range(dF)])]
    pos = {}
    labels = []
    for ci, (name, nodes) in enumerate(cols):
        pos.update(column_positions(nodes, ci * 1.0))
        labels.append((ci * 1.0, name))
    edges = []
    for c in range(2):
        for fi in range(dF):
            t = terr(fi)
            s = h_path if t == "h" else d_path
            edges.append((f"in{c}", f"s0_{fi}",
                          t if ("embed.W_in", c,
                                old(fi)) in s else "f"))
    for nm, tg in (("W_q", "q"), ("W_k", "k"),
                   ("W_v", "v")):
        for fi in range(dF):
            for fj in range(dF):
                edges.append((f"s0_{fi}", f"{tg}_{fj}",
                              mat_cls(f"b0.{nm}",
                                      fi, fj)))
    par = S2["parents"]
    cond = {}
    resid = {}
    logit = {}
    for tag, P in par.items():
        for e in P["edges"]:
            if e["kind"] == "conduit" and \
                    e["entry"] in ("attnQ", "attnK",
                                   "attnA"):
                cond[(tag, e["entry"], e["j"])] = \
                    e["on_path"]
            if e["kind"] == "residual":
                resid[(tag, e["entry"], e["i"])] = \
                    e["on_path"]
            if e["entry"] == "logit":
                logit[(tag, e["j"])] = e["on_path"]
    for fj in range(dF):
        t = terr(fj)
        tag = "host" if t == "h" else "donor"
        for entry, src in (("attnQ", "q"),
                           ("attnK", "k"),
                           ("attnA", "v")):
            on = cond.get((tag, entry, old(fj)), False)
            edges.append((f"{src}_{fj}", f"mix_{fj}",
                          t if on else "f"))
    for fi in range(dF):
        for fj in range(dF):
            edges.append((f"mix_{fi}", f"s1_{fj}",
                          mat_cls("b0.W_o", fi, fj)))
    for fi in range(dF):
        t = terr(fi)
        tag = "host" if t == "h" else "donor"
        on = resid.get((tag, "residual1", old(fi)),
                       False)
        edges.append((f"s0_{fi}", f"s1_{fi}",
                      t if on else "f"))
        on = resid.get((tag, "residual2", old(fi)),
                       False)
        edges.append((f"s1_{fi}", f"s2_{fi}",
                      t if on else "f"))
    for fi in range(dF):
        for ui in range(U):
            tu = "h" if ui < len(hu_map) else "d"
            if terr(fi) != tu:
                cls = "f"
            else:
                u_old = inv_hu[ui] if tu == "h" \
                    else inv_du[ui]
                s = h_path if tu == "h" else d_path
                cls = tu if ("b0.W_1", old(fi),
                             u_old) in s else "f"
            edges.append((f"s1_{fi}", f"ffn_{ui}", cls))
    for ui in range(U):
        for fj in range(dF):
            tu = "h" if ui < len(hu_map) else "d"
            if terr(fj) != tu:
                cls = "f"
            else:
                u_old = inv_hu[ui] if tu == "h" \
                    else inv_du[ui]
                s = h_path if tu == "h" else d_path
                cls = tu if ("b0.W_2", u_old,
                             old(fj)) in s else "f"
            edges.append((f"ffn_{ui}", f"s2_{fj}", cls))
    ctx = {"dF": dF, "U": U, "NH": NH, "terr": terr,
           "old": old, "h_path": h_path,
           "d_path": d_path, "logit": logit,
           "hu_n": len(hu_map)}
    return pos, labels, edges, ctx


def frank_dots(ax, pos, ctx, extra_join_nodes=()):
    NH, hu_n = ctx["NH"], ctx["hu_n"]

    def color_of(nd):
        if nd == "pick" or nd.startswith("in"):
            return C_PICK
        base = nd.split("_")[0]
        if base in ("jq", "jk", "jv", "jmix", "s3",
                    "jffn", "s4", "join"):
            return C_DOT_J
        if base == "ffn":
            return C_DOT if int(nd.split("_")[1]) < hu_n \
                else C_DOT_D
        if base in ("dech",):
            return C_DOT
        if base in ("decd",):
            return C_DOT_D
        idx = int(nd.split("_")[1])
        return C_DOT if idx < NH else C_DOT_D

    draw_dots(ax, pos, color_of)


def make_fig():
    fig = plt.figure(figsize=(17, 38), dpi=200)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(3, 1,
                          height_ratios=[10, 10, 18],
                          hspace=0.28)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(BG)
    return fig, axes


def add_legend(fig, host_sp, donor_sp, extra):
    handles = [
        Line2D([0], [0], color=C_HOST, lw=3,
               label=f"species {host_sp} path synapses "
                     "(kept exact)"),
        Line2D([0], [0], color=C_DONOR, lw=3,
               label=f"species {donor_sp} path synapses "
                     "(kept exact)"),
        Line2D([0], [0], color=C_GRAY, lw=2,
               label="off-path synapses (parents) / "
                     "tiny-random filler "
                     "(Frankenstein)"),
    ] + extra
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=12, frameon=False,
               bbox_to_anchor=(0.5, -0.02))


hostP = S2["parents"]["host"]
donorP = S2["parents"]["donor"]
n_host_path = R2["parents"]["host"][
    "n_path_weight_synapses"]
n_donor_path = R2["parents"]["donor"][
    "n_path_weight_synapses"]
t_host = (f"Parent A: species {hostP['species']} "
          f"champion\nstream 2 wide, FFN 8, 1 head, "
          f"gap {hostP['gap']*100:.2f}%\ncorrected "
          f"path: {n_host_path} of 56 synapses")
t_donor = (f"Parent B: species {donorP['species']} "
           f"champion\nstream 2 wide, FFN 8, 1 head, "
           f"gap {donorP['gap']*100:.2f}%\ncorrected "
           f"path: {n_donor_path} of 56 synapses")

# ================================================================
# Frankenstein 4 figure (merged decoder)
# ================================================================
fig, axes = make_fig()
parent_panel(axes[0], hostP, "h", t_host)
parent_panel(axes[1], donorP, "d", t_donor)
ax = axes[2]
pos, labels, edges, ctx = frank_block1(S4["frank"])
dF, U = ctx["dF"], ctx["U"]
U2 = S4["frank"]["U2"]
x0 = 9.0  # first joining column x
extra_cols = [("Join Q", [f"jq_{i}" for i in range(dF)]),
              ("Join K", [f"jk_{i}" for i in range(dF)]),
              ("Join V", [f"jv_{i}" for i in range(dF)]),
              ("Join mix",
               [f"jmix_{i}" for i in range(dF)]),
              ("Stream", [f"s3_{i}" for i in range(dF)]),
              ("Joining FF",
               [f"jffn_{i}" for i in range(U2)]),
              ("Stream", [f"s4_{i}" for i in range(dF)]),
              ("Decoder",
               [f"dec_{i}" for i in range(dF)]),
              ("Pick", ["pick"])]
for ci, (name, nodes) in enumerate(extra_cols):
    pos.update(column_positions(nodes, x0 + ci * 1.0))
    labels.append((x0 + ci * 1.0, name))
for fi in range(dF):
    for tg in ("jq", "jk", "jv"):
        for fj in range(dF):
            edges.append((f"s2_{fi}", f"{tg}_{fj}", "f"))
    for src in ("jq", "jk", "jv"):
        edges.append((f"{src}_{fi}", f"jmix_{fi}", "f"))
    for fj in range(dF):
        edges.append((f"jmix_{fi}", f"s3_{fj}", "f"))
    edges.append((f"s2_{fi}", f"s3_{fi}", "f"))
    for ui in range(U2):
        edges.append((f"s3_{fi}", f"jffn_{ui}", "f"))
    edges.append((f"s3_{fi}", f"s4_{fi}", "f"))
for ui in range(U2):
    for fj in range(dF):
        edges.append((f"jffn_{ui}", f"s4_{fj}", "f"))
# merged decoder: coloured by block, NOT by path.
# In-territory entries are the parent's own decoder
# weights, kept exact (orange = species 8 block,
# blue = species 9 block); cross entries are the
# tiny-random filler (gray).
for fi in range(dF):
    for fj in range(dF):
        if ctx["terr"](fi) == ctx["terr"](fj):
            cls = ctx["terr"](fi)
        else:
            cls = "f"
        edges.append((f"s4_{fi}", f"dec_{fj}", cls))
for fj in range(dF):
    edges.append((f"dec_{fj}", "pick", ctx["terr"](fj)))
draw_panel_edges(ax, pos, edges)
frank_dots(ax, pos, ctx)
ymax = max(abs(y) for _, y in pos.values())
x_dec = x0 + 7.0
ax.text(x_dec, -ymax - 0.95,
        "ONE merged decoder: orange block is "
        f"species {hostP['species']}'s decoder, "
        "blue block is species "
        f"{donorP['species']}'s, both kept exact.\n"
        "Gray cross entries are tiny random "
        "(std 0.03). No second decoder, no "
        "averaging after.",
        ha="center", va="center", fontsize=11,
        color="#5a666b")
gap4 = R4["frankenstein4"]["validation_gap_pct"]
finish_panel(ax, pos, labels,
             f"Frankenstein 4: ONE merged decoder "
             f"(both parents' blocks)\nstream {dF} "
             f"wide, FFN {U} + joining FFN {U2}, "
             f"gap {gap4:.2f}%")
ax.set_ylim(-ymax - 2.3, ymax + 1.5)
add_legend(fig, hostP["species"], donorP["species"], [])
fig.suptitle("Frankenstein 4: two corrected paths, "
             "joining layer, one merged decoder",
             fontsize=18, color="#333333",
             fontweight="bold", y=0.99)
out4 = HERE / "diagrams" / "frankenstein_4_tall.png"
fig.savefig(out4, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
print("saved", out4)
plt.close(fig)
print("DONE render")
