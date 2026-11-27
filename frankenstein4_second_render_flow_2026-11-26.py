#!/usr/bin/env python3
"""Render frankenstein4_second_flow_tall.png.

Same tall stacked layout as frankenstein_4_tall.png
(species 8, species 9, Frankenstein 4), but every
synapse is drawn GRADED by its measured signal
flow instead of path = coloured, off-path = near
invisible gray.

Grading: for each synapse, rel = std of the flow
through it divided by the largest flow std among
the synapses feeding the same node (fan-in group),
taken from the per-synapse measurements saved in
frankenstein4_second_stitch_2026-11-26.json (all_edges:
mean/std/share per synapse; share there is
std / sum-of-std in the group, so rel is
recomputed as std / max-std in the group).
Line width = LW0 + (LW1 - LW0) * rel, so a synapse
at 40% of its node's winner is drawn at about 40%
of the full thickness (plus a small floor so no
real synapse is invisible). Path synapses keep
their territory colour (orange species 8, blue
species 9); off-path synapses are dark gray, the
shade darkening with rel. Tiny-random filler and
the joining block have no measured flow and stay
thin light gray. The merged decoder blocks keep
orange/blue for the two parents' blocks and light
gray for the cross entries.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
S2 = json.load(open(HERE / "frankenstein4_second_state.json"))
S4 = json.load(open(HERE / "frankenstein4_second_merged_state.json"))
R2 = json.load(open(HERE / "frankenstein4_second_stitch_2026-11-26.json"))
R4 = json.load(open(HERE / "frankenstein4_second_stitch_2026-11-26.json"))

C_HOST = "#e4572e"
C_DONOR = "#2166c7"
C_GRAY = "#c6cbce"
C_OFF_LO = "#a8b0b4"   # off-path gray at rel 0
C_OFF_HI = "#2b3033"   # off-path gray at rel 1
C_DOT = "#14666b"
C_DOT_D = "#6a4ba8"
C_DOT_J = "#8a939a"
C_PICK = "#444c50"
BG = "#f7f3ec"
LW0, LW1 = 0.55, 3.4   # line width at rel 0 and rel 1


def _lerp(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        round(a[k] + (b[k] - a[k]) * t) for k in range(3))


# ---------- measured per-synapse flow lookups ----------
def build_lookup(tag):
    """(entry, i, j) -> (on_path, rel) for one parent,
    rel = std / max std among edges feeding the
    same node (dst)."""
    edges = R2["parents"][tag]["all_edges"]
    by_dst = {}
    for e in edges:
        by_dst.setdefault(e["dst"], []).append(e)
    look = {}
    for e in edges:
        mx = max(x["std"] for x in by_dst[e["dst"]])
        rel = e["std"] / mx if mx > 0 else 0.0
        look[(e["entry"], e["i"], e["j"])] = \
            (bool(e["on_path"]), rel)
    return look


LOOK = {tag: build_lookup(tag)
        for tag in ("host", "donor")}


def column_positions(node_ids, x):
    m = len(node_ids)
    step = 0.55 if m >= 4 else (0.95 if m == 2 else 0.0)
    top = (m - 1) * step / 2.0
    return {nd: (x, top - i * step)
            for i, nd in enumerate(node_ids)}


def _style(cls, rel):
    """(color, lw, alpha, draw priority)."""
    if cls in ("h", "d"):
        col = C_HOST if cls == "h" else C_DONOR
        lw = 2.2 if rel is None else \
            LW0 + (LW1 - LW0) * rel
        return col, lw, 1.0, 2
    if cls == "o":
        r = 0.0 if rel is None else rel
        return _lerp(C_OFF_LO, C_OFF_HI, r), \
            LW0 + (LW1 - LW0) * r, 1.0, 1
    return C_GRAY, 0.65, 0.75, 0


def draw_panel_edges(ax, pos, edge_list):
    """edge_list: (src, dst, cls, rel); cls in
    'h', 'd' (path / territory block), 'o'
    (off-path, graded), 'f' (filler, no flow)."""
    styled = []
    for src, dst, cls, rel in edge_list:
        if src not in pos or dst not in pos:
            continue
        color, lw, alpha, prio = _style(cls, rel)
        styled.append((prio, lw, src, dst, color,
                       alpha))
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


def parent_panel(ax, P, tag, path_color_cls, title):
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
    look = LOOK[tag]
    edges = []
    n_hit = 0
    for e in P["edges"]:
        key = (e["entry"], e["i"], e["j"])
        if key in look:
            on, rel = look[key]
            n_hit += 1
        else:
            on, rel = bool(e["on_path"]), 0.0
        edges.append((e["src"], e["dst"],
                      path_color_cls if on else "o",
                      rel))
    print(f"parent {tag}: flow lookup matched "
          f"{n_hit} of {len(P['edges'])} edges")
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
    NH = len(h_map)
    inv_h = {v: k for k, v in h_map.items()}
    inv_d = {v: k for k, v in d_map.items()}
    inv_hu = {v: k for k, v in hu_map.items()}
    inv_du = {v: k for k, v in du_map.items()}
    stats = {"hit": 0, "miss": 0}

    def terr(fi):
        return "h" if fi < NH else "d"

    def old(fi):
        return inv_h[fi] if fi < NH else inv_d[fi]

    def rec(tag, entry, i, j):
        look = LOOK[tag]
        key = (entry, i, j)
        if key in look:
            stats["hit"] += 1
            return look[key]
        stats["miss"] += 1
        return (False, 0.0)

    def mat_style(entry, fi, fj):
        if terr(fi) != terr(fj):
            return ("f", None)
        t = terr(fi)
        on, rel = rec("host" if t == "h" else "donor",
                      entry, old(fi), old(fj))
        return (t, rel) if on else ("o", rel)

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
            on, rel = rec(
                "host" if t == "h" else "donor",
                "embed.W_in", c, old(fi))
            edges.append((f"in{c}", f"s0_{fi}",
                          *((t, rel) if on else
                            ("o", rel))))
    for nm, tg in (("W_q", "q"), ("W_k", "k"),
                   ("W_v", "v")):
        for fi in range(dF):
            for fj in range(dF):
                edges.append((f"s0_{fi}", f"{tg}_{fj}",
                              *mat_style(f"b0.{nm}",
                                         fi, fj)))
    # conduits + residuals: parent flow records
    cond = {}
    resid = {}
    for tag in ("host", "donor"):
        for e in R2["parents"][tag]["all_edges"]:
            r = LOOK[tag][(e["entry"], e["i"],
                           e["j"])]
            if e["kind"] == "conduit" and \
                    e["entry"] in ("attnQ", "attnK",
                                   "attnA"):
                cond[(tag, e["entry"], e["j"])] = r
            if e["kind"] == "residual":
                resid[(tag, e["entry"], e["i"])] = r
    for fj in range(dF):
        t = terr(fj)
        tag = "host" if t == "h" else "donor"
        for entry, src in (("attnQ", "q"),
                           ("attnK", "k"),
                           ("attnA", "v")):
            on, rel = cond.get((tag, entry, old(fj)),
                               (False, 0.0))
            edges.append((f"{src}_{fj}", f"mix_{fj}",
                          *((t, rel) if on else
                            ("o", rel))))
    for fi in range(dF):
        for fj in range(dF):
            edges.append((f"mix_{fi}", f"s1_{fj}",
                          *mat_style("b0.W_o",
                                     fi, fj)))
    for fi in range(dF):
        t = terr(fi)
        tag = "host" if t == "h" else "donor"
        on, rel = resid.get((tag, "residual1",
                             old(fi)), (False, 0.0))
        edges.append((f"s0_{fi}", f"s1_{fi}",
                      *((t, rel) if on else
                        ("o", rel))))
        on, rel = resid.get((tag, "residual2",
                             old(fi)), (False, 0.0))
        edges.append((f"s1_{fi}", f"s2_{fi}",
                      *((t, rel) if on else
                        ("o", rel))))
    for fi in range(dF):
        for ui in range(U):
            tu = "h" if ui < len(hu_map) else "d"
            if terr(fi) != tu:
                edges.append((f"s1_{fi}", f"ffn_{ui}",
                              "f", None))
            else:
                u_old = inv_hu[ui] if tu == "h" \
                    else inv_du[ui]
                on, rel = rec(
                    "host" if tu == "h" else "donor",
                    "b0.W_1", old(fi), u_old)
                edges.append((f"s1_{fi}", f"ffn_{ui}",
                              *((tu, rel) if on
                                else ("o", rel))))
    for ui in range(U):
        for fj in range(dF):
            tu = "h" if ui < len(hu_map) else "d"
            if terr(fj) != tu:
                edges.append((f"ffn_{ui}", f"s2_{fj}",
                              "f", None))
            else:
                u_old = inv_hu[ui] if tu == "h" \
                    else inv_du[ui]
                on, rel = rec(
                    "host" if tu == "h" else "donor",
                    "b0.W_2", u_old, old(fj))
                edges.append((f"ffn_{ui}", f"s2_{fj}",
                              *((tu, rel) if on
                                else ("o", rel))))
    print(f"frank block1: flow lookup {stats['hit']} "
          f"hits, {stats['miss']} misses")
    ctx = {"dF": dF, "U": U, "NH": NH, "terr": terr,
           "old": old, "hu_n": len(hu_map)}
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
        Line2D([0], [0], color=C_OFF_HI, lw=2.5,
               label="off-path synapses: real "
                     "synapses, thinner and lighter "
                     "the less signal they carry"),
        Line2D([0], [0], color=C_GRAY, lw=2,
               label="tiny-random filler "
                     "(Frankenstein; no measured "
                     "flow)"),
        Line2D([0], [0], color="none", lw=0,
               label="line thickness = measured "
                     "share of signal at the node "
                     "it feeds"),
    ] + extra
    fig.legend(handles=handles, loc="lower center",
               ncol=2, fontsize=12, frameon=False,
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
# Frankenstein 4 figure (merged decoder), graded flow
# ================================================================
fig, axes = make_fig()
parent_panel(axes[0], hostP, "host", "h", t_host)
parent_panel(axes[1], donorP, "donor", "d", t_donor)
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
            edges.append((f"s2_{fi}", f"{tg}_{fj}",
                          "f", None))
    for src in ("jq", "jk", "jv"):
        edges.append((f"{src}_{fi}", f"jmix_{fi}",
                      "f", None))
    for fj in range(dF):
        edges.append((f"jmix_{fi}", f"s3_{fj}",
                      "f", None))
    edges.append((f"s2_{fi}", f"s3_{fi}", "f", None))
    for ui in range(U2):
        edges.append((f"s3_{fi}", f"jffn_{ui}",
                      "f", None))
    edges.append((f"s3_{fi}", f"s4_{fi}", "f", None))
for ui in range(U2):
    for fj in range(dF):
        edges.append((f"jffn_{ui}", f"s4_{fj}",
                      "f", None))
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
        edges.append((f"s4_{fi}", f"dec_{fj}", cls,
                      None))
for fj in range(dF):
    edges.append((f"dec_{fj}", "pick", ctx["terr"](fj),
                  None))
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
gap4 = R4["frankenstein4_second"]["validation_gap_pct"]
finish_panel(ax, pos, labels,
             f"Second Frankenstein 4: ONE merged decoder "
             f"(both parents' blocks)\nstream {dF} "
             f"wide, FFN {U} + joining FFN {U2}, "
             f"gap {gap4:.2f}%")
ax.set_ylim(-ymax - 2.3, ymax + 1.5)
add_legend(fig, hostP["species"], donorP["species"], [])
fig.suptitle("Second Frankenstein 4: two corrected paths, "
             "joining layer, one merged decoder",
             fontsize=18, color="#333333",
             fontweight="bold", y=0.99)
out4 = HERE / "diagrams" / "frankenstein4_second_flow_tall.png"
fig.savefig(out4, facecolor=fig.get_facecolor(),
            bbox_inches="tight")
print("saved", out4)
plt.close(fig)
print("DONE render flow")
