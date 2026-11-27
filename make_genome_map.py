#!/usr/bin/env python3
"""Genome map: one vaulted champion drawn three ways.

1. Whole genome as one labelled grid (every gene, coloured by weight).
2. Every named weight matrix in its true shape, grouped by component.
3. Per-neuron clusters: each FFN hidden unit's incoming weights, bias
   and outgoing weights as one tile; each attention head's four
   matrices as one tile.

Usage: python3 make_genome_map.py [run_dir] [out_html]
Defaults: engine_runs/test_minimal_a -> genome_map.html
"""
import sys, json, html
import numpy as np
import bert_engine as be
import bert_domain_tsp as tsp

run_dir = sys.argv[1] if len(sys.argv) > 1 else "engine_runs/test_minimal_a"
out_html = sys.argv[2] if len(sys.argv) > 2 else "genome_map.html"

ck = np.load(f"{run_dir}/checkpoint.npz", allow_pickle=True)
arch = be.ArchSpec.from_signature(str(ck["arch_signature"]))
layout = be.Layout(arch, tsp.head_spec(arch.d))
ch = np.load(f"{run_dir}/champion.npz")
g = np.asarray(ch["genome"], dtype=float)
val = float(ch["val_score"])
assert len(g) == layout.n_params, (len(g), layout.n_params)
parts = layout.unpack_dict(g)

vmax = float(np.percentile(np.abs(g), 99.5)) or 1.0

def colour(v):
    """Diverging: negative steel blue, zero near-white, positive brick."""
    t = max(-1.0, min(1.0, v / vmax))
    if t >= 0:
        r, gg, b = 255, int(255 - 150 * t), int(255 - 190 * t)
    else:
        r, gg, b = int(255 + 190 * t), int(255 - 150 * (-t)), 255
    return f"rgb({r},{gg},{b})"

def grid(mat, cell=14, title=None, note=None):
    """One matrix as an SVG grid. 1-D arrays drawn as a single row."""
    a = np.asarray(mat, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    rows, cols = a.shape
    w, h = cols * cell, rows * cell
    out = [f'<svg width="{w}" height="{h}" '
           f'style="border:1px solid #999;display:block">']
    for i in range(rows):
        for j in range(cols):
            out.append(
                f'<rect x="{j*cell}" y="{i*cell}" width="{cell-1}" '
                f'height="{cell-1}" fill="{colour(a[i, j])}">'
                f'<title>{a[i, j]:+.4f}</title></rect>')
    out.append("</svg>")
    head = f"<div class='mtitle'>{html.escape(title)}</div>" if title else ""
    tail = f"<div class='mnote'>{html.escape(note)}</div>" if note else ""
    return f"<div class='mat'>{head}{''.join(out)}{tail}</div>"

# ---------- 1. whole-genome overview, coloured by segment ----------
SEG_COLOURS = {}
palette = ["#f4a259", "#8ecae6", "#b565d8", "#90be6d", "#f94144",
           "#43aa8b", "#577590", "#e9c46a"]
seg_of_gene = np.empty(layout.n_params, dtype=object)
for ci, segs in enumerate(layout.segments):
    for si, (sname, idx) in enumerate(segs):
        SEG_COLOURS[sname] = palette[(ci * 3 + si) % len(palette)]
        seg_of_gene[idx] = sname
# the domain head is not part of layout.segments; label it directly
for name, sl, sh in layout.entries:
    if ".head." in name:
        sname = name.split(".head.")[0] + ".head"
        if sname not in SEG_COLOURS:
            SEG_COLOURS[sname] = palette[len(SEG_COLOURS) % len(palette)]
        seg_of_gene[sl] = sname
assert all(s is not None for s in seg_of_gene)
COLS = 47
cell = 13
rows = int(np.ceil(layout.n_params / COLS))
ov = [f'<svg width="{COLS*cell}" height="{rows*cell}" '
      f'style="border:1px solid #999;display:block">']
for k, v in enumerate(g):
    i, j = divmod(k, COLS)
    ov.append(
        f'<rect x="{j*cell}" y="{i*cell}" width="{cell-1}" '
        f'height="{cell-1}" fill="{colour(v)}" '
        f'stroke="{SEG_COLOURS[seg_of_gene[k]]}" stroke-width="1.6">'
        f'<title>gene {k} ({seg_of_gene[k]}): {v:+.4f}</title></rect>')
ov.append("</svg>")
legend = "".join(
    f"<span class='chip' style='background:{c}'></span>{html.escape(n)} "
    for n, c in SEG_COLOURS.items())

# ---------- 2. matrices by component ----------
comp_html = []
for name, sl, sh in layout.entries:
    comp_html.append(grid(parts[name], cell=13, title=f"{name}  {sh}"))
matrices_html = "\n".join(comp_html)

# ---------- 3. per-neuron clusters ----------
d, H = arch.d, arch.ffn_hidden
neuron_html = []
for c, (n_blocks, heads) in enumerate(arch.couples):
    for b in range(n_blocks):
        p = f"c{c}.b{b}"
        tiles = []
        for u in range(H):
            tile = [f"<div class='mtitle'>FFN unit {u}</div>",
                    "<div class='row'>",
                    grid(parts[f"{p}.W_1"][:, u], cell=11),
                    f"<div class='bias'>bias<br>{parts[f'{p}.b_1'][u]:+.3f}</div>",
                    grid(parts[f"{p}.W_2"][u, :], cell=11),
                    "</div>",
                    f"<div class='mnote'>in (left, top to bottom) "
                    f"out (right, left to right)</div>"]
            tiles.append(f"<div class='mat'>{''.join(tile)}</div>")
        neuron_html.append(
            f"<h3>{p} feed-forward neurons: what flows in, what flows out</h3>"
            f"<div class='flex'>{''.join(tiles)}</div>")
        htiles = []
        for hd in range(heads):
            hp = f"{p}.h{hd}"
            tile = [f"<div class='mtitle'>attention head {hd}</div>",
                    "<div class='flex'>"]
            for nm in ("W_q", "W_k", "W_v", "W_o"):
                tile.append(grid(parts[f"{hp}.{nm}"], cell=9, title=nm))
            tile.append("</div>")
            htiles.append(f"<div class='mat'>{''.join(tile)}</div>")
        neuron_html.append(
            f"<h3>{p} attention heads: the four matrices of each head</h3>"
            f"<div class='flex'>{''.join(htiles)}</div>")
neurons_html = "\n".join(neuron_html)

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Genome map</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a;
       background: #fdfdfd; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 34px; }}
h3 {{ font-size: 14px; margin-top: 22px; }}
.flex {{ display: flex; flex-wrap: wrap; gap: 14px; }}
.mat {{ background: #fff; padding: 6px; }}
.mtitle {{ font: 11px ui-monospace, monospace; margin-bottom: 4px; }}
.mnote {{ font-size: 10px; color: #666; margin-top: 3px; max-width: 240px; }}
.row {{ display: flex; gap: 8px; align-items: flex-start; }}
.bias {{ font: 11px ui-monospace, monospace; text-align: center;
        padding-top: 8px; }}
.chip {{ display: inline-block; width: 11px; height: 11px; margin: 0 3px 0 10px;
        border: 1px solid #888; vertical-align: -1px; }}
.small {{ font-size: 12px; color: #444; max-width: 900px; }}
</style></head><body>
<h1>Genome map: champion of {html.escape(run_dir)}</h1>
<p class="small">Travelling-salesman side. Network shape: stream width d {arch.d},
feed-forward width {arch.ffn_hidden}, couples {list(arch.couples)} (blocks, heads),
<b>{layout.n_params} genes</b> in total. Validation score {val:+.4f}
(the score is a tour-length gap figure; sign as stored).
Every square is one gene (one weight or bias). Colour is the weight value:
blue negative, white near zero, red positive, scaled to the 99.5th percentile
(|w| = {vmax:.3f}). Hover any square for its exact value.</p>

<h2>1. The whole genome in one grid</h2>
<p class="small">Genes in genome order, left to right, top to bottom. The
coloured outline of each square names the component the gene belongs to:
{legend}</p>
{''.join(ov)}

<h2>2. Every weight matrix in its true shape</h2>
<p class="small">The same genes, regrouped: each named matrix drawn at its
real dimensions, biases as single rows.</p>
<div class="flex">{matrices_html}</div>

<h2>3. Per-neuron clusters</h2>
<p class="small">Each feed-forward neuron owns one column of the incoming
matrix (what it listens to), one bias (how easily it fires) and one row of
the outgoing matrix (what it says, and how loudly, to each stream position).
Each attention head owns four square matrices: what it asks for (W_q), what
it matches against (W_k), what it carries (W_v) and what it writes back
(W_o).</p>
{neurons_html}
</body></html>"""

with open(out_html, "w") as f:
    f.write(page)
print(f"wrote {out_html}: {len(page)} bytes, {layout.n_params} genes, "
      f"{len(layout.entries)} named arrays")
