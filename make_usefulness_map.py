#!/usr/bin/env python3
"""Usefulness map for one vaulted champion, SCALABLE layout (v2).

The v1 per-neuron tile wall does not scale past a few dozen units.
This version never draws one tile per neuron. It draws:
  1. density by zone (per block, per kind) as bars: O(zones) ink;
  2. unit strips: ONE row of small cells per block and kind, one
     cell per unit, coloured by its silencing drop. 200 units is
     one row of 200 cells; 2,000 units is still one row. Hover a
     cell for the unit's name and drop;
  3. the drop distribution as a histogram: the shape of the whole
     population of units in one small chart;
  4. the top-15 units by drop, and the dead/harmful counts. The
     long tail is summarised, never listed.

Silencing itself is unchanged: be.silence_table on the domain's
fixed validation set.

Usage: python3 make_usefulness_map.py [run_dir] [config_json] [out]
"""
import sys, json, html, dataclasses
import numpy as np
import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

run_dir = sys.argv[1] if len(sys.argv) > 1 else "engine_runs/test_minimal_a"
cfg_json = sys.argv[2] if len(sys.argv) > 2 else "bert_engine_config_test.json"
out_html = sys.argv[3] if len(sys.argv) > 3 else "genome_map_usefulness.html"

raw = json.load(open(cfg_json))
fields = {f.name for f in dataclasses.fields(be.EngineConfig)}
cfg = be.EngineConfig(**{k: v for k, v in raw.items() if k in fields})
backend = get_backend(cfg.backend, cfg.device, cfg.precision)
domain = tsp.TSPDomain(cfg, backend)

ck = np.load(f"{run_dir}/checkpoint.npz", allow_pickle=True)
arch = be.ArchSpec.from_signature(str(ck["arch_signature"]))
layout = be.Layout(arch, tsp.head_spec(arch.d))
g = np.asarray(np.load(f"{run_dir}/champion.npz")["genome"], dtype=float)

table = be.silence_table(
    layout, g,
    lambda G: domain.validation_scores(layout, G),
    zero_tol=domain.silence_zero_tol)
units = table["units"]
champ = table["champion_score"]

# ---- neutral band: paired per-problem noise -------------------
# The aggregate table alone cannot say whether a small drop is
# real: the validation score is a mean over n_val problems, so a
# drop inside the noise of that mean is NEUTRAL, neither
# contributing nor harmful. Paired per-problem gaps (same
# problems, champion vs silenced copy) give each unit its own
# standard error; the band is 2 SE (about 95%).
_unit_list = be.silence_units(layout)
G = np.tile(g, (len(_unit_list) + 1, 1))
for k, u in enumerate(_unit_list, start=1):
    G[k, be._silence_indices(layout, u)] = 0.0
_lens, _ = domain.evaluate_lengths(layout, G, domain.X_val)
_gaps = _lens / domain.optima - 1.0
assert abs(-_gaps[0].mean() - champ) < 1e-12
for k, u in enumerate(units):
    diffs = _gaps[k + 1] - _gaps[0]          # per-problem drop parts
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    u["se"] = se
    u["band"] = 2.0 * se
    u["cat"] = ("contributing" if u["drop"] > u["band"]
                else "harmful" if u["drop"] < -u["band"]
                else "neutral")
dmax = max((u["drop"] for u in units), default=0.0) or 1.0
alive = [u for u in units if u["cat"] == "contributing"]
neutral = [u for u in units if u["cat"] == "neutral"]
hurt = [u for u in units if u["cat"] == "harmful"]
density = len(alive) / len(units) if units else 0.0

def dcolour(u):
    drop = u["drop"]
    if u["cat"] == "contributing":
        t = min(1.0, drop / dmax)
        return f"rgb(255,{int(255 - 205 * t)},{int(255 - 205 * t)})"
    if u["cat"] == "harmful":
        return "rgb(170,220,170)"
    return "rgb(196,214,228)"

# ---- 1. density by zone ----
zones = {}
for u in units:
    z = (u["couple"], u["block"], u["kind"])
    zones.setdefault(z, []).append(u)
zone_rows = []
for (c, b, kind) in sorted(zones):
    us = zones[(c, b, kind)]
    a = sum(1 for u in us if u["cat"] == "contributing")
    n = sum(1 for u in us if u["cat"] == "neutral")
    h = sum(1 for u in us if u["cat"] == "harmful")
    share = a / len(us)
    zone_rows.append(
        f"<tr><td>couple {c}, block {b}</td><td>{kind}</td>"
        f"<td>{len(us)}</td><td>{a}</td><td>{n}</td><td>{h}</td>"
        f"<td><div class='bar'><div class='fill' "
        f"style='width:{share:.0%}'></div></div></td>"
        f"<td>{share:.0%}</td></tr>")
zones_html = "".join(zone_rows)

# ---- 2. unit strips: one row per zone, one cell per unit ----
strips = []
for (c, b, kind) in sorted(zones):
    us = sorted(zones[(c, b, kind)], key=lambda u: u["index"])
    cell = 12 if len(us) <= 90 else (8 if len(us) <= 200 else 5)
    cells = "".join(
        f"<rect x='{u['index']*cell}' y='0' width='{cell-1}' "
        f"height='16' fill='{dcolour(u)}' stroke='#ccc' "
        f"stroke-width='0.4'><title>{html.escape(u['name'])}: "
        f"drop {u['drop']:+.4f}, {u['cat']} "
        f"(band +/-{u['band']:.4f})</title></rect>"
        for u in us)
    strips.append(
        f"<div class='strip'><div class='slabel'>couple {c} block {b} "
        f"{kind} ({len(us)} units)</div>"
        f"<svg width='{len(us)*cell}' height='16'>{cells}</svg></div>")
strips_html = "".join(strips)

# ---- 3. drop histogram ----
drops = np.array([u["drop"] for u in units])
lo = min(drops.min(), -0.01)
hi = max(drops.max(), 0.01)
bins = np.linspace(lo, hi, 25)
counts, _ = np.histogram(drops, bins=bins)
cmax = counts.max() or 1
bw, bh = 22, 90
bars = []
for i, ct in enumerate(counts):
    x = i * (bw + 2)
    hgt = int(bh * ct / cmax)
    zero = bins[i] <= 0 <= bins[i + 1]
    fillc = "#c0392b" if bins[i] >= 0 else "#7a7a7a"
    if zero:
        fillc = "#444"
    bars.append(
        f"<rect x='{x}' y='{bh - hgt}' width='{bw}' height='{hgt}' "
        f"fill='{fillc}'><title>drop in [{bins[i]:+.3f}, "
        f"{bins[i+1]:+.3f}): {ct} units</title></rect>")
hist_html = (f"<svg width='{24*(bw+2)}' height='{bh + 4}'>"
             f"{''.join(bars)}</svg>"
             f"<div class='mnote'>x: silencing drop from {lo:+.3f} to "
             f"{hi:+.3f}; y: number of units. Grey left of zero: units "
             f"whose silencing helps or does nothing. Red right of "
             f"zero: units carrying the network.</div>")

# ---- 4. top 15 ----
ranked = sorted(units, key=lambda u: -u["drop"])
rows = "".join(
    f"<tr><td>{i+1}</td><td>{html.escape(u['name'])}</td>"
    f"<td>{u['kind']}</td><td style='text-align:right'>"
    f"{u['drop']:+.4f}</td></tr>"
    for i, u in enumerate(ranked[:15]))
top_html = rows

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Usefulness map</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 32px; }}
.small {{ font-size: 12px; color: #444; max-width: 920px; }}
.big {{ font-size: 19px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #ccc; padding: 3px 8px; }}
.bar {{ width: 140px; height: 10px; background: #eee; border: 1px solid #bbb; }}
.fill {{ height: 100%; background: #c0392b; }}
.strip {{ margin: 8px 0; }}
.slabel {{ font: 11px ui-monospace, monospace; margin-bottom: 2px; }}
.mnote {{ font-size: 10px; color: #666; margin-top: 3px; max-width: 640px; }}
</style></head><body>
<h1>Usefulness map: champion of {html.escape(run_dir)}</h1>
<p class="small">Each unit (one feed-forward neuron or one whole attention
head) had its outgoing weights zeroed, one at a time, and the champion was
re-scored on the same fixed validation set of {cfg.n_val} problems. The
<b>drop</b> is champion score minus silenced score. Champion score
{champ:+.4f}; {len(units)} units measured in one batched pass. This layout
scales: nothing on the page is drawn per neuron except one small cell in a
strip, so a network with thousands of units produces a page of the same
height.</p>
<p class="big"><b>Information density: {len(alive)} of {len(units)} units
contribute ({density:.0%})</b>; {len(neutral)} are neutral and
{len(hurt)} actively hurt (silencing them improves the score). The
sponsor's bar for dense enough is above 50%.</p>
<p class="small"><b>Neutral is a measured category, not a rounding
bucket.</b> The validation score is a mean over {cfg.n_val} problems, so
every drop carries noise. Each unit's drop was also computed problem by
problem (same problems, champion against silenced copy), giving it a
standard error; a unit is neutral when its drop sits within 2 standard
errors of zero, meaning this test set cannot distinguish its effect
from nothing. Contributing means the drop clears that band upward,
harmful means it clears it downward. The band is per unit, typically
about +/-{float(np.median([u['band'] for u in units])):.4f} here.</p>

<h2>1. Density by zone</h2>
<table><tr><th>zone</th><th>kind</th><th>units</th><th>contributing</th>
<th>neutral</th><th>harmful</th><th></th><th>density</th></tr>{zones_html}</table>

<h2>2. Unit strips: every unit, one cell each</h2>
<p class="small">Red: contributing, deeper is a bigger drop. Blue-grey:
neutral, no measurable effect. Green: harmful, the network scores better
without it. Hover any cell for the unit's name, exact drop, category and
its neutral band.</p>
{strips_html}

<h2>3. The shape of it: distribution of drops</h2>
{hist_html}

<h2>4. The carriers: top 15 units by drop</h2>
<table><tr><th>#</th><th>unit</th><th>kind</th><th>drop</th></tr>
{top_html}</table>
<p class="small">The remaining {max(0, len(units) - 15)} units are
summarised in the strips and the histogram, not listed: at production
sizes a full ranked table is a phone book, and the decisions (what to
protect, what to prune, where density is low) read off the zones and the
distribution.</p>

<p class="small">Scope notes. This is the per-UNIT read, the confirmed
one. A per-WEIGHT usefulness score also exists in the engine, accumulated
from the population's history during a run; this champion's run predates
that machinery, so its per-weight scores do not exist and cannot be
rebuilt after the fact. Any run launched now records them from
generation one, and the monitor plots density as the run goes. The
neutral band above is a reporting read (measurement confidence on this
validation set); the engine's own mutation marking keeps its separate,
exact criterion for which units count as useless, and the two should
not be conflated.</p>
</body></html>"""

with open(out_html, "w") as f:
    f.write(page)
print(f"wrote {out_html}: {len(page)} bytes")
print(f"champion {champ:+.4f}; units {len(units)}; "
      f"contributing {len(alive)}; neutral {len(neutral)}; "
      f"harmful {len(hurt)}; density {density:.1%}")
for u in sorted(units, key=lambda u: -u["drop"])[:3]:
    print("  top:", u["name"], f"{u['drop']:+.4f}", u["cat"])
for u in sorted(units, key=lambda u: u["drop"])[:3]:
    print("  bottom:", u["name"], f"{u['drop']:+.4f}", u["cat"])
