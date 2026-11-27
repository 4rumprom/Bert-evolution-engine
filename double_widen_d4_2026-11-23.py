#!/usr/bin/env python3
"""DOUBLE-WIDEN test on the 11.85% champion of the new
single-layer lineage (2026-11-23). The branch-graft
test found this champion has NO cleared space (every
unit exact-zero positive), so growth is the indicated
next step. Same default widen (Roman, 2026-11-23):
d x2, FFN x2, heads x2, Net2WiderNet splits.
  - stream d 4 -> 8 (engine transplant_widen, step 4);
  - FFN 16 -> 32 by split (duplicate W_1 column and
    b_1, halve W_2 rows in both copies);
  - heads 2 -> 4 by split (Q/K/V copied, W_o and b_o
    halved in both copies).
SUBJECT: the prune-graft round's control champion
(prune_graft_widened_2026-11-23_control_champion.npz,
11.85% mean gap; d 4, FFN 16, 2 heads, 360 genes, 18
units). Arms: treatment 100 generations evolving
ONLY the added material (sponsor rule 2026-11-23:
growth modifies only what it added; the base stays
100% the champion it grew from, so there is NO
whole-network phase in a growth step) vs a
same-budget plain control (the subject evolved
wholly, frozen units excepted), freeze marks = the
subject's band-contributing units (both split copies
frozen).

    cd ~/workspace/bert-engine && \
    setsid nohup python3 double_widen_d4_2026-11-23.py \
        >> double_widen_d4_2026-11-23.log 2>&1 < /dev/null &
"""
import dataclasses
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cw = _load("cw", "coupled_widen_d2_2026-11-23.py")
dw = _load("dw", "double_widen_d2_2026-11-23.py")
pg = _load("pg", "prune_graft_d2_l1_2026-11-23.py")

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
SUBJECT_PATH = HERE / "prune_graft_widened_2026-11-23_" \
    "control_champion.npz"
OUT_JSON = HERE / "double_widen_d4_2026-11-23.json"
STEP_SEEDS = {"run_seed": 20262500, "draw_seed": 20262501}
SUBJECT_STORED_VAL = -0.118467


def log(msg):
    print(f"[dw-d4] {msg}", flush=True)


def build_double_transplant(arch0, layout0, subject, cfg,
                            marked_ffn, marked_heads):
    """d-widen (x2) via the engine, then FFN split and
    head split, Net2WiderNet style. marked_* lists the
    subject units whose split daughters keep the old
    classes (they are frozen by mask in any case)."""
    arch1, layout1, template1, classes1 = be.make_transplant(
        "widen", arch0, layout0, subject, tsp.head_spec, cfg,
        tsp.widen_head, variant="A")
    assert arch1.d == 2 * arch0.d
    n_heads1 = arch1.couples[0][1]
    couples2 = tuple((b, 2 * h) for b, h in arch1.couples)
    arch2 = be.ArchSpec(input_dim=arch1.input_dim, d=arch1.d,
                        ffn_hidden=2 * arch1.ffn_hidden,
                        couples=couples2, ln_eps=arch1.ln_eps,
                        attn_scale=arch1.attn_scale)
    layout2 = be.Layout(arch2, tsp.head_spec(arch2.d))
    d1 = layout1.unpack_dict(template1)
    d2 = {n: np.zeros(sh) for n, _, sh in layout2.entries}
    classes2 = np.zeros(layout2.n_params, dtype=np.int8)
    n_ffn = arch1.ffn_hidden
    marked = set(marked_ffn)
    for name, _, shape in layout2.entries:
        s2 = layout2.sl(name)
        if name.endswith(".W_1"):
            old = d1[name]
            d2[name] = np.concatenate([old, old], axis=1)
            c_old = classes1[layout1.sl(name)].reshape(
                old.shape)
            c_new = np.concatenate([c_old, c_old], axis=1)
            for j in range(n_ffn, 2 * n_ffn):
                if (j - n_ffn) not in marked:
                    c_new[:, j] = be.CLS_IFACE
            classes2[s2] = c_new.reshape(-1)
        elif name.endswith(".b_1"):
            old = d1[name]
            d2[name] = np.concatenate([old, old])
            c_old = classes1[layout1.sl(name)].copy()
            c_new = np.concatenate([c_old, c_old])
            for j in range(n_ffn, 2 * n_ffn):
                if (j - n_ffn) not in marked:
                    c_new[j] = be.CLS_IFACE
            classes2[s2] = c_new
        elif name.endswith(".W_2"):
            old = d1[name]
            d2[name] = np.concatenate(
                [old / 2.0, old / 2.0], axis=0)
            c_old = classes1[layout1.sl(name)].reshape(
                old.shape)
            c_new = np.concatenate([c_old, c_old], axis=0)
            for j in range(n_ffn, 2 * n_ffn):
                if (j - n_ffn) not in marked:
                    c_new[j, :] = be.CLS_IFACE
            classes2[s2] = c_new.reshape(-1)
        elif re.search(r"\.h\d+\.", name) and name not in d1:
            # new head hH: copy of source head h(H - n_heads1)
            h_new = int(name.split(".h")[1].split(".")[0])
            h_src = h_new - n_heads1
            src = re.sub(r"\.h\d+\.", f".h{h_src}.", name,
                         count=1)
            arr = d1[src].copy()
            if name.endswith(".W_o") or name.endswith(".b_o"):
                arr = arr / 2.0
            d2[name] = arr
            classes2[s2] = classes1[layout1.sl(src)]
        elif name in d1 and d1[name].shape == shape:
            arr = d1[name]
            m = re.search(r"\.h(\d+)\.", name)
            if m and int(m.group(1)) in marked_heads and (
                    name.endswith(".W_o")
                    or name.endswith(".b_o")):
                arr = arr / 2.0        # old head's half
            d2[name] = arr
            classes2[s2] = classes1[layout1.sl(name)]
        else:
            raise ValueError(f"unhandled entry {name} {shape}")
    template2 = layout2.pack_dict(d2)
    return arch1, layout1, template1, classes1, \
        arch2, layout2, template2, classes2


def audit_split(layout1, template1, layout2, template2,
                n_ffn, n_heads):
    d1 = layout1.unpack_dict(template1)
    d2 = layout2.unpack_dict(template2)
    checks = {}
    checks["ffn_W1_daughters_exact"] = bool(np.array_equal(
        d2["c0.b0.W_1"][:, n_ffn:],
        d2["c0.b0.W_1"][:, :n_ffn]))
    checks["ffn_W2_halved"] = bool(np.allclose(
        d2["c0.b0.W_2"][:n_ffn], d1["c0.b0.W_2"] / 2.0)
        and np.allclose(d2["c0.b0.W_2"][n_ffn:],
                        d1["c0.b0.W_2"] / 2.0))
    ok = True
    for h in range(n_heads):
        ok &= bool(np.array_equal(
            d2[f"c0.b0.h{h + n_heads}.W_q"],
            d1[f"c0.b0.h{h}.W_q"]))
        ok &= bool(np.allclose(
            d2[f"c0.b0.h{h}.W_o"],
            d1[f"c0.b0.h{h}.W_o"] / 2.0))
        ok &= bool(np.allclose(
            d2[f"c0.b0.h{h + n_heads}.W_o"],
            d1[f"c0.b0.h{h}.W_o"] / 2.0))
    checks["heads_split"] = ok
    checks["all_ok"] = bool(all(checks.values()))
    return checks


def main():
    t_start = time.time()
    cfg = dataclasses.replace(
        be.EngineConfig.load(CFG_PATH), widen_step=4)
    backend = get_backend(cfg.backend, cfg.device,
                          cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    # the subject's own layout is rebuilt through the same
    # widen chain that produced it (attn_scale and entry
    # order must match exactly; a hand-built ArchSpec
    # gets the scale wrong)
    ck = np.load(HERE / "engine_runs" / "fresh_d2_l1" /
                 "checkpoint.npz", allow_pickle=True)
    arch_d2 = be.ArchSpec.from_signature(
        str(ck["arch_signature"]))
    layout_d2 = be.Layout(arch_d2, tsp.head_spec(arch_d2.d))
    pg_subject = np.asarray(
        np.load(HERE / "prune_graft_d2_l1_2026-11-23_"
                       "treatment_champion.npz")["genome"],
        dtype=float)
    cfg_w = dataclasses.replace(cfg, widen_step=2)
    _, _, _, _, arch0, layout0, template2_old, _ = \
        dw.build_double_transplant(arch_d2, layout_d2,
                                   pg_subject, cfg_w)
    champ_z = np.load(SUBJECT_PATH)
    subject = np.asarray(champ_z["genome"], dtype=float)
    stored_val = float(champ_z["val_score"])
    classes0 = be._base_classes(layout0, 0)

    results = {
        "experiment": "double-widen test on the 11.85% "
                      "champion (default widen: d x2, "
                      "FFN x2, heads x2, Net2WiderNet "
                      "splits)",
        "date": "2026-11-23",
        "subject": {"file": str(SUBJECT_PATH),
                    "stored_val_score": stored_val,
                    "n_genes": int(layout0.n_params)},
        "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
        "growth_rule": "sponsor 2026-11-23: a growth "
                       "step modifies ONLY the added "
                       "material; the treatment arm runs "
                       "100 generations on the phase-A "
                       "prob vector (new-material genes "
                       "only); no whole-network phase in "
                       "this step",
    }

    repro_val = domain.validation_score(layout0, subject)
    assert abs(repro_val - stored_val) < 1e-9
    assert abs(stored_val - SUBJECT_STORED_VAL) < 1e-6
    log(f"subject val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% gap)")

    before = dw.cloud_classify(cw.density_read(
        domain, layout0, subject))
    results["before"] = before
    log(f"BEFORE: band {before['band_counts']}; cloud "
        f"{before['cloud']['counts']}")

    useful = [u for u in before["units"]
              if u["cat"] == "contributing"]
    marked_ffn = sorted(int(re.search(r"\.u(\d+)$",
                                      u["name"]).group(1))
                        for u in before["units"]
                        if u["cat"] == "contributing"
                        and u["kind"] == "ffn")
    marked_heads = sorted(int(re.search(r"\.h(\d+)$",
                                       u["name"]).group(1))
                          for u in before["units"]
                          if u["cat"] == "contributing"
                          and u["kind"] == "head")
    frozen0 = cw.frozen_mask_from_names(
        layout0, [u["name"] for u in useful])
    results["freeze_prep"] = {
        "useful_units": [u["name"] for u in useful],
        "n_frozen_genes": int(frozen0.sum())}
    log(f"FREEZE prep: {[u['name'] for u in useful]}; "
        f"{int(frozen0.sum())} genes")

    arch1, layout1, template1, classes1, arch2, layout2, \
        template2, classes2 = build_double_transplant(
            arch0, layout0, subject, cfg, marked_ffn,
            marked_heads)
    pres = domain.preservation(layout0, subject, layout2,
                               template2)
    audit = audit_split(layout1, template1, layout2,
                        template2, arch1.ffn_hidden,
                        arch1.couples[0][1])
    n_ffn2 = arch2.ffn_hidden
    frozen_names2 = \
        [f"c0.b0.ffn.u{j}" for j in marked_ffn] \
        + [f"c0.b0.ffn.u{j + arch1.ffn_hidden}"
           for j in marked_ffn] \
        + [f"c0.b0.head.h{h}"
           for h in range(2 * arch1.couples[0][1])]
    frozen2 = cw.frozen_mask_from_names(layout2,
                                        frozen_names2)
    results["transplant"] = {
        "d_old": int(arch0.d), "d_new": int(arch2.d),
        "ffn_old": int(arch0.ffn_hidden),
        "ffn_new": int(arch2.ffn_hidden),
        "heads_old": 2, "heads_new": 4,
        "n_genes_old": int(layout0.n_params),
        "n_genes_new": int(layout2.n_params),
        "n_units_old": int(before["n_units"]),
        "n_units_new": int(len(be.silence_units(layout2))),
        "preservation_gate": pres,
        "split_audit": audit,
        "n_frozen_genes_carried": int(frozen2.sum()),
        "arch_signature_new": layout2.arch.signature()}
    log(f"TRANSPLANT: d {arch0.d} -> {arch2.d}, FFN "
        f"{arch0.ffn_hidden} -> {arch2.ffn_hidden}, heads "
        f"2 -> 4, {layout0.n_params} -> "
        f"{layout2.n_params} genes, {before['n_units']} -> "
        f"{len(be.silence_units(layout2))} units")
    log(f"GATE: {'PASS' if pres['pass'] else 'REJECT'} "
        f"(max abs diff "
        f"{pres['max_abs_length_difference']:.3g}, tours "
        f"identical {pres['tours_identical']}); split audit "
        f"{audit['all_ok']}; frozen genes carried "
        f"{int(frozen2.sum())}")
    if not pres["pass"] or not audit["all_ok"]:
        results["verdict"] = "gate-rejected; arms not run"
        results["wall_seconds"] = time.time() - t_start
        OUT_JSON.write_text(json.dumps(results, indent=1))
        log("DONE (gate-rejected)")
        print("DOUBLE_WIDEN_D4_DONE")
        return

    pv_t = cw.prob_vectors(classes2, cfg, frozen2)
    # sponsor growth rule 2026-11-23: a growth step
    # modifies ONLY the added material. The treatment
    # arm therefore runs 100 generations on the phase-A
    # prob vector (new-material genes only); there is
    # no whole-network phase in this step.
    pv_t_growth = {"A": pv_t["A"], "B": pv_t["A"]}
    pv_c = cw.prob_vectors(classes0, cfg, frozen0)

    rng_t = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_t = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_t = be.create_species(
        template2, classes2, cfg.n_species, cfg.per_species,
        draw_rng_t, rng_t, pv_t["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_t = cw.evolve(domain, layout2, cfg, classes2,
                      species_t, rng_t, pv_t_growth)
    wall_t = time.time() - t0
    after_t = dw.cloud_classify(cw.density_read(
        domain, layout2, out_t["champion_genome"]))
    results["treatment"] = cw.arm_block(out_t, after_t,
                                        repro_val, wall_t)
    results["treatment"]["density_cloud_counts"] = \
        after_t["cloud"]["counts"]
    np.savez(HERE / "double_widen_d4_2026-11-23_"
                    "treatment_champion.npz",
             genome=out_t["champion_genome"],
             val_score=out_t["champion_val_score"])
    log(f"TREATMENT done: {out_t['champion_val_score']:.6f} "
        f"({-100.0 * out_t['champion_val_score']:.2f}% gap, "
        f"{wall_t:.1f}s); band {after_t['band_counts']}; "
        f"cloud {after_t['cloud']['counts']}")

    rng_c = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_c = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_c = be.create_species(
        subject, classes0, cfg.n_species, cfg.per_species,
        draw_rng_c, rng_c, pv_c["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_c = cw.evolve(domain, layout0, cfg, classes0,
                      species_c, rng_c, pv_c)
    wall_c = time.time() - t0
    after_c = dw.cloud_classify(cw.density_read(
        domain, layout0, out_c["champion_genome"]))
    results["control"] = cw.arm_block(out_c, after_c,
                                      repro_val, wall_c)
    results["control"]["density_cloud_counts"] = \
        after_c["cloud"]["counts"]
    np.savez(HERE / "double_widen_d4_2026-11-23_"
                    "control_champion.npz",
             genome=out_c["champion_genome"],
             val_score=out_c["champion_val_score"])
    log(f"CONTROL done: {out_c['champion_val_score']:.6f} "
        f"({-100.0 * out_c['champion_val_score']:.2f}% gap, "
        f"{wall_c:.1f}s); band {after_c['band_counts']}; "
        f"cloud {after_c['cloud']['counts']}")

    results["margin_treatment_over_control"] = cw.margins(
        out_t["champion_val_score"],
        out_c["champion_val_score"])
    results["wall_seconds"] = time.time() - t_start
    OUT_JSON.write_text(json.dumps(results, indent=1))
    m = results["margin_treatment_over_control"]
    log(f"MARGIN: {m['points']:+.6f} points, "
        f"{m['relative_percent_of_control_gap']:+.2f}% "
        f"relative to the control's gap")
    log(f"DONE in {results['wall_seconds']:.1f}s")
    print("DOUBLE_WIDEN_D4_DONE")


if __name__ == "__main__":
    main()
