#!/usr/bin/env python3
"""DOUBLE-WIDEN test on the new single-layer lineage
(2026-11-23). Sponsor revision of the coupled-widen test
before it ran: the widen step DOUBLES everything, and
that is the default from now on:
  - stream d 2 -> 4 (engine transplant_widen, step 2);
  - FFN 8 -> 16 by Net2WiderNet split: every unit is
    duplicated (W_1 column and b_1 copied exactly) and
    its outgoing W_2 row is halved in both copies, so
    the function is preserved exactly;
  - heads 1 -> 2 by the same split: the head's
    Q/K/V copied exactly, W_o and b_o halved in both
    copies (head outputs are summed in the forward
    pass, verified in bert_engine.py, so the sum is
    preserved exactly).
Subject: the prune-graft treatment champion (18.80%
gap, 88 genes), same arms as the coupled-widen test:
treatment 50 + 50 generations (new material alone,
then whole network) vs a same-budget plain control,
same freeze marks (band-contributing units of the
subject: FFN u3-u7 and head h0; in the widened layout
their split daughters are frozen too, both copies at
mutation rate 0). Density reads report BOTH the
paired-noise band categories and the sponsor's
near-zero cloud classification (2026-11-23 17:34):
drops as relative % of the champion's gap, cloud =
|drop| <= 2 robust sigmas (MAD) of the drops, outside
positive = contributing, outside negative = harmful.

    cd ~/workspace/bert-engine && \
    setsid nohup python3 double_widen_d2_2026-11-23.py \
        >> double_widen_d2_2026-11-23.log 2>&1 < /dev/null &
"""
import dataclasses
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import bert_engine as be
import bert_domain_tsp as tsp
from bert_backend import get_backend

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cw", HERE / "coupled_widen_d2_2026-11-23.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)

CFG_PATH = HERE / "bert_engine_config_d2_l1.json"
SUBJECT_PATH = HERE / "prune_graft_d2_l1_2026-11-23_" \
    "treatment_champion.npz"
OUT_JSON = HERE / "double_widen_d2_cloudmarks_2026-11-23.json"
STEP_SEEDS = {"run_seed": 20262200, "draw_seed": 20262201}
MARKED_FFN = [4]          # subject CLOUD read: only u4
MARKED_HEADS = [0]


def log(msg):
    print(f"[dw-cm] {msg}", flush=True)


def cloud_classify(read):
    """Sponsor's near-zero cloud classification on top of
    a density read: rel drops (% of the champion's gap),
    cloud half-width = 2 * 1.4826 * MAD of the rel drops."""
    gap = -read["champion_score"]
    rel = np.array([100.0 * u["drop"] / gap
                    for u in read["units"]])
    med = float(np.median(rel))
    mad = float(np.median(np.abs(rel - med)))
    sigma = 1.4826 * mad
    lim = 2.0 * sigma
    counts = {"contributing": 0, "neutral": 0, "harmful": 0}
    for u, r in zip(read["units"], rel):
        u["rel_drop_pct"] = float(r)
        u["cloud_cat"] = ("neutral" if abs(r) <= lim
                          else "contributing" if r > 0
                          else "harmful")
        counts[u["cloud_cat"]] += 1
    read["cloud"] = {"limit_pct": float(lim),
                     "sigma_pct": float(sigma),
                     "counts": counts}
    return read


def build_double_transplant(arch0, layout0, subject, cfg):
    """d-widen (step 2) via the engine, then FFN split
    8 -> 16 and head split 1 -> 2, Net2WiderNet style."""
    arch1, layout1, template1, classes1 = be.make_transplant(
        "widen", arch0, layout0, subject, tsp.head_spec, cfg,
        tsp.widen_head, variant="A")
    assert arch1.d == 2 * arch0.d
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
    marked = set(MARKED_FFN)
    for name, _, shape in layout2.entries:
        s2 = layout2.sl(name)
        if name.endswith(".W_1"):
            old = d1[name]
            new = np.concatenate([old, old], axis=1)
            d2[name] = new
            c_old = classes1[layout1.sl(name)].reshape(old.shape)
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
            new = np.concatenate([old / 2.0, old / 2.0], axis=0)
            d2[name] = new
            c_old = classes1[layout1.sl(name)].reshape(old.shape)
            c_new = np.concatenate([c_old, c_old], axis=0)
            for j in range(n_ffn, 2 * n_ffn):
                if (j - n_ffn) not in marked:
                    c_new[j, :] = be.CLS_IFACE
            classes2[s2] = c_new.reshape(-1)
        elif ".h" in name and name not in d1:
            # the new head h1: exact copy of h0 with W_o, b_o halved
            src = name.replace(".h1.", ".h0.")
            arr = d1[src].copy()
            if name.endswith(".W_o") or name.endswith(".b_o"):
                arr = arr / 2.0
            d2[name] = arr
            classes2[s2] = classes1[layout1.sl(src)]
        elif name in d1 and d1[name].shape == shape:
            arr = d1[name]
            if name.endswith(".W_o") or name.endswith(".b_o"):
                arr = arr / 2.0            # h0's half of the split
            d2[name] = arr
            classes2[s2] = classes1[layout1.sl(name)]
        else:
            raise ValueError(f"unhandled entry {name} {shape}")
    template2 = layout2.pack_dict(d2)
    return arch1, layout1, template1, classes1, \
        arch2, layout2, template2, classes2


def audit_split(layout1, template1, layout2, template2):
    d1 = layout1.unpack_dict(template1)
    d2 = layout2.unpack_dict(template2)
    checks = {}
    checks["ffn_W1_daughters_exact"] = bool(np.array_equal(
        d2["c0.b0.W_1"][:, 8:], d2["c0.b0.W_1"][:, :8]))
    checks["ffn_W1_matches_widened"] = bool(np.array_equal(
        d2["c0.b0.W_1"][:, :8], d1["c0.b0.W_1"]))
    checks["ffn_W2_halved"] = bool(np.allclose(
        d2["c0.b0.W_2"][:8], d1["c0.b0.W_2"] / 2.0)
        and np.allclose(d2["c0.b0.W_2"][8:],
                        d1["c0.b0.W_2"] / 2.0))
    checks["head_Wq_copied"] = bool(np.array_equal(
        d2["c0.b0.h1.W_q"], d1["c0.b0.h0.W_q"]))
    checks["head_Wo_halved"] = bool(np.allclose(
        d2["c0.b0.h0.W_o"], d1["c0.b0.h0.W_o"] / 2.0)
        and np.allclose(d2["c0.b0.h1.W_o"],
                        d1["c0.b0.h0.W_o"] / 2.0))
    checks["head_bo_halved"] = bool(np.allclose(
        d2["c0.b0.h0.b_o"], d1["c0.b0.h0.b_o"] / 2.0)
        and np.allclose(d2["c0.b0.h1.b_o"],
                        d1["c0.b0.h0.b_o"] / 2.0))
    checks["all_ok"] = bool(all(checks.values()))
    return checks


def main():
    t_start = time.time()
    cfg = dataclasses.replace(
        be.EngineConfig.load(CFG_PATH), widen_step=2)
    backend = get_backend(cfg.backend, cfg.device, cfg.precision)
    domain = tsp.TSPDomain(cfg, backend)

    ck = np.load(HERE / "engine_runs" / "fresh_d2_l1" /
                 "checkpoint.npz", allow_pickle=True)
    arch0 = be.ArchSpec.from_signature(str(ck["arch_signature"]))
    layout0 = be.Layout(arch0, tsp.head_spec(arch0.d))
    champ_z = np.load(SUBJECT_PATH)
    subject = np.asarray(champ_z["genome"], dtype=float)
    stored_val = float(champ_z["val_score"])
    classes0 = be._base_classes(layout0, 0)

    results = {
        "experiment": "double-widen test on the new "
                      "single-layer lineage (sponsor's "
                      "default widen: d x2, FFN x2, heads "
                      "x2, Net2WiderNet splits)",
        "date": "2026-11-23",
        "subject": {"file": str(SUBJECT_PATH),
                    "stored_val_score": stored_val,
                    "mean_gap_percent": -100.0 * stored_val,
                    "n_genes": int(layout0.n_params)},
        "seeds": dict(STEP_SEEDS, val_seed=cfg.val_seed),
    }

    repro_val = domain.validation_score(layout0, subject)
    assert abs(repro_val - stored_val) < 1e-9
    log(f"champion val reproduced: {repro_val:.6f} "
        f"({-100.0 * repro_val:.2f}% gap)")
    before = cloud_classify(cw.density_read(domain, layout0,
                                            subject))
    results["before"] = before
    log(f"BEFORE: band {before['band_counts']}; cloud "
        f"{before['cloud']['counts']} (limit "
        f"+/-{before['cloud']['limit_pct']:.1f}% of gap)")

    useful = [u for u in before["units"]
              if u["cloud_cat"] == "contributing"]
    frozen0 = cw.frozen_mask_from_names(
        layout0, [u["name"] for u in useful])
    results["freeze_prep"] = {
        "basis": "near-zero cloud contributing units "
                 "of the subject (the sponsor's 17:34 rule)",
        "useful_units": [u["name"] for u in useful],
        "n_frozen_genes": int(frozen0.sum())}
    log(f"FREEZE prep: {[u['name'] for u in useful]}; "
        f"{int(frozen0.sum())} genes")

    arch1, layout1, template1, classes1, arch2, layout2, \
        template2, classes2 = build_double_transplant(
            arch0, layout0, subject, cfg)
    pres = domain.preservation(layout0, subject, layout2,
                               template2)
    audit = audit_split(layout1, template1, layout2, template2)
    frozen_names2 = [f"c0.b0.ffn.u{j}" for j in MARKED_FFN] \
        + [f"c0.b0.ffn.u{j + 8}" for j in MARKED_FFN] \
        + ["c0.b0.head.h0", "c0.b0.head.h1"]
    frozen2 = cw.frozen_mask_from_names(layout2, frozen_names2)
    results["transplant"] = {
        "d_old": int(arch0.d), "d_new": int(arch2.d),
        "ffn_old": int(arch0.ffn_hidden),
        "ffn_new": int(arch2.ffn_hidden),
        "heads_old": 1, "heads_new": 2,
        "n_genes_old": int(layout0.n_params),
        "n_genes_new": int(layout2.n_params),
        "n_units_old": int(before["n_units"]),
        "n_units_new": int(len(be.silence_units(layout2))),
        "preservation_gate": pres,
        "split_audit": audit,
        "frozen_unit_names_new_layout": frozen_names2,
        "n_frozen_genes_carried": int(frozen2.sum()),
        "arch_signature_new": layout2.arch.signature()}
    log(f"TRANSPLANT: d {arch0.d} -> {arch2.d}, FFN "
        f"{arch0.ffn_hidden} -> {arch2.ffn_hidden} (split), "
        f"heads 1 -> 2 (split), {layout0.n_params} -> "
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
        print("DOUBLE_WIDEN_D2_DONE")
        return

    pv_t = cw.prob_vectors(classes2, cfg, frozen2)
    pv_c = cw.prob_vectors(classes0, cfg, frozen0)

    rng_t = np.random.default_rng(STEP_SEEDS["run_seed"])
    draw_rng_t = np.random.default_rng(STEP_SEEDS["draw_seed"])
    species_t = be.create_species(
        template2, classes2, cfg.n_species, cfg.per_species,
        draw_rng_t, rng_t, pv_t["A"], cfg, lineage=None,
        init_mode="transplant", next_id=0)
    t0 = time.time()
    out_t = cw.evolve(domain, layout2, cfg, classes2,
                      species_t, rng_t, pv_t)
    wall_t = time.time() - t0
    after_t = cloud_classify(cw.density_read(
        domain, layout2, out_t["champion_genome"]))
    results["treatment"] = cw.arm_block(out_t, after_t,
                                        repro_val, wall_t)
    results["treatment"]["density_cloud_counts"] = \
        after_t["cloud"]["counts"]
    np.savez(HERE / "double_widen_d2_2026-11-23_"
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
    after_c = cloud_classify(cw.density_read(
        domain, layout0, out_c["champion_genome"]))
    results["control"] = cw.arm_block(out_c, after_c,
                                      repro_val, wall_c)
    results["control"]["density_cloud_counts"] = \
        after_c["cloud"]["counts"]
    np.savez(HERE / "double_widen_d2_2026-11-23_"
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
    print("DOUBLE_WIDEN_D2_DONE")


if __name__ == "__main__":
    main()
