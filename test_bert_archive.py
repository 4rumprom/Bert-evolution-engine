"""Verification for bert_archive.py (Addendum 2: full-population
archive, per-weight usefulness scoring, lineage retention,
heatmap).

Run: python3 test_bert_archive.py
Prints one PASS/FAIL line per leg. Exit code 0 iff every
leg passes. Tiny configs only -- NO full-budget run.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

import bert_archive as ba
import bert_engine as be
import run_bert_engine as runner

RESULTS = []


def leg(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}",
          flush=True)
    return ok


def tiny_cfg(**over):
    cfg = runner.smoke_config("tsp")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def attach_recording(eng, tracker):
    """Install the tracker-maintenance wrapper on
    eng.generation (the exact hook placement the runner will
    use: record after the fold-back, before culling/breeding,
    from the pre-generation population snapshot paired with
    the scores just folded back). The RECORDING itself is the
    runner's real hook now (it writes the entry shapes the
    analysis needs); this wrapper only maintains a SEPARATE
    tracker for the online-vs-post-hoc comparison, plus
    parent_species_id + tracker inheritance across
    cataclysms, and drops tracker vectors for species that
    disappear. Returns a restore callable."""
    out_dir = eng.out_dir
    orig_gen = runner.Engine.generation
    state = {"events_seen": 0}

    def recording_gen(self, phase):
        stage = self.meta["stage_index"]
        gen = self.meta["gen_in_phase"]
        total = self.meta["total_gens"]
        snap = [(sp, sp["pop"].copy())
                for sp in list(self.species)]
        orig_gen(self, phase)
        # 1. tracker update only (snapshot pops, folded-back
        # lens); the recording is the runner's real hook
        for sp, pop in snap:
            lens = sp.get("lens")
            if lens is not None:
                tracker.update(sp["id"], pop, lens)
        # 2. cataclysm inheritance for regenerated species
        events = self.meta["events"]
        for ev in events[state["events_seen"]:]:
            if ev["event"] == "cataclysm" and ev.get("founders"):
                fmap = {f["lineage"]: f for f in ev["founders"]}
                for sp in self.species:
                    if sp.get("parent_species_id") is None \
                            and sp["lineage"] in fmap:
                        # the cataclysm EVENT stores founders
                        # under "species_id" (select_founders'
                        # own dicts use "source_species_id")
                        f = fmap[sp["lineage"]]
                        sp["parent_species_id"] = int(
                            f["species_id"])
                        tracker.inherit(
                            sp["id"], int(f["species_id"]))
        state["events_seen"] = len(events)
        # 3. drop tracker vectors for species that are gone
        alive = {sp["id"] for sp in self.species}
        for sid in tracker.species_ids():
            if sid not in alive:
                tracker.remove(sid)

    eng.generation = types.MethodType(recording_gen, eng)

    def restore():
        del eng.generation

    return restore


def drive_generations(eng, tracker, n_gens, phase="A"):
    """Drive eng.generation() n times with recording attached."""
    restore = attach_recording(eng, tracker)
    try:
        for _ in range(n_gens):
            eng.generation(phase)
    finally:
        restore()


def main():
    rng = np.random.default_rng(20261122)

    # ------------------------------------------ leg 1: round trip
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run1"
        cfg = tiny_cfg()
        eng = runner.Engine(cfg, out)
        eng.setup_fresh()
        n_params = eng.layout.n_params
        batch = eng.domain.train_batch("A", 0, 0)
        pops, all_scores = [], []
        for sp in eng.species:
            pops.append(sp["pop"])
            all_scores.append(eng.domain.evaluate(
                eng.layout, sp["pop"], batch))
        G0 = np.concatenate(pops)
        S0 = np.concatenate(all_scores)
        sids0 = np.concatenate(
            [np.full(len(sp["pop"]), sp["id"], dtype=np.int64)
             for sp in eng.species])
        lins0 = np.concatenate(
            [np.full(len(sp["pop"]), sp["lineage"])
             for sp in eng.species])
        pars0 = np.full(len(S0), -1, dtype=np.int64)
        r1 = ba.record_generation(out, 0, "A", 0, 0, G0, S0,
                                  sids0, lins0, pars0, cfg=cfg)
        # a second generation: one breeding step, re-scored
        pv = be.prob_vector(eng.classes, "A", cfg)
        pops2, scores2 = [], []
        for sp, sc in zip(eng.species, all_scores):
            pop2 = be.breed_species(sp["pop"], sc, sp["draw"],
                                    eng.classes, pv, eng.rng,
                                    cfg, sp["target"])
            pops2.append(pop2)
            scores2.append(eng.domain.evaluate(
                eng.layout, pop2, batch))
        G1 = np.concatenate(pops2)
        S1 = np.concatenate(scores2)
        r2 = ba.record_generation(out, 0, "A", 1, 1, G1, S1,
                                  sids0, lins0, pars0, cfg=cfg)
        recs = list(ba.iter_archive_records(out / "archive"))
        ok_rt = (r1["written"] and r2["written"]
                 and len(recs) == 2
                 and np.array_equal(recs[0]["genomes"], G0)
                 and np.array_equal(recs[0]["scores"], S0)
                 and np.array_equal(recs[0]["species_ids"], sids0)
                 and np.array_equal(recs[1]["genomes"], G1)
                 and np.array_equal(recs[1]["scores"], S1)
                 and recs[0]["stage"] == 0
                 and recs[0]["phase"] == "A"
                 and recs[0]["gen_in_phase"] == 0
                 and recs[0]["total_gens"] == 0
                 and recs[1]["total_gens"] == 1
                 and list(recs[0]["parent_species_ids"])
                 == [-1] * len(S0))
        leg("archive round trip: writer -> reader bit-identical "
            "(genomes, scores, species ids, scalars)",
            ok_rt,
            f"(n_params={n_params}, records={len(recs)}, "
            f"bytes_stage={r2['bytes_used_stage']})")

    # ------------------------------------------ leg 2: usefulness
    # 40 members; gene 0 perfectly separates winners from
    # losers, gene 1 is pure noise, gene 2 is frozen.
    P, n = 40, 3
    scores_syn = np.linspace(-1.0, 1.0, P)
    G_syn = np.zeros((P, n))
    G_syn[:, 0] = scores_syn            # perfect separator
    G_syn[:, 1] = rng.normal(0, 1, P)   # noise
    G_syn[:, 2] = 0.7                   # frozen
    log0 = np.zeros(n)
    log1 = ba.update_log_scores(log0, G_syn, scores_syn, 0.02)
    ok_use = (log1[0] > 0.01
              and log1[1] < 0.02
              and log1[2] == 0.0)
    # a second generation of the same evidence keeps pushing
    # the separator up and never moves the frozen gene
    log2 = ba.update_log_scores(log1, G_syn, scores_syn, 0.02)
    ok_use = ok_use and log2[0] > log1[0] and log2[2] == 0.0
    leg("usefulness: separating gene rises, noise gene falls "
        "or stays near neutral, frozen gene stays exactly 0.0",
        ok_use,
        f"(log-scores after 1 gen: {np.round(log1, 4)}, "
        f"after 2 gens: {np.round(log2, 4)})")

    # --------------------------- legs 3-6: tiny real runs + analysis
    # leg 3: online tracker vs post-hoc recompute (no cataclysm)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run3"
        cfg = tiny_cfg(phase_a_gens=6, phase_b_gens=0,
                       max_growth_rounds=0)
        eng = runner.Engine(cfg, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        for sp in eng.species:
            sp["parent_species_id"] = -1
        tracker = ba.UsefulnessTracker(eng.layout.n_params)
        eng.phase_cache = {}
        eng.elim_series = []
        drive_generations(eng, tracker, 6, phase="A")
        n_recs = len(list(ba.iter_archive_records(
            out / "archive")))
        rep = ba.analyse_run(out)
        winner = rep["winner_species_id"]
        z = np.load(out / "lineage" / "stage000.npz")
        final_post = z["final_log_scores"]
        final_online = tracker.vector(winner)
        dmax = float(np.max(np.abs(final_post - final_online))) \
            if final_online.size else float("nan")
        leg("online vs post-hoc agreement: tracker final vector "
            "== analysis recomputed final vector (max diff 0.0)",
            dmax == 0.0 and n_recs == 6,
            f"(winner species {winner}, max diff {dmax}, "
            f"records {n_recs}, generations covered "
            f"{rep['generations_covered']})")

    # legs 4-6: forced cataclysm, lineage chain, retention, heatmap
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run4"
        cfg = tiny_cfg(phase_a_gens=9, phase_b_gens=0,
                       max_growth_rounds=0, founder_count=1,
                       cataclysm_every_n_generations=4,
                       cull_interval=100)
        eng = runner.Engine(cfg, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        for sp in eng.species:
            sp["parent_species_id"] = -1
        tracker = ba.UsefulnessTracker(eng.layout.n_params)
        eng.phase_cache = {}
        eng.elim_series = []
        drive_generations(eng, tracker, 9, phase="A")
        cats = [e for e in eng.meta["events"]
                if e["event"] == "cataclysm"]
        # pre-analysis snapshot of the archive (for the
        # retention-content check below)
        pre_recs = list(ba.iter_archive_records(
            out / "archive"))
        pre_sizes = [p.stat().st_size for p in
                     sorted((out / "archive").glob(
                         "stage*_gen*.npz"))]
        pre_total_bytes = sum(pre_sizes)
        rep = ba.analyse_run(out)
        # --- leg 4: chain unbroken 0..8, retention keeps chain only
        chain = rep["chain"]
        covered = []
        for entry in chain:
            gr = entry["generations"]
            covered.extend(range(gr[0], gr[1] + 1))
        chain_ok = (sorted(covered) == list(range(9))
                    and rep["generations_covered"] == 9
                    and len(cats) == 2)
        # lineage file content == the archive rows of the chain species
        chain_at = {}
        for entry in chain:
            gr, sid = entry["generations"], entry["species_id"]
            for g in range(gr[0], gr[1] + 1):
                chain_at[g] = sid
        z = np.load(out / "lineage" / "stage000.npz")
        lin_ok = True
        for rec in pre_recs:
            g = rec["total_gens"]
            sid = chain_at[g]
            idx = np.where(rec["species_ids"] == sid)[0]
            rows = np.where(z["generations"] == g)[0]
            if len(idx) != len(rows) \
                    or not np.array_equal(
                        z["genomes"][rows], rec["genomes"][idx]) \
                    or not np.array_equal(
                        z["scores"][rows], rec["scores"][idx]) \
                    or not np.all(z["species_ids"][rows] == sid):
                lin_ok = False
                break
        leg("lineage chain: unbroken from generation 0 to the "
            "end across 2 cataclysms; retention keeps only "
            "chain members, bit-identical",
            chain_ok and lin_ok,
            f"(chain {chain}, cataclysms {len(cats)}, "
            f"winner {rep['winner_species_id']}, fallback "
            f"{rep['fallback_used']})")
        # --- leg 5: disposal + retention accounting
        remaining = list((out / "archive").glob(
            "stage*_gen*.npz"))
        ret = json.loads(
            (out / "lineage" / "retention.json").read_text())
        kept_npz = list((out / "lineage").glob("stage*.npz"))
        kept_bytes = sum(p.stat().st_size for p in kept_npz)
        ok_ret = (len(remaining) == 0
                  and len(kept_npz) == 1
                  and ret["deleted_bytes"] == pre_total_bytes
                  and ret["deleted_count"] == 9
                  and ret["kept_bytes"] == kept_bytes
                  and ret["winner_species_id"]
                  == rep["winner_species_id"])
        leg("retention/disposal: non-lineage archive files "
            "gone, lineage files exist, retention.json "
            "accounts for the bytes",
            ok_ret,
            f"(deleted {ret['deleted_bytes']} bytes in "
            f"{ret['deleted_count']} files, kept "
            f"{ret['kept_bytes']} bytes in {len(kept_npz)} "
            f"file(s))")
        # --- leg 6: heatmap
        html_path = Path(rep["heatmap_path"])
        html = html_path.read_text()
        arch = be.initial_arch(cfg, 2)
        layout = be.Layout(arch,
                           __import__("bert_domain_tsp").head_spec(
                               arch.d))
        matrices = [nm for nm, _sl, sh in layout.entries
                    if len(sh) == 2]
        panels = re.findall(r"class='panel' data-matrix='([^']+)'",
                            html)
        zone_rows = re.findall(
            r"<tr class='zone-row'><td>[^<]+</td><td>[^<]+</td>"
            r"<td>([0-9.]+)</td><td>([0-9.]+)</td>"
            r"<td>([0-9.]+)</td><td>([0-9.]+)</td></tr>", html)
        sums_ok = (len(zone_rows) == len(matrices)
                   and all(abs(sum(float(x) for x in row) - 1.0)
                           < 1e-3 for row in zone_rows))
        ok_hm = (html_path.exists()
                 and "http://" not in html
                 and "https://" not in html
                 and "<script src" not in html
                 and sorted(panels) == sorted(matrices)
                 and sums_ok
                 and "SERIES=" in html
                 and "genSlider" in html)
        leg("heatmap: self-contained, one panel per weight "
            "matrix, density table sums to 1 per zone, "
            "generation slider embedded",
            ok_hm,
            f"(panels {len(panels)} vs matrices "
            f"{len(matrices)}, zone rows {len(zone_rows)}, "
            f"series embedded: {'SERIES=' in html})")

    # ------------------------------------------ leg 7: size cap
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run7"
        cfg = tiny_cfg()
        Gc = rng.normal(0, 0.5, (30, 728))
        Sc = rng.normal(0, 1, 30)
        sids = np.repeat(np.arange(6), 5).astype(np.int64)
        lins = np.array([f"L{i}" for i in sids])
        pars = np.full(30, -1, dtype=np.int64)
        cfg_big = tiny_cfg()
        cfg_big.archive_cap_bytes_per_stage = 10 ** 9
        r_ok = ba.record_generation(out, 0, "A", 0, 0, Gc, Sc,
                                    sids, lins, pars, cfg=cfg_big)
        used = r_ok["bytes_used_stage"]
        cfg_tight = tiny_cfg()
        cfg_tight.archive_cap_bytes_per_stage = used + 10
        r_cap = ba.record_generation(out, 0, "A", 1, 1, Gc, Sc,
                                     sids, lins, pars,
                                     cfg=cfg_tight)
        n_files = len(list((out / "archive").glob(
            "stage*_gen*.npz")))
        ok_cap = (r_ok["written"] and not r_ok["capped"]
                  and r_cap["capped"] and not r_cap["written"]
                  and n_files == 1
                  and r_cap["bytes_used_stage"] == used)
        leg("size cap: recording stops and reports capped=True "
            "instead of writing past the cap",
            ok_cap,
            f"(first record {r_ok['record_bytes']} bytes, cap "
            f"for second {used + 10}, files on disk {n_files})")

    # ------------------------------------------ leg 8: tracker state
    tr = ba.UsefulnessTracker(3)
    G8 = np.zeros((40, 3))
    G8[:, 0] = np.linspace(-1, 1, 40)
    G8[:, 1] = rng.normal(0, 1, 40)
    G8[:, 2] = 0.7
    S8 = np.linspace(-1.0, 1.0, 40)
    tr.update(5, G8, S8)
    tr.update(5, G8, S8)
    tr.inherit(9, 5)
    tr.inherit(11, 999)  # unknown parent -> neutral
    tr.update(9, G8, S8)
    sd = tr.state_dict()
    blob = json.dumps(sd)
    tr2 = ba.UsefulnessTracker(3)
    tr2.load_state_dict(json.loads(blob))
    ok_state = (
        np.array_equal(tr2.vector(5), tr.vector(5))
        and np.array_equal(tr2.vector(9), tr.vector(9))
        and np.array_equal(tr2.vector(11),
                           np.zeros(3))
        and tr2.density(5) == tr.density(5)
        and tr2.density(999)["never_judged"] == 1.0)
    leg("tracker state round trip: state_dict -> JSON -> "
        "load_state_dict reproduces vectors bit-identically",
        ok_state,
        f"(species {tr.species_ids()}, density sp5 "
        f"{tr.density(5)})")

    # ------------------------------------------ leg 9: full runner path
    # eng.run() end to end with the REAL hooks: the runner
    # records, tracks, and analyses at finish() (a CLI re-run
    # would find the archive already disposed).
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run9"
        cfg = tiny_cfg(max_growth_rounds=0)
        eng = runner.Engine(cfg, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        res = eng.run()
        done = (out / "COMPLETE.txt").exists() \
            and (out / "results.json").exists()
        ret_path = out / "lineage" / "retention.json"
        ret9 = json.loads(ret_path.read_text()) \
            if ret_path.exists() else {}
        want_winner = res["stage_results"][-1][
            "champion_species_id"] if done else None
        arch_left = list((out / "archive").glob("*.npz")) \
            if (out / "archive").is_dir() else []
        ok9 = (done
               and ret9.get("winner_species_id") == want_winner
               and ret9.get("fallback_used") == []
               and ret9.get("n_archive_records", 0) > 0
               and ret9["n_archive_records"]
               == ret9["generations_covered"]
               and Path(ret9.get("heatmap_path") or "").exists()
               and not arch_left)
        leg("full runner path: eng.run() with the real hooks "
            "analyses at finish() (results.json champion, no "
            "fallback), heatmap written, archive disposed",
            ok9,
            f"(winner {ret9.get('winner_species_id')} vs stage "
            f"champion {want_winner}, records "
            f"{ret9.get('n_archive_records')}, archive left "
            f"{len(arch_left)})")

    # ------------------------------------------ leg 10: CLI on a
    # stopped run (records present, no results.json: the CLI
    # falls back to the last generation's best species and
    # rebuilds the layout from the checkpoint).
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run10"
        cfg = tiny_cfg(max_growth_rounds=0)
        eng = runner.Engine(cfg, out, stop_after=4)
        eng.setup_fresh()
        eng.save_checkpoint()
        out10 = eng.run()
        stopped = bool(out10.get("stopped_early")) \
            and not (out / "results.json").exists()
        n_rec = len(list((out / "archive").glob("*.npz"))) \
            if (out / "archive").is_dir() else 0
        r = subprocess.run(
            [sys.executable, str(HERE / "bert_archive.py"),
             str(out)], capture_output=True, text=True,
            timeout=300)
        rep10 = json.loads(r.stdout) if r.stdout.strip() else {}
        ok10 = (stopped and r.returncode == 0
                and rep10.get("n_archive_records", 0) == n_rec
                and rep10.get("n_archive_records", 0) > 0
                and Path(rep10.get("heatmap_path") or "").exists()
                and any("last generation" in f
                        for f in rep10.get("fallback_used", [])))
        leg("CLI on a stopped run: fallback winner from the "
            "last generation, layout from the checkpoint, "
            "heatmap written",
            ok10,
            f"(returncode {r.returncode}, stopped {stopped}, "
            f"records {rep10.get('n_archive_records')} vs "
            f"{n_rec}, fallback "
            f"{rep10.get('fallback_used')}, stderr tail: "
            f"{r.stderr[-200:]!r})")

    # ------------------------------------------ leg 11: growing run
    # (the rebase leg): the online tracker and the post-hoc
    # analysis must agree across a growth step -- both rebase
    # the usefulness vectors by entry name -- and the heatmap
    # must be written for the final layout with the chain
    # unbroken across the growth.
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run11"
        cfg = tiny_cfg(max_growth_rounds=1)
        eng = runner.Engine(cfg, out)
        eng.setup_fresh()
        eng.save_checkpoint()
        res = eng.run()
        done = (out / "COMPLETE.txt").exists()
        ret_path = out / "lineage" / "retention.json"
        ret11 = json.loads(ret_path.read_text()) \
            if ret_path.exists() else {}
        lin_files = sorted((out / "lineage").glob("stage*.npz")) \
            if (out / "lineage").is_dir() else []
        # online tracker final state (checkpoint meta) vs the
        # post-hoc final vector (last lineage npz)
        z = np.load(out / "checkpoint.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        trk = meta.get("archive_tracker") or {}
        winner = ret11.get("winner_species_id")
        max_diff = None
        if lin_files and winner is not None \
                and winner in trk.get("species_ids", []):
            final_log = np.load(lin_files[-1],
                                allow_pickle=False)[
                                    "final_log_scores"]
            online = np.array(trk["log_scores"][
                trk["species_ids"].index(winner)])
            max_diff = float(np.max(np.abs(online - final_log)))
        ok11 = (done and len(res["stage_results"]) >= 2
                and ret11.get("fallback_used") == []
                and ret11.get("winner_species_id")
                == res["stage_results"][-1]["champion_species_id"]
                and len(ret11.get("chain", [])) >= 2
                and max_diff == 0.0
                and Path(ret11.get("heatmap_path") or "").exists())
        leg("growing run: tracker and analysis agree across "
            "the growth (max diff 0.0), chain unbroken, "
            "heatmap written",
            ok11,
            f"(stages {len(res['stage_results'])}, chain "
            f"{ret11.get('chain')}, max diff {max_diff}, "
            f"fallback {ret11.get('fallback_used')})")

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_pass}/{len(RESULTS)} legs passed ====",
          flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
