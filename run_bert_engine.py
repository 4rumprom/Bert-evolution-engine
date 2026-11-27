"""Runner for the parametric BERT growth engine.

One engine, two domains (TSP and stock allocation), two
backends (numpy reference / JAX portable). The runner owns:
the two-phase stage protocol, the growth-rules sequencer
(least-change first, exact preservation gate, keep/revert),
the founder mechanism and cataclysm events, Schedule A
culling, per-generation atomic checkpoints with full RNG
state (resume refuses a mismatched config), champion and
founder vaulting, and the completion marker (written only
after evolution, validation and vaulting are on disk).

Usage:
  python3 run_bert_engine.py --config bert_engine_config.json
  python3 run_bert_engine.py --smoke [--domain stock]
  python3 run_bert_engine.py --config X --fresh
  python3 run_bert_engine.py --config X --stop-after 3   (resume test)
  python3 run_bert_engine.py --lifecycle [--lifecycle-config bert_lifecycle_config.json] [--out-dir DIR]
      (the Frankenstein lifecycle, the engine's growth
      sequencer since the sponsor froze the method on
      2026-11-27; drives bert_lifecycle.py's state machine.
      --out-dir names the lifecycle run directory. Without
      --lifecycle the runner behaves exactly as before.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bert_engine as be
import bert_archive as ba
import bert_monitor as bm
from bert_backend import get_backend
import bert_domain_tsp as tsp
import bert_domain_stock as stock
import bert_stock_data as bsd


def log(msg):
    print(msg, flush=True)


def _config_relative(cfg, p):
    """Resolve a domain_config path relative to the config
    file's directory (recorded by main() as cfg.config_dir),
    then this runner's directory, then as given; the first
    existing candidate wins."""
    p = Path(p)
    if p.is_absolute():
        return p
    bases = [getattr(cfg, "config_dir", None), HERE, Path.cwd()]
    for base in bases:
        if base is not None and (Path(base) / p).exists():
            return Path(base) / p
    return Path(bases[1]) / p


def build_domain(cfg, backend):
    if cfg.domain == "tsp":
        return tsp.TSPDomain(cfg, backend), tsp.head_spec, tsp.widen_head
    if cfg.domain == "stock":
        dc = cfg.domain_config
        panel = None
        val_eps = None
        if dc.get("synthetic_panel"):
            panel = stock.make_synthetic_panel(
                int(dc.get("synthetic_tickers", 30)),
                int(dc.get("synthetic_days", 500)),
                int(dc.get("synthetic_seed", 20261120)))
        elif dc.get("panel_bundle"):
            bundle_path = _config_relative(cfg,
                                           dc["panel_bundle"])
            panel = bsd.load_bundle_panel(bundle_path)
            log(f"[engine] stock panel from bundle "
                f"{bundle_path}: {len(panel.tickers)} tickers "
                f"x {len(panel.dates)} days "
                f"({panel.dates[0]}..{panel.dates[-1]}), "
                f"fundamentals "
                f"{sorted(panel.fundamentals)}, "
                f"train_end_index {panel.train_end_index}")
        elif dc.get("panel", "real") == "real":
            raise SystemExit(
                "stock domain: the real genetic_lab panel is "
                "loaded by the caller in production runs; for "
                "this build use domain_config.synthetic_panel "
                "= true (smoke scale) or domain_config."
                "panel_bundle = a bundle built by "
                "build_stock_bundle.py. See design record.")
        if dc.get("frozen_episodes_path"):
            val_eps = stock.load_frozen_episodes(
                dc["frozen_episodes_path"])
            if val_eps is not None and panel is not None \
                    and dc.get("panel_bundle"):
                # the frozen baskets' stored indices address
                # the 622-ticker genetic_lab panel ordering,
                # not the bundle's: resolve by ticker name
                val_eps = bsd.remap_episodes(
                    val_eps, panel,
                    episode_days=int(dc.get("episode_days", 60)))
        return (stock.StockDomain(cfg, backend, panel=panel,
                                  val_episodes=val_eps),
                stock.head_spec, stock.widen_head)
    raise ValueError(cfg.domain)


def initial_classes(layout):
    """Founding-stage gene classes (sponsor correction,
    2026-11-24): stage zero is one brand-new BERT and
    NOTHING in it is frozen -- every gene mutates, the
    whole network evolves as one. No gene is classed
    as a species draw here. The old version marked
    every non-head gene CLS_DRAW, which froze the
    whole encoder for the founding run and left only
    the output head free to change; the sponsor struck
    that down ("the species isn't frozen; if you have
    88 genes you mutate all 88 in the first run").
    Base genes are CLS_OLD; the output head entries
    stay CLS_DECODER, the crossover-eligible class."""
    classes = np.full(layout.n_params, be.CLS_OLD, dtype=np.int8)
    for c in range(len(layout.arch.couples)):
        for name in layout.head_names(c):
            classes[layout.sl(name)] = be.CLS_DECODER
    return classes


class Engine:
    def __init__(self, cfg: be.EngineConfig, out_dir: Path,
                 stop_after=None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vault_dir = self.out_dir / "vault"
        self.vault_dir.mkdir(exist_ok=True)
        self.backend = get_backend(cfg.backend, cfg.device,
                                   cfg.precision)
        self.domain, self.head_spec_fn, self.widen_head = \
            build_domain(cfg, self.backend)
        input_dim = cfg.input_dim or self.domain.input_dim
        self.input_dim = input_dim
        self.stop_after = stop_after
        self.gens_this_invocation = 0
        self.t0 = time.time()
        self.ckpt_path = self.out_dir / "checkpoint.npz"
        # silencing / marking state (plan 2026-11-22): the
        # current champion's silencing table and the per-gene
        # mutation band rates built from it; both ride in the
        # checkpoint meta and are restored on resume
        self.marks = None
        self.frozen = None
        self.silencing_table = None
        # archive / usefulness state (Addendum 2): created in
        # setup_fresh, restored in setup_resume; None when
        # the archive is disabled
        self.tracker = None
        self.archive_bytes_used = {}
        # monitor state (Addendum 3, stock domain only):
        # the cached random ladder, per-generation timings,
        # the last generation's leader, and the snapshot log
        self.monitor_ladder = None
        self._gen_durations = []
        self._gen_t0 = None
        self._gen_leader = None
        self._snap_keys = set()
        self._snapshot_count = 0
        self._last_snapshot = None

    # ---------------------------------------------------------- vault
    def vault(self, genome, val_score, kind, stage, species_id,
              lineage, detail=None):
        fn = (f"{kind}_stage{stage}_sp{species_id}_"
              f"{int(time.time() * 1000) % 100000000}.npz")
        be.atomic_npz(
            self.vault_dir / fn, genome=np.asarray(genome),
            val_score=val_score, kind=kind, stage=stage,
            species_id=species_id, lineage=lineage,
            arch_signature=self.arch.signature(),
            detail=json.dumps(detail or {}))
        return fn

    # ----------------------------------------------------- silencing
    def _silence_champion(self, genome, stage, vault_file):
        """Champion silencing (plan 2026-11-22 step 1): one
        batched silencing table for the stage champion, saved
        as a JSON artifact beside the vault entry, and the
        mutation marks built from it. The table and the marks
        are the live state from here on (stored in meta, so
        they ride the checkpoint); the accepted champion also
        keeps its own copy, so a later revert restores the
        pre-growth marks. Returns the table."""
        cfg, m = self.cfg, self.meta
        table = be.silence_table(
            self.layout, genome,
            lambda G: self.domain.validation_scores(self.layout,
                                                    G),
            zero_tol=self.domain.silence_zero_tol)
        silence_file = vault_file.replace(".npz",
                                          "_silence.json")
        be.atomic_json(self.vault_dir / silence_file, table)
        marks = be.marks_from_table(
            self.layout, table, cfg,
            zero_tol=self.domain.silence_zero_tol)
        self.marks = marks
        self.silencing_table = table
        m["marks"] = marks.tolist()
        m["silencing_table"] = table
        # freeze rule (sponsor, 2026-11-23): proven units
        # and their blocks' layer-norm genes mutate at
        # rate 0; the mask rides the checkpoint like marks
        self.frozen = be.frozen_mask_from_table(
            self.layout, table,
            zero_tol=self.domain.silence_zero_tol)
        m["frozen"] = self.frozen.tolist()
        log(f"[engine] freeze mask: "
            f"{int(self.frozen.sum())}/{len(self.frozen)} "
            f"genes frozen (proven units + their blocks' "
            f"layer-norm genes)")
        n_real = sum(1 for u in table["units"]
                     if u["drop"] > self.domain.silence_zero_tol)
        n_useless = sum(1 for u in table["units"]
                        if u["drop"]
                        <= self.domain.silence_zero_tol)
        density = n_real / len(table["units"]) \
            if table["units"] else 0.0
        m["events"].append({
            "event": "champion_silenced", "stage": stage,
            "vault_file": vault_file,
            "silence_file": silence_file,
            "champion_score": table["champion_score"],
            "n_units": len(table["units"]),
            "n_real_drops": n_real, "n_useless": n_useless,
            "masking_density": density})
        log(f"[engine] stage {stage} champion silenced: "
            f"{n_real}/{len(table['units'])} units read real "
            f"(masking density {density:.2f}), table "
            f"{silence_file}")
        return table

    # ------------------------------------------------------ checkpoint
    def save_checkpoint(self):
        # archive (Addendum 2): the usefulness tracker state
        # rides in the checkpoint meta (JSON-able by
        # construction), so a resume keeps the per-weight
        # evidence; the per-stage archive byte counts ride
        # with it. At smoke scale this is a few KB; at full
        # scale it grows with genome size x species -- see
        # the build log.
        if getattr(self, "tracker", None) is not None:
            self.meta["archive_tracker"] = \
                self.tracker.state_dict()
            self.meta["archive_bytes_used"] = {
                str(k): int(v)
                for k, v in self.archive_bytes_used.items()}
        state = {"species": self.species, "classes": self.classes,
                 "rng": self.rng, "draw_rng": self.draw_rng,
                 "meta": self.meta}
        payload = be.checkpoint_payload(self.cfg, self.arch,
                                        self.layout, state)
        be.atomic_npz(self.ckpt_path, **payload)

    def peek_arch(self):
        z = np.load(self.ckpt_path, allow_pickle=False)
        return be.ArchSpec.from_signature(str(z["arch_signature"]))

    # ------------------------------------------------------------ setup
    def setup_fresh(self):
        cfg = self.cfg
        self.arch = be.initial_arch(cfg, self.input_dim)
        self.layout = be.Layout(self.arch,
                                self.head_spec_fn(self.arch.d))
        self.classes = initial_classes(self.layout)
        self.rng = np.random.default_rng(cfg.run_seed)
        self.draw_rng = np.random.default_rng(cfg.draw_seed)
        pv = be.prob_vector(self.classes, "A", cfg,
                            marks=self.marks,
                            frozen=self.frozen,
                            new_only=self._growth_new_only(),
                            flat_nominal=True)
        self.species = be.create_species(
            np.zeros(self.layout.n_params), self.classes,
            cfg.n_species, cfg.per_species, self.draw_rng,
            self.rng, pv, cfg, lineage=None, init_mode="random",
            next_id=0)
        self.meta = {
            "phase": "A", "gen_in_phase": 0, "stage_gen": 0,
            "stage_index": 0, "growth_round": 0, "rule": None,
            "bar_val_score": None, "founders": None,
            "champion": None, "history": [], "events": [],
            "next_species_id": cfg.n_species,
            "stage_results": [], "total_gens": 0,
            "tried_rules": [], "phase_timings": [],
            "done": False, "marks": None, "frozen": None,
            "silencing_table": None, "widen_variant": "A",
            "rule_variant": "A",
        }
        # archive (Addendum 2): fresh usefulness tracker and
        # per-stage byte accounting; parent pointers are -1
        # for generation-0 species (create_species default)
        if getattr(cfg, "archive_enabled", True):
            self.tracker = ba.UsefulnessTracker(
                self.layout.n_params,
                usefulness_step=getattr(cfg, "usefulness_step",
                                        0.02))
            self.archive_bytes_used = {}
        # monitor (Addendum 3): fresh ladder cache + snapshot log
        self.monitor_ladder = None
        self._snap_keys = set()
        self._snapshot_count = 0
        self._last_snapshot = None
        log(f"[engine] fresh run '{cfg.run_name}': domain "
            f"{cfg.domain}, arch {self.arch.signature()}, "
            f"{self.layout.n_params} genes, "
            f"{cfg.n_species} species x {cfg.per_species}, "
            f"backend {self.backend.describe()}")

    def setup_resume(self):
        arch = self.peek_arch()
        self.arch = arch
        self.layout = be.Layout(arch, self.head_spec_fn(arch.d))
        state = be.load_checkpoint(self.ckpt_path, self.cfg,
                                   arch, self.layout)
        self.species = state["species"]
        self.classes = state["classes"]
        self.rng = state["rng"]
        self.draw_rng = state["draw_rng"]
        self.meta = state["meta"]
        # founders/champion genomes ride in meta as lists
        if self.meta.get("champion"):
            self.meta["champion"]["genome"] = np.array(
                self.meta["champion"]["genome"])
        if self.meta.get("founders"):
            for f in self.meta["founders"]:
                f["genome"] = np.array(f["genome"])
        # silencing state rides in meta (marks as a list);
        # load_checkpoint already refused a resume whose
        # marks do not match the layout
        if self.meta.get("marks") is not None:
            self.marks = np.array(self.meta["marks"],
                                  dtype=float)
        if self.meta.get("frozen") is not None:
            self.frozen = np.array(self.meta["frozen"],
                                   dtype=bool)
        if self.meta.get("silencing_table") is not None:
            self.silencing_table = self.meta["silencing_table"]
        # archive (Addendum 2): restore the usefulness
        # tracker from the checkpoint meta when the archive
        # is on; parent pointers ride in the checkpoint
        # (bert_engine) and survive the resume
        if getattr(self.cfg, "archive_enabled", True):
            self.tracker = ba.UsefulnessTracker(
                self.layout.n_params,
                usefulness_step=getattr(
                    self.cfg, "usefulness_step", 0.02))
            if self.meta.get("archive_tracker"):
                self.tracker.load_state_dict(
                    self.meta["archive_tracker"])
            self.archive_bytes_used = {
                int(k): int(v)
                for k, v in self.meta.get(
                    "archive_bytes_used", {}).items()}
        for sp in self.species:
            sp.setdefault("parent_species_id", -1)
        # monitor (Addendum 3): pick the cached ladder up by
        # architecture signature, and the snapshot log (the
        # reader tolerates a torn trailing line, so a resume
        # after a kill mid-generation never chokes)
        self.monitor_ladder = bm.load_ladder_cache(
            self.out_dir / "monitor_ladder.json")
        snaps = bm.read_snapshots(self.out_dir)
        self._snap_keys = {
            (s["total_gens"], s["label"]) for s in snaps}
        self._snapshot_count = len(snaps)
        self._last_snapshot = snaps[-1] if snaps else None
        log(f"[engine] RESUMED: stage {self.meta['stage_index']} "
            f"phase {self.meta['phase']} gen "
            f"{self.meta['gen_in_phase']}, "
            f"{len(self.species)} species")

    # ---------------------------------------------------- one generation
    def generation(self, phase):
        cfg, m = self.cfg, self.meta
        # monitor (Addendum 3): generation timing starts here
        self._gen_t0 = time.time()
        stage = m["stage_index"]
        gen = m["gen_in_phase"]
        batch = self.domain.train_batch(phase, stage, gen)
        groups, G = [], []
        pos = 0
        for sp in self.species:
            groups.append((pos, pos + len(sp["pop"])))
            G.append(sp["pop"])
            pos += len(sp["pop"])
        G = np.concatenate(G)
        cache = self.phase_cache if phase == "A" \
            else self.gen_cache
        scores = self.domain.evaluate(self.layout, G, batch,
                                      groups=groups, cache=cache)
        # fold back per species
        elim_best_scores = []
        off = 0
        for si, sp in enumerate(self.species):
            s = scores[off:off + len(sp["pop"])]
            off += len(sp["pop"])
            sp["lens"] = s
            bi = int(np.argmax(s))
            if s[bi] > sp["best"]["score"]:
                sp["best"] = {"score": float(s[bi]),
                              "genome": sp["pop"][bi].copy(),
                              "generation": m["total_gens"]}
        # archive hook (Addendum 2): record the evaluated
        # generation BEFORE culling/breeding, while sp["pop"]
        # is still the evaluated population and sp["lens"]
        # holds its scores; the tracker update uses the same
        # pairing. The entry shapes ride in the record so the
        # analysis can rebase the vectors across a growth.
        if getattr(cfg, "archive_enabled", True) \
                and self.tracker is not None:
            rec = ba.record_species_generation(
                self.out_dir, stage, phase, gen,
                m["total_gens"], self.species, cfg=cfg,
                entry_shapes=[(n, sh) for n, _, sh
                              in self.layout.entries])
            if rec["capped"]:
                m["events"].append({
                    "event": "archive_capped", "stage": stage,
                    "generation": m["total_gens"],
                    "bytes_used_stage":
                        rec["bytes_used_stage"]})
                log(f"[engine] archive CAP reached at stage "
                    f"{stage} (used {rec['bytes_used_stage']} "
                    "bytes); recording stops for this stage")
            self.archive_bytes_used[stage] = \
                rec["bytes_used_stage"]
            for sp in self.species:
                self.tracker.update(sp["id"], sp["pop"],
                                    sp["lens"])
        # elimination scores of each species' generation-best
        elim_batch = self.domain.elim_batch()
        for si, sp in enumerate(self.species):
            bi = int(np.argmax(sp["lens"]))
            es = self.domain.evaluate(
                self.layout, sp["pop"][bi][None, :], elim_batch,
                cache=cache)
            elim_best_scores.append(float(es[0]))
        elim_median = float(np.median(elim_best_scores))
        self.elim_series.append(elim_median)
        # monitor (Addendum 3): capture the generation leader
        # here, before breeding/culling -- the species
        # generation-best with the best elimination score
        if self.cfg.domain == "stock" \
                and self.cfg.monitor_enabled:
            bi = int(np.argmax(elim_best_scores))
            sp = self.species[bi]
            self._gen_leader = {
                "genome":
                    sp["pop"][int(np.argmax(sp["lens"]))].copy(),
                "species_id": int(sp["id"]),
                "elim_score": float(elim_best_scores[bi])}
        # culling on schedule (needs full-pop elim champions)
        culled = False
        if (gen + 1) % cfg.cull_interval == 0 \
                and len(self.species) > cfg.species_floor:
            champ_scores = []
            for sp in self.species:
                es = self.domain.evaluate(
                    self.layout, sp["pop"], elim_batch,
                    groups=[(0, len(sp["pop"]))], cache=cache)
                champ_scores.append(float(np.max(es)))
            self.species, entry = be.cull_species(
                self.species, champ_scores, cfg,
                cfg.n_species * cfg.per_species)
            if entry:
                entry["generation"] = m["total_gens"]
                entry["stage"] = stage
                m["events"].append(entry)
                culled = True
                # archive (Addendum 2): drop the culled
                # species' tracker vectors (bounded memory)
                if getattr(self, "tracker", None) is not None:
                    for did in entry["eliminated_species_ids"]:
                        self.tracker.remove(int(did))
        m["history"].append({
            "generation": m["total_gens"], "stage": stage,
            "phase": phase, "gen_in_phase": gen,
            "n_species": len(self.species),
            "pop_total": int(sum(len(sp["pop"])
                                 for sp in self.species)),
            "best_score": float(np.max(scores)),
            "median_score": float(np.median(scores)),
            "elim_best": float(np.max(elim_best_scores)),
            "elim_median": elim_median, "cull": culled})
        # cataclysm on the generation interval (within stage)
        m["gen_in_phase"] += 1
        m["stage_gen"] += 1
        m["total_gens"] += 1
        self.gens_this_invocation += 1
        cataclysm_fired = False
        if cfg.cataclysm_enabled \
                and cfg.cataclysm_every_n_generations \
                and m["stage_gen"] % \
                cfg.cataclysm_every_n_generations == 0 \
                and not self._stage_ending():
            self.do_cataclysm(within_stage=True)
            cataclysm_fired = True
        # breed for the next generation (not after the last,
        # and not right after a cataclysm: the regenerated
        # species were just created with their initial
        # mutation application and have no fitness yet)
        if not self._phase_ending(phase) and not cataclysm_fired:
            pv = be.prob_vector(self.classes, phase, cfg,
                                marks=self.marks,
                            frozen=self.frozen,
                            new_only=self._growth_new_only(),
                            flat_nominal=self._founding_stage())
            for sp in self.species:
                sp["pop"] = be.breed_species(
                    sp["pop"], sp["lens"], sp["draw"],
                    self.classes, pv, self.rng, cfg, sp["target"])
        self.save_checkpoint()
        # monitor (Addendum 3): status rewrite + snapshot
        # trigger, AFTER the checkpoint is saved
        self._monitor_after_generation(phase)

    def _phase_budget(self, phase):
        return self.cfg.phase_a_gens if phase == "A" \
            else self.cfg.phase_b_gens

    def _founding_stage(self):
        """Stage zero (sponsor correction, 2026-11-24):
        the founding stage evolves the WHOLE network at
        the nominal rate. Nothing is frozen, no gene is
        exempt, and there is no separate encoder or
        decoder evolution: one BERT, all of it evolving."""
        m = getattr(self, "meta", None) or {}
        return m.get("stage_index", 0) == 0

    def _growth_new_only(self):
        """Sponsor growth rule (2026-11-23): a stage that
        began with a growth transplant evolves ONLY the
        added material; the base stays untouched. Stages
        with no growth rule (the initial stage, and
        prune-and-graft / consolidation stages) evolve the
        whole network except the frozen proven genes."""
        m = getattr(self, "meta", None) or {}
        return bool(self.cfg.growth_new_only
                    and m.get("rule") is not None)

    def _phase_ending(self, phase):
        return self.meta["gen_in_phase"] >= self._phase_budget(phase)

    def _stage_ending(self):
        m = self.meta
        if m["phase"] == "A":
            return False
        return m["gen_in_phase"] >= self.cfg.phase_b_gens - 1

    # ------------------------------------------------- archive rebase
    def _rebase_tracker(self, layout_old, layout_new):
        """After a transplant changes the layout: remap every
        tracked log-score vector into the new layout by entry
        name (the same by-name prefix-overlap rule as the
        mutation marks), so the usefulness evidence survives
        the growth step and new genes start at 0 (never
        judged). Without this the tracker (fixed-length
        vectors) would raise on the first post-growth
        generation, and the analysis would reset instead of
        continuing."""
        if getattr(self, "tracker", None) is None:
            return
        entries_old = [(n, sh) for n, _, sh
                       in layout_old.entries]
        entries_new = [(n, sh) for n, _, sh
                       in layout_new.entries]
        new_tracker = ba.UsefulnessTracker(
            layout_new.n_params,
            usefulness_step=self.tracker.usefulness_step)
        for sid in self.tracker.species_ids():
            new_tracker.set_vector(
                sid, be.remap_vector_by_entries(
                    self.tracker.vector(sid),
                    entries_old, entries_new))
        self.tracker = new_tracker

    def _inherit_tracker(self):
        """After species are regenerated from founders: each
        child inherits its founder's usefulness vector. The
        parent id is authoritative from regenerate_species
        (set per founder, so sibling founders sharing a
        lineage stay distinct); no lineage lookup here. Also
        drops tracker vectors for species no longer live."""
        if getattr(self, "tracker", None) is None:
            return
        for sp in self.species:
            self.tracker.inherit(
                sp["id"], int(sp.get("parent_species_id", -1)))
        live = {sp["id"] for sp in self.species}
        for sid in self.tracker.species_ids():
            if sid not in live:
                self.tracker.remove(sid)

    # --------------------------------------------------------- monitor
    def _monitor_densities(self, species_id):
        """Density readings for a snapshot. Population density
        comes from the archive tracker when it exists
        (optional dependency -- the monitor works without
        it); masking density comes from the current
        champion's silencing table in meta. Either may be
        None."""
        pop = None
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            try:
                pop = tracker.density(int(species_id))
            except Exception:
                pop = None
        mask = None
        table = self.meta.get("silencing_table")
        if table and table.get("units"):
            n_real = sum(1 for u in table["units"]
                         if u["drop"]
                         > self.domain.silence_zero_tol)
            mask = {"real": n_real / len(table["units"]),
                    "n_real": n_real,
                    "n_units": len(table["units"])}
        return pop, mask

    def _monitor_status_dict(self):
        """Everything cheap, from Engine.meta / history, the
        backend describe() call, and the growth-rule readout
        (meta["rule"] / meta["tried_rules"] / meta["events"]).
        Timing: seconds per generation (mean of the last 20)
        and ETA for the current phase from the phase budget."""
        m = self.meta
        h = m["history"][-1] if m["history"] else {}
        durations = self._gen_durations[-20:]
        spg = float(np.mean(durations)) if durations else None
        budget = self._phase_budget(m["phase"])
        remaining = max(0, budget - m["gen_in_phase"])
        eta_s = None if spg is None else spg * remaining
        eta = "n/a" if eta_s is None else (
            f"{int(eta_s) // 60}m {int(eta_s) % 60:02d}s"
            if eta_s >= 60 else f"{int(round(eta_s))}s")
        verdicts = []
        for ev in m["events"]:
            e = ev.get("event")
            if e == "growth_kept":
                verdicts.append({
                    "rule": ev.get("rule"),
                    "variant": ev.get("variant", "A"),
                    "verdict": "kept",
                    "detail": f"val {ev.get('val_score'):.6f} > "
                              f"bar {ev.get('bar'):.6f}"})
            elif e == "growth_reverted":
                verdicts.append({
                    "rule": ev.get("rule"),
                    "variant": ev.get("variant", "A"),
                    "verdict": "reverted",
                    "detail": f"val {ev.get('val_score'):.6f} "
                              f"<= bar {ev.get('bar'):.6f}"})
            elif e == "growth_transplant":
                verdicts.append({
                    "rule": ev.get("rule"),
                    "variant": ev.get("variant", "A"),
                    "verdict": "transplanted",
                    "detail": f"preservation PASS, "
                              f"{ev.get('n_params_before')} -> "
                              f"{ev.get('n_params_after')} genes"})
            elif str(e).startswith("growth_rejected"):
                verdicts.append({
                    "rule": ev.get("rule"),
                    "variant": ev.get("variant", "A"),
                    "verdict": "rejected",
                    "detail": str(e)})
        verdicts = verdicts[-8:]
        last = self._last_snapshot
        return {
            "run_name": self.cfg.run_name,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": self.backend.describe(),
            "stage": m["stage_index"],
            "phase": m["phase"],
            "gen_in_phase": m["gen_in_phase"],
            "total_gens": m["total_gens"],
            "n_species": len(self.species),
            "population": int(sum(len(sp["pop"])
                                 for sp in self.species)),
            "best_score": h.get("best_score"),
            "median_score": h.get("median_score"),
            "elim_median": h.get("elim_median"),
            "seconds_per_generation": spg,
            "eta_seconds_phase": eta_s,
            "eta_phase": eta,
            "structure": {
                "stacks": [
                    {"stack": c, "blocks": int(b),
                     "heads": int(hd), "d": int(self.arch.d),
                     "ffn_hidden": int(self.arch.ffn_hidden)}
                    for c, (b, hd) in
                    enumerate(self.arch.couples)],
                "total_genes": int(self.layout.n_params),
                "arch_signature": self.arch.signature(),
                "d": int(self.arch.d),
                "ffn_hidden": int(self.arch.ffn_hidden),
                "n_stacks": len(self.arch.couples)},
            "growth": {
                "rule": m["rule"],
                "growth_round": m["growth_round"],
                "tried_rules": list(m["tried_rules"]),
                "verdicts": verdicts},
            "snapshots_count": self._snapshot_count,
            "latest_snapshot": None if last is None else {
                "total_gens": last["total_gens"],
                "label": last["label"],
                "champion_median_cagr":
                    last["cagr"]["champion_median"],
                "buyhold_median_cagr":
                    last["cagr"]["buyhold_median"],
                "percentile": last["percentile"]["value"]},
            "ladder": None if self.monitor_ladder is None
            else {
                "arch_signature":
                    self.monitor_ladder.get("arch_signature"),
                "n_draws": self.monitor_ladder.get("n_draws"),
                "unit": self.monitor_ladder.get("unit")},
        }

    def _monitor_snapshot(self, label, genome, role,
                          species_id=None, recompute_ladder=False):
        """Build one detailed snapshot, append it, and refresh
        the page. Stock domain only (build_snapshot raises
        for other domains, so the guard below is the only
        gate). Deduped by (total_gens, label): a second call
        with the same key is a no-op. The cached ladder is
        used only when its arch signature matches the
        current one; after an architecture change the
        percentile stays None (with its reason) until the
        stage-end recompute. With recompute_ladder=True the
        ladder is rebuilt from this snapshot's mean invested
        % and attached BEFORE the append, so the filed record
        carries the fresh percentile."""
        if self.cfg.domain != "stock" \
                or not self.cfg.monitor_enabled \
                or genome is None:
            return None
        key = (self.meta["total_gens"], label)
        if key in self._snap_keys:
            return None
        ladder = self.monitor_ladder
        if ladder is not None \
                and ladder.get("arch_signature") \
                != self.arch.signature():
            ladder = None  # stale arch: wait for stage-end recompute
        pop, mask = self._monitor_densities(species_id)
        snap = bm.build_snapshot(
            self.domain, self.layout, genome,
            total_gens=self.meta["total_gens"],
            stage=self.meta["stage_index"],
            phase=self.meta["phase"],
            gen_in_phase=self.meta["gen_in_phase"],
            label=label, genome_role=role,
            density_population=pop, density_masking=mask,
            ladder=ladder)
        if recompute_ladder:
            self._monitor_recompute_ladder(snap)
            bm.attach_ladder(snap, self.monitor_ladder)
        bm.append_snapshot(self.out_dir, snap)
        self._snap_keys.add(key)
        self._snapshot_count += 1
        self._last_snapshot = snap
        # refresh the page AND the status file, so neither
        # lags the snapshot just filed (the per-generation
        # status rewrite in _monitor_after_generation runs
        # before the snapshot is appended)
        bm.write_html(
            self.out_dir,
            bm.render_monitor_html(
                self._monitor_status_dict(),
                bm.read_snapshots(self.out_dir)))
        bm.write_status(self.out_dir,
                        self._monitor_status_dict())
        return snap

    def _monitor_recompute_ladder(self, snapshot):
        """Recompute the exposure-matched random ladder when
        the cached one is missing or belongs to another
        architecture; otherwise return the cache untouched.
        target_invested_pct is the champion's mean invested %
        from the snapshot that triggered the recompute. The
        ladder seed is fixed per run (run_seed + 99991) so
        the ladder is reproducible and independent of the
        breeding RNG stream. Cost: 200 draws x 64 episodes x
        60 days of the domain's own _execute_day (tens of
        seconds at full scale) -- stage end only, never per
        snapshot."""
        arch_sig = self.arch.signature()
        cached = self.monitor_ladder
        if cached is not None \
                and cached.get("arch_signature") == arch_sig:
            return cached
        target = float(snapshot["trades"]["mean_invested_pct"])
        rng = np.random.default_rng(self.cfg.run_seed + 99991)
        medians, unit = bm.random_ladder(
            self.domain, self.layout, self.domain.val_episodes(),
            target, rng, n_draws=bm.LADDER_N_DRAWS)
        self.monitor_ladder = bm.save_ladder_cache(
            self.out_dir / "monitor_ladder.json",
            arch_sig, medians, unit)
        log(f"[engine] random ladder recomputed for arch "
            f"{arch_sig[:12]}...: unit {unit:.4f}, "
            f"{bm.LADDER_N_DRAWS} draws")
        return self.monitor_ladder

    def _monitor_after_generation(self, phase):
        """End-of-generation hook: timing, status rewrite
        every generation, and the snapshot trigger
        (total_gens % monitor_every_gens == 0, phase end,
        stage end -- the latter two are called from
        run_stage). Called at the very end of generation(),
        AFTER save_checkpoint, so a kill during the
        (expensive) snapshot evaluation leaves a valid
        checkpoint."""
        if self.cfg.domain != "stock" \
                or not self.cfg.monitor_enabled:
            return
        if self._gen_t0 is not None:
            self._gen_durations.append(time.time() - self._gen_t0)
            self._gen_t0 = None
        bm.write_status(self.out_dir,
                        self._monitor_status_dict())
        if self.meta["total_gens"] % self.cfg.monitor_every_gens \
                == 0:
            leader = self._gen_leader
            if leader is not None:
                self._monitor_snapshot(
                    "snapshot", leader["genome"], "leader",
                    species_id=leader["species_id"])

    # --------------------------------------------------------- cataclysm
    def do_cataclysm(self, within_stage):
        cfg, m = self.cfg, self.meta
        val_scores = [self.domain.validation_score(
            self.layout, sp["best"]["genome"])
            for sp in self.species]
        founders = be.select_founders(self.species, val_scores,
                                      cfg.founder_count)
        for f in founders:
            self.vault(f["genome"], f["val_score"],
                       "cataclysm_founder", m["stage_index"],
                       f["source_species_id"], f["lineage"])
        before = float(max(f["val_score"] for f in founders))
        pv = be.prob_vector(self.classes, m["phase"], cfg,
                            marks=self.marks,
                            frozen=self.frozen,
                            new_only=self._growth_new_only(),
                            flat_nominal=self._founding_stage())
        self.species = be.regenerate_species(
            founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
        # archive (Addendum 2): children inherit the
        # founder's usefulness vector
        self._inherit_tracker()
        # validation after: best unmutated representative of
        # the regenerated species (founder base + species draw)
        draw_mask = self.classes == be.CLS_DRAW
        after = -np.inf
        for sp in self.species:
            f = next(f for f in founders
                     if f["lineage"] == sp["lineage"])
            rep = f["genome"].copy()
            rep[draw_mask] = sp["draw"][draw_mask]
            after = max(after, self.domain.validation_score(
                self.layout, rep))
        ev = {"event": "cataclysm",
              "generation": m["total_gens"],
              "stage": m["stage_index"], "phase": m["phase"],
              "within_stage": within_stage,
              "founders": [{"species_id": f["source_species_id"],
                            "lineage": f["lineage"],
                            "val_score": f["val_score"]}
                           for f in founders],
              "val_before": before, "val_after": float(after),
              "species_after": len(self.species)}
        m["events"].append(ev)
        log(f"[engine] CATACLYSM at gen {m['total_gens']}: "
            f"founders {[f['source_species_id'] for f in founders]}"
            f" val before {before:.6f} after {after:.6f}; "
            f"regenerated {len(self.species)} species")
        return ev

    # ------------------------------------------------------------- stage
    def run_stage(self):
        cfg, m = self.cfg, self.meta
        stage = m["stage_index"]
        for phase in ("A", "B"):
            if m["phase"] != phase:
                continue
            t_phase = time.time()
            self.phase_cache = {} if phase == "A" else None
            self.elim_series = []
            while m["gen_in_phase"] < self._phase_budget(phase):
                self.gen_cache = {}
                self.generation(phase)
                h = m["history"][-1]
                log(f"[engine] stage {stage} phase {phase} gen "
                    f"{h['gen_in_phase']}: species "
                    f"{h['n_species']} pop {h['pop_total']} best "
                    f"{h['best_score']:.4f} elim median "
                    f"{h['elim_median']:.4f}"
                    f"{' (cull)' if h['cull'] else ''}")
                if self.stop_after is not None \
                        and self.gens_this_invocation \
                        >= self.stop_after:
                    return "stopped"
                pr = be.plateau_read(self.elim_series,
                                     cfg.plateau_window,
                                     cfg.plateau_threshold)
                if pr["plateaued"]:
                    log(f"[engine] stage {stage} phase {phase}: "
                        f"PLATEAU ({pr['relative_improvement_over_window']:.4f} "
                        f"rel improvement over window "
                        f"{cfg.plateau_window})")
                    m.setdefault("plateau", {})[phase] = pr
                    break
            m["phase_timings"].append({
                "stage": stage, "phase": phase,
                "seconds": round(time.time() - t_phase, 2),
                "generations": m["gen_in_phase"]})
            # monitor (Addendum 3): phase-end snapshot of the
            # last generation's leader (runs after a plateau
            # break too, since the hook sits after the loop)
            if self.cfg.domain == "stock" \
                    and self.cfg.monitor_enabled:
                leader = self._gen_leader
                if leader is not None:
                    self._monitor_snapshot(
                        "phase_end", leader["genome"], "leader",
                        species_id=leader["species_id"])
            if phase == "A":
                m["phase"] = "B"
                m["gen_in_phase"] = 0
                self.save_checkpoint()
        # stage champion on the fixed validation set
        val_scores = [self.domain.validation_score(
            self.layout, sp["best"]["genome"])
            for sp in self.species]
        bi = int(np.argmax(val_scores))
        champ = self.species[bi]["best"]["genome"]
        detail = self.domain.validation_detail(self.layout, champ)
        fn = self.vault(champ, float(val_scores[bi]),
                        "stage_champion", stage,
                        self.species[bi]["id"],
                        self.species[bi]["lineage"], detail)
        if cfg.silence_enabled:
            self._silence_champion(champ, stage, fn)
        # monitor (Addendum 3): stage-end snapshot of the
        # official stage champion, with the ladder recomputed
        # from the champion's mean invested %
        if self.cfg.domain == "stock" \
                and self.cfg.monitor_enabled:
            self._monitor_snapshot(
                "stage_end", champ, "stage_champion",
                species_id=int(self.species[bi]["id"]),
                recompute_ladder=True)
        result = {"stage": stage, "rule": m["rule"],
                  "champion_species_id":
                      int(self.species[bi]["id"]),
                  "champion_val_score": float(val_scores[bi]),
                  "champion_detail": detail,
                  "bar_val_score": m["bar_val_score"],
                  "vault_file": fn,
                  "plateau": m.pop("plateau", {})}
        m["stage_results"].append(result)
        m["events"].append({"event": "stage_end", **result})
        log(f"[engine] stage {stage} end: champion species "
            f"{result['champion_species_id']} val score "
            f"{result['champion_val_score']:.6f} "
            f"(bar {m['bar_val_score']})")
        return result

    # ------------------------------------------------------- growth step
    def growth_attempt(self, rule, founders, variant="A",
                       split_units=None):
        """Transplant every founder through `rule`, gate on
        the domain's exact preservation check (founder 0's
        representative), then regenerate the species set
        split among the transplanted founders. Variant B of
        widen is split-widening (plan 2026-11-22 step 2):
        the split units come from the champion's silencing
        table, and the split noise is applied to the
        templates AFTER the preservation gate passes, as the
        first change of the new-part phase. The live marks
        are remapped across the transplant by gene name, so
        the next stage's phase B still marks the champion's
        units."""
        cfg, m = self.cfg, self.meta
        n_params_before = int(self.layout.n_params)
        layout_old = self.layout
        if rule == "widen" and variant == "B":
            table = m.get("silencing_table")
            if table is None:
                m["events"].append({
                    "event": "growth_rejected_no_table",
                    "rule": rule, "variant": variant,
                    "stage": m["stage_index"] + 1})
                log("[engine] growth rule widen variant B: no "
                    "silencing table (silencing off or no "
                    "champion yet); rule skipped")
                return False
            if split_units is None:
                split_units = be.split_units_from_table(
                    table, cfg.widen_step)
            real = any(u["drop"] > self.domain.silence_zero_tol
                       for u in table["units"]
                       if u["kind"] == "ffn")
            if not split_units or not real:
                m["events"].append({
                    "event": "growth_rejected_no_split_units",
                    "rule": rule, "variant": variant,
                    "stage": m["stage_index"] + 1})
                log("[engine] growth rule widen variant B: no "
                    "FFN unit with a real silencing drop; "
                    "rule skipped")
                return False
        arch_new, layout_new, classes_new = None, None, None
        templates = []
        for f in founders:
            a2, l2, tmpl, cl2 = be.make_transplant(
                rule, self.arch, self.layout, f["genome"],
                self.head_spec_fn, cfg, self.widen_head,
                variant=variant, split_units=split_units)
            arch_new, layout_new, classes_new = a2, l2, cl2
            templates.append(tmpl)
        # preservation gate: representative species genome =
        # template + founder 0's new draw, vs the founder
        rng0 = np.random.default_rng(cfg.draw_seed + 999)
        draw_mask = classes_new == be.CLS_DRAW
        rep = templates[0].copy()
        dvals = rng0.normal(0, cfg.draw_sigma, int(draw_mask.sum()))
        rep[draw_mask] = dvals
        pres = self.domain.preservation(
            self.layout, founders[0]["genome"], layout_new, rep)
        if not pres["pass"]:
            m["events"].append({
                "event": "growth_rejected_preservation",
                "rule": rule, "variant": variant,
                "preservation": pres,
                "stage": m["stage_index"] + 1})
            log(f"[engine] growth rule {rule} variant {variant}: "
                f"preservation check FAILED ({pres}); rule "
                "skipped")
            return False
        # split noise: AFTER the gate passes, on the copies'
        # genes only. The measured move of the gate
        # representative's validation score under the noise is
        # recorded in the event (verification leg 2).
        noise_val_delta = None
        if rule == "widen" and variant == "B":
            templates = [be.apply_split_noise(
                t, layout_new, split_units, self.rng,
                cfg.split_noise_lo, cfg.split_noise_hi)
                for t in templates]
            rep_noisy = templates[0].copy()
            rep_noisy[draw_mask] = dvals
            noise_val_delta = float(
                self.domain.validation_score(layout_new,
                                             rep_noisy)
                - self.domain.validation_score(layout_new, rep))
            log(f"[engine] growth rule widen variant B: split "
                f"noise moved the representative validation "
                f"score by {noise_val_delta:.6g}")
        # commit the transplant
        self.arch, self.layout, self.classes = \
            arch_new, layout_new, classes_new
        if self.marks is not None:
            self.marks = be.remap_marks(self.marks, layout_old,
                                        layout_new)
            m["marks"] = self.marks.tolist()
        if self.frozen is not None:
            self.frozen = be.remap_frozen(self.frozen,
                                          layout_old,
                                          layout_new)
            m["frozen"] = self.frozen.tolist()
        # archive (Addendum 2): rebase the usefulness tracker
        # into the new layout BEFORE regenerating, so the
        # founders' vectors are already remapped when the
        # children inherit them
        self._rebase_tracker(layout_old, layout_new)
        new_founders = []
        for f, tmpl in zip(founders, templates):
            nf = dict(f)
            nf["genome"] = tmpl
            new_founders.append(nf)
        pv = be.prob_vector(self.classes, "A", cfg,
                            marks=self.marks,
                            frozen=self.frozen,
                            new_only=self._growth_new_only())
        self.species = be.regenerate_species(
            new_founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
        # archive (Addendum 2): children inherit the
        # founder's (remapped) usefulness vector
        self._inherit_tracker()
        m["events"].append({
            "event": "growth_transplant", "rule": rule,
            "variant": variant,
            "preservation": pres,
            "noise_val_delta": noise_val_delta,
            "split_units": [list(u) for u in split_units]
                           if split_units else None,
            "n_params_before": n_params_before,
            "n_params_after": int(self.layout.n_params),
            "stage": m["stage_index"] + 1})
        log(f"[engine] growth rule {rule} variant {variant}: "
            f"preservation PASS ({pres}); new arch "
            f"{self.arch.signature()}, "
            f"{self.layout.n_params} genes")
        return True

    @staticmethod
    def _next_widen_variant(variant, kept):
        """Widen variant bookkeeping (plan step 2): a kept
        widen (either variant) restarts the next round with
        variant A; a reverted variant-A widen arms variant B
        (split-widening) for the next widen attempt; a
        reverted variant B goes back to A."""
        if kept:
            return "A"
        return "B" if variant == "A" else "A"

    # -------------------------------------------------------------- main
    def run(self):
        """Explicit state machine, safe to resume at any
        checkpoint: a stage in progress is finished first
        (with its keep/revert decision if it was a growth
        attempt), then the growth decision point advances
        rule by rule."""
        cfg, m = self.cfg, self.meta
        while True:
            if not m.get("_stage_done"):
                out = self.run_stage()
                if out == "stopped":
                    return {"stopped_early": True}
                m["_stage_done"] = True
                val_scores = [self.domain.validation_score(
                    self.layout, sp["best"]["genome"])
                    for sp in self.species]
                founders_new = be.select_founders(
                    self.species, val_scores, cfg.founder_count)
                if m["rule"] is None:
                    # initial stage: accepted by definition
                    m["champion"] = {
                        "genome": self._champion_genome(out),
                        "val_score": out["champion_val_score"],
                        "marks": m.get("marks"),
                        "frozen": m.get("frozen"),
                        "silencing_table":
                            m.get("silencing_table")}
                    m["champion_arch_signature"] = \
                        self.arch.signature()
                    m["champion_classes"] = self.classes.tolist()
                    m["founders"] = founders_new
                    m["events"].append({
                        "event": "initial_stage_accepted",
                        "val_score": out["champion_val_score"]})
                elif out["champion_val_score"] > m["bar_val_score"]:
                    bar = m["bar_val_score"]
                    m["champion"] = {
                        "genome": self._champion_genome(out),
                        "val_score": out["champion_val_score"],
                        "marks": m.get("marks"),
                        "frozen": m.get("frozen"),
                        "silencing_table":
                            m.get("silencing_table")}
                    m["champion_arch_signature"] = \
                        self.arch.signature()
                    m["champion_classes"] = self.classes.tolist()
                    m["founders"] = founders_new
                    m["events"].append({
                        "event": "growth_kept", "rule": m["rule"],
                        "variant": m.get("rule_variant", "A"),
                        "val_score": out["champion_val_score"],
                        "bar": bar})
                    log(f"[engine] growth rule {m['rule']} KEPT: "
                        f"{out['champion_val_score']:.6f} > bar "
                        f"{bar:.6f}")
                    m["growth_round"] += 1
                    m["tried_rules"] = []
                    if m["rule"] == "widen":
                        # a fresh round starts with variant A
                        m["widen_variant"] = \
                            self._next_widen_variant(
                                m.get("rule_variant", "A"),
                                kept=True)
                else:
                    bar = m["bar_val_score"]
                    m["events"].append({
                        "event": "growth_reverted",
                        "rule": m["rule"],
                        "variant": m.get("rule_variant", "A"),
                        "val_score": out["champion_val_score"],
                        "bar": bar})
                    log(f"[engine] growth rule {m['rule']} "
                        f"REVERTED: "
                        f"{out['champion_val_score']:.6f} <= bar "
                        f"{bar:.6f}")
                    if m["rule"] == "widen":
                        # a reverted variant-A widen arms
                        # variant B for the next widen attempt;
                        # a reverted variant B goes back to A
                        m["widen_variant"] = \
                            self._next_widen_variant(
                                m.get("rule_variant", "A"),
                                kept=False)
                    self._revert_to_champion()
                for f in m["founders"]:
                    f["genome"] = np.asarray(f["genome"]).tolist()
                self.save_checkpoint()
                continue
            # ---- growth decision point ----------------------
            if m["growth_round"] >= cfg.max_growth_rounds:
                break
            remaining = [r for r in cfg.growth_order
                         if r not in m["tried_rules"]]
            if not remaining:
                m["events"].append({
                    "event": "growth_exhausted",
                    "growth_round": m["growth_round"]})
                log("[engine] all growth rules exhausted for "
                    "this round; stopping")
                break
            rule = remaining[0]
            founders = [dict(f, genome=np.array(f["genome"]))
                        for f in m["founders"]]
            bar = float(max(f["val_score"] for f in founders))
            if not m["tried_rules"] and cfg.cataclysm_enabled \
                    and cfg.cataclysm_every_n_generations is None:
                # cataclysm at the growth transition (the
                # default trigger): founders vaulted before
                # the event; regeneration IS the growth
                # attempt's regeneration
                for f in founders:
                    self.vault(f["genome"], f["val_score"],
                               "cataclysm_founder",
                               m["stage_index"],
                               f["source_species_id"],
                               f["lineage"])
                m["events"].append({
                    "event": "cataclysm",
                    "generation": m["total_gens"],
                    "stage": m["stage_index"],
                    "phase": "transition",
                    "within_stage": False,
                    "founders": [
                        {"species_id": f["source_species_id"],
                         "lineage": f["lineage"],
                         "val_score": f["val_score"]}
                        for f in founders],
                    "val_before": bar, "val_after": None,
                    "note": "transition cataclysm: species "
                            "regeneration is the growth "
                            "attempt's regeneration"})
            # widen variant selection (plan step 2): a
            # variant-A widen that reverted arms variant B
            # (split-widening) for the next widen attempt;
            # variant B needs the champion's silencing table,
            # without it the attempt falls back to A
            variant = "A"
            split_units = None
            if rule == "widen":
                variant = m.get("widen_variant", "A")
                if variant == "B" \
                        and m.get("silencing_table") is None:
                    variant = "A"
                if variant == "B":
                    split_units = be.split_units_from_table(
                        m["silencing_table"], cfg.widen_step)
            m["rule_variant"] = variant
            ok = self.growth_attempt(rule, founders,
                                     variant=variant,
                                     split_units=split_units)
            m["tried_rules"].append(rule)
            if not ok:
                if rule == "widen":
                    # a failed widen attempt arms the other
                    # variant for the next widen attempt
                    m["widen_variant"] = \
                        self._next_widen_variant(variant,
                                                 kept=False)
                self.save_checkpoint()
                continue
            m["stage_index"] += 1
            m["rule"] = rule
            m["bar_val_score"] = bar
            m["phase"] = "A"
            m["gen_in_phase"] = 0
            m["stage_gen"] = 0
            m["_stage_done"] = False
            self.save_checkpoint()
        m["done"] = True
        return self.finish()

    def _champion_genome(self, stage_result):
        for sp in self.species:
            if sp["id"] == stage_result["champion_species_id"]:
                return sp["best"]["genome"].tolist()
        raise RuntimeError("champion species not found")

    def _revert_to_champion(self):
        """Restore the pre-growth architecture and rebuild
        the species set from the stage founders (the revert
        returns to the champion architecture; the failed
        stage's species are discarded)."""
        cfg, m = self.cfg, self.meta
        # restore the champion architecture / layout / gene
        # classes recorded when the champion was accepted,
        # and rebuild the species set from the stage founders
        # (which live in that architecture)
        layout_old = self.layout
        arch_sig = m["champion_arch_signature"]
        self.arch = be.ArchSpec.from_signature(arch_sig)
        self.layout = be.Layout(self.arch,
                                self.head_spec_fn(self.arch.d))
        # archive (Addendum 2): rebase the usefulness tracker
        # back into the champion layout before regenerating
        self._rebase_tracker(layout_old, self.layout)
        # classes for the champion arch: rebuild from the
        # stage's class record
        self.classes = np.array(m["champion_classes"],
                                dtype=np.int8)
        # the accepted champion keeps its own silencing table
        # and marks (in this architecture): restore them as
        # the live state, discarding the failed stage's
        # table and marks (which belong to the grown layout)
        champ = m["champion"]
        if champ.get("marks") is not None:
            self.marks = np.array(champ["marks"], dtype=float)
            m["marks"] = champ["marks"]
            self.silencing_table = champ.get("silencing_table")
            m["silencing_table"] = champ.get("silencing_table")
        else:
            self.marks = None
            m["marks"] = None
            self.silencing_table = None
            m["silencing_table"] = None
        if champ.get("frozen") is not None:
            self.frozen = np.array(champ["frozen"], dtype=bool)
            m["frozen"] = champ["frozen"]
        else:
            self.frozen = None
            m["frozen"] = None
        founders = [dict(f, genome=np.array(f["genome"]))
                    for f in m["founders"]]
        pv = be.prob_vector(self.classes, "A", cfg,
                            marks=self.marks,
                            frozen=self.frozen,
                            new_only=self._growth_new_only())
        self.species = be.regenerate_species(
            founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
        # archive (Addendum 2): children inherit the
        # founder's (remapped) usefulness vector
        self._inherit_tracker()
        m["phase"] = "A"
        m["gen_in_phase"] = 0
        m["stage_gen"] = 0
        m["_stage_done"] = True   # no stage to run until a
        # growth attempt commits

    def finish(self):
        cfg, m = self.cfg, self.meta
        results = {
            "run_name": cfg.run_name, "domain": cfg.domain,
            "backend": self.backend.describe(),
            "config": json.loads(cfg.canonical()),
            "final_arch": json.loads(self.arch.signature()),
            "n_params": int(self.layout.n_params),
            "champion": ({"val_score":
                          m["champion"]["val_score"]}
                         if m["champion"] else None),
            "stage_results": m["stage_results"],
            "events": m["events"],
            "history_tail": m["history"][-50:],
            "phase_timings": m["phase_timings"],
            "total_generations": m["total_gens"],
            "wall_seconds": round(time.time() - self.t0, 1),
        }
        be.atomic_json(self.out_dir / "results.json", results)
        # archive (Addendum 2): lineage retention + heatmap,
        # best-effort -- never fail the run
        if getattr(cfg, "archive_enabled", True):
            try:
                rep = ba.analyse_run(self.out_dir)
                log(f"[engine] archive analysis: winner "
                    f"species {rep['winner_species_id']}, chain "
                    f"{rep['chain']}, kept {rep['bytes_kept']} "
                    f"bytes, deleted {rep['bytes_deleted']} "
                    f"bytes, heatmap {rep['heatmap_path']}")
            except Exception as exc:  # noqa: BLE001 - best effort
                log("[engine] archive analysis failed "
                    f"(non-fatal): {exc}")
        # monitor (Addendum 3): final snapshot safety net --
        # normally a no-op (the stage-end hook already filed
        # a stage_end snapshot at the same total_gens, and
        # the (total_gens, label) dedupe skips the second one)
        if self.cfg.domain == "stock" \
                and self.cfg.monitor_enabled:
            champ_m = self.meta.get("champion")
            if champ_m is not None \
                    and champ_m.get("genome") is not None:
                self._monitor_snapshot(
                    "stage_end", np.asarray(champ_m["genome"]),
                    "stage_champion", recompute_ladder=True)
            # final status rewrite, so the file carries the
            # run's last verdict (kept/reverted) and the
            # final snapshot count, not the last generation's
            bm.write_status(self.out_dir,
                            self._monitor_status_dict())
        champ = m["champion"]
        if champ is not None:
            be.atomic_npz(self.out_dir / "champion.npz",
                          genome=np.array(champ["genome"]),
                          val_score=champ["val_score"])
        sentinel = self.out_dir / "COMPLETE.txt"
        sentinel.write_text(
            f"completed {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"domain {cfg.domain} stages "
            f"{len(m['stage_results'])} champion val score "
            f"{champ['val_score'] if champ else None}\n")
        log(f"[engine] DONE: {len(m['stage_results'])} stages, "
            f"champion val score "
            f"{champ['val_score'] if champ else None}, "
            f"{results['wall_seconds']}s")
        return results


def smoke_config(domain="tsp"):
    cfg = be.EngineConfig(
        domain=domain, backend="numpy", device="cpu",
        precision="x64", d=8, ffn_hidden=16, initial_blocks=1,
        initial_heads=1, n_species=6, per_species=5, chunk=16,
        n_train=16, n_elim=16, n_val=40, phase_a_gens=10,
        phase_b_gens=10, cull_interval=3, species_floor=2,
        plateau_window=6, founder_count=1, max_growth_rounds=1,
        run_name=f"bert_engine_smoke_{domain}")
    if domain == "stock":
        cfg.domain_config = {
            "synthetic_panel": True, "synthetic_tickers": 30,
            "synthetic_days": 500, "episode_days": 20,
            "basket_n": 6, "n_train_episodes": 2,
            "n_elim_episodes": 3}
    else:
        cfg.domain_config = {"n_cities": 10}
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lifecycle", action="store_true",
                    help="run the Frankenstein lifecycle "
                         "(bert_lifecycle.py) as the engine's "
                         "growth sequencer, instead of a "
                         "single engine run")
    ap.add_argument("--lifecycle-config", default=None,
                    help="lifecycle config JSON for "
                         "--lifecycle (default: "
                         "bert_lifecycle_config.json beside "
                         "this runner)")
    args = ap.parse_args(argv)
    if args.lifecycle:
        # The Frankenstein lifecycle is the engine's growth
        # sequencer (method frozen by the sponsor,
        # 2026-11-27; FRANKENSTEIN_METHOD.md). The stage
        # code lives in bert_lifecycle.py and is imported,
        # never duplicated here; the lifecycle's own
        # founding stage launches this runner WITHOUT the
        # flag, so there is no recursion. --out-dir, when
        # given, is the lifecycle run directory.
        import bert_lifecycle as bl
        lc_argv = ["run", "--config",
                   args.lifecycle_config
                   or str(HERE / "bert_lifecycle_config.json")]
        if args.out_dir:
            lc_argv += ["--run-dir", args.out_dir]
        return bl.main(lc_argv)
    if args.smoke:
        cfg = smoke_config(args.domain or "tsp")
    elif args.config:
        cfg = be.EngineConfig.load(args.config)
        cfg.config_dir = str(Path(args.config).resolve().parent)
        if args.domain:
            cfg.domain = args.domain
    else:
        cfg = be.EngineConfig.load(
            HERE / "bert_engine_config.json")
    out_dir = Path(args.out_dir) if args.out_dir else \
        HERE / "engine_runs" / cfg.run_name
    eng = Engine(cfg, out_dir, stop_after=args.stop_after)
    sentinel = out_dir / "COMPLETE.txt"
    if sentinel.exists() and not args.fresh:
        log(f"[engine] run '{cfg.run_name}' already complete "
            f"(sentinel present); use --fresh to rerun")
        return 0
    if eng.ckpt_path.exists() and not args.fresh:
        eng.setup_resume()
    else:
        if args.fresh and eng.ckpt_path.exists():
            eng.ckpt_path.unlink()
        eng.setup_fresh()
        eng.save_checkpoint()
    out = eng.run()
    if isinstance(out, dict) and out.get("stopped_early"):
        log(f"[engine] stopped after {args.stop_after} "
            f"generations; checkpoint holds the state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
