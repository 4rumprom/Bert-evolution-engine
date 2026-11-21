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
    classes = np.full(layout.n_params, be.CLS_DRAW, dtype=np.int8)
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

    # ------------------------------------------------------ checkpoint
    def save_checkpoint(self):
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
        pv = be.prob_vector(self.classes, "A", cfg)
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
            "done": False,
        }
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
        log(f"[engine] RESUMED: stage {self.meta['stage_index']} "
            f"phase {self.meta['phase']} gen "
            f"{self.meta['gen_in_phase']}, "
            f"{len(self.species)} species")

    # ---------------------------------------------------- one generation
    def generation(self, phase):
        cfg, m = self.cfg, self.meta
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
            pv = be.prob_vector(self.classes, phase, cfg)
            for sp in self.species:
                sp["pop"] = be.breed_species(
                    sp["pop"], sp["lens"], sp["draw"],
                    self.classes, pv, self.rng, cfg, sp["target"])
        self.save_checkpoint()

    def _phase_budget(self, phase):
        return self.cfg.phase_a_gens if phase == "A" \
            else self.cfg.phase_b_gens

    def _phase_ending(self, phase):
        return self.meta["gen_in_phase"] >= self._phase_budget(phase)

    def _stage_ending(self):
        m = self.meta
        if m["phase"] == "A":
            return False
        return m["gen_in_phase"] >= self.cfg.phase_b_gens - 1

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
        pv = be.prob_vector(self.classes, m["phase"], cfg)
        self.species = be.regenerate_species(
            founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
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
    def growth_attempt(self, rule, founders):
        """Transplant every founder through `rule`, gate on
        the domain's exact preservation check (founder 0's
        representative), then regenerate the species set
        split among the transplanted founders."""
        cfg, m = self.cfg, self.meta
        n_params_before = int(self.layout.n_params)
        arch_new, layout_new, classes_new = None, None, None
        templates = []
        for f in founders:
            a2, l2, tmpl, cl2 = be.make_transplant(
                rule, self.arch, self.layout, f["genome"],
                self.head_spec_fn, cfg, self.widen_head)
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
                "rule": rule, "preservation": pres,
                "stage": m["stage_index"] + 1})
            log(f"[engine] growth rule {rule}: preservation "
                f"check FAILED ({pres}); rule skipped")
            return False
        # commit the transplant
        self.arch, self.layout, self.classes = \
            arch_new, layout_new, classes_new
        new_founders = []
        for f, tmpl in zip(founders, templates):
            nf = dict(f)
            nf["genome"] = tmpl
            new_founders.append(nf)
        pv = be.prob_vector(self.classes, "A", cfg)
        self.species = be.regenerate_species(
            new_founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
        m["events"].append({
            "event": "growth_transplant", "rule": rule,
            "preservation": pres,
            "n_params_before": n_params_before,
            "n_params_after": int(self.layout.n_params),
            "stage": m["stage_index"] + 1})
        log(f"[engine] growth rule {rule}: preservation PASS "
            f"({pres}); new arch {self.arch.signature()}, "
            f"{self.layout.n_params} genes")
        return True

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
                        "val_score": out["champion_val_score"]}
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
                        "val_score": out["champion_val_score"]}
                    m["champion_arch_signature"] = \
                        self.arch.signature()
                    m["champion_classes"] = self.classes.tolist()
                    m["founders"] = founders_new
                    m["events"].append({
                        "event": "growth_kept", "rule": m["rule"],
                        "val_score": out["champion_val_score"],
                        "bar": bar})
                    log(f"[engine] growth rule {m['rule']} KEPT: "
                        f"{out['champion_val_score']:.6f} > bar "
                        f"{bar:.6f}")
                    m["growth_round"] += 1
                    m["tried_rules"] = []
                else:
                    bar = m["bar_val_score"]
                    m["events"].append({
                        "event": "growth_reverted",
                        "rule": m["rule"],
                        "val_score": out["champion_val_score"],
                        "bar": bar})
                    log(f"[engine] growth rule {m['rule']} "
                        f"REVERTED: "
                        f"{out['champion_val_score']:.6f} <= bar "
                        f"{bar:.6f}")
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
            ok = self.growth_attempt(rule, founders)
            m["tried_rules"].append(rule)
            if not ok:
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
        arch_sig = m["champion_arch_signature"]
        self.arch = be.ArchSpec.from_signature(arch_sig)
        self.layout = be.Layout(self.arch,
                                self.head_spec_fn(self.arch.d))
        # classes for the champion arch: rebuild from the
        # stage's class record
        self.classes = np.array(m["champion_classes"],
                                dtype=np.int8)
        founders = [dict(f, genome=np.array(f["genome"]))
                    for f in m["founders"]]
        pv = be.prob_vector(self.classes, "A", cfg)
        self.species = be.regenerate_species(
            founders, self.classes, cfg.n_species,
            cfg.per_species, self.draw_rng, self.rng, pv, cfg,
            next_id=m["next_species_id"])
        m["next_species_id"] += cfg.n_species
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
    args = ap.parse_args(argv)
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
