# Monitor hooks: exact code the runner adds

Module: `bert_monitor.py` (same directory). The runner imports
it; the module never imports the runner. All hooks are a few
lines calling the module, plus small `Engine` helper methods
that live in `run_bert_engine.py`. Nothing here edits
`bert_monitor.py`.

## 0. Prerequisites (already in the tree — verify, do not re-add)

`EngineConfig` (`bert_engine.py`) already carries the monitor
knobs (plan Addendum 3):

```python
monitor_enabled: bool = True
monitor_every_gens: int = 10
```

No config or checkpoint change is needed for the monitor:
status/snapshots/html live in the run directory as plain
files, and the ladder cache is keyed by architecture
signature, so a resumed run picks the cache up by reading
`monitor_ladder.json` (hook 2).

## 1. Import

At the top of `run_bert_engine.py`, next to the other imports:

```python
import bert_monitor as bm
```

## 2. Monitor state: `__init__` / `setup_fresh` / `setup_resume`

In `Engine.__init__`, next to `self.marks = None` /
`self.silencing_table = None`:

```python
self.monitor_ladder = None   # cached ladder dict or None
self._gen_durations = []     # seconds per generation (last 20)
self._gen_t0 = None          # generation start timestamp
self._gen_leader = None      # leader genome of the last generation
self._snap_keys = set()      # (total_gens, label) already snapshotted
self._snapshot_count = 0
self._last_snapshot = None
```

In `setup_fresh`, after `self.meta = {...}`:

```python
self.monitor_ladder = None
self._snap_keys = set()
self._snapshot_count = 0
self._last_snapshot = None
```

In `setup_resume`, after `self.meta = state["meta"]`:

```python
self.monitor_ladder = bm.load_ladder_cache(
    self.out_dir / "monitor_ladder.json")
self._snap_keys = {
    (s["total_gens"], s["label"])
    for s in bm.read_snapshots(self.out_dir)}
snaps = bm.read_snapshots(self.out_dir)
self._snapshot_count = len(snaps)
self._last_snapshot = snaps[-1] if snaps else None
```

(`read_snapshots` tolerates a torn trailing line, so a resume
after a kill mid-generation never chokes on the jsonl.)

## 3. Engine helper methods (add to the `Engine` class)

```python
def _monitor_densities(self, species_id):
    """Density readings for a snapshot. Population density
    comes from the archive tracker when it exists (optional
    dependency — the monitor works without it); masking
    density comes from the current champion's silencing
    table in meta. Either may be None."""
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
                     if u["drop"] > self.domain.silence_zero_tol)
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
                "detail": f"val {ev.get('val_score'):.6f} <= "
                          f"bar {ev.get('bar'):.6f}"})
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
        "ladder": None if self.monitor_ladder is None else {
            "arch_signature":
                self.monitor_ladder.get("arch_signature"),
            "n_draws": self.monitor_ladder.get("n_draws"),
            "unit": self.monitor_ladder.get("unit")},
    }

def _monitor_snapshot(self, label, genome, role,
                      species_id=None, recompute_ladder=False):
    """Build one detailed snapshot, append it, and refresh
    the page. Stock domain only (build_snapshot raises for
    other domains, so the guard below is the only gate).
    Deduped by (total_gens, label): a second call with the
    same key is a no-op. The cached ladder is used only
    when its arch signature matches the current one; after
    an architecture change the percentile stays None (with
    its reason) until the stage-end recompute. With
    recompute_ladder=True the ladder is rebuilt from this
    snapshot's mean invested % and attached BEFORE the
    append, so the filed record carries the fresh
    percentile."""
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
    bm.write_html(
        self.out_dir,
        bm.render_monitor_html(
            self._monitor_status_dict(),
            bm.read_snapshots(self.out_dir)))
    return snap

def _monitor_recompute_ladder(self, snapshot):
    """Recompute the exposure-matched random ladder when
    the cached one is missing or belongs to another
    architecture; otherwise return the cache untouched.
    target_invested_pct is the champion's mean invested %
    from the snapshot that triggered the recompute. The
    ladder seed is fixed per run (run_seed + 99991) so the
    ladder is reproducible and independent of the breeding
    RNG stream. Cost: 200 draws x 64 episodes x 60 days of
    the domain's own _execute_day (tens of seconds at full
    scale) — stage end only, never per snapshot."""
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
    """End-of-generation hook: timing, status rewrite every
    generation, and the snapshot trigger (total_gens %
    monitor_every_gens == 0, phase end, stage end — the
    latter two are called from run_stage, see hook 5).
    Called at the very end of generation(), AFTER
    save_checkpoint, so a kill during the (expensive)
    snapshot evaluation leaves a valid checkpoint."""
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
```

## 4. Per-generation hook points in `generation()`

At the START of `generation(phase)` (timing):

```python
self._gen_t0 = time.time()
```

Immediately AFTER the elimination loop (the loop that fills
`elim_best_scores`), BEFORE the culling block — this is the
only point where the evaluated generation's populations are
still intact (breeding and culling happen later in the
method). The leader genome is the species generation-best
with the best elimination score:

```python
if self.cfg.domain == "stock" and self.cfg.monitor_enabled:
    bi = int(np.argmax(elim_best_scores))
    sp = self.species[bi]
    self._gen_leader = {
        "genome": sp["pop"][int(np.argmax(sp["lens"]))].copy(),
        "species_id": int(sp["id"]),
        "elim_score": float(elim_best_scores[bi])}
```

At the VERY END of `generation(phase)`, after
`self.save_checkpoint()`:

```python
self._monitor_after_generation(phase)
```

## 5. Phase-end and stage-end snapshots in `run_stage()`

After the phase `while` loop ends (after
`m["phase_timings"].append({...})` for that phase), one
phase-end snapshot of the last generation's leader:

```python
if self.cfg.domain == "stock" and self.cfg.monitor_enabled:
    leader = self._gen_leader
    if leader is not None:
        self._monitor_snapshot(
            "phase_end", leader["genome"], "leader",
            species_id=leader["species_id"])
```

At stage end, AFTER the stage champion is vaulted and
silenced (after the `if cfg.silence_enabled:
self._silence_champion(...)` block), BEFORE `result = {...}`.
This is the official stage champion (not the generation
leader), and it is the point where the ladder is recomputed
(the champion's mean invested % is now known):

```python
if self.cfg.domain == "stock" and self.cfg.monitor_enabled:
    self._monitor_snapshot(
        "stage_end", champ, "stage_champion",
        species_id=int(self.species[bi]["id"]),
        recompute_ladder=True)
```

(`champ` and `bi` are the locals already in scope there:
`champ = self.species[bi]["best"]["genome"]`.)

## 6. Final snapshot in `finish()`

After `be.atomic_json(self.out_dir / "results.json", results)`:

```python
if self.cfg.domain == "stock" and self.cfg.monitor_enabled:
    champ = self.meta.get("champion")
    if champ is not None and champ.get("genome") is not None:
        self._monitor_snapshot(
            "stage_end", np.asarray(champ["genome"]),
            "stage_champion", recompute_ladder=True)
```

In the normal path this is a no-op: the stage-end hook
(hook 5) already filed a `stage_end` snapshot at the same
`total_gens`, and the `(total_gens, label)` dedupe skips the
second one. It exists as the safety net for a run that
reaches `finish()` without a stage-end snapshot (e.g. a
resume whose snapshot files were lost).

## 7. Ladder recompute points (exact)

- At stage end (hook 5): the stage-end snapshot is built
  first, then `_monitor_recompute_ladder` runs from that
  snapshot's `trades.mean_invested_pct`, and the fresh
  ladder is attached to the snapshot before it is appended.
- Whenever the architecture changes: the architecture only
  changes in `growth_attempt`, between stages, so the next
  stage end is the recompute point. In between, snapshots
  use the cached ladder ONLY if its `arch_signature`
  matches the current architecture; on mismatch the
  snapshot's percentile is None with the reason "no ladder
  computed yet (ladder is built at stage end and whenever
  the architecture changes)" — the page shows n/a there,
  never a stale percentile.
- The cache file is `monitor_ladder.json` in the run dir,
  written atomically by `save_ladder_cache`, loaded at
  resume (hook 2). No recompute happens when the cached
  signature matches.

## 8. Snapshot cadence (exact)

- Every generation: `monitor_status.json` is rewritten
  (`_monitor_after_generation`); no snapshot.
- Snapshot (label `"snapshot"`, role `"leader"`) when
  `total_gens % monitor_every_gens == 0` (default 10).
- Snapshot (label `"phase_end"`, role `"leader"`) at every
  phase end, including a plateau break (the hook sits after
  the phase loop, so the break path still files it).
- Snapshot (label `"stage_end"`, role `"stage_champion"`)
  at every stage end, with the ladder recomputed and
  attached; plus the `finish()` safety net (hook 6).
- The leader genome is the species generation-best with
  the best elimination score, captured before breeding in
  `generation()` (hook 4); at stage end the genome is the
  official stage champion.
- The page (`monitor.html`) is regenerated on every
  snapshot. It renders from the snapshots file alone if
  `monitor_status.json` is missing (status may be None);
  the page then says so and shows run position from the
  snapshots.

## 9. Semantics the hooks must preserve

- Capture the leader BEFORE breeding/culling in
  `generation()`; capturing it after `generation()` returns
  would pair the next generation's bred genomes with the
  previous generation's scores.
- Never snapshot inside `generation()` before the
  checkpoint is saved: the snapshot evaluation (64
  detailed episodes) is the expensive step, and a kill
  during it must leave a valid checkpoint.
- The ladder's `target_invested_pct` is the champion's
  mean invested % from the snapshot that triggered the
  recompute (`snapshot["trades"]["mean_invested_pct"]`) —
  never a hardcoded number, never the training score.
- `build_snapshot` raises for a non-stock domain; the
  hooks gate on `cfg.domain == "stock"` anyway, so a TSP
  run never pays for (or crashes on) monitor work.
- `monitor_enabled = False` disables every hook (status
  rewrites included); `monitor_every_gens` only sets the
  snapshot cadence.
- The archive tracker (`self.tracker`) is an OPTIONAL
  dependency of `_monitor_densities`: if the archive hooks
  have not landed, `getattr(self, "tracker", None)` is
  None and the population density reads as None (the page
  shows "no data yet" for those curves) — the monitor must
  never fail a run for want of the tracker.
