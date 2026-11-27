# Archive hooks: exact code the runner adds

Module: `bert_archive.py` (same directory). The runner imports it;
the module never imports the runner. All hooks are a few lines
calling the module. Nothing here edits `bert_archive.py`.

## 0. Prerequisites the core build must add (parent's job)

`EngineConfig` gains two fields (plan defaults):

```python
archive_cap_bytes_per_stage: int = 16 * 1024 ** 3  # 16 GiB
usefulness_step: float = 0.02
```

`bert_engine.checkpoint_payload` / `load_checkpoint` gain one
array, so a resumed run keeps the parent pointers (without it
the chain trace breaks at a resume boundary: `setup_resume`
below can only default them to -1). In `checkpoint_payload`,
next to `"lineages"`:

```python
"parent_species_ids": np.array(
    [sp.get("parent_species_id", -1) for sp in species]),
```

and in `load_checkpoint`, in the species-dict construction:

```python
"parent_species_id": int(z["parent_species_ids"][k]),
```

## 1. Import

At the top of `run_bert_engine.py`, next to the other imports:

```python
import bert_archive as ba
```

## 2. Tracker + archive state: `setup_fresh` / `setup_resume`

In `setup_fresh`, after `self.species = be.create_species(...)`:

```python
for sp in self.species:
    sp["parent_species_id"] = -1
self.tracker = ba.UsefulnessTracker(
    self.layout.n_params,
    usefulness_step=getattr(cfg, "usefulness_step", 0.02))
self.archive_bytes_used = {}  # stage -> bytes on disk
```

In `setup_resume`, after `self.meta = state["meta"]`:

```python
self.tracker = ba.UsefulnessTracker(
    self.layout.n_params,
    usefulness_step=getattr(cfg, "usefulness_step", 0.02))
if self.meta.get("archive_tracker"):
    self.tracker.load_state_dict(self.meta["archive_tracker"])
self.archive_bytes_used = {
    int(k): int(v)
    for k, v in self.meta.get("archive_bytes_used", {}).items()}
for sp in self.species:
    sp.setdefault("parent_species_id", -1)
```

## 3. Per-generation record + tracker update: `generation()`

Place immediately AFTER the fold-back loop (the loop that sets
`sp["lens"]` and updates `sp["best"]`), BEFORE the elimination
scores are computed. At that point `sp["pop"]` is still the
evaluated population (breeding and culling happen later in the
method), which is the pairing the archive must hold.

```python
# archive hook (Addendum 2): record the evaluated generation
rec = ba.record_species_generation(
    self.out_dir, stage, phase, gen, m["total_gens"],
    self.species, cfg=cfg)
if rec["capped"]:
    m["events"].append({"event": "archive_capped",
                        "stage": stage,
                        "generation": m["total_gens"],
                        "bytes_used_stage":
                            rec["bytes_used_stage"]})
    log(f"[engine] archive CAP reached at stage {stage} "
        f"(used {rec['bytes_used_stage']} bytes); "
        "recording stops for this stage")
self.archive_bytes_used[stage] = rec["bytes_used_stage"]
for sp in self.species:
    self.tracker.update(sp["id"], sp["pop"], sp["lens"])
```

(`record_species_generation` raises if a species has no `lens`;
that is intentional -- it means the hook is in the wrong place.)

In the same method, inside the `if entry:` block after
`be.cull_species(...)` returns, drop the culled species'
tracker vectors (bounded memory):

```python
for did in entry["eliminated_species_ids"]:
    self.tracker.remove(int(did))
```

## 4. Inheritance on regeneration

Whenever species are regenerated from founders, the child
lineage inherits the founder's usefulness vector AND records
its parent species id. The parent id is the FOUNDER's source
species id. Note the two spellings, which differ by call site:

- `do_cataclysm`: the founders vaulted/logged in the EVENT use
  key `"species_id"`; the `founders` list returned by
  `select_founders` uses `"source_species_id"`. Use the event
  form only if you read it back from `m["events"]`; inside
  `do_cataclysm` the `founders` list is in scope, so use
  `"source_species_id"`.
- `growth_attempt` / `_revert_to_champion`: the founders in
  scope are `select_founders` dicts, key `"source_species_id"`.

In `do_cataclysm`, after `self.species = be.regenerate_species(...)`:

```python
founder_by_lineage = {f["lineage"]: f for f in founders}
for sp in self.species:
    f = founder_by_lineage[sp["lineage"]]
    sp["parent_species_id"] = int(f["source_species_id"])
    self.tracker.inherit(sp["id"], int(f["source_species_id"]))
for sid in self.tracker.species_ids():
    if sid not in {sp["id"] for sp in self.species}:
        self.tracker.remove(sid)
```

In `growth_attempt`, after its `self.species =
be.regenerate_species(...)` (the `new_founders` carry the same
lineage / source_species_id as the input founders):

```python
founder_by_lineage = {f["lineage"]: f for f in new_founders}
for sp in self.species:
    f = founder_by_lineage[sp["lineage"]]
    sp["parent_species_id"] = int(f["source_species_id"])
    self.tracker.inherit(sp["id"], int(f["source_species_id"]))
for sid in self.tracker.species_ids():
    if sid not in {sp["id"] for sp in self.species}:
        self.tracker.remove(sid)
```

In `_revert_to_champion`, after its `self.species =
be.regenerate_species(...)`, the same block with `founders`
(the local variable, `select_founders` dicts):

```python
founder_by_lineage = {f["lineage"]: f for f in founders}
for sp in self.species:
    f = founder_by_lineage[sp["lineage"]]
    sp["parent_species_id"] = int(f["source_species_id"])
    self.tracker.inherit(sp["id"], int(f["source_species_id"]))
for sid in self.tracker.species_ids():
    if sid not in {sp["id"] for sp in self.species}:
        self.tracker.remove(sid)
```

No separate hook for the transition cataclysm in `run()`:
it only vaults and logs; the regeneration IS the following
`growth_attempt`, which is hooked above.

## 5. Checkpoint meta: `save_checkpoint()`

At the top of `save_checkpoint`, before the payload is built
(`meta` is JSON-serialised into the checkpoint, and the tracker
state is JSON-able by construction):

```python
self.meta["archive_tracker"] = self.tracker.state_dict()
self.meta["archive_bytes_used"] = {
    str(k): int(v) for k, v in self.archive_bytes_used.items()}
```

## 6. Analysis at finish: `finish()`

 AFTER `be.atomic_json(self.out_dir / "results.json", results)`
(`analyse_run` reads `results.json` for the champion species id
and the final architecture). Best-effort: never fail the run.

```python
try:
    rep = ba.analyse_run(self.out_dir)
    log(f"[engine] archive analysis: winner species "
        f"{rep['winner_species_id']}, chain "
        f"{rep['chain']}, kept {rep['bytes_kept']} bytes, "
        f"deleted {rep['bytes_deleted']} bytes, heatmap "
        f"{rep['heatmap_path']}")
except Exception as exc:  # noqa: BLE001 - best effort
    log(f"[engine] archive analysis failed (non-fatal): {exc}")
```

## 7. Semantics the hooks must preserve

- Record BEFORE culling/breeding each generation (hook 3).
  Recording after `generation()` returns would pair the next
  generation's bred genomes with the previous generation's
  scores and would miss culled species entirely.
- `parent_species_id` is the founder species the species was
  regenerated from; -1 for generation-0 species. The chain
  trace switches from child to parent when stepping back
  across the child's FIRST generation (a child species
  persists for several generations; its parent only existed
  before the child first appeared).
- The tracker and the post-hoc analysis share the single
  implementation `update_log_scores`; do not reimplement the
  signals in the runner.
- The archive cap is a stop-and-log, never a partial record.
  The past-cap fallback is undecided in the plan; do not
  improvise one.
- `analyse_run` deletes the full-population archive records
  after the lineage archive + heatmap are written. Run it
  once, at finish.
