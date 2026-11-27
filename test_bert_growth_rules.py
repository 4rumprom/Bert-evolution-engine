#!/usr/bin/env python3
"""Tests for the sponsor's 2026-11-23 engine rules:
(1) growth stages evolve only the added material
(prob_vector new_only); (2) the freeze rule -- proven
units mutate at rate 0, and a proven unit freezes its
block's layer-norm genes (frozen_mask_from_table);
(3) branch isolation and whole-branch grafting
(bert_branch). Run: python3 test_bert_growth_rules.py
"""
import unittest

import numpy as np

import bert_engine as be
import bert_branch as bb
from bert_backend import get_backend


def small_cfg(**kw):
    cfg = be.EngineConfig(
        domain="tsp", d=2, ffn_hidden=8, initial_blocks=1,
        initial_heads=1, n_species=3, per_species=4,
        n_train=8, n_elim=8, n_val=12,
        domain_config={"n_cities": 6})
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestProbVectorRules(unittest.TestCase):
    def setUp(self):
        self.cfg = small_cfg()
        self.classes = np.array(
            [be.CLS_OLD, be.CLS_OLD, be.CLS_IFACE,
             be.CLS_DECODER, be.CLS_DRAW], dtype=np.int8)

    def test_phase_b_old_rate_baseline(self):
        pv = be.prob_vector(self.classes, "B", self.cfg)
        self.assertAlmostEqual(
            pv[0], self.cfg.mut_p * self.cfg.old_rate_factor)
        self.assertEqual(pv[4], 0.0)

    def test_new_only_zeroes_old_in_both_phases(self):
        for phase in ("A", "B"):
            pv = be.prob_vector(self.classes, phase, self.cfg,
                                new_only=True)
            self.assertEqual(pv[0], 0.0)
            self.assertEqual(pv[1], 0.0)
            self.assertEqual(pv[2], self.cfg.mut_p)
            self.assertEqual(pv[3], self.cfg.mut_p)
            self.assertEqual(pv[4], 0.0)

    def test_new_only_ignores_marks(self):
        marks = np.full(5, 0.5)
        pv = be.prob_vector(self.classes, "B", self.cfg,
                            marks=marks, new_only=True)
        self.assertEqual(pv[0], 0.0)

    def test_flat_nominal_prune_graft_rate(self):
        # sponsor rule 2026-11-24: a prune-and-graft stage
        # mutates every unfrozen gene at the nominal rate
        # throughout; draw genes never; frozen stay 0
        frozen = np.array([True, False, False, False, False])
        for phase in ("A", "B"):
            pv = be.prob_vector(self.classes, phase, self.cfg,
                                flat_nominal=True, frozen=frozen)
            self.assertEqual(pv[0], 0.0)          # frozen
            self.assertEqual(pv[1], self.cfg.mut_p)   # old
            self.assertEqual(pv[2], self.cfg.mut_p)   # iface
            self.assertEqual(pv[3], self.cfg.mut_p)   # decoder
            self.assertEqual(pv[4], 0.0)          # draw

    def test_frozen_zeroes_any_rate(self):
        frozen = np.array([True, False, True, False, False])
        pv = be.prob_vector(self.classes, "B", self.cfg,
                            frozen=frozen)
        self.assertEqual(pv[0], 0.0)
        self.assertGreater(pv[1], 0.0)
        self.assertEqual(pv[2], 0.0)


class TestFrozenMask(unittest.TestCase):
    def setUp(self):
        cfg = small_cfg()
        self.arch = be.initial_arch(cfg, 2)
        # layout needs the domain head spec; build it the
        # way the runner does, through the TSP domain
        import bert_domain_tsp as tsp
        self.domain = tsp.TSPDomain(cfg, get_backend("numpy"))
        self.layout = be.Layout(
            self.arch, tsp.head_spec(self.arch.d))

    def test_proven_unit_and_block_ln_frozen(self):
        units = be.silence_units(self.layout)
        table = {"champion_score": -0.3, "zero_tol": 0.0,
                 "units": []}
        for u in units:
            table["units"].append(
                {**u, "drop": 0.05 if u["kind"] == "ffn"
                 and u["index"] == 0 else 0.0})
        mask = be.frozen_mask_from_table(self.layout, table)
        proven = next(u for u in units
                      if u["kind"] == "ffn" and u["index"] == 0)
        idx = be.unit_gene_indices(self.layout, proven)
        self.assertTrue(mask[idx].all())
        for nm in ("c0.b0.lnA_gain", "c0.b0.lnA_bias",
                   "c0.b0.lnF_gain", "c0.b0.lnF_bias"):
            self.assertTrue(mask[self.layout.sl(nm)].all(),
                            nm)
        other = next(u for u in units
                     if u["kind"] == "ffn" and u["index"] == 1)
        oidx = be.unit_gene_indices(self.layout, other)
        self.assertFalse(mask[oidx].any())

    def test_no_proven_no_freeze(self):
        units = be.silence_units(self.layout)
        table = {"champion_score": -0.3, "zero_tol": 0.0,
                 "units": [{**u, "drop": 0.0} for u in units]}
        mask = be.frozen_mask_from_table(self.layout, table)
        self.assertFalse(mask.any())

    def test_remap_frozen_keeps_survivors(self):
        import bert_domain_tsp as tsp
        cfg = small_cfg()
        units = be.silence_units(self.layout)
        table = {"champion_score": -0.3, "zero_tol": 0.0,
                 "units": [{**u, "drop": 0.05} for u in units]}
        mask = be.frozen_mask_from_table(self.layout, table)
        from bert_engine import ArchSpec
        spec2 = ArchSpec(d=self.arch.d + cfg.widen_step,
                         ffn_hidden=self.arch.ffn_hidden
                         + cfg.widen_step,
                         couples=self.arch.couples,
                         input_dim=self.arch.input_dim)
        layout2 = be.Layout(spec2,
                            tsp.head_spec(spec2.d))
        out = be.remap_frozen(mask, self.layout, layout2)
        self.assertEqual(len(out), layout2.n_params)
        # surviving prefix stays frozen; the widened
        # extension is new material and starts unfrozen
        sl2 = layout2.sl("c0.b0.lnA_gain")
        self.assertTrue(out[sl2][:self.arch.d].all())
        self.assertFalse(out[sl2][self.arch.d:].any())


class TestBranches(unittest.TestCase):
    def setUp(self):
        cfg = small_cfg(n_val=20)
        self.cfg = cfg
        import bert_domain_tsp as tsp
        self.domain = tsp.TSPDomain(cfg, get_backend("numpy"))
        self.arch = be.initial_arch(cfg, 2)
        self.layout = be.Layout(
            self.arch, tsp.head_spec(self.arch.d))
        rng = np.random.default_rng(3)
        self.genome = rng.normal(
            0, 0.4, self.layout.n_params)

    def test_branches_partition_units(self):
        # signal="silencing": this suite runs under the
        # system python (no jax); it tests the partition
        # and proof mechanics, not the isolation signal
        # (the attribution path is verified in
        # test_bert_attribution.py under the venv python)
        branches = bb.find_branches(self.domain, self.layout,
                                    self.genome,
                                    signal="silencing")
        names = [u["name"] for b in branches
                 for u in b["units"]]
        all_names = [u["name"]
                     for u in be.silence_units(self.layout)]
        self.assertEqual(sorted(names), sorted(all_names))
        for b in branches:
            self.assertIsNotNone(b["group_drop_rel_pct"])

    def test_cleared_slots_order(self):
        rows = [{"name": "a", "kind": "ffn", "drop": 0.2},
                {"name": "b", "kind": "ffn", "drop": -0.1},
                {"name": "c", "kind": "ffn", "drop": 0.0},
                {"name": "d", "kind": "ffn", "drop": -0.3}]
        slots = bb.cleared_slots(rows)
        self.assertEqual([s["name"] for s in slots],
                         ["d", "b", "c"])

    def test_graft_copies_whole_units_only(self):
        branches = bb.find_branches(self.domain, self.layout,
                                    self.genome,
                                    signal="silencing")
        donor = np.random.default_rng(5).normal(
            0, 0.4, self.layout.n_params)
        table = be.silence_table(
            self.layout, self.genome,
            lambda G: self.domain.validation_scores(
                self.layout, G))
        slots = bb.cleared_slots(table["units"])
        if not slots:
            self.skipTest("no cleared slots on random genome")
        out, pairs, _ = bb.graft_branch(
            self.layout, self.genome, donor, branches[0],
            slots)
        by_name = {u["name"]: u
                   for u in be.silence_units(self.layout)}
        touched = set()
        for dn, sn in pairs:
            di = be.unit_gene_indices(self.layout,
                                      by_name[dn])
            si = be.unit_gene_indices(self.layout,
                                      by_name[sn])
            self.assertTrue(np.array_equal(out[si],
                                           donor[di]))
            touched.update(si.tolist())
        rest = np.ones(self.layout.n_params, dtype=bool)
        rest[list(touched)] = False
        self.assertTrue(np.array_equal(out[rest],
                                       self.genome[rest]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
