"""Candidate preparation tests; no GPU, forward pass, or training launch."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import train_s1_bounded_candidate as candidate


def metrics(first=1., chunk=1., yaw=1., torso=1.):
    return dict(all=dict(first_action=dict(rmse_active11=first), valid_chunk=dict(rmse_active11=chunk)),
                reach=dict(first_action=dict(yaw_rmse=yaw, torso_rmse=torso)))


def gate_fixture():
    names = ('first_action_active11_rmse', 'valid_chunk_active11_rmse',
             'reach_yaw_first_rmse', 'reach_torso_first_rmse')
    decision = dict(schema='bvi.s1-bounded-diagnostic-decision/1', status='verified_metric_go', candidate_eligible=True,
                    metric_gate=dict(passed=True, checks={name: {'passed': True} for name in names}),
                    checkpoint_verification=dict(status='passed_cpu_restore', forward_calls=0, gpu_used=False,
                                                 parameters_sha256='a' * 64))
    status = dict(status='completed_diagnostic_requires_review', source_parameters_sha256=candidate.SOURCE_HASH)
    supervisor = dict(status='exited', worker_exit_code=0)
    return decision, status, supervisor


class SamplingTests(unittest.TestCase):
    def test_exact_thresholds_current_actions_head_excluded(self):
        rows = [dict(split='train', parent_id=0, tool_family=f) for f in
                ('reach', 'reach', 'reach', 'move', 'move', 'grasp')]
        actions = np.zeros((6, 13), dtype=np.float64)
        actions[0, 12] = .1; actions[1, 10] = -.1; actions[2, 12] = .099
        actions[3, 7] = -.1; actions[4, 8:10] = 1; actions[5, 10] = 1
        np.testing.assert_array_equal(candidate.sample_weights(rows, actions), [3, 3, 1, 2, 1, 1])

    def test_float32_point_one_is_included(self):
        rows = [dict(split='train', parent_id=0, tool_family='reach')]
        actions = np.zeros((1, 13), dtype=np.float32); actions[0, 12] = .1
        self.assertEqual(candidate.sample_weights(rows, actions).tolist(), [3])

    def test_draws_are_frozen_replacement_4000(self):
        rows = [dict(split='train', parent_id=i, tool_family='reach') for i in range(10)]
        actions = np.zeros((10, 13)); actions[:3, 10] = 1
        w, first = candidate.fixed_sampling(rows, actions)
        _, second = candidate.fixed_sampling(rows, actions)
        expected = np.random.default_rng(19092026).choice(10, size=4000, replace=True, p=w / w.sum())
        np.testing.assert_array_equal(first, second); np.testing.assert_array_equal(first, expected)
        self.assertEqual(len(first), 4000)

    def test_rejects_validation_or_reserved_parent(self):
        for split, parent in [('validation', 3), ('train', 20), ('train', 21)]:
            with self.subTest(split=split, parent=parent), self.assertRaises(ValueError):
                candidate.sample_weights([dict(split=split, parent_id=parent, tool_family='reach')], np.zeros((1, 13)))

    def test_rejects_nonfinite_actions(self):
        rows = [dict(split='train', parent_id=0, tool_family='move')]
        a = np.zeros((1, 13)); a[0, 0] = np.nan
        with self.assertRaises(ValueError): candidate.sample_weights(rows, a)


class DevSelectionTests(unittest.TestCase):
    def fixture(self):
        return [dict(parent_id=20 + i // 4, call_index=i, observation_index=i,
                     split='validation', tool_family=f, action_valid=[True], instruction=f, source_sha256='a' * 64)
                for i, f in enumerate(['reach'] * 4 + ['grasp'] * 2 + ['move'] * 2)]

    def test_selects_only_frozen_validation_eight(self):
        dev = self.fixture()
        train = dict(dev[0], split='train', parent_id=0)
        self.assertEqual(candidate.select_dev([train] + dev, dev + [train]), dev)

    def test_rejects_changed_prompt_mask_or_split(self):
        for field, value in [('instruction', 'wrong'), ('action_valid', [False]), ('split', 'train')]:
            source = self.fixture(); panel = copy.deepcopy(source); panel[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): candidate.select_dev(panel, source)

    def test_rejects_duplicate_or_wrong_family_counts(self):
        for mode in ('duplicate', 'family'):
            source = self.fixture(); panel = copy.deepcopy(source)
            if mode == 'duplicate': panel[0] = panel[1]
            else: panel[0]['tool_family'] = 'move'
            with self.subTest(mode=mode), self.assertRaises(ValueError): candidate.select_dev(panel, source)

    def test_formula_and_guards(self):
        result = candidate.score_dev(metrics(.6, .8, 1.1, 1.1), metrics())
        self.assertAlmostEqual(result['score'], .7); self.assertTrue(result['eligible'])
        self.assertFalse(candidate.score_dev(metrics(.6, .8, 1.10001, 1), metrics())['eligible'])

    def test_equal_baseline_not_eligible_and_zero_guard_strict(self):
        self.assertFalse(candidate.score_dev(metrics(), metrics())['eligible'])
        self.assertTrue(candidate.score_dev(metrics(.8, .8, 0, 0), metrics(yaw=0, torso=0))['eligible'])
        self.assertFalse(candidate.score_dev(metrics(.8, .8, .001, 0), metrics(yaw=0, torso=0))['eligible'])

    def test_zero_or_nonfinite_denominators_rejected(self):
        for bad in (0., float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate.score_dev(metrics(), metrics(first=bad))

    def test_tie_prefers_earlier_never_ineligible(self):
        old = dict(score=.7, step=100, eligible=True)
        self.assertFalse(candidate.prefer(dict(score=.7, step=250, eligible=True), old))
        self.assertTrue(candidate.prefer(dict(score=.6, step=250, eligible=True), old))
        self.assertFalse(candidate.prefer(dict(score=.1, step=250, eligible=False), old))


class LaunchGateTests(unittest.TestCase):
    def test_verified_decision_accepted(self):
        candidate.verify_launch_gate(*gate_fixture())

    def test_boolean_alone_is_not_authorization(self):
        args = gate_fixture(); args[0]['status'] = 'pending'
        with self.assertRaises(ValueError): candidate.verify_launch_gate(*args)

    def test_no_go_or_metric_failure_rejected(self):
        for mode in ('eligibility', 'metric'):
            args = gate_fixture()
            if mode == 'eligibility': args[0]['candidate_eligible'] = False
            else: args[0]['metric_gate']['passed'] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError): candidate.verify_launch_gate(*args)

    def test_cpu_unit_test_is_not_restore_proof(self):
        args = gate_fixture(); args[0]['checkpoint_verification']['status'] = 'unit_tests_passed'
        with self.assertRaises(ValueError): candidate.verify_launch_gate(*args)

    def test_rejects_diagnostic_source_or_timeout(self):
        for mode in ('source', 'timeout'):
            args = gate_fixture()
            if mode == 'source': args[1]['source_parameters_sha256'] = '0' * 64
            else: args[2]['status'] = 'hard_timeout'
            with self.subTest(mode=mode), self.assertRaises(ValueError): candidate.verify_launch_gate(*args)

    def test_evidence_hash_changes_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); decision, status, supervisor = gate_fixture()
            for name, document in [('status.json', status), ('supervisor.json', supervisor),
                                   ('identity.json', {}), ('selected-rows.json', []),
                                   ('before.npz', {}), ('after.npz', {})]:
                (root / name).write_text(json.dumps(document))
            contract = root / 'contract.json'; contract.write_text('{}')
            decision['execution_contract_sha256'] = candidate.base.file_hash(contract)
            decision['evidence_sha256'] = {n: candidate.base.file_hash(root / n) for n in
                ('status.json', 'supervisor.json', 'identity.json', 'selected-rows.json', 'before.npz', 'after.npz')}
            path = root / 'decision.json'; path.write_text(json.dumps(decision))
            args = SimpleNamespace(decision=path, diagnostic_run=root, execution_contract=contract)
            candidate.read_gates(args)
            (root / 'before.npz').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'evidence changed'): candidate.read_gates(args)


class PublishedBestTests(unittest.TestCase):
    def prepare(self, root):
        baseline, selected = metrics(), metrics(.6, .8, 1, 1)
        for name, value in [('dev-step000.json', baseline), ('dev-step100.json', selected)]:
            (root / name).write_text(json.dumps(value))
        for name in ('dev-step000.npz', 'dev-step100.npz'):
            np.savez(root / name, predicted=np.zeros((8, 10, 13)))
        checkpoint = root / 'best' / '100'
        (checkpoint / 'params').mkdir(parents=True)
        (checkpoint / 'train_state').mkdir()
        result = dict(step=100, **candidate.score_dev(selected, baseline))
        return checkpoint, result

    def test_exact_panel_schema_and_all_raw_hashes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, result = self.prepare(root)
            record = candidate.best_record(root, checkpoint, result, 'a' * 64)
            self.assertEqual(record['selection'], 'dev8_only_no_online_selection')
            self.assertFalse(record['diagnostic_only'])
            self.assertFalse(record['fresh_checkpoint_restore_verified'])
            self.assertTrue(record['eligible']); self.assertTrue(record['checkpoint_complete'])
            self.assertEqual(record['checkpoint_parameters_sha256'], 'a' * 64)
            self.assertEqual(record['source_checkpoint_parameters_sha256'], candidate.SOURCE_HASH)
            evidence = record['dev_selection']
            self.assertEqual(evidence['split'], 'development'); self.assertFalse(evidence['used_online_success'])
            for file_key, hash_key in [('evidence_file', 'sha256'), ('raw_evidence_file', 'raw_sha256'),
                    ('baseline_evidence_file', 'baseline_sha256'), ('baseline_raw_evidence_file', 'baseline_raw_sha256')]:
                self.assertEqual(evidence[hash_key], candidate.base.file_hash(root / evidence[file_key]))

    def test_score_cannot_disagree_with_actual_saved_development_metrics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, result = self.prepare(root)
            (root / 'dev-step100.json').write_text(json.dumps(metrics(.9, .9, 1, 1)))
            with self.assertRaisesRegex(ValueError, 'differs from actual saved metrics'):
                candidate.best_record(root, checkpoint, result, 'a' * 64)

    def test_missing_raw_artifact_cannot_be_published(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, result = self.prepare(root)
            (root / 'dev-step000.npz').unlink()
            with self.assertRaises(FileNotFoundError): candidate.best_record(root, checkpoint, result, 'a' * 64)

    def test_missing_train_state_cannot_be_marked_complete(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint, result = self.prepare(root)
            (checkpoint / 'train_state').rmdir()
            with self.assertRaisesRegex(ValueError, 'incomplete checkpoint'):
                candidate.best_record(root, checkpoint, result, 'a' * 64)


if __name__ == '__main__':
    unittest.main()
