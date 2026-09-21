"""CPU-only evidence and frozen-parameter verification tests; no model loads."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import verify_s1_bounded_candidate as verify


def dev_fixture():
    families = np.asarray(['reach'] * 4 + ['grasp'] * 2 + ['move'] * 2)
    rows = [dict(parent_id=20 + i // 4, call_index=i, observation_index=i,
                 split='validation', tool_family=family, action_valid=[True] * 10)
            for i, family in enumerate(families)]
    target = np.zeros((8, 10, 13), dtype=np.float32)
    prediction = np.full_like(target, .5); prediction[:, :, 8:10] = 0
    return rows, dict(predicted=prediction.copy(), predicted_unclipped=prediction.copy(),
        target=target, valid=np.ones((8, 10), dtype=bool), families=families,
        row_keys=verify.row_keys(rows))


def bounds_fixture(updates=250):
    points = [s for s in (100, 250, 500) if s <= updates]
    if updates and updates not in points: points.append(updates)
    status = dict(schema=verify.SCHEMA, additional_updates=updates, checkpoint_selection_complete=True,
                  source_checkpoint_step=855, source_checkpoint_parameters_sha256=verify.SOURCE_HASH,
                  diagnostic_only=False, candidate_protocol=verify.PROTOCOL, optimizer='reset_AdamW',
                  learning_rate=1e-4, progress=False, family_bank=False, api_calls=0,
                  evaluated_steps=points, stop_reason='update_limit' if updates == 500 else 'finalization_reserve')
    supervisor = dict(schema=verify.SCHEMA, elapsed_seconds=8800, budget_seconds=9000)
    identity = dict(schema=verify.SCHEMA, max_updates=500, max_seconds=9000, finalize_reserve=900, accumulation=8,
                    source_checkpoint_parameters_sha256=verify.SOURCE_HASH, trainer_sha256=verify.TRAINER_HASH,
                    helper_sha256=verify.HELPER_HASH, checkpoints=[100, 250, 500, 'budget_stop'],
                    rng_seed=19092026, denoising_steps=10,
                    selection='dev8_relative_rmse_mean_with_reach_yaw_torso_guard_tie_earlier')
    return status, supervisor, identity


class BudgetTests(unittest.TestCase):
    def test_accepts_fixed_and_final_budget_points(self):
        for count, expected in [(0, []), (75, [75]), (100, [100]), (177, [100, 177]),
                                (250, [100, 250]), (500, [100, 250, 500])]:
            with self.subTest(count=count): self.assertEqual(verify.validate_bounds(*bounds_fixture(count)), expected)

    def test_rejects_overbudget_time_or_updates(self):
        for mode in ('time', 'updates', 'nan'):
            args = bounds_fixture()
            if mode == 'updates': args[0]['additional_updates'] = 501
            else: args[1]['elapsed_seconds'] = 9001 if mode == 'time' else float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError): verify.validate_bounds(*args)

    def test_rejects_added_or_skipped_selection_points(self):
        for points in ([100], [100, 200, 250]):
            args = bounds_fixture(); args[0]['evaluated_steps'] = points
            with self.subTest(points=points), self.assertRaises(ValueError): verify.validate_bounds(*args)

    def test_rejects_different_source_or_optimizer_contract(self):
        for field, value in [('source_checkpoint_parameters_sha256', '0' * 64),
                             ('optimizer', 'restored'), ('progress', True), ('diagnostic_only', True)]:
            args = bounds_fixture(); args[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): verify.validate_bounds(*args)


class SamplingTests(unittest.TestCase):
    def fixture(self):
        from collections import Counter
        rows = [dict(parent_id=i, split='train', tool_family=f) for i, f in enumerate(('reach', 'move', 'grasp'))]
        actions = np.zeros((3, 13)); actions[0, 12] = .1; actions[1, 10] = -.1
        weights = np.asarray([3., 2., 1.])
        indices = np.random.default_rng(19092026).choice(3, 4000, replace=True, p=weights / 6)
        archive = dict(current_actions=actions.copy(), weights=weights, indices=indices)
        summary = dict(seed=19092026, draws=4000, replacement=True,
                       rules={'reach_yaw_or_torso_abs_ge_0.1': 3, 'move_any_active11_abs_ge_0.1': 2, 'otherwise': 1},
                       eligible_parent_counts=dict(Counter(str(r['parent_id']) for r in rows)),
                       sampled_parent_counts=dict(Counter(str(rows[i]['parent_id']) for i in indices)),
                       sampled_family_counts=dict(Counter(rows[i]['tool_family'] for i in indices)))
        return rows, archive, summary, actions

    def test_exact_replay(self):
        self.assertEqual(verify.verify_sampling(*self.fixture())['draws'], 4000)

    def test_rejects_forged_actions_weights_draws_or_distribution(self):
        for mode in ('actions', 'weights', 'draws', 'distribution'):
            args = self.fixture()
            if mode == 'actions': args[1]['current_actions'][0, 0] = .3
            elif mode == 'weights': args[1]['weights'][0] = 2
            elif mode == 'draws': args[1]['indices'][0] = (args[1]['indices'][0] + 1) % 3
            else: args[2]['sampled_parent_counts'] = {}
            with self.subTest(mode=mode), self.assertRaises(ValueError): verify.verify_sampling(*args)


class DevArrayTests(unittest.TestCase):
    def test_reads_eight_not_diagnostic_85(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            rows, data = dev_fixture(); path = Path(directory) / 'dev.npz'
            np.savez(path, **data)
            result = verify.read_dev_arrays(path, rows)
            self.assertEqual(result['predicted'].shape, (8, 10, 13))
            self.assertAlmostEqual(verify.dev_metrics(result)['first'], .5)

    def test_rejects_wrong_mask_row_or_applied_actions(self):
        import tempfile
        for mode in ('mask', 'row', 'actions'):
            with tempfile.TemporaryDirectory() as directory:
                rows, data = dev_fixture(); path = Path(directory) / 'dev.npz'
                if mode == 'mask': data['valid'][0, 0] = False
                elif mode == 'row': data['row_keys'][0, 0] = 0
                else: data['predicted'][0, 0, 0] = .7
                np.savez(path, **data)
                with self.subTest(mode=mode), self.assertRaises(ValueError): verify.read_dev_arrays(path, rows)

    def test_recomputes_score_guard_and_refuses_forged_decision(self):
        baseline = dict(first=1., chunk=1., yaw=.2, torso=.1)
        current = dict(first=.7, chunk=.9, yaw=.2, torso=.1)
        score = verify.score_metrics(current, baseline)
        self.assertAlmostEqual(score['score'], .8); self.assertTrue(score['eligible'])
        verify.check_selection(score, score)
        with self.assertRaises(ValueError): verify.check_selection(dict(score, score=.1), score)
        with self.assertRaises(ValueError): verify.check_selection(dict(score, eligible=False), score)
        self.assertFalse(verify.score_metrics(dict(current, yaw=.3), baseline)['eligible'])

    def test_zero_baseline_not_relative_improvement(self):
        with self.assertRaises(ValueError):
            verify.score_metrics(dict(first=0, chunk=0, yaw=0, torso=0), dict(first=0, chunk=0, yaw=0, torso=0))


class FrozenLeafTests(unittest.TestCase):
    def fixture(self):
        leaves = {('llm', f'lora_{i}'): np.asarray([i + 1.], dtype=np.float32) for i in range(20)}
        leaves[('action_projection',)] = np.asarray([.12345679], dtype=np.float32)
        return leaves

    def test_only_lora_bytes_may_change(self):
        source = self.fixture(); selected = {k: v.copy() for k, v in source.items()}
        selected[('llm', 'lora_0')][0] += .5
        result = verify.compare_frozen_leaves(verify.summarize_leaves(source), verify.summarize_leaves(selected))
        self.assertEqual(result['changed_lora_leaves'], 1)
        self.assertEqual(result['non_lora_leaves'], 1)

    def test_rejects_even_tiny_non_lora_byte_change(self):
        source = self.fixture(); selected = {k: v.copy() for k, v in source.items()}
        selected[('action_projection',)][0] = np.nextafter(selected[('action_projection',)][0], np.float32(1))
        with self.assertRaisesRegex(ValueError, 'non-LoRA parameter bytes'):
            verify.compare_frozen_leaves(verify.summarize_leaves(source), verify.summarize_leaves(selected))

    def test_rejects_shape_dtype_or_missing_lora(self):
        original = verify.summarize_leaves(self.fixture())
        for mode in ('shape', 'dtype', 'missing'):
            changed = copy.deepcopy(original)
            if mode == 'missing': del changed[('llm', 'lora_0')]
            else: changed[('llm', 'lora_0')][mode] = [2] if mode == 'shape' else 'bfloat16'
            with self.subTest(mode=mode), self.assertRaises(ValueError): verify.compare_frozen_leaves(original, changed)

    def test_invalid_float_values_fail_before_comparison(self):
        source = self.fixture(); source[('action_projection',)][0] = np.nan
        with self.assertRaises(ValueError): verify.summarize_leaves(source)


class IncompleteRunTests(unittest.TestCase):
    def test_timeout_is_infrastructure_not_policy_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status.json').write_text(json.dumps(dict(status='training')))
            (root / 'supervisor.json').write_text(json.dumps(dict(status='hard_timeout', worker_exit_code=-9)))
            result = verify.verify_run(SimpleNamespace(run_dir=root))
            self.assertEqual(result['status'], 'run_not_evaluable')
            self.assertFalse(result['policy_failure']); self.assertFalse(result['candidate_ready_for_oracle'])


class SyntheticCompletedRunTests(unittest.TestCase):
    def prepare(self, root, improves):
        import train_s1_bounded_diagnostic as base
        import train_s1_bounded_candidate as candidate
        run = root / 'run'; run.mkdir()
        original = root / 'source' / 'best' / '855'
        original_assets = original / 'assets' / verify.REPO_ID
        original_assets.mkdir(parents=True)
        (original / 'params').mkdir()
        (original_assets / 'norm_stats.json').write_text('frozen norm bytes')
        provenance = dict(source_sha256='a' * 64, source_manifest_sha256='b' * 64,
                          normalizer_sha256=verify.sha256(original_assets / 'norm_stats.json'))
        source_contract = dict(robot='fetch', state_dim=24, state_components=['native_qpos12', 'native_qvel12'],
            action_dim=13, base_position_reference='world', base_camera='fetch_head', wrist_camera='fetch_hand',
            state_conditioning=True, training_repo=verify.REPO_ID,
            action_convention='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity',
            state_source='env_native_agent', normalizer_sha256=provenance['normalizer_sha256'],
            training_stage='S1_IA_single_bank_no_progress')
        (original_assets / 'bvi-state-contract.json').write_text(json.dumps(source_contract))
        status, supervisor, identity = bounds_fixture(1)
        metadata = dict(source_contract, diagnostic_only=False, candidate_protocol=verify.PROTOCOL,
                        optimizer_reset=True, source_checkpoint_step=855, lr_clock='reset_constant_1e-4')
        dev, baseline = dev_fixture()
        for row in dev: row['trajectory'] = 'traj_' + str(row['parent_id'])
        train = [dict(parent_id=i, call_index=0, observation_index=0, trajectory='traj_' + str(i),
                      split='train', tool_family=f, action_valid=[True] * 10)
                 for i, f in enumerate(('reach', 'grasp', 'move'))]
        source = root / 'source.h5'
        source.write_bytes(b'synthetic source mocked at H5 boundary')
        audit = root / 'audit-panel.json'; audit.write_text(json.dumps(dev))
        actions = np.zeros((3, 13), dtype=np.float32)
        weights, indices = candidate.fixed_sampling(train, actions)
        np.savez(run / 'sampling.npz', weights=weights, indices=indices, current_actions=actions)
        from collections import Counter
        summary = dict(seed=19092026, draws=4000, replacement=True,
                       rules={'reach_yaw_or_torso_abs_ge_0.1': 3, 'move_any_active11_abs_ge_0.1': 2, 'otherwise': 1},
                       eligible_parent_counts=dict(Counter(str(r['parent_id']) for r in train)),
                       sampled_parent_counts=dict(Counter(str(train[i]['parent_id']) for i in indices)),
                       sampled_family_counts=dict(Counter(train[i]['tool_family'] for i in indices)))
        identity.update(**provenance, checkpoint=str(original), policy_metadata=metadata,
                        audit_panel_sha256=verify.sha256(audit), sampling_sha256=verify.sha256(run / 'sampling.npz'))
        selected = {k: v.copy() for k, v in baseline.items()}
        selected['predicted'] *= .8 if improves else 1.2
        selected['predicted_unclipped'] = selected['predicted'].copy()
        for step, data in ((0, baseline), (1, selected)):
            np.savez(run / f'dev-step{step:03d}.npz', **data)
            result = base.reconstruction_metrics(data['predicted'], data['target'], data['valid'], data['families'])
            (run / f'dev-step{step:03d}.json').write_text(json.dumps(result))
        record = dict(step=1, **verify.score_metrics(verify.dev_metrics(selected), verify.dev_metrics(baseline)))
        (run / 'selection.jsonl').write_text(json.dumps(record) + '\n')
        (run / 'train.jsonl').write_text(json.dumps(dict(step=1, lr=1e-4, loss=.1, grad_norm=.2)) + '\n')
        best = dict(schema=verify.SCHEMA, eligible=False, checkpoint=None)
        if improves:
            checkpoint = run / 'best' / '1'; assets = checkpoint / 'assets' / verify.REPO_ID
            assets.mkdir(parents=True); (checkpoint / 'params').mkdir(); (checkpoint / 'train_state').mkdir()
            (assets / 'norm_stats.json').write_bytes((original_assets / 'norm_stats.json').read_bytes())
            (assets / 'bvi-state-contract.json').write_text(json.dumps(metadata))
            # Producer uses its own reported metrics, independently checked later.
            producer_score = candidate.score_dev(verify.read_json(run / 'dev-step001.json'),
                                                verify.read_json(run / 'dev-step000.json'))
            best = candidate.best_record(run, checkpoint, dict(step=1, **producer_score), 'c' * 64)
        status.update(status='completed_candidate' if improves else 'completed_no_eligible_candidate',
                      best=best if improves else None)
        supervisor.update(status='exited', worker_exit_code=0)
        for name, document in [('status.json', status), ('supervisor.json', supervisor), ('identity.json', identity),
                               ('train-roster.json', train), ('dev-roster.json', dev),
                               ('sampling-summary.json', summary), ('best.json', best)]:
            (run / name).write_text(json.dumps(document))
        args = SimpleNamespace(run_dir=run, source_checkpoint=original, source=source,
                               dataset_manifest=root / 'source-manifest.json', normalizer=original_assets,
                               audit_panel=audit, verify_checkpoint=False)
        return args, provenance, train + dev

    def run_fixture(self, args, provenance, rows):
        class FakeH5(dict):
            def __enter__(self): return self
            def __exit__(self, *exc): pass
        handle = FakeH5({f'traj_{p}': {'actions': np.zeros((20, 13), dtype=np.float32)}
                         for p in (0, 1, 2, 20, 21)})
        with patch('audit_s1_teacher_actions.audit_provenance', return_value=({}, provenance)), \
             patch('bvi.ia_fetch_data.invocation_rows', return_value=rows), \
             patch.dict(sys.modules, {'h5py': SimpleNamespace(File=lambda *a, **kw: handle)}):
            return verify.verify_run(args)

    def test_completed_without_eligible_candidate_is_scoped_no_go(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_fixture(*self.prepare(Path(directory), improves=False))
            self.assertEqual(result['status'], 'verified_no_eligible_candidate', result)
            self.assertFalse(result['policy_failure']); self.assertFalse(result['candidate_ready_for_oracle'])

    def test_valid_selected_producer_artifacts_require_fresh_cpu_restore(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_fixture(*self.prepare(Path(directory), improves=True))
            self.assertEqual(result['status'], 'development_selection_passed_restore_pending', result)
            self.assertFalse(result['candidate_ready_for_oracle'])

    def test_cpu_restore_failure_never_releases_oracle(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            args, provenance, rows = self.prepare(Path(directory), improves=True)
            args.verify_checkpoint = True
            with patch.object(verify, 'verify_checkpoint', side_effect=ValueError('Frozen non-LoRA bytes changed')):
                result = self.run_fixture(args, provenance, rows)
            self.assertEqual(result['status'], 'evidence_invalid', result)
            self.assertFalse(result['policy_failure']); self.assertFalse(result['candidate_ready_for_oracle'])


if __name__ == '__main__':
    unittest.main()
