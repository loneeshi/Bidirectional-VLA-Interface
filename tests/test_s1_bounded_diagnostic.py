"""CPU-only safety and metric checks; no OpenPI, GPU, SSH or downloads."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'bounded_s1', Path(__file__).parents[1] / 'scripts/train_s1_bounded_diagnostic.py')
bounded = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bounded
spec.loader.exec_module(bounded)


def fixture():
    provenance = {k: c * 64 for k, c in [('source_sha256', 'a'),
                    ('source_manifest_sha256', 'b'), ('normalizer_sha256', 'c')]}
    manifest = dict(parent_split=dict(train=[0, 1], validation=[2]), episodes=[
        dict(parent_id=i, split='train' if i < 2 else 'validation', native_success=True)
        for i in range(3)])
    rows = [dict(parent_id=i // 32, call_index=i % 3, observation_index=i,
                 split='train', action_valid=[True, False], tool_family=('reach', 'grasp', 'move')[i % 3])
            for i in range(64)]
    panel = dict(schema_version=1, checkpoint_step=855, **provenance,
                 rows=[{k: r[k] for k in ('parent_id', 'call_index', 'observation_index')} for r in rows])
    return panel, manifest, rows, provenance


class PanelTests(unittest.TestCase):
    def test_preserves_frozen_order(self):
        args = fixture()
        args[0]['rows'].reverse()
        selected = bounded.select_panel(*args)
        self.assertEqual([r['observation_index'] for r in selected], list(reversed(range(64))))

    def test_rejects_all_provenance_mismatches(self):
        for field in fixture()[3]:
            args = fixture()
            args[0][field] = '0' * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'provenance'):
                bounded.select_panel(*args)

    def test_rejects_failed_parent(self):
        args = fixture(); args[1]['episodes'][0]['native_success'] = False
        with self.assertRaisesRegex(ValueError, 'successful train'):
            bounded.select_panel(*args)

    def test_rejects_validation_parent(self):
        args = fixture(); args[1]['parent_split']['train'].remove(1)
        with self.assertRaisesRegex(ValueError, 'successful train'):
            bounded.select_panel(*args)

    def test_rejects_validation_row(self):
        args = fixture(); args[2][0]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'valid train'):
            bounded.select_panel(*args)

    def test_rejects_third_parent(self):
        args = fixture()
        args[0]['rows'][0]['parent_id'] = args[2][0]['parent_id'] = 2
        with self.assertRaisesRegex(ValueError, 'two train parents'):
            bounded.select_panel(*args)

    def test_rejects_duplicate_and_unknown_row(self):
        for mode in ('duplicate', 'unknown'):
            args = fixture()
            if mode == 'duplicate':
                args[0]['rows'][0] = copy.deepcopy(args[0]['rows'][1])
            else:
                args[0]['rows'][0]['observation_index'] = 10000
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                bounded.select_panel(*args)

    def test_rejects_wrong_count(self):
        for count in (0, 63, 129):
            args = fixture(); args[0]['rows'] = (args[0]['rows'] * 3)[:count]
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, '64-128'):
                bounded.select_panel(*args)

    def test_rejects_invalid_actions_or_phase(self):
        for mode in ('invalid', 'missing', 'release'):
            args = fixture()
            if mode == 'invalid':
                args[2][0]['action_valid'][0] = False
            elif mode == 'missing':
                for row in args[2]:
                    row['tool_family'] = 'reach'
            else:
                args[2][0]['tool_family'] = 'release'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                bounded.select_panel(*args)

    def test_rejects_coerced_row_indices(self):
        for bad in ('0', 0.0, True, -1):
            args = fixture(); args[0]['rows'][0]['parent_id'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bounded.select_panel(*args)


class BudgetAndMetricsTests(unittest.TestCase):
    def test_deadline_includes_startup_and_reserve(self):
        # A deadline anchored before loading cannot reset when training starts.
        self.assertTrue(bounded.budget_remaining(3600, 900, clock=lambda: 2699))
        self.assertFalse(bounded.budget_remaining(3600, 900, clock=lambda: 2700))
        self.assertFalse(bounded.budget_remaining(3600, clock=lambda: 3600))

    def test_constant_lr_does_not_inherit_855_clock(self):
        schedule = bounded.ConstantSchedule().create()
        self.assertEqual([schedule(k) for k in (0, 1, 99, 855)], [1e-4] * 4)

    def test_cli_rejects_candidate_budget(self):
        common = [value for name in ('source', 'dataset-manifest', 'normalizer', 'checkpoint', 'panel', 'output')
                  for value in ('--' + name, 'unused')]
        for extra in (['--max-updates', '101'], ['--seconds', '3601'], ['--finalize-reserve', '3600']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                bounded.parse_args(common + extra)

    def test_masks_padding_and_head_and_reports_stage_channels(self):
        target = np.zeros((3, 2, 13)); predicted = target.copy()
        predicted[:, :, 8:10] = 999  # inactive head must not affect active11
        predicted[:, 1] = 999  # invalid tail must not affect either scope
        predicted[0, 0, 10] = .2
        predicted[1, 0, 12] = .3
        metrics = bounded.reconstruction_metrics(predicted, target, [[True, False]] * 3,
                                                  ['reach', 'grasp', 'move'])
        self.assertAlmostEqual(metrics['all']['first_action']['mae_active11'], .5 / 33)
        self.assertEqual(metrics['reach']['valid_chunk']['torso_mae'], .2)
        self.assertEqual(metrics['grasp']['first_action']['yaw_mae'], .3)
        self.assertEqual(metrics['all']['valid_chunk']['moving_sign_denominator'], 0)

    def test_first_action_and_chunk_remain_distinct(self):
        target = np.zeros((1, 2, 13)); predicted = target.copy()
        target[0, 1, 12] = .1; predicted[0, 1, 12] = -.2
        metrics = bounded.reconstruction_metrics(predicted, target, [[True, True]], ['reach'])['all']
        self.assertEqual(metrics['first_action']['mae_active11'], 0)
        self.assertGreater(metrics['valid_chunk']['mae_active11'], 0)
        self.assertEqual(metrics['valid_chunk']['moving_sign_errors'], 1)

    def test_rejects_nonfinite_metrics(self):
        values = np.zeros((1, 1, 13)); values[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            bounded.reconstruction_metrics(values, values, [[True]], ['reach'])


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('flax', 'jax', 'optax', 'openpi')),
                     'Requires pinned OpenPI CPU environment; no dependency installation in this test')
class RealNNXWarmStartTests(unittest.TestCase):
    def test_actual_path_filter_fresh_optimizer_and_source_dtypes(self):
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax
        from types import SimpleNamespace
        from openpi.training import optimizer

        class Block(nnx.Module):
            def __init__(self):
                self.kernel = nnx.Param(jnp.array(2., dtype=jnp.bfloat16))
                self.lora_a = nnx.Param(jnp.array(.25, dtype=jnp.float32))

        class Toy(nnx.Module):
            def __init__(self):
                self.llm = Block()
                self.action_projection = nnx.Param(jnp.array(.12345679, dtype=jnp.float32))

        class ModelConfig:
            def create(self, key):
                return Toy()

        freeze = bounded.lora_only_freeze_filter()
        trainable = nnx.All(nnx.Param, nnx.Not(freeze))
        cfg = SimpleNamespace(model=ModelConfig(), optimizer=optimizer.AdamW(),
                              lr_schedule=bounded.ConstantSchedule(), trainable_filter=trainable)
        loaded = jax.tree.map(np.asarray, nnx.state(Toy()).to_pure_dict())
        state = bounded.initialize_preserving_dtype(cfg, loaded)
        self.assertEqual(int(state.step), 0)
        self.assertEqual(set(state.params.filter(trainable).flat_state()), {('llm', 'lora_a')})
        actual = state.params.to_pure_dict()
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(loaded)):
            self.assertEqual(a.dtype, b.dtype)
            np.testing.assert_array_equal(a, b)
        model = nnx.merge(state.model_def, state.params)
        def loss(m):
            return (m.action_projection.value + m.llm.kernel.value + m.llm.lora_a.value) ** 2
        _, gradients = nnx.value_and_grad(loss, argnums=nnx.DiffState(0, trainable))(model)
        params = state.params.filter(trainable)
        updates, _ = state.tx.update(gradients, state.opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        self.assertNotEqual(float(model.llm.lora_a.value), float(loaded['llm']['lora_a']))
        np.testing.assert_array_equal(model.action_projection.value, loaded['action_projection'])
        np.testing.assert_array_equal(model.llm.kernel.value, loaded['llm']['kernel'])


if __name__ == '__main__':
    unittest.main()
