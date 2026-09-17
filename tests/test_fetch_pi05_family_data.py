"""Contract tests for real endpoint masks and leakage-safe CPU conversion."""
import importlib.util
import copy
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('family_builder', Path(__file__).parents[1] / 'scripts/build_fetch_pi05_family_data.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture():
    episode = dict(state=np.arange(7 * 30, dtype=np.float32).reshape(7, 30),
                   actions=np.full((6, 13), .5, np.float32),
                   workspace_rgb=np.zeros((7, 224, 224, 3), np.uint8),
                   wrist_rgb=np.zeros((7, 128, 128, 3), np.uint8))
    stats = {key: (np.full(dim, -.5, np.float32), np.full(dim, .5, np.float32))
             for key, dim in [('state', 30), ('actions', 13)]}
    window = dict(start=2, end=5, family='grasp', instruction='Grasp the apple securely.', completion_evidence='stable_grasp')
    return episode, stats, window


def test_inclusive_endpoint_never_supervises_fake_action_or_future_progress():
    episode, stats, window = fixture()
    x = builder.invocation_arrays(episode, window, stats)
    assert x['observation_index'].tolist() == [2, 3, 4, 5]
    np.testing.assert_allclose(x['progress'], [0, 1/3, 2/3, 1])
    assert x['action_valid'].tolist() == [True, True, True, False]
    assert x['progress_valid'].all()
    assert x['chunk_action_valid'][-2].tolist() == [True] + [False] * 9
    assert x['chunk_progress_valid'][-2].tolist() == [True, True] + [False] * 8
    assert x['chunk_progress_valid'][-1].tolist() == [True] + [False] * 9
    assert not x['chunk_action_valid'][-1].any()
    assert not x['actions'][-1].any() and not x['normalized_actions'][-1].any()
    assert x['normalized_actions'].shape == (4, 32)
    assert not x['normalized_actions'][:, 13:].any()


def test_raw_v8_state_preserved_no_family_origin_reset_and_unclipped_quantiles():
    episode, stats, window = fixture()
    x = builder.invocation_arrays(episode, window, stats)
    np.testing.assert_array_equal(x['state'], episode['state'][2:6])
    assert x['normalized_state'].shape == (4, 30)
    assert x['normalized_state'].max() > 1
    np.testing.assert_allclose(x['normalized_actions'][:-1, :13], 2 / (1 + 1e-6) - 1)


def test_unverified_or_out_of_range_endpoint_rejected():
    episode, stats, window = fixture()
    with pytest.raises(ValueError, match='unverified'):
        builder.invocation_arrays(episode, {**window, 'completion_verified': False}, stats)
    with pytest.raises(ValueError, match='outside'):
        builder.invocation_arrays(episode, {**window, 'end': 7}, stats)


def test_failed_pick_retains_verified_reach_grasp_but_not_fake_move_completion():
    windows = builder.segment_episode('pick', [False, False, False, True, True, True, True],
        [.3, .2, .07, .02, .02, .02, .02], [1.] * 7, [])
    assert [w['family'] for w in windows] == ['reach', 'grasp']
    assert all(w['completion_evidence'] != 'native_pick_success_while_grasped' for w in windows)


def test_parent_split_leakage_rejected_before_output_creation(tmp_path):
    parent = dict(scene_split='train', task='pick', seed=3000)
    manifest = tmp_path / 'input.json'
    manifest.write_text(json.dumps({'episodes': [dict(directory='a', split='train', parent_episode=parent),
                                               dict(directory='b', split='validation', parent_episode=parent)]}))
    with pytest.raises(ValueError, match='crosses splits'):
        builder.build(manifest, tmp_path / 'unused.json', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_libero_dimension_stats_rejected_and_zero_range_allowed(tmp_path):
    path = tmp_path / 'stats.json'
    stats = {'state': {'q01': [0] * 8, 'q99': [1] * 8},
             'actions': {'q01': [0] * 13, 'q99': [0] * 13}}
    path.write_text(json.dumps({'norm_stats': stats}))
    with pytest.raises(ValueError, match='state'):
        builder.load_stats(path)
    stats['state'] = {'q01': [0] * 30, 'q99': [1] * 30}
    path.write_text(json.dumps({'norm_stats': stats}))
    assert np.isfinite(builder.normalize(np.zeros((1, 13)), builder.load_stats(path)['actions'])).all()


def recorded_contract():
    return dict(split='train', joint_names=list(builder.V8_JOINT_NAMES), camera=copy.deepcopy(builder.V8_CAMERA))


def test_input_manifest_cannot_relabel_recorded_validation_episode():
    with pytest.raises(ValueError, match='split'):
        builder.validate_recorded_contract(recorded_contract(), {'split': 'validation'})


def test_two_consistent_metadata_files_cannot_permute_v8_joint_order():
    report = recorded_contract()
    report['joint_names'][4], report['joint_names'][5] = report['joint_names'][5], report['joint_names'][4]
    with pytest.raises(ValueError, match='Joint order'):
        builder.validate_recorded_contract(report, {'split': 'train'})


@pytest.mark.parametrize('key,value', [('p', [.35, 0, 1.9]), ('q', [1., 0., 0., 0.]),
                                      ('fov', 1.0), ('mount', 'agent.torso_lift_link')])
def test_consistent_but_wrong_camera_configuration_rejected(key, value):
    report = recorded_contract(); report['camera'][key] = value
    with pytest.raises(ValueError, match='Camera'):
        builder.validate_recorded_contract(report, {'split': 'train'})


def full_rows():
    return [dict(split='train' if seed < 3020 else 'validation',
                 parent_episode=dict(scene_split='train', task=task, seed=seed))
            for task in ['pick', 'place'] for seed in range(3000, 3025)]


def test_full50_is_exact_parent_set_not_just_total_count():
    rows = full_rows()
    assert builder.fixed_parent_coverage(rows)
    assert not builder.fixed_parent_coverage(rows[:2])
    rows[-1]['parent_episode']['seed'] = 9999
    assert not builder.fixed_parent_coverage(rows)


def test_explicit_source_collection_completion_is_required(tmp_path):
    assert not builder.verify_source_collection({}, full_rows())['complete']
    path = tmp_path / 'collection.json'
    path.write_text(json.dumps(dict(status='running', episodes=[])))
    source = dict(source_collection=dict(path=str(path)))
    assert not builder.verify_source_collection(source, full_rows())['complete']
    source['source_collection']['status'] = 'completed'
    with pytest.raises(ValueError, match='status differs'):
        builder.verify_source_collection(source, full_rows())
