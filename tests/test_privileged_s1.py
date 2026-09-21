import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bvi.privileged_fetch_data import (
    EXTRA_FIELDS, SCHEMA, STATE_DIM, inspect_episode, policy_frame, privileged_state,
)


def extra():
    return {name: (np.bool_(True) if name == 'is_grasped' else np.arange(size, dtype=np.float32))
            for name, size in EXTRA_FIELDS}


def test_recorded_state42_preserves_all_fields_and_pose_quaternions():
    x = privileged_state(np.arange(12), np.arange(12) + 20, extra())
    assert STATE_DIM == 42 and x.shape == (42,)
    np.testing.assert_array_equal(x[:12], np.arange(12))
    np.testing.assert_array_equal(x[12:24], np.arange(12) + 20)
    np.testing.assert_array_equal(x[24:31], np.arange(7))
    np.testing.assert_array_equal(x[31:38], np.arange(7))
    np.testing.assert_array_equal(x[38:41], np.arange(3))
    assert x[41] == 1 and x.dtype == np.float32


@pytest.mark.parametrize('bad', [np.zeros(3), np.zeros((1, 7)), np.full(7, np.nan)])
def test_reject_cropped_batched_or_nonfinite_pose(bad):
    e = extra(); e['obj_pose_wrt_base'] = bad
    with pytest.raises(ValueError):
        privileged_state(np.zeros(12), np.zeros(12), e)


def test_missing_extra_is_not_silently_filled():
    e = extra(); del e['is_grasped']
    with pytest.raises(KeyError):
        privileged_state(np.zeros(12), np.zeros(12), e)


def test_export_uses_current_extra_and_retains_same_actions():
    n = 3
    g = {'actions': np.zeros((n, 13), np.float32),
         'success': np.array([False, True, True]), 'fail': np.zeros(n, bool),
         'obs/agent/qpos': np.ones((n+1, 12), np.float32),
         'obs/agent/qvel': np.zeros((n+1, 12), np.float32)}
    for name, size in EXTRA_FIELDS:
        g[f'obs/extra/{name}'] = (np.array([False, True, True, True]) if name == 'is_grasped'
                                     else np.arange((n+1)*size, dtype=np.float32).reshape(n+1, size))
    for cam in ('fetch_head', 'fetch_hand'):
        g[f'obs/sensor_data/{cam}/rgb'] = np.zeros((n+1, 128, 128, 3), np.uint8)
    ep = inspect_episode(g, 'pick')
    assert ep['state'].shape == (3, 42) and ep['exported_steps'] == 2
    assert ep['policy_state_schema'] == SCHEMA
    assert not ep['privileged_fields_used_only_for_annotation']
    first = policy_frame(g, ep, 0, 'Pick and stably hold the apple.')
    assert first['state'][41] == 0  # Future grasp must not leak into first state.
    np.testing.assert_array_equal(first['state'][31:38], g['obs/extra/obj_pose_wrt_base'][0])
    np.testing.assert_array_equal(first['actions'], g['actions'][0])
    with pytest.raises(IndexError):
        policy_frame(g, ep, 2, 'Pick apple')


def load_config():
    spec = importlib.util.spec_from_file_location(
        'privileged_s1_config_test', Path(__file__).parents[1] / 'scripts/fetch_privileged_s1_config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('schema,privileged,accepted', [(SCHEMA, True, True),
    (SCHEMA, False, False), ('native24', True, False)])
def test_dedicated_normalizer_contract(tmp_path, schema, privileged, accepted):
    norm = b'{}'; (tmp_path / 'norm_stats.json').write_bytes(norm)
    provenance = dict(status='train_only_stats_ready_not_runtime_validated', validation_frames_used=0,
                      dimensions={'state': 42, 'actions': 13}, delta_transform=False,
                      source_manifest_sha256='a'*64, norm_stats_sha256=hashlib.sha256(norm).hexdigest(),
                      policy_state_schema=schema, privileged_policy_inputs=privileged)
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    if accepted:
        assert load_config().normalizer_provenance(tmp_path)['policy_state_schema'] == SCHEMA
    else:
        with pytest.raises(ValueError):
            load_config().normalizer_provenance(tmp_path)
