import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handoff_replay', ROOT / 'scripts/replay_fetch_pi05_handoffs.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_physical_tolerance_and_boolean_exactness():
    replay.compare({'qpos': np.array([1 + 5e-6])}, {'qpos': [1.0]})
    with pytest.raises(ValueError, match='Replay mismatch'):
        replay.compare({'qpos': np.array([1 + 2e-5])}, {'qpos': [1.0]})
    with pytest.raises(ValueError, match='Replay mismatch'):
        replay.compare({'is_grasped': [False]}, {'is_grasped': [True]})
    with pytest.raises(ValueError, match='Replay mismatch'):
        replay.compare([1.0 + 1e-8], [1.0], exact=True)
    with pytest.raises(ValueError, match='Boolean dtype mismatch'):
        replay.compare([0.0], [False])


def test_hash_catches_bitwise_observation_or_state_difference():
    a = dict(rgb=np.zeros((2, 2, 3), np.uint8), state=np.array([.5], np.float32))
    b = copy.deepcopy(a)
    assert replay.tree_hash(a) == replay.tree_hash(b)
    b['state'][0] = np.nextafter(b['state'][0], np.float32(1))
    assert replay.tree_hash(a) != replay.tree_hash(b)
    b = copy.deepcopy(a); b['rgb'][0, 0, 0] = 1
    assert replay.tree_hash(a) != replay.tree_hash(b)


def test_plan_mapping_is_fixed_no_training_or_substitution():
    plan = json.loads((ROOT / 'docs/results/fetch-pi05-handoff-validation-2026-09-17/plan-v2.json').read_text())
    jobs = replay.prepare_jobs(plan, Path('/lab'))
    assert len(jobs) == 5 and sum(len(cases) for _, _, cases in jobs) == 8
    assert all(str(path).replace('\\', '/').startswith('/lab/runs/') for _, path, _ in jobs)
    with pytest.raises(ValueError, match='training parents forbidden'):
        replay.source_suffix('/lab/runs/fetch-current-handoffs-2026-09-17-run01/train-seed3003')
    plan['fixed_cases'][0]['boundary_step'] += 1
    with pytest.raises(ValueError, match='boundary set changed'):
        replay.prepare_jobs(plan, Path('/lab'))


def test_missing_or_nonfinite_physics_fails_closed():
    with pytest.raises(ValueError, match='Missing recorded fields'):
        replay.compare({}, {'qvel': [1.]})
    with pytest.raises(ValueError, match='Replay mismatch'):
        replay.compare([float('nan')], [0.])


def test_historical_action_cap_is_not_original_environment_horizon():
    suffix = 'fetch-current-handoffs-2026-09-17-run01/validation-seed3020'
    assert replay.replay_horizon(suffix, {'max_steps': 80}) == 200
    with pytest.raises(ValueError, match='action cap differs'):
        replay.replay_horizon(suffix, {'max_steps': 200})
    assert replay.replay_horizon('fetch-tapt-timing-2026-09-17-run01/legacy-pre-action', {'max_steps': 200}) == 200


def test_initial_snapshot_from_wrong_seed_or_checkpoint_rejected():
    result = {'seed': 3020}
    source = {'source_checkpoint_sha256': 'locked-hash'}
    manifest = {'decision_id': '000000', 'metadata': {'prediction_step': 0, 'seed': 3020,
        'family': 'reach', 'checkpoint_sha256': 'locked-hash'}}
    replay.verify_initial_manifest(manifest, result, source)
    manifest['metadata']['seed'] = 3021
    with pytest.raises(ValueError, match='identity differs'):
        replay.verify_initial_manifest(manifest, result, source)
    manifest['metadata']['seed'] = 3020
    manifest['metadata']['checkpoint_sha256'] = 'other-model'
    with pytest.raises(ValueError, match='identity differs'):
        replay.verify_initial_manifest(manifest, result, source)
