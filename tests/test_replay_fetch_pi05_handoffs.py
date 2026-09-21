import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

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


def test_native24_mode_locks_only_two_validation_parents_and_four_classes():
    plan = json.loads((ROOT / 'docs/results/fetch-pi05-handoff-validation-2026-09-17/plan-v2.json').read_text())
    jobs = replay.prepare_jobs(plan, Path('/lab'), native24_validation_only=True)
    assert len(jobs) == 2
    cases = [case for _, _, rows in jobs for case in rows]
    assert len(cases) == 4
    assert {case['classification'] for case in cases} == {
        'far_grasp_diagnostic', 'near_grasp_candidate_not_completion',
        'unheld_move', 'held_move'}
    assert {case['seed'] for case in cases} == {3020, 3021}
    with pytest.raises(ValueError, match='training parents forbidden'):
        replay.source_suffix(
            '/lab/runs/fetch-current-handoffs-2026-09-17-run01/train-seed3000',
            replay.NATIVE24_VALIDATION_ALLOWED,
        )


def test_native24_parent_registry_is_derived_from_archived_partition(tmp_path):
    source = ROOT / 'docs/results/fetch-current-handoffs-2026-09-17-run01'
    target = tmp_path / 'runs/fetch-current-handoffs-2026-09-17-run01'
    target.mkdir(parents=True)
    for name in ('collection-index.json', 'batch.json'):
        shutil.copy2(source / name, target / name)
    for split, seeds in (('train', range(3000, 3004)), ('validation', (3020, 3021))):
        for seed in seeds:
            destination = target / f'{split}-seed{seed}'
            destination.mkdir()
            for name in ('result.json', 'events.jsonl'):
                shutil.copy2(source / f'{split}-seed{seed}' / name, destination / name)
    output = tmp_path / 'evidence'; output.mkdir()
    path = replay.build_native24_parent_registry(tmp_path, output)
    registry = json.loads(path.read_text())
    assert registry['status'] == 'verified_parent_partitions'
    assert {(row['split'], row['parent_episode']['parent_id']) for row in registry['parents']} == {
        *(('train', seed) for seed in range(3000, 3004)),
        ('validation', 3020), ('validation', 3021),
    }


def test_missing_native24_class_writes_not_evaluable_before_gpu_check(tmp_path):
    plan = json.loads((ROOT / 'docs/results/fetch-pi05-handoff-validation-2026-09-17/plan-v2.json').read_text())
    plan['fixed_cases'] = [
        case for case in plan['fixed_cases']
        if case['classification'] != 'held_move'
    ]
    plan_path = tmp_path / 'plan.json'; plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/replay_fetch_pi05_handoffs.py'),
        '--plan', str(plan_path), '--lab-root', str(tmp_path), '--output', str(output),
        '--gpu-uuid', 'unused-before-preflight', '--native24-validation-only',
        '--training-manifest', str(tmp_path / 'not-read-before-preflight.json')],
        text=True, capture_output=True)
    assert result.returncode == 0
    record = json.loads((output / 'result.json').read_text())
    assert record['status'] == 'not_evaluable_missing_fixed_native24_class_or_boundary'
    assert record['handoff_input_ready'] is False


def test_sequence_preregistration_freezes_causal_windows_monitor_and_budget():
    plan = json.loads((
        ROOT / 'docs/results/fetch-pi05-handoff-validation-2026-09-17/plan-v2.json'
    ).read_text())
    training = ROOT / 'docs/results/s2-native24-cache-2026-09-17-run01/manifest.json'
    input_manifest = (
        ROOT / 'docs/results/s2-native24-handoff-2026-09-18-run01/'
        'native24-handoff-manifest.json'
    )
    monitor = ROOT / 'src/bvi/progress_monitor.py'
    record = replay.build_sequence_preregistration(plan, training, input_manifest, monitor)
    windows = {row['anchor_case_id']: row['window'] for row in record['sequences']}
    assert windows['validation-seed3020-grasp-step17']['start'] == 8
    assert windows['validation-seed3020-move-step20']['start'] == 11
    assert windows['validation-seed3021-grasp-step31']['start'] == 22
    assert windows['validation-seed3021-move-step34']['start'] == 25
    assert all(row['window']['length'] == 10 for row in record['sequences'])
    assert all(row['forbidden_events'] == ['learned_threshold'] for row in record['sequences'])
    assert {row['family']: row['invocation_index'] for row in record['sequences']} == {
        'grasp': 1, 'move': 2,
    }
    assert record['source_replay']['exact_simulator_actions'] == 108
    assert record['source_replay']['hard_simulator_action_cap'] == 136
    assert record['monitor_contract']['chunk_progress_consumed_index'] == 0
    assert record['query_scope']['deployment_cadence_evaluated'] is False
