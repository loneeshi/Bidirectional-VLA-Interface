"""Hard entry gates reject thin native success, plan-only inputs, and active hold."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/fetch_tapt_entry_gate.py'
spec = importlib.util.spec_from_file_location('entry_gate', SCRIPT)
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def write(path, value):
    path.write_text(json.dumps(value)); return path


def bind_results(tmp_path, report):
    for row in report['primary_episodes']:
        path = tmp_path / f"episode-{row['seed']}.json"
        actual = {k: v for k, v in row.items() if k not in ('result_path', 'result_sha256')}
        actual.update(training_updates=0, api_calls=0, model_metadata=report['model_metadata'])
        write(path, actual)
        row.update(result_path=path.name, result_sha256=gate.sha(path))
    return report


def native(tmp_path, successes=2):
    info = {k: [True] for k in ['success', 'is_grasped', 'ee_rest', 'robot_rest', 'is_static', 'cumulative_force_within_limit']}
    info.update(fail=[False], robot_cumulative_force=[123.])
    report = dict(status='completed', primary_seeds=gate.PRIMARY_SEEDS,
        training_updates=0, api_calls=0,
        primary_episodes=[dict(seed=s, status='episode_completed', success=i < successes,
            scene_split='val', max_actions=200, ee_rest_threshold_m=.05,
            **({'final_info': info} if i < successes else {})) for i, s in enumerate(gate.PRIMARY_SEEDS)],
        primary_successes=successes, pi05_tapt_stop_this_week=successes <= 1,
        model_metadata=dict(pretrained_parameters_sha256='a' * 64, normalizer_sha256='b' * 64,
                            progress_head=False, tapt=False))
    return bind_results(tmp_path, report)


def handoff(tmp_path):
    cases = []
    for index, kind in enumerate(sorted(gate.HANDOFF_CLASSES)):
        path = tmp_path / f'case{index}.npz'
        np.savez_compressed(path, workspace_rgb=np.full((224, 224, 3), index, np.uint8),
                            wrist_rgb=np.zeros((128, 128, 3), np.uint8), state=np.zeros(30, np.float32))
        cases.append(dict(case_id=f'case{index}', role='validation', classification=kind,
            original_parent=dict(scene_split='train', seed=3020 if 'near' in kind or kind == 'held_move' else 3021),
            workspace_replay_verified=True, input_path=path.name, npz_sha256=gate.sha(path)))
    return dict(status='verified_handoff_inputs', cases=cases, trained_progress_validated=False, training_ready=False)


@pytest.mark.parametrize('successes', [0, 1])
def test_zero_or_one_native_success_stops_before_handoff_read(tmp_path, monkeypatch, successes):
    monkeypatch.setattr(gate, 'DEFAULT_HOLD', tmp_path / 'absent-hold.json')
    path = write(tmp_path / 'native.json', native(tmp_path, successes))
    with pytest.raises(ValueError, match='stopped this week'):
        gate.validate_entry(path, tmp_path / 'nonexistent-handoff.json')


def test_infrastructure_failure_not_counted_as_native_failure(tmp_path):
    value = native(tmp_path); value['primary_episodes'][-1]['status'] = 'infrastructure_failure'
    bind_results(tmp_path, value)
    with pytest.raises(ValueError, match='Infrastructure'):
        gate.validate_native(write(tmp_path / 'native.json', value))


@pytest.mark.parametrize('mutation', ['count', 'duplicate', 'threshold', 'split', 'fake_success'])
def test_native_integrity_and_actual_predicates_required(tmp_path, mutation):
    value = native(tmp_path)
    if mutation == 'count': value['primary_successes'] = 3
    if mutation == 'duplicate': value['primary_episodes'][-1]['seed'] = 2025
    if mutation == 'threshold': value['primary_episodes'][0]['ee_rest_threshold_m'] = .15
    if mutation == 'split': value['primary_episodes'][0]['scene_split'] = 'train'
    if mutation == 'fake_success': value['primary_episodes'][0]['final_info']['is_static'] = [False]
    with pytest.raises(ValueError):
        gate.validate_native(write(tmp_path / 'native.json', value))


def test_plan_only_and_missing_wrong_handoff_class_block(tmp_path):
    value = handoff(tmp_path); value['status'] = 'plan_only'
    path = write(tmp_path / 'handoff.json', value)
    with pytest.raises(ValueError, match='plan_only'):
        gate.validate_handoff(path)
    value['status'] = 'verified_handoff_inputs'; value['cases'].pop()
    with pytest.raises(ValueError, match='Need far-grasp'):
        gate.validate_handoff(write(path, value))


def test_workspace_boolean_cannot_override_changed_input_bytes(tmp_path):
    value = handoff(tmp_path); path = write(tmp_path / 'handoff.json', value)
    with (tmp_path / value['cases'][0]['input_path']).open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        gate.validate_handoff(path)


def test_training_and_locked_diagnostic_parents_cannot_enter_validation(tmp_path):
    value = handoff(tmp_path); value['cases'][0]['original_parent']['seed'] = 3000
    with pytest.raises(ValueError, match='parent'):
        gate.validate_handoff(write(tmp_path / 'handoff.json', value))


def test_active_default_hold_cannot_be_bypassed_by_another_allowed_hold(tmp_path, monkeypatch):
    default = write(tmp_path / 'default-hold.json', dict(training_allowed=False))
    allowed = write(tmp_path / 'other-hold.json', dict(training_allowed=True))
    monkeypatch.setattr(gate, 'DEFAULT_HOLD', default)
    with pytest.raises(ValueError, match='remains on hold'):
        gate.validate_entry(None, None, allowed)


def test_verified_inputs_do_not_authorize_automatic_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'DEFAULT_HOLD', tmp_path / 'absent-hold.json')
    result = gate.validate_entry(write(tmp_path / 'native.json', native(tmp_path)),
                                 write(tmp_path / 'handoff.json', handoff(tmp_path)))
    assert result['native']['primary_successes'] == 2
    assert len(result['handoff']['input_sha256']) == 4
    assert result['authorizes_automatic_resume'] is False


def test_even_twenty_update_trainer_requires_gates_before_reading_dataset(tmp_path):
    trainer = SCRIPT.with_name('train_fetch_pi05_family.py')
    result = subprocess.run([sys.executable, str(trainer), '--data', str(tmp_path / 'absent-dataset'), '--steps', '20'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert '--native-capability-report' in result.stderr or 'remains on hold' in result.stderr
    assert 'nvidia-smi' not in result.stderr and 'FileNotFoundError' not in result.stderr


@pytest.mark.parametrize('key', ['result_path', 'result_sha256'])
def test_unbacked_handwritten_native_summary_rejected(tmp_path, key):
    report = native(tmp_path)
    del report['primary_episodes'][0][key]
    with pytest.raises(ValueError, match='result_path|SHA256'):
        gate.validate_native(write(tmp_path / 'native.json', report))


@pytest.mark.parametrize('key,value', [('pretrained_parameters_sha256', 'c' * 64),
    ('normalizer_sha256', 'c' * 64), ('progress_head', True), ('tapt', True)])
def test_episode_model_identity_checked_even_with_fresh_valid_result_hash(tmp_path, key, value):
    report = native(tmp_path); row = report['primary_episodes'][0]
    path = tmp_path / row['result_path']; actual = json.loads(path.read_text())
    actual['model_metadata'][key] = value
    write(path, actual); row['result_sha256'] = gate.sha(path)
    with pytest.raises(ValueError, match='identity|progress head'):
        gate.validate_native(write(tmp_path / 'native.json', report))


@pytest.mark.parametrize('scope,key', [('batch', 'training_updates'), ('batch', 'api_calls'),
                                    ('episode', 'training_updates'), ('episode', 'api_calls')])
def test_native_gate_excludes_training_and_api_usage(tmp_path, scope, key):
    report = native(tmp_path)
    if scope == 'batch':
        report[key] = 1
    else:
        row = report['primary_episodes'][0]; path = tmp_path / row['result_path']
        actual = json.loads(path.read_text()); actual[key] = 1
        write(path, actual); row['result_sha256'] = gate.sha(path)
    with pytest.raises(ValueError, match=key):
        gate.validate_native(write(tmp_path / 'native.json', report))


def test_stale_native_episode_hash_rejected(tmp_path):
    report = native(tmp_path)
    path = tmp_path / report['primary_episodes'][0]['result_path']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='bytes changed'):
        gate.validate_native(write(tmp_path / 'native.json', report))
