import importlib.util
from pathlib import Path

import pytest

from bvi.protocol import ProtocolError
from bvi.sac_interface_baseline import (begin_attempt, bind_row, finish_attempt,
    interface_metrics, new_manifest, resume_or_create_batch, summarize, validate_manifest)


def test_recovery_metrics_require_evidence_and_native_success():
    assert interface_metrics({})['interventions'] is None
    def call(status, reason=None, target='apple'):
        return {'skill': 'pick', 'target_id': target,
                'feedback': {'status': status, 'reason': reason}}
    result = interface_metrics({'skill_results': [
        call('timed_out', 'step_limit'), call('interrupted', 'missed_grasp'),
        call('timed_out', 'step_limit'), call('succeeded')]})
    assert (result['interventions'], result['replans_attempted'], result['replans_succeeded']) == (1, 1, 1)
    result = interface_metrics({'skill_results': [
        call('interrupted', 'grasp_lost'), call('succeeded', target='other')]})
    assert result['replans_attempted'] == 0
    assert result['replans_succeeded'] == 0


def test_metrics_missing_rows_are_not_silent_zero_denominators():
    manifest = bound_manifest(2)
    batch = resume_or_create_batch(manifest, 2)
    for index, row in enumerate(manifest['episodes'][:2]):
        begin_attempt(row, f'/run/{index}')
        finish_attempt(row, 'completed', {'api_requests': 2, **(
            interface_metrics({'skill_results': []}) if index else interface_metrics({}))})
    result = summarize(manifest, batch)
    assert result['interface_metrics_observed_episodes'] == 1
    assert result['intervene_frequency'] == 0
    assert result['replan_success_rate'] is None
    assert result['average_calls_per_completed_episode'] == 2


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'run_sac_interface_baseline', ROOT / 'scripts/run_sac_interface_baseline.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def bound_manifest(count=20):
    manifest = new_manifest()
    for seed in range(count):
        bind_row(manifest, seed, f'plan-{seed}', {'method': 'fixture'})
    return manifest


def test_manifest_keeps_official_1000_row_denominator():
    manifest = new_manifest()
    rows = validate_manifest(manifest)
    assert len(rows) == 1000
    assert [row['seed'] for row in rows] == list(range(1000))
    manifest['episodes'].pop()
    with pytest.raises(ProtocolError, match='1000'):
        validate_manifest(manifest)


def test_twenty_episode_batch_and_hot_resume_preserve_attempt_history():
    manifest = bound_manifest()
    batch = resume_or_create_batch(manifest, 20)
    assert batch['seeds'] == list(range(20))
    row = manifest['episodes'][0]
    first = begin_attempt(row, '/run/seed-000/attempt-001')
    assert first['attempt'] == 1
    resumed = resume_or_create_batch(manifest, 20)
    assert resumed['batch_id'] == batch['batch_id']
    second = begin_attempt(row, '/run/seed-000/attempt-002')
    assert row['attempts'][0]['status'] == 'interrupted'
    assert second['attempt'] == 2
    finish_attempt(row, 'completed', {'completed_objects': 3, 'task_success': False,
                                      'api_requests': 4})
    summary = summarize(manifest, batch)
    assert summary['validation_total'] == 1000
    assert summary['batch_planned'] == 20
    assert summary['batch_completed'] == 1
    assert summary['completed_objects'] == 3
    assert summary['benchmark_result'] is False


def test_next_batch_requires_bound_rows_and_does_not_replace_finished_rows():
    manifest = bound_manifest(40)
    first = resume_or_create_batch(manifest, 20)
    for seed in first['seeds']:
        row = manifest['episodes'][seed]
        begin_attempt(row, f'/run/{seed}')
        finish_attempt(row, 'completed', {'completed_objects': 0,
                                          'task_success': False, 'api_requests': 1})
    second = resume_or_create_batch(manifest, 20)
    assert second['seeds'] == list(range(20, 40))
    for seed in second['seeds']:
        manifest['episodes'][seed]['status'] = 'unbound'
        manifest['episodes'][seed]['plan_uid'] = None
    second['status'] = 'finished'
    with pytest.raises(ProtocolError, match='unbound episode'):
        resume_or_create_batch(manifest, 20)


def test_command_freezes_single_baseline_and_shared_bridge(tmp_path):
    argv = RUNNER.command(Path('/usr/bin/python3'), Path('/ckpt'), Path('/out/seed'),
                          Path('/out/bridge'), 'authorized', 7, 'uid-7')
    text = ' '.join(map(str, argv))
    for item in ('--benchmark-episode', '--tool-family-interface', 'gpt-5.6-luna',
                 '--navigation-policy lightnav', '--manipulation-policy official',
                 '--policy-type rl_per_obj', '--max-calls 40', '--max-env-steps 7000',
                 '--max-wall-seconds 900', '--max-input-bytes 512000'):
        assert item in text
    assert argv[argv.index('--bridge-dir') + 1] == str(Path('/out/bridge'))
    assert '--dry-run' not in argv
    assert '--record-demonstrations' not in argv


def test_explicit_model_rejection_is_a_valid_failed_rollout():
    status, result = RUNNER.classify(1, {'evaluation_eligible': True,
        'benchmark_episode': True, 'reason': 'error:ModelResponseError', 'api_requests': 2,
        'task_success': False, 'completed_objects': 0, 'planned_objects': 5,
        'steps': 0, 'wall_seconds': 1.0})
    assert status == 'completed'
    assert result['invalid_requests'] == 1


@pytest.mark.parametrize('reason', ['adapter_error:RuntimeError', 'error:ProtocolError'])
def test_infrastructure_error_is_not_completed_even_after_api_calls(reason):
    raw = dict(evaluation_eligible=True, benchmark_episode=True, reason=reason, api_requests=3)
    assert RUNNER.classify(0, raw)[0] == 'infrastructure_failure'
    if reason == 'error:ProtocolError':
        assert RUNNER.classify(1, raw, model_rejection=True)[0] == 'completed'


def test_accounting_includes_failed_and_prior_attempts_and_reports_censoring():
    manifest = bound_manifest(2)
    batch = resume_or_create_batch(manifest, 2)
    row = manifest['episodes'][0]
    begin_attempt(row, '/old')
    finish_attempt(row, 'infrastructure_failure', {'api_requests': 3})
    row['status'] = 'pending'
    begin_attempt(row, '/new')
    finish_attempt(row, *RUNNER.classify(0, dict(evaluation_eligible=True,
        benchmark_episode=True, reason='experiment_call_limit', api_requests=40)))
    second = manifest['episodes'][1]
    begin_attempt(second, '/failed')
    finish_attempt(second, 'infrastructure_failure', {'api_requests': 2})
    report = summarize(manifest, batch)
    assert report['batch_planned'] == 2
    assert report['api_requests_recorded'] == 45
    assert report['average_calls_per_completed_episode'] == 40
    assert report['budget_terminated_seeds'] == [0]


def test_runtime_pin_replaces_inherited_fork(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/wrong/acdit/fork')
    with pytest.raises(ValueError, match='Missing pinned'):
        RUNNER.runtime_environment(tmp_path)
    path = tmp_path / 'mshab/envs/sequential_task.py'
    path.parent.mkdir(parents=True)
    path.touch()
    env = RUNNER.runtime_environment(tmp_path)
    assert '/wrong/acdit/fork' not in env['PYTHONPATH']
    assert env['PYTHONPATH'].startswith(str(tmp_path.resolve()))


def test_retry_flag_is_exposed_without_changing_frozen_commands(tmp_path, monkeypatch, capsys):
    manifest = bound_manifest()
    batch = resume_or_create_batch(manifest, 20)
    row = manifest['episodes'][0]
    begin_attempt(row, '/run/attempt-001')
    finish_attempt(row, 'infrastructure_failure', {'reason': 'missing_dependency'})
    path = tmp_path / 'manifest.json'
    path.write_text(__import__('json').dumps(manifest))
    monkeypatch.setattr('sys.argv', ['runner', '--manifest', str(path),
        '--checkpoint-root', str(tmp_path), '--output', str(tmp_path / 'out'),
        '--authorization-id', 'test', '--retry-infrastructure'])
    RUNNER.main()
    output = __import__('json').loads(capsys.readouterr().out)
    assert output['jobs'][0]['seed'] == 0
    assert output['jobs'][0]['output'].endswith('attempt-002')
