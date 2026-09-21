"""CPU-only launcher contract, admission and failure-accounting tests."""
import importlib.util
import json
from pathlib import Path
import signal
import shutil
import subprocess
import types

import pytest

SPEC = importlib.util.spec_from_file_location('oracle_panel', Path(__file__).parents[1] / 'scripts/run_s1_oracle_panel.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # Match direct script execution, where sibling config modules are importable.
    monkeypatch.syspath_prepend(str(mod.SCRIPTS))
    checkpoint = tmp_path / 's1-ia-epoch-2026-09-18-run01/best/855'
    (checkpoint / 'params').mkdir(parents=True)
    assets = checkpoint / 'assets/bvi/s1-official-pick-medium-train'
    normalizer = tmp_path / 'normalizer'
    put(normalizer / 'norm_stats.json', {'state': 'fixture'})
    put(normalizer / 'provenance.json', dict(
        status='train_only_stats_ready_not_runtime_validated', validation_frames_used=0,
        dimensions={'state': 24, 'actions': 13}, delta_transform=False,
        norm_stats_sha256=mod.digest(normalizer / 'norm_stats.json'),
        source_manifest_sha256='a' * 64))
    put(assets / 'norm_stats.json', {'state': 'fixture'})
    put(assets / 'bvi-state-contract.json', dict(state_dim=24, state_source='env_native_agent',
        training_stage=mod.STAGE, normalizer_sha256=mod.digest(normalizer / 'norm_stats.json')))
    reference = tmp_path / 'reference'
    for seed in mod.SEEDS:
        path = reference / f'seed{seed}/initial-state.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(str(seed).encode())
    executable = tmp_path / 'python'; executable.write_text('fixture')
    a = types.SimpleNamespace(checkpoint=checkpoint, normalizer=normalizer,
        output=tmp_path / 'output', model_python=executable, sim_python=executable,
        reference_panel=reference, protocol_manifest=tmp_path / 'frozen.json',
        variant='baseline', best_json=None, execute=False)
    freeze(a)
    return a


def freeze(a):
    put(a.protocol_manifest, dict(frozen=True, protocol=mod.PROTOCOL,
        variant=a.variant, checkpoint=str(a.checkpoint.resolve()), training_stage=mod.STAGE,
        seeds=list(mod.SEEDS), max_actions=200, shader='minimal', sim_backend='gpu',
        normalizer_sha256=mod.digest(a.normalizer / 'norm_stats.json'),
        reference_sha256={str(s): mod.digest(a.reference_panel / f'seed{s}/initial-state.pt') for s in mod.SEEDS},
        source_sha256={n: mod.digest(mod.SCRIPTS.parent / n) for n in (
            'scripts/eval_native_s1_oracle.py', 'scripts/serve_native_s1.py',
            'src/bvi/s1_oracle_protocol.py', 'scripts/run_s1_oracle_panel.py')}))
    frozen = mod.read(a.protocol_manifest)
    if a.variant == 'candidate':
        frozen['best_json_sha256'] = mod.digest(a.best_json)
    if getattr(a, 'repeat_seed', None) is not None:
        frozen.update(mod.validate_repeat(a, {k: v for k, v in frozen.items() if k != 'frozen'}))
    put(a.protocol_manifest, frozen)


def test_preflight_never_queries_gpu_or_starts_process(inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU preflight attempted external execution')
    monkeypatch.setattr(mod.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(mod.subprocess, 'Popen', forbidden)
    result = mod.preflight(inputs)
    assert result['seeds'] == [2024, 2025, 2026, 2027, 2028]
    assert result['inference_call_cap'] == 1000
    assert not result['g1_passed']
    assert not inputs.output.exists()


def test_changed_initial_state_rejects_frozen_manifest(inputs):
    (inputs.reference_panel / 'seed2024/initial-state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen protocol manifest mismatch'):
        mod.preflight(inputs)


def test_preflight_preserves_both_venv_python_symlinks(inputs, tmp_path, monkeypatch):
    target = inputs.model_python.resolve()
    for name in ('model_python', 'sim_python'):
        link = tmp_path / name / 'bin' / 'python'
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(target)
        except OSError as exc:
            pytest.skip(f'Creating symlinks is unavailable: {exc}')
        setattr(inputs, name, link.relative_to(tmp_path))
    monkeypatch.chdir(tmp_path)
    mod.preflight(inputs)
    for name in ('model_python', 'sim_python'):
        actual = getattr(inputs, name)
        assert actual == tmp_path / name / 'bin' / 'python'
        assert actual.is_absolute() and actual.is_symlink()
        assert actual.resolve() == target
        assert actual != target
    assert inputs.checkpoint == inputs.checkpoint.resolve()


def test_missing_native_initial_state_rejected(inputs):
    (inputs.reference_panel / 'seed2024/initial-state.pt').unlink()
    with pytest.raises(ValueError, match='Missing native initial-state'):
        mod.preflight(inputs)


def test_normalizer_mismatch_rejected(inputs):
    (inputs.checkpoint / 'assets/bvi/s1-official-pick-medium-train/norm_stats.json').write_text('{}')
    with pytest.raises(ValueError, match='contract mismatch'):
        mod.preflight(inputs)


def test_normalizer_file_rejected(inputs):
    inputs.normalizer = inputs.normalizer / 'norm_stats.json'
    with pytest.raises(ValueError, match='Normalizer must be a directory'):
        mod.preflight(inputs)


@pytest.mark.parametrize('filename', ['provenance.json', 'norm_stats.json'])
def test_normalizer_missing_required_file_rejected(inputs, filename):
    (inputs.normalizer / filename).unlink()
    with pytest.raises(FileNotFoundError):
        mod.preflight(inputs)


@pytest.mark.parametrize('field,value', [
    ('validation_frames_used', 1), ('dimensions', {'state': 30, 'actions': 13}),
    ('norm_stats_sha256', '0' * 64), ('delta_transform', True),
    ('source_manifest_sha256', 'invalid'), ('status', 'unvalidated')])
def test_normalizer_bad_provenance_rejected(inputs, field, value):
    path = inputs.normalizer / 'provenance.json'
    provenance = mod.read(path)
    provenance[field] = value
    put(path, provenance)
    with pytest.raises(ValueError, match='normalizer provenance'):
        mod.preflight(inputs)


def metric(value):
    return {'all': {'first_action': {'rmse_active11': value},
                    'valid_chunk': {'rmse_active11': value}},
            'reach': {'first_action': {'yaw_rmse': value, 'torso_rmse': value}}}


def candidate(a, diagnostic=False, online=False):
    a.variant = 'candidate'
    selected = a.output.parent / 'candidate-run/best/125'
    shutil.copytree(a.checkpoint, selected)
    (selected / 'train_state').mkdir()
    a.checkpoint = selected
    a.best_json = selected.parent.parent / 'best.json'
    directory = a.best_json.parent
    put(directory / 'dev-step125.json', metric(.8))
    put(directory / 'dev-step000.json', metric(1.))
    (directory / 'dev-step125.npz').write_bytes(b'raw_fixture_selected')
    (directory / 'dev-step000.npz').write_bytes(b'raw_fixture_baseline')
    evidence = dict(split='development', used_online_success=online)
    for key, hash_key, name in (
            ('evidence_file', 'sha256', 'dev-step125.json'),
            ('raw_evidence_file', 'raw_sha256', 'dev-step125.npz'),
            ('baseline_evidence_file', 'baseline_sha256', 'dev-step000.json'),
            ('baseline_raw_evidence_file', 'baseline_raw_sha256', 'dev-step000.npz')):
        evidence[key] = name
        evidence[hash_key] = mod.digest(directory / name)
    put(a.best_json, dict(checkpoint=str(a.checkpoint), selection='dev8_only_no_online_selection',
        schema='bvi.s1-bounded-candidate/1', candidate_protocol=mod.CANDIDATE_PROTOCOL,
        diagnostic_only=diagnostic, eligible=True, checkpoint_complete=True,
        source_checkpoint_parameters_sha256=mod.SOURCE_HASH, checkpoint_parameters_sha256='a' * 64,
        step=125, score=.8, first_rmse_ratio=.8, chunk_rmse_ratio=.8,
        guards={'yaw_rmse': True, 'torso_rmse': True}, dev_selection=evidence))
    freeze(a)


@pytest.mark.parametrize('diagnostic,online', [(True, False), (False, True)])
def test_candidate_rejects_diagnostic_or_online_selection(inputs, diagnostic, online):
    candidate(inputs, diagnostic, online)
    with pytest.raises(ValueError, match='development selection evidence'):
        mod.preflight(inputs)


def test_candidate_requires_real_hash_bound_development_evidence(inputs):
    candidate(inputs)
    assert mod.preflight(inputs)['variant'] == 'candidate'
    (inputs.best_json.parent / 'dev-step125.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence missing/hash mismatch'):
        mod.preflight(inputs)


def test_gpu_guard_queries_only_gpu1_and_rejects_active_process(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        if '--query-gpu=uuid,memory.used,utilization.gpu,pci.bus_id' in command:
            return mod.GPU + ', 15, 0, 00000000:81:00.0'
        return mod.GPU + ', 1234'
    monkeypatch.setattr(mod.subprocess, 'check_output', query)
    with pytest.raises(RuntimeError, match='compute process'):
        mod.gpu_guard()
    assert calls[0][calls[0].index('-i') + 1] == '1'
    assert all('kill' not in x for command in calls for x in command)


def test_result_accounting_does_not_turn_partial_success_into_policy_success(tmp_path):
    case = mod.planned_cases(tmp_path)[0]
    put(Path(case['result']), dict(status='infrastructure_failure', success=True))
    result = mod.classify(case, 1, '00000000:81:00.0')
    assert result['status'] == 'infrastructure_failure'
    assert result['native_success'] is None
    assert [x['status'] for x in mod.planned_cases(tmp_path)] == ['not_run'] * 5


@pytest.mark.parametrize('success', [True, False])
def test_completed_native_result_keeps_policy_failure_separate(tmp_path, success):
    case = mod.planned_cases(tmp_path)[0]
    put(Path(case['result']), dict(status='episode_completed', seed=2024,
        instruction_protocol=mod.PROTOCOL, max_actions=200, steps=200,
        renderer_pci='0000:81:00.0', success=success, native_failure_causes=['force']))
    result = mod.classify(case, 0, '00000000:81:00.0')
    assert result['status'] == ('success' if success else 'policy_failure')
    assert result['native_success'] is success


def test_supervisor_timeout_kills_owned_group_and_retains_denominator(inputs, monkeypatch):
    killed = []
    fake_os = types.SimpleNamespace(name='posix', environ={}, getpid=lambda: 800,
                                   killpg=lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(mod, 'os', fake_os)
    monkeypatch.setattr(mod.signal, 'SIGKILL', 9, raising=False)
    monkeypatch.setattr(mod.signal, 'signal', lambda *args: None)
    class Process:
        pid = 900
        def wait(self, timeout):
            if timeout == mod.WALL_SECONDS:
                raise subprocess.TimeoutExpired('worker', timeout)
            return -9
    launches = []
    def popen(command, **kwargs):
        launches.append(kwargs)
        state = mod.read(inputs.output / 'panel.json')
        state['cases'][0]['status'] = 'running'
        mod.write(inputs.output / 'panel.json', state)
        return Process()
    monkeypatch.setattr(mod.subprocess, 'Popen', popen)
    assert mod.supervise(inputs, {'frozen': True}) == 1
    assert killed == [(900, 9)]
    assert launches[0]['start_new_session'] is True
    result = mod.read(inputs.output / 'panel.json')
    assert result['infrastructure_failures'] == 1
    assert result['not_run'] == 4
    assert result['native_successes'] == 0
    assert len(result['cases']) == 5


@pytest.mark.parametrize('field,value', [('score', 1.), ('score', float('nan')),
    ('score', True), ('eligible', False), ('checkpoint_complete', False),
    ('selection', 'heldout_action_loss_only')])
def test_candidate_rejects_ineligible_or_fabricated_metric(inputs, field, value):
    candidate(inputs)
    best = mod.read(inputs.best_json); best[field] = value; put(inputs.best_json, best)
    with pytest.raises(ValueError):
        mod.preflight(inputs)


def test_recomputes_reach_guard_even_when_claimed_passed(inputs):
    candidate(inputs)
    best = mod.read(inputs.best_json)
    evidence = inputs.best_json.parent / 'dev-step125.json'
    changed = metric(.8); changed['reach']['first_action']['yaw_rmse'] = 1.11
    put(evidence, changed)
    best['dev_selection']['sha256'] = mod.digest(evidence); put(inputs.best_json, best)
    with pytest.raises(ValueError, match='reach guard failed'):
        mod.preflight(inputs)


def test_rejects_changed_baseline_raw_evidence(inputs):
    candidate(inputs)
    (inputs.best_json.parent / 'dev-step000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence missing/hash mismatch'):
        mod.preflight(inputs)


def test_server_must_restore_selected_parameter_identity():
    manifest = dict(variant='candidate', normalizer_sha256='n',
                    selection={'checkpoint_parameters_sha256': 'a' * 64})
    metadata = dict(pretrained_parameters_sha256='a' * 64, gpu_uuid=mod.GPU,
                    normalizer_sha256='n', state_contract={'training_stage': mod.STAGE})
    mod.validate_server_metadata(metadata, manifest)
    metadata['pretrained_parameters_sha256'] = mod.SOURCE_HASH
    with pytest.raises(ValueError, match='Loaded server identity'):
        mod.validate_server_metadata(metadata, manifest)


def test_real_candidate_producer_matches_launcher_gate(inputs, monkeypatch):
    candidate(inputs)
    monkeypatch.syspath_prepend(str(mod.SCRIPTS))
    spec = importlib.util.spec_from_file_location('candidate_producer_contract',
        mod.SCRIPTS / 'train_s1_bounded_candidate.py')
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    score = dict(step=125, **producer.score_dev(metric(.8), metric(1.)))
    record = producer.best_record(inputs.best_json.parent, inputs.checkpoint, score, 'a' * 64)
    put(inputs.best_json, record); freeze(inputs)
    result = mod.preflight(inputs)
    assert result['selection']['score'] == .8
    assert 'validation_action_loss' not in result['selection']
    assert result['selection']['fresh_checkpoint_restore_verified'] is False


def original_success_panel(a):
    candidate(a)
    launch = mod.preflight(a)
    original = a.output.parent / 'original-panel'
    put(original / 'launch.json', launch)
    cases = mod.planned_cases(original)
    for c in cases:
        c.update(status='policy_failure', native_success=False)
    result = dict(status='episode_completed', seed=2025, instruction_protocol=mod.PROTOCOL,
        max_actions=200, steps=61, renderer_pci='0000:81:00.0', success=True,
        reference_state_sha256=launch['reference_sha256']['2025'],
        model_metadata={'pretrained_parameters_sha256': 'a' * 64})
    path = original / 'seed2025/result.json'; put(path, result)
    cases[1].update(status='success', native_success=True, result_sha256=mod.digest(path))
    put(original / 'panel.json', dict(planned_count=5, status='panel_finished',
        supervisor_outcome='completed', cases=cases, gpu_preflight={'pci': '00000000:81:00.0'}))
    a.original_panel = original
    a.repeat_seed = 2025
    return original


def test_repeat_one_success_uses_separate_denominator_and_600_seconds(inputs):
    original_success_panel(inputs)
    freeze(inputs)
    result = mod.preflight(inputs)
    assert result['execution_seeds'] == [2025]
    assert result['total_wall_seconds'] == 600
    assert result['inference_call_cap'] == 200
    assert result['seeds'] == list(mod.SEEDS)  # Frozen original regression conditions.
    assert not result['g1_passed']
    assert len(mod.planned_cases(inputs.output, result['execution_seeds'])) == 1


def test_repeat_rejects_unsuccessful_seed(inputs):
    original_success_panel(inputs)
    inputs.repeat_seed = 2024
    with pytest.raises((ValueError, FileNotFoundError)):
        freeze(inputs)


def test_repeat_rejects_changed_source_conditions(inputs):
    original = original_success_panel(inputs)
    launch = mod.read(original / 'launch.json')
    launch['source_sha256']['scripts/eval_native_s1_oracle.py'] = 'changed'
    put(original / 'launch.json', launch)
    with pytest.raises(ValueError, match='different frozen conditions'):
        freeze(inputs)


def test_repeat_rejects_mutated_original_success_result(inputs):
    original = original_success_panel(inputs)
    result = mod.read(original / 'seed2025/result.json'); result['steps'] = 62
    put(original / 'seed2025/result.json', result)
    with pytest.raises(ValueError, match='native success evidence'):
        freeze(inputs)


def test_baseline_cannot_be_candidate_g1_repeat(inputs):
    inputs.repeat_seed = 2025
    inputs.original_panel = inputs.output.parent / 'anything'
    with pytest.raises(ValueError, match='Repeat requires candidate'):
        mod.preflight(inputs)
