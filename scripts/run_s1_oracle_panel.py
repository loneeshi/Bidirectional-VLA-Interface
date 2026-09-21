"""CPU preflight by default; --execute runs a frozen five-seed oracle diagnostic.

A POSIX parent supervises one process group containing only this batch's worker,
model server and simulator descendants. No automatic repeat, navigation or API dispatch.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

from bvi.s1_oracle_protocol import PROTOCOL

SEEDS = (2024, 2025, 2026, 2027, 2028)
GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'
STAGE = 'S1_IA_single_bank_no_progress'
SOURCE_HASH = '7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e'
CANDIDATE_PROTOCOL = 'best855_lora_only_weighted_train_dev8_v1'
WALL_SECONDS = 2100
CASE_SECONDS = 300
SCRIPTS = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'normalizer', 'output', 'model-python', 'sim-python',
                 'reference-panel', 'protocol-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
    p.add_argument('--best-json', type=Path)
    p.add_argument('--repeat-seed', type=int, choices=SEEDS)
    p.add_argument('--original-panel', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def finite_nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def validate_candidate(best_path, checkpoint):
    selection = read(best_path)
    evidence = selection.get('dev_selection', {})
    step = selection.get('step')
    if (selection.get('schema') != 'bvi.s1-bounded-candidate/1'
            or selection.get('candidate_protocol') != CANDIDATE_PROTOCOL
            or Path(selection.get('checkpoint') or '').resolve() != checkpoint
            or selection.get('selection') != 'dev8_only_no_online_selection'
            or selection.get('eligible') is not True
            or selection.get('checkpoint_complete') is not True
            or selection.get('diagnostic_only') is not False
            or selection.get('source_checkpoint_parameters_sha256') != SOURCE_HASH
            or not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= 500
            or checkpoint.name != str(step) or checkpoint.parent.name != 'best'
            or evidence.get('split') != 'development'
            or evidence.get('used_online_success') is not False):
        raise ValueError('Candidate lacks non-diagnostic development selection evidence')
    parameter_hash = selection.get('checkpoint_parameters_sha256', '')
    if not isinstance(parameter_hash, str) or len(parameter_hash) != 64 or any(c not in '0123456789abcdef' for c in parameter_hash):
        raise ValueError('Candidate needs checkpoint parameter identity')
    paths = {}
    for name, hash_key, filename in (
            ('evidence_file', 'sha256', f'dev-step{step:03d}.json'),
            ('raw_evidence_file', 'raw_sha256', f'dev-step{step:03d}.npz'),
            ('baseline_evidence_file', 'baseline_sha256', 'dev-step000.json'),
            ('baseline_raw_evidence_file', 'baseline_raw_sha256', 'dev-step000.npz')):
        value = evidence.get(name)
        if not isinstance(value, str):
            raise ValueError('Candidate development evidence missing/hash mismatch')
        path = Path(value)
        if not path.is_absolute():
            path = best_path.resolve().parent / path
        if path.name != filename or not path.is_file() or evidence.get(hash_key) != digest(path):
            raise ValueError('Candidate development evidence missing/hash mismatch')
        paths[name] = path
    current, baseline = read(paths['evidence_file']), read(paths['baseline_evidence_file'])
    ratios = []
    for scope in ('first_action', 'valid_chunk'):
        numerator = current['all'][scope]['rmse_active11']
        denominator = baseline['all'][scope]['rmse_active11']
        if not finite_nonnegative(numerator) or not finite_nonnegative(denominator) or denominator <= 0:
            raise ValueError('Invalid dev8 RMSE metric')
        ratios.append(numerator / denominator)
    score = sum(ratios) / 2
    guards = selection.get('guards', {})
    for channel in ('yaw_rmse', 'torso_rmse'):
        old, new = baseline['reach']['first_action'][channel], current['reach']['first_action'][channel]
        if (not finite_nonnegative(old) or not finite_nonnegative(new)
                or new > 1.1 * old or guards.get(channel) is not True):
            raise ValueError('Candidate reach guard failed')
    for name, actual in zip(('score', 'first_rmse_ratio', 'chunk_rmse_ratio'), (score, *ratios)):
        value = selection.get(name)
        if not finite_nonnegative(value) or not math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Candidate dev8 score disagrees with frozen evidence')
    if score >= 1:
        raise ValueError('Candidate dev8 score must improve below 1')
    return selection


def validate_server_metadata(metadata, manifest):
    expected = (manifest['selection']['checkpoint_parameters_sha256']
                if manifest['variant'] == 'candidate' else SOURCE_HASH)
    if (metadata.get('pretrained_parameters_sha256') != expected
            or metadata.get('gpu_uuid') != GPU
            or metadata.get('state_contract', {}).get('training_stage') != STAGE
            or metadata.get('normalizer_sha256') != manifest['normalizer_sha256']):
        raise ValueError('Loaded server identity differs from frozen baseline/candidate')


def validate_repeat(a, expected):
    if a.variant != 'candidate' or a.original_panel is None:
        raise ValueError('Repeat requires candidate and original completed five-seed panel')
    original = a.original_panel.resolve()
    if original == a.output or a.output.is_relative_to(original):
        raise ValueError('Repeat output must be separate from original panel')
    launch, panel = read(original / 'launch.json'), read(original / 'panel.json')
    if (any(launch.get(k) != v for k, v in expected.items())
            or launch.get('repeat_seed') is not None
            or panel.get('planned_count') != 5
            or panel.get('supervisor_outcome') != 'completed'
            or panel.get('status') != 'panel_finished'
            or sorted(c.get('seed') for c in panel.get('cases', [])) != list(SEEDS)):
        raise ValueError('Original panel is incomplete or has different frozen conditions')
    case = next(c for c in panel['cases'] if c['seed'] == a.repeat_seed)
    result_path = original / f'seed{a.repeat_seed}' / 'result.json'
    result = read(result_path)
    if (case.get('status') != 'success' or case.get('native_success') is not True
            or case.get('result_sha256') != digest(result_path)
            or result.get('reference_state_sha256') != expected['reference_sha256'][str(a.repeat_seed)]
            or result.get('model_metadata', {}).get('pretrained_parameters_sha256')
                != launch['selection']['checkpoint_parameters_sha256']
            or classify(dict(case, result=str(result_path)), 0,
                        panel.get('gpu_preflight', {}).get('pci', 'missing'))['status'] != 'success'):
        raise ValueError('Repeat seed lacks matching native success evidence')
    return dict(repeat_seed=a.repeat_seed, original_panel=str(original),
        original_launch_sha256=digest(original / 'launch.json'),
        original_panel_sha256=digest(original / 'panel.json'),
        original_result_sha256=digest(result_path))


def preflight(a):
    for name in ('checkpoint', 'normalizer', 'output',
                 'reference_panel', 'protocol_manifest'):
        setattr(a, name, getattr(a, name).resolve())
    # A venv Python may symlink to a base interpreter: its invocation path selects
    # pyvenv.cfg and site-packages, so make it absolute without dereferencing it.
    for name in ('model_python', 'sim_python'):
        setattr(a, name, Path(os.path.abspath(getattr(a, name))))
    if not a.checkpoint.is_dir() or not (a.checkpoint / 'params').is_dir():
        raise ValueError('Missing complete checkpoint directory/params')
    for path in (a.model_python, a.sim_python):
        if not path.is_file():
            raise ValueError('Missing required input file: ' + str(path))
    if not a.normalizer.is_dir():
        raise ValueError('Normalizer must be a directory containing provenance.json and norm_stats.json')
    from fetch_native_s1_config import normalizer_provenance
    provenance = normalizer_provenance(a.normalizer)
    assets = a.checkpoint / 'assets/bvi/s1-official-pick-medium-train'
    contract = read(assets / 'bvi-state-contract.json')
    normalizer_hash = provenance['norm_stats_sha256']
    if (contract.get('state_dim') != 24 or contract.get('state_source') != 'env_native_agent'
            or contract.get('training_stage') != STAGE
            or contract.get('normalizer_sha256') != normalizer_hash
            or digest(assets / 'norm_stats.json') != normalizer_hash):
        raise ValueError('S1-IA native24/normalizer contract mismatch')
    selection = None
    if a.variant == 'baseline':
        if a.checkpoint.parts[-3:] != ('s1-ia-epoch-2026-09-18-run01', 'best', '855'):
            raise ValueError('Baseline must be original S1-IA best/855')
    else:
        if a.best_json is None or a.best_json.name != 'best.json':
            raise ValueError('Candidate requires its real best.json')
        selection = validate_candidate(a.best_json, a.checkpoint)
    if os.name == 'posix' and len(os.fsencode(str(a.output / 'server/model.sock'))) >= 108:
        raise ValueError('Output path exceeds Linux Unix-socket limit; choose a shorter run directory')
    reference_hashes = {}
    for seed in SEEDS:
        path = a.reference_panel / f'seed{seed}' / 'initial-state.pt'
        if not path.is_file():
            raise ValueError('Missing native initial-state.pt for seed ' + str(seed))
        reference_hashes[str(seed)] = digest(path)
    source_hashes = {name: digest(SCRIPTS.parent / name) for name in (
        'scripts/eval_native_s1_oracle.py', 'scripts/serve_native_s1.py',
        'src/bvi/s1_oracle_protocol.py', 'scripts/run_s1_oracle_panel.py')}
    expected = dict(protocol=PROTOCOL, variant=a.variant, checkpoint=str(a.checkpoint),
        training_stage=STAGE, seeds=list(SEEDS), max_actions=200,
        shader='minimal', sim_backend='gpu', normalizer_sha256=normalizer_hash,
        reference_sha256=reference_hashes, source_sha256=source_hashes)
    if selection is not None:
        expected['best_json_sha256'] = digest(a.best_json)
    repeat_seed = getattr(a, 'repeat_seed', None)
    if repeat_seed is not None:
        expected.update(validate_repeat(a, expected))
    elif getattr(a, 'original_panel', None) is not None:
        raise ValueError('--original-panel requires --repeat-seed')
    frozen = read(a.protocol_manifest)
    if frozen.get('frozen') is not True or any(frozen.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen protocol manifest mismatch; expected: ' + json.dumps(expected))
    return dict(**expected, protocol_manifest_sha256=digest(a.protocol_manifest),
        checkpoint_contract_sha256=digest(assets / 'bvi-state-contract.json'),
        selection=selection, gpu_uuid=GPU, gpu_physical_index=1,
        per_case_wall_seconds=CASE_SECONDS, total_wall_seconds=600 if repeat_seed is not None else WALL_SECONDS,
        execution_seeds=[repeat_seed] if repeat_seed is not None else list(SEEDS),
        inference_call_cap=200 if repeat_seed is not None else 1000, api_calls=0, training_updates=0,
        interpretation='oracle-assisted development regression; not historical fixed-sentence gate',
        g1_passed=False, repeat='single_seed_fresh_process_separate_denominator' if repeat_seed is not None
        else 'not_run_separate_fresh_process_evidence_required')


def gpu_guard():
    def query(fields):
        return subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=' + fields,
            '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    uuid, used, util, pci = (x.strip() for x in query('uuid,memory.used,utilization.gpu,pci.bus_id').split(','))
    if uuid != GPU or int(used) >= 1024 or int(util) != 0:
        raise RuntimeError('Physical GPU1 identity/idle guard failed')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    if any(line.split(',')[0].strip() == GPU for line in apps.splitlines()):
        raise RuntimeError('Physical GPU1 has a compute process')
    return dict(uuid=uuid, memory_mib=int(used), utilization_percent=int(util), pci=pci)


def planned_cases(output, seeds=SEEDS):
    return [dict(seed=seed, status='not_run', native_success=None, failure_causes=[],
        output=str(output / f'seed{seed}'), result=str(output / f'seed{seed}' / 'result.json'))
        for seed in seeds]


def classify(case, code, expected_pci):
    try:
        result = read(case['result'])
    except (OSError, ValueError):
        return dict(case, status='infrastructure_failure', native_success=None,
                    failure_causes=['missing_or_invalid_result'], returncode=code)
    valid = (code == 0 and result.get('status') == 'episode_completed'
             and result.get('seed') == case['seed']
             and result.get('instruction_protocol') == PROTOCOL
             and result.get('max_actions') == 200
             and 0 < result.get('steps', 0) <= 200
             and str(result.get('renderer_pci', '')).lower()[-7:] == expected_pci.lower()[-7:]
             and isinstance(result.get('success'), bool))
    if not valid:
        return dict(case, status='infrastructure_failure', native_success=None,
            failure_causes=['incomplete_or_mismatched_evaluator_result'], returncode=code)
    return dict(case, status='success' if result['success'] else 'policy_failure',
        native_success=result['success'], failure_causes=result.get('native_failure_causes', []),
        returncode=code, steps=result['steps'], result_sha256=digest(case['result']))


def worker(a):
    # The outer process owns the hard deadline and the complete child process group.
    state = read(a.output / 'panel.json')
    manifest = read(a.output / 'launch.json')
    runtime_path = os.pathsep.join([str(SCRIPTS.parent / 'src'),
        str(Path.home() / 'bvi-research'), os.environ.get('PYTHONPATH', '')])
    env = dict(os.environ, PYTHONPATH=runtime_path, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', TZ='America/New_York')
    auth = a.output / 'socket-auth.local'
    try:
        identity = gpu_guard()
        state['gpu_preflight'] = identity
        write(a.output / 'panel.json', state)
        auth.write_bytes(secrets.token_bytes(32)); auth.chmod(0o600)
        server_dir = a.output / 'server'; server_dir.mkdir()
        socket = server_dir / 'model.sock'
        with (server_dir / 'stdout.log').open('w') as log:
            server = subprocess.Popen([str(a.model_python), str(SCRIPTS / 'serve_native_s1.py'),
                '--checkpoint', str(a.checkpoint), '--normalizer', str(a.normalizer),
                '--output', str(server_dir), '--socket', str(socket), '--auth-file', str(auth),
                '--gpu-uuid', GPU, '--training-stage', STAGE,
                '--max-inference-calls', str(manifest['inference_call_cap'])], env=env, stdout=log, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while not (server_dir / 'ready.json').is_file():
                if server.poll() is not None:
                    raise RuntimeError('Model server exited before readiness')
                if time.monotonic() - start >= CASE_SECONDS:
                    raise TimeoutError('Model loading exceeded 300 seconds')
                time.sleep(.5)
            validate_server_metadata(read(server_dir / 'metadata.json'), manifest)
            for index, case in enumerate(state['cases']):
                if server.poll() is not None:
                    raise RuntimeError('Model server exited between cases')
                case['status'] = 'running'; write(a.output / 'panel.json', state)
                command = [str(a.sim_python), str(SCRIPTS / 'eval_native_s1_oracle.py'),
                    '--seed', str(case['seed']), '--output', case['output'],
                    '--socket', str(socket), '--auth-file', str(auth),
                    '--shader', 'minimal', '--sim-backend', 'gpu', '--reference-state',
                    str(a.reference_panel / f"seed{case['seed']}" / 'initial-state.pt')]
                with (a.output / f"seed{case['seed']}.log").open('w') as case_log:
                    process = subprocess.Popen(command, env=dict(env, PYTHONHASHSEED=str(case['seed'])),
                        stdout=case_log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait(timeout=CASE_SECONDS)
                    except subprocess.TimeoutExpired:
                        case.update(status='infrastructure_failure', native_success=None,
                                    failure_causes=['per_case_300_second_timeout'])
                        write(a.output / 'panel.json', state)
                        raise
                state['cases'][index] = classify(case, code, identity['pci'])
                write(a.output / 'panel.json', state)
                if state['cases'][index]['status'] == 'infrastructure_failure':
                    raise RuntimeError('Infrastructure failure; remaining planned cases retained as not_run')
        state['status'] = 'panel_finished'
    except Exception as exc:
        state['status'] = 'infrastructure_failure'
        state['error_type'] = type(exc).__name__
        raise
    finally:
        write(a.output / 'panel.json', state)
        # Parent also removes this if the worker is forcibly killed. Never report its bytes.
        auth.unlink(missing_ok=True)


def kill_group(pgid):
    # Always kill our dedicated group, even when its original worker already exited.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(a, manifest):
    if os.name != 'posix':
        raise RuntimeError('GPU execution requires POSIX process-group supervision')
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'launch.json', manifest)
    seeds = manifest.get('execution_seeds', list(SEEDS))
    wall_limit = manifest.get('total_wall_seconds', WALL_SECONDS)
    state = dict(status='starting', cases=planned_cases(a.output, seeds), g1_passed=False,
                 planned_count=len(seeds), fresh_process_repeat=manifest.get('repeat', 'not_run'))
    write(a.output / 'panel.json', state)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    process = None
    outcome = 'supervisor_error'
    started = time.monotonic()
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (a.output / 'supervisor.log').open('w') as log:
            process = subprocess.Popen(command, start_new_session=True,
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, BVI_ORACLE_SUPERVISOR_PID=str(os.getpid())))
            try:
                code = process.wait(timeout=wall_limit)
                outcome = 'completed' if code == 0 else 'worker_failure'
            except subprocess.TimeoutExpired:
                outcome = f'total_wall_{wall_limit}_second_timeout'
    finally:
        if process is not None:
            kill_group(process.pid)
            process.wait(timeout=10)
        signal.signal(signal.SIGTERM, previous)
        (a.output / 'socket-auth.local').unlink(missing_ok=True)
        state = read(a.output / 'panel.json')
        for case in state['cases']:
            if case['status'] == 'running':
                case.update(status='infrastructure_failure', native_success=None,
                            failure_causes=[outcome])
        state.update(supervisor_outcome=outcome, wall_seconds=time.monotonic() - started,
            native_successes=sum(c['native_success'] is True for c in state['cases']),
            infrastructure_failures=sum(c['status'] == 'infrastructure_failure' for c in state['cases']),
            not_run=sum(c['status'] == 'not_run' for c in state['cases']))
        if outcome != 'completed':
            state['status'] = 'infrastructure_failure'
        write(a.output / 'panel.json', state)
    return 0 if outcome == 'completed' else 1


def main():
    a = parser().parse_args()
    if a.worker:
        if (os.name != 'posix' or os.environ.get('BVI_ORACLE_SUPERVISOR_PID') != str(os.getppid())
                or os.getsid(0) != os.getpid()):
            raise ValueError('Worker must run in its supervisor-created process group')
        preflight(a)
        # Only the outer launcher may invoke the internal worker with a prewritten plan.
        if not a.execute or not (a.output / 'launch.json').is_file():
            raise ValueError('Internal worker requires a frozen executed launch')
        worker(a)
        return 0
    manifest = preflight(a)
    if not a.execute:
        print(json.dumps(dict(status='preflight_only', **manifest), indent=2))
        return 0
    return supervise(a, manifest)


if __name__ == '__main__':
    raise SystemExit(main())
