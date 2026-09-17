"""Bounded sequential Fetch author gate -> fixed teacher data -> 20-update SFT.

Joins an already running transfer/stager and teacher batch; never duplicates them.
Every GPU subprocess has an external timeout and must exit before the next phase.
No GPT/API, cloud provisioning, original-source edits, or online success claims.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.home() / 'bvi-research'
OUT = ROOT / 'runs/fetch-pi05-author-pipeline-2026-09-17-run01'
TEACHER = ROOT / 'runs/fetch-pi05-workspace-teacher-2026-09-17-run01'
GATE = ROOT / 'runs/fetch-pi05-author-gate-2026-09-17-run01'
CHECKPOINT = ROOT / 'checkpoints/fetch-v8'
GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    start = time.monotonic()
    result = dict(status='starting', api_calls=0, new_rental_usd=0, lab_charge_usd=None,
                  started_utc=datetime.now(timezone.utc).isoformat(), phases=[])
    scripts = ['check_lab_fetch_pi05.py', 'collect_lab_fetch_pi05_batch.py',
               'run_lab_fetch_pi05_teacher.py', 'build_fetch_pi05_family_data.py',
               'train_fetch_pi05_family.py', 'fetch_openpi.py']
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in scripts}
    result['script_sha256'] = hashes

    def save(status):
        result.update(status=status, wall_seconds=time.monotonic() - start)
        tmp = OUT / 'pipeline.tmp'
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(OUT / 'pipeline.json')

    def unchanged():
        for name, expected in hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Script changed during fixed pipeline: {name}')

    def gpu_idle():
        memory = int(subprocess.check_output(['nvidia-smi', '-i', GPU,
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory >= 1024:
            raise RuntimeError(f'GPU1 remains occupied: {memory} MiB; no overlapping launch')
        return memory

    def run(name, command, seconds, gpu=False):
        unchanged()
        if gpu:
            gpu_idle()
        save(name)
        phase = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat(), limit_seconds=seconds)
        result['phases'].append(phase)
        before = time.monotonic()
        with (OUT / f'{name}.log').open('x') as log:
            phase['exit_code'] = subprocess.run(
                ['timeout', '-k', '15s', f'{seconds}s', *map(str, command)],
                stdout=log, stderr=subprocess.STDOUT).returncode
        phase['wall_seconds'] = time.monotonic() - before
        if gpu:
            phase['gpu_memory_after_mib'] = gpu_idle()
        save(name + '_exited')
        if phase['exit_code'] != 0:
            raise RuntimeError(f'{name} exited {phase["exit_code"]}; see retained log')

    try:
        save('waiting_for_existing_stage_and_collection')
        deadline = time.monotonic() + 1200
        while True:
            stage = load(ROOT / 'fetch-v8-stage-result.json')
            collection = load(TEACHER / 'collection.json')
            if stage and stage['status'] == 'failed':
                raise RuntimeError(f'Checkpoint staging failed: {stage}')
            if collection and collection['status'] == 'infrastructure_failure':
                raise RuntimeError('Existing teacher batch infrastructure failure')
            if stage and stage['status'] == 'verified_checkpoint_staged' and collection and collection['status'] in ('paused_for_model_gate', 'completed'):
                result['stage'] = stage
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Existing jobs did not reach model-gate boundary within1200s')
            time.sleep(10)
        # The collector writes its terminal report just before process exit.
        time.sleep(2)
        run('model-gate', [ROOT / 'envs/openpi/bin/python', ROOT / 'check_lab_fetch_pi05.py',
            '--input-npz', ROOT / 'pi05-gate-input.npz', '--checkpoint', CHECKPOINT,
            '--output', GATE, '--gpu-uuid', GPU], 600, gpu=True)
        gate = load(GATE / 'result.json')
        result['model_gate'] = gate
        if not gate or gate['status'] != 'passed_interface_gate_untrained_head_not_task_success':
            raise RuntimeError('Strict model gate did not pass')
        if collection['status'] != 'completed':
            run('teacher-resume', [ROOT / 'envs/acdit/bin/python', ROOT / 'collect_lab_fetch_pi05_batch.py',
                '--output', TEACHER, '--resume', '--wall-seconds', '3600'], 3650, gpu=True)
        collection = load(TEACHER / 'collection.json')
        if not collection or collection['status'] != 'completed':
            raise RuntimeError('Fixed collection incomplete; no training')
        source = dict(source_collection={'path': str(TEACHER / 'collection.json')}, episodes=[
            dict(directory=e['directory'], split=e['split'], parent_episode=dict(
                scene_split='train', task=e['task'], seed=e['seed'])) for e in collection['episodes']])
        source_path = OUT / 'teacher-manifest.json'
        source_path.write_text(json.dumps(source, indent=2))
        data = OUT / 'data'
        run('build-data', [ROOT / 'envs/acdit/bin/python', ROOT / 'build_fetch_pi05_family_data.py',
            '--manifest', source_path, '--norm-stats', CHECKPOINT / 'assets/bvi/fetch-seed1-workspace-recovery-v8/norm_stats.json',
            '--output', data, '--max-episodes', '50', '--max-seconds', '600'], 630)
        manifest = load(data / 'manifest.json')
        result['data_status'] = {k: manifest.get(k) for k in ['status', 'training_ready', 'counts']}
        if not manifest.get('training_ready'):
            raise RuntimeError('Dataset not training ready; no label or threshold workaround')
        run('train20-gate', [ROOT / 'envs/openpi/bin/python', ROOT / 'train_fetch_pi05_family.py',
            '--data', data, '--checkpoint', CHECKPOINT, '--output', OUT / 'training20',
            '--gpu-uuid', GPU, '--steps', '20', '--seconds', '1800'], 2400, gpu=True)
        result['training'] = load(OUT / 'training20/result.json')
        save('pipeline_finished_no_online_success_evaluated')
    except Exception as exc:
        result['error'] = repr(exc)
        save('stopped_at_failed_gate')
        raise
    finally:
        result['gpu1_snapshot'] = subprocess.check_output(['nvidia-smi', '-i', GPU,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(result['status'])


if __name__ == '__main__':
    main()
