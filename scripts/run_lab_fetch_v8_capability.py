"""Six fixed frozen-V8 native episodes; no TAPT launch, even if the gate passes."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time

ROOT = Path.home() / 'bvi-research'
OUT = ROOT / 'runs/fetch-v8-native-capability-2026-09-17-run01'
GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT / 'runs'):
        parser.error('Output must be a fresh directory under the laboratory runs directory')
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    report = dict(status='waiting_for_existing_teacher_exit', primary_seeds=[2024, 2025, 2026, 2027, 2028],
        auxiliary_seeds=[2025, 2030], unique_seeds=[2024, 2025, 2026, 2027, 2028, 2030],
        primary_episodes=[], auxiliary_episodes=[], episodes=[],
        started_utc=datetime.now(timezone.utc).isoformat(), training_updates=0, api_calls=0,
        max_actions=200, scene_split='val', ee_rest_threshold_m=.05,
        historical_pairing='qpos/head/hand RGB+depth; no original AC full-state snapshot',
        pi05_tapt_stop_this_week=None, never_launches_training=True)
    server = None
    auth = OUT / 'auth.bin'
    server_out = OUT / 'server'; socket = server_out / 'policy.sock'

    def save():
        tmp = OUT / 'batch.tmp'; tmp.write_text(json.dumps(report, indent=2)); tmp.replace(OUT / 'batch.json')

    save()
    try:
        deadline = time.monotonic() + 1200
        collection_path = ROOT / 'runs/fetch-pi05-workspace-teacher-2026-09-17-run01/collection.json'
        while True:
            collection = json.loads(collection_path.read_text())
            if collection['status'] != 'running':
                used = int(subprocess.check_output(['nvidia-smi', '-i', GPU,
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                if used < 1024: break
            if time.monotonic() > deadline: raise TimeoutError('Existing collection has not released GPU')
            time.sleep(10)
        hold = json.loads((ROOT / 'fetch-pi05-training-hold.json').read_text())
        if hold.get('training_allowed') is not False:
            raise RuntimeError('Training hold must remain active during capability evaluation')
        auth.write_bytes(os.urandom(32)); auth.chmod(0o600)
        report['status'] = 'loading_frozen_v8'; save()
        with (OUT / 'server.log').open('x') as server_log:
            server = subprocess.Popen(['timeout', '-k', '20s', '4300s', ROOT / 'envs/openpi/bin/python',
                ROOT / 'serve_lab_fetch_v8_native.py', '--checkpoint', ROOT / 'checkpoints/fetch-v8',
                '--output', server_out, '--socket', socket, '--auth-file', auth, '--gpu-uuid', GPU],
                stdout=server_log, stderr=subprocess.STDOUT)
            until = time.monotonic() + 300
            while not (server_out / 'ready.json').exists():
                if server.poll() is not None: raise RuntimeError('Frozen model server failed to initialize')
                if time.monotonic() > until: raise TimeoutError('Frozen model server startup exceeded300s')
                time.sleep(3)
            report['model_metadata'] = json.loads((server_out / 'metadata.json').read_text())
            for seed in report['unique_seeds']:
                historical = (ROOT / 'runs/acdit-native-2026-09-16/episode01/reset-observation.npz' if seed == 2024 else
                    ROOT / 'runs/fetch-current-progress-eval-2026-09-17-run01/seed2030/reset-observation.npz' if seed == 2030 else
                    ROOT / f'runs/acdit-fixed-seeds-2026-09-16/seed{seed}/reset-observation.npz')
                directory = OUT / f'seed{seed}'
                report.update(status='evaluating', current_seed=seed); save()
                before = time.monotonic()
                with (OUT / f'seed{seed}.log').open('x') as log:
                    code = subprocess.run(['timeout', '-k', '15s', '600s', ROOT / 'envs/acdit/bin/python',
                        ROOT / 'eval_lab_fetch_v8_native.py', '--seed', str(seed), '--output', directory,
                        '--socket', socket, '--auth-file', auth, '--historical-reset', historical],
                        env=dict(os.environ, PYTHONHASHSEED=str(seed)), stdout=log, stderr=subprocess.STDOUT).returncode
                path = directory / 'result.json'
                result = json.loads(path.read_text()) if path.exists() else {}
                row = dict(seed=seed, status=result.get('status', 'no_report'), success=result.get('success'),
                    ever_grasped=result.get('ever_grasped'), stable_grasp_3_observations=result.get('stable_grasp_3_observations'),
                    steps=result.get('steps'), exit_code=code, seconds=time.monotonic() - before,
                    scene_split='val', max_actions=200, ee_rest_threshold_m=.05, result_path=str(path),
                    final_info=result.get('final_info'),
                    result_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
                report['episodes'].append(row)
                if seed in report['primary_seeds']: report['primary_episodes'].append(row)
                if seed in report['auxiliary_seeds']: report['auxiliary_episodes'].append(row)
                save()
                if code != 0 or row['status'] != 'episode_completed':
                    raise RuntimeError(f'Seed{seed} infrastructure/protocol error; not a native failure score')
            from multiprocessing.connection import Client
            with Client(str(socket), family='AF_UNIX', authkey=auth.read_bytes()) as conn:
                conn.send({'op': 'shutdown'})
                if not conn.poll(10): raise TimeoutError('Model server shutdown did not acknowledge')
                conn.recv()
            server.wait(timeout=30)
        successes = sum(e['success'] for e in report['primary_episodes'])
        report.update(status='completed', primary_successes=successes,
            pi05_tapt_stop_this_week=successes <= 1,
            decision='STOP_WEEKLY_PI05_TAPT' if successes <= 1 else 'NATIVE_GATE_PASSED_HANDOFF_VALIDATION_STILL_REQUIRED',
            auxiliary_successes=sum(e['success'] for e in report['auxiliary_episodes']),
            overlap_note='seed2025 is reused in two panels, not an additional independent episode')
    except Exception as exc:
        report.update(status='infrastructure_or_protocol_block', error=repr(exc)); raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=25)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
        if auth.exists(): auth.unlink()  # Only this batch's generated32-byte socket credential.
        report['gpu1_final_snapshot'] = subprocess.check_output(['nvidia-smi', '-i', GPU,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
