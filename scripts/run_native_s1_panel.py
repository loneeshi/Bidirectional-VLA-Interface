"""Default CPU preflight; --execute runs a serialized, one-hour native S1 panel.

No evaluation during training, no checkpoint choice from test success, no API.
Own child processes only are terminated on completion or timeout.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from bvi.s1_capability_gate import SEEDS
from bvi.s1_evidence import build_report

GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-run', 'normalizer', 'output', 'model-python', 'sim-python'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--shader', choices=['default','minimal'], default='default')
    parser.add_argument('--sim-backend', choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--reference-panel', type=Path)
    args = parser.parse_args()
    status = json.loads((args.training_run/'status.json').read_text())
    best_path = args.training_run/'best.json'
    best = json.loads(best_path.read_text())
    if best.get('selection') != 'heldout_action_loss_only':
        raise ValueError('Missing heldout-selected checkpoint')
    checkpoint = Path(best['checkpoint'])
    assets = checkpoint/'assets/bvi/s1-official-pick-medium-train'
    contract = json.loads((assets/'bvi-state-contract.json').read_text())
    if contract.get('state_dim') != 24 or contract.get('state_source') != 'env_native_agent':
        raise ValueError('Not a native24 checkpoint')
    manifest = dict(stage='S1', question='Native Pick ability on ten fixed validation seeds',
                    checkpoint=str(checkpoint), seeds=SEEDS, per_episode_action_cap=200,
                    total_wall_seconds=3600, api_calls=0, rental_usd=0,
                    lab_charge_usd=None, execute=args.execute, training_status=status['status'],
                    shader=args.shader, sim_backend=args.sim_backend,
                    reference_panel=str(args.reference_panel) if args.reference_panel else None)
    if args.reference_panel:
        for seed in SEEDS:
            if not (args.reference_panel/f'seed{seed}'/'initial-state.pt').is_file():
                raise ValueError('Reference panel lacks a complete initial-state snapshot')
    if not args.execute:
        print(json.dumps(manifest)); return
    if status['status'] != 'completed_one_epoch':
        raise ValueError('Training must finish before native evaluation')
    used = int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError('GPU1 still occupied; no concurrent inference/render')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'launch.json').write_text(json.dumps(manifest, indent=2))
    auth = args.output/'socket-auth.local'
    auth.write_bytes(secrets.token_bytes(32)); auth.chmod(0o600)
    server_dir = args.output/'server'; server_dir.mkdir()
    socket = server_dir/'model.sock'
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ, TZ='America/New_York', CUDA_VISIBLE_DEVICES=GPU)
    started = time.monotonic()
    server = None
    def remaining():
        value = 3600-(time.monotonic()-started)
        if value <= 0:
            raise TimeoutError('Native panel exceeded one hour')
        return value
    try:
        with (server_dir/'stdout.log').open('w') as log:
            server = subprocess.Popen([str(args.model_python), str(scripts/'serve_native_s1.py'),
                '--checkpoint', str(checkpoint), '--normalizer', str(args.normalizer),
                '--output', str(server_dir), '--socket', str(socket), '--auth-file', str(auth),
                '--gpu-uuid', GPU], env=env, stdout=log, stderr=subprocess.STDOUT)
            while not (server_dir/'ready.json').exists():
                if server.poll() is not None:
                    raise RuntimeError('Model server failed before readiness')
                if time.monotonic()-started > 300:
                    raise TimeoutError('Server loading exceeded five minutes')
                time.sleep(2)
            for seed in SEEDS:
                command = [str(args.sim_python), str(scripts/'eval_native_s1.py'), '--seed', str(seed),
                           '--output', str(args.output/f'seed{seed}'), '--socket', str(socket),
                           '--auth-file', str(auth), '--shader', args.shader, '--sim-backend', args.sim_backend]
                if args.reference_panel:
                    command += ['--reference-state',str(args.reference_panel/f'seed{seed}'/'initial-state.pt')]
                with (args.output/f'seed{seed}.log').open('w') as log:
                    result = subprocess.run(command, env=dict(env, PYTHONHASHSEED=str(seed)),
                        stdout=log, stderr=subprocess.STDOUT, timeout=min(600, remaining()))
                if result.returncode:
                    raise RuntimeError(f'Seed {seed} infrastructure failure; retain evidence and stop')
            report = build_report(args.output, best_path)
            (args.output/'native-capability.json').write_text(json.dumps(report, indent=2))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        auth.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
