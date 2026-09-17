"""Predeclared two-seed check after current-frame TAPT calibration, no GPT/API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training = json.loads((args.checkpoint.parent / 'result.json').read_text())
    if (args.checkpoint.name != 'best.pt'
            or training.get('status') != 'bounded_calibration_complete_not_online_evaluated'
            or not training.get('frozen_native_unchanged')
            or not training.get('all_four_banks_changed')
            or not training.get('progress_head_changed')
            or not training.get('best_step')):
        raise ValueError('Require the validation-selected checkpoint from a completed, audited calibration')
    root = Path.home() / 'bvi-research'
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  api_calls=0, seeds=[2025, 2030], max_steps=200,
                  selected_step=training['best_step'], runs=[])
    def save():
        (out / 'batch.json').write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    save()
    try:
        for seed in report['seeds']:
            label = f'seed{seed}'
            command = ['timeout', '-k', '15s', '300s', str(root / 'envs/acdit/bin/python'),
                       str(root / 'run_lab_fetch_tapt_online.py'), '--gpu-index', '1',
                       '--seed', str(seed), '--output', str(out / label), '--scene-split', 'val',
                       '--max-steps', '200', '--progress-timing', 'current_observation_v2',
                       '--checkpoint', str(args.checkpoint), '--record-decisions']
            with (out / f'{label}.log').open('w') as log:
                result = subprocess.run(command, cwd=root,
                    env={**os.environ, 'PYTHONHASHSEED': str(seed), 'PYTHONUNBUFFERED': '1'},
                    stdout=log, stderr=subprocess.STDOUT)
            episode_path = out / label / 'result.json'
            episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
            report['runs'].append(dict(seed=seed, role='known regression' if seed == 2025 else 'predeclared holdout',
                                       returncode=result.returncode, command=command, result=episode))
            save()
            if result.returncode or episode.get('status') != 'episode_completed':
                raise RuntimeError(f'{label} execution incomplete; preserved result')
        report['status'] = 'complete'
        report['successes'] = sum(row['result']['success'] for row in report['runs'])
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        save()


if __name__ == '__main__':
    main()
