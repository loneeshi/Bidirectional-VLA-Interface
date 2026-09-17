"""Bounded frozen-policy collection with parent-episode partitions fixed in advance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cases = [('train', seed) for seed in range(3000, 3004)] + [('validation', seed) for seed in (3020, 3021)]
    rows = [dict(path=str(out / f'{split}-seed{seed}'), split=split, seed=seed,
                 scene_split='train', task='pick') for split, seed in cases]
    (out / 'collection-index.json').write_text(json.dumps(rows, indent=2))
    report = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  api_calls=0, training_updates=0, max_seconds_per_episode=150,
                  scope='native-start policy-generated handoffs, NOT LightNav handoffs', runs=[])
    def save():
        (out / 'batch.json').write_text(json.dumps(report, indent=2))
    save()
    started = time.monotonic()
    try:
        for row in rows:
            seed = str(row['seed'])
            command = ['timeout', '-k', '10s', '150s', str(root / 'envs/acdit/bin/python'),
                       str(root / 'run_lab_fetch_tapt_online.py'), '--gpu-index', '1',
                       '--seed', seed, '--output', row['path'], '--scene-split', 'train',
                       '--max-steps', '80', '--progress-timing', 'post_action_v1', '--record-decisions']
            with (out / f"{row['split']}-seed{seed}.log").open('w') as log:
                result = subprocess.run(command, cwd=root,
                    env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONUNBUFFERED': '1'},
                    stdout=log, stderr=subprocess.STDOUT)
            report['runs'].append({**row, 'returncode': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(f'Collection process failed seed{seed}: {result.returncode}')
            episode = json.loads((Path(row['path']) / 'result.json').read_text())
            if episode['status'] != 'episode_completed':
                raise RuntimeError(f'Unexpected episode status: {episode["status"]}')
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        save()


if __name__ == '__main__':
    main()
