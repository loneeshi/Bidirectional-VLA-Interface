"""Fixed parent-level workspace teacher collection; no learned-model training.

The same seed belongs to the same split for both tasks. A checkpoint staging
marker can pause collection between episodes so the model gate gets GPU priority.
Completed/failed native outcomes are retained. Infrastructure errors stop the run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--collector', type=Path, default=Path(__file__).with_name('run_lab_fetch_pi05_teacher.py'))
    p.add_argument('--pause-marker', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wall-seconds', type=int, default=3600)
    a = p.parse_args()
    if not 240 <= a.wall_seconds <= 3600:
        p.error('wall-seconds must be in [240,3600]')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=a.resume)
    report_path = root / 'collection.json'
    plan = [dict(task=task, seed=seed, split='train' if seed < 3020 else 'validation')
            for seed in range(3000, 3025) for task in ['pick', 'place']]
    collector_hash = sha(a.collector)
    if a.resume:
        report = json.loads(report_path.read_text())
        if report['plan'] != plan or report['collector_sha256'] != collector_hash:
            raise ValueError('Resume would change the fixed data protocol')
    else:
        report = dict(status='planned', plan=plan, collector_sha256=collector_hash,
                      parent_split='seeds3000:3019 train;3020:3024 validation; no segment split',
                      api_calls=0, training_updates=0, navigation_handoff_coverage=False,
                      scope='frozen official SAC native-start data, not VLA success', episodes=[], launches=[])
    consumed = sum(float(x.get('wall_seconds', 0)) for x in report['launches'])
    remaining = min(a.wall_seconds, max(0, 3600 - consumed))
    launch = dict(started_utc=datetime.now(timezone.utc).isoformat(), wall_limit=remaining)
    report['launches'].append(launch)
    start = time.monotonic()

    def save(status):
        report['status'] = status
        launch['wall_seconds'] = time.monotonic() - start
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        tmp.replace(report_path)

    save('running')
    for item in plan:
        directory = root / item['task'] / f"seed{item['seed']}"
        result_path = directory / 'result.json'
        known = [r for r in report['episodes'] if r['task'] == item['task'] and r['seed'] == item['seed']]
        if known:
            result = json.loads(result_path.read_text())
            if result['status'] != 'completed' or sha(result_path) != known[0]['result_sha256']:
                raise ValueError('Existing episode evidence changed; do not overwrite')
            continue
        if a.pause_marker and a.pause_marker.exists():
            save('paused_for_model_gate'); return
        if time.monotonic() - start + 250 > remaining:
            save('wall_limit_before_next_episode'); return
        if directory.exists():
            raise FileExistsError(f'Unrecorded partial episode preserved: {directory}')
        log_path = root / f"{item['task']}-seed{item['seed']}.log"
        env = dict(os.environ, PYTHONHASHSEED=str(item['seed']))
        cmd = ['timeout', '-k', '10s', '240s', sys.executable, str(a.collector.resolve()),
               '--task', item['task'], '--seed', str(item['seed']), '--data-split', item['split'],
               '--output', str(directory), '--max-steps', '200']
        with log_path.open('x') as log:
            result_code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        entry = dict(**item, trajectory=str(directory / 'trajectory.h5'), directory=str(directory),
                     exit_code=result_code, success=result.get('success'), steps=result.get('steps'),
                     status=result.get('status'), result_sha256=sha(result_path) if result_path.exists() else None)
        report['episodes'].append(entry)
        if result_code != 0 or result.get('status') != 'completed':
            save('infrastructure_failure'); return
        save('running')
    report['counts'] = {task: dict(total=sum(e['task'] == task for e in report['episodes']),
        success=sum(e['task'] == task and e['success'] for e in report['episodes'])) for task in ['pick', 'place']}
    save('completed')


if __name__ == '__main__':
    main()
