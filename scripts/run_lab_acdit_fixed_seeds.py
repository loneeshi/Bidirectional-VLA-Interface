"""Complete a predeclared native capability screen, preserving the first failure.

The first seed 2024 is the previously archived episode. The remaining four are
run once each, in order. No retries, success filtering, training or paid APIs.
"""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path.home() / 'bvi-research'
    batch = root / 'runs/acdit-fixed-seeds-2026-09-16'
    batch.mkdir(exist_ok=False)
    report = {'seeds': [2024, 2025, 2026, 2027, 2028], 'task': 'set_table/pick/013_apple',
              'max_actions': 200, 'per_episode_wall_seconds': 600, 'status': 'running', 'episodes': []}
    report_path = batch / 'batch.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    first = root / 'runs/acdit-native-2026-09-16/episode01/result.json'
    result = json.loads(first.read_text())
    assert result['seed'] == 2024 and result['status'] == 'episode_completed'
    report['episodes'].append({'seed': 2024, 'source': str(first), 'success': result['success'],
                               'steps': result['steps'], 'previously_recorded': True})
    save()
    for seed in report['seeds'][1:]:
        out = batch / f'seed{seed}'
        start = time.monotonic()
        with (batch / f'seed{seed}.log').open('w') as log:
            completed = subprocess.run(['timeout', '--kill-after=10', '600',
                str(root / 'envs/acdit/bin/python'), '-u', str(root / 'run_lab_acdit_native.py'),
                '--gpu-index', '1', '--seed', str(seed), '--output', str(out)],
                env={**os.environ, 'PYTHONHASHSEED': str(seed)}, stdout=log, stderr=subprocess.STDOUT)
        outcome = json.loads((out / 'result.json').read_text()) if (out / 'result.json').exists() else {}
        report['episodes'].append({'seed': seed, 'source': str(out / 'result.json'),
            'returncode': completed.returncode, 'status': outcome.get('status', 'no_report'),
            'success': outcome.get('success'), 'steps': outcome.get('steps'),
            'seconds': time.monotonic()-start})
        save()
        if completed.returncode or outcome.get('status') != 'episode_completed':
            report['status'] = 'stopped_on_infrastructure_failure'
            save()
            raise SystemExit(1)
    report['status'] = 'completed'
    report['successes'] = sum(x['success'] for x in report['episodes'])
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
