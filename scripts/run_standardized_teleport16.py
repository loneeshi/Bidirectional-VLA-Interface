"""Run 16 current-runtime teleport/SAC episodes paired to the fixed panel."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from run_sac_interface_baseline import classify, runtime_environment, OFFICIAL_MSHAB


def select_plans(manifest):
    seen, rows = set(), []
    for row in manifest['episodes'][:20]:
        if row['plan_uid'] in seen:
            continue
        if not row.get('plan_uid') or len(row['binding']['subtasks']) != 20:
            raise ValueError('Missing real five-object task binding')
        seen.add(row['plan_uid'])
        rows.append(dict(seed=row['seed'], plan_uid=row['plan_uid']))
    if len(rows) != 16:
        raise ValueError('Expected exactly 16 unique historical plans')
    return rows


def command(row, output, checkpoints, expected_initial_hash):
    return [sys.executable, str(Path(__file__).with_name('run_coordinator.py')),
        '--paired-ppo-episode', '--seed', str(row['seed']),
        '--expected-plan-uid', row['plan_uid'],
        '--expected-initial-state-sha256', expected_initial_hash,
        '--checkpoint-root', str(checkpoints), '--policy-type', 'rl_per_obj',
        '--navigation-policy', 'teleport', '--manipulation-policy', 'official',
        '--navigation-camera', 'fetch_nav', '--workspace-camera',
        '--max-env-steps', '7000', '--max-wall-seconds', '900',
        '--skill-wall-seconds', '180', '--organizer-slice-steps', '40',
        '--dry-run', '--max-calls', '175', '--output', str(output)]


def panel_summary(rows):
    completed = [row for row in rows if row['status'] == 'completed']
    return dict(planned=16, completed=len(completed),
        infrastructure_failed=sum(row['status'] == 'infrastructure_failure' for row in rows),
        not_run=sum(row['status'] == 'not_run' for row in rows),
        running=sum(row['status'] == 'running' for row in rows),
        successes=sum(bool(row.get('result', {}).get('task_success')) for row in completed),
        completed_objects=sum(int(row.get('result', {}).get('completed_objects', 0))
                              for row in completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--paired-panel', type=Path, required=True,
                        help='Completed fixed/GPT panel whose fixed initial states are binding')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-new', type=int, default=2,
                        help='Small-batch checkpoint boundary; rerun to resume')
    parser.add_argument('--retry-infrastructure', action='store_true',
                        help='Preserve failed attempts and retry after a verified repair')
    args = parser.parse_args()
    selected = select_plans(json.loads(args.source_manifest.read_text()))
    paired = json.loads((args.paired_panel / 'panel-status.json').read_text())
    fixed = {row['seed']: row for row in paired['episodes']
             if row['arm'] == 'fixed' and row['status'] == 'completed'}
    if len(fixed) != 16:
        parser.error('Paired panel must contain 16 completed fixed rows')
    bindings = []
    for row in selected:
        attempt = fixed[row['seed']]['attempts'][-1]
        initial = json.loads((Path(attempt['directory']) / 'initial-state.json').read_text())
        if initial['plan_uid'] != row['plan_uid']:
            parser.error('Fixed panel plan binding changed')
        bindings.append({**row, 'expected_initial_state_sha256': initial['state_sha256']})
    if not args.execute:
        print(json.dumps({'plans': bindings, 'planned': 16, 'api_requests': 0}, indent=2))
        return
    if args.max_new < 1:
        parser.error('--max-new must be positive')

    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / 'runner.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    status_path = args.output / 'panel-status.json'
    state = (json.loads(status_path.read_text()) if status_path.exists() else
             dict(schema='standardized-teleport16/1', status='running',
                  source_manifest=str(args.source_manifest), paired_panel=str(args.paired_panel),
                  runtime_commit='e9ff3d23496d38e4431c8d913e147ffa007f7f72',
                  teleport_source_commit='4729821db3fc94a2470cfd625e6f8ab439f01478',
                  rgbd_setting='fetch_nav+fetch_workspace; identical to fixed/gpt',
                  episodes=[dict(**row, status='not_run', attempts=[])
                            for row in bindings]))
    env = runtime_environment(OFFICIAL_MSHAB)
    env.update(MS_ASSET_DIR='/home/pshuai/bvi-research/assets',
               PYTHONHASHSEED='0', BVI_DEBUG_ADAPTER_EXCEPTION='1')
    started = 0

    def save():
        state['summary'] = panel_summary(state['episodes'])
        atomic_json(status_path, state)

    try:
        state['status'] = 'running'
        save()
        for row in state['episodes']:
            if row['status'] == 'completed':
                continue
            if row['status'] == 'infrastructure_failure' and not args.retry_infrastructure:
                state['status'] = 'stopped_infrastructure'
                break
            if started >= args.max_new:
                state['status'] = 'chunk_complete'
                break
            for attempt in row['attempts']:
                if attempt['status'] == 'running':
                    attempt['status'] = 'interrupted'
            dest = (args.output / f"seed-{row['seed']:03d}" /
                    f"attempt-{len(row['attempts']) + 1:03d}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            argv = command(row, dest, args.checkpoint_root,
                           row['expected_initial_state_sha256'])
            attempt = dict(directory=str(dest), argv=argv, status='running',
                           started_unix=time.time())
            row['attempts'].append(attempt)
            row['status'] = 'running'
            started += 1
            save()
            with dest.with_suffix('.log').open('w') as log:
                child = subprocess.Popen(argv, env=env, stdout=log,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
                attempt['pid'] = child.pid
                save()
                try:
                    code = child.wait(timeout=990)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            raw = json.loads((dest / 'summary.json').read_text()) if (dest / 'summary.json').exists() else None
            status, result = classify(code, raw)
            if result.get('api_requests', 0) == 0:
                result.update(api_cost_usd=0, api_cost_status='no_api_calls')
            row.update(status=status, result=result)
            attempt.update(status=status, result=result, finished_unix=time.time())
            save()
            if status == 'infrastructure_failure':
                state['status'] = 'stopped_infrastructure'
                break
        else:
            state['status'] = 'finished'
    except BaseException as exc:
        state.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        save()
        lock.unlink()


if __name__ == '__main__':
    main()
