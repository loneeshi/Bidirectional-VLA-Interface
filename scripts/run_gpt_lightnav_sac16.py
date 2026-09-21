"""Resume 16 bound plans in two-episode chunks with GPT + LightNav + SAC."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from run_ppo_sac_paired16 import select_plans
from run_sac_interface_baseline import classify, runtime_environment, OFFICIAL_MSHAB


def command(row, output, checkpoints, bridge, authorization, initial_hash, lightnav_url):
    return [sys.executable, str(Path(__file__).with_name('run_coordinator.py')),
        '--benchmark-episode', '--goal-tools', '--tool-family-interface', '--organizer',
        '--seed', str(row['seed']), '--expected-plan-uid', row['plan_uid'],
        '--expected-initial-state-sha256', initial_hash,
        '--checkpoint-root', str(checkpoints), '--policy-type', 'rl_per_obj',
        '--navigation-policy', 'lightnav', '--navigation-control', 'waypoint_velocity',
        # Replanning every five physical steps means 1400 predictions is the
        # exact 7000-step episode ceiling, not an additional method-specific cap.
        '--lightnav-url', lightnav_url, '--max-navigation-predictions', '1400',
        '--manipulation-policy', 'official', '--navigation-camera', 'fetch_nav',
        '--workspace-camera', '--max-env-steps', '7000', '--max-wall-seconds', '900',
        '--skill-wall-seconds', '180', '--organizer-slice-steps', '40',
        '--max-calls', '40', '--provider', 'openai', '--model', 'gpt-5.6-luna',
        '--transport', 'bridge', '--bridge-dir', str(bridge),
        '--bridge-timeout-seconds', '120', '--authorization-id', authorization,
        '--max-api-cost-usd', '0.05', '--request-cost-ceiling-usd', '0.00125',
        '--max-output-tokens', '2048', '--max-input-bytes', '512000',
        '--output', str(output)]


def aggregate(state):
    rows = state['episodes']
    completed = [row for row in rows if row['status'] == 'completed']
    return dict(planned=16, completed=len(completed),
        infrastructure_failed=sum(row['status'] == 'infrastructure_failure' for row in rows),
        not_run=sum(row['status'] == 'not_run' for row in rows),
        running=sum(row['status'] == 'running' for row in rows),
        task_successes=sum(row.get('result', {}).get('task_success', False) for row in completed),
        completed_objects=sum(row.get('result', {}).get('completed_objects', 0) for row in completed),
        api_requests=sum(attempt.get('result', {}).get('api_requests', 0)
                         for row in rows for attempt in row['attempts']),
        navigation_predictions=sum(row.get('result', {}).get('navigation_predictions', 0)
                                   for row in completed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--reference-panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--bridge-dir', type=Path, required=True)
    parser.add_argument('--authorization-id', required=True)
    parser.add_argument('--lightnav-url', default='ws://127.0.0.1:8050')
    parser.add_argument('--max-new', type=int, default=2)
    parser.add_argument('--retry-infrastructure', action='store_true')
    parser.add_argument('--retry-code-failures', action='store_true',
                        help='Retry preserved attempts invalidated by a verified interface defect')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_new <= 2:
        parser.error('--max-new must be one or two')
    plans = select_plans(json.loads(args.source_manifest.read_text()))
    reference = json.loads(args.reference_panel.read_text())
    fixed = {(row['seed'], row['plan_uid']): row for row in reference['episodes']
             if row['arm'] == 'fixed' and row['status'] == 'completed'}
    bindings = []
    for row in plans:
        paired = fixed.get((row['seed'], row['plan_uid']))
        if paired is None:
            raise ValueError('Every LightNav row requires a completed fixed binding')
        initial_path = Path(paired['attempts'][-1]['directory']) / 'initial-state.json'
        initial = json.loads(initial_path.read_text())
        if initial['plan_uid'] != row['plan_uid']:
            raise ValueError('Reference initial state plan mismatch')
        bindings.append({**row, 'initial_state_sha256': initial['state_sha256']})
    if not args.execute:
        print(json.dumps(dict(schema='gpt-lightnav-sac16/1', plans=bindings,
                              max_api_requests=640), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / 'runner.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    status_path = args.output / 'panel-status.json'
    state = (json.loads(status_path.read_text()) if status_path.exists() else
             dict(schema='gpt-lightnav-sac16/1', status='running',
                  source_manifest=str(args.source_manifest),
                  reference_panel=str(args.reference_panel),
                  authorization_id=args.authorization_id,
                  episodes=[dict(**row, status='not_run', attempts=[]) for row in bindings]))
    env = runtime_environment(OFFICIAL_MSHAB)
    env.update(MS_ASSET_DIR='/home/pshuai/bvi-research/assets', PYTHONHASHSEED='0',
               BVI_DEBUG_ADAPTER_EXCEPTION='1')

    def save():
        state['summary'] = aggregate(state)
        atomic_json(status_path, state)

    started = 0
    state['status'] = 'running'
    save()
    try:
        for row in state['episodes']:
            retry_code = (args.retry_code_failures and row['status'] == 'completed'
                          and row.get('result', {}).get('reason') == 'error:ModelResponseError')
            if row['status'] == 'completed' and not retry_code:
                continue
            if row['status'] == 'infrastructure_failure' and not args.retry_infrastructure:
                continue
            if started >= args.max_new:
                break
            if retry_code:
                row['attempts'][-1].setdefault('validity_adjudications', []).append(dict(
                    classification='excluded_interface_defect',
                    reason='semantically_identical_repeated_json_rejected',
                    time_unix=time.time(), original_status=row['status']))
            elif row['status'] == 'infrastructure_failure':
                prior_events = Path(row['attempts'][-1]['directory']) / 'events.jsonl'
                prediction_cap = (prior_events.is_file() and any(
                    json.loads(line).get('event') == 'navigation_rejected'
                    and json.loads(line).get('reason') == 'prediction_cap'
                    for line in prior_events.read_text().splitlines()))
                row['attempts'][-1].setdefault('validity_adjudications', []).append(dict(
                    classification=('excluded_configuration_defect' if prediction_cap
                                    else 'excluded_infrastructure_failure'),
                    reason=('episode_level_lightnav_prediction_cap' if prediction_cap
                            else 'operator_authorized_infrastructure_retry'),
                    time_unix=time.time(), original_status=row['status']))
            for attempt in row['attempts']:
                if attempt['status'] == 'running':
                    attempt['status'] = 'interrupted'
            destination = (args.output / f"seed-{row['seed']:03d}" /
                           f"attempt-{len(row['attempts']) + 1:03d}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            argv = command(row, destination, args.checkpoint_root, args.bridge_dir,
                           args.authorization_id, row['initial_state_sha256'], args.lightnav_url)
            attempt = dict(directory=str(destination), argv=argv, status='running',
                           started_unix=time.time())
            row['attempts'].append(attempt)
            row['status'] = 'running'
            save()
            started += 1
            with destination.with_suffix('.log').open('w') as log:
                child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
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
            raw = json.loads((destination / 'summary.json').read_text()) \
                if (destination / 'summary.json').exists() else None
            status, result = classify(code, raw)
            row.update(status=status, result=result)
            attempt.update(status=status, result=result, finished_unix=time.time())
            save()
            if status == 'infrastructure_failure':
                state['status'] = 'stopped_infrastructure'
                break
        if state['status'] == 'running':
            state['status'] = ('finished' if all(row['status'] == 'completed'
                                                for row in state['episodes'])
                               else 'chunk_complete')
    except BaseException as exc:
        state.update(status='interrupted', error_type=type(exc).__name__)
        raise
    finally:
        save()
        lock.unlink()


if __name__ == '__main__':
    main()
