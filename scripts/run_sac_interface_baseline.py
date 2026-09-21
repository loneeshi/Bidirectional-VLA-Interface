"""Run/resume one bounded batch from the 1000-row TidyHouse manifest.

The simulator and policies run on laboratory GPU1. Model credentials remain on
the local computer; all episodes share one persistent SSH bridge spool so the
existing bridge server can serve the whole batch and retain request claims.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from bvi.sac_interface_baseline import (DEFAULT_BATCH_SIZE, begin_attempt,
    finish_attempt, interface_metrics, resume_or_create_batch, summarize, validate_manifest)


GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'
OFFICIAL_MSHAB = Path('/home/pshuai/bvi-research/src/official-mshab-runtime/mshab')


def runtime_environment(root: Path) -> dict[str, str]:
    """Pin the official 42D runtime ahead of the acdit venv's editable fork."""
    root = root.resolve()
    if not (root / 'mshab/envs/sequential_task.py').is_file():
        raise ValueError('Missing pinned official MS-HAB runtime')
    return dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                PYTHONPATH=os.pathsep.join((str(root), str(Path(__file__).resolve().parents[1] / 'src'))))


def command(python: Path, checkpoint_root: Path, output: Path, bridge: Path,
            authorization_id: str, seed: int, plan_uid: str) -> list[str]:
    script = Path(__file__).with_name('run_coordinator.py')
    return [str(python), str(script), '--benchmark-episode', '--organizer',
        '--tool-family-interface', '--provider', 'openai', '--model', 'gpt-5.6-luna',
        '--transport', 'bridge', '--bridge-dir', str(bridge), '--bridge-timeout-seconds', '300',
        '--authorization-id', authorization_id, '--max-api-cost-usd', '0.05',
        '--request-cost-ceiling-usd', '0.00125', '--max-output-tokens', '2048',
        '--max-input-bytes', '512000',
        '--max-calls', '40', '--max-env-steps', '7000', '--max-wall-seconds', '900',
        '--skill-wall-seconds', '180', '--organizer-slice-steps', '40',
        '--seed', str(seed), '--expected-plan-uid', plan_uid,
        '--checkpoint-root', str(checkpoint_root), '--policy-type', 'rl_per_obj',
        '--navigation-policy', 'lightnav', '--navigation-camera', 'fetch_nav',
        '--workspace-camera', '--manipulation-policy', 'official', '--output', str(output)]


def classify(returncode: int, summary: dict | None, *, model_rejection: bool = False) -> tuple[str, dict]:
    if summary and summary.get('evaluation_eligible') and summary.get('benchmark_episode'):
        reason = summary.get('reason') or ''
        # Historical ProtocolError is ambiguous: prior API calls do not prove
        # this failure came from the model. Require its rejection event.
        protocol_failure = reason == 'error:ModelResponseError' or (
            reason == 'error:ProtocolError' and model_rejection)
        budget_failure = reason == 'error:APIBudgetExhausted'
        infrastructure_failure = reason.startswith(('adapter_error:', 'error:')) and not (
            protocol_failure or budget_failure)
        if not infrastructure_failure and (returncode == 0 or protocol_failure or budget_failure):
            return 'completed', {
                'returncode': returncode,
                **interface_metrics(summary),
                'reason': reason,
                'budget_terminated': budget_failure or reason in {
                    'experiment_call_limit', 'experiment_step_limit', 'experiment_wall_clock_limit',
                    'request_exceeds_remaining_experiment_steps',
                    'request_exceeds_remaining_experiment_wall_budget'},
                'task_success': bool(summary.get('task_success')),
                'completed_objects': int(summary.get('completed_objects', 0)),
                'planned_objects': int(summary.get('planned_objects', 5)),
                'steps': int(summary.get('steps', 0)),
                'api_requests': int(summary.get('api_requests', 0)),
                'invalid_requests': int(protocol_failure),
                'wall_seconds': float(summary.get('wall_seconds', 0)),
                'api_cost_usd': None,
                'api_cost_status': summary.get('api_cost_status', 'pending_reconciliation'),
            }
    return 'infrastructure_failure', {'returncode': returncode,
        'reason': summary.get('reason') if summary else 'missing_summary',
        'api_requests': int(summary.get('api_requests', 0)) if summary else 0,
        'steps': summary.get('steps') if summary else None,
        'wall_seconds': summary.get('wall_seconds') if summary else None,
        'api_cost_usd': None, 'api_cost_status': 'pending_provider_reconciliation'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--authorization-id', required=True)
    parser.add_argument('--mshab-root', type=Path, default=OFFICIAL_MSHAB)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--retry-infrastructure', action='store_true',
                        help='Create a new preserved attempt for this batch\'s infrastructure failures')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    rows = validate_manifest(manifest)
    batch = resume_or_create_batch(manifest, args.batch_size)
    if args.retry_infrastructure:
        for seed in batch['seeds']:
            if rows[seed]['status'] == 'infrastructure_failure':
                rows[seed]['status'] = 'pending'
    jobs = []
    bridge = args.output.resolve() / 'bridge'
    for seed in batch['seeds']:
        row = rows[seed]
        if row['status'] not in {'completed', 'infrastructure_failure'}:
            attempt_number = len(row['attempts']) + 1
            directory = args.output.resolve() / f'seed-{seed:03d}' / f'attempt-{attempt_number:03d}'
            jobs.append({'seed': seed, 'plan_uid': row['plan_uid'], 'output': str(directory),
                         'argv': command(Path(sys.executable), args.checkpoint_root.resolve(),
                                         directory, bridge, args.authorization_id,
                                         seed, row['plan_uid'])})
    if not args.execute:
        print(json.dumps({'status': 'commands_only_no_execution', 'batch': batch,
                          'jobs': jobs}, indent=2))
        return
    if os.name != 'posix':
        raise ValueError('Execution requires the lab Linux process-group timeout handling')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest, manifest)
    env = runtime_environment(args.mshab_root)
    subprocess.run([sys.executable, '-c',
        'import pathlib,sys,mshab.envs.sequential_task as m; '
        'assert pathlib.Path(m.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve(), '
        '"MS-HAB import escaped the pinned runtime"',
        str(args.mshab_root / 'mshab/envs/sequential_task.py')], env=env, check=True)
    consecutive_infra = 0
    try:
        for job in jobs:
            row = rows[job['seed']]
            output = Path(job['output'])
            output.parent.mkdir(parents=True, exist_ok=True)
            attempt = begin_attempt(row, str(output))
            atomic_json(args.manifest, manifest)
            log_path = output.parent / f"attempt-{attempt['attempt']:03d}.log"
            with log_path.open('w', encoding='utf-8') as log:
                child = subprocess.Popen(job['argv'], stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                try:
                    returncode = child.wait(timeout=990)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            summary_path = output / 'summary.json'
            raw = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
            status, result = classify(returncode, raw)
            finish_attempt(row, status, result)
            consecutive_infra = consecutive_infra + 1 if status == 'infrastructure_failure' else 0
            atomic_json(args.manifest, manifest)
            atomic_json(args.output / 'batch-summary.json', summarize(manifest, batch))
            if consecutive_infra >= 3:
                batch['status'] = 'stopped_after_three_consecutive_infrastructure_failures'
                break
        else:
            batch['status'] = 'finished'
    except BaseException as exc:
        batch['status'] = 'interrupted'
        batch['last_error_type'] = type(exc).__name__
        raise
    finally:
        atomic_json(args.manifest, manifest)
        atomic_json(args.output / 'batch-summary.json', summarize(manifest, batch))


if __name__ == '__main__':
    main()
