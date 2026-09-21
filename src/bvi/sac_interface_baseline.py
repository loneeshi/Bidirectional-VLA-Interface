"""Persistent accounting for the 1000-episode TidyHouse SAC-interface baseline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .protocol import ProtocolError


SCHEMA = 'bvi-sac-interface-baseline/1'
TOTAL_ROLLOUTS = 1000
DEFAULT_BATCH_SIZE = 20
TARGET_COVERAGE = 50  # Staged target, not permission to exceed API/resource caps.


def interface_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Local recovery proxies reconstructed from serialized invocation results.

    An intervention is an interrupted call with missed_grasp/grasp_lost feedback.
    Recovery is attempted if the next call repeats that skill and target; success
    requires native success within at most two subsequent matching calls. Normal
    step-limit continuation is not an intervention. This is not Table 4 replication.
    """
    history = summary.get('skill_results')
    keys = ('interventions', 'replans_attempted', 'replans_succeeded')
    if not isinstance(history, list):
        return {**dict.fromkeys(keys), 'interface_metrics_status': 'unavailable_missing_skill_results'}
    interventions = attempted = succeeded = 0
    for index, call in enumerate(history):
        feedback = call.get('feedback', {})
        if feedback.get('status') != 'interrupted' or feedback.get('reason') not in {'missed_grasp', 'grasp_lost'}:
            continue
        interventions += 1
        following = history[index + 1:index + 3]
        for offset, next_call in enumerate(following):
            if (next_call.get('skill'), next_call.get('target_id')) != (call.get('skill'), call.get('target_id')):
                break
            if offset == 0:
                attempted += 1
            if next_call.get('feedback', {}).get('status') == 'succeeded':
                succeeded += 1
                break
            if next_call.get('feedback', {}).get('reason') != 'step_limit':
                break
    return dict(zip(keys, (interventions, attempted, succeeded)),
                interface_metrics_status='local_grasp_recovery_proxy_v1')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_manifest() -> dict[str, Any]:
    return {
        'schema': SCHEMA,
        'task': 'tidy_house',
        'split': 'val',
        'total_rollouts': TOTAL_ROLLOUTS,
        'target_coverage': TARGET_COVERAGE,
        'created_utc': utc_now(),
        'contract': {
            'organizer': 'gpt-5.6-luna',
            'navigate': 'LightNav-0',
            'pick_place': 'official_rl_per_obj_sac',
            'training_updates': 0,
            'max_env_steps': 7000,
            'max_organizer_calls': 40,
            'max_episode_wall_seconds': 900,
            'max_skill_wall_seconds': 180,
            'gpu': 'GPU1_only',
        },
        'episodes': [
            {'seed': seed, 'plan_uid': None, 'status': 'unbound', 'attempts': []}
            for seed in range(TOTAL_ROLLOUTS)
        ],
        'batches': [],
    }


def validate_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get('schema') != SCHEMA or data.get('task') != 'tidy_house' or data.get('split') != 'val':
        raise ProtocolError('Wrong SAC-interface manifest identity')
    if data.get('total_rollouts') != TOTAL_ROLLOUTS:
        raise ProtocolError('The validation manifest must retain all 1000 rollouts')
    rows = data.get('episodes')
    if not isinstance(rows, list) or len(rows) != TOTAL_ROLLOUTS:
        raise ProtocolError('The validation manifest must contain exactly 1000 rows')
    if [row.get('seed') for row in rows] != list(range(TOTAL_ROLLOUTS)):
        raise ProtocolError('Manifest seeds must be ordered 0..999')
    allowed = {'unbound', 'pending', 'running', 'completed', 'infrastructure_failure'}
    for row in rows:
        if row.get('status') not in allowed or not isinstance(row.get('attempts'), list):
            raise ProtocolError('Invalid episode state')
        uid = row.get('plan_uid')
        if row['status'] != 'unbound' and (not isinstance(uid, str) or not uid.strip()):
            raise ProtocolError('Bound episode is missing its real plan UID')
    if not isinstance(data.get('batches'), list):
        raise ProtocolError('Manifest batches must be a list')
    return rows


def bind_row(data: dict[str, Any], seed: int, plan_uid: str, evidence: dict[str, Any]) -> None:
    rows = validate_manifest(data)
    row = rows[seed]
    if row['status'] == 'unbound':
        if not isinstance(plan_uid, str) or not plan_uid.strip():
            raise ProtocolError('Real reset did not provide a plan UID')
        row.update(plan_uid=plan_uid, status='pending', binding=evidence)
    elif row['plan_uid'] != plan_uid:
        raise ProtocolError('Deterministic seed rebound to a different plan UID')


def resume_or_create_batch(data: dict[str, Any], count: int) -> dict[str, Any]:
    rows = validate_manifest(data)
    if not 1 <= count <= DEFAULT_BATCH_SIZE:
        raise ProtocolError(f'Batch size must be within 1..{DEFAULT_BATCH_SIZE}')
    by_seed = {row['seed']: row for row in rows}
    for batch in reversed(data['batches']):
        unfinished = [seed for seed in batch['seeds']
                      if by_seed[seed]['status'] not in {'completed', 'infrastructure_failure'}]
        if unfinished:
            if any(by_seed[seed]['status'] == 'unbound' for seed in unfinished):
                raise ProtocolError('An active batch contains an unbound episode')
            batch['status'] = 'running'
            batch['resumed_utc'] = utc_now()
            return batch
    available = [row['seed'] for row in rows if row['status'] == 'pending'][:count]
    if not available:
        raise ProtocolError('No bound pending episode is available; bind the next batch first')
    batch = {'batch_id': len(data['batches']) + 1, 'seeds': available,
             'created_utc': utc_now(), 'status': 'running'}
    data['batches'].append(batch)
    return batch


def begin_attempt(row: dict[str, Any], directory: str) -> dict[str, Any]:
    if row['status'] not in {'pending', 'running'}:
        raise ProtocolError('Only pending or interrupted-running episodes can start')
    for old in row['attempts']:
        if old.get('status') == 'running':
            old.update(status='interrupted', finished_utc=utc_now(),
                       reason='runner_restarted_without_terminal_record')
    attempt = {'attempt': len(row['attempts']) + 1, 'status': 'running',
               'started_utc': utc_now(), 'directory': directory}
    row['attempts'].append(attempt)
    row['status'] = 'running'
    return attempt


def finish_attempt(row: dict[str, Any], status: str, result: dict[str, Any]) -> None:
    if status not in {'completed', 'infrastructure_failure'} or row['status'] != 'running':
        raise ProtocolError('Invalid terminal episode transition')
    attempt = row['attempts'][-1]
    if attempt.get('status') != 'running':
        raise ProtocolError('No running attempt to finish')
    attempt.update(status=status, finished_utc=utc_now(), result=result)
    row['status'] = status


def summarize(data: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    rows = validate_manifest(data)
    selected = [rows[seed] for seed in batch['seeds']]
    completed = [row for row in selected if row['status'] == 'completed']
    infrastructure = [row for row in selected if row['status'] == 'infrastructure_failure']
    completed_objects = sum(row['attempts'][-1]['result'].get('completed_objects', 0)
                            for row in completed)
    task_successes = sum(bool(row['attempts'][-1]['result'].get('task_success')) for row in completed)
    completed_api_requests = sum(row['attempts'][-1]['result'].get('api_requests', 0) for row in completed)
    api_requests = sum(attempt.get('result', {}).get('api_requests', 0)
                       for row in selected for attempt in row['attempts'])
    budget_rows = [row['seed'] for row in completed
                   if row['attempts'][-1]['result'].get('budget_terminated') is True]
    budget_unknown = [row['seed'] for row in completed
                      if row['attempts'][-1]['result'].get('budget_terminated') is None]
    metric_rows = [row['attempts'][-1]['result'] for row in completed
                   if all(row['attempts'][-1]['result'].get(key) is not None
                          for key in ('interventions', 'replans_attempted', 'replans_succeeded'))]
    metrics = {key: sum(result[key] for result in metric_rows) if metric_rows else None
               for key in ('interventions', 'replans_attempted', 'replans_succeeded')}
    return {
        'schema': SCHEMA,
        'benchmark_result': False,
        'result_scope': 'partial_1000_episode_validation_baseline',
        'batch_id': batch['batch_id'],
        'batch_planned': len(selected),
        'batch_completed': len(completed),
        'batch_infrastructure_failures': len(infrastructure),
        'batch_not_run': [row['seed'] for row in selected
                          if row['status'] not in {'completed', 'infrastructure_failure'}],
        'validation_total': TOTAL_ROLLOUTS,
        'validation_coverage': len(completed) / TOTAL_ROLLOUTS,
        'completed_objects': completed_objects,
        'planned_objects_in_batch': 5 * len(selected),
        'infrastructure_invalid_episodes': len(infrastructure),
        'planned_objects_in_completed_rows': 5 * len(completed),
        'completed_object_rate_completed_rows': (
            completed_objects / (5 * len(completed)) if completed else None),
        'native_task_successes': task_successes,
        'native_task_success_rate_completed_rows': task_successes / len(completed) if completed else None,
        'api_requests_recorded': api_requests,
        **metrics,
        'interface_metrics_observed_episodes': len(metric_rows),
        'intervene_frequency': (sum(result['interventions'] > 0 for result in metric_rows)
                               / len(metric_rows) if metric_rows else None),
        'replan_success_rate': (metrics['replans_succeeded'] / metrics['replans_attempted']
                               if metrics['replans_attempted'] else None),
        'average_calls_per_completed_episode': completed_api_requests / len(completed) if completed else None,
        'api_requests_scope': 'all_recorded_attempts_including_infrastructure_failures',
        'budget_terminated_episode_count': len(budget_rows),
        'budget_terminated_seeds': budget_rows,
        'budget_termination_unknown_seeds': budget_unknown,
        'call_average_interpretation': 'observed_consumption; budget termination right-censors completion demand',
        'api_cost_usd': None,
        'api_cost_status': 'pending_provider_reconciliation',
        'training_updates': 0,
    }
