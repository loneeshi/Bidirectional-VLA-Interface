"""Observation-only diagnosis; never changes actions or success thresholds.

STATUS: active — evaluation
"""
import numpy as np
from bvi.s1_capability_gate import SEEDS


def state_max_errors(reference, actual, prefix=''):
    """Compare every serialized simulator/controller leaf, not only qpos."""
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(reference) != set(actual):
            raise ValueError(f'State keys differ at {prefix}')
        result = {}
        for key in reference:
            result.update(state_max_errors(reference[key], actual[key], f'{prefix}/{key}'))
        return result
    left, right = np.asarray(reference), np.asarray(actual)
    if left.shape != right.shape:
        raise ValueError(f'State shape differs at {prefix}')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f'Nonfinite state at {prefix}')
    return {prefix: float(np.max(np.abs(left.astype(float)-right.astype(float)))) if left.size else 0.}


def validate_episode(result, events):
    """Do not turn an incomplete infrastructure run into a policy failure."""
    if result.get('status') != 'episode_completed':
        raise ValueError('Incomplete native episode')
    if result.get('steps') != len(events) or not events:
        raise ValueError('Native result/event count mismatch')
    if [e.get('step') for e in events] != list(range(1, len(events)+1)):
        raise ValueError('Missing or duplicated action events')


def validate_seed_roster(rows):
    if sorted(row['seed'] for row in rows) != sorted(SEEDS):
        raise ValueError('Exact unique preregistered seed roster required')


def scalar(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError('Single-environment evidence required')
    return value.item()


def native_failure_causes(info):
    if scalar(info.get('success', False)):
        return []
    causes = []
    if 'subtasks_steps_left' in info and scalar(info['subtasks_steps_left']) <= 0:
        causes.append('native_horizon_exhausted')
    if 'cumulative_force_within_limit' in info and not scalar(info['cumulative_force_within_limit']):
        causes.append('native_cumulative_force_limit')
    if scalar(info.get('fail', False)) and not causes:
        causes.append('other_native_failure')
    return causes


def summarize_actions(events):
    if not events:
        raise ValueError('No executed actions')
    # All geometry is diagnostic only, never a policy input or a new success rule.
    distances = []
    for e in events:
        tcp = np.asarray(e['extra']['tcp_pose_wrt_base']).reshape(-1)[:3]
        obj = np.asarray(e['extra']['obj_pose_wrt_base']).reshape(-1)[:3]
        distances.append(float(np.linalg.norm(tcp-obj)))
    raw = np.stack([np.asarray(e['raw_actions'])[0] for e in events])
    applied = np.stack([np.asarray(e['action']) for e in events])
    qpos = np.stack([np.asarray(e['qpos']).reshape(-1) for e in events])
    if raw.shape != (len(events),13) or applied.shape != raw.shape:
        raise ValueError('Expected Fetch13 action evidence')
    closest = int(np.argmin(distances))
    return dict(actions=len(events), minimum_tcp_object_distance_m=distances[closest],
        closest_action=events[closest]['step'], first_post_action_distance_m=distances[0],
        final_distance_m=distances[-1],
        clipped_steps=int(np.any(np.abs(raw)>1,axis=1).sum()),
        clipped_scalars=int((np.abs(raw)>1).sum()),
        clipped_scalars_per_channel=(np.abs(raw)>1).sum(axis=0).tolist(),
        raw_head_abs_max=float(np.abs(raw[:,8:10]).max()),
        mean_finger_qpos_range_m=[float(qpos[:,-2:].mean(axis=1).min()),float(qpos[:,-2:].mean(axis=1).max())],
        base_path_post_action_m=float(np.linalg.norm(np.diff(qpos[:,:2],axis=0),axis=1).sum()),
        final_native_failure_causes=native_failure_causes(events[-1]['info']))
