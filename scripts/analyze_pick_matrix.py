"""Read-only attribution from archived Pick matrix logs; no simulator or API.

Distances are TCP-origin to object-origin distances, not surface clearances.
Contact forces are samples at control-step boundaries, not substep histories.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ('template-stop', 'gpt-stop', 'template-observe', 'gpt-observe', 'sac-reference')


def vector(d, key):
    a = np.asarray(d[key], dtype=float).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite diagnostic field: {key}')
    return a


def analyze(folder):
    source = folder / 'events.jsonl'
    events = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    contacts = [e for e in events if e['event'] == 'contact_diagnostic']
    if not contacts:
        raise ValueError(f'Missing contact records: {folder}')
    steps = {int(e['frame_id'].rsplit('-', 1)[1]): e for e in events if e['event'] == 'mshab_step'}
    predictions = {e['frame_id']: e for e in events if e['event'] == 'fetch_pi_inference_started'}
    actions = [e for e in events if e['event'] == 'fetch_pi_action']
    config = next(e for e in events if e['event'] == 'diagnostic_config')
    summary = json.loads((folder / 'summary.json').read_text())
    initial = contacts[0]['before']
    origin = vector(initial, 'qpos')[:2]
    trace, max_state_error, max_action_error = [], 0., 0.
    for i, event in enumerate(contacts, 1):
        before, after = event['before'], event['after']
        qpos, action = vector(after, 'qpos'), np.asarray(event['action'])
        step = steps[event['step']]
        info = step['info']
        def scalar(key):
            return np.asarray(info[key]).item()
        trace.append(dict(case=folder.name, pick_step=i, env_step=event['step'],
            tcp_object_m=float(np.linalg.norm(vector(after, 'tcp_pose')[:3]-vector(after, 'object_pose')[:3])),
            grip_command=float(action[7]), finger_qpos_sum_m=float(qpos[-2:].sum()),
            left_force_norm=float(np.linalg.norm(vector(after, 'left_force'))),
            right_force_norm=float(np.linalg.norm(vector(after, 'right_force'))),
            object_displacement_m=float(np.linalg.norm(vector(after, 'object_pose')[:3]-vector(initial, 'object_pose')[:3])),
            base_x=float(qpos[0]), base_y=float(qpos[1]), base_yaw=float(qpos[2]),
            grasp30=bool(np.asarray(after['grasp30']).item()),
            native_grasp=bool(scalar('is_grasped')), native_fail=bool(scalar('fail')),
            cumulative_force_within_limit=bool(scalar('cumulative_force_within_limit')),
            robot_cumulative_force=float(scalar('robot_cumulative_force')),
            robot_force=float(scalar('robot_force')),
            subtask_steps_left=int(scalar('subtasks_steps_left'))))
        # Compare the client input against the measured PRE-action state.
        frame = f"seed-1-step-{event['step']-1}"
        if predictions:
            expected = np.concatenate([vector(before, 'qpos'), vector(before, 'qvel')])
            expected[:2] -= origin
            max_state_error = max(max_state_error, float(np.max(np.abs(expected-np.asarray(predictions[frame]['state'])))))
        controller = np.asarray(step['controller_action']).reshape(-1)
        expected_action = action.copy()
        expected_action[8:10] = 0.  # Declared official stationary-head wrapper.
        max_action_error = max(max_action_error, float(np.max(np.abs(controller-expected_action))))
    def first(predicate):
        return next((r for r in trace if predicate(r)), None)
    raw = np.array([a['raw'] for a in actions])
    result = dict(case=folder.name, events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prompt=config['prompt'], interrupt=config['interrupt'], outcome=summary['reason'],
        steps=len(trace), start_equality=summary['start_errors'],
        closest=min(trace, key=lambda r:r['tcp_object_m']),
        first_close=first(lambda r:r['grip_command'] < -.5),
        first_sampled_contact=first(lambda r:max(r['left_force_norm'],r['right_force_norm']) > 1e-6),
        first_grasp=first(lambda r:r['native_grasp']), final=trace[-1],
        max_object_displacement_m=max(r['object_displacement_m'] for r in trace),
        max_client_state_error=max_state_error if predictions else None,
        max_wrapper_action_error=max_action_error,
        clipped_action_rows=int(np.any(np.abs(raw)>1, axis=1).sum()) if actions else None,
        clipped_by_channel=(np.abs(raw)>1).sum(axis=0).tolist() if actions else None,
        raw_action_min=raw.min(axis=0).tolist() if actions else None,
        raw_action_max=raw.max(axis=0).tolist() if actions else None,
        initial_action=contacts[0]['action'])
    return result, trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results, traces = [], []
    for name in CASES:
        result, trace = analyze(args.input/name)
        results.append(result)
        traces.extend(trace)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'contact-summary.json').write_text(json.dumps(results, indent=2)+'\n')
    with (args.output/'contact-trace.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(traces[0]))
        w.writeheader()
        w.writerows(traces)
    for r in results:
        print(r['case'], 'min_distance_m=', round(r['closest']['tcp_object_m'], 4),
              'state_error=', r['max_client_state_error'], 'action_error=', r['max_wrapper_action_error'])


if __name__ == '__main__':
    main()
