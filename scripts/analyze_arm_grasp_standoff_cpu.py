"""CPU-only grasp standoff analysis for the 2026-10-01 arm capability runs.

For every set_gripper(0) call, measure how far the commanded gripper_link origin
stopped short of the object surface point Astra itself had located, along the
gripper's local +x (the direction through the fingers). Reads archived
result.json files only; no simulator, GPU or API.

The reference point is the latest locate_point result before the closure, taken
after the last move_base (base motion changes the frame). Points more than 5 cm
from the true target surface are excluded as target references, except when the
model never located the true target (case r1 arm-dev-003), where its own
mislocated object point is used and flagged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RUNS = (
    ('r3-fixed-base', 'research/c2/diagnostics/2026-10-01-arm-capability-development-r3', 'V'),
    ('r1-mobile-base', 'research/c2/diagnostics/2026-10-01-arm-mobile-five-r1', 'V-mobile'),
)


def closures(result: dict) -> list[dict]:
    frames = {f['step']: f for f in result['frames']}
    loc = {item['turn']: item for item in result.get('localization_eval_only', [])}
    goal = target = fallback = None
    out = []
    for command in result['commands']:
        call = command.get('call') or {}
        res = command.get('result') or {}
        tool = call.get('tool')
        if tool == 'move_base':
            target = fallback = None
        if tool == 'locate_point' and res.get('point_base_m') is not None:
            point = np.asarray(res['point_base_m'], dtype=float)
            fallback = point
            if command['turn'] in loc and loc[command['turn']]['surface_distance_m'] < 0.05:
                target = point
        if tool in ('move_to', 'move_eef_chunk') and call.get('poses'):
            goal = call['poses'][-1]
        if tool == 'set_gripper' and call.get('gripper') == 0:
            reference, flag = (target, 'true_target') if target is not None else (fallback, 'mislocated_object')
            end = command.get('start_step', 0) + (res.get('steps_used') or 0)
            frame = frames.get(end, {})
            row = {'turn': command['turn'], 'step': end, 'reason': res.get('reason'),
                   'stable': res.get('stable'), 'reference': flag if reference is not None else None,
                   'finger_distance_m': frame.get('finger_distance_m'),
                   'is_grasped_target': frame.get('is_grasped'),
                   'tcp_box_distance_m': frame.get('tcp_box_distance_m')}
            if goal is not None and reference is not None:
                x_axis = Rotation.from_quat(goal['quaternion_xyzw']).as_matrix()[:, 0]
                v = reference - np.asarray(goal['position'], dtype=float)
                along = float(v @ x_axis)
                row['standoff_along_x_m'] = along
                row['lateral_m'] = float(np.linalg.norm(v - along * x_axis))
            out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'bvi.arm-grasp-standoff/1', 'cpu_only': True, 'simulator_actions': 0,
              'api_requests': 0, 'definition': 'standoff = (located surface point - commanded '
              'gripper_link origin) projected on gripper local +x; positive means the origin '
              'stopped short of the surface', 'sources': {}, 'closures': []}
    for run, root, condition in RUNS:
        for path in sorted(Path(root).glob(f'results/arm-dev-*/{condition}/result.json')):
            raw = path.read_bytes()
            report['sources'][path.as_posix()] = hashlib.sha256(raw).hexdigest()
            result = json.loads(raw)
            for row in closures(result):
                report['closures'].append({'run': run, 'case': result['case_id'], **row})
    closed = [c for c in report['closures'] if c['reason'] == 'ok']
    near = [c for c in closed if c.get('standoff_along_x_m') is not None and c['standoff_along_x_m'] <= 0.03]
    far = [c for c in closed if c.get('standoff_along_x_m') is not None and c['standoff_along_x_m'] > 0.03]
    report['summary'] = {
        'close_commands': len(report['closures']),
        'executed_closures': len(closed),
        'alignment_refused': len(report['closures']) - len(closed),
        'near_le_3cm': {'n': len(near), 'grasped_any_object': sum((c['finger_distance_m'] or 0) > 0.01 for c in near),
                        'range_m': [min(c['standoff_along_x_m'] for c in near), max(c['standoff_along_x_m'] for c in near)] if near else None},
        'far_gt_3cm': {'n': len(far), 'grasped_any_object': sum((c['finger_distance_m'] or 0) > 0.01 for c in far),
                       'range_m': [min(c['standoff_along_x_m'] for c in far), max(c['standoff_along_x_m'] for c in far)] if far else None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
