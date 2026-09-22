"""Export candidate-level pose evidence from named train-spawn geometry.

Missing geometry is a hard error, never inferred from distance intervals. This
does not certify online reachability or generalization to a new scene.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from bvi.pose_recovery import audit_pose_prior


def yaw(pose):
    w, x, y, z = pose[3:]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def vector(value):
    return value[0] if isinstance(value[0], list) else value


def candidate(binding, summary, geometry, source_sha):
    if summary.get('status') != 'completed':
        raise ValueError('Candidate source must be a completed recorded probe')
    if (geometry.get('world_frame') != 'simulator_world' or
            geometry.get('pose_layout') != 'xyz_quaternion_wxyz'):
        raise ValueError('Unverified pose frame/layout')
    base = vector(geometry['base_link_pose'])
    target = vector(geometry['object_pose'] if binding['skill']=='pick' else geometry['goal_pose'])
    theta = yaw(target)
    dx, dy = base[0]-target[0], base[1]-target[1]
    heading = yaw(base)-theta
    groups = geometry['controller_qpos']
    return {'id': source_sha[:24], 'skill': binding['skill'],
            'object_category': binding['category'], 'frame': 'target_se2',
            'units': {'position': 'm', 'angle': 'rad'},
            'base_pose': [math.cos(theta)*dx+math.sin(theta)*dy,
                          -math.sin(theta)*dx+math.cos(theta)*dy,
                          math.atan2(math.sin(heading),math.cos(heading))],
            'arm_qpos': vector(groups['arm']), 'body_qpos': vector(groups['body']),
            'target_height_m': target[2],
            'object_pose_wrt_tcp': (vector(geometry['object_pose_wrt_tcp'])
                                    if binding['skill']=='place' else None),
            'joint_names': geometry['controller_joint_names'],
            'evidence': {'source_sha256': source_sha,
                'checkpoint_sha256': binding['checkpoint_sha'],
                'parent_uid': binding['uid'], 'scene': binding['scene'],
                'spawn_index': binding['spawn_index'], 'samples': 1,
                'successes': int(summary['ever_native_success']),
                'force_violation': summary['force_violation'],
                'final_native_success': summary['final_native_success']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.probe_root.glob('case-*/attempt-*/summary.json')):
        files = [path, path.with_name('binding.json'), path.with_name('pose-geometry.json')]
        raw = [p.read_bytes() for p in files]  # Missing geometry blocks export.
        summary, binding, geometry = [json.loads(b) for b in raw]
        records.append(candidate(binding, summary, geometry,
                                 hashlib.sha256(b'\n'.join(raw)).hexdigest()))
    data = {'schema': 'spawn-prior/2', 'candidates': records,
            'training_plan_uids': sorted({c['evidence']['parent_uid'] for c in records}),
            'limitations': ['One recorded outcome per candidate, not a calibrated probability.',
                'Requires live scene/checkpoint/grasp and collision-path validation.',
                'Failed source candidates are retained; no cross-scene optimum claim.']}
    audit = audit_pose_prior(data)
    if not audit['ready']:
        raise ValueError(str(audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
