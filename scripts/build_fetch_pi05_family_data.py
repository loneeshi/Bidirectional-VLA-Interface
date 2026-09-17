"""CPU-only V8-workspace teacher -> invocation dataset; no model/simulator.

Input JSON: {"episodes": [{"directory": "...", "split": "train|validation",
"parent_episode": {"scene_split":"train", "task":"pick|place", "seed":3000}}]}.
Parent split assignments must be supplied, never inferred or reshuffled here.
Every invocation NPZ includes its verified terminal observation and independent
action/progress validity. State/action normalization uses the supplied V8 stats.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from bvi.fetch_current_labels import CONTRACT, validate_manifest_splits, verified_teacher_targets
from bvi.fetch_segments import segment_episode

FAMILIES = ('reach', 'grasp', 'move', 'release')
ACTION_CONVENTION = 'Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity'
V8_JOINT_NAMES = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint',
    'torso_lift_joint', 'head_pan_joint', 'shoulder_pan_joint', 'head_tilt_joint',
    'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint',
    'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint',
    'r_gripper_finger_joint', 'l_gripper_finger_joint']
V8_CAMERA = dict(uid='fetch_workspace', p=[.35, .35, 1.5],
    q=[math.cos(-.35 / 2) * math.cos(.65 / 2), -math.sin(-.35 / 2) * math.sin(.65 / 2),
       math.cos(-.35 / 2) * math.sin(.65 / 2), math.sin(-.35 / 2) * math.cos(.65 / 2)],
    width=224, height=224, fov=math.pi / 2, near=.01, far=100, mount='agent.base_link')


def validate_recorded_contract(report, row):
    if report['split'] != row['split']:
        raise ValueError('Recorded split cannot be relabeled by input manifest')
    if report['joint_names'] != V8_JOINT_NAMES:
        raise ValueError('Joint order is not the declared V8 order')
    camera = report['camera']
    for key, expected in V8_CAMERA.items():
        actual = camera.get(key)
        if isinstance(expected, (str, int)):
            valid = actual == expected
        else:
            valid = np.asarray(actual).shape == np.asarray(expected).shape and np.allclose(actual, expected, rtol=0, atol=1e-12)
        if not valid:
            raise ValueError(f'Camera is not the declared V8 configuration: {key}')


def fixed_parent_coverage(rows):
    actual = {(r['parent_episode']['scene_split'], r['parent_episode']['task'],
               r['parent_episode']['seed'], r['split']) for r in rows}
    expected = {('train', task, seed, 'train' if seed < 3020 else 'validation')
                for task in ['pick', 'place'] for seed in range(3000, 3025)}
    return len(rows) == 50 and actual == expected


def verify_source_collection(source, rows):
    declared = source.get('source_collection')
    if declared is None:
        return dict(status='not_supplied', complete=False)
    path = Path(declared['path']).resolve()
    collection = json.loads(path.read_text())
    if declared.get('status', collection['status']) != collection['status']:
        raise ValueError('Declared source collection status differs from evidence')
    result = dict(path=str(path), sha256=sha(path), status=collection['status'], complete=False)
    if collection['status'] != 'completed':
        return result
    collected = collection['episodes']
    by_parent = {(e['task'], e['seed']): e for e in collected}
    if len(collected) != 50 or len(by_parent) != 50 or not fixed_parent_coverage(rows):
        raise ValueError('Completed source collection requires the exact fixed50 parent set')
    for row in rows:
        parent = row['parent_episode']; evidence = by_parent.get((parent['task'], parent['seed']))
        directory = Path(row['directory']).resolve()
        if evidence is None or evidence['status'] != 'completed' or evidence['split'] != row['split'] or Path(evidence['directory']).resolve() != directory:
            raise ValueError('Source collection and dataset parent mapping differ')
        if evidence['result_sha256'] != sha(directory / 'result.json'):
            raise ValueError('Source collection result evidence changed')
    result['complete'] = True
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_stats(path):
    stats = json.loads(Path(path).read_text())['norm_stats']
    result = {}
    for key, dim in [('state', 30), ('actions', 13)]:
        low = np.asarray(stats[key]['q01'], np.float32)
        high = np.asarray(stats[key]['q99'], np.float32)
        if low.shape != (dim,) or high.shape != (dim,) or not np.isfinite([low, high]).all() or np.any(high < low):
            raise ValueError(f'Invalid V8 quantiles: {key}')
        result[key] = (low, high)
    return result


def normalize(values, stats):
    low, high = stats
    # Same formula as author Normalize(use_quantiles=True); intentionally unclipped.
    return (2 * (values - low) / (high - low + 1e-6) - 1).astype(np.float32)


def invocation_arrays(episode, window, stats):
    start, end = window['start'], window['end']
    if not 0 <= start < end < len(episode['state']):
        raise ValueError('Window is outside inclusive recorded observations')
    times = np.arange(start, end + 1)
    targets = [verified_teacher_targets(window, int(t), horizon=10) for t in times]
    actions = np.zeros((len(times), 13), np.float32)
    actions[:-1] = episode['actions'][start:end]
    normalized_actions = np.zeros((len(times), 32), np.float32)
    normalized_actions[:-1, :13] = normalize(actions[:-1], stats['actions'])
    return dict(workspace_rgb=episode['workspace_rgb'][times], wrist_rgb=episode['wrist_rgb'][times],
                state=episode['state'][times], actions=actions,
                normalized_state=normalize(episode['state'][times], stats['state']),
                normalized_actions=normalized_actions,
                observation_index=times, progress=np.asarray([t['progress_target'][0] for t in targets], np.float32),
                action_valid=np.asarray([t['action_valid'][0] for t in targets], bool),
                progress_valid=np.asarray([t['progress_valid'][0] for t in targets], bool),
                chunk_progress=np.asarray([t['progress_target'] for t in targets], np.float32),
                chunk_action_valid=np.asarray([t['action_valid'] for t in targets], bool),
                chunk_progress_valid=np.asarray([t['progress_valid'] for t in targets], bool))


def read_episode(row):
    import h5py  # CPU dependency only; OpenPI/JAX/Torch are deliberately absent.
    directory = Path(row['directory']).resolve()
    report = json.loads((directory / 'result.json').read_text())
    validate_recorded_contract(report, row)
    parent = row['parent_episode']
    if report['status'] != 'completed':
        raise ValueError('Incomplete raw trajectory must be retained but cannot enter this dataset')
    if (report['task'], report['seed'], report.get('scene_split', report['split'])) != (parent['task'], parent['seed'], parent['scene_split']):
        raise ValueError('Recorded parent identity differs from declared manifest')
    n = report['steps']
    if not 1 <= n <= 200:
        raise ValueError('Episode outside bounded 1..200 action contract')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    if [e['step'] for e in events] != list(range(1, n + 1)):
        raise ValueError('Events must cover every applied action exactly once')
    success = [e['step'] for e in events if e['info']['success'][0]]
    if bool(events[-1]['info']['success'][0]) != bool(report['success']):
        raise ValueError('Native final success differs between evidence files')
    arrays = {k: [] for k in ('state', 'workspace_rgb', 'wrist_rgb', 'actions')}
    grasp, distance, goal = [], [], []
    with h5py.File(directory / 'trajectory.h5', 'r') as h:
        if h.attrs['status'] != 'completed' or bool(h.attrs['native_success']) != bool(report['success']):
            raise ValueError('H5 completion metadata mismatch')
        if h.attrs['controller_action_convention'] != ACTION_CONVENTION:
            raise ValueError('Controller action convention mismatch')
        camera = json.loads(h.attrs['camera_contract'])
        if camera != report['camera'] or camera.get('uid') != 'fetch_workspace' or camera.get('width') != 224 or camera.get('height') != 224:
            raise ValueError('Workspace camera provenance mismatch')
        if list(h['observations']) != [f'{t:04d}' for t in range(n + 1)] or list(h['actions']) != [f'{t:04d}' for t in range(n)]:
            raise ValueError('Missing/noncontiguous observation or action')
        origin = np.asarray(h.attrs['original_subtask_start_base_xy'], np.float32)
        if origin.shape != (2,) or not np.array_equal(origin, np.asarray(report['original_subtask_start_base_xy'], np.float32)):
            raise ValueError('Original subtask XY origin mismatch')
        if json.loads(h.attrs['joint_names']) != V8_JOINT_NAMES:
            raise ValueError('Fetch joint order mismatch')
        for t in range(n + 1):
            g = h[f'observations/{t:04d}']
            # Never use agent/qpos: the native SAC observation contains only12 joints.
            qpos, qvel = g['recorded_robot/qpos'][0], g['recorded_robot/qvel'][0]
            qpos = qpos.copy(); qpos[:2] -= origin
            expected = np.r_[qpos, qvel].astype(np.float32)
            state = g['recorded_robot/v8_state30'][:]
            if state.shape != (30,) or not np.array_equal(state, expected) or not np.isfinite(state).all():
                raise ValueError('Stored V8 state differs from original-start robot qpos/qvel')
            arrays['state'].append(state)
            for key, camera, shape in [('workspace_rgb', 'fetch_workspace', (224, 224, 3)),
                                       ('wrist_rgb', 'fetch_hand', (128, 128, 3))]:
                if f'sensor_data/{camera}/rgb' not in g:
                    raise ValueError(f'Missing required V8 camera: {camera}; old AC H5 is not compatible')
                image = g[f'sensor_data/{camera}/rgb'][0]
                if image.shape != shape or image.dtype != np.uint8:
                    raise ValueError(f'Invalid RGB contract: {camera}')
                arrays[key].append(image)
            extra = g['extra']
            grasp.append(bool(extra['is_grasped'][0]))
            distance.append(float(np.linalg.norm(extra['tcp_pose_wrt_base'][0, :3] - extra['obj_pose_wrt_base'][0, :3])))
            goal.append(float(np.linalg.norm(extra['goal_pos_wrt_base'][0] - extra['obj_pose_wrt_base'][0, :3])))
            if t < n:
                action = h[f'actions/{t:04d}'][0]
                if action.shape != (13,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1) or np.any(action[8:10] != 0):
                    raise ValueError('Invalid applied Fetch controller action')
                if not np.array_equal(action, np.asarray(events[t]['action'][0], np.float32)):
                    raise ValueError('H5/actions do not match applied event controls')
                arrays['actions'].append(action)
    windows = segment_episode(parent['task'], grasp, distance, goal, success)
    return {k: np.stack(v) for k, v in arrays.items()}, windows, report


def build(manifest_path, stats_path, output, max_episodes=100, max_seconds=300):
    started = time.monotonic()
    source = json.loads(Path(manifest_path).read_text())
    rows = source['episodes']
    if not 1 <= len(rows) <= max_episodes:
        raise ValueError('Episode count exceeds declared bound')
    validate_manifest_splits(rows)
    identities = [json.dumps(r['parent_episode'], sort_keys=True) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate original trajectory identity')
    stats = load_stats(stats_path)
    collection = verify_source_collection(source, rows)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    result = dict(status='building', schema='fetch_pi05_family_v1', label_contract=CONTRACT,
        scope='native-start SAC teacher; actual navigation handoffs not covered',
        action_horizon=10, internal_action_dim=32, external_action_dim=13,
        state_contract=dict(robot='fetch', state_dim=30, state_components=['qpos', 'qvel'],
            base_position_reference='skill_start_xy', base_camera='fetch_workspace', wrist_camera='fetch_hand',
            state_conditioning=True, action_dim=13, action_convention=ACTION_CONVENTION,
            tokenization='normalize raw30 then tokenize discrete state30 before padding state to32'),
        progress_semantics='local current-observation elapsed fraction to independently verified boundary; not physical completion probability',
        normalizer_sha256=sha(stats_path), input_manifest_sha256=sha(manifest_path),
        normalizer_path=str(Path(stats_path).resolve()), invocations=[], episodes=[])
    result.update(source_collection=collection, source_collection_complete=collection['complete'],
                  fixed_collection_complete=fixed_parent_coverage(rows), training_ready=False,
                  parent_counts={task: {split: sum(r['parent_episode']['task'] == task and r['split'] == split for r in rows)
                                       for split in ['train', 'validation']} for task in ['pick', 'place']})
    try:
        for ei, row in enumerate(rows):
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError('CPU dataset-building wall bound exceeded')
            episode, windows, report = read_episode(row)
            directory = Path(row['directory']).resolve()
            evidence = {name: sha(directory / name) for name in ['trajectory.h5', 'events.jsonl', 'result.json']}
            result['episodes'].append(dict(**row, native_success=report['success'], windows=len(windows), source_sha256=evidence))
            for wi, window in enumerate(windows):
                arrays = invocation_arrays(episode, window, stats)
                path = output / f'episode-{ei:03d}-window-{wi:02d}-{window["family"]}.npz'
                np.savez_compressed(path, **arrays)
                result['invocations'].append(dict(**window, path=str(path), split=row['split'],
                    parent_episode=row['parent_episode'], completion_verified=True,
                    sample_count=len(arrays['state']), sha256=sha(path)))
        result['counts'] = {split: {f: sum(w['split'] == split and w['family'] == f for w in result['invocations'])
                                 for f in FAMILIES} for split in ['train', 'validation']}
        result['all_families_present'] = all(n > 0 for c in result['counts'].values() for n in c.values())
        result['training_ready'] = result['all_families_present'] and result['fixed_collection_complete'] and result['source_collection_complete']
        result['status'] = 'built_all_families' if result['training_ready'] else 'built_not_training_ready'
    except Exception as exc:
        result.update(status='failed', error=repr(exc)); raise
    finally:
        result['wall_seconds'] = time.monotonic() - started
        (output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--norm-stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-episodes', type=int, default=100)
    p.add_argument('--max-seconds', type=int, default=300)
    a = p.parse_args()
    if not 1 <= a.max_episodes <= 100 or not 1 <= a.max_seconds <= 600:
        p.error('Bounds: episodes1..100, seconds1..600')
    result = build(a.manifest, a.norm_stats, a.output, a.max_episodes, a.max_seconds)
    print(json.dumps({k: v for k, v in result.items() if k not in ['invocations', 'episodes']}, indent=2))


if __name__ == '__main__':
    main()
