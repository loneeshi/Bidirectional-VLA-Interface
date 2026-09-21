"""CPU read-only evidence summary of an ended oracle panel; no success redefinition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from bvi.native_pick_audit import validate_episode, scalar


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def optional_scalar(mapping, key):
    if key not in mapping or mapping[key] is None: return None
    try:
        value = scalar(mapping[key])
        return value if isinstance(value, (bool, int, float)) and np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def geometry(extra):
    try:
        tcp = np.asarray(extra['tcp_pose_wrt_base'], dtype=float).reshape(-1)[:3]
        obj = np.asarray(extra['obj_pose_wrt_base'], dtype=float).reshape(-1)[:3]
        if tcp.shape != (3,) or obj.shape != (3,) or not np.isfinite(np.r_[tcp, obj]).all(): return None
        return tcp, obj, float(np.linalg.norm(tcp - obj))
    except (KeyError, TypeError, ValueError):
        return None


def displacement_windows(initial_extra, events):
    initial = geometry(initial_extra) if isinstance(initial_extra, dict) else None
    by_step = {e['step']: e for e in events}
    windows = {}
    for step in (1, 5, 10, 20):
        row = dict(available=False, frame='robot_base', tcp_displacement_vector_m=None,
                   tcp_displacement_norm_m=None, object_displacement_vector_m=None,
                   object_displacement_norm_m=None, distance_change_m=None)
        if initial is None:
            row['reason'] = 'initial_extra_missing_or_invalid_no_step0_reconstruction'
        elif step not in by_step:
            row['reason'] = 'episode_did_not_reach_requested_step'
        else:
            current = geometry(by_step[step].get('extra', {}))
            if current is None:
                row['reason'] = 'post_action_geometry_missing_or_invalid'
            else:
                tcp, obj = current[0] - initial[0], current[1] - initial[1]
                row.update(available=True, tcp_displacement_vector_m=tcp.tolist(),
                           tcp_displacement_norm_m=float(np.linalg.norm(tcp)),
                           object_displacement_vector_m=obj.tolist(), object_displacement_norm_m=float(np.linalg.norm(obj)),
                           distance_change_m=current[2] - initial[2])
        windows[str(step)] = row
    return windows


def summarize_case(root, case, expected_protocol=None):
    seed = case['seed']; directory = root / f'seed{seed}'
    row = dict(seed=seed, status=case.get('status'), native_success=None, policy_failure=False,
               ever_grasped=None, steps=None, evidence={}, video={'path': None, 'sha256': None},
               unavailable=[], final_native_info={k: None for k in (
                   'success', 'fail', 'ee_rest', 'robot_rest', 'is_static', 'robot_cumulative_force',
                   'cumulative_force_within_limit')})
    result_path, events_path = directory / 'result.json', directory / 'events.jsonl'
    for name, path in (('result', result_path), ('events', events_path)):
        if path.is_file(): row['evidence'][name] = dict(path=str(path.resolve()), sha256=digest(path))
    if case.get('status') == 'not_run': return row
    if not result_path.is_file():
        row.update(status='infrastructure_failure', evidence_error='result_missing'); return row
    try:
        result = read(result_path)
        if case.get('result_sha256') is not None and digest(result_path) != case['result_sha256']:
            raise ValueError('Panel-bound result hash changed')
        video = result.get('video')
        if isinstance(video, str):
            video_path = Path(video)
            if not video_path.is_absolute(): video_path = directory / video_path
            video_path = video_path.resolve()
            row['video']['path'] = str(video_path)
            if video_path.is_file():
                row['video']['sha256'] = digest(video_path)
                expected_video = result.get('artifact_sha256', {}).get(video_path.name)
                if expected_video is not None and row['video']['sha256'] != expected_video:
                    raise ValueError('Result-bound video hash changed')
            else: row['unavailable'].append('video_file_missing')
        else:
            row['unavailable'].append('video_filename_not_recorded')
        if case.get('status') == 'infrastructure_failure' or result.get('status') != 'episode_completed':
            row.update(status='infrastructure_failure', episode_status=result.get('status')); return row
        events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        validate_episode(result, events)
        if result.get('seed') != seed or type(result.get('success')) is not bool:
            raise ValueError('Episode seed/native-success contract mismatch')
        if expected_protocol is not None and result.get('instruction_protocol') != expected_protocol:
            raise ValueError('Episode instruction protocol differs from launch manifest')
        if case.get('native_success') is not result['success']:
            raise ValueError('Panel/episode native-success mismatch')
        expected_events_hash = result.get('artifact_sha256', {}).get('events.jsonl')
        if expected_events_hash is not None and expected_events_hash != digest(events_path):
            raise ValueError('Result-bound event hash changed')
        final_info = events[-1].get('info', {})
        final_success = optional_scalar(final_info, 'success')
        if final_success is not None and bool(final_success) != result['success']:
            raise ValueError('Final native success differs from episode result')
        row.update(status='success' if result['success'] else 'policy_failure', native_success=result['success'],
                   policy_failure=not result['success'], steps=len(events),
                   instruction_protocol=result.get('instruction_protocol'),
                   native_failure_causes=result.get('native_failure_causes', []),
                   stop_reason=result.get('stop_reason'),
                   final_native_info={k: optional_scalar(final_info, k) for k in row['final_native_info']})
        held = [optional_scalar(e.get('extra', {}), 'is_grasped') for e in events]
        row['ever_grasped'] = True if any(v is True for v in held) else False if all(v is False for v in held) else None
        if (result.get('ever_grasped') is not None and row['ever_grasped'] is not None
                and result['ever_grasped'] is not row['ever_grasped']):
            raise ValueError('Result/event ever-grasped evidence differs')
        distances = [geometry(e.get('extra', {})) for e in events]
        known = [(i + 1, g[2]) for i, g in enumerate(distances) if g is not None]
        row['distance_m'] = dict(first_post_action=None if distances[0] is None else distances[0][2],
            minimum=min((d for _, d in known), default=None), final=None if distances[-1] is None else distances[-1][2],
            observed_steps=len(known), total_steps=len(events),
            closest_step=min(known, key=lambda x: x[1])[0] if known else None)
        row['displacements_from_step0'] = displacement_windows(result.get('initial_extra'), events)
        row['displacement_interpretation'] = 'Endpoint changes relative to each observation robot base; not world-frame travel or causal effect'
        raw = []
        for e in events:
            action = np.asarray(e.get('raw_actions', []), dtype=float)
            if action.shape != (10, 13) or not np.isfinite(action).all(): break
            raw.append(action[0])
        row['clipping'] = dict(steps=None, scalars=None, scalars_per_channel=None)
        if len(raw) == len(events):
            clipped = np.abs(np.asarray(raw)) > 1
            row['clipping'] = dict(steps=int(clipped.any(axis=1).sum()), scalars=int(clipped.sum()),
                                   scalars_per_channel=clipped.sum(axis=0).tolist(), scope='executed_chunk_index0')
        else: row['unavailable'].append('raw_chunk_evidence_incomplete')
        counts = Counter(e.get('instruction_family') or 'unavailable' for e in events)
        row['family_action_counts'] = dict(counts)
        switches = [e['prompt_switch'] for e in events if e.get('prompt_switch') is not None]
        row['prompt_switches'] = switches
        row['prompt_switch_evidence_available'] = all('prompt_switch' in e for e in events)
        if result.get('prompt_switches') is not None and result['prompt_switches'] != switches:
            raise ValueError('Result/event prompt switches differ')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        row.update(status='evidence_invalid', native_success=None, policy_failure=False, evidence_error=str(exc))
    return row


def summarize(panel_dir):
    root = Path(panel_dir).resolve(); panel_path = root / 'panel.json'; panel = read(panel_path)
    if not panel.get('supervisor_outcome') or any(c.get('status') in ('running', 'starting') for c in panel['cases']):
        raise ValueError('Panel has not ended; refuse to summarize an active run')
    seeds = [c['seed'] for c in panel['cases']]
    if len(set(seeds)) != len(seeds) or panel.get('planned_count') != len(seeds):
        raise ValueError('Invalid planned seed roster')
    if any(c.get('status') not in ('success', 'policy_failure', 'infrastructure_failure', 'not_run') for c in panel['cases']):
        raise ValueError('Unrecognized final episode status')
    launch_path = root / 'launch.json'; launch = read(launch_path)
    if seeds != launch.get('execution_seeds', launch.get('seeds')):
        raise ValueError('Panel roster differs from launch manifest')
    cases = [summarize_case(root, case, launch.get('protocol')) for case in panel['cases']]
    return dict(schema='bvi.s1-oracle-panel-evidence-summary/1', panel_dir=str(root),
        interpretation='Single-arm evidence for later paired comparison; no significance test or automatic G1 admission',
        variant=launch.get('variant'), checkpoint=launch.get('checkpoint'), protocol=launch.get('protocol'),
        pairing_identity={k: launch.get(k) for k in ('seeds', 'execution_seeds', 'shader', 'sim_backend',
                          'max_actions', 'reference_sha256', 'normalizer_sha256', 'source_sha256')},
        panel_status=panel.get('status'), supervisor_outcome=panel['supervisor_outcome'],
        evidence={'panel': {'path': str(panel_path), 'sha256': digest(panel_path)},
                  'launch': {'path': str(launch_path), 'sha256': digest(launch_path)}},
        planned_count=len(cases), completed_count=sum(c['native_success'] is not None for c in cases),
        native_successes=sum(c['native_success'] is True for c in cases),
        policy_failures=sum(c['policy_failure'] for c in cases),
        infrastructure_failures=sum(c['status'] == 'infrastructure_failure' for c in cases),
        evidence_invalid=sum(c['status'] == 'evidence_invalid' for c in cases),
        not_run=sum(c['status'] == 'not_run' for c in cases), cases=cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('Output exists; preserve existing evidence')
    report = summarize(args.panel_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('planned_count', 'completed_count', 'native_successes', 'policy_failures',
                                           'infrastructure_failures', 'evidence_invalid', 'not_run')}))


if __name__ == '__main__': main()
