"""Offline motion statistics, never evidence of a repaired controller or new run."""
import numpy as np


def preclose_metrics(history, belief_events):
    """Process metrics only. Never infer task success or identity correctness."""
    decisions=[row for row in history if row.get('action')]
    refinements=[e for e in belief_events if e['kind']=='pose_update']
    measured=[e for e in belief_events if e['kind']=='detection']
    checks=[e for e in belief_events if e['kind']=='verification']
    motions=[row for row in decisions if row['action']['tool'] in ('move_to','move_eef_chunk')]
    return {'refine_calls':sum(row['action']['tool']=='update_target_pose' for row in decisions),
        'refine_checked':sum(e['pose']['status']=='checked' for e in refinements),
        'refine_multiview_checked':sum(e['pose']['status']=='checked' and e['pose'].get('validation_scope')=='held_out_multiview' for e in refinements),
        'explicit_intent_motion_commands':sum(bool(row['action'].get('grasp_intent')) and row['result']['accepted'] for row in motions),
        'association_ambiguous':sum(e['view'].association=='uncertain' for e in measured),
        'confirmed_target_movements':sum(bool(e.get('moved')) for e in measured),
        'identity_checks':sum(e['value'].get('purpose')=='identity' for e in checks),
        'preclose_checks':sum(e['value'].get('purpose')=='preclose' for e in checks),
        'denied_actions':sum(not row['result']['accepted'] for row in history),
        'denial_reasons':[row['result']['reason'] for row in history if not row['result']['accepted']],
        'grasp_evidence_self_reports':[e['value'] for e in belief_events if e['kind']=='grasp_evidence'],
        'interpretation':'process evidence only; no official score or semantic correctness inferred'}


def _motion_rows(groups, dt, trace, commands):
    rows = []
    for phase, group in groups.items():
        steps = np.array([f['step'] for f in group])
        contiguous = np.diff(steps) == 1
        q = np.array([f['qpos'] for f in group], float) if all('qpos' in f for f in group) else None
        tcp = np.array([f['tcp_base'] for f in group], float)[:, :3, 3] if all('tcp_base' in f for f in group) else None
        action = np.array([f['action'] for f in group], float) if all('action' in f for f in group) else None
        row = {'phase': phase, 'samples': len(group), 'first_step': int(steps[0]), 'last_step': int(steps[-1])}
        if tcp is not None:
            velocity = np.diff(tcp, axis=0)[contiguous] / dt
            row['tcp_speed_rms_m_s'] = float(np.sqrt(np.mean(velocity**2))) if len(velocity) else None
            row['tcp_span_m'] = np.ptp(tcp, axis=0).tolist()
            # Reversals only within consecutive samples; curved intended motion can also reverse.
            deltas = np.diff(tcp, axis=0)
            mask = contiguous[:-1] & contiguous[1:]
            row['tcp_direction_reversals'] = int(np.sum((deltas[:-1]*deltas[1:] < -1e-8)[mask]))
        velocities = [f['qvel'] for f in group if 'qvel' in f]
        row['joint_velocity_rms'] = float(np.sqrt(np.mean(np.array(velocities)[:,3:]**2))) if velocities else None
        if q is not None:
            row['base_xy_span_m'] = np.ptp(q[:,:2], axis=0).tolist()
        row['arm_action_saturation_fraction'] = float(np.mean(abs(action[:,:7]) >= .99)) if action is not None else None
        local_trace = [t for t in trace if t.get('step') in set(map(int,steps))]
        ik_trace = [t for t in local_trace if 'ik_reason' in t]
        row['ik_failure_samples'] = sum(t['ik_reason'] not in ('ok', 'planned_joint_endpoint') for t in ik_trace) if ik_trace else None
        row['ik_trace_available'] = bool(ik_trace)
        contacts = [f['contact_diagnostics_eval_only'] for f in group if 'contact_diagnostics_eval_only' in f]
        forces_sampled = [c['sampled_force_n'] for d in contacts for c in d.get('contacts',[]) if 'sampled_force_n' in c]
        row['contact_sampled_force_peak_n'] = max(forces_sampled,default=0.) if contacts else None
        row['contact_samples_available'] = len(contacts)
        notes = [c.get('call',{}).get('note') for c in commands if c.get('call') and
                 c.get('start_step',-1) < int(steps[-1]) and
                 c.get('start_step',-1)+c.get('result',{}).get('steps_used',0) >= int(steps[0])]
        row['model_notes'] = [note for note in notes if note]
        row['attribution'] = 'undeterminable_from_motion_statistics_alone'
        forces = [f.get('official_info_eval_only', {}).get('robot_cumulative_force') for f in group]
        forces = [float(np.asarray(x).reshape(-1)[0]) for x in forces if x is not None]
        row['cumulative_force_max'] = max(forces) if forces else None
        rows.append(row)
    return rows


EX_STAGES = frozenset({'closing', 'opening', 'lifting', 'empty_hold', 'loaded_hold',
                       'return_to_rest', 'navigation', 'observing', 'unknown'})


def _stage_annotations(report):
    annotations = report.get('diagnostic_stage_annotations', [])
    result = []
    ids = set()
    for item in annotations:
        label, start, end = item.get('stage'), item.get('start_step'), item.get('end_step')
        identifier, source = item.get('id'), item.get('evidence_source')
        if (label not in EX_STAGES or not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool) or start < 0 or end <= start
                or not isinstance(identifier, str) or not identifier or identifier in ids
                or not isinstance(source, str) or not source.strip()):
            raise ValueError('Invalid EX annotation; require unique id, stage, (start,end] and evidence_source')
        if any(max(start, old['start_step']) < min(end, old['end_step']) for old in result):
            raise ValueError('Overlapping EX stage annotations')
        ids.add(identifier)
        result.append(dict(item))
    return result


def _command_stage(call):
    tool = call.get('tool')
    if tool == 'set_gripper' and isinstance(call.get('gripper'), (int, float)):
        return ('closing' if call['gripper'] < .5 else 'opening'), 'command_gripper_direction'
    fixed = {'return_to_rest': 'return_to_rest', 'move_base': 'navigation', 'look': 'observing'}
    if tool in fixed:
        return fixed[tool], 'command_tool'
    # A move command or an approximately motionless trajectory is insufficient
    # to distinguish intended hold, lift, failed lift, approach or return.
    return 'unknown', 'NA: no explicit lift/hold stage evidence'


def _stage_segments(report, dt, trace, commands):
    annotations = _stage_annotations(report)
    segments = []
    previous_key = None
    for frame in report['frames']:
        step = frame['step']
        owners = [(i, c) for i, c in enumerate(commands)
                  if c.get('start_step', -1) < step <= c.get('start_step', -1) + c.get('result', {}).get('steps_used', 0)]
        if len(owners) > 1:
            raise ValueError('Overlapping physical command intervals')
        index, command = owners[0] if owners else (None, {})
        annotation = next((a for a in annotations if a['start_step'] < step <= a['end_step']), None)
        if annotation:
            stage, basis = annotation['stage'], 'explicit_annotation'
            evidence_source, annotation_id = annotation['evidence_source'], annotation['id']
        else:
            stage, basis = _command_stage(command.get('call') or {})
            evidence_source = f'/commands/{index}/call' if index is not None else None
            annotation_id = None
        key = (index, stage, basis, evidence_source, annotation_id)
        if key != previous_key or (segments and step != segments[-1]['frames'][-1]['step'] + 1):
            segments.append({'key': key, 'frames': []})
        segments[-1]['frames'].append(frame)
        previous_key = key
    rows = []
    for number, segment in enumerate(segments):
        index, stage, basis, evidence_source, annotation_id = segment['key']
        row = _motion_rows({stage: segment['frames']}, dt, trace, commands)[0]
        row.update({'segment_id': f'segment-{number:03d}', 'stage': stage, 'command_index': index,
                    'stage_evidence_basis': basis, 'stage_evidence_source': evidence_source,
                    'annotation_id': annotation_id,
                    'stage_status': 'NA' if stage == 'unknown' else 'labelled',
                    'sampling_convention': 'physical frames in (command_start, command_end]'})
        rows.append(row)
    return rows


def _base_pose(report, step):
    state = report.get('initial_robot_state', {}) if step == 0 else next(
        (f for f in report.get('frames', []) if f.get('step') == step), {})
    matrix = state.get('base_world_eval_only')
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    return matrix if matrix.shape == (4, 4) and np.isfinite(matrix).all() else None


def _base_command_rows(report, commands):
    rows = []
    for i, command in enumerate(commands):
        call, result = command.get('call') or {}, command.get('result') or {}
        if call.get('tool') != 'move_base':
            continue
        feedback = {**result, **result.get('feedback', {})}
        turn, forward = call.get('turn_rad'), call.get('forward_m')
        actual_turn, actual_forward = feedback.get('actual_turn_rad'), feedback.get('actual_forward_m')
        turn_source = 'executor_feedback' if actual_turn is not None else None
        forward_source = 'executor_feedback' if actual_forward is not None else None
        start = command.get('start_step', 0)
        before, after = _base_pose(report, start), _base_pose(report, start + result.get('steps_used', 0))
        lateral = None
        if before is not None and after is not None:
            yaw0, yaw1 = np.arctan2(before[1, 0], before[0, 0]), np.arctan2(after[1, 0], after[0, 0])
            if actual_turn is None:
                actual_turn = float(np.arctan2(np.sin(yaw1-yaw0), np.cos(yaw1-yaw0)))
                turn_source = 'exact_boundary_base_world_eval_only'
            if turn is not None:
                heading = yaw0 + turn
                direction = np.array([np.cos(heading), np.sin(heading)])
                displacement = after[:2, 3] - before[:2, 3]
                if actual_forward is None:
                    actual_forward = float(displacement @ direction)
                    forward_source = 'exact_boundary_base_world_eval_only_projected_requested_heading'
                lateral = float(displacement @ np.array([-direction[1], direction[0]]))
        yaw_error = None if turn is None or actual_turn is None else float(
            np.arctan2(np.sin(actual_turn-turn), np.cos(actual_turn-turn)))
        rows.append({'command_index': i, 'source_json_pointer': f'/commands/{i}',
                     'start_step': start, 'end_step': start + result.get('steps_used', 0),
                     'requested_turn_rad': turn, 'actual_turn_rad': actual_turn,
                     'requested_forward_m': forward, 'actual_forward_m': actual_forward,
                     'turn_residual_rad': yaw_error,
                     'forward_residual_m': None if forward is None or actual_forward is None else actual_forward-forward,
                     'lateral_residual_m_eval_only': lateral,
                     'turn_measurement_source': turn_source, 'forward_measurement_source': forward_source,
                     'arrived': feedback.get('arrived'), 'note': call.get('note'),
                     'missing_residual_reason': 'NA: missing request or feedback and exact pose boundary' if yaw_error is None or actual_forward is None else None})
    return rows


def analyze(report):
    frames = report.get('frames', [])
    dt = report.get('control_dt', .05)
    if not frames or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        return {'status': 'missing_trajectory', 'case_id': report.get('case_id')}
    first_grasp = next((f['step'] for f in frames if f.get('is_grasped') is True), None)
    commands = report.get('commands', [])
    trace = report.get('executor_trace', [])
    def command_at(step):
        candidates = [c for c in commands if c.get('start_step', -1) < step]
        return candidates[-1].get('call') or {} if candidates else {}
    groups = {}
    for f in frames:
        call = command_at(f['step']); tool = call.get('tool', 'SAC_or_unrecorded')
        phase = ('pre_grasp' if first_grasp is None or f['step'] < first_grasp else 'post_grasp') + '/' + tool
        groups.setdefault(phase, []).append(f)
    return {'status': 'offline_statistics', 'case_id': report.get('case_id'),
        'first_designated_grasp_step_eval_only': first_grasp,
        'phases': _motion_rows(groups, dt, trace, commands),
        'stage_segments': _stage_segments(report, dt, trace, commands),
        'base_commands': _base_command_rows(report, commands),
        'stage_annotation_count': len(report.get('diagnostic_stage_annotations', [])),
        'limitations': ['Motion spans/reversals include intended motion; they do not prove oscillation.',
                       'No new simulation or causal controller comparison was performed.',
                       'Unavailable traces/contacts remain unavailable, never imputed as zero.',
                       'Lift and hold require explicit evidence annotations; unlabelled motion remains unknown.',
                       'The legacy phases view is retained; stage_segments preserves individual command boundaries.']}


def paired_sac_reference(candidate, reference, binding):
    """Compare explicitly bound same-state stages; never infer a matched reset."""
    case = candidate.get('case_id')
    reset = binding.get('candidate_reset_sha256')
    if (case is None or case != reference.get('case_id') or case != binding.get('case_id')
            or binding.get('reference_policy') != 'official_sac'
            or not isinstance(reset, str) or len(reset) != 64
            or any(c not in '0123456789abcdef' for c in reset)
            or reset != binding.get('reference_reset_sha256')):
        raise ValueError('SAC reference requires same case and explicit matching reset hash')
    comparisons = []
    for pair in binding.get('stage_pairs', []):
        left = next((s for s in candidate.get('stage_segments', []) if s['segment_id'] == pair['candidate_segment_id']), None)
        right = next((s for s in reference.get('stage_segments', []) if s['segment_id'] == pair['reference_segment_id']), None)
        if left is None or right is None or left['stage'] != right['stage'] or left['stage'] == 'unknown':
            raise ValueError('Explicit stage pair is missing, ambiguous or has different stage labels')
        comparisons.append({'stage': left['stage'], 'candidate_segment_id': left['segment_id'],
                            'reference_segment_id': right['segment_id'],
                            'candidate': left, 'reference': right})
    return {'status': 'paired_reference_bound' if comparisons else 'reset_bound_stage_pairs_NA',
            'case_id': case, 'reset_sha256': reset, 'reference_policy': 'official_sac',
            'stage_comparisons': comparisons,
            'limitation': 'Same reset does not make SAC and Astra controllers or trajectories equivalent; phase diagnostic only.'}
