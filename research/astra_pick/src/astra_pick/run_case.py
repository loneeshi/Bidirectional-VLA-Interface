"""CPU archive checks and explicit live environment wiring.

Evaluation and transport are owned here, never passed as objects to pick.py.
All helpers are extracted from the hash-pinned C0/C1 deployment, not imported
from the former bvi package. Live execution requires fresh candidate provenance.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from . import pick
from .pick import save, canonical, vector, SENSOR_FIELDS, PROPRIO_KEYS, RESOLUTION
from .robot.eef_tools import RobotFK, validate_official_controller
from .robot.motion import validate_base_runtime
from .robot.executor import ArmCoordinatedV2
from .robot.arm import ParametersV3
from .robot.sensing import ObservationDepth

GPU='GPU-b7ebba23-7824-7601-df32-be55628936c3'
# Evaluation-only labels from frozen C0 src/bvi/eef_arm_scoring.py.
FUNNEL = ('check_path_accepted', 'tcp_near_bbox', 'is_grasped', 'grasped_lift_5cm', 'strict_success')


def search_phase_usage(report):
    """Evaluation-side accounting, never fed back as a robot observation.

    Phase is the model's declared intent, not a ground-truth task label. Keep
    each actual command and its note so a reviewer can check that declaration.
    Unclassified calls/steps remain visible instead of being guessed away.
    """
    phases=('search','approach','grasp','lift_rest','unclassified')
    rows={p:dict(steps=0,main_requests=0,verification_requests=0,
                verification_attempts=0,unsettled_verification_attempts=0,
                unsent_verification_attempts=0,give_up_reasons=[])
          for p in phases}
    commands=report.get('commands',[]);actions=[];trace_by_step={}
    for tick in report.get('executor_trace',[]):
        trace_by_step.setdefault(tick['step'],[]).append(tick.get('kind','unknown'))
    verification_by_turn={}
    for event in report.get('search_phase_verifications',[]):
        phase=event.get('phase')
        row=rows[phase if phase in phases else 'unclassified'];row['verification_attempts']+=1
        if event.get('status')=='settled':
            turn=event['decision_turn']
            verification_by_turn[turn]=verification_by_turn.get(turn,0)+1
            row['verification_requests']+=1
        elif event.get('status')=='not_sent':row['unsent_verification_attempts']+=1
        else:row['unsettled_verification_attempts']+=1
    for command in commands:
        call=command.get('call') or {};result=command.get('result') or {}
        memory=command.get('search_memory') or call.get('search_memory') or {}
        phase=memory.get('phase');phase=phase if phase in phases else 'unclassified'
        interval=command.get('ack_interval')
        steps=(interval['end_step']-interval['start_step'] if interval is not None
               else result.get('steps_used',0))
        if type(steps) is not int or steps<0:raise ValueError('Invalid acknowledged phase steps')
        rows[phase]['steps']+=steps;rows[phase]['main_requests']+=1
        entry=dict(turn=command.get('turn'),phase=phase,tool=call.get('tool'),
            start_step=command.get('start_step'),steps_used=steps,
            main_requests=1,verification_requests=verification_by_turn.get(command.get('turn'),0),
            result_reason=result.get('reason'),note=call.get('note'),
            steps_basis='executor_ack_interval' if interval is not None else 'legacy_feedback',
            feedback_steps_used=result.get('steps_used'),feedback_matches_ack=result.get('steps_used')==steps,
            phase_basis='model_self_report' if phase!='unclassified' else 'missing_or_invalid_call')
        start=command.get('start_step');executed={}
        if type(start) is int:
            for step in range(start+1,start+steps+1):
                for kind in trace_by_step.get(step,[]):
                    executed[kind]=executed.get(kind,0)+1
        entry['executed_controller_phases']=executed
        if call.get('tool')=='give_up':
            reason=dict(turn=command.get('turn'),hindsight=call.get('hindsight'),
                result_reason=result.get('reason'),search_memory=copy.deepcopy(memory))
            rows[phase]['give_up_reasons'].append(reason)
        actions.append(entry)
    acknowledged_steps=sum(r['steps'] for r in rows.values())
    pending=report.get('pending_command');incomplete=None
    if pending is not None and pending.get('turn') not in {c.get('turn') for c in commands}:
        memory=(pending.get('call') or {}).get('search_memory') or {}
        phase=memory.get('phase');phase=phase if phase in phases else 'unclassified'
        partial_steps=report.get('simulator_actions',0)-pending['start_step']
        if type(partial_steps) is not int or partial_steps<0:raise ValueError('Invalid pending physical interval')
        rows[phase]['steps']+=partial_steps;rows[phase]['main_requests']+=1
        incomplete=dict(turn=pending['turn'],phase=phase,steps=partial_steps,main_requests=1,
            steps_basis='physical_step_counter_pending_ack_or_completion',completed=False)
    allocated_steps=sum(r['steps'] for r in rows.values())
    allocated_calls=sum(r['main_requests']+r['verification_requests'] for r in rows.values())
    accounting=report.get('weekly_accounting') or {}
    ledger_count=accounting.get('provider_requests')
    reported_calls=report.get('api_requests')
    return dict(version='search-phase-usage-v1',evaluation_only=True,
        phase_basis='model_self_report_cross_check_notes_actions_and_video',phases=rows,actions=actions,
        acknowledged_steps=acknowledged_steps,attributed_steps=allocated_steps,
        incomplete_execution=incomplete,attributed_requests=allocated_calls,
        reported_physical_steps=report.get('simulator_actions'),reported_api_requests=report.get('api_requests'),
        unattributed_steps=(report.get('simulator_actions',0)-allocated_steps),
        unattributed_requests=(reported_calls-allocated_calls if type(reported_calls) is int else None),
        ledger_requests=ledger_count,ledger_unresolved_requests=accounting.get('unresolved_requests'),
        ledger_unattributed_requests=(ledger_count-allocated_calls if type(ledger_count) is int else None),
        ledger_snapshot_available=bool(accounting),
        boundary='Live counts are provisional; final weekly_accounting is the independent ledger. Missing ledger is unknown, never zero. Pending physical steps are not declared complete ACKs. Unknown provider outcomes require reconciliation; never infer zero cost.')


def staged_metrics(report):
    """Evaluation-only extension, using contemporaneous surface scores, never boxes."""
    commands = report.get('commands')
    command_evidence = isinstance(commands, list) and all(isinstance(c, dict) and isinstance(c.get('result'), dict) for c in commands)
    if not command_evidence:
        commands = []
    selected = [c for c in commands if (c.get('call') or {}).get('tool') == 'locate_point' and c['result'].get('valid')]
    diagnostics = [d for d in report.get('localization_eval_only', []) if 'pixel' not in d]
    evidence, missing = [], []
    if not command_evidence:
        missing.append(dict(metric='command_metrics', reason='missing or incomplete command history'))
    for c in selected:
        candidates = [d for d in diagnostics if d.get('turn') == c['turn']]
        d = candidates[0] if len(candidates) == 1 else {}
        distance = d.get('surface_distance_m')
        observation_id = c['result'].get('observation_id')
        start, used = c.get('start_step'), c['result'].get('steps_used')
        bound = (isinstance(observation_id, str) and bool(observation_id)
                 and d.get('observation_id') == observation_id
                 and (c.get('call') or {}).get('observation_id') == observation_id
                 and type(start) is int and type(used) is int and used == 0
                 and type(d.get('step')) is int and d['step'] == start)
        ok = bound and d.get('status') == 'scored' and type(distance) in (int, float) and math.isfinite(distance) and distance >= 0
        evidence.append(distance <= .05 if ok else None)
        if not ok:
            reason = ('duplicate target surface evidence' if len(candidates) > 1 else
                      'missing or mismatched contemporaneous target surface evidence' if not bound else
                      'missing exact target surface distance: '+str(d.get('reason', d.get('status'))))
            missing.append(dict(metric='selected_target', turn=c['turn'], reason=reason))
    frames = report.get('frames', [])
    grasp = [f.get('grasped_objects_eval_only') for f in frames]
    steps = report.get('simulator_actions')
    complete = (type(steps) is int and steps > 0 and
                [f.get('step') for f in frames] == list(range(1, steps+1))
                and all(isinstance(x, list) for x in grasp))
    any_grasp = True if any(isinstance(x, list) and bool(x) for x in grasp) else False if complete else None
    if any_grasp is None:
        missing.append(dict(metric='grasped_any_object', reason='missing or incomplete per-step grasped object evidence'))
    final = report.get('official_final_info')
    if final is None:
        final = frames[-1].get('official_info_eval_only', {}) if frames and type(steps) is int and frames[-1].get('step') == steps else {}
    force = final.get('robot_cumulative_force')
    while isinstance(force, list) and len(force) == 1:
        force = force[0]
    force_ok = type(force) in (int, float) and math.isfinite(force) and force >= 0
    if not force_ok:
        missing.append(dict(metric='force_budget_fraction', reason='missing final cumulative force'))
    return dict(evaluation_only=True, selected_target_any=None if not command_evidence else True if True in evidence else None if None in evidence else False,
                selected_target_final=None if not command_evidence else evidence[-1] if evidence else False, selected_points=len(selected) if command_evidence else None,
                grasped_any_object=any_grasp, force_budget_fraction=force/5000 if force_ok else None,
                base_motion_aborts=sum(c['result'].get('reason') == 'base_motion_aborted' for c in commands) if command_evidence else None,
                missing_evidence=missing, surface_threshold_m=.05)


class ProtocolError(ValueError):
    pass

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def target_semantics(object_id: str) -> str:
    parts = object_id.rsplit('-', 1)
    if len(parts) != 2 or not parts[1].isdigit() or parts[0] not in pick.OBJECT_NAMES:
        raise ValueError('unrecognized target object class')
    return 'target object: ' + pick.OBJECT_NAMES[parts[0]]


def navigate_pick_spawn(navigate_plan, pick_plan, spawn):
    """Evaluator-side reset adapter. Preserve the official robot start exactly.

    Navigate ignores its zero carried-object placeholder when unladen. Pick
    would apply that placeholder to its target, so never forward it to Pick.
    This does not execute a reset or establish simulator-state equivalence.
    """
    for key in ('build_config_name', 'init_config_name'):
        if navigate_plan[key] != pick_plan[key]:
            raise ValueError('Navigate/Pick scene mismatch')
    if len(navigate_plan['subtasks']) != 1 or len(pick_plan['subtasks']) != 1:
        raise ValueError('single official subtask required')
    nav, target = navigate_plan['subtasks'][0], pick_plan['subtasks'][0]
    if nav['type'] != 'navigate' or target['type'] != 'pick' or not target.get('obj_id'):
        raise ValueError('Navigate to Pick types required')
    if any(nav.get(k) is not None for k in ('obj_id', 'articulation_config', 'remove_obj_id')):
        raise ValueError('only unladen unmodified Navigate starts supported')
    required = {'robot_pos', 'robot_qpos'}
    if not required <= spawn.keys() or set(spawn) - required - {'obj_raw_pose_wrt_tcp'}:
        raise ValueError('unsupported Navigate reset fields')
    pos, qpos = array(spawn['robot_pos']), array(spawn['robot_qpos'])
    if pos.ndim != 2 or pos.shape[1] != 3 or qpos.shape != (len(pos), 15) or not len(pos):
        raise ValueError('official Navigate spawn shape')
    if not np.isfinite(pos).all() or not np.isfinite(qpos).all():
        raise ValueError('nonfinite Navigate start')
    if 'obj_raw_pose_wrt_tcp' in spawn:
        sentinel = array(spawn['obj_raw_pose_wrt_tcp'])
        if sentinel.shape != (len(pos), 7) or np.any(sentinel != 0):
            raise ValueError('unladen object placeholder must be zero')
    return {target['composite_subtask_uids'][0]:
            {key: copy.deepcopy(spawn[key]) for key in ('robot_pos', 'robot_qpos')}}

# Robot sensor capture, copied from eef_arm_sensing and render helpers.
def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

def state_copy(value):
    if isinstance(value, dict):
        return {k: state_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_copy(v) for v in value]
    return array(value).copy()

INITIALIZATION_SDK_MODULES = {
    'mshab.envs.make': 'src/official-mshab-runtime/mshab/mshab/envs/make.py',
    'mshab.envs.pick': 'src/official-mshab-runtime/mshab/mshab/envs/pick.py',
    'mshab.envs.subtask': 'src/official-mshab-runtime/mshab/mshab/envs/subtask.py',
    'mshab.envs.sequential_task': 'src/official-mshab-runtime/mshab/mshab/envs/sequential_task.py',
    'mani_skill.envs.sapien_env': 'src/official-mshab-runtime/ManiSkill/mani_skill/envs/sapien_env.py',
    'mani_skill.utils.scene_builder.replicacad.rearrange.scene_builder':
        'src/official-mshab-runtime/ManiSkill/mani_skill/utils/scene_builder/replicacad/rearrange/scene_builder.py',
}


def initialization_observability_metadata(args, auth, row, cfg, env_config, *, seed,
                                         reset_options, files, modules):
    """Read real worker inputs only; never inspect environment variables or credentials."""
    def bound(path):
        path = Path(path).resolve()
        for root, manifest in ((Path(args.lab_root).resolve(), auth['asset_sha256']),
                               (Path(args.code).resolve(), auth['source_sha256'])):
            if path.is_relative_to(root):
                key = path.relative_to(root).as_posix()
                if key in manifest and path.is_file() and sha(path) == manifest[key]:
                    return dict(path=str(path), key=key, sha256=manifest[key])
        raise ValueError('initialization input is not a frozen installed asset/source: '+str(path))
    if (not isinstance(sys.argv, list) or not sys.argv or not all(isinstance(v, str) for v in sys.argv) or not sys.argv[0]):
        raise ValueError('initialization actual argv missing or malformed')
    if (not is_dataclass(env_config) or isinstance(env_config, type)
            or not isinstance(cfg, dict) or not {'plan', 'spawn', 'checkpoint_config'} <= set(files)
            or set(files)-{'plan', 'spawn', 'checkpoint_config', 'navigate_plan', 'navigate_spawn'}
            or set(modules) != set(INITIALIZATION_SDK_MODULES)
            or seed != row['seed'] or reset_options != {'reconfigure': True, 'spawn_selection_idxs': [row['spawn_index']]}):
        raise ValueError('initialization metadata incomplete or reset inputs changed')
    required_args = ('case_id', 'profile', 'authorization', 'output', 'mode', 'condition')
    if any(not hasattr(args, key) for key in required_args) or not row.get('uid'):
        raise ValueError('initialization worker identity/authorization arguments missing')
    for key in ('case_id', 'profile', 'authorization', 'output', 'mode'):
        flag = '--'+key.replace('_', '-')
        if sys.argv.count(flag) != 1 or sys.argv.index(flag)+1 >= len(sys.argv) or sys.argv[sys.argv.index(flag)+1] != str(getattr(args, key)):
            raise ValueError('initialization actual argv differs from parsed arguments')
    if (Path(cfg.get('task_plan_fp', '')).resolve() != (Path(args.output)/'single-plan.json').resolve()
            or Path(cfg.get('spawn_data_fp', '')).resolve() !=
                ((Path(args.output)/'official-navigate-pick-spawn.pt') if row.get('spawn_source') == 'official_navigate' else Path(files['spawn'])).resolve()):
        raise ValueError('initialization final configuration input paths differ')
    final_config = asdict(env_config)
    if (type(env_config).__module__+'.'+type(env_config).__qualname__ != 'mshab.envs.make.EnvConfig'
            or set(final_config) != set(pick.INITIALIZATION_ENV_CONFIG_FIELDS)
            or not set(pick.INITIALIZATION_CONFIG_INPUT_FIELDS) <= set(cfg)):
        raise ValueError('initialization actual EnvConfig schema incomplete')
    if any(k not in final_config or jsonable(final_config[k]) != jsonable(v) for k,v in cfg.items()):
        raise ValueError('initialization EnvConfig differs from final configuration')
    if row.get('spawn_source') == 'official_navigate' and not {'navigate_plan', 'navigate_spawn'} <= set(files):
        raise ValueError('initialization Navigate source evidence missing')
    file_bindings = {key: bound(path) for key,path in files.items()}
    installed = {}
    for name, key in INITIALIZATION_SDK_MODULES.items():
        module = modules[name]
        path = Path(getattr(module, '__file__', '')).resolve()
        if path != (Path(args.lab_root)/key).resolve():
            raise ValueError('initialization imported SDK module path mismatch: '+name)
        installed[name] = bound(path)
    derived = {}
    for label, name in (('single_plan', 'single-plan.json'), ('adapted_spawn', 'official-navigate-pick-spawn.pt')):
        if label == 'adapted_spawn' and row.get('spawn_source') != 'official_navigate':
            continue
        path = Path(args.output)/name
        if not path.is_file():
            raise ValueError('initialization derived input missing: '+name)
        derived[label] = dict(path=str(path.resolve()), sha256=sha(path),
                              provenance='derived by frozen worker from the bound input files; not an independent asset')
    config = jsonable(final_config)
    return dict(evaluation_only=True, protocol='init-boundary-eval-v1',
                source_manifest_sha256=auth['source_manifest_sha256'], actual_sys_argv=list(sys.argv),
                parsed_args=jsonable(vars(args)), final_config_input=jsonable(cfg),
                worker_identity=dict(case_id=args.case_id, uid=row['uid'], spawn_source=row.get('spawn_source', 'official_pick'),
                                     object_category=row['object_category'], dataset_split=official_dataset_split(row),
                                     plan_filename='all.json' if 'dataset_split' in row else row['object_category']+'.json'),
                authorization=dict(path=str(Path(args.authorization).resolve()), sha256=sha(args.authorization)),
                env_config_dataclass=type(env_config).__module__+'.'+type(env_config).__qualname__,
                env_config=config, env_config_sha256=hashlib.sha256(canonical(config)).hexdigest(),
                seed=seed, reset_options=copy.deepcopy(reset_options), files=file_bindings,
                derived_files=derived, installed_modules=installed,
                limits='Outer boundaries only; CPU copies can synchronize GPU and change timing.')


def _initialization_json_exclusive(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def capture_initialization_boundary(output, stage, *, state_getter=None, info=None,
                                    snapshot_writer, metadata=None):
    """One export per boundary; partial files/marker remain on error, never retried."""
    if stage not in pick.INITIALIZATION_OBSERVABILITY_STAGES:
        raise ValueError('unknown initialization observation stage')
    base = Path(output)/('initialization-'+stage)
    marker = Path(str(base)+'.marker.eval-only.json')
    _initialization_json_exclusive(marker, dict(stage=stage, status='capture_started', evaluation_only=True))
    receipt = dict(stage=stage, evaluation_only=True, files={},
                   marker=dict(path=marker.name, sha256=sha(marker)))
    if stage == pick.INITIALIZATION_OBSERVABILITY_STAGES[0]:
        if state_getter is not None or metadata is None or info is not None:
            raise ValueError('construction boundary requires metadata only')
        data = metadata() if callable(metadata) else metadata
        path = Path(str(base)+'.metadata.eval-only.json')
        _initialization_json_exclusive(path, data)
        receipt['files']['metadata'] = dict(path=path.name, sha256=sha(path))
    else:
        if state_getter is None or metadata is not None:
            raise ValueError('reset boundary requires state only')
        snapshot = state_copy(state_getter())
        path = Path(str(base)+'.state.eval-only.pt')
        with path.open('xb') as stream:
            snapshot_writer(snapshot, stream)
        receipt['files']['state'] = dict(path=path.name, sha256=sha(path))
        if stage == pick.INITIALIZATION_OBSERVABILITY_STAGES[2]:
            if info is None:
                raise ValueError('post-reset boundary requires existing reset info')
            path = Path(str(base)+'.info.eval-only.json')
            _initialization_json_exclusive(path, jsonable(info))
            receipt['files']['info'] = dict(path=path.name, sha256=sha(path))
        elif info is not None:
            raise ValueError('pre-reset boundary has no returned reset info')
    receipt['status'] = 'capture_complete'
    receipt_path = Path(str(base)+'.receipt.eval-only.json')
    _initialization_json_exclusive(receipt_path, receipt)
    return dict(stage=stage, path=receipt_path.name, sha256=sha(receipt_path))


def record_initialization_error(output, stage, exc, *, primary):
    """Return first-cause evidence even when its own exclusive write fails."""
    import traceback
    evidence = dict(stage=stage, primary=primary, error_type=type(exc).__name__,
                    error_message=str(exc), traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    label = 'primary' if primary else 'secondary-'+re.sub(r'[^a-zA-Z0-9_-]', '_', stage)
    path = Path(output)/('initialization-'+label+'-error.eval-only.json')
    try:
        _initialization_json_exclusive(path, evidence)
    except Exception as write_error:
        evidence['recording_secondary_error'] = dict(error_type=type(write_error).__name__,
            error_message=str(write_error), traceback=''.join(traceback.format_exception(type(write_error), write_error, write_error.__traceback__)))
    return evidence


def initialization_final_action(output, report, stage, action):
    """Run one closeout action without replacing an already-recorded first cause."""
    try:
        return action()
    except Exception as exc:
        primary = 'initialization_error_eval_only' not in report and 'error_type' not in report
        evidence = record_initialization_error(output, stage, exc, primary=primary)
        report.setdefault('initialization_secondary_errors_eval_only', []).append(evidence)
        if primary:
            report.update(status='infrastructure_censored', error_type=type(exc).__name__,
                          error_message=str(exc), initialization_error_eval_only=evidence)
        return None


def state_equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all((state_equal(left[k], right[k]) for k in left))
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all((state_equal(a, b) for a, b in zip(left, right)))
    return np.array_equal(left, right)

def camera_in_base(fk, q, camera):
    links = fk.links(q)
    if camera == 'head':
        return links['head_camera_link']
    if camera != 'hand':
        raise ValueError('robot camera only')
    mount = np.eye(4)
    mount[:3, 3] = [-0.1, 0.0, 0.1]
    return links['gripper_link'] @ mount


def grasp_sensor_feedback(q, fk, raw_depth, intrinsics):
    """Measured opening and an FK-projected region between the finger tips."""
    links = fk.links(q)
    transform = np.linalg.inv(camera_in_base(fk, q, 'hand'))
    pixels = []
    for name in ('l_gripper_finger_link', 'r_gripper_finger_link'):
        # Finger link origin is the robot-model finger reference, not scene geometry.
        p = transform @ links[name][:, 3]
        if p[0] <= 0:
            continue
        uv = intrinsics @ np.array([-p[1], -p[2], p[0]])
        pixels.append(uv[:2]/uv[2])
    out = dict(finger_joint_positions_m=list(map(float, q[-2:])),
               measured_opening_joint_sum_m=float(sum(q[-2:])), camera='hand',
               depth_region='between FK-projected finger references, 5 pixel vertical band',
               depth_median_m=None, depth_min_m=None, depth_max_m=None, valid_pixels=0)
    if len(pixels) == 2:
        x0, x1 = sorted(int(round(p[0])) for p in pixels)
        y = int(round(np.mean([p[1] for p in pixels])))
        if 0 <= y < 640 and x1 >= 0 and x0 < 640:
            box = [max(0,x0), max(0,y-2), min(639,x1), min(639,y+2)]
            values = np.asarray(raw_depth)[box[1]:box[3]+1, box[0]:box[2]+1]
            values = values[values > 0].astype(float)/1000
            out['pixel_roi_xyxy'] = box
            if len(values):
                out.update(depth_median_m=float(np.median(values)), depth_min_m=float(values.min()),
                           depth_max_m=float(values.max()), valid_pixels=int(len(values)))
    return out

def history_for_observation_audit(history, profile):
    """Project authorized extensions for the legacy sensor audit; retain raw history elsewhere."""
    if getattr(profile,'stage_plan',False):
        # S2's separate audit validates each command result and the new local
        # query result. A generated decision is not a sensor-history row.
        return copy.deepcopy(history)
    observed_history = pick.project_history(history, profile) if profile.measure else copy.deepcopy(history)
    if profile.base_motion_abort or (profile.base_progress_abort and (profile.base_progress_monitor or profile.base_response_monitor or profile.base_response_debt_monitor)):
        for entry in observed_history:
            entry['result'].pop('base_abort_rule', None)
    if profile.verification:
        for entry in observed_history:
            entry['result'].pop('identity_verification', None)
            if profile.identity_revision == 2:
                entry['result'].pop('target_record', None)
                entry['result'].pop('grasp_sensor_feedback', None)
                entry['result'].pop('target_observation_feedback', None)
                if entry['call']:
                    entry['call'].pop('target_observation', None)
                if entry['call'] and entry['call']['tool'] in ('verify_target', 'recover_pick'):
                    entry['call'] = None
    if profile.stall_feedback:
        for entry in observed_history:
            for key in ('execution_phase', 'consecutive_ik_failures'):
                entry['result'].pop(key, None)
    if profile.target_association or profile.target_evidence_diagnostics:
        for entry in observed_history:
            entry['result'].pop('target_evidence_diagnostics', None)
            if entry['call']:
                entry['call'].pop('target_association', None)
    return observed_history


def observe(u, fk, qpos, semantic, history, remaining, folder, odometry, depth_store, previous_state=None, *, call_cap=25, profile=pick.C0, decision_turn=None, verification_requests=None):
    before = state_copy(u.get_state_dict())
    q = vector(qpos(), 15).copy()
    data = read_sensor_data_without_evaluate(u)
    if not state_equal(before, state_copy(u.get_state_dict())):
        raise ValueError('render changed simulator state')
    if not np.array_equal(q, qpos()):
        raise ValueError('render changed joints')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    sensors = {}
    manifest = {}
    cameras = {}
    for field, (camera, modality) in SENSOR_FIELDS.items():
        value = array(data[camera][modality])
        if value.shape[0] != 1 or value.shape[1:3] != (RESOLUTION, RESOLUTION):
            raise ValueError('native 640 sensor required')
        pixels = value[0]
        if modality == 'depth':
            raw = pixels[..., 0]
            np.save(folder / (field + '-raw.npy'), raw)
            pixels = depth_visualization_8bit(raw)
            k = array(u._sensors[camera].camera.get_intrinsic_matrix())
            if k.shape == (1, 3, 3):
                k = k[0]
            expected_focal = RESOLUTION / (2 * np.tan(1.0))
            if k.shape != (3, 3) or not np.allclose(k[[0, 1], [0, 1]], expected_focal, atol=0.001, rtol=0):
                raise ValueError('runtime 640/FOV intrinsic mismatch')
            short = 'head' if camera == 'fetch_head' else 'hand'
            cameras[short] = {'depth_mm': raw, 'intrinsics': k, 'camera_in_base': camera_in_base(fk, q, short)}
        else:
            if pixels.shape != (RESOLUTION, RESOLUTION, 3):
                raise ValueError('RGB channels')
            pixels = pixels.astype(np.uint8)
        file = folder / (field + '.png')
        Image.fromarray(pixels).save(file)
        encoded = file.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        sensors[field] = {'camera': camera, 'modality': modality, 'sha256': digest, 'data_base64': base64.b64encode(encoded).decode()}
        manifest[field] = {'camera': camera, 'modality': modality, 'sha256': digest, 'width': RESOLUTION, 'height': RESOLUTION}
    if decision_turn is not None and (not getattr(profile,'stage_plan',False) or type(decision_turn) is not int or decision_turn<0):
        raise ValueError('Decision-turn override requires the explicit S2 profile')
    turn = len(history) if decision_turn is None else decision_turn
    identifier = f'obs-{turn:03d}-' + hashlib.sha256(canonical(manifest) + q.tobytes()).hexdigest()[:16]
    depth_store.set(identifier, q, cameras)
    tcp = fk.tcp(q)
    velocity = array(u.agent.robot.qvel)[0]
    obs = {'target_semantics': semantic, 'observation_id': identifier, 'joint_positions': dict(zip(PROPRIO_KEYS, map(float, q[3:]))), 'joint_velocities': dict(zip(PROPRIO_KEYS, map(float, velocity[3:]))), 'tcp_pose': {'position': tcp[:3, 3].tolist(), 'quaternion_xyzw': Rotation.from_matrix(tcp[:3, :3]).as_quat().tolist()}, 'sensors': sensors, 'steps_remaining': int(remaining), 'calls_remaining': call_cap - turn - (sum((row.get('call') or {}).get('tool')=='verify_target' and row.get('result',{}).get('reason')=='review_sent' for row in history) if profile.r2a_identity else 0), 'history': copy.deepcopy(history), 'previous_state': copy.deepcopy(previous_state), 'odometry': dict(zip(('dx_m', 'dy_m', 'dyaw_rad'), map(float, odometry)))}
    if getattr(profile,'stage_plan',False) and verification_requests is not None:
        if type(verification_requests) is not int or verification_requests<0:raise ValueError('S2 verifier count')
        obs['calls_remaining']=call_cap-turn-verification_requests
    audit_profile = pick.replace(profile, model_target_boxes=False, target_association=False) if profile.active_confirmation else profile
    if getattr(profile,'stage_plan',False):
        # Sensor rendering precedes case/authorization bundle assembly. Reuse
        # S2's actual per-command history and decision-budget validator here;
        # the completed bundle receives its separate case/attempt check later.
        pick.s2_audit_observation(obs,manifest,turn,audit_profile)
    else:
        pick.audit_observation(obs, manifest, turn, audit_profile,
                               hot_recovery=profile == pick.SOL_HOT_RECOVERY)
    (folder / 'sensor-manifest.json').write_bytes(canonical(manifest))
    (folder / 'robot-camera-calibration.json').write_bytes(canonical({n: {k: v.tolist() for k, v in c.items() if k != 'depth_mm'} for n, c in cameras.items()}))
    return (obs, manifest)

def depth_visualization_8bit(depth_mm):
    """0 marks invalid; 1..255 linearly encodes 0..4000 mm, saturating above."""
    import numpy as np
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise ValueError('depth must be a two-dimensional integer millimeter array')
    clipped = np.clip(depth.astype(np.float32), 0, 4000)
    return np.where(depth > 0, 1 + np.rint(clipped * 254 / 4000), 0).astype(np.uint8)

def read_sensor_data_without_evaluate(uenv) -> dict:
    """Use ManiSkill's camera pathway without MS-HAB's mutating get_info()."""
    data = uenv._get_obs_sensor_data()
    if not isinstance(data, dict) or not {'fetch_head', 'fetch_hand'} <= set(data):
        raise ValueError('expected official head and hand sensor data')
    return data


# Robot-only calibration, derived on the laboratory CPU from the installed URDF.
EPISODE_GEOMETRY_PATH = 'research/c2/diagnostics/2026-10-14-episode-memory-clearance-cpu/robot-geometry.json'
EPISODE_GEOMETRY_SHA256 = 'e95482e16590f23a8d000513ba81d0504b95ae5aa3c00095dbfe77fce169fa7f'


def episode_geometry_pin(code, *, installed_robot_root=None, refine_base=False, refine_fingers=False):
    path=Path(code)/EPISODE_GEOMETRY_PATH
    if sha(path)!=EPISODE_GEOMETRY_SHA256:raise ValueError('Robot geometry pin changed')
    value=read(path)
    if (value['contract']!='fetch-link-local-collision-aabb-v1' or value['robot_only'] is not True
            or sha(Path(code)/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf')!=value['urdf_sha256']):
        raise ValueError('Robot FK and geometry pin mismatch')
    if installed_robot_root is not None:
        root=Path(installed_robot_root).resolve()
        for name,digest in value['source_files'].items():
            candidate=(root/name).resolve()
            if not candidate.is_relative_to(root) or sha(candidate)!=digest:
                raise ValueError('Installed robot geometry differs from calibration')
    geometry={name:dict(record,sha256=EPISODE_GEOMETRY_SHA256) for name,record in value['links'].items()}
    if refine_base:
        from .robot.clearance import RobotGeometry
        if installed_robot_root is None:raise ValueError('Base refinement requires installed robot hash check')
        geometry['base_link']['convex_vertices']=RobotGeometry.base_convex_vertices(installed_robot_root,value).tolist()
    if refine_fingers:
        from .robot.clearance import RobotGeometry,FINGERS
        if installed_robot_root is None:raise ValueError('Finger refinement requires installed robot hash check')
        for name in FINGERS | {'gripper_link'}:
            geometry[name]['convex_vertices']=RobotGeometry.link_convex_vertices(installed_robot_root,value,name).tolist()
    return geometry


def bind_episode_modules(state, executor, fk, *, geometry, calibration_binding,
                         local_support_floor=False):
    """Bind transient geometry consumers to the sole checkpointed memory owner.

    Must run after restoring policy/controller state. No current scene data or
    evaluator values are arguments to this function.
    """
    from .memory import EpisodeMemory
    from .robot.clearance import SensorClearance
    if state.response_session is None:raise ValueError('Episode memory needs a response session')
    episode=state.response_session.attempt_id
    binding=dict(sensor_calibration_sha256=calibration_binding,
        robot_geometry_sha256=EPISODE_GEOMETRY_SHA256,memory_version='episode-memory-v1')
    if type(local_support_floor) is not bool:
        raise ValueError('Boolean local support floor required')
    if local_support_floor:
        binding['geometry_classifier']='local-support-floor-v1'
    previous=getattr(state,'episode_memory',None)
    if previous is None:
        if state.archive:raise ValueError('Historical state needs explicit episode-memory migration')
        state.episode_memory=EpisodeMemory(episode,episode+':odometry',binding)
    else:
        state.episode_memory=EpisodeMemory.from_snapshot(previous.snapshot(),
            episode_id=episode,odom_frame_id=episode+':odometry',source_binding=binding)
    from .robot.clearance import ClearanceConfig
    # Resolve all observed surfaces (including nearby furniture) finely in
    # the approaching hand workspace. This changes voxel enclosure accuracy,
    # not occupied/free classification, contact ownership or the 15mm margin.
    config=ClearanceConfig(precision_reach_m=.30) if getattr(executor,'carried_envelope_enabled',False) else None
    if local_support_floor:
        from dataclasses import replace
        config=replace(config or ClearanceConfig(),local_support_floor=True)
    if getattr(executor,'s11_system_fixes',False):
        from dataclasses import replace
        config=replace(config or ClearanceConfig(),s11_report=True)
    checker=SensorClearance(fk,state.episode_memory,geometry=geometry,config=config)
    if getattr(executor,'local_clearance_guard',False) or getattr(executor,'r2b_clearance',False):executor.clearance=checker
    return checker


def ingest_episode_frame(state, checker, depth, *, step, odometry, sensor_sha256, record_views=True):
    """Register calibrated robot evidence, then store geometry in EpisodeMemory."""
    if depth.frame is None:raise ValueError('Episode memory requires current depth')
    oid,q,cameras=depth.frame
    # RGB is the retained encoded sensor image; geometry must bind the raw mm
    # depths, never their clipped 8-bit display PNG. Canonical int32 little-endian
    # encoding plus shape makes the hash independent of native sensor dtype.
    sensor_sha256=dict(sensor_sha256)
    for name,camera in cameras.items():
        raw=np.asarray(camera['depth_mm'],dtype='<i4')
        sensor_sha256[name+'_depth']=hashlib.sha256(canonical(dict(shape=list(raw.shape),
            encoding='little-endian-int32-mm'))+raw.tobytes()).hexdigest()
    calibration={name:{key:np.asarray(value).tolist() for key,value in camera.items()
        if key!='depth_mm'} for name,camera in cameras.items()}
    digest=hashlib.sha256(canonical(calibration)).hexdigest()
    if record_views:state.episode_memory.begin_observation(step=step)
    state.episode_memory.record_observation(oid,step=step,odometry=odometry,
        calibration_sha256=digest,sensor_sha256=sensor_sha256)
    records=checker.ingest_observation(oid,q,odometry,cameras,control_step=step,calibration_sha256=digest)
    state.episode_memory.record_geometry(records,step=step)
    if state.episode_memory.attachment() is not None:
        checker.associate_carried_surface(oid,q,odometry,cameras,control_step=step)
    from .robot.clearance import odometry_transform, OPTICAL_TO_CAMERA
    for camera,values in (cameras.items() if record_views else ()):
        mount=odometry_transform(odometry)@values['camera_in_base']
        direction=(mount[:3,:3]@OPTICAL_TO_CAMERA)[:,2]
        key=[int(math.floor(float(x)/.5)) for x in mount[:3,3]]
        key.extend([int(math.floor(math.atan2(direction[1],direction[0])/.35)),
                    int(math.floor(math.asin(float(np.clip(direction[2],-1,1)))/.35))])
        state.episode_memory.observe_view(dict(id='view:'+camera+':'+','.join(map(str,key)),
            evidence=state.episode_memory.observation_evidence(oid,camera),
            search_result='not_assessed',origin_odom_m=mount[:3,3].tolist(),
            rotation_odom_from_optical=(mount[:3,:3]@OPTICAL_TO_CAMERA).tolist(),
            intrinsics=np.asarray(values['intrinsics']).tolist()),step=step)
    return digest


def episode_contact_region(state, executor, fk, q):
    """A narrow, robot-calibrated finger corridor; never a scene-object mask."""
    target=state.identity_target;entry=state.pick_entry
    goal=getattr(executor,'grasp_goal_local',None)
    if (not target or target.get('verdict')!='confirmed' or state.target_association_unresolved
            or not entry or entry.get('candidate_id')!=target['id'] or goal is None):
        return None
    # Only the intersection of the two inner pad spans may contact the target.
    # Finger link poses and mesh bounds are own-robot calibration, not object size.
    links=fk.links(q);grip_inv=np.linalg.inv(links['gripper_link']);bounds=[]
    from itertools import product
    for side in ('l','r'):
        name=side+'_gripper_finger_link';shape=executor.clearance.geometry[name]['bounds']
        points=np.array(list(product(*zip(shape[0],shape[1]))))
        mount=grip_inv@links[name]
        points=points@mount[:3,:3].T+mount[:3,3];bounds.append((points.min(0),points.max(0)))
    lo=np.array([max(b[0][0] for b in bounds),bounds[0][1][1],max(b[0][2] for b in bounds)])
    hi=np.array([min(b[1][0] for b in bounds),bounds[1][0][1],min(b[1][2] for b in bounds)])
    if np.any(hi<=lo):return None
    return dict(confirmed=True,locked=True,localization_executed=True,
        center_odom_m=(goal[:3,:3]@((lo+hi)/2)+goal[:3,3]).tolist(),
        rotation_odom=goal[:3,:3].tolist(),half_extents_m=((hi-lo)/2).tolist())


# Evaluation-only helpers. These objects never cross the decision boundary.
def jsonable(value: Any) -> Any:
    if is_dataclass(value) and (not isinstance(value, type)):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return str(value)

def scalar(value: Any) -> Any:
    value = jsonable(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, dict)):
        raise ProtocolError('This adapter supports exactly one environment')
    return value

def strict_pick(info):
    keys = ('is_grasped', 'ee_rest', 'robot_rest', 'is_static', 'cumulative_force_within_limit')

    def value(v):
        if isinstance(v, list):
            v = v[0]
        return v
    return all((bool(value(info.get(k, False))) for k in keys)) and float(value(info.get('robot_cumulative_force', float('inf')))) < 5000

def tcp_box_distance(tcp_object, bbox_min_object, bbox_max_object):
    triples = list(zip(tcp_object, bbox_min_object, bbox_max_object))
    if len(triples) != 3 or any((not math.isfinite(float(x)) for values in triples for x in values)):
        raise ValueError('three finite coordinates required')
    if any((lo > hi for _, lo, hi in triples)):
        raise ValueError('invalid box')
    return math.sqrt(sum((max(lo - p, 0, p - hi) ** 2 for p, lo, hi in triples)))

def funnel(samples, accepted_checks, initial_object_z):
    """samples are per-step, including official final step; missing != false.

    object_z is the same target actor-origin world z at reset and every step.
    Distance uses that step's box and TCP. Lift and grasp must coincide.
    """
    if not samples:
        raise ValueError('missing trajectory; mark unscorable')
    required = {'tcp_box_distance_m', 'is_grasped', 'object_z_m', 'strict_success'}
    for sample in samples:
        if not required <= set(sample):
            raise ValueError('incomplete evaluator sample')
        if any((type(sample[k]) is not bool for k in ('is_grasped', 'strict_success'))):
            raise ValueError('official booleans required')
        if not all((math.isfinite(sample[k]) for k in ('tcp_box_distance_m', 'object_z_m'))) or sample['tcp_box_distance_m'] < 0:
            raise ValueError('invalid measurements')
    raw = dict(zip(FUNNEL, [any(accepted_checks), any((s['tcp_box_distance_m'] <= 0.05 for s in samples)), any((s['is_grasped'] for s in samples)), any((s['is_grasped'] and s['object_z_m'] - initial_object_z >= 0.05 for s in samples)), any((s['strict_success'] for s in samples))]))
    return {'raw': raw, 'cumulative': {key: all((raw[k] for k in FUNNEL[:i + 1])) for i, key in enumerate(FUNNEL)}}

def horizon_results(samples):
    successful = [s['step'] for s in samples if s['strict_success']]
    first = min(successful) if successful else None
    return {'first_strict_success_step': first, 'success_within_200': first is not None and first <= 200, 'success_within_600': first is not None and first <= 600, 'horizon_description': 'nonofficial 600-step horizon; 200-step prefix of same run'}


def weekly_accounting(report, entries, state, profile):
    """Evaluation-only counters, with no conversion of API waits into motion."""
    trace=report.get('executor_trace',[])
    steps=report.get('simulator_actions',0)
    sequence=[r['step'] for r in trace]
    if sequence!=list(range(1,steps+1)):
        raise ValueError('each acknowledged control step must occur exactly once')
    settle=sum(r['kind'] in ('base_settle','settling') for r in trace)
    calls=[e for e in entries if e['case_id']==report['case_id']]
    return dict(provider_requests=len(calls),main_requests=sum(e['request_kind']=='main' for e in calls),
        verification_requests=sum(e['request_kind']=='verification' for e in calls),
        tool_decisions=len(report.get('commands',[])),control_steps=steps,
        settling_steps=settle,motion_steps=steps-settle,physics_substeps=steps*5,
        control_hz=20,physics_hz=100,simulation_seconds=steps/20,
        wall_seconds=report['wall_seconds'],
        api_latency_seconds=sum(e.get('latency_seconds',0) for e in calls),
        step_cap=profile.step_cap,request_cap=profile.call_cap,
        extended_horizon=profile.step_cap>600,
        recovery_event=None if state is None else state.recovery_event,
        pick_entry=None if state is None else state.pick_entry,
        target_confirmed_by_model=state is not None and any(r.get('verdict')=='confirmed' and r.get('status')=='completed' for r in state.target_reviews),
        grasp_localization_executed=any((c.get('call') or {}).get('tool') in ('prepare_grasp','replan_grasp') and c['result'].get('steps_used',0)>0 for c in report.get('commands',[])),
        close_command_completed=any((c.get('call') or {}).get('tool')=='set_gripper' and c['call'].get('gripper')==0 and c['result'].get('closure_command_completed') is True for c in report.get('commands',[])),
        committed_usd=str(pick.committed_cost(calls)),
        unresolved_requests=sum(e['status'] in ('reserved','unknown','censored') for e in calls))


def baseline_prefix_results(samples, commands, profile):
    """Score the observed prefix; do not infer a separate shortened rollout."""
    first = min((s['step'] for s in samples if s['strict_success']), default=None)
    decision = 0 if first == 0 else None
    if first is not None and first > 0:
        matching = [i+1 for i, c in enumerate(commands)
                    if c['start_step'] < first <= c['start_step'] + c['result'].get('steps_used', 0)]
        if len(matching) != 1:
            raise ValueError('success step lacks unique executed command')
        decision = matching[0]
    return dict(main_decision_cap=profile.call_cap, physical_step_cap=profile.step_cap,
        first_success_decision=decision,
        success_within_25_decisions_600_steps=first is not None and first <= 600 and decision <= 25,
        success_within_full_budget=first is not None and first <= profile.step_cap and decision <= profile.call_cap,
        prefix_definition='same observed trajectory; both decision <=25 and step <=600; not a separate rollout')

def _component_and_shapes(body):
    import sapien.physx as physx
    if len(body._objs) != 1:
        raise ValueError('body_wrapper_not_single_environment')
    obj = body._objs[0]
    components = [obj] if isinstance(obj, physx.PhysxRigidBaseComponent) else [comp for comp in obj.components if isinstance(comp, physx.PhysxRigidBaseComponent)]
    if len(components) > 1:
        raise ValueError('multiple_rigid_components_in_body')
    component = components[0] if components else None
    return (component, [] if component is None else list(component.get_collision_shapes()))

def diagnose_location(tool_result, triangles_object, object_world, base_world):
    from trimesh.triangles import closest_point
    point = tool_result.get('point_base_m')
    meta = {'evaluation_only': True, 'observation_id': tool_result.get('observation_id'), 'surface_distance_m': None, 'definition': 'unsigned minimum Euclidean distance to all target collision-shape triangle surfaces at this call'}
    if not tool_result.get('valid') or point is None:
        return dict(meta, status='no_valid_returned_point')
    triangles = np.asarray(triangles_object, dtype=float)
    point = np.asarray(point, dtype=float)
    object_world = np.asarray(object_world, dtype=float)
    base_world = np.asarray(base_world, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0 or (point.shape != (3,)) or (object_world.shape != (4, 4)) or (base_world.shape != (4, 4)) or (not all((np.isfinite(x).all() for x in (triangles, point, object_world, base_world)))):
        raise ValueError('invalid evaluator geometry')
    local = (np.linalg.inv(object_world) @ base_world @ np.r_[point, 1])[:3]
    closest = closest_point(triangles, np.repeat(local[None, :], len(triangles), axis=0))
    distance = float(np.linalg.norm(closest - local, axis=1).min())
    return dict(meta, status='scored', surface_distance_m=distance, point_object_eval_only=local.tolist())

def r15_localization_scores(call,result,triangles,entry,turn,step):
    """Evaluation-side only; point and transforms share the command-entry frame."""
    if call['tool']=='locate_points':
        samples=[(row['pixel'],row['result']) for row in result.get('located_points',[])]
    elif call['tool']=='approach_point':
        point=result.get('approach_feedback',{}).get('surface_point_base_m')
        samples=[(call['pixel'],dict(valid=point is not None,point_base_m=point,observation_id=call['observation_id']))]
    else:raise ValueError('Unsupported evaluation tool')
    return [dict(diagnose_location(sample,triangles,entry['object_world'],entry['base_world']),
        turn=turn,step=step,tool=call['tool'],pixel=pixel) for pixel,sample in samples]


def state_differences(before, after, path='state'):
    """Evaluation-only exact diff; do not infer physical meaning or loosen gates."""
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                rows.append({'path': path + '/' + str(key), 'kind': 'missing_key'})
            else:
                rows.extend(state_differences(before[key], after[key], path + '/' + str(key)))
        return rows
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [{'path': path, 'kind': 'length_changed'}]
        return [row for i, (a, b) in enumerate(zip(before, after)) for row in state_differences(a, b, f'{path}/{i}')]
    a, b = (np.asarray(before), np.asarray(after))
    if np.array_equal(a, b):
        return []
    row = {'path': path, 'before_shape': list(a.shape), 'after_shape': list(b.shape), 'before': a.tolist(), 'after': b.tolist()}
    if a.shape == b.shape and a.dtype.kind in 'biuf' and (b.dtype.kind in 'biuf'):
        finite = np.isfinite(a) & np.isfinite(b)
        row['finite_max_abs_delta'] = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)), initial=0))
        row['nonfinite_count'] = int(np.size(finite) - np.count_nonzero(finite))
    return [row]

class AttemptRecording:

    def __init__(self, directory, fps):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.count = 0

    def capture_environment(self, environment):
        """Read the evaluator camera without render() marker/state mutation."""
        before = state_copy(environment.get_state_dict())
        environment.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        images = environment.scene.get_human_render_camera_images()
        if len(images) != 1:
            raise ValueError('expected one frozen evaluator camera')
        self.capture(lambda: next(iter(images.values())))
        after = state_copy(environment.get_state_dict())
        if not state_equal(before, after):
            (self.directory.parent / 'recording-state-diff.eval-only.json').write_text(json.dumps({'frame': self.count - 1, 'differences': state_differences(before, after)}, indent=2), encoding='utf-8')
            raise ValueError('evaluation recording changed simulator state after direct camera capture')

    def capture(self, render):
        from PIL import Image
        frame = render()
        if hasattr(frame, 'detach'):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('invalid evaluation video frame')
        Image.fromarray(frame).save(self.directory / f'{self.count:06d}.png')
        self.count += 1

def bind_initial(path, binding):
    """All four conditions must start from the same frozen physical state."""
    import numpy as np
    if path.exists():
        prior = json.loads(path.read_text())
        for key in ('uid', 'spawn_index', 'seed'):
            if prior[key] != binding[key]:
                raise ValueError('paired spawn metadata mismatch')
        for key in ('qpos', 'base_world', 'target_world'):
            if not np.allclose(prior[key], binding[key], rtol=0, atol=1e-06):
                raise ValueError('paired physical initial state mismatch: ' + key)
    else:
        save(path, binding)


def memory_ab_initial_snapshot(case_id, row, seed, qpos, qvel, root_pose, environment_state, runtime_identity):
    """Evaluation-side snapshot only. Caller reads a natural reset; never sets state."""
    def plain(value):
        if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        if hasattr(value,'tolist'):return value.tolist()
        return value
    snapshot=dict(metadata=dict(case_id=case_id,uid=row['uid'],seed=seed,spawn_index=row['spawn_index'],
        source_sha256=pick.ab_source_fingerprint(),runtime_sha256=hashlib.sha256(canonical(runtime_identity)).hexdigest()),
        state=dict(qpos=plain(qpos),qvel=plain(qvel),root_pose=plain(root_pose),environment=plain(environment_state)))
    pick.ab_initial_check(snapshot,snapshot)
    return snapshot

AB_REFERENCE_SCOPE = 'sol-memory-reference-initialization-v1'
AB_REFERENCE_CAPS = dict(workers=6, worker_seconds=600, total_seconds=3600,
                         api_calls=0, controller_actions=0, automatic_retries=0)


def validate_ab_reference(auth, args, *, worker=True):
    """Separate zero-provider authorization; cannot authorize an A/B episode."""
    cases = [r['case_id'] for r in pick.FROZEN['memory_ab']['cases']]
    if (auth.get('scope') != AB_REFERENCE_SCOPE or auth.get('status') != 'approved'
            or auth.get('case_ids') != cases or args.case_id not in cases
            or args.profile != 'Sol-session-v1' or auth.get('resource_caps') != AB_REFERENCE_CAPS
            or auth.get('source_fingerprint') != pick.ab_source_fingerprint()
            or not auth.get('scheduling_clearance_confirmed')
            or time.time() >= auth.get('expires_at_epoch', 0)
            or getattr(args, 'mode', None) not in ('ab-reference', 'ab-reference-supervise')):
        raise ValueError('AB reference-only scope/source/caps gate')
    if worker and (args.mode != 'ab-reference' or os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU):
        raise ValueError('AB reference worker/device gate')
    root = Path(auth['output_root']).resolve()
    if args.output.resolve() != (root / args.case_id).resolve():
        raise ValueError('AB reference exact output path')
    if sha(args.roster) != auth['roster_sha256']:
        raise ValueError('AB reference roster changed')
    rows = read(args.roster)['dev']
    if [r['case_id'] for r in rows] != cases:
        raise ValueError('AB reference six-case roster required')
    for row in rows:
        expected_seed = 20260930 + int(row['case_id'].rsplit('-', 1)[1])
        if row.get('ab_seed') != expected_seed:
            raise ValueError('AB reference must use actual unchanged A/B seed branch')
    if not auth.get('asset_sha256'):
        raise ValueError('AB reference assets missing')
    validate_candidate_sources(auth, args.code)
    for name, expected in auth['source_sha256'].items():
        if sha(args.code / name) != expected:
            raise ValueError('AB reference source changed: ' + name)


def capture_ab_reference(output, case_id, row, seed, agent, environment_state, runtime_identity, info):
    """Read the natural reset; no actions, model calls or state writes."""
    if int(scalar(info['subtasks_steps_left'])) != 600:
        raise ValueError('AB reference requires 600 remaining steps')
    snapshot = memory_ab_initial_snapshot(case_id, row, seed,
        array(agent.robot.qpos)[0].astype(float), array(agent.robot.qvel)[0].astype(float),
        np.asarray(agent.base_link.pose[0].sp.to_transformation_matrix()), environment_state, runtime_identity)
    _initialization_json_exclusive(Path(output)/'initial-reference.eval-only.json', snapshot)
    return snapshot


def supervise_ab_references(args):
    """One durable serial batch; any unknown/failure stops it without retry."""
    import fcntl
    import subprocess
    import signal
    auth = read(args.authorization)
    validate_ab_reference(auth, args, worker=False)
    root = Path(auth['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (args.lab_root/'runs/arm-capability.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _initialization_json_exclusive(root/'launch-claim.json', dict(pid=os.getpid(),
            authorization_sha256=sha(args.authorization), started_epoch=time.time()))
        ledger = dict(status='running', scope=AB_REFERENCE_SCOPE, batch_id=auth['batch_id'],
                      resource_caps=AB_REFERENCE_CAPS, entries=[], api_calls=0, controller_actions=0)
        save(root/'ledger.json', ledger)
        total = 0.
        references = {}
        for case in auth['case_ids']:
            if total >= 3600 or time.time() >= auth['expires_at_epoch']:
                raise ValueError('reference batch time budget expired')
            gpu = subprocess.check_output(['nvidia-smi', '-i', GPU,
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15)
            memory, utilization = [int(x.strip()) for x in gpu.strip().split(',')]
            if memory >= 1024 or utilization != 0:
                raise ValueError('GPU 1 not idle for next reference')
            entry = dict(case_id=case, status='launch_intent', started_epoch=time.time(), gpu_preflight=gpu.strip())
            ledger['entries'].append(entry)
            save(root/'ledger.json', ledger)
            _initialization_json_exclusive(root/(case+'.launch-intent.json'),
                dict(authorization_sha256=sha(args.authorization), case_id=case, index=len(ledger['entries'])-1))
            command = [sys.executable, '-B', '-m', 'astra_pick.run_case', '--mode', 'ab-reference',
                '--profile', args.profile, '--case-id', case, '--initialization-lock-fd', str(lock.fileno())]
            for name, value in dict(authorization=args.authorization, roster=args.roster, code=args.code,
                                   lab_root=args.lab_root, output=root/case, ledger=root/'ledger.json').items():
                command += ['--'+name.replace('_','-'), str(value)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=str(args.code/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for key in list(env):
                if any(x in key.upper() for x in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN')):
                    env.pop(key)
            began = time.monotonic()
            with (root/(case+'.worker.log')).open('xb') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lock.fileno(),))
                entry.update(status='running', pid=process.pid)
                save(root/'ledger.json', ledger)
                try:
                    process.wait(timeout=min(600, 3600-total))
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    entry['status'] = 'unknown_or_timeout'
                    raise
                finally:
                    elapsed = time.monotonic()-began
                    total += elapsed
                    entry.update(wall_seconds=elapsed, returncode=process.returncode)
                    ledger['worker_seconds'] = total
                    save(root/'ledger.json', ledger)
            result = read(root/case/'result.json')
            reference_path = root/case/'initial-reference.eval-only.json'
            if (process.returncode != 0 or elapsed > 600 or total > 3600
                    or result.get('status') != 'reference_captured' or not result.get('environment_closed')
                    or result.get('api_requests') != 0 or result.get('simulator_actions') != 0
                    or result.get('source_fingerprint') != auth['source_fingerprint']):
                raise ValueError('reference worker failed; no retry')
            snapshot = read(reference_path)
            pick.ab_initial_check(snapshot, snapshot)
            references[case] = snapshot
            entry.update(status='completed', reference_sha256=sha(reference_path), result_sha256=sha(root/case/'result.json'))
            save(root/'ledger.json', ledger)
        _initialization_json_exclusive(root/'references.eval-only.json', references)
        ledger.update(status='completed', references_sha256=sha(root/'references.eval-only.json'))
        save(root/'ledger.json', ledger)
        return ledger
    except BaseException as error:
        if 'ledger' in locals():
            ledger.update(status='stopped_no_retry', error_type=type(error).__name__, error_message=str(error))
            save(root/'ledger.json', ledger)
        raise
    finally:
        lock.close()


def configure_runtime(lab_root):
    runtime = lab_root / 'src/official-mshab-runtime'
    for package in ('mshab', 'mani_skill'):
        if any((key == package or key.startswith(package + '.') for key in sys.modules)):
            raise ValueError('runtime pin must precede environment imports')
    sys.path[:0] = [str(runtime / 'mshab'), str(runtime / 'ManiSkill')]

def configure_arm_runtime(lab_root, asset_root=None):
    lab_root = Path(lab_root)
    configure_runtime(lab_root)
    os.environ['MS_ASSET_DIR'] = str(Path(asset_root) if asset_root is not None else lab_root / 'assets')
    from mani_skill import ASSET_DIR
    if asset_root is not None:
        if Path(ASSET_DIR).resolve() != (Path(asset_root)/'data').resolve() or not Path(ASSET_DIR).is_dir():
            raise ValueError('C2 asset directory mismatch')
        return Path(ASSET_DIR)
    return validate_asset_directory(lab_root, ASSET_DIR)

def validate_asset_directory(lab_root, actual):
    expected = Path(lab_root) / 'assets/data'
    actual = Path(actual)
    if actual.resolve() != expected.resolve():
        raise ValueError('fixed ManiSkill asset directory mismatch')
    if not actual.is_dir():
        raise FileNotFoundError('fixed ManiSkill asset directory missing')
    return actual


# Authorized single-case runner; environment imports stay inside live_run.
R31_STAGED_SCOPE = 'c2-r31-staged-20261008'
FULL_B_PREFLIGHT_SCOPE = 'sol-memory-full-b-preflight-001-v1'
ALIGNED_PREFLIGHT_SCOPE = 'sol-aligned-002-preflight-v1'


def validate_aligned_preflight(auth,args,*,worker=True):
    expected=dict(scope=ALIGNED_PREFLIGHT_SCOPE,status='approved',case_ids=['baseline-002'],
        execution_profile=pick.SOL_PICK20_ALIGNED.name,physical_steps_cap=600,
        gpu1_process_seconds_cap=1200,provider_requests_cap=0,automatic_retries=0,
        initializations_cap=1,prefix_steps_cap=300,controller_version='full-B-r4-plus-task-rules-v1')
    if any(type(auth.get(k)) is not type(v) or auth[k]!=v for k,v in expected.items()):
        raise ValueError('Aligned preflight exact scope mismatch')
    if args.mode not in ('aligned-preflight','aligned-preflight-supervise') or args.case_id!='baseline-002' or args.profile!=pick.SOL_PICK20_ALIGNED.name:
        raise ValueError('Aligned preflight mode/case/profile')
    if time.time()>=auth['expires_at_epoch'] or not auth.get('scheduling_clearance_confirmed') or args.condition!='V-mobile':
        raise ValueError('Aligned preflight time/scheduling')
    expected_output=Path(auth['output_root'])/args.case_id if worker else Path(auth['output_root'])
    if args.output.resolve()!=expected_output.resolve() or sha(args.roster)!=auth['roster_sha256']:
        raise ValueError('Aligned preflight output/roster')
    for name,digest in auth['source_sha256'].items():
        if sha(args.code/name)!=digest:raise ValueError('Aligned preflight source changed: '+name)
    if worker:
        lock=args.lab_root/'runs/arm-capability.lock';fd=args.initialization_lock_fd
        if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('BVI_NO_API')!='1':
            raise ValueError('Aligned preflight requires GPU1 and API denial')
        if type(fd) is not int or (os.fstat(fd).st_dev,os.fstat(fd).st_ino)!=(lock.stat().st_dev,lock.stat().st_ino):
            raise ValueError('Aligned preflight inherited lock required')


def validate_aligned_preflight_inputs(auth,args):
    """Read-only downstream contracts, checked before any GPU/SDK startup."""
    validate_candidate_sources(auth,args.code)
    def bound(name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or name not in auth['source_sha256']:
            raise ValueError('Aligned preflight input is not source-bound')
        path=args.code/relative
        if sha(path)!=auth['source_sha256'][name]:raise ValueError('Aligned preflight input changed')
        return read(path)
    script=bound(auth['preflight_script'])
    if script.get('case_id')!='baseline-002' or not isinstance(script.get('prefix'),list) or not script['prefix']:
        raise ValueError('Aligned preflight script identity/prefix')
    for call in script['prefix']+[script['grasp']]:pick.validate_call(call,pick.SOL_PICK20)
    if script['grasp']['tool']!='move_to' or any(c['tool'] not in ('look','move_base','move_to','move_eef_chunk') for c in script['prefix']):
        raise ValueError('Aligned preflight script phase/tool mismatch')
    reference=bound(auth['initial_references']['baseline-002'])
    if 'qpos' not in reference or not ('root_pose' in reference or 'base_world' in reference):
        raise ValueError('Aligned preflight initial reference missing robot state')
    rows=[row for row in read(args.roster)['dev'] if row['case_id']=='baseline-002']
    if len(rows)!=1 or any(k not in rows[0] for k in ('seed','spawn_index','uid')):
        raise ValueError('Aligned preflight roster identity missing')
    for key in ('seed','spawn_index','uid'):
        if key in reference and rows[0][key]!=reference[key]:raise ValueError('Aligned preflight roster/reference mismatch')
    return script


def execute_aligned_preflight(ex,script,save_event):
    """One archived command prefix, one original goal; no policy/API/retry."""
    if ex.version!='full-B-r4-grasp-alignment-v1' or ex.steps!=0 or ex.max_steps!=600:
        raise ValueError('Fresh aligned executor required')
    events=[]
    def invoke(label,function):
        before=ex.steps;result=function()
        events.append(dict(tool=label,start_step=before,end_step=ex.steps,result=result,
            rest_reserved=ex.max_steps-ex.steps>=122 if label!='return_to_rest' else None))
        save_event(events)
        if ex.steps>600:raise RuntimeError('Preflight physical cap')
        return result
    def finish(reason,convergence=False,rest=False):
        return dict(passed=convergence and rest,reason=reason,convergence_passed=convergence,
            rest_passed=rest,events=events,api_requests=0)
    ex.phase_step_ceiling=300
    try:
        for call in script['prefix']:
            if call['tool'] not in ('look','move_base','move_to','move_eef_chunk','set_gripper'):
                raise ValueError('Unexpected physical prefix tool')
            result=invoke(call['tool'],lambda call=call:pick.dispatch(ex,call,None,profile=pick.SOL_PICK20_ALIGNED)[1])
            if result.get('reason')!='completed' or result.get('accepted') is not True:
                return finish('prefix_not_completed')
    finally:ex.phase_step_ceiling=None
    target=script['grasp']['poses'][0]
    result=invoke('prepare_grasp',lambda:ex.prepare_grasp(pick.pose(target['position'],target['quaternion_xyzw']),reason=script['grasp']['note']))
    converged=result.get('reason')=='completed' and result.get('arrived') is True
    if not converged and result.get('reason') not in ('grasp_alignment_timeout','compliance_blocked'):return finish('localization_stopped')
    if converged:
        result=invoke('set_gripper',lambda:ex.set_gripper(0))
        if result.get('reason') not in ('completed','grasp_alignment_timeout','compliance_blocked','close_alignment_failed'):
            return finish('close_stopped',convergence=True)
        converged=result.get('reason')=='completed' and result.get('closure_command_completed') is True
    result=invoke('return_to_rest',ex.return_to_rest)
    rest=result.get('reason')=='completed'
    return finish('completed' if converged and rest else 'convergence_or_rest_failed',converged,rest)


def finish_aligned_preflight(ex,report,env):
    """Always attempt environment close, preserving the first failure."""
    def failed(kind,exc):
        report[kind]=dict(error_type=type(exc).__name__,error_message=str(exc))
        report.setdefault('error_type',type(exc).__name__)
        report.setdefault('error_message',str(exc))
        report['status']='infrastructure_censored'
    if ex is not None:
        if hasattr(ex,'controller_diagnostics'):
            report['controller_diagnostics']=ex.controller_diagnostics
            report['alignment_samples']=ex.alignment_samples
        try:finish_memory_recording(ex,report)
        except Exception as exc:failed('recording_error',exc)
    if env is not None:
        try:
            env.close();report['environment_closed']=True
        except Exception as exc:
            report['environment_closed']=False;failed('close_error',exc)


def validate_full_b_preflight(auth,args,*,worker=True):
    if (auth.get('scope')!=FULL_B_PREFLIGHT_SCOPE or args.mode!='full-b-preflight'
            or auth.get('status')!='approved' or auth.get('case_ids')!=['baseline-001']
            or args.case_id!='baseline-001' or args.profile!='Sol-session-v1'
            or auth.get('physical_steps_cap')!=160 or auth.get('gpu1_process_seconds_cap')!=600
            or auth.get('provider_requests_cap')!=0 or auth.get('automatic_retries')!=0
            or auth.get('controller_version')!='full-B-r4-plus-task-rules-v1'):
        raise ValueError('Full B preflight exact scope mismatch')
    if time.time()>=auth['expires_at_epoch']:raise ValueError('Full B preflight expired')
    if worker:
        if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('GPU1 lab worker required')
        fd=args.initialization_lock_fd
        if type(fd) is not int or (os.fstat(fd).st_dev,os.fstat(fd).st_ino)!=((args.lab_root/'runs/arm-capability.lock').stat().st_dev,(args.lab_root/'runs/arm-capability.lock').stat().st_ino):
            raise ValueError('Preflight inherited scheduler lock required')
    if not auth.get('scheduling_clearance_confirmed') or args.condition!='V-mobile':raise ValueError('Scheduling/condition mismatch')
    if args.output.resolve()!=(Path(auth['output_root'])/'preflight').resolve():raise ValueError('Preflight output mismatch')
    if sha(args.roster)!=auth['roster_sha256']:raise ValueError('Preflight roster changed')
    for name,digest in auth['source_sha256'].items():
        if sha(args.code/name)!=digest:raise ValueError('Preflight source changed: '+name)


def execute_memory_preflight(ex,qpos,fk,save_event):
    """One bounded no-provider script; same online tools, no retries."""
    if ex.steps!=0 or ex.max_steps!=600:raise ValueError('Fresh600-step executor required')
    ex.max_steps=160  # calibration-only aggregate action cap, not scored AB horizon
    target=fk.tcp(qpos()).copy()
    commands=[('check_path',lambda:ex.check_path([target])),
              ('move_to',lambda:ex.move_to(target,max_steps=10)),
              ('look',lambda:ex.look(float(qpos()[4]),float(qpos()[6]))),
              ('move_base',lambda:ex.move_base(0.,0.)),
              ('open',lambda:ex.set_gripper(1)),
              ('close',lambda:ex.set_gripper(0)),
              ('return_to_rest',ex.return_to_rest)]
    events=[]
    for name,call in commands:
        before=ex.steps;result=call()
        event=dict(tool=name,start_step=before,end_step=ex.steps,result=result)
        events.append(event);save_event(events)
        if ex.steps>160:raise RuntimeError('Preflight action limit violation')
        if result.get('accepted') is not True or (name!='check_path' and result.get('reason')!='completed'):
            return dict(passed=False,reason='tool_not_completed',events=events)
    return dict(passed=True,reason='all_tools_completed',events=events)


def validate(auth, args, *, worker=True):
    if args.mode in ('aligned-preflight','aligned-preflight-supervise') or auth.get('scope')==ALIGNED_PREFLIGHT_SCOPE:
        return validate_aligned_preflight(auth,args,worker=worker)
    if auth.get('scope') == pick.TRANSPORT_SCOPE or getattr(args, 'profile', None) in (pick.SOL_TRANSPORT_A.name, pick.SOL_TRANSPORT_B.name):
        raise ValueError('Transport CPU candidate: native launch awaits semantic, asset and paired-state gates')
    if getattr(args, 'profile', None) == pick.SOL_HANDOFF.name:
        raise ValueError('Sol-handoff-v1 is a CPU candidate; freeze the multistage task and full-B binding before live execution')
    if args.mode=='full-b-preflight' or auth.get('scope')==FULL_B_PREFLIGHT_SCOPE:
        return validate_full_b_preflight(auth,args,worker=worker)
    reference_mode = getattr(args, 'mode', None) in ('ab-reference', 'ab-reference-supervise')
    if reference_mode or auth.get('scope') == AB_REFERENCE_SCOPE:
        validate_ab_reference(auth, args, worker=worker)
        return
    init_only = getattr(args, 'mode', None) == 'init-only'
    if (auth.get('scope') == pick.R33_INIT_ONLY_SCOPE) != init_only:
        raise ValueError('initialization-only scope and mode must be paired')
    if 'replay_initial_anchor' in auth and auth.get('scope') not in (pick.R33_REPLAY_SCOPE,pick.R33_STAGED_SCOPE,pick.R33_INIT_ONLY_SCOPE):
        raise ValueError('actual-anchor option is forbidden in legacy scope')
    if init_only:
        pick.validate_init_only_scope(auth,pick.PROFILES[args.profile],case_id=args.case_id,observation=args.initialization_observation)
        if args.output.resolve() != (Path(auth['output_root'])/args.case_id/args.initialization_observation).resolve():
            raise ValueError('initialization-only exact output identity')
    elif getattr(args, 'mode', None) == 'l1-replay':
        profile = pick.PROFILES[args.profile]
        if profile in (pick.C2_R32, pick.C2_R32_FAR,pick.C2_R33,pick.C2_R33_FAR) or auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
            pick.validate_r32_replay_scope(auth, profile, case_id=args.case_id)
            replay_executor_config(auth, profile, args.replay_arm)
            if getattr(args, 'replay_group', 'L1-A') not in ('L1-A','L1-B'):
                raise ValueError('r32 replay group')
            if getattr(args, 'replay_group', 'L1-A') == 'L1-B':
                l1b_replay_entry(args, auth)
        elif profile in (pick.C2_R31, pick.C2_R31_FAR) or auth.get('scope') == 'c2-r31-l1-replay-20261008':
            pick.validate_r31_replay_scope(auth, profile, case_id=args.case_id)
            replay_executor_config(auth, profile, args.replay_arm)
        else:
            pick.validate_r3_scope(auth, profile, replay=True, case_id=args.case_id)
    else:
        profile = pick.PROFILES[args.profile]
        if profile in (pick.C2_R33, pick.C2_R33_FAR) or auth.get('scope') == pick.R33_STAGED_SCOPE:
            pick.validate_r33_paid_scope(auth, profile, case_id=args.case_id)
            if auth.get('stage') not in ('L3', 'L4'):
                raise ValueError('r33 physical entry requires L3 or L4; L2 is API-only')
        elif profile in (pick.C2_R31, pick.C2_R31_FAR) or auth.get('scope') == R31_STAGED_SCOPE:
            pick.validate_r31_paid_scope(auth, profile, case_id=args.case_id)
            if auth.get('stage') not in ('L3', 'L4'):
                raise ValueError('r31 physical entry requires L3 or L4; L2 is API-only')
        elif auth.get('scope') in (pick.FROZEN['memory_ab']['scope'], pick.FROZEN['task_handoff']['scope']):
            pick.validate_session_scope(auth, profile, require_start=False)
        else:
            pick.validate_scope(auth, profile)
    if auth.get('scope') == 'baseline20-supervised-20261006':
        index = auth['case_ids'].index(args.case_id)
        expected_profile = ('C2-r2.5' if index < 10 else 'C2-r2.5-far') if auth.get('system_revision') == 'C2-r2.5' else ('C2-baseline-prep' if index < 10 else 'C2-baseline-far-prep')
        if args.profile != expected_profile:
            raise ValueError('baseline distance profile mismatch')
    if (worker and (os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU)) or args.case_id not in auth['case_ids'] or (args.condition != 'V-mobile') or (not auth.get('scheduling_clearance_confirmed')):
        raise ValueError('lab worker authorization/device/case gate')
    if sha(args.roster) != auth['roster_sha256']:
        raise ValueError('roster changed')
    if not args.output.resolve().is_relative_to(Path(auth['output_root']).resolve()):
        raise ValueError('output outside approved root')
    for name, digest in auth['source_sha256'].items():
        if sha(args.code / name) != digest:
            raise ValueError('source changed: ' + name)
    if auth.get('checkpoint_branch'):
        checkpoint_branch_seed(auth, pick.PROFILES[args.profile])
    elif auth.get('cold_resume'):
        cold_resume_seed(auth, pick.PROFILES[args.profile])

def validate_candidate_sources(auth, code):
    """The original source gate must bind the candidate, not authorize old code."""
    root=Path(__file__).parent
    required={'src/astra_pick/'+str(p.relative_to(root)).replace('\\','/')
              for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json')}
    if not required <= set(auth['source_sha256']):
        raise ValueError('candidate source manifest incomplete; frozen experiment authorization is not transferable')
    for name in required:
        if sha(Path(code)/name)!=auth['source_sha256'][name]:raise ValueError('candidate source changed')


def record_final_anchor_reset(output, auth, case_id, report, nominal, qpos, root_pose, state, *, snapshot_writer):
    """One read-only comparison of the existing reset sample; no simulator method call."""
    anchor = pick.replay_initial_anchor(auth, case_id)
    if anchor is None or nominal != anchor['nominal']:
        raise ValueError('selected actual anchor does not match original nominal reference')
    report['initial_anchor_eval_only'] = anchor['binding']
    for value,shape in ((array(qpos),(15,)),(array(root_pose),(7,))):
        if np.asarray(value).shape!=shape or not np.isfinite(value).all():
            raise ValueError('actual anchor reset nonfinite or shape')
    for name in anchor['reference']['objects']:
        actor=array(state['actors'].get('env-0_'+name, []))
        if actor.shape!=(1,13) or not np.isfinite(actor).all():
            raise ValueError('actual anchor reset actor missing/nonfinite/shape')
    actual = dict(qpos=array(qpos), root_pose=array(root_pose))
    expected = dict(qpos=nominal['qpos'], root_pose=nominal['root_pose'])
    for name, value in nominal['objects'].items():
        expected[name] = value
        actual[name] = array(state['actors']['env-0_'+name])[0][:7]
    checks = {name: bool(np.allclose(actual[name], value, rtol=0, atol=1e-6)) for name,value in expected.items()}
    save(Path(output)/'nominal-reset-diagnostic.eval-only.json', dict(
        decision_role='diagnostic_only_not_selected_reference', binding=anchor['binding'], checks=checks))
    record_reset_comparison(output, anchor['reference'], qpos, root_pose, state, snapshot_writer=snapshot_writer)


def record_reset_comparison(output, archived, qpos, root_pose, state, *, snapshot_writer):
    """Retain actual reset evidence before the unchanged full-scene equality gate.

    All inputs/outputs are evaluator-only. No reset, actor mutation or retry occurs.
    """
    actual = dict(qpos=array(qpos), root_pose=array(root_pose))
    reference = dict(qpos=archived['qpos'], root_pose=archived['root_pose'])
    for name, pose in archived['objects'].items():
        reference[name] = pose
        actual[name] = array(state['actors']['env-0_'+name])[0][:7]
    checks, fields = {}, {}
    def finite_json(value):
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value if math.isfinite(value) else str(value)
    for name, expected in reference.items():
        expected, observed = np.asarray(expected), actual[name]
        checks[name] = bool(np.allclose(expected, observed, rtol=0, atol=1e-6))
        with np.errstate(invalid='ignore', over='ignore'):
            residual = observed - expected
        fields[name] = dict(reference=finite_json(expected.tolist()), actual=finite_json(observed.tolist()),
            residual=finite_json(residual.tolist()),
            reference_finite=np.isfinite(expected).tolist(), actual_finite=np.isfinite(observed).tolist(),
            max_abs_residual=float(np.max(abs(residual))) if np.isfinite(residual).all() else None,
            failed_components=np.argwhere(~np.isclose(expected, observed, rtol=0, atol=1e-6)).tolist(),
            passed=checks[name])
    save(output/'baseline-reset-check.eval-only.json', checks)
    diagnostics = dict(evaluation_only=True, rtol=0, atol=1e-6, fields=fields)
    save(output/'baseline-reset-diagnostics.eval-only.json', diagnostics)
    snapshot = output/'reset-state.eval-only.pt'
    try:
        snapshot_writer(state_copy(state), snapshot)
        diagnostics['snapshot'] = dict(path=snapshot.name, sha256=sha(snapshot), status='saved')
    except Exception as error:
        diagnostics['snapshot'] = dict(status='unavailable', error_type=type(error).__name__)
    save(output/'baseline-reset-diagnostics.eval-only.json', diagnostics)
    if not all(checks.values()):
        raise ValueError('baseline differs from frozen initial state')
    return checks


def record_replay_attempt_identity(args, auth, report):
    """Bind requested replay identity before environment creation, not completion."""
    if auth.get('scope') not in (pick.R32_REPLAY_SCOPE, pick.R33_REPLAY_SCOPE):
        return
    group = getattr(args, 'replay_group', 'L1-A')
    if group == 'L1-B':
        row = l1b_replay_entry(args, auth)
        snap = row['snapshots']['precommand']
        source = dict(result_sha256=row['source_result']['sha256'], snapshot_sha256=snap['sha256'],
            prefix_steps=row['start_step'], source_path=row['source_result']['path'],
            l1b_evidence_sha256=auth['l1b_sources_and_scores']['sha256'],
            command_sha256=row['command_sha256'], target_list_sha256=row['target_list_sha256'])
        snapshot_path = snap['path']
    elif group == 'L1-A':
        if sha(args.replay_roster) != auth['replay_roster_sha256']:
            raise ValueError('L1 command roster changed')
        rows = [r for r in read(args.replay_roster)['roster']
                if 'baseline-'+r['case'] == args.case_id and r['turn'] == args.replay_turn]
        if len(rows) != 1:
            raise ValueError('L1 command/arm binding')
        row = rows[0]
        source = dict(result_sha256=row['result_sha256'], snapshot_sha256=row['snapshot_sha256'],
            prefix_steps=row['prefix_steps'], source_path=row['result_path'])
        snapshot_path = row['precommand_snapshot']
    else:
        raise ValueError('r32 replay group')
    flags = replay_executor_config(auth, pick.PROFILES[args.profile], args.replay_arm)
    if (sha(args.code/source['source_path']) != source['result_sha256']
            or sha(args.code/snapshot_path) != source['snapshot_sha256']):
        raise ValueError('L1 archived evidence changed')
    report.update(replay=True, replay_group=group, replay_arm=args.replay_arm, replay_turn=args.replay_turn,
        replay_source=source, system_revision=auth['system_revision'],
        source_manifest_sha256=auth['source_manifest_sha256'], authorization_sha256=sha(args.authorization),
        replay_variant=auth['replay_variant'], effective_controller_flags=flags,
        replay_identity_status='validated_requested_attempt_not_execution')

def s2_robot_snapshot(executor):
    """Whitelist robot-only values for static query and command-frame binding."""
    port=getattr(executor,'ports',None)
    measured=port.observe() if port is not None else {}
    q=measured.get('qpos')
    if q is None:q=executor.read_qpos()
    B=measured.get('local_from_base')
    if B is None:
        x,y,yaw=map(float,executor.odometry());B=np.eye(4)
        B[:2,:2]=[[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]]
        B[:2,3]=[x,y]
    q=np.asarray(q,float);B=np.asarray(B,float)
    if q.ndim!=1 or B.shape!=(4,4) or not np.isfinite(q).all() or not np.isfinite(B).all():
        raise ValueError('Invalid S2 measured robot snapshot')
    snapshot=dict(step=int(executor.steps),qpos=q.tolist(),local_from_base=B.tolist(),
        closed=bool(getattr(executor,'closed',False)),pending=copy.deepcopy(getattr(executor,'pending',None)))
    velocity=measured.get('qvel')
    if velocity is not None:
        velocity=np.asarray(velocity,float)
        if velocity.shape!=q.shape or not np.isfinite(velocity).all():raise ValueError('Invalid S2 velocity snapshot')
        snapshot['qvel']=velocity.tolist()
    for name in ('last_goal_local','grasp_goal_local'):
        value=getattr(executor,name,None)
        if value is not None:
            value=np.asarray(value,float)
            if value.shape!=(4,4) or not np.isfinite(value).all():raise ValueError('Invalid S2 controller goal snapshot')
            snapshot[name]=value.tolist()
    if hasattr(executor,'grasp_ready'):snapshot['grasp_ready']=bool(executor.grasp_ready)
    return snapshot


def _s2_journal(state,binding):
    session=getattr(state,'response_session',None)
    path=None if session is None or session.path is None else Path(session.path).with_name('s2-execution-journal.json')
    value=getattr(state,'s2_execution_journal',None)
    if path is not None and path.exists():
        persisted=read(path)
        if value is not None and canonical(value)!=canonical(persisted):
            raise RuntimeError('S2 execution journal diverged; no replay')
        value=persisted
    if value is None:value=dict(protocol='sol-pick-command-journal-v1',binding=copy.deepcopy(binding),plans=[])
    if value.get('protocol')!='sol-pick-command-journal-v1' or not isinstance(value.get('plans'),list):
        raise RuntimeError('Invalid S2 execution journal; no recovery')
    if value.get('binding')!=binding:raise RuntimeError('S2 execution journal case/attempt/authorization mismatch')
    state.s2_execution_journal=value
    return path,value


def _s2_persist_journal(state,path,journal):
    # The caller must not dispatch when this durable write fails.
    if path is not None:save(path,journal)
    state.s2_execution_journal=copy.deepcopy(journal)


def _s2_transform_call(call,anchor,current):
    """Convert explicitly proposed poses, using measured robot odometry only."""
    result=copy.deepcopy(call)
    transform=np.linalg.inv(np.asarray(current,float))@np.asarray(anchor,float)
    for item in result.get('poses',[]):
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
        T[:3,3]=item['position'];T=transform@T
        item['position']=T[:3,3].tolist()
        item['quaternion_xyzw']=Rotation.from_matrix(T[:3,:3]).as_quat().tolist()
    return result


def _s2_unknown_result(value):
    """Unknown stays unknown; this predicate selects a return, never a route."""
    if isinstance(value,dict):
        if value.get('accepted') is False or value.get('valid') is False:return True
        if value.get('reachable','absent') is None or value.get('complete') is False:return True
        if value.get('path_complete') is False:return True
        if value.get('unknown_regions'):return True
        for key in ('status','reason','clearance_status','verification_status'):
            text=str(value.get(key,''))
            if any(term in text for term in ('unknown','unresolved','insufficient','not_verified','unverified')):return True
        return any(_s2_unknown_result(v) for v in value.values())
    if isinstance(value,list):return any(_s2_unknown_result(v) for v in value)
    return False


def s2_unknown_scope(call,result):
    """Exact limitation identity; provenance is checked separately before reuse."""
    ignored={'observation_id','frame_id','query_seconds','steps_remaining','note','task_state','hindsight'}
    def content(value):
        if isinstance(value,dict):return {k:content(v) for k,v in value.items() if k not in ignored}
        if isinstance(value,list):return [content(v) for v in value]
        return value
    return 'query:'+hashlib.sha256(canonical(dict(call=content(call),result=content(result)))).hexdigest()


def _s2_unknown_acknowledged(plan,call,result,journal,current,sensors):
    scope=s2_unknown_scope(call,result)
    for known in plan['known_unknowns']:
        if known['scope']!=scope:continue
        for previous in journal['plans'][:-1]:
            if (previous.get('source_frame_id')!=known['evidence_ref']
                    or previous.get('target_ref')!=plan['target_ref']
                    or previous.get('entry_robot')!=current or previous.get('source_sensor_sha256')!=sensors):continue
            for row in (previous.get('result') or {}).get('entries',[]):
                if row['call'] and _s2_unknown_result(row['result']) and s2_unknown_scope(row['call'],row['result'])==scope:
                    return True
    return False


def s2_evaluate_plan(state,bundle,call,executor,depth,profile,*,sink=None):
    """A static set of explicit local checks. No semantic review or candidate search."""
    args=call if 'query_id' in call else call['args']
    snapshot=s2_robot_snapshot(executor)
    start_hash=hashlib.sha256(canonical(snapshot)).hexdigest()
    candidates=[];queries=0;started=time.monotonic()
    base=pick.s2_base_profile(profile)
    allowed={'check_path','check_clearance','locate_point','locate_points','measure'}
    for candidate in args['candidates']:
        checks=[]
        for check in candidate['checks']:
            if check['tool'] not in allowed:raise ValueError('S2 joint query is not a local read-only tool')
            command=dict(command_id=check['check_id'],tool=check['tool'],args=check['args'],
                note=call.get('note','Evaluate this explicitly proposed candidate from current robot evidence.'),
                dependency='independent')
            child=pick.s2_command_call(command,profile)
            before=s2_robot_snapshot(executor)
            actual,result=pick.dispatch(executor,child,depth,state.observation_ids,base,
                sink=sink,observation_id=bundle['observation']['observation_id'])
            after=s2_robot_snapshot(executor)
            if before!=after or after!=snapshot or result.get('steps_used',0)!=0:
                raise RuntimeError('S2 read-only query changed robot state; no continuation')
            queries+=1
            checks.append(dict(check_id=check['check_id'],tool=check['tool'],result=result))
        candidates.append(dict(candidate_id=candidate['candidate_id'],checks=checks))
    return dict(accepted=True,readonly=True,reason='joint_evaluation_completed',steps_used=0,
        steps_remaining=executor.max_steps-executor.steps,query_id=args['query_id'],candidates=candidates,
        query_count=queries,query_seconds=time.monotonic()-started,
        binding=dict(observation_id=bundle['observation']['observation_id'],
            frame_id=bundle['observation']['observation_id'],start_state_hash=start_hash,
            route_hash=hashlib.sha256(canonical(args['candidates'])).hexdigest()),
        limitations=['Robot IK is not scene clearance.','Unobserved space remains unknown.',
            'No candidate is generated, ranked, selected or executed.'])


def execute_s2_plan(state,bundle,plan,executor,depth,profile,*,sink=None,verification_deliver=None):
    """Execute one delivered decision's bounded, already-specified command prefix.

    The policy loop owns the one archive entry and provider receipt per decision.
    This executor owns command claims/ACKs. An unfinished claim is never replayed.
    """
    if not getattr(profile,'stage_plan',False):raise ValueError('S2 plan profile required')
    session=getattr(state,'response_session',None)
    attempt=getattr(session,'attempt_id',None)
    if session is not None and session.path is not None and not attempt:raise ValueError('S2 durable attempt ID required')
    binding=dict(attempt_id=attempt,case_id=bundle.get('position'),
        authorization_sha256=bundle.get('authorization_sha256'),profile=profile.name)
    session_binding=getattr(session,'binding',None)
    if session_binding is not None and any(session_binding.get(key)!=binding[key] for key in
            ('attempt_id','case_id','authorization_sha256')):
        raise RuntimeError('S2 command plan differs from the bound response session')
    path,journal=_s2_journal(state,binding)
    plan_hash=hashlib.sha256(canonical(plan)).hexdigest()
    prior=next((r for r in journal['plans'] if (r['plan_id'],r['revision'])==
        (plan.get('plan_id'),plan.get('revision'))),None)
    if prior is not None:
        if prior['plan_sha256']!=plan_hash:raise RuntimeError('S2 command identity reused with changed parameters')
        if prior.get('status')!='closed':raise RuntimeError('S2 execution is unknown; no automatic recovery')
        return copy.deepcopy(prior['result'])
    obs=bundle['observation'];oid=obs['observation_id']
    version=len(obs.get('history',getattr(state,'history',[])))
    previous=max((r['revision'] for r in journal['plans'] if r['plan_id']==plan.get('plan_id')),default=0)
    pick.s2_validate_plan(plan,profile,observation=obs,state_version=version,frame_id=oid,previous_revision=previous)
    known_ids=set(getattr(state,'observation_ids',[]))|{oid}
    known_ids.update(item['bundle']['observation']['observation_id'] for item in state.archive)
    pick.s2_validate_plan_evidence(plan,known_ids)
    if getattr(executor,'pending',None) is not None or getattr(executor,'failed',False):
        raise RuntimeError('S2 cannot execute after an unknown action')
    entry=s2_robot_snapshot(executor);anchor=entry['local_from_base'];start=executor.steps
    sensors={key:value['sha256'] for key,value in bundle.get('sensor_manifest',{}).items()}
    started=time.monotonic();rows=[];completed=[];interrupted=None;event=None
    record=dict(plan_id=plan['plan_id'],revision=plan['revision'],plan_sha256=plan_hash,
        decision_turn=bundle['turn'],status='executing',source_frame_id=oid,
        entry_robot=entry,target_ref=plan['target_ref'],source_sensor_sha256=sensors,commands=[])
    journal['plans'].append(record);_s2_persist_journal(state,path,journal)
    original_cap=executor.max_steps
    old_expired=getattr(executor,'action_budget_expired',None)
    old_ceiling=getattr(executor,'_s2_plan_step_ceiling',None)
    plan_step_cap=min(original_cap,start+plan['limits']['max_steps'])
    executor._s2_plan_step_ceiling=plan_step_cap
    executor.action_budget_expired=lambda:(time.monotonic()-started>=plan['limits']['max_seconds']
        or (old_expired is not None and old_expired()))
    try:
        for index,command in enumerate(plan['commands']):
            cid=command['command_id'];before=executor.steps
            call=pick.s2_command_call(command,profile)
            physical=call['tool'] in ('look','move_base','approach_point','move_to','move_eef_chunk',
                'prepare_grasp','replan_grasp','set_gripper','return_to_rest')
            if executor.is_stopped() or getattr(executor,'ended',False):
                event=dict(code='official_terminated',scope='case',decision_needed=False);break
            if (physical and executor.steps>=plan_step_cap) or time.monotonic()-started>=plan['limits']['max_seconds']:
                event=dict(code='plan_limit_reached',scope='plan',decision_needed=True);break
            current=s2_robot_snapshot(executor)
            delta=np.linalg.inv(np.asarray(anchor,float))@np.asarray(current['local_from_base'],float)
            translation=float(np.linalg.norm(delta[:2,3]));yaw=float(math.atan2(delta[1,0],delta[0,0]))
            if translation>.006 or abs(yaw)>.02:
                event=dict(code='base_deviation',scope='plan',decision_needed=True,
                    trigger_evidence=dict(translation_m=translation,turn_rad=yaw));break
            sensor_query=call['tool'] in ('locate_point','locate_points','measure','check_clearance','evaluate_plan')
            if sensor_query and executor.steps!=start:
                event=dict(code='evidence_invalidated',scope='plan',decision_needed=True);break
            call=_s2_transform_call(call,anchor,current['local_from_base'])
            claim=dict(command_id=cid,status='claimed',command_sha256=hashlib.sha256(canonical(command)).hexdigest(),
                requested=copy.deepcopy(command),actual_call=copy.deepcopy(call),start_step=before,
                source_frame_id=oid,command_entry_frame_id=f'{oid}/command/{cid}/step/{before}',
                anchor_to_command=(np.linalg.inv(np.asarray(current['local_from_base']))@np.asarray(anchor)).tolist())
            record['commands'].append(claim);_s2_persist_journal(state,path,journal)
            try:
                if call['tool']=='evaluate_plan':
                    actual=call
                    result=s2_evaluate_plan(state,bundle,call,executor,depth,profile,sink=sink)
                else:
                    actual,result=pick.dispatch_with_identity(state,executor,call,depth,bundle,
                        verification_deliver,pick.s2_base_profile(profile),sink=sink)
                if getattr(executor,'pending',None) is not None:
                    raise RuntimeError('S2 tool returned with an unknown pending ACK')
                result=dict(result,steps_remaining=original_cap-executor.steps)
                row=dict(call=actual,result=result,command_id=cid,decision_turn=bundle['turn'],
                    plan_id=plan['plan_id'],revision=plan['revision'],start_step=before,end_step=executor.steps,
                    source_frame_id=oid,command_entry_frame_id=claim['command_entry_frame_id'])
                claim.update(status='acknowledged',end_step=executor.steps,result=copy.deepcopy(row))
                _s2_persist_journal(state,path,journal)
            except BaseException:
                claim.update(status='unknown',last_observed_step=executor.steps);record['status']='unknown'
                _s2_persist_journal(state,path,journal)
                raise
            rows.append(row)
            def command_sink(event):
                if sink is not None:
                    event.update(decision_turn=bundle['turn'],command_id=cid,plan_id=plan['plan_id'],
                        revision=plan['revision'],source_frame_id=oid,
                        command_entry_frame_id=claim['command_entry_frame_id'])
                    sink(event)
            pick.record(command_sink,'decision_result',call,(actual,result),before,executor.steps,oid)
            failure=(result.get('accepted') is False or result.get('arrived') is False
                or result.get('requires_replan') is True or result.get('localization_invalidated') is True
                or result.get('reason') in ('grip_lost','base_pushed','motion_stalled','needs_base_reposition',
                    'waypoint_ik_unresolved','step_cap_reached','official_terminated','wall_time_limit'))
            unknown=call['tool'] in ('evaluate_plan','check_clearance','check_path','locate_point','locate_points','measure') and _s2_unknown_result(result)
            if unknown and not failure and _s2_unknown_acknowledged(plan,call,result,journal,current,sensors):unknown=False
            if failure or unknown:
                interrupted=cid;event=dict(code=result.get('reason','blocked') if failure else 'clearance_unknown',
                    scope='case' if result.get('reason')=='official_terminated' else 'plan',decision_needed=True)
                if unknown:event['trigger_evidence']=dict(evidence_ref=oid,scope=s2_unknown_scope(call,result))
                break
            completed.append(cid)
            if call['tool'] in ('done','give_up'):
                state.terminal_reason=call['tool']
            barrier=(cid in plan['observation_barriers'] or call['tool'] in
                ('move_base','approach_point','look','set_gripper','verify_target','recall_observation','done','give_up','return_to_rest'))
            if barrier:
                event=dict(code='observation_barrier' if not state.terminal_reason else state.terminal_reason,
                    scope='case' if state.terminal_reason else 'plan',decision_needed=not bool(state.terminal_reason));break
    finally:
        executor.max_steps=original_cap
        if old_ceiling is None:del executor._s2_plan_step_ceiling
        else:executor._s2_plan_step_ceiling=old_ceiling
        if old_expired is None:
            del executor.action_budget_expired
        else:executor.action_budget_expired=old_expired
    executed={r['command_id'] for r in rows}
    unexecuted=[c['command_id'] for c in plan['commands'] if c['command_id'] not in executed]
    result=dict(plan_id=plan['plan_id'],revision=plan['revision'],decision_turn=bundle['turn'],
        execution_status='interrupted' if interrupted or unexecuted else 'completed',entries=rows,
        completed_command_ids=completed,interrupted_command_id=interrupted,unexecuted_command_ids=unexecuted,
        execution_cursor=dict(command_index=len(rows),control_step=executor.steps,steps_used=executor.steps-start),
        event=event,observation_id=oid,requires_new_observation=True,
        steps_used=executor.steps-start,steps_remaining=original_cap-executor.steps)
    record.update(status='closed',result=copy.deepcopy(result));_s2_persist_journal(state,path,journal)
    pick.record(sink,'s2_plan_result',plan,(None,result),start,executor.steps,oid)
    return result


s2_execute_plan=execute_s2_plan


def run_with_ports(state, observe, deliver, executor, depth, stopped, profile, sink=None, *, verification_deliver=None):
    """Single-case decision connection; caller owns sensing, transport and evaluation."""
    try:
        return pick.run(state, observe, deliver, executor, depth, stopped, profile, sink=sink, verification_deliver=verification_deliver)
    finally:
        if profile.task_handoff and state.response_session is not None:
            session = state.response_session
            if session.path is not None:
                save(session.path.with_name('task-handoff.json'), dict(
                    episode_id=session.attempt_id, session_status=session.status,
                    journal=session.handoff_journal,
                    injections=[dict(turn=r['turn'], handoff_id=r.get('handoff_injected_id'),
                                     wire_sha256=r['wire_sha256']) for r in session.receipts
                                if r.get('handoff_injected_id')]))


def new_attempt_state(profile, output, *, attempt_id=None, handoff_plan=None, transport_mission=None,
                      case_id=None, authorization_sha256=None):
    """Create a fresh chain only; a persisted journal is evidence, not a resume token."""
    bind_initial = case_id is not None or authorization_sha256 is not None
    if bind_initial and (not profile.cold_checkpoint or not profile.response_session
            or not isinstance(case_id,str) or not case_id
            or not isinstance(authorization_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',authorization_sha256)):
        raise ValueError('Initial cold session requires authorized case and authorization hash')
    state = pick.State()
    if profile.response_session:
        if not attempt_id:
            raise ValueError('session requires authorized attempt_id')
        path = Path(output)/'response-session.json'
        plan = None
        if profile.task_handoff:
            plan = copy.deepcopy(pick.validate_handoff_plan(
                pick.FROZEN['task_handoff']['default_plan'] if handoff_plan is None else handoff_plan))
        elif handoff_plan is not None:
            raise ValueError('task handoff plan requires the explicit candidate profile')
        if profile.transport_task:
            transport_mission = copy.deepcopy(pick.validate_transport_mission(transport_mission))
        elif transport_mission is not None:
            raise ValueError('transport mission requires an explicit transport profile')
        state.response_session = pick.ResponseSession(attempt_id=attempt_id, path=path, handoff_plan=plan,
                                                      transport_mission=transport_mission)
        if bind_initial:
            # Bind before any API-free action/checkpoint. A later resume must
            # retain this origin even when no provider response exists yet.
            state.response_session.bind(dict(position=case_id,authorization_sha256=authorization_sha256,
                turn=0,observation=dict(history=[])),[])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as stream:
            json.dump(dict(attempt_id=attempt_id, status='ready', binding=state.response_session.binding, receipts=[]), stream)
            stream.flush(); os.fsync(stream.fileno())
        if profile.weekly_budget:
            state.response_session.single_pick=dict(pick_entry=None,pick_closure=None,recovery_event=None)
        if profile.task_handoff or profile.transport_task or profile.weekly_budget:
            state.response_session.persist()
    return state


def transport_plan(source):
    """Extract the frozen four-subtask development case; all geometry is eval-only."""
    if source.get('dataset') != 'ReplicaCADTidyHouseVal' or not source.get('plans'):
        raise ValueError('transport dataset mismatch')
    plan = copy.deepcopy(source['plans'][0])
    tasks = plan['subtasks'][:4]
    if (plan.get('build_config_name') != 'v3_sc3_staging_14.scene_instance.json'
            or plan.get('init_config_name') != 'val/tidy_house/episode_0.json'
            or [t.get('type') for t in tasks] != ['navigate','pick','navigate','place']
            or [t.get('uid') for t in tasks] != [f'tidy_house-sequential-val-0-{i}' for i in range(4)]
            or tasks[1].get('obj_id') != '002_master_chef_can-0'
            or tasks[3].get('obj_id') != tasks[1]['obj_id']
            or tasks[3].get('validate_goal_rectangle_corners') is not True):
        raise ValueError('transport task binding mismatch')
    pick.vector(tasks[3]['goal_pos'], 3)
    corners = tasks[3]['goal_rectangle_corners']
    if len(corners) != 4:
        raise ValueError('transport placement zone requires four corners')
    for corner in corners: pick.vector(corner, 3)
    plan['subtasks'] = tasks
    return dict(dataset=source['dataset'], plans=[plan])


def transport_semantics(record, evidence_root, plan):
    """Require reviewed robot-visible descriptions with file-bound provenance.

    File hashes establish provenance, not semantic correctness. The reviewed flag
    must only be recorded after inspection; CPU fixtures are never case evidence.
    """
    if (record.get('status') != 'reviewed' or record.get('synthetic') is not False
            or record.get('task_sha256') != hashlib.sha256(canonical(transport_plan(plan))).hexdigest()):
        raise ValueError('transport object/destination semantics remain unverified')
    mission = copy.deepcopy(pick.validate_transport_mission(record.get('mission')))
    root = Path(evidence_root).resolve()
    for role in ('object_description','destination_description'):
        item = record.get('evidence', {}).get(role, {})
        if item.get('kind') not in ('robot_rgbd_annotation','task_asset_semantic_annotation'):
            raise ValueError('transport semantic evidence type')
        if item.get('unambiguous_in_robot_view') is not True:
            raise ValueError('transport semantic ambiguity unresolved')
        path = (root/item.get('path', '')).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != item.get('sha256'):
            raise ValueError('transport semantic evidence missing or changed')
    return mission


TRANSPORT_PREFLIGHT_SCOPE = 'transport-no-api-preflight-20261011-v1'
TRANSPORT_PREFLIGHT_CAPS = dict(resets=2, steps=160, seconds=1200, api_calls=0, retries=0)
TRANSPORT_CONTINUOUS_SCOPE = 'transport-no-api-continuous-20261011-v1'


def bind_transport_fetch_factory(make_env, runtime, fetch_class):
    """Private SequentialTask binding; frozen Pick factory/controller stay untouched."""
    import types, functools
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils import registration
    def sequential_spec(gym_module, env_id, agent_class):
        if env_id != 'SequentialTask-v0':
            raise ValueError('Transport factory requires SequentialTask-v0')
        native = registration.REGISTERED_ENVS[env_id]
        if (native.cls.__module__, native.cls.__name__) != ('mshab.envs.sequential_task', 'SequentialTaskEnv'):
            raise ValueError('Unexpected native SequentialTask registration')
        if not issubclass(native.cls, BaseEnv):
            raise ValueError('Unexpected SequentialTask base class')
        loader = BaseEnv._load_agent
        agents = dict(loader.__globals__['REGISTERED_AGENTS'])
        spec = copy.copy(agents['fetch']); spec.agent_cls = agent_class; agents['fetch'] = spec
        private_loader = types.FunctionType(loader.__code__, dict(loader.__globals__, REGISTERED_AGENTS=agents),
            loader.__name__, loader.__defaults__, loader.__closure__)
        private_loader.__kwdefaults__ = loader.__kwdefaults__
        private_base = type('PrivateFetchLookup', (BaseEnv,), {'_load_agent': private_loader})
        candidate = type(native.cls.__name__, (native.cls, private_base),
            {'__module__': runtime.__name__, 'flowsde_private_agent_lookup': True})
        if candidate.__mro__.index(private_base)+1 != candidate.__mro__.index(BaseEnv):
            raise ValueError('Unexpected SequentialTask loader inheritance')
        env_spec = copy.copy(native); env_spec.cls = candidate
        envs = dict(registration.REGISTERED_ENVS); envs[env_id] = env_spec
        maker = registration.make
        private_make = types.FunctionType(maker.__code__, dict(maker.__globals__, REGISTERED_ENVS=envs),
            maker.__name__, maker.__defaults__, maker.__closure__)
        private_make.__kwdefaults__ = maker.__kwdefaults__
        gym_spec = copy.deepcopy(gym_module.spec(env_id))
        gym_spec.entry_point = functools.partial(private_make, env_id=env_id)
        return gym_spec
    # Retain every frozen factory robot/control/wrapper check. Replace only its
    # private task-spec resolver in a new function scope, never its module globals.
    binder = runtime.bind_fetch_factory
    private_binder = types.FunctionType(binder.__code__, dict(binder.__globals__, private_fetch_spec=sequential_spec),
        binder.__name__, binder.__defaults__, binder.__closure__)
    private_binder.__kwdefaults__ = binder.__kwdefaults__
    return private_binder(make_env, fetch_class)


def transport_preflight_ports(*, stage, output, construct, capture, hold, finish,
        deadline, reference=None, semantic_review=None):
    """One explicit reset per fresh process. No provider, state restore or retry."""
    if stage not in ('reference', 'survey', 'check', 'plumbing'):
        raise ValueError('transport preflight stage')
    output = Path(output)
    if stage in ('check','plumbing'):
        if reference is None or (stage=='check' and semantic_review is None):
            raise ValueError('reviewed reference required before second initialization')
        completed = json.loads((reference.parent/'result.json').read_text())
        if (completed.get('status') not in ('reference_captured_visual_review_pending','survey_complete_visual_review_pending')
                or completed.get('environment_closed') is not True):
            raise ValueError('incomplete reference stage cannot authorize second initialization')
        if stage=='check':
            transport_semantics(semantic_review, reference.parent, json.loads(
                (reference.parent/'single-plan.eval-only.json').read_text()))
            if semantic_review.get('reference_sha256') != sha(reference):
                raise ValueError('visual review is not bound to this reference')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', explicit_resets=0, steps=0, api_calls=0,
                  stage=stage, environment_closed=False, sdk_internal_steps='not independently counted')
    env = None
    try:
        deadline(); env = construct(output)
        report['explicit_resets'] = 1
        save(output/'result.json', report)
        obs, info = env.reset(seed=0, options={'reconfigure': True})
        deadline()
        snapshot = capture(env, obs, info, output)
        save(output/'initial-reference.eval-only.json', snapshot)
        if stage in ('check','plumbing'):
            comparison = pick.ab_initial_check(json.loads(reference.read_text()), snapshot)
            save(output/'paired-initial-check.json', comparison)
            if not comparison['passed']:
                raise ValueError('paired initial state mismatch')
        if stage in ('check','survey','plumbing'):
            report['steps'] = hold(env, info, output, deadline)
            if type(report['steps']) is not int or not 0 < report['steps'] <= 600:
                raise ValueError('preflight physical count invalid')
            report['status'] = 'plumbing_passed' if stage=='check' else 'survey_complete_visual_review_pending'
            if stage=='plumbing':report['status']='plumbing_passed_semantics_pending'
        else:
            report['status'] = 'reference_captured_visual_review_pending'
        deadline()
    except BaseException as error:
        report.update(status='stopped_no_retry', error_type=type(error).__name__, error=str(error))
        import traceback
        (output/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        try:
            finish(report)
        except BaseException as error:
            report.update(status='stopped_no_retry', finish_error=repr(error))
        if env is not None:
            try:
                env.close(); report['environment_closed'] = True
            except BaseException as error:
                report.update(status='stopped_no_retry', close_error=repr(error))
        save(output/'result.json', report)
    return report


def transport_preflight_native(args):
    """Dedicated authorized no-network entry, independent of paid AB bootstrap."""
    import socket
    def denied(*a, **k): raise RuntimeError('network disabled in transport preflight')
    socket.socket.connect = denied
    socket.create_connection = denied
    auth = json.loads(args.authorization.read_text())
    stage = args.transport_stage
    continuous=auth.get('scope')==TRANSPORT_CONTINUOUS_SCOPE
    scope_valid=(continuous and auth.get('caps')==dict(steps=600,seconds=3600,api_calls=0,retries=True)
        and type(auth.get('attempt_steps')) is int and 1<=auth['attempt_steps']<=600
        and 30<=auth.get('attempt_seconds',0)<=900) or (
        auth.get('scope')==TRANSPORT_PREFLIGHT_SCOPE and auth.get('caps')==TRANSPORT_PREFLIGHT_CAPS)
    if (not scope_valid or auth.get('status') != 'approved'
            or stage not in (('reference','survey','check','plumbing') if continuous else ('reference','check'))
            or auth.get('gpu_uuid') != GPU or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU
            or time.time() >= auth.get('expires_at_epoch', 0)
            or args.output.resolve() != (Path(auth['output_root'])/stage).resolve()):
        raise ValueError('transport no-API authorization/device/output mismatch')
    if args.initialization_lock_fd is None:
        raise ValueError('preflight inherited scheduler lock required')
    os.fstat(args.initialization_lock_fd)
    for name,digest in auth['source_sha256'].items():
        if sha(args.code/name) != digest: raise ValueError('preflight source changed: '+name)
    for name,digest in auth['asset_sha256'].items():
        if sha(args.lab_root/name) != digest: raise ValueError('preflight asset changed: '+name)
    source = json.loads((args.code/auth['plan_path']).read_text())
    started = time.monotonic()
    cap = auth['attempt_seconds'] if continuous else (300 if stage == 'reference' else 900)
    def deadline():
        if time.monotonic()-started > cap-10: raise TimeoutError('preflight stage deadline')
    configure_arm_runtime(args.lab_root)
    import torch, random
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from dacite import from_dict
    from mshab.envs.make import EnvConfig, make_env
    for name,path in auth['sdk_modules'].items():
        import importlib
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (args.lab_root/path).resolve():
            raise ValueError('loaded SDK path mismatch: '+name)
    full = load_memory_full_b(args.code, auth)
    factory = bind_transport_fetch_factory(make_env, full[1], full[1].agent_class())
    return transport_preflight_runtime(args, auth, source, full=full, factory=factory,
        config_factory=lambda cfg:from_dict(EnvConfig,cfg), tensor_api=torch, deadline=deadline)


def transport_preflight_runtime(args, auth, source, *, full, factory, config_factory,
        tensor_api, deadline):
    """Shared real observation/score/media chain; CPU tests replace device boundaries."""
    source = transport_plan(source)
    semantic = target_semantics(source['plans'][0]['subtasks'][1]['obj_id'])
    stage = args.transport_stage
    step_cap=auth.get('attempt_steps',160)
    state = {}
    def construct(output):
        save(output/'single-plan.eval-only.json', transport_plan(source))
        env = factory(config_factory(transport_environment_config(output/'single-plan.eval-only.json')))
        env.env.auto_reset = False
        return env
    def capture(env, obs, info, output):
        u = env.unwrapped
        if len(u.task_plan)!=4 or [t.composite_subtask_uids for t in u.task_plan] != [[t['uid']] for t in source['plans'][0]['subtasks']]:
            raise ValueError('native task identity mismatch')
        evaluation = TransportEvaluation(info)
        timer = MemoryPhaseTimer(output)
        base = activate_memory_full_b(env,full[1],full[2],output,timed=timer)
        fk = RobotFK(args.code/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',
                     [j.name for j in u.agent.robot.active_joints])
        qpos=lambda:array(u.agent.robot.qpos)[0].astype(float)
        qvel=lambda:array(u.agent.robot.qvel)[0].astype(float)
        from bvi.pick_system.snapshot_transport import capture_transport
        whole=capture_transport(base)
        rng={k:v for k,v in whole['state'].items() if 'rng' in k}
        snapshot=transport_initial_snapshot(seed=0,qpos=qpos(),qvel=qvel(),
            root_pose=array(u.agent.robot.pose.to_transformation_matrix())[0],environment_state=u.get_state_dict(),
            rng_state=rng,runtime_identity=auth['asset_sha256'],task_plan=source,initial_info=info)
        tensor_api.save(whole,output/'full-state.eval-only.pt')
        # Preserve diagnostic state even when later sensor/audit work fails. The
        # completed-stage check above prevents a partial reference being reused.
        save(output/'initial-reference.eval-only.json',snapshot)
        observe(u,fk,qpos,semantic,[],600,output/'sensors',[0,0,0],
                ObservationDepth(),call_cap=60,profile=pick.SOL_TRANSPORT_A)
        save(output/'scoring.eval-only.json',dict(goal_type=u.place_cfg.goal_type,
            obj_goal_thresh=u.place_cfg.obj_goal_thresh,initial_info=jsonable(info)))
        state.update(base=base,fk=fk,qpos=qpos,qvel=qvel,info=info,evaluation=evaluation,timer=timer)
        return snapshot
    def hold(env, info, output, deadline):
        evaluation=state['evaluation']
        state['physical_count']=0
        def step(action):
            deadline()
            if state['physical_count']>=step_cap: raise ValueError('preflight allocated step cap')
            state['physical_count']+=1
            if auth.get('scope')==TRANSPORT_CONTINUOUS_SCOPE:
                _initialization_json_exclusive(output/f"step-{state['physical_count']:04d}.intent.json",
                    dict(attempted_step=state['physical_count']))
            save(output/'physical-count.json',dict(attempted=state['physical_count']))
            result=env.step(tensor_api.as_tensor(action,device=env.unwrapped.device,dtype=tensor_api.float32).reshape(1,-1))
            state['info']=result[4]
            return evaluation.accept_step(result[4],terminated=bool(scalar(result[2])),truncated=bool(scalar(result[3])))
        recording=AttemptRecording(output/'video-frames',env.unwrapped.control_freq)
        ex=make_transport_full_b_executor(*full,state['base'],arm='A',output=output,
            case_id=pick.TransportABBatch.case_id,step=step,qpos=state['qpos'],qvel=state['qvel'],
            info=lambda:state['info'],stopped=lambda:evaluation.stopped,deadline=deadline,
            fk=state['fk'],recording=recording,timed=state['timer'])
        state['ex']=ex
        # Use the frozen controller's native hold stage; no grasp, navigation or oracle command.
        if stage=='survey':
            commands=auth.get('survey_commands',[dict(kind='look',values=x) for x in auth.get('survey_views',[[-1.2,0.],[0.,0.],[1.2,0.]])])
            for index,command in enumerate(commands):
                kind=command['kind'];values=command['values']
                if kind not in ('look','base'):raise ValueError('unsupported survey command')
                if state['physical_count']+(40 if kind=='look' else 150)>step_cap:break
                (ex.look if kind=='look' else ex.move_base)(*map(float,values))
                if evaluation.stopped:break
                observe(env.unwrapped,state['fk'],state['qpos'],semantic,[],600-ex.steps,
                    output/f'view-{index:02d}',ex.odometry(),ObservationDepth(),call_cap=60,profile=pick.SOL_TRANSPORT_A)
        else:
            ex._run([ex.Stage('hold',step_cap)],tool_cap=step_cap)
        if evaluation.stopped: raise ValueError('official termination during neutral preflight')
        return state['physical_count']
    def finish(report):
        if 'physical_count' in state: report['steps']=state['physical_count']
        if 'ex' in state:
            receipt=TransportMedia(args.output/'media',arm='A',case_id=pick.TransportABBatch.case_id).finish(
                state['ex'],state['evaluation'],'preflight-'+report['status'],[])
            save(args.output/'media.json',receipt)
            if receipt.get('status')!='complete':raise ValueError('incomplete preflight media')
    reference=Path(auth.get('reference_path',str(Path(auth['output_root'])/'reference/initial-reference.eval-only.json')))
    review_path=reference.parent/'semantic-review.json'
    return transport_preflight_ports(stage=stage,output=args.output,construct=construct,capture=capture,
        hold=hold,finish=finish,deadline=deadline,reference=reference if stage in ('check','plumbing') else None,
        semantic_review=json.loads(review_path.read_text()) if stage=='check' else None)


def transport_environment_config(task_plan_path):
    """Native SequentialTask config, no Pick spawns or relaxed scoring overrides.

    This function is pure. Construction/reset is owned by a future authorized
    lab launcher; CPU tests never import or construct the simulator.
    """
    return dict(env_id='SequentialTask-v0', num_envs=1, max_episode_steps=600,
        task_plan_fp=str(task_plan_path), obs_mode='rgbd', render_mode='rgb_array',
        record_video=False, info_on_video=False, continuous_task=True, frame_stack=1,
        stationary_base=False, stationary_torso=False, stationary_head=False,
        env_kwargs=dict(require_build_configs_repeated_equally_across_envs=False,
            add_event_tracker_info=True, sensor_configs=dict(width=640,height=640)))


def initialize_transport_environment(*, make_env, config_factory, source, semantic_record,
        evidence_root, output, seed, auth):
    """Single native reset through injected SDK ports, after semantic gates.

    No import, retry, pointer write, Pick-spawn adaptation or warm-up step. The
    caller owns SDK/source hashes and deployment authorization before invoking.
    """
    mission = transport_semantics(semantic_record, evidence_root, source)
    profiles={p.name:p for p in (pick.SOL_TRANSPORT_A,pick.SOL_TRANSPORT_B)}
    if auth.get('execution_profile') not in profiles:
        raise ValueError('transport initialization profile binding')
    pick.validate_session_scope(auth,profiles[auth['execution_profile']])
    if auth['transport_mission']!=mission:raise ValueError('transport initialization mission mismatch')
    pick.TransportABBatch.from_auth(auth).preflight(auth['ab_attempt_id'])
    plan = transport_plan(source)
    if type(seed) is not int or seed < 0:
        raise ValueError('explicit nonnegative transport seed required')
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/'single-plan.eval-only.json'
    save(path, plan)
    cfg = transport_environment_config(path)
    env = None
    try:
        env = make_env(config_factory(cfg))
        env.env.auto_reset = False
        _obs, info = env.reset(seed=seed, options={'reconfigure': True})
        actual = env.unwrapped.task_plan
        if (len(actual) != 4 or [s.composite_subtask_uids for s in actual] != [[s['uid']] for s in plan['plans'][0]['subtasks']]
                or actual[1].obj_id != actual[3].obj_id):
            raise ValueError('native transport task differs from frozen four-step plan')
        evaluation = TransportEvaluation(info)
        return env, evaluation, mission
    except BaseException:
        if env is not None:
            env.close()
        raise


class TransportEvaluation:
    """Consume normal reset/step info once, never call get_info/evaluate ourselves.

    Official pointer/outcome remain evaluation-only. Intermediate Pick success
    cannot end a four-subtask mission or trigger a model-memory handoff.
    """
    def __init__(self, initial_info, *, max_steps=600):
        if max_steps != 600:
            raise ValueError('unchanged full-B transport interface requires 600 steps')
        self.max_steps = max_steps
        self.steps = 0
        self.events = []
        self.info = self._checked(initial_info)
        self.last_info_eval_only = initial_info
        if self.info['subtask'] != 0 or self.info['success'] or self.info['fail']:
            raise ValueError('transport requires a fresh native initial state')
        self.stopped = False
        self.success = False
        self.reason = None

    @staticmethod
    def _checked(info):
        result = {}
        for field in ('success','fail','subtask','subtasks_steps_left','robot_cumulative_force'):
            if field not in info: raise ValueError('missing official transport field: '+field)
            result[field] = scalar(info[field])
        if (type(result['success']) is not bool or type(result['fail']) is not bool
                or type(result['subtask']) is not int or not 0 <= result['subtask'] <= 4
                or type(result['subtasks_steps_left']) is not int
                or type(result['robot_cumulative_force']) not in (int,float)
                or not math.isfinite(result['robot_cumulative_force'])
                or result['robot_cumulative_force'] < 0):
            raise ValueError('malformed official transport info')
        if result['success'] != (result['subtask'] == 4) or (result['success'] and result['fail']):
            raise ValueError('official whole-task success/pointer disagreement')
        return result

    def accept_step(self, info, terminated, truncated):
        if self.stopped: raise RuntimeError('transport step after terminal')
        current = self._checked(info)
        previous = self.info
        delta = current['subtask']-previous['subtask']
        if delta not in (0,1): raise ValueError('official transport stage skipped or reversed')
        if delta == 0 and (current['subtasks_steps_left'] != previous['subtasks_steps_left']-1
                or current['robot_cumulative_force']+1e-5 < previous['robot_cumulative_force']):
            raise ValueError('transport clock/force discontinuity within official stage')
        self.steps += 1
        self.info = current
        self.last_info_eval_only = info
        self.success = current['success']
        self.reason = ('official_success' if self.success else 'official_failure' if current['fail']
                       else 'time_limit' if bool(scalar(truncated)) or self.steps >= self.max_steps
                       else 'official_termination' if bool(scalar(terminated)) else None)
        self.stopped = self.reason is not None
        self.events.append(dict(step=self.steps, info=copy.deepcopy(current), stage_advanced=bool(delta),
                                termination=self.reason))
        return dict(terminated=self.stopped and self.reason != 'time_limit',
                    truncated=self.reason == 'time_limit')

    def step_port(self, env_step):
        """Connect this port to FullBOnlineExecutor.step_once; tensorize outside."""
        def step(action):
            if self.stopped: raise RuntimeError('transport step after terminal')
            _obs, _reward, terminated, truncated, info = env_step(action)
            return self.accept_step(info, terminated, truncated)
        return step


class TransportCarryPort:
    """Pass commands unchanged to one persistent full-B executor.

    closed is the controller's commanded gripper state, never object possession.
    The check only detects unexpected command-state loss; no auto-close/retry.
    """
    def __init__(self, executor):
        if (getattr(executor,'version',None) != 'full-B-r4-task-rules-v1-online-v1'
                or executor.max_steps != 600 or executor.steps != 0
                or type(getattr(executor,'closed',None)) is not bool):
            raise ValueError('transport requires fresh unchanged full-B executor')
        self.executor = executor
        self.command_log = []

    def __getattr__(self, name):
        target = getattr(self.executor, name)
        if name not in ('look','move_base','check_path','move_to','move_eef_chunk','set_gripper','return_to_rest'):
            return target
        def call(*args, **kwargs):
            before = self.executor.closed
            result = target(*args, **kwargs)
            after = self.executor.closed
            self.command_log.append(dict(tool=name, commanded_closed_before=before,commanded_closed_after=after))
            if name != 'set_gripper' and after != before:
                raise RuntimeError('transport gripper command changed during non-gripper tool')
            return result
        return call


def transport_initial_snapshot(*, seed, qpos, qvel, root_pose, environment_state,
        rng_state, runtime_identity, task_plan, initial_info):
    """Read-only full-state snapshot including task and RNG, never a restore."""
    plan=transport_plan(task_plan)
    if not isinstance(rng_state,dict) or not rng_state:
        raise ValueError('transport paired RNG evidence required')
    environment=copy.deepcopy(environment_state)
    if 'transport_context' in environment:
        raise ValueError('transport snapshot context already set')
    environment['transport_context']=dict(rng=rng_state,
        initial_evaluation=TransportEvaluation(initial_info).info,
        task_sha256=hashlib.sha256(canonical(plan)).hexdigest())
    return memory_ab_initial_snapshot(pick.TransportABBatch.case_id,
        dict(uid='tidy_house-sequential-val-0-0',spawn_index=0),seed,qpos,qvel,root_pose,
        environment,runtime_identity)


class TransportMedia:
    """Finalize the existing full-B capture once; no render or environment reads.

    CPU encoding uses retained frames. Failure attempts keep the same artifacts
    and explicit outcome labels; incomplete capture is never called complete.
    """
    def __init__(self, output, *, arm, case_id):
        if arm not in ('A','B') or case_id!=pick.TransportABBatch.case_id:
            raise ValueError('transport media identity')
        self.output=Path(output);self.arm=arm;self.case_id=case_id
        self.finished=False

    def validate_executor(self, executor):
        native=executor.recorder.native
        if native.case_id!=self.case_id or native.condition!=self.arm or native.action_size!=20:
            raise ValueError('transport native recorder identity/action size')

    def finish(self, executor, evaluation, outcome, history):
        if self.finished:raise ValueError('transport media already finalized')
        self.finished=True
        self.validate_executor(executor)
        self.output.mkdir(parents=True,exist_ok=False)
        receipt=dict(case_id=self.case_id,arm=self.arm,outcome=outcome,
            physical_steps=evaluation.steps,evaluation_only=True,replay=False,artifacts={})
        try:
            executor.finish(outcome)  # Original capture/state checks run before encoding.
            native=executor.recorder.native
            source=native.root/'frames.json';data=json.loads(source.read_bytes())
            frames=data['frames']
            if not frames and evaluation.steps==0:
                receipt['status']='no_physical_steps'
                return receipt
            if (len(frames)!=evaluation.steps+1 or [f['step'] for f in frames]!=list(range(len(frames)))
                    or data.get('evaluation_only') is not True or data.get('attempt_outcome')!=outcome):
                raise ValueError('transport media frame coverage/outcome mismatch')
            for frame in frames:
                if frame['case_id']!=self.case_id or frame['condition']!=self.arm:
                    raise ValueError('transport media cross-arm frame')
                required={frame[k] for k in ('scene','head','hand','depth')}
                if not required<=set(frame['files_sha256']):
                    raise ValueError('transport media missing file hashes')
                for name,digest in frame['files_sha256'].items():
                    path=(native.root/name).resolve()
                    if not path.is_relative_to(native.root.resolve()) or sha(path)!=digest:
                        raise ValueError('transport media file missing or changed')
            import importlib.util
            template=Path(__file__).resolve().parents[2]/'research/c2/diagnostics/2026-10-10-flowsde-controller-cpu/video_template.py'
            spec=importlib.util.spec_from_file_location('_transport_video_template',template)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            import imageio.v2 as imageio
            import imageio_ffmpeg
            recorded=datetime.strptime(frames[0]['utc'],'%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%dT%H%M%SZ')
            video=self.output/f'{self.case_id}-{self.arm}-{recorded}-{outcome}.mp4'
            fps=float(data['fps'])
            if not math.isfinite(fps) or fps<=0:raise ValueError('invalid recording fps')
            with imageio.get_writer(str(video),fps=fps,codec='libx264',macro_block_size=1) as writer:
                for row in frames:writer.append_data(np.asarray(module.render(row,native.root)))
            decoder=imageio_ffmpeg.read_frames(str(video));next(decoder)
            decoded=sum(1 for _ in decoder)
            if decoded!=len(frames):raise ValueError('encoded video frame count mismatch')
            trace=self.output/'trajectory.eval-only.json'
            save(trace,dict(frames=frames,history=history,official_stages=evaluation.events,
                controller_trace=executor.trace,frame_source=str(source.resolve()),evaluation_only=True))
            import html
            page=self.output/'trajectory.html'
            rows=''.join('<tr><td>'+str(f['step'])+'</td><td>'+html.escape(json.dumps(f['base_world_eval_only']))+
                '</td><td>'+html.escape(json.dumps(f['trajectory_world'][-1]))+'</td><td>'+html.escape(json.dumps(f['qpos'][-2:]))+
                '</td><td>'+html.escape(str(f.get('cumulative_force')))+'</td></tr>' for f in frames)
            page.write_text('<!doctype html><meta charset="utf-8"><title>Transport trajectory</title>'+
                '<h1>Evaluation only — '+html.escape(self.arm+' / '+outcome)+'</h1><p>Recorded run; TCP overlay is X-ray.</p>'+
                '<video controls width="960" src="'+html.escape(video.name,quote=True)+'"></video>'+
                '<p><a href="trajectory.eval-only.json">Full trajectory, tool history and scoring</a></p>'+
                '<table><tr><th>Step</th><th>Base pose</th><th>TCP world</th><th>Finger joints</th><th>Force</th></tr>'+rows+'</table>',encoding='utf8')
            for path in (video,trace,page,source):
                receipt['artifacts'][path.name]=dict(path=str(path.resolve()),sha256=sha(path))
            receipt.update(status='complete',recorded_at_utc=frames[0]['utc'],frames=len(frames),decoded_frames=decoded)
            return receipt
        except BaseException as error:
            receipt.update(status='incomplete',error_type=type(error).__name__)
            raise
        finally:
            save(self.output/'media-receipt.json',receipt)


def run_transport_with_ports(*, arm, mission, output, auth, auth_sha, ledger,
        observe, send, executor, depth, evaluation, deadline, initial_snapshot, media, sink=None):
    """Production decision/delivery/dispatch loop; CPU tests inject inert ports.

    No native launcher is enabled by this function. Real semantic provenance,
    paired initial state, SDK/full-B hashes and capture gates belong to launch.
    """
    if arm not in ('A','B'): raise ValueError('transport arm must be A or B')
    profile = pick.SOL_TRANSPORT_A if arm == 'A' else pick.SOL_TRANSPORT_B
    pick.validate_session_scope(auth, profile)
    batch=pick.TransportABBatch.from_auth(auth)
    attempt=auth['ab_attempt_id']
    batch.preflight(attempt)
    if mission != auth.get('transport_mission'):
        raise ValueError('transport mission differs from authorization')
    if evaluation.steps != 0 or evaluation.stopped:
        raise ValueError('transport evaluator must be fresh')
    carry = TransportCarryPort(executor)
    media.validate_executor(executor)
    Path(output).mkdir(parents=True, exist_ok=False)
    state = new_attempt_state(profile, output, attempt_id=auth['attempt_ids']['handoff-transport-val000-object0'],
        handoff_plan=auth.get('handoff_plan') if profile.task_handoff else None, transport_mission=mission)
    def stop():
        deadline()
        if evaluation.steps != carry.steps:
            raise ValueError('transport evaluator/controller step mismatch')
        return evaluation.stopped or carry.steps >= profile.step_cap
    def deliver(wire, bundle, archive):
        deadline()
        result = pick.process(bundle, archive, 'V-mobile', Path(output)/'provider'/f'turn-{bundle["turn"]:03d}',
                              ledger, auth, auth_sha, send, profile, session=state.response_session)
        deadline()  # A response arriving after the deadline never starts motion.
        return result
    failure = None
    try:
        initial=observe(0,[],None)
        batch.register_start(attempt,initial_snapshot,initial)
        def cached_observe(turn,history,previous):
            return copy.deepcopy(initial) if turn==0 else observe(turn,history,previous)
        run_with_ports(state, cached_observe, deliver, carry, depth, stop, profile, sink=sink)
        stop()
        return state
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        outcome='execution_error' if failure else evaluation.reason or state.terminal_reason or 'tool_call_limit'
        report=dict(
            strict_transport_success=evaluation.success, physical_steps=evaluation.steps,
            termination=outcome,
            error_type=failure,
            official_stages=evaluation.events, carry_commands=carry.command_log,
            provider_receipts=len(state.response_session.receipts))
        final_error=None
        try:
            report['media']=media.finish(executor,evaluation,outcome,state.history)
            save(batch.root/(arm+'-media.json'),report['media'])
            if failure:batch.abort(attempt,evaluation.steps)
            else:
                terminal=('call_limit' if outcome=='tool_call_limit' else
                          'step_limit' if evaluation.steps==600 and not evaluation.success else
                          outcome if outcome in ('done','give_up') else 'official_termination')
                batch.finish(attempt,evaluation.steps,terminal)
        except BaseException as error:
            report['finalization_error']=type(error).__name__
            final_error=error
            try:batch.abort(attempt,evaluation.steps)
            except BaseException as secondary:
                report['batch_abort_error']=type(secondary).__name__
        finally:
            save(Path(output)/'transport-result.eval-only.json', report)
        if final_error is not None and failure is None:raise final_error


def request_accounting(ledger, case_id, *, prefix=None):
    """Ledger slots are not a claim that every reserved HTTP request was sent."""
    entries = [e for e in ledger['entries'] if e.get('case_id') == case_id and
               (prefix is None or e['request_id'].startswith(prefix))]
    return dict(request_slots=len(entries), main_slots=sum(e['request_kind']=='main' for e in entries),
        verification_slots=sum(e['request_kind']=='verification' for e in entries),
        known_not_sent=sum(e['status']=='not_sent' for e in entries),
        outcome_unknown=sum(e['status'] in ('unknown','reserved') for e in entries),
        committed_usd=str(pick.committed_cost(entries)),
        definition='Durable logical slots, including not-sent and unknown entries; not HTTP-send count. Unknown cost retains reservation.')


def batch_physical_steps(root, batch_id):
    """Original same-batch results only, independent of attempt directory depth."""
    total = 0; seen = set()
    for path in Path(root).rglob('result.json'):
        if 'media-source' in path.relative_to(root).parts:
            continue
        value = read(path)
        if value.get('batch_id') != batch_id:
            raise ValueError('unbound result in fresh batch output root')
        key = (value.get('case_id'),value.get('attempt_index'))
        steps = value.get('simulator_actions')
        if key in seen or key[1] not in (0,1) or not isinstance(key[0],str) or type(steps) is not int or steps < 0:
            raise ValueError('duplicate or invalid original attempt accounting')
        seen.add(key); total += steps
    return total


def finalize_request_accounting(report, ledger_path, auth):
    """Accounting errors must not bypass environment closure or trajectory retention."""
    try:
        prefix = (auth['batch_id']+'-'+report['case_id']+'-'+auth['attempt_ids'][report['case_id']]+'-'
                  if auth.get('scope') == pick.WEEKLY_SCOPE
                  else auth['batch_id']+f"-attempt-{report['attempt_index']}-")
        ledger=read(ledger_path)
        grant=auth.get('checkpoint_branch') or auth.get('cold_resume')
        if grant:
            count=(grant['prefix_requests'] if auth.get('checkpoint_branch') else
                len(read(Path(grant['parent_ledger_path']))['entries']))
            parent=read(Path(grant['parent_ledger_path']))
            inherited=parent['entries'][:count]
            if sha(grant['parent_ledger_path'])!=grant['parent_ledger_sha256'] or ledger['entries'][:count]!=inherited:
                raise ValueError('Inherited request accounting changed')
            if auth.get('paid_delivery_resume'):
                inherited,_=paid_delivery_evidence(auth,pick.SOL_HOT_RECOVERY,inherited)
                count=len(inherited)
                if ledger['entries'][:count]!=inherited:raise ValueError('Paid inherited accounting changed')
            accounting=request_accounting(ledger,report['case_id'])
            report['inherited_request_accounting']=request_accounting({'entries':inherited},report['case_id'])
            report['segment_request_accounting']=request_accounting({'entries':ledger['entries'][count:]},report['case_id'],prefix=prefix)
            if (accounting['request_slots']!=report['inherited_request_accounting']['request_slots']+
                    report['segment_request_accounting']['request_slots']):
                raise ValueError('New request accounting outside bound branch segment')
            report['accounting_scope']='Cumulative execution counts; inherited fees already booked, book only segment fees.'
        else:accounting=request_accounting(ledger,report['case_id'],prefix=prefix)
    except Exception as error:
        report.update(api_request_accounting=dict(status='accounting_incomplete',error_type=type(error).__name__),
                      api_requests=None,verification_requests=None,status='infrastructure_censored',
                      accounting_error='Ledger could not be reconciled; costs and counts remain unknown.')
    else:
        report.update(api_request_accounting=accounting,api_requests=accounting['request_slots'],
                      verification_requests=accounting['verification_slots'])

def configure_pick_runtime(auth, lab_root, profile):
    # Independent identity review does not opt a Sol run into C2's separate
    # high-resolution texture condition. Sol retains its frozen r2 asset tree;
    # live_run still checks every authorized asset_sha256 before env creation.
    # The offline rest-feedback candidate retains the v7+HD asset condition
    # without inheriting the later HD/search/guard profile hierarchy.
    if profile.hd_textures or profile == pick.SOL_REST_PROGRESS:
        verify_sol_hd_assets(auth)
        configure_arm_runtime(lab_root, auth['asset_root'])
    elif profile.verification and not profile.weekly_budget:
        verify_c2_assets(auth)
        configure_arm_runtime(lab_root, auth['asset_root'])
    else:
        configure_arm_runtime(lab_root)


def verify_c2_assets(auth):
    manifest_path=Path(auth['asset_manifest_path'])
    if sha(manifest_path)!=auth['asset_manifest_sha256']:
        raise ValueError('C2 asset manifest changed')
    manifest=json.loads(manifest_path.read_text())
    if Path(manifest['asset_root']).resolve()!=Path(auth['asset_root']).resolve():
        raise ValueError('C2 asset root not bound')
    root=Path(auth['asset_root'])/'data/assets/mani_skill2_ycb'
    for row in manifest['rows']:
        path=root/row['path']
        if not path.resolve().is_relative_to(root.resolve()) or sha(path)!=row['new_sha256']:
            raise ValueError('C2 asset changed')
        if row['changed'] and path.suffix!='.png':
            raise ValueError('C2 non-texture change')
    return manifest


def verify_sol_hd_assets(auth):
    """Bind the HD tree explicitly; never fall back to the low-resolution tree."""
    import struct
    manifest = verify_c2_assets(auth)
    root = Path(auth['asset_root'])/'data/assets/mani_skill2_ycb'
    pngs = {row['path'] for row in manifest['rows'] if Path(row['path']).suffix == '.png'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
    if not pngs or pngs != actual:
        raise ValueError('Sol HD texture manifest incomplete')
    # The YCB builder loads textured.obj; material_0.png can be an unused export.
    referenced = set()
    bound = {row['path'] for row in manifest['rows']}
    upgraded_models = {Path(row['path']).parts[1] for row in manifest['rows']
                       if row['changed'] and row['path'].startswith('models/') and Path(row['path']).suffix == '.png'}
    checked_models = set()
    for obj in root.glob('models/*/textured.obj'):
        if obj.relative_to(root).as_posix() not in bound:
            raise ValueError('Sol HD visual model not bound')
        for line in obj.read_text().splitlines():
            if not line.startswith('mtllib '):
                continue
            material = obj.parent/line.split(maxsplit=1)[1].strip()
            if not material.resolve().is_relative_to(root.resolve()) or material.relative_to(root).as_posix() not in bound:
                raise ValueError('Sol HD material not bound')
            for entry in material.read_text().splitlines():
                if entry.startswith('map_Kd '):
                    texture = material.parent/entry.split(maxsplit=1)[1].strip()
                    if not texture.resolve().is_relative_to(root.resolve()) or texture.relative_to(root).as_posix() not in pngs:
                        raise ValueError('Sol HD referenced texture not bound')
                    # The frozen HD pack covers its declared scene categories,
                    # not every unused model shipped in the YCB asset archive.
                    if obj.parent.name in upgraded_models:
                        referenced.add(texture.relative_to(root).as_posix())
                        checked_models.add(obj.parent.name)
    if not referenced or checked_models != upgraded_models:
        raise ValueError('Sol HD referenced textures missing')
    for name in sorted(referenced):
        with (root/name).open('rb') as f:
            header = f.read(24)
        if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
                or header[12:16] != b'IHDR' or struct.unpack('>II', header[16:24]) != (4096, 4096)):
            raise ValueError('Sol HD texture must be 4096x4096')
    return manifest



class SupervisionStopped(RuntimeError):
    pass


def automatic_ledger_check(ledger, *, authorization_sha256=None, authorization_path=None, batch_id=None, case_ids=None, checkpoint_branch=None, cold_resume=None, attempt_ids=None):
    """Reject ambiguous or malformed accounting before automatic admission."""
    try:
        if isinstance(ledger, (str, os.PathLike)):
            ledger = read(ledger)
        if authorization_path is not None:
            authorization_sha256 = sha(authorization_path)
            authorization=read(Path(authorization_path))
            authorized_branch=authorization.get('checkpoint_branch')
            if checkpoint_branch is not None and checkpoint_branch!=authorized_branch:
                raise ValueError('checkpoint branch authorization differs')
            checkpoint_branch=authorized_branch
            authorized_resume=authorization.get('cold_resume')
            if cold_resume is not None and cold_resume!=authorized_resume:
                raise ValueError('cold resume authorization differs')
            cold_resume=authorized_resume
            if authorization.get('scope')==pick.WEEKLY_SCOPE:
                if attempt_ids is not None and attempt_ids!=authorization.get('attempt_ids'):
                    raise ValueError('weekly logical session authorization differs')
                attempt_ids=authorization.get('attempt_ids')
        if not isinstance(ledger, dict) or ledger.get('status') != 'running' or not isinstance(ledger.get('entries'), list):
            raise ValueError('ledger structure or status')
        if authorization_sha256 is not None and ledger.get('authorization_sha256') != authorization_sha256:
            raise ValueError('ledger authorization hash')
        inherited=0
        if checkpoint_branch is not None and cold_resume is not None:
            raise ValueError('multiple inherited ledger grants')
        if checkpoint_branch is not None:
            inherited=checkpoint_branch['prefix_requests']
            if (type(inherited) is not int or inherited<1 or len(ledger['entries'])<inherited
                    or hashlib.sha256(canonical(ledger['entries'][:inherited])).hexdigest()!=checkpoint_branch['prefix_ledger_sha256']
                    or checkpoint_branch['parent_batch_id']==batch_id):
                raise ValueError('checkpoint branch inherited ledger changed')
        elif cold_resume is not None:
            # A subsequent ordinary cold resume may inherit a mixed-batch
            # branch ledger. Exact parent evidence, never an arbitrary ID
            # prefix exception, binds every inherited row.
            for name in ('parent_ledger','parent_result'):
                if sha(cold_resume[name+'_path'])!=cold_resume[name+'_sha256']:
                    raise ValueError('cold inherited ledger evidence changed')
            parent=read(Path(cold_resume['parent_ledger_path']))
            previous=parent['entries'];inherited=len(previous)
            checkpoint=Path(cold_resume['checkpoint'])/'receipt.json'
            receipt=read(checkpoint)
            paid=None
            parent_closed=parent.get('status')=='closed'
            if authorization_path is not None and authorization.get('paid_delivery_resume'):
                paid,_=paid_delivery_evidence(authorization,pick.SOL_HOT_RECOVERY,previous)
                # The original safe budget exit retains its raw stopped status.
                # The later paid segment must be independently closed and bound.
                parent_closed=parent_closed or (parent.get('status')=='stopped_budget_exhausted'
                    and read(Path(cold_resume['parent_result_path'])).get('environment_closed') is True)
            if (sha(checkpoint)!=cold_resume['receipt_sha256'] or receipt['requests']!=inherited
                    or not parent_closed or ledger['entries'][:inherited]!=previous
                    or len(ledger['entries'])<inherited):
                raise ValueError('cold inherited ledger prefix changed')
            if paid is not None:
                if ledger['entries'][:len(paid)]!=paid:raise ValueError('Paid inherited ledger changed')
                inherited=len(paid)
        seen = set()
        for index,e in enumerate(ledger['entries']):
            entry_batch=None if index<inherited else batch_id
            if not isinstance(e, dict) or e.get('status') not in ('valid', 'invalid_call', 'invalid_verification', 'not_sent'):
                raise ValueError('delivery outcome incomplete')
            key = e.get('request_id')
            if not isinstance(key, str) or not key or key in seen or (entry_batch is not None and not key.startswith(entry_batch+'-')):
                raise ValueError('request identity or duplicate slot')
            seen.add(key)
            if not isinstance(e.get('case_id'), str) or not e['case_id'] or e.get('request_kind') not in ('main', 'verification'):
                raise ValueError('request accounting identity')
            if case_ids is not None and e['case_id'] not in case_ids:
                raise ValueError('request outside authorized cases')
            if attempt_ids is not None:
                if e.get('attempt_id')!=attempt_ids[e['case_id']]:raise ValueError('weekly logical session differs')
                if entry_batch is not None:
                    expected=entry_batch+'-'+e['case_id']+'-'+e['attempt_id']+('-review-' if e['request_kind']=='verification' else '-')
                    if not re.fullmatch(re.escape(expected)+r'\d{3}',key):raise ValueError('weekly new request identity differs')
            elif entry_batch is not None and not re.fullmatch(re.escape(entry_batch+'-attempt-0-'+e['case_id']+'-'+e['request_kind'])+r'-\d{3}', key):
                raise ValueError('automatic request cannot recover a prior attempt')
            if ((attempt_ids is None and e.get('condition') != 'V-mobile')
                    or not isinstance(e.get('wire_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', e['wire_sha256'])):
                raise ValueError('request content identity')
            for field in ('estimated_usd', 'reserve_usd'):
                value = e.get(field)
                if not isinstance(value, str):
                    raise ValueError('missing monetary accounting')
                amount = pick.Decimal(value)
                if not amount.is_finite() or amount < 0:
                    raise ValueError('invalid monetary accounting')
            if pick.Decimal(e['estimated_usd']) > pick.Decimal(e['reserve_usd']) or (e['status'] == 'not_sent' and pick.Decimal(e['estimated_usd']) != 0):
                raise ValueError('monetary accounting exceeds reservation or unsent cost')
    except (OSError, ValueError, TypeError, KeyError, ArithmeticError) as error:
        raise pick.VerificationOutcomeUnknown('automatic supervision requires reconciled delivery and budget: '+str(error)) from None
    return ledger


def resource_accounting_snapshot(ledger_path, auth, authorization_sha256, case_id):
    """Read local spending headroom; never treat unavailable provider bills as zero."""
    unknown = dict(status='unknown', currency='USD', campaign_remaining_usd=None,
                   case_remaining_usd=None, provider_bill_status='unknown',
                   basis='local estimated usage and reservations; not provider billed charges',
                   ledger_sha256=None)
    if not Path(ledger_path).is_file():
        return unknown
    ledger = automatic_ledger_check(ledger_path, authorization_sha256=authorization_sha256,
                                   batch_id=auth.get('batch_id'), case_ids=auth.get('case_ids'),
                                   checkpoint_branch=auth.get('checkpoint_branch'),cold_resume=auth.get('cold_resume'),
                                   attempt_ids=auth.get('attempt_ids') if auth.get('scope')==pick.WEEKLY_SCOPE else None)
    if case_id not in auth.get('case_ids', []):
        raise ValueError('budget snapshot case outside authorized roster')
    try:
        campaign_cap = pick.Decimal(auth['usd_cap'])
        case_cap = pick.Decimal(auth['per_case_usd_cap'])
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return unknown
    if any(not value.is_finite() or value <= 0 for value in (campaign_cap, case_cap)):
        raise ValueError('invalid budget snapshot ceiling')
    spent = pick.committed_cost(ledger['entries'])
    case_spent = pick.committed_cost([row for row in ledger['entries'] if row['case_id'] == case_id])
    if spent > campaign_cap or case_spent > case_cap:
        raise ValueError('budget snapshot ledger exceeds authorized ceiling')
    return dict(unknown, status='estimated_ledger_headroom',
                campaign_remaining_usd=str(campaign_cap-spent),
                case_remaining_usd=str(case_cap-case_spent), ledger_sha256=sha(ledger_path))


def supervise_command(output, pending, deadline, *, automatic=False):
    """Operator review never adds advice to the experiment model."""
    root = Path(output) / 'supervision'
    root.mkdir(exist_ok=True)
    turn = pending['turn']
    digest = hashlib.sha256(canonical(pending)).hexdigest()
    save(root / f'turn-{turn:03d}.pending.json', dict(pending, command_sha256=digest))
    if automatic:
        deadline()
        save(root / f'turn-{turn:03d}.automatic.json', dict(command_sha256=digest,
             action='continue', mode='automatic', reason='dispatch retains parameter validation and zero-step feedback'))
        save(root / f'turn-{turn:03d}.wait.json', dict(operator_wait_seconds=0))
        return
    decision_path = root / f'turn-{turn:03d}.decision.json'
    wait_started = time.monotonic()
    try:
        while not decision_path.exists():
            deadline()
            time.sleep(0.2)
    finally:
        save(root / f'turn-{turn:03d}.wait.json', dict(operator_wait_seconds=time.monotonic()-wait_started))
    deadline()
    decision = json.loads(decision_path.read_text())
    if decision.get('command_sha256') != digest or not decision.get('reason'):
        raise SupervisionStopped('invalid supervision decision')
    if decision.get('action') != 'continue':
        raise SupervisionStopped(decision['reason'])


def capture_detailed_step(u, output, step, dt):
    """Full sensor/state/contact archive, strictly evaluator-only, no feedback."""
    before = state_copy(u.get_state_dict())
    data = read_sensor_data_without_evaluate(u)
    folder = Path(output) / 'trajectory' / f'step-{step:04d}'
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/'robot-rgbd.npz', **{
        f'{camera}_{kind}': array(data[camera][kind])
        for camera in ('fetch_head','fetch_hand') for kind in ('rgb','depth')})
    import torch
    torch.save(before, folder/'state.eval-only.pt')
    contacts = []
    for name, actor in u.scene_builder.movable_objects.items():
        sides = []
        for finger, sign in ((u.agent.finger1_link,-1), (u.agent.finger2_link,1)):
            force = array(u.scene.get_pairwise_contact_forces(finger,actor))[0].astype(float)
            direction = sign * array(finger.pose.to_transformation_matrix())[0,:3,1]
            norm = float(np.linalg.norm(force))
            angle = None if norm == 0 else float(np.degrees(np.arccos(np.clip(np.dot(force,direction)/(norm*np.linalg.norm(direction)),-1,1))))
            sides.append(dict(force_world_N=force.tolist(),opening_direction_world=direction.tolist(),force_N=norm,angle_deg=angle,force_pass=norm>=.5,angle_pass=angle is not None and angle<=30))
        contacts.append(dict(object=name,fingers=sides))
    save(folder/'contacts.eval-only.json',dict(step=step,simulation_seconds=step*dt,captured_at_utc=datetime.now(timezone.utc).isoformat(),objects=contacts))
    if not state_equal(before,state_copy(u.get_state_dict())):
        raise ValueError('detailed recording changed simulator state')


def official_dataset_split(row):
    split = row.get('dataset_split', 'val')
    if split not in ('train', 'val'):
        raise ValueError('unsupported official dataset split')
    if 'dataset_split' in row and (not row['uid'].startswith(f'tidy_house-pick-{split}-') or not row['init_config_name'].startswith(split + '/')):
        raise ValueError('official dataset split disagrees with task binding')
    return split


def replay_executor_config(auth, profile, arm):
    """Resolve a bound L1 arm without leaking treatment flags into control."""
    if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
        r33=auth.get('scope')==pick.R33_REPLAY_SCOPE
        expected=pick.R33_REPLAY_FLAGS if r33 else pick.R32_REPLAY_FLAGS
        revision='C2-r3.3' if r33 else 'C2-r3.2'
        profiles=(pick.C2_R33,pick.C2_R33_FAR) if r33 else (pick.C2_R32,pick.C2_R32_FAR)
        if (profile not in profiles or arm not in expected
                or auth.get('system_revision')!=revision
                or auth.get('replay_variant')!=(pick.R33_REPLAY_VARIANT if r33 else pick.R32_REPLAY_VARIANT)
                or auth.get('effective_controller_flags_exact')!=expected
                or any(type(v) is not bool for row in auth.get('effective_controller_flags_exact',{}).values() for v in row.values())):
            raise ValueError('r32 replay variant/profile/controller flags mismatch')
        return expected[arm].copy()
    if arm not in ('control', 'abort'):
        raise ValueError('unknown replay arm')
    flags = ('base_motion_abort', 'base_heading_hold', 'continuous_base_speed',
             'schedule_precheck', 'robot_diagnostics')
    off = dict.fromkeys(flags, False)
    if auth.get('scope') == 'c2-r31-l1-replay-20261008':
        expected = {'control': off, 'abort': dict(off, base_motion_abort=True,
                    base_heading_hold=True, continuous_base_speed=True)}
        bound = auth.get('effective_controller_flags_exact')
        if (profile not in (pick.C2_R31, pick.C2_R31_FAR)
                or auth.get('system_revision') != 'C2-r3.1'
                or auth.get('replay_variant') != 'r31-heading-envelope-abort'
                or bound != expected
                or any(type(v) is not bool for values in (bound or {}).values() for v in values.values())):
            raise ValueError('r31 replay variant/profile/controller flags mismatch')
        return expected[arm].copy()
    if auth.get('scope') != pick.R3_SCOPE or profile not in (pick.C2_R3, pick.C2_R3_FAR):
        raise ValueError('unknown replay scope/profile')
    return dict(off, base_motion_abort=bool(profile.base_motion_abort and arm == 'abort'))


def replay_command_state(source, entry, initial_qvel):
    """Recover base-command state from bound evidence, never from default values."""
    prefix = entry['prefix_steps']
    q = vector(source['initial_robot_state']['qpos'], 15)
    prior = [c for c in source['commands'] if c['turn'] < entry['turn']]
    if any(c.get('call', {}).get('tool') == 'set_gripper' for c in prior if c.get('call')):
        raise ValueError('L1 roster has no gripper-state reconstruction for this prefix')
    if prefix:
        trace = [t for t in source['executor_trace'] if t['step'] == prefix]
        if len(trace) != 1:
            raise ValueError('ambiguous controller delta evidence')
        delta = vector(trace[0]['delta'], 15)
        last = vector(source['frames'][prefix-1]['qpos'], 15)
        before = q if prefix == 1 else vector(source['frames'][prefix-2]['qpos'], 15)
        if not np.array_equal(delta, last-before):
            raise ValueError('controller delta differs from recorded joint displacement')
    else:
        delta = vector(initial_qvel, 15) * source['control_dt']
    return dict(previous_delta=delta, initial_base=q[:3].copy(),
                gripper=float(np.clip((q[-2:].mean()+.01)/.06, 0, 1)), steps=prefix)


def record_r31_end_observation(output, auth, report, control_step, observation, manifest):
    """Bind the command-end sensor bundle before the physical post-command hold."""
    output = Path(output)
    folder = output/'post-command-sensors'
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    bundle = dict(case_id=report['case_id'], condition=report['condition'],
                  system_revision=report['system_revision'], replay_variant=report['replay_variant'],
                  replay_turn=report['replay_turn'], replay_arm=report['replay_arm'],
                  source_manifest_sha256=auth['source_manifest_sha256'],
                  authorization_sha256=report['authorization_sha256'], control_step=control_step,
                  observation=observation, sensor_manifest=manifest, sensor_files_sha256=files)
    if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
        bundle['replay_group']=report['replay_group']
    path = output/'post-command-observation.json'
    save(path, bundle)
    report['l1_end_observation_binding'] = dict(path=path.name, sha256=sha(path),
        control_step=control_step, observation_id=observation['observation_id'],
        sensor_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest())


def l1b_replay_entry(args, auth):
    """Select one immutable non-base regression; no source or candidate generation."""
    if auth.get('scope') not in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
        raise ValueError('L1-B requires fresh r32 scope')
    binding=auth['l1b_sources_and_scores'];path=Path(binding['path'])
    if sha(path)!=binding['sha256']:
        raise ValueError('L1-B sources/score binding changed')
    evidence=read(path)
    rows=[r for r in evidence['rows'] if r['scope']=='L1-B_proposed_pair'
          and 'baseline-'+r['case']==args.case_id and r['turn']==args.replay_turn]
    if len(rows)!=1 or args.replay_arm not in ('control','candidate'):
        raise ValueError('L1-B command/arm not registered')
    return rows[0]


def replay_arm_step_cap(args, auth):
    if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE) and getattr(args,'replay_group','L1-A')=='L1-B':
        row=l1b_replay_entry(args,auth)
        tool_cap=15 if row['original_command']['call']['tool']=='set_gripper' else 150
        return row['start_step']+tool_cap+max(20,row['raw_action_segments']['suffix']['count'])
    rows=[r for r in read(args.replay_roster)['roster'] if 'baseline-'+r['case']==args.case_id and r['turn']==args.replay_turn]
    if len(rows)!=1:
        raise ValueError('L1 command step cap binding')
    return rows[0]['prefix_steps']+100


def l1b_schedule_probe(args, ex, goals, state, recipe, label):
    """Independent CPU robot clone; no environment callback or live-executor mutation."""
    original={k:copy.deepcopy(getattr(ex,k)) for k in
        ('steps','previous_delta','last_goal','last_arm_target','last_plan_endpoint','gripper')}
    q=vector(state['qpos'],15).copy();v=vector(state['qvel'],15).copy()
    urdf=args.code/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf'
    clone_fk=RobotFK(urdf)
    calls=[]
    def forbidden_step(action):
        calls.append(True)
        raise ValueError('L1-B readonly schedule probe attempted a physical step')
    clone=ArmCoordinatedV2(clone_fk,lambda:q.copy(),forbidden_step,
        lambda:float(q[-2:].sum()),lambda:{},copy.deepcopy(ex.model),
        max_steps=ex.max_steps,extended_attempt=ex.max_steps==1000,parameters=copy.deepcopy(ex.p),
        read_qvel=lambda:v.copy(),control_dt=ex.control_dt,
        initial_delta=vector(state['previous_delta'],15),gripper=recipe['gripper'],schedule_precheck=True)
    clone.steps=state['step'];clone._initial_base=vector(recipe['initial_base'],3).copy()
    clone.last_arm_target=vector(recipe['last_arm_target'],15).copy()
    clone.last_plan_endpoint=vector(recipe['last_plan_endpoint'],15).copy()
    goal=recipe['last_goal']['pose'];clone.last_goal=pick.pose(goal['position'],goal['quaternion_xyzw'])
    before=clone.steps;result=clone.check_path(goals)
    unchanged=all(np.array_equal(getattr(ex,k),value) for k,value in original.items())
    if calls or clone.steps!=before or not unchanged:
        raise ValueError('L1-B schedule probe is not readonly')
    return dict(label=label,control_step=state['step'],steps_used=clone.steps-before,
        live_executor_unchanged=unchanged,clone_owns_fk=True,urdf_sha256=sha(urdf),
        input_state=jsonable(state),parameters=jsonable(ex.p),model_sha256=hashlib.sha256(canonical(jsonable(ex.model))).hexdigest(),
        exact_target_list_sha256=hashlib.sha256(canonical([g.tolist() for g in goals])).hexdigest(),
        result=jsonable(result),scope='robot_model_only_no_scene_collision_or_success_claim')


def execute_l1b_replay(args, auth, report, ex, qpos, qvel, observe_end, is_stopped):
    """Registered close/path and original suffix, with twenty post-command observations."""
    row=l1b_replay_entry(args,auth)
    source_path=args.code/row['source_result']['path']
    snapshot=args.code/row['snapshots']['precommand']['path']
    if sha(source_path)!=row['source_result']['sha256'] or sha(snapshot)!=row['snapshots']['precommand']['sha256']:
        raise ValueError('L1-B archived evidence changed')
    source=read(source_path);command=row['original_command'];call=command['call'];prefix=row['start_step']
    actual=[c for c in source['commands'] if c['turn']==row['turn']]
    if (actual!=[command] or hashlib.sha256(canonical(command)).hexdigest()!=row['command_sha256']
            or call['poses']!=row['exact_target_list']
            or hashlib.sha256(canonical(call['poses'])).hexdigest()!=row['target_list_sha256']):
        raise ValueError('L1-B fixed metric command/list provenance')
    if call['tool'] not in ('set_gripper','move_to','move_eef_chunk'):
        raise ValueError('unsupported registered L1-B tool')
    frames={f['step']:f for f in source['frames']}
    segments={}
    for label in ('prefix','suffix'):
        segment=row['raw_action_segments'][label]
        selected=[frames[i] for i in range(segment['first_step'],segment['last_step']+1)]
        if len(selected)!=segment['count'] or hashlib.sha256(canonical([dict(step=f['step'],action=f['action']) for f in selected])).hexdigest()!=segment['canonical_actions_sha256']:
            raise ValueError('L1-B original action segment changed')
        segments[label]=selected
    import torch
    frozen=torch.load(snapshot,map_location='cpu',weights_only=False)
    robot=np.asarray(frozen['articulations']['fetch'])
    if robot.shape!=(1,43):raise ValueError('L1-B snapshot robot layout')
    expected_q=robot[0,13:28];expected_v=robot[0,28:43]
    snap=row['snapshots']['precommand']
    if not np.array_equal(expected_q,snap['qpos']) or not np.array_equal(expected_v,snap['qvel']):
        raise ValueError('L1-B snapshot joint/velocity binding')
    recipe=row['precommand_controller_recipe']
    if recipe['steps']!=prefix or recipe['max_steps']!=ex.max_steps:
        raise ValueError('L1-B controller horizon mismatch')
    report.update(stage='L1',replay=True,replay_group='L1-B',replay_arm=args.replay_arm,replay_turn=args.replay_turn,
        system_revision=auth['system_revision'],source_manifest_sha256=auth['source_manifest_sha256'],
        authorization_sha256=sha(args.authorization),replay_variant=auth['replay_variant'],
        effective_controller_flags=replay_executor_config(auth,pick.PROFILES[args.profile],args.replay_arm),
        replay_source=dict(result_sha256=row['source_result']['sha256'],snapshot_sha256=snap['sha256'],
            prefix_steps=prefix,source_path=row['source_result']['path'],l1b_evidence_sha256=auth['l1b_sources_and_scores']['sha256'],
            command_sha256=row['command_sha256'],target_list_sha256=row['target_list_sha256']),
        api_requests=0,verification_requests=0)
    ex._base_active=True
    try:
        for frame in segments['prefix']:
            if not ex._step(vector(frame['action'],13),'recorded_action_prefix'):
                raise ValueError('official termination before L1-B command')
    finally:ex._base_active=False
    checks=dict(qpos=bool(np.allclose(qpos(),expected_q,rtol=0,atol=1e-6)),qvel=bool(np.allclose(qvel(),expected_v,rtol=0,atol=1e-6)),
        qpos_max_error=float(np.max(abs(qpos()-expected_q))),qvel_max_error=float(np.max(abs(qvel()-expected_v))))
    report['precommand_state_check_eval_only']=checks;save(args.output/'precommand-state-check.eval-only.json',checks)
    if not checks['qpos'] or not checks['qvel'] or ex.steps!=prefix:
        raise ValueError('L1-B precommand state failed exact registered tolerance')
    ex.previous_delta=vector(recipe['previous_delta'],15).copy();ex._initial_base=vector(recipe['initial_base'],3).copy()
    ex.gripper=recipe['gripper'];ex.last_arm_target=vector(recipe['last_arm_target'],15).copy()
    ex.last_plan_endpoint=vector(recipe['last_plan_endpoint'],15).copy()
    goal=recipe['last_goal']['pose'];ex.last_goal=pick.pose(goal['position'],goal['quaternion_xyzw'])
    ex.deadline=ex._guard_deadline=ex.max_steps
    report['controller_state_restored']=jsonable(dict(recipe,reconstruction_sha256=auth['l1b_sources_and_scores']['sha256']))
    report['commands'].append(dict(turn=-1,call={'tool':'recorded_action_prefix'},result=dict(steps_used=prefix),start_step=0))
    if call['tool']!='set_gripper':
        goals=[pick.pose(p['position'],p['quaternion_xyzw']) for p in call['poses']]
        report['l1_schedule_probes_eval_only']=[l1b_schedule_probe(args,ex,goals,
            dict(qpos=qpos().tolist(),qvel=qvel().tolist(),previous_delta=ex.previous_delta.tolist(),step=ex.steps),recipe,'pre_entry_settle')]
    force_before=float(scalar(report.get('official_final_info',report['initial_info_eval_only'])['robot_cumulative_force']))
    if call['tool']=='set_gripper':feedback=pick.set_gripper(ex,call['gripper'])
    else:
        goals=[pick.pose(p['position'],p['quaternion_xyzw'])for p in call['poses']]
        feedback=pick.move_to(ex,goals[0]) if call['tool']=='move_to' else pick.move_eef_chunk(ex,goals)
    report['commands'].append(dict(turn=row['turn'],call=call,result=feedback,start_step=prefix))
    if call['tool']!='set_gripper':
        settled=[t for t in ex.trace if t['step']>prefix and t['kind']=='base_settle' and t.get('phase')=='tool_entry']
        if not settled:
            raise ValueError('L1-B post-entry-settle state unavailable')
        t=settled[-1]
        report['l1_schedule_probes_eval_only'].append(l1b_schedule_probe(args,ex,goals,
            dict(qpos=t['qpos'],qvel=t['qvel'],previous_delta=t['delta'],step=t['step']),recipe,'post_entry_settle'))
    force_end=float(scalar(report['official_final_info']['robot_cumulative_force']))
    report['l1_command_eval_only']=dict(force_before=force_before,force_end=force_end,command_force=force_end-force_before,
        feedback=feedback,end_step=ex.steps,group='nonbase_regression')
    # Preserve command-end sensors before any suffix; a media/sensor failure aborts here.
    observe_end()
    suffix_start=ex.steps;ex.deadline=ex._guard_deadline=ex.max_steps
    ex._base_active=True
    try:
        for frame in segments['suffix']:
            if is_stopped() or not ex._available():break
            if not ex._step(vector(frame['action'],13),'recorded_action_suffix',original_step=frame['step']):break
    finally:ex._base_active=False
    suffix_used=ex.steps-suffix_start
    report['l1_suffix_eval_only']=dict(expected_steps=len(segments['suffix']),steps=suffix_used,
        complete=suffix_used==len(segments['suffix']),strict_success=bool(report['strict_pick_success']),
        original_action_sha256=row['raw_action_segments']['suffix']['canonical_actions_sha256'],
        unavailable_reason=None if suffix_used==len(segments['suffix']) else 'official termination or original horizon',
        start_step=suffix_start,end_step=ex.steps)
    hold=vector(qpos(),15).copy();previous=np.zeros(13);post_start=ex.steps
    remaining_observations=max(0,20-suffix_used)
    ex.deadline=ex._guard_deadline=min(ex.max_steps,ex.steps+remaining_observations)
    while ex.steps-post_start<remaining_observations and not is_stopped() and ex._available():
        action=ex.settling_action(hold,previous);action[11:]=0.
        ex._step(action,'replay_post_command_hold',reference_qpos=hold.tolist());previous=action
    report['l1_post_command_eval_only']=dict(steps=ex.steps-suffix_start,
        cumulative_force=float(scalar(report['official_final_info']['robot_cumulative_force'])),
        suffix_steps=suffix_used,hold_steps=ex.steps-post_start,original_suffix_steps=len(segments['suffix']),
        start_step=suffix_start,required_minimum_steps=20,
        observation_definition='continuous command-end observations including original suffix; zero-target hold only fills remaining steps to20',
        unavailable_reason='official termination or original horizon' if ex.steps-suffix_start<20 else None)
    report.update(status='completed',termination_category='official_success' if report['strict_pick_success'] else
        'cumulative_force_limit' if float(scalar(report['official_final_info']['robot_cumulative_force']))>=5000 else
        'time_limit' if is_stopped() else 'recorded_action_replay_complete',executor_trace=ex.trace)


def execute_l1_replay(args, auth, report, ex, qpos, qvel, observe_end, is_stopped):
    """One registered arm, one execution; shared live wiring records all physics."""
    if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
        pick.validate_r32_replay_scope(auth,pick.PROFILES[args.profile],case_id=args.case_id)
        if getattr(args,'replay_group','L1-A')=='L1-B':
            return execute_l1b_replay(args,auth,report,ex,qpos,qvel,observe_end,is_stopped)
        if getattr(args,'replay_group','L1-A')!='L1-A':raise ValueError('r32 replay group')
    if sha(args.replay_roster) != auth['replay_roster_sha256']:
        raise ValueError('L1 command roster changed')
    roster = read(args.replay_roster)
    matches = [r for r in roster['roster'] if 'baseline-'+r['case'] == args.case_id and r['turn'] == args.replay_turn]
    if len(matches) != 1 or args.replay_arm not in (('control','candidate') if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE) else ('control','abort')):
        raise ValueError('L1 command/arm binding')
    entry = matches[0]
    result_path = args.code / entry['result_path']
    snapshot = args.code / entry['precommand_snapshot']
    if sha(result_path) != entry['result_sha256'] or sha(snapshot) != entry['snapshot_sha256']:
        raise ValueError('L1 archived evidence changed')
    source = read(result_path)
    commands = [c for c in source['commands'] if c['turn'] == entry['turn']]
    if len(commands) != 1 or commands[0]['call'] != entry['command'] or commands[0]['start_step'] != entry['prefix_steps']:
        raise ValueError('L1 command provenance mismatch')
    import torch
    frozen = torch.load(snapshot, map_location='cpu', weights_only=False)
    robot = np.asarray(frozen['articulations']['fetch'])
    if robot.shape != (1, 43):
        raise ValueError('L1 snapshot robot layout must be exactly (1,43)')
    expected_q = robot[0, 13:28]
    expected_v = robot[0, 28:43]
    if not np.array_equal(expected_q, entry['precommand_qpos']):
        raise ValueError('L1 snapshot joint binding')
    if entry['precommand_qvel'] is not None and not np.array_equal(expected_v, entry['precommand_qvel']):
        raise ValueError('L1 snapshot velocity binding')
    recovered = replay_command_state(source, entry, expected_v if not entry['prefix_steps'] else None)
    reconstruction_path = Path(auth['controller_reconstruction_path'])
    if sha(reconstruction_path) != auth['controller_reconstruction_sha256']:
        raise ValueError('L1 controller reconstruction evidence changed')
    reconstruction = read(reconstruction_path)
    states = [s for s in reconstruction['rows'] if s['case'] == entry['case'] and s['turn'] == entry['turn']]
    if len(states) != 1:
        raise ValueError('L1 controller reconstruction missing')
    original = states[0]
    if original['result_sha256'] != entry['result_sha256'] or original['snapshot_sha256'] != entry['snapshot_sha256']:
        raise ValueError('L1 controller reconstruction source mismatch')
    if original['max_steps'] != ex.max_steps or original['steps'] != entry['prefix_steps']:
        raise ValueError('L1 controller reconstruction horizon mismatch')
    for name in ('previous_delta', 'initial_base', 'gripper'):
        if not np.array_equal(original[name], recovered[name]):
            raise ValueError('L1 controller reconstruction mismatch: '+name)
    report.update(stage='L1', replay=True, replay_arm=args.replay_arm, replay_turn=args.replay_turn,
                  replay_source=dict(result_sha256=entry['result_sha256'], snapshot_sha256=entry['snapshot_sha256'],
                                     prefix_steps=entry['prefix_steps'], source_path=entry['result_path']),
                  api_requests=0, verification_requests=0)
    if auth.get('scope') in (pick.R31_REPLAY_SCOPE,pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
        if auth.get('scope') in (pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
            report['replay_group']='L1-A'
        report.update(system_revision=auth['system_revision'],
                      source_manifest_sha256=auth['source_manifest_sha256'],
                      authorization_sha256=sha(args.authorization),
                      replay_variant=auth['replay_variant'], effective_controller_flags=
                      replay_executor_config(auth, pick.PROFILES[args.profile], args.replay_arm))
    prefix = entry['prefix_steps']
    if len(source['frames']) < prefix or [f['step'] for f in source['frames'][:prefix]] != list(range(1, prefix+1)):
        raise ValueError('L1 prefix action evidence incomplete')
    # Use the original action channels. _step also maintains the authoritative counter.
    ex._base_active = True
    try:
        for frame in source['frames'][:prefix]:
            if not ex._step(vector(frame['action'], 13), 'recorded_action_prefix'):
                raise ValueError('official termination before L1 command')
    finally:
        ex._base_active = False
    checks = dict(qpos=bool(np.allclose(qpos(), expected_q, rtol=0, atol=1e-6)),
                  qvel=bool(np.allclose(qvel(), expected_v, rtol=0, atol=1e-6)),
                  qpos_max_error=float(np.max(abs(qpos()-expected_q))),
                  qvel_max_error=float(np.max(abs(qvel()-expected_v))))
    report['precommand_state_check_eval_only'] = checks
    save(args.output/'precommand-state-check.eval-only.json', checks)
    if not checks['qpos'] or not checks['qvel']:
        raise ValueError('L1 precommand state failed exact registered tolerance')
    ex.previous_delta = recovered['previous_delta'].copy()
    ex._initial_base = recovered['initial_base'].copy()
    ex.gripper = recovered['gripper']
    ex.last_arm_target = vector(original['last_arm_target'], 15).copy()
    ex.last_plan_endpoint = vector(original['last_plan_endpoint'], 15).copy()
    recipe = original['last_goal_recipe']
    ex.last_goal = ex.fk.tcp(vector(recipe['qpos'],15)) if recipe['kind'] == 'fk_tcp' else pick.pose(recipe['pose']['position'], recipe['pose']['quaternion_xyzw'])
    ex._guard_deadline = ex.deadline = ex.max_steps
    report['controller_state_restored'] = jsonable(dict(recovered, last_arm_target=ex.last_arm_target,
        last_plan_endpoint=ex.last_plan_endpoint, last_goal=ex.last_goal,
        reconstruction_sha256=auth['controller_reconstruction_sha256']))
    if prefix:
        report['commands'].append(dict(turn=-1, call={'tool':'recorded_action_prefix'},
                                       result=dict(steps_used=prefix), start_step=0))
    force_before = float(scalar(report.get('official_final_info',report['initial_info_eval_only'])['robot_cumulative_force']))
    feedback = pick.move_base(ex, entry['command']['turn_rad'], entry['command']['forward_m'])
    report['commands'].append(dict(turn=entry['turn'], call=entry['command'], result=feedback, start_step=prefix))
    force_end = float(scalar(report['official_final_info']['robot_cumulative_force']))
    report['l1_command_eval_only'] = dict(archived_force=entry['archived_command_force'],
        force_before=force_before, force_end=force_end, command_force=force_end-force_before,
        feedback=feedback, end_step=ex.steps, group=entry['group'])
    observe_end()
    hold = vector(qpos(), 15).copy()
    previous = np.zeros(13)
    post_start = ex.steps
    ex.deadline = ex._guard_deadline = min(ex.max_steps, ex.steps+20)
    while ex.steps-post_start < 20 and not is_stopped() and ex._available():
        action = ex.settling_action(hold, previous)
        action[11:] = 0.
        ex._step(action, 'replay_post_command_hold', reference_qpos=hold.tolist())
        previous = action
    report['l1_post_command_eval_only'] = dict(steps=ex.steps-post_start,
        cumulative_force=float(scalar(report['official_final_info']['robot_cumulative_force'])),
        unavailable_reason='official termination or original horizon' if ex.steps-post_start < 20 else None)
    report.update(status='completed', termination_category='official_success' if report['strict_pick_success'] else
        'cumulative_force_limit' if float(scalar(report['official_final_info']['robot_cumulative_force'])) >= 5000 else
        'time_limit' if is_stopped() else 'recorded_action_replay_complete', executor_trace=ex.trace)


class StrictInitializationMismatch(ValueError):
    """Only a completed, finite strict comparison may produce this classification."""


def check_initialization_only(env, info, args, auth, row, horizon, report, snapshot_writer):
    """Original identity/controller/FK and same-final-attempt hard comparison, no actions."""
    u=env.unwrapped
    if str(u.task_plan[0].composite_subtask_uids[0])!=row['uid'] or u.spawn_selection_idxs[0]!=row['spawn_index']:
        raise ValueError('runtime spawn binding')
    if int(scalar(info['subtasks_steps_left']))!=horizon:
        raise ValueError('reset must leave exactly authorized horizon; no counter repair')
    if float(u.pick_cfg.robot_cumulative_force_limit)!=5000 or float(u.pick_cfg.ee_rest_thresh)!=0.05:
        raise ValueError('official criteria changed')
    validate_official_controller(u.agent.controller)
    validate_base_runtime(u.agent.controller,u.control_freq)
    q=array(u.agent.robot.qpos)[0].astype(float)
    fk=RobotFK(args.code/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',[j.name for j in u.agent.robot.active_joints])
    base=np.asarray(u.agent.base_link.pose[0].sp.to_transformation_matrix())
    tcp=np.asarray(u.agent.tcp.pose[0].sp.to_transformation_matrix())
    fk_tcp=np.asarray(fk.tcp(q))
    if q.shape!=(15,) or not np.isfinite(q).all() or any(v.shape!=(4,4) or not np.isfinite(v).all() for v in (base,tcp,fk_tcp)):
        raise ValueError('FK finite shape')
    difference=np.max(abs(fk_tcp-np.linalg.inv(base)@tcp))
    if not np.isfinite(difference) or difference>1e-5:
        raise ValueError('FK identity')
    configs={}
    for name in ('arm','body','gripper','base'):
        config=u.agent.controller.controllers[name].config
        keys=('joint_names','normalize_action','lower','upper')+(('use_delta','use_target') if name in ('arm','body') else ('use_delta',) if name=='gripper' else ())
        configs[name]={key:jsonable(getattr(config,key)) for key in keys}
    evidence=dict(uid=str(u.task_plan[0].composite_subtask_uids[0]),spawn_index=int(u.spawn_selection_idxs[0]),
        horizon=horizon,steps_left=int(scalar(info['subtasks_steps_left'])),internal_pick_horizon=int(u.pick_cfg.horizon),
        robot_cumulative_force_limit=float(u.pick_cfg.robot_cumulative_force_limit),ee_rest_thresh=float(u.pick_cfg.ee_rest_thresh),
        action_mapping=jsonable(u.agent.controller.action_mapping),controller_configs=configs,control_freq=float(u.control_freq),
        joint_names=[j.name for j in u.agent.robot.active_joints],qpos=q.tolist(),base_world=base.tolist(),tcp_world=tcp.tolist(),fk_tcp=fk_tcp.tolist())
    path=args.output/'initialization-runtime-checks.eval-only.json'
    _initialization_json_exclusive(path,evidence)
    report['initialization_runtime_checks_eval_only']=dict(path=path.name,sha256=sha(path))
    report['initialization_checks_eval_only']=dict(identity=True,horizon=True,official_criteria=True,controller=True,base_runtime=True,FK=True)
    report['initial_binding']=dict(uid=row['uid'],spawn_index=row['spawn_index'],seed=row['seed'],qpos=q.tolist(),base_world=base.tolist())
    save(args.output/'initial-binding.json',report['initial_binding'])
    report.update(configured_horizon=horizon,internal_pick_horizon=int(u.pick_cfg.horizon),initial_steps_left=int(scalar(info['subtasks_steps_left'])))
    nominal=json.loads(Path(row['initial_binding_path']).read_bytes())
    try:
        record_final_anchor_reset(args.output,auth,args.case_id,report,nominal,q,
            array(u.agent.robot.pose.raw_pose)[0],u.get_state_dict(),snapshot_writer=snapshot_writer)
    except ValueError as error:
        # The exact residual exception alone is insufficient: require its complete saved evidence.
        if str(error)!='baseline differs from frozen initial state':
            raise
        verify_initialization_only_comparison(args.output)
        raise StrictInitializationMismatch(str(error)) from error
    verify_initialization_only_comparison(args.output)


def verify_initialization_only_comparison(output):
    diag=read(Path(output)/'baseline-reset-diagnostics.eval-only.json')
    snapshot=diag.get('snapshot',{})
    if snapshot.get('status')!='saved' or snapshot.get('path')!='reset-state.eval-only.pt':
        raise ValueError('initialization-only full strict snapshot required')
    path=Path(output)/snapshot['path']
    if sha(path)!=snapshot.get('sha256'):
        raise ValueError('initialization-only strict snapshot hash')
    pick.replay_anchor_snapshot(path)
    if diag.get('rtol')!=0 or diag.get('atol')!=1e-6 or not diag.get('fields'):
        raise ValueError('initialization-only strict comparison missing')
    for item in diag['fields'].values():
        if not np.asarray(item['reference_finite']).all() or not np.asarray(item['actual_finite']).all():
            raise ValueError('initialization-only nonfinite strict evidence')


def run_initialization_only(output, report, *, make_env, env_config, seed, reset_options,
                            metadata, strict_check, snapshot_writer):
    """The complete action-free core, also exercised directly by CPU substitute tests."""
    env=None;stage=pick.INITIALIZATION_OBSERVABILITY_STAGES[0]
    report.update(explicit_reset_calls=0,environment_closed=False,strict_initialization_match=None,
                  initialization_boundaries_eval_only=[],started_epoch=time.time(),pid=os.getpid())
    def capture(**kwargs):
        report['initialization_boundaries_eval_only'].append(capture_initialization_boundary(
            output,stage,snapshot_writer=snapshot_writer,**kwargs))
    try:
        capture(metadata=metadata)
        stage='construction'
        _initialization_json_exclusive(Path(output)/'initialization-construction.entered.eval-only.json',dict(stage=stage,evaluation_only=True))
        env=make_env(env_config)
        env.env.auto_reset=False
        stage=pick.INITIALIZATION_OBSERVABILITY_STAGES[1]
        capture(state_getter=env.unwrapped.get_state_dict)
        stage='explicit_reset'
        _initialization_json_exclusive(Path(output)/'initialization-explicit-reset.entered.eval-only.json',dict(stage=stage,evaluation_only=True))
        report['explicit_reset_calls']+=1
        obs,info=env.reset(seed=seed,options=reset_options)
        stage=pick.INITIALIZATION_OBSERVABILITY_STAGES[2]
        capture(state_getter=env.unwrapped.get_state_dict,info=info)
        stage='strict_initialization_check'
        strict_check(env,info)
        report.update(status='initialization_match',strict_initialization_match=True)
    except Exception as error:
        mismatch=isinstance(error,StrictInitializationMismatch) and stage=='strict_initialization_check'
        report.update(status='initialization_mismatch' if mismatch else 'infrastructure_censored',
            strict_initialization_match=False if mismatch else None,error_type=type(error).__name__,error_message=str(error))
        evidence=record_initialization_error(output,stage,error,primary=True)
        report['initialization_error_eval_only']=evidence
        if evidence.get('recording_secondary_error'):
            report['status']='infrastructure_censored'
    finally:
        if env is not None:
            try:
                env.close();report['environment_closed']=True
            except Exception as error:
                primary='initialization_error_eval_only' not in report
                evidence=record_initialization_error(output,'close',error,primary=primary)
                report['initialization_close_error_eval_only']=evidence
                report.setdefault('initialization_secondary_errors_eval_only',[]).append(evidence)
                if primary:
                    report.update(error_type=type(error).__name__,error_message=str(error),initialization_error_eval_only=evidence)
                report['status']='infrastructure_censored'
        report['ended_epoch']=time.time()
        if report['ended_epoch']-report['started_epoch']>600:
            report['status']='infrastructure_censored'
            report['resource_error']='worker wall limit exceeded'
    return report


def load_memory_full_b(code, auth):
    """Hash-bound import only; never constructs a task or silently falls back."""
    import importlib
    code=Path(code).resolve()
    relative=Path('research/c2/diagnostics/2026-10-10-flowsde-controller-cpu')
    manifest=code/relative/'task-rules-cpu/controller-memory-handoff.json'
    if auth.get('controller_version')!='full-B-r4-plus-task-rules-v1':
        raise ValueError('Memory A/B requires the adopted full B executor')
    if auth.get('controller_manifest_sha256')!=sha(manifest):
        raise ValueError('Full B handoff manifest not bound to authorization')
    receipt=json.loads(manifest.read_text())
    for name,digest in receipt['files'].items():
        if sha(code/name)!=digest:raise ValueError('Full B source changed: '+name)
    full=code/relative/'full-stack'
    for name in ('fetch_runtime','fetch_episode','fetch_tracker','fetch_driver','preflight','task_rules','flowsde_ctrl'):
        loaded=sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(code/relative):
            raise ValueError('Conflicting full B module: '+name)
    sys.path[:0]=[str(full/'upstream'),str(full),str(full.parent),str(code/'scripts')]
    return full,importlib.import_module('fetch_runtime'),importlib.import_module('preflight')


def pick20_reset_check(reference, actual, root_pose, state):
    """Compare every available archived field; missing fields are not equality."""
    checks, missing = {}, []
    def compare(name, expected, value):
        a, b = np.asarray(expected), np.asarray(value)
        checks[name] = bool(a.shape == b.shape and np.isfinite(a).all() and
            np.isfinite(b).all() and np.allclose(a, b, rtol=0, atol=1e-6))
    for key in ('uid', 'spawn_index', 'seed'):
        if key in reference: checks[key] = reference[key] == actual[key]
        else: missing.append(key)
    for key in ('qpos', 'base_world', 'target_world'):
        if key in reference: compare(key, reference[key], actual[key])
        else: missing.append(key)
    if 'root_pose' in reference: compare('root_pose', reference['root_pose'], root_pose)
    else: missing.append('root_pose')
    for name, pose in reference.get('objects', {}).items():
        actor = state['actors'].get('env-0_'+name)
        if actor is None: checks['object:'+name] = False
        else: compare('object:'+name, pose, array(actor)[0][:7])
    if not reference.get('objects'): missing.append('object_poses')
    if 'qpos' not in reference or not ('root_pose' in reference or 'base_world' in reference):
        raise ValueError('pick20 reference lacks required robot initial state')
    return dict(passed=all(checks.values()), checks=checks, missing_historical_fields=missing,
                comparison='available historical fields only; not full state/RNG equivalence',
                evaluation_only=True)


def pick20_initial_admission(auth, case_id, comparison):
    """A user-approved evaluation-only exception; never change historical checks."""
    policy=auth.get('initial_state_exception')
    failed={k for k,v in comparison['checks'].items() if not v}
    if not failed:return dict(allowed=True,historical_match=True,exception_used=False)
    expected={'object:024_bowl-0','object:008_pudding_box-2'}
    valid=(auth.get('scope')==pick.PICK20_BATCH_SCOPE and auth.get('pick20_remaining') is not None
        and case_id=='baseline-017' and isinstance(policy,dict)
        and policy.get('case_id')==case_id and policy.get('approved') is True
        and policy.get('mode')=='natural_reset_non_target_exception'
        and set(policy.get('allowed_mismatch_keys',[]))==expected and failed<=expected)
    return dict(allowed=bool(valid),historical_match=False,exception_used=bool(valid),
                mismatched_keys=sorted(failed),evaluation_only=True)


def pick20_factory(make_env, code, auth, profile):
    """Fail closed on profile/scope mismatch before native environment creation."""
    scope=(ALIGNED_PREFLIGHT_SCOPE if auth.get('scope')==ALIGNED_PREFLIGHT_SCOPE else pick.PICK20_BATCH_SCOPE if auth.get('scope')==pick.PICK20_BATCH_SCOPE else pick.PICK20_ALIGNMENT_SCOPE) if profile==pick.SOL_PICK20_ALIGNED else pick.PICK20_SCOPE
    if profile in pick.DEV_PROFILES:

        pick.validate_session_scope(auth,profile)

        scope=auth['scope'] if profile in pick.PAIRED_PROFILES else pick.DEV_R1_BUDGET_SCOPE if profile.r1_extended_budget else pick.DEV_R1_SCOPE

    if profile.weekly_budget:
        pick.validate_weekly_scope(auth,profile)
        scope=pick.WEEKLY_SCOPE
    if profile not in (pick.SOL_PICK20,pick.SOL_PICK20_ALIGNED)+pick.DEV_PROFILES+pick.WEEKLY_PROFILES or auth.get('scope') != scope:

        raise ValueError('pick20 explicit factory binding required')
    full = load_memory_full_b(code, auth)
    return full, full[1].bind_fetch_factory(make_env, full[1].agent_class())


def direct_timed_call(name, function, *args, **kwargs):
    return function(*args, **kwargs)


class MemoryPhaseTimer:
    """Evaluation-only wall timings; unmatched begin survives a killed worker.

    Nested durations are inclusive and must not be summed across levels.
    No device synchronization is introduced; these are host wall times.
    """
    def __init__(self, output, clock=None):
        self.path = Path(output) / 'phase-timing.eval-only.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or time.monotonic
        self.origin = self.clock()
        self.sequence = 0

    def emit(self, event):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
            stream.flush()

    def __call__(self, name, function, *args, **kwargs):
        self.sequence += 1
        ident = self.sequence
        start = self.clock()
        self.emit(dict(id=ident, phase=name, event='begin', elapsed_s=start-self.origin))
        outcome = 'completed'
        try:
            return function(*args, **kwargs)
        except BaseException:
            outcome = 'error'
            raise
        finally:
            end = self.clock()
            self.emit(dict(id=ident, phase=name, event='end', elapsed_s=end-self.origin,
                           wall_s=end-start, outcome=outcome))


def memory_recording_snapshot(base, spawn_reference, spawn_sha256):
    """Full dynamic snapshot, replacing only the identified static spawn dataset.

    The environment is never changed. The immutable dataset has separate full
    boundary hashes; every other captured field uses the original clone policy.
    This is an audit snapshot, not a restorable transport or a paired initial state.
    """
    from bvi.pick_system.snapshot_transport import clone_snapshot_value, native_buffer_layout
    from check_c2_snapshot_restore import capture, targets
    if base.uenv.spawn_data is not spawn_reference:
        raise ValueError('recording spawn_data object replaced')
    holders=[(owner,key,obj) for owner,obj in targets(base)
             for key,value in vars(obj).items() if value is spawn_reference]
    if len(holders)!=1 or holders[0][1]!='spawn_data' or holders[0][2] is not base.uenv:
        raise ValueError('recording spawn_data ownership changed')
    def clone(value):
        if value is spawn_reference:
            return dict(contract='immutable-spawn-dataset-token-v1',sha256=spawn_sha256,
                        type=type(value).__module__+'.'+type(value).__qualname__,entries=len(value))
        return clone_snapshot_value(value)
    return dict(contract='memory-recording-dynamic-audit-v1',
                layout=native_buffer_layout(base.uenv.scene),state=capture(base,clone_fn=clone))


class MemoryRecordingState:
    """Keep full dynamic coverage; audit the large static spawn cache separately."""
    def __init__(self, base, preflight, output, timed=direct_timed_call):
        from bvi.pick_system.snapshot_transport import capture_transport
        from check_c2_snapshot_restore import targets
        self.base,self.preflight,self.timed=base,preflight,timed
        self.capture_full=capture_transport
        self.spawn_reference=base.uenv.spawn_data
        owners=[name for name,obj in targets(base) if obj is base.uenv]
        if len(owners)!=1:raise ValueError('recording environment owner identity')
        self.spawn_sha256=timed('recording.static_spawn.initial_hash',
                                preflight.state_fingerprint,self.spawn_reference)
        self.path=Path(output)/'recording-state-audit.eval-only.json'
        self.incremental_audit=False
        self.full=True;self.first_rest=False;self.index=None;self.phase=None
        self.audit=dict(contract='memory-recording-dynamic-audit-v1',evaluation_only=True,
            dynamic_coverage='original capture: simulator, all native buffers/layout, all other attributes, controller, task, RNG',
            static_fields=[dict(owner=owners[0],field='spawn_data',sha256=self.spawn_sha256,
                type=type(self.spawn_reference).__module__+'.'+type(self.spawn_reference).__qualname__,
                entries=len(self.spawn_reference))],
            full_state_frames=[],frame_guards=[],static_checks=[],finished=False)
        save(self.path,self.audit)

    def check_static(self, stage):
        if self.base.uenv.spawn_data is not self.spawn_reference:
            raise ValueError('recording spawn_data object replaced')
        current=self.timed('recording.static_spawn.boundary_hash',
                           self.preflight.state_fingerprint,self.spawn_reference)
        check=dict(stage=stage,sha256=current,passed=current==self.spawn_sha256)
        self.audit['static_checks'].append(check);save(self.path,self.audit)
        if not check['passed']:raise ValueError('recording static spawn_data mutated')

    def rebind_after_restore(self):
        if self.audit['frame_guards'] or self.index is not None:
            raise ValueError('Recording restore must precede the first frame')
        restored=self.base.uenv.spawn_data
        digest=self.preflight.state_fingerprint(restored)
        if digest!=self.spawn_sha256:
            raise ValueError('Restored recording spawn_data content changed')
        self.spawn_reference=restored
        self.check_static('cold_restore_rebind')

    def before_frame(self, index, phase):
        self.index,self.phase=index,phase
        self.full=index in (0,1) or (phase=='rest' and not self.first_rest)
        if phase=='rest':self.first_rest=True
        if self.full:
            self.check_static('before_frame_'+str(index))
            self.audit['full_state_frames'].append(index)

    def __call__(self):
        if self.base.uenv.spawn_data is not self.spawn_reference:
            raise ValueError('recording spawn_data object replaced')
        value=(self.capture_full(self.base) if self.full else
               memory_recording_snapshot(self.base,self.spawn_reference,self.spawn_sha256))
        digest=self.preflight.state_fingerprint(value)
        self.audit['frame_guards'].append(dict(frame=self.index,phase=self.phase,
            coverage='full_transport' if self.full else 'all_dynamic_plus_static_token',sha256=digest))
        if self.incremental_audit:
            with self.path.with_suffix('.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(self.audit['frame_guards'][-1],separators=(',',':'))+'\n')
                stream.flush();os.fsync(stream.fileno())
        else:
            save(self.path,self.audit)
        return digest

    def finish(self):
        self.check_static('finish')
        guards=self.audit['frame_guards']
        if not guards or len(guards)%2 or any(a['frame']!=b['frame'] or a['coverage']!=b['coverage'] or a['sha256']!=b['sha256']
                               for a,b in zip(guards[::2],guards[1::2])):
            raise ValueError('recording state audit incomplete or changed')
        frames=guards[::2]
        if [row['frame'] for row in frames]!=list(range(len(frames))):
            raise ValueError('recording state audit frames not consecutive')
        required={0,1}.intersection(range(len(frames)))
        first_rest=next((row['frame'] for row in frames if row['phase']=='rest'),None)
        if first_rest is not None:required.add(first_rest)
        if any(frames[index]['coverage']!='full_transport' for index in required):
            raise ValueError('recording state audit full boundary missing')
        self.audit['finished']=True;save(self.path,self.audit)


class TransportRecordingState(MemoryRecordingState):
    """SequentialTask has no spawn cache; retain full transport at every frame."""
    def __init__(self,base,preflight,output,timed=direct_timed_call):
        from bvi.pick_system.snapshot_transport import capture_transport
        self.base,self.preflight,self.timed=base,preflight,timed
        self.capture_full=capture_transport
        self.path=Path(output)/'recording-state-audit.eval-only.json'
        self.index=None;self.phase=None
        self.audit=dict(contract='transport-recording-full-audit-v1',evaluation_only=True,
            static_fields=[],full_state_frames=[],frame_guards=[],static_checks=[],finished=False)
        save(self.path,self.audit)

    def check_static(self,stage):
        pass  # No omitted/static-token fields: every capture is complete.

    def before_frame(self,index,phase):
        self.index,self.phase=index,phase
        self.audit['full_state_frames'].append(index)

    def __call__(self):
        digest=self.preflight.state_fingerprint(self.capture_full(self.base))
        self.audit['frame_guards'].append(dict(frame=self.index,phase=self.phase,
            coverage='full_transport',sha256=digest))
        save(self.path,self.audit)
        return digest


class AlignedContactDiagnostics:
    """Evaluation-only pair impulses for one GPU environment, never tool feedback.

    One batched query covers every robot link (including fingers) against every
    rigid/articulation body, including self contacts. Only the last physics
    substep is sampled; zero net impulse does not prove absence of contact.
    """
    def __init__(self,base,output):
        self.base=base;self.path=Path(output)/'contacts.eval-only.jsonl'
        self.query=None;self.pairs=None;self.last=-1

    def __call__(self,index):
        u=self.base.uenv;scene=u.scene;px=scene.px
        if index!=self.last+1:raise ValueError('Nonconsecutive contact diagnostic')
        if not scene.gpu_sim_enabled or int(u.num_envs)!=1:
            raise ValueError('Contact diagnostic requires one GPU environment')
        if self.query is None:
            bodies=list(px.rigid_dynamic_components)+list(px.rigid_static_components)+list(px.articulation_link_components)
            bodies=list({id(b):b for b in bodies}.values())
            links=list(u.agent.robot.get_links());pairs=[];labels=[]
            for link in links:
                if len(link._bodies)!=1:raise ValueError('Contact link batch mismatch')
                body=link._bodies[0]
                for number,other in enumerate(bodies):
                    if body is other:continue
                    pairs.append((body,other))
                    labels.append(dict(robot_link=link.name,other_body_index=number,
                        other_name=other.entity.name))
            if not pairs:raise ValueError('Empty contact inventory')
            self.query=px.gpu_create_contact_pair_impulse_query(pairs)
            self.pairs=labels
            save(self.path.with_suffix('.manifest.json'),dict(evaluation_only=True,
                pairs=labels,includes_fingers=True,includes_self_contacts=True,
                sampling='last physics substep per recorded control step; not all substeps',
                zero_semantics='omitted rows have exactly zero net impulse, not proof of no contact'))
        px.gpu_query_contact_pair_impulses(self.query)
        impulses=array(self.query.cuda_impulses.torch().clone()).astype(float)
        dt=float(px.timestep)
        if impulses.shape!=(len(self.pairs),3) or not np.isfinite(impulses).all() or dt<=0:
            raise ValueError('Invalid contact diagnostic readback')
        rows=[dict(pair=i,impulse_ns=v.tolist(),force_n=(v/dt).tolist())
              for i,v in enumerate(impulses) if np.any(v!=0)]
        with self.path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(step=index,evaluation_only=True,physics_dt=dt,
                pairs_queried=len(self.pairs),nonzero_pairs=rows))+'\n');stream.flush()
        self.last=index


class IncrementalFrameManifest:
    """Instance-local writer seam for the frozen native recorder.

    Keep native capture/state checks, pixels, per-file hashes and frame values.
    A compact fsynced journal is authoritative between atomic 100-frame snapshots.
    Native record/finish methods receive a private globals dictionary; the frozen
    module and every other recorder retain their original save function.
    """
    def __init__(self, native, checkpoint_every=100, *, current_step_only=False):
        import types
        if type(checkpoint_every) is not int or checkpoint_every < 1:
            raise ValueError('Positive checkpoint interval required')
        if native.frames or (native.root/'frames.incremental.jsonl').exists():
            raise ValueError('Incremental recording requires a fresh empty recorder')
        self.current_step_only=current_step_only
        if type(current_step_only) is not bool:raise ValueError('Boolean incremental mode required')
        if current_step_only:
            if native.trajectory:raise ValueError('Fresh trajectory required')
            class CurrentPoint(list):
                # Frozen recorder appends, then deep-copies this list. Keep only
                # this frame; the durable journal owns all previous points.
                def append(self, point):
                    self[:]=[point]
                def __deepcopy__(self, memo):
                    return copy.deepcopy(list(self),memo)
            native.trajectory=CurrentPoint()
        self.native, self.checkpoint_every, self.count = native, checkpoint_every, 0
        self.journal = native.root/'frames.incremental.jsonl'
        for name in ('record', 'finish'):
            method = getattr(native, name)
            fn = method.__func__
            if 'save' not in fn.__globals__ or 'save' not in fn.__code__.co_names:
                raise ValueError('Native recorder writer contract changed')
            original = fn.__globals__['save']
            def writer(path, value, original=original):
                if Path(path) == native.root/'frames.json':
                    return self.write(path, value)
                return original(path, value)
            bound = types.FunctionType(fn.__code__, dict(fn.__globals__, save=writer),
                fn.__name__, fn.__defaults__, fn.__closure__)
            bound.__kwdefaults__ = fn.__kwdefaults__
            setattr(native, name, types.MethodType(bound, native))

    def _append(self, value):
        with self.journal.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())

    def write(self, path, value):
        frames = value['frames']
        if len(frames) == self.count + 1:
            row = dict(frames[-1]); trajectory = row.pop('trajectory_world')
            if row['step'] != self.count or len(trajectory) != (1 if self.current_step_only else self.count + 1):
                raise ValueError('Nonconsecutive incremental frame')
            self._append(dict(format='native-frame-delta-v1', fps=value['fps'],
                evaluation_only=True, frame=row, tcp_world=trajectory[-1]))
            if self.current_step_only:frames[-1].pop('trajectory_world')
            self.count += 1
        elif len(frames) != self.count or 'attempt_outcome' not in value:
            raise ValueError('Unexpected native manifest write')
        if 'attempt_outcome' in value:
            self._append(dict(format='native-frame-outcome-v1', count=self.count, fps=value['fps'],
                attempt_outcome=value['attempt_outcome']))
        if self.current_step_only:
            # Pointer only, never serialize the growing frame list during capture.
            if self.count==1 or 'attempt_outcome' in value:
                small=dict(format='native-frame-journal-v1',fps=value['fps'],evaluation_only=True,
                    journal=self.journal.name,complete_frames=self.count)
                if 'attempt_outcome' in value:small['attempt_outcome']=value['attempt_outcome']
                temporary=Path(path).with_suffix('.tmp');save(temporary,small);os.replace(temporary,path)
            return
        if self.count == 1 or self.count % self.checkpoint_every == 0 or 'attempt_outcome' in value:
            temporary = Path(path).with_suffix('.tmp')
            save(temporary, value); os.replace(temporary, path)

    @staticmethod
    def recover(journal):
        """Rebuild the exact manifest; a torn final line is explicitly reported.

        The journal is never edited. Malformed complete lines and sequence gaps
        are errors, not silently discarded evidence.
        """
        raw = Path(journal).read_bytes(); lines = raw.splitlines(keepends=True)
        incomplete = bool(lines and not lines[-1].endswith(b'\n'))
        if incomplete: lines = lines[:-1]
        frames=[]; trajectory=[]; fps=None; outcome=None
        for line in lines:
            value=json.loads(line)
            if value['format']=='native-frame-delta-v1':
                row=value['frame']
                if row['step']!=len(frames) or (fps is not None and fps!=value['fps']):
                    raise ValueError('Incremental frame sequence/fps mismatch')
                fps=value['fps']; trajectory.append(value['tcp_world'])
                row['trajectory_world']=[list(point) for point in trajectory]; frames.append(row)
                outcome=None
            elif value['format']=='native-frame-outcome-v1':
                if value['count']!=len(frames):raise ValueError('Incremental outcome sequence mismatch')
                outcome=value['attempt_outcome']; fps=value['fps']
                if frames:frames[-1]['outcome']=outcome
            else:raise ValueError('Unknown incremental manifest format')
        manifest=dict(fps=fps, evaluation_only=True, frames=frames)
        if outcome is not None:manifest['attempt_outcome']=outcome
        return manifest, dict(complete_frames=len(frames), incomplete_final_line=incomplete,
            terminal_recorded=outcome is not None)


class MemoryRecordingPort:
    """One native capture; legacy PNG names share the already checked scene image."""
    action_size = 20

    def __init__(self, native, recording, info, timed=direct_timed_call, state_guard=None, contact_diagnostics=None):
        self.native, self.recording, self.info, self.timed = native, recording, info, timed
        self.state_guard=state_guard
        self.contact_diagnostics=contact_diagnostics
        self.segment_start_step=0
        capture, state = native.capture, state_guard or native.state
        native.capture = lambda: timed('recording.render_and_readback', capture)
        native.state = lambda: timed('recording.state_snapshot', state)

    def record(self, index, action, **kwargs):
        segment_index=index-self.segment_start_step
        if segment_index != self.recording.count:
            raise ValueError('nonconsecutive shared recording')
        if self.state_guard is not None:
            self.state_guard.before_frame(segment_index,kwargs.get('phase'))
        row = self.timed('recording.total', self.native.record, segment_index, action, self.info(), **kwargs)
        if self.contact_diagnostics is not None:
            self.timed('recording.contacts_eval_only',self.contact_diagnostics,segment_index)
        source = self.native.root / row['scene']
        target = self.recording.directory / f'{segment_index:06d}.png'
        # Same output filesystem: fail closed on link failure; never render again.
        self.timed('recording.link_existing_scene', os.link, source, target)
        self.recording.count += 1
        return row

    def finish(self, status):
        if self.state_guard is not None:
            try:self.timed('recording.finish_state_audit',self.state_guard.finish)
            except Exception as error:
                try:self.timed('recording.finish',self.native.finish,'recording_state_invalid')
                except Exception as secondary:error.diagnostic_error=repr(secondary)
                raise
        return self.timed('recording.finish', self.native.finish, status)


def finish_memory_recording(executor, report):
    """Close the recorder before env.close; preserve any earlier failure details."""
    try:executor.finish(report.get('status','failed'))
    except Exception as error:
        report['full_b_recording_error']=repr(error)
        if hasattr(error,'diagnostic_error'):
            report['full_b_recording_secondary_error']=error.diagnostic_error
        report['status']='infrastructure_censored'


def activate_memory_full_b(env, runtime, preflight, output, timed=direct_timed_call):
    """Preserve full plant/task/RNG; serialize large snapshots only on failure.

    All four native captures and exact plant fingerprints are still required.
    Successful activation retains their digests; failure retains full bundles.
    """
    from types import SimpleNamespace
    from bvi.pick_system.snapshot_transport import capture_transport
    base=SimpleNamespace(env=env,uenv=env.unwrapped)
    captures=[]; fingerprints=[]
    def capture():
        value=timed('activation.snapshot_capture',capture_transport,base)
        captures.append(value)
        digest=timed('activation.fingerprint',lambda:preflight.state_fingerprint(runtime.plant_transition_view(value)))
        fingerprints.append(digest)
        return digest
    try:
        before=capture()
        transition=runtime.activate_full_controller(base.uenv.agent,capture_plant_state=capture,states_equal=lambda a,b:a==b)
        transition['action_caches']=runtime.refresh_full_action_spaces(env)
        if capture()!=before:raise ValueError('Full B activation altered plant/task/RNG')
        transition['driver']=runtime.validate_full_driver(base.uenv)
        transition['snapshot_audit']=dict(captures=len(captures),plant_fingerprints=fingerprints,
            passed=len(captures)==4 and len(set(fingerprints))==1,
            retained='exact plant digests; full bundles on failure only',evaluation_only=True)
        if not transition['snapshot_audit']['passed']:
            raise ValueError('Full B activation snapshot audit coverage')
    except BaseException as error:
        for index,value in enumerate(captures):
            try:
                timed('activation.failure_snapshot_write',preflight.write_diagnostic_bundle,
                      Path(output)/'full-b-activation',str(index),value)
            except BaseException as secondary:
                error.diagnostic_error=repr(secondary)
        raise
    save(Path(output)/'full-b-activation.json',transition)
    return base


class IncrementalProgress:
    """Synchronous durable deltas; compact current state, exact recoverable history."""
    fields=('trace','controller_diagnostics')

    def __init__(self, output, timed=direct_timed_call, *, stem='full-b-progress', fields=None):
        if fields is not None:self.fields=tuple(fields)
        self.path=Path(output)/(stem+'.json')
        self.journal=Path(output)/(stem+'.incremental.jsonl')
        if self.path.exists() or self.journal.exists():
            raise ValueError('Fresh progress output required')
        self.counts={k:0 for k in self.fields};self.sequence=0;self.timed=timed

    def snapshot(self, value):
        small={k:v for k,v in value.items() if k not in self.fields}
        deltas={}
        for k in self.fields:
            rows=value.get(k,[])
            if len(rows)<self.counts[k]:raise ValueError('Progress history shrank')
            deltas[k]=rows[self.counts[k]:]
        return copy.deepcopy(dict(state=small,deltas=deltas))

    def __call__(self, value):
        return self.timed('progress.persist',self.write,value)

    def write(self,value):
        event=dict(sequence=self.sequence,**value)
        with self.journal.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        for k in self.fields:self.counts[k]+=len(value['deltas'][k])
        self.sequence+=1
        state=dict(value['state'],history_format='progress-delta-v1',
            history_journal=self.journal.name,history_counts=dict(self.counts),journal_sequence=self.sequence-1)
        temporary=self.path.with_suffix('.tmp')
        with temporary.open('w',encoding='utf8') as stream:
            json.dump(state,stream,separators=(',',':'),allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,self.path)

    def finish(self):
        state,audit=self.recover(self.journal)
        if audit['incomplete_final_line']:raise ValueError('Torn live progress journal')
        temporary=self.path.with_suffix('.tmp')
        save(temporary,state);os.replace(temporary,self.path)

    @staticmethod
    def recover(journal):
        lines=Path(journal).read_bytes().splitlines(keepends=True)
        torn=bool(lines and not lines[-1].endswith(b'\n'))
        if torn:lines=lines[:-1]
        histories=None;state=None
        for index,line in enumerate(lines):
            row=json.loads(line)
            if row['sequence']!=index:raise ValueError('Progress journal sequence mismatch')
            if histories is None:histories={k:[] for k in row['deltas']}
            if set(row['deltas'])!=set(histories):raise ValueError('Progress delta fields changed')
            for k in histories:histories[k].extend(row['deltas'][k])
            state=dict(row['state'],**histories)
        return state,dict(incomplete_final_line=torn,events=len(lines))


def clearance_sensor_joints(raw_qpos, measured_qpos):
    """Match FetchMeasuredPorts' root-stripped joint convention exactly."""
    q=vector(raw_qpos,15).copy();measured=vector(measured_qpos,15)
    q[:3]=0.
    if not np.array_equal(q,measured):
        raise ValueError('Clearance sensor/proprioception mismatch')
    return q


def run_head_observation_probe(executor, auth):
    """Explicit bounded lab diagnostic; no provider, restore or guard bypass."""
    targets=auth.get('head_observation_probe')
    if (os.environ.get('BVI_NO_API')!='1' or auth.get('cold_resume') or auth.get('checkpoint_branch')
            or not isinstance(targets,list) or not 1<=len(targets)<=8
            or any(not isinstance(t,list) or len(t)!=2
                   or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1 for v in t)
                   for t in targets)):
        raise ValueError('Invalid no-API head observation probe')
    results=[];start=executor.steps
    for pan,tilt in targets:
        row=executor.look(pan,tilt);results.append(row)
        if not row.get('accepted') or row.get('reason')!='completed' or row.get('steps_used',0)==0:break
    return dict(accepted=len(results)==len(targets) and all(x.get('accepted') and x.get('reason')=='completed' for x in results),
                results=results,steps_used=executor.steps-start,no_api=True)


def run_clearance_action_probe(executor, auth):
    """Bounded authored sensor/control admission; no model call or retry."""
    actions=auth.get('clearance_action_probe')
    if os.environ.get('BVI_NO_API')!='1' or not isinstance(actions,list) or not 1<=len(actions)<=3:
        raise ValueError('Invalid no-API clearance action probe')
    # Validate the complete sequence before any action. No resume or direct ACK.
    if any(auth.get(k) for k in ('resume_from','branch_from','cold_resume','checkpoint_branch')):
        raise ValueError('Fresh probe required')
    from .robot.eef_tools import pose as tool_pose
    commands=[]
    for a in actions:
        if not isinstance(a,dict):raise ValueError('Invalid clearance probe action')
        tool=a.get('tool')
        if tool=='move_base' and set(a)=={'tool','forward_m'}:
            d=a['forward_m']
            if type(d) not in (float,int) or not math.isfinite(d) or not 0<abs(d)<=.15:
                raise ValueError('Bounded base probe required')
            commands.append(lambda d=d:executor.move_base(0.,d))
        elif tool=='move_to' and set(a)=={'tool','position','quaternion_xyzw'}:
            target=tool_pose(a['position'],a['quaternion_xyzw'])
            if np.linalg.norm(target[:3,3])>1.5:raise ValueError('Bounded arm probe required')
            commands.append(lambda t=target:executor.move_to(t,max_steps=100))
        else:raise ValueError('Unsupported clearance probe action')
    start=executor.steps;results=[]
    for command in commands:
        row=command();results.append(row)
        if not row.get('accepted') or row.get('reason')!='completed' or row.get('steps_used',0)==0:break
    return dict(accepted=len(results)==len(commands) and all(x.get('accepted') and x.get('reason')=='completed' for x in results),
                results=results,steps_used=executor.steps-start,no_api=True)



def run_carried_grasp_probe(executor,state,depth,observe,fk,auth,controller_factory):
    """Authored no-API mechanical acceptance after a hash-bound recorded prefix.

    Prefix replay is explicitly NOT candidate-guard acceptance. No simulator
    object pose/contact/grasp signal is consumed here. Scoring stays in caller.
    """
    from .robot.clearance import odometry_transform,OPTICAL_TO_CAMERA
    from .robot.eef_tools import pose as tool_pose
    from scipy.spatial.transform import Rotation
    spec=auth.get('carried_grasp_probe')
    if (os.environ.get('BVI_NO_API')!='1' or not isinstance(spec,dict)
            or not getattr(executor,'carried_envelope_enabled',False)
            or executor.steps!=0 or any(auth.get(k) for k in ('cold_resume','checkpoint_branch'))):
        raise ValueError('Fresh no-API carried probe required')
    path=Path(spec['source_result'])
    if sha(path)!=spec['source_sha256']:raise ValueError('Grasp probe archive hash changed')
    source=read(path);prefix=spec['prefix_step'];cap=spec['physical_steps_cap']
    if type(prefix)is not int or not 0<prefix<=232 or type(cap)is not int or not prefix<cap<=500:
        raise ValueError('Bounded grasp probe required')
    rows=[r for r in source['frames'] if r['step']<=prefix]
    if [r['step'] for r in rows]!=list(range(1,prefix+1)):raise ValueError('Incomplete recorded prefix')
    actions=[np.asarray(r['action'],float) for r in rows]
    if any(a.shape!=(20,) or not np.isfinite(a).all() or np.max(abs(a))>1 for a in actions):
        raise ValueError('Invalid recorded prefix action')
    if not np.allclose(executor.read_qpos(),source['initial_robot_state']['qpos'],atol=1e-6,rtol=0):
        raise ValueError('Recorded prefix initial encoders mismatch')
    target=copy.deepcopy(next(r for r in source['commands'] if r['turn']==26)['result']['target_record'])
    if target['verdict']!='confirmed':raise ValueError('Archived visual confirmation required')
    goal=tool_pose(spec['position'],spec['quaternion_xyzw'])
    # Validate author-selected bounds before any physical action.
    half=np.asarray(spec['half_extents_base_m'],float)
    if half.shape!=(3,) or not np.isfinite(half).all() or np.any(half<.005) or np.any(half>.15):
        raise ValueError('Bounded sensor estimate required')
    result=dict(no_api=True,prefix_replay_steps=prefix,controller_reinitialized_after_prefix=True,
                prefix_guard_evaluated=False,actions=[],accepted=False)
    for action in actions:
        executor.deadline()
        if not executor._available():raise ValueError('Prefix cannot continue after terminal/unknown')
        executor.pending=dict(sequence=executor.steps,action=action.tolist(),phase='recorded_prefix_replay')
        executor._flush()
        ack=executor.step_once(action)
        if not isinstance(ack,dict) or any(type(ack.get(k))is not bool for k in ('terminated','truncated')):
            raise RuntimeError('Recorded prefix acknowledgement unknown')
        executor.steps+=1;executor.trace.append(dict(step=executor.steps,kind='recorded_prefix_replay',action=action.tolist()))
        executor.pending=None;executor._flush();executor._capture(action,'recorded_prefix_replay')
        if ack['terminated'] or ack['truncated']:
            executor.ended=True;executor.session.latch_official_end(**ack)
            raise ValueError('Official terminal during diagnostic prefix')
    error=float(np.max(np.abs(np.asarray(executor.read_qpos())-np.asarray(rows[-1]['qpos']))))
    result['prefix_encoder_max_error']=error
    if error>1e-4:raise ValueError('Recorded prefix encoder drift')
    executor.session=controller_factory()
    state.identity_target=target;state.identity_sequence=target['id'];state.target_association_unresolved=False
    def estimate_current(turn):
        bundle=observe(turn,[],None);oid,q,cameras=depth.frame
        state.episode_memory.record_target(dict(id=target['id'],status='confirmed',visible=True,
            evidence=state.episode_memory.observation_evidence(oid,'hand')),step=executor.steps)
        B=odometry_transform(dict(zip(('dx_m','dy_m','dyaw_rad'),map(float,executor.odometry()))))
        anchor=np.asarray(target['position_odometry_m']);point_base=B[:3,:3].T@(anchor-B[:3,3])
        camera=cameras['hand'];T=np.asarray(camera['camera_in_base']);opt=OPTICAL_TO_CAMERA.T@(T[:3,:3].T@(point_base-T[:3,3]))
        uv=np.asarray(camera['intrinsics'])@opt;pixel=np.rint(uv[:2]/uv[2]).astype(int).tolist()
        measured=depth.locate('hand',pixel,oid,q)
        if not measured['valid'] or np.linalg.norm(np.asarray(measured['point_base_m'])-point_base)>.015:
            raise ValueError('Diagnostic target sensor tracking mismatch')
        state.target_points=[dict(measured,point_id=1,candidate_id=target['id'],frame=oid,camera='hand')]
        G=fk.links(q)['gripper_link'];center=np.asarray(measured['point_base_m']).copy();center[2]-=spec['center_below_surface_m']
        value=dict(observation_id=oid,camera='hand',point_id=1,center_gripper_m=((center-G[:3,3])@G[:3,:3]).tolist(),
            quaternion_xyzw=Rotation.from_matrix(G[:3,:3].T).as_quat().tolist(),half_extents_m=half.tolist(),
            uncertainty_m=.005,reason='Authored bound from original robot depth and visible tabletop; no scene truth')
        estimated=pick.estimate_carried_envelope(state,executor,dict(carried_envelope=value),depth)
        result['actions'].append(dict(tool='estimate_carried_envelope',result=estimated,pixel=pixel,estimate=value))
        return estimated,bundle
    for attempt in range(2):
        estimated,bundle=estimate_current(attempt)
        if not estimated.get('accepted'):return result
        oid=bundle['observation']['observation_id'];before=executor.steps
        localized=executor.prepare_grasp(goal,reason='No-API sensor-authored grasp acceptance; recorded confirmed carton',replan=attempt>0)
        result['actions'].append(dict(tool='prepare_grasp' if not attempt else 'replan_grasp',result=localized))
        if localized.get('localization_motion_executed'):
            state.pick_entry=dict(candidate_id=target['id'],step=before,observation_id=oid)
        if localized.get('arrived'):break
        if attempt or not localized.get('localization_motion_executed') or executor.is_stopped():return result
    if not localized.get('arrived') or not state.pick_entry:return result
    estimated,_=estimate_current(2)
    if not estimated.get('accepted'):return result
    closed=executor.set_gripper(0);result['actions'].append(dict(tool='set_gripper',result=closed))
    if not closed.get('closure_command_completed') or executor.is_stopped():return result
    lift=fk.tcp(executor.read_qpos());lift[2,3]+=.12
    lifted=executor.move_to(lift,max_steps=100);result['actions'].append(dict(tool='move_to',result=lifted))
    if lifted.get('reason')!='completed' or executor.is_stopped():return result
    rested=executor.return_to_rest();result['actions'].append(dict(tool='return_to_rest',result=rested))
    result['accepted']=bool(rested.get('accepted') and rested.get('reason') in ('completed','official_terminated'))
    return result


def worker_wall_cap(auth, profile, *, replay_mode=False):
    if profile.name in getattr(pick,'S2_SCOPES',{}) and not replay_mode:
        if (auth.get('scope')!=pick.S2_SCOPES[profile.name]
                or type(auth.get('gpu1_process_seconds_cap')) is not int
                or auth['gpu1_process_seconds_cap']!=4800):
            raise ValueError('S2 or independent-candidate wall scope mismatch')
        return auth['gpu1_process_seconds_cap']
    if profile.r1_extended_budget and not replay_mode:
        if (auth.get('scope') not in ((pick.V3_S13_SCOPE,) if profile==pick.SOL_V3_LEAN_S13 else (pick.V3_S12_SCOPE,) if profile==pick.SOL_V3_LEAN_S12 else (pick.V3_S11_SCOPE,) if profile==pick.SOL_V3_LEAN_S11 else (pick.V3_LEAN_SCOPE,) if profile==pick.SOL_V3_LEAN else (pick.R2E_SCOPE,) if profile in (pick.SOL_R2E,pick.SOL_R2E_RECOVERY,pick.SOL_R2E_CANDIDATE_PRIORITY,pick.SOL_R2E_IDENTITY_CONFIRMATION,pick.SOL_R2E_IDENTITY_EVIDENCE,pick.SOL_R2E_I1_STRUCTURED) else (pick.R2D_SCOPE,) if profile==pick.SOL_R2D else (pick.R2C_SCOPE,) if profile==pick.SOL_R2C else (pick.R2B_SCOPE,) if profile==pick.SOL_R2B else (pick.R2A_SCOPE,pick.R2B_SCOPE) if profile==pick.SOL_R2A else (pick.R15_SCOPE,pick.R2A_SCOPE) if profile==pick.SOL_R15_BASELINE else (pick.R15_SCOPE,) if profile==pick.SOL_R15 else (pick.DEV_R1_BUDGET_SCOPE,))
                or type(auth.get('gpu1_process_seconds_cap')) is not int
                or auth['gpu1_process_seconds_cap'] != 4800):
            raise ValueError('R1 1600 worker wall scope mismatch')
        return auth['gpu1_process_seconds_cap']
    return (auth['gpu1_process_seconds_cap'] if auth.get('scope') == pick.WEEKLY_SCOPE else
            min(1800, auth.get('worker_seconds_cap', 600) if replay_mode else auth.get('gpu1_process_seconds_cap', 1800)))


def make_memory_full_b_executor(full, runtime, preflight, base, *, output, case_id,
        step, qpos, qvel, info, stopped, deadline, fk, recording, timed=direct_timed_call,
        recording_state_factory=MemoryRecordingState, grasp_alignment=False, weekly_profile=None, rest_depth_checker=None, r1_budget_profile=None):
    if r1_budget_profile is not None and (r1_budget_profile not in (pick.SOL_DEV_R1_1600,pick.SOL_DEV_R1_PARALLEL)+pick.PAIRED_PROFILES
            or weekly_profile is not None or not grasp_alignment):
        raise ValueError('Explicit R1 1600 aligned profile required')
    r15_recording=r1_budget_profile is not None and r1_budget_profile.r15_incremental
    from .robot.executor import FullBOnlineExecutor, AlignedFullBOnlineExecutor
    from video_template import full_action_labels
    native=preflight.native_ports(base,Path(output)/'full-b-frames',case_id,'memory',info(),action_size=20)['recorder']
    native.action_labels=full_action_labels(base.uenv.agent.controller.action_mapping)
    if r15_recording:
        native.incremental_manifest=IncrementalFrameManifest(native,current_step_only=True)
    elif weekly_profile is not None and weekly_profile.incremental_recording:
        native.incremental_manifest=IncrementalFrameManifest(native)
    guard=recording_state_factory(base,preflight,output,timed)
    if r15_recording or (weekly_profile is not None and weekly_profile.efficient_io):
        guard.incremental_audit=True
    recorder=MemoryRecordingPort(native,recording,info,timed,state_guard=guard,
        **({'contact_diagnostics':AlignedContactDiagnostics(base,output)} if grasp_alignment else {}))
    session=runtime.make_controller_session(full,qpos(),sim_freq=base.uenv.sim_freq,control_freq=base.uenv.control_freq)
    ports=runtime.FetchMeasuredPorts(qpos,qvel,lambda:{k:bool(scalar(info()[k])) for k in ('ee_rest','robot_rest','is_static')},fk)
    def persist(value):
        path=Path(output)/'full-b-progress.json';temporary=path.with_suffix('.tmp')
        timed('progress.persist',save,temporary,value);temporary.replace(path)
    if r15_recording or (weekly_profile is not None and weekly_profile.efficient_io):
        persist=IncrementalProgress(Path(output),timed,**({'fields':('trace','controller_diagnostics','alignment_samples')} if r15_recording else {}))
    if weekly_profile is not None and (weekly_profile not in pick.WEEKLY_PROFILES or not grasp_alignment):
        raise ValueError("Explicit weekly aligned profile required")
    executor_class=AlignedFullBOnlineExecutor if grasp_alignment else FullBOnlineExecutor
    ex = executor_class(session,ports,base.uenv.agent.controller.action_mapping,
        step_once=step,read_qpos=qpos,is_stopped=stopped,recorder=recorder,persist=persist,deadline=deadline,timed=timed,
        **(dict(rest_depth_checker=rest_depth_checker) if rest_depth_checker is not None else {}),
        **(dict(weekly_budget=True,max_steps=weekly_profile.step_cap) if weekly_profile is not None else {}),
        **(dict(weekly_budget=True,max_steps=r1_budget_profile.step_cap) if r1_budget_profile is not None else {}),
        **({k:getattr(r1_budget_profile,k) for k in ('r15_protocol','r15_completion','r15_pregrasp','r15_carry')} if r1_budget_profile in pick.PAIRED_PROFILES else {}),
        **(dict(smooth_localization=True) if weekly_profile is not None and weekly_profile.smooth_grasp_localization else {}),
        **(dict(grasp_direction_feedback=True) if weekly_profile == pick.SOL_HOT_RECOVERY else {}),
        **(dict(rest_progress_feedback=True) if weekly_profile is not None and weekly_profile.rest_progress_feedback else {}),
        **(dict(motion_guard=True) if weekly_profile is not None and weekly_profile.motion_guard else {}),
        **(dict(self_collision_guard=True) if (weekly_profile is not None and weekly_profile.self_collision_guard) or r1_budget_profile==pick.SOL_V3_LEAN_S11 else {}),
        **(dict(look_only_self_guard=True) if any(p in pick.V3_SYSTEM_PROFILES and p!=pick.SOL_V3_LEAN_S11
                for p in (weekly_profile,r1_budget_profile)) else {}),
        **(dict(base_tracking_guard=True) if weekly_profile is not None and weekly_profile.base_tracking_guard else {}),
        **(dict(local_clearance_guard=True,defer_clearance_binding=True,
                clearance_unknown_advisory=weekly_profile.clearance_unknown_advisory,
                carried_envelope_enabled=weekly_profile.carried_envelope)
           if weekly_profile is not None and weekly_profile.local_clearance else {}),
        **(dict(fast_base_approach=True) if weekly_profile is not None and weekly_profile.fast_base_approach else {}))
    if weekly_profile in pick.V3_SYSTEM_PROFILES or r1_budget_profile in pick.V3_SYSTEM_PROFILES:ex.s11_system_fixes=True
    if weekly_profile is not None and weekly_profile.active_confirmation:ex.configure_recovery(weekly_profile.primary_steps)
    if r1_budget_profile is not None and r1_budget_profile.arm_base_explicit:ex.arm_base_explicit=True
    if r1_budget_profile is not None and r1_budget_profile.arm_tools_lock_base:ex.arm_tools_lock_base=True
    if r1_budget_profile is not None:
        for flag in ('compound_feedback','r2d_base_clearance','base_tracking_exit','arm_base_push_exit','g1_dispatch_semantics','rest_progress_feedback','rest_joint_target')+pick.R2E_FLAGS:
            if getattr(r1_budget_profile,flag):setattr(ex,flag,True)
        if getattr(r1_budget_profile,'rest_base_push_report_only',False):ex.rest_base_push_report_only=True
    if r1_budget_profile is not None and r1_budget_profile.motion_progress_only:ex.motion_progress_only=True
    if r1_budget_profile is not None and r1_budget_profile.r2b_grip_monitor:ex.r2b_grip_monitor=True
    if r1_budget_profile is not None and r1_budget_profile.r2b_clearance:ex.r2b_clearance=True
    return ex


def make_transport_full_b_executor(*args, arm, **kwargs):
    """Use the unchanged factory/controller; bind recording labels before capture."""
    if arm not in ('A','B') or kwargs.get('case_id')!=pick.TransportABBatch.case_id:
        raise ValueError('transport executor case/arm binding')
    executor=make_memory_full_b_executor(*args,recording_state_factory=TransportRecordingState,**kwargs)
    native=executor.recorder.native
    if native.frames:raise ValueError('transport recording must be bound before initial capture')
    native.condition=arm
    return executor


def live_run(args):
    started = time.monotonic()
    auth = json.loads(args.authorization.read_text())
    validate(auth, args)
    validate_candidate_sources(auth, args.code)
    is_pick20 = pick.PROFILES[args.profile] in (pick.SOL_PICK20,pick.SOL_PICK20_ALIGNED)+pick.DEV_PROFILES+pick.WEEKLY_PROFILES

    if is_pick20:
        if auth.get('pick20_continuation') and args.case_id == 'baseline-000':
            raise ValueError('completed baseline-000 must not rerun')
        job=pick.r15_job_id(args.case_id,pick.PROFILES[args.profile],auth.get('scope'))
        if args.output.resolve() != (Path(auth['output_root'])/job).resolve():
            raise ValueError('pick20 exact output identity required')
        fd = getattr(args, 'initialization_lock_fd', None)
        lock_path = args.lab_root/'runs/arm-capability.lock'
        if type(fd) is not int or (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (lock_path.stat().st_dev, lock_path.stat().st_ino):
            raise ValueError('pick20 requires inherited batch scheduler lock')
        intent = read(Path(auth['output_root'])/(job+'.launch-intent.json'))
        if intent != dict(case_id=args.case_id, authorization_sha256=sha(args.authorization)):
            raise ValueError('pick20 launch identity mismatch')
        _initialization_json_exclusive(Path(auth['output_root'])/(job+'.worker-entered.json'),
            dict(pid=os.getpid(), authorization_sha256=sha(args.authorization)))
    roster = json.loads(args.roster.read_text())
    row = next((r for r in roster['dev'] if r['case_id'] == args.case_id))
    reference_only = args.mode == 'ab-reference'
    full_b_preflight = args.mode == 'full-b-preflight'
    aligned_preflight = args.mode == 'aligned-preflight'
    if aligned_preflight:validate_aligned_preflight_inputs(auth,args)
    if reference_only:
        fd = getattr(args, 'initialization_lock_fd', None)
        if type(fd) is not int or (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != ((args.lab_root/'runs/arm-capability.lock').stat().st_dev, (args.lab_root/'runs/arm-capability.lock').stat().st_ino):
            raise ValueError('reference worker requires inherited scheduler lock')
        intent = read(Path(auth['output_root'])/(args.case_id+'.launch-intent.json'))
        if intent != dict(authorization_sha256=sha(args.authorization), case_id=args.case_id, index=auth['case_ids'].index(args.case_id)):
            raise ValueError('reference worker launch identity mismatch')
        _initialization_json_exclusive(Path(auth['output_root'])/(args.case_id+'.worker-entered.json'), dict(pid=os.getpid(), authorization_sha256=sha(args.authorization)))
    if args.mode=='init-only':
        completed=[] if args.initialization_previous is None else read(args.initialization_previous)
        if pick.validate_init_only_sequence(auth,completed,authorization_path=args.authorization)!=args.initialization_observation:
            raise ValueError('initialization-only observation order')
        fd=getattr(args,'initialization_lock_fd',None)
        if type(fd) is not int or (os.fstat(fd).st_dev,os.fstat(fd).st_ino)!=((args.lab_root/'runs/arm-capability.lock').stat().st_dev,(args.lab_root/'runs/arm-capability.lock').stat().st_ino):
            raise ValueError('initialization-only inherited scheduler lock required')
        intent=read(Path(auth['output_root'])/(args.initialization_observation+'.launch-intent.json'))
        if (intent.get('source_manifest_sha256')!=auth['source_manifest_sha256'] or intent.get('authorization_sha256')!=sha(args.authorization)
                or intent.get('observation')!=args.initialization_observation):
            raise ValueError('initialization-only dedicated launch intent missing/mismatched')
        claim=Path(auth['output_root'])/(args.initialization_observation+'.worker-entered.json')
        _initialization_json_exclusive(claim,dict(source_manifest_sha256=auth['source_manifest_sha256'],batch_id=auth['batch_id'],observation=args.initialization_observation,pid=os.getpid(),pgid=os.getpgrp()))
    args.output.mkdir(parents=True, exist_ok=False)
    # Claim this attempt before timer setup can create any output paths.
    timed = MemoryPhaseTimer(args.output) if is_pick20 or full_b_preflight or auth.get('scope')==pick.FROZEN['memory_ab']['scope'] else direct_timed_call
    report = {'case_id': args.case_id, 'position': int(args.case_id.rsplit('-', 1)[1]), 'condition': args.condition, 'status': 'running', 'information_condition': pick.PROFILES[args.profile].name, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'strict_pick_success': False, 'frames': [], 'commands': [], 'simulator_actions': 0, 'api_requests': 0}
    r15_journal=(IncrementalProgress(args.output,timed,stem='result',fields=('frames','executor_trace','ik_records'))
                 if pick.PROFILES[args.profile].r15_incremental else None)
    def write_report():
        if pick.PROFILES[args.profile] in pick.V3_PROFILES:
            report['outcome_class'] = v3_outcome_class(report)
        if pick.PROFILES[args.profile].search_recessed_view:
            report['phase_usage_eval_only']=search_phase_usage(report)
        if r15_journal is None:save(args.output/'result.json',report)
        else:r15_journal(r15_journal.snapshot(report))
    init_only = args.mode == 'init-only'
    replay_mode = args.mode == 'l1-replay'
    initialization_observability = 'initialization_observability' in auth and auth.get('stage') in ('L1', 'L4', 'INIT')
    initialization_stage = 'before_construction_metadata_only'
    if 'initialization_observability' in auth:
        pick.validate_initialization_observability(auth)
        report['initialization_observability'] = copy.deepcopy(auth['initialization_observability'])
        report['initialization_capture_active'] = initialization_observability
        if initialization_observability:
            report['initialization_boundaries_eval_only'] = []
    actual_anchor_mode = 'replay_initial_anchor' in auth and (init_only or replay_mode or (auth.get('scope')==pick.R33_STAGED_SCOPE and auth.get('stage')=='L4'))
    if 'replay_initial_anchor' in auth:
        report['replay_initial_anchor']=auth['replay_initial_anchor']
    if actual_anchor_mode:
        report['initial_anchor_eval_only']=pick.replay_initial_anchor(auth,args.case_id)['binding']
    if pick.PROFILES[args.profile].model_target_boxes:
        retry = auth.get('infrastructure_retry')
        report['attempt_index'] = 1 if retry and retry['case_id'] == args.case_id else 0
        report['batch_id'] = auth['batch_id']
    cap = worker_wall_cap(auth, pick.PROFILES[args.profile], replay_mode=replay_mode)
    env = None
    recording = None
    ex = None
    full_b = None
    full_b_base = None

    prior_steps = int(auth.get('prior_scope_physical_steps', 0))
    if pick.PROFILES[args.profile].model_target_boxes and not replay_mode and not init_only:
        prior_steps += batch_physical_steps(Path(auth['output_root']), auth['batch_id'])
    elif auth.get('scope') == 'baseline20-supervised-20261006':
        prior_steps += sum(json.loads(p.read_text()).get('simulator_actions', 0)
                           for p in args.output.parent.parent.glob('baseline-*/V-mobile/result.json'))

    def deadline():
        if (args.output / 'STOP').exists() or (pick.PROFILES[args.profile].parallel_case_workers and (Path(auth['output_root'])/'STOP').exists()):
            raise SupervisionStopped('operator stop requested')
        if pick.r15_case_isolation(auth) and ((args.output/'CASE_WALL_LIMIT').exists() or time.monotonic()-started>=cap-10):
            raise pick.CaseLimitReached('wall_time_limit')
        if time.monotonic() - started >= cap - 10:
            raise TimeoutError('authorized worker wall limit')
    try:
        if replay_mode:
            record_replay_attempt_identity(args, auth, report)
            if 'replay_identity_status' in report:
                write_report()
        configure_pick_runtime(auth, args.lab_root, pick.PROFILES[args.profile])
        import numpy as np
        import torch
        from dacite import from_dict
        from omegaconf import OmegaConf
        from mshab.envs.make import EnvConfig, make_env
        if is_pick20:
            full_b, make_env = pick20_factory(make_env, args.code, auth, pick.PROFILES[args.profile])
        if auth.get('scope') in (pick.FROZEN['memory_ab']['scope'],FULL_B_PREFLIGHT_SCOPE):
            full_b=load_memory_full_b(args.code,auth)
            make_env=full_b[1].bind_fetch_factory(make_env,full_b[1].agent_class())
        from types import SimpleNamespace
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
        import subprocess
        runtime = args.lab_root / 'src/official-mshab-runtime/mshab'
        if subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip() != 'e9ff3d23496d38e4431c8d913e147ffa007f7f72':
            raise ValueError('runtime pin')
        for name, digest in auth['asset_sha256'].items():
            if sha(args.lab_root / name) != digest:
                raise ValueError('frozen asset changed: ' + name)
        data = args.lab_root / 'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        split = official_dataset_split(row)
        plan_file = "all.json" if "dataset_split" in row else row["object_category"] + ".json"
        plan_path = data / f"task_plans/tidy_house/pick/{split}/{plan_file}"
        source = json.loads(plan_path.read_text())
        plans = [p for p in source['plans'] if p['subtasks'][0]['composite_subtask_uids'][0] == row['uid']]
        if len(plans) != 1:
            raise ValueError('spawn UID ambiguity')
        save(args.output / 'single-plan.json', {'dataset': source['dataset'], 'plans': plans})
        spawn_path = data / f'spawn_data/tidy_house/pick/{split}/spawn_data.pt'
        if row.get('spawn_source') == 'official_navigate':
            if pick.PROFILES[args.profile].stall_guard_mode != 'reference_progress' and auth.get('scope') not in (pick.PICK20_SCOPE, pick.PICK20_BATCH_SCOPE, *pick.DEV_R1_SCOPES, *pick.V3_SCOPES, pick.WEEKLY_SCOPE, ALIGNED_PREFLIGHT_SCOPE, pick.FROZEN['memory_ab']['scope'], AB_REFERENCE_SCOPE,FULL_B_PREFLIGHT_SCOPE):

                raise ValueError('Navigate start requires explicit baseline preparation profile')
            nav_path = data / f'task_plans/tidy_house/navigate/{split}/all.json'
            nav_source = json.loads(nav_path.read_text())
            nav_plans = [p for p in nav_source['plans']
                         if p['subtasks'][0]['composite_subtask_uids'][0] == row['navigate_uid']]
            if len(nav_plans) != 1:
                raise ValueError('Navigate UID ambiguity')
            nav_spawn_path = data / f'spawn_data/tidy_house/navigate/{split}/spawn_data.pt'
            nav_spawns = torch.load(nav_spawn_path, map_location='cpu', weights_only=False)
            adapted = navigate_pick_spawn(nav_plans[0], plans[0], nav_spawns[row['navigate_uid']])
            spawn_path = args.output / 'official-navigate-pick-spawn.pt'
            torch.save(adapted, spawn_path)
            save(args.output / 'navigate-reset-provenance.eval-only.json', dict(
                navigate_uid=row['navigate_uid'], pick_uid=row['uid'], spawn_index=row['spawn_index'],
                plan_sha256=sha(nav_path), original_spawn_sha256=sha(nav_spawn_path),
                adapted_spawn_sha256=sha(spawn_path), robot_fields_unchanged=True,
                object_override=False, actual_reset_equivalence='requires frozen binding comparison'))
        elif row.get('spawn_source', 'official_pick') != 'official_pick':
            raise ValueError('unknown official start source')
        checkpoint = Path(auth['checkpoint_root']) / 'rl/tidy_house/pick' / row['object_category']
        raw = OmegaConf.to_container(OmegaConf.load(checkpoint / 'config.yml'), resolve=True)
        cfg = dict(raw['eval_env'])
        horizon = pick.PROFILES[args.profile].step_cap
        cfg.update(num_envs=1, max_episode_steps=horizon, continuous_task=True, record_video=False, task_plan_fp=str(args.output / 'single-plan.json'), spawn_data_fp=str(spawn_path))
        kwargs = dict(cfg.get('env_kwargs', {}))
        task_cfgs = dict(kwargs.get('task_cfgs', {}))
        task_cfgs['pick'] = {'horizon': horizon + 1}
        kwargs.update(require_build_configs_repeated_equally_across_envs=False, target_randomization=False, add_event_tracker_info=True, task_cfgs=task_cfgs, human_render_camera_configs={'width': 512, 'height': 512})
        kwargs['sensor_configs'] = {'width': 640, 'height': 640}
        cfg.update(stationary_base=False, stationary_torso=False, stationary_head=False, obs_mode='rgbd')
        cfg['env_kwargs'] = kwargs
        if cfg['env_id'] != 'PickSubtaskTrain-v0':
            raise ValueError('official Pick checkpoint environment mismatch')
        if init_only:
            report.update(scope=pick.R33_INIT_ONLY_SCOPE,batch_id=auth['batch_id'],
                observation=args.initialization_observation,source_manifest_sha256=auth['source_manifest_sha256'],authorization_sha256=sha(args.authorization))
            inputs=dict(plan=plan_path,spawn=data/f'spawn_data/tidy_house/pick/{split}/spawn_data.pt',checkpoint_config=checkpoint/'config.yml')
            if row.get('spawn_source')=='official_navigate':
                inputs.update(navigate_plan=nav_path,navigate_spawn=nav_spawn_path)
            env_config=from_dict(EnvConfig,cfg)
            options={'reconfigure':True,'spawn_selection_idxs':[row['spawn_index']]}
            return run_initialization_only(args.output,report,make_env=make_env,env_config=env_config,
                seed=row['seed'],reset_options=options,snapshot_writer=torch.save,
                metadata=lambda: initialization_observability_metadata(args,auth,row,cfg,env_config,
                    seed=row['seed'],reset_options=options,files=inputs,
                    modules={name:sys.modules.get(name) for name in INITIALIZATION_SDK_MODULES}),
                strict_check=lambda environment,info: check_initialization_only(environment,info,args,auth,row,horizon,report,torch.save))
        if initialization_observability:
            env_config = from_dict(EnvConfig, cfg)
            seed = row['seed']
            reset_options = {'reconfigure': True, 'spawn_selection_idxs': [row['spawn_index']]}
            inputs = dict(plan=plan_path, spawn=data / f'spawn_data/tidy_house/pick/{split}/spawn_data.pt',
                          checkpoint_config=checkpoint/'config.yml')
            if row.get('spawn_source') == 'official_navigate':
                inputs.update(navigate_plan=nav_path, navigate_spawn=nav_spawn_path)
            report['initialization_boundaries_eval_only'].append(capture_initialization_boundary(args.output, initialization_stage, snapshot_writer=torch.save,
                metadata=lambda: initialization_observability_metadata(args, auth, row, cfg, env_config,
                    seed=seed, reset_options=reset_options, files=inputs,
                    modules={name: sys.modules.get(name) for name in INITIALIZATION_SDK_MODULES})))
            initialization_stage = 'construction'
            _initialization_json_exclusive(args.output/'initialization-construction.entered.eval-only.json',
                dict(stage=initialization_stage, evaluation_only=True))
            env = make_env(env_config)
            env.env.auto_reset = False
            initialization_stage = pick.INITIALIZATION_OBSERVABILITY_STAGES[1]
            report['initialization_boundaries_eval_only'].append(capture_initialization_boundary(args.output, initialization_stage,
                state_getter=env.unwrapped.get_state_dict, snapshot_writer=torch.save))
            initialization_stage = 'explicit_reset'
            _initialization_json_exclusive(args.output/'initialization-explicit-reset.entered.eval-only.json',
                dict(stage=initialization_stage, evaluation_only=True))
            obs, info = env.reset(seed=seed, options=reset_options)
            initialization_stage = pick.INITIALIZATION_OBSERVABILITY_STAGES[2]
            report['initialization_boundaries_eval_only'].append(capture_initialization_boundary(args.output, initialization_stage,
                state_getter=env.unwrapped.get_state_dict, info=info, snapshot_writer=torch.save))
            initialization_stage = 'post_initialization_checks_and_execution'
        else:
            env = timed('initialization.construct',make_env,from_dict(EnvConfig, cfg))
            env.env.auto_reset = False
            seed = row['seed'] if is_pick20 or pick.PROFILES[args.profile].verification else 20260930 + int(args.case_id.rsplit('-', 1)[1])
            obs, info = timed('initialization.reset',env.reset,seed=seed, options={'reconfigure': True, 'spawn_selection_idxs': [row['spawn_index']]})
        u = env.unwrapped
        target = u.subtask_objs[0]
        if str(u.task_plan[0].composite_subtask_uids[0]) != row['uid'] or u.spawn_selection_idxs[0] != row['spawn_index']:
            raise ValueError('runtime spawn binding')
        if int(scalar(info['subtasks_steps_left'])) != horizon:
            raise ValueError(f'reset must leave exactly {horizon} steps; no counter repair')
        if float(u.pick_cfg.robot_cumulative_force_limit) != 5000 or float(u.pick_cfg.ee_rest_thresh) != 0.05:
            raise ValueError('official criteria changed')
        validate_official_controller(u.agent.controller)
        validate_base_runtime(u.agent.controller, u.control_freq)
        if pick.PROFILES[args.profile].weekly_budget and (float(u.control_freq)!=20 or float(u.sim_freq)!=100):
            raise ValueError('weekly accounting requires unchanged 20 Hz control and 100 Hz physics')
        if full_b is not None:
            full_b_base=timed('initialization.activation',activate_memory_full_b,env,full_b[1],full_b[2],args.output,timed=timed)
        if reference_only:
            snapshot = capture_ab_reference(args.output, args.case_id, row, seed, u.agent,
                u.get_state_dict(), dict(runtime_commit='e9ff3d23496d38e4431c8d913e147ffa007f7f72', asset_sha256=auth['asset_sha256']), info)
            report.update(status='reference_captured', source_fingerprint=auth['source_fingerprint'],
                seed=seed, configured_horizon=horizon, internal_pick_horizon=int(u.pick_cfg.horizon),
                initial_steps_left=int(scalar(info['subtasks_steps_left'])), explicit_resets=1,
                sdk_internal_physics_steps='not counted', reference_sha256=sha(args.output/'initial-reference.eval-only.json'))
            return report
        fk = RobotFK(args.code / 'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf', [j.name for j in u.agent.robot.active_joints])

        def qpos():
            return u.agent.robot.qpos[0].detach().cpu().numpy().astype(float)

        def qvel():
            return u.agent.robot.qvel[0].detach().cpu().numpy().astype(float)

        def matrix(pose):
            return np.asarray(pose[0].sp.to_transformation_matrix())

        def world_base():
            return matrix(u.agent.base_link.pose)
        if np.max(abs(fk.tcp(qpos()) - np.linalg.inv(world_base()) @ matrix(u.agent.tcp.pose))) > 1e-05:
            raise ValueError('FK identity')
        component, shapes = _component_and_shapes(target)
        meshes = get_component_meshes(component)
        if not meshes or len(meshes) != len(shapes):
            raise ValueError('target collision coverage')
        vertices = np.concatenate([m.vertices for m in meshes])
        lo = vertices.min(0)
        hi = vertices.max(0)
        base_from_object = np.linalg.inv(world_base()) @ matrix(target.pose)
        surface_triangles = np.concatenate([m.triangles for m in meshes])
        corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        box_base = (corners @ base_from_object.T)[:, :3]
        initial_z = float(matrix(target.pose)[2, 3])
        stopped = False
        report.update(control_dt=1 / float(u.control_freq), initial_robot_state={'qpos': qpos().tolist(), 'tcp_base': fk.tcp(qpos()).tolist(), 'base_world_eval_only': world_base().tolist(), 'finger_distance_m': float(qpos()[-2:].sum())}, initial_binding={'uid': row['uid'], 'spawn_index': row['spawn_index'], 'seed': seed, 'qpos': qpos().tolist(), 'base_world': world_base().tolist(), 'target_world': matrix(target.pose).tolist()}, configured_horizon=horizon, internal_pick_horizon=int(u.pick_cfg.horizon), initial_steps_left=int(scalar(info['subtasks_steps_left'])))
        save(args.output / 'initial-binding.json', report['initial_binding'])
        if is_pick20:
            reference = args.code/auth['initial_references'][args.case_id]
            if sha(reference) != auth['source_sha256'].get(auth['initial_references'][args.case_id]):
                raise ValueError('pick20 initial reference not frozen')
            comparison = pick20_reset_check(read(reference), report['initial_binding'],
                array(u.agent.robot.pose.raw_pose)[0], u.get_state_dict())
            save(args.output/'initial-check.eval-only.json', comparison)
            admission=pick20_initial_admission(auth,args.case_id,comparison)
            report['initial_state_admission_eval_only']=admission
            save(args.output/'initial-admission.eval-only.json',admission)
            if auth.get('initial_state_exception') and args.case_id=='baseline-017':
                # Save every actual object pose for later comparison; never inject
                # archived poses into the environment or expose them to the model.
                state=u.get_state_dict()
                snapshot=dict(root_pose=array(u.agent.robot.pose.raw_pose)[0].tolist(),
                    qpos=qpos().tolist(),objects={k:array(v)[0][:7].tolist() for k,v in state['actors'].items()})
                save(args.output/'natural-reset.eval-only.json',snapshot)
            if not admission['allowed']: raise ValueError('pick20 historical initial state mismatch')
        # AB uses its complete reference receipt, not the historical r3 binding.
        if auth.get('scope') not in (pick.PICK20_SCOPE, pick.PICK20_BATCH_SCOPE, *pick.DEV_R1_SCOPES, *pick.V3_SCOPES, pick.WEEKLY_SCOPE, ALIGNED_PREFLIGHT_SCOPE, pick.FROZEN['memory_ab']['scope'],FULL_B_PREFLIGHT_SCOPE):

            reference = Path(row['initial_binding_path']) if pick.PROFILES[args.profile].verification else Path(auth['reference_result_root']) / args.case_id / 'V/initial-binding.json'
            if not reference.is_file():
                raise ValueError('missing frozen r3 initial state')
            if (auth.get('scope') == 'baseline20-supervised-20261006' or replay_mode
                    or (auth.get('scope') in (pick.R3_SCOPE, pick.R31_STAGED_SCOPE, pick.R33_STAGED_SCOPE) and auth.get('stage') == 'L4')):
                archived = json.loads(reference.read_text())
                if pick.PROFILES[args.profile] in (pick.C2_R32, pick.C2_R32_FAR, pick.C2_R33, pick.C2_R33_FAR):
                    if actual_anchor_mode:
                        record_final_anchor_reset(args.output, auth, args.case_id, report, archived,
                            qpos(), array(u.agent.robot.pose.raw_pose)[0], u.get_state_dict(), snapshot_writer=torch.save)
                    else:
                        record_reset_comparison(args.output, archived, qpos(), array(u.agent.robot.pose.raw_pose)[0],
                            u.get_state_dict(), snapshot_writer=torch.save)
                else:
                    checks = dict(qpos=np.allclose(archived['qpos'],qpos(),rtol=0,atol=1e-6),
                        root_pose=np.allclose(archived['root_pose'],array(u.agent.robot.pose.raw_pose)[0],rtol=0,atol=1e-6))
                    actors = u.get_state_dict()['actors']
                    for name,pose in archived['objects'].items():
                        checks[name] = np.allclose(pose,array(actors['env-0_'+name])[0][:7],rtol=0,atol=1e-6)
                    save(args.output/'baseline-reset-check.eval-only.json',{k:bool(v) for k,v in checks.items()})
                    if not all(checks.values()):
                        raise ValueError('baseline differs from frozen initial state')
            else:
                bind_initial(reference, report['initial_binding'])
        recording = AttemptRecording(args.output / 'recording-frames', float(u.control_freq))
        if full_b is None:
            recording.capture_environment(u)
        if auth.get('detailed_trajectory'):
            capture_detailed_step(u, args.output, 0, report['control_dt'])
        initial_info = jsonable(info)
        if replay_mode:
            report['initial_info_eval_only'] = initial_info
        accepted_checks = []
        report['initial_evaluation'] = {'step': 0, 'strict_success': strict_pick(initial_info), 'is_grasped': bool(scalar(info['is_grasped'])), 'object_z_m': initial_z, 'tcp_box_distance_m': tcp_box_distance((np.linalg.inv(matrix(target.pose)) @ matrix(u.agent.tcp.pose))[:3, 3], lo, hi)}

        def step(action):
            nonlocal info, obs, stopped
            deadline()
            if replay_mode:
                if report['simulator_actions'] >= replay_arm_step_cap(args,auth):
                    raise SupervisionStopped('registered L1 arm physical step cap')
            if auth.get('scope') == 'baseline20-supervised-20261006' and prior_steps + report['simulator_actions'] >= auth['physical_steps_cap']:
                raise SupervisionStopped('cumulative authorized physical step cap')
            if full_b_preflight and report['simulator_actions']>=160:
                raise SupervisionStopped('preflight160-step cap')
            if aligned_preflight and report['simulator_actions']>=600:
                raise SupervisionStopped('aligned preflight600-step cap')
            if auth.get('carried_grasp_probe') and report['simulator_actions']>=auth['carried_grasp_probe']['physical_steps_cap']:
                raise SupervisionStopped('registered no-API grasp probe step cap')
            if stopped:
                raise RuntimeError('step after official terminal')
            old_left = int(scalar(info['subtasks_steps_left']))
            old_force = float(scalar(info['robot_cumulative_force']))
            obs, _, terminated, truncated, info = timed('physics.env_step',env.step,torch.as_tensor(action, device=u.device, dtype=torch.float32).reshape(1, -1))
            report['simulator_actions'] += 1
            values = jsonable(info)
            success = strict_pick(values)
            if int(scalar(info['subtasks_steps_left'])) != old_left - 1 or float(scalar(info['robot_cumulative_force'])) + 1e-05 < old_force:
                raise ValueError('clock/force discontinuity')
            tcp_object = np.linalg.inv(matrix(target.pose)) @ matrix(u.agent.tcp.pose)
            frame = {'step': report['simulator_actions'], 'qpos': qpos().tolist(), 'qvel': qvel().tolist(), 'base_world_eval_only': world_base().tolist(), 'tcp_base': fk.tcp(qpos()).tolist(), 'finger_distance_m': float(qpos()[-2:].sum()), 'action': list(map(float, action)), 'official_info_eval_only': values, 'strict_success': success, 'is_grasped': bool(scalar(info['is_grasped'])), 'object_z_m': float(matrix(target.pose)[2, 3]), 'tcp_box_distance_m': tcp_box_distance(tcp_object[:3, 3], lo, hi)}
            if pick.PROFILES[args.profile].verification:
                frame['grasped_objects_eval_only'] = [name for name, actor in u.scene_builder.movable_objects.items() if bool(scalar(u.agent.is_grasping(actor, max_angle=30)))]
            report['frames'].append(frame)
            if pick.PROFILES[args.profile].r2d_extended_calls and success:
                report.setdefault('strict_success_request_count',report['api_requests'])
                report['success_within_50_total_requests']=report['strict_success_request_count']<=50
            report['strict_pick_success'] |= success
            stopped = success or bool(scalar(info.get('fail', False))) or bool(scalar(terminated)) or bool(scalar(truncated)) or (report['simulator_actions'] >= horizon)
            report['official_final_info'] = values
            write_report()
            if full_b is None:
                recording.capture_environment(u)
            if auth.get('detailed_trajectory'):
                capture_detailed_step(u, args.output, report['simulator_actions'], report['control_dt'])
            return {'terminated': stopped, 'truncated': bool(scalar(truncated))}
        model = json.loads((args.code / 'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/response-model.json').read_text())['model']
        profile = pick.PROFILES[args.profile]
        report['execution_profile'] = args.profile
        executor_class = ArmCoordinatedV2
        executor_options = {}
        effective_flags = replay_executor_config(auth, profile, args.replay_arm) if replay_mode else None
        for flag in ('schedule_precheck', 'robot_diagnostics', 'base_heading_hold', 'continuous_base_speed'):
            if (effective_flags[flag] if replay_mode else getattr(profile, flag)):
                executor_options[flag] = True
        for flag in ('base_progress_monitor', 'base_response_monitor', 'base_response_debt_monitor', 'base_response_corroboration', 'base_progress_abort', 'execution_lifecycle'):
            if (effective_flags.get(flag,False) if replay_mode else getattr(profile,flag)):
                executor_options[flag] = True
        if (effective_flags['base_motion_abort'] if replay_mode else profile.base_motion_abort):
            from .robot.motion import BaseAbortMixin
            class AbortExecutor(BaseAbortMixin, ArmCoordinatedV2):
                pass
            executor_class = AbortExecutor
            executor_options['zero_base_velocity'] = lambda: u.agent.controller.controllers['base'].set_action(torch.zeros((1, 2), device=u.device))
        if executor_options.get('base_progress_abort') and 'zero_base_velocity' not in executor_options:
            executor_options['zero_base_velocity'] = lambda: u.agent.controller.controllers['base'].set_action(torch.zeros((1, 2), device=u.device))
        report['close_uses_endpoint_tolerance'] = profile.close_uses_endpoint_tolerance
        rest_checker=None
        if (auth.get('checkpoint_branch',{}).get('migration_contract')=='rest-brake-branch-v1' or auth.get('cold_resume',{}).get('source_migration',{}).get('contract')=='rest-brake-budget-resume-v1'):
            from .memory import EpisodeMemory, VERSION
            from .robot.clearance import SensorClearance
            robot_root=args.lab_root/'src/official-mshab-runtime/ManiSkill/mani_skill/assets/robots/fetch'
            geometry=episode_geometry_pin(args.code,installed_robot_root=robot_root,refine_base=True,refine_fingers=True)
            rest_memory=EpisodeMemory(auth['attempt_ids'][args.case_id],'relative-odom',dict(
                sensor_calibration_sha256=sha(robot_root/'fetch.urdf'),
                robot_geometry_sha256=EPISODE_GEOMETRY_SHA256,memory_version=VERSION))
            rest_checker=SensorClearance(fk,rest_memory,geometry=geometry)
        if full_b is not None:
            ex=make_memory_full_b_executor(*full_b,full_b_base,output=args.output,case_id=args.case_id,
                step=step,qpos=qpos,qvel=qvel,info=lambda:info,stopped=lambda:stopped,deadline=deadline,fk=fk,recording=recording,timed=timed,
                rest_depth_checker=rest_checker,
                **({'grasp_alignment':True} if profile.grasp_alignment else {}),
                **({'weekly_profile':profile} if profile.weekly_budget else {}),
                **({'r1_budget_profile':profile} if profile.r1_extended_budget else {}))
            if not (auth.get('cold_resume') or auth.get('checkpoint_branch')):
                ex._capture(None,'tool_loop_initial');ex.initial_recorded=True
            report['controller_version']='full-B-r4-plus-task-rules-v1'
        else:
            ex = executor_class(fk, qpos, step, lambda: float(qpos()[-2:].sum()), lambda: {k: bool(scalar(info.get(k, False))) for k in ('ee_rest', 'robot_rest', 'is_static')}, model, max_steps=profile.step_cap, extended_attempt=profile.step_cap == 1000, parameters=ParametersV3(close_uses_endpoint_tolerance=profile.close_uses_endpoint_tolerance, stall_feedback=profile.stall_feedback, stall_guard_mode=profile.stall_guard_mode), read_qvel=qvel, control_dt=report['control_dt'], initial_delta=qvel() * report['control_dt'], gripper=float(np.clip((qpos()[-2:].mean() + 0.01) / 0.06, 0, 1)), **executor_options)

        if profile.parallel_case_workers:
            pick.atomic_ledger_save(args.output/'initialized.json',dict(pid=os.getpid(),
                authorization_sha256=sha(args.authorization),steps=ex.steps,ready_at_epoch=time.time()))
        if aligned_preflight:
            script=read(args.code/auth['preflight_script'])
            report['preflight']=execute_aligned_preflight(ex,script,lambda events:save(args.output/'preflight-events.json',events))
            report.update(status='preflight_passed' if report['preflight']['passed'] else 'preflight_failed',
                explicit_resets=1,execution_profile=profile.name,alignment_samples=ex.alignment_samples,
                controller_diagnostics=ex.controller_diagnostics,
                recording_kind='archived-command-prefix with new grasp alignment; no API; not an SR trial')
            return report

        if full_b_preflight:
            snapshot=memory_ab_initial_snapshot(args.case_id,row,seed,qpos(),qvel(),world_base(),u.get_state_dict(),
                dict(runtime_commit='e9ff3d23496d38e4431c8d913e147ffa007f7f72',asset_sha256=auth['asset_sha256']))
            save(args.output/'initial-reference.eval-only.json',snapshot)
            report['preflight']=execute_memory_preflight(ex,qpos,fk,lambda events:save(args.output/'preflight-events.json',events))
            report.update(status='preflight_passed' if report['preflight']['passed'] else 'preflight_failed',
                          explicit_resets=1,source_fingerprint=pick.ab_source_fingerprint(),reference_sha256=sha(args.output/'initial-reference.eval-only.json'))
            return report

        if replay_mode:
            def replay_observe_end():
                observation, manifest = observe(u, fk, qpos, target_semantics(row['object_id']), [],
                    profile.step_cap-ex.steps, args.output/'post-command-sensors', ex.odometry(),
                    ObservationDepth(), call_cap=profile.call_cap, profile=profile)
                if auth.get('scope') in (pick.R31_REPLAY_SCOPE,pick.R32_REPLAY_SCOPE,pick.R33_REPLAY_SCOPE):
                    record_r31_end_observation(args.output, auth, report, ex.steps, observation, manifest)
                else:
                    save(args.output/'post-command-observation.json', dict(observation=observation, sensor_manifest=manifest))
            execute_l1_replay(args, auth, report, ex, qpos, qvel, replay_observe_end, lambda: stopped)
            return report

        def record(call, result):
            report['commands'].append({'turn': len(report['commands']), 'call': call, 'result': result, 'start_step': ex.steps - result.get('steps_used', 0)})
            if profile.search_recessed_view:
                start=report['pending_command']['start_step']
                report['commands'][-1].update(start_step=start,
                    ack_interval=dict(start_step=start,end_step=ex.steps))
                report['commands'][-1]['search_memory']=copy.deepcopy(
                    (report.get('pending_command',{}).get('call') or {}).get('search_memory'))
            if call and call['tool'] == 'check_path':
                accepted_checks.append(result.get('accepted') is True)
            write_report()
        profile = pick.PROFILES[args.profile]
        state = new_attempt_state(profile, args.output,
            attempt_id=pick.session_attempt_id(auth,args.case_id,profile),
            **({'case_id':args.case_id,'authorization_sha256':sha(args.authorization)} if profile.cold_checkpoint else {}),
            **({'handoff_plan': auth.get('handoff_plan')} if profile.task_handoff else {}))
        depth = ObservationDepth()
        if profile.identity_revision == 2:
            def read_grasp_sensors():
                q = qpos().copy()
                data = read_sensor_data_without_evaluate(u)
                raw = array(data['fetch_hand']['depth'])[0, ..., 0]
                k = array(u._sensors['fetch_hand'].camera.get_intrinsic_matrix()).reshape(3, 3)
                result = grasp_sensor_feedback(q, fk, raw, k)
                result['control_step'] = ex.steps
                return result
            ex.grasp_sensor_feedback = read_grasp_sensors
        if auth.get('cold_resume') or auth.get('checkpoint_branch'):
            branch=auth.get('checkpoint_branch')
            (checkpoint_branch_seed if branch else cold_resume_seed)(auth,profile)
            grant=branch or auth['cold_resume']
            payload=read_cold_checkpoint(grant['checkpoint'],receipt_sha256=grant['receipt_sha256'],binding=grant['binding'])
            expected_resume_entries=paid_delivery_evidence(auth,profile,payload['ledger']['entries'])[0]
            if expected_resume_entries!=read(args.ledger)['entries']:
                raise ValueError('Resume ledger differs from checkpoint')
            if payload['runtime_state']['stopped']:raise ValueError('Official terminal checkpoint')
            if branch:validate_checkpoint_branch_payload(payload,auth,profile)
            elif grant.get('source_migration'):validate_carry_resume_payload(payload,grant)
            migrated_memory=None
            if grant.get('source_migration',{}).get('contract')=='local-support-floor-resume-v1':
                artifact=load_local_floor_artifact(auth,grant)
                migrated_memory=validate_local_floor_memory_artifact(payload['state'].episode_memory,artifact,
                    checkpoint_receipt_sha256=grant['receipt_sha256'],source_sha256=auth['source_sha256'])
            native_snapshot=payload['native'];canonical_receipt=None
            velocity_grant=(grant.get('source_migration') or {}).get('derived_velocity') or grant.get('derived_velocity')
            if velocity_grant:
                native_snapshot,canonical_receipt=canonical_fetch_velocity_snapshot(native_snapshot,
                    urdf=args.lab_root/'src/official-mshab-runtime/ManiSkill/mani_skill/assets/robots/fetch/fetch.urdf',
                    urdf_sha256=velocity_grant['urdf_sha256'])
                save(args.output/'cold-derived-velocity.json',canonical_receipt)
            restore_receipt=full_b[2].checked_restore(full_b_base,native_snapshot,diagnostics=args.output/'cold-restore-audit')
            if branch:
                state,migration=restore_checkpoint_branch_python(payload,ex,auth,profile,
                    session_path=args.output/'response-session.json',authorization_sha256=sha(args.authorization))
            else:
                retained_before=carry_resume_digest(payload) if grant.get('source_migration') else None
                state=restore_cold_python(payload,ex,session_path=args.output/'response-session.json')
                if retained_before is not None:
                    restored=dict(payload,state=state,executor={k:getattr(ex,k) for k in payload['executor']})
                    if carry_resume_digest(restored)!=retained_before:
                        raise ValueError('Carry resume changed retained state')
                if migrated_memory is not None:
                    state.episode_memory=migrated_memory
                    save(args.output/'local-floor-memory-migration.json',dict(
                        artifact_sha256=grant['source_migration']['memory_artifact_sha256'],
                        checkpoint_receipt_sha256=grant['receipt_sha256'],audit=artifact['audit']))
                if not auth.get('paid_delivery_resume'):
                    state.response_session.rebind_checkpoint_authorization(sha(args.authorization),
                        state.archive,state.history,checkpoint_receipt_sha256=grant['receipt_sha256'])
            info=payload['runtime_state']['info']
            report.clear();report.update(payload['runtime_state']['report'])
            report.update(status='running',environment_closed=False,batch_id=auth['batch_id'],
                execution_profile=profile.name,information_condition=profile.name,
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                cold_resume=dict(checkpoint=grant['checkpoint'],receipt_sha256=grant['receipt_sha256'],
                    parent_steps=ex.steps,parent_requests=len(state.archive)+state.verification_requests,
                    parent_recorded_at_utc=payload['runtime_state']['report'].get('recorded_at_utc'),
                    prior_wall_seconds=payload['runtime_state']['wall_seconds'],native_restore=restore_receipt))
            if branch:
                report['checkpoint_branch']=dict(label=branch['label'],parent_batch_id=branch['parent_batch_id'],
                    parent_result_sha256=branch['parent_result_sha256'],parent_result_preserved=True,
                    from_checkpoint_steps=ex.steps,from_checkpoint_requests=branch['prefix_requests'],
                    source_delta=branch['source_delta'],migration=migration,
                    parent_full_committed_usd=str(pick.committed_cost(read(Path(branch['parent_ledger_path']))['entries'])),
                    inherited_prefix_usd=str(pick.committed_cost(payload['ledger']['entries'])),
                    additional_usd_cap=branch['additional_usd_cap'],
                    outcome_scope='diagnostic branch from saved prefix; not fresh-start benchmark success')
                save(args.output/'checkpoint-branch.json',report['checkpoint_branch'])
            report.pop('error_type',None);report.pop('error_message',None)
            save(args.output/'cold-resume.json',report['cold_resume'])
            if grant.get('source_migration'):
                report['cold_resume']['source_migration']=dict(grant['source_migration'],
                    retained_state_sha256=retained_before,invariant_state_equal=True)
                if canonical_receipt is not None:
                    report['cold_resume']['derived_velocity']=canonical_receipt
                    report['cold_resume']['equivalence_scope']='saved generalized state; URDF-derived link velocity caches'
                save(args.output/'cold-resume.json',report['cold_resume'])
            ex.recorder.segment_start_step=ex.steps
            ex.recorder.state_guard.rebind_after_restore()
            save(args.output/'recording-segment.json',dict(start_step=ex.steps,
                parent_checkpoint=grant['checkpoint'],frame_step='start_step + frame_index'))
            ex._capture(None,'checkpoint_branch' if branch else 'cold_resume');ex.initial_recorded=True
        episode_checker=None
        if profile.episode_memory or profile.r2b_clearance:
            robot_root=args.lab_root/'src/official-mshab-runtime/ManiSkill/mani_skill/assets/robots/fetch'
            geometry=episode_geometry_pin(args.code,installed_robot_root=robot_root,
                refine_base=profile.clearance_unknown_advisory,refine_fingers=profile.carried_envelope)
            calibration_binding=hashlib.sha256(canonical(dict(
                robot_geometry_sha256=EPISODE_GEOMETRY_SHA256,head_link='head_camera_link',
                hand_link='gripper_link',hand_mount_m=[-.1,0.,.1],depth_unit='mm',
                optical_axes='right_down_forward',sensor_protocol='robot-rgbd-640-v1'))).hexdigest()
            episode_checker=bind_episode_modules(state,ex,fk,geometry=geometry,
                calibration_binding=calibration_binding,local_support_floor=profile.local_support_floor)
            refresh_step=[None]
            def refresh_episode_sensors(*,step,measured):
                # Five control steps at most between fresh robot views; checking
                # still occurs before every proposed packet. This is not an
                # observed latency/braking safety guarantee.
                if refresh_step[0] is not None and step-refresh_step[0]<5:
                    episode_checker.contact_region=episode_contact_region(state,ex,fk,measured['qpos'])
                    return
                before=state_copy(u.get_state_dict());q=vector(qpos(),15).copy()
                data=read_sensor_data_without_evaluate(u)
                if not state_equal(before,state_copy(u.get_state_dict())) or not np.array_equal(q,qpos()):
                    raise ValueError('Clearance sensor refresh changed physical state')
                q=clearance_sensor_joints(q,measured['qpos'])
                cameras={};hashes={}
                for short,long in (('head','fetch_head'),('hand','fetch_hand')):
                    raw=array(data[long]['depth'])[0,...,0]
                    rgb=array(data[long]['rgb'])[0]
                    k=array(u._sensors[long].camera.get_intrinsic_matrix()).reshape(3,3)
                    cameras[short]=dict(depth_mm=raw,intrinsics=k,camera_in_base=camera_in_base(fk,q,short))
                    for modality,pixels in (('depth',raw),('rgb',rgb)):
                        hashes[short+'_'+modality]=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
                oid='obs-motion-'+str(step)+'-'+hashlib.sha256(canonical(hashes)).hexdigest()[:16]
                frame=ObservationDepth();frame.set(oid,q,cameras)
                odom=dict(zip(('dx_m','dy_m','dyaw_rad'),map(float,ex.odometry())))
                ingest_episode_frame(state,episode_checker,frame,step=step,odometry=odom,sensor_sha256=hashes,record_views=False)
                episode_checker.contact_region=episode_contact_region(state,ex,fk,q)
                refresh_step[0]=step
            if profile.local_clearance:ex.refresh_clearance=refresh_episode_sensors
        def observe_turn(turn, history, previous):
            observed_history = history_for_observation_audit(history, profile)
            observation, manifest = observe(u, fk, qpos, target_semantics(row['object_id']),
                observed_history, profile.step_cap-ex.steps, args.output/'sensors'/f'turn-{turn:03d}',
                ex.odometry(), depth, previous, call_cap=profile.call_cap, profile=profile,
                **(dict(decision_turn=turn,verification_requests=state.verification_requests)
                   if getattr(profile,'stage_plan',False) else {}))
            if profile.measure or profile.verification or profile.stall_feedback: observation['history'] = copy.deepcopy(history)
            if profile.cold_checkpoint and report.get('cold_resume'):
                observation['resumed_execution'] = dict(
                    status='resumed_with_new_wall_window',
                    restored_control_step=report['cold_resume']['parent_steps'],
                    segment_wall_seconds_remaining=max(0,int(cap-(time.monotonic()-started))),
                    history_policy='prior_limits_are_historical; current_limits_still_apply')
                priority = (auth.get('cold_resume') or auth.get('checkpoint_branch') or {}).get('user_priority')
                if priority is not None:
                    observation['resumed_execution']['user_priority'] = priority
                    pick.validate_resumed_execution(observation['resumed_execution'])
            if profile.target_evidence_diagnostics:
                observation['accounting_remaining'] = resource_accounting_snapshot(args.ledger, auth,
                    sha(args.authorization), args.case_id)
            bundle = dict(position=args.case_id, turn=turn, kind='tool',
                authorization_sha256=sha(args.authorization), observation=observation, sensor_manifest=manifest)
            if rest_checker is not None:
                from types import SimpleNamespace
                ingest_episode_frame(SimpleNamespace(episode_memory=rest_memory),rest_checker,depth,
                    step=ex.steps,odometry=observation['odometry'],
                    sensor_sha256={key:value['sha256'] for key,value in manifest.items()},record_views=False)
            if profile.episode_memory or profile.r2b_clearance:
                ingest_episode_frame(state,episode_checker,depth,step=ex.steps,odometry=observation['odometry'],
                    sensor_sha256={key:value['sha256'] for key,value in manifest.items()})
                refresh_step[0]=ex.steps
            if turn==0 and auth.get('scope')==pick.FROZEN['memory_ab']['scope']:
                snapshot=memory_ab_initial_snapshot(args.case_id,row,seed,qpos(),qvel(),world_base(),u.get_state_dict(),
                    dict(runtime_commit='e9ff3d23496d38e4431c8d913e147ffa007f7f72',asset_sha256=auth['asset_sha256']))
                pick.MemoryABBatch.from_auth(auth).register_start(auth['ab_attempt_id'],snapshot,bundle)
            return bundle
        def deliver(wire, bundle, archive):
            if profile.automatic_supervision:
                automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
            request_dir = args.output/'provider'/f"turn-{bundle['turn']:03d}"
            request_dir.mkdir(parents=True)
            response = pick.process(bundle, archive, args.condition, request_dir, args.ledger,
                auth, sha(args.authorization), runtime_sender(request_dir, lambda: cap-(time.monotonic()-started)-10, connection_retries=2 if not profile.automatic_supervision and auth.get('scope')=='baseline20-supervised-20261006' else 0), profile,
                **({'session': state.response_session} if profile.response_session else {}))
            report['api_requests'] += 1
            if response['status']=='censored': raise RuntimeError('provider censored')
            if profile.automatic_supervision:
                automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
            report['pending_command'] = dict(turn=bundle['turn'], call=response.get('call'),
                start_step=ex.steps, reasoning_summaries=response.get('reasoning_summaries', []))
            if profile.r15_localization_eval:
                report['pending_command']['localization_frame_eval_only']=dict(base_world=world_base().tolist(),object_world=matrix(target.pose).tolist())
            write_report()
            if auth.get('command_supervision') or profile.automatic_supervision:
                supervise_command(args.output, report['pending_command'], deadline, automatic=profile.automatic_supervision)
            return response
        def record_transition(event):
            if getattr(profile,'stage_plan',False) and event['function']=='s2_plan_result':
                report.setdefault('plans',[]).append(copy.deepcopy(event['output'][1]))
                report.pop('pending_command',None)
                write_report()
                with (args.output/'tool-trace.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
                return
            if (profile.verification or getattr(profile,'stage_plan',False)) and event['function'] != 'decision_result':
                with (args.output/'tool-trace.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
                if event['function'] in ('identity_review', 'target_review'):
                    report.setdefault('verification_events', []).append(copy.deepcopy(event))
                return
            actual, result = event['output']
            turn = len(report['commands'])
            if actual and actual['tool']=='locate_point':
                diagnostic = diagnose_location(result, surface_triangles, matrix(target.pose), world_base())
                report.setdefault('localization_eval_only', []).append(dict(diagnostic, turn=turn, step=ex.steps))
            if profile.r15_localization_eval and actual and actual['tool'] in ('locate_points','approach_point'):
                bound=report['pending_command']['localization_frame_eval_only']
                report.setdefault('localization_eval_only',[]).extend(r15_localization_scores(
                    actual,result,surface_triangles,bound,turn,report['pending_command']['start_step']))
            if actual and actual['tool']=='measure':
                for sample in result.get('samples', []):
                    diagnostic = diagnose_location(sample, surface_triangles, matrix(target.pose), world_base())
                    report.setdefault('localization_eval_only', []).append(dict(diagnostic, turn=turn, step=ex.steps, pixel=sample['pixel']))
            if profile.verification and actual and actual['tool']=='locate_point' and result.get('valid'):
                record_id = (result.get('target_record') or {}).get('id') if profile.identity_revision == 2 else result.get('identity_verification',{}).get('target_id')
                report.setdefault('verification_truth_eval_only', []).append(dict(target_id=record_id,turn=turn,**diagnostic))
            record(actual, result)
            if getattr(profile,'stage_plan',False):
                report['commands'][-1].update({k:event[k] for k in
                    ('decision_turn','command_id','plan_id','revision','source_frame_id','command_entry_frame_id') if k in event})
                report['commands'][-1]['ack_interval']=dict(start_step=event['start_step'],end_step=event['end_step'])
            report['commands'][-1]['reasoning_summaries'] = report['pending_command']['reasoning_summaries']
            if not getattr(profile,'stage_plan',False):report.pop('pending_command', None)
            if profile in pick.PAIRED_PROFILES:
                issue=r15_batch_issue(report, isolate_case=pick.r15_case_isolation(auth))
                if issue:
                    write_report()
                    write_batch_stop(auth['output_root'],'R1.5 issue: '+issue,preserve_first=pick.r15_case_isolation(auth))
            with (args.output/'tool-trace.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        def deliver_verification(wire):
            if profile.automatic_supervision:
                automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
            sequence = state.verification_requests - 1
            request_dir = args.output/'verifier'/f'request-{sequence:03d}'
            sender=runtime_sender(request_dir,lambda:cap-(time.monotonic()-started)-10)
            phase_event=None
            if profile.search_recessed_view:
                memory=((report.get('pending_command') or {}).get('call') or {}).get('search_memory') or {}
                phase_event=dict(sequence=sequence,
                    decision_turn=len(report['commands']),phase=memory.get('phase'),
                    status='prepared_no_retry')
                report.setdefault('search_phase_verifications',[]).append(phase_event)
                write_report()
                transport_sender=sender
                def sender(payload,key):
                    phase_event.update(status='sending',request_id=key)
                    write_report()
                    return transport_sender(payload,key)
            try:
                if profile.r2a_identity:
                    response=pick.process_r2a_verification(wire,args.case_id,sequence,request_dir,args.ledger,auth,sha(args.authorization),sender,state.response_session,profile=profile)
                elif profile.weekly_budget:
                    response=pick.process_weekly_verification(wire,args.case_id,sequence,request_dir,args.ledger,
                        auth,sha(args.authorization),sender,state.response_session,profile=profile)
                else:
                    response=pick.process_c2_wire(wire,'verification',args.case_id,sequence,request_dir,args.ledger,
                        auth,sha(args.authorization),sender)
            except Exception:
                if phase_event is not None:
                    phase_event['status']='unknown' if phase_event['status']=='sending' else 'not_sent'
                    write_report()
                raise
            if phase_event is not None:
                phase_event.update(status='settled',provider_status=response['status'],
                    request_id=response.get('request_id'),provider_id=response.get('provider_id'))
            report['api_requests'] += 1
            report['verification_requests'] = report.get('verification_requests',0) + 1
            write_report()
            if profile.automatic_supervision:
                automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
            if response['status'] != 'valid':
                raise ValueError('completed verifier reply invalid; recorded without retry')
            return response['reply']
        if profile.hot_continuation:
            def checkpoint_live(continuing_state):
                from bvi.pick_system.snapshot_transport import capture_transport
                deadline()
                if stopped or ex.pending is not None:raise ValueError('Unsafe continuation boundary')
                destination=args.output/'hot-checkpoint-001'
                if destination.exists():raise ValueError('Checkpoint already exists; no replay')
                preflight=full_b[2]
                before=capture_transport(full_b_base)
                fingerprint=preflight.state_fingerprint(before)
                memory_record=episode_checkpoint_record(continuing_state,ex.steps)
                preflight.write_diagnostic_bundle(destination,'native-state',before)
                continuing_state.response_session.persist()
                search_memory=dict(archive=continuing_state.archive,
                    history=continuing_state.history,observation_ids=continuing_state.observation_ids,
                    identity_target=continuing_state.identity_target,target_reviews=continuing_state.target_reviews,
                    session=read(continuing_state.response_session.path),executor=ex._state())
                if memory_record is not None:search_memory['episode_memory']=memory_record
                save(destination/'search-memory.json',search_memory)
                after=preflight.state_fingerprint(capture_transport(full_b_base))
                if fingerprint!=after:raise ValueError('Live checkpoint mutated native state')
                if memory_record!=episode_checkpoint_record(continuing_state,ex.steps):
                    raise ValueError('Live checkpoint mutated episode memory')
                receipt=dict(mode='same-process-live-continuation',native_state_unchanged=True,
                    native_sha256=fingerprint,steps=ex.steps,requests=len(continuing_state.archive)+continuing_state.verification_requests,
                    environment_recreated=False,controller_recreated=False,session_recreated=False,
                    cold_process_restore_supported=False)
                if memory_record is not None:
                    receipt['episode_memory_sha256']=memory_record['sha256']
                    receipt['episode_memory_unchanged']=True
                save(destination/'live-continuation.json',receipt)
                report['hot_continuation']=receipt
                write_report()
                return dict(path=destination.name,receipt_sha256=sha(destination/'live-continuation.json'))
            ex.hot_checkpoint=checkpoint_live
        if profile.cold_checkpoint:
            # Reserve shutdown time for the acknowledged-action receipt and
            # complete native/session checkpoint. The hard deadline is unchanged.
            ex.action_budget_expired=lambda:time.monotonic()-started>=max(0.,cap-60.)
            def checkpoint_cold(continuing_state):
                from bvi.pick_system.snapshot_transport import capture_transport
                if stopped:raise ValueError('Official terminal state is not resumable')
                ledger=read(args.ledger)
                root=args.output/'cold-checkpoints'/f'{ex.steps:06d}-{len(ledger["entries"]):04d}-{ex.usable_step_cap():04d}-{continuing_state.response_session.status}'
                if root.exists():return read(root/'receipt.json')
                native=capture_transport(full_b_base)
                fingerprint=full_b[2].state_fingerprint(native)
                binding=dict(case_id=args.case_id,profile=profile.name,
                    source_sha256=auth['source_sha256'],asset_manifest_sha256=auth.get('asset_manifest_sha256'),
                    authorization_sha256=sha(args.authorization),attempt_id=continuing_state.response_session.attempt_id)
                receipt=write_cold_checkpoint(root,state=continuing_state,executor=ex,native=native,
                    ledger=ledger,binding=binding,runtime_state=dict(info=info,report=report,
                        wall_seconds=time.monotonic()-started+float((report.get('cold_resume') or {}).get('prior_wall_seconds',0.)),stopped=stopped))
                if full_b[2].state_fingerprint(capture_transport(full_b_base))!=fingerprint:
                    raise ValueError('Checkpoint capture changed live state')
                save(args.output/'cold-checkpoint-latest.json',dict(path=str(root),receipt_sha256=sha(root/'receipt.json'),
                    steps=ex.steps,requests=len(ledger['entries']),native_unchanged=True))
                return receipt
            ex.cold_checkpoint=checkpoint_cold
        if auth.get('cold_probe') is True:
            if not profile.cold_checkpoint:raise ValueError('Cold probe profile required')
            if 'carried_grasp_probe' in auth:
                result=run_carried_grasp_probe(ex,state,depth,observe_turn,fk,auth,
                    lambda:full_b[1].make_controller_session(full_b[0],qpos(),
                        sim_freq=full_b_base.uenv.sim_freq,control_freq=full_b_base.uenv.control_freq))
                if not stopped:checkpoint_cold(state)
                report.update(status='completed',cold_probe=dict(no_api=True,result=result),
                    executor_trace=ex.trace,ik_records=ex.ik_records,
                    **horizon_results([report['initial_evaluation']]+report['frames']))
                return report
            if 'clearance_action_probe' in auth:
                result=run_clearance_action_probe(ex,auth)
            elif 'head_observation_probe' in auth:
                result=run_head_observation_probe(ex,auth)
            else:
                result=(dict(accepted=True,reason='zero_action_branch_restore',steps_used=0)
                    if auth.get('checkpoint_branch') else ex.look(-.1 if auth.get('cold_resume') else 0.,.7))
            checkpoint_cold(state)
            report.update(status='completed',cold_probe=dict(no_api=True,result=result),strict_pick_success=False)
            return report
        if auth.get('paid_delivery_resume'):
            recovered=restore_paid_delivery(state,auth,profile,
                observe_turn(len(state.archive),state.history,state.previous),args.output)
            bundle,response=recovered
            report['api_requests']=len(read(args.ledger)['entries'])
            report['pending_command']=dict(turn=bundle['turn'],call=response['call'],start_step=ex.steps,
                reasoning_summaries=response.get('reasoning_summaries',[]),recovered_paid_response=True)
            write_report()
            claim=Path(auth['paid_delivery_resume']['claim_path'])
            with claim.open('x',encoding='utf8') as stream:
                json.dump(dict(authorization_sha256=sha(args.authorization),response_id=response['provider_id'],
                    step=ex.steps,status='execution_claimed_no_automatic_replay'),stream)
                stream.flush();os.fsync(stream.fileno())
            pick.advance(state,bundle,response,ex,depth,profile,sink=record_transition,
                verification_deliver=deliver_verification)
            state.response_session.rebind_checkpoint_authorization(sha(args.authorization),
                state.archive,state.history,checkpoint_receipt_sha256=auth['cold_resume']['receipt_sha256'])
            save(args.output/'paid-delivery-recovery.json',dict(response_id=response['provider_id'],
                request_regenerated=False,new_api_requests=0,start_step=1906,end_step=ex.steps,
                session_status=state.response_session.status,claim_sha256=sha(claim)))
            if not stopped:checkpoint_cold(state)
        run_with_ports(state, observe_turn, deliver, ex, depth, lambda: stopped, profile, record_transition,
                       verification_deliver=deliver_verification if (profile.verification or profile.r2a_identity) else None)
        if state.terminal_reason: report['termination_category'] = state.terminal_reason
        if profile.r2d_extended_calls:
            report.setdefault('success_within_50_total_requests',False)
            report['request_prefix_comparison']='same-run <=50 total requests; approximate, not a matched 50-call policy'
        report['reflection_status'] = 'not_requested; terminal hindsight and per-call notes retained'
        report.update(executor_trace=ex.trace, ik_records=ex.ik_records)
        scored = [report['initial_evaluation']] + report['frames']
        report.update(status='completed', **horizon_results(scored))
        if profile.stall_guard_mode == 'reference_progress':
            report['baseline_budget_results'] = baseline_prefix_results(scored, report['commands'], profile)
            report['horizon_description'] = f'nonofficial {profile.step_cap}-step horizon; same-run 25-decision/600-step prefix'
        report['termination_category'] = report.get('termination_category') or ('official_success' if report['strict_pick_success'] else 'cumulative_force_limit' if float(scalar(info['robot_cumulative_force'])) >= 5000 else 'time_limit' if stopped else 'tool_call_limit')
        report['funnel'] = funnel(scored, accepted_checks, initial_z)
    except pick.CaseLimitReached as exc:
        report.update(status='completed',termination_category=exc.category,case_limit_reached=True,stop_reason=str(exc))
    except pick.C2BudgetExhausted as exc:
        report.update(budget_exhaustion_report(auth,exc,profile=pick.PROFILES[args.profile]))
        if initialization_observability:
            report['initialization_error_eval_only'] = record_initialization_error(args.output, initialization_stage, exc, primary=True)
    except SupervisionStopped as exc:
        report.update(status='supervision_stopped', termination_category='operator_stop', stop_reason=str(exc))
        if initialization_observability:
            report['initialization_error_eval_only'] = record_initialization_error(args.output, initialization_stage, exc, primary=True)
    except Exception as exc:
        report.update(status='infrastructure_censored', error_type=type(exc).__name__, error_message=str(exc))
        if initialization_observability:
            report['initialization_error_eval_only'] = record_initialization_error(
                args.output, initialization_stage, exc, primary=True)
    finally:
        if pick.PROFILES[args.profile].parallel_case_workers and report.get('status')!='completed':
            write_batch_stop(auth['output_root'],'worker failure: '+args.output.name+' '+str(report.get('error_type',''))+': '+str(report.get('error_message',report.get('status'))),preserve_first=pick.r15_case_isolation(auth))
        if ex is not None and hasattr(ex,'cold_checkpoint') and 'state' in locals():
            try:
                if not stopped and (state.response_session.status=='ready' or state.terminal_reason in ('give_up','request_limit')):ex.cold_checkpoint(state)
            except Exception as error:report['cold_checkpoint_error']=repr(error)
        if not init_only and pick.PROFILES[args.profile].staged_metrics:
            if initialization_observability:
                initialization_final_action(args.output, report, 'staged_metrics',
                    lambda: report.update(staged_metrics_eval_only=staged_metrics(report)))
            else:
                report['staged_metrics_eval_only'] = staged_metrics(report)
        if not init_only and pick.PROFILES[args.profile].model_target_boxes and not replay_mode:
            if initialization_observability:
                initialization_final_action(args.output, report, 'request_accounting',
                    lambda: finalize_request_accounting(report,args.ledger,auth))
            else:
                finalize_request_accounting(report,args.ledger,auth)
        if ex is not None:
            report.update(executor_trace=ex.trace, ik_records=ex.ik_records)
            if hasattr(ex,'controller_diagnostics'):
                report.update(controller_diagnostics=ex.controller_diagnostics,alignment_samples=ex.alignment_samples)
        if full_b is not None and ex is not None:
            if not aligned_preflight:finish_memory_recording(ex,report)
        if aligned_preflight:finish_aligned_preflight(ex,report,env)
        if pick.PROFILES[args.profile].stall_guard_mode == 'reference_progress' and 'initial_evaluation' in report:
            try:
                report['baseline_budget_results'] = baseline_prefix_results(
                    [report['initial_evaluation']] + report['frames'], report['commands'], pick.PROFILES[args.profile])
                report['baseline_budget_results']['record_status'] = report['status']
            except ValueError as exc:
                report['baseline_budget_results'] = dict(status='incomplete_evidence', reason=str(exc))
        if env is not None and not aligned_preflight:
            try:
                env.close()
                if reference_only or is_pick20:
                    report['environment_closed'] = True
            except Exception as exc:
                report['close_error_type'] = type(exc).__name__
                if pick.PROFILES[args.profile].parallel_case_workers:
                    write_batch_stop(auth['output_root'],'worker close failure: '+args.output.name+' '+type(exc).__name__+': '+str(exc),preserve_first=pick.r15_case_isolation(auth))
                if reference_only or is_pick20:
                    report['status'] = 'infrastructure_censored'
                if initialization_observability:
                    primary = 'initialization_error_eval_only' not in report and 'error_type' not in report
                    evidence = record_initialization_error(args.output, 'close', exc, primary=primary)
                    report['initialization_close_error_eval_only'] = evidence
                    if primary:
                        report.update(status='infrastructure_censored', error_type=type(exc).__name__,
                                      error_message=str(exc), initialization_error_eval_only=evidence)
        if auth.get('scope')==pick.FROZEN['memory_ab']['scope']:
            try:
                batch=pick.MemoryABBatch.from_auth(auth)
                if report.get('status')=='completed' and not report.get('close_error_type'):
                    outcome=report.get('termination_category')
                    outcome=(outcome if outcome in ('done','give_up') else 'call_limit' if outcome=='tool_call_limit'
                             else 'step_limit' if ex.steps>=600 else 'official_termination')
                    batch.finish(auth['ab_attempt_id'],ex.steps,outcome)
                else:batch.abort(auth['ab_attempt_id'],None if ex is None else ex.steps)
            except Exception as error:
                report['ab_closeout_error']=type(error).__name__+': '+str(error)
                report['status']='infrastructure_censored'
        segment_wall=time.monotonic()-started
        report.update(wall_seconds=segment_wall+float((report.get('cold_resume') or {}).get('prior_wall_seconds',0.)),
            segment_wall_seconds=segment_wall,
            recording={'kind': ('recorded-action replay ' if replay_mode else
                'checkpoint diagnostic branch segment ' if report.get('checkpoint_branch') else
                'cold resumed segment ' if report.get('cold_resume') else 'online official-spawn ') + args.condition,
                'frames': 0 if recording is None else recording.count, 'directory': 'recording-frames'})
        if pick.PROFILES[args.profile].r1_extended_budget:
            report['horizon_description'] = f'nonofficial {horizon}-step horizon at 20 Hz; 600-step prefix of same trajectory; unchanged success and force limits'
            report['budget_condition'] = dict(control_steps_cap=horizon, model_calls_cap=(pick.PROFILES[args.profile].call_cap if pick.PROFILES[args.profile] in pick.V3_PROFILES else 50), worker_seconds_cap=cap)
        if pick.PROFILES[args.profile].weekly_budget:
            try:
                report['weekly_accounting']=weekly_accounting(report,read(args.ledger)['entries'],
                    locals().get('state'),pick.PROFILES[args.profile])
                report['horizon_description']=f'nonofficial {horizon}-step horizon at 20 Hz; unchanged strict success and cumulative force limit'
            except Exception as error:
                report.update(status='infrastructure_censored',accounting_error=str(error))
        if pick.PROFILES[args.profile] in pick.V3_PROFILES:
            try:
                report['v3_cost_accounting']=v3_cost_accounting(read(args.ledger),args.case_id)
            except Exception as error:
                report.update(status='infrastructure_censored',accounting_error=str(error))
        if initialization_observability:
            try:
                write_report()
            except Exception as exc:
                primary = 'initialization_error_eval_only' not in report and 'error_type' not in report
                evidence = record_initialization_error(args.output, 'final_result_write', exc, primary=primary)
                report['initialization_result_write_error_eval_only'] = evidence
                if primary:
                    report.update(status='infrastructure_censored', error_type=type(exc).__name__,
                                  error_message=str(exc), initialization_error_eval_only=evidence)
                if init_only:
                    report['status']='infrastructure_censored'
        else:
            write_report()
    if r15_journal is not None:r15_journal.finish()
    return report


# Transport construction is reachable only after live authorization checks.
def transient_before_send(error):
    """Only transient connection failures; no certificate/auth/config retries."""
    import socket, ssl, errno
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    return isinstance(error, (TimeoutError, ConnectionError, ssl.SSLEOFError)) or (
        isinstance(error, OSError) and error.errno in
        (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT))


def send_with_connection_retries(send, wire, key, remaining, emit, *, retries=2, sleep=time.sleep):
    """Retry only proven-unsent connections; one unchanged model decision."""
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError('connection retries must be 0..2')
    for attempt in range(retries + 1):
        emit(dict(event='connection_attempt', attempt=attempt + 1))
        try:
            return send(wire, key)
        except pick.RequestNotSent:
            if attempt == retries:
                raise
            delay = 2 ** (attempt + 1)
            if remaining() <= delay + 1:
                raise
            emit(dict(event='connection_retry', delay_seconds=delay, next_attempt=attempt + 2))
            sleep(delay)


def retain_http_error_response(folder, request_id, status, raw, provider_request_id):
    """Retain provider rejection evidence without credentials or header dumps.

    Exclusive writes preserve prior evidence. A metadata receipt exists only
    after the complete raw bytes have been flushed to disk.
    """
    folder=Path(folder)
    raw_path=folder/'http-error-response.bin'
    with raw_path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    receipt=dict(status_code=status,client_request_id=request_id,
        provider_request_id=provider_request_id,body_file=raw_path.name,
        body_bytes=len(raw),body_sha256=hashlib.sha256(raw).hexdigest(),body_complete=True)
    with (folder/'http-error-response.json').open('x',encoding='utf8') as stream:
        json.dump(receipt,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    return receipt


def runtime_sender(folder, remaining, *, connection_retries=0):
    if os.environ.get('BVI_NO_API')=='1':raise RuntimeError('Provider disabled for no-API preflight')
    if os.name!='posix' or not Path(os.environ.get('BVI_LAB_ROOT') or Path.home()/'bvi-research').is_dir():
        raise RuntimeError('API transport is lab-server-only')
    import socket
    import ssl
    import stat
    import http.client
    from urllib.error import HTTPError
    class LeaseExpired(RuntimeError):
        pass

    def credential_metadata(path):
        path = Path(path)
        info = path.lstat()
        parent = path.parent.lstat()
        return {'path': str(path), 'regular_file': stat.S_ISREG(info.st_mode), 'not_symlink': not stat.S_ISLNK(info.st_mode), 'mode': stat.S_IMODE(info.st_mode), 'parent_mode': stat.S_IMODE(parent.st_mode), 'owner_matches': info.st_uid == os.getuid(), 'parent_owner_matches': parent.st_uid == os.getuid(), 'content_read': False}

    def validate_metadata(meta):
        if not (meta['regular_file'] and meta['not_symlink'] and meta['owner_matches'] and meta['parent_owner_matches'] and (meta['mode'] == 384) and (meta['parent_mode'] == 448)):
            raise PermissionError('credential file must be owned regular 600 file in owned 700 directory')

    def load_runtime_key(path):
        """Only the approved broker runtime calls this; never shell-source an env file."""
        meta = credential_metadata(path)
        validate_metadata(meta)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'r', encoding='utf-8') as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 384:
                raise PermissionError('credential changed during open')
            lines = stream.read(16384).splitlines()
        values = [line.split('=', 1)[1].strip() for line in lines if line.startswith('OPENAI_API_KEY=')]
        if len(values) != 1 or not values[0] or any((c.isspace() for c in values[0])):
            raise ValueError('one nonempty OPENAI_API_KEY assignment required')
        return values[0]

    class PhaseTransport:
        """One verified TLS connection per send; same wire and policy as local broker.

        A single absolute deadline covers all phases. A phase receipt deliberately
        contains only an enum, duration, byte count and status. TLS errors propagate
        to the existing bounded-transport-v2 classifier.
        """

        def __init__(self, key, emit, lease_remaining, timeout=180.0, connect=socket.create_connection, context_factory=ssl.create_default_context, clock=time.monotonic):
            self.key, self.emit, self.remaining = (key, emit, lease_remaining)
            self.timeout, self.connect, self.context_factory, self.clock = (timeout, connect, context_factory, clock)

        def count_input(self, wire, request_id):
            return self(wire, request_id, endpoint='/v1/responses/input_tokens')

        def __call__(self, wire, request_id, *, endpoint='/v1/responses'):
            if endpoint not in ('/v1/responses', '/v1/responses/input_tokens'):
                raise ValueError('unsupported provider endpoint')
            deadline = self.clock() + min(self.timeout, self.remaining())

            def timeout():
                left = min(deadline - self.clock(), self.remaining())
                if left <= 0:
                    raise LeaseExpired('request lease expired')
                return left

            def phase(name, operation, size=None):
                start = self.clock()
                row = {'phase': name, 'status': 'started'}
                self.emit(row)
                try:
                    value = operation()
                except BaseException:
                    self.emit({'phase': name, 'status': 'failed', 'seconds': self.clock() - start})
                    raise
                row = {'phase': name, 'status': 'completed', 'seconds': self.clock() - start}
                if size is not None:
                    row['bytes'] = size(value) if callable(size) else size
                self.emit(row)
                return value
            sock = None
            conn = None
            send_started = False
            try:
                sock = phase('tcp', lambda: self.connect(('api.openai.com', 443), timeout()))
                sock.settimeout(timeout())
                sock = phase('tls', lambda: self.context_factory().wrap_socket(sock, server_hostname='api.openai.com'))
                conn = http.client.HTTPConnection('api.openai.com', timeout=timeout())
                conn.sock = sock

                def send():
                    sock.settimeout(timeout())
                    conn.request('POST', endpoint, body=wire, headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json', 'X-Client-Request-Id': request_id, 'Connection': 'close'})
                send_started = True  # Even a partial POST is outcome-unknown.
                phase('send', send, len(wire))
                sock.settimeout(timeout())
                response = phase('response_headers', conn.getresponse)
                self.emit({'phase': 'http_status', 'status_code': response.status})
                chunks = []

                def read():
                    total = 0
                    while not response.isclosed():
                        sock.settimeout(timeout())
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > 32 * 1024 * 1024:
                            raise ValueError('response body exceeds transport bound')
                    return b''.join(chunks)
                raw = phase('response_body', read, len)
                if response.status != 200:
                    error=HTTPError('https://api.openai.com'+endpoint, response.status, 'provider status', {}, None)
                    try:
                        retain_http_error_response(folder,request_id,response.status,raw,response.getheader('x-request-id'))
                    except Exception as secondary:
                        # Keep HTTP status classification and retry policy unchanged.
                        error.diagnostic_error=repr(secondary)
                    raise error
                return raw
            except Exception as error:
                if connection_retries and not send_started and transient_before_send(error):
                    raise pick.RequestNotSent('transient connection failure before HTTP send') from None
                raise
            finally:
                if conn is not None:
                    conn.close()
                elif sock is not None:
                    sock.close()

    path=Path(folder)/'transport-phases.jsonl'
    def emit(value):
        with path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(value)+'\n')
    transport = PhaseTransport(load_runtime_key(Path.home()/'.config/bvi/openai.env'),emit,remaining,timeout=180.)
    if not connection_retries:
        return transport
    def guarded_send(wire, key):
        return send_with_connection_retries(transport,wire,key,remaining,emit,retries=connection_retries)
    guarded_send.count_input = transport.count_input  # Counting never retries generation.
    return guarded_send


# Offline evaluator: archived outcomes are playback, never physical evidence.
def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def differences(a, b, path='$'):
    """Field paths, without dumping images or provider content."""
    if type(a) != type(b):
        return [path]
    if isinstance(a, dict):
        return [p for k in sorted(a.keys() | b.keys()) for p in ([path + '.' + k] if k not in a or k not in b else differences(a[k], b[k], path + '.' + k))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + '.length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, f'{path}[{i}]')]
    if a == b:
        return []
    if isinstance(a, str) and path.endswith('.text'):
        try:
            return differences(json.loads(a), json.loads(b), path + '(json)')
        except (ValueError, TypeError):
            pass
    return [path]

def normalized(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [normalized(x) for x in value]
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    return value

def feedback_differences(archived, regenerated, paths):
    values = []
    for path in paths:
        parts = re.findall('[^.\\[\\]]+', path.removeprefix('$.'))
        a, b = (archived, regenerated)
        try:
            for key in parts:
                a = a[int(key)] if isinstance(a, list) else a[key]
                b = b[int(key)] if isinstance(b, list) else b[key]
        except (KeyError, IndexError, TypeError):
            continue
        row = dict(field=path, archived=a, regenerated=b)
        if type(a) in (float, int) and type(b) in (float, int):
            row['absolute_difference'] = abs(a - b)
        values.append(row)
    return values

class RecordingExecutor:
    """Replay only archived outcomes; record exact method arguments, no physics."""

    def __init__(self, command):
        self.steps = command['start_step']
        self.max_steps = 600
        self.trace = []
        self.calls = []
        self.command = command

    def read_qpos(self):
        return None

    def __getattr__(self, method):
        if method not in ('look', 'move_base', 'check_path', 'move_to', 'move_eef_chunk', 'set_gripper', 'return_to_rest'):
            raise AttributeError(method)

        def execute(*args, **kwargs):
            self.calls.append(dict(method=method, args=normalized(args), kwargs=normalized(kwargs)))
            result = copy.deepcopy(self.command['result'])
            before = self.steps
            self.steps += result.get('steps_used', 0)
            self.trace = [dict(kind='base_settle', step=before + i + 1) for i in range(result.get('settle_steps', 0))]
            return {k: v for k, v in result.items() if k not in ('steps_used', 'steps_remaining', 'motion_steps', 'settle_steps')}
        return execute

class RecordedDepth:
    """Sensor-output playback, not regenerated depth or invented geometry."""

    def __init__(self, command, observation_id):
        self.command = command
        self.calls = []
        self.frame = (observation_id,)

    def locate(self, camera, pixel, observation_id, measured):
        self.calls.append(dict(camera=camera, pixel=pixel, observation_id=observation_id))
        result = self.command['result']
        if 'samples' in result:
            sample = result['samples'][len(self.calls) - 1]
            if sample['pixel'] != pixel:
                raise ValueError('sensor query differs from archive')
            return {k: copy.deepcopy(v) for k, v in sample.items() if k != 'pixel'}
        return {k: copy.deepcopy(v) for k, v in result.items() if k not in ('steps_used', 'steps_remaining', 'motion_steps', 'settle_steps')}

def expected_commands(call):
    if not call or call['tool'] in ('done', 'give_up', 'recall_observation', 'locate_point', 'measure'):
        return []
    tool = call['tool']
    kwargs = {}
    if tool == 'look':
        args = [call['pan_rad'], call['tilt_rad']]
    elif tool == 'move_base':
        args = [call['turn_rad'], call['forward_m']]
    elif tool == 'set_gripper':
        args = [call['gripper']]
    elif tool == 'return_to_rest':
        args = []
    else:
        goals = [pick.pose(p['position'], p['quaternion_xyzw']).tolist() for p in call['poses']]
        args = [goals[0] if tool == 'move_to' else goals]
        if tool in ('move_to', 'move_eef_chunk'):
            kwargs = dict(max_steps=150)
    return [dict(method=tool, args=args, kwargs=kwargs)]

def replay(label, batch, profile, cache):
    rows = []
    ledger = read(batch / 'api-ledger.json')
    request_hashes = {entry['request_id']: entry['wire_sha256'] for entry in ledger['entries']}
    for case in sorted((cache).glob('arm-dev-*')):
        result = read(batch / 'data' / case.name / 'V-mobile/result.json')
        state = pick.State()
        for folder in sorted((case / 'V-mobile/provider').glob('turn-*')):
            row = dict(profile=label, case=case.name, turn=int(folder.name[-3:]), errors=[])
            rows.append(row)
            try:
                bundle = read(folder / 'request-bundle.json')
                archived = (folder / 'request-wire.json').read_bytes()
                regenerated = pick.build_wire(bundle, state.archive, profile)
                row.update(observation_id=bundle['observation']['observation_id'], archived_sha256=hashlib.sha256(archived).hexdigest(), regenerated_sha256=hashlib.sha256(regenerated).hexdigest(), request_equal=archived == regenerated, request_difference_fields=differences(json.loads(archived), json.loads(regenerated)))
                request_id = f"{case.name}-V-mobile-tool-{row['turn']:03d}"
                row['ledger_sha256'] = request_hashes[request_id]
                row['ledger_hash_equal'] = row['archived_sha256'] == row['ledger_sha256']
                row['request_equal'] = row['request_equal'] and row['ledger_hash_equal']
                delivery = pick.parse_response((folder / 'response-raw.json').read_bytes(), pick.estimate_wire(archived), profile=profile)
                stored = read(folder / 'delivery.json')
                row['parse_equal'] = delivery == stored
                row['parse_difference_fields'] = differences(stored, delivery)
                command = result['commands'][row['turn']]
                executor = RecordingExecutor(command)
                depth = RecordedDepth(command, bundle['observation']['observation_id'])
                traces = []
                actual, feedback = pick.advance(state, bundle, delivery, executor, depth, profile, sink=traces.append)
                row['call_equal'] = actual == command['call']
                row['feedback_equal'] = feedback == command['result']
                row['feedback_difference_fields'] = differences(command['result'], feedback)
                row['feedback_difference_values'] = feedback_differences(command['result'], feedback, row['feedback_difference_fields'])
                if delivery['call'] is not None:
                    pass
                expected = expected_commands(command['call'])
                row.update(executor_commands=executor.calls, expected_executor_commands=expected, commands_equal=executor.calls == expected, sensor_queries=depth.calls, trace=traces)
                call = command['call']
                expected_queries = []
                if call and call['tool'] in ('measure', 'locate_point'):
                    pixels = call['pixels'] if call['tool'] == 'measure' else [call['pixel']]
                    expected_queries = [dict(camera=call['camera'], pixel=p, observation_id=bundle['observation']['observation_id'] if profile.measure else call['observation_id']) for p in pixels]
                row['sensor_queries_equal'] = depth.calls == expected_queries
                row['dispatch_equal'] = all((row[k] for k in ('parse_equal', 'call_equal', 'feedback_equal', 'commands_equal', 'sensor_queries_equal')))
            except Exception as error:
                row['errors'].append(dict(type=type(error).__name__, reason=str(error)))
                row.setdefault('request_equal', False)
                row.setdefault('dispatch_equal', False)
    return rows

def verify_archive_cache(archive, cache):
    names={'request-bundle.json','request-wire.json','response-raw.json','delivery.json'}
    with tarfile.open(archive) as stream:
        members=stream.getmembers()
        index={m.name:m for m in members}
        selected=[m for m in members if '/provider/' in m.name and Path(m.name).name in names]
        def target(member):
            seen=set()
            while member.islnk():
                if member.name in seen: raise ValueError('cyclic archive link')
                seen.add(member.name)
                member=index[member.linkname]
            return member.name
        needed={target(m) for m in selected}
    hashes={}
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            if member.name in needed and member.isfile():
                hashes[member.name]=hashlib.sha256(stream.extractfile(member).read()).hexdigest()
    rows=[]
    for member in selected:
        relative=Path(*Path(member.name).parts[1:])
        path=(cache/relative).resolve()
        if not path.is_relative_to(cache.resolve()): raise ValueError('archive path escape')
        expected=hashes[target(member)]
        actual=sha(path)
        rows.append(dict(path=relative.as_posix(),archive_sha256=expected,cache_sha256=actual))
        if actual!=expected: raise ValueError('cached archive differs: '+relative.as_posix())
    if not selected: raise ValueError('archive has no provider materials')
    return dict(archive=str(archive),sha256=sha(archive),files=rows)

def archive_check(profile, root, cache, output):
    """Replay cached archive payloads, recording every strict difference."""
    provenance=verify_archive_cache(root/'raw-run.tar.gz',cache)
    save(output/'archive-sources.json',provenance)
    rows = replay(profile, root, pick.PROFILES[profile], cache)
    expected = 93 if profile == 'C0' else 96
    summary = dict(profile=profile, expected=expected, found=len(rows),
        G1=sum(r.get('request_equal',False) for r in rows),
        G2=sum(r.get('dispatch_equal',False) for r in rows),
        parser=sum(r.get('parse_equal',False) for r in rows),
        commands=sum(r.get('commands_equal',False) for r in rows),
        sensors=sum(r.get('sensor_queries_equal',False) for r in rows),
        unreplayable=[r for r in rows if r['errors']],
        API=0, simulation=0, GPU=0, lab=0,
        playback='Executor outcomes and sensor samples are archived playback; geometry is recomputed.',
        accepted=len(rows)==expected and all(r.get('request_equal') and r.get('dispatch_equal') for r in rows))
    save(output/'request-dispatch-parity.json', rows)
    save(output/'feedback-differences.json', [dict(case=r['case'],turn=r['turn'],**v)
        for r in rows for v in r.get('feedback_difference_values',[])])
    save(output/'summary.json', summary)
    return summary


def unique_storage_bytes(root):
    """Count regular-file bytes once per device/inode, including media hardlinks."""
    seen = set(); total = 0
    for path in Path(root).rglob('*'):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
        except FileNotFoundError:  # atomic receipt replacement during traversal
            continue
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            seen.add(key); total += stat.st_size
    return total


def supervision_budget(auth, prior):
    """Explicit cumulative consumption, never inferred from a historical constant."""
    limits = dict(gpu1_process_seconds_cap=36000, worker_seconds_cap=1800,
                  cpu_media_seconds_cap=12000, media_worker_seconds_cap=600,
                  output_bytes_cap=12*1024**3)
    if any(type(auth.get(k)) not in (int, float) or not 0 < auth[k] <= v for k,v in limits.items()):
        raise ValueError('missing or expanded supervision limits')
    if not isinstance(prior.get('evidence'), list) or not prior['evidence']:
        raise ValueError('explicit prior budget evidence required, including a fresh-zero declaration')
    consumed = {}
    for name in ('gpu1_process_seconds', 'cpu_media_seconds'):
        value = prior.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid prior '+name)
        consumed[name] = value
    worker = min(auth['worker_seconds_cap'], auth['gpu1_process_seconds_cap']-consumed['gpu1_process_seconds'])
    media = min(auth['media_worker_seconds_cap'], auth['cpu_media_seconds_cap']-consumed['cpu_media_seconds'])
    if worker <= 0 or media <= 0:
        raise ValueError('cumulative supervision budget exhausted')
    return dict(worker_seconds=worker, media_seconds=media, prior=copy.deepcopy(prior))


def media_readiness(folder, media_exit_code):
    """A successful process alone is not a complete retained recording."""
    folder = Path(folder); errors = []
    try:
        if media_exit_code != 0:
            raise ValueError('media process failed or was not run')
        result_path = folder/'result.json'; result = read(result_path)
        receipt = read(folder/'media-receipt.json')
        if receipt.get('complete') is not True or receipt.get('errors'):
            raise ValueError('media receipt incomplete')
        if receipt.get('raw_result_sha256') != sha(result_path):
            raise ValueError('raw result hash mismatch')
        out = folder/'media-source'/'delivery'
        required = {'history.html'}
        frames = result.get('frames')
        if not isinstance(frames, list):
            raise ValueError('original frame list missing')
        if frames:
            if [frame.get('step') for frame in frames] != list(range(1,len(frames)+1)):
                raise ValueError('original trajectory step sequence mismatch')
            for step in range(len(frames)+1):
                for name in ('robot-rgbd.npz','state.eval-only.pt','contacts.eval-only.json'):
                    if not (folder/'trajectory'/f'step-{step:04d}'/name).is_file():
                        raise ValueError('detailed trajectory file missing')
            required.update(('trajectory.svg','trajectory.html','online-raw.mp4','online-tcp-demo.mp4',
                'online-command-vs-executed.mp4','astra-commanded-vs-executed-3d-demo.mp4',
                'demo-receipt.json','command-overlay-receipt.json','dual-path-receipt.json'))
            recorded = sorted((folder/'recording-frames').glob('*.png'))
            if [p.name for p in recorded] != [f'{i:06d}.png' for i in range(len(frames)+1)]:
                raise ValueError('recording frame count or sequence mismatch')
            if read(out/'dual-path-receipt.json').get('frames') != len(frames)+1:
                raise ValueError('rendered frame count mismatch')
            required.update(('execution-rgbd.mp4','execution-rgbd.json'))
            rgbd = read(out/'execution-rgbd.json')
            if (receipt.get('preferred_demo') != 'execution-rgbd.mp4' or
                receipt.get('video_format') != 'execution-rgbd-v1' or
                rgbd.get('format') != 'execution-rgbd-v1' or
                rgbd.get('frames') != len(frames)+1 or rgbd.get('resolution') != [1720,800] or
                rgbd.get('depth_frames') != dict(head=len(frames)+1, hand=len(frames)+1) or
                rgbd.get('sha256') != sha(out/'execution-rgbd.mp4') or
                rgbd.get('source_result_sha256') != sha(out.parent/'result.json')):
                raise ValueError('required default RGB-D format/coverage/hash mismatch')
        elif receipt.get('no_motion') is not True:
            raise ValueError('empty trajectory lacks no-motion receipt')
        artifacts = receipt.get('artifacts', {})
        for name in required:
            if name not in artifacts or not (out/name).is_file() or sha(out/name) != artifacts[name].get('sha256'):
                raise ValueError('missing or changed required artifact: '+name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return dict(ready=not errors, errors=errors)


def validate_automatic_attempt(receipt, auth, case, root, ledger):
    """The last case has the same completion checks as every predecessor."""
    if (receipt.get('batch_id') != auth.get('batch_id') or receipt.get('case_id') != case
            or receipt.get('status') != 'ready_for_review' or type(receipt.get('attempt_index')) is not int
            or receipt['attempt_index'] != 0 or receipt.get('worker_forced_stop') or receipt.get('media_forced_stop')
            or receipt.get('exit_code') != 0 or receipt.get('media_exit_code') != 0):
        raise ValueError('automatic attempt stopped or identity incomplete')
    result_path = Path(receipt['result_path'])
    if result_path.resolve() != (Path(root)/case/'V-mobile'/'result.json').resolve():
        raise ValueError('automatic result outside the frozen case output')
    if sha(result_path) != receipt.get('result_sha256'):
        raise ValueError('automatic result changed after receipt')
    result = read(result_path)
    if receipt.get('replay_initial_anchor')!=auth.get('replay_initial_anchor') or result.get('replay_initial_anchor')!=auth.get('replay_initial_anchor'):
        raise ValueError('supervised initial-anchor protocol mismatch')
    if not (receipt.get('initialization_observability') == result.get('initialization_observability') == auth.get('initialization_observability')):
        raise ValueError('supervised initialization observability protocol mismatch')
    if 'initialization_observability' in auth:
        if auth.get('stage') == 'L4':
            pick.validate_initialization_observability_result(auth, result, result_path.parent)
        elif auth.get('stage') == 'L3':
            if result.get('initialization_capture_active') is not False or 'initialization_boundaries_eval_only' in result:
                raise ValueError('L3 initialization option is passive only')
    elif any(k in result for k in ('initialization_observability', 'initialization_capture_active', 'initialization_boundaries_eval_only')):
        raise ValueError('initialization evidence without authorization option')
    if auth.get('scope')==pick.R33_STAGED_SCOPE and auth.get('stage')=='L4' and 'replay_initial_anchor' in auth:
        pick.validate_final_anchor_attempt(auth,result,case,result_path.parent)
    if (result.get('status') != 'completed' or result.get('batch_id') != auth.get('batch_id')
            or result.get('case_id') != case or result.get('attempt_index') != 0 or result.get('close_error_type')):
        raise ValueError('automatic continuation stops on budget or infrastructure interruption')
    if not media_readiness(result_path.parent, receipt['media_exit_code'])['ready']:
        raise ValueError('prior media incomplete')
    expected = request_accounting(ledger, case, prefix=auth['batch_id']+'-attempt-0-')
    actual = result.get('api_request_accounting', {})
    for field in ('request_slots', 'main_slots', 'verification_slots', 'known_not_sent', 'outcome_unknown'):
        if actual.get(field) != expected[field]:
            raise ValueError('result and ledger request accounting differ')
    try:
        if pick.Decimal(actual.get('committed_usd', 'NaN')) != pick.Decimal(expected['committed_usd']):
            raise ValueError('result and ledger cost accounting differ')
    except ArithmeticError:
        raise ValueError('result monetary accounting incomplete') from None
    if result.get('api_requests') != expected['request_slots'] or result.get('verification_requests') != expected['verification_slots']:
        raise ValueError('result and ledger request counts differ')
    return result


def validate_continuation(root, auth, case, review_path, *, automatic=False, ledger_path=None, authorization_sha256=None):
    index = auth['case_ids'].index(case)
    if index == 0:
        return
    if automatic:
        if ledger_path is None:
            raise ValueError('automatic continuation requires ledger')
        ledger = automatic_ledger_check(ledger_path, authorization_sha256=authorization_sha256, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
        review = dict(batch_id=auth.get('batch_id'), action='continue', next_case_id=case,
                      review_evidence='automatic checks; not human review', receipt_sha256={})
    elif review_path is None:
        raise ValueError('explicit continuation review required')
    else:
        review = read(review_path)
    if review.get('batch_id') != auth.get('batch_id') or review.get('action') != 'continue' or review.get('next_case_id') != case or not review.get('review_evidence'):
        raise ValueError('continuation review identity mismatch')
    for previous in auth['case_ids'][:index]:
        receipts = sorted(Path(root).glob(previous+'-attempt-*-supervision.json'))
        if not receipts:
            raise ValueError('prior case missing; frozen order cannot skip')
        path = receipts[-1]; receipt = read(path)
        if automatic:
            validate_automatic_attempt(receipt, auth, previous, root, ledger)
            review['receipt_sha256'][path.name] = sha(path)
        if receipt.get('batch_id') != auth.get('batch_id') or receipt.get('status') != 'ready_for_review' or review.get('receipt_sha256',{}).get(path.name) != sha(path):
            raise ValueError('prior media or continuation approval incomplete')
        result = read(Path(receipt['result_path']))
        if sha(Path(receipt['result_path'])) != receipt.get('result_sha256'):
            raise ValueError('prior result changed after review')
        if result.get('status') != 'completed' and not (result.get('status') == 'budget_exhausted' and result.get('budget_scope') == 'case'):
            raise ValueError('prior infrastructure/system interruption requires separate recovery')
    if automatic:
        save(Path(root)/(case+'-automatic-continuation.json'), dict(review, mode='automatic'))


def supervise_attempt(args):
    """One authorized Linux worker and bounded CPU finalizer; no retries."""
    auth = read(args.authorization); profile = pick.PROFILES[args.profile]
    validate(auth, args, worker=False)
    import subprocess
    import signal
    import fcntl
    validate_candidate_sources(auth, args.code)
    verify_c2_assets(auth)
    if sha(args.roster) != auth['roster_sha256'] or args.case_id not in auth['case_ids']:
        raise ValueError('supervised roster mismatch')
    root = args.ledger.parent
    root.mkdir(parents=True, exist_ok=True)
    if profile.model_target_boxes and root.resolve() != Path(auth['output_root']).resolve():
        raise ValueError('r25 ledger and storage accounting must use frozen batch output root')
    prior = read(args.prior_budget)
    if prior.get('batch_id') != auth.get('batch_id') or prior.get('authorization_sha256') != sha(args.authorization):
        raise ValueError('prior budget must bind exact batch and authorization')
    lock_path = args.lab_root/'runs'/'arm-capability.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_continuation(root,auth,args.case_id,getattr(args,'continuation_review',None),
                              automatic=profile.automatic_supervision, ledger_path=args.ledger,
                              authorization_sha256=sha(args.authorization))
        if profile.model_target_boxes and batch_physical_steps(root,auth['batch_id']) >= auth['physical_steps_cap']:
            raise ValueError('cumulative physical budget exhausted before worker launch')
        for receipt_path in root.glob('*-supervision.json'):
            receipt = read(receipt_path)
            if receipt.get('status') in ('running', 'worker_finished'):
                raise ValueError('unfinished prior supervision needs explicit reconciliation')
            for field in ('gpu1_process_seconds','cpu_media_seconds'):
                value = receipt.get(field)
                if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                    raise ValueError('prior receipt has no complete resource accounting')
                prior[field] += value
        budget = supervision_budget(auth, prior)
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
        if GPU in gpu:
            raise ValueError('GPU1 occupied')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError('attempt output already exists; never rerun implicitly')
        retry = auth.get('infrastructure_retry')
        attempt_index = 1 if retry and retry['case_id'] == args.case_id else 0
        attempt_name = args.case_id+f'-attempt-{attempt_index}'
        with (root/(attempt_name+'-started.lock')).open('x') as started_lock:
            started_lock.write(str(os.getpid()))
        receipt_path = root/(attempt_name+'-supervision.json')
        if not args.ledger.exists():
            save(args.ledger, dict(authorization_sha256=sha(args.authorization),status='running',entries=[]))
        else:
            ledger = read(args.ledger)
            if ledger.get('authorization_sha256') != sha(args.authorization) or ledger.get('status') != 'running':
                raise ValueError('ledger needs explicit authorization/accounting reconciliation before launch')
            if profile.model_target_boxes and any(not entry['request_id'].startswith(auth['batch_id']+'-') for entry in ledger['entries']):
                raise ValueError('old batch ledger cannot transfer')
        if profile.automatic_supervision:
            automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=auth['case_ids'])
        state = dict(status='running', case_id=args.case_id, attempt_index=attempt_index, batch_id=auth.get('batch_id'), budget=budget, started_at_epoch=time.time())
        if 'replay_initial_anchor' in auth:
            state['replay_initial_anchor']=auth['replay_initial_anchor']
        if 'initialization_observability' in auth:
            state['initialization_observability'] = copy.deepcopy(auth['initialization_observability'])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(args.code/'src'),str(args.code/'scripts'))),
                   CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        command = [sys.executable,'-m','astra_pick.run_case','--mode','live','--profile',args.profile]
        for name in ('authorization','roster','code','lab_root','output','ledger','case_id'):
            command += ['--'+name.replace('_','-'),str(getattr(args,name))]
        def execute(command, cap, label):
            start = time.monotonic()
            with (root/(attempt_name+'-'+label+'.log')).open('w') as log:
                process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    state[label+'_pid'] = process.pid; save(receipt_path,state)
                    while process.poll() is None:
                        if time.monotonic()-start >= cap or unique_storage_bytes(root) > auth['output_bytes_cap']:
                            state[label+'_forced_stop'] = 'wall_time_or_storage_cap'
                            break
                        time.sleep(.25)
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # process exited between poll and kill
                        process.wait()
            return process.returncode, time.monotonic()-start
        save(receipt_path,state)
        state['exit_code'],state['gpu1_process_seconds'] = execute(command,budget['worker_seconds'],'worker')
        state['status'] = 'worker_finished'; save(receipt_path,state)
        state['media_exit_code'] = None; state['cpu_media_seconds'] = 0
        if (args.output/'result.json').is_file():
            env['CUDA_VISIBLE_DEVICES'] = ''
            command = [sys.executable,'-c','import sys; from finalize_arm_capability_attempt import finalize; r=finalize(sys.argv[1]); sys.exit(0 if r["complete"] else 1)',str(args.output)]
            state['media_exit_code'],state['cpu_media_seconds'] = execute(command,budget['media_seconds'],'media')
        state['media'] = media_readiness(args.output,state['media_exit_code'])
        state['status'] = 'ready_for_review' if state['media']['ready'] else 'media_failed'
        if (args.output/'result.json').is_file():
            state.update(result_path=str((args.output/'result.json').resolve()),result_sha256=sha(args.output/'result.json'))
        entries = read(args.ledger)['entries']
        case_entries = [e for e in entries if e.get('case_id') == args.case_id]
        waits = [read(p)['operator_wait_seconds'] for p in (args.output/'supervision').glob('*.wait.json')]
        state['wait_accounting'] = dict(api_wait_seconds=sum(e.get('api_wait_seconds',0) for e in case_entries),
            operator_wait_seconds=sum(waits),
            breakdown_complete=all('api_wait_seconds' in e for e in case_entries) and 'worker_forced_stop' not in state,
            note='API and operator waiting remains included in worker wall time; forced interruption can truncate the breakdown.')
        state['storage_bytes'] = unique_storage_bytes(root)
        save(receipt_path,state)
        return state


def l4_stage1_check(root, auth, ledger_path, authorization_path):
    """Revalidate all first-nine artifacts before admitting any second-stage case."""
    first = [f'baseline-{i:03d}' for i in (2, 3, 6, 11, 12, 15, 17, 18, 5)]
    if auth.get('scope') == R31_STAGED_SCOPE:
        pick.validate_r31_paid_scope(auth, pick.C2_R31)
    if auth.get('scope') == pick.R33_STAGED_SCOPE:
        pick.validate_r33_paid_scope(auth, pick.C2_R33)
    if (auth.get('scope') not in (pick.R3_SCOPE, R31_STAGED_SCOPE, pick.R33_STAGED_SCOPE) or auth.get('stage') != 'L4'
            or auth.get('case_ids', [])[:9] != first or auth.get('stage1_cases') != first
            or auth.get('stage1_force_limit_stop_count') != 4):
        raise ValueError('L4 first-stage frozen order or stop threshold mismatch')
    root = Path(root)
    ledger = automatic_ledger_check(ledger_path, authorization_path=authorization_path,
                                   batch_id=auth['batch_id'], case_ids=auth['case_ids'])
    receipt_hashes, result_hashes, outcomes = {}, {}, {}
    for case in first:
        paths = list(root.glob(case+'-attempt-*-supervision.json'))
        expected = root/(case+'-attempt-0-supervision.json')
        if paths != [expected]:
            raise ValueError('L4 first stage needs exactly one original supervised receipt per case')
        result = validate_automatic_attempt(read(expected), auth, case, root, ledger)
        category = result.get('termination_category')
        if category not in ('official_success', 'cumulative_force_limit', 'time_limit', 'tool_call_limit', 'done', 'give_up'):
            raise ValueError('L4 first-stage outcome incomplete')
        receipt_hashes[expected.name] = sha(expected)
        result_hashes[case] = sha(root/case/'V-mobile'/'result.json')
        outcomes[case] = category
    count = sum(outcomes[case] == 'cumulative_force_limit' for case in first[:8])
    gate = dict(stage='L4-stage1', batch_id=auth['batch_id'], authorization_sha256=sha(authorization_path),
                status='passed' if count < 4 else 'stopped', formerly_force_limit_count=count,
                counted_cases=first[:8], success_control_case=first[8], outcomes=outcomes,
                receipt_sha256=receipt_hashes, result_sha256=result_hashes)
    path = root/'L4-stage1-gate.json'
    if path.exists() and read(path) != gate:
        raise ValueError('L4 first-stage evidence changed after its bound gate receipt')
    if not path.exists():
        save(path, gate)
    return gate


def automatic_batch(args, *, attempt=None):
    """Ordered CPU-testable orchestration; each attempt rechecks authorization."""
    attempt = attempt or supervise_attempt
    auth = read(args.authorization)
    profiles = auth.get('case_profiles', {})
    cases = auth['case_ids']
    r31 = auth.get('scope') == R31_STAGED_SCOPE
    r33 = auth.get('scope') == pick.R33_STAGED_SCOPE
    allowed_profiles = ('C2-r3.3', 'C2-r3.3-far') if r33 else ('C2-r3.1', 'C2-r3.1-far') if r31 else ('C2-r3', 'C2-r3-far')
    if set(profiles) != set(cases) or any(p not in allowed_profiles for p in profiles.values()):
        raise ValueError('automatic batch requires explicit per-case profiles')
    if r31 or r33:
        for case in cases:
            validator = pick.validate_r33_paid_scope if r33 else pick.validate_r31_paid_scope
            validator(auth, pick.PROFILES[profiles[case]], case_id=case)
        if auth.get('stage') not in ('L3', 'L4'):
            raise ValueError(('r33' if r33 else 'r31')+' automatic physical batch requires L3 or L4')
    if auth.get('infrastructure_retry') or auth.get('recovery_budget'):
        raise ValueError('automatic batch cannot retry or recover a truncated attempt')
    if args.output != args.ledger.parent/args.case_id/'V-mobile':
        raise ValueError('automatic batch output must be ledger-root/case/V-mobile')
    receipts = []
    for case in cases[cases.index(args.case_id):]:
        if auth.get('stage') == 'L4' and cases.index(case) >= 9:
            try:
                gate = l4_stage1_check(args.ledger.parent, auth, args.ledger, args.authorization)
            except (OSError, ValueError, KeyError, TypeError, pick.VerificationOutcomeUnknown) as error:
                return dict(status='automatic_stopped', receipts=receipts, stop_reason=str(error))
            if gate['status'] != 'passed':
                return dict(status='automatic_stopped', receipts=receipts,
                            stop_reason='L4 first eight former force-limit cases reached the four-case stop threshold',
                            stage1_gate=gate)
        if args.ledger.exists():
            try:
                automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=cases)
            except (OSError, ValueError, TypeError, pick.VerificationOutcomeUnknown) as error:
                return dict(status='automatic_stopped', receipts=receipts, stop_reason=str(error))
        current = copy.copy(args)
        current.case_id, current.profile = case, profiles[case]
        current.output = args.ledger.parent/case/'V-mobile'
        receipt = attempt(current)
        receipts.append(receipt)
        try:
            ledger = automatic_ledger_check(args.ledger, authorization_path=args.authorization, batch_id=auth['batch_id'], case_ids=cases)
            validate_automatic_attempt(receipt, auth, case, args.ledger.parent, ledger)
        except (OSError, ValueError, KeyError, TypeError, pick.VerificationOutcomeUnknown) as error:
            return dict(status='automatic_stopped', receipts=receipts, stop_reason=str(error))
    complete = len(receipts) == len(cases[cases.index(args.case_id):])
    return dict(status='ready_for_review' if complete else 'automatic_stopped', receipts=receipts)


def _initialization_process_group(pgid):
    """Read actual Linux process-group membership, never infer exit from a result file."""
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2])==pgid:
            members.append(int(entry.name))
    return sorted(members)


def supervise_initialization_only(args):
    """Exactly two serial fresh workers; lock and independent GNU timeout survive parent exit."""
    import subprocess
    import signal
    import fcntl
    import shutil
    started=time.monotonic()
    auth=read(args.authorization)
    pick.validate_init_only_scope(auth,pick.PROFILES[args.profile],case_id=args.case_id,observation='init-a')
    validate_candidate_sources(auth,args.code)
    if any(sha(args.code/name)!=digest for name,digest in auth['source_sha256'].items()):
        raise ValueError('initialization-only supervisor source21 changed')
    if os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or not auth.get('scheduling_clearance_confirmed'):
        raise ValueError('initialization-only supervisor lab/device/clearance')
    root=Path(auth['output_root'])
    if args.output.resolve()!=root.resolve() or root.exists():
        raise ValueError('initialization-only fresh output root required')
    if sha(args.roster)!=auth['roster_sha256'] or Path(auth['roster_path']).resolve()!=args.roster.resolve():
        raise ValueError('initialization-only supervisor roster')
    timeout=shutil.which('timeout')
    if not timeout or not subprocess.check_output([timeout,'--version'],text=True).startswith('timeout (GNU coreutils)'):
        raise ValueError('initialization-only independent GNU timeout required')
    lock_path=args.lab_root/'runs/arm-capability.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if GPU in active:
            raise ValueError('initialization-only approved GPU occupied')
        root.mkdir(parents=True,exist_ok=False)
        def durable(path,value):
            with path.open('x',encoding='utf8') as stream:
                json.dump(value,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        durable(root/'batch-start.intent.json',dict(batch_id=auth['batch_id'],source_manifest_sha256=auth['source_manifest_sha256'],authorization_sha256=sha(args.authorization)))
        completed=[];total=0.;status='infrastructure_censored';cause=None;unknown=False;attempts=[]
        env=os.environ.copy();env.update(PYTHONPATH=str(args.code/'src'),PYTHONUNBUFFERED='1')
        lease=pick.r3_bound_json(auth['resource_lease'])
        def resource_check():
            nonworker=time.monotonic()-started-total
            if total>1200 or nonworker>600 or time.time()>=lease['expires_at_epoch'] or unique_storage_bytes(root)>8*1024**3 or (root/'STOP').exists():
                raise ValueError('initialization-only wall/storage/lease/CPU-postprocess upper bound')
        try:
            for observation in pick.INIT_ONLY_CONTRACT['observations']:
                if pick.validate_init_only_sequence(auth,completed,authorization_path=args.authorization)!=observation:
                    raise ValueError('initialization-only sequence stopped')
                resource_check()
                output=root/args.case_id/observation
                previous=root/(observation+'-previous.json');durable(previous,completed)
                now=time.time()
                watchdog_cap=min(600.,1200.-total,lease['expires_at_epoch']-now)
                # Leave a bounded launch margin, and refuse rather than extend the deadline.
                watchdog_seconds=watchdog_cap-1.
                watchdog_deadline=now+watchdog_cap
                if not math.isfinite(watchdog_seconds) or watchdog_seconds<=0:
                    raise ValueError('initialization-only independent watchdog budget exhausted')
                command=[timeout,'--signal=KILL',f'{watchdog_seconds:.9f}s',sys.executable,'-m','astra_pick.run_case','--mode','init-only','--profile',args.profile,
                    '--initialization-observation',observation,'--initialization-previous',str(previous),
                    '--initialization-lock-fd',str(lock.fileno()),'--output',str(output)]
                for name in ('authorization','roster','code','lab_root','ledger','case_id'):
                    command+=['--'+name.replace('_','-'),str(getattr(args,name))]
                durable(root/(observation+'.launch-intent.json'),dict(command=command,observation=observation,
                    authorization_sha256=sha(args.authorization),source_manifest_sha256=auth['source_manifest_sha256'],automatic_retries=0,
                    watchdog_duration_seconds=watchdog_seconds,watchdog_deadline_epoch=watchdog_deadline,lease_expires_at_epoch=lease['expires_at_epoch']))
                process=None;forced=None;begin=time.monotonic();launch_error=None;secondary=[]
                try:
                    with (root/(observation+'.worker.log')).open('x',encoding='utf8') as log:
                        if time.time()+watchdog_seconds>watchdog_deadline:
                            raise ValueError('initialization-only watchdog launch deadline elapsed')
                        try:
                            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                        except Exception:
                            unknown=True
                            raise
                        durable(root/(observation+'.handle.json'),dict(pid=process.pid,pgid=process.pid,observation=observation,role='GNU_timeout_watchdog'))
                        while process.poll() is None:
                            elapsed=time.monotonic()-begin
                            if elapsed>=min(600,1200-total) or time.time()>=lease['expires_at_epoch'] or unique_storage_bytes(root)>8*1024**3 or (root/'STOP').exists():
                                forced='wall_storage_lease_or_operator_stop';break
                            time.sleep(.1)
                except Exception as error:
                    launch_error=error
                    raise
                finally:
                    try:
                        if process is not None:
                            if process.poll() is None:
                                try:os.killpg(process.pid,signal.SIGKILL)
                                except ProcessLookupError:pass
                            process.wait()
                            if _initialization_process_group(process.pid):
                                try:os.killpg(process.pid,signal.SIGKILL)
                                except ProcessLookupError:pass
                                forced=forced or 'surviving_process_group'
                    except Exception as error:
                        secondary.append(record_initialization_error(root,observation+'-group-close',error,primary=launch_error is None))
                        if launch_error is None:raise
                    finally:
                        elapsed=time.monotonic()-begin;total+=elapsed
                        accounting=dict(observation=observation,launch_attempt_wall_seconds=elapsed,
                            gpu_process_wall_seconds=elapsed if process is not None else None,
                            startup_outcome='known_pid' if process is not None else 'unknown',
                            pid=None if process is None else process.pid,secondary_errors=secondary)
                        attempts.append(accounting)
                        try:durable(root/(observation+'.attempt-accounting.json'),accounting)
                        except Exception as error:
                            if launch_error is None:raise
                            record_initialization_error(root,observation+'-accounting',error,primary=False)
                resource_check()
                if forced or elapsed>600 or secondary:
                    raise ValueError('initialization-only worker resource/termination fault: '+str(forced))
                result_path=output/'result.json'
                if process.returncode!=0 or not result_path.is_file():
                    raise ValueError('initialization-only worker failed or result unknown')
                if _initialization_process_group(process.pid):
                    raise ValueError('initialization-only process group not empty')
                result=read(result_path);claim=read(root/(observation+'.worker-entered.json'))
                if claim.get('pid')!=result.get('pid') or claim.get('pgid')!=process.pid:
                    raise ValueError('initialization-only actual worker/watchdog process binding')
                process_path=root/(observation+'.process.json')
                durable(process_path,dict(status='exited',pid=result['pid'],pgid=process.pid,exit_code=process.returncode,process_group=[],
                    wall_seconds=elapsed,result_sha256=sha(result_path)))
                completed.append(dict(result=dict(path=str(result_path),sha256=sha(result_path)),
                    process=dict(path=str(process_path),sha256=sha(process_path))))
                pick.validate_init_only_sequence(auth,completed,authorization_path=args.authorization)
                resource_check()
            status='initialization_observations_complete'
        except Exception as error:
            cause=record_initialization_error(root,'initialization_supervision',error,primary=True)
        finally:
            receipt=dict(scope=pick.R33_INIT_ONLY_SCOPE,status=status,batch_id=auth['batch_id'],
                source_manifest_sha256=auth['source_manifest_sha256'],completed=completed,error=cause,attempts=attempts,
                gpu_process_wall_seconds=None if unknown else total,known_launch_attempt_wall_seconds=total,
                startup_outcome_unknown=unknown,cpu_postprocess_wall_upper_bound=max(0.,time.monotonic()-started-total),
                exact_CPU_seconds=None,api_requests=0,explicit_control_steps=0,sdk_internal_steps=None,
                automatic_retries=0,physical_L1_passed=False)
            durable(root/'initialization-supervision.json',receipt)
        return receipt


def v3_cost_accounting(ledger, case_id=None):
    """Keep unbilled counting risk separate from generation usage and actual charges."""
    entries=[e for e in ledger.get('entries',[]) if case_id is None or e['case_id']==case_id]
    counts=[e for e in ledger.get('count_entries',[]) if case_id is None or e['case_id']==case_id]
    generation=sum((pick.Decimal(str(e['estimated_usd'])) for e in entries
                    if e.get('estimated_usd') is not None),pick.Decimal(0))
    actual=[e.get('actual_charge_usd') for e in counts]
    actual_total=(None if any(x is None for x in actual)
                  else str(sum((pick.Decimal(str(x)) for x in actual),pick.Decimal(0))))
    generation_committed=pick.committed_cost(entries)
    count_committed=pick.committed_cost(counts)
    return dict(generation_requests=len(entries),
        generation_usage_estimated_usd=str(generation),
        generation_committed_usd=str(generation_committed),
        input_count_requests=len(counts),
        input_count_completed=sum(e.get('status')=='valid' for e in counts),
        input_count_unresolved_delivery=sum(e.get('status') in ('reserved','unknown','censored') for e in counts),
        input_count_risk_reserved_usd=str(count_committed),input_count_actual_usd=actual_total,
        total_committed_usd=str(generation_committed+count_committed),
        all_endpoint_requests=len(entries)+len(counts),
        fee_basis='generation token usage estimate plus retained input-count risk reserve; not a provider bill',
        counting_policy='usd-0.01-per-count-risk-reserve-v1')


def pick20_summary(reports, ledger, case_ids, *, weekly=False):
    entries = ledger.get('entries', [])
    known = sum((pick.Decimal(str(e['estimated_usd'])) for e in entries if e.get('estimated_usd') is not None), pick.Decimal(0))
    result=dict(planned_cases=case_ids, strict_successes=sum(r.get('strict_pick_success') is True for r in reports),
        denominator=len(case_ids), completed=sum(r.get('status')=='completed' for r in reports),
        censored=sum(r.get('status')!='completed' for r in reports), not_started=len(case_ids)-len(reports),
        usage_estimated_usd=str(known), committed_usd=str(pick.committed_cost(entries)),
        unresolved_requests=sum(e.get('status') in ('unknown','reserved','censored') for e in entries),
        provider_requests=len(entries), physical_steps=sum(r.get('simulator_actions',0) for r in reports),
        cases=[{k:r.get(k) for k in ('case_id','status','strict_pick_success','termination_category','simulator_actions','wall_seconds','error_message')} for r in reports],
        billing_status='token usage estimate, not provider bill', lab_billed_amount=None)
    lean_reports=[r for r in reports if r.get('execution_profile') in tuple(p.name for p in pick.V3_PROFILES)
                  or r.get('information_condition') in tuple(p.name for p in pick.V3_PROFILES)]
    if lean_reports:
        result['v3_cost_accounting']=v3_cost_accounting(ledger)
        result['committed_usd']=result['v3_cost_accounting']['total_committed_usd']
        result['billing_status']='generation usage estimate; counting fee unbilled with retained risk reserve'
        classes=('success','task_failure','case_budget_failure','batch_budget_failure',
                 'infrastructure_failure','censored','pending')
        result['v3_outcome_counts']={name:sum(v3_outcome_class(r)==name for r in lean_reports)
                                     for name in classes}
        result['v3_outcome_denominator']=len(case_ids)
        result['v3_reported_outcomes']=len(lean_reports)
        result['v3_missing_outcomes']=len(case_ids)-len(lean_reports)
        result['budget_failure_scoring']='case_budget_failure is separate from task_failure; planned-case denominator unchanged'
        for source,row in zip(reports,result['cases']):
            if source.get('execution_profile') in tuple(p.name for p in pick.V3_PROFILES) or source.get('information_condition') in tuple(p.name for p in pick.V3_PROFILES):
                row['outcome_class']=v3_outcome_class(source)
                row['cost_accounting']=v3_cost_accounting(ledger,source['case_id'])
                if 'budget_stop' in source:row['budget_stop']=copy.deepcopy(source['budget_stop'])
                if 'budget_scope' in source:row['budget_scope']=source['budget_scope']
    if weekly:
        accounted=[r for r in reports if 'weekly_accounting' in r]
        recovered=[r for r in accounted if r['weekly_accounting'].get('recovery_event')]
        counts={key:sum(r['weekly_accounting'].get(key) is True for r in accounted)
            for key in ('target_confirmed_by_model','grasp_localization_executed','close_command_completed')}
        result['weekly']=dict(stage_counts=counts,stage_denominator=len(accounted),
            missing_accounting=len(reports)-len(accounted),
            stage_rates={k:v/len(accounted) if accounted else None for k,v in counts.items()},
            force_terminations=sum(r.get('termination_category')=='cumulative_force_limit' for r in reports),
            false_done_claims=sum(r.get('termination_category')=='done' and r.get('strict_pick_success') is False for r in reports),
            recovery_episodes=len(recovered),recovered_strict_successes=sum(r.get('strict_pick_success') is True for r in recovered),
            recovery_conditional_success_rate=(sum(r.get('strict_pick_success') is True for r in recovered)/len(recovered) if recovered else None),
            recovery_causal_effect='not identified by conditional rate; compare frozen conditions',
            per_case=[dict(case_id=r['case_id'],accounting=r['weekly_accounting']) for r in accounted])
    return result


def wait_pick20_idle(read_gpu, record, *, clock=time.monotonic, sleep=time.sleep):
    """Read-only release observation, never a worker/API retry or GPU intervention."""
    started=clock(); samples=[]; consecutive=0
    while True:
        raw=read_gpu()
        sample=dict(elapsed_s=clock()-started, raw=raw)
        samples.append(sample)
        record(samples)  # Preserve even the sample that rejects the next launch.
        rows=[line.split(', ') for line in raw.splitlines() if line.startswith('1,')]
        if len(rows)!=1 or len(rows[0])!=4 or rows[0][1]!=GPU:
            raise ValueError('GPU 1 identity changed')
        gpu=rows[0]
        idle=int(gpu[2])<1024 and int(gpu[3])==0
        consecutive=consecutive+1 if idle else 0
        if consecutive>=2:return samples
        if clock()-started>=15:raise ValueError('GPU 1 did not become stably idle within 15 seconds')
        sleep(min(1,15-(clock()-started)))


def pick20_remaining_seed(auth):
    """Explicitly approved remaining cases; completed results and costs stay immutable."""
    c=auth.get('pick20_remaining')
    if c is None:return pick20_continuation_seed(auth)
    if auth.get('scope')!=pick.PICK20_BATCH_SCOPE or auth.get('pick20_continuation'):
        raise ValueError('remaining scope must be aligned20 only')
    if c.get('run_case_ids')!=list(pick.PICK20_CASES[17:]):
        raise ValueError('remaining scope permits only017 through019')
    def bound(item):
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('remaining prior evidence changed')
        return read(path)
    ledger=bound(c['ledger']);batch=bound(c['batch'])
    reports=[bound(x) for x in c['completed_results']]
    failed=bound(c['failed_initial_result'])
    if (ledger.get('status')!='closed' or batch.get('status')!='stopped_no_retry'
        or [x.get('case_id') for x in reports]!=list(pick.PICK20_CASES[:17])
        or any(x.get('status')!='completed' or x.get('environment_closed') is not True for x in reports)
        or failed.get('case_id')!='baseline-017' or failed.get('status')!='infrastructure_censored'
        or failed.get('api_requests')!=0 or failed.get('simulator_actions')!=0):
        raise ValueError('remaining scope requires17 completed and zero-action017')
    entries=ledger['entries']
    if (len({e['request_id'] for e in entries})!=len(entries)
        or any(e.get('case_id') not in pick.PICK20_CASES[:17] or e.get('status') not in ('valid','invalid_call')
               or e.get('estimated_usd') is None for e in entries)
        or any(sum(e['case_id']==x['case_id'] for e in entries)!=x['api_requests'] for x in reports)):
        raise ValueError('remaining prior calls are unsettled or incomplete')
    if pick.committed_cost(entries)!=pick.Decimal(c['prior_cost_usd']):
        raise ValueError('remaining prior cost mismatch')
    return entries,reports


def pick20_continuation_seed(auth):
    """Carry immutable settled 000 evidence into a fresh continuation ledger."""
    continuation=auth.get('pick20_continuation')
    if continuation is None:return [],[]
    if continuation.get('run_case_ids') != list(pick.PICK20_CASES[1:5]):
        raise ValueError('continuation may run only 001 through 004')
    for key in ('ledger','result'):
        if sha(Path(continuation[key+'_path'])) != continuation[key+'_sha256']:
            raise ValueError('continuation prior evidence changed')
    ledger=read(Path(continuation['ledger_path']));report=read(Path(continuation['result_path']))
    entries=ledger['entries']
    if (ledger.get('status')!='closed' or report.get('case_id')!='baseline-000'
            or report.get('status')!='completed' or report.get('environment_closed') is not True
            or not entries or len(entries)!=report.get('api_requests')
            or len({e['request_id'] for e in entries})!=len(entries)
            or any(e.get('case_id')!='baseline-000' or e.get('status') not in ('valid','invalid_call')
                   or e.get('estimated_usd') is None for e in entries)):
        raise ValueError('continuation requires settled completed 000 only')
    cost=pick.committed_cost(entries)
    if not cost.is_finite() or cost<0 or cost!=pick.Decimal(continuation['prior_cost_usd']) or cost>8:
        raise ValueError('continuation prior cost mismatch')
    return entries,[report]


def aligned_preflight_worker_command(args,lock_fd):
    """Single source for production argv and real CLI subprocess regression."""
    root=args.output
    return [sys.executable,'-B','-m','astra_pick.run_case','--mode','aligned-preflight','--profile',args.profile,
        '--case-id','baseline-002','--authorization',str(args.authorization),'--roster',str(args.roster),
        '--code',str(args.code),'--lab-root',str(args.lab_root),'--ledger',str(root/'ledger.json'),
        '--output',str(root/'baseline-002'),'--initialization-lock-fd',str(lock_fd)]


def aligned_preflight_worker_result(root,returncode):
    """Keep the worker's first failure; missing JSON must not mask stderr."""
    path=root/'baseline-002/result.json';report=None
    log=root/'worker.log'
    tail=log.read_text(encoding='utf8',errors='replace')[-4000:] if log.is_file() else ''
    try:
        if path.is_file():report=read(path)
    except (ValueError,OSError) as exc:
        return dict(status='stopped_no_retry',returncode=returncode,error_type='InvalidWorkerReport',
                    error_message=str(exc),worker_log_tail=tail,report_present=True)
    if not isinstance(report,dict):
        return dict(status='stopped_no_retry',returncode=returncode,error_type='WorkerExitedWithoutReport',
                    error_message=tail.strip() or 'worker exited without report',worker_log_tail=tail,report_present=False)
    preflight=report.get('preflight')
    if not isinstance(preflight,dict):preflight={}
    passed=(returncode==0 and report.get('status')=='preflight_passed' and report.get('environment_closed') is True
        and type(report.get('api_requests')) is int and report['api_requests']==0
        and type(report.get('explicit_resets')) is int and report['explicit_resets']==1
        and type(report.get('simulator_actions')) is int and 0<=report['simulator_actions']<=600
        and all(preflight.get(k) is True for k in ('passed','convergence_passed','rest_passed')))
    return dict(status='preflight_passed' if passed else 'stopped_no_retry',returncode=returncode,
        preflight=preflight,simulator_actions=report.get('simulator_actions'),environment_closed=report.get('environment_closed'),
        error_type=report.get('error_type'),error_message=report.get('error_message') or report.get('stop_reason') or
        (None if passed else preflight.get('reason','worker result incomplete or failed')),worker_log_tail=tail,report_present=True)


def supervise_aligned_preflight(args):
    import fcntl,subprocess,signal
    auth=read(args.authorization);validate_aligned_preflight(auth,args,worker=False)
    validate_aligned_preflight_inputs(auth,args)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    save(root/'ledger.json',dict(status='no_api',entries=[],provider_requests_cap=0))
    result=dict(status='stopped_no_retry',api_requests=0,automatic_retries=0)
    process=None;lock=None
    try:
        lock=(args.lab_root/'runs/arm-capability.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
        save(root/'gpu-admission.json',dict(gpu=gpu,apps=apps))
        row=next(line.split(',') for line in gpu.splitlines() if GPU in line)
        if row[0].strip()!='1' or int(row[2])>100 or int(row[3])!=0 or GPU in apps:
            raise RuntimeError('GPU1 unavailable; no launch/retry')
        case='baseline-002';started=time.monotonic()
        _initialization_json_exclusive(root/(case+'.launch-intent.json'),dict(case_id=case,authorization_sha256=sha(args.authorization)))
        command=aligned_preflight_worker_command(args,lock.fileno())
        env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
        env.update(CUDA_VISIBLE_DEVICES=GPU,BVI_NO_API='1')
        with (root/'worker.log').open('x') as log:
            process=subprocess.Popen(command,cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
            save(root/'process.json',dict(pid=process.pid,command=command))
            try:result['returncode']=process.wait(timeout=max(1,1200-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise TimeoutError('1200-second hard cap; no retry')
        result['wall_seconds']=time.monotonic()-started
        result.update(aligned_preflight_worker_result(root,result['returncode']))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error_message=str(exc))
    finally:
        if process is not None:
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
            process.wait();result['worker_exited']=True
        if lock is not None:lock.close()
        save(root/'preflight-supervisor.json',result)
    return result


def parallel_admission(sample, active, root, config, *, now_epoch=None):
    """Resource-only scheduling; no scene or task state influences commands."""
    now_epoch=time.time() if now_epoch is None else now_epoch
    if len(active)>=config['max_workers']:return False,'worker_cap'
    if any(not (Path(root)/case/'initialized.json').exists() for case in active):
        return False,'initialization_in_progress'
    if sample['foreign_pids']:return False,'foreign_gpu_process'
    if sample['free_mib']<config['gpu_min_free_mib']:return False,'gpu_memory_headroom'
    if sample['utilization']>config['gpu_admission_utilization_max']:return False,'gpu_busy'
    if sample['load1']>max(1,sample['cpu_count']//2):return False,'cpu_busy'
    for case in active:
        p=Path(root)/(case+'.request-wait.json')
        if p.exists():
            wait=read(p)
            if wait.get('waiting') and now_epoch-wait['started_epoch']>30:
                return False,'request_queue_backpressure'
    return True,'resource_headroom'


def parallel_resource_sample(own_pids):
    import subprocess
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()
    row=[x.split(',') for x in rows if x.split(',')[0].strip()==GPU]
    if len(row)!=1:raise ValueError('Parallel GPU identity changed')
    fields=[x.strip() for x in row[0]]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()
    pids=[int(x.split(',')[0]) for x in apps if x.split(',')[-1].strip()==GPU]
    return dict(epoch=time.time(),gpu_uuid=GPU,free_mib=int(fields[1])-int(fields[2]),
        used_mib=int(fields[2]),utilization=int(fields[3]),load1=os.getloadavg()[0],
        cpu_count=os.cpu_count() or 1,foreign_pids=sorted(set(pids)-set(own_pids)))


def v3_outcome_class(report):
    """Separate financial admission failures from observed task outcomes."""
    if report.get('status') == 'infrastructure_censored' or any(report.get(k) for k in
            ('close_error_type','accounting_error','initialization_result_write_error_eval_only')):
        return 'infrastructure_failure'
    if report.get('termination_category') == 'budget_exhausted':
        return 'case_budget_failure' if report.get('budget_scope') == 'case' else 'batch_budget_failure'
    if report.get('status') == 'completed':
        return 'success' if report.get('strict_pick_success') is True else 'task_failure'
    return 'pending' if report.get('status') == 'running' else 'censored'


def budget_exhaustion_report(auth,error,*,profile=None):
    """A proven case limit finishes only that worker; unknown scope stays a batch issue."""
    result=dict(status='budget_exhausted',termination_category='budget_exhausted',
        budget_scope=getattr(error,'scope','unknown'),stop_reason=str(error))
    if pick.r15_case_isolation(auth) and result['budget_scope']=='case':
        result.update(status='completed',case_limit_reached=True,strict_pick_success=False)
    if profile in pick.V3_PROFILES:
        # The admission decision owns this snapshot. Reading the live ledger
        # here could mix another worker's later reservations into the evidence.
        snapshot=getattr(error,'budget_snapshot',None)
        result['budget_stop']=copy.deepcopy(snapshot) if snapshot is not None else dict(
            case_spent_usd=None,case_committed_usd=None,request_reserve_usd=None,
            case_remaining_usd=None,batch_remaining_usd=None,reason=str(error),
            evidence_status='admission_snapshot_unavailable')
        if result['budget_scope']=='case':
            result.update(status='completed',case_limit_reached=True,strict_pick_success=False)
        result['outcome_class']=v3_outcome_class(result)
    return result


def r15_batch_issue(report, *, isolate_case=False):
    """Independent report-side check; no scene facts enter motion admission."""
    if not isolate_case and report.get('termination_category') in ('cumulative_force_limit','wall_time_limit'):
        return report['termination_category']
    if report.get('execution_profile')!=pick.SOL_R15.name:return None
    motions=('look','move_base','approach_point','move_to','move_eef_chunk','prepare_grasp','replan_grasp','set_gripper','return_to_rest')
    for row in report.get('commands',[]):
        tool=(row.get('call') or {}).get('tool');result=row.get('result',{});f=result.get('motion_completion')
        if tool not in motions or result.get('reason') not in ('completed','arrived'):continue
        if result['reason']=='completed' or not f or f.get('status')!='arrived':return 'false_completion'
        try:
            actual=f['actual'];residual=f['residual']
            if tool=='look':ok=max(abs(v) for v in residual['head_rad'])<.02
            elif tool in ('move_base','approach_point'):
                ok=abs(residual['base_delta'][0])<.015 and abs(residual['base_delta'][2])<.02
            elif tool in ('move_to','move_eef_chunk','prepare_grasp','replan_grasp'):
                strict=tool in ('prepare_grasp','replan_grasp')
                ok=residual['position_error_m']<=(.002 if strict else .01) and residual['rotation_error_rad']<=(.02 if strict else .05)
            elif tool=='return_to_rest':ok=all(actual['rest_flags'][k] is True for k in ('ee_rest','robot_rest','is_static'))
            else:
                closed=f['requested']['gripper_command']==0
                ok=(actual['gripper_command']==0 and actual['closure_command_completed'] is True) if closed else abs(residual['finger_distance_m'])<=.002
        except (KeyError,TypeError,ValueError):ok=False
        if not ok:return 'false_completion'
    return None


def write_batch_stop(root, reason, *, preserve_first=False):
    """Keep the originating exception when peer workers observe shared STOP."""
    root=Path(root)
    if not preserve_first:
        (root/'STOP').write_text(reason,encoding='utf8')
        return reason
    import fcntl
    with (root/'stop-cause.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=root/'stop-cause.json'
        if receipt.exists():reason=read(receipt)['reason']
        else:
            existing=(root/'STOP').read_text(encoding='utf8') if (root/'STOP').exists() else ''
            reason=existing or reason
            pick.atomic_ledger_save(receipt,dict(reason=reason,recorded_at_utc=datetime.now(timezone.utc).isoformat()))
        (root/'STOP').write_text(reason,encoding='utf8')
        return reason


def pick20_parallel_jobs(auth, profile):
    """Resolve actual workers before launch; lean never expands into paired arms."""
    if profile in pick.V3_PROFILES:
        if auth.get('scope') not in pick.V3_SCOPES or auth.get('case_ids') != list(pick.PICK20_CASES[:10]):
            raise ValueError('v3 requires exactly ten candidate cases')
        return [(pick.r15_job_id(case, profile, auth['scope']), case, profile) for case in auth['case_ids']]
    return [(pick.r15_job_id(case,p,auth.get('scope')),case,p) for case in auth['case_ids'] for p in ((profile,) if auth['scope']==pick.R2E_SCOPE else (pick.SOL_R2D,) if auth['scope']==pick.R2D_SCOPE else (pick.SOL_R2C,) if auth['scope']==pick.R2C_SCOPE else (pick.R2B_PROFILES if auth['scope']==pick.R2B_SCOPE else pick.R2A_PROFILES if auth['scope']==pick.R2A_SCOPE else pick.R15_PROFILES) if profile in pick.PAIRED_PROFILES else (profile,))]


def supervise_pick20_parallel(args):
    """Fresh bounded workers, staggered initialization, bounded durable sends.

    Worker failures stop the whole batch. Normal task outcomes free a slot; there
    are no retries, extra cases, or repeated initialization probes.
    """
    import fcntl,signal,subprocess,shutil
    auth=read(args.authorization);args.condition='V-mobile'
    profile=pick.PROFILES[args.profile]
    validate(auth,args,worker=False);validate_candidate_sources(auth,args.code)
    pick.validate_session_scope(auth,profile)
    if not profile.parallel_case_workers:raise ValueError('Parallel profile required')
    config=auth['parallel_execution'];root=Path(auth['output_root'])
    jobs=pick20_parallel_jobs(auth, profile)
    jobs_by_id={job[0]:job for job in jobs}
    job_keys=[job[0] for job in jobs]
    if (args.output.resolve()!=root.resolve() or args.ledger.resolve()!=(root/'ledger.json').resolve()
            or [r['case_id'] for r in read(args.roster)['dev']]!=list(pick.PICK20_CASES)):
        raise ValueError('Parallel root, ledger or roster mismatch')
    root.mkdir(parents=True,exist_ok=False)
    ledger=dict(status='running',authorization_sha256=sha(args.authorization),batch_id=auth['batch_id'],entries=[],prior_evidence=None)
    pick.atomic_ledger_save(args.ledger,ledger)
    active={};started_cases=[];reports={};workers={};lock=None
    started=time.monotonic();last_launch=-float('inf');last_sample=-float('inf');last_report=-float('inf')
    stop_at=None;stop_error=None;peak=0;finished_seconds=0.;launch_enabled=True
    receipt=dict(status='running',automatic_retries=0,recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        concurrency_limit=config['max_workers'],model_requests_serialized=pick.api_concurrency(auth)==1)
    if 'api_concurrency' in config:receipt['api_concurrency_limit']=pick.api_concurrency(auth)
    def request_stop(reason):
        nonlocal stop_at,stop_error,launch_enabled
        launch_enabled=False
        if pick.r15_case_isolation(auth):reason=write_batch_stop(root,reason,preserve_first=True)
        if stop_at is None:stop_at=time.monotonic();stop_error=reason
        (root/'STOP').write_text(reason,encoding='utf-8')
        for case in active:
            folder=root/case
            if folder.is_dir():(folder/'STOP').write_text(reason,encoding='utf-8')
    def summarize():
        snapshot=read(args.ledger)
        receipt.update(pick20_summary(list(reports.values()),snapshot,job_keys),
            active_cases=list(active),started_cases=started_cases,peak_concurrency=peak,
            worker_process_seconds=finished_seconds+sum(time.monotonic()-v['started'] for v in active.values()))
        pick.atomic_ledger_save(root/'batch-result.json',receipt)
    try:
        lock=(args.lab_root/'runs/arm-capability.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wait_pick20_idle(lambda:subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=5),lambda samples:save(root/'initial-idle.json',samples))
        while active or (launch_enabled and len(started_cases)<len(jobs)):
            now=time.monotonic()
            for case,item in list(active.items()):
                process=item['process'];elapsed=now-item['started']
                isolated=pick.r15_case_isolation(auth)
                if isolated and process.poll() is None and elapsed>=config['worker_seconds_cap']-195:
                    folder=root/case
                    if folder.is_dir():(folder/'CASE_WALL_LIMIT').write_text('case wall limit',encoding='utf8')
                if process.poll() is None and elapsed>=config['worker_seconds_cap']-15:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                    item['wall_cap']=True
                    if not isolated:request_stop('worker wall cap; no retry: '+case)
                if process.poll() is not None:
                    process.wait();item['log'].close();finished_seconds+=time.monotonic()-item['started']
                    path=root/case/'result.json'
                    try:report=read(path) if path.exists() else dict(case_id=case,status='infrastructure_censored')
                    except (OSError,ValueError):report=dict(case_id=case,status='infrastructure_censored',reason='unreadable worker result')
                    local_wall=isolated and item.get('wall_cap',False)
                    if local_wall:
                        report.update(status='wall_time_censored',termination_category='wall_time_limit',
                            case_limit_reached=True,environment_closed=False,worker_killed_at_wall_cap=True)
                        save(root/(case+'.wall-cap-receipt.json'),report)
                        job_profile=jobs_by_id[case][2]
                        physical_case=jobs_by_id[case][1]
                        attempt=pick.session_attempt_id(auth,physical_case,job_profile)
                        unresolved=[e for e in pick.budget_entries(read(args.ledger)) if e.get('attempt_id')==attempt
                                    and e.get('status') in ('reserved','unknown','censored')]
                        if unresolved:request_stop('unresolved request in timed-out worker: '+case)
                    if not local_wall and (process.returncode or report.get('status')!='completed' or report.get('environment_closed') is not True):
                        request_stop('worker stopped: '+case+' '+report.get('status','missing'))
                        if report.get('status') in ('running',None):report['status']='infrastructure_censored'
                    if profile in pick.PAIRED_PROFILES and r15_batch_issue(report,isolate_case=isolated):request_stop('R1.5 issue: '+case+' '+r15_batch_issue(report,isolate_case=isolated))
                    reports[case]=report;workers[case]=dict(pid=process.pid,seconds=time.monotonic()-item['started'],returncode=process.returncode)
                    del active[case]
            elapsed_total=finished_seconds+sum(time.monotonic()-v['started'] for v in active.values())
            if (root/'STOP').exists() and stop_at is None:request_stop('operator stop requested')
            if elapsed_total>=config['total_worker_seconds_cap']-20*max(1,len(active)):request_stop('total GPU process seconds cap')
            if time.time()>=auth['expires_at_epoch']-10 or now-started>=auth['batch_wall_seconds_cap']-10:
                request_stop('batch lease or wall cap')
            snapshot=read(args.ledger)
            if snapshot.get('status')!='running':request_stop('request ledger stopped: '+str(snapshot.get('status')))
            if stop_at is not None:
                # Allow in-flight HTTP to resolve under its original 180s bound.
                # No new request can pass the STOP-aware request lock.
                grace=0 if 'seconds cap' in stop_error or 'lease or wall' in stop_error else 185
                if now-stop_at>=grace:
                    for item in active.values():
                        if item['process'].poll() is None:
                            os.killpg(item['process'].pid,signal.SIGKILL)
                if not active:break
            if stop_at is None and now-last_sample>=10:
                sample=parallel_resource_sample([x['process'].pid for x in active.values()]);last_sample=now
                allowed,why=parallel_admission(sample,active,root,config)
                with (root/'resource-samples.jsonl').open('a') as f:
                    f.write(json.dumps(dict(sample,active=list(active),admit=allowed,reason=why))+'\n')
                if sample['foreign_pids']:request_stop('foreign GPU process appeared; no launch')
            if (launch_enabled and len(started_cases)<len(jobs) and allowed
                    and now-last_launch>=config['initialization_stagger_seconds']):
                # Re-evaluate initialization/queue state immediately before launch.
                allowed,why=parallel_admission(sample,active,root,config)
                if allowed:
                    if shutil.disk_usage(root).free<30*1024**3:raise ValueError('insufficient parallel recording disk space')
                    case,physical_case,worker_profile=jobs[len(started_cases)]
                    _initialization_json_exclusive(root/(case+'.launch-intent.json'),dict(case_id=physical_case,authorization_sha256=sha(args.authorization)))
                    command=[sys.executable,'-B','-m','astra_pick.run_case','--mode','live','--profile',worker_profile.name,
                        '--case-id',physical_case,'--authorization',str(args.authorization),'--roster',str(args.roster),
                        '--code',str(args.code),'--lab-root',str(args.lab_root),'--ledger',str(args.ledger),
                        '--output',str(root/case),'--initialization-lock-fd',str(lock.fileno())]
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    log=(root/(case+'.worker.log')).open('x');begin=time.monotonic()
                    process=subprocess.Popen(command,env=env,cwd=args.code,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,pass_fds=(lock.fileno(),))
                    active[case]=dict(process=process,log=log,started=begin);started_cases.append(case)
                    peak=max(peak,len(active));last_launch=begin
                    save(root/(case+'.process.json'),dict(pid=process.pid,command=command,started_at_epoch=time.time()))
            if now-last_report>=5:summarize();last_report=now
            time.sleep(.2)
        receipt['status']='completed' if stop_error is None else 'stopped_no_retry'
        if stop_error:receipt['error_message']=stop_error
    except BaseException as exc:
        receipt.update(status='stopped_no_retry',error_type=type(exc).__name__,error_message=str(exc))
        request_stop(str(exc))
        # Unexpected supervisor faults fail closed; reserved requests remain unknown.
        for item in active.values():
            if item['process'].poll() is None:os.killpg(item['process'].pid,signal.SIGKILL)
    finally:
        for case,item in active.items():
            process=item['process']
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
            process.wait();item['log'].close();finished_seconds+=time.monotonic()-item['started']
            path=root/case/'result.json'
            try:report=read(path) if path.exists() else dict(case_id=case)
            except (OSError,ValueError):report=dict(case_id=case)
            if report.get('status') not in ('completed','supervision_stopped'):report['status']='infrastructure_censored'
            reports[case]=report;workers[case]=dict(pid=process.pid,seconds=time.monotonic()-item['started'],returncode=process.returncode)
        active.clear()
        ledger=read(args.ledger)
        if ledger['status']=='running':ledger['status']='closed'
        pick.atomic_ledger_save(args.ledger,ledger)
        receipt.update(own_worker_exited=True,workers=workers)
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        receipt['lock_released']=True;summarize()
    return receipt


def supervise_pick20(args):
    """One five-case A-only batch; no retries, recovery, or implicit later fifteen."""
    if pick.PROFILES[args.profile].parallel_case_workers:
        return supervise_pick20_parallel(args)
    import fcntl, signal, subprocess, shutil
    auth = read(args.authorization)
    args.condition = 'V-mobile'
    validate(auth, args, worker=False)
    validate_candidate_sources(auth, args.code)
    aligned20=auth.get('scope')==pick.PICK20_BATCH_SCOPE
    weekly=auth.get('scope')==pick.WEEKLY_SCOPE
    dev=auth.get('scope') in pick.DEV_R1_SCOPES

    batch_profile=pick.PROFILES[args.profile] if weekly or dev else pick.SOL_PICK20_ALIGNED if aligned20 else pick.SOL_PICK20

    if weekly:pick.validate_weekly_scope(auth,batch_profile)
    if not weekly and (args.profile != batch_profile.name or auth['case_ids'] != pick.authorized_pick_cases(auth,pick.PICK20_CASES[:10] if dev else pick.PICK20_CASES if aligned20 else pick.PICK20_CASES[:5])):

        raise ValueError('pick20 exact batch scope required')
    if aligned20 and auth.get('pick20_continuation'):raise ValueError('fresh twenty-case batch required')
    rows = read(args.roster)['dev']
    if [r['case_id'] for r in rows] != list(pick.PICK20_CASES):
        raise ValueError('original ordered twenty-case roster required')
    root = Path(auth['output_root'])
    if args.output.resolve() != root.resolve() or args.ledger.resolve() != (root/'ledger.json').resolve():
        raise ValueError('pick20 shared ledger/root mismatch')
    prior_entries,reports=pick20_remaining_seed(auth)
    if auth.get('cold_resume') or auth.get('checkpoint_branch'):
        if prior_entries or reports:raise ValueError('Multiple continuation mechanisms')
        prior_entries=(checkpoint_branch_seed if auth.get('checkpoint_branch') else cold_resume_seed)(auth,batch_profile)
    root.mkdir(parents=True, exist_ok=False)
    ledger = dict(status='running', authorization_sha256=sha(args.authorization), batch_id=auth['batch_id'], entries=prior_entries,
                  prior_evidence=auth.get('checkpoint_branch') or auth.get('pick20_remaining') or auth.get('pick20_continuation'))
    save(args.ledger, ledger)
    process=None; lock=None
    batch_started=time.monotonic()
    receipt = dict(status='running', automatic_retries=0, recorded_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        lock=(args.lab_root/'runs/arm-capability.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(root).free < 10*1024**3: raise ValueError('insufficient recording disk space')
        for case in auth['case_ids']:
            if any(r['case_id']==case for r in reports):continue
            if (weekly or dev) and time.monotonic()-batch_started>=auth['batch_wall_seconds_cap']:

                raise TimeoutError('weekly batch wall cap; no retry')
            case_started=time.monotonic()
            pick.validate_session_scope(auth, batch_profile)
            wait_pick20_idle(lambda:subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5),
                lambda samples:save(root/(case+'.gpu.json'),samples))
            _initialization_json_exclusive(root/(case+'.launch-intent.json'),dict(case_id=case,authorization_sha256=sha(args.authorization)))
            command=[sys.executable,'-B','-m','astra_pick.run_case','--mode','live',
                '--profile',args.profile,'--case-id',case,'--authorization',str(args.authorization),
                '--roster',str(args.roster),'--code',str(args.code),'--lab-root',str(args.lab_root),
                '--ledger',str(args.ledger),'--output',str(root/case),'--initialization-lock-fd',str(lock.fileno())]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU)
            with (root/(case+'.worker.log')).open('x') as log:
                process=subprocess.Popen(command,env=env,cwd=args.code,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(lock.fileno(),))
                save(root/(case+'.process.json'),dict(pid=process.pid,command=command))
                remaining=(min(auth['gpu1_process_seconds_cap']-(time.monotonic()-case_started),
                    auth['batch_wall_seconds_cap']-(time.monotonic()-batch_started),
                    auth['expires_at_epoch']-time.time()) if weekly or dev else 1800-(time.monotonic()-case_started))

                try: returncode=process.wait(timeout=max(1,remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise TimeoutError('pick20 authorized worker wall cap; no retry')
            path=root/case/'result.json'
            report=read(path) if path.is_file() else dict(case_id=case,status='infrastructure_censored')
            reports.append(report)
            receipt.update(pick20_summary(reports,read(args.ledger),auth['case_ids'],weekly=weekly))
            save(root/'batch-result.json',receipt)
            if returncode or report.get('status')!='completed' or report.get('environment_closed') is not True:
                raise RuntimeError('pick20 worker stopped: '+case+' '+report.get('status','missing'))
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='stopped_no_retry',error_type=type(exc).__name__,error_message=str(exc))
    finally:
        if process is not None:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        # Retain a timed-out worker's partial record without counting it as completed.
        for case in auth['case_ids']:
            if (root/(case+'.launch-intent.json')).exists() and not any(r['case_id']==case for r in reports):
                path=root/case/'result.json'
                r=read(path) if path.exists() else dict(case_id=case)
                r['status']='infrastructure_censored';reports.append(r)
        ledger=read(args.ledger)
        if ledger['status']=='running': ledger['status']='closed';save(args.ledger,ledger)
        receipt.update(pick20_summary(reports,ledger,auth['case_ids'],weekly=weekly), own_worker_exited=process is None or process.poll() is not None)
        if lock is not None: fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        receipt['lock_released']=True
        save(root/'batch-result.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('archive-check','live','supervise','l1-replay','init-only','init-supervise','ab-reference','ab-reference-supervise','full-b-preflight','transport-preflight','pick20-supervise','aligned-preflight','aligned-preflight-supervise'),default='archive-check')
    parser.add_argument('--transport-stage',choices=('reference','survey','check','plumbing'))
    parser.add_argument('--profile',choices=tuple(pick.PROFILES),default=None,
        help='Sol-mainline for new single-pick runs; historical profiles remain explicit')
    parser.add_argument('--archive-root',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    for name in ('authorization','roster','code','lab-root','ledger'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--initialization-observation',choices=('init-a','init-b'))
    parser.add_argument('--initialization-previous',type=Path)
    parser.add_argument('--initialization-lock-fd',type=int)
    parser.add_argument('--prior-budget',type=Path)
    parser.add_argument('--continuation-review',type=Path)
    parser.add_argument('--replay-roster', type=Path)
    parser.add_argument('--replay-turn', type=int)
    parser.add_argument('--replay-arm', choices=('control','abort','candidate'))
    parser.add_argument('--replay-group',choices=('L1-A','L1-B'),default='L1-A')
    args=parser.parse_args(argv)
    if args.profile is None:
        if args.mode == 'archive-check':
            args.profile = 'C0'
        elif args.mode in ('live', 'pick20-supervise', 'aligned-preflight', 'aligned-preflight-supervise'):
            args.profile = pick.SOL_MAINLINE.name
        else:
            parser.error('this specialized mode requires an explicit --profile')
    elif args.profile == 'Sol-mainline':
        args.profile = pick.SOL_MAINLINE.name
    if args.mode=='transport-preflight':
        for name in ('authorization','code','lab_root','transport_stage'):
            if getattr(args,name) is None: parser.error('transport preflight requires '+name)
        result=transport_preflight_native(args)
        return 0 if result['status'] in ('reference_captured_visual_review_pending','survey_complete_visual_review_pending','plumbing_passed','plumbing_passed_semantics_pending') else 1
    if args.mode=='archive-check':
        if args.profile not in ('C0', 'C1'):
            parser.error('experimental execution profile has no historical parity baseline')
        if args.archive_root is None or args.cache is None:
            parser.error('archive-check requires --archive-root and --cache (extracted provider archive)')
        summary=archive_check(args.profile,args.archive_root,args.cache,args.output)
        print(json.dumps(summary,ensure_ascii=False))
        return 0 if summary['accepted'] else 1
    for name in ('authorization','roster','code','lab_root','ledger','case_id'):
        if getattr(args,name) is None: parser.error('live requires '+name)
    args.condition='V-mobile'
    if args.mode=='aligned-preflight-supervise':
        result=supervise_aligned_preflight(args)
        return 0 if result['status']=='preflight_passed' else 1
    if args.mode=='aligned-preflight':
        result=live_run(args)
        return 0 if result['status']=='preflight_passed' else 1
    if args.mode=='pick20-supervise':
        result=supervise_pick20(args)
        return 0 if result['status']=='completed' else 1
    if args.mode == 'full-b-preflight':
        result=live_run(args)
        return 0 if result['status']=='preflight_passed' else 1
    if args.mode == 'ab-reference-supervise':
        result = supervise_ab_references(args)
        return 0 if result['status'] == 'completed' else 1
    if args.mode == 'ab-reference':
        result = live_run(args)
        return 0 if result['status'] == 'reference_captured' and result.get('environment_closed') else 1
    if args.mode=='init-supervise':
        result=supervise_initialization_only(args)
        return 0 if result['status']=='initialization_observations_complete' else 1
    if args.mode=='init-only':
        auth=read(args.authorization)
        validate(auth,args)
        result=live_run(args)
        return 0 if result['status'] in ('initialization_match','initialization_mismatch') else 1
    if args.mode != 'l1-replay' and pick.PROFILES[args.profile] in (pick.C2_R33, pick.C2_R33_FAR):
        auth = read(args.authorization)
        pick.validate_r33_paid_scope(auth, pick.PROFILES[args.profile], case_id=args.case_id)
        if auth.get('stage') not in ('L3', 'L4'):
            raise ValueError('r33 physical CLI requires L3 or L4')
    if args.mode != 'l1-replay' and pick.PROFILES[args.profile] in (pick.C2_R31, pick.C2_R31_FAR):
        auth = read(args.authorization)
        pick.validate_r31_paid_scope(auth, pick.PROFILES[args.profile], case_id=args.case_id)
        if auth.get('stage') not in ('L3', 'L4'):
            raise ValueError('r31 physical CLI requires L3 or L4')
    if args.mode == 'l1-replay' and (args.replay_roster is None or args.replay_turn is None or args.replay_arm is None):
        parser.error('l1-replay requires bound command roster, turn and arm')
    if args.mode == 'supervise':
        if args.prior_budget is None: parser.error('supervise requires explicit --prior-budget')
        result = automatic_batch(args) if pick.PROFILES[args.profile].automatic_supervision else supervise_attempt(args)
        print(json.dumps(result))
        return 0 if result['status']=='ready_for_review' else 1
    result=live_run(args)
    return 0 if result['status']=='completed' else 1


def cold_state_bytes(value):
    """Trusted local checkpoint codec; imported modules are bound by frozen sources."""
    import pickle, types, io
    class Writer(pickle.Pickler):
        def persistent_id(self,obj):
            if isinstance(obj,types.ModuleType):return ('module',obj.__name__)
    stream=io.BytesIO();Writer(stream,protocol=5).dump(value);return stream.getvalue()


def cold_state_loads(data):
    """Only call after checking the checkpoint manifest against authorized hashes."""
    import pickle, io, importlib
    class Reader(pickle.Unpickler):
        def persistent_load(self,key):
            if not isinstance(key,tuple) or len(key)!=2 or key[0]!='module':
                raise ValueError('Unknown checkpoint reference')
            return importlib.import_module(key[1])
    return Reader(io.BytesIO(data)).load()


COLD_CHECKPOINT_CONTRACT = 'sol-cold-checkpoint-v1'
COLD_RUNTIME_PORTS = frozenset(('ports','step_once','read_qpos','is_stopped','recorder',
    'persist','deadline','timed','hot_checkpoint','cold_checkpoint','action_budget_expired','grasp_sensor_feedback',
    'clearance','refresh_clearance','rest_depth_checker',
    # Source-bound output formatting only; no controller or policy state.
    # The constructor binds it from the frozen execution profile on resume.
    'grasp_direction_feedback'))


def validate_episode_checkpoint_memory(state, steps):
    memory=getattr(state,'episode_memory',None)
    if memory is None:return
    from .memory import EpisodeMemory
    value=memory.snapshot()
    if value['episode_id']!=state.response_session.attempt_id or value['control_step']>steps:
        raise ValueError('Episode memory differs from completed physical boundary')
    EpisodeMemory.from_snapshot(value,episode_id=memory.episode_id,
        odom_frame_id=memory.odom_frame_id,source_binding=memory.source_binding)


def episode_checkpoint_record(state, steps):
    """Auditable memory payload shared by live continuation and cold receipts."""
    if getattr(state,'episode_memory',None) is None:return None
    validate_episode_checkpoint_memory(state,steps)
    snapshot=state.episode_memory.snapshot()
    return dict(snapshot=snapshot,sha256=hashlib.sha256(canonical(snapshot)).hexdigest())


def write_cold_checkpoint(root, *, state, executor, native, ledger, binding, runtime_state):
    """Save a completed-action boundary; receipt is the final commit record."""
    import hashlib, os
    root=Path(root)
    if root.exists():raise FileExistsError('Checkpoint destination must be fresh')
    if (executor.pending is not None or executor.failed or executor.ended
            or state.response_session is None or (state.response_session.status!='ready' and not (state.response_session.status=='stopped' and state.terminal_reason in ('give_up','request_limit')))
            or state.terminal_reason not in (None,'give_up','request_limit')):
        raise ValueError('Checkpoint requires a resumable completed-action boundary')
    if any(x.get('status') not in ('valid','invalid_call','invalid_verification','not_sent') for x in ledger['entries']):
        raise ValueError('Unresolved delivery cannot be resumed')
    if len(state.archive)+state.verification_requests!=len(ledger['entries']):
        raise ValueError('Checkpoint request accounting mismatch')
    memory_record=episode_checkpoint_record(state,executor.steps)
    fields={k:v for k,v in vars(executor).items() if k not in COLD_RUNTIME_PORTS}
    payload=dict(contract=COLD_CHECKPOINT_CONTRACT,native=native,state=state,
        executor=fields,ledger=ledger,binding=binding,runtime_state=runtime_state)
    data=cold_state_bytes(payload)
    root.mkdir(parents=True)
    with (root/'state.bin').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    receipt=dict(contract=COLD_CHECKPOINT_CONTRACT,sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),steps=executor.steps,requests=len(ledger['entries']),binding=binding,
        trusted_local_pickle=True,complete=True)
    if memory_record is not None:receipt['episode_memory_sha256']=memory_record['sha256']
    save(root/'receipt.json',receipt)
    return receipt


def read_cold_checkpoint(root, *, receipt_sha256, binding):
    """Authorization pins receipt, payload and frozen origin before unpickling."""
    root=Path(root)
    if sha(root/'receipt.json')!=receipt_sha256:raise ValueError('Checkpoint receipt hash mismatch')
    receipt=read(root/'receipt.json')
    if receipt.get('contract')!=COLD_CHECKPOINT_CONTRACT or receipt.get('binding')!=binding:
        raise ValueError('Checkpoint provenance mismatch')
    data=(root/'state.bin').read_bytes()
    if len(data)!=receipt['bytes'] or hashlib.sha256(data).hexdigest()!=receipt['sha256']:
        raise ValueError('Checkpoint payload hash mismatch')
    payload=cold_state_loads(data)
    if payload['contract']!=COLD_CHECKPOINT_CONTRACT or payload['binding']!=binding:
        raise ValueError('Checkpoint payload provenance mismatch')
    memory_record=episode_checkpoint_record(payload['state'],receipt['steps'])
    if memory_record is not None and receipt.get('episode_memory_sha256')!=memory_record['sha256']:
        raise ValueError('Checkpoint memory hash mismatch')
    return payload


def restore_cold_python(payload, executor, *, session_path):
    """Bind fresh IO ports around the retained control and policy state."""
    fields=payload['executor'];state=payload['state']
    if fields['pending'] is not None or fields['failed'] or fields['ended']:
        raise ValueError('Cannot restore an incomplete action')
    if fields['max_steps']!=executor.max_steps or fields['mapping']!=executor.mapping:
        raise ValueError('Controller configuration changed')
    expected=set(vars(executor))-COLD_RUNTIME_PORTS
    # Historical payloads and constructors both omit the default-off field.
    # This is compatibility with false, never an implicit profile migration.
    retained_feedback=fields.get('rest_progress_feedback',False)
    bound_feedback=getattr(executor,'rest_progress_feedback',False)
    if (type(retained_feedback) is not bool or type(bound_feedback) is not bool
            or retained_feedback != bound_feedback):
        raise ValueError('Rest progress feedback checkpoint profile mismatch')
    # Created by the first clearance preview, not by the fresh constructor.
    # Restore its retained value; do not silently discard diagnostic state or
    # permit arbitrary extra checkpoint fields.
    if ('clearance_last_codec' in fields and
            getattr(executor,'local_clearance_guard',False) and
            isinstance(fields['clearance_last_codec'],dict)):
        expected.add('clearance_last_codec')
    if set(fields)!=expected:
        raise ValueError('Executor checkpoint field coverage mismatch')
    validate_episode_checkpoint_memory(state,fields['steps'])
    for k,v in fields.items():setattr(executor,k,v)
    state.response_session.path=Path(session_path)
    state.response_session.status='ready'
    state.terminal_reason=None
    state.response_session.persist()
    return state


def prepare_episode_memory_migration_cpu(root, *, receipt_sha256, binding,
        source_root, destination_root, destination_sources, source_delta,
        profile, geometry_sha256, calibration_sha256):
    """Build an offline migration candidate; never authorize live continuation.

    Old evidence stays in its original policy archive. New memory starts empty
    at the inherited step, without inferred past geometry or target evidence.
    This function is deliberately absent from all live resume dispatch paths.
    """
    from .memory import EpisodeMemory
    if profile not in (pick.SOL_EPISODIC,pick.SOL_CLEARANCE):
        raise ValueError('Explicit episode-memory destination required')
    if binding.get('profile') not in (pick.SOL_SELF.name,pick.SOL_BASE.name) or binding.get('case_id')!='baseline-014':
        raise ValueError('Only archived old 014 SELF/BASE migration supported')
    old=binding.get('source_sha256',{});new=destination_sources
    changed_allowed=CHECKPOINT_BRANCH_CHANGED | {'tests/test_astra_pick_measure.py','tests/test_astra_pick_parity.py'}
    added_allowed={'src/astra_pick/memory.py','src/astra_pick/robot/clearance.py',
        'research/c2/diagnostics/2026-10-14-episode-memory-clearance-cpu/robot-geometry.json'}
    delta={name:dict(old=old.get(name),new=new.get(name))
        for name in sorted(set(old)|set(new)) if old.get(name)!=new.get(name)}
    if not old or not delta or delta!=source_delta:
        raise ValueError('Exact migration source delta required')
    for name in delta:
        if name not in new or (name in old and name not in changed_allowed) or (name not in old and name not in added_allowed):
            raise ValueError('Unsupported episode migration source change: '+name)
    for sources,directory in ((old,source_root),(new,destination_root)):
        directory=Path(directory).resolve()
        package={p.relative_to(directory).as_posix() for p in (directory/'src/astra_pick').rglob('*.py')}
        if not package or not package<=set(sources):
            raise ValueError('Migration must bind every package source')
        for name,digest in sources.items():
            path=(directory/name).resolve()
            if (not path.is_relative_to(directory) or not re.fullmatch(r'[0-9a-f]{64}',str(digest))
                    or not path.is_file() or sha(path)!=digest):
                raise ValueError('Migration source hash mismatch: '+name)
    calibration=hashlib.sha256(canonical(dict(robot_geometry_sha256=EPISODE_GEOMETRY_SHA256,
        head_link='head_camera_link',hand_link='gripper_link',hand_mount_m=[-.1,0.,.1],
        depth_unit='mm',optical_axes='right_down_forward',sensor_protocol='robot-rgbd-640-v1'))).hexdigest()
    geometry_path='research/c2/diagnostics/2026-10-14-episode-memory-clearance-cpu/robot-geometry.json'
    if (geometry_sha256!=EPISODE_GEOMETRY_SHA256 or calibration_sha256!=calibration
            or new.get(geometry_path)!=geometry_sha256):
        raise ValueError('Migration geometry or calibration binding mismatch')
    # Source verification precedes trusted unpickling. No simulator is created.
    payload=read_cold_checkpoint(root,receipt_sha256=receipt_sha256,binding=binding)
    receipt=read(Path(root)/'receipt.json');state=payload['state'];fields=payload['executor']
    entries=payload['ledger']['entries'];session=state.response_session
    if (receipt.get('complete') is not True or payload['runtime_state'].get('stopped') is not False
            or fields.get('pending') is not None or fields.get('failed') is not False or fields.get('ended') is not False
            or state.terminal_reason is not None or session is None or session.status!='ready'
            or fields['steps']!=receipt['steps'] or fields['max_steps']!=profile.step_cap
            or len(entries)!=receipt['requests'] or len(state.archive)+state.verification_requests!=len(entries)
            or any(e.get('status') not in ('valid','invalid_call','invalid_verification','not_sent') for e in entries)
            or session.attempt_id!=binding['attempt_id']
            or session.binding.get('authorization_sha256')!=binding['authorization_sha256']
            or getattr(state,'episode_memory',None) is not None
            or not fields.get('motion_guard') or not fields.get('self_collision_guard')):
        raise ValueError('Memory migration needs an exact safe completed checkpoint')
    session.bind(dict(position='baseline-014',authorization_sha256=binding['authorization_sha256'],
        turn=len(state.archive),observation={'history':state.history}),state.archive)
    if bool(fields.get('base_tracking_guard',False))!=(binding['profile']==pick.SOL_BASE.name):
        raise ValueError('Archived profile and base guard differ')
    additions=dict(base_tracking_guard=True)
    if profile.local_clearance:
        additions.update(local_clearance_guard=True,motion_goal_local=None,hold_reference_local=None,
            motion_suspended=False,clearance_last_feedback={})
    if set(fields)&(set(additions)-{'base_tracking_guard'}):
        raise ValueError('Unexpected preexisting clearance migration state')
    def retained(value):
        policy={k:v for k,v in vars(value['state']).items() if k!='episode_memory'}
        executor={k:v for k,v in value['executor'].items() if k not in additions}
        return dict(policy=policy,executor=executor,native=value['native'],runtime=value['runtime_state'],
            ledger=value['ledger'],binding=value['binding'])
    before=hashlib.sha256(cold_state_bytes(retained(payload))).hexdigest()
    memory=EpisodeMemory(session.attempt_id,session.attempt_id+':odometry',dict(
        sensor_calibration_sha256=calibration,robot_geometry_sha256=geometry_sha256,memory_version='episode-memory-v1'))
    memory.begin_observation(step=fields['steps'])
    state.episode_memory=memory
    payload['executor']=dict(fields,**additions)
    after=hashlib.sha256(cold_state_bytes(retained(payload))).hexdigest()
    if before!=after:raise ValueError('Episode migration changed retained state')
    return payload,dict(contract='episode-memory-migration-cpu-v1',cpu_only=True,live_resume_authorized=False,
        checkpoint_receipt_sha256=receipt_sha256,from_profile=binding['profile'],to_profile=profile.name,
        source_delta=delta,geometry_sha256=geometry_sha256,calibration_sha256=calibration,
        preserved_state_sha256=before,restored_state_sha256=after,invariant_state_equal=True,
        steps=fields['steps'],requests=len(entries),memory_initialization='empty_at_inherited_step',
        added_executor_fields=sorted(additions),api_requests=0,physics_steps=0)


def canonical_fetch_velocity_snapshot(checkpoint, *, urdf, urdf_sha256):
    """Reconstruct derived Fetch link velocities from saved generalized state.

    Evaluation-side restore only. Never read a destination scene or modify the
    archived checkpoint. This is an explicit canonical-state restore, not a
    claim that historical link velocity caches were reproduced byte for byte.
    All other buffers, task state, poses and RNG remain unchanged.
    """
    import numpy as np
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from .robot.eef_tools import RobotFK
    from bvi.pick_system.snapshot_transport import clone_snapshot_value
    if sha(Path(urdf))!=urdf_sha256 or checkpoint.get('contract')!='bvi-eval-snapshot-transport-v2':
        raise ValueError('Canonical velocity calibration/contract mismatch')
    fk=RobotFK(Path(urdf));layout=checkpoint['layout'];buffers=checkpoint['state']['gpu_buffers']
    def array(value):
        return value.detach().cpu().numpy().copy() if hasattr(value,'detach') else np.asarray(value).copy()
    def unique(values,predicate):
        matches=[i for i,value in enumerate(values) if predicate(json.loads(value))]
        if len(matches)!=1:raise ValueError('Canonical velocity requires one exact Fetch binding')
        return matches[0]
    index=unique(layout['cuda_articulation_qpos'],lambda v: v[-1]=='fetch')
    if layout['cuda_articulation_qpos']!=layout['cuda_articulation_qvel']:
        raise ValueError('Canonical velocity joint layouts differ')
    q=array(buffers['cuda_articulation_qpos'])[index].astype(float)
    dq=array(buffers['cuda_articulation_qvel'])[index].astype(float)
    data=array(buffers['cuda_rigid_body_data']);labels=layout['cuda_rigid_body_data']
    root_index=unique(labels,lambda v:v==[0,'link','fetch',fk.root])
    root=data[root_index]
    if q.shape!=(len(fk.order),) or dq.shape!=q.shape or not all(np.isfinite(v).all() for v in (q,dq,data)):
        raise ValueError('Canonical velocity invalid saved arrays')
    com={}
    for node in ET.parse(urdf).getroot().findall('link'):
        origin=node.find('inertial/origin')
        com[node.get('name')]=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in com.values()):
        raise ValueError('Canonical velocity invalid inertial origins')
    rotation=Rotation.from_quat(root[[4,5,6,3]]).as_matrix()
    rotations={fk.root:rotation};angular={fk.root:root[7:10].astype(float)}
    linear={fk.root:root[10:13].astype(float)-np.cross(angular[fk.root],rotation@com[fk.root])}
    pending=list(fk.joints)
    while pending:
        remaining=[]
        for name,kind,parent,child,origin,axis in pending:
            if parent not in rotations:
                remaining.append((name,kind,parent,child,origin,axis));continue
            joint_rotation=rotations[parent]@origin[:3,:3]
            offset=rotations[parent]@origin[:3,3]
            if kind=='prismatic':offset+=joint_rotation@axis*q[fk.index[name]]
            linear[child]=linear[parent]+np.cross(angular[parent],offset)
            angular[child]=angular[parent].copy();motion=np.eye(3)
            if kind=='prismatic':linear[child]+=joint_rotation@axis*dq[fk.index[name]]
            if kind in ('continuous','revolute'):
                angular[child]+=joint_rotation@axis*dq[fk.index[name]]
                motion=Rotation.from_rotvec(axis*q[fk.index[name]]).as_matrix()
            rotations[child]=joint_rotation@motion
        if len(remaining)==len(pending):raise ValueError('Canonical velocity disconnected calibration')
        pending=remaining
    expected_links=set(rotations)
    bound=[json.loads(v)[3] for v in labels if json.loads(v)[0:3]==[0,'link','fetch']]
    if len(bound)!=len(set(bound)) or set(bound)!=expected_links:
        raise ValueError('Canonical velocity Fetch link coverage mismatch')
    output=clone_snapshot_value(checkpoint);target=output['state']['gpu_buffers']['cuda_rigid_body_data']
    changes=[]
    for i,label in enumerate(labels):
        entity=json.loads(label)
        if entity[:3]!=[0,'link','fetch'] or entity[3]==fk.root:continue
        name=entity[3]
        velocity=np.concatenate((angular[name],linear[name]+np.cross(angular[name],rotations[name]@com[name])))
        values=velocity.astype(data.dtype)
        if hasattr(target,'new_tensor'):target[i,7:13]=target.new_tensor(values)
        else:target[i,7:13]=values
        changes.append(dict(link=name,max_abs_cache_change=float(np.max(np.abs(data[i,7:13]-values)))))
    return output,dict(contract='fetch-urdf-com-velocity-v1',evaluation_only=True,
        urdf_sha256=urdf_sha256,cache_changes=changes,physical_actions=0,
        independent_generalized_state_preserved=True,original_cache_preserved_in_parent=True)


def rebuild_local_floor_memory(memory, checker, evidence):
    """CPU migration from hash-bound robot depth; never modify the parent map.

    Evidence maps each retained geometry observation to qpos and calibrated
    cameras. Exact old occupied reconstruction also checks the supplied frame
    transforms and robot self-filter before applying the new classification.
    This produces a candidate, not permission to restore or spend money.
    """
    import numpy as np
    from dataclasses import replace
    from .memory import EpisodeMemory
    from .robot.clearance import SensorClearance, odometry_transform
    original=memory.snapshot()
    if (checker.memory is not memory or checker.config.local_support_floor
            or 'geometry_classifier' in original['source_binding']):
        raise ValueError('Local floor migration requires the original classifier')
    groups={}
    for row in original['geometry']:
        groups.setdefault(row['evidence']['observation_id'],[]).append(row)
    if not groups or set(evidence)!=set(groups):
        raise ValueError('Exact retained observation evidence required')
    rebuilt=[];audit=[]
    geometry={name:dict(row,**({'convex_vertices':checker.convex_vertices[name]}
        if name in checker.convex_vertices else {})) for name,row in checker.geometry.items()}
    candidate=SensorClearance(checker.fk,memory,geometry=geometry,
        config=replace(checker.config,local_support_floor=True))
    for oid in sorted(groups,key=lambda key:(memory.observation_binding(key)['control_step'],key)):
        rows=groups[oid]
        frame=evidence[oid];binding=memory.observation_binding(oid)
        cameras={name:frame['cameras'][name] for name in sorted(frame['cameras'])}
        if set(cameras)!={row['evidence']['camera'] for row in rows}:
            raise ValueError('Migration camera coverage mismatch')
        for name,camera in cameras.items():
            raw=np.asarray(camera['depth_mm'])
            if (raw.ndim!=2 or not np.issubdtype(raw.dtype,np.integer)
                    or np.any(raw<0) or np.any(raw>np.iinfo(np.int32).max)):
                raise ValueError('Migration requires original integer depth')
            digest=hashlib.sha256(canonical(dict(shape=list(raw.shape),
                encoding='little-endian-int32-mm'))+raw.astype('<i4').tobytes()).hexdigest()
            if digest!=binding['sensor_sha256'][name+'_depth']:
                raise ValueError('Migration raw depth hash mismatch')
        args=(oid,frame['qpos'],odometry_transform(binding['odometry']),cameras)
        kwargs=dict(control_step=binding['control_step'],
                    calibration_sha256=binding['calibration_sha256'])
        previous=checker.ingest_observation(*args,**kwargs)
        expected={r['id']:r for r in rows if r['status']=='occupied'}
        actual={r['id']:r for r in previous if r['status']=='occupied'}
        if canonical(expected)!=canonical(actual):
            raise ValueError('Migration original occupied reconstruction mismatch')
        rebuilt.extend(candidate.ingest_observation(*args,**kwargs))
        audit.append(dict(observation_id=oid,control_step=binding['control_step'],
                          raw_depth_verified=True,original_occupied_exact=True))
    # Every non-geometry collection is copied verbatim, including targets,
    # search regions, actions, route, attachment and eviction accounting.
    updated=dict(original,geometry=rebuilt,source_binding=dict(original['source_binding'],
        geometry_classifier='local-support-floor-v1'))
    output=EpisodeMemory.from_snapshot(updated,episode_id=memory.episode_id,
        odom_frame_id=memory.odom_frame_id,source_binding=updated['source_binding'])
    if canonical(memory.snapshot())!=canonical(original):
        raise ValueError('Migration modified parent memory')
    if any(canonical(output.snapshot()[key])!=canonical(value)
           for key,value in original.items() if key not in ('geometry','source_binding')):
        raise ValueError('Migration changed retained memory')
    return output,dict(contract='local-support-floor-memory-v1',observations=audit,
        parent_snapshot_sha256=hashlib.sha256(canonical(original)).hexdigest(),
        snapshot_sha256=hashlib.sha256(canonical(output.snapshot())).hexdigest(),
        physical_steps=0,api_requests=0,live_resume_authorized=False)


def validate_local_floor_memory_artifact(memory, artifact, *, checkpoint_receipt_sha256,
                                         source_sha256):
    """Validate a separately hash-pinned migration artifact before replacement.

    The caller must verify the artifact file hash against its authorization.
    This function never changes policy, native state, the ledger or parent map.
    """
    from .memory import EpisodeMemory
    if (set(artifact)!={'contract','checkpoint_receipt_sha256','source_sha256','audit','snapshot'}
            or artifact['contract']!='local-support-floor-artifact-v1'
            or artifact['checkpoint_receipt_sha256']!=checkpoint_receipt_sha256
            or artifact['source_sha256']!=source_sha256):
        raise ValueError('Local floor artifact source binding mismatch')
    old=memory.snapshot();new=artifact['snapshot'];audit=artifact['audit']
    if (audit.get('contract')!='local-support-floor-memory-v1'
            or audit.get('physical_steps')!=0 or audit.get('api_requests')!=0
            or audit.get('live_resume_authorized') is not False
            or audit.get('parent_snapshot_sha256')!=hashlib.sha256(canonical(old)).hexdigest()
            or audit.get('snapshot_sha256')!=hashlib.sha256(canonical(new)).hexdigest()
            or 'geometry_classifier' in old['source_binding']):
        raise ValueError('Local floor artifact snapshot mismatch')
    expected=dict(old['source_binding'],geometry_classifier='local-support-floor-v1')
    restored=EpisodeMemory.from_snapshot(new,episode_id=memory.episode_id,
        odom_frame_id=memory.odom_frame_id,source_binding=expected)
    if any(canonical(new[k])!=canonical(v) for k,v in old.items()
           if k not in ('geometry','source_binding')):
        raise ValueError('Local floor artifact changed retained memory')
    ids={row['evidence']['observation_id'] for row in old['geometry']}
    records=audit.get('observations',[])
    if (len(records)!=len(ids) or {v.get('observation_id') for v in records}!=ids
            or any(v.get('raw_depth_verified') is not True
                   or v.get('original_occupied_exact') is not True
                   or v.get('control_step')!=memory.observation_binding(v['observation_id'])['control_step']
                   for v in records)
            or {r['evidence']['observation_id'] for r in new['geometry']}!=ids):
        raise ValueError('Local floor artifact incomplete observations')
    return restored


def carry_resume_digest(payload):
    """IO path may change; control, memory, provider history and accounting may not."""
    state=payload['state'];session=state.response_session
    return hashlib.sha256(cold_state_bytes(dict(
        policy={k:v for k,v in vars(state).items() if k!='response_session'},
        session={k:v for k,v in vars(session).items() if k!='path'},
        executor=payload['executor'],native=payload['native'],ledger=payload['ledger'],
        runtime=payload['runtime_state']))).hexdigest()


def validate_carry_resume_payload(payload, grant):
    state=payload['state'];fields=payload['executor'];session=state.response_session
    receipt=read(Path(grant['checkpoint'])/'receipt.json')
    if (payload['binding']!=grant['binding'] or payload['runtime_state'].get('stopped') is not False
            or fields.get('pending') is not None or fields.get('failed') is not False
            or fields.get('ended') is not False or state.terminal_reason is not None
            or session is None or session.status!='ready'
            or (state.episode_memory is None and grant.get('source_migration',{}).get('contract') not in ('hot-recovery-resume-v1','rest-brake-budget-resume-v1'))
            or fields['steps']!=receipt['steps']
            or len(state.archive)+state.verification_requests!=receipt['requests']
            or payload['ledger']['entries']!=read(Path(grant['parent_ledger_path']))['entries']):
        raise ValueError('Carry migration needs exact ready completed state')
    if state.episode_memory is not None:validate_episode_checkpoint_memory(state,fields['steps'])
    session.bind(dict(position=grant['binding']['case_id'],
        authorization_sha256=grant['binding']['authorization_sha256'],
        turn=len(state.archive),observation={'history':state.history}),state.archive)


def validate_carry_source_migration(auth, profile, grant, ledger):
    migration=grant['source_migration']
    brake_budget = migration.get('contract') == 'rest-brake-budget-resume-v1'
    hot = migration.get('contract') in ('hot-recovery-resume-v1','rest-brake-budget-resume-v1')
    lifecycle = migration.get('contract') == 'carry-resume-lifecycle-v1'
    floor = migration.get('contract') == 'local-support-floor-resume-v1'
    if (profile!=(pick.SOL_HOT_RECOVERY if hot else pick.SOL_LOCAL_FLOOR if floor else pick.SOL_CARRY)
            or migration.get('contract') not in ('carry-memory-copy-v1','carry-resume-lifecycle-v1','local-support-floor-resume-v1','hot-recovery-resume-v1','rest-brake-budget-resume-v1')
            or auth.get('case_ids')!=['baseline-014']):
        raise ValueError('Unsupported carry source migration')
    if 'derived_velocity' in migration and migration['derived_velocity']!=dict(
            contract='fetch-urdf-com-velocity-v1',
            urdf_sha256='7503ea2dc96eb3d1130c81f7ff1e397f30df46459b3dbf10cf5d0d247c5b1256'):
        raise ValueError('Unsupported derived velocity contract or calibration')
    path=Path(migration['parent_authorization_path']);parent=read(path)
    b=grant['binding'];old=b['source_sha256'];new=auth['source_sha256']
    if (sha(path)!=migration['parent_authorization_sha256'] or sha(path)!=b['authorization_sha256']
            or parent['source_sha256']!=old or parent['execution_profile']!=(pick.SOL_CARRY.name if floor else profile.name)
            or parent['batch_id']==auth['batch_id']
            or ledger.get('status') not in (('closed','stopped_budget_exhausted') if lifecycle or floor or hot else ('closed',))
            or Path(parent['output_root']).resolve()!=path.resolve().parent/'output'
            or Path(grant['parent_ledger_path']).resolve()!=Path(parent['output_root']).resolve()/'ledger.json'
            or Path(grant['parent_result_path']).resolve()!=Path(parent['output_root']).resolve()/'baseline-014/result.json'
            or not Path(grant['checkpoint']).resolve().is_relative_to(Path(parent['output_root']).resolve()/'baseline-014/cold-checkpoints')
            or Path(auth['output_root']).resolve().is_relative_to(path.resolve().parent)):
        raise ValueError('Carry migration parent provenance mismatch')
    for name in ('model','reasoning','controller_version','asset_manifest_sha256','pricing',
                 'per_case_steps_cap','per_case_decisions_cap','per_case_verification_cap'):
        if brake_budget and name=='per_case_decisions_cap':
            if parent.get(name)!=120 or auth.get(name)!=130 or migration.get('request_cap_amendment')!={'from':120,'to':130}:raise ValueError('Explicit 130 request amendment required')
        elif parent.get(name)!=auth.get(name):raise ValueError('Carry migration configuration changed: '+name)
    allowed={'src/astra_pick/memory.py','src/astra_pick/run_case.py','src/astra_pick/pick.py',
             'src/astra_pick/README.md','tests/test_astra_pick_contract.py',
             'tests/test_astra_pick_measure.py','tests/test_astra_pick_parity.py'}
    delta={n:dict(old=old.get(n),new=new.get(n)) for n in sorted(set(old)|set(new)) if old.get(n)!=new.get(n)}
    required = {'src/astra_pick/run_case.py', 'src/astra_pick/pick.py' if lifecycle or hot else 'src/astra_pick/memory.py'}
    if hot:
        allowed -= {'src/astra_pick/memory.py'}
        if 'history_feedback_audit' in migration:
            if (migration['history_feedback_audit']!='hot-profile-scope-v1'
                    or not set(delta)<={'src/astra_pick/pick.py','src/astra_pick/run_case.py',
                                       'tests/test_astra_pick_contract.py','src/astra_pick/README.md'}):
                raise ValueError('History feedback migration scope mismatch')
        if migration.get('robot_feedback_migration') == 'grasp-direction-only-v1':
            name='src/astra_pick/robot/executor.py'
            original=(path.parent/'code'/name).read_text()
            destination=Path(auth['output_root']).parent/'code'/name
            edits=[
                ('def __init__(self,*args,smooth_localization=False,**kwargs):',
                 "def __init__(self,*args,smooth_localization=False,grasp_direction_feedback=False,**kwargs):\n        if type(grasp_direction_feedback) is not bool:raise ValueError('Boolean grasp feedback required')"),
                ('        self.smooth_localization=smooth_localization',
                 '        self.smooth_localization=smooth_localization\n        self.grasp_direction_feedback=grasp_direction_feedback'),
                ('                    alignment_feedback=dict(position_error_m=pe,rotation_error_rad=re,stable=bool(stable))',
                 "                    alignment_feedback=dict(position_error_m=pe,rotation_error_rad=re,stable=bool(stable))\n                    if getattr(self,'grasp_direction_feedback',False):\n                        # Command minus measured robot FK, not an object/contact estimate.\n                        alignment_feedback['grasp_error_base_m']=(target[:3,3]-p).tolist()")]
            for before,after in edits:
                if original.count(before)!=1:raise ValueError('Grasp feedback migration parent mismatch')
                original=original.replace(before,after)
            if destination.read_text()!=original:
                raise ValueError('Grasp feedback migration changed controller behavior')
            allowed.add(name)
        elif 'robot_feedback_migration' in migration:
            raise ValueError('Unknown robot feedback migration')
        if 'derived_velocity' not in migration:
            raise ValueError('Hot recovery migration requires independently derived velocity')
    if floor:
        allowed.add('src/astra_pick/robot/clearance.py')
        required.update(('src/astra_pick/pick.py','src/astra_pick/robot/clearance.py'))
    if (set(old)!=set(new) or not required<=set(delta)
            or not set(delta)<=allowed or delta!=migration.get('source_delta')):
        raise ValueError('Carry migration source delta mismatch')
    root=path.resolve().parent/'code'
    for name,digest in old.items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or sha(source)!=digest:
            raise ValueError('Carry frozen parent source changed')
    additional=pick.Decimal(migration['additional_usd_cap'])
    cap=pick.committed_cost(ledger['entries'])+additional
    limit='40' if brake_budget else '7.3360691' if hot else '3.8187658' if floor else ('3.3413161' if lifecycle else '1.9983796')
    if (not additional.is_finite() or additional<=0 or additional>pick.Decimal(limit)
            or pick.Decimal(auth['usd_cap'])!=cap or pick.Decimal(auth['per_case_usd_cap'])!=cap):
        raise ValueError('Carry migration cumulative budget mismatch')
    if floor:
        load_local_floor_artifact(auth,grant)


def load_local_floor_artifact(auth, grant):
    migration=grant['source_migration']
    path=Path(migration['memory_artifact_path'])
    if sha(path)!=migration['memory_artifact_sha256']:
        raise ValueError('Local floor artifact file changed')
    artifact=read(path)
    if (artifact.get('contract')!='local-support-floor-artifact-v1'
            or artifact.get('checkpoint_receipt_sha256')!=grant['receipt_sha256']
            or artifact.get('source_sha256')!=auth['source_sha256']):
        raise ValueError('Local floor artifact source binding mismatch')
    return artifact


def cold_resume_seed(auth, profile):
    grant=auth.get('cold_resume')
    if grant is None:return []
    if 'user_priority' in grant and (profile!=pick.SOL_HOT_RECOVERY or
            grant['user_priority'] not in pick.RESUME_PRIORITIES):
        raise ValueError('Unsupported resume user priority')
    if auth.get('checkpoint_branch'):raise ValueError('Resume and diagnostic branch are exclusive')
    if not profile.cold_checkpoint:raise ValueError('Cold resume requires explicit profile')
    for name in ('parent_ledger','parent_result'):
        if sha(Path(grant[name+'_path']))!=grant[name+'_sha256']:
            raise ValueError('Parent closeout changed')
    ledger=read(Path(grant['parent_ledger_path']));result=read(Path(grant['parent_result_path']))
    receipt=read(Path(grant['checkpoint'])/'receipt.json')
    if sha(Path(grant['checkpoint'])/'receipt.json')!=grant['receipt_sha256']:
        raise ValueError('Checkpoint receipt changed')
    if result.get('environment_closed') is not True or result.get('status')=='running':
        raise ValueError('Parent must be closed before resuming')
    if result.get('termination_category') in ('cumulative_force_limit','official_success','time_limit','official_terminated','official_truncated'):
        raise ValueError('Official terminal parent cannot resume')
    if (receipt['steps']!=result['simulator_actions'] or receipt['requests']!=len(ledger['entries'])
            or any(e['status'] not in ('valid','invalid_call','invalid_verification','not_sent') for e in ledger['entries'])):
        raise ValueError('Checkpoint is stale or delivery unresolved; no replay')
    b=receipt['binding']
    if grant.get('source_migration'):
        validate_carry_source_migration(auth,profile,grant,ledger)
    floor=(profile==pick.SOL_LOCAL_FLOOR and b['profile']==pick.SOL_CARRY.name
           and grant.get('source_migration',{}).get('contract')=='local-support-floor-resume-v1')
    if (b!=grant['binding'] or (b['source_sha256']!=auth['source_sha256'] and not grant.get('source_migration'))
            or (b['profile']!=profile.name and not floor) or b['case_id'] not in auth['case_ids']
            or b['attempt_id']!=auth['attempt_ids'][b['case_id']]
            or b['asset_manifest_sha256']!=auth.get('asset_manifest_sha256')):
        raise ValueError('Resume source, asset or task mismatch')
    return paid_delivery_evidence(auth,profile,ledger['entries'])[0]


def paid_delivery_evidence(auth,profile,prefix):
    """An explicit, hash-bound delivered-but-never-dispatched suffix; no network."""
    grant=auth.get('paid_delivery_resume')
    if grant is None:return copy.deepcopy(prefix),None
    if profile!=pick.SOL_HOT_RECOVERY or grant.get('contract')!='paid-undispatched-response-v1':
        raise ValueError('Unsupported paid delivery recovery')
    paths={}
    for name in ('authorization','ledger','result','resource','session','bundle','delivery','wire','raw'):
        p=Path(grant[name+'_path'])
        if sha(p)!=grant[name+'_sha256']:raise ValueError('Paid delivery evidence changed: '+name)
        paths[name]=p
    old=read(paths['authorization']);ledger=read(paths['ledger']);result=read(paths['result']);resource=read(paths['resource'])
    session=read(paths['session']);bundle=read(paths['bundle']);response=read(paths['delivery'])
    entries=ledger['entries'];last=entries[-1]
    if (ledger['status']!='closed' or entries[:-1]!=prefix or len(entries)!=len(prefix)+1
            or last['status']!='valid' or last.get('request_kind')!='main'
            or result.get('environment_closed') is not True or result.get('simulator_actions')!=1906
            or result.get('error_message')!='session response not durably delivered or already executed'
            or not all(resource['own_processes_exited'].values()) or not resource['lock_available']
            or session['status']!='awaiting_execution'
            or bundle['authorization_sha256']!=sha(paths['authorization'])
            or bundle['position']!=auth['case_ids'][0] or response['status']!='valid'
            or last['provider_id']!=response['provider_id']
            or last['wire_sha256']!=sha(paths['wire'])
            or old['cold_resume']['receipt_sha256']!=auth['cold_resume']['receipt_sha256']
            or Path(grant['claim_path']).parent!=paths['delivery'].parent):
        raise ValueError('Paid response is not an exact undispatched suffix')
    receipt=session['receipts'][-1]
    if (receipt.get('execution') is not None or receipt.get('execution_sha256') is not None
            or receipt['response_id']!=response['provider_id']
            or receipt['response_sha256']!=sha(paths['raw'])
            or receipt['wire_sha256']!=sha(paths['wire'])
            or receipt['archive_sha256']!=hashlib.sha256(canonical(dict(bundle=bundle,call=response['call']))).hexdigest()):
        raise ValueError('Paid response receipt mismatch or execution uncertainty')
    return copy.deepcopy(entries),dict(session=session,bundle=bundle,response=response)


def restore_paid_delivery(state,auth,profile,fresh,output):
    prefix=read(Path(auth['cold_resume']['parent_ledger_path']))['entries']
    _,evidence=paid_delivery_evidence(auth,profile,prefix)
    bundle=evidence['bundle'];saved=evidence['session'];response=evidence['response']
    if Path(auth['paid_delivery_resume']['claim_path']).exists():
        raise ValueError('Paid response execution already claimed; no replay')
    # Rendered sensors must be identical at the physically restored boundary.
    for key,item in bundle['sensor_manifest'].items():
        if fresh['sensor_manifest'][key]['sha256']!=item['sha256']:
            raise ValueError('Paid command sensor boundary differs')
    if (bundle['turn']!=len(state.archive) or bundle['observation']['history']!=state.history
            or len(saved['receipts'])!=len(state.archive)+1):
        raise ValueError('Paid command history boundary differs')
    session=state.response_session
    session.rebind_checkpoint_authorization(bundle['authorization_sha256'],state.archive,state.history,
        checkpoint_receipt_sha256=auth['cold_resume']['receipt_sha256'])
    expected=dict(attempt_id=session.attempt_id,binding=session.binding,receipts=session.receipts,
        status='ready',authorization_lineage=session.authorization_lineage,cache_binding=session.cache_binding)
    for key,value in expected.items():
        actual=saved.get(key)
        if key=='receipts':actual=actual[:-1]
        if key=='status':actual='ready' if actual=='awaiting_execution' else actual
        if actual!=value:raise ValueError('Paid session prefix differs: '+key)
    if saved.get('single_pick')!=session.single_pick:raise ValueError('Paid recovery flags changed')
    session.receipts=copy.deepcopy(saved['receipts']);session.status='awaiting_execution';session.persist()
    save(Path(output)/'paid-delivery-validation.json',dict(sensor_hashes_equal=True,
        history_prefix_equal=True,response_id=response['provider_id'],new_api_requests=0))
    return copy.deepcopy(bundle),copy.deepcopy(response)


CHECKPOINT_BRANCH_CHANGED = frozenset((
    'src/astra_pick/pick.py','src/astra_pick/run_case.py','src/astra_pick/robot/executor.py',
    'src/astra_pick/README.md','tests/test_astra_pick_contract.py'))
CHECKPOINT_BRANCH_ADDED = frozenset((
    'research/c2/diagnostics/2026-10-12-sol-weekly-diagnostic/014-self-r3-result.json',))


def checkpoint_branch_seed(auth, profile):
    """Authorize one explicit source migration, preserving the complete parent run.

    The inherited ledger prefix is unchanged. Charges after that prefix remain
    in the parent/campaign ledger; they are not a credit to this branch.
    """
    grant=auth.get('checkpoint_branch')
    if not isinstance(grant,dict) or auth.get('cold_resume') or auth.get('pick20_continuation') or auth.get('pick20_remaining'):
        raise ValueError('Explicit exclusive checkpoint branch required')
    brake_branch=(profile==pick.SOL_HOT_RECOVERY and grant.get('migration_contract')=='rest-brake-branch-v1')
    if ((profile!=pick.SOL_BASE and not brake_branch) or not profile.cold_checkpoint or (not brake_branch and not profile.base_tracking_guard)
            or grant.get('label')!='diagnostic-checkpoint-branch'
            or auth.get('case_ids')!=['baseline-014']):
        raise ValueError('Unsupported checkpoint source migration')
    for name in ('parent_authorization','parent_ledger','parent_result'):
        if sha(Path(grant[name+'_path']))!=grant[name+'_sha256']:
            raise ValueError('Checkpoint branch parent evidence changed')
    parent=read(Path(grant['parent_authorization_path']));ledger=read(Path(grant['parent_ledger_path']))
    result=read(Path(grant['parent_result_path']));root=Path(grant['parent_authorization_path']).resolve().parent
    output=Path(parent['output_root']).resolve();checkpoint=Path(grant['checkpoint']).resolve()
    if (Path(grant['parent_output_root']).resolve()!=output or output!=root/'output'
            or Path(grant['parent_result_path']).resolve()!=output/'baseline-014'/'result.json'
            or Path(grant['parent_ledger_path']).resolve()!=output/'ledger.json'
            or not checkpoint.is_relative_to(output/'baseline-014'/'cold-checkpoints')
            or Path(auth['output_root']).resolve().is_relative_to(root)):
        raise ValueError('Checkpoint branch parent/output paths mismatch')
    receipt=read(checkpoint/'receipt.json')
    if sha(checkpoint/'receipt.json')!=grant['receipt_sha256'] or receipt.get('complete') is not True:
        raise ValueError('Checkpoint branch receipt changed or incomplete')
    b=receipt['binding'];case='baseline-014';logical=grant.get('logical_parent_attempt_id')
    if (b!=grant['binding'] or b.get('profile')!=(pick.SOL_HOT_RECOVERY.name if brake_branch else pick.SOL_SELF.name) or parent.get('execution_profile')!=b.get('profile')
            or b.get('case_id')!=case or parent.get('case_ids')!=[case]
            or b.get('authorization_sha256')!=grant['parent_authorization_sha256']
            or b.get('source_sha256')!=parent['source_sha256']
            or b.get('asset_manifest_sha256')!=auth.get('asset_manifest_sha256')
            or parent.get('asset_manifest_sha256')!=auth.get('asset_manifest_sha256')
            or logical!=b.get('attempt_id') or logical!=parent['attempt_ids'][case]
            or auth['attempt_ids'][case]!=logical
            or grant.get('parent_batch_id')!=parent['batch_id'] or parent['batch_id']==auth['batch_id']):
        raise ValueError('Checkpoint branch source, asset, case or logical session mismatch')
    if (result.get('environment_closed') is not True or result.get('status')=='running'
            or ledger.get('status')!='closed'
            or result.get('simulator_actions',-1)<receipt['steps']):
        raise ValueError('Checkpoint branch parent must be closed and retain its full outcome')
    old=parent['source_sha256'];new=auth['source_sha256'];changed={}
    for name in sorted(set(old)|set(new)):
        if old.get(name)==new.get(name):continue
        if (name not in new or (name in old and name not in (CHECKPOINT_BRANCH_CHANGED | (frozenset(('src/astra_pick/robot/clearance.py',)) if brake_branch else frozenset())))
                or (name not in old and name not in CHECKPOINT_BRANCH_ADDED)):
            raise ValueError('Unsupported checkpoint source change: '+name)
        changed[name]=dict(old=old.get(name),new=new.get(name))
    if (changed!=grant.get('source_delta') or not changed
            or not {'src/astra_pick/pick.py','src/astra_pick/run_case.py','src/astra_pick/robot/executor.py'}<=set(changed)):
        raise ValueError('Checkpoint migration delta must exactly bind changed sources')
    for name,digest in old.items():
        source=(root/'code'/name).resolve()
        if not source.is_relative_to(root/'code') or sha(source)!=digest:
            raise ValueError('Checkpoint parent frozen source changed: '+name)
    count=grant.get('prefix_requests')
    if (type(count) is not int or count<1 or count!=receipt['requests'] or count>len(ledger['entries'])
            or any(e.get('status') not in ('valid','invalid_call','invalid_verification','not_sent') for e in ledger['entries'])):
        raise ValueError('Checkpoint branch request boundary is incomplete or unknown')
    prefix=ledger['entries'][:count]
    if hashlib.sha256(canonical(prefix)).hexdigest()!=grant.get('prefix_ledger_sha256'):
        raise ValueError('Checkpoint branch request prefix changed')
    additional=pick.Decimal(grant['additional_usd_cap']);cap=pick.committed_cost(prefix)+additional
    if (not additional.is_finite() or additional<=0 or additional>pick.Decimal('3.75')
            or pick.Decimal(auth['usd_cap'])!=cap or pick.Decimal(auth['per_case_usd_cap'])!=cap
            or auth['per_case_steps_cap']!=profile.step_cap or auth['per_case_decisions_cap']!=profile.call_cap):
        raise ValueError('Checkpoint branch cumulative budget mismatch')
    if brake_branch and (grant.get('parent_requests')!=len(ledger['entries']) or grant.get('new_requests_cap')!=profile.call_cap-len(ledger['entries'])):
        raise ValueError('Brake branch must retain all consumed parent request slots')
    return copy.deepcopy(prefix)


def validate_checkpoint_branch_payload(payload, auth, profile):
    """Validate the safe prefix before restoring any native state."""
    grant=auth['checkpoint_branch'];fields=payload['executor'];state=payload['state']
    receipt=read(Path(grant['checkpoint'])/'receipt.json')
    prefix=read(Path(grant['parent_ledger_path']))['entries'][:grant['prefix_requests']]
    brake_branch=(profile==pick.SOL_HOT_RECOVERY and grant.get('migration_contract')=='rest-brake-branch-v1')
    if ((profile!=pick.SOL_BASE and not brake_branch) or payload['binding']!=grant['binding']
            or payload['runtime_state'].get('stopped') is not False
            or fields['pending'] is not None or fields['failed'] or fields['ended']
            or state.terminal_reason is not None or state.response_session.status!='ready'
            or fields['steps']!=receipt['steps'] or fields['max_steps']!=profile.step_cap
            or len(state.archive)+state.verification_requests!=receipt['requests']
            or payload['ledger']['entries']!=prefix
            or state.response_session.attempt_id!=grant['logical_parent_attempt_id']
            or state.response_session.binding.get('authorization_sha256')!=grant['parent_authorization_sha256']):
        raise ValueError('Checkpoint branch requires an exact safe completed prefix')
    # Verify all archived decisions still bind to the retained provider chain.
    state.response_session.bind(dict(position='baseline-014',authorization_sha256=grant['parent_authorization_sha256'],
        turn=len(state.archive),observation={'history':state.history}),state.archive)
    if fields.get('base_tracking_guard',False) is not False:
        raise ValueError('Checkpoint branch may only add the declared base tracking guard')
    if brake_branch:
        if fields.get('self_collision_guard',False) or fields.get('local_clearance_guard',False) or fields.get('rest_brake_in_progress',False):
            raise ValueError('Brake branch source has incompatible guards or pending brake')
        return
    if not fields.get('motion_guard') or not fields.get('self_collision_guard'):
        raise ValueError('Checkpoint branch requires preserved motion and self guards')


def restore_checkpoint_branch_python(payload, executor, auth, profile, *, session_path, authorization_sha256):
    """Retain old state, add one guard flag, and rebind the same logical session."""
    validate_checkpoint_branch_payload(payload,auth,profile)
    if auth['checkpoint_branch'].get('migration_contract')=='rest-brake-branch-v1':
        return restore_rest_brake_branch_python(payload,executor,auth,session_path=session_path,authorization_sha256=authorization_sha256)
    fields=payload['executor'];expected=set(vars(executor))-COLD_RUNTIME_PORTS
    if set(fields)-expected or expected-set(fields)-{'base_tracking_guard'}:
        raise ValueError('Checkpoint migration has unsupported executor fields')
    if executor.base_tracking_guard is not True:
        raise ValueError('Checkpoint migration destination guard missing')
    if not isinstance(authorization_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',authorization_sha256):
        raise ValueError('Exact new authorization file hash required')
    state=payload['state'];session=state.response_session
    invariant_state={k:v for k,v in vars(state).items() if k!='response_session'}
    invariant_session={k:v for k,v in vars(session).items() if k not in ('path','binding')}
    before=hashlib.sha256(cold_state_bytes((invariant_state,invariant_session,fields))).hexdigest()
    migrated=dict(payload,executor=dict(fields,base_tracking_guard=True))
    restored=restore_cold_python(migrated,executor,session_path=session_path)
    after_fields={k:getattr(executor,k) for k in fields}
    if 'base_tracking_guard' in after_fields:after_fields['base_tracking_guard']=fields['base_tracking_guard']
    after=hashlib.sha256(cold_state_bytes((
        {k:v for k,v in vars(restored).items() if k!='response_session'},
        {k:v for k,v in vars(restored.response_session).items() if k not in ('path','binding')},after_fields))).hexdigest()
    if before!=after:raise ValueError('Checkpoint migration changed controller, policy or accounting state')
    restored.response_session.rebind_checkpoint_authorization(authorization_sha256,
        restored.archive,restored.history,checkpoint_receipt_sha256=auth['checkpoint_branch']['receipt_sha256'])
    return restored,dict(from_profile=pick.SOL_SELF.name,to_profile=profile.name,
        added_executor_fields=['base_tracking_guard'],state_before_sha256=before,state_after_sha256=after,
        invariant_state_equal=True,logical_attempt_id=restored.response_session.attempt_id,
        provider_receipts_preserved=len(restored.response_session.receipts),steps=executor.steps,
        requests=len(restored.archive)+restored.verification_requests)


def restore_rest_brake_branch_python(payload, executor, auth, *, session_path, authorization_sha256):
    """Add only brake control fields; retain exact controller and complete policy."""
    fields=payload['executor'];added={'rest_depth_guard':True,'rest_brake_in_progress':False,'rest_obstruction_guard':False}
    expected=set(vars(executor))-COLD_RUNTIME_PORTS
    if set(fields)&set(added) or expected!=set(fields)|set(added):
        raise ValueError('Brake migration executor fields differ')
    if any(getattr(executor,k)!=v for k,v in added.items()):
        raise ValueError('Brake migration destination not enabled')
    before=carry_resume_digest(payload)
    restored=restore_cold_python(dict(payload,executor=dict(fields,**added)),executor,session_path=session_path)
    after=carry_resume_digest(dict(payload,state=restored,executor={k:getattr(executor,k) for k in fields}))
    if before!=after:raise ValueError('Brake migration changed retained state')
    restored.response_session.rebind_checkpoint_authorization(authorization_sha256,
        restored.archive,restored.history,checkpoint_receipt_sha256=auth['checkpoint_branch']['receipt_sha256'])
    return restored,dict(invariant_state_equal=True,state_before_sha256=before,state_after_sha256=after,
        added_executor_fields=list(added),steps=executor.steps,requests=len(restored.archive)+restored.verification_requests)

if __name__=='__main__':
    raise SystemExit(main())

