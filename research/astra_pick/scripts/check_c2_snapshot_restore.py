"""API-free original-vs-three-restores SAC control, using historical source."""
import argparse
from collections import deque
import copy
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def clone(value):
    import numpy as np
    import torch
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, (np.generic, np.random.RandomState, np.random.Generator, torch.Generator)):
        return copy.deepcopy(value)
    if value is None or isinstance(value,(str,bytes,int,float,bool,Enum)): return value
    if isinstance(value,dict): return {k:clone(v) for k,v in value.items()}
    if isinstance(value,deque): return deque((clone(v) for v in value),maxlen=value.maxlen)
    if isinstance(value,(tuple,list,set)): return type(value)(clone(v) for v in value)
    if is_dataclass(value) and not isinstance(value,type):
        return type(value)(**{f.name:clone(getattr(value,f.name)) for f in fields(value) if f.init})
    if type(value).__name__ in ('BatchedRNG','Pose'): return copy.deepcopy(value)
    raise TypeError(type(value).__name__)


def targets(base):
    result=[('adapter',base)]; obj=base.env; seen=set()
    while id(obj) not in seen:
        seen.add(id(obj));result.append((f'wrapper{len(result)}:{type(obj).__name__}',obj))
        if obj is base.uenv: break
        child=vars(obj).get('env',vars(obj).get('_env'))
        if child is None:raise RuntimeError('Unknown wrapper child: '+type(obj).__name__)
        obj=child
    ctrl=base.uenv.agent.controller
    result.append(('controller',ctrl))
    result.extend(('controller:'+name,c) for name,c in ctrl.controllers.items())
    return result


def capture(base, *, clone_fn=None):
    import numpy as np
    import torch
    clone = clone_fn or globals()['clone']
    data={};skipped={};batched_rng_states={}
    for name,obj in targets(base):
        values={};ignored={}
        for key,value in vars(obj).items():
            if type(value).__name__=='BatchedRNG':
                batched_rng_states[name+'|'+key]=[clone(r.get_state()) for r in value.rngs]
                continue
            try: values[key]=clone(value)
            except (TypeError,ValueError): ignored[key]=type(value).__name__
        data[name]=values;skipped[name]=ignored
    buffers={key:clone(getattr(base.uenv.scene.px,key).torch()) for key in (
        'cuda_rigid_body_data','cuda_articulation_qpos','cuda_articulation_qvel',
        'cuda_articulation_qf','cuda_articulation_target_qpos','cuda_articulation_target_qvel')}
    return dict(simulator=clone(base.uenv.get_state_dict()),gpu_buffers=buffers,attributes=data,skipped=skipped,batched_rng_states=batched_rng_states,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())


def local_pose_delta_to_world(base, pose_delta):
    """Map one base-local planar variable to the two world root joint axes."""
    import numpy as np
    axis, amount = pose_delta
    if axis == 2:
        return (0.0, 0.0, float(amount))
    if axis not in (0, 1):
        raise ValueError('Unknown base-local pose axis')
    matrix = np.asarray(base.uenv.agent.base_link.pose[0].sp.to_transformation_matrix())
    local = np.asarray([float(amount), 0.0] if axis == 0 else [0.0, float(amount)])
    world = matrix[:2, :2] @ local
    return (float(world[0]), float(world[1]), 0.0)


def restore(base,snap,mode='public',pose_delta=None, *, clone_fn=None):
    import numpy as np
    import torch
    clone = clone_fn or globals()['clone']
    if mode in ('qpos_only','qpos_physics_only'):
        scene=base.uenv.scene
        scene.px.cuda_articulation_qpos.torch().copy_(snap['gpu_buffers']['cuda_articulation_qpos'])
        if pose_delta is not None:
            row=int(base.uenv.agent.robot._data_index[0])
            dx,dy,dyaw=local_pose_delta_to_world(base,pose_delta)
            qpos=scene.px.cuda_articulation_qpos.torch()
            qpos[row,0]+=dx
            qpos[row,1]+=dy
            qpos[row,2]+=dyaw
        scene.px.gpu_apply_articulation_qpos()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()
        if mode=='qpos_physics_only':return
    elif mode=='buffers':
        scene=base.uenv.scene
        for key,value in snap['gpu_buffers'].items():getattr(scene.px,key).torch().copy_(value)
        scene._gpu_apply_all()
        scene.px.gpu_update_articulation_kinematics()
        scene._gpu_fetch_all()
    elif mode=='python_only':
        pass
    elif mode=='scene':
        from mani_skill.envs.sapien_env import BaseEnv
        BaseEnv.set_state_dict(base.uenv,clone(snap['simulator']))
    else:
        base.uenv.set_state_dict(clone(snap['simulator']))
    for name,obj in targets(base):
        for key,value in snap['attributes'][name].items(): setattr(obj,key,clone(value))
        for label,states in snap['batched_rng_states'].items():
            owner,key=label.split('|',1)
            if owner==name:
                rng=getattr(obj,key)
                for r,state in zip(rng.rngs,states):r.set_state(clone(state))
    base.uenv._episode_rng=base.uenv._batched_episode_rng[0]
    random.setstate(snap['python_rng']);np.random.set_state(snap['numpy_rng'])
    torch.set_rng_state(snap['torch_rng']);torch.cuda.set_rng_state_all(snap['cuda_rng'])


def max_error(a,b):
    import numpy as np
    import torch
    if isinstance(a,dict):
        if set(a)!=set(b): return float('inf')
        return max((max_error(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple,deque)):
        if len(a)!=len(b):return float('inf')
        return max((max_error(x,y) for x,y in zip(a,b)),default=0.)
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return float('inf')
    if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':return float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0))
    return 0. if np.array_equal(a,b) else float('inf')


SHADOW_SCENE_BODIES = 128
SHADOW_CONTACT_SCOPE = 'official_aggregate_robot_and_target_force_only'


def _shadow_scalar(info, key, *, boolean=False, integer=False):
    if key not in info:
        raise ValueError(f'geometry_shadow_missing_info:{key}')
    value = info[key]
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f'geometry_shadow_not_single_env:{key}')
        value = value[0]
    if boolean:
        if not isinstance(value, bool):
            raise ValueError(f'geometry_shadow_not_bool:{key}')
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'geometry_shadow_not_finite_number:{key}')
    if integer:
        if int(value) != value:
            raise ValueError(f'geometry_shadow_not_integer:{key}')
        return int(value)
    return float(value)


def _shadow_scene(scene, expected_ids=None):
    if not isinstance(scene, dict) or len(scene) != SHADOW_SCENE_BODIES:
        raise ValueError('geometry_shadow_scene_body_count')
    ids = frozenset(scene)
    if expected_ids is not None and ids != expected_ids:
        raise ValueError('geometry_shadow_scene_body_ids_changed')
    for key, body in scene.items():
        if not isinstance(key, str) or not isinstance(body, dict):
            raise ValueError('geometry_shadow_bad_body_entry')
        pose = body.get('pose_world')
        if (not isinstance(pose, list) or len(pose) != 4 or
                any(not isinstance(row, list) or len(row) != 4 for row in pose) or
                any(not isinstance(value, (int, float)) or not math.isfinite(value)
                    for row in pose for value in row)):
            raise ValueError(f'geometry_shadow_bad_pose:{key}')
        if not isinstance(body.get('dynamic'), bool) or not isinstance(body.get('body_type'), str):
            raise ValueError(f'geometry_shadow_bad_body_type:{key}')
        if body['dynamic']:
            for field in ('linear_velocity_world', 'angular_velocity_world'):
                values = body.get(field)
                if (not isinstance(values, list) or len(values) != 3 or
                        any(not isinstance(v, (int, float)) or not math.isfinite(v)
                            for v in values)):
                    raise ValueError(f'geometry_shadow_bad_velocity:{key}:{field}')
    return ids


def _shadow_row(step, info, pointer, scene, expected_ids=None):
    ids = _shadow_scene(scene, expected_ids)
    forces = {key: _shadow_scalar(info, key) for key in
              ('robot_force', 'robot_cumulative_force', 'robot_target_pairwise_force')}
    if any(value < 0 for value in forces.values()):
        raise ValueError('geometry_shadow_negative_force')
    official = {key: _shadow_scalar(info, key, boolean=True) for key in
                ('is_grasped', 'cumulative_force_within_limit', 'fail', 'success')}
    official.update({key: _shadow_scalar(info, key, integer=True) for key in
                     ('subtask', 'subtasks_steps_left')})
    if not isinstance(pointer, int):
        raise ValueError('geometry_shadow_bad_pointer')
    return ids, {'step': step, 'task_pointer': pointer, 'official': official,
                 'contact_summary': {**forces, 'scope': SHADOW_CONTACT_SCOPE,
                                     'robot_force_nonzero': forces['robot_force'] > 0,
                                     'target_pairwise_force_nonzero':
                                         forces['robot_target_pairwise_force'] > 0},
                 'scene_body_states': scene}


def run_geometry_shadow(base, *, settled_info, steps, output, report,
                        scene_registry, jsonable, scalar, started, seconds_limit):
    """Record a no-policy counterfactual; never certify candidate motion."""
    output = Path(output)
    artifact = output / 'geometry-shadow-trajectory.jsonl'
    shadow = {'status': 'running', 'steps_requested': steps, 'steps_recorded': 0,
              'scene_bodies_expected': SHADOW_SCENE_BODIES,
              'contact_scope': SHADOW_CONTACT_SCOPE,
              'dynamic_motion_proven': False,
              'scope': 'zero_action_counterfactual_only_not_SAC_motion'}
    report['geometry_shadow'] = shadow
    try:
        remaining = _shadow_scalar(settled_info, 'subtasks_steps_left', integer=True)
        if steps <= 0 or steps != remaining or steps > 200:
            raise ValueError('geometry_shadow_steps_must_equal_official_pick_remaining')
        if _shadow_scalar(settled_info, 'is_grasped', boolean=True):
            raise ValueError('geometry_shadow_starts_holding_object')
        pointer = int(scalar(base.uenv.subtask_pointer))
        ids = None
        with artifact.open('w', encoding='utf-8') as stream:
            for step in range(steps + 1):
                if step:
                    if time.monotonic() - started > seconds_limit:
                        raise TimeoutError('Geometry shadow process time limit')
                    transition = base.step((0.0,) * 13)
                    report['simulator_actions'] += 1
                    info = jsonable(transition.info)
                else:
                    info = settled_info
                ids, row = _shadow_row(step, info,
                                       int(scalar(base.uenv.subtask_pointer)),
                                       scene_registry(base.uenv), ids)
                if row['task_pointer'] != pointer:
                    raise ValueError('geometry_shadow_task_pointer_changed')
                if row['official']['subtasks_steps_left'] != remaining - step:
                    raise ValueError('geometry_shadow_official_clock_diverged')
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                        separators=(',', ':')) + '\n')
                stream.flush()
                if step:
                    shadow['steps_recorded'] = step
                if row['official']['is_grasped'] or not row['official']['cumulative_force_within_limit']:
                    raise ValueError('geometry_shadow_strict_state_changed')
                if step < steps and (row['official']['fail'] or row['official']['success'] or base.ended):
                    raise ValueError('geometry_shadow_early_terminal')
        shadow.update(status='completed_diagnostic_only',
                      scene_body_ids_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      artifact=artifact.name,
                      final_official=row['official'])
    except Exception as exc:
        shadow.update(status='incomplete', error=f'{type(exc).__name__}: {exc}')
        if artifact.exists():
            shadow['sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            shadow['artifact'] = artifact.name
        raise
    return shadow


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--pilot-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--restore-mode',choices=('public','scene','buffers','qpos_only','qpos_physics_only','python_only'),default='public')
    p.add_argument('--reference-only',action='store_true')
    p.add_argument('--immediate-write',action='store_true')
    p.add_argument('--settle-actions',type=int,choices=(0,1),default=0)
    p.add_argument('--geometry-candidate',choices=('C0','C1','C2','C3','C4','C5','C6'))
    p.add_argument('--geometry-run-sac',action='store_true')
    p.add_argument('--a1-c0-repeat',action='store_true',
                   help='Run a C0 repeat despite a base XY settle drift over the A2 intervention limit; record the drift')
    p.add_argument('--paired-c0-summary',type=Path,
                   help='For A2, compare the settled candidate XY pose with settled C0 plus the frozen requested world delta')
    p.add_argument('--export-pre-sac-images',action='store_true',
                   help='Save post-settle camera PNGs before any SAC action, including image-only prechecks')
    p.add_argument('--geometry-shadow-steps',type=int,default=0,
                   help='Independent zero-action diagnostic; must equal the post-settle official Pick steps left')
    p.add_argument('--export-collision-bundle',action='store_true',
                   help='Write a read-only post-settle collision and state bundle for a frozen seed8 candidate')
    p.add_argument('--seconds-limit',type=int,default=480)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.restore_mode in ('qpos_only','qpos_physics_only','python_only') and not (a.reference_only and a.immediate_write):
        p.error('Partial modes are immediate identity-write diagnostics, not snapshot restores')
    if a.settle_actions and not (a.reference_only and a.immediate_write and a.restore_mode=='qpos_physics_only'):
        p.error('Settling is only defined for fresh-process minimal qpos C0 controls')
    if a.geometry_candidate and not (a.reference_only and a.immediate_write and a.restore_mode=='qpos_physics_only' and a.settle_actions==1):
        p.error('Geometry precheck requires the frozen C0 minimal write and one settling action')
    if a.geometry_run_sac and not a.geometry_candidate:
        p.error('A2 SAC requires a frozen C0-C6 candidate')
    if a.a1_c0_repeat and not (a.geometry_candidate=='C0' and
            (a.geometry_run_sac or a.export_pre_sac_images) and
            a.reference_only and a.immediate_write and a.settle_actions==1 and
            a.restore_mode=='qpos_physics_only'):
        p.error('A1 C0 repeat requires the frozen C0 treatment and SAC')
    if a.export_pre_sac_images and not a.geometry_candidate:
        p.error('Pre-SAC image export requires a frozen C0-C6 candidate')
    if a.paired_c0_summary and (a.geometry_candidate not in ('C1','C2','C3','C4','C5','C6') or
            a.a1_c0_repeat or not (a.reference_only and a.immediate_write and
            a.settle_actions==1 and a.restore_mode=='qpos_physics_only')):
        p.error('Paired C0 settle check requires a frozen new candidate and the same treatment')
    if a.geometry_shadow_steps < 0 or (a.geometry_shadow_steps and
            (not a.geometry_candidate or a.geometry_run_sac)):
        p.error('Geometry shadow needs C0-C6 precheck and cannot run SAC')
    if a.export_collision_bundle and a.geometry_candidate not in ('C0','C1','C2','C3','C4','C5','C6'):
        p.error('Collision bundle export requires a frozen C0-C6 geometry precheck')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic();base=None;bundle_dir=None
    report=dict(status='running',simulator_actions=0,policy_inferences=0,provider_attempts=0,restores=[])
    report['restore_mode']=a.restore_mode
    report['reference_only']=a.reference_only
    report['immediate_write']=a.immediate_write
    report['settle_actions']=a.settle_actions
    report['geometry_candidate']=a.geometry_candidate
    report['geometry_run_sac']=a.geometry_run_sac
    report['a1_c0_repeat']=a.a1_c0_repeat
    report['paired_c0_summary']=str(a.paired_c0_summary) if a.paired_c0_summary else None
    report['export_pre_sac_images']=a.export_pre_sac_images
    report['geometry_shadow_steps']=a.geometry_shadow_steps
    def save():
        report['wall_seconds']=time.monotonic()-started
        (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-b7ebba23-7824-7601-df32-be55628936c3' or os.environ.get('PYTHONHASHSEED')!='0':
            raise RuntimeError('GPU1 / PYTHONHASHSEED guards not satisfied')
        import numpy as np
        import torch
        from bvi.mshab_adapter import make_mshab_adapter,jsonable,scalar
        from bvi.logging import JsonlLogger
        from bvi.goal_tools import GoalToolAdapter,GoalRLSkill
        from bvi.protocol import SkillRequest,Requirement,SkillStatus
        from mshab.envs.make import EnvConfig
        import bvi.nav_camera_env
        import bvi.mshab_adapter as adapter_module
        import mshab.envs.sequential_task as runtime_module
        proposal=json.loads(a.audit.read_text())['pilot_proposals'][a.pilot_index]
        src=Path(proposal['source_directory']);meta=json.loads((src/'run-metadata.json').read_text())
        report['proposal']=proposal
        for path,expected in [(Path(adapter_module.__file__),meta['runtime_source_sha256']['src/bvi/mshab_adapter.py']),
                              (Path(runtime_module.__file__),meta['mshab_runtime']['sha256'])]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Historical source hash mismatch')
        seed=proposal['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
        cfg=EnvConfig(**meta['config']);cfg.record_video=False
        base=make_mshab_adapter(cfg,JsonlLogger(a.output/'events.jsonl','snapshot-control'),a.output,seed=seed)
        original=json.loads((src/'initial-state.json').read_text())
        digest=hashlib.sha256(json.dumps(jsonable(base.uenv.get_state_dict()),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
        report['initial_hash_match']=digest==original['state_sha256']
        if not report['initial_hash_match']:raise RuntimeError('Initial hash differs; no approximate seed reconstruction')
        events=[json.loads(line) for line in (src/'events.jsonl').read_text().splitlines()]
        recorded=[e for e in events if e['event']=='mshab_step']
        for event in recorded[:proposal['preceding_steps']]:
            t=base.step(tuple(event['controller_action'][0]));report['simulator_actions']+=1
            for key in ('subtask','fail','success','is_grasped','cumulative_force_within_limit'):
                if jsonable(t.info.get(key))!=event['info'].get(key):raise RuntimeError('Historical prefix categorical divergence')
        adapter=GoalToolAdapter(base);q=proposal['request']
        request=SkillRequest('snapshot-pick','pick',q['target_id'],adapter.observe().frame_id,
                             (Requirement('benchmark_success','benchmark_success'),),200,180.)
        skill=GoalRLSkill('pick',adapter,Path(os.environ.get('BVI_LAB_ROOT') or Path.home()/'bvi-research')/'checkpoints'/'mshab','rl_per_obj')
        skill.start(request,adapter.observe())
        if skill.selected_index!=int(scalar(base.uenv.subtask_pointer)):
            raise RuntimeError('Requested object differs from native task pointer')
        snap=capture(base);torch.save(snap,a.output/'full-handoff-capture.pt')
        report['batched_rng_fields']=sorted(snap['batched_rng_states'])
        (a.output/'capture-inventory.json').write_text(json.dumps({
            'saved':{k:sorted(v) for k,v in snap['attributes'].items()},'skipped':snap['skipped']},indent=2))
        def rollout(label):
            rows=[];status='timed_out'
            for step in range(200):
                if time.monotonic()-started>a.seconds_limit:raise TimeoutError('Process time limit')
                inp=adapter.conditioned_policy(adapter.observe(),skill.selected_index).policy
                state_input=clone(inp['state'])
                pixel_hash={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in inp['pixels'].items()}
                action=skill.act(adapter.observe());report['policy_inferences']+=1
                transition=adapter.step(action);report['simulator_actions']+=1
                fb=skill.feedback(request,transition)
                rows.append(dict(action=list(action),conditioned_state=state_input,pixel_sha256=pixel_hash,qpos=jsonable(base.uenv.agent.robot.get_qpos()),
                    qvel=jsonable(base.uenv.agent.robot.get_qvel()),info=jsonable(transition.info),
                    simulator=clone(base.uenv.get_state_dict())))
                if fb.status is not SkillStatus.EXECUTING or base.ended:
                    status=fb.status.value;break
            torch.save(rows,a.output/f'{label}-trajectory.pt')
            return rows,dict(status=status,steps=len(rows),final_info=rows[-1]['info'])
        if a.immediate_write:
            delta={'C0':None,'C1':(0,0.20),'C2':(0,-0.20),
                   'C3':(1,0.20),'C4':(1,-0.20),
                   'C5':(2,0.35),'C6':(2,-0.35)}[a.geometry_candidate] if a.geometry_candidate else None
            start_base=clone(base.uenv.agent.base_link.pose.p)
            start_qpos=clone(base.uenv.agent.robot.get_qpos())
            if a.export_collision_bundle:
                from export_c2_collision_bundle import capture_reference
                reference_capture=capture_reference(base.uenv)
            world_delta=local_pose_delta_to_world(base,delta) if delta is not None else (0.0,0.0,0.0)
            restore(base,snap,a.restore_mode,delta)
            report['immediate_write_state_error']=max_error(snap['simulator'],base.uenv.get_state_dict()) if delta is None else None
            if a.geometry_candidate:
                from c2_floor_triangles import FloorTriangles
                u=base.uenv
                build=u.scene_builder.build_configs[int(u.build_config_idxs[0])]
                floor_path=Path(os.environ['MS_ASSET_DIR'])/'data/scene_datasets/replica_cad_dataset/configs/scenes'/(Path(build).stem+'.fetch.navigable_positions_simplified.obj')
                floor=FloorTriangles(floor_path)
                edited_base=clone(u.agent.base_link.pose.p)
                edited_qpos=clone(u.agent.robot.get_qpos())
                report['geometry']=dict(candidate_id=a.geometry_candidate,
                    requested_axis_delta=None if delta is None else list(delta),
                    coordinate_frame='base_link_local',
                    applied_world_xy_yaw_delta=list(world_delta),
                    floor_mesh=str(floor_path),floor_sha256=floor.sha256,
                    floor_origin_clear=floor.contains_segment(start_base[0,:2].detach().cpu().tolist(),start_base[0,:2].detach().cpu().tolist()),
                    floor_endpoint_clear=floor.contains_segment(edited_base[0,:2].detach().cpu().tolist(),edited_base[0,:2].detach().cpu().tolist()),
                    floor_segment_clear=floor.contains_segment(start_base[0,:2].detach().cpu().tolist(),edited_base[0,:2].detach().cpu().tolist()),
                    base_world_before=start_base[0].detach().cpu().tolist(),
                    base_world_after_edit=edited_base[0].detach().cpu().tolist(),
                    robot_qpos_before=start_qpos[0].detach().cpu().tolist(),
                    robot_qpos_after_edit=edited_qpos[0].detach().cpu().tolist(),
                    other_qpos_max_error=max_error(start_qpos[:,3:],edited_qpos[:,3:]))
            if a.settle_actions:
                before_qpos=clone(base.uenv.agent.robot.get_qpos())
                before_pointer=int(scalar(base.uenv.subtask_pointer))
                settled=base.step((0.0,)*13)
                report['simulator_actions']+=1
                settle_info=jsonable(settled.info)
                after_qpos=clone(base.uenv.agent.robot.get_qpos())
                base_xy_drift=float(torch.linalg.vector_norm(
                    after_qpos[0,:2]-before_qpos[0,:2]))
                base_yaw_drift=float(torch.abs(after_qpos[0,2]-before_qpos[0,2]))
                arm_drift=max_error(before_qpos[:,3:10],after_qpos[:,3:10])
                paired_xy_residual=None
                if a.paired_c0_summary:
                    c0_bytes=a.paired_c0_summary.read_bytes()
                    c0=json.loads(c0_bytes)
                    if c0.get('status')!='completed_reference_only' or c0.get('geometry_candidate')!='C0' or not c0.get('a1_c0_repeat'):
                        raise RuntimeError('Paired baseline is not the verified seed9 C0 repeat')
                    if c0.get('proposal')!=proposal or c0.get('initial_hash_match') is not True:
                        raise RuntimeError('Paired baseline proposal or initial state differs')
                    reference_xy=c0['geometry']['base_world_after_settle'][:2]
                    actual_xy=base.uenv.agent.base_link.pose.p[0,:2].detach().cpu().tolist()
                    target_xy=[reference_xy[i]+world_delta[i] for i in range(2)]
                    paired_xy_residual=float(sum((actual_xy[i]-target_xy[i])**2 for i in range(2))**0.5)
                    report['paired_c0_settle']=dict(
                        reference_summary_sha256=hashlib.sha256(c0_bytes).hexdigest(),
                        reference_settled_xy=reference_xy,
                        expected_candidate_settled_xy=target_xy,
                        actual_candidate_settled_xy=actual_xy,
                        residual_m=paired_xy_residual,
                        residual_limit_m=0.03)
                report['stabilization']=dict(action=[0.0]*13,info=settle_info,
                    qpos_max_delta=max_error(before_qpos,after_qpos),
                    arm_qpos_max_delta=arm_drift,
                    base_xy_drift_m=base_xy_drift,
                    base_xy_drift_exceeds_a2_limit=base_xy_drift>0.03,
                    a1_c0_repeat_xy_override=bool(a.a1_c0_repeat and base_xy_drift>0.03),
                    base_yaw_drift_rad=base_yaw_drift,
                    pose_drift_limit=0.03,
                    held_after=settle_info.get('is_grasped'),
                    pointer_unchanged=int(scalar(base.uenv.subtask_pointer))==before_pointer)
                if (not report['stabilization']['pointer_unchanged'] or
                    bool(scalar(settled.info.get('fail',False))) or
                    bool(scalar(settled.info.get('is_grasped',False))) or
                    not bool(scalar(settled.info.get('cumulative_force_within_limit',True))) or
                    (base_xy_drift>0.03 and not a.a1_c0_repeat and
                     (paired_xy_residual is None or paired_xy_residual>0.03)) or
                    base_yaw_drift>0.03 or arm_drift>0.03 or
                    base.ended):
                    report['status']='stabilization_infeasible'
                    report['candidate_scan_admitted']=False
                    if a.export_collision_bundle:
                        report['collision_bundle']={'status':'not_exported_stabilization_infeasible'}
                    return
            if a.geometry_candidate:
                u=base.uenv;obj=u.subtask_objs[skill.selected_index]
                distance=torch.norm(obj.pose.p[:,:2]-u.agent.base_link.pose.p[:,:2],dim=1)
                rel=(u.agent.base_link.pose.inv()*obj.pose).p
                angle=torch.atan2(rel[:,1],rel[:,0]).abs()
                geometry=report['geometry']
                geometry['object_distance_m']=float(distance[0])
                geometry['object_angle_rad']=float(angle[0])
                geometry['official_nav_distance_limit_m']=float(u.navigate_cfg.navigated_successfully_dist)
                geometry['official_nav_angle_limit_rad']=float(u.navigate_cfg.navigated_successfully_rot)
                geometry['target_close']=bool(distance[0]<=u.navigate_cfg.navigated_successfully_dist)
                geometry['target_oriented']=bool(angle[0]<=u.navigate_cfg.navigated_successfully_rot)
                geometry['base_world_after_settle']=u.agent.base_link.pose.p[0].detach().cpu().tolist()
                from c2_pairwise_contacts import audit_pairwise_contacts
                from c2_mesh_collisions import audit_mesh_collisions
                geometry['collision_meshes']=audit_mesh_collisions(u)
                geometry['pairwise_contacts']=audit_pairwise_contacts(
                    u,selected_pairs=geometry['collision_meshes']['overlaps'])
                geometry['contact_query_available']=(
                    geometry['pairwise_contacts']['status']=='measured')
                # Impulse queries do not measure positive clearance or prove
                # that no contact happened during the stabilization step.
                geometry['full_body_collision_certified']=False
                if a.export_collision_bundle:
                    from export_c2_collision_bundle import export_bundle
                    bundle_dir=a.output/'collision-bundle'
                    try:
                        report['collision_bundle']=export_bundle(
                            u,base.capture,bundle_dir,
                            candidate=a.geometry_candidate,proposal=proposal,
                            reference=reference_capture)
                    except Exception as exc:
                        report['collision_bundle']={
                            'status':'export_error',
                            'error':f'{type(exc).__name__}: {exc}',
                            'path':str(bundle_dir)}
                report['status']='completed_geometry_precheck'
                report['candidate_scan_admitted']=False
                if a.export_pre_sac_images:
                    camera_dir=a.output/'pre-sac-images'
                    camera_dir.mkdir()
                    report['pre_sac_images']={}
                    for frame in adapter.observe().images:
                        frame.validate()
                        path=camera_dir/(frame.camera+'.png')
                        path.write_bytes(frame.data)
                        report['pre_sac_images'][frame.camera]={
                            'path':str(path.relative_to(a.output)),
                            'sha256':hashlib.sha256(frame.data).hexdigest(),
                            'bytes':len(frame.data)}
                if a.geometry_shadow_steps:
                    from export_c2_collision_bundle import _scene_registry
                    run_geometry_shadow(
                        base, settled_info=report['stabilization']['info'],
                        steps=a.geometry_shadow_steps, output=a.output,
                        report=report, scene_registry=_scene_registry,
                        jsonable=jsonable, scalar=scalar,
                        started=started, seconds_limit=a.seconds_limit)
                    report['status']='completed_geometry_shadow_diagnostic'
                    return
                if not a.geometry_run_sac:
                    return
                report['candidate_scan_admitted']=True
        reference,report['reference']=rollout('unrestored-reference');save()
        if a.reference_only:
            report['status']='completed_reference_only'
            report['candidate_scan_admitted']=bool(a.geometry_run_sac and a.geometry_candidate)
            return
        for repeat in range(3):
            restore(base,snap,a.restore_mode)
            state_error=max_error(snap['simulator'],base.uenv.get_state_dict())
            obs_error=max_error(snap['attributes']['adapter']['_observation'].policy,base.observe().policy)
            row=dict(repeat=repeat,restore_state_max_error=state_error,restore_observation_max_error=obs_error)
            report['restores'].append(row)
            if state_error>1e-6 or obs_error!=0:raise RuntimeError('Snapshot round-trip gate failed')
            trajectory,outcome=rollout(f'restored-{repeat}')
            row.update(outcome)
            n=min(len(reference),len(trajectory))
            row['action_max_error']=max(max_error(reference[i]['action'],trajectory[i]['action']) for i in range(n))
            row['first_action_max_error']=max_error(reference[0]['action'],trajectory[0]['action'])
            row['first_conditioned_state_error']=max_error(reference[0]['conditioned_state'],trajectory[0]['conditioned_state'])
            row['first_pixels_equal']=reference[0]['pixel_sha256']==trajectory[0]['pixel_sha256']
            row['first_pixel_divergence_step']=next((i+1 for i in range(n) if reference[i]['pixel_sha256']!=trajectory[i]['pixel_sha256']),None)
            row['qpos_max_error']=max(max_error(reference[i]['qpos'],trajectory[i]['qpos']) for i in range(n))
            row['state_max_error']=max(max_error(reference[i]['simulator'],trajectory[i]['simulator']) for i in range(n))
            row['force_max_error']=max(max_error(reference[i]['info']['robot_force'],trajectory[i]['info']['robot_force']) for i in range(n))
            keys=('subtask','fail','success','is_grasped','cumulative_force_within_limit')
            row['categorical_trajectory_equal']=len(reference)==len(trajectory) and all(
                reference[i]['info'].get(k)==trajectory[i]['info'].get(k) for i in range(n) for k in keys)
            row['same_terminal_status']=outcome['status']==report['reference']['status'];save()
        report['repeat_outcomes_stable']=all(r['categorical_trajectory_equal'] and r['same_terminal_status'] for r in report['restores'])
        report['bit_exact_actions']=all(r['action_max_error']==0 for r in report['restores'])
        report['status']='completed_controls'
        report['candidate_scan_admitted']=False
        report['note']='Control evidence only; assess numerical drift and historical final outcome before candidate admission'
    except Exception as exc:
        import traceback
        report['status']='error';report['error']=f'{type(exc).__name__}: {exc}'
        (a.output/'error.txt').write_text(traceback.format_exc())
    finally:
        if base is not None:base.close()
        save()
        if bundle_dir is not None:
            from export_c2_collision_bundle import finalize_summary_hash
            finalize_summary_hash(bundle_dir,a.output/'summary.json')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
