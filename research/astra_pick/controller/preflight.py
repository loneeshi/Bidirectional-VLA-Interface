"""CPU-importable preparation ports. No environment creation, reset or CLI launch.

All state/camera material is evaluation-only. run_prepared_arm is a future physical
entry point and must only be called by an explicitly authorized laboratory runner.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CONTRACT = 'bvi-eval-snapshot-transport-v2'


def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, 'shape'): return array(value).tolist()
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def driver_readback(u):
    """Read actual active configuration; defaults in saved SDK are NOT attestation."""
    from astra_pick.robot.eef_tools import validate_official_controller
    from astra_pick.robot.motion import validate_base_runtime
    controller = u.agent.controller
    validate_official_controller(controller)
    validate_base_runtime(controller, u.control_freq)
    if u.sim_freq != 100 or u.control_freq != 20:
        raise ValueError('driver frequency mismatch')
    if u.pick_cfg.robot_cumulative_force_limit != 5000 or u.pick_cfg.ee_rest_thresh != .05:
        raise ValueError('official criteria mismatch')
    result = dict(sim_freq=u.sim_freq, control_freq=u.control_freq, substeps=5,
                  official_force_limit=5000, ee_rest_thresh=.05,
                  action_mapping=plain(controller.action_mapping), controllers={})
    for name in ('arm', 'body', 'gripper', 'base'):
        config = controller.controllers[name].config
        expected = dict(damping=1000 if name == 'base' else 100,
                        force_limit=500 if name == 'base' else 100,
                        friction=0., normalize_action=True)
        if name != 'base':
            expected.update(stiffness=1000, drive_mode='force', interpolate=False,
                            use_target=False, use_delta=name != 'gripper')
        for key, wanted in expected.items():
            if not hasattr(config, key) or not np.all(array(getattr(config, key)) == wanted):
                raise ValueError(f'driver {name}.{key} mismatch')
        keys = tuple(expected) + ('joint_names', 'lower', 'upper')
        result['controllers'][name] = {k: plain(getattr(config, k)) for k in keys}
        result['controllers'][name]['class'] = type(controller.controllers[name]).__name__
    return result


def world_to_opencv(world_from_sapien):
    from astra_pick.robot.eef_tools import transform
    # SAPIEN camera axes x=forward,y=left,z=up; OpenCV z=forward,x=right,y=down.
    basis = np.array([[0., -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    return basis @ np.linalg.inv(transform(world_from_sapien))



class CameraPoseError(ValueError):
    """Evaluation-camera failure with replayable CPU evidence; never a command pose."""
    def __init__(self, receipt, raw_matrix):
        super().__init__('invalid evaluation camera pose: ' + ', '.join(receipt['failed_checks']))
        self.receipt, self.raw_matrix = diagnostic_json(receipt), np.asarray(raw_matrix).copy()


def camera_pose_to_opencv(raw_matrix, position, quaternion_wxyz, camera_id):
    """Use SDK p/q in float64; validate float32 SDK serialization independently.

    16 eps allows bounded arithmetic roundoff in the quaternion matrix formula;
    32 eps bounds its Gram/determinant residual. These are evaluation-camera
    serialization gates, not robot command, snapshot, or restoration tolerances.
    """
    from astra_pick.robot.eef_tools import transform
    from scipy.spatial.transform import Rotation
    raw = np.asarray(raw_matrix)
    p, q = np.asarray(position), np.asarray(quaternion_wxyz)
    t = raw.astype(np.float64)
    eps = float(np.finfo(np.float32).eps)
    checks = dict(shape=raw.shape == (4, 4), dtype=raw.dtype == np.dtype('float32'),
                  finite=bool(np.isfinite(t).all()), position_shape=p.shape == (3,),
                  quaternion_shape=q.shape == (4,),
                  position_finite=bool(np.isfinite(p).all()), quaternion_finite=bool(np.isfinite(q).all()))
    receipt = dict(schema='eval-camera-pose-v1', camera_id=str(camera_id),
                   raw_matrix=plain(raw), raw_dtype=str(raw.dtype), raw_shape=list(raw.shape),
                   source_position=plain(p), source_quaternion_wxyz=plain(q),
                   source_position_dtype=str(p.dtype), source_quaternion_dtype=str(q.dtype),
                   thresholds=dict(quaternion_norm_abs=8*eps, matrix_abs=16*eps,
                                   gram_abs=32*eps, determinant_abs=32*eps, homogeneous_abs=0., rtol=0.),
                   checks=checks, residuals={})
    if checks['shape'] and checks['finite']:
        gram = t[:3, :3].T @ t[:3, :3] - np.eye(3)
        det = float(np.linalg.det(t[:3, :3]))
        receipt['residuals'].update(gram=gram.tolist(), gram_max_abs=float(np.max(abs(gram))),
            determinant=det, determinant_abs=abs(det-1), homogeneous=(t[3]-[0,0,0,1]).tolist())
        checks.update(homogeneous=bool(np.array_equal(t[3], [0,0,0,1])),
                      gram=bool(np.max(abs(gram)) <= 32*eps), determinant=abs(det-1) <= 32*eps)
        receipt['legacy_checks'] = dict(
            homogeneous=bool(np.allclose(t[3], [0,0,0,1], atol=1e-8)),
            gram=bool(np.allclose(gram+np.eye(3), np.eye(3), atol=1e-7)),
            determinant=bool(np.isclose(det, 1, atol=1e-7)))
    canonical = None
    if all(checks[k] for k in ('position_shape','quaternion_shape','position_finite','quaternion_finite')):
        norm = float(np.linalg.norm(q.astype(np.float64)))
        receipt['residuals']['quaternion_norm_abs'] = abs(norm-1)
        checks['quaternion_norm'] = abs(norm-1) <= 8*eps
        if checks['quaternion_norm']:
            canonical = np.eye(4)
            # SDK order is wxyz. Rotation normalizes only after the explicit norm gate.
            canonical[:3,:3] = Rotation.from_quat(q.astype(np.float64)[[1,2,3,0]]).as_matrix()
            canonical[:3,3] = p
            if checks['shape'] and checks['finite']:
                delta = t-canonical
                receipt['residuals']['matrix_delta'] = delta.tolist()
                receipt['residuals']['matrix_max_abs'] = float(np.max(abs(delta)))
                checks['matrix_agrees_with_pq'] = bool(np.max(abs(delta)) <= 16*eps)
    receipt['failed_checks'] = [key for key, passed in checks.items() if not passed]
    if receipt['failed_checks'] or canonical is None:
        raise CameraPoseError(receipt, raw)
    transform(canonical)  # Preserve the existing strict command-pose validator.
    receipt.update(canonical_world_from_camera=canonical.tolist(), passed=True)
    return world_to_opencv(canonical), receipt


def diagnostic_json(value):
    """Portable JSON for failed NaN/Inf inputs; exact bytes retained in .npy."""
    if isinstance(value, dict): return {k: diagnostic_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [diagnostic_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return repr(value)
    return value


def native_capture(u):
    """Same non-evaluate sensor pathway as run_case; no get_info()/render()."""
    data = u._get_obs_sensor_data()
    u.scene.update_render(update_sensors=False, update_human_render_cameras=True)
    images = u.scene.get_human_render_camera_images()
    cameras = u._human_render_cameras
    if len(images) != 1 or len(cameras) != 1 or set(images) != set(cameras):
        raise ValueError('exactly one identified evaluator camera required')
    key = next(iter(cameras))
    camera = cameras[key].camera
    sdk_pose = camera.global_pose[0].sp
    pose = sdk_pose.to_transformation_matrix()
    camera_transform, camera_receipt = camera_pose_to_opencv(pose, sdk_pose.p, sdk_pose.q, key)
    k = array(camera.get_intrinsic_matrix())
    if k.shape == (1, 3, 3): k = k[0]
    result = dict(scene=array(images[key]), K=k,
                  world_to_camera=camera_transform, camera_pose_receipt=camera_receipt,
                  tcp_world=array(u.agent.tcp.pose[0].sp.to_transformation_matrix())[:3, 3],
                  base_world=array(u.agent.base_link.pose[0].sp.to_transformation_matrix()),
                  qpos=array(u.agent.robot.qpos)[0], qvel=array(u.agent.robot.qvel)[0])
    for short, name in (('head', 'fetch_head'), ('hand', 'fetch_hand')):
        result[short] = array(data[name]['rgb'])
        result[short + '_depth'] = array(data[name]['depth'])
    return result


def exact_tree_equal(a, b):
    """No tolerance for recorder-induced state changes."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(exact_tree_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, 'shape') or hasattr(b, 'shape'):
        return np.array_equal(array(a), array(b), equal_nan=True)
    return type(a) == type(b) and a == b


class Recorder:
    """Incremental raw frames + JSON manifest; terminal/error frames are retained."""
    def __init__(self, root, case_id, condition, capture, state, *, fps=20, action_size=13):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.case_id, self.condition = case_id, condition
        if action_size not in (13,20):raise ValueError('unsupported recorder action dimension')
        self.action_size=action_size
        self.capture, self.state, self.fps = capture, state, fps
        self.frames, self.trajectory = [], []

    def record(self, step, action, info, *, phase='execution', outcome='in_progress'):
        if type(step) is not int or step != len(self.frames): raise ValueError('nonconsecutive capture')
        if action is not None:
            action = np.asarray(action, float)
            if action.shape != (self.action_size,) or not np.isfinite(action).all() or np.max(abs(action)) > 1 + 1e-8:
                raise ValueError('invalid recorded action')
        before = copy.deepcopy(self.state())
        try:
            data = self.capture()
        except CameraPoseError as error:
            raw_path = self.root / f'{step:04d}-camera-pose-failure.npy'
            np.save(raw_path, error.raw_matrix, allow_pickle=False)
            evidence = dict(case_id=self.case_id, condition=self.condition, step=step,
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase,
                actions=plain(action), official_info_eval_only=plain(info),
                camera=error.receipt, matrix_file=raw_path.name,
                matrix_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                frame_completed=False)
            try: evidence['capture_state_unchanged'] = exact_tree_equal(before, self.state())
            except Exception as state_error: evidence['state_check_error'] = repr(state_error)
            save(self.root / f'{step:04d}-camera-pose-failure.json', diagnostic_json(evidence))
            raise
        if not exact_tree_equal(before, self.state()): raise ValueError('capture mutated evaluation state')
        from video_template import project
        point = np.asarray(data['tcp_world'], float)
        if point.shape != (3,): raise ValueError('invalid TCP')
        project([point], data['world_to_camera'], data['K'])
        row = dict(case_id=self.case_id, condition=self.condition, step=step,
                   elapsed_s=step / self.fps, actions=plain(action), phase=phase, outcome=outcome,
                   cumulative_force=plain(info.get('robot_cumulative_force')),
                   official_info_eval_only=plain(info), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        if hasattr(self,'action_labels'):row['action_labels']=list(self.action_labels)
        for name in ('scene', 'head', 'hand'):
            im = array(data[name])
            if im.ndim == 4 and im.shape[0] == 1: im = im[0]
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] not in (3, 4):
                raise ValueError(f'invalid {name} RGB')
            path = self.root / f'{step:04d}-{name}.png'
            Image.fromarray(im[:, :, :3]).save(path, compress_level=1)
            row[name] = path.name
        depth_path = self.root / f'{step:04d}-depth.npz'
        # Lossless level-1 compression reduces recorder latency; dtype/data unchanged.
        import zipfile
        with zipfile.ZipFile(depth_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for key, values in [('head', data['head_depth']), ('hand', data['hand_depth'])]:
                with z.open(key + '.npy', 'w') as member:
                    np.save(member, array(values), allow_pickle=False)
        self.trajectory.append(point.tolist())
        row.update(trajectory_world=copy.deepcopy(self.trajectory),
                   world_to_camera=plain(data['world_to_camera']), K=plain(data['K']),
                   qpos=plain(data['qpos']), qvel=plain(data['qvel']),
                   base_world_eval_only=plain(data['base_world']), depth=depth_path.name)
        from task_rules import evaluate_state
        row['evaluation_only']=evaluate_state(row['qvel'],row['official_info_eval_only'])
        if 'camera_pose_receipt' in data:
            row['camera_pose_receipt'] = data['camera_pose_receipt']
        row['files_sha256'] = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                              for name in [row['scene'], row['head'], row['hand'], row['depth']]}
        self.frames.append(row)
        save(self.root / 'frames.json', dict(fps=self.fps, evaluation_only=True, frames=self.frames))
        return row

    def finish(self, outcome):
        if self.frames: self.frames[-1]['outcome'] = outcome
        save(self.root / 'frames.json', dict(fps=self.fps, evaluation_only=True,
             attempt_outcome=outcome, frames=self.frames))


def validate_restore_targets(base,payload):
    """Reject Python/RNG topology mismatch before transport touches GPU buffers."""
    from check_c2_snapshot_restore import targets
    pairs=targets(base);actual=[name for name,obj in pairs]
    expected=list(payload['state']['attributes'])
    if len(set(actual))!=len(actual) or set(actual)!=set(expected):
        error=ValueError('snapshot target identity mismatch before native writes')
        error.receipt=dict(expected=expected,actual=actual,
            missing=sorted(set(expected)-set(actual)),unexpected=sorted(set(actual)-set(expected)))
        raise error
    owners=dict(pairs)
    for label,states in payload['state']['batched_rng_states'].items():
        owner,key=label.split('|',1)
        obj=owners.get(owner);rng=getattr(obj,key,None)
        if rng is None or not hasattr(rng,'rngs') or len(rng.rngs)!=len(states):
            error=ValueError('snapshot RNG owner/count mismatch before native writes: '+label)
            error.receipt=dict(owner=owner,attribute=key,expected_count=len(states),actual_count=len(rng.rngs) if hasattr(rng,'rngs') else None)
            raise error
    return dict(targets=actual,rng_owners=list(payload['state']['batched_rng_states']))


def write_diagnostic_bundle(root, name, value):
    """Lossless array leaves plus typed JSON tree; evaluation-only, never policy input.

    Publish the hash receipt last. An interrupted write is not a valid bundle.
    No pickle, object arrays, numeric rounding, or array-size truncation.
    """
    import os
    from collections import deque
    from bvi.pick_system.executor_diagnostics import state_plain
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Unsafe diagnostic name')
    if (root/(name+'.receipt.json')).exists():raise FileExistsError('Diagnostic bundle already published')
    arrays={}
    def encode(v):
        if hasattr(v,'detach') or isinstance(v,np.ndarray):
            a=array(v).copy()
            if a.dtype.hasobject:raise TypeError('Object array in diagnostics')
            key='a'+str(len(arrays));arrays[key]=a
            return dict(kind='array',key=key,dtype=str(a.dtype),shape=list(a.shape),source_dtype=str(v.dtype))
        if isinstance(v,dict):return dict(kind='dict',items=[[encode(k),encode(x)] for k,x in v.items()])
        if isinstance(v,(list,tuple,deque)):
            return dict(kind=type(v).__name__,items=[encode(x) for x in v],maxlen=v.maxlen if isinstance(v,deque) else None)
        if isinstance(v,np.generic):return encode(np.asarray(v))
        if v is None or type(v) in (str,int,bool):return dict(kind='scalar',value=v)
        if type(v) is float:return dict(kind='float',value=v if np.isfinite(v) else repr(v))
        return dict(kind='state_plain',source_type=type(v).__name__,value=encode(state_plain(v)))
    tree=encode(value)
    def publish(path,writer):
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('xb') as stream:writer(stream);stream.flush();os.fsync(stream.fileno())
        tmp.replace(path)
    matrix=root/(name+'.npz');meta=root/(name+'.tree.json')
    publish(matrix,lambda stream:np.savez_compressed(stream,**arrays))
    publish(meta,lambda stream:stream.write(json.dumps(tree,ensure_ascii=False,allow_nan=False).encode('utf-8')))
    receipt=dict(evaluation_only=True,arrays=len(arrays),files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (matrix,meta)})
    publish(root/(name+'.receipt.json'),lambda stream:stream.write(json.dumps(receipt,indent=2).encode()))
    return receipt


def diagnostic_residuals(expected,actual):
    """Full leaf inventory, independent of the acceptance comparison/thresholds."""
    from bvi.pick_system.executor_diagnostics import state_plain
    from collections import deque
    rows=[]
    def walk(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b),key=str):
                p=path+'/'+str(k)
                if k not in a or k not in b:rows.append(dict(path=p,kind='missing',expected=k in a,actual=k in b))
                else:walk(a[k],b[k],p)
        elif isinstance(a,(tuple,list,deque)) and isinstance(b,(tuple,list,deque)):
            if len(a)!=len(b):rows.append(dict(path=path,kind='length',expected=len(a),actual=len(b)))
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif isinstance(a,(np.ndarray,np.generic)) or isinstance(b,(np.ndarray,np.generic)) or hasattr(a,'detach') or hasattr(b,'detach') or (type(a) in (int,float,bool) and type(b) in (int,float,bool)):
            x=array(a);y=array(b);r=dict(path=path,expected_shape=list(x.shape),actual_shape=list(y.shape),expected_dtype=str(x.dtype),actual_dtype=str(y.dtype))
            r['same_shape']=x.shape==y.shape;r['exact']=bool(r['same_shape'] and x.dtype==y.dtype and np.array_equal(x,y))
            if r['same_shape'] and x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf':
                finite=np.isfinite(x)&np.isfinite(y);r['nonfinite_elements']=int((~finite).sum())
                delta=np.abs(x.astype(np.float64)-y.astype(np.float64));values=np.where(finite,delta,0.)
                r['max_abs_finite']=float(np.max(values,initial=0));r['different_elements']=int(np.count_nonzero(x!=y))
                r['max_abs_index']=list(np.unravel_index(int(np.argmax(values)),values.shape)) if values.size else None
                if r['max_abs_index'] is not None:r['max_abs_index']=[int(v) for v in r['max_abs_index']]
            rows.append(r)
        else:
            rows.append(dict(path=path,kind='value',exact=state_plain(a)==state_plain(b)))
    walk(expected,actual,'')
    return rows


def restore_diagnostic_observer(root):
    def observe(name,value):
        if name=='restore_comparison':
            value=dict(value,leaf_residuals=diagnostic_residuals(value['expected'],value['actual']))
        write_diagnostic_bundle(root,name,value)
    return observe


def checked_restore(base, payload, *, diagnostics=None):
    """Reuse complete native transport. Never accept qpos-only or legacy row IDs."""
    if not isinstance(payload, dict) or payload.get('contract') != CONTRACT:
        raise ValueError('bound native v2 snapshot required')
    validate_restore_targets(base,payload)
    validate_skipped_targets(base,payload)
    from bvi.pick_system.snapshot_transport import restore_transport
    if diagnostics is None:return restore_transport(base, payload)
    return restore_transport(base, payload, observer=restore_diagnostic_observer(diagnostics))


def validate_skipped_targets(base,payload):
    """Check all non-serialized runtime object identities before any GPU write."""
    from check_c2_snapshot_restore import targets
    from bvi.pick_system.snapshot_transport import clone_snapshot_value
    actual={}
    for name,obj in targets(base):
        actual[name]={}
        for key,value in vars(obj).items():
            if type(value).__name__=='BatchedRNG':continue
            try:clone_snapshot_value(value)
            except (TypeError,ValueError):actual[name][key]=type(value).__name__
    expected=payload['state']['skipped']
    if actual!=expected:
        error=ValueError('snapshot skipped-object identity mismatch before native writes')
        error.receipt=dict(expected=expected,actual=actual,differences=diagnostic_residuals(expected,actual))
        raise error
    return dict(passed=True,owners=len(actual),objects=sum(map(len,actual.values())))


def restore_bound_entry(base, receipt_path, case, *, diagnostics=None):
    """Verify case and compressed hash before deserializing any native checkpoint."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt['case_id'] != case['case_id'] or receipt['source_sha256'] != case['source_sha256']
            or receipt['prefix_steps'] != case['entry_step']):
        raise ValueError('entry case/source binding')
    snapshot = receipt['snapshot']
    if snapshot['contract'] != CONTRACT: raise ValueError('entry v2 required')
    name = snapshot['file']
    if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('checkpoint must be a sibling file')
    path = receipt_path.parent / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot['sha256']:
        raise ValueError('checkpoint SHA mismatch')
    u = base.uenv
    def identity():
        if (str(u.task_plan[0].composite_subtask_uids[0]) != case['uid']
                or int(u.spawn_selection_idxs[0]) != case['spawn_index']):
            raise ValueError('runtime task identity')
    identity(); driver_readback(u)
    from bvi.pick_system.executor_diagnostics import load_snapshot
    payload = load_snapshot(path, u.device, receipt=snapshot)
    checks = checked_restore(base, payload, diagnostics=diagnostics)
    identity(); driver_readback(u)
    return dict(restore=checks, initial_info=receipt['entry_info_eval_only'],
                executor_recipe=receipt['executor_recipe'])


def write_native_snapshot(base, output):
    """Future approved capture only. Save the real complete v2 payload + hashes."""
    import gzip
    import io
    import torch
    from bvi.pick_system.snapshot_transport import capture_transport
    from bvi.pick_system.executor_diagnostics import validate_snapshot
    payload = capture_transport(base)
    validate_snapshot(payload['state'])
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    compressed = gzip.compress(raw, mtime=0)
    output = Path(output)
    with output.open('xb') as file: file.write(compressed)
    receipt = dict(contract=CONTRACT, file=output.name, compression='gzip',
                   sha256=hashlib.sha256(compressed).hexdigest(), payload_bytes=len(raw),
                   payload_sha256=hashlib.sha256(raw).hexdigest())
    save(output.with_suffix(output.suffix + '.json'), receipt)
    return receipt


def prepare_entry(*, prefix, driver_guard, initial_guard, entry_guard,
                  recorder, initial_info, step_once, snapshot_writer,
                  wall_seconds=600):
    """Replay only frozen prefix, then save v2; no API or new planning.

    The caller must bind the three guards to actual case identity, archived reset
    state, and archived entry state. Navigation in this PREPARATION phase is explicit.
    """
    actions = np.asarray(prefix['actions'], float)
    if not 0 < wall_seconds <= 600: raise ValueError('prefix wall cap')
    if actions.shape != (prefix['entry_step'], 13) or len(actions) > 117:
        raise ValueError('prefix shape/budget')
    if not np.isfinite(actions).all() or np.max(abs(actions)) > 1+1e-8:
        raise ValueError('prefix action')
    driver_guard(); initial_guard()
    recorder.record(0, None, initial_info, phase='prefix_preparation')
    start = time.monotonic()
    for number, action in enumerate(actions, 1):
        if time.monotonic() - start >= wall_seconds: raise RuntimeError('prefix wall cap')
        status, info = step_once(action)
        terminal = bool(status.get('terminated') or status.get('truncated'))
        recorder.record(number, action, info, phase='prefix_preparation',
                        outcome='official_terminal' if terminal else 'in_progress')
        if status.get('clock_valid') is False: raise RuntimeError('official clock discontinuity')
        if status.get('force_valid') is False: raise RuntimeError('official force discontinuity')
        if terminal: raise RuntimeError('prefix terminated; no snapshot or A/B')
    entry_guard()
    receipt = snapshot_writer()
    if receipt.get('contract') != CONTRACT: raise ValueError('snapshot writer must produce native v2')
    delta = np.asarray(recorder.frames[-1]['qpos']) - np.asarray(recorder.frames[-2]['qpos'])
    save(recorder.root / 'entry-receipt.json', dict(snapshot=receipt, case_id=prefix['case_id'], prefix_steps=len(actions),
        source_sha256=prefix['source_sha256'], entry_info_eval_only=plain(info),
        executor_recipe=dict(initial_delta=delta.tolist(), gripper=1., control_dt=.05,
                             initialization='new high-level executor; native controller restored in full'),
        provenance='fresh historical-prefix replay; not original full snapshot'))
    recorder.finish('entry_captured_not_pick_result')
    return receipt


class RecordingGCScope:
    """Bound a short recorder batch; retain GC and restore its original settings.

    Large immutable task caches otherwise cause repeated generation-2 scans.
    Collect at both boundaries; those costs belong in the batch wall budget.
    No environment or numerical state is changed, and gen0/gen1 remain enabled.
    """
    def __enter__(self):
        import gc
        self.previous=gc.get_threshold()
        gc.collect()
        gc.set_threshold(self.previous[0],self.previous[1],max(self.previous[2],1000))
        return self
    def __exit__(self,*exc):
        import gc
        try:gc.collect()
        finally:gc.set_threshold(*self.previous)
        return False


def state_fingerprint(value, *, _audit=None):
    """Stream all typed state into SHA256, without per-pixel or JSON expansion.

    CPU arrays include dtype, shape and every byte. Signed zero/NaN payloads are
    exact. Optional audit events expose actual visited objects/array byte feeds;
    production uses no hook. Snapshot capture/restore itself is unchanged.
    """
    import dataclasses
    import struct
    from collections import deque, defaultdict
    from enum import Enum
    h = hashlib.sha256()
    array_headers = {}  # immutable dtype/shape bytes, cached only within this call
    def token(tag, data=b''):
        h.update(tag); h.update(struct.pack('>Q', len(data))); h.update(data)
    def text(tag, data): token(tag, data.encode('utf-8'))
    def typename(v): return type(v).__module__ + '.' + type(v).__qualname__
    def child(path, kind, key=None): return path + ((kind, key),) if _audit is not None else ()
    def encode(v, path=()):
        if _audit is not None: _audit('visit', (path, v))
        t = type(v)
        if v is None: token(b'N'); return
        if t is bool: token(b'B', b'1' if v else b'0'); return
        if t is int: text(b'I', str(v)); return
        if t is float: token(b'F', struct.pack('>d', v)); return
        if t is str: text(b'S', v); return
        if t is bytes: token(b'Y', v); return
        if hasattr(v, 'detach'):
            text(b'T', str(v.dtype)); encode(v.detach().cpu().numpy(), child(path, 'tensor_array')); return
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject: raise ValueError('object arrays cannot be fingerprinted')
            if _audit is None and v.dtype.fields is None and v.dtype.metadata is None:
                key=(v.dtype.str, tuple(v.shape), v.dtype.isalignedstruct)
                header=array_headers.get(key)
                if header is None:
                    def pack(tag, data=b''): return tag + struct.pack('>Q', len(data)) + data
                    header=b''.join([pack(b'A',v.dtype.str.encode()), pack(b'D',repr(v.dtype.descr).encode()),
                        pack(b'L',b'1' if v.dtype.isalignedstruct else b'0'),pack(b'N'),
                        pack(b'V',b'builtins.tuple'),pack(b'I',str(v.ndim).encode()),
                        *[pack(b'I',str(n).encode()) for n in v.shape]])
                    array_headers[key]=header
                h.update(header)
                token(b'R', memoryview(np.ascontiguousarray(v)).cast('B') if v.nbytes else b'')
                return
            text(b'A', v.dtype.str); text(b'D', repr(v.dtype.descr))
            token(b'L', b'1' if v.dtype.isalignedstruct else b'0')
            encode(dict(v.dtype.metadata) if v.dtype.metadata else None, child(path, 'dtype_metadata'))
            encode(tuple(v.shape), child(path, 'shape'))
            data = memoryview(np.ascontiguousarray(v)).cast('B') if v.nbytes else b''
            if _audit is not None: _audit('array_bytes', (path, data))
            token(b'R', data); return
        if isinstance(v, np.generic): token(b'G'); encode(np.asarray(v), child(path, 'scalar_array')); return
        if isinstance(v, np.random.RandomState): token(b'RS'); encode(v.get_state(), child(path, 'rng_state')); return
        if isinstance(v, np.random.Generator): token(b'RG'); encode(v.bit_generator.state, child(path, 'rng_state')); return
        if t.__name__ == 'Generator' and hasattr(v, 'get_state'):
            token(b'TG'); encode(v.get_state(), child(path, 'rng_state')); return
        if isinstance(v, Enum): text(b'E', typename(v)); encode(v.value, child(path, 'enum_value')); return
        if dataclasses.is_dataclass(v):
            text(b'C', typename(v)); fields=dataclasses.fields(v);encode(len(fields), child(path, 'length'))
            for f in fields: text(b'K', f.name); encode(getattr(v, f.name), child(path, 'field', f.name))
            return
        if t.__name__ == 'Pose':
            token(b'P')
            if hasattr(v, 'raw_pose'): encode(v.raw_pose, child(path, 'pose')); return
            if hasattr(v, 'p') and hasattr(v, 'q'): encode([v.p, v.q], child(path, 'pose')); return
        if t.__name__ == 'BatchedRNG' and hasattr(v, 'rngs'):
            token(b'BR'); encode([r.get_state() for r in v.rngs], child(path, 'rng_state')); return
        if isinstance(v, dict):
            text(b'M', typename(v)); encode(len(v), child(path, 'length'))
            if isinstance(v, defaultdict):
                factory = v.default_factory
                if factory is not None and factory not in (list, dict, set, tuple, int, float, bool):
                    raise ValueError('unverified defaultdict factory')
                text(b'Q', '' if factory is None else factory.__module__ + '.' + factory.__qualname__)
                if _audit is not None: _audit('factory', (path, factory))
            for i, (k, x) in enumerate(v.items()):
                encode(k, child(path, 'dict_key', i)); encode(x, child(path, 'dict_value', i))
            return
        if isinstance(v, (list, tuple, deque)):
            text(b'V', typename(v)); encode(len(v), child(path, 'length'))
            if isinstance(v, deque): encode(v.maxlen, child(path, 'maxlen'))
            for i, x in enumerate(v): encode(x, child(path, 'item', i))
            return
        if isinstance(v, set):
            token(b'SET'); encode(len(v), child(path, 'length'))
            for i, x in enumerate(sorted(v, key=state_fingerprint)): encode(x, child(path, 'item', i))
            return
        raise ValueError('unverified recorder state type: ' + typename(v))
    encode(value)
    return h.hexdigest()


def recording_state(base):
    """Full simulator/controller/task/RNG guard with array-byte fingerprints."""
    from bvi.pick_system.snapshot_transport import capture_transport
    return state_fingerprint(capture_transport(base))


def native_ports(base, output, case_id, condition, initial_info, *, action_size=13):
    """Concrete ManiSkill binding for an already created/restored approved env.

    Returns ports only; creating these ports does not execute an action. Official
    info comes from reset/checkpoint or env.step, never a second evaluate call.
    """
    from astra_pick.run_case import strict_pick
    u = base.uenv
    info = copy.deepcopy(initial_info)
    stopped = False
    def scalar(value):
        result = array(value)
        if result.size != 1: raise ValueError('single environment required')
        return result.item()
    if bool(scalar(info['success'])) or bool(scalar(info['fail'])) or int(scalar(info['subtasks_steps_left'])) <= 0:
        raise ValueError('entry already terminal')
    def qpos(): return array(u.agent.robot.qpos)[0].astype(float)
    def qvel(): return array(u.agent.robot.qvel)[0].astype(float)
    def rest(): return {k: bool(scalar(info[k])) for k in ('ee_rest', 'robot_rest', 'is_static')}
    def step_once(action):
        nonlocal info, stopped
        import torch
        if stopped: raise RuntimeError('step after official terminal')
        old_left = int(scalar(info['subtasks_steps_left']))
        old_force = float(scalar(info['robot_cumulative_force']))
        if old_left <= 0: raise RuntimeError('no remaining official steps')
        _, _, terminated, truncated, info = base.env.step(
            torch.as_tensor(action, device=u.device, dtype=torch.float32).reshape(1, -1))
        success = strict_pick(plain(info))
        failed = bool(scalar(info['fail']))
        stopped = success or failed or bool(scalar(terminated)) or bool(scalar(truncated))
        # Discontinuity is returned to the recorder and then checked by the caller,
        # so the last physically executed frame is not lost.
        return dict(terminated=stopped, truncated=bool(scalar(truncated)),
                    clock_valid=int(scalar(info['subtasks_steps_left'])) == old_left - 1,
                    force_valid=float(scalar(info['robot_cumulative_force'])) + 1e-5 >= old_force), dict(plain(info), strict_pick_eval_only=success)
    recorder = Recorder(output, case_id, condition, lambda: native_capture(u),
                        lambda: recording_state(base), fps=u.control_freq, action_size=action_size)
    return dict(recorder=recorder, step_once=step_once, read_qpos=qpos, read_qvel=qvel,
                read_finger=lambda: float(qpos()[-2:].sum()), read_rest_status=rest)


def run_prepared_arm(*, condition, commands, executor_args, executor_kwargs,
                     driver_guard, restore_guard, recorder_factory, step_once,
                     initial_info, wall_seconds=600, step_cap=600):
    """Dependency wiring; never creates an env. CPU tests supply inert ports.

    A future approved runner owns source/identity/snapshot SHA binding and passes
    a step_once returning (official_status, official_info) without calling evaluate
    again. Its initial_info must come from the restored checkpoint's last step.
    """
    from adapter import make_executor, execute_script
    if condition not in ('A', 'B'): raise ValueError('condition')
    if not 0 < wall_seconds <= 1500: raise ValueError('wall cap')
    if type(step_cap) is not int or not 0 < step_cap <= 600: raise ValueError('step cap')
    if 'initial_delta' not in executor_kwargs or executor_kwargs.get('gripper') != 1.:
        raise ValueError('explicit shared entry executor recipe required')
    driver_guard()
    restore_guard()
    recorder = recorder_factory()
    recorder.record(0, None, initial_info, phase='restored_entry')
    start = time.monotonic()
    count = 0
    terminal = False
    phase = 'execution'
    command_events = []
    def step(action):
        nonlocal count, terminal
        if terminal or count >= step_cap or time.monotonic() - start >= wall_seconds:
            raise RuntimeError('attempt stopped or budget exhausted')
        action = np.asarray(action, float)
        if action.shape != (13,) or not np.isfinite(action).all() or np.max(abs(action)) > 1+1e-8 or np.any(action[11:] != 0):
            raise ValueError('invalid fixed-base action')
        status, info = step_once(action)
        count += 1
        terminal = bool(status.get('terminated') or status.get('truncated'))
        recorder.record(count, action, info, phase=phase, outcome='official_terminal' if terminal else 'in_progress')
        if status.get('clock_valid') is False: raise RuntimeError('official clock discontinuity')
        if status.get('force_valid') is False: raise RuntimeError('official force discontinuity')
        return status
    # step is the third positional argument of FetchToolsV3.
    args = list(executor_args)
    if len(args) != 5: raise ValueError('provide fk,qpos,finger,rest,model ports')
    args.insert(2, step)
    options = dict(executor_kwargs)
    options['max_steps'] = step_cap
    executor = make_executor(condition, *args, **options)
    class TracedCommands:
        def __getattr__(self, name):
            value = getattr(executor, name)
            if name not in ('move_eef_chunk', 'set_gripper', 'return_to_rest'): return value
            def call(*args, **kwargs):
                nonlocal phase
                phase = name
                event = dict(tool=name, start_step=count, start_wall_s=time.monotonic()-start)
                try:
                    result = value(*args, **kwargs)
                    event['result'] = result
                    grip=args[0] if args else kwargs.get('command')
                    if name=='set_gripper' and grip==0:
                        from task_rules import close_progress
                        closed_steps=sum(1 for f in recorder.frames[event['start_step']+1:]
                                         if f.get('actions') is not None and f['actions'][7]<0)
                        event.update(close_progress(result,closed_steps,executor.p.close_wait_cap))
                    return result
                finally:
                    event.update(end_step=count, end_wall_s=time.monotonic()-start)
                    command_events.append(event)
                    save(recorder.root / 'command-events.json', command_events)
            return call
    try:
        rows = execute_script(TracedCommands(), commands)
        outcome = 'official_terminal' if terminal else ('tool_failure' if len(rows) < len(commands) or
            any(r['result'].get('arrived') is False or r['result'].get('accepted') is False for r in rows)
            else 'script_complete_not_official_success')
        recorder.finish(outcome)
        save(recorder.root / 'result.json', dict(status='finished', commands=rows,
             physical_steps=count, trace=executor.trace, official_terminal=terminal,
             strict_pick_success=any(f['official_info_eval_only'].get('strict_pick_eval_only', False) for f in recorder.frames[1:]),
             strict_success_claim='read official_info; never infer from IK'))
        return rows
    except Exception as error:
        recorder.finish('interrupted')
        save(recorder.root / 'result.json', dict(status='interrupted', physical_steps=count,
             error_type=type(error).__name__, error=str(error), trace=executor.trace))
        raise
