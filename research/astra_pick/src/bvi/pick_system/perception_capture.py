"""E1 r2 evaluator-side synchronous capture. No policy, sender, IK or env.step.

Only ``sensor`` references may be handed to a request builder. Native segmentation,
Position, world transforms, actors and checkpoints are deliberately separate files.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

SCHEMA = 'bvi-e1-sync-capture-v1'
PROTOCOL = 'pick-perception-e1-auto-gt-v3-r2'
LIMITS = dict(runs=10, steps=2277, seconds=6000, per_run_seconds=600,
              captures=320, output_bytes=10 * 1024**3, api_requests=0)
RENDERER_PROFILE = dict(shader_pack='minimal', render_system='3.0',
    texture_names={'Color': ['rgb'], 'PositionSegmentation': ['position', 'depth', 'segmentation']},
    shader_pack_config={}, position_unit='mm', position_dtype='int16',
    position_convention='opengl_camera', transformed_segmentation_channel=0,
    raw_segmentation_source='PositionSegmentation[...,3] actor/link ID')
SAPIEN_FROM_GL = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]])


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)) or hasattr(value, 'detach'): return array(value).tolist()
    return value


def canonical(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(canonical(value)); os.replace(temp, path)


def reference(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path), bytes=path.stat().st_size)


def bound_path(root, ref):
    root = Path(root).resolve(); path = (root / ref['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != ref['sha256']:
        raise ValueError('missing, escaped or changed bound file: ' + str(ref.get('path')))
    return path


def transform(value):
    if hasattr(value, 'to_transformation_matrix'): value = value.to_transformation_matrix()
    m = array(value).astype(float)
    if m.shape == (1, 4, 4): m = m[0]
    if (m.shape != (4, 4) or not np.isfinite(m).all() or
            not np.allclose(m[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-4)):
        raise ValueError('invalid rigid transform')
    return m


def action_sequence(result):
    frames = result['frames']; actions = []
    for step, frame in enumerate(frames, 1):
        a = np.asarray(frame['action'], dtype=float)
        if frame['step'] != step or a.shape != (13,) or not np.isfinite(a).all():
            raise ValueError('source must contain contiguous finite thirteen-dimensional controls')
        actions.append(frame['action'])
    if len(actions) != result['simulator_actions']: raise ValueError('source step count mismatch')
    return actions


def capture_schedule(count, original_steps):
    if not 0 <= count <= 600 or any(type(s) is not int or not 0 <= s <= count for s in original_steps):
        raise ValueError('invalid frozen observation steps')
    return sorted({0, count, *range(0, count + 1, 10), *original_steps})


def validate_shader_config(sensors, expected, prebuilt, render_system):
    """Check effective config, including texture transforms, without changing it."""
    if expected != RENDERER_PROFILE or render_system != expected['render_system']:
        raise ValueError('unknown or unfrozen renderer profile')
    frozen = prebuilt.get(expected['shader_pack'])
    if frozen is None: raise ValueError('frozen shader pack unavailable')
    records = {}
    for name in ('head', 'hand'):
        config = sensors['fetch_' + name].config
        shader = config.shader_config
        if (config.shader_pack != expected['shader_pack'] or shader.shader_pack != expected['shader_pack'] or
                shader.texture_names != expected['texture_names'] or
                shader.shader_pack_config != expected['shader_pack_config'] or
                set(shader.texture_transforms) != set(frozen.texture_transforms) or
                any(shader.texture_transforms[k] is not frozen.texture_transforms[k] for k in frozen.texture_transforms)):
            raise ValueError('effective camera shader config differs from frozen prebuilt shader')
        records[name] = dict(shader_pack=shader.shader_pack, texture_names=shader.texture_names,
                             shader_pack_config=shader.shader_pack_config,
                             texture_transform_binding='identical callables from source-pinned PREBUILT_SHADER_CONFIGS')
    return dict(profile=expected, cameras=records)


class OutputBudget:
    """Account writes before commit, including failed/incomplete captures."""
    def __init__(self, root, maximum=LIMITS['output_bytes'], *, account_root=None, media_reserve_bytes=0,
                 receipt_reserve_bytes=0):
        self.root, self.maximum = Path(root), int(maximum)
        self.account_root = Path(account_root) if account_root is not None else None
        self.media_reserve_bytes = int(media_reserve_bytes)
        self.receipt_reserve_bytes = int(receipt_reserve_bytes)
        self.finalizing = False
        if self.media_reserve_bytes < 0 or self.media_reserve_bytes >= self.maximum:
            raise ValueError('invalid final-media reservation')
        if not 0 <= self.receipt_reserve_bytes < self.maximum-self.media_reserve_bytes:
            raise ValueError('invalid durable-receipt reservation')
        self.used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
    def usage(self):
        root = self.account_root or self.root
        files = [(p, p.stat().st_size) for p in root.rglob('*') if p.is_file()]
        media = sum(n for p, n in files if any(part.startswith('diagnostic-media-') for part in p.relative_to(root).parts))
        total = sum(n for _, n in files)
        return dict(total=total, media=media, capture=total-media)
    def check(self, additional=0, *, receipt=False):
        if self.account_root is None:
            if self.used + additional > self.maximum: raise ValueError('output byte budget exhausted')
            return
        usage = self.usage()
        if usage['total'] + additional > self.maximum: raise ValueError('total output byte budget exhausted')
        margin = 0 if receipt or self.finalizing else self.receipt_reserve_bytes
        if usage['capture'] + additional > self.maximum-self.media_reserve_bytes-margin:
            raise ValueError('capture output exhausted; final-media reservation retained')
    def media_guard(self, additional=0):
        usage = self.usage()
        if (usage['total'] + additional > self.maximum or
                usage['media'] + additional > self.media_reserve_bytes):
            raise ValueError('reserved final-media byte budget exhausted')
    def finish_capture(self): self.finalizing = True
    def write(self, path, data):
        path = Path(path)
        if not path.resolve().is_relative_to(self.root.resolve()): raise ValueError('output escaped root')
        if path.exists(): raise FileExistsError(path)
        self.check(len(data))
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data); self.used += len(data)
        return reference(path, self.root)
    def replace_json(self, path, value):
        """Reserve the atomic temporary file as well as the old version."""
        path = Path(path); data = canonical(value)
        if not path.resolve().is_relative_to(self.root.resolve()): raise ValueError('output escaped root')
        self.check(len(data), receipt=True); old = path.stat().st_size if path.exists() else 0
        save(path, value); self.used += len(data)-old
    def json(self, path, value): return self.write(path, canonical(value))
    def npy(self, path, value):
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
        return self.write(path, stream.getvalue())
    def png(self, path, value):
        stream = io.BytesIO(); Image.fromarray(value).save(stream, format='PNG')
        return self.write(path, stream.getvalue())


def write_checkpoint(budget, path, snapshot, *, compression=None, torch_api=None):
    """Old raw encoding is unchanged; recovery gzip retains byte-exact payload."""
    if torch_api is None:
        import torch as torch_api
    stream = io.BytesIO(); torch_api.save(snapshot, stream); payload = stream.getvalue()
    if compression is None: return budget.write(path, payload)
    if compression != 'gzip': raise ValueError('unknown checkpoint encoding')
    import gzip
    target = Path(str(path)+'.gz')
    encoded = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=encoded, compresslevel=6, mtime=0) as zipped:
        zipped.write(payload)
    return {**budget.write(target, encoded.getvalue()), 'compression': 'gzip',
            'payload_sha256': hashlib.sha256(payload).hexdigest(), 'payload_bytes': len(payload),
            'compression_level': 6}


def _native_id(body):
    objects = getattr(body, '_objs', ())
    if len(objects) != 1: raise ValueError('actor must belong to one environment')
    obj = objects[0]
    identifier = getattr(obj, 'per_scene_id', None)
    if identifier is None: identifier = getattr(getattr(obj, 'entity', None), 'per_scene_id', None)
    if identifier is None or int(identifier) <= 0: raise ValueError('unbound native actor segmentation ID')
    return int(identifier)


def bind_pick_target(u, target_instance_id):
    """MS-HAB's obj_0 is a merged wrapper: bind by its underlying entity ID."""
    key = 'env-0_' + target_instance_id
    original = u.scene_builder.movable_objects.get(key)
    if original is None or _native_id(original) != _native_id(u.subtask_objs[0]):
        raise ValueError('Pick merged target does not match frozen task instance')
    return original


def actor_inventory(u, *, target, target_instance_id, asset_bindings=None):
    """Rebuild IDs for this reset; no assumptions about IDs shared by two runs."""
    from export_c2_collision_bundle import _other_bodies
    asset_bindings = asset_bindings or {}
    target_id = _native_id(target); rows = []; bodies = {}
    inventory = _other_bodies(u) + [(f'robot_link:{i}:{b.name}', 'robot_link', b)
                                   for i, b in enumerate(u.agent.robot.links)]
    for stable_name, kind, body in inventory:
        identifier = _native_id(body); key = str(identifier)
        if key in bodies:
            if bodies[key]._objs[0] is body._objs[0]: continue
            raise ValueError('ambiguous segmentation ID')
        bodies[key] = body
        instance = target_instance_id if identifier == target_id else str(body.name)
        binding = asset_bindings.get(instance, asset_bindings.get('native:' + key, {}))
        rows.append(dict(actor_id=key, instance_id=binding.get('instance_id', instance), name=str(body.name), kind=kind,
                         category=binding.get('category'), segmentation_ids=[identifier],
                         is_target=identifier == target_id, asset_binding=binding,
                         scene_reference_key=stable_name))
    if sum(row['is_target'] for row in rows) != 1: raise ValueError('Pick target missing from actor map')
    return dict(schema=SCHEMA, target_actor_id=str(target_id), actors=rows), bodies


def body_states(mapping, bodies):
    from export_c2_collision_bundle import _body_state
    rows = []
    for row in mapping['actors']:
        body = bodies[row['actor_id']]
        dynamic = row['kind'] in ('other_link', 'robot_link') or getattr(body, 'px_body_type', '') in ('dynamic', 'kinematic')
        state = _body_state(body, dynamic=dynamic)
        rows.append({**row, 'pose_world': state['pose_world'],
                     'linear_velocity_m_s': state.get('linear_velocity_world', [0., 0., 0.]),
                     'angular_velocity_rad_s': state.get('angular_velocity_world', [0., 0., 0.]),
                     'body_type': state['body_type']})
    return rows


def obj_triangles(path):
    """Read the frozen Wavefront visual surface, without textures or network IO."""
    vertices = []; faces = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'): continue
        if fields[0] == 'v':
            if len(fields) != 4: raise ValueError('unsupported homogeneous OBJ vertex')
            vertices.append([float(x) for x in fields[1:]])
        elif fields[0] == 'f':
            # Exported YCB files are triangles; do not guess a nonconvex polygon triangulation.
            if len(fields) != 4: raise ValueError('non-triangular visual OBJ')
            raw = [int(x.split('/')[0]) for x in fields[1:]]
            if 0 in raw: raise ValueError('zero OBJ vertex index')
            faces.append([x - 1 if x > 0 else len(vertices) + x for x in raw])
    v = np.asarray(vertices, dtype=float); f = np.asarray(faces, dtype=np.int64)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or f.ndim != 2 or f.shape[1] != 3 or
            not len(f) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('invalid visual OBJ')
    return v, f


def export_bound_ycb_visuals(mapping, budget, lab_root, asset_pins):
    """Exact pinned YCB loader recipe: textured.obj, metadata scales[0], identity.

    This is visual geometry, not collision.ply. The recipe is bound to both loader
    sources and the metadata/OBJ hashes; it is not inferred from object pose.
    """
    ycb = 'assets/data/assets/mani_skill2_ycb'
    loader = 'src/official-mshab-runtime/ManiSkill/mani_skill/utils/building/actors/ycb.py'
    scene_builder = 'src/official-mshab-runtime/ManiSkill/mani_skill/utils/scene_builder/replicacad/rearrange/scene_builder.py'
    metadata_path = ycb + '/info_pick_v0.json'
    for name in (loader, scene_builder, metadata_path):
        if name not in asset_pins: raise ValueError('visual loader/scale provenance missing')
        bound_path(lab_root, dict(path=name, sha256=asset_pins[name]))
    # Pin the known recipe, not an arbitrary future loader that happens to exist.
    source = (Path(lab_root) / loader).read_text(encoding='utf-8')
    recipe = ('scale = model_scales[0]', 'model_scales = metadata.get("scales", [1.0])',
              'visual_file = str(model_dir / "textured.obj")',
              'builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)')
    if not all(line in source for line in recipe): raise ValueError('frozen visual recipe changed')
    metadata = json.loads((Path(lab_root) / metadata_path).read_text(encoding='utf-8'))
    result = {}; cache = {}
    for actor in mapping['actors']:
        category = actor.get('category')
        if category is None: continue
        if category not in metadata: raise ValueError('category missing from frozen scale metadata')
        if category not in cache:
            asset = f'{ycb}/models/{category}/textured.obj'
            if asset not in asset_pins:
                if actor.get('is_target'): raise ValueError('target visual mesh source pin missing')
                result[actor['actor_id']] = dict(visuals=[], visual_coverage='unavailable',
                    visual_coverage_reasons=['non-target visual mesh outside bounded D0 export'])
                continue
            path = bound_path(lab_root, dict(path=asset, sha256=asset_pins[asset]))
            scale = float(metadata[category].get('scales', [1.])[0])
            if not np.isfinite(scale) or scale <= 0: raise ValueError('invalid frozen YCB scale')
            vertices, faces = obj_triangles(path)
            stream = io.BytesIO(); np.savez_compressed(stream, vertices=vertices, faces=faces)
            mesh = budget.write(budget.root / 'evaluation' / 'visuals' / (category + '.npz'), stream.getvalue())
            cache[category] = dict(mesh=mesh, format='npz', vertices_key='vertices', faces_key='faces',
                                  scale_xyz=[scale] * 3, local_transform=np.eye(4).tolist(), scale_applied=False,
                                  source_asset_sha256=asset_pins[asset], provenance='frozen_YCB_visual_loader_recipe',
                                  loader_sha256=asset_pins[loader], metadata_sha256=asset_pins[metadata_path],
                                  scene_builder_sha256=asset_pins[scene_builder])
        result[actor['actor_id']] = dict(visuals=[cache[category]], visual_coverage='complete', visual_coverage_reasons=[])
    return result


def export_visuals(mapping, bodies, budget):
    """Export actual render geometry when the pinned renderer exposes triangles.

    Unsupported render shapes stay explicit identity-only data, never collider substitutes.
    Mesh vertices are unscaled; shape scale and local pose are recorded separately.
    """
    result = {}
    for row in mapping['actors']:
        key = row['actor_id']; body = bodies[key]; visuals = []; reasons = []
        obj = body._objs[0]; entity = getattr(obj, 'entity', obj)
        components = [c for c in getattr(entity, 'components', ()) if type(c).__name__ == 'RenderBodyComponent']
        for component in components:
            shapes = getattr(component, 'render_shapes', ())
            for shape in shapes:
                try:
                    scale = np.asarray(shape.scale, dtype=float)
                    local = transform(shape.local_pose)
                    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0): raise ValueError('render scale')
                    for part in shape.parts:
                        mesh = part.mesh
                        vertices = np.asarray(mesh.vertices, dtype=float)
                        faces = np.asarray(mesh.triangles)
                        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or
                                faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or
                                not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
                            raise ValueError('render triangle format')
                        stream = io.BytesIO(); np.savez_compressed(stream, vertices=vertices, faces=faces)
                        ref = budget.write(budget.root / 'evaluation' / 'visuals' / f'{key}-{len(visuals):03d}.npz', stream.getvalue())
                        visuals.append(dict(mesh=ref, format='npz', vertices_key='vertices', faces_key='faces',
                                            scale_xyz=scale.tolist(), local_transform=local.tolist(), scale_applied=False,
                                            source_asset_sha256=row['asset_binding'].get('source_asset_sha256'),
                                            provenance='runtime_render_shape_not_collision'))
                except (AttributeError, TypeError, ValueError) as exc:
                    reasons.append(type(shape).__name__ + ': ' + str(exc))
        # Partial geometry cannot be passed off as a complete visual mesh.
        result[key] = dict(visuals=[] if reasons else visuals,
                           visual_coverage='complete' if visuals and not reasons else 'unavailable',
                           visual_coverage_reasons=reasons or ([] if visuals else ['no_exportable_render_shapes']))
    return result


def _capture_token(u, read_step):
    # get_state_dict covers physics; counters cover non-physics evaluate mutation.
    return dict(control_step=int(read_step()), simulator=jsonable(u.get_state_dict()),
                counters={name: jsonable(getattr(u, name)) for name in
                          ('elapsed_steps', '_elapsed_steps', 'subtask_pointer', 'subtask_steps_left', 'subtasks_steps_left',
                           'robot_cumulative_force') if hasattr(u, name)})


def capture_native(u, fk, *, run, step, read_step, mapping, bodies, visuals,
                   odometry, budget, read_rgbd=None, scene_reader=None, renderer=None):
    """One camera capture followed by segmentation/Position reads, without stepping."""
    from bvi.eef_arm_sensing import camera_in_base
    if read_rgbd is None:
        from export_real_handoff_spatial_render_case import read_sensor_data_without_evaluate
        read_rgbd = read_sensor_data_without_evaluate
    if scene_reader is None:
        from export_c2_collision_bundle import capture_reference
        scene_reader = capture_reference
    before = _capture_token(u, read_step)
    if before['control_step'] != step: raise ValueError('capture step mismatch')
    q = array(u.agent.robot.qpos)[0].astype(float)
    qvel = array(u.agent.robot.qvel)[0].astype(float)
    if q.shape != (15,) or qvel.shape != (15,) or not np.isfinite(q).all() or not np.isfinite(qvel).all():
        raise ValueError('Fetch proprioception format')
    rgbd = read_rgbd(u)
    captured = {}; calibration = {}; camera_eval = {}; known_ids = {0, *map(int, bodies)}
    base_world = transform(u.agent.base_link.pose)
    for name in ('head', 'hand'):
        sensor = u._sensors['fetch_' + name]
        evaluation = sensor.get_obs(rgb=False, depth=False, position=True, segmentation=True)
        rgb = array(rgbd['fetch_' + name]['rgb']); depth = array(rgbd['fetch_' + name]['depth'])
        segmentation = array(evaluation['segmentation']); position = array(evaluation['position'])
        if rgb.shape != (1, 640, 640, 3) or rgb.dtype != np.uint8: raise ValueError('native RGB must be 640 uint8')
        if depth.shape != (1, 640, 640, 1) or depth.dtype.kind not in 'iu': raise ValueError('native depth must be integer mm')
        if np.any(depth < 0): raise ValueError('negative native depth')
        if segmentation.ndim != 4 or segmentation.shape[:3] != (1, 640, 640) or segmentation.shape[3] < 1 or segmentation.dtype.kind not in 'iu':
            raise ValueError('native segmentation format')
        if position.shape not in ((1, 640, 640, 3), (1, 640, 640, 4)) or position.dtype.kind not in 'iu':
            raise ValueError('native Position must be integer mm OpenGL')
        if renderer is not None and position.dtype != np.dtype(RENDERER_PROFILE['position_dtype']):
            raise ValueError('native Position dtype differs from frozen shader format')
        ids = set(map(int, np.unique(segmentation[0, ..., 0])))
        if not ids <= known_ids: raise ValueError('unmapped visible segmentation IDs')
        intrinsics = array(sensor.camera.get_intrinsic_matrix()).reshape(3, 3).astype(float)
        expected = 640 / (2 * np.tan(1.))
        if not np.isfinite(intrinsics).all() or not np.allclose(intrinsics[[0, 1], [0, 1]], expected, atol=1e-3, rtol=0):
            raise ValueError('native intrinsics/FOV changed')
        camera_base = transform(camera_in_base(fk, q, name))
        camera_world_gl = transform(sensor.camera.get_model_matrix())
        if not np.allclose(np.linalg.inv(base_world) @ camera_world_gl, camera_base @ SAPIEN_FROM_GL, atol=1e-5, rtol=0):
            raise ValueError('FK camera convention does not match native renderer')
        calibration[name] = dict(intrinsics=intrinsics.tolist(), camera_in_base=camera_base.tolist(),
                                 camera_to_base=camera_base.tolist(), convention='sapien')
        camera_eval[name] = dict(camera_to_world=(camera_world_gl @ np.linalg.inv(SAPIEN_FROM_GL)).tolist(),
                                 convention='sapien', intrinsics=intrinsics.tolist())
        captured[name] = (rgb[0].copy(), depth[0, ..., 0].copy(), segmentation[0, ..., 0].copy(), position[0, ..., :3].copy())
    scene = scene_reader(u)
    actors = body_states(mapping, bodies)
    after = _capture_token(u, read_step)
    if canonical(before) != canonical(after): raise ValueError('RGB/segmentation capture crossed or mutated a physical state')
    capture_id = f"{run['run_id']}-s{step:04d}"
    common = dict(schema=SCHEMA, run_id=run['run_id'], capture_id=capture_id, control_step=step)
    sdir = budget.root / 'sensors' / capture_id; edir = budget.root / 'evaluation' / capture_id
    sensors = {}; depths = {}; segs = {}; positions = {}
    for name, (rgb, depth, seg, position) in captured.items():
        sensors[name + '_rgb'] = budget.png(sdir / (name + '_rgb.png'), rgb)
        display = np.where(depth > 0, 1 + np.rint(np.clip(depth.astype(float), 0, 4000) * 254 / 4000), 0).astype(np.uint8)
        sensors[name + '_depth'] = budget.png(sdir / (name + '_depth.png'), display)
        depths[name] = budget.npy(sdir / (name + '_depth-raw.npy'), depth)
        segs[name] = budget.npy(edir / (name + '-segmentation.npy'), seg)
        positions[name] = budget.npy(edir / (name + '-position.npy'), position)
    from scipy.spatial.transform import Rotation
    tcp = transform(fk.tcp(q))
    proprio = dict(observation_id=capture_id, control_step=step, joint_positions=q[3:].tolist(),
                   joint_velocities=qvel[3:].tolist(), qpos=q.tolist(), qvel=qvel.tolist(),
                   tcp_pose=dict(position=tcp[:3, 3].tolist(), quaternion_xyzw=Rotation.from_matrix(tcp[:3, :3]).as_quat().tolist()),
                   odometry=dict(zip(('dx_m', 'dy_m', 'dyaw_rad'), map(float, odometry))))
    sensor_refs = dict(sensors=sensors, depth_raw=depths,
                       calibration=budget.json(sdir / 'robot-camera-calibration.json', calibration),
                       proprioception=budget.json(sdir / 'proprioception.json', proprio))
    actor_receipt = dict(**common, target_actor_id=mapping['target_actor_id'],
                         actors=[{**a, **visuals.get(a['actor_id'], {'visuals': []})} for a in actors])
    evaluation_refs = dict(segmentation=segs, position=positions,
                           position_unit='mm', position_convention='opengl_camera', segmentation_channel=0,
                           actors=budget.json(edir / 'actors.json', actor_receipt),
                           actor_mapping=budget.json(edir / 'actor-mapping.json', {**common, **mapping}),
                           cameras=budget.json(edir / 'cameras-world.json', {**common, 'cameras': camera_eval}),
                           scene_state=budget.json(edir / 'scene-state.json', {**common, 'reference': scene, 'base_world': base_world.tolist()}))
    entry = dict(**common, sensor=sensor_refs, evaluation=evaluation_refs,
                 renderer=renderer,
                 synchronization=dict(before_sha256=digest(before), after_sha256=digest(after),
                                      single_capture=True, physical_steps_added=0))
    entry['binding_sha256'] = digest(entry)
    budget.json(edir / 'binding.json', entry)
    return entry


def verify_entry(entry, root):
    payload = {k: v for k, v in entry.items() if k != 'binding_sha256'}
    if entry.get('schema') != SCHEMA or digest(payload) != entry.get('binding_sha256'):
        raise ValueError('capture binding changed')
    sync = entry['synchronization']
    if sync['before_sha256'] != sync['after_sha256'] or sync['physical_steps_added'] != 0:
        raise ValueError('unsynchronized capture')
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value: bound_path(root, value)
            else:
                for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(entry['sensor']); walk(entry['evaluation'])
    for name in ('actors', 'actor_mapping', 'scene_state', 'cameras'):
        content = json.loads(bound_path(root, entry['evaluation'][name]).read_text(encoding='utf-8'))
        if any(content[k] != entry[k] for k in ('run_id', 'capture_id', 'control_step')):
            raise ValueError('cross-run or cross-step truth transfer')
    return True
