"""Robot sensors only. No target actor, world-pose or scene-geometry access."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .eef_tools import vector
from .eef_compact_config import RESOLUTION
from .eef_arm_visual_contract import audit_observation, canonical, SENSOR_FIELDS, old


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def state_copy(value):
    if isinstance(value, dict): return {k: state_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [state_copy(v) for v in value]
    return array(value).copy()


def state_equal(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return np.array_equal(left, right)


def camera_in_base(fk, q, camera):
    links = fk.links(q)
    if camera == 'head': return links['head_camera_link']
    if camera != 'hand': raise ValueError('robot camera only')
    mount = np.eye(4); mount[:3, 3] = [-.1, 0., .1]
    return links['gripper_link'] @ mount


def observe(u, fk, qpos, semantic, history, remaining, folder, odometry, depth_store, previous_state=None):
    from export_real_handoff_spatial_render_case import read_sensor_data_without_evaluate, depth_visualization_8bit
    before = state_copy(u.get_state_dict())
    q = vector(qpos(), 15).copy()
    data = read_sensor_data_without_evaluate(u)
    if not state_equal(before, state_copy(u.get_state_dict())): raise ValueError('render changed simulator state')
    if not np.array_equal(q, qpos()): raise ValueError('render changed joints')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    sensors = {}; manifest = {}; cameras = {}
    for field, (camera, modality) in SENSOR_FIELDS.items():
        value = array(data[camera][modality])
        if value.shape[0] != 1 or value.shape[1:3] != (RESOLUTION, RESOLUTION): raise ValueError('native 640 sensor required')
        pixels = value[0]
        if modality == 'depth':
            raw = pixels[..., 0]
            np.save(folder/(field+'-raw.npy'), raw)
            pixels = depth_visualization_8bit(raw)
            k = array(u._sensors[camera].camera.get_intrinsic_matrix())
            if k.shape == (1, 3, 3): k = k[0]
            # Pinned FOV is checked, not silently replaced by a guessed intrinsic.
            expected_focal = RESOLUTION / (2*np.tan(1.))
            if k.shape != (3, 3) or not np.allclose(k[[0, 1], [0, 1]], expected_focal, atol=1e-3, rtol=0):
                raise ValueError('runtime 640/FOV intrinsic mismatch')
            short = 'head' if camera == 'fetch_head' else 'hand'
            cameras[short] = {'depth_mm': raw, 'intrinsics': k, 'camera_in_base': camera_in_base(fk, q, short)}
        else:
            if pixels.shape != (RESOLUTION, RESOLUTION, 3): raise ValueError('RGB channels')
            pixels = pixels.astype(np.uint8)
        file = folder/(field+'.png'); Image.fromarray(pixels).save(file)
        encoded = file.read_bytes(); digest = hashlib.sha256(encoded).hexdigest()
        sensors[field] = {'camera': camera, 'modality': modality, 'sha256': digest, 'data_base64': base64.b64encode(encoded).decode()}
        manifest[field] = {'camera': camera, 'modality': modality, 'sha256': digest, 'width': RESOLUTION, 'height': RESOLUTION}
    turn = len(history)
    identifier = f'obs-{turn:03d}-' + hashlib.sha256(canonical(manifest)+q.tobytes()).hexdigest()[:16]
    depth_store.set(identifier, q, cameras)
    tcp = fk.tcp(q); velocity = array(u.agent.robot.qvel)[0]
    obs = {'target_semantics': semantic, 'observation_id': identifier,
        'joint_positions': dict(zip(old.PROPRIO_KEYS, map(float, q[3:]))),
        'joint_velocities': dict(zip(old.PROPRIO_KEYS, map(float, velocity[3:]))),
        'tcp_pose': {'position': tcp[:3, 3].tolist(), 'quaternion_xyzw': Rotation.from_matrix(tcp[:3, :3]).as_quat().tolist()},
        'sensors': sensors, 'steps_remaining': int(remaining), 'calls_remaining': 25-turn,
        'history': copy.deepcopy(history), 'previous_state': copy.deepcopy(previous_state),
        'odometry': dict(zip(('dx_m', 'dy_m', 'dyaw_rad'), map(float, odometry)))}
    audit_observation(obs, manifest, turn)
    (folder/'sensor-manifest.json').write_bytes(canonical(manifest))
    (folder/'robot-camera-calibration.json').write_bytes(canonical({n: {k: v.tolist() for k, v in c.items() if k != 'depth_mm'} for n, c in cameras.items()}))
    return obs, manifest
