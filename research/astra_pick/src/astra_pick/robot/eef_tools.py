"""Robot-only Fetch tool core. No simulator, scene geometry, or API imports.

Poses are homogeneous transforms in base_link; public quaternions use xyzw.
The adapter supplies measured qpos and an official step callback, never poses
of scene objects. Numerical IK failure means no solution found, not unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

ARM = ('shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
       'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint')
BODY = ('head_pan_joint', 'head_tilt_joint', 'torso_lift_joint')
REST = np.array([0, 0, 0, .386, 0, -.370, .562, -1.032, .695, .955,
                 -.1, 2.077, 0, .015, .015])
FETCH_ORDER = ('root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint',
               'torso_lift_joint', 'head_pan_joint', 'shoulder_pan_joint',
               'head_tilt_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
               'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
               'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint')


def validate_official_controller(controller):
    """Fail before stepping if the live combined controller differs from the pin."""
    expected = {'arm': (0, 7), 'gripper': (7, 8), 'body': (8, 11), 'base': (11, 13)}
    if {k: tuple(v) for k, v in controller.action_mapping.items()} != expected:
        raise ValueError('official controller action layout mismatch')
    for key, names in (('arm', ARM), ('body', BODY)):
        config = controller.controllers[key].config
        if (tuple(config.joint_names) != names or not config.use_delta
                or config.use_target or not config.normalize_action
                or not np.all(np.asarray(config.lower) == -.1)
                or not np.all(np.asarray(config.upper) == .1)):
            raise ValueError('official joint delta controller mismatch')
    grip = controller.controllers['gripper'].config
    if (tuple(grip.joint_names) != ('l_gripper_finger_joint', 'r_gripper_finger_joint')
            or grip.use_delta or not grip.normalize_action
            or not np.all(np.asarray(grip.lower) == -.01)
            or not np.all(np.asarray(grip.upper) == .05)):
        raise ValueError('official gripper controller mismatch')


def vector(value, size):
    out = np.asarray(value, dtype=float)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError('invalid finite vector')
    return out


def transform(value):
    out = np.asarray(value, dtype=float)
    if (out.shape != (4, 4) or not np.isfinite(out).all()
            or not np.allclose(out[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(out[:3, :3]), 1, atol=1e-7)):
        raise ValueError('invalid rigid pose')
    return out.copy()


def pose(position, quaternion_xyzw):
    p, q = vector(position, 3), vector(quaternion_xyzw, 4)
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError('quaternion must be unit xyzw')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q).as_matrix()
    result[:3, 3] = p
    return result


class RobotFK:
    """URDF kinematics only; mesh elements are never read."""
    def __init__(self, urdf: Path, order=FETCH_ORDER):
        self.order = tuple(order)
        self.index = {name: i for i, name in enumerate(order)}
        self.joints, self.limits = [], {}
        xml = ET.parse(urdf).getroot()
        links = {x.get('name') for x in xml.findall('link')}
        children = set()
        for j in xml.findall('joint'):
            name, kind = j.get('name'), j.get('type')
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            children.add(child)
            origin = np.eye(4)
            node = j.find('origin')
            if node is not None:
                origin[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
                origin[:3, :3] = Rotation.from_euler('xyz', np.fromstring(
                    node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            axis_node = j.find('axis')
            axis = np.fromstring(axis_node.get('xyz') if axis_node is not None
                                 else '1 0 0', sep=' ')
            if kind != 'fixed':
                if axis.shape != (3,) or np.linalg.norm(axis) == 0:
                    raise ValueError('invalid joint axis')
                axis /= np.linalg.norm(axis)
                limit = j.find('limit')
                if kind == 'continuous':
                    lo, hi = -np.inf, np.inf
                elif kind in ('revolute', 'prismatic') and limit is not None:
                    lo, hi = float(limit.get('lower')), float(limit.get('upper'))
                else:
                    raise ValueError('unsupported or unbounded joint')
                self.limits[name] = (lo, hi, kind)
            self.joints.append((name, kind, parent, child, origin, axis))
        roots = links - children
        if len(roots) != 1 or set(self.index) != set(self.limits) or len(self.index) != len(order):
            raise ValueError('joint ordering or URDF root mismatch')
        self.root = roots.pop()

    def links(self, qpos):
        qpos = vector(qpos, len(self.order))
        result, pending = {self.root: np.eye(4)}, list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, origin, axis in pending:
                if parent not in result:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind in ('continuous', 'revolute'):
                    motion[:3, :3] = Rotation.from_rotvec(axis*qpos[self.index[name]]).as_matrix()
                elif kind == 'prismatic':
                    motion[:3, 3] = axis*qpos[self.index[name]]
                result[child] = result[parent] @ origin @ motion
            if len(remaining) == len(pending):
                raise ValueError('disconnected URDF')
            pending = remaining
        base_inv = np.linalg.inv(result['base_link'])
        return {name: base_inv @ value for name, value in result.items()}

    def tcp(self, qpos):
        return self.links(qpos)['gripper_link']


@dataclass(frozen=True)
class IKResult:
    accepted: bool
    qpos: np.ndarray
    position_error_m: float
    rotation_error_rad: float


class FetchIK:
    def __init__(self, fk: RobotFK, *, torso=True, position_tol=.002,
                 rotation_tol=.02, max_nfev=180, method='trf'):
        if method not in ('trf', 'dogbox'):
            raise ValueError('bounded IK method must be trf or dogbox')
        self.fk = fk
        self.names = (('torso_lift_joint',) if torso else ()) + ARM
        self.indices = [fk.index[n] for n in self.names]
        self.position_tol, self.rotation_tol = position_tol, rotation_tol
        self.max_nfev = max_nfev
        self.method = method

    def solve(self, target, seed):
        target, seed = transform(target), vector(seed, len(self.fk.order)).copy()
        bounds = []
        for name, i in zip(self.names, self.indices):
            lo, hi, kind = self.fk.limits[name]
            bounds.append((seed[i]-2*np.pi, seed[i]+2*np.pi) if kind == 'continuous' else (lo, hi))
        lo, hi = np.asarray(bounds).T
        def residual(x):
            q = seed.copy()
            q[self.indices] = x
            actual = self.fk.tcp(q)
            return np.r_[actual[:3, 3]-target[:3, 3], Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).as_rotvec()]
        initial = np.clip(seed[self.indices], lo, hi)
        fit = least_squares(residual, initial, bounds=(lo, hi), max_nfev=self.max_nfev, method=self.method,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = seed.copy()
        q[self.indices] = fit.x
        error = residual(fit.x)
        pe, re = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        return IKResult(pe <= self.position_tol and re <= self.rotation_tol, q, pe, re)


def densify(start, targets, translation_step=.02, rotation_step=.1):
    if translation_step <= 0 or rotation_step <= 0:
        raise ValueError('positive sampling steps required')
    previous, output = transform(start), []
    for raw in targets:
        target = transform(raw)
        rotations = Rotation.from_matrix(np.stack([previous[:3, :3], target[:3, :3]]))
        angle = (rotations[1]*rotations[0].inv()).magnitude()
        count = max(1, int(np.ceil(np.linalg.norm(target[:3, 3]-previous[:3, 3])/translation_step)),
                    int(np.ceil(angle/rotation_step)))
        if len(output)+count > 2000:
            raise ValueError('path sampling limit exceeded')
        slerp = Slerp([0, 1], rotations)
        for fraction in np.arange(1, count+1)/count:
            point = np.eye(4)
            point[:3, 3] = previous[:3, 3]*(1-fraction)+target[:3, 3]*fraction
            point[:3, :3] = slerp(fraction).as_matrix()
            output.append(point)
        previous = target
    return output


class FetchDeltaExecutor:
    """Pinned official action layout: arm7, gripper1, head2+torso1, base2.

    read_qpos() supplies fresh measurements; step(action) returns only
    terminated/truncated. All step budgets are shared across tool calls.
    Runtime must validate controller layout/scales/use_target=False before use.
    """
    def __init__(self, ik, read_qpos, step, *, max_steps=200, gripper=1.):
        if ik.fk.order != FETCH_ORDER or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('executor requires pinned Fetch ordering and finite step budget')
        self.ik, self.read_qpos, self.step = ik, read_qpos, step
        self.max_steps, self.steps, self.stopped = max_steps, 0, False
        self.set_gripper_command(gripper)

    def set_gripper_command(self, openness):
        if not np.isfinite(openness) or not 0 <= openness <= 1:
            raise ValueError('gripper openness must be in [0,1]')
        self.gripper = float(openness)

    def action(self, target, measured):
        target, measured = vector(target, 15), vector(measured, 15)
        delta = target-measured
        arm = [self.ik.fk.index[n] for n in ARM]
        body = [self.ik.fk.index[n] for n in BODY]
        # Official joint delta scale 0.1 rad / m; gripper is absolute mimic.
        return np.r_[np.clip(delta[arm]/.1, -1, 1), 2*self.gripper-1,
                     np.clip(delta[body]/.1, -1, 1), 0., 0.]

    def check_path(self, targets):
        q = vector(self.read_qpos(), 15).copy()
        points = densify(self.ik.fk.tcp(q), targets)
        if not points:
            raise ValueError('empty movement path')
        results, minimum_steps = [], 0
        controlled = [self.ik.fk.index[n] for n in (*ARM, *BODY)]
        for point in points:
            result = self.ik.solve(point, q)
            results.append(result)
            if not result.accepted:
                return {'accepted': False, 'reason': 'ik_not_found', 'results': results,
                        'minimum_steps': minimum_steps}
            minimum_steps += max(1, int(np.ceil(np.max(np.abs(result.qpos[controlled]-q[controlled]))/.1)))
            q = result.qpos
        feasible = minimum_steps <= self.max_steps-self.steps
        return {'accepted': feasible, 'reason': 'ok' if feasible else 'step_budget',
                'results': results, 'minimum_steps': minimum_steps, 'final_target': points[-1]}

    def _step(self, action):
        if self.stopped or self.steps >= self.max_steps:
            return False
        status = self.step(action)
        self.steps += 1
        self.stopped = bool(status.get('terminated') or status.get('truncated'))
        return not self.stopped

    def _joint_target(self, target, indices, tolerance=.002):
        while not self.stopped and self.steps < self.max_steps:
            measured = vector(self.read_qpos(), 15)
            if np.max(np.abs(measured[indices]-target[indices])) <= tolerance:
                return True
            if not self._step(self.action(target, measured)):
                return False
        return False

    def move_eef_chunk(self, targets):
        plan = self.check_path(targets)
        if not plan['accepted'] or self.stopped:
            return {'arrived': False, 'plan': plan, 'steps': self.steps}
        indices = [self.ik.fk.index[n] for n in (*ARM, *BODY)]
        for result in plan['results']:
            if not self._joint_target(result.qpos, indices):
                return {'arrived': False, 'plan': plan, 'steps': self.steps}
        measured = self.ik.fk.tcp(self.read_qpos())
        if not plan['results']:
            raise ValueError('empty movement path')
        goal = plan['final_target']
        pe = float(np.linalg.norm(measured[:3, 3]-goal[:3, 3]))
        re = float(Rotation.from_matrix(goal[:3, :3] @ measured[:3, :3].T).magnitude())
        return {'arrived': pe <= self.ik.position_tol and re <= self.ik.rotation_tol,
                'measured_tcp': measured, 'position_error_m': pe, 'rotation_error_rad': re,
                'steps': self.steps, 'plan': plan}

    def move_to(self, target):
        return self.move_eef_chunk([target])

    def set_gripper(self, openness, steps=1):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('positive gripper step count required')
        self.set_gripper_command(openness)
        for _ in range(steps):
            q = vector(self.read_qpos(), 15)
            if not self._step(self.action(q, q)):
                break
        return {'steps': self.steps, 'finger_distance_m': float(sum(vector(self.read_qpos(), 15)[-2:]))}

    def return_to_rest(self, settle_steps=3):
        if not isinstance(settle_steps, int) or settle_steps < 0:
            raise ValueError('nonnegative settling step count required')
        q = vector(self.read_qpos(), 15).copy()
        q[3:-2] = REST[3:-2]
        reached = self._joint_target(q, list(range(3, 13)))
        for _ in range(settle_steps if reached else 0):
            measured = vector(self.read_qpos(), 15)
            if not self._step(self.action(q, measured)):
                break
        measured = vector(self.read_qpos(), 15)
        return {'joint_target_reached': reached, 'max_rest_error': float(np.max(
            np.abs(measured[3:-2]-REST[3:-2]))), 'steps': self.steps,
            'stopped': self.stopped}


def locate_point(pixel, depth_m, intrinsics, camera_in_base, *, convention='opencv'):
    """Depth is optical z metres. SAPIEN camera local axes: x forward,y left,z up."""
    pixel = vector(pixel, 2)
    k = np.asarray(intrinsics, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError('invalid camera intrinsics')
    if not np.isfinite(depth_m) or depth_m <= 0:
        return {'valid': False, 'point_base': None}
    optical = np.linalg.solve(k, np.r_[pixel, 1.])*depth_m
    if convention == 'sapien':
        local = np.array([optical[2], -optical[0], -optical[1]])
    elif convention == 'opencv':
        local = optical
    else:
        raise ValueError('unknown camera convention')
    point = (transform(camera_in_base) @ np.r_[local, 1.])[:3]
    return {'valid': True, 'point_base': point}
