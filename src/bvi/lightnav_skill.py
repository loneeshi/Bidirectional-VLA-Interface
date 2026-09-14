"""Experimental LightNav -> Fetch serial skill. Live control is unvalidated.

Pinned controller semantics: ManiSkill17121e3f. Robot proprioception is used by
the tracker/hold controller; no simulator goal positions enter LightNav prompts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .mshab_adapter import benchmark_feedback, jsonable
from .protocol import ProtocolError


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def world_waypoint(anchor, waypoint):
    """Each predicted row is cumulative in the capture frame, never an increment."""
    x, y, yaw = anchor
    dx, dy, dyaw = waypoint
    c, s = math.cos(yaw), math.sin(yaw)
    return x+c*dx-s*dy, y+s*dx+c*dy, wrap(yaw+dyaw)


@dataclass(frozen=True)
class TrackerLimits:
    linear: float = .25
    angular: float = .6
    position_tolerance: float = .04
    yaw_tolerance: float = .06


def track_waypoint(pose, target, limits=TrackerLimits()):
    """Conservative forward-only controller, rotating before advancing sideways."""
    if not all(math.isfinite(v) for v in (*pose, *target)):
        raise ProtocolError("Nonfinite planar pose")
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    distance = math.hypot(dx, dy)
    yaw_error = wrap(target[2]-pose[2])
    if distance < limits.position_tolerance:
        reached = abs(yaw_error) < limits.yaw_tolerance
        return 0., max(-limits.angular, min(limits.angular, 2*yaw_error)), reached
    heading = wrap(math.atan2(dy, dx)-pose[2])
    angular = max(-limits.angular, min(limits.angular, 2*heading))
    linear = min(limits.linear, distance) * max(0., math.cos(heading))
    if abs(heading) > math.pi/4:
        linear = 0.
    return linear, angular, False


def normalize(values, low, high):
    if len(values) != len(low) or len(low) != len(high):
        raise ProtocolError("Controller/action dimensions differ")
    if any(not math.isfinite(v) for v in (*values, *low, *high)):
        raise ProtocolError("Nonfinite physical control value")
    if any(hi <= lo for lo, hi in zip(low, high)):
        raise ProtocolError("Invalid physical controller limits")
    return tuple(max(-1., min(1., 2*(v-lo)/(hi-lo)-1))
                 for v, lo, hi in zip(values, low, high))


def vector(value):
    return tuple(float(x) for x in jsonable(value)[0])


class FetchNavigationControl:
    """Discover and assert the installed controller contract before any inference."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.controllers = adapter.uenv.agent.controller.controllers
        if tuple(self.controllers) != ('arm', 'gripper', 'body', 'base'):
            raise ProtocolError("Unknown Fetch flattened controller order")
        expected = {
            'arm': (7, 'PDJointPosController'),
            'gripper': (1, 'PDJointPosMimicController'),
            'body': (3, 'PDJointPosController'),
            'base': (2, 'PDBaseForwardVelController'),
        }
        self.ranges = {}
        for name, (dim, cls) in expected.items():
            ctrl = self.controllers[name]
            if type(ctrl).__name__ != cls or not ctrl._normalize_action:
                raise ProtocolError("Installed controller differs from pinned contract")
            space = ctrl._original_single_action_space
            if tuple(space.shape) != (dim,):
                raise ProtocolError("Unexpected physical action dimension")
            self.ranges[name] = (tuple(float(v) for v in space.low),
                                 tuple(float(v) for v in space.high))
            if name in ('arm', 'body') and (not ctrl.config.use_delta or ctrl.config.use_target):
                raise ProtocolError("Hold requires deltas relative to measured qpos")
        if self.controllers['gripper'].config.use_delta:
            raise ProtocolError("Gripper must accept an absolute position")
        if list(self.controllers['base'].config.joint_names) != [
            'root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint']:
            raise ProtocolError("Base coordinate convention changed")
        if list(self.controllers['body'].config.joint_names) != [
            'head_pan_joint', 'head_tilt_joint', 'torso_lift_joint']:
            raise ProtocolError("Body joint ordering changed")
        if self.controllers['base'].control_freq != 20:
            raise ProtocolError("Tracker cadence is defined for20Hz control")
        adapter.logger.emit('navigation_controller_contract', ranges=self.ranges,
                            base_frame='root_joint_planar_frame', frequency_hz=20)
        self.hold = None

    def capture_hold(self):
        self.hold = {name: vector(self.controllers[name].qpos) for name in ('arm', 'body')}
        grip = self.controllers['gripper']
        # Preserve the last commanded closing force, not merely finger separation.
        target = vector(grip._target_qpos)
        if len(target) != 2 or abs(target[0]-target[1]) > 1e-5:
            raise ProtocolError("Mimic gripper targets disagree")
        self.hold['gripper'] = (target[0],)
        self.adapter.logger.emit('navigation_hold_captured', targets=self.hold)

    def pose(self):
        return vector(self.controllers['base'].qpos)

    def action(self, linear, angular):
        if self.hold is None:
            raise ProtocolError("Missing navigation hold state")
        result = []
        for name in ('arm', 'gripper', 'body', 'base'):
            if name in ('arm', 'body'):
                current = vector(self.controllers[name].qpos)
                physical = tuple(t-q for t, q in zip(self.hold[name], current))
            elif name == 'gripper':
                physical = self.hold[name]
            else:
                physical = (linear, angular)
            result.extend(normalize(physical, *self.ranges[name]))
        return self.adapter.action_bounds.validate(result)


class LightNavSkill:
    def __init__(self, adapter, client, instructions, max_predictions=40, replan_steps=5):
        if max_predictions < 1 or replan_steps < 1:
            raise ProtocolError("Positive prediction and cadence limits required")
        self.adapter, self.client, self.instructions = adapter, client, instructions
        self.control = FetchNavigationControl(adapter)
        self.max_predictions, self.replan_steps = max_predictions, replan_steps
        self.total_predictions = 0
        self.targets = ()
        self.index = None

    def start(self, request, observation):
        self.index = int(observation.metadata['subtask_index'])
        if request.skill != 'navigate':
            raise ProtocolError("LightNav only implements navigation")
        self.instruction = self.instructions.get(str(self.index), '')
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ProtocolError("An explicit visual navigation instruction is required")
        self.client.reset()  # Also clear history between distinct navigation goals.
        self.control.capture_hold()
        self.targets, self.target_index = (), 0
        self.last_inference = -self.replan_steps
        self.last_step = None
        self.call_id = request.call_id

    def act(self, observation):
        if self.index is None or self.last_step == observation.sim_step:
            raise ProtocolError("Missing start or duplicate action request")
        self.last_step = observation.sim_step
        pose = self.control.pose()
        if not self.targets or observation.sim_step-self.last_inference >= self.replan_steps:
            if self.total_predictions >= self.max_predictions:
                raise ProtocolError("LightNav experiment prediction cap reached")
            camera = next(image for image in observation.images if image.camera == 'fetch_head')
            self.total_predictions += 1
            self.adapter.logger.emit('navigation_inference_started', call_id=self.call_id,
                attempt=self.total_predictions, frame_id=observation.frame_id,
                instruction=self.instruction, capture_pose=pose)
            prediction = self.client.infer(camera, self.instruction)
            self.adapter.logger.emit('navigation_prediction', call_id=self.call_id,
                frame_id=observation.frame_id, prediction=prediction)
            if prediction.stop:
                raise ProtocolError("Model stopped without benchmark completion")
            self.targets = tuple(world_waypoint(pose, row) for row in prediction.waypoints)
            self.target_index, self.last_inference = 0, observation.sim_step
        if not self.targets:
            raise ProtocolError("No predicted trajectory")
        linear, angular, reached = track_waypoint(pose, self.targets[self.target_index])
        while reached and self.target_index < len(self.targets)-1:
            self.target_index += 1
            linear, angular, reached = track_waypoint(pose, self.targets[self.target_index])
        action = self.control.action(linear, angular)
        self.adapter.logger.emit('navigation_control', call_id=self.call_id,
            pose=pose, target=self.targets[self.target_index], linear_m_s=linear,
            angular_rad_s=angular, action=action)
        return action

    def feedback(self, request, transition):
        return benchmark_feedback(request, transition, self.index)
