"""C2 structured base goals and audited static-floor local preparation.

STATUS: active — executors

Independent implementation inspired by GPT-Policy's target/measurement boundary.
The official floor mesh checks static base navigability, NOT arm/dynamic collision.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .lightnav_skill import world_waypoint, wrap
from .pose_recovery import PoseRecovery, FetchPosePreparation, RepositionSkill
from .protocol import ProtocolError, SkillFeedback, SkillStatus, RequirementResult, RequirementState


def validate_goal(goal):
    if not isinstance(goal, dict) or set(goal) != {'frame', 'x_m', 'y_m', 'yaw_rad'}:
        raise ProtocolError('Recovery goal requires frame/x_m/y_m/yaw_rad only')
    if goal['frame'] != 'base_at_request':
        raise ProtocolError('Unsupported recovery coordinate frame')
    values = [goal[k] for k in ('x_m', 'y_m', 'yaw_rad')]
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise ProtocolError('Nonfinite recovery goal')
    if math.hypot(*values[:2]) > .5 or abs(values[2]) > math.pi / 2:
        raise ProtocolError('Recovery goal exceeds 0.5m / pi/2 bound')
    if math.hypot(*values[:2]) <= .04 and abs(values[2]) <= .06:
        raise ProtocolError('Recovery goal is equivalent to current pose')
    return values


def goal_schema():
    props = {'frame': {'type': 'string', 'enum': ['base_at_request']},
             'x_m': {'type': 'number'}, 'y_m': {'type': 'number'},
             'yaw_rad': {'type': 'number'}}
    return {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': props,
            'required': list(props), 'additionalProperties': False}]}


def same_base(a, b):
    return (math.hypot(a[0]-b[0], a[1]-b[1]) <= .04 and abs(wrap(a[2]-b[2])) <= .06)


def recovery_velocity(pose, target):
    """Bounded reversible base tracking: retreat need not spin a held object."""
    from .lightnav_skill import track_waypoint
    if not all(math.isfinite(x) for x in (*pose,*target)):
        raise ProtocolError('Nonfinite recovery tracker pose')
    dx,dy=target[0]-pose[0],target[1]-pose[1]
    distance=math.hypot(dx,dy)
    if distance < .04:return track_waypoint(pose,target)
    heading=wrap(math.atan2(dy,dx)-pose[2]); direction=1.
    if abs(heading)>math.pi/2:
        heading=wrap(heading+math.pi);direction=-1.
    angular=max(-.6,min(.6,2*heading))
    linear=direction*min(.15,distance)*max(0.,math.cos(heading))
    if abs(heading)>math.pi/4:linear=0.
    return linear,angular,False


class GoalPoseRecovery(PoseRecovery):
    """Same single-use tickets as legacy candidates; targets proposed by GPT."""
    def __init__(self, adapter, backend=None):
        super().__init__(adapter, {}, backend)

    def eligible(self):
        ledger = self.adapter.retry_ledger
        return sorted(pair for pair in ledger.attempted if pair[0] in ('pick', 'place')
                      and ledger.can_call(pair)
                      and sum(a['pair'] == pair for a in self.attempts) < 3)

    def offers(self):
        return []

    def snapshot(self):
        return {'mode': 'structured_base_goal/1', 'candidates': [],
                'eligible_targets': [dict(skill=s, target_id=t) for s,t in self.eligible()],
                'backend_available': self.backend is not None,
                'robot': self.backend.snapshot() if self.backend else None,
                'geometry_scope': 'official_static_base_floor_mesh_not_full_body_collision',
                'attempted_poses': self.attempts, 'sac_start_poses': self.sac_starts,
                'verified_retry': self.ready,
                'limits': {'preparations_per_manipulation_target': 3,
                           'preparation_actions': 200, 'translation_m': .5,
                           'rotation_rad': math.pi/2},
                'coordinate_convention': 'base_at_request: x forward, y left, yaw CCW; m/rad'}

    def prepare(self, request):
        delta = validate_goal(request.recovery_goal)
        matches = [p for p in self.eligible() if p[1] == request.target_id]
        if len(matches) != 1 or self.backend is None:
            raise ProtocolError('No failed manipulation target available for recovery')
        pair = matches[0]
        current = self.backend.snapshot()
        goal = list(world_waypoint(current['base_pose'], delta))
        previous = [a['pose']['base_pose'] for a in self.sac_starts if a['pair'] == pair]
        previous += [a['pose']['base_pose'] for a in self.attempts if a['pair'] == pair]
        if any(same_base(goal, old) for old in [current['base_pose'], *previous]):
            raise ProtocolError('Recovery goal repeats a previously attempted base pose')
        candidate = dict(id=request.call_id, skill=pair[0], frame='root_joint_planar_frame',
                         base_pose=goal, arm_qpos=current['arm_qpos'], body_qpos=current['body_qpos'])
        if not self.backend.feasible(candidate, request.target_id):
            raise ProtocolError('Recovery goal unavailable: static floor or held-object check failed')
        return pair, candidate, current

    def begin(self, request):
        pair, candidate, current = self.prepare(request)
        self.backend.start(candidate, request.target_id)
        self.invalidate()
        self.active = {'pair': pair, 'candidate_id': request.call_id, 'pose': candidate,
                       'requested_goal': dict(request.recovery_goal), 'call_id': request.call_id,
                       'status': 'running', 'steps': 0, 'before': current}
        self.attempts.append(self.active)
        self.adapter.logger.emit('pose_preparation_started', **self.active)


class FloorMesh:
    """Test a short straight segment against the *union* of official triangles.

    Interval clipping checks continuous containment, not just sampled endpoints.
    No snapping, path substitution, Delaunay filling, convex hull or state changes.
    """
    def __init__(self, path):
        self.path = Path(path)
        vertices, faces = [], []
        for line in self.path.read_text().splitlines():
            parts = line.split()
            if not parts: continue
            if parts[0] == 'v': vertices.append(tuple(map(float, parts[1:3])))
            if parts[0] == 'f':
                if len(parts) != 4: raise ProtocolError('Expected triangular official floor map')
                faces.append([int(x.split('/')[0])-1 for x in parts[1:]])
        if not vertices or not faces: raise ProtocolError('Missing official floor triangles')
        if any(i < 0 or i >= len(vertices) for f in faces for i in f):
            raise ProtocolError('Invalid floor mesh index')
        self.triangles = [tuple(vertices[i] for i in f) for f in faces]
        if any(not math.isfinite(v) for p in vertices for v in p):
            raise ProtocolError('Nonfinite floor map')
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def contains_segment(self, start, end):
        def cross(a,b): return a[0]*b[1]-a[1]*b[0]
        d = (end[0]-start[0], end[1]-start[1])
        intervals = []
        for tri in self.triangles:
            a,b,c = tri
            orient = cross((b[0]-a[0], b[1]-a[1]), (c[0]-a[0], c[1]-a[1]))
            if abs(orient) < 1e-12: continue
            sign = 1 if orient > 0 else -1
            lo, hi = 0., 1.
            for a,b in zip(tri, (*tri[1:], tri[0])):
                edge = b[0]-a[0], b[1]-a[1]
                v = sign*cross(edge, (start[0]-a[0], start[1]-a[1]))
                rate = sign*cross(edge, d)
                if abs(rate) < 1e-12:
                    if v < -1e-9: hi = -1.; break
                elif rate > 0: lo = max(lo, (-1e-9-v)/rate)
                else: hi = min(hi, (-1e-9-v)/rate)
            if lo <= hi: intervals.append((lo,hi))
        covered = 0.
        for lo,hi in sorted(intervals):
            if lo > covered + 1e-7: return False
            covered = max(covered,hi)
            if covered >= 1.-1e-7: return True
        return False


class StaticFloorValidator:
    def __init__(self, adapter, asset_dir):
        self.adapter = adapter
        u = adapter.uenv
        if len(u.build_config_idxs) != 1: raise ProtocolError('One environment required')
        bc = u.scene_builder.build_configs[int(u.build_config_idxs[0])]
        filename = Path(bc).stem + '.fetch.navigable_positions_simplified.obj'
        self.mesh = FloorMesh(Path(asset_dir)/'scene_datasets/replica_cad_dataset/configs/scenes'/filename)
        adapter.logger.emit('recovery_floor_map', path=str(self.mesh.path), sha256=self.mesh.sha256,
                            scope='static_base_only_not_full_body_or_dynamic_collision')

    def __call__(self, candidate, target, snapshot):
        from .mshab_adapter import jsonable, scalar
        u = self.adapter.uenv
        if candidate['frame'] != 'root_joint_planar_frame': return None
        pair = candidate['skill'], target
        index = self.adapter.catalog.bindings[pair]
        if pair[0] == 'place' and not bool(scalar(u.agent.is_grasping(u.subtask_objs[index], max_angle=30))):
            return None
        # Derive root->world SE2 from fresh base_link measurement, never assume identity.
        p = jsonable(u.agent.base_link.pose.raw_pose)[0]
        w,x,y,z = p[3:]
        world_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        root = snapshot['base_pose']; end = candidate['base_pose']
        rotation = world_yaw-root[2]
        dx,dy = end[0]-root[0],end[1]-root[1]
        world_end = (p[0]+math.cos(rotation)*dx-math.sin(rotation)*dy,
                     p[1]+math.sin(rotation)*dx+math.cos(rotation)*dy)
        return [end] if self.mesh.contains_segment(p[:2],world_end) else None


class GoalFetchPreparation(FetchPosePreparation):
    def snapshot(self):
        state = super().snapshot()
        from .lightnav_skill import vector
        state['gripper_command']=vector(self.control.controllers['gripper']._target_qpos)
        if getattr(self,'target_id',None) is not None:
            from .mshab_adapter import scalar,jsonable
            adapter=self.control.adapter
            pair=(self.candidate['skill'],self.target_id)
            obj=adapter.uenv.subtask_objs[adapter.catalog.bindings[pair]]
            state['held_target']={'target_id':self.target_id,
                'is_grasped':bool(scalar(adapter.uenv.agent.is_grasping(obj,max_angle=30))),
                'object_pose_wrt_tcp':jsonable((adapter.uenv.agent.tcp.pose.inv()*obj.pose).raw_pose)}
        state.update(frame='root_joint_planar_frame', position_unit='m', angle_unit='rad',
                     body_joint_names=list(self.control.controllers['body'].config.joint_names),
                     arm_joint_names=list(self.control.controllers['arm'].config.joint_names))
        if getattr(self,'route',None):
            target=self.route[-1]; pose=state['base_pose']
            state['arrival_error']={'position_m':math.hypot(target[0]-pose[0],target[1]-pose[1]),
                                    'yaw_rad':abs(wrap(target[2]-pose[2]))}
        return state

    def start(self, candidate, target):
        super().start(candidate, target)
        self.target_id = target

    def action(self):
        # Revalidate live drift and, for Place, holding before each physical step.
        if self._route(self.candidate, self.target_id) is None:
            raise ProtocolError('Preparation geometry/holding became invalid')
        linear,angular,_=recovery_velocity(self.control.pose(),self.route[-1])
        if linear < 0 and self.control.ranges['base'][0][0] >= 0:
            raise ProtocolError('Installed controller does not permit reverse velocity')
        return self.control.action(linear,angular)

    def arrived(self):
        return self._route(self.candidate, self.target_id) is not None and super().arrived()


class GoalRepositionSkill(RepositionSkill):
    def start(self, request, observation):
        self.blocked=False
        super().start(request, observation)

    def act(self, observation):
        backend=self.adapter.recovery.backend
        if backend._route(backend.candidate,backend.target_id) is None:
            self.blocked=True
            return backend.control.action(0.,0.)
        return backend.action()

    def feedback(self, request, transition):
        if self.blocked:
            self.adapter.recovery.active['steps'] += 1
            return SkillFeedback(SkillStatus.INTERRUPTED,tuple(RequirementResult(
                r.id,RequirementState.UNSATISFIED) for r in request.requirements),
                'recovery_geometry_or_holding_changed','measured_robot_pose')
        return super().feedback(request,transition)
