"""C2-only physical preparation, arrival verification and single-use SAC retry tickets.

No simulator setters, resets, teleport or policy changes. A scene motion validator
must approve each path from the live state; legacy distance bins cannot supply it.

STATUS: active — executors
"""
from __future__ import annotations

import math

from .protocol import (ProtocolError, RequirementResult, RequirementState,
                       SkillFeedback, SkillStatus)


def _vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size and
            all(type(v) in (float, int) and math.isfinite(v) for v in value))


def audit_pose_prior(data):
    """Report missing evidence without upgrading historical assets in place."""
    reasons = []
    candidates = data.get('candidates', [])
    if data.get('schema') != 'spawn-prior/2':
        reasons.append('legacy_distance_bins_are_not_pose_candidates')
    if not isinstance(candidates, list) or not candidates:
        reasons.append('no_pose_candidates')
        candidates = []
    seen = set()
    for c in candidates:
        if not isinstance(c, dict):
            reasons.append('invalid_candidate'); continue
        key = c.get('id')
        if not isinstance(key, str) or not key or key in seen:
            reasons.append('missing_or_duplicate_candidate_id')
        else:
            seen.add(key)
        if (c.get('skill') not in ('pick', 'place') or
                not c.get('object_category') or c.get('frame') != 'target_se2' or
                c.get('units') != {'position': 'm', 'angle': 'rad'} or
                not _vector(c.get('base_pose'), 3) or
                not _vector(c.get('arm_qpos'), 7) or
                not _vector(c.get('body_qpos'), 3) or
                type(c.get('target_height_m')) not in (float, int) or
                not math.isfinite(c['target_height_m'])):
            reasons.append('incomplete_pose_geometry')
        if c.get('skill') == 'place' and not _vector(c.get('object_pose_wrt_tcp'), 7):
            reasons.append('missing_place_grasp_transform')
        evidence = c.get('evidence', {})
        if (not isinstance(evidence, dict) or not evidence.get('source_sha256') or
                not evidence.get('checkpoint_sha256') or
                evidence.get('parent_uid') not in data.get('training_plan_uids', []) or
                type(evidence.get('samples')) is not int or evidence['samples'] < 1 or
                type(evidence.get('successes')) is not int or
                not 0 <= evidence['successes'] <= evidence['samples']):
            reasons.append('missing_candidate_provenance')
    return {'ready': not reasons, 'candidate_count': len(candidates),
            'reasons': sorted(set(reasons)),
            'scope': 'pose_data_only_live_scene_and_controller_validation_still_required'}


class PoseRecovery:
    def __init__(self, adapter, prior, backend=None):
        self.adapter, self.prior, self.backend = adapter, prior, backend
        self.audit = audit_pose_prior(prior)
        self.attempts = []
        self.sac_starts = []
        self.used = set()
        self.ready = None
        self.active = None

    @staticmethod
    def pose_key(c):
        # Millimetre/milliradian resolution prevents aliases bypassing dedup.
        return (c['skill'], c['frame'], *[round(v, 3) for name in
                ('base_pose', 'arm_qpos', 'body_qpos') for v in c[name]])

    @staticmethod
    def equivalent(a, b):
        from .lightnav_skill import wrap
        return (a['skill'] == b['skill'] and a['frame'] == b['frame'] and
                math.hypot(a['base_pose'][0]-b['base_pose'][0],
                           a['base_pose'][1]-b['base_pose'][1]) <= .04 and
                abs(wrap(a['base_pose'][2]-b['base_pose'][2])) <= .06 and
                all(abs(x-y) <= .03 for field in ('arm_qpos', 'body_qpos')
                    for x,y in zip(a[field], b[field])))

    def offers(self):
        if not self.audit['ready'] or self.backend is None:
            return []
        offers = []
        for goal in self.adapter.catalog.goals:
            for skill, target in (('pick', goal['object_id']), ('place', goal['destination_id'])):
                pair = skill, target
                ledger = self.adapter.retry_ledger
                if (pair not in ledger.attempted or not ledger.can_call(pair) or
                        sum(a['pair'] == pair for a in self.attempts) >= 3):
                    continue
                for c in self.prior['candidates']:
                    if (c['skill'] != skill or c['object_category'] != goal['object_name'].split('-')[0]
                            or (target, self.pose_key(c)) in self.used):
                        continue
                    if any(a['pair'] == pair and self.equivalent(c, a['pose'])
                           for a in self.attempts):
                        continue
                    # The backend must check checkpoint identity, current geometry,
                    # joint limits and Place grasp compatibility without stepping.
                    excluded = [a['pose'] for a in self.sac_starts if a['pair'] == pair]
                    if self.backend.feasible(c, target, excluded):
                        offers.append({**c, 'target_id': target})
        return offers

    def snapshot(self):
        return {'prior_audit': self.audit, 'backend_available': self.backend is not None,
                'candidates': self.offers(), 'attempted_poses': self.attempts,
                'sac_start_poses': self.sac_starts,
                'verified_retry': self.ready,
                'limits': {'preparations_per_manipulation_target': 3,
                           'preparation_actions': 200}}

    def can_retry(self, pair, step):
        return bool(self.ready and tuple(self.ready['pair']) == pair and self.ready['step'] == step)

    def invalidate(self):
        self.ready = None

    def record_sac_start(self, request):
        if self.backend is not None and request.skill in ('pick', 'place'):
            record = {'pair': (request.skill, request.target_id), 'call_id': request.call_id,
                      'pose': self.backend.snapshot()}
            self.sac_starts.append(record)
            self.adapter.logger.emit('sac_start_pose', **record)

    def begin(self, request):
        candidate = next((c for c in self.offers() if c['id'] == request.recovery_candidate_id
                          and c['target_id'] == request.target_id), None)
        if candidate is None:
            raise ProtocolError('Recovery candidate is no longer feasible')
        self.invalidate()
        self.used.add((request.target_id, self.pose_key(candidate)))
        self.active = {'pair': (candidate['skill'], request.target_id),
                       'candidate_id': candidate['id'], 'pose': candidate,
                       'call_id': request.call_id, 'status': 'running', 'steps': 0,
                       'before': self.backend.snapshot()}
        self.attempts.append(self.active)
        self.backend.start(candidate, request.target_id)
        self.adapter.logger.emit('pose_preparation_started', **self.active)

    def finish(self, request, succeeded):
        if self.active is None or self.active['call_id'] != request.call_id:
            return
        self.active['status'] = 'arrived' if succeeded else 'not_arrived'
        self.active['after'] = self.backend.snapshot()
        if succeeded:
            self.ready = {'pair': self.active['pair'], 'step': self.adapter.steps,
                          'candidate_id': self.active['candidate_id']}
        self.adapter.logger.emit('pose_preparation_finished', **self.active)
        self.active = None


class RepositionSkill:
    """A separate GPT-selected physical tool; SAC is never called on failed arrival."""
    def __init__(self, adapter):
        self.adapter = adapter

    def start(self, request, observation):
        self.adapter.recovery.begin(request)

    def act(self, observation):
        return self.adapter.recovery.backend.action()

    def feedback(self, request, transition):
        recovery = self.adapter.recovery
        recovery.active['steps'] += 1
        arrived = recovery.backend.arrived()
        evidence = f'events.jsonl:pose_arrival:{request.call_id}:{transition.observation.frame_id}'
        self.adapter.logger.emit('pose_arrival', call_id=request.call_id,
                                 frame_id=transition.observation.frame_id,
                                 arrived=arrived, measured=recovery.backend.snapshot())
        return SkillFeedback(SkillStatus.SUCCEEDED if arrived else SkillStatus.EXECUTING,
            tuple(RequirementResult(r.id, RequirementState.SATISFIED if arrived else
                  RequirementState.UNSATISFIED, (evidence,)) for r in request.requirements),
            'pose_arrived' if arrived else None, 'measured_robot_pose')


class FetchPosePreparation:
    """Closed-loop base/arm/body actions through the existing Fetch controller.

    validate_motion(candidate, target_id, snapshot) returns a validated route in
    root-joint coordinates, or None. It must recheck live collision geometry,
    checkpoint/target identity, support height, and (for Place) held-object pose.
    No validator is installed implicitly: unverified scene planning fails closed.
    """
    def __init__(self, adapter, validate_motion=None):
        from .lightnav_skill import FetchNavigationControl
        self.control = FetchNavigationControl(adapter)
        self.validate_motion = validate_motion
        self.route = None
        self.settled = 0

    def snapshot(self):
        from .lightnav_skill import vector
        return {'base_pose': self.control.pose(),
                'arm_qpos': vector(self.control.controllers['arm'].qpos),
                'body_qpos': vector(self.control.controllers['body'].qpos)}

    def _route(self, candidate, target):
        if self.validate_motion is None:
            return None
        route = self.validate_motion(candidate, target, self.snapshot())
        if route is None:
            return None
        if (not isinstance(route, list) or not route or
                any(not _vector(p, 3) for p in route)):
            raise ProtocolError('Motion validator returned an invalid root-frame route')
        return route

    def feasible(self, candidate, target, excluded=()):
        from .lightnav_skill import wrap
        route = self._route(candidate, target)
        if route is None:
            return False
        def changed(state):
            return (math.hypot(state['base_pose'][0]-route[-1][0],
                               state['base_pose'][1]-route[-1][1]) > .04 or
                    abs(wrap(state['base_pose'][2]-route[-1][2])) > .06 or
                    any(abs(a-b) > .03 for name in ('arm_qpos', 'body_qpos')
                        for a, b in zip(state[name], candidate[name])))
        return all(changed(state) for state in [self.snapshot(), *excluded])

    def start(self, candidate, target):
        self.route = self._route(candidate, target)
        if self.route is None:
            raise ProtocolError('No validated physical recovery route')
        self.control.capture_hold()  # Preserves the real gripper command.
        self.candidate = candidate
        self.waypoint = 0
        self.settled = 0

    def action(self):
        from .lightnav_skill import track_waypoint
        linear, angular, reached = track_waypoint(self.control.pose(), self.route[self.waypoint])
        if reached and self.waypoint + 1 < len(self.route):
            self.waypoint += 1
        elif reached:
            self.control.hold['arm'] = tuple(self.candidate['arm_qpos'])
            self.control.hold['body'] = tuple(self.candidate['body_qpos'])
        return self.control.action(linear, angular)

    def arrived(self):
        from .lightnav_skill import track_waypoint
        state = self.snapshot()
        reached = track_waypoint(state['base_pose'], self.route[-1])[2]
        joints = all(abs(a-b) <= .03 for name in ('arm_qpos', 'body_qpos')
                     for a, b in zip(state[name], self.candidate[name]))
        self.settled = self.settled + 1 if reached and joints else 0
        return self.settled >= 3
