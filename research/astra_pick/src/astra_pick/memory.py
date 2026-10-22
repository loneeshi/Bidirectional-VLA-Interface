"""Bounded episode evidence, independent of planning and collision checking.

All positions use one episode-local odometry frame.  This module owns stored
evidence and aging; it never extracts depth, plans a path, or declares safety.
Missing, evicted and expired geometry are unknown, never observed free space.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import re


VERSION = "episode-memory-v1"
DEFAULT_LIMITS = dict(max_observations=256, max_targets=32, max_regions=32,
    max_geometry=8192, max_voxel_points=131072, max_actions=32, max_route=64, free_ttl_steps=40,
    occupied_ttl_steps=200, unknown_ttl_steps=40)
_HARD_LIMITS = dict(max_observations=1024, max_targets=128, max_regions=128,
    max_geometry=16384, max_voxel_points=131072, max_actions=128, max_route=256, free_ttl_steps=200,
    occupied_ttl_steps=1200, unknown_ttl_steps=200)
_SENSOR_NAMES = {"head_rgb", "head_depth", "hand_rgb", "hand_depth"}
_MAX_CLOUD_POINTS = 32768
_MAX_OBSERVATION_POINTS = 65536
_SUMMARY_MEANINGS = "positions:last_seen_not_grasp;regions:visible_part_only;route:not_free;surfaces:not_full_obstacle_shape"
_SURFACE_CELL_M = .25
_SURFACE_NEAR_M = 3.
_MAX_SUMMARY_SURFACES = 4


def plan_state_card(observation, history, archive, *, previous=None, budget=None):
    """S2 evidence index, reconstructed from public records, never scene truth.

    Beliefs retain their model provenance. The card neither authorizes a close
    nor claims possession/safety from completion. A planned command without a
    dispatched/measured prefix is not counted as an attempted strategy.
    """
    oid=observation['observation_id']
    version=len(history)
    measured={key:dict(value=_copy(observation[key]),source='robot_sensor',
        observation_id=oid,state_version=version,frame_id=oid,validity='current')
        for key in ('joint_positions','joint_velocities','tcp_pose','odometry')}
    targets={name:dict(status='unknown',source=None,evidence_refs=[])
             for name in ('identity','localization','alignment','closure','possession','rest')}
    invalidations=[];strategies=[];beliefs={};pending=[];target_changes={};last_target=None
    history_sources={}
    blocked_arm=False;repositioned=False
    arm_tools=('prepare_grasp','replan_grasp','move_to','move_eef_chunk','set_gripper','return_to_rest')
    # One archive entry is a provider decision. Its previous history prefix
    # locates that decision's actual commands; plans alone never count as tried.
    for index,item in enumerate(archive):
        plan=item.get('call')
        if not isinstance(plan,dict) or plan.get('schema_version')!='sol-pick-plan-v1':
            continue
        start=len(item['bundle']['observation']['history'])
        stop=(len(archive[index+1]['bundle']['observation']['history']) if index+1<len(archive) else len(history))
        entries=history[start:stop]
        for offset,row in enumerate(entries):
            source=item['bundle']['observation']['observation_id']
            commands=plan.get('commands',[])
            cid=commands[offset]['command_id'] if offset<len(commands) else None
            remaining=row.get('result',{}).get('steps_remaining')
            used=row.get('result',{}).get('steps_used',0)
            # S2's frozen contract has 1600 shared control steps. A partial
            # command's remaining count is written against that full horizon.
            end=1600-remaining if type(remaining) is int else None
            before=end-used if type(end) is int and type(used) is int else None
            history_sources[start+offset]=dict(observation_id=source,source_turn=index,
                source_frame_id=plan.get('frame_id',source),control_step=end,
                command_entry_frame_id=(f'{source}/command/{cid}/step/{before}'
                    if cid is not None and before is not None else None))
        target=plan.get('target_ref')
        if target is not None and last_target is not None and target!=last_target:
            target_changes[start]=dict(source_turn=index,from_target_ref=last_target,to_target_ref=target)
        if target is not None:last_target=target
        attempted=[x for x in entries if x.get('call') and
            (x['result'].get('action_dispatched') is True or x['result'].get('steps_used',0)>0)]
        if attempted:
            strategies.append(dict(source_turn=index,plan_id=plan['plan_id'],revision=plan['revision'],
                category=plan['strategy']['category'],parameters=plan['strategy']['parameters'],
                changed_conditions=plan['strategy']['changed_conditions'],
                executed_tools=[x['call']['tool'] for x in attempted],
                results=[x['result'].get('reason','unknown') for x in attempted]))
        beliefs=dict(source='model_self_report',source_turn=index,
            evidence_refs=_copy(plan['evidence_refs']),**_copy(plan['model_beliefs']))
        pending=_copy(plan['model_beliefs']['pending_checks'])
    for index,row in enumerate(history):
        if index in target_changes:
            for name in targets:targets[name]=dict(status='invalidated',source='model_target_switch',evidence_refs=[])
            invalidations.append(dict(command_index=index,reason='candidate_switch',fields=list(targets),
                                      **target_changes[index]))
        call=row.get('call') or {};result=row.get('result') or {};tool=call.get('tool')
        provenance=history_sources.get(index,{})
        source_oid=call.get('observation_id') or provenance.get('observation_id')
        evidence=dict(source='tool_feedback',command_index=index,observation_id=source_oid,
                      evidence_refs=[source_oid] if source_oid else [])
        if tool=='locate_point' and result.get('valid') is True:
            targets['localization']=dict(status='measured',**evidence)
        comparison=result.get('identity_comparison') or {}
        # Only the existing independent verifier's verdict is a confirmation.
        continuity=result.get('identity_continuity') or {}
        verdict=continuity.get('effective_status') or comparison.get('identity_status') or comparison.get('verdict') or comparison.get('status')
        if tool=='verify_target' and verdict in ('confirmed','rejected'):
            targets['identity']=dict(status=verdict,**evidence)
        elif tool=='verify_target' and verdict=='uncertain' and targets['identity']['status']!='confirmed':
            targets['identity']=dict(status='uncertain',**evidence)
        if tool in ('prepare_grasp','replan_grasp'):
            targets['alignment']=dict(status='arrived' if result.get('arrived') else 'not_arrived',**evidence)
        if tool=='set_gripper' and call.get('gripper')==0:
            targets['closure']=dict(status='completed' if result.get('closure_command_completed') else 'incomplete',**evidence)
            targets['possession']=dict(status='unknown',**evidence)
        if tool=='return_to_rest':
            targets['rest']=dict(status='arrived' if result.get('arrived') else 'incomplete',**evidence)
        moved=tool=='move_base' and result.get('steps_used',0)>0
        drift=result.get('localization_invalidated') is True or result.get('reason')=='base_pushed'
        loss=result.get('reason')=='grip_lost'
        completion=result.get('motion_completion') or {}
        dispatched=(result.get('steps_used',0)>0 or result.get('action_dispatched') is True)
        open_arrived=(result.get('arrived') is True or completion.get('status')=='arrived')
        opened=(tool=='set_gripper' and call.get('gripper')==1
                and (dispatched or (result.get('accepted') is True and open_arrived)))
        if moved or drift:
            for name in ('localization','alignment'):
                targets[name]=dict(status='invalidated',**evidence)
            invalidations.append(dict(command_index=index,reason='base_motion' if moved else 'base_drift',
                fields=['localization','alignment','path_evidence']))
        if opened or loss:
            targets['possession']=dict(status='invalidated',**evidence)
            targets['alignment']=dict(status='invalidated',**evidence)
            targets['localization']=dict(status='invalidated',**evidence)
            targets['closure']=dict(status=('open' if open_arrived else 'open_command_incomplete')
                if opened else 'unknown',**evidence)
            invalidations.append(dict(command_index=index,reason='grip_lost' if loss else 'open_command',
                                      fields=['possession','alignment','localization','path_evidence','closure']))
        if tool in arm_tools and (result.get('accepted') is False or result.get('requires_replan') is True
                or result.get('reason') in ('blocked','needs_base_reposition','grip_lost','motion_stalled','base_pushed','ik_unresolved')):
            blocked_arm=True
        if blocked_arm and moved:repositioned=True
        if result.get('motion_completion'):
            actual=completion.get('actual') or {}
            # Cartesian completion poses are expressed in the measured base at
            # command end. Base displacements use the command-entry frame;
            # joint/finger scalars have no Cartesian frame. Never relabel any
            # historical reading as the latest observation's base coordinates.
            frame=(f"{provenance['command_entry_frame_id']}/completion-base/{provenance['control_step']}"
                if 'poses' in actual and provenance.get('command_entry_frame_id') is not None
                else provenance.get('command_entry_frame_id') if 'base_delta' in actual else None)
            measured['last_motion']=dict(value=_copy(result['motion_completion']),source='tool_feedback',
                command_index=index,state_version=index+1,frame_id=frame,validity='historical',
                **_copy(provenance))
        if 'finger_distance_m' in result:
            measured['finger_distance_m']=dict(value=result['finger_distance_m'],source='tool_feedback',
                command_index=index,state_version=index+1,frame_id=None,validity='historical',
                **_copy(provenance))
    if len(history) in target_changes:
        for name in targets:targets[name]=dict(status='invalidated',source='model_target_switch',evidence_refs=[])
        invalidations.append(dict(command_index=len(history),reason='candidate_switch',fields=list(targets),
                                  **target_changes[len(history)]))
    for key in ('identity','possession'):
        if beliefs:
            targets[key]['model_belief']=beliefs.get(key,'unknown')
            targets[key]['model_belief_source_turn']=beliefs['source_turn']
    budgets=dict(steps_remaining=observation['steps_remaining'],generation_requests_remaining=observation['calls_remaining'])
    if budget is not None:budgets['usd']=_copy(budget)
    card=dict(schema_version='sol-pick-state-card-v1',state_version=version,
        parent_version=previous.get('state_version') if isinstance(previous,dict) else None,
        observation_id=oid,frame_id=oid,measured=measured,target_state=targets,
        model_beliefs=beliefs,pending_checks=pending,strategy_history=strategies,
        invalidations=invalidations,unresolved_refs=pending,budget=budgets,
        recovery_metrics=dict(arm_failure_observed=blocked_arm,
                              base_reposition_after_arm_failure=repositioned),
        delivery='full_current_card' if not isinstance(previous,dict) or previous.get('state_version',-1)>version else 'full_current_card_with_parent')
    return _copy(card)


def _json(value):
    """Reject non-JSON values rather than silently coercing keys/arrays/NaN."""
    def check(item, depth=0):
        if depth > 16:
            raise ValueError("memory JSON nesting")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():
                check(child, depth + 1)
            return
        raise ValueError("memory requires finite strict JSON")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _geometry_copy(value, depth=0):
    """Validate and isolate JSON geometry without serializing it twice.

    Match _copy's strict types, depth limit and sorted dictionary order.
    Keep validation on reads, including restored or internally damaged state.
    """
    if depth > 16:
        raise ValueError("memory JSON nesting")
    kind = type(value)
    if kind is int and value.bit_length() > 2048:
        # Preserve the runtime JSON integer digit limit on damaged records.
        return _copy(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is list:
        return [_geometry_copy(child, depth + 1) for child in value]
    if kind is dict and all(type(key) is str for key in value):
        return {key: _geometry_copy(value[key], depth + 1) for key in sorted(value)}
    raise ValueError("memory requires finite strict JSON")


def _fields(value, required, optional=()):
    if type(value) is not dict or set(value) - set(required) - set(optional) or not set(required) <= set(value):
        raise ValueError("memory field schema")


def _text(value, maximum=160):
    if type(value) is not str or not value.strip() or len(value.encode("utf8")) > maximum:
        raise ValueError("memory text length/type")
    return value


def _prose(value):
    # The public rationale contract limits characters, including Chinese text.
    if type(value) is not str or not value.strip() or len(value) > 160:
        raise ValueError("memory prose length/type")
    return value


def _integer(value, minimum=0, maximum=10**9):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("memory integer range/type")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("memory finite number required")
    return value


def _vector(value, size=3):
    if type(value) is not list or len(value) != size:
        raise ValueError("memory vector shape")
    for x in value:
        _number(x)
    return value


def _sha(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("memory source SHA256")
    return value


def _identifier(value):
    if type(value) is int:
        return _integer(value, minimum=1)
    return _text(value, 160)


def _odom(value):
    _fields(value, ("dx_m", "dy_m", "dyaw_rad"))
    for x in value.values():
        _number(x)
    return value


def _camera(rotation, intrinsics):
    for matrix in (rotation, intrinsics):
        if type(matrix) is not list or len(matrix) != 3:
            raise ValueError("memory geometry matrix shape")
        for row in matrix:
            _vector(row)
    for i in range(3):
        for j in range(3):
            if abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - (1 if i == j else 0)) > 1e-5:
                raise ValueError("memory optical rotation must be orthonormal")
    determinant = sum(rotation[0][i] * (rotation[1][(i+1)%3] * rotation[2][(i+2)%3] -
        rotation[1][(i+2)%3] * rotation[2][(i+1)%3]) for i in range(3))
    if abs(determinant - 1) > 1e-5:
        raise ValueError("memory optical rotation handedness")
    if intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0 or intrinsics[2] != [0, 0, 1]:
        raise ValueError("memory camera intrinsics")


def _voxel_points(record):
    if record["kind"] == "voxel_cloud":
        return len(record["centers_odom_m"])
    return int(record["kind"] == "point")


def validate_summary(value, *, max_bytes=2048):
    """Strict public wire contract; bounded observed surfaces, no hidden fields."""
    value = _copy(value)
    _integer(max_bytes, 512, 2048)
    if len(_json(value).encode("utf8")) > max_bytes:
        raise ValueError("memory summary byte limit")
    _fields(value, ("version", "step", "frame", "meanings", "geometry_counts", "voxel_points", "evicted", "targets", "nearby_surfaces", "regions", "failures", "route", "omitted"))
    if value["version"] != VERSION or value["meanings"] != _SUMMARY_MEANINGS:
        raise ValueError("memory summary semantics")
    _integer(value["step"]); _text(value["frame"])
    _fields(value["geometry_counts"], ("occupied", "free", "unknown"))
    for number in value["geometry_counts"].values():
        _integer(number, maximum=_HARD_LIMITS["max_geometry"])
    if sum(value["geometry_counts"].values()) > _HARD_LIMITS["max_geometry"]:
        raise ValueError("memory summary geometry count")
    _integer(value["voxel_points"], maximum=_HARD_LIMITS["max_voxel_points"])
    _fields(value["evicted"], ("observations", "geometry", "regions", "actions", "route"))
    _fields(value["omitted"], ("targets", "nearby_surfaces", "regions", "failures", "route"))
    for group in ("evicted", "omitted"):
        for number in value[group].values():
            _integer(number)
    for group in ("targets", "regions", "failures", "route"):
        if type(value[group]) is not list:
            raise ValueError("memory summary list")
        maximum = _HARD_LIMITS["max_" + ("actions" if group == "failures" else group)]
        if len(value[group]) + value["omitted"][group] > maximum:
            raise ValueError("memory summary collection count")
    for row in value["targets"]:
        _fields(row, ("id", "status", "visible", "observation_id", "age_steps"),
            ("label", "last_visible_position_odom_m", "position_age_steps", "confirmation_observation_id", "confirmation_age_steps", "rejection_reason"))
        _identifier(row["id"]); _text(row["observation_id"]); _integer(row["age_steps"], maximum=value["step"])
        if row["status"] not in ("confirmed", "rejected", "unknown") or type(row["visible"]) is not bool:
            raise ValueError("memory summary target identity")
        if "label" in row:
            _prose(row["label"])
        if ("confirmation_observation_id" in row) != ("confirmation_age_steps" in row):
            raise ValueError("memory summary confirmation provenance")
        if row["status"] == "confirmed" and "confirmation_observation_id" not in row:
            raise ValueError("memory summary confirmed identity needs source")
        if "confirmation_observation_id" in row:
            _text(row["confirmation_observation_id"])
            _integer(row["confirmation_age_steps"], maximum=value["step"])
            if row["status"] != "confirmed":
                raise ValueError("memory summary invalidated confirmation")
        if "rejection_reason" in row:
            _prose(row["rejection_reason"])
            if row["status"] != "rejected":
                raise ValueError("memory summary rejection status")
        if ("last_visible_position_odom_m" in row) != ("position_age_steps" in row):
            raise ValueError("memory summary position age")
        if "last_visible_position_odom_m" in row:
            _vector(row["last_visible_position_odom_m"]); _integer(row["position_age_steps"], maximum=value["step"])
    for row in value["regions"]:
        _fields(row, ("id", "result", "observation_id", "age_steps"), ("origin_odom_m", "view_direction_odom", "bounds_odom_m"))
        _text(row["id"]); _text(row["observation_id"]); _integer(row["age_steps"], maximum=value["step"])
        if row["result"] not in ("not_assessed", "candidate_seen", "no_match_in_visible_part"):
            raise ValueError("memory summary observed region")
        if ("origin_odom_m" in row) != ("view_direction_odom" in row):
            raise ValueError("memory summary camera view")
        if "origin_odom_m" in row:
            _vector(row["origin_odom_m"]); _vector(row["view_direction_odom"])
            if abs(sum(v*v for v in row["view_direction_odom"])-1) > 1e-5 or "bounds_odom_m" in row:
                raise ValueError("memory summary view direction")
        if "bounds_odom_m" in row:
            bounds = row["bounds_odom_m"]
            if type(bounds) is not list or len(bounds) != 2:
                raise ValueError("memory summary region bounds")
            _vector(bounds[0]); _vector(bounds[1])
            if any(a > b for a, b in zip(*bounds)):
                raise ValueError("memory summary region bounds order")
        if "origin_odom_m" not in row and "bounds_odom_m" not in row:
            raise ValueError("memory summary region needs measured location")
    for row in value["failures"]:
        _fields(row, ("tool", "outcome", "reason", "step"))
        _text(row["tool"]); _prose(row["reason"]); _integer(row["step"], maximum=value["step"])
        if row["outcome"] not in ("rejected", "stalled", "deviated", "failed"):
            raise ValueError("memory summary failure")
    for row in value["route"]:
        _fields(row, ("step", "odometry"))
        _integer(row["step"], maximum=value["step"]); _odom(row["odometry"])
    if type(value["nearby_surfaces"]) is not list or len(value["nearby_surfaces"]) > _MAX_SUMMARY_SURFACES:
        raise ValueError("memory summary surface count")
    if len(value["nearby_surfaces"]) + value["omitted"]["nearby_surfaces"] > _HARD_LIMITS["max_voxel_points"]:
        raise ValueError("memory summary surface collection count")
    cells = set()
    for row in value["nearby_surfaces"]:
        _fields(row, ("point_odom_m", "radius_m", "observation_id", "camera", "age_steps"))
        _vector(row["point_odom_m"]); _text(row["observation_id"])
        _integer(row["age_steps"], maximum=min(value["step"], _HARD_LIMITS["occupied_ttl_steps"]))
        if row["camera"] not in ("head", "hand") or not 0 <= _number(row["radius_m"]) <= 11:
            raise ValueError("memory summary observed surface")
        cell = tuple(row["point_odom_m"])
        if cell in cells:
            raise ValueError("memory summary duplicate surface cell")
        cells.add(cell)
    return value


class EpisodeMemory:
    """Single owner of portable, source-bound short-term episode memory.

    Register a robot observation before adding records which cite it.  Producers
    may read copies of geometry but must not keep a second persistent map.
    Snapshot restoration requires exact episode, odometry-frame and calibration
    bindings; source-version migration is a separate runner authorization gate.
    """

    def __init__(self, episode_id, odom_frame_id, source_binding, *, limits=None):
        self.episode_id = _text(episode_id)
        self.odom_frame_id = _text(odom_frame_id)
        _fields(source_binding, ("sensor_calibration_sha256", "robot_geometry_sha256", "memory_version"),
                ("geometry_classifier",))
        if ('geometry_classifier' in source_binding
                and source_binding['geometry_classifier'] != 'local-support-floor-v1'):
            raise ValueError('geometry classifier binding')
        _sha(source_binding["sensor_calibration_sha256"])
        _sha(source_binding["robot_geometry_sha256"])
        if source_binding["memory_version"] != VERSION:
            raise ValueError("memory version binding")
        self._source_binding = _copy(source_binding)
        if limits is not None:
            _fields(limits, (), DEFAULT_LIMITS)
        self._limits = dict(DEFAULT_LIMITS, **(limits or {}))
        for key, value in self.limits.items():
            _integer(value, 1, _HARD_LIMITS[key])
        if self.limits["max_observations"] < self.limits["max_targets"] * 4 + 2:
            raise ValueError("memory observations must retain target provenance")
        self._step = 0
        self._observations = {}
        self._targets = {}
        self._regions = {}
        self._geometry = {}
        self._actions = {}
        self._route = []
        self._evicted = {k: 0 for k in ("observations", "geometry", "regions", "actions", "route")}

    @property
    def source_binding(self):
        return dict(self._source_binding)

    @property
    def limits(self):
        return dict(self._limits)

    def _time(self, step):
        _integer(step)
        if step < self._step:
            raise ValueError("memory time cannot move backwards")

    def _evidence(self, value, step, *, depth=False):
        _fields(value, ("observation_id", "camera", "control_step", "calibration_sha256"))
        _text(value["observation_id"])
        _integer(value["control_step"])
        _sha(value["calibration_sha256"])
        if value["camera"] not in ("head", "hand"):
            raise ValueError("memory robot sensor camera only")
        frame = self._observations.get(value["observation_id"])
        if (frame is None or value["control_step"] != frame["control_step"] or
                value["calibration_sha256"] != frame["calibration_sha256"] or
                value["control_step"] > step):
            raise ValueError("memory observation provenance mismatch")
        keys = frame["sensor_sha256"]
        if depth and value["camera"] + "_depth" not in keys:
            raise ValueError("memory geometry needs raw depth source")
        if not any(value["camera"] + suffix in keys for suffix in ("_depth", "_rgb")):
            raise ValueError("memory evidence sensor missing")
        return value

    def _trim(self, records, limit, category):
        while len(records) > limit:
            records.pop(next(iter(records)))
            self._evicted[category] += 1

    def _trim_observations(self):
        pinned = set()
        attachment=getattr(self,'_attachment',None)
        if attachment is not None:
            pinned.add(attachment['evidence']['observation_id'])
            pinned.add(attachment['confirmation_evidence']['observation_id'])
        for target in self._targets.values():
            for key in ("evidence", "confirmation_evidence", "last_visible_evidence"):
                if key in target:
                    pinned.add(target[key]["observation_id"])
            if "contradiction" in target:
                pinned.add(target["contradiction"]["evidence"]["observation_id"])
        while len(self._observations) > self.limits["max_observations"]:
            oldest = next(key for key in self._observations if key not in pinned)
            self._observations.pop(oldest)
            self._evicted["observations"] += 1
            for category in ("geometry", "regions", "actions"):
                records = getattr(self, "_" + category)
                for key in list(records):
                    if records[key]["evidence"]["observation_id"] == oldest:
                        records.pop(key)
                        self._evicted[category] += 1
            keep = [row for row in self._route if row["evidence"]["observation_id"] != oldest]
            self._evicted["route"] += len(self._route) - len(keep)
            self._route = keep

    def record_observation(self, observation_id, *, step, odometry,
                           calibration_sha256, sensor_sha256):
        self._time(step)
        _text(observation_id)
        _odom(odometry)
        _sha(calibration_sha256)
        _fields(sensor_sha256, (), _SENSOR_NAMES)
        if not sensor_sha256:
            raise ValueError("memory observation needs robot sensor hashes")
        for value in sensor_sha256.values():
            _sha(value)
        record = _copy(dict(observation_id=observation_id, control_step=step,
            odometry=odometry, calibration_sha256=calibration_sha256, sensor_sha256=sensor_sha256))
        old = self._observations.get(observation_id)
        if old is not None and old != record:
            raise ValueError("memory observation ID reused with different source")
        self._observations[observation_id] = record
        self._step = step
        self._trim_observations()

    def observation_evidence(self, observation_id, camera):
        frame = self._observations.get(observation_id)
        if frame is None:
            raise ValueError("memory observation not retained")
        evidence = dict(observation_id=observation_id, camera=camera,
            control_step=frame["control_step"], calibration_sha256=frame["calibration_sha256"])
        self._evidence(evidence, self._step)
        return _copy(evidence)

    def observation_binding(self, observation_id):
        frame=self._observations.get(observation_id)
        if frame is None:raise ValueError('memory observation not retained')
        return _copy(frame)

    def begin_observation(self, *, step):
        """Clear current visibility, without changing identity or evidence age.

        Call once when beginning a new observation.  Subsequent sensor-supported
        target records mark what is visible in that observation.  This does not
        interpret absence from a view as absence from the scene.
        """
        self._time(step)
        for row in self._targets.values():
            row["visible"] = False
        self._step = step

    def record_target(self, record, *, step):
        self._time(step)
        _fields(record, ("id", "status", "visible", "evidence"),
                ("label", "last_visible_position_odom_m", "position_evidence", "contradiction"))
        _identifier(record["id"])
        if record["status"] not in ("confirmed", "rejected", "unknown") or type(record["visible"]) is not bool:
            raise ValueError("memory target identity status")
        self._evidence(record["evidence"], step)
        if "label" in record:
            _prose(record["label"])
        if "last_visible_position_odom_m" in record:
            _vector(record["last_visible_position_odom_m"])
            if not record["visible"]:
                raise ValueError("memory unseen target cannot update visible position")
            position_evidence = record.get("position_evidence", record["evidence"])
            self._evidence(position_evidence, step, depth=True)
            if any(position_evidence[k] != record["evidence"][k]
                   for k in ("observation_id", "control_step", "calibration_sha256")):
                raise ValueError("memory position source must belong to the same observation")
        elif "position_evidence" in record:
            raise ValueError("memory position evidence requires a visible position")
        if "contradiction" in record:
            contradiction = record["contradiction"]
            _fields(contradiction, ("kind", "reason", "evidence"))
            if contradiction["kind"] != "identity_mismatch" or contradiction["evidence"] != record["evidence"]:
                raise ValueError("memory identity contradiction needs current evidence")
            _prose(contradiction["reason"])
            self._evidence(contradiction["evidence"], step)
            if record["status"] != "rejected" or not record["visible"]:
                raise ValueError("memory identity contradiction must reject")
        key = _json(record["id"])
        old = self._targets.get(key, {})
        if record["status"] == "confirmed" and not record["visible"]:
            raise ValueError("memory fresh confirmation requires visible identity evidence")
        if not old and len(self._targets) >= self.limits["max_targets"]:
            raise ValueError("memory target capacity; do not discard identity evidence")
        new = dict(old, **_copy(record), control_step=step)
        new.pop("position_evidence", None)
        if old.get("status") == "confirmed" and record["status"] != "confirmed" and "contradiction" not in record:
            new["status"] = "confirmed"
        if old.get("status") == "rejected" and record["status"] == "unknown":
            new["status"] = "rejected"
        if record["status"] == "confirmed":
            new["confirmation_evidence"] = _copy(record["evidence"])
            new.pop("contradiction", None)
        if "last_visible_position_odom_m" in record:
            new["last_visible_evidence"] = _copy(position_evidence)
            new["position_role"] = "last_visible_surface_not_grasp_pose"
        self._targets[key] = new
        self._step = step

    def observe_region(self, record, *, step):
        self._time(step)
        _fields(record, ("id", "bounds_odom_m", "coverage", "search_result", "evidence"), ("description",))
        _text(record["id"])
        if type(record["bounds_odom_m"]) is not list or len(record["bounds_odom_m"]) != 2:
            raise ValueError("memory region bounds")
        low, high = record["bounds_odom_m"]
        _vector(low); _vector(high)
        if any(a > b for a, b in zip(low, high)):
            raise ValueError("memory region bounds order")
        if record["coverage"] not in ("partial", "observed") or record["search_result"] not in (
                "not_assessed", "candidate_seen", "no_match_in_visible_part"):
            raise ValueError("memory looked-at region is not excluded or free")
        if "description" in record:
            _prose(record["description"])
        self._evidence(record["evidence"], step)
        self._regions.pop(record["id"], None)
        self._regions[record["id"]] = dict(_copy(record), control_step=step)
        self._trim(self._regions, self.limits["max_regions"], "regions")
        self._step = step

    def observe_view(self, record, *, step):
        """Remember only the measured view, without inventing a cleared region."""
        self._time(step)
        _fields(record, ("id", "evidence", "search_result", "origin_odom_m", "rotation_odom_from_optical", "intrinsics"))
        _text(record["id"]); _vector(record["origin_odom_m"])
        _camera(record["rotation_odom_from_optical"], record["intrinsics"])
        self._evidence(record["evidence"], step)
        if record["search_result"] not in ("not_assessed", "candidate_seen"):
            raise ValueError("memory camera view cannot exclude an unseen target")
        self._regions.pop(record["id"], None)
        self._regions[record["id"]] = dict(_copy(record), control_step=step, kind="view", coverage="observed_only")
        self._trim(self._regions, self.limits["max_regions"], "regions")
        self._step = step

    def _geometry_record(self, record, step):
        common = ("id", "kind", "status", "evidence")
        fields = {"point": ("point_odom_m", "radius_m"),
                  "voxel_cloud": ("centers_odom_m", "radii_m"),
                  "ray": ("origin_odom_m", "end_odom_m", "radius_m"),
                  "frustum": ("origin_odom_m", "rotation_odom_from_optical", "intrinsics", "pixel_bounds", "near_m", "far_m")}
        if type(record) is not dict or record.get("kind") not in fields:
            raise ValueError("memory geometry kind")
        _fields(record, common + fields[record["kind"]],
            ("support_plane_odom_z_m",) if record["kind"] == "frustum" else ())
        _text(record["id"])
        allowed = ("occupied", "unknown") if record["kind"] in ("point", "voxel_cloud") else ("free", "unknown")
        if record["status"] not in allowed:
            raise ValueError("memory geometry status/kind")
        self._evidence(record["evidence"], step, depth=True)
        for name in ("point_odom_m", "origin_odom_m", "end_odom_m"):
            if name in record:
                _vector(record[name])
        if "radius_m" in record and not 0 <= _number(record["radius_m"]) <= 10:
            raise ValueError("memory geometry radius")
        if record["kind"] == "voxel_cloud":
            centers, radii = record["centers_odom_m"], record["radii_m"]
            if (type(centers) is not list or not 1 <= len(centers) <= _MAX_CLOUD_POINTS or
                    type(radii) is not list or len(centers) != len(radii)):
                raise ValueError("memory voxel cloud shape/capacity")
            for center, radius in zip(centers, radii):
                _vector(center)
                if not 0 <= _number(radius) <= 10:
                    raise ValueError("memory voxel cloud radius")
        if record["kind"] == "frustum":
            rotation, intrinsics = record["rotation_odom_from_optical"], record["intrinsics"]
            _camera(rotation, intrinsics)
            pixels = _vector(record["pixel_bounds"], 4)
            if not (0 <= pixels[0] < pixels[2] <= 640 and 0 <= pixels[1] < pixels[3] <= 640):
                raise ValueError("memory camera pixel bounds")
            if not 0 <= _number(record["near_m"]) < _number(record["far_m"]) <= 100:
                raise ValueError("memory frustum measured range")
            if "support_plane_odom_z_m" in record:
                # An additional lower bound may only narrow the measured free
                # frustum. It cannot erase an occupied record or clear a map.
                _number(record["support_plane_odom_z_m"])
        return _copy(record)

    def record_geometry(self, records, *, step):
        self._time(step)
        if type(records) is not list or len(records) > _HARD_LIMITS["max_geometry"]:
            raise ValueError("memory geometry batch limit")
        checked = {}
        for record in records:
            row = self._geometry_record(record, step)
            old = checked.get(row["id"], self._geometry.get(row["id"]))
            if old is not None and old != row:
                raise ValueError("memory geometry ID reused with different evidence")
            checked[row["id"]] = row
        merged = dict(self._geometry, **checked)
        observation_sizes, observation_points = {}, {}
        for row in merged.values():
            key = row["evidence"]["observation_id"]
            observation_sizes[key] = observation_sizes.get(key, 0) + 1
            observation_points[key] = observation_points.get(key, 0) + _voxel_points(row)
        if max(observation_sizes.values(), default=0) > self.limits["max_geometry"]:
            raise ValueError("memory geometry one observation exceeds capacity")
        if max(observation_points.values(), default=0) > min(self.limits["max_voxel_points"], _MAX_OBSERVATION_POINTS):
            raise ValueError("memory voxel points one observation exceeds capacity")
        self._geometry = merged
        point_count = sum(observation_points.values())
        while len(self._geometry) > self.limits["max_geometry"] or point_count > self.limits["max_voxel_points"]:
            oldest = next(iter(self._geometry.values()))["evidence"]["observation_id"]
            point_count -= observation_points[oldest]
            for key in list(self._geometry):
                if self._geometry[key]["evidence"]["observation_id"] == oldest:
                    self._geometry.pop(key)
                    self._evicted["geometry"] += 1
        if getattr(self,'_attachment',None) is not None and 'surface_memberships' in self._attachment:
            self._attachment['surface_memberships']=[r for r in self._attachment['surface_memberships']
                if r['geometry_id'] in self._geometry]
        if getattr(self,'_attachment',None) is not None and 'support_plane' in self._attachment:
            support=self._attachment['support_plane']
            support['memberships']=[r for r in support['memberships'] if r['geometry_id'] in self._geometry]
        self._step = step

    def record_action(self, record, *, step):
        self._time(step)
        _fields(record, ("id", "tool", "outcome", "reason", "entry_odometry", "exit_odometry", "steps_used", "evidence"), ("target_id",))
        for key in ("id", "tool"):
            _text(record[key])
        _prose(record["reason"])
        if record["outcome"] not in ("completed", "rejected", "stalled", "deviated", "failed"):
            raise ValueError("memory action outcome")
        _integer(record["steps_used"], maximum=step)
        _odom(record["entry_odometry"]); _odom(record["exit_odometry"])
        self._evidence(record["evidence"], step)
        if "target_id" in record:
            _identifier(record["target_id"])
        row = dict(_copy(record), control_step=step)
        if record["id"] in self._actions:
            if self._actions[record["id"]] != row:
                raise ValueError("memory action ID reused")
            return
        if record["steps_used"] and self._route and step-record["steps_used"] < self._route[-1]["end_step"]:
            raise ValueError("memory executed route overlaps previous action")
        if any(r["action_id"] == record["id"] for r in self._route):
            raise ValueError("memory action ID already in retained route")
        self._actions[record["id"]] = row
        self._trim(self._actions, self.limits["max_actions"], "actions")
        if record["steps_used"]:
            self._route.append(dict(action_id=record["id"], start_step=step-record["steps_used"],
                end_step=step, entry_odometry=_copy(record["entry_odometry"]),
                exit_odometry=_copy(record["exit_odometry"]), evidence=_copy(record["evidence"]),
                meaning="executed_trace_not_free_space"))
            while len(self._route) > self.limits["max_route"]:
                self._route.pop(0)
                self._evicted["route"] += 1
        self._step = step

    def geometry(self, *, step):
        self._time(step)
        result = []
        # Limits are immutable during this query. Copy/validate the public
        # configuration once, not once for every stored sensor record.
        limits = self.limits
        for record in self._geometry.values():
            row = _geometry_copy(record)
            age = step - row["evidence"]["control_step"]
            stale = age > limits[row["status"] + "_ttl_steps"]
            row.update(observed_status=row["status"], age_steps=age, stale=stale)
            if stale:
                row["status"] = "unknown"
            result.append(row)
        return result

    def _nearby_surfaces(self, step):
        """Transient sensor summary, never an occupancy map or collision query.

        Use fixed 25 cm cells and sphere envelopes of the observed samples.
        Each envelope cites one camera/frame; newer observations of the same
        cell replace older ones only in this lossy summary, never in memory.
        Rank within 3 m of the latest registered odometry origin. This is not
        distance to the robot's hull and cannot certify a passage is clear.
        """
        if not self._observations:
            return [], 0
        observation_order = {name: index for index, name in enumerate(self._observations)}
        latest = next(reversed(self._observations.values()))["odometry"]
        base_x, base_y = latest["dx_m"], latest["dy_m"]
        cells = {}
        ttl = self._limits["occupied_ttl_steps"]
        half_diagonal = math.sqrt(3) * _SURFACE_CELL_M / 2
        for row in self._geometry.values():
            evidence = row["evidence"]
            if row["status"] != "occupied" or step-evidence["control_step"] > ttl:
                continue
            if row["kind"] == "voxel_cloud":
                samples = zip(row["centers_odom_m"], row["radii_m"])
            elif row["kind"] == "point":
                samples = ((row["point_odom_m"], row["radius_m"]),)
            else:
                continue
            priority = (evidence["control_step"], observation_order[evidence["observation_id"]], evidence["camera"])
            for point, radius in samples:
                x, y, z = point
                # Cheap conservative prefilter includes cell quantization.
                reach = _SURFACE_NEAR_M + radius + half_diagonal
                if (x-base_x)**2 + (y-base_y)**2 + z*z > reach*reach:
                    continue
                cell = (math.floor(x/_SURFACE_CELL_M), math.floor(y/_SURFACE_CELL_M), math.floor(z/_SURFACE_CELL_M))
                prior = cells.get(cell)
                if prior is not None and prior[0] > priority:
                    continue
                center = [(axis+.5)*_SURFACE_CELL_M for axis in cell]
                envelope = math.sqrt(sum((a-b)**2 for a, b in zip(point, center))) + radius
                if prior is None or prior[0] != priority:
                    cells[cell] = [priority, evidence, center, envelope]
                elif envelope > prior[3]:
                    prior[3] = envelope
        candidates = []
        for priority, evidence, center, radius in cells.values():
            distance = max(0., math.sqrt((center[0]-base_x)**2 + (center[1]-base_y)**2 + center[2]**2)-radius)
            if distance <= _SURFACE_NEAR_M:
                candidates.append((distance, -priority[0], tuple(center), evidence, radius))
        chosen = heapq.nsmallest(_MAX_SUMMARY_SURFACES, candidates, key=lambda item: item[:3])
        return [dict(point_odom_m=list(center), radius_m=math.ceil(radius*1000)/1000,
            observation_id=evidence["observation_id"], camera=evidence["camera"],
            age_steps=step-evidence["control_step"]) for _, _, center, evidence, radius in chosen], len(candidates)

    def summary(self, *, step, max_bytes=2048):
        """Compact model-facing JSON; dense geometry stays outside the context."""
        self._time(step)
        _integer(max_bytes, 512, 2048)
        counts = {status: 0 for status in ("occupied", "free", "unknown")}
        # No copies of dense clouds enter the model-facing path. At most four
        # transient observed-surface envelopes accompany the record counts.
        for row in self._geometry.values():
            stale = step-row["evidence"]["control_step"] > self.limits[row["status"]+"_ttl_steps"]
            counts["unknown" if stale else row["status"]] += 1
        surfaces, surface_count = self._nearby_surfaces(step)
        result = dict(version=VERSION, step=step, frame=self.odom_frame_id,
            meanings=_SUMMARY_MEANINGS,
            geometry_counts=counts, voxel_points=sum(_voxel_points(r) for r in self._geometry.values()),
            evicted=_copy(self._evicted), targets=[], nearby_surfaces=[], regions=[], failures=[], route=[])
        totals = dict(targets=len(self._targets), regions=len(self._regions),
            nearby_surfaces=surface_count,
            failures=sum(r["outcome"] != "completed" for r in self._actions.values()), route=len(self._route))
        result["omitted"] = dict(totals)
        if len(_json(result).encode("utf8")) > max_bytes:
            raise ValueError("memory summary metadata exceeds byte limit")
        def add(key, row):
            result[key].append(row)
            result["omitted"][key] -= 1
            if len(_json(result).encode("utf8")) > max_bytes:
                result[key].pop()
                result["omitted"][key] += 1
        groups = {"targets": [], "nearby_surfaces": surfaces, "regions": [], "failures": [], "route": []}
        for row in reversed(list(self._targets.values())):
            item = dict(id=row["id"], status=row["status"], visible=row["visible"],
                observation_id=row["evidence"]["observation_id"], age_steps=step-row["evidence"]["control_step"])
            if "label" in row:
                item["label"] = row["label"]
            if row["status"] == "confirmed":
                item.update(confirmation_observation_id=row["confirmation_evidence"]["observation_id"],
                    confirmation_age_steps=step-row["confirmation_evidence"]["control_step"])
            if row["status"] == "rejected" and "contradiction" in row:
                item["rejection_reason"] = row["contradiction"]["reason"]
            if "last_visible_position_odom_m" in row:
                item["last_visible_position_odom_m"] = row["last_visible_position_odom_m"]
                item["position_age_steps"] = step-row["last_visible_evidence"]["control_step"]
            groups["targets"].append(item)
        for row in reversed(list(self._regions.values())):
            item = dict(id=row["id"], result=row["search_result"],
                observation_id=row["evidence"]["observation_id"], age_steps=step-row["evidence"]["control_step"])
            if row.get("kind") == "view":
                item.update(origin_odom_m=row["origin_odom_m"],
                    view_direction_odom=[r[2] for r in row["rotation_odom_from_optical"]])
            else:
                item["bounds_odom_m"] = row["bounds_odom_m"]
            groups["regions"].append(item)
        for row in reversed(list(self._actions.values())):
            if row["outcome"] != "completed":
                groups["failures"].append(dict(tool=row["tool"], outcome=row["outcome"],
                    reason=row["reason"], step=row["control_step"]))
        for row in reversed(self._route):
            groups["route"].append(dict(step=row["end_step"], odometry=row["exit_odometry"]))
        # Round-robin keeps one domain from consuming all available context.
        for index in range(max((len(rows) for rows in groups.values()), default=0)):
            for key, rows in groups.items():
                if index < len(rows):
                    add(key, rows[index])
        return validate_summary(result, max_bytes=max_bytes)

    def record_attachment(self, record, *, step):
        """Store a source-bound estimated envelope, never a grasp-success claim.

        Geometry is fixed in gripper_link coordinates. A separate execution
        event may mark possible attachment; neither event asserts actual grip.
        No obstacle record is removed by this operation.
        """
        self._time(step)
        value=self._attachment_value(record,step,require_current_confirmation=True)
        if getattr(self,'_attachment',{}).get('status')=='possibly_attached':
            raise ValueError('Open before replacing possible attachment')
        previous=getattr(self,'_attachment',{})
        if previous.get('status')=='estimated' and previous.get('candidate_id')==value['candidate_id']:
            # A fresh estimate of the SAME candidate retains exact old source
            # labels; changing its bounds must not relabel any other voxels.
            value['surface_memberships']=_copy(previous.get('surface_memberships',[]))
            if 'support_plane' in previous:value['support_plane']=_copy(previous['support_plane'])
        self._attachment=value
        self._step=step

    def _attachment_value(self, record, step, *, require_current_confirmation):
        _fields(record,('candidate_id','evidence','center_gripper_m','rotation_gripper',
                        'half_extents_m','uncertainty_m','basis','reason'),('surface_memberships','surface_seed_gripper_m','estimate_gripper_odom','support_plane'))
        _identifier(record['candidate_id'])
        target=self._targets.get(_json(record['candidate_id']))
        if target is None or (require_current_confirmation and target['status']!='confirmed'):
            raise ValueError('Attachment estimate requires confirmed candidate')
        self._evidence(record['evidence'],step,depth=True)
        _vector(record['center_gripper_m']);_vector(record['half_extents_m'])
        _camera(record['rotation_gripper'],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        uncertainty=_number(record['uncertainty_m']);_prose(record['reason'])
        if (record['basis']!='sensor_supported_estimate' or not .005<=uncertainty<=.15
                or any(not .002<=x<=.5 for x in record['half_extents_m'])
                or sum(x*x for x in record['center_gripper_m'])>.75**2):
            raise ValueError('Invalid estimated attachment bounds')
        if 'surface_memberships' in record:self._validate_surface_memberships(record['surface_memberships'],step)
        if 'surface_seed_gripper_m' in record:_vector(record['surface_seed_gripper_m'])
        if 'support_plane' in record:self._validate_attachment_support(record['support_plane'],step)
        if 'estimate_gripper_odom' in record:
            frame=record['estimate_gripper_odom']
            if type(frame)is not list or len(frame)!=4:raise ValueError('Estimate frame shape')
            for row in frame:_vector(row,4)
            if frame[3]!=[0.,0.,0.,1.]:raise ValueError('Estimate frame affine row')
            _camera([row[:3] for row in frame[:3]],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        return _copy(dict(record,status='estimated',control_step=step,
                          confirmation_evidence=target['confirmation_evidence']))

    def _validate_surface_memberships(self, rows, step):
        if type(rows) is not list or len(rows)>64:raise ValueError('Surface membership capacity')
        seen=set()
        for row in rows:
            _fields(row,('geometry_id','indices','evidence','mask_sha256'))
            source=self._geometry.get(row['geometry_id'])
            if source is None or source['kind']!='voxel_cloud' or source['status']!='occupied':
                raise ValueError('Surface membership needs retained occupied source')
            if row['geometry_id'] in seen or row['evidence']!=source['evidence']:
                raise ValueError('Surface membership source mismatch')
            seen.add(row['geometry_id']);self._evidence(row['evidence'],step,depth=True)
            _sha(row['mask_sha256'])
            indices=row['indices']
            if type(indices) is not list or not indices or indices!=sorted(set(indices)):
                raise ValueError('Surface membership indices')
            for index in indices:_integer(index,maximum=len(source['centers_odom_m'])-1)

    def record_surface_memberships(self, rows, *, candidate_id, step):
        """Annotate exact source voxels; never erase geometry or create free space."""
        self._time(step)
        attachment=getattr(self,'_attachment',None)
        if (attachment is None or attachment['candidate_id']!=candidate_id
                or attachment['status'] not in ('estimated','possibly_attached')):
            raise ValueError('Surface membership candidate mismatch')
        self._validate_surface_memberships(rows,step)
        merged={r['geometry_id']:r for r in attachment.get('surface_memberships',[])}
        for row in rows:
            if row['geometry_id'] in merged and merged[row['geometry_id']]!=row:
                raise ValueError('Surface membership cannot be silently reassigned')
            merged[row['geometry_id']]=_copy(row)
        self._validate_surface_memberships(list(merged.values()),step)
        self._attachment=dict(attachment,surface_memberships=list(merged.values()))
        self._step=step

    def _validate_attachment_support(self, value, step):
        _fields(value,('height_odom_m','anchor_object_odom','memberships'))
        _number(value['height_odom_m']);frame=value['anchor_object_odom']
        if type(frame)is not list or len(frame)!=4:raise ValueError('Support anchor shape')
        for row in frame:_vector(row,4)
        if frame[3]!=[0.,0.,0.,1.]:raise ValueError('Support anchor affine row')
        _camera([row[:3] for row in frame[:3]],[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self._validate_surface_memberships(value['memberships'],step)

    def record_attachment_support(self, value, *, candidate_id, step):
        self._time(step);record=getattr(self,'_attachment',None)
        if record is None or record['candidate_id']!=candidate_id or record['status']=='released':raise ValueError('Support candidate mismatch')
        self._validate_attachment_support(value,step)
        old=record.get('support_plane')
        if old is not None:
            if abs(old['height_odom_m']-value['height_odom_m'])>.005:return
            merged={row['geometry_id']:row for row in old['memberships']}
            for row in value['memberships']:merged.setdefault(row['geometry_id'],_copy(row))
            value=dict(old,memberships=list(merged.values()))
        self._validate_attachment_support(value,step)
        self._attachment=dict(record,support_plane=_copy(value));self._step=step

    def attachment_event(self, event, *, candidate_id, step):
        self._time(step)
        record=getattr(self,'_attachment',None)
        if record is None or record['candidate_id']!=candidate_id:
            raise ValueError('Attachment candidate mismatch')
        if event=='close_started' and record['status']=='estimated':
            status='possibly_attached'
        elif event=='open_completed':status='released'
        else:raise ValueError('Invalid attachment transition')
        self._attachment=dict(record,status=status,event_step=step)
        self._step=step

    def attachment(self):
        record=getattr(self,'_attachment',None)
        return None if record is None else _copy(record)

    def contact_attachment(self):
        """An empty-hand contact intent needs a currently confirmed identity."""
        record=getattr(self,'_attachment',None)
        target=None if record is None else self._targets.get(_json(record['candidate_id']))
        if (record is None or record['status']!='estimated' or not record.get('surface_memberships')
                or target is None or target['status']!='confirmed'):return None
        return _copy(record)

    def snapshot(self):
        value=dict(version=VERSION, episode_id=self.episode_id,
            odom_frame_id=self.odom_frame_id, source_binding=self.source_binding,
            limits=self.limits, control_step=self._step, observations=list(self._observations.values()),
            targets=list(self._targets.values()), regions=list(self._regions.values()),
            geometry=list(self._geometry.values()), actions=list(self._actions.values()),
            route=self._route, evicted=self._evicted)
        if getattr(self,'_attachment',None) is not None:value['attachment']=self._attachment
        return _copy(value)

    @classmethod
    def from_snapshot(cls, value, *, episode_id, odom_frame_id, source_binding):
        """Validate every field and source reference, with no implicit migration."""
        value = _copy(value)
        _fields(value, ("version", "episode_id", "odom_frame_id", "source_binding", "limits",
            "control_step", "observations", "targets", "regions", "geometry", "actions", "route", "evicted"),('attachment',))
        if (value["version"] != VERSION or value["episode_id"] != episode_id or
                value["odom_frame_id"] != odom_frame_id or value["source_binding"] != source_binding):
            raise ValueError("memory snapshot binding mismatch")
        memory = cls(episode_id, odom_frame_id, source_binding, limits=value["limits"])
        final_step = _integer(value["control_step"])
        for name in ("observations", "targets", "regions", "geometry", "actions", "route"):
            if type(value[name]) is not list or len(value[name]) > memory.limits["max_" + name]:
                raise ValueError("memory snapshot capacity")
        _fields(value["evicted"], memory._evicted)
        for number in value["evicted"].values():
            _integer(number)
        for row in value["observations"]:
            _fields(row, ("observation_id", "control_step", "odometry", "calibration_sha256", "sensor_sha256"))
            _text(row["observation_id"])
            if row["observation_id"] in memory._observations:
                raise ValueError("memory duplicate observation")
            memory.record_observation(row["observation_id"], step=row["control_step"],
                odometry=row["odometry"], calibration_sha256=row["calibration_sha256"], sensor_sha256=row["sensor_sha256"])
        # Each collection is independently ordered; replay validation at the final
        # step then restore its original source timestamps without advancing time.
        memory._time(final_step)
        target_ids = set()
        for row in value["targets"]:
            _fields(row, ("id", "status", "visible", "evidence", "control_step"),
                ("label", "last_visible_position_odom_m", "contradiction", "confirmation_evidence", "last_visible_evidence", "position_role"))
            original = _copy(row)
            target_key = _json(row["id"])
            if target_key in target_ids:
                raise ValueError("memory duplicate target")
            target_ids.add(target_key)
            step = _integer(row["control_step"], maximum=final_step)
            for key in ("evidence", "confirmation_evidence", "last_visible_evidence"):
                if key in row:
                    memory._evidence(row[key], step, depth=key == "last_visible_evidence")
            if "contradiction" in row:
                contradiction = row["contradiction"]
                _fields(contradiction, ("kind", "reason", "evidence"))
                if contradiction["kind"] != "identity_mismatch" or row["status"] != "rejected":
                    raise ValueError("memory snapshot identity contradiction")
                _prose(contradiction["reason"])
                memory._evidence(contradiction["evidence"], step)
            if row["status"] == "confirmed" and "confirmation_evidence" not in row:
                raise ValueError("memory confirmed identity lost evidence")
            has_position = "last_visible_position_odom_m" in row
            if has_position != ("last_visible_evidence" in row) or has_position != ("position_role" in row):
                raise ValueError("memory visible position provenance")
            if has_position and row["position_role"] != "last_visible_surface_not_grasp_pose":
                raise ValueError("memory position is not a grasp point")
            input_row = {k: v for k, v in row.items() if k not in ("control_step", "confirmation_evidence", "last_visible_evidence", "position_role", "contradiction")}
            if has_position:
                _vector(row["last_visible_position_odom_m"])
                if all(row["last_visible_evidence"][k] == row["evidence"][k]
                       for k in ("observation_id", "control_step", "calibration_sha256")):
                    input_row["position_evidence"] = row["last_visible_evidence"]
                else:
                    # The target may now be visible elsewhere while the last
                    # metric position still comes from an older measured frame.
                    input_row.pop("last_visible_position_odom_m", None)
            # An occluded confirmed target may retain its older visible position.
            if not input_row["visible"]:
                input_row.pop("last_visible_position_odom_m", None)
                input_row.pop("position_evidence", None)
                if has_position:
                    _vector(row["last_visible_position_odom_m"])
                if input_row["status"] == "confirmed":
                    input_row["status"] = "unknown"
            memory._step = 0
            memory.record_target(input_row, step=step)
            memory._targets[target_key] = original
        for name, method in (("regions", memory.observe_region), ("actions", memory.record_action)):
            target = getattr(memory, "_" + name)
            for row in value[name]:
                record = dict(row)
                step = _integer(record.pop("control_step", None), maximum=final_step)
                _text(record.get("id"))
                if record.get("id") in target:
                    raise ValueError("memory duplicate record")
                memory._step = 0
                if name == "regions" and record.get("kind") == "view":
                    if record.pop("coverage", None) != "observed_only":
                        raise ValueError("memory view coverage")
                    record.pop("kind")
                    memory.observe_view(record, step=step)
                else:
                    method(record, step=step)
        memory._step = final_step
        ids = [_text(row.get("id")) for row in value["geometry"] if type(row) is dict]
        if len(ids) != len(value["geometry"]) or len(set(ids)) != len(ids):
            raise ValueError("memory duplicate geometry")
        memory.record_geometry(value["geometry"], step=final_step)
        route_ids = set()
        previous_end = 0
        for row in value["route"]:
            _fields(row, ("action_id", "start_step", "end_step", "entry_odometry", "exit_odometry", "evidence", "meaning"))
            _text(row["action_id"])
            _integer(row["start_step"], maximum=final_step)
            _integer(row["end_step"], minimum=row["start_step"]+1, maximum=final_step)
            if row["action_id"] in route_ids or row["start_step"] < previous_end:
                raise ValueError("memory route sequence")
            route_ids.add(row["action_id"])
            previous_end = row["end_step"]
            _odom(row["entry_odometry"]); _odom(row["exit_odometry"])
            memory._evidence(row["evidence"], row["end_step"])
            if row["meaning"] != "executed_trace_not_free_space":
                raise ValueError("memory route is not free-space evidence")
            action = memory._actions.get(row["action_id"])
            if action is not None and (row["end_step"] != action["control_step"] or
                    row["end_step"]-row["start_step"] != action["steps_used"] or
                    any(row[k] != action[k] for k in ("entry_odometry", "exit_odometry", "evidence"))):
                raise ValueError("memory route/action evidence mismatch")
        memory._route = value["route"]
        memory._evicted = value["evicted"]
        if 'attachment' in value:
            row=_copy(value['attachment'])
            memberships=row.pop('surface_memberships',None)
            support=row.pop('support_plane',None)
            status=row.pop('status',None);created=_integer(row.pop('control_step',None),maximum=final_step)
            event=row.pop('event_step',None)
            confirmed=row.pop('confirmation_evidence',None)
            memory._evidence(confirmed,created)
            memory._step=0
            memory._attachment=memory._attachment_value(row,created,require_current_confirmation=False)
            memory._attachment['confirmation_evidence']=confirmed
            if status=='estimated':
                if event is not None:raise ValueError('Unexpected estimated attachment event')
            elif status in ('possibly_attached','released'):
                _integer(event,minimum=created,maximum=final_step)
                memory.attachment_event('close_started' if status=='possibly_attached' else 'open_completed',
                                        candidate_id=row['candidate_id'],step=event)
            else:raise ValueError('Invalid attachment snapshot status')
            if memberships is not None:
                memory._validate_surface_memberships(memberships,final_step)
                memory._attachment['surface_memberships']=memberships
            if support is not None:
                memory._validate_attachment_support(support,final_step)
                memory._attachment['support_plane']=support
            memory._step=final_step
        if memory.snapshot() != value:
            raise ValueError("memory snapshot not canonical")
        return memory

    def sha256(self):
        return hashlib.sha256(_json(self.snapshot()).encode("utf8")).hexdigest()
