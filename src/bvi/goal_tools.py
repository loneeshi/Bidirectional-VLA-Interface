"""Goal-grounded PPO/SAC tools with an independent, unchanged native scorer.

No pointer writes, reset, or evaluate calls. A requested target selects the
policy's object/goal features and checkpoint, not the benchmark's current task.
The native sequential scorer still imposes the disclosed object-order constraint.
"""
from dataclasses import replace
import hashlib
import json

from .mshab_adapter import OfficialRLSkill, jsonable, scalar
from .protocol import (AllowedCall, Target, ProtocolError, SkillFeedback,
                       SkillStatus, RequirementResult, RequirementState)


PROGRESS_SOURCE = 'heuristic_policy_observation_relative_geometry_v1'


def relative_policy_progress(skill, start, current):
    """Bounded local progress using only quantities already in policy observations.

    This is a disclosed rule proxy, not a learned head and not a benchmark
    completion predicate.  The start snapshot makes progress invocation-local.
    """
    if skill not in ('navigate', 'pick', 'place'):
        raise ProtocolError('Unsupported progress skill')
    initial = float(start['distance'])
    distance = float(current['distance'])
    if not all(map(lambda value: value >= 0 and value < float('inf'), (initial, distance))):
        raise ProtocolError('Invalid policy-observation distance')
    reduction = max(0., min(1., (initial - distance) / max(initial, 1e-6)))
    if skill == 'navigate':
        return min(.99, reduction)
    if skill == 'pick':
        return 1. if bool(current['grasped']) else min(.9, reduction * .9)
    # Place progress combines transport toward the goal and release.  It never
    # calls containment or the native Place checker.
    release = .1 if bool(start['grasped']) and not bool(current['grasped']) else 0.
    return min(1., reduction * .9 + release)


class GoalCatalog:
    def __init__(self, plan):
        self.bindings = {}
        self.targets = []
        self.goals = []
        tasks = plan.subtasks
        if len(tasks) != 20:
            raise ProtocolError('Goal tools require five-object TidyHouse')
        for offset in range(0, 20, 4):
            nav, pick, delivery, place = tasks[offset:offset+4]
            if [x.type for x in (nav,pick,delivery,place)] != ['navigate','pick','navigate','place'] or pick.obj_id != place.obj_id:
                raise ProtocolError('Unsupported goal binding')
            key = hashlib.sha256(str(pick.obj_id).encode()).hexdigest()[:12]
            obj, dest = f'object-{key}', f'destination-{key}'
            if any(t.id == obj for t in self.targets):
                raise ProtocolError('Ambiguous repeated object')
            self.targets.extend((Target(obj, f'Object {pick.obj_id}', 'native_goal_catalog'),
                Target(dest, f'Placement region assigned to {pick.obj_id}', 'native_goal_catalog')))
            for skill, target, index in [('navigate',obj,offset),('pick',obj,offset+1),
                                         ('navigate',dest,offset+2),('place',dest,offset+3)]:
                self.bindings[(skill,target)] = index
            self.goals.append(dict(object_id=obj, object_name=pick.obj_id, destination_id=dest,
                                   destination=jsonable(getattr(place,'goal_pos',None)),
                                   region=jsonable(getattr(place,'goal_rectangle_corners',None))))
        self.calls = tuple(AllowedCall(*pair) for pair in sorted(self.bindings))

    def resolve(self, request):
        try:
            return self.bindings[(request.skill, request.target_id)]
        except KeyError as exc:
            raise ProtocolError('Unknown skill/grounded target pair') from exc


class GoalToolAdapter:
    def __init__(self, adapter):
        self.base = adapter
        self.catalog = GoalCatalog(adapter.original_plan)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def resolve_request_index(self, request):
        return self.catalog.resolve(request)

    def view(self, observation):
        if self.base.ended:
            return replace(observation, targets=(), allowed_calls=(), metadata={'ended':True})
        # Deliberately omit subtask_index, subtask_type and the official plan.
        return replace(observation, targets=tuple(self.catalog.targets), allowed_calls=self.catalog.calls,
            task=('Move all five specified objects to their assigned destinations. Complete objects in '
                  'the listed order, as required by the native sequential benchmark. Decompose each '
                  'object goal into physical tool calls yourself using images and feedback. You may '
                  'invoke any catalog tool/target; no hidden scheduler substitutes your choice. '
                  'navigate approaches an object or destination; pick grasps its object; place releases '
                  'the assigned object in its destination. Manipulation requires suitable proximity '
                  'and grasp state. A slice timeout is not a completed goal. Goals: '+json.dumps(self.catalog.goals)),
            metadata={'interface':'goal-grounded-ppo-sac/1', 'goal_source':'native_goal_catalog',
                      'native_scoring':'unchanged_sequential', 'ended':False})

    def observe(self):
        return self.view(self.base.observe())

    def step(self, action):
        transition = self.base.step(action)
        return replace(transition, observation=self.view(transition.observation))

    def conditioned_policy(self, observation, index):
        """Official 42D feature construction, conditioned on the requested binding.

        Preserve the real depth history. Recompute only the current state with
        upstream flattening and poses; never render/step/evaluate for observe.
        """
        import torch
        from mani_skill.utils.common import flatten_state_dict
        from mani_skill.utils.structs.pose import vectorize_pose
        u = self.uenv
        inv = u.agent.base_link.pose.inv()
        obj, goal = u.subtask_objs[index], u.subtask_goals[index]
        extra = dict(tcp_pose_wrt_base=vectorize_pose(inv * u.agent.tcp.pose),
            obj_pose_wrt_base=torch.zeros((1,7),device=u.device),
            goal_pos_wrt_base=torch.zeros((1,3),device=u.device),
            is_grasped=torch.zeros(1,dtype=torch.bool,device=u.device))
        if obj is not None:
            extra['obj_pose_wrt_base'] = vectorize_pose(inv * obj.pose)
            extra['is_grasped'] = u.agent.is_grasping(obj,max_angle=30)
        if goal is not None:
            extra['goal_pos_wrt_base'] = (inv * goal.pose).p
        state = torch.cat([flatten_state_dict(u._get_obs_agent(),use_torch=True),
                           flatten_state_dict(extra,use_torch=True)],dim=1).float()
        if tuple(state.shape) != (1,42) or not torch.isfinite(state).all():
            raise ProtocolError('Invalid grounded policy state')
        return replace(observation,policy={**observation.policy,'state':state})

    def predicate(self, index):
        """Use pure upstream predicate helpers, never stateful evaluate()."""
        import torch
        u=self.uenv; task=u.task_plan[index]
        env_idx=torch.zeros(1,dtype=torch.long,device=u.device)
        obj,goal=u.subtask_objs[index],u.subtask_goals[index]
        if task.type=='pick':
            return u._pick_check_success(obj,env_idx)
        if task.type=='place':
            return u._place_check_success(obj,goal,task.goal_rectangle_corners,env_idx)
        if task.type=='navigate':
            return u._navigate_check_success(obj,goal,None,env_idx)
        raise ProtocolError('Unsupported grounded predicate')

    def progress_snapshot(self, index):
        """Read geometry/grasp state already represented in the official 42D input."""
        import torch
        u=self.uenv; task=u.task_plan[index]
        obj,goal=u.subtask_objs[index],u.subtask_goals[index]
        if task.type == 'navigate':
            target = goal.pose.p if goal is not None else obj.pose.p
            delta = u.agent.base_link.pose.p[..., :2] - target[..., :2]
            distance = torch.linalg.vector_norm(delta, dim=-1)
        elif task.type == 'pick':
            distance = torch.linalg.vector_norm(u.agent.tcp.pose.p - obj.pose.p, dim=-1)
        elif task.type == 'place':
            distance = torch.linalg.vector_norm(obj.pose.p - goal.pose.p, dim=-1)
        else:
            raise ProtocolError('Unsupported grounded progress target')
        grasped = False if obj is None else bool(scalar(u.agent.is_grasping(obj,max_angle=30)))
        value = float(scalar(distance))
        if not (value >= 0 and value < float('inf')):
            raise ProtocolError('Invalid grounded progress geometry')
        return {'distance': value, 'grasped': grasped}


class GoalRLSkill(OfficialRLSkill):
    def start(self, request, observation):
        self.selected_index = self.adapter.resolve_request_index(request)
        conditioned=self.adapter.conditioned_policy(observation,self.selected_index)
        super().start(request,conditioned)
        self.adapter.logger.emit('grounded_tool_binding',call_id=request.call_id,
            skill=request.skill,target_id=request.target_id,selected_index=self.selected_index,
            native_index=int(scalar(self.adapter.uenv.subtask_pointer)),
            state_sha256=hashlib.sha256(json.dumps(jsonable(conditioned.policy['state'])).encode()).hexdigest())

    def act(self, observation):
        return super().act(self.adapter.conditioned_policy(observation,self.selected_index))

    def feedback(self, request, transition):
        satisfied,checkers=self.adapter.predicate(self.selected_index)
        if bool(scalar(transition.info.get('fail',False))):
            status,state,reason=SkillStatus.FAILED,RequirementState.UNSATISFIED,'benchmark_fail'
        elif bool(scalar(satisfied)):
            status,state,reason=SkillStatus.SUCCEEDED,RequirementState.SATISFIED,'requested_native_predicate'
        elif transition.truncated:
            status,state,reason=SkillStatus.TIMED_OUT,RequirementState.UNKNOWN,'environment_horizon'
        else:
            status,state,reason=SkillStatus.EXECUTING,RequirementState.UNSATISFIED,None
        self.adapter.logger.emit('grounded_predicate',call_id=request.call_id,
            frame_id=transition.observation.frame_id,
            target_id=request.target_id,selected_index=self.selected_index,checkers=jsonable(checkers))
        evidence=(f'events.jsonl:grounded_predicate:{request.call_id}:{transition.observation.frame_id}',)
        return SkillFeedback(status,tuple(RequirementResult(r.id,state,evidence) for r in request.requirements),
                             reason,'native_requested_target_predicate')


class ProgressGoalRLSkill(GoalRLSkill):
    """Goal-grounded PPO/SAC with non-evaluator continuous progress feedback."""

    def start(self, request, observation):
        super().start(request, observation)
        self.progress_start = self.adapter.progress_snapshot(self.selected_index)

    def feedback(self, request, transition):
        current = self.adapter.progress_snapshot(self.selected_index)
        progress = relative_policy_progress(self.name, self.progress_start, current)
        frame_id = transition.observation.frame_id
        evidence=(f'events.jsonl:tool_progress:{request.call_id}:{frame_id}',)
        self.adapter.logger.emit('tool_progress', call_id=request.call_id,
            frame_id=frame_id, target_id=request.target_id,
            selected_index=self.selected_index, value=progress,
            source=PROGRESS_SOURCE, features=current)
        if bool(scalar(transition.info.get('fail',False))):
            status,state,reason=(SkillStatus.FAILED,RequirementState.UNSATISFIED,
                                 'benchmark_fail')
        elif transition.truncated:
            status,state,reason=(SkillStatus.TIMED_OUT,RequirementState.UNKNOWN,
                                 'environment_horizon')
        else:
            status,state,reason=(SkillStatus.EXECUTING,RequirementState.UNKNOWN,None)
        return SkillFeedback(status, tuple(RequirementResult(r.id,state,evidence)
                                            for r in request.requirements),
                             reason, 'policy_observation_progress', progress,
                             PROGRESS_SOURCE)
