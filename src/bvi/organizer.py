"""Execution organizer: bounded replanning without changing benchmark pointers.

The benchmark chooses which physical skills are feasible. The VLM chooses their
invocation budget or explicitly aborts. Sensor-derived grasp events yield control
back to it. This is execution organization, not free task-order planning.
"""
from dataclasses import replace
from .protocol import AllowedCall, Target, SkillSpec, SkillStatus, SkillFeedback, RequirementResult, RequirementState

ABORT='abort_task'

class InjectedClosureFault:
    """Explicit test-only six-step closure before the first Pick policy action.

    Never enabled by default. Does not reset after a retry. This is a synthetic
    low-level fault, not evidence of a natural SAC or pi05 failure.
    """
    def __init__(self,skill,logger):self.skill=skill;self.logger=logger;self.used=0
    def start(self,request,observation):self.skill.start(request,observation)
    def act(self,observation):
        if self.used<6:
            self.used+=1
            self.logger.emit('injected_closure_fault',frame_id=observation.frame_id,step=self.used,synthetic=True)
            return (0.,)*7+(-1.,)+(0.,)*5
        return self.skill.act(observation)
    def feedback(self,request,transition):return self.skill.feedback(request,transition)

class OrganizerView:
    def __init__(self, env, specs, max_slice_steps=40):
        if not 1<=max_slice_steps<=500:raise ValueError('Invalid organizer slice')
        self.env=env
        self.specs={k:replace(v,max_steps=min(v.max_steps,max_slice_steps)) for k,v in specs.items()}
        self.specs[ABORT]=SkillSpec(ABORT,('task_stopped',),('task_stopped',),1,1.,())
        self.last_event=None

    def observe(self):
        o=self.env.observe()
        if not o.allowed_calls:return o
        instruction=(' Organize execution using the current feasible skills. Their order is constrained by the benchmark. '
            'Choose a bounded invocation; a step_limit means yield, not task failure. '
            'After missed_grasp or grasp_lost, inspect the new images and feedback before deciding to retry '
            'the currently feasible skill or abort_task. abort_task stops the episode without claiming success. '
            'Do not claim that a retry includes a repositioning controller. Avoid repeated futile attempts.')
        event='' if self.last_event is None else f' Latest execution event: {self.last_event}.'
        selected=tuple(x for x in o.images if x.camera in ('fetch_workspace','fetch_hand'))
        images=selected if len(selected)==2 else o.images[:2]
        return replace(o,images=images,task=o.task+instruction+event,
            targets=o.targets+(Target('episode','Stop this episode unsuccessfully','organizer'),),
            allowed_calls=o.allowed_calls+(AllowedCall(ABORT,'episode'),))

    def note_result(self,result):
        self.last_event={'skill':result.request.skill,'status':result.feedback.status.value,
                         'reason':result.feedback.reason,'steps':result.steps}

class GraspMonitor:
    """Yield on sustained loss or failed closure; does not alter native scoring.

    is_grasped is privileged benchmark feedback, disclosed in event logs. The
    closure threshold is a configurable engineering heuristic, not ground truth.
    """
    def __init__(self,skill,closure_steps=6,loss_steps=2):
        if closure_steps<1 or loss_steps<1:raise ValueError('Monitor windows must be positive')
        self.skill=skill;self.closure_steps=closure_steps;self.loss_steps=loss_steps

    def start(self,request,observation):
        self.closed=0;self.lost=0;self.seen_grasp=False;self.action=None
        self.skill.start(request,observation)

    def act(self,observation):
        self.action=self.skill.act(observation)
        return self.action

    def feedback(self,request,transition):
        f=self.skill.feedback(request,transition)
        if f.status is not SkillStatus.EXECUTING:return f
        raw=transition.info.get('is_grasped')
        if raw is None:return f
        while isinstance(raw,(list,tuple)) and len(raw)==1:raw=raw[0]
        if hasattr(raw,'item'):raw=raw.item()
        if not isinstance(raw,bool):return f
        if raw:
            self.seen_grasp=True;self.closed=0;self.lost=0;return f
        self.lost=self.lost+1 if self.seen_grasp else 0
        self.closed=self.closed+1 if self.action is not None and float(self.action[7])<-.5 else 0
        reason='grasp_lost' if self.lost>=self.loss_steps else ('missed_grasp' if not self.seen_grasp and self.closed>=self.closure_steps else None)
        if reason is None:return f
        return SkillFeedback(SkillStatus.INTERRUPTED,tuple(RequirementResult(r.id,RequirementState.UNSATISFIED) for r in request.requirements),reason,'organizer_monitor_using_benchmark_grasp')
