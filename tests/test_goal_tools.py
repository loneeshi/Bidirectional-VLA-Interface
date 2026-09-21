from types import SimpleNamespace as NS
import pytest
from bvi.goal_tools import GoalCatalog, GoalToolAdapter
from bvi.protocol import Observation, SkillRequest, ProtocolError


@pytest.mark.parametrize('name', ['navigate','pick','place'])
def test_success_feedback_has_resolvable_evidence(name):
    from bvi.goal_tools import GoalRLSkill
    from bvi.protocol import Requirement,Transition,validate_feedback,SkillStatus
    records=[]
    adapter=NS(predicate=lambda i:(True,{'native_success':True}),
               logger=NS(emit=lambda event,**kw:records.append(dict(event=event,**kw))))
    skill=GoalRLSkill(name,adapter,'/unused');skill.selected_index=3
    req=SkillRequest('call',name,'target','f0',(Requirement('done','benchmark_success'),))
    transition=Transition(Observation('f1',1),info={'fail':False})
    feedback=skill.feedback(req,transition)
    validate_feedback(req,feedback)
    assert feedback.status is SkillStatus.SUCCEEDED
    assert feedback.requirements[0].evidence==('events.jsonl:grounded_predicate:call:f1',)
    assert records[-1]['call_id']=='call' and records[-1]['frame_id']=='f1'
    failed=skill.feedback(req,Transition(Observation('f2',2),info={'fail':True}))
    validate_feedback(req,failed)
    assert failed.status is SkillStatus.FAILED


@pytest.mark.parametrize('skill,start,current,expected', [
    ('navigate', {'distance': 4., 'grasped': False}, {'distance': 2., 'grasped': False}, .5),
    ('pick', {'distance': .4, 'grasped': False}, {'distance': .1, 'grasped': False}, .675),
    ('pick', {'distance': .4, 'grasped': False}, {'distance': .2, 'grasped': True}, 1.),
    ('place', {'distance': 1., 'grasped': True}, {'distance': .2, 'grasped': False}, .82),
])
def test_policy_observation_progress_is_bounded_and_not_a_native_predicate(
        skill, start, current, expected):
    from bvi.goal_tools import relative_policy_progress
    assert relative_policy_progress(skill, start, current) == pytest.approx(expected)


def test_progress_feedback_never_calls_native_predicate_or_declares_success():
    from bvi.goal_tools import ProgressGoalRLSkill, PROGRESS_SOURCE
    from bvi.protocol import Requirement,Transition,validate_feedback,SkillStatus,RequirementState
    records=[]
    adapter=NS(progress_snapshot=lambda i:{'distance': .25, 'grasped': False},
               predicate=lambda i:(_ for _ in ()).throw(AssertionError('native predicate leak')),
               logger=NS(emit=lambda event,**kw:records.append(dict(event=event,**kw))))
    skill=ProgressGoalRLSkill('navigate',adapter,'/unused')
    skill.selected_index=3
    skill.progress_start={'distance': 1., 'grasped': False}
    req=SkillRequest('call','navigate','target','f0',(Requirement('done','benchmark_success'),))
    feedback=skill.feedback(req,Transition(Observation('f1',1),info={'fail':False}))
    validate_feedback(req,feedback)
    assert feedback.status is SkillStatus.EXECUTING
    assert feedback.requirements[0].state is RequirementState.UNKNOWN
    assert feedback.progress == pytest.approx(.75)
    assert feedback.progress_source == PROGRESS_SOURCE
    assert records[-1]['event'] == 'tool_progress'


def plan():
    return NS(subtasks=[NS(type=skill,obj_id=f'obj-{i}',goal_pos=[i,0,0],goal_rectangle_corners=[])
                       for i in range(5) for skill in ('navigate','pick','navigate','place')])


def test_catalog_all_targets_and_skill_bindings():
    c=GoalCatalog(plan())
    assert len(c.calls)==20 and len(c.targets)==10
    for i,g in enumerate(c.goals):
        for skill,target,index in [('navigate',g['object_id'],4*i),('pick',g['object_id'],4*i+1),
                                  ('navigate',g['destination_id'],4*i+2),('place',g['destination_id'],4*i+3)]:
            assert c.resolve(NS(skill=skill,target_id=target))==index
    with pytest.raises(ProtocolError):c.resolve(NS(skill='pick',target_id=c.goals[0]['destination_id']))


def test_no_next_subtask_hint_and_catalog_does_not_follow_pointer():
    base=NS(original_plan=plan(),ended=False)
    a=GoalToolAdapter(base)
    first=a.view(Observation('f',0,metadata={'subtask_index':0,'benchmark_info':{'subtask_type':'navigate'}}))
    later=a.view(Observation('g',10,metadata={'subtask_index':8}))
    assert first.allowed_calls==later.allowed_calls and len(first.allowed_calls)==20
    assert 'subtask_index' not in first.metadata and 'benchmark_info' not in first.metadata
    assert 'Decompose' in first.task and 'listed order' in first.task
    base.ended=True
    assert not a.view(first).allowed_calls


def test_goal_flag_only_changes_gpt_dispatch():
    import importlib.util,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import run_ppo_sac_paired16 as p
    args=(dict(seed=0,plan_uid='p'),'gpt',Path('/o'),Path('/c'),Path('/b'),'auth','sha')
    assert p.command(*args,goal_tools=True)==p.command(*args)+['--goal-tools','--progress-feedback']
