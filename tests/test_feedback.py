from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from bvi.feedback import FeedbackView, apply_view, summarize
from bvi.feedback.trajectory import TrajectoryAccumulator
from bvi.feedback.spawn_prior import load_prior
from bvi.protocol import SkillFeedback, SkillStatus


def test_raw_bytes_and_distinct_navigation_bindings():
    history = [dict(skill='navigate', target_id=t, steps=4,
                    feedback=SkillFeedback(SkillStatus.TIMED_OUT))
               for t in ('object-abc', 'destination-abc')]
    encode = lambda x: json.dumps(x, default=asdict, ensure_ascii=False)
    assert encode(summarize(history, 'raw_v0')) == encode(list(history))
    summary = summarize(history, 'object_v1')
    assert len(summary['objects']['abc']['last_by_family']) == 2
    assert summary['steps_total'] == 8


def test_trajectory_retreat_stall_and_no_success_inference():
    trace = TrajectoryAccumulator('pick', 'object-a', dict(distance=.4, grasped=False))
    for distance in (.05, .2, .2):
        trace.update(dict(distance=distance, grasped=False))
    result = trace.finish('step_limit')
    assert result['distance_min'] == .05 and result['distance_end'] == .2
    assert result['distance_at_min_step'] == 1 and result['stalled_steps'] == 1
    assert result['steps'] == 3 and not result['grasped_ever']


def test_views_remove_dataclass_progress_without_mutating_input():
    context = dict(task='task', targets=[dict(id='a', description='oracle')], image_order=['cam'],
        feedback_history=[dict(feedback=SkillFeedback(SkillStatus.TIMED_OUT, progress=.8,
                                                    progress_source='fixture'), trajectory={'distance':.1})])
    before = deepcopy(context)
    result, images = apply_view(context, ('image',), FeedbackView(False, False, False, False))
    assert images == () and result['image_order'] == []
    assert 'progress' not in result['feedback_history'][0]['feedback']
    assert 'trajectory' not in result['feedback_history'][0]
    assert result['targets'] == [{'id':'a'}] and context == before


def test_prior_requires_independent_provenance(tmp_path):
    path = tmp_path/'prior.json'
    path.write_text(json.dumps(dict(schema='spawn-prior/1', training_plan_uids=['train-a'],
                                    bins=[dict(samples=10, successes=4)])))
    assert len(load_prior(path, ['val-a'])['sha256']) == 64
    with pytest.raises(ValueError, match='overlaps'):
        load_prior(path, ['train-a'])


def test_structured_goal_mask_removes_embedded_coordinates():
    goal=dict(object_id='object-a',object_name='apple',destination_id='destination-a',destination=[123,456,789])
    context=dict(task='Move objects. Goals: '+json.dumps([goal])+' Suffix.',targets=[dict(id='object-a')])
    result,_=apply_view(context,(),FeedbackView(structured_goals=False))
    assert '123' not in result['task'] and 'apple' in result['task'] and result['task'].endswith(' Suffix.')


def test_instrumentation_does_not_change_action_or_feedback():
    from types import SimpleNamespace as NS
    from bvi.feedback.trajectory import TrajectorySkill
    feedback=SkillFeedback(SkillStatus.SUCCEEDED)
    snapshots=iter([dict(distance=.5,grasped=False),dict(distance=.1,grasped=True)])
    delegate=NS(start=lambda *args:None,act=lambda obs:(.1,)*13,feedback=lambda *args:feedback)
    adapter=NS(resolve_request_index=lambda req:1,progress_snapshot=lambda i:next(snapshots))
    wrapper=TrajectorySkill(delegate,adapter)
    wrapper.start(NS(skill='pick',target_id='object-a'),None)
    assert wrapper.act(None)==delegate.act(None)
    assert wrapper.feedback(None,None) is feedback
    assert wrapper.invocation_digest('done')['grasped_ever']
