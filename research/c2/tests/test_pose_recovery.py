from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bvi.continuation import ContinuationGoalAdapter
from bvi.coordinator import request_schema, parse_request
from bvi.pose_recovery import FetchPosePreparation, RepositionSkill, audit_pose_prior
from bvi.protocol import (ActionBounds, Observation, ProtocolError, Requirement,
                          SkillRequest, SkillSpec, SkillStatus, Transition, validate_request)
from bvi.runtime import SerialRuntime


def prior():
    c = dict(id='pose-a', skill='pick', object_category='024_bowl', frame='target_se2',
             units={'position':'m','angle':'rad'}, base_pose=[1.,0.,0.],
             arm_qpos=[0.]*7, body_qpos=[0.]*3, target_height_m=.7,
             evidence=dict(source_sha256='a'*64,checkpoint_sha256='b'*64,
                           parent_uid='train-1',samples=1,successes=1))
    return dict(schema='spawn-prior/2',training_plan_uids=['train-1'],candidates=[c])


class Backend:
    def __init__(self, base, reaches=True):
        self.base, self.reaches = base, reaches
    def feasible(self, c, target, excluded=()): return True
    def snapshot(self): return {'measured_step':self.base.steps}
    def start(self, c, target): self.start_step=self.base.steps
    def action(self): return (.1,)*13
    def arrived(self): return self.reaches and self.base.steps-self.start_step>=2


def setup(condition='C2', reaches=True):
    tasks=[]
    for i in range(5):
        for skill in ('navigate','pick','navigate','place'):
            tasks.append(SimpleNamespace(type=skill,obj_id=f'024_bowl-{i}'))
    events=[]
    logger=SimpleNamespace(emit=lambda event, **kw: events.append((event,deepcopy(kw))))
    base=SimpleNamespace(original_plan=SimpleNamespace(subtasks=tasks),steps=0,ended=False,
                         logger=logger, action_bounds=ActionBounds((-1.,)*13,(1.,)*13))
    base.observe=lambda: Observation(str(base.steps),base.steps)
    def step(action):
        base.steps+=1
        return Transition(base.observe())
    base.step=step
    data=prior() if condition=='C2' else None
    adapter=ContinuationGoalAdapter(base,condition,data)
    if condition=='C2': adapter.recovery.backend=Backend(base,reaches)
    target=adapter.catalog.goals[0]['object_id']
    return adapter,target,events


def fail_pick(adapter,target):
    adapter.retry_ledger.begin(('pick',target))
    adapter.retry_ledger.finish(('pick',target),False)


def reposition(adapter,target,steps=3):
    req=SkillRequest('prepare', 'reposition', target,str(adapter.steps),
                     (Requirement('arrival','pose_arrived'),),steps,10,
                     recovery_candidate_id='pose-a')
    spec=SkillSpec('reposition',('pose_arrived',),('pose_arrived',),steps,10)
    runtime=SerialRuntime(adapter,{'reposition':RepositionSkill(adapter)},
                          {'reposition':spec},adapter.logger)
    result=runtime.execute(req)
    adapter.finish_request(req,result.feedback.status is SkillStatus.SUCCEEDED)
    return result


def test_legacy_prior_cannot_become_executable_pose():
    # Synthetic legacy shape: public tests do not depend on private run assets.
    audit=audit_pose_prior({'schema':'spawn-prior/1','training_plan_uids':['train-example'],
                           'bins':[{'distance_m':0.5,'success_rate':0.8}]})
    assert not audit['ready']
    assert 'no_pose_candidates' in audit['reasons']
    assert audit_pose_prior(prior())['ready']


def test_failed_pick_requires_physical_arrival_then_single_use_ticket():
    adapter,target,events=setup(); fail_pick(adapter,target)
    req=SkillRequest('retry','pick',target,'0',(Requirement('done','benchmark_success'),),200,10)
    with pytest.raises(ProtocolError): adapter.begin_request(req)
    assert not any(c.skill=='pick' and c.target_id==target for c in adapter.observe().allowed_calls)
    result=reposition(adapter,target)
    assert result.feedback.status is SkillStatus.SUCCEEDED and result.steps==2
    assert adapter.recovery.can_retry(('pick',target),adapter.steps)
    adapter.begin_request(req)
    assert adapter.retry_ledger.retries[('pick',target)]==1
    assert adapter.recovery.sac_starts[-1]['call_id']=='retry'
    assert adapter.recovery.ready is None
    adapter.finish_request(req,False)
    with pytest.raises(ProtocolError): adapter.begin_request(req)


def test_preparation_timeout_never_grants_sac_retry_or_counts_as_sac_attempt():
    adapter,target,_=setup(reaches=False); fail_pick(adapter,target)
    result=reposition(adapter,target)
    assert result.feedback.status is SkillStatus.TIMED_OUT and result.steps==3
    assert adapter.recovery.ready is None
    assert adapter.retry_ledger.retries=={}
    assert adapter.recovery.attempts[0]['status']=='not_arrived'


def test_alias_pose_and_three_failed_preparations_are_bounded():
    adapter,target,_=setup(reaches=False); fail_pick(adapter,target)
    alias=deepcopy(adapter.recovery.prior['candidates'][0]); alias['id']='alias'
    adapter.recovery.prior['candidates'].append(alias)
    reposition(adapter,target)
    assert not adapter.recovery.offers()
    for i in (1,2):
        c=deepcopy(alias); c['id']=f'pose-{i}'; c['base_pose'][0]+=i
        adapter.recovery.prior['candidates'].append(c)
        req=SkillRequest(str(i),'reposition',target,str(adapter.steps),
                         (Requirement('arrival','pose_arrived'),),1,10,recovery_candidate_id=c['id'])
        adapter.recovery.begin(req); adapter.recovery.finish(req,False)
    c=deepcopy(alias); c['id']='fourth'; c['base_pose'][0]=9
    adapter.recovery.prior['candidates'].append(c)
    assert not adapter.recovery.offers()


def test_ticket_stale_after_another_action():
    adapter,target,_=setup(); fail_pick(adapter,target); reposition(adapter,target)
    adapter.step((0.,)*13)
    assert not adapter.recovery.can_retry(('pick',target),adapter.steps)


def test_c0_schema_tools_and_retry_are_unchanged():
    adapter,target,_=setup('C0')
    obs=adapter.observe(); specs={s:SkillSpec(s) for s in ('navigate','pick','place')}
    assert len(obs.allowed_calls)==20
    assert 'recovery' not in obs.metadata and adapter.recovery is None
    assert 'recovery_candidate_id' not in request_schema(obs,specs)['properties']
    fail_pick(adapter,target)
    assert not any(c.skill=='pick' and c.target_id==target for c in adapter.observe().allowed_calls)


def test_candidate_enters_gpt_schema_and_invalid_target_is_rejected():
    adapter,target,_=setup(); fail_pick(adapter,target)
    obs=adapter.observe()
    spec=SkillSpec('reposition',('pose_arrived',),('pose_arrived',),200,10)
    schema=request_schema(obs,{'reposition':spec},executor_horizons={'reposition':200})
    assert schema['properties']['recovery_candidate_id']['enum']==[None,'pose-a']
    payload=dict(call_id='a',skill='reposition',target_id=target,observation_id='0',
                 requirements=[dict(id='arrive',predicate='pose_arrived')],
                 timeout_seconds=10,recovery_candidate_id='pose-a')
    request=parse_request(json.dumps(payload),obs,{'reposition':spec},
                          executor_horizons={'reposition':200})
    assert request.recovery_candidate_id=='pose-a'
    payload['target_id']=adapter.catalog.goals[1]['object_id']
    with pytest.raises(ProtocolError):
        parse_request(json.dumps(payload),obs,{'reposition':spec},
                      executor_horizons={'reposition':200})


def test_missing_scene_validator_offers_nothing():
    adapter,target,_=setup(); fail_pick(adapter,target)
    adapter.recovery.backend=None
    assert adapter.recovery.offers()==[]


def test_fetch_preparation_uses_measured_arrival_and_preserves_gripper():
    # Exercise the real action/arrival state machine with a fake controller;
    # this deliberately does not claim real simulator collision validation.
    class Control:
        def __init__(self):
            self.xy=(0.,0.,0.); self.hold={}; self.calls=[]
        def pose(self): return self.xy
        def capture_hold(self):
            self.hold={'arm':(0.,)*7,'body':(0.,)*3,'gripper':(-.01,)}
        def action(self,v,w):
            self.calls.append((v,w)); return (v,w)
    control=Control()
    backend=FetchPosePreparation.__new__(FetchPosePreparation)
    backend.control=control
    backend.validate_motion=lambda c,t,s: [(1.,0.,0.)]
    backend.snapshot=lambda: {'base_pose':control.pose(), 'arm_qpos':[.2]*7,'body_qpos':[.1]*3}
    c=prior()['candidates'][0]; c['arm_qpos']=[.2]*7;c['body_qpos']=[.1]*3
    assert backend.feasible(c,'target')
    backend.start(c,'target')
    assert backend.action()[0]>0
    assert not backend.arrived()
    control.xy=(1.,0.,0.)
    backend.action()
    assert control.hold['arm']==(.2,)*7 and control.hold['gripper']==(-.01,)
    assert not backend.arrived(); assert not backend.arrived(); assert backend.arrived()
    assert not backend.feasible(c,'target')  # Already at same pose: not a recovery.
    backend.validate_motion=None
    assert not backend.feasible(c,'target')
    with pytest.raises(ProtocolError): backend.start(c,'target')


def test_failed_sources_retained_and_no_distance_only_export():
    from build_sac_pose_prior import candidate
    geometry=dict(world_frame='simulator_world', pose_layout='xyz_quaternion_wxyz',
        base_link_pose=[[2.,1.,0.,1.,0.,0.,0.]],
        object_pose=[[1.,1.,.7,1.,0.,0.,0.]],
        controller_qpos={'arm':[[0.]*7],'body':[[0.]*3]}, controller_joint_names={})
    binding=dict(skill='pick',category='024_bowl',checkpoint_sha='b'*64,
                 uid='train-1',scene=1,spawn_index=0)
    summary=dict(status='completed',ever_native_success=False,
                 force_violation=True,final_native_success=False)
    row=candidate(binding,summary,geometry,'a'*64)
    assert row['base_pose']==[1.,0.,0.]
    assert row['evidence']['successes']==0 and row['evidence']['force_violation']
    del geometry['base_link_pose']
    with pytest.raises(KeyError): candidate(binding,summary,geometry,'a'*64)
