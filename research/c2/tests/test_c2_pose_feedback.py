import hashlib
import io
import json
import math
from dataclasses import replace
from types import SimpleNamespace

from PIL import Image
import pytest

from bvi.pose_goal import validate_goal, FloorMesh, GoalPoseRecovery
from bvi.feedback.c2_context import CaseBank, C2ContextCompiler, InvocationFrames
from bvi.protocol import ImageFrame, Observation, ProtocolError, SkillRequest, Requirement, SkillSpec
from bvi.coordinator import parse_request, request_schema, VLMRequest
from bvi.bridge import encode_request, decode_request
from test_pose_recovery import setup, fail_pick


def goal(**kw): return {'frame':'base_at_request','x_m':.2,'y_m':0.,'yaw_rad':0.,**kw}

@pytest.mark.parametrize('change', [dict(x_m=float('nan')),dict(y_m=True),dict(frame='world'),
    dict(x_m=.51),dict(yaw_rad=2.),dict(x_m=0.),dict(extra=0)])
def test_goal_rejects_ambiguous_nonfinite_unbounded_or_empty(change):
    with pytest.raises(ProtocolError): validate_goal(goal(**change))


def test_floor_segment_does_not_bridge_holes(tmp_path):
    p=tmp_path/'floor.obj'
    p.write_text('v 0 0\nv 1 0\nv 1 1\nv 0 1\nv 2 0\nv 3 0\nv 3 1\nv 2 1\n'
                 'f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\n')
    mesh=FloorMesh(p)
    assert mesh.contains_segment((.1,.1),(.9,.9))
    assert mesh.contains_segment((.9,.9),(.1,.1))
    assert not mesh.contains_segment((.5,.5),(2.5,.5))
    assert not mesh.contains_segment((1.5,.5),(1.5,.5))


def structured_setup():
    adapter,target,_=setup();fail_pick(adapter,target)
    backend=SimpleNamespace(snapshot=lambda:dict(base_pose=[1.,2.,math.pi/2],arm_qpos=[0.]*7,body_qpos=[0.]*3),
                            feasible=lambda c,t:True, start=lambda c,t:None)
    adapter.recovery=GoalPoseRecovery(adapter,backend)
    return adapter,target


def request(adapter,target):
    return SkillRequest('a','reposition',target,'0',(Requirement('arrival','pose_arrived'),),200,10,recovery_goal=goal())


def test_goal_binds_request_frame_and_cannot_repeat_sac_start():
    adapter,target=structured_setup(); req=request(adapter,target)
    pair,c,current=adapter.recovery.prepare(req)
    assert c['base_pose']==pytest.approx([1.,2.2,math.pi/2])
    adapter.recovery.sac_starts.append({'pair':pair,'pose':c})
    with pytest.raises(ProtocolError,match='repeats'):adapter.recovery.prepare(req)


def test_recovery_retreat_does_not_rotate_held_object():
    from bvi.pose_goal import recovery_velocity
    v,w,done=recovery_velocity([0.,0.,0.],[-.2,0.,0.])
    assert v<0 and abs(w)<1e-8 and not done
    assert recovery_velocity([0.,0.,0.],[.2,0.,0.])[0]>0
    assert recovery_velocity([0.,0.,0.],[0.,0.,0.])[2]
    assert recovery_velocity([0.,0.,0.],[0.,.2,0.])[0]==0


def test_structured_schema_and_c0_separation():
    adapter,target=structured_setup();obs=adapter.observe()
    spec={'reposition':SkillSpec('reposition',('pose_arrived',),('pose_arrived',),200,10)}
    schema=request_schema(obs,spec,executor_horizons={'reposition':200})
    assert 'recovery_goal' in schema['properties'] and 'recovery_candidate_id' not in schema['properties']
    payload=dict(call_id='a',skill='reposition',target_id=target,observation_id='0',
                 requirements=[dict(id='arrived',predicate='pose_arrived')],timeout_seconds=10,recovery_goal=goal())
    assert parse_request(json.dumps(payload),obs,spec,executor_horizons={'reposition':200}).recovery_goal==goal()
    c0,_,_=setup('C0')
    assert 'recovery_goal' not in request_schema(c0.observe(),{})['properties']


def image():
    b=io.BytesIO();Image.new('RGB',(20,20),'blue').save(b,format='PNG')
    return ImageFrame('fetch_workspace',b.getvalue())


def bank(tmp_path, uid='train-a'):
    im=image(); (tmp_path/'frame.png').write_bytes(im.data)
    case=dict(parent_uid=uid,call_id='old',skill='pick',object_category='bowl',
              checkpoint_sha256='a'*64,source_sha256='b'*64,
              start_state={'distance':.2},result={'status':'failed'},
              frames=[dict(path='frame.png',step=1,event='start',sha256=hashlib.sha256(im.data).hexdigest())])
    p=tmp_path/'bank.json';p.write_text(json.dumps({'cases':[case]}))
    return p


def test_case_bank_excludes_all_attempts_and_checks_hash(tmp_path):
    p=bank(tmp_path, 'test-1')
    with pytest.raises(ProtocolError,match='Leaked'):CaseBank(p,{'test-1'})
    (tmp_path/'frame.png').write_bytes(b'bad')
    with pytest.raises(ProtocolError,match='SHA'):CaseBank(p,{'another'})


def test_feedback_variants_have_no_trace_leak_to_basic(tmp_path):
    obs=Observation('step-10',10,images=(image(),image()))
    rows=[dict(step=0,event='start',image=image(),state={'distance':.2})]
    history=[dict(call_id='c',skill='pick',target_id='obj',steps=10,
                  trajectory={'SECRET':'trace'},pose_recovery={'SECRET':'trace'},feedback={'status':'timed_out'})]
    outputs=[]
    for variant in ('basic','trace','experience'):
        compiler=C2ContextCompiler(variant,tmp_path/variant,
            CaseBank(bank(tmp_path),{'eval'}) if variant=='experience' else None,
            {'obj':'bowl'},{('pick','obj'):'a'*64})
        compiler.latest={'skill':'pick','target_id':'obj','call_id':'c','rows':rows}
        context,ims=compiler.compile({},obs,history)
        assert 'SECRET' not in json.dumps(context)
        outputs.append((context,ims))
    assert [len(o[1]) for o in outputs]==[2,3,4]
    assert outputs[2][0]['c2_feedback']['experience']['available']
    assert not outputs[0][0]['c2_feedback']['trajectory']['available']


def test_trace_is_bounded_chronological_and_end_aligned():
    trace=InvocationFrames()
    for step in range(200):
        trace.update(Observation(str(step),step,images=(image(),)),
                     dict(grasped=step>40 and step<80,distance=abs(100-step)*.01))
    rows=trace.finish()
    assert [r['step'] for r in rows]==[0,41,80,100,199]
    assert rows[-1]['event']=='end'


def test_bridge_extra_images_are_only_c2():
    req=VLMRequest('system','prompt',(image(),)*5,{'properties':{'recovery_goal':{}}},2048,'a'*32)
    assert len(decode_request(encode_request(req,'openai','model','auth')).images)==5
    legacy=replace(req,schema={'properties':{}})
    with pytest.raises(ProtocolError):decode_request(encode_request(legacy,'openai','model','auth'))


def test_rejected_admission_is_zero_step_and_not_a_retry():
    from bvi.runtime import SerialRuntime
    from bvi.pose_recovery import RepositionSkill
    adapter,target=structured_setup()
    adapter.recovery.backend.feasible=lambda c,t:False
    spec=SkillSpec('reposition',('pose_arrived',),('pose_arrived',),200,10)
    result=SerialRuntime(adapter,{'reposition':RepositionSkill(adapter)},
                         {'reposition':spec},adapter.logger).execute(request(adapter,target))
    assert result.feedback.status.value=='rejected' and result.steps==0
    assert adapter.recovery.attempts==[] and adapter.retry_ledger.retries=={}


def test_runner_changes_only_c2_caps_and_binds_initial_hash():
    from run_c2_feedback import command
    row=dict(seed=0,plan_uid='p',initial_state_sha256='f'*64)
    argv=command(row,'trace','out','cp','bridge','auth','manifest','bank',.005,.6)
    def value(key):return argv[argv.index(key)+1]
    assert value('--continuation-condition')=='C2'
    assert value('--max-calls')=='90' and value('--max-wall-seconds')=='1800'
    assert value('--feedback-profile')=='raw_v0' and '--spawn-prior' not in argv
    assert value('--expected-initial-state-sha256')=='f'*64


def test_structured_c2_navigation_is_one_shot():
    from bvi.continuation import ContinuationGoalAdapter
    old,target,_=setup()
    adapter=ContinuationGoalAdapter(old.base,'C2',structured_recovery=True)
    pair=('navigate',target)
    adapter.retry_ledger.begin(pair);adapter.retry_ledger.finish(pair,False)
    assert not adapter.retry_ledger.can_call(pair)
    with pytest.raises(ProtocolError):adapter.retry_ledger.begin(pair)


def test_retrieve_does_not_prefer_success(tmp_path):
    p=bank(tmp_path); data=json.loads(p.read_text())
    other=json.loads(json.dumps(data['cases'][0]));other['call_id']='z';other['result']['status']='succeeded'
    data['cases'].append(other);p.write_text(json.dumps(data))
    b=CaseBank(p,{'eval'})
    assert b.retrieve('pick','bowl',{'distance':.2},'a'*64,1)[0]['call_id']=='old'
    assert not b.retrieve('pick','bowl',{'distance':.2},'c'*64)


def test_full_coordinator_input_output_with_compiler_and_mock_model(tmp_path):
    from bvi.coordinator import VLMCoordinator, VLMResponse, APIBudget
    from bvi.logging import JsonlLogger
    adapter,target=structured_setup();obs=replace(adapter.observe(),images=(image(),image()))
    spec={'reposition':SkillSpec('reposition',('pose_arrived',),('pose_arrived',),200,10)}
    # Keep only this tool for the mock roundtrip, not a restriction on the real GPT catalog.
    obs=replace(obs,allowed_calls=tuple(c for c in obs.allowed_calls if c.skill=='reposition'))
    payload=dict(call_id='model-1',skill='reposition',target_id=target,observation_id='0',
                 requirements=[dict(id='arrival',predicate='pose_arrived')],timeout_seconds=10,recovery_goal=goal())
    seen=[]
    def generate(req):
        seen.append(req);return VLMResponse(json.dumps(payload),'fake-id',{})
    transport=SimpleNamespace(provider='mock',model='mock',generate=generate)
    compiler=C2ContextCompiler('basic',tmp_path/'context')
    coordinator=VLMCoordinator(transport,spec,JsonlLogger(tmp_path/'events.jsonl','test'),
        APIBudget('cpu-only-mock',1,2048,.01,.01),executor_horizons={'reposition':200},context_compiler=compiler)
    result=coordinator.decide(obs,[])
    assert result.recovery_goal==goal() and result.max_steps==200
    prompt=json.loads(seen[0].prompt)
    assert prompt['c2_feedback']['variant']=='basic'
    assert prompt['recovery']['coordinate_convention'].startswith('base_at_request')
    assert len(seen[0].images)==2
    assert seen[0].episode_scope['max_calls']==1


def test_physical_retries_obey_episode_cap_before_global_cap(tmp_path):
    from bvi.bridge import BridgeProcessor
    from bvi.coordinator import APIBudget
    class APIConnectionError(Exception):pass
    attempts=[]
    def fail(r):attempts.append(r);raise APIConnectionError()
    processor=BridgeProcessor(SimpleNamespace(provider='mock',model='mock',generate=fail),
        APIBudget('auth',50,2048,1.,.01),tmp_path,max_provider_retries=10,retry_delay_seconds=0)
    request=VLMRequest('s','p',(image(),),{'properties':{}},2048,'b'*32,
                       dict(id='c'*64,max_calls=2,max_cost_usd=.02))
    result=processor.process(encode_request(request,'mock','mock','auth'))
    assert result['error_type']=='APIBudgetExhausted' and len(attempts)==2
    assert len(list(tmp_path.glob('*.claim.json')))==2
