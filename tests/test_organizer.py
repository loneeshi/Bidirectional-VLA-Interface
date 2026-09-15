import unittest,tempfile,json,base64
from pathlib import Path
from dataclasses import replace
from bvi.organizer import OrganizerView,GraspMonitor
from bvi import *
from test_runtime import FakeEnv,FakeSkill

class ContactEnv(FakeEnv):
    def __init__(self,grasps):super().__init__();self.grasps=grasps
    def step(self,action):
        t=super().step(action)
        return replace(t,info={'is_grasped':[self.grasps[min(self.steps-1,len(self.grasps)-1)]]})

class OrganizerTests(unittest.TestCase):
    def run_monitor(self,grasps,success=100):
        env=ContactEnv(grasps);skill=FakeSkill(successful_at=success,action=[0.]*7+[-1.]+[0.]*5)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        log=JsonlLogger(Path(temp.name)/'events.jsonl','offline')
        specs={'pick':SkillSpec('pick')};view=OrganizerView(env,specs,10)
        runtime=SerialRuntime(env,{'pick':GraspMonitor(skill,closure_steps=3,loss_steps=2)},specs,log)
        req=SkillRequest('one','pick','cup','f0',(Requirement('r','benchmark_success'),),10,30)
        result=runtime.execute(req);view.note_result(result)
        return env,result,view,log

    def test_missed_grasp_yields_before_full_skill_horizon(self):
        env,r,v,_=self.run_monitor([False]*10)
        self.assertEqual(env.steps,3);self.assertEqual(r.feedback.reason,'missed_grasp')
        self.assertEqual(r.feedback.status,SkillStatus.TIMED_OUT)
        self.assertIn('missed_grasp',v.observe().task)
        self.assertEqual(len(v.observe().allowed_calls),2)
        self.assertEqual(env.observe().allowed_calls,(AllowedCall('pick','cup'),))

    def test_transient_contact_is_not_sustained_loss(self):
        env,r,_,_=self.run_monitor([True,False,True,False,False])
        self.assertEqual(env.steps,5);self.assertEqual(r.feedback.reason,'grasp_lost')

    def test_native_success_has_priority(self):
        env,r,_,_=self.run_monitor([False]*8,success=3)
        self.assertEqual(r.feedback.status,SkillStatus.SUCCEEDED)

    def test_native_failure_has_priority(self):
        class Fail(FakeSkill):
            def feedback(self,r,t):return SkillFeedback(SkillStatus.FAILED,reason='benchmark_fail')
        m=GraspMonitor(Fail(),closure_steps=1);r=SkillRequest('x','pick','cup','f0',())
        e=ContactEnv([False]);m.start(r,e.observe());m.act(e.observe())
        self.assertEqual(m.feedback(r,e.step([0.]*13)).reason,'benchmark_fail')

    def test_vlm_can_choose_abort_after_event_without_extra_physics(self):
        env,result,view,logger=self.run_monitor([False]*10)
        png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5e0AAAAASUVORK5CYII=')
        obs=replace(view.observe(),images=(ImageFrame('fixture',png),))
        class Transport:
            provider='offline';model='fixture'
            def generate(self,request):
                self.prompt=request.prompt
                return VLMResponse(json.dumps(dict(call_id='stop',skill='abort_task',target_id='episode',observation_id='f3',requirements=[dict(id='stop',predicate='task_stopped')],max_steps=1,timeout_seconds=1)),None,None)
        transport=Transport();coordinator=VLMCoordinator(transport,view.specs,logger,APIBudget('offline',1,200,.01,.01))
        request=coordinator.decide(obs,[view.last_event])
        self.assertEqual(request.skill,'abort_task');self.assertIn('missed_grasp',transport.prompt)
        self.assertEqual(env.steps,3)

if __name__=='__main__':unittest.main()
