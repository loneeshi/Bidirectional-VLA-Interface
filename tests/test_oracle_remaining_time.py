import importlib.util
from pathlib import Path
import tempfile
import unittest
from bvi import JsonlLogger,SerialRuntime,SkillSpec,SkillRequest,Requirement,SkillStatus
from test_runtime import FakeEnv,FakeSkill

spec=importlib.util.spec_from_file_location('runner',Path(__file__).resolve().parents[1]/'scripts/run_coordinator.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class OracleRemainingTimeTests(unittest.TestCase):
    def test_short_remaining_window_runs_but_still_stops_an_overlong_policy(self):
        for action_seconds,expected_steps,expected_status in [(1,2,SkillStatus.SUCCEEDED),(3,1,SkillStatus.TIMED_OUT)]:
            with self.subTest(action_seconds=action_seconds),tempfile.TemporaryDirectory() as tmp:
                now=[0.];env=FakeEnv();skill=FakeSkill();original=skill.act
                def act(obs):
                    now[0]+=action_seconds
                    return original(obs)
                skill.act=act
                runtime=SerialRuntime(env,{'pick':skill},{'pick':SkillSpec('pick',max_steps=10,timeout_seconds=90)},JsonlLogger(Path(tmp)/'events.jsonl','test'),clock=lambda:now[0])
                timeout=runner.oracle_skill_timeout(90,5)
                request=SkillRequest('c','pick','cup','f0',(Requirement('r','benchmark_success'),),max_steps=10,timeout_seconds=timeout)
                result=runtime.execute(request)
                self.assertEqual(result.feedback.status,expected_status)
                self.assertEqual(env.steps,expected_steps)
                self.assertLess(timeout,5)

    def test_no_request_after_budget_exhaustion(self):
        for remaining in (0,-1,.5,float('nan'),float('inf')):
            self.assertIsNone(runner.oracle_skill_timeout(90,remaining))
