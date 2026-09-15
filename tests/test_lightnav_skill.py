import math
import unittest
from types import SimpleNamespace as NS

from bvi.lightnav_skill import (FetchNavigationControl, LightNavSkill, IndexedNavigationSkill, normalize, track_waypoint,
                               world_waypoint)
from bvi.protocol import ActionBounds, ProtocolError, Observation, ImageFrame, Requirement, SkillStatus
from bvi.vla_clients import NavigationPrediction


class TrackerTests(unittest.TestCase):
    def test_forward_left_and_in_place_turn(self):
        self.assertEqual(track_waypoint((0, 0, 0), (1, 0, 0)), (.25, 0., False))
        v, w, _ = track_waypoint((0, 0, 0), (0, 1, 0))
        self.assertEqual(v, 0)
        self.assertGreater(w, 0)
        v, w, done = track_waypoint((0, 0, 0), (0, 0, -.3))
        self.assertEqual(v, 0)
        self.assertLess(w, 0)
        self.assertFalse(done)

    def test_world_rotation_and_yaw_wrap(self):
        x, y, yaw = world_waypoint((4, 5, math.pi/2), (1, 0, math.pi))
        self.assertAlmostEqual(x, 4)
        self.assertAlmostEqual(y, 6)
        self.assertAlmostEqual(yaw, -math.pi/2)
        # Row2 is cumulative relative to capture, not relative to row1.
        self.assertEqual(world_waypoint((1, 1, 0), (2, 0, 0))[:2], (3, 1))

    def test_nonfinite_and_normalization(self):
        with self.assertRaises(ProtocolError):
            track_waypoint((0, 0, float('nan')), (1, 0, 0))
        self.assertEqual(normalize((-.01,), (-.01,), (.05,)), (-1.,))
        self.assertAlmostEqual(normalize((0.,), (-.01,), (.05,))[0], -2/3)
        self.assertEqual(normalize((.25,), (-1.,), (1.,)), (.25,))

    def adapter(self):
        def ctrl(cls, qpos, low, high, names=(), delta=True):
            c = type(cls, (), {})()
            c._normalize_action = True
            c._original_single_action_space = NS(shape=(len(low),), low=low, high=high)
            c.config = NS(use_delta=delta, use_target=False, joint_names=list(names))
            c.control_freq = 20
            c.qpos = [qpos]
            return c
        controllers = dict(
            arm=ctrl('PDJointPosController', [0]*7, [-.1]*7, [.1]*7),
            gripper=ctrl('PDJointPosMimicController', [.005]*2, [-.01], [.05], delta=False),
            body=ctrl('PDJointPosController', [0,0,.3], [-.1]*3, [.1]*3,
                      ['head_pan_joint','head_tilt_joint','torso_lift_joint']),
            base=ctrl('PDBaseForwardVelController', [0,0,0], [-1,-3.14], [1,3.14],
                      ['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint']))
        controllers['gripper']._target_qpos = [[-.01, -.01]]
        return NS(uenv=NS(agent=NS(controller=NS(controllers=controllers))),
                  logger=NS(emit=lambda *a, **k: None),
                  action_bounds=ActionBounds((-1.,)*13, (1.,)*13))

    def test_hold_preserves_closed_target_and_compensates_torso_drift(self):
        adapter = self.adapter()
        control = FetchNavigationControl(adapter)
        control.capture_hold()
        control.controllers['body'].qpos = [[0,0,.28]]
        action = control.action(.25, .314)
        self.assertEqual(len(action), 13)
        self.assertEqual(action[7], -1)  # Not zero, which would command opening.
        self.assertAlmostEqual(action[10], .2)
        self.assertAlmostEqual(action[11], .25)
        self.assertAlmostEqual(action[12], .1)

    def test_changed_controller_contract_is_rejected(self):
        adapter = self.adapter()
        adapter.uenv.agent.controller.controllers['base'].config.joint_names.reverse()
        with self.assertRaises(ProtocolError):
            FetchNavigationControl(adapter)

    def test_replan_materializes_current_capture_and_rejects_stale_camera(self):
        adapter = self.adapter()
        seen = []
        client = NS(reset=lambda:None, infer=lambda image, instruction:
                    seen.append(image.data) or NavigationPrediction(((1.,0.,0.),),False,True))
        skill = LightNavSkill(adapter,client,{'0':'Approach the counter.'})
        initial = Observation('f0',0,metadata={'subtask_index':0})
        skill.start(NS(skill='navigate',call_id='nav1'),initial)
        for step in (0,5):
            # The transition itself has no encoded images, just as in MS-HAB.
            transition_obs = Observation(f'f{step}',step)
            adapter.observe = lambda step=step: Observation(f'f{step}',step,
                images=(ImageFrame('fetch_head',bytes([step])),))
            skill.act(transition_obs)
        self.assertEqual(seen,[b'\x00',b'\x05'])
        adapter.observe = lambda: Observation('f5',5)
        with self.assertRaisesRegex(ProtocolError,'does not match'):
            skill.act(Observation('f10',10))
        self.assertEqual(len(seen),2)

    def test_index_routing_is_explicit_and_never_falls_back_after_failure(self):
        calls=[]
        def child(name):
            return NS(start=lambda *args:calls.append(name),act=lambda obs:name,
                      feedback=lambda *args:'failed')
        router=IndexedNavigationSkill(child('lightnav'),child('official'),[2],NS(emit=lambda *a,**k:None))
        router.start(NS(call_id='a'),Observation('f0',0,metadata={'subtask_index':0}))
        self.assertEqual(router.act(None),'official')
        router.start(NS(call_id='b'),Observation('f1',1,metadata={'subtask_index':2}))
        self.assertEqual(router.act(None),'lightnav')
        self.assertEqual(router.feedback(None,None),'failed')
        self.assertEqual(router.act(None),'lightnav')
        self.assertEqual(calls,['official','lightnav'])

    def test_model_stop_brakes_and_requires_environment_success(self):
        adapter=self.adapter();calls=[]
        client=NS(reset=lambda:None,infer=lambda *a:
                  calls.append('infer') or NavigationPrediction((),True,False))
        skill=LightNavSkill(adapter,client,{'0':'counter'},settle_steps=2)
        request=NS(skill='navigate',call_id='a',requirements=(Requirement('done','benchmark_success'),))
        obs=Observation('f0',0,images=(ImageFrame('fetch_head',b'x'),),metadata={'subtask_index':0})
        adapter.observe=lambda:obs;skill.start(request,obs)
        self.assertEqual(skill.act(obs)[-2:],(0.,0.))
        def transition(after):return NS(info={'adapter_subtask_before':0,'adapter_subtask_after':after},
            observation=Observation('f1',1),truncated=False)
        self.assertEqual(skill.feedback(request,transition(0)).status,SkillStatus.EXECUTING)
        self.assertEqual(skill.feedback(request,transition(1)).status,SkillStatus.SUCCEEDED)
        skill.act(Observation('f1',1))
        self.assertEqual(skill.feedback(request,transition(0)).status,SkillStatus.FAILED)
        self.assertEqual(calls,['infer'])

    def test_orientation_replan_is_bounded_and_does_not_claim_success(self):
        adapter=self.adapter();resets=[]
        client=NS(reset=lambda:resets.append(1),infer=lambda *a:NavigationPrediction((),True,True))
        skill=LightNavSkill(adapter,client,{'0':'chair'},settle_steps=1,
            recovery_instructions={'0':'Turn to face the chair.'},max_stop_replans=1)
        request=NS(skill='navigate',call_id='a',requirements=(Requirement('done','benchmark_success'),))
        obs=Observation('f0',0,images=(ImageFrame('fetch_head',b'x'),),metadata={'subtask_index':0})
        adapter.observe=lambda:obs;skill.start(request,obs);skill.act(obs)
        transition=NS(info={'adapter_subtask_before':0,'adapter_subtask_after':0,
            'navigated_close':[True],'oriented_correctly':[False]},observation=obs,truncated=False)
        self.assertEqual(skill.feedback(request,transition).status,SkillStatus.EXECUTING)
        self.assertTrue(skill.rotation_only)
        self.assertEqual(len(resets),2)
        skill.stopping=True;skill.stopped_steps=1
        self.assertEqual(skill.feedback(request,transition).status,SkillStatus.FAILED)
