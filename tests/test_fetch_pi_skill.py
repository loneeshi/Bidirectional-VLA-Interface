import unittest
from types import SimpleNamespace as NS
from bvi.fetch_pi_skill import FetchPiSkill,JOINT_NAMES,CONVENTION
from bvi.protocol import ProtocolError,ActionBounds
from bvi.lightnav_skill import waypoint_velocity


class FetchContractTests(unittest.TestCase):
    def fixture(self):
        adapter=NS(uenv=NS(agent=NS(robot=NS(active_joints=[NS(name=x) for x in JOINT_NAMES]))),
            logger=NS(emit=lambda *a,**k:None),action_bounds=ActionBounds((-1.,)*13,(1.,)*13))
        client=NS(metadata={'robot':'fetch','state_dim':15,'action_dim':13,'action_convention':CONVENTION,
                            'state_conditioning':True})
        return adapter,client

    def test_rejects_droid_and_wrong_joint_order(self):
        adapter,client=self.fixture()
        client.metadata['robot']='panda'
        with self.assertRaises(ProtocolError): FetchPiSkill('pick',adapter,client)
        client.metadata['robot']='fetch'
        client.metadata['state_conditioning']=False
        with self.assertRaises(ProtocolError): FetchPiSkill('pick',adapter,client)
        client.metadata['state_conditioning']=True
        adapter.uenv.agent.robot.active_joints.reverse()
        with self.assertRaises(ProtocolError): FetchPiSkill('pick',adapter,client)

    def test_cached_actions_are_bounded_without_hidden_policy_call(self):
        adapter,client=self.fixture()
        skill=FetchPiSkill('pick',adapter,client)
        skill.index=1
        skill.call_id='test'
        skill.actions.append((2.,-2.)+(0.,)*11)
        self.assertEqual(skill.act(None),(1.,-1.)+(0.,)*11)

    def test_velocity_checkpoint_requires_explicit_state_contract(self):
        adapter,client=self.fixture();client.metadata['state_dim']=30
        with self.assertRaises(ProtocolError):FetchPiSkill('pick',adapter,client)
        client.metadata['state_components']=['qpos','qvel']
        self.assertEqual(FetchPiSkill('pick',adapter,client).state_dim,30)
        client.metadata['state_components']=['qvel','qpos']
        with self.assertRaises(ProtocolError):FetchPiSkill('pick',adapter,client)

    def test_velocity_values_follow_positions_in_wire_observation(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest('Optional image/array dependencies')
        import io
        adapter,client=self.fixture()
        client.metadata.update(state_dim=30,state_components=['qpos','qvel'])
        robot=adapter.uenv.agent.robot
        robot.qpos=[list(range(15))];robot.qvel=[list(range(100,115))]
        buffer=io.BytesIO();Image.new('RGB',(4,4)).save(buffer,format='PNG')
        observation=NS(frame_id='test-0',images=[NS(camera=c,data=buffer.getvalue()) for c in ('fetch_head','fetch_hand')])
        adapter.observe=lambda:observation;adapter.save_observation_images=lambda:None
        sent=[]
        def infer(data,action_dim):
            sent.append(data);self.assertEqual(action_dim,13);return ((0.,)*13,)
        client.infer=infer
        skill=FetchPiSkill('pick',adapter,client);skill.index=1;skill.call_id='test';skill.prompt='Pick.'
        self.assertEqual(skill.act(observation),(0.,)*13)
        np.testing.assert_array_equal(sent[0]['observation/state'],list(range(15))+list(range(100,115)))

    def test_navigation_displacement_uses_execution_duration(self):
        v,w=waypoint_velocity(((.15,0,.3),),.25)
        self.assertAlmostEqual(v,.6)
        self.assertAlmostEqual(w,1.2)
        self.assertEqual(waypoint_velocity(((0.,0.,0.),(.1,.1,0.)),.5),(.2,0.))

    def test_native24_uses_environment_observation_not_full_robot_state(self):
        import io
        import numpy as np
        from PIL import Image
        adapter,client=self.fixture()
        client.metadata.update(state_dim=24,state_components=['native_qpos12','native_qvel12'])
        with self.assertRaises(ProtocolError):FetchPiSkill('pick',adapter,client)
        client.metadata['state_source']='env_native_agent'
        # Robot has no qpos/qvel here: reading the legacy path must fail.
        adapter.uenv._get_obs_agent=lambda:{'qpos':[list(range(12))],'qvel':[list(range(50,62))]}
        buf=io.BytesIO();Image.new('RGB',(128,128)).save(buf,format='PNG')
        obs=NS(frame_id='native-0',images=[NS(camera=c,data=buf.getvalue()) for c in ('fetch_head','fetch_hand')])
        adapter.observe=lambda:obs;adapter.save_observation_images=lambda:None
        sent=[]
        client.infer=lambda data,action_dim:(sent.append(data) or ((0.,)*13,))
        skill=FetchPiSkill('pick',adapter,client);skill.index=1;skill.call_id='native';skill.prompt='Grasp the apple.'
        skill.act(obs)
        np.testing.assert_array_equal(sent[0]['observation/state'],list(range(12))+list(range(50,62)))
        self.assertEqual(sent[0]['prompt'],'Grasp the apple.')
        client.metadata['base_position_reference']='skill_start_xy'
        with self.assertRaises(ProtocolError):FetchPiSkill('pick',adapter,client)


if __name__=='__main__': unittest.main()
