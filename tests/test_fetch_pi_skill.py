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

    def test_navigation_displacement_uses_execution_duration(self):
        v,w=waypoint_velocity(((.15,0,.3),),.25)
        self.assertAlmostEqual(v,.6)
        self.assertAlmostEqual(w,1.2)
        self.assertEqual(waypoint_velocity(((0.,0.,0.),(.1,.1,0.)),.5),(.2,0.))


if __name__=='__main__': unittest.main()
