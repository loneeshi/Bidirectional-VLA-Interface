import io
import unittest
from types import SimpleNamespace as NS
from bvi.fetch_pi_skill import FetchPiSkill,JOINT_NAMES,CONVENTION
from bvi.protocol import ActionBounds,ProtocolError
try:
    import numpy as np
    from PIL import Image
except ImportError:
    np=None

@unittest.skipIf(np is None,'Optional image/array dependencies')
class ManipulationInstructionTests(unittest.TestCase):
    def make(self,instructions):
        buffer=io.BytesIO();Image.new('RGB',(4,4)).save(buffer,format='PNG')
        obs=NS(frame_id='s',metadata={'subtask_index':1},images=[NS(camera=k,data=buffer.getvalue()) for k in ('fetch_head','fetch_hand')])
        robot=NS(active_joints=[NS(name=x) for x in JOINT_NAMES],qpos=[[0.]*15])
        events=[];inputs=[]
        adapter=NS(uenv=NS(agent=NS(robot=robot)),original_plan=NS(subtasks=[NS(type='navigate'),NS(type='pick',uid='pick-1',obj_id='can-id')]),logger=NS(emit=lambda event,**kw:events.append((event,kw))),action_bounds=ActionBounds((-1.,)*13,(1.,)*13),observe=lambda:obs,save_observation_images=lambda:None)
        def infer(data,action_dim):inputs.append(data);return ((0.,)*13,)
        client=NS(metadata={'robot':'fetch','state_dim':15,'action_dim':13,'action_convention':CONVENTION,'state_conditioning':True},infer=infer)
        return FetchPiSkill('pick',adapter,client,instructions=instructions),obs,inputs,events

    def test_explicit_instruction_reaches_real_client_input_and_provenance(self):
        skill,obs,inputs,events=self.make({'1':'Pick up the blue can.'})
        skill.start(NS(skill='pick',call_id='p'),obs);skill.act(obs)
        self.assertEqual(inputs[0]['prompt'],'Pick up the blue can.')
        self.assertEqual(inputs[0]['observation/state'].shape,(15,))
        self.assertEqual(events[0][1]['instruction_source'],'explicit_scene_config')

    def test_missing_target_rejects_before_inference_and_default_is_preserved(self):
        skill,obs,inputs,_=self.make({'3':'Place the can.'})
        with self.assertRaisesRegex(ProtocolError,'Missing explicit'):
            skill.start(NS(skill='pick',call_id='p'),obs)
        self.assertEqual(inputs,[])
        skill,obs,inputs,events=self.make(None)
        skill.start(NS(skill='pick',call_id='p'),obs);skill.act(obs)
        self.assertEqual(inputs[0]['prompt'],'Pick and stably hold object can-id.')
        self.assertEqual(events[0][1]['instruction_source'],'task_plan_id_template')
