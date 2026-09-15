import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np=None


@unittest.skipIf(np is None,'Optional array/image dependencies')
class RelativeFetchStateTests(unittest.TestCase):
    def test_recovery_keeps_original_skill_origin_not_teacher_takeover(self):
        path=Path(__file__).resolve().parents[1]/'scripts/fetch_openpi.py'
        spec=importlib.util.spec_from_file_location('fetch_pipeline',path)
        pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();recovery=root/'recovery';recovery.mkdir()
            events=[{'event':'fetch_pi_started','skill':'pick','call_id':'p','model_metadata':{}},
                    {'event':'fetch_pi_inference_started','call_id':'p','state':[10.,20.]+[0.]*13}]
            data='\n'.join(map(json.dumps,events)).encode();(source/'events.jsonl').write_bytes(data)
            meta={'mixed_teacher_collection':True,'source_run':str(source),
                  'source_events_sha256':hashlib.sha256(data).hexdigest()}
            (recovery/'run-metadata.json').write_text(json.dumps(meta))
            rows=[{'skill':'pick','qpos':[[10.2,20.3]+[0.]*13]}]
            origin,label=pipeline.segment_base_origin(recovery,[],rows)
            np.testing.assert_array_equal(origin,[10.,20.])
            self.assertEqual(label,'original_learner_world_state')
            (source/'events.jsonl').write_bytes(data+b' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                pipeline.segment_base_origin(recovery,[],rows)

    def test_wire_state_is_translation_invariant_and_preserves_other_joints(self):
        from bvi.fetch_pi_skill import FetchPiSkill,JOINT_NAMES,CONVENTION
        from bvi.protocol import ActionBounds
        buffer=io.BytesIO();Image.new('RGB',(4,4)).save(buffer,format='PNG')
        obs=NS(frame_id='s',images=[NS(camera=c,data=buffer.getvalue()) for c in ('fetch_head','fetch_hand')])
        states=[]
        for offset in [0.,100.]:
            qpos=[10.25+offset,20.5+offset]+list(range(2,15));qvel=list(range(100,115))
            robot=NS(active_joints=[NS(name=x) for x in JOINT_NAMES],qpos=[qpos],qvel=[qvel])
            adapter=NS(uenv=NS(agent=NS(robot=robot)),logger=NS(emit=lambda *a,**k:None),
                       action_bounds=ActionBounds((-1.,)*13,(1.,)*13),observe=lambda:obs,save_observation_images=lambda:None)
            def infer(data,action_dim):
                states.append(data['observation/state'].copy());return ((0.,)*13,)
            client=NS(metadata={'robot':'fetch','state_dim':30,'state_components':['qpos','qvel'],
                'action_dim':13,'action_convention':CONVENTION,'state_conditioning':True,
                'base_position_reference':'skill_start_xy'},infer=infer)
            skill=FetchPiSkill('pick',adapter,client);skill.index=1;skill.call_id='p';skill.prompt='Pick'
            skill.base_xy_origin=(10.+offset,20.+offset);skill.act(obs)
        np.testing.assert_array_equal(states[0],states[1])
        np.testing.assert_array_equal(states[0], [.25,.5]+list(range(2,15))+list(range(100,115)))

    def test_ensemble_counts_every_draw_and_applies_mean(self):
        from bvi.fetch_pi_skill import FetchPiSkill,JOINT_NAMES,CONVENTION
        from bvi.protocol import ActionBounds,ProtocolError
        buffer=io.BytesIO();Image.new('RGB',(4,4)).save(buffer,format='PNG')
        obs=NS(frame_id='s',images=[NS(camera=c,data=buffer.getvalue()) for c in ('fetch_head','fetch_hand')])
        robot=NS(active_joints=[NS(name=x) for x in JOINT_NAMES],qpos=[[0.]*15])
        adapter=NS(uenv=NS(agent=NS(robot=robot)),logger=NS(emit=lambda *a,**k:None),
            action_bounds=ActionBounds((-1.,)*13,(1.,)*13),observe=lambda:obs,save_observation_images=lambda:None)
        draws=[]
        def infer(data,action_dim):
            draws.append(data);return (([1.,-1.,.2,.2][len(draws)-1],)+(0.,)*12,)
        client=NS(metadata={'robot':'fetch','state_dim':15,'action_dim':13,
            'action_convention':CONVENTION,'state_conditioning':True},infer=infer)
        skill=FetchPiSkill('pick',adapter,client,max_predictions=4,chunk_steps=1,ensemble_samples=4)
        skill.index=1;skill.call_id='p';skill.prompt='Pick'
        self.assertAlmostEqual(skill.act(obs)[0],.1)
        self.assertEqual(skill.total_predictions,4)
        self.assertEqual(len(draws),4)
        self.assertTrue(all(d is draws[0] for d in draws))
        with self.assertRaisesRegex(ProtocolError,'cap reached'):skill.act(obs)
        self.assertEqual(len(draws),4)


if __name__=='__main__':unittest.main()
