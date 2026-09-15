import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class TrainingExclusionTests(unittest.TestCase):
    def test_successful_teacher_chain_is_not_an_evaluation_result(self):
        path=Path(__file__).resolve().parents[1]/'scripts/audit_chain.py'
        spec=importlib.util.spec_from_file_location('chain_audit',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frames').mkdir()
            events=[{'event':'mshab_reset','auto_reset':False}]
            for i in range(4):
                events.append(dict(event='mshab_step',subtask_before=i,subtask_after=i+1,
                    controller_action=[[0.]*13],info={'elapsed_steps':[i+1],
                    'is_grasped':[i in (1,2)],'obj_at_goal':[i==3],'fail':[False]}))
            (root/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
            (root/'summary.json').write_text(json.dumps(dict(steps=4,benchmark_result=False,
                first_object_chain_success=True,skill_results=[{'skill':s,'feedback':{'status':'succeeded'}}
                    for s in ('navigate','pick','navigate','place')])))
            meta=root/'run-metadata.json';meta.write_text('{}')
            self.assertTrue(module.audit(root)['automated_evidence_pass'])
            meta.write_text(json.dumps({'training_collection':True}))
            result=module.audit(root)
            self.assertFalse(result['automated_evidence_pass'])
            self.assertFalse(result['checks']['not_training_collection'])


if __name__=='__main__':unittest.main()
