import unittest
from types import SimpleNamespace as NS
from bvi.recovery_collection import RecoveryCollectionSkill


class RecoveryCollectionTests(unittest.TestCase):
    def test_records_only_expert_tail_and_marks_takeover(self):
        calls=[];events=[]
        def policy(name,value):
            return NS(name='pick',start=lambda r,o:calls.append((name,'start',o.frame_id)),
                      act=lambda o:(calls.append((name,'act',o.frame_id)) or (value,)),feedback=lambda *a:'feedback')
        adapter=NS(record_demonstrations=True,logger=NS(emit=lambda event,**kw:events.append((event,kw))))
        skill=RecoveryCollectionSkill(policy('learner',1),policy('teacher',2),adapter,2)
        request=NS(call_id='collect');skill.start(request,NS(frame_id='f0'))
        for i in range(4):
            self.assertEqual(skill.act(NS(frame_id=f'f{i}')),(1 if i<2 else 2,))
            self.assertEqual(adapter.record_demonstrations,i>=2)
        self.assertEqual([x for x in calls if x[1]=='start'],[('learner','start','f0'),('teacher','start','f2')])
        self.assertEqual(adapter.demonstration_source,'official_sac_recovery')
        takeover=[e for e in events if e[0]=='recovery_teacher_takeover']
        self.assertEqual(len(takeover),1)
        self.assertFalse(takeover[0][1]['evaluation_eligible'])
        skill.start(request,NS(frame_id='new'))
        self.assertFalse(adapter.record_demonstrations)
