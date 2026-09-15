import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('resource_deadline',Path(__file__).parents[1]/'scripts/resource_deadline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class DeadlineTests(unittest.TestCase):
    def test_waits_then_stops_only_named_pod_and_verifies(self):
        now=[100.];calls=[]
        def api(pod,key,stop=False):
            calls.append((pod,stop,now[0]))
            return {'id':pod,'desiredStatus':'EXITED' if len(calls)>2 else 'RUNNING'}
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'log';ready=Path(d)/'ready'
            m.watch('abcdefgh',104.,'secret',log,ready,clock=lambda:now[0],
                    sleep=lambda t:now.__setitem__(0,now[0]+t),request=api)
            self.assertEqual(calls,[('abcdefgh',False,100.),('abcdefgh',True,104.),('abcdefgh',False,104.)])
            self.assertNotIn('secret',log.read_text())
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])['event'],'stop_verified')

    def test_identity_mismatch_never_arms_or_mutates(self):
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            ready=Path(d)/'ready'
            def api(*a,**kw):
                calls.append(kw);return {'id':'different'}
            with self.assertRaises(ValueError):
                m.watch('abcdefgh',101.,'secret',Path(d)/'log',ready,
                        clock=lambda:100.,request=api)
            self.assertFalse(ready.exists());self.assertEqual(calls,[{}])
