import unittest
from bvi.progress_monitor import ProgressMonitor
class ProgressTest(unittest.TestCase):
    def test_threshold_requires_two_predictions(self):
        m=ProgressMonitor('reach',0)
        self.assertIsNone(m.update(.95));self.assertEqual(m.update(.96),'learned_threshold')
    def test_stagnation(self):
        m=ProgressMonitor('move',1)
        for _ in range(9):self.assertIsNone(m.update(.2))
        self.assertEqual(m.update(.2),'learned_stagnation')
    def test_drop_and_cooldown(self):
        m=ProgressMonitor('grasp',1)
        for x in [.2,.3,.4]:self.assertIsNone(m.update(x))
        self.assertEqual(m.update(.2),'learned_drop')
        m=ProgressMonitor('grasp',1,15)
        for x in [.2,.3,.4,.2]:self.assertIsNone(m.update(x))
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):ProgressMonitor('reach',0).update(float('nan'))
