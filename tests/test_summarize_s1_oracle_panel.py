"""Behavior tests for ended-panel evidence, missing step0, and denominators."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import summarize_s1_oracle_panel as summary


def extra(tcp=0., obj=1., held=False):
    return dict(tcp_pose_wrt_base=[[tcp, 0, 0, 1, 0, 0, 0]],
                obj_pose_wrt_base=[[obj, 0, 0, 1, 0, 0, 0]], is_grasped=[held])


class SummaryTests(unittest.TestCase):
    def fixture(self, root, initial=True):
        directory = root / 'seed2024'; directory.mkdir()
        events = []
        for step in range(1, 6):
            raw = np.zeros((10, 13)); raw[0, 12] = 2
            events.append(dict(step=step, raw_actions=raw.tolist(), action=np.zeros(13).tolist(),
                extra=extra(step * .1, 1 + step * .02, held=step >= 4), info={'success': [False], 'robot_rest': [False]},
                instruction_family='reach' if step < 4 else 'grasp', prompt_switch=None))
        path = directory / 'events.jsonl'; path.write_text('\n'.join(json.dumps(e) for e in events))
        video = directory / 'exact-seed2024-failed.mp4'; video.write_bytes(b'fixture media bytes')
        result = dict(status='episode_completed', seed=2024, steps=5, success=False,
                      video=video.name, prompt_switches=[], artifact_sha256={'events.jsonl': summary.digest(path)})
        if initial: result['initial_extra'] = extra()
        result_path = directory / 'result.json'; result_path.write_text(json.dumps(result))
        case = dict(seed=2024, status='policy_failure', native_success=False, result_sha256=summary.digest(result_path))
        panel = dict(status='panel_finished', supervisor_outcome='completed', planned_count=3,
                     cases=[case, dict(seed=2025, status='infrastructure_failure'), dict(seed=2026, status='not_run')])
        (root / 'panel.json').write_text(json.dumps(panel))
        (root / 'launch.json').write_text(json.dumps(dict(variant='baseline', execution_seeds=[2024, 2025, 2026])))
        return result, events

    def test_partial_denominators_video_hash_and_initial_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            report = summary.summarize(root); case = report['cases'][0]
            self.assertEqual((report['planned_count'], report['completed_count'], report['policy_failures']), (3, 1, 1))
            self.assertEqual((report['infrastructure_failures'], report['not_run']), (1, 1))
            self.assertTrue(case['ever_grasped']); self.assertIsNone(case['final_native_info']['is_static'])
            self.assertEqual(case['clipping']['steps'], 5); self.assertEqual(case['clipping']['scalars'], 5)
            self.assertEqual(case['family_action_counts'], {'reach': 3, 'grasp': 2})
            self.assertAlmostEqual(case['distance_m']['first_post_action'], .92)
            self.assertAlmostEqual(case['displacements_from_step0']['5']['tcp_displacement_norm_m'], .5)
            self.assertAlmostEqual(case['displacements_from_step0']['5']['object_displacement_norm_m'], .1)
            self.assertAlmostEqual(case['displacements_from_step0']['5']['distance_change_m'], -.4)
            self.assertFalse(case['displacements_from_step0']['10']['available'])
            self.assertEqual(case['video']['path'], str((root / 'seed2024/exact-seed2024-failed.mp4').resolve()))
            self.assertEqual(case['video']['sha256'], summary.digest(case['video']['path']))

    def test_missing_initial_extra_never_substitutes_first_post_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root, initial=False)
            case = summary.summarize(root)['cases'][0]
            for window in case['displacements_from_step0'].values():
                self.assertFalse(window['available']); self.assertIsNone(window['distance_change_m'])
                self.assertIn('no_step0_reconstruction', window['reason'])

    def test_event_gap_and_hash_mutation_are_not_policy_failure(self):
        for mutation in ('gap', 'hash'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); result, events = self.fixture(root)
                path = root / 'seed2024/events.jsonl'
                events[1]['step'] = 99
                path.write_text('\n'.join(json.dumps(e) for e in events))
                if mutation == 'gap':
                    result['artifact_sha256']['events.jsonl'] = summary.digest(path)
                    result_path = root / 'seed2024/result.json'; result_path.write_text(json.dumps(result))
                    panel = summary.read(root / 'panel.json'); panel['cases'][0]['result_sha256'] = summary.digest(result_path)
                    (root / 'panel.json').write_text(json.dumps(panel))
                report = summary.summarize(root)
                self.assertEqual(report['evidence_invalid'], 1); self.assertEqual(report['policy_failures'], 0)

    def test_active_panel_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            panel = summary.read(root / 'panel.json'); panel.pop('supervisor_outcome')
            (root / 'panel.json').write_text(json.dumps(panel))
            with self.assertRaisesRegex(ValueError, 'has not ended'): summary.summarize(root)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            output = root / 'summary.json'; output.write_text('preserved')
            with self.assertRaises(SystemExit): summary.main(['--panel-dir', str(root), '--output', str(output)])
            self.assertEqual(output.read_text(), 'preserved')


if __name__ == '__main__': unittest.main()
