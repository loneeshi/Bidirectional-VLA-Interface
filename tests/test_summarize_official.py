"""Diagnostic trace summaries must not turn reset artifacts into success."""
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("summarize_official",
    Path(__file__).resolve().parents[1] / "scripts/summarize_official.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def reset():
    return {"event": "reset", "seed": 0, "build_config_idxs": [69], "task_plan_idxs": [23],
            "task_plan": [{"type": x} for x in ("navigate", "pick", "navigate", "place", "navigate")]}


def step(number, before, after, **extra):
    return {"event": "step", "step": number, "subtask_before": [before], "subtask_after": [after],
            "raw_action": [[0.0] * 13], "raw_outside_unit_box": False,
            "terminated": [False], "truncated": [False],
            "info": {"fail": [False], "success": [False]}, **extra}


class SummaryTest(unittest.TestCase):
    def test_continuous_chain_is_evidence_but_not_an_unqualified_gate_pass(self):
        records = [reset(), step(0, 0, 0), step(1, 0, 1), step(2, 1, 2),
                   step(3, 2, 3), step(4, 3, 4)]
        value = summary.summarize(records)
        self.assertTrue(value["g2"]["navigate_pick_place_each_observed"])
        self.assertEqual(value["g3"]["chain_count"], 1)
        self.assertIsNone(value["g3"]["passed"])
        self.assertEqual([e["step"] for e in value["chain_evidence"][0]["transitions"]], [1, 2, 3, 4])
        attested = summary.summarize(records, controller_provenance="official_learned_navigation")
        self.assertTrue(attested["g3"]["passed"])
        self.assertEqual(attested["controller_provenance"]["source"], "explicit_operator_attestation")
        self.assertFalse(attested["controller_provenance"]["inferred_from_trace"])

    def test_autoreset_final_success_does_not_create_place_completion(self):
        final = {"fail": [False], "success": [True], "subtask": [4]}
        records = [reset(), step(0, 0, 1), step(1, 1, 2), step(2, 2, 3),
                   step(3, 3, 0, truncated=[True],
                        info={"fail": [False], "success": [False], "final_info": final, "_final_info": [True]})]
        value = summary.summarize(records)
        self.assertTrue(value["fulltask_success_once_observed"])
        self.assertEqual(value["success_evidence"][0]["source"], "final_info")
        self.assertFalse(value["g3"]["continuous_chain_observed"])
        self.assertNotIn("place", value["g2"]["successful_transition_counts"])

    def test_gap_reset_failure_and_missing_flags_break_the_chain(self):
        variants = [
            [reset(), step(0, 0, 1), step(2, 1, 2), step(3, 2, 3), step(4, 3, 4)],
            [reset(), step(0, 0, 1), step(1, 1, 2), reset(), step(2, 2, 3), step(3, 3, 4)],
            [reset(), step(0, 0, 1), step(1, 1, 2, info={"fail": [True], "success": [False]}),
             step(2, 2, 3), step(3, 3, 4)],
            [reset(), step(0, 0, 1), step(1, 1, 2, info={"success": [False]}),
             step(2, 2, 3), step(3, 3, 4)],
        ]
        for records in variants:
            with self.subTest(records=records):
                self.assertFalse(summary.summarize(records)["g3"]["continuous_chain_observed"])

    def test_actions_are_recomputed_and_nonfinite_values_are_serializable(self):
        raw = [2.0] + [0.0] * 12
        value = summary.summarize([reset(), step(0, 0, 1, raw_action=[raw], raw_outside_unit_box=True),
                                   step(1, 1, 2, raw_action=[[float("nan")] + [0.0] * 12])])
        self.assertEqual(value["action_counts"]["raw_outside_unit_box_steps"], 1)
        self.assertEqual(value["action_counts"]["nonfinite_steps"], 1)
        self.assertEqual(len(value["skill_completion_evidence"]), 1)
        json.dumps(summary.json_safe(value), allow_nan=False)

    def test_masked_final_info_and_missing_stages_remain_unknown(self):
        records = [dict(reset(), task_plan=[]), step(0, 0, 1),
                   step(1, 1, 0, truncated=[True], info={"final_info": {"success": [True]}, "_final_info": [False]})]
        value = summary.summarize(records)
        self.assertIsNone(value["fulltask_success_once_observed"])
        self.assertFalse(value["g3"]["continuous_chain_observed"])
        self.assertEqual(value["g2"]["successful_transition_counts"], {})

    def test_compact_output_keeps_first_failure_and_completion_locations(self):
        value = summary.summarize([reset(), step(0, 0, 1),
            step(1, 1, 1, info={"fail": [True], "success": [False],
                               "cumulative_force_within_limit": [False], "robot_cumulative_force": [5100]})])
        compact = summary.compact_stdout(value)
        self.assertEqual(compact["first_failure"]["step"], 1)
        self.assertEqual(compact["first_failure"]["robot_cumulative_force"], [5100])
        self.assertEqual(compact["completion_evidence"][0]["step"], 0)
        self.assertEqual(compact["failure_step_count"], 1)
        self.assertNotIn("failures", compact)


if __name__ == "__main__":
    unittest.main()
