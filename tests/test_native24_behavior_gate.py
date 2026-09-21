from copy import deepcopy

import numpy as np
import pytest

from bvi.native24_behavior_gate import (
    CLASSIFICATIONS,
    COMBINATIONS,
    SCHEMA,
    adjudicate_behavior_rows,
    aggregate_gate_status,
    combination_specs,
    progress_scalar,
    rng_contract,
    roster_seeds,
    run_monitor_sequence,
)


SEQUENCES = (
    ("far", "far_grasp_diagnostic", "grasp"),
    ("near", "near_grasp_candidate_not_completion", "grasp"),
    ("unheld", "unheld_move", "move"),
    ("held", "held_move", "move"),
)


def _chunks(values, *, dtype=np.float64, tail=0.0):
    chunks = []
    for value in values:
        chunk = np.full((1, 10), tail, dtype=dtype)
        chunk[0, 0] = value
        chunks.append(chunk)
    return chunks


def _matrix(*, false_complete=None):
    specs = {row["name"]: row for row in combination_specs()}
    rows, queue_events = [], []
    expected_ids = [row[0] for row in SEQUENCES]
    for combination in COMBINATIONS:
        for sequence_index, (sequence_id, classification, family) in enumerate(SEQUENCES):
            values = np.linspace(0.1, 0.4, 10)
            if false_complete == (combination, sequence_id):
                values = np.array([0.95, 0.95, *np.linspace(0.2, 0.4, 8)])
            trace = run_monitor_sequence(family, 1, _chunks(values))
            rows.append({
                "combination": combination,
                "sequence_id": sequence_id,
                "sequence_index": sequence_index,
                "classification": classification,
                "family": family,
                "head_step": specs[combination]["head_step"],
                "bank_step": specs[combination]["bank_steps"][family],
                "monitor": trace,
                "physical_completion": [False] * trace["prediction_count_input"],
                "rng_seeds": [
                    roster_seeds(sequence_index, frame_index)
                    for frame_index in range(trace["prediction_count_input"])
                ],
            })
            for prediction in trace["predictions"]:
                queue_events.append({
                    "combination": combination,
                    "sequence_id": sequence_id,
                    "prediction_index": prediction["prediction_index"],
                    "reason": prediction["queue_clear_reason"],
                    "discarded_actions": prediction["expected_discarded_actions"],
                    "queue_empty": True,
                })
    return rows, queue_events, expected_ids


def test_combination_routes_include_selective_move_only_bank():
    specs = {row["name"]: row for row in combination_specs()}
    assert tuple(specs) == COMBINATIONS
    assert specs["H0_B0"]["head_step"] == 0
    assert set(specs["H0_B20"]["bank_steps"].values()) == {20}
    assert specs["H20_BSEL_MOVE20"]["bank_steps"] == {
        "reach": 0, "grasp": 0, "move": 20, "release": 0,
    }
    assert all(row["training_updates_during_gate"] == 0 for row in specs.values())
    specs["H0_B0"]["bank_steps"]["move"] = 20
    assert combination_specs()[0]["bank_steps"]["move"] == 0


def test_fixed_rng_formula_identity_and_input_bounds():
    first = roster_seeds(2, 7)
    assert first["sample_rng_seed"] == 702_130
    assert first["noise_rng_seed"] == 902_130
    assert first == roster_seeds(2, 7)
    assert first["rng_identity"] != roster_seeds(2, 8)["rng_identity"]
    assert rng_contract()["shared_across_combinations"] is True
    for sequence_index, frame_index in ((True, 0), (0, True), (-1, 0), (0, -1), (0, 1000)):
        with pytest.raises(ValueError):
            roster_seeds(sequence_index, frame_index)


def test_progress_requires_exact_finite_real_shape_and_preserves_float32_value():
    value, dtype = progress_scalar(np.full((1, 10), np.float32(0.9), np.float32))
    assert dtype == "float32"
    assert value == float(np.float32(0.9))
    assert value < 0.9
    for invalid in (
        np.zeros(10),
        np.zeros((1, 9)),
        np.zeros((2, 10)),
        np.full((1, 10), np.nan),
        np.array([[0.0] * 9 + [np.inf]]),
        np.zeros((1, 10), dtype=bool),
        np.zeros((1, 10), dtype=np.complex64),
    ):
        with pytest.raises(ValueError):
            progress_scalar(invalid)


def test_float32_threshold_boundaries_are_not_rounded():
    reach = run_monitor_sequence("reach", 0, _chunks([0.9, 0.9], dtype=np.float32))
    assert reach["terminal_event"] is None
    assert [row["value"] for row in reach["predictions"]] == [float(np.float32(0.9))] * 2

    grasp = run_monitor_sequence("grasp", 0, _chunks([0.6, 0.6], dtype=np.float32))
    assert float(np.float32(0.6)) > 0.6
    assert grasp["terminal_event"] == "learned_threshold"
    assert grasp["prediction_count_consumed"] == 2


def test_monitor_uses_only_prefix_zero_and_requires_two_consecutive_hits():
    base = _chunks([0.95, 0.89, 0.95, 0.95], tail=0.0)
    changed_tail = _chunks([0.95, 0.89, 0.95, 0.95], tail=123.0)
    left = run_monitor_sequence("reach", 0, base)
    right = run_monitor_sequence("reach", 0, changed_tail)
    assert left["terminal_event"] == right["terminal_event"] == "learned_threshold"
    assert left["prediction_count_consumed"] == right["prediction_count_consumed"] == 4
    assert [row["value"] for row in left["predictions"]] == [
        row["value"] for row in right["predictions"]
    ]


def test_drop_is_strictly_greater_than_point_zero_three():
    exact = run_monitor_sequence("move", 1, _chunks([0.0, 0.03, 0.02, 0.0]))
    assert exact["terminal_event"] is None
    greater = np.nextafter(np.float64(0.03), np.float64(np.inf))
    dropped = run_monitor_sequence("move", 1, _chunks([0.0, greater, 0.02, 0.0]))
    assert dropped["terminal_event"] == "learned_drop"
    assert dropped["prediction_count_consumed"] == 4


def test_stagnation_is_strictly_less_than_point_zero_three():
    exact = run_monitor_sequence("move", 1, _chunks([0.0] * 9 + [0.03]))
    assert exact["terminal_event"] is None
    less = np.nextafter(np.float64(0.03), np.float64(0.0))
    stagnant = run_monitor_sequence("move", 1, _chunks([0.0] * 9 + [less]))
    assert stagnant["terminal_event"] == "learned_stagnation"
    assert stagnant["prediction_count_consumed"] == 10


def test_first_event_stops_runtime_monitor_but_keeps_diagnostic_suffix():
    trace = run_monitor_sequence("grasp", 1, _chunks([0.7, 0.7, 0.1, 0.9]))
    assert trace["terminal_event"] == "learned_threshold"
    assert trace["prediction_count_consumed"] == 2
    assert trace["post_terminal_diagnostic_count"] == 2
    assert [row["runtime_disposition"] for row in trace["predictions"]] == [
        "runtime_monitor_input",
        "runtime_monitor_input",
        "post_terminal_diagnostic_only",
        "post_terminal_diagnostic_only",
    ]
    assert trace["predictions"][1]["queue_clear_reason"] == "learned_threshold"
    assert trace["predictions"][2]["queue_clear_reason"] == "post_terminal_diagnostic_only"


def test_monitor_entry_rejects_bool_unknown_negative_and_bad_first_value():
    valid = _chunks([0.2])
    for family, invocation, cooldown in (
        (True, 0, 0),
        ("unknown", 0, 0),
        ("reach", True, 0),
        ("reach", -1, 0),
        ("reach", 0, True),
        ("reach", 0, -1),
    ):
        with pytest.raises(ValueError):
            run_monitor_sequence(family, invocation, valid, cooldown)
    with pytest.raises(ValueError, match="Invalid learned progress"):
        run_monitor_sequence("reach", 0, _chunks([1.1]))
    with pytest.raises(ValueError, match="finite"):
        run_monitor_sequence("reach", 0, [
            np.zeros((1, 10)), np.array([[0.0] * 9 + [np.nan]])
        ])


def test_complete_matrix_passes_offline_but_full_gate_remains_not_evaluable():
    rows, queues, sequence_ids = _matrix()
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["schema"] == SCHEMA
    assert report["expected_row_count"] == len(COMBINATIONS) * len(sequence_ids)
    assert report["offline_behavior_status"] == "passed"
    assert report["full_gate_status"] == report["status"] == "not_evaluable"
    assert report["full_behavior_admission"] is False
    assert report["native_success_evaluated"] is False
    assert report["deployment_cadence_evaluated"] is False
    assert report["checks"]["routing"]["status"] == "passed"
    assert report["checks"]["queue_clear"]["status"] == "passed"
    assert report["checks"]["negative_false_completion"]["status"] == "passed"
    for name in ("physical_stagnation", "physical_rollback", "deployment_cadence", "native_success"):
        assert report["checks"][name]["status"] == "not_evaluable"


def test_selective_routing_queue_clear_and_negative_false_completion_failures():
    rows, queues, sequence_ids = _matrix(false_complete=("H20_BSEL_MOVE20", "unheld"))
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["offline_behavior_status"] == "failed"
    assert report["checks"]["negative_false_completion"]["status"] == "failed"
    # Full scope prioritizes its known not-evaluable dimensions over a validly
    # measured offline failure, while preserving the offline failure above.
    assert report["full_gate_status"] == "not_evaluable"

    rows, queues, sequence_ids = _matrix()
    selective = next(
        row for row in rows
        if row["combination"] == "H20_BSEL_MOVE20" and row["sequence_id"] == "unheld"
    )
    selective["bank_step"] = 0
    queues.pop()
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["routing"]["status"] == "failed"
    assert report["checks"]["queue_clear"]["status"] == "failed"
    assert report["offline_behavior_status"] == "failed"


def test_false_completion_uses_event_frame_physics_even_for_held_move():
    rows, queues, sequence_ids = _matrix(false_complete=("H20_B20", "held"))
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["negative_false_completion"]["status"] == "failed"
    assert "before physical completion" in report["checks"]["negative_false_completion"][
        "reasons"
    ][0]

    target = next(
        row for row in rows
        if row["combination"] == "H20_B20" and row["sequence_id"] == "held"
    )
    target["physical_completion"][1] = True
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["negative_false_completion"]["status"] == "passed"


def test_post_terminal_diagnostic_predictions_still_clear_all_ten_actions():
    rows, queues, sequence_ids = _matrix(false_complete=("H0_B0", "unheld"))
    event = next(
        row for row in queues
        if row["combination"] == "H0_B0"
        and row["sequence_id"] == "unheld"
        and row["prediction_index"] == 5
    )
    assert event["reason"] == "post_terminal_diagnostic_only"
    event["discarded_actions"] = 9
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["queue_clear"]["status"] == "failed"


def test_single_high_negative_value_is_not_false_completion():
    rows, queues, sequence_ids = _matrix()
    target = next(
        row for row in rows
        if row["combination"] == "H0_B0" and row["sequence_id"] == "unheld"
    )
    old_count = target["monitor"]["prediction_count_input"]
    target["monitor"] = run_monitor_sequence(
        "move", 1, _chunks([0.95, 0.85, 0.86, 0.87, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89])
    )
    target["physical_completion"] = [False] * 10
    target["rng_seeds"] = [
        roster_seeds(target["sequence_index"], frame_index) for frame_index in range(10)
    ]
    queues[:] = [
        event for event in queues
        if not (event["combination"] == "H0_B0" and event["sequence_id"] == "unheld")
    ]
    assert old_count == 10
    for prediction in target["monitor"]["predictions"]:
        queues.append({
            "combination": "H0_B0",
            "sequence_id": "unheld",
            "prediction_index": prediction["prediction_index"],
            "reason": prediction["queue_clear_reason"],
            "discarded_actions": 10,
            "queue_empty": True,
        })
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert target["monitor"]["terminal_event"] == "learned_drop"
    assert report["checks"]["negative_false_completion"]["status"] == "passed"
    assert report["offline_behavior_status"] == "passed"


def test_adjudication_rejects_a_short_sequence_even_when_two_hit_cannot_fire():
    rows, queues, sequence_ids = _matrix()
    target = rows[0]
    target["monitor"] = run_monitor_sequence(target["family"], 1, _chunks([0.1]))
    target["physical_completion"] = [False]
    target["rng_seeds"] = [roster_seeds(target["sequence_index"], 0)]
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["integrity"]["status"] == "invalid"
    assert "must have 10 predictions" in " ".join(report["checks"]["integrity"]["reasons"])


def test_invalid_rng_or_trace_wins_status_precedence():
    rows, queues, sequence_ids = _matrix()
    rows[0]["rng_seeds"][0]["sample_rng_seed"] += 1
    rows[1]["monitor"]["predictions"][0]["above_after"] = 99
    queues.pop()
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["integrity"]["status"] == "invalid"
    assert report["checks"]["rng_identity"]["status"] == "invalid"
    assert report["checks"]["queue_clear"]["status"] == "failed"
    assert report["offline_behavior_status"] == report["full_gate_status"] == "invalid"


def test_missing_or_misaligned_physical_completion_is_invalid():
    rows, queues, sequence_ids = _matrix()
    rows[0].pop("physical_completion")
    rows[1]["physical_completion"] = [False] * 9
    rows[2]["physical_completion"][0] = 0
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["integrity"]["status"] == "invalid"
    assert report["full_gate_status"] == "invalid"


def test_missing_row_is_not_evaluable_before_measured_failures():
    rows, queues, sequence_ids = _matrix()
    missing = rows.pop()
    queues[:] = [
        event for event in queues
        if not (
            event["combination"] == missing["combination"]
            and event["sequence_id"] == missing["sequence_id"]
        )
    ]
    rows[0]["bank_step"] = 999
    report = adjudicate_behavior_rows(rows, queues, sequence_ids)
    assert report["checks"]["coverage"]["status"] == "not_evaluable"
    assert report["checks"]["routing"]["status"] == "failed"
    assert report["offline_behavior_status"] == "not_evaluable"


def test_status_aggregation_priority_is_explicit():
    assert aggregate_gate_status(["passed", "failed"]) == "failed"
    assert aggregate_gate_status(["failed", "not_evaluable"]) == "not_evaluable"
    assert aggregate_gate_status(["not_evaluable", "invalid", "failed"]) == "invalid"
    with pytest.raises(ValueError):
        aggregate_gate_status([])


def test_classification_roster_is_fixed():
    assert CLASSIFICATIONS == (
        "far_grasp_diagnostic",
        "near_grasp_candidate_not_completion",
        "unheld_move",
        "held_move",
    )
    rows, queues, sequence_ids = _matrix()
    changed = deepcopy(rows)
    for row in changed:
        if row["sequence_id"] == "held":
            row["classification"] = "unheld_move"
    report = adjudicate_behavior_rows(changed, queues, sequence_ids)
    assert report["checks"]["coverage"]["status"] == "not_evaluable"
