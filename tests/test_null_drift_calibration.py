import copy

import pytest

from bvi.null_drift_calibration import (
    CALIBRATION_ACTIONS,
    RULE_VERSION,
    generate_null_drift_thresholds,
    validate_recorded_actions,
)


def source_rows():
    return [
        {"step": step, "action": [[0.1] * 8 + [0.0, 0.0] + [0.1] * 3]}
        for step in range(1, CALIBRATION_ACTIONS + 1)
    ]


def repeat(actions, offset0, offset1):
    return [
        {
            "step": step,
            "action": list(actions[step - 1]),
            "qpos": [offset0 + step * 1e-6, offset1 - step * 1e-6],
            "tcp_target_distance_m": 1.0 - step * 0.01 + offset0,
            "robot_cumulative_force": step + offset1,
        }
        for step in range(1, CALIBRATION_ACTIONS + 1)
    ]


def test_v1_uses_three_repeat_pairwise_envelope_and_unit_floor():
    actions = validate_recorded_actions(source_rows())
    repeats = [
        repeat(actions, 0.0, 0.0),
        repeat(actions, 2e-4, 1e-5),
        repeat(actions, 4e-4, 3e-5),
    ]
    result = generate_null_drift_thresholds(
        repeats,
        actions,
        channel_names=("linear", "angular"),
        units={"linear": "m", "angular": "rad"},
    )
    assert result["generation_rule"]["version"] == RULE_VERSION
    assert result["generation_rule"]["repeat_pair_count"] == 3
    assert result["qpos_null_max_pairwise_abs"]["linear"] == pytest.approx(4e-4)
    assert result["qpos_abs_error"]["linear"] == pytest.approx(8e-4)
    assert result["qpos_abs_error"]["angular"] == pytest.approx(1e-4)
    assert result["tcp_target_distance_null_max_pairwise_abs_m"] == pytest.approx(4e-4)
    assert result["consumes_policy_under_test"] is False


def test_fewer_than_three_repeats_fail_closed():
    actions = validate_recorded_actions(source_rows())
    rows = repeat(actions, 0.0, 0.0)
    with pytest.raises(ValueError, match="At least 3"):
        generate_null_drift_thresholds(
            [rows, rows],
            actions,
            channel_names=("linear", "angular"),
            units={"linear": "m", "angular": "rad"},
        )


def test_changed_action_in_a_repeat_fails_closed():
    actions = validate_recorded_actions(source_rows())
    repeats = [repeat(actions, 0, 0) for _ in range(3)]
    repeats[2][10]["action"][0] += 0.01
    with pytest.raises(ValueError, match="differs from frozen SAC source"):
        generate_null_drift_thresholds(
            repeats,
            actions,
            channel_names=("linear", "angular"),
            units={"linear": "m", "angular": "rad"},
        )


def test_incomplete_repeat_fails_closed():
    actions = validate_recorded_actions(source_rows())
    repeats = [repeat(actions, 0, 0) for _ in range(3)]
    repeats[1].pop()
    with pytest.raises(ValueError, match="exactly 35"):
        generate_null_drift_thresholds(
            repeats,
            actions,
            channel_names=("linear", "angular"),
            units={"linear": "m", "angular": "rad"},
        )


def test_source_action_contract_rejects_head_motion_and_wrong_horizon():
    rows = source_rows()
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["action"][0][8] = 0.1
    with pytest.raises(ValueError, match="stationary-head"):
        validate_recorded_actions(rows)
    with pytest.raises(ValueError, match="only 34"):
        validate_recorded_actions(source_rows()[:-1])
