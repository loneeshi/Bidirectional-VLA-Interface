import pytest

from bvi.closed_loop_divergence import (
    ClosedLoopDivergenceRecorder,
    analyze_closed_loop_divergence,
    strict_pairing,
)


SHA = "a" * 64


def identity(sha=SHA, restore=1e-7):
    return {
        "task": "set_table/pick/013_apple",
        "seed": 2025,
        "reference_state_sha256": sha,
        "restore_max_abs_error": restore,
    }


def event(step, qpos, distance=None, force=None):
    row = {"step": step, "qpos": [qpos]}
    if distance is not None:
        row["tcp_target_distance_m"] = distance
    if force is not None:
        row["info"] = {"robot_cumulative_force": [force]}
    return row


def analyze(policy, expert, policy_identity=None, expert_identity=None):
    return analyze_closed_loop_divergence(
        policy,
        expert,
        policy_identity or identity(),
        expert_identity or identity(),
        {"joint0": 0.1, "joint1": 0.2},
        channel_names=("joint0", "joint1"),
        units={"joint0": "m", "joint1": "rad"},
    )


def test_reports_first_crossing_channel_and_paired_curves():
    policy = [
        event(1, [0.02, 0.01], 0.9, 10),
        event(2, [0.03, 0.31], 0.8, 30),
        event(3, [0.25, 0.35], 0.7, 60),
    ]
    expert = [
        event(1, [0.0, 0.0], 1.0, 0),
        event(2, [0.0, 0.0], 0.9, 5),
        event(3, [0.0, 0.0], 0.8, 10),
    ]
    report = analyze(policy, expert)
    assert report["status"] == "comparable"
    assert report["first_divergence"]["step"] == 2
    assert report["first_divergence"]["channels"] == ["joint1"]
    assert report["first_divergence"]["primary"]["unit"] == "rad"
    paired = report["curves"]["paired"]
    assert paired["tcp_target_distance_m"]["available"] is True
    assert paired["tcp_target_distance_m"]["values"][0]["delta"] == pytest.approx(-0.1)
    assert paired["robot_cumulative_force"]["values"][1]["delta"] == pytest.approx(25)


def test_hash_mismatch_fails_closed_without_state_comparison():
    rows = [event(1, [0.0, 0.0], 1.0, 0)]
    report = analyze(rows, rows, expert_identity=identity("b" * 64))
    assert report["status"] == "not_comparable"
    assert report["comparison_performed"] is False
    assert report["first_divergence"] is None
    assert report["qpos_error_curve"] == []
    assert "reference_state_sha256_mismatch" in report["pairing"]["reasons"]
    assert report["curves"]["paired"]["status"] == "not_comparable"


def test_missing_expert_state_fails_closed():
    policy = [event(1, [0.0, 0.0])]
    expert = [{"step": 1, "info": {"robot_cumulative_force": [0]}}]
    report = analyze(policy, expert)
    assert report["comparison_performed"] is False
    assert "expert_qpos_missing_at_step_1" in report["trajectory_contract_reasons"]


def test_restore_error_is_required_and_bounded():
    pairing = strict_pairing(identity(), identity(restore=None))
    assert pairing["comparable"] is False
    assert "expert_restore_error_missing_or_above_tolerance" in pairing["reasons"]


def test_thresholds_are_explicit_and_complete():
    with pytest.raises(ValueError, match="every and only"):
        analyze_closed_loop_divergence(
            [event(1, [0.0, 0.0])],
            [event(1, [0.0, 0.0])],
            identity(),
            identity(),
            {"joint0": 0.1},
            channel_names=("joint0", "joint1"),
            units={"joint0": "m", "joint1": "rad"},
        )


def test_incremental_recorder_rejects_step_gaps_and_summarizes():
    expert = [event(1, [0.0, 0.0]), event(2, [0.0, 0.0])]
    recorder = ClosedLoopDivergenceRecorder(
        expert,
        identity(),
        identity(),
        {"joint0": 0.1, "joint1": 0.2},
        channel_names=("joint0", "joint1"),
        units={"joint0": "m", "joint1": "rad"},
    )
    with pytest.raises(ValueError, match="Expected policy step 1"):
        recorder.record(event(2, [0.0, 0.0]))
    recorder.record(event(1, [0.0, 0.0]))
    recorder.record(event(2, [0.0, 0.3]))
    assert recorder.summary()["first_divergence"]["step"] == 2
