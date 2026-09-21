import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_s1_offline_action_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("s1_offline_action_reconstruction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controller_to_physical_matches_pinned_fetch_ranges():
    low = MODULE.controller_to_physical(np.full(13, -1.0))
    high = MODULE.controller_to_physical(np.full(13, 1.0))
    assert np.allclose(low[:7], -0.1)
    assert np.allclose(high[:7], 0.1)
    assert low[7] == -0.01 and high[7] == 0.05
    assert np.allclose(low[8:11], -0.1)
    assert np.allclose(high[8:11], 0.1)
    assert np.allclose(low[11:], [-1.0, -3.14])
    assert np.allclose(high[11:], [1.0, 3.14])
    midpoint = MODULE.controller_to_physical(np.zeros(13))
    assert np.isclose(midpoint[7], 0.02)  # absolute gripper joint target, not a delta
    assert np.allclose(np.delete(midpoint, 7), 0.0)


def test_applied_actions_clip_then_mask_stationary_head():
    raw = np.zeros((2, 13))
    raw[0, 0] = 1.5
    raw[0, 8:10] = [-0.7, 0.9]
    raw[1, 12] = -2.0
    applied = MODULE.applied_controller_actions(raw)
    assert applied[0, 0] == 1
    assert np.array_equal(applied[:, 8:10], np.zeros((2, 2)))
    assert applied[1, 12] == -1


def _row(parent, call, frame):
    return dict(parent_id=parent, call_index=call, observation_index=frame)


def test_balanced_selection_is_deterministic_and_spans_groups_and_time():
    rows = [_row(0, 0, i) for i in range(5)] + [_row(1, 0, i) for i in range(5)]
    selected = MODULE.balanced_selection(rows, 6)
    assert selected == MODULE.balanced_selection(list(reversed(rows)), 6)
    assert {row["parent_id"] for row in selected} == {0, 1}
    for parent in (0, 1):
        frames = {row["observation_index"] for row in selected if row["parent_id"] == parent}
        assert {0, 4}.issubset(frames)


def test_channel_statistics_separates_distribution_mean_and_sample_error():
    # Sample mean is exact but each stochastic draw has unit error.
    samples = np.zeros((3, 2, 13), dtype=np.float64)
    samples[:, 0, 0] = -1
    samples[:, 1, 0] = 1
    target = np.zeros((3, 13), dtype=np.float64)
    stats = MODULE.channel_statistics(samples, target)
    assert stats[0]["rmse"] == 0
    assert stats[0]["expected_sample_rmse"] == 1
    assert np.isclose(stats[0]["mean_stochastic_std"], np.sqrt(2))
    assert stats[0]["pearson"] is None


def test_per_channel_records_use_valid_mask_and_keep_raw_head_diagnostic():
    predictions = np.zeros((2, 2, 3, 13), dtype=np.float64)
    expert = np.zeros((2, 3, 13), dtype=np.float64)
    valid = np.array([[True, True, False], [True, False, False]])
    predictions[..., 8] = 0.75
    predictions[:, :, 0, 11] = 0.5
    records = MODULE.per_channel_records(predictions, expert, valid)
    first_head = next(r for r in records if r["scope"] == "first_action" and r["channel"] == 8)
    first_base = next(r for r in records if r["scope"] == "first_action" and r["channel"] == 11)
    chunk_base = next(r for r in records if r["scope"] == "valid_chunk" and r["channel"] == 11)
    assert first_head["physical_rmse"] == 0  # actual wrapper command is masked
    assert first_head["controller_raw_rmse"] == 0.75
    assert first_head["wrapper_changed_fraction"] == 1
    assert first_base["physical_rmse"] == 0.5  # normalized 0.5 -> 0.5 m/s
    assert first_base["physical_n_targets"] == 2
    assert chunk_base["physical_n_targets"] == 3
