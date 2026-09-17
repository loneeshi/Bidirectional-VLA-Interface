import pytest

from bvi.fetch_current_labels import (
    CONTRACT,
    first_call_anchor,
    sample_times,
    teacher_targets,
    validate_manifest_splits,
    verified_teacher_targets,
)


def test_author_current_frame_math_has_no_post_action_offset():
    assert CONTRACT == "current_observation_v2"
    assert teacher_targets(10, 14, 10)["progress_target"] == [0.0, 0.25]
    assert teacher_targets(10, 14, 12)["progress_target"] == [0.5, 0.75]
    penultimate = teacher_targets(10, 14, 13, horizon=3)
    assert penultimate == {
        "progress_target": [0.75, 1.0, 0.0],
        "progress_valid": [True, True, False],
        "action_valid": [True, False, False],
    }


def test_endpoint_has_completion_progress_without_action_supervision():
    assert teacher_targets(3, 9, 9) == {
        "progress_target": [1.0, 0.0],
        "progress_valid": [True, False],
        "action_valid": [False, False],
    }


def test_shortest_interval_and_single_horizon():
    assert teacher_targets(0, 1, 0, 1)["progress_target"] == [0.0]
    assert teacher_targets(0, 1, 1, 1)["action_valid"] == [False]


@pytest.mark.parametrize("args", [
    (-1, 4, 0), (2, 2, 2), (3, 2, 3), (2, 4, 1), (2, 4, 5),
    (0.0, 4, 0), (0, 4.5, 0), (0, 4, True), (0, 4, 0, 0),
    (0, 4, 0, -1), (0, 4, 0, 1.5), (0, 4, 0, True),
])
def test_invalid_teacher_dimensions_and_bounds(args):
    with pytest.raises(ValueError):
        teacher_targets(*args)


def test_failure_end_cannot_be_promoted_to_verified_teacher_completion():
    failure = {"start": 0, "end": 24, "success": False}
    with pytest.raises(ValueError, match="completion_evidence"):
        verified_teacher_targets(failure, 24)
    failure.update(completion_evidence="timeout", completion_verified=False)
    with pytest.raises(ValueError, match="unverified"):
        verified_teacher_targets(failure, 24)
    # A failed full trajectory can still have an independently verified subskill.
    subskill = {"start": 0, "end": 5, "completion_evidence": "stable_grasp"}
    assert verified_teacher_targets(subskill, 5)["progress_target"][0] == 1.0


def test_invocation_start_anchor_has_no_future_or_action_labels():
    anchor = first_call_anchor(4)
    assert anchor["progress_target"] == [0.0] * 4
    assert anchor["progress_valid"] == [True, False, False, False]
    assert not any(anchor["action_valid"])
    with pytest.raises(TypeError):
        first_call_anchor(end=24)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_anchor_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        first_call_anchor(horizon)


def test_sample_times_are_unique_sorted_and_include_observed_endpoint():
    assert sample_times(10, 18) == [10, 12, 14, 16, 17, 18]
    assert sample_times(0, 1) == [0, 1]
    assert sample_times(1, 4) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        sample_times(4, 4)


def _row(split, seed=3000, task="pick", scene_split="train", **extra):
    return dict(split=split, scene_split=scene_split, task=task, seed=seed, **extra)


def test_all_episode_descendants_share_split_despite_distinct_paths_and_seeds():
    parent = dict(scene_split="train", task="pick", seed=3000)
    rows = [_row("train", trajectory="teacher.h5"),
            _row("train", seed=9999, parent_episode=parent, trajectory="recovery.h5")]
    assert validate_manifest_splits(rows) == {("train", "pick", 3000): "train"}
    rows.append(_row("validation", seed=8888, parent_episode=parent, path="replay.npz"))
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest_splits(rows)


@pytest.mark.parametrize("split", ["validation", "test", "diagnostic"])
def test_episode_overlap_rejected_for_every_nontraining_partition(split):
    with pytest.raises(ValueError, match="crosses splits"):
        validate_manifest_splits([_row("train"), _row(split)])


@pytest.mark.parametrize("seed", [2025, 2030])
def test_diagnostic_seed_never_trains_even_through_parent_identity(seed):
    parent = dict(scene_split="val", task="pick", seed=seed)
    with pytest.raises(ValueError, match="Diagnostic seed"):
        validate_manifest_splits([_row("train", parent_episode=parent)])
    assert validate_manifest_splits([_row("diagnostic", seed=seed)])


def test_distinct_tasks_and_scene_splits_are_distinct_parent_episodes():
    rows = [_row("train"), _row("validation", task="place"),
            _row("test", scene_split="val")]
    assert len(validate_manifest_splits(rows)) == 3


@pytest.mark.parametrize("row", [
    {}, _row("unknown"), _row("train", seed="3000"),
    _row("train", scene_split=""), _row("train", parent_episode={}),
])
def test_incomplete_or_ambiguous_manifest_identity_rejected(row):
    with pytest.raises(ValueError):
        validate_manifest_splits([row])
