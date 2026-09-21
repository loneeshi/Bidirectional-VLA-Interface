import hashlib

import numpy as np
import pytest

from bvi.s1_preprocessing_parity import (
    array_sha256,
    compare_shared_leaves,
    flatten_leaves,
    load_saved_request,
    quantile_state_ood,
)


def test_array_hash_binds_dtype_shape_and_bytes():
    raw = np.arange(6, dtype=np.uint8)
    assert array_sha256(raw) == array_sha256(raw.copy())
    assert array_sha256(raw) != array_sha256(raw.reshape(2, 3))
    assert array_sha256(raw) != array_sha256(raw.astype(np.int16))
    changed = raw.copy(); changed[-1] += 1
    assert array_sha256(raw) != array_sha256(changed)


def test_flatten_and_exact_shared_comparison_excludes_action_only_leaves():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    left = {"image": {"base": image}, "state": np.array([1, 2], np.float32),
            "actions": np.ones((2, 13), np.float32)}
    right = {"image": {"base": image.copy()}, "state": np.array([1, 2], np.float32)}
    assert sorted(flatten_leaves(left)) == ["actions", "image/base", "state"]
    result = compare_shared_leaves(left, right, required_roots=("image", "state"))
    assert result["exact"]
    assert result["left_only_non_action_leaves"] == []
    assert result["exact_leaf_count"] == 2


def test_comparison_reports_value_dtype_and_required_path_failures():
    left = {"observation": {"image": np.zeros((2, 2), np.uint8)},
            "state": np.array([1], np.float32)}
    right = {"observation": {"image": np.ones((2, 2), np.uint8)},
             "state": np.array([1], np.float64)}
    result = compare_shared_leaves(
        left, right, required_paths=("observation/image", "observation/wrist_image"))
    assert not result["exact"]
    assert result["missing_required_paths"] == ["observation/wrist_image"]
    image = next(row for row in result["leaves"] if row["path"] == "observation/image")
    state = next(row for row in result["leaves"] if row["path"] == "state")
    assert not image["array_equal"] and image["max_abs"] == 1
    assert state["array_equal"] and not state["sha256_equal"] and not state["exact"]


def test_comparison_fails_closed_on_unmatched_non_action_leaf():
    left = {"state": np.zeros(2, np.float32), "unexpected": np.asarray(1)}
    right = {"state": np.zeros(2, np.float32)}
    result = compare_shared_leaves(left, right, required_roots=("state",))
    assert not result["exact"]
    assert result["left_only_non_action_leaves"] == ["unexpected"]


def test_quantile_state_ood_uses_pinned_formula_and_reports_distances():
    result = quantile_state_ood(
        np.array([-1.0, 0.5, 3.0]), np.array([0.0, 0.0, 1.0]),
        np.array([2.0, 1.0, 2.0]), epsilon=1e-6)
    assert result["outside_indices"] == [0, 2]
    assert result["outside_count"] == 2
    assert result["channels"][0]["distance_below_q01"] == 1
    assert result["channels"][2]["distance_above_q99"] == 1
    assert np.isclose(result["channels"][1]["normalized"], -1e-6 / 1.000001)


def test_saved_request_contract_roundtrip(tmp_path):
    path = tmp_path / "step000.npz"
    prompt = "Reach the apple with the gripper open."
    np.savez_compressed(
        path,
        head_rgb=np.zeros((128, 128, 3), np.uint8),
        wrist_rgb=np.full((128, 128, 3), 7, np.uint8),
        state=np.arange(24, dtype=np.float32),
        prompt=prompt,
    )
    request = load_saved_request(path)
    assert request["prompt"] == prompt
    assert request["state"].dtype == np.float32
    assert request["wrist_rgb"][0, 0, 0] == 7
    assert hashlib.sha256(prompt.encode()).hexdigest()


@pytest.mark.parametrize("mutation", ["key", "image_dtype", "state_dtype", "prompt"])
def test_saved_request_rejects_contract_drift(tmp_path, mutation):
    values = dict(
        head_rgb=np.zeros((128, 128, 3), np.uint8),
        wrist_rgb=np.zeros((128, 128, 3), np.uint8),
        state=np.zeros(24, np.float32),
        prompt="Reach.",
    )
    if mutation == "key":
        values["extra"] = np.array(1)
    elif mutation == "image_dtype":
        values["head_rgb"] = values["head_rgb"].astype(np.float32)
    elif mutation == "state_dtype":
        values["state"] = values["state"].astype(np.float64)
    else:
        values["prompt"] = ""
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        load_saved_request(path)
