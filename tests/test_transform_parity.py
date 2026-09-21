import numpy as np

from bvi.transform_parity import REQUIRED_MODEL_LEAVES, compare_shared_model_inputs


def model_input(fill=0):
    return {
        "image": {
            "base_0_rgb": np.full((4, 4, 3), fill, np.uint8),
            "left_wrist_0_rgb": np.full((4, 4, 3), fill + 1, np.uint8),
            "right_wrist_0_rgb": np.zeros((4, 4, 3), np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.asarray(True),
            "left_wrist_0_rgb": np.asarray(True),
            "right_wrist_0_rgb": np.asarray(False),
        },
        "state": np.arange(32, dtype=np.float32),
        "tokenized_prompt": np.arange(8, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(8, dtype=bool),
    }


def test_transform_parity_is_bit_exact_and_ignores_training_action_label():
    training = model_input()
    training["actions"] = np.ones((10, 32), np.float32)
    server = model_input()
    result = compare_shared_model_inputs(training, server)
    assert result["status"] == "bit_exact"
    assert result["bit_exact"] is True
    assert {row["leaf"] for row in result["leaves"]} == set(REQUIRED_MODEL_LEAVES)
    assert all(row["training_sha256"] == row["server_sha256"] for row in result["leaves"])


def test_transform_parity_detects_one_value_and_dtype_mismatches():
    training = model_input()
    server = model_input()
    server["state"] = server["state"].copy()
    server["state"][3] += 1
    server["tokenized_prompt"] = server["tokenized_prompt"].astype(np.int64)
    result = compare_shared_model_inputs(training, server)
    assert result["status"] == "mismatch"
    rows = {row["leaf"]: row for row in result["leaves"]}
    assert rows["state"]["max_abs_difference"] == 1
    assert rows["state"]["array_equal"] is False
    assert rows["tokenized_prompt"]["dtype_equal"] is False


def test_transform_parity_fails_closed_on_missing_leaf():
    training = model_input()
    server = model_input()
    del server["image_mask"]["right_wrist_0_rgb"]
    result = compare_shared_model_inputs(training, server)
    assert result["bit_exact"] is False
    assert "image_mask/right_wrist_0_rgb" in result["missing_server"]
    assert "image_mask/right_wrist_0_rgb" in result["training_only"]
