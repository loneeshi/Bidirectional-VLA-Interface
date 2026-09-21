"""Bit-exact comparison helpers for training and inference transform leaves."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


REQUIRED_MODEL_LEAVES = (
    "image/base_0_rgb",
    "image/left_wrist_0_rgb",
    "image/right_wrist_0_rgb",
    "image_mask/base_0_rgb",
    "image_mask/left_wrist_0_rgb",
    "image_mask/right_wrist_0_rgb",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
)


def _flatten(value, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        leaves = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError("Transform mappings must use nonempty string keys")
            path = f"{prefix}/{key}" if prefix else key
            leaves.update(_flatten(value[key], path))
        return leaves
    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"Object transform leaf is not auditable: {prefix}")
    return {prefix: array}


def _leaf_sha256(path: str, array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(path.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def compare_shared_model_inputs(training_output: Mapping, server_output: Mapping) -> dict:
    """Compare every non-action leaf and require the frozen model-input set."""
    training = _flatten(training_output)
    server = _flatten(server_output)
    training = {key: value for key, value in training.items() if key != "actions"}
    server = {key: value for key, value in server.items() if key != "actions"}
    missing_training = sorted(set(REQUIRED_MODEL_LEAVES) - set(training))
    missing_server = sorted(set(REQUIRED_MODEL_LEAVES) - set(server))
    training_only = sorted(set(training) - set(server))
    server_only = sorted(set(server) - set(training))
    rows = []
    for key in sorted(set(training) & set(server)):
        left, right = training[key], server[key]
        shape_equal = left.shape == right.shape
        dtype_equal = left.dtype == right.dtype
        values_equal = bool(shape_equal and dtype_equal and np.array_equal(left, right))
        max_abs = None
        if shape_equal and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
            if left.size:
                max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            else:
                max_abs = 0.0
        rows.append(
            {
                "leaf": key,
                "shape_equal": shape_equal,
                "dtype_equal": dtype_equal,
                "array_equal": values_equal,
                "training_shape": list(left.shape),
                "server_shape": list(right.shape),
                "training_dtype": str(left.dtype),
                "server_dtype": str(right.dtype),
                "max_abs_difference": max_abs,
                "training_sha256": _leaf_sha256(key, left),
                "server_sha256": _leaf_sha256(key, right),
            }
        )
    comparable = not missing_training and not missing_server and not training_only and not server_only
    equal = comparable and all(row["array_equal"] for row in rows)
    return {
        "status": "bit_exact" if equal else "mismatch",
        "bit_exact": equal,
        "required_leaves": list(REQUIRED_MODEL_LEAVES),
        "missing_training": missing_training,
        "missing_server": missing_server,
        "training_only": training_only,
        "server_only": server_only,
        "leaves": rows,
    }
