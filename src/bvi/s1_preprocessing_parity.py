"""Pure helpers for exact S1 training/inference preprocessing audits.

STATUS: frozen — historical training and diagnostics (retained)

This module deliberately has no OpenPI, JAX, simulator, or GPU dependency.  The
runtime audit imports those dependencies only after validating the saved raw
request files with these helpers.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUEST_KEYS = frozenset({"head_rgb", "wrist_rgb", "state", "prompt"})
LIBERO_REQUIRED_PATHS = (
    "image/base_0_rgb",
    "image/left_wrist_0_rgb",
    "image/right_wrist_0_rgb",
    "image_mask/base_0_rgb",
    "image_mask/left_wrist_0_rgb",
    "image_mask/right_wrist_0_rgb",
    "state",
    "prompt",
)
MODEL_REQUIRED_PATHS = (
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


def array_sha256(value: Any) -> str:
    """Hash an array together with its dtype and shape."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot enter a deterministic audit")
    digest = hashlib.sha256()
    digest.update(b"bvi-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt(value: Any) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError("Saved prompt must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Saved prompt must be nonempty UTF-8 text")
    result.encode("utf-8")
    return result


def load_saved_request(path: str | Path) -> dict[str, Any]:
    """Load the exact four-field request written by ``eval_ia_s1_call.py``."""
    request_path = Path(path)
    with np.load(request_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUEST_KEYS:
            raise ValueError(
                f"Saved request keys differ: expected {sorted(REQUEST_KEYS)}, "
                f"found {sorted(archive.files)}"
            )
        head = np.asarray(archive["head_rgb"])
        wrist = np.asarray(archive["wrist_rgb"])
        state = np.asarray(archive["state"])
        prompt = _prompt(archive["prompt"])
    for name, image in (("head_rgb", head), ("wrist_rgb", wrist)):
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 HWC RGB128")
    if state.shape != (24,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("state must be finite float32 native24")
    return {
        "head_rgb": head.copy(),
        "wrist_rgb": wrist.copy(),
        "state": state.copy(),
        "prompt": prompt,
    }


def flatten_leaves(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested mappings/sequences into deterministic array leaves."""
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten_leaves(value[key], child))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}/{index}" if prefix else str(index)
            result.update(flatten_leaves(item, child))
        return result
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"Object leaf is not auditable: {prefix}")
    return {prefix: array}


def leaf_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": array_sha256(array),
    }
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        numeric = array.astype(np.float64, copy=False)
        result.update(
            finite=bool(np.isfinite(numeric).all()),
            minimum=float(np.min(numeric)) if numeric.size else None,
            maximum=float(np.max(numeric)) if numeric.size else None,
            mean=float(np.mean(numeric)) if numeric.size else None,
            std=float(np.std(numeric)) if numeric.size else None,
        )
    else:
        # Keep this useful for either a scalar prompt or an unexpected string
        # array without relying on ``item()``, which only accepts one element.
        text = json.dumps(array.tolist(), ensure_ascii=False, separators=(",", ":"))
        result["utf8_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return result


def tree_summary(value: Any) -> dict[str, dict[str, Any]]:
    return {path: leaf_summary(array) for path, array in flatten_leaves(value).items()}


def _root(path: str) -> str:
    return path.split("/", 1)[0]


def compare_shared_leaves(
    left: Any,
    right: Any,
    *,
    excluded_roots: tuple[str, ...] = ("actions", "actions_is_pad"),
    required_roots: tuple[str, ...] = (),
    required_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Compare every shared non-action leaf with both array equality and SHA."""
    left_flat = {
        path: value for path, value in flatten_leaves(left).items()
        if _root(path) not in excluded_roots
    }
    right_flat = {
        path: value for path, value in flatten_leaves(right).items()
        if _root(path) not in excluded_roots
    }
    shared = sorted(left_flat.keys() & right_flat.keys())
    rows = []
    for path in shared:
        a, b = left_flat[path], right_flat[path]
        same_shape = a.shape == b.shape
        array_equal = bool(same_shape and np.array_equal(a, b))
        left_sha, right_sha = array_sha256(a), array_sha256(b)
        max_abs = None
        if (same_shape and a.size and
                np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        rows.append({
            "path": path,
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
            "left_dtype": a.dtype.str,
            "right_dtype": b.dtype.str,
            "array_equal": array_equal,
            "left_sha256": left_sha,
            "right_sha256": right_sha,
            "sha256_equal": left_sha == right_sha,
            "max_abs": max_abs,
            "exact": array_equal and left_sha == right_sha,
        })
    shared_roots = {_root(path) for path in shared}
    missing_roots = sorted(set(required_roots) - shared_roots)
    missing_paths = sorted(set(required_paths) - set(shared))
    left_only = sorted(left_flat.keys() - right_flat.keys())
    right_only = sorted(right_flat.keys() - left_flat.keys())
    exact = (
        bool(shared)
        and not missing_roots
        and not missing_paths
        and not left_only
        and not right_only
        and all(row["exact"] for row in rows)
    )
    return {
        "exact": exact,
        "shared_leaf_count": len(shared),
        "exact_leaf_count": sum(row["exact"] for row in rows),
        "missing_required_roots": missing_roots,
        "missing_required_paths": missing_paths,
        "left_only_non_action_leaves": left_only,
        "right_only_non_action_leaves": right_only,
        "leaves": rows,
    }


def quantile_state_ood(state: Any, q01: Any, q99: Any, epsilon: float = 1e-6) -> dict[str, Any]:
    """Report the exact quantile normalization and marginal OOD channels."""
    value = np.asarray(state, dtype=np.float64)
    lower = np.asarray(q01, dtype=np.float64)
    upper = np.asarray(q99, dtype=np.float64)
    if value.ndim != 1 or lower.shape != value.shape or upper.shape != value.shape:
        raise ValueError("state/q01/q99 must be same-width vectors")
    if not np.isfinite(value).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("state quantile audit requires finite values")
    if np.any(upper < lower) or not epsilon > 0:
        raise ValueError("Invalid quantile bounds/epsilon")
    normalized = 2.0 * (value - lower) / (upper - lower + epsilon) - 1.0
    outside = (value < lower) | (value > upper)
    zero_span = upper == lower
    channels = [
        {
            "index": int(index),
            "value": float(value[index]),
            "q01": float(lower[index]),
            "q99": float(upper[index]),
            "normalized": float(normalized[index]),
            "outside_q01_q99": bool(outside[index]),
            "distance_below_q01": float(max(lower[index] - value[index], 0.0)),
            "distance_above_q99": float(max(value[index] - upper[index], 0.0)),
        }
        for index in range(len(value))
    ]
    return {
        "dimension": len(value),
        "epsilon": epsilon,
        "outside_count": int(np.sum(outside)),
        "outside_indices": np.flatnonzero(outside).astype(int).tolist(),
        "zero_span_indices": np.flatnonzero(zero_span).astype(int).tolist(),
        "max_abs_normalized": float(np.max(np.abs(normalized))),
        "channels": channels,
    }
