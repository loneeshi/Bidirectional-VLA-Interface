"""Build and validate held-out native24 wrong-handoff evidence.

STATUS: frozen — historical training and diagnostics (retained)

``source_replay_verified`` is deliberately derived, not trusted as a naked
boolean.  A case is replay verified only when a hash-bound replay record ties
the exact parent trajectory and observation index to the exact NPZ bytes and
native observation content, records at least two identical reconstructions,
and affirms the frozen physical checks.  The training-cache manifest is also
hash bound so a training parent cannot later be relabelled as held out.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "bvi.native24-handoff/1"
ACQUISITION_SCHEMA = "bvi.native24-handoff-acquisition/1"
TRAJECTORY_SCHEMA = "bvi.native24-source-trajectory/1"
REPLAY_SCHEMA = "bvi.native24-source-replay/1"
PARENT_REGISTRY_SCHEMA = "bvi.native24-heldout-parent-registry/1"
LABEL_CONTRACT = "current_observation_v2"
REQUIRED_CLASSES = frozenset(
    {
        "far_grasp_diagnostic",
        "near_grasp_candidate_not_completion",
        "unheld_move",
        "held_move",
    }
)
CLASSIFICATION_DISTANCE_M = 0.10
MAX_REPLAY_ATOL = 1e-5
OBSERVATION_KEYS = ("head_rgb", "wrist_rgb", "state")


class NotEvaluableError(ValueError):
    """Required frozen evidence is absent; callers must not substitute a case."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _artifact_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a nonempty relative path")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the evidence directory") from error
    if not path.is_file():
        raise ValueError(f"Missing {label}: {relative}")
    return path


def _portable_relative(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be stored under the manifest directory") from error
    return relative.as_posix()


def _training_parent(parent: Any) -> dict[str, Any]:
    keys = {"task", "source_sha256", "trajectory", "parent_id"}
    if not isinstance(parent, dict) or set(parent) != keys:
        raise ValueError("Parent episode must have exactly task/source_sha256/trajectory/parent_id")
    if parent["task"] not in {"pick", "place"} or not _is_sha256(parent["source_sha256"]):
        raise ValueError("Invalid parent task or source hash")
    if type(parent["parent_id"]) is not int or parent["parent_id"] < 0:
        raise ValueError("Invalid parent id")
    if parent["trajectory"] != f"traj_{parent['parent_id']}":
        raise ValueError("Parent trajectory and id disagree")
    return dict(parent)


def _handoff_parent(parent: Any) -> dict[str, Any]:
    keys = {"source_kind", "task", "source_sha256", "trajectory", "parent_id"}
    if not isinstance(parent, dict) or set(parent) != keys:
        raise ValueError(
            "Handoff parent must have exactly source_kind/task/source_sha256/trajectory/parent_id"
        )
    if parent["source_kind"] != "archived_online_replay" or parent["task"] != "pick":
        raise ValueError("Only the archived online Pick replay source is currently admissible")
    if not _is_sha256(parent["source_sha256"]):
        raise ValueError("Invalid handoff parent source hash")
    if type(parent["parent_id"]) is not int or parent["parent_id"] < 0:
        raise ValueError("Invalid handoff parent id")
    if parent["trajectory"] not in {
        f"train-seed{parent['parent_id']}",
        f"validation-seed{parent['parent_id']}",
    }:
        raise ValueError("Handoff parent trajectory and id disagree")
    return dict(parent)


def _parent_key(parent: Mapping[str, Any]) -> str:
    return json.dumps(parent, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parent_rosters(training_manifest_path: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    manifest = _read_json(training_manifest_path, "training manifest")
    if (
        manifest.get("schema") != "fetch_native24_family_v1"
        or manifest.get("label_contract") != LABEL_CONTRACT
    ):
        raise ValueError("Training manifest is not the native24 current-observation cache")
    train: set[str] = set()
    validation: set[str] = set()
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Training manifest has no parent roster")
    for row in episodes:
        if not isinstance(row, dict) or row.get("split") not in {"train", "validation"}:
            raise ValueError("Invalid parent split in training manifest")
        key = _parent_key(_training_parent(row.get("parent_episode")))
        target = train if row["split"] == "train" else validation
        if key in target:
            raise ValueError("Duplicate parent in training manifest")
        target.add(key)
    if train & validation:
        raise ValueError("Training manifest leaks parents across splits")
    if not train or not validation:
        raise ValueError("Both train and validation parent rosters are required")
    return manifest, train, validation


def _load_parent_registry(
    root: Path, binding: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Held-out parent registry binding is incomplete")
    path = _artifact_path(root, binding["path"], "held-out parent registry")
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Held-out parent registry hash mismatch")
    registry = _read_json(path, "held-out parent registry")
    if (
        registry.get("schema") != PARENT_REGISTRY_SCHEMA
        or registry.get("status") != "verified_parent_partitions"
        or registry.get("source_batch") != "fetch-current-handoffs-2026-09-17-run01"
    ):
        raise ValueError("Held-out parent registry is not the frozen replay batch")
    if not _is_sha256(registry.get("collection_index_sha256")) or not _is_sha256(
        registry.get("batch_sha256")
    ):
        raise ValueError("Held-out parent registry lacks source artifact hashes")
    rows = registry.get("parents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Held-out parent registry has no parents")
    train: dict[str, dict[str, Any]] = {}
    validation: dict[str, dict[str, Any]] = {}
    seeds = {"train": set(), "validation": set()}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in seeds:
            raise ValueError("Invalid held-out parent registry row")
        parent = _handoff_parent(row.get("parent_episode"))
        if parent["trajectory"] != f"{row['split']}-seed{parent['parent_id']}":
            raise ValueError("Parent registry partition and trajectory disagree")
        if row.get("events_sha256") != parent["source_sha256"] or not _is_sha256(
            row.get("result_sha256")
        ):
            raise ValueError("Parent registry source hashes disagree")
        key = _parent_key(parent)
        target = train if row["split"] == "train" else validation
        if key in target:
            raise ValueError("Duplicate parent in held-out registry")
        target[key] = row
        seeds[row["split"]].add(parent["parent_id"])
    if set(train) & set(validation):
        raise ValueError("Held-out parent registry leaks parents across splits")
    # This is a previously archived, fixed collection.  Reject substituted
    # origins instead of accepting a freshly curated favorable roster.
    if seeds != {"train": {3000, 3001, 3002, 3003}, "validation": {3020, 3021}}:
        raise ValueError("Frozen replay parent roster changed")
    return registry, train, validation


def observation_sha256(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in OBSERVATION_KEYS:
        value = np.ascontiguousarray(values[key])
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_observation(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(OBSERVATION_KEYS):
            raise ValueError("Native24 handoff NPZ must contain only head_rgb/wrist_rgb/state")
        values = {key: archive[key].copy() for key in OBSERVATION_KEYS}
    for key in ("head_rgb", "wrist_rgb"):
        value = values[key]
        if value.shape != (128, 128, 3) or value.dtype != np.uint8:
            raise ValueError(f"{key} must be one uint8 RGB128 observation")
    state = values["state"]
    if state.shape != (24,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("state must be one finite float32 native24 observation")
    return values, observation_sha256(values)


def _validate_ground_truth(classification: str, family: str, ground_truth: Any) -> None:
    if not isinstance(ground_truth, dict) or set(ground_truth) != {
        "source",
        "tcp_object_distance_m",
        "is_grasped",
        "physical_completion",
    }:
        raise ValueError("Ground truth must contain only frozen physical predicates")
    if ground_truth["source"] != "source_replay_physical_predicates":
        raise ValueError("Ground truth must come from source replay physical predicates")
    distance = ground_truth["tcp_object_distance_m"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("TCP-object distance must be numeric")
    if not math.isfinite(float(distance)) or float(distance) < 0:
        raise ValueError("TCP-object distance must be finite and nonnegative")
    if type(ground_truth["is_grasped"]) is not bool or type(ground_truth["physical_completion"]) is not bool:
        raise ValueError("Physical predicates must be booleans")
    # These are pre-completion handoff boundaries.  A completed endpoint belongs
    # to completion calibration, not this wrong-handoff gate.
    if ground_truth["physical_completion"]:
        raise ValueError("Wrong-handoff cases must precede physical completion")
    held = ground_truth["is_grasped"]
    distance = float(distance)
    expected_family = "grasp" if "grasp" in classification else "move"
    if family != expected_family:
        raise ValueError("Classification and family disagree")
    if classification == "far_grasp_diagnostic" and (held or distance < CLASSIFICATION_DISTANCE_M):
        raise ValueError("Far grasp classification contradicts physical predicates")
    if classification == "near_grasp_candidate_not_completion" and (
        held or distance >= CLASSIFICATION_DISTANCE_M
    ):
        raise ValueError("Near grasp classification contradicts physical predicates")
    if classification == "unheld_move" and held:
        raise ValueError("Unheld move classification contradicts physical predicates")
    if classification == "held_move" and not held:
        raise ValueError("Held move classification contradicts physical predicates")


def _validate_trajectory_record(path: Path, expected_parent: dict[str, Any]) -> dict[str, Any]:
    record = _read_json(path, "source trajectory record")
    if record.get("schema") != TRAJECTORY_SCHEMA or record.get("status") != "verified_source_trajectory":
        raise ValueError("Source trajectory record is not verified")
    if _handoff_parent(record.get("parent_episode")) != expected_parent:
        raise ValueError("Source trajectory record belongs to another parent")
    if record.get("source_events_sha256") != expected_parent["source_sha256"]:
        raise ValueError("Source trajectory events hash differs from parent provenance")
    if not _is_sha256(record.get("source_result_sha256")):
        raise ValueError("Source trajectory result hash is missing")
    if record.get("digest_contract") != "historical-events-jsonl-v1" or not _is_sha256(
        record.get("trajectory_sha256")
    ):
        raise ValueError("Source trajectory content digest is missing")
    if record["trajectory_sha256"] != record["source_events_sha256"]:
        raise ValueError("Historical trajectory digest must bind the complete events log")
    return record


def _validate_replay_record(
    path: Path,
    case_id: str,
    instruction: str,
    parent: dict[str, Any],
    trajectory_sha256: str,
    npz_sha256: str,
    observation_digest: str,
) -> dict[str, Any]:
    record = _read_json(path, "source replay record")
    if record.get("schema") != REPLAY_SCHEMA or record.get("status") != "verified_source_replay":
        raise ValueError("Source replay record is not verified")
    if record.get("case_id") != case_id or _handoff_parent(record.get("parent_episode")) != parent:
        raise ValueError("Source replay record identity differs from case")
    if record.get("instruction") != instruction:
        raise ValueError("Source replay instruction differs from case")
    if record.get("trajectory_sha256") != trajectory_sha256:
        raise ValueError("Source replay record uses another trajectory")
    if record.get("npz_sha256") != npz_sha256:
        raise ValueError("Source replay record is not bound to the exact NPZ")
    if record.get("boundary_observation_sha256") != observation_digest:
        raise ValueError("Source replay observation differs from NPZ contents")
    index = record.get("observation_index")
    if type(index) is not int or index < 0:
        raise ValueError("Replay observation index must be a nonnegative integer")
    hashes = record.get("reconstruction_observation_sha256")
    if (
        not isinstance(hashes, list)
        or len(hashes) < 2
        or any(value != observation_digest for value in hashes)
    ):
        raise ValueError("At least two identical source reconstructions are required")
    atol = record.get("physical_atol")
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not (0 < atol <= MAX_REPLAY_ATOL):
        raise ValueError("Replay physical tolerance is missing or too loose")
    required_checks = (
        "initial_state_verified",
        "action_prefix_verified",
        "boundary_observation_verified",
        "physical_predicates_verified",
    )
    if any(record.get(name) is not True for name in required_checks):
        raise ValueError("Source replay did not verify every frozen physical check")
    if record.get("policy_calls") != 0 or record.get("training_updates") != 0:
        raise ValueError("Source replay evidence must not contain policy calls or training updates")
    return record


def _validate_case(
    case: Any,
    root: Path,
    registry_train: Mapping[str, dict[str, Any]],
    registry_validation: Mapping[str, dict[str, Any]],
    training_source_hashes: set[str],
) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError("Handoff case must be an object")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id:
        raise ValueError("Invalid case id")
    expected_case_sha = canonical_sha256({key: value for key, value in case.items() if key != "case_sha256"})
    if case.get("case_sha256") != expected_case_sha:
        raise ValueError("Case metadata hash mismatch")
    if case.get("role") != "validation" or case.get("classification") not in REQUIRED_CLASSES:
        raise ValueError("Only the four held-out validation classifications are accepted")
    if case.get("split") != "validation":
        raise ValueError("Native24 handoff cases must use the validation split")
    if case.get("label_contract") != LABEL_CONTRACT:
        raise ValueError("Case label contract differs")
    instruction = case.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Handoff case requires a nonempty frozen instruction")
    if case.get("policy_input_privileged") is not False:
        raise ValueError("Privileged policy input is forbidden")
    if case.get("model_progress_used_as_ground_truth") is not False:
        raise ValueError("Model progress cannot supply ground truth")
    parent = _handoff_parent(case.get("parent_episode"))
    parent_key = _parent_key(parent)
    if parent_key in registry_train or parent_key not in registry_validation:
        raise ValueError("Case parent is not in the frozen held-out roster")
    if parent["source_sha256"] in training_source_hashes:
        raise ValueError("Held-out replay source overlaps the native24 training source")
    registry_row = registry_validation[parent_key]
    npz_path = _artifact_path(root, case.get("path"), "handoff NPZ")
    npz_sha = sha256_file(npz_path)
    if case.get("sha256") != npz_sha:
        raise ValueError("Handoff NPZ hash mismatch")
    values, observation_digest = _load_observation(npz_path)
    trajectory = case.get("source_trajectory")
    if not isinstance(trajectory, dict) or set(trajectory) != {"path", "sha256", "trajectory_sha256"}:
        raise ValueError("Source trajectory artifact binding is incomplete")
    trajectory_path = _artifact_path(root, trajectory["path"], "source trajectory record")
    if sha256_file(trajectory_path) != trajectory["sha256"]:
        raise ValueError("Source trajectory record hash mismatch")
    trajectory_record = _validate_trajectory_record(trajectory_path, parent)
    if (
        trajectory_record["source_events_sha256"] != registry_row["events_sha256"]
        or trajectory_record["source_result_sha256"] != registry_row["result_sha256"]
    ):
        raise ValueError("Source trajectory differs from the frozen parent registry")
    if trajectory["trajectory_sha256"] != trajectory_record["trajectory_sha256"]:
        raise ValueError("Source trajectory content hash mismatch")
    replay = case.get("source_replay")
    if not isinstance(replay, dict) or set(replay) != {"path", "sha256"}:
        raise ValueError("Source replay artifact binding is incomplete")
    replay_path = _artifact_path(root, replay["path"], "source replay record")
    if sha256_file(replay_path) != replay["sha256"]:
        raise ValueError("Source replay record hash mismatch")
    replay_record = _validate_replay_record(
        replay_path,
        case_id,
        instruction,
        parent,
        trajectory_record["trajectory_sha256"],
        npz_sha,
        observation_digest,
    )
    if case.get("observation_index") != replay_record["observation_index"]:
        raise ValueError("Case observation index differs from replay evidence")
    if case.get("source_replay_verified") is not True:
        raise ValueError("Derived source replay verification flag is absent")
    if case.get("ground_truth") != replay_record.get("ground_truth"):
        raise ValueError("Case ground truth differs from replay evidence")
    _validate_ground_truth(case["classification"], case.get("family"), case["ground_truth"])
    return {
        "case_id": case_id,
        "classification": case["classification"],
        "parent_episode": parent,
        "npz_sha256": npz_sha,
        "observation_sha256": observation_digest,
        "trajectory_sha256": trajectory_record["trajectory_sha256"],
        "source_replay_sha256": replay["sha256"],
    }


def validate_manifest_data(
    manifest: Mapping[str, Any], root: str | Path, training_manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(root)
    training_manifest_path = Path(training_manifest_path)
    training_manifest, _, _ = _parent_rosters(training_manifest_path)
    training_source_hashes = {
        row["parent_episode"]["source_sha256"] for row in training_manifest["episodes"]
    }
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "verified_handoff_inputs":
        raise ValueError("Native24 handoff manifest is not verified")
    if manifest.get("label_contract") != LABEL_CONTRACT:
        raise ValueError("Native24 handoff label contract differs")
    training = manifest.get("training_data")
    if not isinstance(training, dict) or training != {
        "schema": "fetch_native24_family_v1",
        "manifest_sha256": sha256_file(training_manifest_path),
    }:
        raise ValueError("Handoff manifest is not bound to this training cache")
    _, registry_train, registry_validation = _load_parent_registry(
        root, manifest.get("parent_registry")
    )
    observation = manifest.get("observation_contract")
    expected_observation = {
        "head_rgb": {"shape": [128, 128, 3], "dtype": "uint8"},
        "wrist_rgb": {"shape": [128, 128, 3], "dtype": "uint8"},
        "state": {"shape": [24], "dtype": "float32", "source": "env_native_agent"},
        "policy_input_privileged": False,
    }
    if observation != expected_observation:
        raise ValueError("Native24 handoff observation contract differs")
    if manifest.get("classification_contract") != {
        "schema": "bvi.native24-handoff-classification/1",
        "far_grasp_distance_m": CLASSIFICATION_DISTANCE_M,
        "ground_truth_source": "source_replay_physical_predicates",
        "physical_completion_required": False,
    }:
        raise ValueError("Handoff classification contract differs")
    if manifest.get("offline_behavior_scope") != {
        "status": "input_only_behavior_not_evaluated",
        "single_observation_per_case": True,
        "thresholds_bound": False,
        "false_completion_evaluable": False,
        "stagnation_evaluable": False,
        "rollback_evaluable": False,
    }:
        raise ValueError("Input manifest must not claim the separate learned-feedback behavior gate")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Native24 handoff manifest has no cases")
    results = [
        _validate_case(
            case,
            root,
            registry_train,
            registry_validation,
            training_source_hashes,
        )
        for case in cases
    ]
    identifiers = [result["case_id"] for result in results]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate native24 handoff case id")
    seen = {result["classification"] for result in results}
    if seen != REQUIRED_CLASSES:
        raise NotEvaluableError("Missing native24 wrong-handoff validation classes")
    return {
        "schema": SCHEMA,
        "status": "verified_handoff_inputs",
        "case_count": len(results),
        "classifications": sorted(seen),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "cases": results,
        "handoff_parents": [result["parent_episode"] for result in results],
    }


def validate_manifest(
    manifest_path: str | Path, training_manifest_path: str | Path
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path, "native24 handoff manifest")
    result = validate_manifest_data(manifest, manifest_path.parent, training_manifest_path)
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def build_manifest(
    acquisition_index_path: str | Path,
    training_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    acquisition_index_path = Path(acquisition_index_path)
    training_manifest_path = Path(training_manifest_path)
    output_path = Path(output_path)
    acquisition = _read_json(acquisition_index_path, "native24 handoff acquisition index")
    if (
        acquisition.get("schema") != ACQUISITION_SCHEMA
        or acquisition.get("status") != "capture_complete"
        or acquisition.get("label_contract") != LABEL_CONTRACT
    ):
        raise ValueError("Native24 handoff acquisition is incomplete")
    training_sha = sha256_file(training_manifest_path)
    if acquisition.get("training_manifest_sha256") != training_sha:
        raise ValueError("Acquisition was not locked to this training cache")
    training_manifest, _, _ = _parent_rosters(training_manifest_path)
    training_source_hashes = {
        row["parent_episode"]["source_sha256"] for row in training_manifest["episodes"]
    }
    source_root = acquisition_index_path.parent
    registry_path = _artifact_path(
        source_root, acquisition.get("parent_registry_path"), "held-out parent registry"
    )
    registry_binding = {
        "path": _portable_relative(registry_path, output_path.parent, "held-out parent registry"),
        "sha256": sha256_file(registry_path),
    }
    _, registry_train, registry_validation = _load_parent_registry(
        output_path.parent, registry_binding
    )
    rows = acquisition.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Acquisition index has no cases")
    if {row.get("classification") for row in rows if isinstance(row, dict)} != REQUIRED_CLASSES:
        raise NotEvaluableError("Missing native24 wrong-handoff validation classes")
    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Acquisition case must be an object")
        case_id = row.get("case_id")
        classification = row.get("classification")
        if classification not in REQUIRED_CLASSES:
            raise ValueError("Unknown native24 handoff classification")
        parent = _handoff_parent(row.get("parent_episode"))
        key = _parent_key(parent)
        if key in registry_train or key not in registry_validation:
            raise ValueError("Acquisition case parent is not held out")
        if parent["source_sha256"] in training_source_hashes:
            raise ValueError("Held-out replay source overlaps the native24 training source")
        if row.get("policy_input_privileged") is not False:
            raise ValueError("Acquisition must attest nonprivileged policy inputs")
        if row.get("model_progress_used_as_ground_truth") is not False:
            raise ValueError("Acquisition cannot use model progress as ground truth")
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Acquisition case requires a nonempty frozen instruction")
        npz_path = _artifact_path(source_root, row.get("npz_path"), "handoff NPZ")
        trajectory_path = _artifact_path(
            source_root, row.get("source_trajectory_record_path"), "source trajectory record"
        )
        replay_path = _artifact_path(
            source_root, row.get("source_replay_record_path"), "source replay record"
        )
        npz_sha = sha256_file(npz_path)
        _, observation_digest = _load_observation(npz_path)
        trajectory_record = _validate_trajectory_record(trajectory_path, parent)
        registry_row = registry_validation[key]
        if (
            trajectory_record["source_events_sha256"] != registry_row["events_sha256"]
            or trajectory_record["source_result_sha256"] != registry_row["result_sha256"]
        ):
            raise ValueError("Source trajectory differs from the frozen parent registry")
        replay_record = _validate_replay_record(
            replay_path,
            case_id,
            instruction,
            parent,
            trajectory_record["trajectory_sha256"],
            npz_sha,
            observation_digest,
        )
        family = row.get("family")
        _validate_ground_truth(classification, family, replay_record.get("ground_truth"))
        case = {
            "case_id": case_id,
            "role": "validation",
            "split": "validation",
            "classification": classification,
            "family": family,
            "instruction": instruction,
            "label_contract": LABEL_CONTRACT,
            "observation_index": replay_record["observation_index"],
            "parent_episode": parent,
            "path": _portable_relative(npz_path, output_path.parent, "handoff NPZ"),
            "sha256": npz_sha,
            "source_trajectory": {
                "path": _portable_relative(
                    trajectory_path, output_path.parent, "source trajectory record"
                ),
                "sha256": sha256_file(trajectory_path),
                "trajectory_sha256": trajectory_record["trajectory_sha256"],
            },
            "source_replay": {
                "path": _portable_relative(replay_path, output_path.parent, "source replay record"),
                "sha256": sha256_file(replay_path),
            },
            "source_replay_verified": True,
            "policy_input_privileged": False,
            "model_progress_used_as_ground_truth": False,
            "ground_truth": replay_record["ground_truth"],
        }
        case["case_sha256"] = canonical_sha256(case)
        cases.append(case)
    manifest = {
        "schema": SCHEMA,
        "status": "verified_handoff_inputs",
        "label_contract": LABEL_CONTRACT,
        "training_data": {
            "schema": "fetch_native24_family_v1",
            "manifest_sha256": training_sha,
        },
        "parent_registry": registry_binding,
        "observation_contract": {
            "head_rgb": {"shape": [128, 128, 3], "dtype": "uint8"},
            "wrist_rgb": {"shape": [128, 128, 3], "dtype": "uint8"},
            "state": {"shape": [24], "dtype": "float32", "source": "env_native_agent"},
            "policy_input_privileged": False,
        },
        "classification_contract": {
            "schema": "bvi.native24-handoff-classification/1",
            "far_grasp_distance_m": CLASSIFICATION_DISTANCE_M,
            "ground_truth_source": "source_replay_physical_predicates",
            "physical_completion_required": False,
        },
        "offline_behavior_scope": {
            "status": "input_only_behavior_not_evaluated",
            "single_observation_per_case": True,
            "thresholds_bound": False,
            "false_completion_evaluable": False,
            "stagnation_evaluable": False,
            "rollback_evaluable": False,
        },
        "cases": cases,
    }
    validate_manifest_data(manifest, output_path.parent, training_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return validate_manifest(output_path, training_manifest_path)
