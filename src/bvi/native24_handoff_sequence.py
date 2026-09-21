"""Contracts for the zero-training native24 wrong-handoff sequence gate.

The sequence gate is separate from the single-frame input gate.  It binds the
existing progress-monitor source before any model query and requires at least
ten contiguous observations per class, enabling the frozen two-hit, drop and
stagnation rules to be replayed without claiming native task success.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import progress_monitor
from .native24_handoff import (
    CLASSIFICATION_DISTANCE_M,
    LABEL_CONTRACT,
    NotEvaluableError,
    REQUIRED_CLASSES,
    _artifact_path,
    _handoff_parent,
    _load_parent_registry,
    _parent_key,
    _parent_rosters,
    _portable_relative,
    _validate_trajectory_record,
    canonical_sha256,
    observation_sha256,
    sha256_file,
)


SCHEMA = "bvi.native24-handoff-sequence/1"
ACQUISITION_SCHEMA = "bvi.native24-handoff-sequence-acquisition/1"
REPLAY_SCHEMA = "bvi.native24-source-sequence-replay/1"
PREDICATE_SCHEMA = "bvi.native24-physical-predicate-sequence/1"
MIN_SEQUENCE_LENGTH = 10
NEGATIVE_COMPLETION_CLASSES = frozenset(
    {"far_grasp_diagnostic", "near_grasp_candidate_not_completion", "unheld_move"}
)
FALSE_COMPLETION_FORBIDDEN_CLASSES = frozenset(REQUIRED_CLASSES)
FIXED_WINDOWS = {
    "validation-seed3020-grasp-step17": (8, 17),
    "validation-seed3020-move-step20": (11, 20),
    "validation-seed3021-grasp-step31": (22, 31),
    "validation-seed3021-move-step34": (25, 34),
}
INVOCATION_INDEX = {"grasp": 1, "move": 2}


def monitor_contract(source_path: str | Path) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    actual = Path(inspect.getfile(progress_monitor)).resolve()
    if source_path != actual:
        raise ValueError("Progress-monitor source path differs from the imported implementation")
    thresholds = dict(progress_monitor.THRESHOLDS)
    if thresholds != {"reach": 0.9, "grasp": 0.6, "move": 0.9, "release": 0.6}:
        raise ValueError("Frozen progress thresholds changed")
    return {
        "schema": "bvi.progress-monitor-contract/1",
        "source_sha256": sha256_file(actual),
        "evaluation_frequency": "once_per_policy_prediction",
        "thresholds": thresholds,
        "consecutive_threshold_hits": 2,
        "drop_after_minimum_history": 4,
        "drop_delta_strictly_greater_than": 0.03,
        "stagnation_minimum_predictions": 10,
        "stagnation_delta_strictly_less_than": 0.03,
        "drop_and_stagnation_require_invocation_index_gt": 0,
        "replan_cooldown_required": 0,
        "comparison_semantics": "exact_runtime_python_float_no_rounding_no_epsilon",
        "chunk_progress_consumed_index": 0,
        "threshold_precedes_drop_and_stagnation": True,
        "cooldown_decremented_before_checks": True,
    }


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"head_rgb", "wrist_rgb", "state"}:
            raise ValueError("Sequence NPZ must contain only nonprivileged native24 policy inputs")
        values = {key: archive[key].copy() for key in archive.files}
    length = len(values["state"])
    expected = {
        "head_rgb": (length, 128, 128, 3),
        "wrist_rgb": (length, 128, 128, 3),
        "state": (length, 24),
    }
    if length < MIN_SEQUENCE_LENGTH:
        raise NotEvaluableError("Native24 handoff sequence has fewer than ten observations")
    for key, shape in expected.items():
        value = values[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid sequence {key} shape/finiteness")
        if key.endswith("rgb") and value.dtype != np.uint8:
            raise ValueError("Sequence RGB must be uint8")
    if values["state"].dtype != np.float32:
        raise ValueError("Sequence state24 must be float32")
    hashes = [
        observation_sha256({key: values[key][index] for key in values})
        for index in range(length)
    ]
    return values, hashes


def _load_predicates(path: Path, sequence_id: str, window: Mapping[str, Any]) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema") != PREDICATE_SCHEMA
        or document.get("status") != "source_replay_physical_predicates"
        or document.get("sequence_id") != sequence_id
    ):
        raise ValueError("Physical predicate sequence identity differs")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != window["length"]:
        raise ValueError("Physical predicate sequence length differs")
    indices = []
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {
            "observation_index", "tcp_object_distance_m", "is_grasped",
            "physical_completion", "terminated", "truncated",
        }:
            raise ValueError("Physical predicate frame fields differ")
        index = frame["observation_index"]
        distance = frame["tcp_object_distance_m"]
        if type(index) is not int or isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError("Invalid physical predicate frame")
        if not math.isfinite(float(distance)) or float(distance) < 0:
            raise ValueError("Invalid physical distance")
        if any(type(frame[name]) is not bool for name in (
            "is_grasped", "physical_completion", "terminated", "truncated"
        )):
            raise ValueError("Physical predicates must be booleans")
        if frame["physical_completion"] or frame["terminated"] or frame["truncated"]:
            raise ValueError("Sequence gate accepts only pre-completion, nonterminal frames")
        indices.append(index)
    expected = list(range(window["start"], window["end"] + 1))
    if indices != expected:
        raise ValueError("Sequence observations are not contiguous or window-bound")
    return frames


def _validate_class_sequence(case: Mapping[str, Any], frames: list[dict[str, Any]]) -> None:
    classification = case["classification"]
    anchor = next(
        (frame for frame in frames if frame["observation_index"] == case["anchor_observation_index"]),
        None,
    )
    if anchor is None:
        raise ValueError("Anchor observation is outside the sequence window")
    distance, held = float(anchor["tcp_object_distance_m"]), anchor["is_grasped"]
    if classification == "far_grasp_diagnostic" and (held or distance < CLASSIFICATION_DISTANCE_M):
        raise ValueError("Far-grasp anchor contradicts physical predicates")
    if classification == "near_grasp_candidate_not_completion" and (
        held or distance >= CLASSIFICATION_DISTANCE_M
    ):
        raise ValueError("Near-grasp anchor contradicts physical predicates")
    if classification == "unheld_move" and held:
        raise ValueError("Unheld-move anchor contradicts physical predicates")
    if classification == "held_move" and not held:
        raise ValueError("Held-move anchor contradicts physical predicates")
    if classification in {"far_grasp_diagnostic", "near_grasp_candidate_not_completion"}:
        if any(frame["is_grasped"] for frame in frames):
            raise ValueError("Negative grasp sequence contains a completed grasp")
    if classification == "unheld_move" and any(frame["is_grasped"] for frame in frames):
        raise ValueError("Unheld move sequence contains a held frame")
    # A sequence is a fixed causal tail ending at the classified handoff.  Only
    # its final/anchor frame owns the class label; earlier frames need not.  In
    # particular the held-move tail intentionally contains pre-grasp frames.


def _validate_case(
    case: Mapping[str, Any],
    root: Path,
    registry_train: Mapping[str, dict],
    registry_validation: Mapping[str, dict],
    training_source_hashes: set[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    sequence_id = case.get("sequence_id")
    if not isinstance(sequence_id, str) or Path(sequence_id).name != sequence_id:
        raise ValueError("Invalid sequence id")
    if case.get("case_sha256") != canonical_sha256(
        {key: value for key, value in case.items() if key != "case_sha256"}
    ):
        raise ValueError("Sequence case metadata hash mismatch")
    if case.get("role") != "validation" or case.get("split") != "validation":
        raise ValueError("Sequence case is not held out")
    if case.get("label_contract") != LABEL_CONTRACT:
        raise ValueError("Sequence case label contract differs")
    classification = case.get("classification")
    if classification not in REQUIRED_CLASSES:
        raise ValueError("Unknown sequence classification")
    family = case.get("family")
    if family != ("grasp" if "grasp" in classification else "move"):
        raise ValueError("Sequence family/classification mismatch")
    if case.get("policy_input_privileged") is not False or case.get(
        "model_progress_used_as_ground_truth"
    ) is not False:
        raise ValueError("Sequence policy inputs/ground truth provenance differs")
    instruction = case.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Sequence instruction is missing")
    invocation_index = case.get("invocation_index")
    if type(invocation_index) is not int or invocation_index != INVOCATION_INDEX[family]:
        raise ValueError("Sequence invocation index differs from the fixed family order")
    parent = _handoff_parent(case.get("parent_episode"))
    key = _parent_key(parent)
    if key in registry_train or key not in registry_validation:
        raise ValueError("Sequence parent is not in the frozen validation partition")
    if parent["source_sha256"] in training_source_hashes:
        raise ValueError("Sequence source overlaps the native24 training source")
    expected_window = FIXED_WINDOWS.get(case.get("anchor_case_id"))
    window = case.get("window")
    if not isinstance(window, dict) or expected_window is None or window != {
        "start": expected_window[0],
        "end": expected_window[1],
        "length": expected_window[1] - expected_window[0] + 1,
        "selection_rule": "previous_9_through_anchor_source_observations",
    }:
        raise ValueError("Sequence window differs from the pre-registered fixed window")
    if case.get("anchor_observation_index") != expected_window[1]:
        raise ValueError("Sequence anchor index differs")
    npz_path = _artifact_path(root, case.get("path"), "native24 sequence NPZ")
    if sha256_file(npz_path) != case.get("sha256"):
        raise ValueError("Native24 sequence NPZ hash mismatch")
    values, frame_hashes = _load_npz(npz_path)
    if len(set(frame_hashes)) != len(frame_hashes):
        raise ValueError("Sequence contains a repeated native24 observation")
    predicate_binding = case.get("physical_predicates")
    if not isinstance(predicate_binding, dict) or set(predicate_binding) != {"path", "sha256"}:
        raise ValueError("Physical predicate binding is incomplete")
    predicate_path = _artifact_path(root, predicate_binding["path"], "physical predicates")
    if sha256_file(predicate_path) != predicate_binding["sha256"]:
        raise ValueError("Physical predicate artifact hash mismatch")
    frames = _load_predicates(predicate_path, sequence_id, window)
    _validate_class_sequence(case, frames)
    trajectory_binding = case.get("source_trajectory")
    if not isinstance(trajectory_binding, dict) or set(trajectory_binding) != {
        "path", "sha256", "trajectory_sha256"
    }:
        raise ValueError("Sequence trajectory binding is incomplete")
    trajectory_path = _artifact_path(root, trajectory_binding["path"], "source trajectory record")
    if sha256_file(trajectory_path) != trajectory_binding["sha256"]:
        raise ValueError("Sequence trajectory record hash mismatch")
    trajectory = _validate_trajectory_record(trajectory_path, parent)
    registry = registry_validation[key]
    if (
        trajectory["source_events_sha256"] != registry["events_sha256"]
        or trajectory["source_result_sha256"] != registry["result_sha256"]
        or trajectory_binding["trajectory_sha256"] != trajectory["trajectory_sha256"]
    ):
        raise ValueError("Sequence trajectory differs from the frozen registry")
    replay_binding = case.get("source_replay")
    if not isinstance(replay_binding, dict) or set(replay_binding) != {"path", "sha256"}:
        raise ValueError("Sequence replay binding is incomplete")
    replay_path = _artifact_path(root, replay_binding["path"], "source sequence replay")
    if sha256_file(replay_path) != replay_binding["sha256"]:
        raise ValueError("Source sequence replay hash mismatch")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (
        replay.get("schema") != REPLAY_SCHEMA
        or replay.get("status") != "verified_source_sequence_replay"
        or replay.get("sequence_id") != sequence_id
        or replay.get("parent_episode") != parent
        or replay.get("trajectory_sha256") != trajectory["trajectory_sha256"]
        or replay.get("instruction") != instruction
        or replay.get("window") != window
        or replay.get("npz_sha256") != case["sha256"]
        or replay.get("physical_predicates_sha256") != predicate_binding["sha256"]
        or replay.get("frame_observation_sha256") != frame_hashes
        or replay.get("sequence_observation_sha256") != canonical_sha256(frame_hashes)
        or replay.get("physical_predicate_sequence_sha256") != canonical_sha256(frames)
    ):
        raise ValueError("Source sequence replay identity/hash binding differs")
    reconstructions = replay.get("reconstruction_frame_observation_sha256")
    if not isinstance(reconstructions, list) or len(reconstructions) < 2 or any(
        item != frame_hashes for item in reconstructions
    ):
        raise ValueError("Sequence requires two identical reconstructions")
    reconstructed_predicates = replay.get("reconstruction_physical_predicate_sha256")
    if not isinstance(reconstructed_predicates, list) or len(reconstructed_predicates) < 2 or any(
        item != canonical_sha256(frames) for item in reconstructed_predicates
    ):
        raise ValueError("Sequence requires two identical predicate reconstructions")
    checks = (
        "initial_state_verified", "action_prefix_verified",
        "boundary_observation_verified", "physical_predicates_verified",
    )
    if any(replay.get(name) is not True for name in checks):
        raise ValueError("Sequence replay physical verification is incomplete")
    if replay.get("physical_atol") != 1e-5 or replay.get("policy_calls") != 0 or replay.get(
        "training_updates"
    ) != 0:
        raise ValueError("Sequence replay scope/tolerance differs")
    return ({
        "sequence_id": sequence_id,
        "anchor_case_id": case["anchor_case_id"],
        "anchor_observation_index": case["anchor_observation_index"],
        "classification": classification,
        "family": family,
        "instruction": instruction,
        "invocation_index": invocation_index,
        "parent_episode": parent,
        "window": window,
        "frame_observation_sha256": frame_hashes,
        "sequence_observation_sha256": canonical_sha256(frame_hashes),
        "negative_completion_case": classification in NEGATIVE_COMPLETION_CLASSES,
        "learned_threshold_forbidden_while_physical_completion_false": (
            classification in FALSE_COMPLETION_FORBIDDEN_CLASSES
        ),
        "shadow_query_cadence": "one_query_per_contiguous_source_replay_observation",
        "starts_at_deployed_family_invocation": False,
    }, values, frames)


def load_manifest(
    manifest_path: str | Path,
    training_manifest_path: str | Path,
    input_manifest_path: str | Path,
    progress_monitor_source: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict[str, list[dict]]]:
    manifest_path = Path(manifest_path)
    training_manifest_path, input_manifest_path = Path(training_manifest_path), Path(input_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "verified_sequence_inputs":
        raise ValueError("Native24 handoff sequence manifest is not verified")
    if manifest.get("label_contract") != LABEL_CONTRACT:
        raise ValueError("Sequence label contract differs")
    expected_monitor = monitor_contract(progress_monitor_source)
    if manifest.get("monitor_contract") != expected_monitor:
        raise ValueError("Progress monitor was not frozen before sequence evaluation")
    preregistration_binding = manifest.get("preregistration")
    if not isinstance(preregistration_binding, dict) or set(preregistration_binding) != {
        "path", "sha256", "canonical_sha256"
    }:
        raise ValueError("Sequence preregistration binding is incomplete")
    preregistration_path = _artifact_path(
        manifest_path.parent, preregistration_binding["path"], "sequence preregistration"
    )
    if sha256_file(preregistration_path) != preregistration_binding["sha256"]:
        raise ValueError("Sequence preregistration hash mismatch")
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    claimed_preregistration_hash = preregistration.get("preregistration_sha256")
    if (
        preregistration.get("schema") != "bvi.native24-handoff-sequence-preregistration/1"
        or preregistration.get("status") != "frozen_before_render_or_model_query"
        or preregistration.get("monitor_contract") != expected_monitor
        or preregistration.get("training_manifest_sha256") != sha256_file(training_manifest_path)
        or preregistration.get("input_manifest_sha256") != sha256_file(input_manifest_path)
        or claimed_preregistration_hash != canonical_sha256({
            key: value for key, value in preregistration.items()
            if key != "preregistration_sha256"
        })
        or preregistration_binding["canonical_sha256"] != claimed_preregistration_hash
    ):
        raise ValueError("Sequence preregistration content/hash differs")
    training, _, _ = _parent_rosters(training_manifest_path)
    if manifest.get("training_data") != {
        "schema": "fetch_native24_family_v1",
        "manifest_sha256": sha256_file(training_manifest_path),
    }:
        raise ValueError("Sequence manifest is not bound to the training cache")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("schema") != "bvi.native24-handoff/1" or input_manifest.get(
        "status"
    ) != "verified_handoff_inputs":
        raise ValueError("Sequence source is not the verified single-frame B input gate")
    if manifest.get("input_gate") != {
        "schema": "bvi.native24-handoff/1",
        "manifest_sha256": sha256_file(input_manifest_path),
    }:
        raise ValueError("Sequence manifest is not bound to the B input gate")
    expected_scope = {
        "optimizer_updates": 0,
        "simulator_steps_during_offline_evaluation": 0,
        "native_success_evaluated": False,
        "transport_evaluated": False,
        "deployment_cadence_evaluated": False,
        "physical_stagnation_correctness_evaluable": False,
        "physical_rollback_correctness_evaluable": False,
        "full_behavior_admission_evaluable": False,
        "shadow_query_cadence": "one_query_per_contiguous_source_replay_observation",
    }
    if manifest.get("evaluation_scope") != expected_scope:
        raise ValueError("Sequence evaluation scope differs")
    anchors = {row.get("case_id"): row for row in input_manifest.get("cases", [])}
    training_sources = {row["parent_episode"]["source_sha256"] for row in training["episodes"]}
    _, registry_train, registry_validation = _load_parent_registry(
        manifest_path.parent, manifest.get("parent_registry")
    )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Sequence manifest has no cases")
    validated, tables, predicates = [], {}, {}
    for case in cases:
        row, values, frames = _validate_case(
            case, manifest_path.parent, registry_train, registry_validation, training_sources
        )
        anchor = anchors.get(row["anchor_case_id"])
        if anchor is None or any(
            row.get(name) != anchor.get(anchor_name)
            for name, anchor_name in (
                ("classification", "classification"), ("family", "family"),
                ("instruction", "instruction"), ("parent_episode", "parent_episode"),
            )
        ) or row["window"]["end"] != anchor.get("observation_index"):
            raise ValueError("Sequence case differs from its verified B input anchor")
        validated.append(row); tables[row["sequence_id"]] = values; predicates[row["sequence_id"]] = frames
    if {row["classification"] for row in validated} != REQUIRED_CLASSES:
        raise NotEvaluableError("Missing native24 handoff sequence class")
    if len({row["sequence_id"] for row in validated}) != len(validated):
        raise ValueError("Duplicate native24 handoff sequence")
    preregistered = {
        row.get("anchor_case_id"): row for row in preregistration.get("sequences", [])
    }
    if set(preregistered) != {row["anchor_case_id"] for row in validated}:
        raise ValueError("Preregistered and captured sequence rosters differ")
    for row in validated:
        frozen = preregistered[row["anchor_case_id"]]
        if any(row.get(name) != frozen.get(name) for name in (
            "sequence_id", "classification", "family", "instruction", "invocation_index",
            "anchor_observation_index", "parent_episode", "window",
        )) or frozen.get("replan_cooldown") != 0 or frozen.get(
            "forbidden_events"
        ) != ["learned_threshold"]:
            raise ValueError("Captured sequence differs from preregistered adjudication inputs")
    report = {
        "schema": SCHEMA,
        "status": "verified_sequence_inputs",
        "manifest_sha256": sha256_file(manifest_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "monitor_contract": expected_monitor,
        "sequence_count": len(validated),
        "frame_count": sum(row["window"]["length"] for row in validated),
        "negative_false_completion_evaluable": True,
        "false_completion_scope": (
            "every learned_threshold frame with physical_completion=false, including held_move"
        ),
        "monitor_stagnation_signal_mechanics_evaluable": all(
            row["window"]["length"] >= 10 for row in validated
        ),
        "monitor_drop_signal_mechanics_evaluable": all(
            row["window"]["length"] >= 4 and row["invocation_index"] > 0 for row in validated
        ),
        "positive_two_hit_correctness_evaluable": False,
        "physical_stagnation_correctness_evaluable": False,
        "physical_rollback_correctness_evaluable": False,
        "deployment_cadence_evaluated": False,
        "shadow_query_cadence": "one_query_per_contiguous_source_replay_observation",
        "full_behavior_admission_evaluable": False,
        "native_success_evaluated": False,
    }
    return report, validated, tables, predicates


def build_manifest(
    acquisition_path: str | Path,
    training_manifest_path: str | Path,
    input_manifest_path: str | Path,
    progress_monitor_source: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    acquisition_path, output_path = Path(acquisition_path), Path(output_path)
    input_manifest_path = Path(input_manifest_path)
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if (
        acquisition.get("schema") != ACQUISITION_SCHEMA
        or acquisition.get("status") != "capture_complete"
        or acquisition.get("label_contract") != LABEL_CONTRACT
    ):
        raise ValueError("Sequence acquisition is incomplete")
    if acquisition.get("training_manifest_sha256") != sha256_file(training_manifest_path):
        raise ValueError("Sequence acquisition training-cache binding differs")
    expected_monitor = monitor_contract(progress_monitor_source)
    if acquisition.get("monitor_contract") != expected_monitor:
        raise ValueError("Sequence acquisition did not bind the frozen monitor contract")
    if (
        acquisition.get("exact_simulator_actions") != 108
        or acquisition.get("hard_simulator_action_cap") != 136
        or acquisition.get("policy_calls") != 0
        or acquisition.get("training_updates") != 0
    ):
        raise ValueError("Sequence acquisition scope/budget differs")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("schema") != "bvi.native24-handoff/1"
        or input_manifest.get("status") != "verified_handoff_inputs"
        or acquisition.get("input_manifest_sha256") != sha256_file(input_manifest_path)
    ):
        raise ValueError("Sequence acquisition B-input binding differs")
    anchors = {row.get("case_id"): row for row in input_manifest.get("cases", [])}
    rows = acquisition.get("cases")
    if not isinstance(rows, list) or {row.get("classification") for row in rows} != REQUIRED_CLASSES:
        raise NotEvaluableError("Missing native24 handoff sequence class")
    root = acquisition_path.parent
    preregistration_path = _artifact_path(
        root, acquisition.get("preregistration_path"), "sequence preregistration"
    )
    if sha256_file(preregistration_path) != acquisition.get("preregistration_sha256"):
        raise ValueError("Sequence preregistration hash differs")
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if (
        preregistration.get("schema") != "bvi.native24-handoff-sequence-preregistration/1"
        or preregistration.get("status") != "frozen_before_render_or_model_query"
        or preregistration.get("monitor_contract") != expected_monitor
        or preregistration.get("training_manifest_sha256") != sha256_file(training_manifest_path)
        or preregistration.get("input_manifest_sha256") != sha256_file(input_manifest_path)
        or preregistration.get("preregistration_sha256") != canonical_sha256({
            key: value for key, value in preregistration.items()
            if key != "preregistration_sha256"
        })
    ):
        raise ValueError("Sequence preregistration identity/scope differs")
    preregistered = {
        row.get("anchor_case_id"): row for row in preregistration.get("sequences", [])
    }
    if set(preregistered) != {row.get("anchor_case_id") for row in rows}:
        raise ValueError("Acquired and preregistered sequence rosters differ")
    registry_path = _artifact_path(root, acquisition.get("parent_registry_path"), "parent registry")
    cases = []
    for row in rows:
        anchor = anchors.get(row.get("anchor_case_id"))
        if anchor is None or any(
            row.get(name) != anchor.get(anchor_name)
            for name, anchor_name in (
                ("classification", "classification"), ("family", "family"),
                ("instruction", "instruction"), ("anchor_observation_index", "observation_index"),
                ("parent_episode", "parent_episode"),
            )
        ):
            raise ValueError("Sequence anchor differs from the verified B input gate")
        frozen = preregistered[row["anchor_case_id"]]
        if any(row.get(name) != frozen.get(name) for name in (
            "sequence_id", "classification", "family", "instruction", "invocation_index",
            "anchor_observation_index", "parent_episode", "window",
        )) or frozen.get("replan_cooldown") != 0 or frozen.get(
            "forbidden_events"
        ) != ["learned_threshold"]:
            raise ValueError("Acquired sequence differs from preregistration")
        case = {
            key: row[key]
            for key in (
                "sequence_id", "anchor_case_id", "classification", "family", "instruction",
                "invocation_index", "anchor_observation_index", "parent_episode", "window",
            )
        }
        case.update(
            role="validation", split="validation", label_contract=LABEL_CONTRACT,
            policy_input_privileged=False, model_progress_used_as_ground_truth=False,
        )
        for source_key, target_key in (
            ("npz_path", "path"),
            ("physical_predicates_path", "physical_predicates"),
            ("source_replay_record_path", "source_replay"),
        ):
            path = _artifact_path(root, row[source_key], source_key)
            relative = _portable_relative(path, output_path.parent, source_key)
            if target_key == "path":
                case["path"] = relative; case["sha256"] = sha256_file(path)
            else:
                case[target_key] = {"path": relative, "sha256": sha256_file(path)}
        trajectory_path = _artifact_path(root, row["source_trajectory_record_path"], "trajectory")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        case["source_trajectory"] = {
            "path": _portable_relative(trajectory_path, output_path.parent, "trajectory"),
            "sha256": sha256_file(trajectory_path),
            "trajectory_sha256": trajectory.get("trajectory_sha256"),
        }
        case["case_sha256"] = canonical_sha256(case)
        cases.append(case)
    manifest = {
        "schema": SCHEMA,
        "status": "verified_sequence_inputs",
        "label_contract": LABEL_CONTRACT,
        "training_data": {
            "schema": "fetch_native24_family_v1",
            "manifest_sha256": sha256_file(training_manifest_path),
        },
        "input_gate": {
            "schema": "bvi.native24-handoff/1",
            "manifest_sha256": sha256_file(input_manifest_path),
        },
        "parent_registry": {
            "path": _portable_relative(registry_path, output_path.parent, "parent registry"),
            "sha256": sha256_file(registry_path),
        },
        "monitor_contract": expected_monitor,
        "preregistration": {
            "path": _portable_relative(preregistration_path, output_path.parent, "preregistration"),
            "sha256": sha256_file(preregistration_path),
            "canonical_sha256": preregistration["preregistration_sha256"],
        },
        "evaluation_scope": {
            "optimizer_updates": 0,
            "simulator_steps_during_offline_evaluation": 0,
            "native_success_evaluated": False,
            "transport_evaluated": False,
            "deployment_cadence_evaluated": False,
            "physical_stagnation_correctness_evaluable": False,
            "physical_rollback_correctness_evaluable": False,
            "full_behavior_admission_evaluable": False,
            "shadow_query_cadence": "one_query_per_contiguous_source_replay_observation",
        },
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
    report, _, _, _ = load_manifest(
        output_path, training_manifest_path, input_manifest_path, progress_monitor_source
    )
    return report
