import json
from pathlib import Path

import numpy as np
import pytest

from bvi.native24_handoff import (
    ACQUISITION_SCHEMA,
    LABEL_CONTRACT,
    PARENT_REGISTRY_SCHEMA,
    REPLAY_SCHEMA,
    TRAJECTORY_SCHEMA,
    build_manifest,
    canonical_sha256,
    observation_sha256,
    sha256_file,
    validate_manifest,
)


CLASSES = (
    "far_grasp_diagnostic",
    "near_grasp_candidate_not_completion",
    "unheld_move",
    "held_move",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parent(parent_id: int) -> dict:
    return {
        "task": "pick",
        "source_sha256": canonical_sha256({"source": "pick-h5"}),
        "trajectory": f"traj_{parent_id}",
        "parent_id": parent_id,
    }


def handoff_parent(seed: int, split: str = "validation") -> dict:
    return {
        "source_kind": "archived_online_replay",
        "task": "pick",
        "source_sha256": canonical_sha256({"events": seed}),
        "trajectory": f"{split}-seed{seed}",
        "parent_id": seed,
    }


def make_fixture(root: Path) -> tuple[Path, Path]:
    training = root / "training-manifest.json"
    write_json(
        training,
        {
            "schema": "fetch_native24_family_v1",
            "label_contract": LABEL_CONTRACT,
            "episodes": [
                {"split": "train", "parent_episode": parent(0)},
                {"split": "validation", "parent_episode": parent(20)},
            ],
        },
    )
    registry_path = root / "parent-registry.json"
    registry_rows = []
    for split, seeds in (("train", range(3000, 3004)), ("validation", (3020, 3021))):
        for seed in seeds:
            replay_parent = handoff_parent(seed, split)
            registry_rows.append(
                {
                    "split": split,
                    "parent_episode": replay_parent,
                    "events_sha256": replay_parent["source_sha256"],
                    "result_sha256": canonical_sha256({"result": seed}),
                }
            )
    write_json(
        registry_path,
        {
            "schema": PARENT_REGISTRY_SCHEMA,
            "status": "verified_parent_partitions",
            "source_batch": "fetch-current-handoffs-2026-09-17-run01",
            "collection_index_sha256": canonical_sha256("collection-index"),
            "batch_sha256": canonical_sha256("batch"),
            "parents": registry_rows,
        },
    )
    rows = []
    for offset, classification in enumerate(CLASSES):
        seed = 3020 if classification in {
            "near_grasp_candidate_not_completion",
            "held_move",
        } else 3021
        parent_episode = handoff_parent(seed)
        case_id = f"heldout-{classification}"
        family = "grasp" if "grasp" in classification else "move"
        instruction = "Grasp the apple." if family == "grasp" else "Move the held apple."
        case_dir = root / "cases" / case_id
        case_dir.mkdir(parents=True)
        head = np.full((128, 128, 3), offset, dtype=np.uint8)
        wrist = np.full((128, 128, 3), offset + 1, dtype=np.uint8)
        state = np.arange(24, dtype=np.float32) + offset
        npz_path = case_dir / "observation.npz"
        np.savez_compressed(npz_path, head_rgb=head, wrist_rgb=wrist, state=state)
        trajectory_sha = parent_episode["source_sha256"]
        trajectory_path = case_dir / "source-trajectory.json"
        write_json(
            trajectory_path,
            {
                "schema": TRAJECTORY_SCHEMA,
                "status": "verified_source_trajectory",
                "parent_episode": parent_episode,
                "source_events_sha256": parent_episode["source_sha256"],
                "source_result_sha256": canonical_sha256({"result": seed}),
                "digest_contract": "historical-events-jsonl-v1",
                "trajectory_sha256": trajectory_sha,
            },
        )
        held = classification == "held_move"
        distance = 0.25 if classification == "far_grasp_diagnostic" else 0.04
        replay_path = case_dir / "source-replay.json"
        observation_digest = observation_sha256(
            {"head_rgb": head, "wrist_rgb": wrist, "state": state}
        )
        write_json(
            replay_path,
            {
                "schema": REPLAY_SCHEMA,
                "status": "verified_source_replay",
                "case_id": case_id,
                "instruction": instruction,
                "parent_episode": parent_episode,
                "trajectory_sha256": trajectory_sha,
                "observation_index": 3 + offset,
                "npz_sha256": sha256_file(npz_path),
                "boundary_observation_sha256": observation_digest,
                "reconstruction_observation_sha256": [observation_digest, observation_digest],
                "physical_atol": 1e-5,
                "initial_state_verified": True,
                "action_prefix_verified": True,
                "boundary_observation_verified": True,
                "physical_predicates_verified": True,
                "policy_calls": 0,
                "training_updates": 0,
                "ground_truth": {
                    "source": "source_replay_physical_predicates",
                    "tcp_object_distance_m": distance,
                    "is_grasped": held,
                    "physical_completion": False,
                },
            },
        )
        rows.append(
            {
                "case_id": case_id,
                "classification": classification,
                "family": family,
                "instruction": instruction,
                "parent_episode": parent_episode,
                "policy_input_privileged": False,
                "model_progress_used_as_ground_truth": False,
                "npz_path": npz_path.relative_to(root).as_posix(),
                "source_trajectory_record_path": trajectory_path.relative_to(root).as_posix(),
                "source_replay_record_path": replay_path.relative_to(root).as_posix(),
            }
        )
    acquisition = root / "acquisition.json"
    write_json(
        acquisition,
        {
            "schema": ACQUISITION_SCHEMA,
            "status": "capture_complete",
            "label_contract": LABEL_CONTRACT,
            "training_manifest_sha256": sha256_file(training),
            "parent_registry_path": registry_path.relative_to(root).as_posix(),
            "cases": rows,
        },
    )
    return training, acquisition


def test_build_and_validate_four_class_native24_manifest(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    built = build_manifest(acquisition, training, output)
    assert built["case_count"] == 4
    assert set(built["classifications"]) == set(CLASSES)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema"] == "bvi.native24-handoff/1"
    assert manifest["offline_behavior_scope"] == {
        "status": "input_only_behavior_not_evaluated",
        "single_observation_per_case": True,
        "thresholds_bound": False,
        "false_completion_evaluable": False,
        "stagnation_evaluable": False,
        "rollback_evaluable": False,
    }
    assert all(case["split"] == "validation" for case in manifest["cases"])
    assert all(case["source_replay_verified"] for case in manifest["cases"])
    assert validate_manifest(output, training)["manifest_sha256"] == sha256_file(output)


def test_training_parent_is_rejected_before_artifact_admission(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    value["cases"][0]["parent_episode"] = handoff_parent(3000, "train")
    write_json(acquisition, value)
    with pytest.raises(ValueError, match="not held out"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_h5_validation_parent_cannot_claim_exact_source_replay(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    value["cases"][0]["parent_episode"] = parent(20)
    write_json(acquisition, value)
    with pytest.raises(ValueError, match="Handoff parent must have exactly"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_changed_npz_fails_closed_after_manifest_is_built(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    build_manifest(acquisition, training, output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    npz_path = tmp_path / manifest["cases"][0]["path"]
    np.savez_compressed(
        npz_path,
        head_rgb=np.ones((128, 128, 3), np.uint8),
        wrist_rgb=np.zeros((128, 128, 3), np.uint8),
        state=np.zeros(24, np.float32),
    )
    with pytest.raises(ValueError, match="NPZ hash mismatch"):
        validate_manifest(output, training)


def test_state30_or_extra_privileged_payload_is_rejected(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    npz_path = tmp_path / value["cases"][0]["npz_path"]
    np.savez_compressed(
        npz_path,
        head_rgb=np.zeros((128, 128, 3), np.uint8),
        wrist_rgb=np.zeros((128, 128, 3), np.uint8),
        state=np.zeros(30, np.float32),
        goal_pose=np.zeros(7, np.float32),
    )
    with pytest.raises(ValueError, match="must contain only"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_missing_class_is_not_evaluable(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    value["cases"].pop()
    write_json(acquisition, value)
    with pytest.raises(ValueError, match="Missing native24"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_replay_verified_cannot_be_supplied_as_an_unbound_boolean(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    replay_path = tmp_path / value["cases"][0]["source_replay_record_path"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["reconstruction_observation_sha256"] = [replay["boundary_observation_sha256"]]
    replay["source_replay_verified"] = True
    write_json(replay_path, replay)
    with pytest.raises(ValueError, match="two identical"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_case_metadata_hash_detects_relabeling(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    build_manifest(acquisition, training, output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest["cases"][0]["instruction"] = "Changed after locking."
    write_json(output, manifest)
    with pytest.raises(ValueError, match="Case metadata hash mismatch"):
        validate_manifest(output, training)


def test_classification_must_match_physical_predicates(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    replay_path = tmp_path / value["cases"][0]["source_replay_record_path"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["ground_truth"]["tcp_object_distance_m"] = 0.01
    write_json(replay_path, replay)
    with pytest.raises(ValueError, match="Far grasp classification contradicts"):
        build_manifest(acquisition, training, tmp_path / "manifest.json")


def test_changed_training_roster_invalidates_existing_manifest(tmp_path):
    training, acquisition = make_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    build_manifest(acquisition, training, output)
    value = json.loads(training.read_text(encoding="utf-8"))
    value["episodes"].append({"split": "validation", "parent_episode": parent(24)})
    write_json(training, value)
    with pytest.raises(ValueError, match="not bound to this training cache"):
        validate_manifest(output, training)


def test_artifacts_cannot_escape_manifest_directory(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    training, acquisition = make_fixture(evidence)
    value = json.loads(acquisition.read_text(encoding="utf-8"))
    outside = tmp_path / "outside.npz"
    outside.write_bytes((evidence / value["cases"][0]["npz_path"]).read_bytes())
    value["cases"][0]["npz_path"] = "../outside.npz"
    write_json(acquisition, value)
    with pytest.raises(ValueError, match="escapes"):
        build_manifest(acquisition, training, evidence / "manifest.json")
