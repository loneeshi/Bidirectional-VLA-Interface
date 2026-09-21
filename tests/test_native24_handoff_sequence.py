import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from bvi import progress_monitor
from bvi.native24_handoff import (
    LABEL_CONTRACT,
    PARENT_REGISTRY_SCHEMA,
    TRAJECTORY_SCHEMA,
    NotEvaluableError,
    canonical_sha256,
    observation_sha256,
    sha256_file,
)
from bvi.native24_handoff_sequence import (
    ACQUISITION_SCHEMA,
    PREDICATE_SCHEMA,
    REPLAY_SCHEMA,
    build_manifest,
    load_manifest,
    monitor_contract,
)


MONITOR_SOURCE = Path(inspect.getfile(progress_monitor)).resolve()
CASE_SPECS = (
    (
        "validation-seed3020-grasp-step17",
        "near_grasp_candidate_not_completion",
        3020,
        8,
        17,
    ),
    ("validation-seed3020-move-step20", "held_move", 3020, 11, 20),
    ("validation-seed3021-grasp-step31", "far_grasp_diagnostic", 3021, 22, 31),
    ("validation-seed3021-move-step34", "unheld_move", 3021, 25, 34),
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def training_parent(parent_id: int, source_sha256: str | None = None) -> dict:
    return {
        "task": "pick",
        "source_sha256": source_sha256 or canonical_sha256({"training": parent_id}),
        "trajectory": f"traj_{parent_id}",
        "parent_id": parent_id,
    }


def replay_parent(seed: int, split: str = "validation") -> dict:
    return {
        "source_kind": "archived_online_replay",
        "task": "pick",
        "source_sha256": canonical_sha256({"events": seed}),
        "trajectory": f"{split}-seed{seed}",
        "parent_id": seed,
    }


def sequence_values(case_offset: int, length: int, duplicate_frame: bool) -> dict[str, np.ndarray]:
    head = np.empty((length, 128, 128, 3), dtype=np.uint8)
    wrist = np.empty_like(head)
    state = np.empty((length, 24), dtype=np.float32)
    for frame_index in range(length):
        head[frame_index].fill(case_offset * 20 + frame_index)
        wrist[frame_index].fill(100 + case_offset * 20 + frame_index)
        state[frame_index] = (
            np.arange(24, dtype=np.float32) + case_offset * 1000 + frame_index * 25
        )
    if duplicate_frame and length >= 2:
        head[-1] = head[-2]
        wrist[-1] = wrist[-2]
        state[-1] = state[-2]
    return {"head_rgb": head, "wrist_rgb": wrist, "state": state}


def make_fixture(
    root: Path,
    *,
    sequence_length: int = 10,
    duplicate_frame: bool = False,
    training_overlap: bool = False,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    overlapping_source = replay_parent(3020)["source_sha256"] if training_overlap else None
    training_path = root / "training-manifest.json"
    write_json(
        training_path,
        {
            "schema": "fetch_native24_family_v1",
            "label_contract": LABEL_CONTRACT,
            "episodes": [
                {
                    "split": "train",
                    "parent_episode": training_parent(0, overlapping_source),
                },
                {"split": "validation", "parent_episode": training_parent(20)},
            ],
        },
    )

    registry_rows = []
    for split, seeds in (("train", range(3000, 3004)), ("validation", (3020, 3021))):
        for seed in seeds:
            parent = replay_parent(seed, split)
            registry_rows.append(
                {
                    "split": split,
                    "parent_episode": parent,
                    "events_sha256": parent["source_sha256"],
                    "result_sha256": canonical_sha256({"result": seed}),
                }
            )
    registry_path = root / "parent-registry.json"
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

    input_rows = []
    frozen_rows = []
    acquisition_rows = []
    case_artifacts: dict[str, dict[str, Path]] = {}
    for case_offset, (anchor_id, classification, seed, start, end) in enumerate(CASE_SPECS):
        family = "grasp" if "grasp" in classification else "move"
        instruction = "Grasp the apple." if family == "grasp" else "Move the held apple."
        parent = replay_parent(seed)
        sequence_id = f"{anchor_id}-causal-tail10"
        window = {
            "start": start,
            "end": end,
            "length": 10,
            "selection_rule": "previous_9_through_anchor_source_observations",
        }
        input_rows.append(
            {
                "case_id": anchor_id,
                "classification": classification,
                "family": family,
                "instruction": instruction,
                "parent_episode": parent,
                "observation_index": end,
            }
        )
        frozen = {
            "sequence_id": sequence_id,
            "anchor_case_id": anchor_id,
            "classification": classification,
            "family": family,
            "instruction": instruction,
            "invocation_index": 1 if family == "grasp" else 2,
            "anchor_observation_index": end,
            "parent_episode": parent,
            "window": window,
            "replan_cooldown": 0,
            "forbidden_events": ["learned_threshold"],
        }
        frozen_rows.append(frozen)

        case_dir = root / "cases" / sequence_id
        case_dir.mkdir(parents=True)
        values = sequence_values(case_offset, sequence_length, duplicate_frame)
        npz_path = case_dir / "observations.npz"
        np.savez_compressed(npz_path, **values)
        frame_hashes = [
            observation_sha256({key: value[index] for key, value in values.items()})
            for index in range(sequence_length)
        ]

        held = classification == "held_move"
        distance = 0.25 if classification == "far_grasp_diagnostic" else 0.04
        frames = []
        for observation_index in range(start, end + 1):
            frames.append(
                {
                    "observation_index": observation_index,
                    "tcp_object_distance_m": distance,
                    "is_grasped": held and observation_index == end,
                    "physical_completion": False,
                    "terminated": False,
                    "truncated": False,
                }
            )
        predicate_path = case_dir / "physical-predicates.json"
        write_json(
            predicate_path,
            {
                "schema": PREDICATE_SCHEMA,
                "status": "source_replay_physical_predicates",
                "sequence_id": sequence_id,
                "frames": frames,
            },
        )

        trajectory_path = case_dir / "source-trajectory.json"
        write_json(
            trajectory_path,
            {
                "schema": TRAJECTORY_SCHEMA,
                "status": "verified_source_trajectory",
                "parent_episode": parent,
                "source_events_sha256": parent["source_sha256"],
                "source_result_sha256": canonical_sha256({"result": seed}),
                "digest_contract": "historical-events-jsonl-v1",
                "trajectory_sha256": parent["source_sha256"],
            },
        )

        predicate_digest = canonical_sha256(frames)
        replay_path = case_dir / "source-sequence-replay.json"
        write_json(
            replay_path,
            {
                "schema": REPLAY_SCHEMA,
                "status": "verified_source_sequence_replay",
                "sequence_id": sequence_id,
                "parent_episode": parent,
                "trajectory_sha256": parent["source_sha256"],
                "instruction": instruction,
                "window": window,
                "npz_sha256": sha256_file(npz_path),
                "physical_predicates_sha256": sha256_file(predicate_path),
                "frame_observation_sha256": frame_hashes,
                "sequence_observation_sha256": canonical_sha256(frame_hashes),
                "physical_predicate_sequence_sha256": predicate_digest,
                "reconstruction_frame_observation_sha256": [frame_hashes, frame_hashes],
                "reconstruction_physical_predicate_sha256": [
                    predicate_digest,
                    predicate_digest,
                ],
                "initial_state_verified": True,
                "action_prefix_verified": True,
                "boundary_observation_verified": True,
                "physical_predicates_verified": True,
                "physical_atol": 1e-5,
                "policy_calls": 0,
                "training_updates": 0,
            },
        )
        acquisition_rows.append(
            {
                key: frozen[key]
                for key in (
                    "sequence_id",
                    "anchor_case_id",
                    "classification",
                    "family",
                    "instruction",
                    "invocation_index",
                    "anchor_observation_index",
                    "parent_episode",
                    "window",
                )
            }
            | {
                "npz_path": npz_path.relative_to(root).as_posix(),
                "physical_predicates_path": predicate_path.relative_to(root).as_posix(),
                "source_trajectory_record_path": trajectory_path.relative_to(root).as_posix(),
                "source_replay_record_path": replay_path.relative_to(root).as_posix(),
            }
        )
        case_artifacts[anchor_id] = {
            "npz": npz_path,
            "predicates": predicate_path,
            "trajectory": trajectory_path,
            "replay": replay_path,
        }

    input_path = root / "native24-handoff-manifest.json"
    write_json(
        input_path,
        {
            "schema": "bvi.native24-handoff/1",
            "status": "verified_handoff_inputs",
            "cases": input_rows,
        },
    )
    contract = monitor_contract(MONITOR_SOURCE)
    preregistration = {
        "schema": "bvi.native24-handoff-sequence-preregistration/1",
        "status": "frozen_before_render_or_model_query",
        "label_contract": LABEL_CONTRACT,
        "training_manifest_sha256": sha256_file(training_path),
        "input_manifest_sha256": sha256_file(input_path),
        "monitor_contract": contract,
        "sequences": frozen_rows,
    }
    preregistration["preregistration_sha256"] = canonical_sha256(preregistration)
    preregistration_path = root / "sequence-preregistration.json"
    write_json(preregistration_path, preregistration)

    acquisition_path = root / "sequence-acquisition.json"
    write_json(
        acquisition_path,
        {
            "schema": ACQUISITION_SCHEMA,
            "status": "capture_complete",
            "label_contract": LABEL_CONTRACT,
            "training_manifest_sha256": sha256_file(training_path),
            "input_manifest_sha256": sha256_file(input_path),
            "preregistration_path": preregistration_path.relative_to(root).as_posix(),
            "preregistration_sha256": sha256_file(preregistration_path),
            "monitor_contract": contract,
            "parent_registry_path": registry_path.relative_to(root).as_posix(),
            "exact_simulator_actions": 108,
            "hard_simulator_action_cap": 136,
            "policy_calls": 0,
            "training_updates": 0,
            "cases": acquisition_rows,
        },
    )
    return {
        "root": root,
        "training": training_path,
        "input": input_path,
        "registry": registry_path,
        "preregistration": preregistration_path,
        "acquisition": acquisition_path,
        "output": root / "native24-handoff-sequence-manifest.json",
        "case_artifacts": case_artifacts,
    }


def build_fixture(fixture: dict[str, Path]) -> dict:
    return build_manifest(
        fixture["acquisition"],
        fixture["training"],
        fixture["input"],
        MONITOR_SOURCE,
        fixture["output"],
    )


def load_fixture(fixture: dict[str, Path]):
    return load_manifest(
        fixture["output"],
        fixture["training"],
        fixture["input"],
        MONITOR_SOURCE,
    )


def test_build_and_load_complete_four_class_ten_frame_fixture(tmp_path):
    fixture = make_fixture(tmp_path)
    built = build_fixture(fixture)
    report, cases, tables, predicates = load_fixture(fixture)

    assert built == report
    assert report["sequence_count"] == 4
    assert report["frame_count"] == 40
    assert report["monitor_contract"]["source_sha256"] == sha256_file(MONITOR_SOURCE)
    assert report["monitor_stagnation_signal_mechanics_evaluable"] is True
    assert report["monitor_drop_signal_mechanics_evaluable"] is True
    assert report["positive_two_hit_correctness_evaluable"] is False
    assert {case["classification"] for case in cases} == {
        "far_grasp_diagnostic",
        "near_grasp_candidate_not_completion",
        "unheld_move",
        "held_move",
    }
    assert all(table["state"].shape == (10, 24) for table in tables.values())
    assert all(len(frames) == 10 for frames in predicates.values())
    assert all(
        frame["physical_completion"] is False
        and frame["terminated"] is False
        and frame["truncated"] is False
        for frames in predicates.values()
        for frame in frames
    )

    manifest = read_json(fixture["output"])
    input_manifest = read_json(fixture["input"])
    anchors = {row["case_id"]: row for row in input_manifest["cases"]}
    training_sources = {
        row["parent_episode"]["source_sha256"]
        for row in read_json(fixture["training"])["episodes"]
        if row["split"] == "train"
    }
    assert manifest["training_data"]["manifest_sha256"] == sha256_file(fixture["training"])
    assert manifest["input_gate"]["manifest_sha256"] == sha256_file(fixture["input"])
    assert manifest["preregistration"] == {
        "path": fixture["preregistration"].relative_to(fixture["root"]).as_posix(),
        "sha256": sha256_file(fixture["preregistration"]),
        "canonical_sha256": read_json(fixture["preregistration"])[
            "preregistration_sha256"
        ],
    }
    for case in manifest["cases"]:
        anchor = anchors[case["anchor_case_id"]]
        assert case["parent_episode"]["source_sha256"] not in training_sources
        assert case["classification"] == anchor["classification"]
        assert case["family"] == anchor["family"]
        assert case["instruction"] == anchor["instruction"]
        assert case["parent_episode"] == anchor["parent_episode"]
        assert case["window"]["end"] == anchor["observation_index"]
        replay = read_json(fixture["root"] / case["source_replay"]["path"])
        assert replay["reconstruction_frame_observation_sha256"] == [
            case_row["frame_observation_sha256"]
            for case_row in cases
            if case_row["sequence_id"] == case["sequence_id"]
        ] * 2
        expected_predicate_digest = canonical_sha256(predicates[case["sequence_id"]])
        assert replay["reconstruction_physical_predicate_sha256"] == [
            expected_predicate_digest,
            expected_predicate_digest,
        ]


def test_short_sequence_is_not_evaluable(tmp_path):
    fixture = make_fixture(tmp_path, sequence_length=9)
    with pytest.raises(NotEvaluableError, match="fewer than ten observations"):
        build_fixture(fixture)


def test_repeated_native24_frame_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path, duplicate_frame=True)
    with pytest.raises(ValueError, match="repeated native24 observation"):
        build_fixture(fixture)


def test_training_source_overlap_is_rejected(tmp_path):
    fixture = make_fixture(tmp_path, training_overlap=True)
    with pytest.raises(ValueError, match="overlaps the native24 training source"):
        build_fixture(fixture)


@pytest.mark.parametrize(
    ("tamper", "error"),
    (
        ("monitor", "Progress monitor was not frozen"),
        ("preregistration", "preregistration content/hash differs"),
        ("scope", "evaluation scope differs"),
    ),
)
def test_manifest_contract_tampering_is_rejected(tmp_path, tamper, error):
    fixture = make_fixture(tmp_path)
    build_fixture(fixture)
    manifest = read_json(fixture["output"])
    if tamper == "monitor":
        manifest["monitor_contract"]["source_sha256"] = "0" * 64
    elif tamper == "preregistration":
        manifest["preregistration"]["canonical_sha256"] = "0" * 64
    else:
        manifest["evaluation_scope"]["optimizer_updates"] = 1
    write_json(fixture["output"], manifest)

    with pytest.raises(ValueError, match=error):
        load_fixture(fixture)


@pytest.mark.parametrize("source", ("training", "input"))
def test_preregistration_rejects_a_rebound_source_manifest(tmp_path, source):
    fixture = make_fixture(tmp_path)
    build_fixture(fixture)
    source_manifest = read_json(fixture[source])
    source_manifest["post_freeze_note"] = "This changes the source manifest bytes only."
    write_json(fixture[source], source_manifest)
    manifest = read_json(fixture["output"])
    binding = "training_data" if source == "training" else "input_gate"
    manifest[binding]["manifest_sha256"] = sha256_file(fixture[source])
    write_json(fixture["output"], manifest)

    with pytest.raises(ValueError, match="preregistration content/hash differs"):
        load_fixture(fixture)


def test_changed_original_b_anchor_is_rejected_after_all_bindings_are_rehashed(tmp_path):
    fixture = make_fixture(tmp_path)
    build_fixture(fixture)
    input_manifest = read_json(fixture["input"])
    input_manifest["cases"][0]["instruction"] = "A post-freeze substituted instruction."
    write_json(fixture["input"], input_manifest)

    preregistration = read_json(fixture["preregistration"])
    preregistration["input_manifest_sha256"] = sha256_file(fixture["input"])
    preregistration["preregistration_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in preregistration.items()
            if key != "preregistration_sha256"
        }
    )
    write_json(fixture["preregistration"], preregistration)

    manifest = read_json(fixture["output"])
    manifest["input_gate"]["manifest_sha256"] = sha256_file(fixture["input"])
    manifest["preregistration"]["sha256"] = sha256_file(fixture["preregistration"])
    manifest["preregistration"]["canonical_sha256"] = preregistration[
        "preregistration_sha256"
    ]
    write_json(fixture["output"], manifest)

    with pytest.raises(ValueError, match="differs from its verified B input anchor"):
        load_fixture(fixture)


def test_nonterminal_per_frame_predicate_contract_is_enforced(tmp_path):
    fixture = make_fixture(tmp_path)
    acquisition = read_json(fixture["acquisition"])
    anchor_id = acquisition["cases"][0]["anchor_case_id"]
    predicate_path = fixture["case_artifacts"][anchor_id]["predicates"]
    replay_path = fixture["case_artifacts"][anchor_id]["replay"]
    predicates = read_json(predicate_path)
    predicates["frames"][4]["physical_completion"] = True
    write_json(predicate_path, predicates)
    replay = read_json(replay_path)
    predicate_digest = canonical_sha256(predicates["frames"])
    replay["physical_predicates_sha256"] = sha256_file(predicate_path)
    replay["physical_predicate_sequence_sha256"] = predicate_digest
    replay["reconstruction_physical_predicate_sha256"] = [
        predicate_digest,
        predicate_digest,
    ]
    write_json(replay_path, replay)

    with pytest.raises(ValueError, match="pre-completion, nonterminal"):
        build_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("reconstruction_frame_observation_sha256", "two identical reconstructions"),
        (
            "reconstruction_physical_predicate_sha256",
            "two identical predicate reconstructions",
        ),
    ),
)
def test_two_identical_reconstructions_are_required(tmp_path, field, error):
    fixture = make_fixture(tmp_path)
    acquisition = read_json(fixture["acquisition"])
    anchor_id = acquisition["cases"][0]["anchor_case_id"]
    replay_path = fixture["case_artifacts"][anchor_id]["replay"]
    replay = read_json(replay_path)
    replay[field] = replay[field][:1]
    write_json(replay_path, replay)

    with pytest.raises(ValueError, match=error):
        build_fixture(fixture)
