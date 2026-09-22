"""Fail-closed admission for the native S2 20-update diagnostic only.

STATUS: frozen — historical training and diagnostics (retained)

This gate is intentionally separate from the full S2 capability admission.
It does not replace the native >=3/10 capability gate or the native24
wrong-handoff validation used by that path.  Its only purpose is to permit a
bounded gradient/routing integration experiment after exact-start S1-IA
failure attribution has completed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .first_action_probe import GATE_SPEC
from .same_key_confirmation import CONFIRMATION_SPEC, adjudicate_confirmed_tail
from .transform_parity import REQUIRED_MODEL_LEAVES


SCHEMA = "bvi.s2-diagnostic-entry-adjudication/1"
STATUS = "approved_after_exact_start_attribution"
SCOPE = "diagnostic_20_update_only_not_s2_capability_admission"
TRAINING_STAGE = "S1_IA_single_bank_no_progress"
MAX_UPDATES = 20
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CONFIRMATION_SEEDS = [2024, 2025, 2026, 2027, 2028]
_CONFIRMATION_MODEL_FIELDS = (
    "server_source_sha256",
    "author_commit",
    "pretrained_parameters_sha256",
    "normalizer_sha256",
    "state_contract_sha256",
    "rng_contract",
    "checkpoint",
    "precision",
    "denoising_steps",
    "action_horizon",
    "external_action_dim",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _resolve_evidence(manifest_path: Path, record: Mapping, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"Missing {label} evidence record")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Missing {label} evidence path")
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Missing {label} evidence file: {path}")
    expected = _hex64(record.get("sha256"), f"{label}.sha256")
    if sha256(path) != expected:
        raise ValueError(f"Changed {label} evidence")
    return path


def _require_exact(mapping: Mapping, expected: Mapping, label: str) -> None:
    for key, value in expected.items():
        actual = mapping.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Wrong {label}: {key}")


def _json_list(path: Path, label: str) -> list:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array for {label}: {path}")
    return value


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and bool(np.isfinite(value))
    )


def _validate_exploratory_model_identity(row: Mapping, model_identity: Mapping) -> None:
    recorded = row.get("model_identity")
    if not isinstance(recorded, Mapping):
        raise ValueError("Exploratory row lacks model identity evidence")
    _require_exact(
        recorded,
        {"status": "matched", "matched": True, "reasons": []},
        "exploratory per-seed model identity",
    )
    checks = recorded.get("checks")
    expected_fields = (
        "pretrained_parameters_sha256",
        "normalizer_sha256",
        "state_contract_sha256",
    )
    if not isinstance(checks, Mapping) or set(checks) != set(expected_fields):
        raise ValueError("Exploratory per-seed model identity field roster differs")
    for field in expected_fields:
        check = checks.get(field)
        if not isinstance(check, Mapping):
            raise ValueError(f"Missing exploratory model identity check: {field}")
        _require_exact(
            check,
            {
                "equal": True,
                "current": model_identity.get(field),
                "recorded": model_identity.get(field),
            },
            f"exploratory model identity check {field}",
        )


def _validate_probe(
    manifest_path: Path,
    record: Mapping,
    model_identity: Mapping,
) -> dict:
    if not isinstance(record, Mapping):
        raise ValueError("Missing exact-start first-action probe evidence")
    summary_path = _resolve_evidence(
        manifest_path, record.get("summary"), "first_action_probe.summary"
    )
    per_seed_path = _resolve_evidence(
        manifest_path, record.get("per_seed"), "first_action_probe.per_seed"
    )
    summary = _json(summary_path)
    _require_exact(
        summary,
        {
            "status": "completed",
            "stage": "exact_start_first_action_stochastic_attribution",
            "training_updates": 0,
            "simulator_steps": 0,
            "rollout_episodes": 0,
            "api_calls": 0,
        },
        "first-action probe contract",
    )
    planned = summary.get("planned_inference_calls")
    completed = summary.get("completed_inference_calls")
    if (
        isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned < 1
        or completed != planned
    ):
        raise ValueError("First-action probe inference cohort is incomplete")
    cohort = summary.get("cohort_gate")
    if not isinstance(cohort, Mapping):
        raise ValueError("Missing first-action cohort adjudication")

    current = summary.get("current_model_identity")
    if not isinstance(current, Mapping):
        raise ValueError("First-action probe lacks current model identity")
    if str(Path(current.get("checkpoint", "")).resolve()) != model_identity.get("checkpoint"):
        raise ValueError("First-action probe used a different checkpoint path")
    for key in (
        "pretrained_parameters_sha256",
        "normalizer_sha256",
        "state_contract_sha256",
    ):
        if current.get(key) != model_identity.get(key):
            raise ValueError(f"First-action probe model identity mismatch: {key}")

    rows = _json_list(per_seed_path, "first_action_probe.per_seed")
    if not rows:
        raise ValueError("First-action probe per-seed evidence is empty")
    expected_seeds = summary.get("seeds")
    if (
        not isinstance(expected_seeds, list)
        or [row.get("seed") for row in rows] != expected_seeds
    ):
        raise ValueError("First-action per-seed roster differs from summary")
    sample_count = summary.get("samples_per_seed")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("Invalid first-action per-seed sample count")
    decision = cohort.get("decision")
    reasons = cohort.get("contract_reasons")
    probe_mode = None
    request_sha256_by_seed = {}
    rng_by_seed = {}
    samples_path = None
    if decision == "stochastic_tail_not_supported" and reasons == []:
        # Original exact-reproduction path.  Its criteria are deliberately
        # unchanged by the separately handled fresh-confirmation exception.
        for row in rows:
            identity = row.get("identity")
            if (
                row.get("sample_count") != sample_count
                or not isinstance(identity, Mapping)
                or identity.get("valid") is not True
                or identity.get("model_identity_matched") is not True
                or identity.get("first_sample_rng_matches") is not True
                or identity.get("first_sample_reproduced") is not True
            ):
                raise ValueError("First-action exact request/RNG reproduction is incomplete")
            error = identity.get("first_sample_reproduction_max_abs_raw")
            tolerance = identity.get("exact_reproduction_atol")
            if (
                not _finite_number(error)
                or not _finite_number(tolerance)
                or error < 0
                or tolerance < 0
                or error > tolerance
            ):
                raise ValueError("First-action reproduction exceeds its frozen tolerance")
        probe_mode = "exact_reproduction"
    elif decision == "invalid_cohort" and reasons == [
        "one_or_more_exact_reproduction_identity_failures"
    ]:
        if expected_seeds != _CONFIRMATION_SEEDS:
            raise ValueError("Fresh-confirmation exception requires the frozen five-seed cohort")
        if sample_count != GATE_SPEC["samples_per_case"] or completed != len(rows) * sample_count:
            raise ValueError("Fresh-confirmation exception requires the complete 5x16 probe")
        for row in rows:
            identity = row.get("identity")
            if (
                row.get("decision") != "identity_failed"
                or row.get("sample_count") != sample_count
                or not isinstance(identity, Mapping)
                or set(identity)
                != {
                    "valid",
                    "model_identity_matched",
                    "first_sample_rng_matches",
                    "first_sample_reproduction_max_abs_raw",
                    "exact_reproduction_atol",
                    "first_sample_reproduced",
                }
                or identity.get("valid") is not False
                or identity.get("model_identity_matched") is not True
                or identity.get("first_sample_rng_matches") is not True
                or identity.get("first_sample_reproduced") is not False
            ):
                raise ValueError(
                    "Invalid exploratory cohort is not solely a first-sample reproduction failure"
                )
            error = identity.get("first_sample_reproduction_max_abs_raw")
            tolerance = identity.get("exact_reproduction_atol")
            if (
                not _finite_number(error)
                or not _finite_number(tolerance)
                or tolerance < 0
                or error <= tolerance
            ):
                raise ValueError("Exploratory reproduction failure is not above frozen tolerance")
            expected_rng = row.get("expected_first_rng")
            if (
                not isinstance(expected_rng, list)
                or len(expected_rng) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_rng)
                or row.get("observed_first_rng") != expected_rng
            ):
                raise ValueError("Exploratory request RNG identity differs")
            _validate_exploratory_model_identity(row, model_identity)
            request_sha = _hex64(
                row.get("request_sha256"), f"exploratory seed {row.get('seed')} request_sha256"
            )
            request_sha256_by_seed[row["seed"]] = request_sha
            rng_by_seed[row["seed"]] = expected_rng
        if not isinstance(current.get("rng_contract"), str) or not current["rng_contract"]:
            raise ValueError("Exploratory probe lacks its frozen RNG contract")
        samples_path = _resolve_evidence(
            manifest_path, record.get("samples"), "first_action_probe.samples"
        )
        with np.load(samples_path, allow_pickle=False) as arrays:
            if "seeds" not in arrays or "raw_first_actions" not in arrays:
                raise ValueError("Exploratory samples lack seed/action arrays")
            if not np.array_equal(arrays["seeds"], np.asarray(expected_seeds)):
                raise ValueError("Exploratory sample seed roster differs")
            raw_first = np.asarray(arrays["raw_first_actions"])
            if raw_first.shape != (5, sample_count, 13) or not np.isfinite(raw_first).all():
                raise ValueError("Exploratory first-action sample tensor differs")
        probe_mode = "fresh_same_key_confirmation_required"
    else:
        raise ValueError("First-action probe does not support an adaptation diagnostic")
    transform = summary.get("transform_parity")
    if not isinstance(transform, Mapping):
        raise ValueError("First-action probe lacks preprocessing transform parity")
    _require_exact(
        transform,
        {"status": "bit_exact", "bit_exact": True, "artifact": "transform-parity.json"},
        "first-action preprocessing parity summary",
    )
    result = {
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "per_seed_path": str(per_seed_path),
        "per_seed_sha256": sha256(per_seed_path),
        "decision": decision,
        "contract_reasons": reasons,
        "admission_mode": probe_mode,
        "seeds": expected_seeds,
        "inference_calls": completed,
        "transform_parity_artifact": transform["artifact"],
        "request_sha256_by_seed": request_sha256_by_seed,
        "rng_by_seed": rng_by_seed,
        "rng_contract": current.get("rng_contract"),
    }
    if samples_path is not None:
        result.update(samples_path=str(samples_path), samples_sha256=sha256(samples_path))
    return result


def _validate_preprocessing(
    manifest_path: Path,
    record: Mapping,
    repo_id: str,
    expected_seeds: list,
) -> dict:
    path = _resolve_evidence(manifest_path, record, "preprocessing_parity")
    evidence = _json(path)
    _require_exact(
        evidence,
        {"status": "bit_exact", "bit_exact": True, "repo_id": repo_id},
        "preprocessing transform parity",
    )
    cases = evidence.get("cases")
    if not isinstance(cases, list) or [case.get("seed") for case in cases] != expected_seeds:
        raise ValueError("Preprocessing parity seed roster differs from first-action probe")
    required = list(REQUIRED_MODEL_LEAVES)
    for case in cases:
        _require_exact(
            case,
            {
                "status": "bit_exact",
                "bit_exact": True,
                "required_leaves": required,
                "missing_training": [],
                "missing_server": [],
                "training_only": [],
                "server_only": [],
            },
            "per-seed preprocessing transform parity",
        )
        rows = case.get("leaves")
        if not isinstance(rows, list) or [row.get("leaf") for row in rows] != sorted(required):
            raise ValueError("Preprocessing parity leaf roster differs")
        for row in rows:
            if (
                row.get("shape_equal") is not True
                or row.get("dtype_equal") is not True
                or row.get("array_equal") is not True
                or _HEX64.fullmatch(str(row.get("training_sha256", ""))) is None
                or row.get("training_sha256") != row.get("server_sha256")
            ):
                raise ValueError("Preprocessing transform leaf is not bit exact")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": "bit_exact",
        "seeds": expected_seeds,
        "required_leaves": required,
        "scope": "training_and_server_shared_model_input_leaves_plus_exact_request_reproduction",
    }


def _confirmation_manifest_entries(path: Path) -> dict[str, dict]:
    artifact_manifest = _json(path)
    rows = artifact_manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Fresh confirmation artifact manifest is empty")
    entries = {}
    root = path.parent.resolve()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Fresh confirmation artifact manifest row is invalid")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in entries
            or Path(relative).is_absolute()
        ):
            raise ValueError("Fresh confirmation artifact manifest path is invalid")
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError("Fresh confirmation artifact escapes or is missing")
        expected = _hex64(row.get("sha256"), f"fresh confirmation {relative} sha256")
        if sha256(artifact) != expected:
            raise ValueError(f"Changed fresh confirmation artifact: {relative}")
        entries[relative] = dict(row)
    return entries


def _validate_confirmation_identity(
    value: Any,
    model_identity: Mapping,
    checkpoint: Path,
    rng_contract: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Fresh confirmation lacks model identity evidence")
    _require_exact(value, {"matched": True, "reasons": []}, "fresh model identity")
    checks = value.get("checks")
    if (
        not isinstance(checks, list)
        or [check.get("field") for check in checks if isinstance(check, Mapping)]
        != list(_CONFIRMATION_MODEL_FIELDS)
    ):
        raise ValueError("Fresh confirmation model identity field roster differs")
    for check in checks:
        field = check["field"]
        if check.get("equal") is not True or check.get("current") != check.get("recorded"):
            raise ValueError(f"Fresh confirmation model identity mismatch: {field}")
        if field in (
            "pretrained_parameters_sha256",
            "normalizer_sha256",
            "state_contract_sha256",
        ) and check.get("current") != model_identity.get(field):
            raise ValueError(f"Fresh confirmation changed adjudicated identity: {field}")
        if field == "checkpoint" and str(Path(check.get("current", "")).resolve()) != str(
            checkpoint
        ):
            raise ValueError("Fresh confirmation used a different checkpoint")
        if field == "rng_contract" and check.get("current") != rng_contract:
            raise ValueError("Fresh confirmation changed the exploratory RNG contract")


def _validate_same_key_confirmation(
    manifest_path: Path,
    record: Mapping,
    probe: Mapping,
    preprocessing: Mapping,
    model_identity: Mapping,
    checkpoint: Path,
    normalizer_directory: Path,
) -> dict:
    if not isinstance(record, Mapping):
        raise ValueError("Invalid exploratory cohort requires fresh same-key confirmation")
    summary_path = _resolve_evidence(
        manifest_path, record.get("summary"), "same_key_confirmation.summary"
    )
    per_seed_path = _resolve_evidence(
        manifest_path, record.get("per_seed"), "same_key_confirmation.per_seed"
    )
    fresh_actions_path = _resolve_evidence(
        manifest_path, record.get("fresh_actions"), "same_key_confirmation.fresh_actions"
    )
    artifact_manifest_path = _resolve_evidence(
        manifest_path,
        record.get("artifact_manifest"),
        "same_key_confirmation.artifact_manifest",
    )
    root = summary_path.parent.resolve()
    expected_paths = {
        "summary.json": summary_path,
        "per-seed.json": per_seed_path,
        "fresh-actions.npz": fresh_actions_path,
        "artifact-manifest.json": artifact_manifest_path,
    }
    if any(path.resolve() != (root / relative).resolve() for relative, path in expected_paths.items()):
        raise ValueError("Fresh confirmation evidence files do not share one canonical run directory")
    entries = _confirmation_manifest_entries(artifact_manifest_path)
    for relative in (
        "summary.json",
        "per-seed.json",
        "fresh-actions.npz",
        "server/result.json",
        "server/metadata.json",
    ):
        if relative not in entries:
            raise ValueError(f"Fresh confirmation artifact manifest lacks {relative}")
    for relative, path in expected_paths.items():
        if relative != "artifact-manifest.json" and entries[relative]["sha256"] != sha256(path):
            raise ValueError(f"Fresh confirmation record differs from manifest: {relative}")

    summary = _json(summary_path)
    _require_exact(
        summary,
        {
            "status": "completed",
            "stage": "fresh_server_same_key_cluster_confirmation",
            "seeds": _CONFIRMATION_SEEDS,
            "planned_inference_calls": 5,
            "completed_inference_calls": 5,
            "training_updates": 0,
            "simulator_steps": 0,
            "rollout_episodes": 0,
            "api_calls": 0,
            "confirmation_spec_frozen_before_inference": CONFIRMATION_SPEC,
            "tail_spec_unchanged": GATE_SPEC,
            "exploratory_transform_parity": "bit_exact",
            "all_five_same_key_clusters_confirmed": True,
            "outputs": ["per-seed.json", "fresh-actions.npz"],
        },
        "fresh same-key confirmation contract",
    )
    if str(Path(summary.get("exploratory_run", "")).resolve()) != str(
        Path(probe["summary_path"]).parent
    ):
        raise ValueError("Fresh confirmation names a different exploratory run")
    if str(Path(summary.get("checkpoint", "")).resolve()) != str(checkpoint):
        raise ValueError("Fresh confirmation names a different checkpoint")
    if str(Path(summary.get("normalizer", "")).resolve()) != str(normalizer_directory):
        raise ValueError("Fresh confirmation names a different normalizer")
    exploratory_hashes = summary.get("exploratory_artifact_sha256")
    if not isinstance(exploratory_hashes, Mapping):
        raise ValueError("Fresh confirmation lacks exploratory artifact bindings")
    _require_exact(
        exploratory_hashes,
        {
            "summary.json": probe["summary_sha256"],
            "per-seed.json": probe["per_seed_sha256"],
            "samples.npz": probe["samples_sha256"],
            "transform-parity.json": preprocessing["sha256"],
        },
        "fresh confirmation exploratory artifact binding",
    )

    reports = _json_list(per_seed_path, "same_key_confirmation.per_seed")
    if [row.get("seed") for row in reports if isinstance(row, Mapping)] != _CONFIRMATION_SEEDS:
        raise ValueError("Fresh confirmation per-seed roster differs")
    anchor_summary = summary.get("anchor_preflight")
    if (
        not isinstance(anchor_summary, list)
        or [row.get("seed") for row in anchor_summary if isinstance(row, Mapping)]
        != _CONFIRMATION_SEEDS
    ):
        raise ValueError("Fresh confirmation anchor roster differs")
    for report, summary_anchor in zip(reports, anchor_summary, strict=True):
        seed = report["seed"]
        if report.get("request_sha256") != probe["request_sha256_by_seed"].get(seed):
            raise ValueError(f"Fresh confirmation request identity differs for seed {seed}")
        _validate_confirmation_identity(
            report.get("model_identity"), model_identity, checkpoint, probe["rng_contract"]
        )
        expected_rng = report.get("expected_rng")
        if (
            not isinstance(expected_rng, list)
            or len(expected_rng) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_rng)
            or expected_rng != probe["rng_by_seed"].get(seed)
            or report.get("fresh_rng") != expected_rng
            or report.get("rng_matches") is not True
        ):
            raise ValueError(f"Fresh confirmation RNG identity differs for seed {seed}")
        anchor = report.get("anchor_preflight")
        fresh = report.get("fresh_confirmation")
        if not isinstance(anchor, Mapping) or not isinstance(fresh, Mapping):
            raise ValueError(f"Fresh confirmation cluster evidence is missing for seed {seed}")
        _require_exact(
            anchor,
            {
                "passed": True,
                "mutual_unique_nearest": True,
                "material_sign_flip_channels": [],
            },
            f"fresh confirmation anchor for seed {seed}",
        )
        _require_exact(
            fresh,
            {
                "passed": True,
                "closer_to_both_same_key_anchors_than_any_different_key": True,
                "material_sign_flips_vs_historical": [],
                "material_sign_flips_vs_exploratory_sample0": [],
            },
            f"fresh confirmation cluster for seed {seed}",
        )
        if dict(summary_anchor) != {"seed": seed, **dict(anchor)}:
            raise ValueError(f"Fresh confirmation summary anchor differs for seed {seed}")

    adjudication = summary.get("adjudication")
    recomputed = adjudicate_confirmed_tail(reports)
    if not isinstance(adjudication, Mapping) or dict(adjudication) != recomputed:
        raise ValueError("Fresh confirmation final adjudication was not reproduced")
    _require_exact(
        adjudication,
        {"decision": "stochastic_tail_not_supported", "contract_reasons": []},
        "fresh confirmation final decision",
    )
    with np.load(fresh_actions_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"seeds", "raw_action_chunks"}:
            raise ValueError("Fresh confirmation action archive fields differ")
        chunks = np.asarray(arrays["raw_action_chunks"])
        if (
            not np.array_equal(arrays["seeds"], np.asarray(_CONFIRMATION_SEEDS))
            or chunks.shape != (5, 10, 13)
            or not np.isfinite(chunks).all()
        ):
            raise ValueError("Fresh confirmation action archive contract differs")

    server_result = _json(root / "server" / "result.json")
    _require_exact(
        server_result,
        {
            "status": "stopped",
            "training_updates": 0,
            "progress_head": False,
            "tapt": False,
            "inference_calls": 5,
            "max_inference_calls": 5,
            "stop_reason": "explicit_shutdown",
        },
        "fresh confirmation server call budget",
    )
    server_metadata = _json(root / "server" / "metadata.json")
    _require_exact(
        server_metadata,
        {
            "checkpoint": str(checkpoint),
            "pretrained_parameters_sha256": model_identity["pretrained_parameters_sha256"],
            "normalizer_sha256": model_identity["normalizer_sha256"],
            "state_contract_sha256": model_identity["state_contract_sha256"],
            "precision": "bfloat16",
            "denoising_steps": 10,
            "action_horizon": 10,
            "external_action_dim": 13,
            "max_inference_calls": 5,
            "training_updates": 0,
            "progress_head": False,
            "tapt": False,
        },
        "fresh confirmation server identity",
    )
    state_contract = server_metadata.get("state_contract")
    if not isinstance(state_contract, Mapping) or state_contract.get("training_stage") != TRAINING_STAGE:
        raise ValueError("Fresh confirmation server did not load the S1-IA training stage")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "per_seed_path": str(per_seed_path),
        "per_seed_sha256": sha256(per_seed_path),
        "fresh_actions_path": str(fresh_actions_path),
        "fresh_actions_sha256": sha256(fresh_actions_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_manifest_sha256": sha256(artifact_manifest_path),
        "decision": adjudication["decision"],
        "seeds": _CONFIRMATION_SEEDS,
        "inference_calls": 5,
        "all_clusters_confirmed": True,
    }


def validate_diagnostic_entry(
    manifest_path: Path,
    expected_manifest_sha256: str,
    checkpoint: Path,
    normalizer_directory: Path,
    best_record_path: Path,
    steps: int,
) -> dict:
    """Validate an immutable, diagnostic-only S2 entry adjudication.

    The returned shape deliberately contains ``native.checkpoint_sha256`` so
    the trainer's post-restore digest check remains common to both admission
    paths.  ``capability_admission`` is always false.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_UPDATES:
        raise ValueError("Diagnostic S2 entry is limited to 1..20 total optimizer updates")
    manifest_path = Path(manifest_path).resolve()
    expected_manifest_sha256 = _hex64(
        expected_manifest_sha256, "diagnostic adjudication manifest SHA256"
    )
    if not manifest_path.is_file() or sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Diagnostic adjudication manifest is missing or changed")
    manifest = _json(manifest_path)
    _require_exact(
        manifest,
        {"schema": SCHEMA, "status": STATUS, "scope": SCOPE},
        "diagnostic adjudication contract",
    )
    authorization = manifest.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("Missing diagnostic-only authorization boundaries")
    _require_exact(
        authorization,
        {
            "max_optimizer_updates": MAX_UPDATES,
            "capability_admission": False,
            "bounded_sft_expansion": False,
            "online_evaluation": False,
        },
        "diagnostic-only authorization boundary",
    )

    checkpoint = Path(checkpoint).resolve()
    normalizer_directory = Path(normalizer_directory).resolve()
    best_record_path = Path(best_record_path).resolve()
    if not checkpoint.is_dir() or not normalizer_directory.is_dir() or not best_record_path.is_file():
        raise ValueError("Checkpoint, normalizer directory and S1 best record must exist")
    identity = manifest.get("model_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Missing diagnostic model identity")
    _require_exact(
        identity,
        {"checkpoint": str(checkpoint), "training_stage": TRAINING_STAGE},
        "diagnostic S1-IA identity",
    )
    pretrained_sha = _hex64(
        identity.get("pretrained_parameters_sha256"),
        "model_identity.pretrained_parameters_sha256",
    )
    normalizer_sha = _hex64(identity.get("normalizer_sha256"), "model_identity.normalizer_sha256")
    state_contract_sha = _hex64(
        identity.get("state_contract_sha256"), "model_identity.state_contract_sha256"
    )

    best_evidence = manifest.get("s1_best_record")
    if not isinstance(best_evidence, Mapping):
        raise ValueError("Missing S1 heldout-selected best record evidence")
    if str(best_record_path) != str(_resolve_evidence(manifest_path, best_evidence, "s1_best_record")):
        raise ValueError("CLI S1 best record differs from adjudicated evidence")
    best = _json(best_record_path)
    if best.get("selection") != "heldout_action_loss_only":
        raise ValueError("Diagnostic checkpoint was not selected by heldout action loss")
    if str(Path(best.get("checkpoint", "")).resolve()) != str(checkpoint):
        raise ValueError("S1 best record names a different checkpoint")

    repo_id = manifest.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("Missing training repo identity")
    asset_directory = checkpoint / "assets" / Path(repo_id)
    checkpoint_norm = asset_directory / "norm_stats.json"
    state_contract_path = asset_directory / "bvi-state-contract.json"
    if sha256(checkpoint_norm) != normalizer_sha or sha256(state_contract_path) != state_contract_sha:
        raise ValueError("Checkpoint asset identity differs from adjudication")
    if sha256(normalizer_directory / "norm_stats.json") != normalizer_sha:
        raise ValueError("CLI normalizer differs from diagnostic checkpoint")
    contract = _json(state_contract_path)
    _require_exact(
        contract,
        {
            "training_stage": TRAINING_STAGE,
            "training_repo": repo_id,
            "state_dim": 24,
            "action_dim": 13,
            "state_source": "env_native_agent",
            "base_camera": "fetch_head",
            "wrist_camera": "fetch_hand",
        },
        "S1-IA checkpoint state contract",
    )

    probe = _validate_probe(manifest_path, manifest.get("first_action_probe"), identity)
    preprocessing = _validate_preprocessing(
        manifest_path, manifest.get("preprocessing_parity"), repo_id, probe["seeds"]
    )
    expected_transform = (
        Path(probe["summary_path"]).parent / probe["transform_parity_artifact"]
    ).resolve()
    if Path(preprocessing["path"]) != expected_transform:
        raise ValueError("Adjudicated preprocessing evidence is not the probe's P0 artifact")
    confirmation = None
    if probe["admission_mode"] == "fresh_same_key_confirmation_required":
        confirmation = _validate_same_key_confirmation(
            manifest_path,
            manifest.get("same_key_confirmation"),
            probe,
            preprocessing,
            identity,
            checkpoint,
            normalizer_directory,
        )
    elif manifest.get("same_key_confirmation") is not None:
        raise ValueError("Fresh same-key confirmation is only valid for the narrow invalid-cohort path")
    return {
        "scope": SCOPE,
        "schema": SCHEMA,
        "status": STATUS,
        "capability_admission": False,
        "bounded_sft_expansion": False,
        "online_evaluation": False,
        "max_optimizer_updates": MAX_UPDATES,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "native": {
            "checkpoint_sha256": pretrained_sha,
            "normalizer_sha256": normalizer_sha,
            "state_contract_sha256": state_contract_sha,
            "training_stage": TRAINING_STAGE,
        },
        "first_action_probe": probe,
        "preprocessing_parity": preprocessing,
        "same_key_confirmation": confirmation,
        "budget_increase": False,
        "historical_pipeline_resume": False,
    }
