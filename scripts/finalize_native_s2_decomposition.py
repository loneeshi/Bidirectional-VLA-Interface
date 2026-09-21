"""Finalize a completed native24 S2 decomposition after metadata-only failure.

This recovery entry point performs no model inference, checkpoint unpickling,
optimizer update, simulator step, GPU query, or source mutation.  It binds the
pre-existing failed result and every completed run artifact by caller-supplied
SHA256, revalidates all 2x2 rows/predictions/queue events against the immutable
training evidence, and writes new identity/summary/finalization files.  The
original failed ``result.json`` is preserved verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from bvi.s2_offline_decomposition import (
    FAMILIES,
    SCHEMA,
    canonical_sha256,
    combination_spec,
    compare_historical_rows,
    summarize_rows,
)


EXISTING = {
    "failed_result": "result.json",
    "roster": "roster.json",
    "rows": "rows.jsonl",
    "predictions": "predictions.npz",
    "queue": "queue-audit.json",
}
NEW = ("identity.json", "summary.json", "finalization-result.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {label} SHA256")
    return value


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_failed_result(result: dict, *, expected_rows: int) -> dict:
    error = result.get("error")
    if (result.get("schema") != SCHEMA or result.get("status") != "failed"
            or result.get("optimizer_updates") != 0 or result.get("simulator_steps") != 0
            or result.get("native_success_evaluated") is not False
            or result.get("capability_admission") is not False):
        raise ValueError("Run is not the fail-closed zero-update decomposition failure")
    # The failed runner recorded repr(exc), and FileNotFoundError.__repr__ omits
    # its filename on CPython.  The caller-supplied result SHA binds the exact
    # failure bytes; completed artifact/row checks below prove evaluation ended.
    if not isinstance(error, str) or not error.startswith("FileNotFoundError("):
        raise ValueError("Recovery is restricted to the known post-evaluation source-path failure")
    wall = result.get("wall_seconds")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("Failed run lacks a finite positive wall duration")
    return {"status": result["status"], "error": error, "wall_seconds": float(wall),
            "expected_completed_rows": expected_rows}


def validate_roster_and_rows(roster_document: dict, rows: list[dict], *, expected_rows: int) -> tuple[list[dict], dict]:
    roster = roster_document.get("rows")
    if (roster_document.get("schema") != SCHEMA or not isinstance(roster, list)
            or roster_document.get("roster_sha256") != canonical_sha256(roster)):
        raise ValueError("Roster schema/hash mismatch")
    combinations = [item["name"] for item in combination_spec()]
    if expected_rows != len(roster) * len(combinations) or len(rows) != expected_rows:
        raise ValueError("Completed row count does not equal roster x four combinations")
    roster_by_id = {row.get("row_id"): row for row in roster}
    if None in roster_by_id or len(roster_by_id) != len(roster):
        raise ValueError("Roster row identities are missing or duplicated")
    seen = set()
    for row in rows:
        identity = (row.get("combination"), row.get("row_id"))
        if identity in seen or identity[0] not in combinations or identity[1] not in roster_by_id:
            raise ValueError("Duplicate or unknown decomposition row identity")
        seen.add(identity)
        source = roster_by_id[identity[1]]
        for field in ("window", "index", "family", "loss_rng_seed", "sample_rng_seed",
                      "noise_rng_seed", "action_label_valid"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Row differs from frozen roster: {identity}/{field}")
        if row.get("family") not in FAMILIES:
            raise ValueError("Unknown row family")
        spec = next(item for item in combination_spec() if item["name"] == row["combination"])
        if row.get("head_step") != spec["head_step"] or row.get("bank_step") != spec["bank_step"]:
            raise ValueError("Row uses the wrong head/bank step")
        for metric in ("action_loss", "progress_loss", "joint_loss"):
            if not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]):
                raise ValueError("Row contains a nonfinite loss")
        parity = row.get("deployment_parity", {})
        if parity.get("passed") is not True or parity.get("observation_sha256") is None:
            raise ValueError("Deployment adapter parity did not pass")
        if any(float(value) > float(parity.get("atol", -1)) for value in parity.get("max_abs_diff", {}).values()):
            raise ValueError("Deployment parity delta exceeds its frozen tolerance")
        first = row.get("first_action", {})
        predicted = np.asarray(first.get("predicted_raw"), dtype=np.float64)
        if predicted.shape != (13,) or not np.isfinite(predicted).all():
            raise ValueError("Row lacks a finite first Fetch13 action")
    expected_identities = {(combination, row_id) for combination in combinations for row_id in roster_by_id}
    if seen != expected_identities:
        raise ValueError("2x2 row matrix is incomplete")
    head_hashes = {}
    bank_hashes = {0: {}, 20: {}}
    for step in (0, 20):
        head_values = {row["head_sha256"] for row in rows if row["head_step"] == step}
        if len(head_values) != 1:
            raise ValueError("A head step has inconsistent identities")
        head_hashes[step] = require_sha256(head_values.pop(), f"head{step}")
        for family in FAMILIES:
            values = {row["bank_sha256"] for row in rows
                      if row["bank_step"] == step and row["family"] == family}
            if len(values) != 1:
                raise ValueError("A family bank step has inconsistent identities")
            bank_hashes[step][family] = require_sha256(values.pop(), f"bank{step}/{family}")
    return roster, {"head_sha256": head_hashes, "bank_sha256": bank_hashes}


def validate_predictions(path: Path, rows: list[dict], roster: list[dict]) -> dict:
    combinations = [item["name"] for item in combination_spec()]
    expected_keys = {
        f"{combination}_{suffix}" for combination in combinations
        for suffix in ("normalized_actions", "external_actions", "progress")
    }
    by_identity = {(row["combination"], row["row_id"]): row for row in rows}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("Prediction NPZ keys differ from the frozen 2x2 contract")
        shapes = {}
        for combination in combinations:
            normalized = archive[f"{combination}_normalized_actions"]
            external = archive[f"{combination}_external_actions"]
            progress = archive[f"{combination}_progress"]
            expected_shapes = ((len(roster), 10, 32), (len(roster), 10, 13), (len(roster), 10))
            if (normalized.shape, external.shape, progress.shape) != expected_shapes:
                raise ValueError("Prediction NPZ shape mismatch")
            if not all(np.isfinite(value).all() for value in (normalized, external, progress)):
                raise ValueError("Prediction NPZ contains nonfinite values")
            for index, roster_row in enumerate(roster):
                reported = np.asarray(
                    by_identity[(combination, roster_row["row_id"])]["first_action"]["predicted_raw"],
                    dtype=external.dtype,
                )
                if not np.array_equal(reported, external[index, 0]):
                    raise ValueError("rows.jsonl first action differs from predictions.npz")
            shapes[combination] = {
                "normalized_actions": list(normalized.shape),
                "external_actions": list(external.shape),
                "progress": list(progress.shape),
            }
    return {"passed": True, "shapes": shapes}


def validate_queue(queue_document: dict, rows: list[dict], roster: list[dict], identities: dict,
                   *, frozen_sha256: str) -> dict:
    events = queue_document.get("events")
    if (queue_document.get("passed") is not True
            or queue_document.get("deployment_adapter_evaluated") is not True
            or queue_document.get("deployment_transport_evaluated") is not False
            or not isinstance(events, list) or len(events) != len(rows)):
        raise ValueError("Queue audit header/count mismatch")
    by_call = {event.get("call_id"): event for event in events}
    if None in by_call or len(by_call) != len(events):
        raise ValueError("Queue event call IDs are missing or duplicated")
    checked = 0
    for combination in combination_spec():
        name, head_step, bank_step = combination["name"], combination["head_step"], combination["bank_step"]
        combination_sha256 = canonical_sha256({
            "frozen": frozen_sha256,
            "head": identities["head_sha256"][head_step],
            "banks": identities["bank_sha256"][bank_step],
            "head_step": head_step,
            "bank_step": bank_step,
        })
        previous_family = None
        for index, roster_row in enumerate(roster):
            call_id = f"{name}-{roster_row['row_id']}"
            event = by_call.get(call_id)
            expected_bank = identities["bank_sha256"][bank_step][roster_row["family"]]
            if (event is None or event.get("combination") != name
                    or event.get("selected_family") != roster_row["family"]
                    or event.get("previous_family") != previous_family
                    or event.get("selected_bank_sha256") != expected_bank
                    or event.get("response_adapter_sha256") != expected_bank
                    or event.get("response_progress_head_sha256") != identities["head_sha256"][head_step]
                    or event.get("response_combination_sha256") != combination_sha256
                    or event.get("discarded_actions") != (0 if index == 0 else 10)
                    or event.get("queue_empty_before_inference") is not True
                    or event.get("queued_actions_after_inference") != 10
                    or event.get("transport_evaluated") is not False):
                raise ValueError(f"Queue/routing event mismatch: {call_id}")
            previous_family = roster_row["family"]
            checked += 1
    return {"passed": True, "events": checked, "deployment_transport_evaluated": False}


def parse_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank rows.jsonl line: {line_number}")
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    for key in EXISTING:
        parser.add_argument(f"--expected-{key.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--best-record", type=Path, required=True)
    parser.add_argument("--validation-step0", type=Path, required=True)
    parser.add_argument("--validation-step20", type=Path, required=True)
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--step20", type=Path, required=True)
    parser.add_argument("--normalizer-stats", type=Path, required=True)
    parser.add_argument("--state-contract", type=Path, required=True)
    parser.add_argument("--evaluation-audit-source", type=Path, required=True)
    parser.add_argument("--core-source", type=Path, required=True)
    parser.add_argument("--deployment-source", type=Path, required=True)
    parser.add_argument("--expected-evaluation-audit-source-sha256", required=True)
    parser.add_argument("--expected-core-source-sha256", required=True)
    parser.add_argument("--expected-deployment-source-sha256", required=True)
    parser.add_argument("--historical-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.expected_rows <= 0 or args.expected_rows % 4:
        parser.error("--expected-rows must be a positive multiple of four")
    if args.historical_atol < 0:
        parser.error("--historical-atol must be nonnegative")
    run = args.run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory unavailable: {run}")
    if any((run / name).exists() for name in NEW):
        raise FileExistsError("Finalization outputs already exist; recovery is single-shot")
    started = time.monotonic()

    expected_existing = {
        key: require_sha256(getattr(args, f"expected_{key}_sha256"), key)
        for key in EXISTING
    }
    existing_paths = {key: run / name for key, name in EXISTING.items()}
    actual_existing = {key: sha256(path) for key, path in existing_paths.items()}
    if actual_existing != expected_existing:
        raise ValueError(f"Existing run artifact hash mismatch: expected={expected_existing}, actual={actual_existing}")
    failed = validate_failed_result(read_json(existing_paths["failed_result"]), expected_rows=args.expected_rows)
    roster_document = read_json(existing_paths["roster"])
    rows = parse_rows(existing_paths["rows"])
    roster, parameter_identities = validate_roster_and_rows(
        roster_document, rows, expected_rows=args.expected_rows
    )

    evidence_paths = {
        "data_manifest_sha256": args.data_manifest,
        "training_config_sha256": args.training_config,
        "checkpoint_record_sha256": args.checkpoint_record,
        "best_record_sha256": args.best_record,
        "validation_step0_sha256": args.validation_step0,
        "validation_step20_sha256": args.validation_step20,
        "step0_artifact_sha256": args.step0,
        "step20_artifact_sha256": args.step20,
    }
    evidence_hashes = {key: sha256(path) for key, path in evidence_paths.items()}
    training_config = read_json(args.training_config)
    checkpoint_record = read_json(args.checkpoint_record)
    best_record = read_json(args.best_record)
    historical0 = read_json(args.validation_step0)
    historical20 = read_json(args.validation_step20)
    training_identity = training_config.get("identity", {})
    if training_identity.get("manifest_sha256") != evidence_hashes["data_manifest_sha256"]:
        raise ValueError("Training config/data manifest identity mismatch")
    if best_record.get("step") != 20 or best_record.get("sha256") != evidence_hashes["step20_artifact_sha256"]:
        raise ValueError("Best record does not bind step20 artifact")
    if checkpoint_record.get("step") != 20:
        raise ValueError("Checkpoint record is not step20")
    frozen_sha256 = require_sha256(checkpoint_record.get("frozen_sha256"), "frozen")
    if frozen_sha256 != training_config.get("frozen_sha256"):
        raise ValueError("Frozen partition identity changed")
    if parameter_identities["bank_sha256"][0] != training_config.get("initial_bank_sha256"):
        raise ValueError("Rows do not use the frozen step0 bank identities")
    if (parameter_identities["bank_sha256"][20] != checkpoint_record.get("bank_sha256")
            or parameter_identities["head_sha256"][20] != checkpoint_record.get("head_sha256")):
        raise ValueError("Rows do not use the final step20 bank/head identities")
    normalizer_sha256 = sha256(args.normalizer_stats)
    state_contract_sha256 = sha256(args.state_contract)
    if (normalizer_sha256 != training_identity.get("normalizer_sha256")
            or state_contract_sha256 != training_identity.get("state_contract_sha256")):
        raise ValueError("Normalizer/state contract differs from training identity")

    source_paths = {
        "evaluation_audit_source_sha256": args.evaluation_audit_source,
        "core_source_sha256": args.core_source,
        "deployment_adapter_source_sha256": args.deployment_source,
    }
    expected_sources = {
        "evaluation_audit_source_sha256": require_sha256(
            args.expected_evaluation_audit_source_sha256, "evaluation audit source"
        ),
        "core_source_sha256": require_sha256(args.expected_core_source_sha256, "core source"),
        "deployment_adapter_source_sha256": require_sha256(
            args.expected_deployment_source_sha256, "deployment source"
        ),
    }
    source_hashes = {key: sha256(path) for key, path in source_paths.items()}
    if source_hashes != expected_sources:
        raise ValueError(f"Explicit source hash mismatch: {source_hashes}")

    prediction_check = validate_predictions(existing_paths["predictions"], rows, roster)
    queue_check = validate_queue(
        read_json(existing_paths["queue"]), rows, roster, parameter_identities,
        frozen_sha256=frozen_sha256,
    )
    historical_reproduction = {
        "H0_B0": compare_historical_rows(
            [row for row in rows if row["combination"] == "H0_B0"], historical0,
            atol=args.historical_atol,
        ),
        "H20_B20": compare_historical_rows(
            [row for row in rows if row["combination"] == "H20_B20"], historical20,
            atol=args.historical_atol,
        ),
    }
    summary = summarize_rows(rows)
    identity_document = {
        "schema": SCHEMA,
        "recovery_schema": "bvi.native24-s2-offline-decomposition-finalize/1",
        "recovered_from_failed_result_sha256": actual_existing["failed_result"],
        "failed_run": failed,
        "evaluation_audit_source_path": str(args.evaluation_audit_source.resolve()),
        "core_source_path": str(args.core_source.resolve()),
        "deployment_adapter_source_path": str(args.deployment_source.resolve()),
        "finalizer_source_sha256": sha256(Path(__file__)),
        **source_hashes,
        "existing_artifact_sha256": actual_existing,
        "evidence_hashes": evidence_hashes,
        "dataset_identity": {
            "manifest_sha256": training_identity.get("manifest_sha256"),
            "npz_sha256": training_identity.get("npz_sha256"),
        },
        "frozen_sha256": frozen_sha256,
        **parameter_identities,
        "normalizer_sha256": normalizer_sha256,
        "state_contract_sha256": state_contract_sha256,
        "roster_sha256": roster_document["roster_sha256"],
        "rng_contract": "loss=PRNGKey(123+window*1000+index); sample=PRNGKey(700123+window*1000+index); explicit_noise=PRNGKey(900123+window*1000+index)",
        "model_inference_calls_during_finalization": 0,
        "optimizer_updates_during_finalization": 0,
        "simulator_steps_during_finalization": 0,
        "gpu_queried_during_finalization": False,
    }
    (run / "identity.json").write_text(json.dumps(identity_document, indent=2) + "\n", encoding="utf-8")
    summary_document = {
        "schema": SCHEMA,
        "recovery_schema": "bvi.native24-s2-offline-decomposition-finalize/1",
        "status": "passed_in_process_deployment_adapter_parity_transport_not_evaluated_recovered_without_model",
        "historical_reproduction": historical_reproduction,
        "deployment_parity_rows": len(rows),
        "all_deployment_parity_passed": True,
        "deployment_adapter_evaluated": True,
        "deployment_transport_evaluated": False,
        "prediction_check": prediction_check,
        "queue_check": queue_check,
        "combination_summary": summary,
        "interpretation_scope": "revalidated existing offline loss, sampled first-action, and in-process deployment-adapter artifacts only",
        "native_success_evaluated": False,
        "capability_admission": False,
        "model_inference_calls_during_finalization": 0,
    }
    (run / "summary.json").write_text(json.dumps(summary_document, indent=2) + "\n", encoding="utf-8")
    artifact_hashes = {
        **actual_existing,
        "identity": sha256(run / "identity.json"),
        "summary": sha256(run / "summary.json"),
    }
    finalization = {
        "schema": "bvi.native24-s2-offline-decomposition-finalize/1",
        "status": summary_document["status"],
        "wall_seconds": time.monotonic() - started,
        "recovered_completed_rows": len(rows),
        "artifact_hashes": artifact_hashes,
        "original_result_preserved": sha256(existing_paths["failed_result"]) == expected_existing["failed_result"],
        "model_inference_calls": 0,
        "optimizer_updates": 0,
        "simulator_steps": 0,
        "gpu_queried": False,
        "native_success_evaluated": False,
        "capability_admission": False,
        "bounded_sft_expansion_authorized": False,
        "online_evaluation_authorized": False,
    }
    (run / "finalization-result.json").write_text(
        json.dumps(finalization, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": finalization["status"], "rows": len(rows),
                      "output": str(run / "finalization-result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
