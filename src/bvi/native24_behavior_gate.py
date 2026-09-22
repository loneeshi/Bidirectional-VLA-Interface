"""Pure-CPU contracts for the native24 zero-training behavior gate.

STATUS: frozen — historical training and diagnostics (retained)

The gate compares the four head/bank corners and one selective-bank condition
on a fixed observation roster.  It deliberately owns no model, accelerator,
transport, or simulator.  Callers supply model ``progress`` chunks and queue
clear evidence; this module binds their routing/RNG identities, replays the
existing :class:`bvi.progress_monitor.ProgressMonitor`, and adjudicates only
the behavior that an offline current-observation replay can establish.

Behavior-row contract used by :func:`adjudicate_behavior_rows`::

    {
        "combination": "H20_BSEL_MOVE20",
        "sequence_id": "...",
        "sequence_index": 0,
        "classification": "unheld_move",
        "family": "move",
        "head_step": 20,
        "bank_step": 20,
        "monitor": run_monitor_sequence(...),
        "physical_completion": [False] * 10,
        "rng_seeds": [roster_seeds(0, frame) for frame in range(10)],
    }

Every model prediction must have a corresponding queue-event row keyed by
``combination``, ``sequence_id`` and ``prediction_index``.  Its ``reason``
must equal the monitor row's ``queue_clear_reason``; ``discarded_actions``
must be 10 and ``queue_empty`` must be true.  This records that read-only
offline predictions never leak a previous action chunk into the next fixed
observation.  A learned event stops the runtime monitor at its first event;
later fixed-roster predictions may remain in a complete paired matrix, but
are explicitly diagnostic-only and can never change that decision.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from .progress_monitor import ProgressMonitor


SCHEMA = "bvi.native24-behavior-gate/1"
MONITOR_TRACE_SCHEMA = "bvi.native24-behavior-monitor-trace/1"
RNG_SCHEMA = "bvi.native24-behavior-rng/1"
FAMILIES = ("reach", "grasp", "move", "release")
COMBINATIONS = (
    "H0_B0",
    "H20_B0",
    "H0_B20",
    "H20_B20",
    "H20_BSEL_MOVE20",
)
CLASSIFICATIONS = (
    "far_grasp_diagnostic",
    "near_grasp_candidate_not_completion",
    "unheld_move",
    "held_move",
)
NEGATIVE_COMPLETION_CLASSES = frozenset(CLASSIFICATIONS[:3])
STATUS_PRECEDENCE = ("invalid", "not_evaluable", "failed", "passed")
PROGRESS_SHAPE = (1, 10)
FIXED_SEQUENCE_LENGTH = 10

_HEAD_STEPS = {
    "H0_B0": 0,
    "H20_B0": 20,
    "H0_B20": 0,
    "H20_B20": 20,
    "H20_BSEL_MOVE20": 20,
}
_BANK_STEPS = {
    "H0_B0": {family: 0 for family in FAMILIES},
    "H20_B0": {family: 0 for family in FAMILIES},
    "H0_B20": {family: 20 for family in FAMILIES},
    "H20_B20": {family: 20 for family in FAMILIES},
    "H20_BSEL_MOVE20": {
        "reach": 0,
        "grasp": 0,
        "move": 20,
        "release": 0,
    },
}

_RUNTIME_SEMANTICS = {
    "progress_contract": "current_observation_v2",
    "progress_output_shape": [1, 10],
    "monitor_input": "float(progress[0,0])",
    "monitor_input_rounding": "none",
    "unused_chunk_prefix_values": "progress[0,1:10] are not monitor timesteps",
    "monitor_frequency": "once_per_policy_prediction",
    "monitor_lifecycle": "fresh ProgressMonitor per sequence and combination",
    "event_priority": ["learned_threshold", "learned_drop", "learned_stagnation"],
    "event_handling": "first event stops runtime consumption and clears the pending chunk",
    "false_completion_rule": (
        "any learned_threshold at a frame with physical_completion=false, including held_move"
    ),
    "offline_suffix": (
        "fixed-roster predictions after the first event are post-terminal diagnostics only"
    ),
    "queue_semantics": (
        "each read-only prediction clears all 10 pending actions before the next fixed observation"
    ),
    "simulator_steps": 0,
    "training_updates": 0,
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonnegative_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, not bool")
    result = int(value)
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _family(value: object) -> str:
    if not isinstance(value, str) or value not in FAMILIES:
        raise ValueError("Unknown native24 family")
    return value


def combination_specs() -> list[dict[str, Any]]:
    """Return the five immutable head/bank routes as JSON-safe records."""
    return [
        {
            "name": name,
            "head_step": _HEAD_STEPS[name],
            "bank_steps": dict(_BANK_STEPS[name]),
            "selective": name == "H20_BSEL_MOVE20",
            "training_updates_during_gate": 0,
        }
        for name in COMBINATIONS
    ]


def runtime_semantics() -> dict[str, Any]:
    """Return a defensive copy of the exact offline/runtime interpretation."""
    return deepcopy(_RUNTIME_SEMANTICS)


def rng_contract() -> dict[str, Any]:
    """Describe the combination-independent fixed RNG formula.

    ``frame_index`` is intentionally constrained below 1000 so the historical
    ``sequence_index * 1000 + frame_index`` layout has no row collisions.
    """
    return {
        "schema": RNG_SCHEMA,
        "sample_seed_formula": "700123 + sequence_index*1000 + frame_index",
        "noise_seed_formula": "900123 + sequence_index*1000 + frame_index",
        "sample_key_construction": "PRNGKey(sample_rng_seed)",
        "noise_key_construction": "PRNGKey(noise_rng_seed)",
        "frame_index_range": [0, 999],
        "shared_across_combinations": True,
        "implicit_rng_splits": 0,
    }


def roster_seeds(sequence_index: int, frame_index: int) -> dict[str, Any]:
    """Return the fixed sample/noise seeds and a hash-bound row identity."""
    sequence_index = _nonnegative_int(sequence_index, "sequence_index")
    frame_index = _nonnegative_int(frame_index, "frame_index", maximum=999)
    offset = sequence_index * 1000 + frame_index
    identity = {
        "schema": RNG_SCHEMA,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "sample_rng_seed": 700_123 + offset,
        "noise_rng_seed": 900_123 + offset,
    }
    identity["rng_identity"] = _canonical_sha256(identity)
    return identity


def progress_scalar(progress: object) -> tuple[float, str]:
    """Validate one model output and return its unrounded ``progress[0, 0]``.

    Finiteness covers the complete ``[1, 10]`` tensor even though the monitor
    consumes only the first element.  The returned Python float is converted
    directly from the source scalar; no rounding, clipping or epsilon is used.
    """
    try:
        values = np.asarray(progress)
    except Exception as exc:  # pragma: no cover - numpy exception types vary
        raise ValueError("Progress must be a numeric array with shape [1, 10]") from exc
    if values.shape != PROGRESS_SHAPE:
        raise ValueError("Progress must have exact shape [1, 10]")
    if values.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("Progress must contain real non-boolean numbers")
    if not np.isfinite(values).all():
        raise ValueError("Progress must be finite across the complete [1, 10] output")
    return float(values[0, 0]), str(values.dtype)


def run_monitor_sequence(
    family: str,
    invocation_index: int,
    progress_chunks: Sequence[object],
    cooldown: int = 0,
) -> dict[str, Any]:
    """Replay model chunks through a fresh real :class:`ProgressMonitor`.

    All supplied chunks are shape/finiteness validated to support a complete
    paired offline matrix.  Only ``progress[0, 0]`` from predictions through
    the first learned event reaches the monitor.  Remaining chunks are marked
    ``post_terminal_diagnostic_only`` and cannot alter the terminal event.
    """
    family = _family(family)
    invocation_index = _nonnegative_int(invocation_index, "invocation_index")
    cooldown = _nonnegative_int(cooldown, "cooldown")
    if isinstance(progress_chunks, (str, bytes)):
        raise ValueError("progress_chunks must be a nonempty sequence")
    try:
        chunks = list(progress_chunks)
    except TypeError as exc:
        raise ValueError("progress_chunks must be a nonempty sequence") from exc
    if not chunks:
        raise ValueError("progress_chunks must be a nonempty sequence")

    # Validate the complete paired matrix before replaying the runtime prefix.
    scalars = [progress_scalar(chunk) for chunk in chunks]
    monitor = ProgressMonitor(family, invocation_index, cooldown)
    predictions: list[dict[str, Any]] = []
    terminal_event: str | None = None
    for prediction_index, (value, source_dtype) in enumerate(scalars):
        if terminal_event is not None:
            predictions.append({
                "prediction_index": prediction_index,
                "source_dtype": source_dtype,
                "value": value,
                "monitor_consumed": False,
                "runtime_disposition": "post_terminal_diagnostic_only",
                "history_length_before": None,
                "history_length_after": None,
                "above_before": None,
                "above_after": None,
                "cooldown_before": None,
                "cooldown_after": None,
                "event": None,
                "queue_clear_reason": "post_terminal_diagnostic_only",
                "expected_discarded_actions": 10,
            })
            continue

        before = {
            "history_length": len(monitor.history),
            "above": monitor.above,
            "cooldown": monitor.replan_cooldown,
        }
        event = monitor.update(value)
        predictions.append({
            "prediction_index": prediction_index,
            "source_dtype": source_dtype,
            "value": value,
            "monitor_consumed": True,
            "runtime_disposition": "runtime_monitor_input",
            "history_length_before": before["history_length"],
            "history_length_after": len(monitor.history),
            "above_before": before["above"],
            "above_after": monitor.above,
            "cooldown_before": before["cooldown"],
            "cooldown_after": monitor.replan_cooldown,
            "event": event,
            "queue_clear_reason": event or "shadow_advance",
            "expected_discarded_actions": 10,
        })
        if event is not None:
            terminal_event = event

    if terminal_event is None:
        predictions[-1]["queue_clear_reason"] = "shadow_sequence_end"
    consumed = sum(row["monitor_consumed"] for row in predictions)
    return {
        "schema": MONITOR_TRACE_SCHEMA,
        "family": family,
        "invocation_index": invocation_index,
        "initial_cooldown": cooldown,
        "runtime_semantics": runtime_semantics(),
        "prediction_count_input": len(predictions),
        "prediction_count_consumed": consumed,
        "post_terminal_diagnostic_count": len(predictions) - consumed,
        "terminal_event": terminal_event,
        "predictions": predictions,
    }


def _replay_trace(trace: Mapping[str, Any], family: str) -> None:
    """Fail if a serialized monitor trace differs from the real monitor."""
    if trace.get("schema") != MONITOR_TRACE_SCHEMA or trace.get(
        "runtime_semantics"
    ) != runtime_semantics():
        raise ValueError("Monitor trace schema/runtime semantics differ")
    if trace.get("family") != family:
        raise ValueError("Monitor trace family differs")
    invocation_index = _nonnegative_int(trace.get("invocation_index"), "invocation_index")
    cooldown = _nonnegative_int(trace.get("initial_cooldown"), "cooldown")
    predictions = trace.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Monitor trace predictions are missing")
    if trace.get("prediction_count_input") != len(predictions):
        raise ValueError("Monitor trace input count differs")

    monitor = ProgressMonitor(family, invocation_index, cooldown)
    terminal_event = None
    consumed = 0
    for index, row in enumerate(predictions):
        if not isinstance(row, Mapping) or row.get("prediction_index") != index:
            raise ValueError("Monitor prediction identity/order differs")
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Serialized monitor value is invalid")
        if not isinstance(row.get("source_dtype"), str) or not row["source_dtype"]:
            raise ValueError("Serialized monitor dtype is missing")
        if terminal_event is not None:
            expected = {
                "monitor_consumed": False,
                "runtime_disposition": "post_terminal_diagnostic_only",
                "history_length_before": None,
                "history_length_after": None,
                "above_before": None,
                "above_after": None,
                "cooldown_before": None,
                "cooldown_after": None,
                "event": None,
                "queue_clear_reason": "post_terminal_diagnostic_only",
                "expected_discarded_actions": 10,
            }
        else:
            before = (len(monitor.history), monitor.above, monitor.replan_cooldown)
            event = monitor.update(float(value))
            consumed += 1
            queue_reason = event or (
                "shadow_sequence_end" if index == len(predictions) - 1 else "shadow_advance"
            )
            expected = {
                "monitor_consumed": True,
                "runtime_disposition": "runtime_monitor_input",
                "history_length_before": before[0],
                "history_length_after": len(monitor.history),
                "above_before": before[1],
                "above_after": monitor.above,
                "cooldown_before": before[2],
                "cooldown_after": monitor.replan_cooldown,
                "event": event,
                "queue_clear_reason": queue_reason,
                "expected_discarded_actions": 10,
            }
            if event is not None:
                terminal_event = event
        for key, expected_value in expected.items():
            if row.get(key) != expected_value:
                raise ValueError(f"Monitor trace field differs: {key}")

    if terminal_event is None and predictions[-1].get("queue_clear_reason") != "shadow_sequence_end":
        raise ValueError("Nonterminal sequence must record its final queue clear")
    if terminal_event is None:
        # The loop compared the provisional reason before knowing this was the
        # final row, so all earlier rows must remain shadow_advance.
        for row in predictions[:-1]:
            if row.get("queue_clear_reason") != "shadow_advance":
                raise ValueError("Nonterminal queue-clear ordering differs")
    if (
        trace.get("prediction_count_consumed") != consumed
        or trace.get("post_terminal_diagnostic_count") != len(predictions) - consumed
        or trace.get("terminal_event") != terminal_event
    ):
        raise ValueError("Monitor trace terminal/count summary differs")


def aggregate_gate_status(statuses: Sequence[str]) -> str:
    """Aggregate statuses using invalid > not_evaluable > failed > passed."""
    if isinstance(statuses, (str, bytes)):
        raise ValueError("statuses must be a nonempty sequence")
    values = list(statuses)
    if not values or any(value not in STATUS_PRECEDENCE for value in values):
        raise ValueError("Unknown or empty gate status sequence")
    return next(status for status in STATUS_PRECEDENCE if status in values)


def _check(status: str, reasons: Sequence[str]) -> dict[str, Any]:
    return {"status": status, "reasons": list(reasons)}


def _invalid_report(reason: str) -> dict[str, Any]:
    checks = {
        "integrity": _check("invalid", [reason]),
        "coverage": _check("not_evaluable", ["Input structure could not be inspected"]),
        "rng_identity": _check("invalid", [reason]),
        "routing": _check("invalid", [reason]),
        "queue_clear": _check("invalid", [reason]),
        "negative_false_completion": _check("invalid", [reason]),
        **_not_evaluable_scope_checks(),
    }
    return _adjudication_report(checks, expected_rows=0, observed_rows=0)


def _not_evaluable_scope_checks() -> dict[str, dict[str, Any]]:
    return {
        "physical_stagnation": _check(
            "not_evaluable",
            ["Learned-progress stagnation is not a physical-stagnation measurement"],
        ),
        "physical_rollback": _check(
            "not_evaluable",
            ["Learned-progress drop is not a measured physical rollback"],
        ),
        "deployment_cadence": _check(
            "not_evaluable",
            ["Fixed offline observations do not establish deployment prediction cadence"],
        ),
        "native_success": _check(
            "not_evaluable",
            ["The pure-CPU gate performs zero simulator steps and no native episode"],
        ),
    }


def _adjudication_report(
    checks: Mapping[str, Mapping[str, Any]], *, expected_rows: int, observed_rows: int
) -> dict[str, Any]:
    offline_names = (
        "integrity", "coverage", "rng_identity", "routing", "queue_clear",
        "negative_false_completion",
    )
    offline_status = aggregate_gate_status([checks[name]["status"] for name in offline_names])
    full_status = aggregate_gate_status([row["status"] for row in checks.values()])
    return {
        "schema": SCHEMA,
        "status": full_status,
        "full_gate_status": full_status,
        "offline_behavior_status": offline_status,
        "status_precedence": list(STATUS_PRECEDENCE),
        "expected_row_count": expected_rows,
        "observed_row_count": observed_rows,
        "checks": dict(checks),
        "runtime_semantics": runtime_semantics(),
        "rng_contract": rng_contract(),
        "training_updates": 0,
        "simulator_steps": 0,
        "physical_stagnation_evaluated": False,
        "physical_rollback_evaluated": False,
        "deployment_cadence_evaluated": False,
        "native_success_evaluated": False,
        "transport_evaluated": False,
        "full_behavior_admission": False,
    }


def adjudicate_behavior_rows(
    rows: Sequence[Mapping[str, Any]],
    queue_events: Sequence[Mapping[str, Any]],
    expected_sequence_ids: Sequence[str],
) -> dict[str, Any]:
    """Adjudicate a complete five-combination offline behavior matrix.

    Structural/provenance defects are ``invalid``; an incomplete predeclared
    matrix is ``not_evaluable``; correctly formed routing, queue, or negative
    false-completion violations are ``failed``.  The full gate additionally
    includes physical behavior, deployment cadence, and native success, which
    this offline CPU module always reports as ``not_evaluable``.
    """
    if isinstance(rows, (str, bytes)) or isinstance(queue_events, (str, bytes)):
        return _invalid_report("Rows and queue events must be sequences")
    if isinstance(expected_sequence_ids, (str, bytes)):
        return _invalid_report("expected_sequence_ids must be a sequence")
    try:
        rows = list(rows)
        queue_events = list(queue_events)
        expected_ids = list(expected_sequence_ids)
    except TypeError:
        return _invalid_report("Gate inputs must be finite sequences")
    if not expected_ids or any(not isinstance(value, str) or not value for value in expected_ids):
        return _invalid_report("Expected sequence ids must be nonempty strings")
    if len(set(expected_ids)) != len(expected_ids):
        return _invalid_report("Expected sequence ids must be unique")

    expected_index = {sequence_id: index for index, sequence_id in enumerate(expected_ids)}
    expected_keys = {(combination, sequence_id) for combination in COMBINATIONS for sequence_id in expected_ids}
    invalid_reasons: list[str] = []
    coverage_reasons: list[str] = []
    rng_reasons: list[str] = []
    routing_reasons: list[str] = []
    queue_reasons: list[str] = []
    false_completion_reasons: list[str] = []
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    valid_traces: dict[tuple[str, str], Mapping[str, Any]] = {}
    sequence_metadata: dict[str, tuple[str, str]] = {}

    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            invalid_reasons.append(f"row[{position}] is not a mapping")
            continue
        combination, sequence_id = row.get("combination"), row.get("sequence_id")
        if combination not in COMBINATIONS:
            invalid_reasons.append(f"row[{position}] has unknown combination")
            continue
        if sequence_id not in expected_index:
            invalid_reasons.append(f"row[{position}] has unknown sequence_id")
            continue
        key = (combination, sequence_id)
        if key in by_key:
            invalid_reasons.append(f"duplicate behavior row: {combination}/{sequence_id}")
            continue
        by_key[key] = row

        family, classification = row.get("family"), row.get("classification")
        if not isinstance(family, str) or family not in FAMILIES:
            invalid_reasons.append(f"{combination}/{sequence_id}: unknown family")
            continue
        if classification not in CLASSIFICATIONS:
            invalid_reasons.append(f"{combination}/{sequence_id}: unknown classification")
            continue
        expected_family = "grasp" if "grasp" in classification else "move"
        if family != expected_family:
            invalid_reasons.append(f"{combination}/{sequence_id}: family/classification mismatch")
            continue
        metadata = (classification, family)
        if sequence_id in sequence_metadata and sequence_metadata[sequence_id] != metadata:
            invalid_reasons.append(f"{sequence_id}: classification/family differs across combinations")
            continue
        sequence_metadata[sequence_id] = metadata
        try:
            sequence_index = _nonnegative_int(row.get("sequence_index"), "sequence_index")
        except ValueError as exc:
            invalid_reasons.append(f"{combination}/{sequence_id}: {exc}")
            continue
        if sequence_index != expected_index[sequence_id]:
            invalid_reasons.append(f"{combination}/{sequence_id}: sequence index differs")
            continue

        head_step, bank_step = row.get("head_step"), row.get("bank_step")
        if (
            isinstance(head_step, bool) or not isinstance(head_step, Integral)
            or isinstance(bank_step, bool) or not isinstance(bank_step, Integral)
        ):
            invalid_reasons.append(f"{combination}/{sequence_id}: route steps are invalid")
            continue
        if int(head_step) != _HEAD_STEPS[combination] or int(bank_step) != _BANK_STEPS[
            combination
        ][family]:
            routing_reasons.append(f"{combination}/{sequence_id}: selected head/bank route differs")

        trace = row.get("monitor")
        if not isinstance(trace, Mapping):
            invalid_reasons.append(f"{combination}/{sequence_id}: monitor trace missing")
            continue
        try:
            _replay_trace(trace, family)
        except (TypeError, ValueError) as exc:
            invalid_reasons.append(f"{combination}/{sequence_id}: {exc}")
            continue
        if trace["prediction_count_input"] != FIXED_SEQUENCE_LENGTH:
            invalid_reasons.append(
                f"{combination}/{sequence_id}: fixed behavior sequence must have 10 predictions"
            )
            continue
        valid_traces[key] = trace

        seeds = row.get("rng_seeds")
        if not isinstance(seeds, list) or len(seeds) != trace["prediction_count_input"]:
            rng_reasons.append(f"{combination}/{sequence_id}: RNG row count differs")
        else:
            for frame_index, seed_row in enumerate(seeds):
                if seed_row != roster_seeds(sequence_index, frame_index):
                    rng_reasons.append(
                        f"{combination}/{sequence_id}/frame{frame_index}: RNG identity differs"
                    )

        physical_completion = row.get("physical_completion")
        if (
            not isinstance(physical_completion, list)
            or len(physical_completion) != trace["prediction_count_input"]
            or any(type(value) is not bool for value in physical_completion)
        ):
            invalid_reasons.append(
                f"{combination}/{sequence_id}: physical completion frames are invalid"
            )
            continue
        threshold_prediction = next(
            (
                prediction["prediction_index"]
                for prediction in trace["predictions"]
                if prediction["event"] == "learned_threshold"
            ),
            None,
        )
        if (
            threshold_prediction is not None
            and physical_completion[threshold_prediction] is False
        ):
            false_completion_reasons.append(
                f"{combination}/{sequence_id}/prediction{threshold_prediction}: "
                "learned threshold before physical completion"
            )

    missing = sorted(expected_keys - set(by_key))
    if missing:
        coverage_reasons.extend(f"missing behavior row: {combination}/{sequence_id}" for combination, sequence_id in missing)
    if set(sequence_metadata) == set(expected_ids):
        classifications = {value[0] for value in sequence_metadata.values()}
        if classifications != set(CLASSIFICATIONS):
            coverage_reasons.append("Expected roster does not cover all four handoff classifications")
    else:
        coverage_reasons.append("Sequence metadata is incomplete")

    queue_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for position, event in enumerate(queue_events):
        if not isinstance(event, Mapping):
            invalid_reasons.append(f"queue_event[{position}] is not a mapping")
            continue
        combination, sequence_id = event.get("combination"), event.get("sequence_id")
        prediction_index = event.get("prediction_index")
        if (
            combination not in COMBINATIONS
            or sequence_id not in expected_index
            or isinstance(prediction_index, bool)
            or not isinstance(prediction_index, Integral)
            or prediction_index < 0
        ):
            invalid_reasons.append(f"queue_event[{position}] identity is invalid")
            continue
        key = (combination, sequence_id, int(prediction_index))
        if key in queue_by_key:
            invalid_reasons.append(
                f"duplicate queue event: {combination}/{sequence_id}/{prediction_index}"
            )
            continue
        queue_by_key[key] = event

    expected_queue_keys: set[tuple[str, str, int]] = set()
    for (combination, sequence_id), trace in valid_traces.items():
        for prediction in trace["predictions"]:
            index = prediction["prediction_index"]
            key = (combination, sequence_id, index)
            expected_queue_keys.add(key)
            event = queue_by_key.get(key)
            label = f"{combination}/{sequence_id}/prediction{index}"
            if event is None:
                queue_reasons.append(f"{label}: queue clear is missing")
                continue
            discarded = event.get("discarded_actions")
            if (
                event.get("reason") != prediction["queue_clear_reason"]
                or event.get("queue_empty") is not True
                or isinstance(discarded, bool)
                or not isinstance(discarded, Integral)
                or int(discarded) != prediction["expected_discarded_actions"]
            ):
                queue_reasons.append(f"{label}: queue clear semantics differ")
    extra_queue = sorted(set(queue_by_key) - expected_queue_keys)
    invalid_reasons.extend(
        f"unexpected queue event: {combination}/{sequence_id}/{index}"
        for combination, sequence_id, index in extra_queue
    )

    checks = {
        "integrity": _check("invalid" if invalid_reasons else "passed", invalid_reasons),
        "coverage": _check("not_evaluable" if coverage_reasons else "passed", coverage_reasons),
        "rng_identity": _check("invalid" if rng_reasons else "passed", rng_reasons),
        "routing": _check("failed" if routing_reasons else "passed", routing_reasons),
        "queue_clear": _check("failed" if queue_reasons else "passed", queue_reasons),
        "negative_false_completion": _check(
            "failed" if false_completion_reasons else "passed", false_completion_reasons
        ),
        **_not_evaluable_scope_checks(),
    }
    return _adjudication_report(
        checks, expected_rows=len(expected_keys), observed_rows=len(rows)
    )
