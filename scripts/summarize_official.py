#!/usr/bin/env python3
"""Summarize passive official traces as diagnostic evidence, not benchmark scores.

Only explicitly successful, nonterminal +1 pointer transitions count as skill
completion evidence. Autoresets, gaps, missing flags, and unknown stage names
cannot manufacture a continuous navigation/pick/navigation/place chain.

--controller-provenance=official_learned_navigation is an explicit operator
attestation after inspecting the pinned evaluator/config: it dispatches the
learned navigation policy and does not invoke teleport navigation. The flag is
not a fact inferred from the trace. Without it, a no-teleport G3 verdict is unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


STAGES = {0: "pick", 1: "place", 2: "navigate", 3: "open", 4: "close"}


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def scalar(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return None if isinstance(value, (list, dict)) else value


def flag(value: Any) -> bool | None:
    value = scalar(value)
    return value if isinstance(value, bool) else None


def integer(value: Any) -> int | None:
    value = scalar(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def stage_names(plan: Any) -> list[str | None]:
    if isinstance(plan, dict):
        plan = plan.get("subtasks", [])
    if not isinstance(plan, list):
        return []
    return [entry.get("type") if isinstance(entry, dict) and entry.get("type") in STAGES.values()
            else None for entry in plan]


def effective_info(info: Any) -> tuple[dict, bool, str]:
    """Use valid final_info for terminal outcomes, never a new reset's outcome."""
    if not isinstance(info, dict):
        return {}, False, "missing_info"
    if "final_info" not in info:
        return info, False, "info"
    if "_final_info" in info and flag(info["_final_info"]) is not True:
        return {}, True, "masked_or_ambiguous_final_info"
    final = info["final_info"]
    while isinstance(final, list) and len(final) == 1:
        final = final[0]
    return (final if isinstance(final, dict) else {}), True, "final_info"


def action_evidence(raw: Any) -> dict:
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        return {"valid_shape": False, "finite": None, "out_of_bounds_indices": [], "raw_action": raw}
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw)
    finite = all(math.isfinite(x) for x in raw) if numeric else None
    return {"valid_shape": len(raw) == 13 and numeric, "finite": finite,
            "out_of_bounds_indices": [i for i, x in enumerate(raw) if isinstance(x, (int, float))
                                       and not isinstance(x, bool) and math.isfinite(x) and abs(x) > 1],
            "raw_action": raw}


def summarize(records: list[dict], metadata: dict | None = None,
              controller_provenance: str = "unknown") -> dict:
    if controller_provenance not in ("unknown", "official_learned_navigation"):
        raise ValueError("Unknown controller provenance")
    metadata = metadata or {}
    summary = {
        "schema_version": 1, "kind": "official_trace_diagnostics", "benchmark_score": False,
        "dispatcher": metadata.get("dispatcher"), "vlm": metadata.get("vlm"),
        "steps": 0, "reset_events": [], "invalid_action_events": [], "boundaries": [],
        "pointer_advances": [], "skill_completion_evidence": [], "chain_evidence": [],
        "failures": [], "success_evidence": [], "unknowns": [], "stage_actions": {},
    }
    action_counts = Counter()
    stage_counts: dict[str, Counter] = {}
    failures = Counter()
    active_reset = None
    episode = -1
    names: list[str | None] = []
    last_step = last_after = None
    chain = None
    # None means no explicit successful transition is observed; absence is not a failure score.
    success_known = success_unknown = 0

    def boundary(line, step, reasons):
        nonlocal chain
        chain = None
        summary["boundaries"].append({"line": line, "step": step, "reasons": reasons, "episode": episode})

    for line, row in enumerate(records, 1):
        event = row.get("event")
        if event == "reset":
            episode += 1
            names = stage_names(row.get("task_plan"))
            active_reset = {"line": line, "episode": episode, "seed": row.get("seed"),
                            "build_config_idxs": row.get("build_config_idxs"),
                            "task_plan_idxs": row.get("task_plan_idxs"), "stage_names": names}
            summary["reset_events"].append(active_reset)
            last_step = last_after = chain = None
            continue
        if event == "invalid_action":
            summary["invalid_action_events"].append({"line": line, **row})
            boundary(line, row.get("step"), ["invalid_action_event"])
            continue
        if event != "step":
            continue
        summary["steps"] += 1
        step, before, after = (integer(row.get(k)) for k in ("step", "subtask_before", "subtask_after"))
        info, autoreset, info_source = effective_info(row.get("info"))
        fail, success = flag(info.get("fail")), flag(info.get("success"))
        terminated, truncated = flag(row.get("terminated")), flag(row.get("truncated"))
        action = action_evidence(row.get("raw_action"))
        stage = names[before] if before is not None and 0 <= before < len(names) else None
        # info.subtask_type can refer to the post-step state; it is retained as evidence,
        # but is not used to fill a missing reset-time action-stage label.
        evidence = {"line": line, "step": step, "episode": episode,
                    "reset_line": active_reset["line"] if active_reset else None,
                    "before": before, "after": after, "stage": stage,
                    "info_subtask_type": info.get("subtask_type"), "fail": fail,
                    "terminated": terminated, "truncated": truncated, "autoreset": autoreset,
                    "info_source": info_source, "action": action}
        counts = stage_counts.setdefault(stage or "unknown", Counter())
        for counter in (action_counts, counts):
            counter["steps"] += 1
            if not action["valid_shape"]:
                counter["invalid_shape_steps"] += 1
            if action["finite"] is True:
                counter["finite_steps"] += 1
            elif action["finite"] is False:
                counter["nonfinite_steps"] += 1
            else:
                counter["unknown_finiteness_steps"] += 1
            if action["out_of_bounds_indices"]:
                counter["raw_outside_unit_box_steps"] += 1
                counter["raw_outside_unit_box_values"] += len(action["out_of_bounds_indices"])
        reported_bounds = flag(row.get("raw_outside_unit_box"))
        if reported_bounds is not None and reported_bounds != bool(action["out_of_bounds_indices"]):
            summary["unknowns"].append({"line": line, "reason": "raw_bound_flag_disagrees_with_values"})
        if success is None:
            success_unknown += 1
        else:
            success_known += 1
            if success:
                summary["success_evidence"].append({"line": line, "step": step, "source": info_source,
                                                    "episode": episode, "success": True})
        if fail is True:
            types = ["benchmark_fail"]
            if flag(info.get("cumulative_force_within_limit")) is False:
                types.append("cumulative_force_limit")
            remaining = scalar(info.get("subtasks_steps_left"))
            if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining <= 0:
                types.append("subtask_budget_exhausted")
            failures.update(types)
            summary["failures"].append({**evidence, "types": types,
                "subtasks_steps_left": info.get("subtasks_steps_left"),
                "robot_cumulative_force": info.get("robot_cumulative_force"),
                "false_checkers": [k for k, value in info.items() if flag(value) is False]})
        issues = []
        if active_reset is None:
            issues.append("no_observed_reset_for_episode")
        if step is None or before is None or after is None:
            issues.append("missing_or_multienvironment_pointer")
        if last_step is not None and step != last_step + 1:
            issues.append("step_sequence_gap")
        if last_after is not None and before != last_after:
            issues.append("pointer_discontinuity")
        if before is not None and after is not None and after not in (before, before + 1):
            issues.append("pointer_reset_or_jump")
        if fail is not False:
            issues.append("benchmark_fail" if fail else "unknown_fail_flag")
        if terminated is not False or truncated is not False or autoreset:
            issues.append("terminal_autoreset_or_unknown_terminal_flags")
        if not action["valid_shape"] or action["finite"] is not True:
            issues.append("invalid_or_unknown_action")
        if issues:
            boundary(line, step, issues)
        if before is not None and after is not None and after > before:
            summary["pointer_advances"].append({**evidence, "qualifies": not issues,
                                                 "exclusion_reasons": issues})
        if not issues:
            if chain is None and before == 0 and names[:4] == ["navigate", "pick", "navigate", "place"]:
                chain = {"episode": episode, "reset_line": active_reset["line"],
                         "first_step": step, "transitions": [], "steps": 0}
            if chain is not None:
                chain["steps"] += 1
            if after == before + 1:
                summary["skill_completion_evidence"].append(evidence)
                if chain is not None:
                    expected = len(chain["transitions"])
                    if before != expected:
                        chain = None
                    else:
                        chain["transitions"].append(evidence)
                        if after == 4:
                            summary["chain_evidence"].append({**chain, "last_step": step,
                                "pointer_sequence": [0, 1, 2, 3, 4],
                                "stage_sequence": ["navigate", "pick", "navigate", "place"],
                                "no_observed_reset_or_gap_within_chain": True,
                                "teleport_absence_verified_by_trace": None})
                            chain = None
        last_step, last_after = step, after
        if autoreset or terminated is True or truncated is True:
            # Internal auto-reset is not intercepted by run_official's outer reset logger.
            episode += 1
            active_reset = None
            names = []
            last_step = last_after = chain = None

    summary["action_counts"] = dict(action_counts)
    summary["stage_actions"] = {stage: dict(counts) for stage, counts in stage_counts.items()}
    summary["failure_type_counts"] = dict(failures)
    full_success = (True if summary["success_evidence"] else
                    False if success_known and not success_unknown else None)
    summary["fulltask_success_once_observed"] = full_success
    summary["success_observation_coverage"] = {"known_steps": success_known, "unknown_steps": success_unknown}
    observed = Counter(e["stage"] for e in summary["skill_completion_evidence"] if e["stage"])
    summary["g2"] = {"passed": None, "successful_transition_counts": dict(observed),
                     "navigate_pick_place_each_observed": all(observed[s] > 0 for s in ("navigate", "pick", "place")),
                     "limitation": "Trace records raw policy actions, not post-wrapper/controller bounds; gate requires independent interface checks."}
    summary["g3"] = {"passed": None, "continuous_chain_observed": bool(summary["chain_evidence"]),
                     "chain_count": len(summary["chain_evidence"]),
                     "limitation": "Pointer evidence does not prove absence of teleport calls; verify pinned runner/environment provenance separately."}
    attested = controller_provenance == "official_learned_navigation"
    summary["controller_provenance"] = {
        "mode": controller_provenance,
        "source": "explicit_operator_attestation" if attested else "unknown",
        "learned_navigation_without_teleport_attested": True if attested else None,
        "inferred_from_trace": False,
    }
    if attested and summary["chain_evidence"]:
        summary["g3"]["passed"] = True
        summary["g3"]["verdict_basis"] = "continuous_trace_evidence_plus_explicit_controller_provenance_attestation"
    return summary


def compact_stdout(result: dict) -> dict:
    first = result["failures"][0] if result["failures"] else None
    first_failure = ({key: first.get(key) for key in (
        "line", "step", "stage", "before", "after", "types", "robot_cumulative_force",
        "subtasks_steps_left", "false_checkers"
    )} if first else None)
    completions = [{key: event.get(key) for key in ("line", "step", "episode", "stage", "before", "after")}
                   for event in result["skill_completion_evidence"]]
    return {
        "steps": result["steps"], "action_counts": result["action_counts"],
        "failure_step_count": len(result["failures"]),
        "failure_type_counts": result["failure_type_counts"],
        "reset_event_count": len(result["reset_events"]),
        "first_failure": first_failure, "completion_evidence": completions,
        "g2": result["g2"], "g3": result["g3"],
        "fulltask_success_once_observed": result["fulltask_success_once_observed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="run_official steps.jsonl")
    parser.add_argument("--metadata", type=Path, help="Defaults to sibling run-metadata.json if present")
    parser.add_argument("--output", type=Path, help="Defaults to sibling diagnostic-summary.json")
    parser.add_argument("--controller-provenance", choices=("unknown", "official_learned_navigation"),
                        default="unknown", help="Explicit operator attestation of inspected runner/config; not trace inference")
    parser.add_argument("--compact-stdout", action="store_true",
                        help="Print counts, first failure, completion steps and gate diagnostics; JSON artifact retains all evidence")
    args = parser.parse_args()
    records = []
    with args.trace.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            try:
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("expected JSON object")
            except (ValueError, json.JSONDecodeError) as error:
                raise SystemExit(f"Invalid trace at line {line}: {error}; do not silently skip incomplete rows")
            records.append(value)
    metadata_path = args.metadata or args.trace.with_name("run-metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    result = summarize(records, metadata, args.controller_provenance)
    result["trace"] = str(args.trace.resolve())
    result["metadata"] = str(metadata_path.resolve()) if metadata_path.exists() else None
    output = args.output or args.trace.with_name("diagnostic-summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    display = (compact_stdout(result) if args.compact_stdout else
               {"steps": result["steps"], "skill_counts": result["g2"]["successful_transition_counts"],
                "chain_count": result["g3"]["chain_count"],
                "fulltask_success_once_observed": result["fulltask_success_once_observed"]})
    print(json.dumps(json_safe({"summary": str(output.resolve()), **display}), allow_nan=False))


if __name__ == "__main__":
    main()
