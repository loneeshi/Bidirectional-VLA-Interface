"""Pure helpers for paired-prefix recovery experiments; no simulator truth in prompts.

STATUS: frozen — historical training and diagnostics (retained)
"""

import copy
import hashlib
import json


def visible_history(history, feedback_enabled):
    result = copy.deepcopy(history)
    if not feedback_enabled:
        for item in result:
            item.pop("progress", None)
            item.pop("rule_completed", None)
            item["source"] = "bounded_call"
            if item.get("reason", "").startswith("learned_"):
                item["reason"] = (
                    "prefix_boundary" if item.get("shared_prefix") else "step_limit"
                )
    return result


def extract_prefix(records):
    """Stop before the first cream-cheese grasp request, never select by outcome."""
    boundary = None
    for index, record in enumerate(records):
        if record["event"] == "vlm_raw_response":
            decision = json.loads(record["raw_text"])
            if (
                decision["tool_family"] == "grasp"
                and decision["target"] == "cream_cheese_1"
            ):
                boundary = index
                break
    if boundary is None:
        raise ValueError("No first cream-cheese grasp boundary")
    # Exclude the request that selected grasp: branching planner decides anew.
    while boundary and records[boundary]["event"] != "vlm_request":
        boundary -= 1
    prior = records[:boundary]
    history = [
        dict(r, shared_prefix=True)
        for r in prior
        if r["event"] == "invocation_finished"
    ]
    actions = [r["action"] for r in prior if r["event"] == "action"]
    if not actions or not history or history[-1]["family"] != "reach":
        raise ValueError("Prefix does not end at a reach boundary")
    return {
        "actions": actions,
        "history": history,
        "calls": sum(r["event"] == "vlm_request" for r in prior),
        "predictions": sum(r["event"] == "learned_progress" for r in prior),
        "last_action_diagnostic": next(
            r["diagnostic_only"] for r in reversed(prior) if r["event"] == "action"
        ),
    }


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def public_feedback(feedback, enabled):
    return visible_history([feedback], enabled)[0]


def progress_event(monitor, value, feedback_enabled):
    """Disabled feedback must not even mutate the scheduler's monitor state."""
    return monitor.update(value) if feedback_enabled else None
