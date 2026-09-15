"""Bounded, source-labelled invocation memory; never asserts task success."""

from .progress_monitor import THRESHOLDS


def object_memory(history):
    objects = {}
    for item in history:
        target, family = item.get("target"), item.get("family")
        if target is None or family not in THRESHOLDS:
            continue
        obj = objects.setdefault(
            target, {"native_goal": "unknown", "last_by_family": {}}
        )
        obj["last_by_family"][family] = {
            key: item[key]
            for key in ("reason", "progress", "source", "executed_steps", "end_step")
            if key in item
        }
        obj["last_family"] = family
    return objects


def transition_rejection(history, family, target):
    """Require re-localization after an explicitly incomplete reach.

    This is a disclosed coordinator guard, not a simulator oracle or learned
    success detector. Rejected decisions consume a request but execute no action.
    """
    last = next((x for x in reversed(history) if x.get("target") == target), None)
    if not last or last.get("family") != "reach" or family != "grasp":
        return None
    progress = last.get("progress")
    incomplete = last.get("reason") in ("learned_drop", "learned_stagnation") or (
        last.get("reason") == "step_limit"
        and progress is not None
        and progress < THRESHOLDS["reach"]
    )
    if incomplete:
        return "Previous reach is incomplete; inspect both images and re-localize with reach before grasp."
    return None
