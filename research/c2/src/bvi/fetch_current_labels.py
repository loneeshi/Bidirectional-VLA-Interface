"""Current-observation labels for verified local teacher intervals.

STATUS: frozen — historical training and diagnostics (retained)

Observation indices include both interval endpoints; action indices exclude the
last endpoint. A failed rollout's end is not a verified completion boundary.
Invocation-start anchors are supplementary reconstructed temporal supervision,
not physical failure labels or author-released negative labels.
"""

from collections.abc import Mapping
from numbers import Integral


CONTRACT = "current_observation_v2"
ANCHOR_SUPERVISION = "reconstructed_invocation_start_temporal_anchor"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _interval(start, end):
    start = _integer("start", start)
    end = _integer("end", end)
    if end <= start:
        raise ValueError("A verified interval must have end > start")
    return start, end


def teacher_targets(start, end, t, horizon=2):
    """Build targets for an externally verified teacher completion boundary.

    This arithmetic function cannot establish physical completion. Use
    ``verified_teacher_targets`` for manifests, after auditing their evidence.
    """
    start, end = _interval(start, end)
    t = _integer("t", t)
    horizon = _integer("horizon", horizon, minimum=1)
    if not start <= t <= end:
        raise ValueError("t must lie in the inclusive observation interval")
    return {
        "progress_target": [
            (t + j - start) / (end - start) if t + j <= end else 0.0
            for j in range(horizon)
        ],
        "progress_valid": [t + j <= end for j in range(horizon)],
        "action_valid": [t + j < end for j in range(horizon)],
    }


def verified_teacher_targets(window, t, horizon=2):
    """Require recorded boundary evidence; never infer it from rollout length.

    Nonempty evidence is a provenance requirement, not independent verification
    of the stated predicate. The collection/segmentation audit supplies that.
    """
    evidence = window.get("completion_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Teacher completion requires recorded completion_evidence")
    if window.get("completion_verified") is False:
        raise ValueError("An explicitly unverified endpoint cannot be completion")
    return teacher_targets(window["start"], window["end"], t, horizon)


def first_call_anchor(horizon=2):
    """Label only the first observation's local invocation clock as zero.

    Future progress and all actions remain unsupervised. This does not assign a
    physical failure score and must not be applied to later failed observations.
    """
    horizon = _integer("horizon", horizon, minimum=1)
    return {
        "progress_target": [0.0] * horizon,
        "progress_valid": [True] + [False] * (horizon - 1),
        "action_valid": [False] * horizon,
    }


def sample_times(start, end):
    """Deterministic inclusive support, using floor for quarter positions."""
    start, end = _interval(start, end)
    length = end - start
    return sorted({start, *(start + k * length // 4 for k in (1, 2, 3)), end - 1, end})


def validate_manifest_splits(manifests):
    """Reject split leakage among all descendants of an original episode.

    Each row has ``split`` and either a ``parent_episode`` mapping containing
    ``scene_split, task, seed``, or those three fields directly. Descendant rows
    must retain the original parent identity, including across teacher recovery,
    decision capture and replay; paths and derived run IDs are not identities.
    Returns the parent-to-split mapping after validation.
    """
    assignments = {}
    for row in manifests:
        if not isinstance(row, Mapping):
            raise ValueError("Manifest entries must be mappings")
        split = row.get("split")
        if split not in ("train", "validation", "test", "diagnostic"):
            raise ValueError("Unknown or missing manifest split")
        parent = row.get("parent_episode", row)
        if not isinstance(parent, Mapping):
            raise ValueError("parent_episode must be a mapping")
        scene_split, task = parent.get("scene_split"), parent.get("task")
        if not isinstance(scene_split, str) or not scene_split.strip():
            raise ValueError("Parent scene_split is required")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Parent task is required")
        seed = _integer("parent seed", parent.get("seed"))
        key = (scene_split, task, seed)
        if split == "train" and seed in (2025, 2030):
            raise ValueError(f"Diagnostic seed {seed} cannot enter training")
        if key in assignments and assignments[key] != split:
            raise ValueError(f"Parent episode {key!r} crosses splits")
        assignments[key] = split
    return assignments
