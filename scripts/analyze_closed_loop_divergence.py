"""Analyze first closed-loop divergence only under strict exact-start pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bvi.closed_loop_divergence import (
    FETCH_QPOS_CHANNELS,
    FETCH_QPOS_UNITS,
    analyze_closed_loop_divergence,
    run_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text())


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-events", type=Path, required=True)
    parser.add_argument("--expert-events", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--expert-result", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        required=True,
        help="JSON object with qpos_abs_error keys for all 15 Fetch qpos channels",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    paths = (
        args.policy_events,
        args.expert_events,
        args.policy_result,
        args.expert_result,
        args.thresholds,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f"Missing input files: {missing}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    threshold_document = _json(args.thresholds)
    thresholds = threshold_document.get("qpos_abs_error")
    if not isinstance(thresholds, dict):
        parser.error("Threshold file must contain a qpos_abs_error object")
    result = analyze_closed_loop_divergence(
        _events(args.policy_events),
        _events(args.expert_events),
        run_identity(_json(args.policy_result)),
        run_identity(_json(args.expert_result)),
        thresholds,
        channel_names=FETCH_QPOS_CHANNELS,
        units=FETCH_QPOS_UNITS,
        restore_tolerance=args.restore_tolerance,
    )
    result["provenance"] = {
        "policy_events": {"path": str(args.policy_events), "sha256": _sha256(args.policy_events)},
        "expert_events": {"path": str(args.expert_events), "sha256": _sha256(args.expert_events)},
        "policy_result": {"path": str(args.policy_result), "sha256": _sha256(args.policy_result)},
        "expert_result": {"path": str(args.expert_result), "sha256": _sha256(args.expert_result)},
        "thresholds": {"path": str(args.thresholds), "sha256": _sha256(args.thresholds)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["comparison_performed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
