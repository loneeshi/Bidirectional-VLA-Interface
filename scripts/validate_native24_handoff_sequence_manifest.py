"""Validate a native24 B-prime sequence manifest without loading a model."""

import argparse
import json
from pathlib import Path

from bvi.native24_handoff_sequence import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--progress-monitor-source", type=Path, required=True)
    args = parser.parse_args()
    report, _, _, _ = load_manifest(
        args.manifest, args.training_manifest, args.input_manifest,
        args.progress_monitor_source,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
