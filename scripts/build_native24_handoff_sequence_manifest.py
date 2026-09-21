"""Build and immediately validate a frozen native24 B-prime sequence manifest."""

import argparse
import json
from pathlib import Path

from bvi.native24_handoff_sequence import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-index", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--progress-monitor-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_manifest(
        args.acquisition_index, args.training_manifest, args.input_manifest,
        args.progress_monitor_source, args.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
