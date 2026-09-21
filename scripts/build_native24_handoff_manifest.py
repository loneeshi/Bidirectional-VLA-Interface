"""Build a hash-bound bvi.native24-handoff/1 input manifest on CPU.

The command consumes already collected, replay-verified evidence.  It does not
run a simulator, call a policy, update weights, or infer labels from progress.
All referenced artifacts must live below the output manifest directory.
"""

import argparse
import json
from pathlib import Path

from bvi.native24_handoff import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-index", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.acquisition_index, args.training_manifest, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
