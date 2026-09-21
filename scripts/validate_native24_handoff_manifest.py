"""Validate bvi.native24-handoff/1 and all bound artifacts on CPU."""

import argparse
import json
from pathlib import Path

from bvi.native24_handoff import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_manifest(args.manifest, args.training_manifest),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
