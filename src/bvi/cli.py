"""Public command line for the three frozen MS-HAB evaluation settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import PROFILES, render_markdown, run_panel, summarize_panels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bvi-eval", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Preview or execute one frozen evaluation setting")
    run.add_argument("--setting", choices=tuple(PROFILES), required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--reference-panel", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--mshab-root", type=Path)
    run.add_argument("--asset-dir", type=Path)
    run.add_argument("--bridge-dir", type=Path)
    run.add_argument("--authorization-id")
    run.add_argument("--max-new", type=int, default=2)
    run.add_argument("--retry-infrastructure", action="store_true")
    run.add_argument("--no-video", action="store_true")
    run.add_argument("--execute", action="store_true", help="Required for any simulator/API execution")

    summarize = subparsers.add_parser("summarize", help="Validate evidence and generate the public table")
    summarize.add_argument("--paired-panel", type=Path, required=True)
    summarize.add_argument("--teleport-panel", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    subparsers.add_parser(
        "bridge",
        add_help=False,
        help="Serve the bounded GPT provider bridge (GPT setting only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bridge":
        from .bridge_server import main as bridge_main
        return int(bridge_main(argv[1:]) or 0)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.execute and args.mshab_root is None:
            parser.error("--execute requires --mshab-root")
        try:
            return run_panel(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "summarize":
        try:
            summary = summarize_panels(args.paired_panel, args.teleport_panel)
        except ValueError as exc:
            parser.error(str(exc))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
