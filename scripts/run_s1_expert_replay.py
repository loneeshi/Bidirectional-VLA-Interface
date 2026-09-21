"""Run the paired Level-2 expert replay: CPU reset and GPU snapshot restore."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=int, default=0)
    parser.add_argument(
        "--source-env-index",
        type=int,
        default=0,
        help="recording worker for the selected parent (traj_0 was worker 0)",
    )
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=720)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    child = Path(__file__).with_name("replay_s1_expert.py")
    report: dict[str, Any] = {
        "stage": "S1-Level2",
        "question": "CPU metadata-reset versus reconstructed GPU snapshot-restore expert replay",
        "status": "running",
        "parent": args.parent,
        "source_env_index": args.source_env_index,
        "original_snapshot_available": False,
        "gpu_snapshot_scope": "new metadata reconstruction saved/restored; not original collection snapshot",
        "max_actions_per_path": args.max_actions,
        "max_seconds": args.max_seconds,
        "training_updates": 0,
        "api_calls": 0,
        "new_rental_usd": 0,
        "lab_charge_usd": None,
        "runner_sha256": sha256(Path(__file__)),
        "child_runner_sha256": sha256(child),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rows": [],
    }

    def save() -> None:
        report["wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))

    save()
    failures: list[str] = []
    try:
        for path in ("cpu_reset", "gpu_snapshot_restore"):
            name = f"parent{args.parent:03d}-{path}"
            output = args.output / name
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining < 60:
                raise TimeoutError("insufficient remaining paired-replay window")
            timeout = min(330, max(60, int(remaining - 15)))
            command = [
                sys.executable,
                str(child),
                "--parent",
                str(args.parent),
                "--source-env-index",
                str(args.source_env_index),
                "--path",
                path,
                "--shader",
                "minimal",
                "--max-actions",
                str(args.max_actions),
                "--max-seconds",
                str(max(30, timeout - 15)),
                "--output",
                str(output),
            ]
            log_path = args.output / f"{name}.log"
            with log_path.open("x") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            result = (
                json.loads((output / "result.json").read_text())
                if (output / "result.json").exists()
                else {"status": "no_result"}
            )
            report["rows"].append(
                {
                    "name": name,
                    "path": path,
                    "exit_code": completed.returncode,
                    "log": log_path.name,
                    "result": result,
                }
            )
            if completed.returncode != 0:
                failures.append(name)
            save()
        if failures:
            report.update(status="partial_infrastructure_failure", failed_paths=failures)
            raise RuntimeError(f"expert replay path(s) failed: {failures}")
        report["gpu_memory_after_children_mib"] = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    GPU,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        report["status"] = "completed_paired_expert_state_comparison"
    except Exception as error:
        if report["status"] == "running":
            report["status"] = "infrastructure_failure"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
