"""Bounded remote recovery runner. All paired-state audits precede paid GPT calls."""

import argparse
import datetime
import json
import os
import pathlib
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/tapt")
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 7200:
        raise ValueError(
            "Recovery runner limited to two hours; reserve time for backup"
        )
    root = pathlib.Path(args.root)
    evidence = root / "evidence" / "recovery"
    evidence.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "openpi-cache/openpi-assets/checkpoints/pi05_libero"
    metadata = json.loads((root / "new/metadata.json").read_text())
    if (
        metadata["sha256"]
        != "ee48e60dd979985cfd17ee126b1c703bd7c29f82907821cb2e801e66d90b67f2"
    ):
        raise ValueError("Wrong preregistered checkpoint")
    (evidence / "evaluation-lock.json").write_text(
        json.dumps({"adapter_metadata": metadata})
    )
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONPATH=str(root) + ":" + str(root / "author/third_party/libero"),
        OPENPI_DATA_HOME=str(root / "openpi-cache"),
        XLA_PYTHON_CLIENT_MEM_FRACTION=".8",
    )
    deadline = time.monotonic() + args.max_seconds
    sim = str(root / "sim-env/bin/python")
    server = [
        str(root / "author/.venv/bin/python"),
        str(root / "serve_libero_family.py"),
        "--checkpoint",
        str(checkpoint),
        "--adapters",
        str(root / "new"),
    ]
    jobs = []
    for episode in range(5):
        for label, dx in (("sham", 0), ("shift-x-3cm", 0.03)):
            jobs.append((episode, label, dx))
    summary = []

    def run(episode, label, dx, disabled, audit, expected=None):
        name = f"episode{episode:03d}-{label}-{'bounded' if disabled else 'learned'}"
        output = evidence / (name + ("-audit" if audit else ""))
        command = [
            sim,
            str(root / "eval_libero_family.py"),
            "--mode",
            "tapt",
            "--output",
            str(output),
            "--coordinator-mode",
            "memory-recovery",
            "--authorization-id",
            args.authorization_id,
            "--recovery-prefix",
            str(root / "prefixes" / f"episode{episode:03d}.jsonl"),
            "--episode-index",
            str(episode),
            "--displacement-x",
            str(dx),
        ]
        if disabled:
            command.append("--feedback-disabled")
        if audit:
            command.append("--recovery-audit-only")
        if expected:
            command.extend(["--expected-state-hash", expected])
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Batch time budget exhausted")
        with (evidence / (output.name + ".log")).open("w") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
            )
        return output

    with (evidence / "server.log").open("w") as log:
        process = subprocess.Popen(
            server, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError("Policy server exited")
                try:
                    with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Policy server startup")
            gates = {}
            # Fail closed on any invalid injection or state mismatch. No API yet.
            for episode, label, dx in jobs:
                left = run(episode, label, dx, False, True)
                right = run(episode, label, dx, True, True)
                a = json.loads((left / "branch-state.json").read_text())
                b = json.loads((right / "branch-state.json").read_text())
                if a["state_sha256"] != b["state_sha256"]:
                    raise ValueError("Paired simulator/controller/observation mismatch")
                if json.loads((left / "policy-state.json").read_text()) != json.loads(
                    (right / "policy-state.json").read_text()
                ):
                    raise ValueError("Paired policy state mismatch")
                gates[f"{episode}-{label}"] = a["state_sha256"]
            (evidence / "paired-gates.json").write_text(json.dumps(gates, indent=2))
            print("ALL PAIRED GATES PASSED", flush=True)
            for episode, label, dx in jobs:
                # Alternate arm order to reduce systematic time/order effects.
                for disabled in (False, True) if episode % 2 == 0 else (True, False):
                    output = run(
                        episode, label, dx, disabled, False, gates[f"{episode}-{label}"]
                    )
                    rows = json.loads((output / "summary.json").read_text())
                    summary.append(
                        {
                            "condition": "bounded" if disabled else "learned",
                            "intervention": label,
                            "directory": output.name,
                            **rows[0],
                        }
                    )
                    (evidence / "summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                    print("EPISODE COMPLETE", summary[-1], flush=True)
            (evidence / "complete.json").write_text(
                json.dumps(
                    {
                        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "episodes": len(summary),
                    }
                )
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
