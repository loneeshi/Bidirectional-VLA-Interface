"""Serve remote simulator VLM requests with a key that stays on this computer.

Only the configured SSH alias and selected model provider are contacted. The
server runs serially, caches responses, and stops on an uncertain/error request.
Reusing the output directory preserves claims and the spending scope on restart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from bvi import APIBudget, ProtocolError
from bvi.bridge import BridgeProcessor, atomic_json, validate_bridge_id
from bvi.providers import AnthropicTransport, OpenAITransport
from run_coordinator import load_local_credentials


class SSHSpool:
    def __init__(self, config: Path, alias: str, directory: str):
        path = PurePosixPath(directory)
        if (not path.is_absolute() or ".." in path.parts or str(path) == "/"
                or not re.fullmatch(r"/[A-Za-z0-9_./-]+", directory)):
            raise ValueError("Remote bridge directory must be an absolute path without spaces or parent traversal")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", alias):
            raise ValueError("Use an explicit configured SSH alias")
        self.config, self.alias, self.directory = config.resolve(), alias, str(path)

    def ssh_python(self, code: str, *args: str) -> str:
        command = " ".join(shlex.quote(value) for value in ("python3", "-c", code, *args))
        result = subprocess.run(["ssh", "-F", str(self.config), "-o", "BatchMode=yes",
            self.alias, command], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError("SSH spool operation failed; inspect connection separately")
        return result.stdout

    def pending(self) -> list[str]:
        code = ("import json,pathlib,re,sys; p=pathlib.Path(sys.argv[1]); "
                "print(json.dumps(sorted(x.name for x in p.iterdir() if "
                "re.fullmatch('[0-9a-f]{32}',x.name) and (x/'request.json').is_file() "
                "and not (x/'response.json').exists() and not (x/'expired.json').exists()) "
                "if p.exists() else []))")
        return [validate_bridge_id(value) for value in json.loads(self.ssh_python(code, self.directory))]

    def read(self, attempt_id: str) -> dict:
        validate_bridge_id(attempt_id)
        code = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                "assert not (p/'expired.json').exists(), 'expired'; "
                "q=p/'request.json'; assert q.stat().st_size<=2000000, 'oversized'; "
                "print(q.read_text())")
        envelope = json.loads(self.ssh_python(code, f"{self.directory}/{attempt_id}"))
        if envelope.get("bridge_id") != attempt_id:
            raise ProtocolError("Spool filename and envelope ID disagree")
        return envelope

    def reply(self, attempt_id: str, local_file: Path) -> None:
        validate_bridge_id(attempt_id)
        # Retry only delivery of an immutable cached response, never the API.
        for attempt in range(3):
            try:
                self._reply_once(attempt_id, local_file)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == 2:
                    raise
                time.sleep(1)

    def _reply_once(self, attempt_id: str, local_file: Path) -> None:
        target = f"{self.directory}/{attempt_id}/response.upload.json"
        transfer = subprocess.run(["scp", "-F", str(self.config), "-o", "BatchMode=yes",
            str(local_file), f"{self.alias}:{target}"], capture_output=True, text=True, timeout=30)
        if transfer.returncode:
            raise RuntimeError("Response transfer failed; cached response retained, do not repeat API request")
        self.ssh_python("import os,sys; os.replace(sys.argv[1],sys.argv[2])", target,
                        f"{self.directory}/{attempt_id}/response.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-config", type=Path, required=True)
    parser.add_argument("--ssh-alias", required=True)
    parser.add_argument("--remote-bridge-dir", required=True)
    parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--credentials-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--max-api-cost-usd", type=float, required=True)
    parser.add_argument("--request-cost-ceiling-usd", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-input-bytes", type=int, default=128000)
    parser.add_argument("--image-detail", choices=("low", "high", "original", "auto"), default="low")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--idle-timeout-seconds", type=float, default=60)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget = APIBudget(args.authorization_id, args.max_calls, args.max_output_tokens,
                       args.max_api_cost_usd, args.request_cost_ceiling_usd, args.max_input_bytes)
    if args.idle_timeout_seconds <= 0 or args.max_wall_seconds <= 0:
        parser.error("Timeouts must be positive")
    if not args.ssh_config.is_file():
        parser.error("SSH config file is missing")
    load_local_credentials(args.credentials_file)
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        parser.error(f"Missing local {key_name}")
    transport = (OpenAITransport(args.model, image_detail=args.image_detail,
                                reasoning_effort=args.reasoning_effort)
                 if args.provider == "openai" else AnthropicTransport(args.model))
    processor = BridgeProcessor(transport, budget, args.output.resolve())
    spool = SSHSpool(args.ssh_config, args.ssh_alias, args.remote_bridge_dir)
    started = last_work = time.monotonic()
    served = set()
    print("BRIDGE_READY: credentials remain local; no API call until a validated remote request arrives", flush=True)
    try:
        while time.monotonic() - started < args.max_wall_seconds:
            pending = [item for item in spool.pending() if item not in served]
            if not pending:
                if time.monotonic() - last_work >= args.idle_timeout_seconds:
                    print("BRIDGE_STOPPED: idle timeout", flush=True)
                    break
                time.sleep(1)
                continue
            for attempt_id in pending:
                envelope = spool.read(attempt_id)
                try:
                    response = processor.process(envelope)
                except Exception as exc:
                    # Return a terminal rejection promptly; do not leave the
                    # remote simulator waiting after a local budget/scope error.
                    response = {"bridge_id": attempt_id, "ok": False,
                        "error_type": type(exc).__name__, "new_api_call": False,
                        "status": "rejected_or_prior_charge_unknown"}
                    atomic_json(processor.directory / f"{attempt_id}.rejection.json", response)
                    spool.reply(attempt_id, processor.directory / f"{attempt_id}.rejection.json")
                    raise
                spool.reply(attempt_id, processor.directory / f"{attempt_id}.response.json")
                served.add(attempt_id)
                last_work = time.monotonic()
                print(json.dumps({"bridge_id": attempt_id, "ok": response["ok"]}), flush=True)
                if not response["ok"]:
                    print("BRIDGE_STOPPED: provider error or unknown charge; no retry", flush=True)
                    return
                if len(served) >= args.max_calls:
                    print("BRIDGE_STOPPED: configured request count reached", flush=True)
                    return
    except Exception as exc:
        processor.logger.emit("bridge_server_stopped", error_type=type(exc).__name__)
        print(f"BRIDGE_STOPPED: {type(exc).__name__}; inspect cached responses before retrying", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
