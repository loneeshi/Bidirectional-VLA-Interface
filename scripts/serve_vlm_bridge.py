"""Serve remote simulator VLM requests with a key that stays on this computer.

Only the configured SSH alias and selected model provider are contacted. The
server runs serially, caches responses, and stops on an uncertain/error request.
Reusing the output directory preserves claims and the spending scope on restart.
"""
from __future__ import annotations

import argparse
import errno
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


SERVER_CREDENTIAL_KEYS = {'LAB_SSH_HOST', 'LAB_SSH_PORT', 'LAB_SSH_USER',
                          'LAB_SSH_PASSWORD', 'LAB_SSH_KEY_PATH'}


def load_server_credentials(path: Path) -> dict[str, str]:
    """Load only the declared lab SSH fields without executing or logging them."""
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if separator and key in SERVER_CREDENTIAL_KEYS:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            values[key] = value
    required = {'LAB_SSH_HOST', 'LAB_SSH_PORT', 'LAB_SSH_USER'}
    if not required <= values.keys() or not (values.get('LAB_SSH_PASSWORD') or values.get('LAB_SSH_KEY_PATH')):
        raise ValueError('Lab SSH credentials are incomplete')
    return values


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


class ParamikoSpool:
    """Password/key SSH spool for a lab host with a pinned known-host entry."""
    def __init__(self, credentials: Path, known_hosts: Path, directory: str):
        import paramiko
        path = PurePosixPath(directory)
        if (not path.is_absolute() or '..' in path.parts or str(path) == '/'
                or not re.fullmatch(r'/[A-Za-z0-9_./-]+', directory)):
            raise ValueError('Remote bridge directory must be an absolute safe path')
        if not known_hosts.is_file():
            raise ValueError('Pinned lab known-hosts file is missing')
        self.paramiko = paramiko
        self.credentials = load_server_credentials(credentials)
        self.known_hosts, self.directory = known_hosts.resolve(), str(path)
        self.client = self.sftp = None

    def _close(self) -> None:
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()
        self.client = self.sftp = None

    def close(self) -> None:
        self._close()

    def _connect(self) -> None:
        if self.client is not None:
            return
        client = self.paramiko.SSHClient()
        client.load_host_keys(str(self.known_hosts))
        cfg = self.credentials
        options = dict(hostname=cfg['LAB_SSH_HOST'], port=int(cfg['LAB_SSH_PORT']),
            username=cfg['LAB_SSH_USER'], password=cfg.get('LAB_SSH_PASSWORD') or None,
            look_for_keys=False, allow_agent=False, timeout=15,
            auth_timeout=15, banner_timeout=15)
        if cfg.get('LAB_SSH_KEY_PATH'):
            options['key_filename'] = cfg['LAB_SSH_KEY_PATH']
        client.connect(**options)
        self.client, self.sftp = client, client.open_sftp()

    def _run(self, operation):
        for attempt in range(2):
            try:
                self._connect()
                return operation(self.sftp)
            except (EOFError, OSError, self.paramiko.SSHException):
                self._close()
                if attempt:
                    raise

    @staticmethod
    def _exists(sftp, path: str) -> bool:
        try:
            sftp.stat(path)
            return True
        except OSError as exc:
            if getattr(exc, 'errno', None) == errno.ENOENT:
                return False
            raise

    def pending(self) -> list[str]:
        def operation(sftp):
            try:
                names = sftp.listdir(self.directory)
            except OSError as exc:
                if getattr(exc, 'errno', None) == errno.ENOENT:
                    return []
                raise
            return [validate_bridge_id(name) for name in sorted(names)
                    if re.fullmatch(r'[0-9a-f]{32}', name)
                    and self._exists(sftp, f'{self.directory}/{name}/request.json')
                    and not self._exists(sftp, f'{self.directory}/{name}/response.json')
                    and not self._exists(sftp, f'{self.directory}/{name}/expired.json')]
        return self._run(operation)

    def read(self, attempt_id: str) -> dict:
        validate_bridge_id(attempt_id)
        def operation(sftp):
            base = f'{self.directory}/{attempt_id}'
            if self._exists(sftp, f'{base}/expired.json'):
                raise ProtocolError('Remote bridge request already expired')
            path = f'{base}/request.json'
            if sftp.stat(path).st_size > 2_000_000:
                raise ProtocolError('Remote bridge request is oversized')
            with sftp.open(path, 'r') as stream:
                raw = stream.read()
                envelope = json.loads(raw.decode('utf-8') if isinstance(raw, bytes) else raw)
            if envelope.get('bridge_id') != attempt_id:
                raise ProtocolError('Spool filename and envelope ID disagree')
            return envelope
        return self._run(operation)

    def reply(self, attempt_id: str, local_file: Path) -> None:
        validate_bridge_id(attempt_id)
        def operation(sftp):
            base = f'{self.directory}/{attempt_id}'
            temporary, target = f'{base}/response.upload.json', f'{base}/response.json'
            sftp.put(str(local_file), temporary)
            try:
                sftp.posix_rename(temporary, target)
            except (AttributeError, OSError):
                sftp.rename(temporary, target)
        self._run(operation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument('--ssh-config', type=Path)
    connection.add_argument('--server-credentials-file', type=Path)
    parser.add_argument('--ssh-alias')
    parser.add_argument('--known-hosts', type=Path)
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
    # This is intentionally required. A silent zero-retry default previously
    # turned one transient TLS read failure into a terminal panel failure.
    parser.add_argument('--max-provider-retries', type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budget = APIBudget(args.authorization_id, args.max_calls, args.max_output_tokens,
                       args.max_api_cost_usd, args.request_cost_ceiling_usd, args.max_input_bytes)
    if args.idle_timeout_seconds <= 0 or args.max_wall_seconds <= 0:
        parser.error("Timeouts must be positive")
    if args.ssh_config and (not args.ssh_config.is_file() or not args.ssh_alias):
        parser.error('SSH config mode requires an existing config and --ssh-alias')
    if args.server_credentials_file and (not args.server_credentials_file.is_file()
            or args.known_hosts is None or not args.known_hosts.is_file()):
        parser.error('Lab mode requires a credentials file and pinned --known-hosts')
    load_local_credentials(args.credentials_file)
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        parser.error(f"Missing local {key_name}")
    transport = (OpenAITransport(args.model, image_detail=args.image_detail,
                                reasoning_effort=args.reasoning_effort)
                 if args.provider == "openai" else AnthropicTransport(args.model))
    processor = BridgeProcessor(transport, budget, args.output.resolve(),
                                max_provider_retries=args.max_provider_retries)
    spool = (SSHSpool(args.ssh_config, args.ssh_alias, args.remote_bridge_dir)
             if args.ssh_config else ParamikoSpool(args.server_credentials_file,
                                                   args.known_hosts, args.remote_bridge_dir))
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
    finally:
        if hasattr(spool, 'close'):
            spool.close()


if __name__ == "__main__":
    main()
