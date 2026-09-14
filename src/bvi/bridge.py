"""File-spool VLM transport: simulator remote, credentials and paid request local.

Both sides use the remote coordinator's attempt ID. Local records are mirrors,
not additional expenses. A durable local claim prevents a second model request
after an ambiguous failure or process restart. Completed responses can be resent.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .coordinator import APIBudget, VLMRequest, VLMResponse
from .logging import JsonlLogger
from .protocol import ImageFrame, ProtocolError


def validate_bridge_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ProtocolError("Bridge ID must be a 32-character lowercase hex attempt ID")
    return value


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def encode_request(request: VLMRequest, provider: str, model: str,
                   authorization_id: str) -> dict[str, Any]:
    attempt_id = validate_bridge_id(request.attempt_id)
    return {"version": 1, "bridge_id": attempt_id, "provider": provider, "model": model,
        "authorization_id": authorization_id, "request": {
            "system": request.system, "prompt": request.prompt, "schema": dict(request.schema),
            "max_output_tokens": request.max_output_tokens, "images": [
                {"camera": frame.camera, "media_type": frame.media_type,
                 "data_base64": base64.b64encode(frame.data).decode("ascii")}
                for frame in request.images]}}


def decode_request(envelope: dict[str, Any]) -> VLMRequest:
    if envelope.get("version") != 1:
        raise ProtocolError("Unsupported bridge version")
    attempt_id = validate_bridge_id(envelope.get("bridge_id"))
    data = envelope["request"]
    if not isinstance(data.get("system"), str) or not isinstance(data.get("prompt"), str):
        raise ProtocolError("Bridge text must be strings")
    if not isinstance(data.get("schema"), dict):
        raise ProtocolError("Bridge request requires a JSON schema")
    if type(data.get("max_output_tokens")) is not int or data["max_output_tokens"] < 1:
        raise ProtocolError("Bridge output token cap must be a positive integer")
    if not 1 <= len(data.get("images", [])) <= 2:
        raise ProtocolError("Bridge allows one or two actual camera images")
    images = tuple(ImageFrame(frame["camera"], base64.b64decode(frame["data_base64"], validate=True),
                              frame["media_type"]) for frame in data["images"])
    for frame in images:
        frame.validate()
    return VLMRequest(data["system"], data["prompt"], images, data["schema"],
                      data["max_output_tokens"], attempt_id)


class FileBridgeTransport:
    """Runs on the GPU host and never reads credentials or imports a provider SDK."""
    def __init__(self, directory: str | Path, provider: str, model: str, authorization_id: str,
                 timeout_seconds: float = 120.0, image_detail: str = "low",
                 reasoning_effort: str = "none"):
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ProtocolError("Bridge timeout must be in (0, 300] seconds")
        self.directory, self.provider, self.model = Path(directory), provider, model
        self.authorization_id, self.timeout_seconds = authorization_id, timeout_seconds
        self.request_options = {"image_detail": image_detail, "reasoning_effort": reasoning_effort,
                                "transport": "ssh_file_bridge"}

    def generate(self, request: VLMRequest) -> VLMResponse:
        envelope = encode_request(request, self.provider, self.model, self.authorization_id)
        envelope["request_options"] = {k: v for k, v in self.request_options.items() if k != "transport"}
        directory = self.directory / envelope["bridge_id"]
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "request.json", envelope)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            response_path = directory / "response.json"
            if response_path.is_file():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("bridge_id") != envelope["bridge_id"]:
                    raise ProtocolError("Bridge response ID mismatch")
                if not response.get("ok"):
                    raise ProtocolError(f"Local provider failed: {response.get('error_type', 'unknown')}")
                return VLMResponse(**response["response"])
            time.sleep(0.25)
        atomic_json(directory / "expired.json", {"bridge_id": envelope["bridge_id"],
                    "status": "remote_wait_expired_charge_unknown"})
        raise TimeoutError("Bridge response timed out; inspect both logs before any retry")


class BridgeProcessor:
    """Local paid-request executor with durable claims and explicit bounds."""
    def __init__(self, transport: Any, budget: APIBudget, directory: str | Path):
        self.transport, self.budget, self.directory = transport, budget, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(self.directory / "bridge-events.jsonl", self.directory.name)

    def process(self, envelope: dict[str, Any]) -> dict[str, Any]:
        request = decode_request(envelope)
        attempt_id = request.attempt_id
        if (envelope.get("provider") != self.transport.provider
                or envelope.get("model") != self.transport.model
                or envelope.get("authorization_id") != self.budget.authorization_id):
            raise ProtocolError("Remote request is outside local provider/model/authorization scope")
        options = envelope.get("request_options", {})
        for key, value in getattr(self.transport, "request_options", {}).items():
            if options.get(key) != value:
                raise ProtocolError("Remote and local provider options disagree")
        input_bytes = len((request.system + request.prompt + json.dumps(request.schema)).encode("utf-8")) + sum(
            4 * ((len(frame.data) + 2) // 3) for frame in request.images)
        if request.max_output_tokens > self.budget.max_output_tokens or input_bytes > self.budget.max_input_bytes:
            raise ProtocolError("Remote request exceeds local input/output budget")
        digest = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode("utf-8")).hexdigest()
        claim = self.directory / f"{attempt_id}.claim.json"
        cached = self.directory / f"{attempt_id}.response.json"
        if claim.is_file():
            record = json.loads(claim.read_text(encoding="utf-8"))
            if record["request_sha256"] != digest:
                raise ProtocolError("An existing bridge ID was reused for a different request")
            if cached.is_file():
                return json.loads(cached.read_text(encoding="utf-8"))
            raise ProtocolError("Previously started bridge attempt has no response; charge unknown, no retry")
        claims = [json.loads(path.read_text(encoding="utf-8"))
                  for path in self.directory.glob("*.claim.json")]
        relevant = [item for item in claims if item["authorization_id"] == self.budget.authorization_id]
        reserved = sum(item["reserved_cost_usd"] for item in relevant)
        if (len(relevant) >= self.budget.max_calls
                or reserved + self.budget.request_cost_ceiling_usd > self.budget.max_cost_usd + 1e-12):
            raise ProtocolError("Local bridge experiment budget exhausted")
        record = {"bridge_id": attempt_id, "authorization_id": self.budget.authorization_id,
                  "request_sha256": digest, "reserved_cost_usd": self.budget.request_cost_ceiling_usd}
        # Exclusive creation prevents accidental duplicate processing of this ID.
        with claim.open("x", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self.logger.emit("bridge_attempt_started", **record, provider=self.transport.provider,
                         model=self.transport.model, accounting_role="mirror_of_remote_attempt")
        try:
            response = self.transport.generate(request)
            result = {"bridge_id": attempt_id, "ok": True, "response": asdict(response)}
            atomic_json(cached, result)
            self.logger.emit("bridge_api_usage", bridge_id=attempt_id,
                request_id=response.request_id, usage=response.usage, amount=None,
                status="pending_reconciliation", accounting_role="mirror_of_remote_attempt")
        except Exception as exc:
            result = {"bridge_id": attempt_id, "ok": False, "error_type": type(exc).__name__,
                      "status": "charge_unknown_pending_reconciliation"}
            atomic_json(cached, result)
            self.logger.emit("bridge_api_failed", **result, amount=None,
                             accounting_role="mirror_of_remote_attempt")
        return result
