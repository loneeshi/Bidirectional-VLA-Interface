"""Append-only, flushed JSONL events. Callers keep these outside public source.

STATUS: active — core
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class JsonlLogger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def emit(self, event: str, **fields: Any) -> None:
        record = {"schema_version": "0.1", "run_id": self.run_id,
                  "time_utc": datetime.now(timezone.utc).isoformat(),
                  "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_default)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
