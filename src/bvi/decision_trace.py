"""Local, exact policy decision artifacts; loading .pt files requires trusted data.

STATUS: frozen — historical training and diagnostics (retained)

Inputs contain CPU copies of model tensors. A replayer must move those values to
the original model device and restore RNG immediately before model execution.
The recorder captures only explicitly supplied values, never environment vars.
"""

import copy
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch


def recursive_clone(value):
    """Detach tensors, copy them to CPU, and isolate nested mutable values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: recursive_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(recursive_clone(item) for item in value)
    if isinstance(value, list):
        return [recursive_clone(item) for item in value]
    return copy.deepcopy(value)


def capture_rng_state():
    """Read existing generators without initializing CUDA or drawing samples."""
    return recursive_clone({
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    })


def restore_rng_state(state):
    """Restore a trusted snapshot, including CUDA generators when recorded."""
    cuda = state["torch_cuda"]
    if cuda is not None:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError("Replay requires the recorded CUDA device count")
        torch.cuda.set_rng_state_all(cuda)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path, writer):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        writer(handle)
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


class TorchDecisionRecorder:
    """Append sequential decisions using begin -> record_batch -> finish.

    An exclusive reservation prevents concurrent recorders from overwriting the
    same id. Interrupted reservations/artifacts remain as evidence and are never
    reused. The manifest is committed only after all three payloads are saved.
    Metadata must be JSON serializable; payloads may contain tensors and arrays.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        used = [int(match.group(1)) for path in self.output_dir.iterdir()
                if (match := re.match(r"^(\d+)-", path.name))]
        self._next_id = max(used, default=-1) + 1
        self._current = None

    def begin(self, metadata, raw):
        if self._current is not None:
            raise RuntimeError("Finish the active decision before begin")
        # Validate metadata before reserving an id; serialize to isolate copies.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        decision_id = f"{self._next_id:06d}"
        if any(self.output_dir.glob(f"{decision_id}-*")):
            raise FileExistsError(f"Decision {decision_id} already exists")
        reservation = self.output_dir / f"{decision_id}-reservation"
        with reservation.open("xb"):
            pass
        self._next_id += 1
        self._current = {"decision_id": decision_id, "metadata": metadata,
                         "stage": "reserved", "artifacts": {}}
        self._save("raw", {"raw": recursive_clone(raw),
                           "rng_before_preprocessing": capture_rng_state()})
        self._current["stage"] = "raw"
        return decision_id

    def record_batch(self, batch):
        self._require("raw")
        self._save("inputs", {"batch": recursive_clone(batch),
                              "rng_before_model": capture_rng_state()})
        self._current["stage"] = "inputs"

    def finish(self, actions, progress):
        self._require("inputs")
        self._save("outputs", {"actions": recursive_clone(actions),
                               "progress": recursive_clone(progress),
                               "rng_after_model": capture_rng_state()})
        # A failed manifest write must not permit retrying/overwriting outputs.
        self._current["stage"] = "outputs"
        manifest = {"format_version": 1,
                    "decision_id": self._current["decision_id"],
                    "metadata": self._current["metadata"],
                    "tensor_storage": "cpu; restore tensors to model device for replay",
                    "artifacts": self._current["artifacts"]}
        path = self.output_dir / f"{manifest['decision_id']}-manifest.json"
        encoded = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
        _atomic_write(path, lambda handle: handle.write(encoded))
        self._current = None
        return manifest

    def _require(self, stage):
        if self._current is None or self._current["stage"] != stage:
            raise RuntimeError(f"Decision must be at stage {stage}")

    def _save(self, kind, payload):
        current = self._current
        path = self.output_dir / f"{current['decision_id']}-{kind}.pt"
        payload = {"format_version": 1, "decision_id": current["decision_id"],
                   "metadata": current["metadata"], **payload}
        _atomic_write(path, lambda handle: torch.save(payload, handle))
        current["artifacts"][kind] = {"path": path.name,
                                       "sha256": _sha256(path),
                                       "size_bytes": path.stat().st_size}
