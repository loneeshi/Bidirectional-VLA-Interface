"""Optional model-service clients. Predictions are NOT Fetch actions.

LightNav uses JSON; openpi uses its numpy-aware msgpack codec. Dependencies
and model servers stay separate from the pinned MS-HAB environment.
No retries: an uncertain request must not silently run inference twice.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from typing import Any

from .protocol import ImageFrame, ProtocolError


def finite_rows(value: Any, width: int) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(x) for x in row) for row in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Expected a numeric action matrix") from exc
    if not rows or len(rows) > 1024 or any(
        len(row) != width or not all(math.isfinite(x) for x in row) for row in rows
    ):
        raise ProtocolError("Invalid action shape or nonfinite action")
    return rows


@dataclass(frozen=True)
class NavigationPrediction:
    # Cumulative SE(2), relative to the pose at image capture: forward/left/CCW.
    waypoints: tuple[tuple[float, ...], ...]
    stop: bool
    visible: bool | None


class LightNavClient:
    def __init__(self, connection: Any, timeout: float = 30):
        self.connection, self.timeout = connection, timeout
        self.seq = 0

    @classmethod
    def connect(cls, url: str, timeout: float = 30):
        from websockets.sync.client import connect
        return cls(connect(url, open_timeout=timeout, close_timeout=5,
                           max_size=4 * 1024 * 1024), timeout)

    def _exchange(self, action: str, data: dict) -> dict:
        self.connection.send(json.dumps({"action": action, "data": data}))
        result = json.loads(self.connection.recv(timeout=self.timeout))
        if not isinstance(result, dict) or result.get("action") != action:
            raise ProtocolError("Unexpected LightNav response")
        payload = result.get("data")
        if not isinstance(payload, dict) or payload.get("rc") != 0:
            raise ProtocolError("LightNav server rejected request")
        return payload

    def reset(self) -> None:
        self._exchange("reset", {})
        self.seq = 0

    def infer(self, image: ImageFrame, instruction: str) -> NavigationPrediction:
        image.validate()
        if not instruction.strip():
            raise ProtocolError("Prediction requires a nonempty navigation instruction")
        seq = self.seq
        self.seq += 1
        data = self._exchange("next", {
            "seq": seq, "image": base64.b64encode(image.data).decode("ascii"),
            "instruction": instruction,
        })
        if data.get("seq") != seq or type(data.get("stop")) is not bool:
            raise ProtocolError("Missing stop flag or mismatched sequence")
        visible = data.get("visible")
        if visible is not None and type(visible) is not bool:
            raise ProtocolError("Invalid visibility flag")
        block = data.get("actions")
        raw = block.get("actions") if isinstance(block, dict) else None
        if data["stop"] and (raw is None or raw == []):
            rows = ()
        else:
            rows = finite_rows(raw, 3)
        return NavigationPrediction(rows, data["stop"], visible)

    def close(self) -> None:
        self.connection.close()


class OpenPiClient:
    """Wire-compatible openpi client with bounded connection/receive waits."""
    def __init__(self, connection: Any, codec: Any, timeout: float = 30):
        self.connection, self.codec, self.timeout = connection, codec, timeout
        self.packer = codec.Packer()
        self.metadata = self._receive()

    @classmethod
    def connect(cls, url: str, timeout: float = 30):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        connection = connect(url, compression=None, open_timeout=timeout,
                             close_timeout=5, max_size=16 * 1024 * 1024,
                             ping_interval=None)
        # Upstream performs synchronous JAX inference on its websocket event
        # loop. First compilation can delay pong handling beyond the default
        # keepalive deadline. recv(timeout=...) remains the bounded I/O guard.
        try:
            return cls(connection, msgpack_numpy, timeout)
        except Exception:
            connection.close()
            raise

    def _receive(self) -> dict:
        raw = self.connection.recv(timeout=self.timeout)
        if not isinstance(raw, bytes):
            raise ProtocolError("Openpi returned a server error instead of binary data")
        result = self.codec.unpackb(raw)
        if not isinstance(result, dict):
            raise ProtocolError("Openpi response must be a dictionary")
        return result

    def infer(self, observation: dict, action_dim: int) -> tuple[tuple[float, ...], ...]:
        self.connection.send(self.packer.pack(observation))
        return finite_rows(self._receive().get("actions"), action_dim)

    def close(self) -> None:
        self.connection.close()


def droid_probe_observation(head: ImageFrame, hand: ImageFrame,
                            joint_position: Any, gripper_position: float,
                            prompt: str) -> dict:
    """DROID input schema for inference probes ONLY, not a Fetch transfer recipe.

    Fetch images in DROID camera fields are out of distribution. Seven joint
    numbers do not establish compatible kinematics, units or normalization.
    """
    import io
    import numpy as np
    from PIL import Image
    from openpi_client import image_tools

    joints = finite_rows([joint_position], 7)[0]
    if not math.isfinite(gripper_position) or not prompt.strip():
        raise ProtocolError("Invalid gripper state or prompt")

    def pixels(frame):
        frame.validate()
        rgb = np.asarray(Image.open(io.BytesIO(frame.data)).convert("RGB"))
        return image_tools.resize_with_pad(rgb, 224, 224)

    return {
        "observation/exterior_image_1_left": pixels(head),
        "observation/wrist_image_left": pixels(hand),
        "observation/joint_position": np.asarray(joints, dtype=np.float32),
        "observation/gripper_position": np.asarray([gripper_position], dtype=np.float32),
        "prompt": prompt,
    }
