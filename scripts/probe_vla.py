"""Run ONE model inference from saved observations; never steps a simulator.

Needs an already running, budget-authorized model service. A zero-state DROID
probe is explicitly synthetic, not evidence of robot compatibility.
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from bvi.protocol import ImageFrame
from bvi.vla_clients import LightNavClient, OpenPiClient, droid_probe_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["lightnav", "pi05-droid"], required=True)
    parser.add_argument("--url", required=True, help="SSH-forwarded ws://127.0.0.1:PORT")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--hand", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve previous evidence")
    if args.backend == "pi05-droid" and args.hand is None:
        parser.error("pi05-droid requires --hand")
    frames = {"head": ImageFrame("fetch_head", args.head.read_bytes())}
    if args.hand:
        frames["hand"] = ImageFrame("fetch_hand", args.hand.read_bytes())
    record = dict(backend=args.backend, task=args.instruction,
                  status="started", benchmark_result=False, simulator_steps=0,
                  image_sha256={k: hashlib.sha256(v.data).hexdigest() for k, v in frames.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    client = None
    start = time.monotonic()
    try:
        if args.backend == "lightnav":
            client = LightNavClient.connect(args.url)
            client.reset()
            result = asdict(client.infer(frames["head"], args.instruction))
        else:
            observation = droid_probe_observation(frames["head"], frames["hand"],
                                                  [0.] * 7, 0., args.instruction)
            record["state_source"] = "synthetic_zero_state_not_fetch_proprioception"
            client = OpenPiClient.connect(args.url)
            result = {"actions": client.infer(observation, action_dim=8)}
        record.update(status="prediction_received", prediction=result)
    except Exception as exc:
        record.update(status="failed_or_uncertain", error_type=type(exc).__name__)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
