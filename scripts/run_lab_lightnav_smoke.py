"""Bounded real-image HF inference; not a navigation success evaluation."""
import os
import subprocess
import json
import time
import hashlib
from pathlib import Path

ROOT = Path.home() / "bvi-research"
OUT = ROOT / "runs/lightnav-native-2026-09-16"
OUT.mkdir(parents=True, exist_ok=True)
used = int(subprocess.check_output([
    "nvidia-smi", "-i", "1", "--query-gpu=memory.used",
    "--format=csv,noheader,nounits"], text=True).strip())
if used > 1024:
    raise RuntimeError("GPU1 occupied; refusing to start")
os.environ.update(CUDA_VISIBLE_DEVICES="GPU-b7ebba23-7824-7601-df32-be55628936c3",
                  LIGHTNAV_HF_DTYPE="float32", HF_HUB_OFFLINE="1")
report = {"status": "loading", "api_calls": 0, "precision": "float32",
          "task_key": "vlnce", "task_type": "vlnce_traj",
          "scope": "single archived real frame, inference only"}
def save():
    (OUT / "result.json").write_text(json.dumps(report, indent=2))
save()
try:
    import torch
    import numpy as np
    from PIL import Image
    from lightnav.tracking import build_tracking_agent
    torch.manual_seed(2024)
    frame = OUT / "fetch_nav.png"
    report["frame_sha256"] = hashlib.sha256(frame.read_bytes()).hexdigest()
    agent = build_tracking_agent(str(ROOT / "checkpoints/lightnav"),
                                 backend="hf", task_key="vlnce", device="cuda")
    instruction = "Navigate to the kitchen counter."
    agent.reset(instruction=instruction)
    agent.observe(np.array(Image.open(frame).convert("RGB")))
    report.update(status="inferencing", instruction=instruction)
    save()
    start = time.monotonic()
    waypoints, raw, latency = agent.predict_waypoints(instruction, task_type="vlnce_traj")
    if not np.isfinite(waypoints).all():
        raise ValueError("nonfinite waypoints")
    report.update(status="inference_passed", waypoints=waypoints.tolist(), raw_text=raw,
                  latency_ms=latency, elapsed_seconds=time.monotonic()-start,
                  max_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as exc:
    report.update(status="failed", error=repr(exc))
    raise
finally:
    save()
