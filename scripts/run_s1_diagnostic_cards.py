"""Preflight or execute the frozen S1-IA D1/D2 diagnostic cards serially on GPU1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from bvi.s1_diagnostic_cards import d1_adjudication, d2_adjudication


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def trajectory_summary(path: Path, raw_action_key: str = "action") -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    forces = []
    distances = []
    raw_out_of_bounds = 0
    raw_max_abs = 0.0
    force_limit_step = None
    for row in rows:
        info = row.get("info", {})
        force_value = row.get("robot_cumulative_force", info.get("robot_cumulative_force", [0]))
        if isinstance(force_value, list):
            force_value = force_value[0]
        forces.append(float(force_value))
        if "tcp_target_distance_m" in row:
            distances.append(float(row["tcp_target_distance_m"]))
        action = row.get(raw_action_key, row.get("action", []))
        flat = list(flatten(action))
        raw_out_of_bounds += sum(abs(value) > 1.0 for value in flat)
        raw_max_abs = max([raw_max_abs, *(abs(value) for value in flat)])
        within = info.get("cumulative_force_within_limit", [True])
        if isinstance(within, list):
            within = within[0]
        if not bool(within) and force_limit_step is None:
            force_limit_step = int(row["step"])
    return {
        "steps": len(rows), "peak_cumulative_force": max(forces, default=0.0),
        "force_limit_step": force_limit_step, "tcp_target_distance_m": distances,
        "raw_action_out_of_bounds_values": raw_out_of_bounds, "raw_action_max_abs": raw_max_abs,
    }


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--native-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    source = args.source_run.resolve()
    output = args.output.resolve()
    manifest = {
        "cards": ["D1", "D2"],
        "questions": [
            "D1: Is reach 0/5 caused by the force threshold/wrapper or policy behavior?",
            "D2: Is grasp 3/4 distinguishable from closing in place?",
        ],
        "source_run": str(source), "native_panel": str(args.native_panel.resolve()),
        "seeds": list(SEEDS), "d1_a0_action_cap": 40, "d1_sac_action_cap": 200,
        "d2_action_cap": 60, "wall_cap_seconds": 1800,
        "training_updates": 0, "api_calls": 0, "rental_usd": 0, "lab_charge_usd": None,
        "threshold_changes": 0, "new_seeds": 0, "execute": args.execute,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    required = []
    for seed in SEEDS:
        start = source / "starts" / f"seed{seed}"
        required += [start / "reach-state.pt", start / "events.jsonl",
                     source / f"reach-seed{seed}" / "events.jsonl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen source evidence missing: {missing}")
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return
    used = int(subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip())
    if used >= 1024:
        raise RuntimeError("GPU1 occupied; diagnostic launch refused")
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()

    def remaining() -> float:
        value = 1800 - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Diagnostic cards exceeded the frozen 30-minute outer cap")
        return value

    def run(name: str, command: list[str], seed: int, cap: float = 180) -> dict:
        log_path = output / f"{name}.log"
        with log_path.open("x") as log:
            result = subprocess.run(command, env=dict(os.environ, PYTHONHASHSEED=str(seed)),
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=min(cap, remaining()))
        result_path = output / name / "result.json"
        payload = read_json(result_path) if result_path.is_file() else {}
        if result.returncode or payload.get("status") == "infrastructure_failure":
            raise RuntimeError(f"{name} infrastructure failure; evidence retained")
        return payload

    # B0 is a read-only restore audit and fixes the evaluable denominator before controls run.
    audits = {}
    for seed in SEEDS:
        grasp = source / "starts" / f"seed{seed}" / "grasp-state.pt"
        if not grasp.is_file():
            continue
        name = f"d2-b0-seed{seed}"
        audits[seed] = run(name, [str(args.sim_python), str(scripts / "eval_s1_diagnostic_control.py"),
            "--seed", str(seed), "--mode", "audit", "--card", "D2", "--max-actions", "0",
            "--reference-state", str(grasp), "--output", str(output / name)], seed)
    evaluable = [seed for seed, row in audits.items()
                 if not row["initial"]["native_fail"] and not row["initial"]["native_success"]]
    if len(evaluable) != 4:
        raise RuntimeError(f"Frozen D2 denominator changed: expected 4 evaluable starts, got {evaluable}")

    # D1 A0, plus A1 exact action replay because the prior teacher log omitted distance/contact series.
    for seed in SEEDS:
        reach = source / "starts" / f"seed{seed}" / "reach-state.pt"
        for mode in ("zero", "hold"):
            name = f"d1-a0-{mode}-seed{seed}"
            run(name, [str(args.sim_python), str(scripts / "eval_s1_diagnostic_control.py"),
                "--seed", str(seed), "--mode", mode, "--card", "D1", "--max-actions", "40",
                "--reference-state", str(reach), "--output", str(output / name)], seed)
        name = f"d1-a1-sac-seed{seed}"
        run(name, [str(args.sim_python), str(scripts / "eval_s1_diagnostic_control.py"),
            "--seed", str(seed), "--mode", "replay", "--card", "D1", "--max-actions", "200",
            "--actions-jsonl", str(source / "starts" / f"seed{seed}" / "events.jsonl"),
            "--reference-state", str(reach), "--output", str(output / name)], seed, cap=240)

    # D2 B1 and B2. B2 states remain on each seed's same deterministic SAC trajectory.
    b2_collections = {}
    for seed in evaluable:
        grasp = source / "starts" / f"seed{seed}" / "grasp-state.pt"
        name = f"d2-b1-close-seed{seed}"
        run(name, [str(args.sim_python), str(scripts / "eval_s1_diagnostic_control.py"),
            "--seed", str(seed), "--mode", "close", "--card", "D2", "--max-actions", "60",
            "--reference-state", str(grasp), "--output", str(output / name)], seed)
        name = f"d2-b2-starts-seed{seed}"
        b2_collections[seed] = run(name, [str(args.sim_python), str(scripts / "collect_s1_d2_perturbations.py"),
            "--seed", str(seed), "--reach-state", str(source / "starts" / f"seed{seed}" / "reach-state.pt"),
            "--grasp-state", str(grasp), "--output", str(output / name)], seed, cap=240)
        if b2_collections[seed]["status"] != "completed":
            continue
        for offset in (3, 5):
            state = output / name / f"offset-{offset}cm-state.pt"
            child = f"d2-b2-offset{offset}cm-close-seed{seed}"
            run(child, [str(args.sim_python), str(scripts / "eval_s1_diagnostic_control.py"),
                "--seed", str(seed), "--mode", "close", "--card", "D2", "--max-actions", "60",
                "--reference-state", str(state), "--output", str(output / child)], seed)

    d1_controls = []
    force_steps = []
    sac_peaks = []
    for seed in SEEDS:
        for mode in ("zero", "hold"):
            row = read_json(output / f"d1-a0-{mode}-seed{seed}" / "result.json")
            d1_controls.append(row)
            if row.get("reason_code") == "cumulative_force_limit":
                force_steps.append(int(row["steps"]))
        sac_peaks.append(float(read_json(output / f"d1-a1-sac-seed{seed}" / "result.json")["peak_cumulative_force"]))
    a2 = {seed: trajectory_summary(source / f"reach-seed{seed}" / "events.jsonl", "raw_actions")
          for seed in SEEDS}
    b1 = {seed: read_json(output / f"d2-b1-close-seed{seed}" / "result.json") for seed in evaluable}
    b2 = {offset: {seed: read_json(output / f"d2-b2-offset{offset}cm-close-seed{seed}" / "result.json")
                   for seed in evaluable
                   if (output / f"d2-b2-offset{offset}cm-close-seed{seed}" / "result.json").is_file()}
          for offset in (3, 5)}
    b2_success_by_offset = {offset: sum(bool(row.get("success")) for row in rows.values())
                            for offset, rows in b2.items()}
    panel = []
    for seed in (*SEEDS, 2030, 2031, 2032, 2033, 2034):
        result = args.native_panel / f"seed{seed}" / "result.json"
        if result.is_file():
            panel.append(read_json(result))
    distances = [float(audits[seed]["initial"]["tcp_target_distance_m"]) for seed in evaluable]
    summary = {
        "status": "completed", "cards": ["D1", "D2"],
        "d1": {
            "adjudication": d1_adjudication(force_steps, sac_peaks),
            "a0_controls": d1_controls, "a1_sac_peak_forces": sac_peaks,
            "a2_reused_ia": a2, "force_limit": 5000.0,
        },
        "d2": {
            "initial_distances_m": distances, "initial_distance_median_m": sorted(distances)[len(distances) // 2 - 1:len(distances) // 2 + 1],
            "b0": audits, "b1": b1, "b1_successes": sum(bool(row.get("success")) for row in b1.values()),
            "b2": b2, "b2_successes_by_offset": b2_success_by_offset,
            "adjudication": d2_adjudication(sum(bool(row.get("success")) for row in b1.values()),
                                             len(evaluable), max(b2_success_by_offset.values(), default=0)),
        },
        "b3_existing_native_gate": {
            "episodes": len(panel), "successes": sum(bool(row.get("success")) for row in panel),
            "threshold": 3, "rerun": False,
        },
        "wall_seconds": time.monotonic() - started, "api_calls": 0, "training_updates": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
