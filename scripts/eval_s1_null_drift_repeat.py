"""One fresh-env exact-start SAC replay for Level-3 null-drift calibration.

This child performs no model inference, training, API request, or comparison to
an IA trajectory.  The parent runner starts it in a fresh process per repeat.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
CALIBRATION_ACTIONS = 35


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--actions-jsonl", type=Path, required=True)
    parser.add_argument("--expected-actions-sha256", required=True)
    parser.add_argument("--expected-task-plan-sha256", required=True)
    parser.add_argument("--expected-spawn-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=CALIBRATION_ACTIONS)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("PYTHONHASHSEED must match --seed before process start")
    if args.max_actions != CALIBRATION_ACTIONS:
        parser.error(f"Rule v1 requires exactly {CALIBRATION_ACTIONS} actions")
    if not 30 <= args.max_seconds <= 300:
        parser.error("--max-seconds must be within [30, 300]")
    if args.repeat_index < 0:
        parser.error("--repeat-index must be nonnegative")
    for value in (
        args.expected_reference_sha256,
        args.expected_actions_sha256,
        args.expected_task_plan_sha256,
        args.expected_spawn_sha256,
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error("Expected hashes must be lowercase SHA256 values")
    if not args.reference_state.is_file() or not args.actions_jsonl.is_file():
        parser.error("Reference state and recorded action files must exist")
    if sha256(args.reference_state) != args.expected_reference_sha256:
        raise ValueError("Reference-state SHA256 differs from preregistration")
    if sha256(args.actions_jsonl) != args.expected_actions_sha256:
        raise ValueError("Recorded-action SHA256 differs from preregistration")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "stage": "S1-Level3-null-drift",
        "status": "preflight",
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "fresh_process": True,
        "fresh_environment": True,
        "sim_backend": "gpu",
        "shader": "minimal",
        "gpu_uuid": GPU,
        "task": "set_table/pick/013_apple",
        "reference_state_sha256": args.expected_reference_sha256,
        "actions_jsonl_sha256": args.expected_actions_sha256,
        "max_actions": args.max_actions,
        "max_seconds": args.max_seconds,
        "training_updates": 0,
        "api_calls": 0,
        "policy_inference_calls": 0,
        "consumes_ia_events": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256(Path(__file__)),
    }
    env = None

    def save() -> None:
        report["wall_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        used = int(
            subprocess.check_output(
                ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
        )
        if used >= 1024:
            raise RuntimeError("GPU1 is occupied; null repeat refused")
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="2")
        research_root = Path.home() / "bvi-research"
        os.environ["MS_ASSET_DIR"] = str(research_root / "assets")
        repository = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(research_root / "src/AC-DiT"), str(repository / "src")]
        import gymnasium as gym
        import mshab.envs
        import numpy as np
        import torch
        from mshab.envs.planner import plan_data_from_file
        from mani_skill.utils.registration import REGISTERED_ENVS
        from bvi.fetch_pi_skill import JOINT_NAMES
        from bvi.ia_call_predicates import distance
        from bvi.native_pick_audit import state_max_errors
        from bvi.null_drift_calibration import validate_recorded_actions
        from bvi.s1_diagnostic_cards import jsonable

        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rows = [json.loads(line) for line in args.actions_jsonl.read_text().splitlines() if line.strip()]
        source_actions = validate_recorded_actions(rows, args.max_actions)
        mutations = []
        for action in source_actions:
            raw = np.asarray(action, dtype=np.float32)
            applied = np.clip(raw, -1, 1)
            applied[8:10] = 0
            mutations.append(float(np.max(np.abs(raw - applied))))
        if max(mutations, default=math.inf) != 0:
            raise RuntimeError("Frozen SAC prefix would be changed by clip/head masking")
        os.chdir(research_root / "src/AC-DiT")
        rearrange = research_root / "assets/data/scene_datasets/replica_cad_dataset/rearrange"
        plan_path = rearrange / "task_plans/set_table/pick/val/013_apple.json"
        spawn_path = rearrange / "spawn_data/set_table/pick/val/spawn_data.pt"
        if sha256(plan_path) != args.expected_task_plan_sha256:
            raise ValueError("Task-plan SHA256 differs from preregistration")
        if sha256(spawn_path) != args.expected_spawn_sha256:
            raise ValueError("Spawn-data SHA256 differs from preregistration")
        plans = plan_data_from_file(plan_path)
        env = gym.make(
            "PickSubtaskTrain-v0",
            num_envs=1,
            robot_uids="fetch",
            obs_mode="rgbdp",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            reward_mode="dense",
            sensor_configs={"shader_pack": "minimal"},
            human_render_camera_configs={"shader_pack": "minimal"},
            viewer_camera_configs={"shader_pack": "minimal"},
            sim_backend="gpu",
            max_episode_steps=200,
            task_plans=plans.plans,
            scene_builder_cls=plans.dataset,
            spawn_data_fp=spawn_path,
            require_build_configs_repeated_equally_across_envs=False,
        )
        reference = torch.load(args.reference_state, map_location="cpu", weights_only=False)
        required = {"simulator", "controller", "reset_info"}
        if not isinstance(reference, dict) or not required <= set(reference):
            raise ValueError("Reference snapshot lacks simulator/controller/reset_info")
        saved = reference["simulator"]
        options = {
            "reconfigure": True,
            "build_config_idxs": jsonable(saved["build_config_idxs"]),
            "task_plan_idxs": jsonable(saved["task_plan_idxs"]),
        }
        obs, info = env.reset(seed=args.seed, options=options)
        unwrapped = env.unwrapped

        def device(value):
            if isinstance(value, torch.Tensor):
                return value.to(unwrapped.device)
            if isinstance(value, dict):
                return {key: device(item) for key, item in value.items()}
            return value

        unwrapped.set_state_dict(device(saved))
        unwrapped.agent.controller.set_state(device(reference["controller"]))
        info = device(reference["reset_info"])
        obs = unwrapped.get_obs(info=info)
        restored = {
            "simulator": unwrapped.get_state_dict(),
            "controller": unwrapped.agent.controller.get_state(),
        }
        expected = {key: reference[key] for key in restored}
        restore_errors = state_max_errors(jsonable(expected), jsonable(restored))
        restore_max = max(restore_errors.values(), default=math.inf)
        if restore_max > 1e-5:
            raise RuntimeError("Exact-start restoration exceeds 1e-5")
        joints = [joint.name for joint in unwrapped.agent.robot.active_joints]
        if joints != JOINT_NAMES or len(joints) != 15 or unwrapped.control_freq != 20:
            raise RuntimeError("Fetch qpos/control-frequency contract changed")
        if unwrapped.pick_cfg.robot_cumulative_force_limit != 5000:
            raise RuntimeError("Native cumulative-force threshold changed")
        renderer_pci = str(unwrapped._render_device.pci_string)
        expected_pci = subprocess.check_output(
            ["nvidia-smi", "-i", GPU, "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True
        ).strip()
        if renderer_pci.lower()[-7:] != expected_pci.lower()[-7:]:
            raise RuntimeError("Renderer is not on the preregistered physical GPU")
        initial = {
            "qpos": unwrapped.agent.robot.qpos[0].detach().cpu().tolist(),
            "tcp_target_distance_m": distance(jsonable(obs["extra"])),
            "robot_cumulative_force": float(info["robot_cumulative_force"].item()),
        }
        report.update(
            status="replay_running",
            restore_state_max_errors=restore_errors,
            restore_max_abs_error=restore_max,
            joint_names=joints,
            renderer_pci=renderer_pci,
            task_plan_sha256=args.expected_task_plan_sha256,
            spawn_data_sha256=args.expected_spawn_sha256,
            action_wrapper_mutation_max=max(mutations),
            initial=initial,
            runtime_identity={
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "task_cfg": str(unwrapped.task_cfgs["pick"]),
                "controller": str(
                    getattr(
                        unwrapped.agent.controller,
                        "config",
                        getattr(unwrapped.agent.controller, "configs", None),
                    )
                ),
                "control_frequency": unwrapped.control_freq,
                "class_source_sha256": {
                    str(Path(inspect.getfile(cls)).resolve()): sha256(Path(inspect.getfile(cls)))
                    for cls in REGISTERED_ENVS["PickSubtaskTrain-v0"].cls.__mro__
                    if cls.__module__.startswith(("mshab.", "mani_skill."))
                },
            },
        )
        save()
        with (output / "events.jsonl").open("x") as log:
            for index, source_action in enumerate(source_actions):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Null repeat exceeded its wall-time bound")
                action = np.asarray(source_action, dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(
                    torch.as_tensor(action[None], device=unwrapped.device)
                )
                row = {
                    "step": index + 1,
                    "action": action.tolist(),
                    "qpos": unwrapped.agent.robot.qpos[0].detach().cpu().tolist(),
                    "tcp_target_distance_m": distance(jsonable(obs["extra"])),
                    "robot_cumulative_force": float(info["robot_cumulative_force"].item()),
                    "terminated": bool(terminated.item()),
                    "truncated": bool(truncated.item()),
                }
                if len(row["qpos"]) != 15 or not all(math.isfinite(value) for value in row["qpos"]):
                    raise RuntimeError("Nonfinite or wrong-width qpos evidence")
                if not math.isfinite(row["tcp_target_distance_m"]) or not math.isfinite(
                    row["robot_cumulative_force"]
                ):
                    raise RuntimeError("Nonfinite null-drift scalar evidence")
                log.write(json.dumps(row) + "\n")
                log.flush()
                report["steps"] = index + 1
                save()
                if row["terminated"] or row["truncated"]:
                    raise RuntimeError("Null control terminated before the frozen 35-step horizon completed")
        if report.get("steps") != CALIBRATION_ACTIONS:
            raise RuntimeError("Null repeat did not complete all 35 actions")
        report["status"] = "completed_null_repeat"
    except Exception as error:
        report.update(status="infrastructure_failure", error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        save()


if __name__ == "__main__":
    main()
