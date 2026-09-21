"""Run one fixed-action D1/D2 control from an exact saved native state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from bvi.s1_diagnostic_cards import jsonable, sha256


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--mode", choices=("zero", "hold", "close", "audit", "replay"), required=True)
    parser.add_argument("--card", choices=("D1", "D2"), required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, required=True)
    parser.add_argument("--actions-jsonl", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("PYTHONHASHSEED must match seed before process start")
    if args.mode in ("zero", "hold") and not 0 <= args.max_actions <= 40:
        parser.error("D1 A0 controls are capped at 40 actions")
    if args.mode in ("close", "audit") and not 0 <= args.max_actions <= 60:
        parser.error("D2 controls are capped at the frozen 60-action grasp horizon")
    if args.mode == "replay" and (args.actions_jsonl is None or not 0 <= args.max_actions <= 200):
        parser.error("SAC replay requires --actions-jsonl and is capped at 200 actions")

    root = Path.home() / "bvi-research"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "card": args.card,
        "question": "D1 force-limit attribution" if args.card == "D1" else "D2 grasp qualification",
        "seed": args.seed,
        "mode": args.mode,
        "status": "preflight",
        "task": "set_table/pick/013_apple",
        "max_actions": args.max_actions,
        "api_calls": 0,
        "training_updates": 0,
        "threshold_changes": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256(__file__),
        "reference_state_sha256": sha256(args.reference_state),
    }
    env = None

    def save() -> None:
        report["wall_seconds"] = time.monotonic() - started
        (out / "result.json").write_text(json.dumps(jsonable(report), indent=2))

    save()
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU, MS_ASSET_DIR=str(root / "assets"), OMP_NUM_THREADS="2")
        sys.path[:0] = [str(root / "src/AC-DiT"), str(root / "src/Bidirectional-VLA-Interface/src")]
        import gymnasium as gym
        import mshab.envs
        import numpy as np
        import torch
        from mshab.envs.planner import plan_data_from_file
        from mani_skill.utils.registration import REGISTERED_ENVS
        from bvi.fetch_pi_skill import JOINT_NAMES
        from bvi.ia_call_predicates import distance
        from bvi.native_pick_audit import state_max_errors
        from bvi.s1_diagnostic_cards import fixed_action

        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.chdir(root / "src/AC-DiT")
        rearrange = root / "assets/data/scene_datasets/replica_cad_dataset/rearrange"
        plan_path = rearrange / "task_plans/set_table/pick/val/013_apple.json"
        plans = plan_data_from_file(plan_path)
        env = gym.make(
            "PickSubtaskTrain-v0", num_envs=1, robot_uids="fetch", obs_mode="rgbdp",
            control_mode="pd_joint_delta_pos", render_mode="rgb_array", reward_mode="dense",
            sensor_configs={"shader_pack": "minimal"}, human_render_camera_configs={"shader_pack": "minimal"},
            viewer_camera_configs={"shader_pack": "minimal"}, sim_backend="gpu", max_episode_steps=200,
            task_plans=plans.plans, scene_builder_cls=plans.dataset,
            spawn_data_fp=rearrange / "spawn_data/set_table/pick/val/spawn_data.pt",
            require_build_configs_repeated_equally_across_envs=False,
        )
        reference = torch.load(args.reference_state, map_location="cpu", weights_only=False)
        saved = reference["simulator"]
        options = {"reconfigure": True, "build_config_idxs": jsonable(saved["build_config_idxs"]),
                   "task_plan_idxs": jsonable(saved["task_plan_idxs"])}
        obs, info = env.reset(seed=args.seed, options=options)
        u = env.unwrapped

        def device(value):
            if isinstance(value, torch.Tensor):
                return value.to(u.device)
            if isinstance(value, dict):
                return {key: device(item) for key, item in value.items()}
            return value

        u.set_state_dict(device(saved))
        u.agent.controller.set_state(device(reference["controller"]))
        info = device(reference["reset_info"])
        obs = u.get_obs(info=info)
        restored = {"simulator": u.get_state_dict(), "controller": u.agent.controller.get_state()}
        errors = state_max_errors(jsonable({key: reference[key] for key in restored}), jsonable(restored))
        if max(errors.values(), default=0.0) > 1e-5:
            raise RuntimeError("Saved state restoration exceeds 1e-5")
        if [joint.name for joint in u.agent.robot.active_joints] != JOINT_NAMES or u.control_freq != 20:
            raise RuntimeError("Native joint/control contract changed")
        if u.pick_cfg.robot_cumulative_force_limit != 5000:
            raise RuntimeError("Frozen cumulative-force threshold changed")
        target = u.subtask_objs[0]

        def contact() -> list[float]:
            return [float(torch.linalg.norm(u.scene.get_pairwise_contact_forces(link, target), dim=1)[0].item())
                    for link in (u.agent.finger1_link, u.agent.finger2_link)]

        fingers = u.agent.robot.qpos[0, -2:].detach().cpu().numpy()
        replay_actions = []
        if args.mode == "replay":
            for line in args.actions_jsonl.read_text().splitlines():
                value = np.asarray(json.loads(line)["action"], dtype=np.float32).reshape(-1)
                if value.shape != (13,) or not np.isfinite(value).all():
                    raise ValueError("Recorded SAC action contract changed")
                replay_actions.append(value)
            if not replay_actions:
                raise ValueError("No recorded SAC actions")
            action = replay_actions[0]
            report.update(actions_jsonl_sha256=sha256(args.actions_jsonl), policy="official_SAC_action_replay")
        else:
            action = fixed_action(args.mode if args.mode != "audit" else "zero", fingers)
        initial = {
            "tcp_target_distance_m": distance(jsonable(obs["extra"])),
            "finger_qpos_m": fingers.tolist(),
            "grasped": bool(obs["extra"]["is_grasped"].item()),
            "fingertip_target_contact_force_n": contact(),
            "robot_cumulative_force": float(info["robot_cumulative_force"].item()),
            "native_fail": bool(info["fail"].item()),
            "native_success": bool(info["success"].item()),
            "action": action.tolist() if args.mode != "audit" else None,
        }
        report.update(
            status="audit_completed" if args.mode == "audit" else "episode_running",
            initial=initial,
            reset_state_max_errors=errors,
            force_limit=5000.0,
            force_limit_source={
                "task_cfg": str(u.task_cfgs["pick"]),
                "class_sources": {str(Path(inspect.getfile(cls)).resolve()): sha256(inspect.getfile(cls))
                                  for cls in REGISTERED_ENVS["PickSubtaskTrain-v0"].cls.__mro__
                                  if cls.__module__.startswith(("mshab.", "mani_skill."))},
            },
            action_contract={"arm_delta": [0, 7], "gripper_absolute": [7, 8],
                             "body_delta": [8, 11], "base_velocity": [11, 13]},
        )
        save()
        if args.mode == "audit":
            return
        grasp_streak = 0
        with (out / "events.jsonl").open("x") as log:
            for index in range(args.max_actions):
                if args.mode == "replay":
                    if index >= len(replay_actions):
                        report.update(success=False, stop_reason="recorded_action_exhausted",
                                      reason_code="source_replay_incomplete")
                        break
                    action = replay_actions[index]
                obs, _, terminated, truncated, info = env.step(torch.from_numpy(action[None]).to(u.device))
                held = bool(obs["extra"]["is_grasped"].item())
                grasp_streak = grasp_streak + 1 if held else 0
                cumulative = float(info["robot_cumulative_force"].item())
                row = {
                    "step": index + 1, "action": action, "tcp_target_distance_m": distance(jsonable(obs["extra"])),
                    "fingertip_target_contact_force_n": contact(), "grasped": held,
                    "grasp_streak": grasp_streak, "robot_force": info["robot_force"],
                    "robot_cumulative_force": info["robot_cumulative_force"], "info": info,
                    "qpos": u.agent.robot.qpos, "terminated": terminated, "truncated": truncated,
                }
                log.write(json.dumps(jsonable(row)) + "\n")
                log.flush()
                report.update(steps=index + 1, final=row, peak_cumulative_force=max(
                    float(report.get("peak_cumulative_force", initial["robot_cumulative_force"])), cumulative))
                if args.mode == "close" and grasp_streak >= 3:
                    report.update(success=True, stop_reason="stable_grasp_3_observations", reason_code="grasp_control_success")
                    break
                if args.mode == "replay" and bool(info["success"].item()):
                    report.update(success=True, stop_reason="native_pick_success", reason_code="native_pick_success")
                    break
                if bool(terminated.item()) or bool(truncated.item()):
                    force_fail = not bool(info["cumulative_force_within_limit"].item())
                    report.update(success=False,
                                  stop_reason="native_termination" if bool(terminated.item()) else "native_truncation",
                                  reason_code="cumulative_force_limit" if force_fail else "native_other_termination")
                    break
            else:
                report.update(success=False, stop_reason="action_budget", reason_code="no_native_termination")
        report["status"] = "episode_completed"
    except Exception as exc:
        report.update(status="infrastructure_failure", error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        save()


if __name__ == "__main__":
    main()
