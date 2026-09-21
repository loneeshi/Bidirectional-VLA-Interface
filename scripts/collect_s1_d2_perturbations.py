"""Replay the frozen SAC trajectory and save the preregistered D2 +3/+5 cm starts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    parser.add_argument("--reach-state", type=Path, required=True)
    parser.add_argument("--grasp-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise ValueError("PYTHONHASHSEED must match seed before process start")
    root = Path.home() / "bvi-research"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "card": "D2", "seed": args.seed, "status": "preflight",
        "question": "Does grasp persist from +3 cm and +5 cm earlier states on the same SAC trajectory?",
        "api_calls": 0, "training_updates": 0, "threshold_changes": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "reach_state_sha256": sha256(args.reach_state), "grasp_state_sha256": sha256(args.grasp_state),
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
        import yaml
        from gymnasium import spaces
        from mani_skill.utils.common import flatten_state_dict
        from mshab.agents.sac import Agent
        from mshab.envs.planner import plan_data_from_file
        from bvi.ia_call_predicates import distance
        from bvi.native_pick_audit import state_max_errors
        from bvi.s1_diagnostic_cards import perturbation_targets

        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.chdir(root / "src/AC-DiT")
        rearrange = root / "assets/data/scene_datasets/replica_cad_dataset/rearrange"
        plans = plan_data_from_file(rearrange / "task_plans/set_table/pick/val/013_apple.json")
        env = gym.make(
            "PickSubtaskTrain-v0", num_envs=1, robot_uids="fetch", obs_mode="rgbdp",
            control_mode="pd_joint_delta_pos", render_mode="rgb_array", reward_mode="dense",
            sensor_configs={"shader_pack": "minimal"}, human_render_camera_configs={"shader_pack": "minimal"},
            viewer_camera_configs={"shader_pack": "minimal"}, sim_backend="gpu", max_episode_steps=200,
            task_plans=plans.plans, scene_builder_cls=plans.dataset,
            spawn_data_fp=rearrange / "spawn_data/set_table/pick/val/spawn_data.pt",
            require_build_configs_repeated_equally_across_envs=False,
        )
        u = env.unwrapped

        def device(value):
            if isinstance(value, torch.Tensor):
                return value.to(u.device)
            if isinstance(value, dict):
                return {key: device(item) for key, item in value.items()}
            return value

        def restore(path: Path):
            reference = torch.load(path, map_location="cpu", weights_only=False)
            saved = reference["simulator"]
            options = {"reconfigure": True, "build_config_idxs": jsonable(saved["build_config_idxs"]),
                       "task_plan_idxs": jsonable(saved["task_plan_idxs"])}
            obs, info = env.reset(seed=args.seed, options=options)
            u.set_state_dict(device(saved))
            u.agent.controller.set_state(device(reference["controller"]))
            info = device(reference["reset_info"])
            obs = u.get_obs(info=info)
            errors = state_max_errors(jsonable(saved), jsonable(u.get_state_dict()))
            if max(errors.values(), default=0.0) > 1e-5:
                raise RuntimeError("Saved simulator state restoration exceeds 1e-5")
            return obs, info

        grasp_obs, grasp_info = restore(args.grasp_state)
        original_distance = distance(jsonable(grasp_obs["extra"]))
        targets = perturbation_targets(original_distance)
        report.update(original_grasp_distance_m=original_distance, target_distances_m=targets)
        if bool(grasp_info["fail"].item()) or bool(grasp_info["success"].item()):
            report.update(status="not_evaluable_terminal_grasp_start", reason_code="original_start_terminal")
            return
        if not targets:
            report.update(status="not_evaluable_target_band", reason_code="plus_3_5_cm_outside_5_8_cm_band")
            return

        obs, info = restore(args.reach_state)
        from collections import deque
        frames = {camera: deque(maxlen=3) for camera in ("fetch_head", "fetch_hand")}

        def encode(observation, first=False):
            pixels = {}
            for camera, queue in frames.items():
                depth = observation["sensor_data"][camera]["depth"].permute(0, 3, 1, 2)
                for _ in range(3 if first else 1):
                    queue.append(depth)
                pixels[camera + "_depth"] = torch.cat(list(queue), dim=1).to("cuda", dtype=torch.float32).contiguous()
            official_extra = {key: observation["extra"][key] for key in
                              ("tcp_pose_wrt_base", "obj_pose_wrt_base", "goal_pos_wrt_base", "is_grasped")}
            state = torch.cat([flatten_state_dict(observation["agent"], use_torch=True),
                               flatten_state_dict(official_extra, use_torch=True)], dim=1).to("cuda")
            if state.shape != (1, 42):
                raise RuntimeError(f"Official SAC state contract changed: {state.shape}")
            return pixels, state

        pixels, state = encode(obs, True)
        checkpoint = root / "checkpoints/mshab/rl/set_table/pick/013_apple"
        cfg = yaml.safe_load((checkpoint / "config.yml").read_text())["algo"]
        keys = ("actor_hidden_dims", "critic_hidden_dims", "critic_layer_norm", "critic_dropout",
                "encoder_pixels_feature_dim", "encoder_state_feature_dim", "cnn_features", "cnn_filters",
                "cnn_strides", "cnn_padding")
        policy = Agent(
            spaces.Dict({key: spaces.Box(0, 32767, tuple(value.shape[1:]), np.int16)
                         for key, value in pixels.items()}), tuple(state.shape[1:]), (13,),
            **{key: cfg[key] for key in keys}, log_std_min=cfg["actor_log_std_min"],
            log_std_max=cfg["actor_log_std_max"], device="cuda",
        )
        policy.load_state_dict(torch.load(checkpoint / "policy.pt", map_location="cuda", weights_only=False)["agent"], strict=True)
        policy.to("cuda").eval()
        report["sac_checkpoint_sha256"] = sha256(checkpoint / "policy.pt")
        candidates = {index: {"error": float("inf"), "step": None, "distance": None}
                      for index in range(len(targets))}

        def consider(step: int) -> None:
            current = distance(jsonable(obs["extra"]))
            held = bool(obs["extra"]["is_grasped"].item())
            if held or not 0.05 <= current <= 0.08:
                return
            for index, target_distance in enumerate(targets):
                error = abs(current - target_distance)
                if error < candidates[index]["error"]:
                    snapshot = {"simulator": u.get_state_dict(), "controller": u.agent.controller.get_state(),
                                "reset_info": info, "source_sac_step": step,
                                "target_distance_m": target_distance, "actual_distance_m": current}
                    path = out / f"offset-{(index * 2) + 3}cm-state.pt"
                    torch.save(snapshot, path)
                    candidates[index] = {"error": error, "step": step, "distance": current,
                                         "path": path.name, "sha256": sha256(path)}

        consider(0)
        report["status"] = "replay_running"
        with (out / "events.jsonl").open("x") as log:
            for step in range(200):
                with torch.no_grad():
                    action = policy.actor(pixels, state, compute_pi=False, compute_log_pi=False)[0].cpu()
                action[..., 8:10] = 0
                action = action.clamp(-1, 1)
                obs, _, terminated, truncated, info = env.step(action.to(u.device))
                current = distance(jsonable(obs["extra"]))
                consider(step + 1)
                log.write(json.dumps(jsonable({"step": step + 1, "distance_m": current, "action": action,
                                               "grasped": obs["extra"]["is_grasped"], "info": info})) + "\n")
                log.flush()
                if current <= original_distance or bool(terminated.item()) or bool(truncated.item()):
                    break
                pixels, state = encode(obs)
        if any(candidate["step"] is None for candidate in candidates.values()):
            report.update(status="not_evaluable_missing_same_trajectory_state", candidates=candidates)
            return
        report.update(status="completed", candidates=candidates, replay_steps=step + 1)
    except Exception as exc:
        report.update(status="infrastructure_failure", error=repr(exc))
        raise
    finally:
        if env is not None:
            env.close()
        save()


if __name__ == "__main__":
    main()
