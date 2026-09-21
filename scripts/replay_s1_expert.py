"""Replay one recorded MS-HAB expert trajectory and measure state drift.

This is a Level-2 diagnostic, not a VLA evaluation. The source H5 contains
observations and actions but no simulator/controller snapshot. Consequently:

* ``cpu_reset`` reconstructs the recorded reset from episode metadata.
* ``gpu_snapshot_restore`` first reconstructs that reset on the GPU backend,
  serializes the *newly reconstructed* full state, reloads/restores it, and
  only then replays the expert actions.

The second path exercises our snapshot-restore machinery; it is never described
as restoration of the unavailable original collection snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
PATH_CONFIG = {
    "cpu_reset": {"backend": "cpu", "uses_snapshot_restore": False},
    "gpu_snapshot_restore": {"backend": "gpu", "uses_snapshot_restore": True},
}

# Diagnostic thresholds, not benchmark tolerances. Raw values and errors are
# retained in trace.jsonl so thresholds can be changed without rerunning physics.
THRESHOLDS = {
    "qpos": 1e-4,
    "qvel": 1e-3,
    "position_m": 1e-4,
    "rotation_rad": 1e-3,
    "grasp_bool": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quaternion_angle(a: Any, b: Any) -> float:
    """Sign-invariant angular distance in radians for xyzw quaternions."""
    import numpy as np

    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na == 0 or nb == 0:
        return float("inf")
    dot = float(np.clip(abs(np.dot(qa / na, qb / nb)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument(
        "--source-env-index",
        type=int,
        required=True,
        help="parallel environment index that produced this H5 trajectory",
    )
    parser.add_argument("--path", choices=sorted(PATH_CONFIG), required=True)
    parser.add_argument("--shader", choices=["default", "minimal"], default="minimal")
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.max_actions <= 200:
        parser.error("--max-actions must be in [0, 200]")

    path_config = PATH_CONFIG[args.path]
    backend = path_config["backend"]
    used = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                GPU,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if used >= 1024:
        raise RuntimeError("GPU1 not idle; expert replay must remain serialized")

    os.environ.update(
        CUDA_VISIBLE_DEVICES=GPU,
        OMP_NUM_THREADS="2",
        MS_ASSET_DIR=str(Path.home() / "bvi-research/assets"),
    )
    root = Path.home() / "bvi-research"
    sys.path[:0] = [
        str(root / "src/AC-DiT"),
        str(root / "src/Bidirectional-VLA-Interface/src"),
    ]

    import h5py
    import gymnasium as gym
    import mshab.envs  # noqa: F401 - registers environments
    import numpy as np
    import torch
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import FetchActionWrapper

    from bvi.mshab_adapter import jsonable
    from bvi.native_pick_audit import native_failure_causes, state_max_errors

    torch.set_num_threads(2)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = root / "data/fetch-tapt-source/pick/013_apple.h5"
    metadata = json.loads(source.with_suffix(".json").read_text())
    episode = next(
        (item for item in metadata["episodes"] if item["episode_id"] == args.parent),
        None,
    )
    if episode is None:
        raise ValueError(f"episode {args.parent} is absent from metadata")
    episode_seeds = episode["episode_seed"]
    if not isinstance(episode_seeds, list):
        raise ValueError("source metadata lacks the original parallel seed vector")
    if not 0 <= args.source_env_index < len(episode_seeds):
        raise ValueError("source environment index is outside the recorded seed vector")
    reset_seed = int(episode_seeds[args.source_env_index])

    started = time.monotonic()
    report: dict[str, Any] = {
        "stage": "S1-Level2",
        "question": "Do recorded expert actions reproduce recorded states through the current action wrapper and physics path?",
        "status": "initializing",
        "parent": args.parent,
        "source_env_index": args.source_env_index,
        "reset_seed": reset_seed,
        "initialization_path": args.path,
        "backend": backend,
        "shader": args.shader,
        "source_collection_backend": metadata["env_info"]["env_kwargs"]["sim_backend"],
        "source_collection_num_envs": metadata["env_info"]["env_kwargs"]["num_envs"],
        "exact_original_full_state": False,
        "original_snapshot_available": False,
        "missing_source_fields": [
            "simulator_state",
            "controller_state",
            "per_environment_RNG_state",
            "robot_force_curve",
            "robot_cumulative_force_curve",
        ],
        "snapshot_restore_scope": (
            "new GPU metadata reconstruction serialized and restored; not the original collection state"
            if path_config["uses_snapshot_restore"]
            else None
        ),
        "init_config_handling": (
            "MS-HAB derives init_config_idx from the selected task-plan init_config_name; reset has no "
            "init_config_idxs override. The integer mapping is revision-local, so it is reported but the "
            "recorded state at index 0 is the cross-revision validation."
        ),
        "teacher_replay": True,
        "vla_success": False,
        "training_updates": 0,
        "api_calls": 0,
        "new_rental_usd": 0,
        "lab_charge_usd": None,
        "max_actions": args.max_actions,
        "max_seconds": args.max_seconds,
        "gpu_uuid": GPU,
        "steps": 0,
        "success_once": False,
        "success_at_end": False,
        "ever_grasped": False,
        "thresholds": THRESHOLDS,
        "source_episode": {
            key: episode[key]
            for key in (
                "episode_id",
                "subtask_uid",
                "build_config_idx",
                "task_plan_idx",
                "init_config_idx",
                "spawn_selection_idx",
                "control_mode",
                "elapsed_steps",
                "label",
                "success_once",
                "success_at_end",
                "fail",
            )
            if key in episode
        },
        "source_collision_events_from_metadata": [
            event
            for event in episode.get("events_verbose", [])
            if isinstance(event, list) and len(event) >= 2 and event[1] == "collision"
        ],
        "runner_sha256": sha256(Path(__file__)),
        "source_h5_sha256": sha256(source),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_before_mib": used,
    }
    env = None

    def save() -> None:
        report["wall_seconds"] = time.monotonic() - started
        (out / "result.json").write_text(json.dumps(jsonable(report), indent=2))

    def array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def scalar(container: dict[str, Any], key: str, default: Any = None) -> Any:
        if key not in container:
            return default
        value = array(container[key]).reshape(-1)
        return value[0].item() if value.size else default

    save()
    try:
        random.seed(reset_seed)
        np.random.seed(reset_seed)
        torch.manual_seed(reset_seed)
        torch.cuda.manual_seed_all(reset_seed)
        os.chdir(root / "src/AC-DiT")
        rearrange = root / "assets/data/scene_datasets/replica_cad_dataset/rearrange"
        plans = plan_data_from_file(
            rearrange / "task_plans/set_table/pick/train/013_apple.json"
        )
        base_env = gym.make(
            "PickSubtaskTrain-v0",
            num_envs=1,
            robot_uids="fetch",
            obs_mode="rgbd",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            reward_mode="normalized_dense",
            sensor_configs={"shader_pack": args.shader},
            human_render_camera_configs={"shader_pack": args.shader},
            viewer_camera_configs={"shader_pack": args.shader},
            sim_backend=backend,
            max_episode_steps=200,
            task_plans=plans.plans,
            scene_builder_cls=plans.dataset,
            spawn_data_fp=rearrange / "spawn_data/set_table/pick/train/spawn_data.pt",
            require_build_configs_repeated_equally_across_envs=False,
            target_randomization=False,
            add_event_tracker_info=True,
            robot_force_mult=0.001,
            robot_force_penalty_min=0.2,
        )
        # This is the action wrapper used by the known-good official BC path.
        env = FetchActionWrapper(base_env, stationary_head=True)
        reset_options = {
            "reconfigure": True,
            "build_config_idxs": [episode["build_config_idx"]],
            "task_plan_idxs": [episode["task_plan_idx"]],
            "spawn_selection_idxs": [episode["spawn_selection_idx"]],
        }
        obs, info = env.reset(seed=reset_seed, options=reset_options)
        unwrapped = env.unwrapped

        # The recorder wrote composite_subtask_uids, not Subtask.uid. Comparing
        # against actual.subtasks[0].uid caused the earlier false mismatch.
        actual_composite_uid = str(unwrapped.task_plan[0].composite_subtask_uids[0])
        actual_metadata = {
            "composite_subtask_uid": actual_composite_uid,
            "build_config_idx": int(unwrapped.build_config_idxs[0]),
            "task_plan_idx": int(unwrapped.task_plan_idxs[0]),
            "init_config_idx": int(unwrapped.init_config_idxs[0]),
            "init_config_name": str(
                unwrapped.build_config_idx_to_task_plans[
                    int(unwrapped.build_config_idxs[0])
                ][int(unwrapped.task_plan_idxs[0])].init_config_name
            ),
            "spawn_selection_idx": int(unwrapped.spawn_selection_idxs[0]),
        }
        expected_metadata = {
            "composite_subtask_uid": episode["subtask_uid"],
            "build_config_idx": int(episode["build_config_idx"]),
            "task_plan_idx": int(episode["task_plan_idx"]),
            "init_config_idx": int(episode["init_config_idx"]),
            "spawn_selection_idx": int(episode["spawn_selection_idx"]),
        }
        semantic_keys = (
            "composite_subtask_uid",
            "build_config_idx",
            "task_plan_idx",
            "spawn_selection_idx",
        )
        semantic_identity_exact = all(
            actual_metadata[key] == expected_metadata[key] for key in semantic_keys
        )
        report.update(
            reset_options=reset_options,
            actual_reset_metadata=actual_metadata,
            expected_reset_metadata=expected_metadata,
            reset_metadata_exact=all(
                actual_metadata[key] == value for key, value in expected_metadata.items()
            ),
            reset_semantic_identity_exact=semantic_identity_exact,
            init_config_index_exact=(
                actual_metadata["init_config_idx"] == expected_metadata["init_config_idx"]
            ),
            init_config_index_comparison=(
                "diagnostic only: source metadata omits init_config_name and numeric indices changed "
                "between the collection runtime and the AC-DiT vendored MS-HAB revision"
            ),
            controller=str(
                getattr(
                    unwrapped.agent.controller,
                    "config",
                    getattr(unwrapped.agent.controller, "configs", None),
                )
            ),
            action_wrapper="mshab.envs.wrappers.FetchActionWrapper(stationary_head=True)",
        )
        if not semantic_identity_exact:
            raise ValueError("semantic metadata reset reconstruction mismatch")

        expected_pci = subprocess.check_output(
            ["nvidia-smi", "-i", GPU, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True,
        ).strip()
        report["renderer_pci"] = unwrapped._render_device.pci_string
        if str(report["renderer_pci"]).lower()[-7:] != expected_pci.lower()[-7:]:
            raise ValueError("renderer is not on physical GPU1")

        joint_names = [joint.name for joint in unwrapped.agent.robot.active_joints]
        observed_qpos_width = int(array(obs["agent"]["qpos"]).shape[-1])
        observation_joint_names = (
            joint_names[-observed_qpos_width:]
            if len(joint_names) >= observed_qpos_width
            else [f"coordinate_{index}" for index in range(observed_qpos_width)]
        )
        report["observation_qpos_channel_names"] = observation_joint_names

        def to_device(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(unwrapped.device)
            if isinstance(value, dict):
                return {key: to_device(item) for key, item in value.items()}
            return value

        reconstructed_snapshot = {
            "simulator": unwrapped.get_state_dict(),
            "controller": unwrapped.agent.controller.get_state(),
            "reset_info": info,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "provenance": "metadata reconstruction; original source snapshot unavailable",
        }
        snapshot_path = out / f"reconstructed-{args.path}-initial-state.pt"
        torch.save(reconstructed_snapshot, snapshot_path)
        report["reconstructed_snapshot_sha256"] = sha256(snapshot_path)

        if path_config["uses_snapshot_restore"]:
            # Force serialization/deserialization and a fresh env reset before
            # restoration. This exercises the disk snapshot path used by S1
            # paired evaluations while keeping the TimeLimit counter at zero.
            reference = torch.load(snapshot_path, map_location="cpu", weights_only=False)
            env.reset(seed=reset_seed, options=reset_options)
            unwrapped.set_state_dict(to_device(reference["simulator"]))
            unwrapped.agent.controller.set_state(to_device(reference["controller"]))
            info = to_device(reference["reset_info"])
            random.setstate(reference["python_rng"])
            np.random.set_state(reference["numpy_rng"])
            torch.set_rng_state(reference["torch_rng"])
            torch.cuda.set_rng_state_all(reference["cuda_rng"])
            obs = unwrapped.get_obs(info=info)
            restored = {
                "simulator": unwrapped.get_state_dict(),
                "controller": unwrapped.agent.controller.get_state(),
            }
            reference_state = {key: reference[key] for key in restored}
            restore_errors = state_max_errors(jsonable(reference_state), jsonable(restored))
            report["snapshot_restore_max_abs_error_by_leaf"] = restore_errors
            report["snapshot_restore_max_abs_error"] = max(
                restore_errors.values(), default=0.0
            )
            if report["snapshot_restore_max_abs_error"] > 1e-5:
                raise RuntimeError("reconstructed GPU snapshot restoration exceeds 1e-5")

        first_divergence: dict[str, Any] | None = None
        first_divergence_by_channel: dict[str, dict[str, Any]] = {}
        max_error_by_channel: dict[str, float] = {}
        action_mutation_max = 0.0
        peak_robot_force = 0.0
        peak_cumulative_force = 0.0

        def channel(
            values: list[dict[str, Any]],
            name: str,
            source_value: Any,
            replay_value: Any,
            error: float,
            threshold: float,
            unit: str,
        ) -> None:
            values.append(
                {
                    "channel": name,
                    "source": source_value,
                    "replay": replay_value,
                    "abs_error": float(error),
                    "threshold": threshold,
                    "unit": unit,
                }
            )

        def compare_state(group: Any, observation: dict[str, Any], state_index: int) -> dict[str, Any]:
            values: list[dict[str, Any]] = []
            for field, threshold, unit in (
                ("qpos", THRESHOLDS["qpos"], "m_or_rad"),
                ("qvel", THRESHOLDS["qvel"], "m_or_rad_per_s"),
            ):
                expected = np.asarray(group[f"obs/agent/{field}"][state_index], dtype=np.float64)
                actual = array(observation["agent"][field])[0].astype(np.float64)
                if expected.shape != actual.shape:
                    raise ValueError(f"agent/{field} shape mismatch: {expected.shape} != {actual.shape}")
                for index, (source_value, replay_value) in enumerate(zip(expected, actual)):
                    joint = observation_joint_names[index]
                    channel(
                        values,
                        f"agent/{field}/{joint}",
                        float(source_value),
                        float(replay_value),
                        abs(float(replay_value - source_value)),
                        threshold,
                        unit,
                    )

            for pose_name in ("tcp_pose_wrt_base", "obj_pose_wrt_base"):
                expected = np.asarray(group[f"obs/extra/{pose_name}"][state_index], dtype=np.float64)
                actual = array(observation["extra"][pose_name])[0].astype(np.float64)
                for index, axis in enumerate("xyz"):
                    channel(
                        values,
                        f"extra/{pose_name}/position_{axis}",
                        float(expected[index]),
                        float(actual[index]),
                        abs(float(actual[index] - expected[index])),
                        THRESHOLDS["position_m"],
                        "m",
                    )
                channel(
                    values,
                    f"extra/{pose_name}/quaternion_angle",
                    expected[3:].tolist(),
                    actual[3:].tolist(),
                    quaternion_angle(expected[3:], actual[3:]),
                    THRESHOLDS["rotation_rad"],
                    "rad",
                )

            expected_goal = np.asarray(
                group["obs/extra/goal_pos_wrt_base"][state_index], dtype=np.float64
            )
            actual_goal = array(observation["extra"]["goal_pos_wrt_base"])[0].astype(np.float64)
            for index, axis in enumerate("xyz"):
                channel(
                    values,
                    f"extra/goal_pos_wrt_base/{axis}",
                    float(expected_goal[index]),
                    float(actual_goal[index]),
                    abs(float(actual_goal[index] - expected_goal[index])),
                    THRESHOLDS["position_m"],
                    "m",
                )

            expected_grasp = bool(group["obs/extra/is_grasped"][state_index])
            actual_grasp = bool(array(observation["extra"]["is_grasped"])[0])
            channel(
                values,
                "extra/is_grasped",
                expected_grasp,
                actual_grasp,
                float(expected_grasp != actual_grasp),
                THRESHOLDS["grasp_bool"],
                "bool",
            )

            expected_tcp = np.asarray(
                group["obs/extra/tcp_pose_wrt_base"][state_index, :3], dtype=np.float64
            )
            expected_obj = np.asarray(
                group["obs/extra/obj_pose_wrt_base"][state_index, :3], dtype=np.float64
            )
            actual_tcp = array(observation["extra"]["tcp_pose_wrt_base"])[0, :3]
            actual_obj = array(observation["extra"]["obj_pose_wrt_base"])[0, :3]
            return {
                "state_index": state_index,
                "channels": values,
                "source_tcp_object_distance_m": float(np.linalg.norm(expected_tcp - expected_obj)),
                "replay_tcp_object_distance_m": float(np.linalg.norm(actual_tcp - actual_obj)),
            }

        def observe_divergence(state_row: dict[str, Any]) -> None:
            nonlocal first_divergence
            for item in state_row["channels"]:
                name = item["channel"]
                max_error_by_channel[name] = max(
                    max_error_by_channel.get(name, 0.0), item["abs_error"]
                )
                if item["abs_error"] > item["threshold"]:
                    event = {"state_index": state_row["state_index"], **item}
                    first_divergence_by_channel.setdefault(name, event)
                    if first_divergence is None:
                        first_divergence = event

        with h5py.File(source, "r") as handle, (out / "trace.jsonl").open("x") as trace:
            group = handle[f"traj_{args.parent}"]
            source_actions = np.asarray(group["actions"])
            if source_actions.shape[1:] != (13,):
                raise ValueError(f"source action contract changed: {source_actions.shape}")
            report["source_action_abs_max"] = float(np.abs(source_actions).max())
            report["source_head_action_abs_max"] = float(np.abs(source_actions[:, 8:10]).max())
            report["source_first_success_action"] = (
                int(np.flatnonzero(group["success"][:])[0]) + 1
                if np.any(group["success"][:])
                else None
            )
            if report["source_action_abs_max"] > 1.0 + 1e-6:
                raise ValueError("recorded expert actions are outside controller bounds")
            if report["source_head_action_abs_max"] != 0.0:
                raise ValueError("recorded expert actions violate stationary-head contract")

            initial_state = compare_state(group, obs, 0)
            observe_divergence(initial_state)
            initial_row = {
                **initial_state,
                "action_step": None,
                "robot_force": scalar(info, "robot_force", 0.0),
                "robot_cumulative_force": scalar(info, "robot_cumulative_force", 0.0),
            }
            trace.write(json.dumps(jsonable(initial_row)) + "\n")
            trace.flush()

            action_count = min(args.max_actions, len(source_actions))
            for step in range(action_count):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("expert replay wall limit exceeded")
                raw = source_actions[step].astype(np.float32, copy=True)
                bounded = np.clip(raw, -1.0, 1.0)
                action = (
                    torch.as_tensor(bounded[None], device=unwrapped.device)
                    if backend == "gpu"
                    else bounded[None]
                )
                # FetchActionWrapper performs the stationary-head mutation.
                obs, _, terminated, truncated, info = env.step(action)
                applied = array(action)[0]
                mutation = float(np.max(np.abs(applied - raw)))
                action_mutation_max = max(action_mutation_max, mutation)
                state_row = compare_state(group, obs, step + 1)
                observe_divergence(state_row)

                robot_force = float(scalar(info, "robot_force", 0.0))
                cumulative_force = float(scalar(info, "robot_cumulative_force", 0.0))
                peak_robot_force = max(peak_robot_force, robot_force)
                peak_cumulative_force = max(peak_cumulative_force, cumulative_force)
                grasped = bool(array(obs["extra"]["is_grasped"])[0])
                success = bool(scalar(info, "success", False))
                report.update(
                    steps=step + 1,
                    ever_grasped=report["ever_grasped"] or grasped,
                    success_once=report["success_once"] or success,
                    success_at_end=success,
                    final_info=jsonable(info),
                    failure_causes=native_failure_causes(jsonable(info)),
                )
                row = {
                    **state_row,
                    "action_step": step + 1,
                    "raw_action": raw,
                    "applied_action": applied,
                    "action_mutation_max_abs": mutation,
                    "source_success": bool(group["success"][step]),
                    "replay_success": success,
                    "source_terminated": bool(group["terminated"][step]),
                    "replay_terminated": bool(array(terminated).reshape(-1)[0]),
                    "source_truncated": bool(group["truncated"][step]),
                    "replay_truncated": bool(array(truncated).reshape(-1)[0]),
                    "grasped": grasped,
                    "robot_force": robot_force,
                    "robot_cumulative_force": cumulative_force,
                }
                trace.write(json.dumps(jsonable(row)) + "\n")
                trace.flush()

        report.update(
            status="completed_expert_state_comparison",
            trace="trace.jsonl",
            trace_sha256=sha256(out / "trace.jsonl"),
            first_divergence=first_divergence,
            first_divergence_by_channel=first_divergence_by_channel,
            max_abs_error_by_channel=max_error_by_channel,
            replay_matches_recorded_state_within_thresholds=first_divergence is None,
            action_wrapper_mutation_max_abs=action_mutation_max,
            expected_action_wrapper_mutation="zero because recorded actions are already clipped with stationary head",
            peak_robot_force=peak_robot_force,
            peak_robot_cumulative_force=peak_cumulative_force,
            force_comparison_available=False,
            force_comparison_note="source H5 has no force trace; replay force curves are retained in trace.jsonl",
        )
    except Exception as error:
        report.update(status="infrastructure_failure", error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        try:
            after = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        GPU,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            report["gpu_after_mib"] = after
        except Exception as error:  # preserve experiment result if telemetry fails
            report["gpu_after_error"] = repr(error)
        save()


if __name__ == "__main__":
    main()
