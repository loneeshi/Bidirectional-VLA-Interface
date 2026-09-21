"""Bind C1 seeds 0--9 to real TidyHouse validation plan UIDs by environment reset.

This performs no policy actions and no API calls.  It writes a new bound config;
the preregistration draft remains immutable evidence of the pre-binding state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

from bvi import JsonlLogger
from bvi.mshab_adapter import jsonable, make_mshab_adapter


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = config.get("evaluation_episodes")
    if config.get("status") != "blocked_until_plan_uids_and_assets_bound":
        raise ValueError("Expected the immutable unbound preregistration draft")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(range(10)):
        raise ValueError("Expected the frozen ordered seeds 0..9")
    if any(row.get("plan_uid") is not None for row in rows):
        raise ValueError("Draft already contains a UID; refuse ambiguous rebinding")
    used = int(subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip())
    if used >= 1024:
        raise RuntimeError("GPU1 occupied; binding refused")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU)
    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig

    plan_path = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json"
    checkpoint_base = args.checkpoint_root / "rl/tidy_house"
    checkpoint_files = sorted(
        path for skill in ("navigate", "pick", "place")
        for path in (checkpoint_base / skill).glob("*/policy.pt")
    )
    required_configs = [path.with_name("config.yml") for path in checkpoint_files]
    if (not plan_path.is_file() or
            not (checkpoint_base / "navigate/all/policy.pt").is_file() or
            not checkpoint_files or
            any(not path.is_file() for path in required_configs)):
        raise FileNotFoundError("C1 plan or official rl_per_obj checkpoint set is incomplete")
    env_kwargs = {"require_build_configs_repeated_equally_across_envs": False,
                  "add_event_tracker_info": True,
                  "human_render_camera_configs": {"width": 512, "height": 512},
                  "task_cfgs": {"navigate": {"ignore_arm_checkers": True}}}
    # The lab's pinned ManiSkill fork predates this render-only option.  It does
    # not alter policy observations, so omit it only when BaseEnv cannot accept it.
    from mani_skill.envs.sapien_env import BaseEnv
    if "invisible_goals_in_human_render" in inspect.signature(BaseEnv.__init__).parameters:
        env_kwargs["invisible_goals_in_human_render"] = True
    env_config = EnvConfig(
        env_id="SequentialTask-v0", num_envs=1, max_episode_steps=7000,
        task_plan_fp=str(plan_path), obs_mode="rgbd", render_mode="rgb_array",
        record_video=False, info_on_video=False, continuous_task=True, frame_stack=3,
        stationary_base=False, stationary_torso=False, stationary_head=True,
        env_kwargs=env_kwargs,
    )
    evidence = []
    for seed in range(10):
        seed_output = output / f"seed-{seed:03d}"
        seed_output.mkdir()
        logger = JsonlLogger(seed_output / "events.jsonl", f"c1-bind-seed-{seed}")
        adapter = make_mshab_adapter(env_config, logger, seed_output, seed=seed)
        try:
            plan = adapter.original_plan
            subtasks = [{"uid": subtask.uid, "type": subtask.type,
                         "obj_id": getattr(subtask, "obj_id", None)} for subtask in plan.subtasks]
            first_uid = subtasks[0]["uid"]
            if not isinstance(first_uid, str) or not first_uid:
                raise RuntimeError("Real reset did not expose a first-subtask UID")
            evidence.append({"seed": seed, "plan_uid": first_uid, "subtasks": subtasks})
            (seed_output / "binding.json").write_text(json.dumps(evidence[-1], indent=2) + "\n")
        finally:
            adapter.close()
    bound = dict(config)
    bound["status"] = "bound_ready_for_preflight"
    bound["evaluation_episodes"] = [{"seed": row["seed"], "plan_uid": row["plan_uid"]}
                                    for row in evidence]
    bound["binding"] = {
        "method": "real_environment_reset_no_policy_actions",
        "bound_utc": datetime.now(timezone.utc).isoformat(),
        "source_config_sha256": sha256(args.config),
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "official_checkpoint_sha256": {
            str(path.relative_to(args.checkpoint_root)): sha256(path)
            for path in checkpoint_files
        },
        "condition1_progress_thresholds": "not_used_no_feedback_events",
        "policy_mode": "rl_per_obj",
        "recorded_demonstrations": "evaluation_evidence_only_not_progress_head_training",
        "environment_config": jsonable(env_config),
        "evidence": evidence,
        "api_calls": 0,
        "training_updates": 0,
    }
    (output / "weekly-baseline-v2-bound.json").write_text(json.dumps(bound, indent=2) + "\n")


if __name__ == "__main__":
    main()
