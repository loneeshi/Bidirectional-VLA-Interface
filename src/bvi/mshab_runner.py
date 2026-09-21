"""Execute one frozen Fixed, GPT, or Teleport TidyHouse episode.

This evaluation-only module is launched by ``bvi-eval run``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

from . import (
    APIBudget,
    JsonlLogger,
    ProtocolError,
    Requirement,
    SerialRuntime,
    SkillRequest,
    SkillStatus,
    VLMCoordinator,
)
from .evaluation import PROFILES
from .mshab_adapter import OfficialRLSkill, jsonable, make_mshab_adapter, scalar


def oracle_skill_timeout(spec_timeout: float, remaining_wall: float) -> float | None:
    if not math.isfinite(remaining_wall) or remaining_wall <= 0.5:
        return None
    return min(spec_timeout, remaining_wall - 0.5)


def oracle_step_budget(spec_steps: int, remaining_steps: int) -> int:
    """Keep the native PPO/SAC horizon for non-GPT settings."""
    return min(spec_steps, remaining_steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=tuple(PROFILES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-plan-uid", required=True)
    parser.add_argument("--expected-initial-state-sha256")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path)
    parser.add_argument("--authorization-id")
    parser.add_argument("--no-video", action="store_true")
    return parser


def _validate_checkpoints(parser: argparse.ArgumentParser, checkpoint_root: Path) -> None:
    from mshab.evaluate import POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS

    mapping = POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"]["tidy_house"]
    required = {
        "navigate": ["all"],
        "pick": [target for target in mapping["pick"] if target != "all"],
        "place": [target for target in mapping["place"] if target != "all"],
    }
    base = checkpoint_root / "rl/tidy_house"
    for skill, targets in required.items():
        if not targets:
            parser.error(f"No registered {skill} targets for rl_per_obj")
        for target in targets:
            directory = base / skill / target
            for filename in ("policy.pt", "config.yml"):
                if not (directory / filename).is_file():
                    parser.error(f"Missing {skill}/{target}/{filename} under {checkpoint_root}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = PROFILES[args.setting]
    if args.setting == "gpt" and (args.bridge_dir is None or not args.authorization_id):
        parser.error("GPT setting requires --bridge-dir and --authorization-id")
    if args.setting != "fixed" and not args.expected_initial_state_sha256:
        parser.error("GPT and teleport settings require a paired fixed initial-state hash")

    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig

    plan_path = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json"
    if not plan_path.is_file():
        parser.error(f"Missing task plans: {plan_path}")
    checkpoint_root = args.checkpoint_root.resolve()
    _validate_checkpoints(parser, checkpoint_root)

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(args.output / "events.jsonl", args.output.name)

    import bvi.nav_camera_env  # noqa: F401
    from mani_skill.envs.sapien_env import BaseEnv

    env_kwargs = {
        "require_build_configs_repeated_equally_across_envs": False,
        "add_event_tracker_info": True,
        "human_render_camera_configs": {"width": 512, "height": 512},
        "task_cfgs": {"navigate": {"ignore_arm_checkers": True}},
    }
    if "invisible_goals_in_human_render" in inspect.signature(BaseEnv.__init__).parameters:
        env_kwargs["invisible_goals_in_human_render"] = True
    cfg = EnvConfig(
        env_id="BVISequentialWorkspaceCamera-v0",
        num_envs=1,
        max_episode_steps=profile.max_env_steps,
        task_plan_fp=str(plan_path),
        obs_mode="rgbd",
        render_mode="rgb_array",
        record_video=not args.no_video,
        info_on_video=False,
        continuous_task=True,
        frame_stack=3,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=True,
        env_kwargs=env_kwargs,
    )

    budget = None
    if args.setting == "gpt":
        budget = APIBudget(
            args.authorization_id,
            profile.max_calls,
            profile.max_output_tokens,
            profile.max_api_cost_usd,
            profile.request_cost_ceiling_usd,
            profile.max_input_bytes,
        )

    metadata = {
        "schema": "bvi-eval-episode/1",
        "setting": args.setting,
        "profile": profile.__dict__,
        "benchmark_result": False,
        "evaluation_eligible": True,
        "seed": args.seed,
        "plan_uid": args.expected_plan_uid,
        "config": jsonable(cfg),
        "api_budget": jsonable(budget),
        "training_updates": 0,
    }
    source_root = Path(__file__).resolve().parents[2]
    source_files = sorted((source_root / "src/bvi").glob("*.py"))
    metadata["runtime_source_sha256"] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    import mshab.envs.sequential_task as mshab_runtime

    runtime_path = Path(mshab_runtime.__file__).resolve()
    metadata["mshab_runtime"] = {
        "path": str(runtime_path),
        "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    }
    (args.output / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    adapter = None
    coordinator = None
    organizer = None
    history: list[dict] = []
    decisions = 0
    api_calls = 0
    reason = "not_started"
    started = time.monotonic()
    try:
        adapter = make_mshab_adapter(cfg, logger, args.output, seed=args.seed)
        if adapter.original_plan.subtasks[0].uid != args.expected_plan_uid:
            raise ProtocolError("Sampled plan does not match expected-plan-uid")
        initial_state = jsonable(adapter.uenv.get_state_dict())
        initial_hash = hashlib.sha256(
            json.dumps(initial_state, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()
        initial = {
            "seed": args.seed,
            "plan_uid": adapter.original_plan.subtasks[0].uid,
            "state_sha256": initial_hash,
            "state": initial_state,
            "task_plan": jsonable(adapter.original_plan),
            "policy_state_shape": list(adapter.observe().policy["state"].shape),
            "native_horizons": {
                name: int(adapter.uenv.task_cfgs[name].horizon)
                for name in ("navigate", "pick", "place")
            },
        }
        (args.output / "initial-state.json").write_text(json.dumps(initial), encoding="utf-8")
        if args.expected_initial_state_sha256 and initial_hash != args.expected_initial_state_sha256:
            raise ProtocolError("Paired initial simulator state hash mismatch")
        if initial["policy_state_shape"] != [1, 42] or initial["native_horizons"] != {
            "navigate": 500,
            "pick": 200,
            "place": 200,
        }:
            raise ProtocolError("Paired official runtime state/horizon mismatch")

        specs = adapter.skill_specs(profile.skill_wall_seconds)
        skills = {
            name: OfficialRLSkill(name, adapter, checkpoint_root, "rl_per_obj")
            for name in specs
        }
        if args.setting == "gpt":
            from .goal_tools import GoalRLSkill, GoalToolAdapter
            from .organizer import OrganizerView
            from .bridge import FileBridgeTransport

            adapter = GoalToolAdapter(adapter)
            skills = {
                name: GoalRLSkill(name, adapter, checkpoint_root, "rl_per_obj")
                for name in specs
            }
            organizer = OrganizerView(adapter, specs, profile.slice_steps, False)
            transport = FileBridgeTransport(
                args.bridge_dir.resolve(),
                "openai",
                profile.model,
                args.authorization_id,
                120,
                "low",
                "none",
            )
            coordinator = VLMCoordinator(transport, organizer.specs, logger, budget)
        elif args.setting == "teleport":
            from .teleport_skill import StandardizedTeleportSkill

            skills["navigate"] = StandardizedTeleportSkill(adapter, checkpoint_root, "rl_per_obj")

        runtime = SerialRuntime(adapter, skills, specs, logger)
        for index in range(profile.max_calls):
            if adapter.ended:
                reason = "environment_success" if bool(scalar(adapter.last_info.get("success", False))) else "environment_ended"
                break
            if adapter.steps >= profile.max_env_steps:
                reason = "experiment_step_limit"
                break
            if time.monotonic() - started >= profile.max_wall_seconds:
                reason = "experiment_wall_clock_limit"
                break
            observation = organizer.observe() if organizer else adapter.observe()
            adapter.save_observation_images()
            if not observation.allowed_calls:
                reason = "no_admissible_skill"
                break
            if args.setting == "gpt":
                request = coordinator.decide(observation, history)
                api_calls = coordinator.calls_reserved
                if request.skill == "abort_task":
                    logger.emit("organizer_abort", request=request, task_success=False)
                    decisions += 1
                    reason = "organizer_aborted_task"
                    break
                remaining_steps = profile.max_env_steps - adapter.steps
                if request.max_steps > remaining_steps:
                    reason = "request_exceeds_remaining_experiment_steps"
                    logger.emit("experiment_request_rejected", request=request, reason=reason)
                    break
            else:
                admissible = observation.allowed_calls[0]
                spec = specs[admissible.skill]
                timeout = oracle_skill_timeout(
                    spec.timeout_seconds,
                    profile.max_wall_seconds - (time.monotonic() - started),
                )
                if timeout is None:
                    reason = "experiment_wall_clock_limit"
                    break
                request = SkillRequest(
                    f"oracle-{index}",
                    admissible.skill,
                    admissible.target_id,
                    observation.frame_id,
                    (Requirement("benchmark-completion", "benchmark_success"),),
                    oracle_step_budget(spec.max_steps, profile.max_env_steps - adapter.steps),
                    timeout,
                )
                logger.emit("oracle_protocol_decision", request=request, vlm=False)
            remaining_wall = profile.max_wall_seconds - (time.monotonic() - started)
            if request.timeout_seconds > remaining_wall:
                reason = "request_exceeds_remaining_experiment_wall_budget"
                logger.emit("experiment_request_rejected", request=request, reason=reason)
                break
            decisions += 1
            result = runtime.execute(request)
            if organizer:
                organizer.note_result(result)
            history.append({
                "call_id": request.call_id,
                "skill": request.skill,
                "target_id": request.target_id,
                "requirements": request.requirements,
                "feedback": result.feedback,
                "steps": result.steps,
                "elapsed_seconds": result.elapsed_seconds,
            })
            print(json.dumps({
                "call_id": request.call_id,
                "skill": request.skill,
                "status": result.feedback.status.value,
                "steps": result.steps,
                "reason": result.feedback.reason,
            }), flush=True)
            if args.setting == "gpt" and result.feedback.status is SkillStatus.REJECTED:
                continue
            if result.feedback.status in (SkillStatus.FAILED, SkillStatus.REJECTED):
                reason = result.feedback.reason or "skill_failed"
                break
        else:
            reason = "experiment_call_limit"
        if adapter.ended and bool(scalar(adapter.last_info.get("success", False))):
            reason = "environment_success"
        adapter.save_observation_images()
    except Exception as exc:
        reason = f"error:{type(exc).__name__}"
        logger.emit("experiment_error", error_type=type(exc).__name__)
        raise
    finally:
        if adapter is not None:
            if coordinator is not None:
                api_calls = coordinator.calls_reserved
            native_index = int(scalar(adapter.uenv.subtask_pointer))
            completed_objects = sum(
                subtask.type == "place" for subtask in adapter.original_plan.subtasks[:native_index]
            )
            planned_objects = sum(subtask.type == "place" for subtask in adapter.original_plan.subtasks)
            summary = {
                "benchmark_result": False,
                "benchmark_episode": True,
                "evaluation_eligible": True,
                "setting": args.setting,
                "reason": reason,
                "decisions": decisions,
                "api_requests": api_calls,
                "vlm": args.setting == "gpt",
                "vlm_feedback_loop_observed": args.setting == "gpt" and len(history) >= 2,
                "task_success": bool(scalar(adapter.last_info.get("success", False))),
                "completed_objects": min(completed_objects, planned_objects),
                "planned_objects": planned_objects,
                "steps": adapter.steps,
                "wall_seconds": time.monotonic() - started,
                "skill_results": jsonable(history),
                "navigation_policy": profile.navigation_policy,
                "manipulation_policy": profile.manipulation_policy,
                "training_updates": 0,
                "api_cost_usd": None if args.setting == "gpt" else 0,
                "api_cost_status": "pending_reconciliation" if args.setting == "gpt" else "no_api_calls",
            }
            try:
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                logger.emit("experiment_finished", **summary)
            finally:
                adapter.close()
            print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
