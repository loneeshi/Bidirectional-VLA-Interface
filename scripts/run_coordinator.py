"""Run real MS-HAB skills through the constrained coordinator interface.

--dry-run uses an explicitly labelled oracle dispatcher with real simulation and
policies; it makes zero model API requests but still uses the host GPU. Live VLM
mode requires credentials and an explicit bounded API authorization. This
single-scene diagnostic run does not produce an official benchmark score.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

from bvi import (APIBudget, JsonlLogger, ProtocolError, Requirement, SerialRuntime,
                 SkillRequest, SkillStatus, VLMCoordinator)
from bvi.mshab_adapter import OfficialRLSkill, jsonable, make_mshab_adapter, scalar
from bvi.providers import AnthropicTransport, OpenAITransport


def load_local_credentials(path: Path) -> None:
    """Read only known key names; never execute shell text or print credentials."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if value:
            os.environ.setdefault(key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Real GPU rollout, oracle dispatch, no API")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--transport", choices=("direct", "bridge"), default="direct")
    parser.add_argument("--bridge-timeout-seconds", type=float, default=120)
    parser.add_argument("--model", help="Explicit account-accessible vision model ID")
    parser.add_argument("--credentials-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--authorization-id", help="Reference to approved experiment spending scope")
    parser.add_argument("--max-api-cost-usd", type=float)
    parser.add_argument("--request-cost-ceiling-usd", type=float,
                        help="Conservative reservation, verified for model and image budget")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--image-detail", choices=("low", "high", "original", "auto"), default="low")
    parser.add_argument("--reasoning-effort", default="none", help="OpenAI only; model must support it")
    parser.add_argument("--max-input-bytes", type=int, default=128_000,
                        help="Serialized content byte guard; not an exact provider token count")
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--max-env-steps", type=int, default=7000)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    parser.add_argument("--skill-wall-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-type", choices=("rl_all_obj", "rl_per_obj"), default="rl_all_obj")
    parser.add_argument("--checkpoint-root", type=Path, default=os.getenv("MSHAB_CHECKPOINT_DIR"))
    parser.add_argument("--output", type=Path, default=Path("runs/coordinator"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--navigation-policy", choices=("official", "lightnav"), default="official")
    parser.add_argument("--lightnav-url", default="ws://127.0.0.1:8050")
    parser.add_argument("--navigation-instructions", type=Path,
                        help="JSON mapping subtask indices to explicit visual language goals")
    parser.add_argument("--expected-plan-uid", help="Require the exact first subtask UID before LightNav")
    parser.add_argument("--max-navigation-predictions", type=int, default=40)
    parser.add_argument("--lightnav-subtasks", type=int, nargs='+',
                        help="Explicit diagnostic routing: use LightNav only at these indices; official elsewhere")
    parser.add_argument("--video-debug-overlay", action="store_true",
                        help="Burn verbose simulator statistics into diagnostic video")
    parser.add_argument("--show-goal-markers", action="store_true",
                        help="Show benchmark debug goals in human-render video")
    args = parser.parse_args()
    navigation_instructions = {}
    if args.navigation_policy == "lightnav":
        if args.navigation_instructions is None or not args.expected_plan_uid or args.max_navigation_predictions < 1:
            parser.error("LightNav requires instructions, expected-plan-uid and a positive prediction cap")
        navigation_instructions = json.loads(args.navigation_instructions.read_text(encoding="utf-8"))
        if not isinstance(navigation_instructions, dict) or not all(
            isinstance(k, str) and k.isdigit() and isinstance(v, str) and v.strip()
            for k, v in navigation_instructions.items()
        ):
            parser.error("Navigation instructions must map numeric string indices to nonempty strings")
    if args.checkpoint_root is None:
        parser.error("set --checkpoint-root or MSHAB_CHECKPOINT_DIR")
    if any(value <= 0 for value in (args.max_calls, args.max_env_steps, args.max_wall_seconds,
                                    args.skill_wall_seconds, args.max_output_tokens, args.max_input_bytes)):
        parser.error("Step, call, wall-clock and token limits must be positive")
    if args.max_env_steps > 7000:
        parser.error("The diagnostic TidyHouse runner allows at most its official 7000-step horizon")
    budget = None
    if not args.dry_run:
        if not (args.model and args.authorization_id and args.max_api_cost_usd
                and args.request_cost_ceiling_usd):
            parser.error("Live VLM requires --model, --authorization-id, --max-api-cost-usd, "
                         "and --request-cost-ceiling-usd")
        if args.transport == "direct":
            load_local_credentials(args.credentials_file)
            key_name = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
            if not os.getenv(key_name):
                parser.error(f"Missing {key_name}; put it in the process environment or .env.local")
            if importlib.util.find_spec(args.provider) is None:
                parser.error(f"Install the optional SDK first: pip install -e '.[{args.provider}]'")
        budget = APIBudget(args.authorization_id, args.max_calls, args.max_output_tokens,
                           args.max_api_cost_usd, args.request_cost_ceiling_usd, args.max_input_bytes)

    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig

    plan_path = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json"
    if not plan_path.is_file():
        parser.error(f"Missing task plans: {plan_path}")
    checkpoint_root = args.checkpoint_root.resolve()
    for name in ("navigate", "pick", "place"):
        if not (checkpoint_root / "rl/tidy_house" / name / "all/policy.pt").is_file():
            parser.error(f"Missing {name}/all checkpoint under {checkpoint_root}")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(args.output / "events.jsonl", args.output.name)
    cfg = EnvConfig(env_id="SequentialTask-v0", num_envs=1, max_episode_steps=7000,
        task_plan_fp=str(plan_path), obs_mode="rgbd", render_mode="rgb_array",
        record_video=not args.no_video, info_on_video=args.video_debug_overlay, continuous_task=True,
        frame_stack=3, stationary_base=False, stationary_torso=False, stationary_head=True,
        env_kwargs={"require_build_configs_repeated_equally_across_envs": False,
                    "invisible_goals_in_human_render": not args.show_goal_markers,
                    "add_event_tracker_info": True,
                    "human_render_camera_configs": {"width": 512, "height": 512},
                    "task_cfgs": {"navigate": {"ignore_arm_checkers": True}}})
    # Snapshot before official make_env expands task_plans inside env_kwargs.
    metadata = {"stage": "G4_adapter_diagnostic", "benchmark_result": False,
                "vlm": not args.dry_run,
                "dispatcher": "oracle_protocol_dry_run" if args.dry_run else "constrained_vlm",
                "coverage": "one validation scene / one sampled plan",
                "completion_source": "oracle_benchmark", "targets_source": "oracle_task_plan",
                "config": jsonable(cfg), "seed": args.seed, "policy_type": args.policy_type,
                "provider": None if args.dry_run else args.provider,
                "model": None if args.dry_run else args.model, "api_budget": jsonable(budget),
                "transport": args.transport,
                "image_detail": args.image_detail, "reasoning_effort": args.reasoning_effort,
                "max_env_steps": args.max_env_steps, "max_wall_seconds": args.max_wall_seconds}
    metadata.update(navigation_policy=args.navigation_policy,
                    lightnav_subtasks=args.lightnav_subtasks,
                    navigation_instructions=navigation_instructions,
                    max_navigation_predictions=args.max_navigation_predictions)
    (args.output / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    adapter = None
    navigation_client = None
    navigation_skill = None
    started = time.monotonic()
    history = []
    decisions = 0
    api_calls = 0
    reason = "not_started"
    coordinator = None
    try:
        adapter = make_mshab_adapter(cfg, logger, args.output, seed=args.seed)
        specs = adapter.skill_specs(args.skill_wall_seconds)
        skills = {name: OfficialRLSkill(name, adapter, checkpoint_root, args.policy_type) for name in specs}
        if args.navigation_policy == "lightnav":
            from bvi.lightnav_skill import LightNavSkill
            from bvi.vla_clients import LightNavClient
            if adapter.original_plan.subtasks[0].uid != args.expected_plan_uid:
                raise ProtocolError("Sampled plan does not match navigation language instructions")
            navigation_client = LightNavClient.connect(args.lightnav_url)
            navigation_skill = LightNavSkill(adapter, navigation_client, navigation_instructions,
                                             args.max_navigation_predictions)
            if args.lightnav_subtasks is not None:
                from bvi.lightnav_skill import IndexedNavigationSkill
                if any(i < 0 or i >= len(adapter.original_plan.subtasks) or
                       adapter.original_plan.subtasks[i].type != 'navigate'
                       for i in args.lightnav_subtasks):
                    raise ProtocolError('LightNav indices must select existing navigation subtasks')
                skills['navigate'] = IndexedNavigationSkill(navigation_skill, skills['navigate'],
                                                           args.lightnav_subtasks, logger)
            else:
                skills["navigate"] = navigation_skill
        runtime = SerialRuntime(adapter, skills, specs, logger)
        if not args.dry_run:
            if args.transport == "bridge":
                from bvi.bridge import FileBridgeTransport
                transport = FileBridgeTransport(args.output / "bridge", args.provider, args.model,
                    args.authorization_id, args.bridge_timeout_seconds,
                    args.image_detail, args.reasoning_effort)
            else:
                transport = (OpenAITransport(args.model, image_detail=args.image_detail,
                                            reasoning_effort=args.reasoning_effort)
                             if args.provider == "openai" else AnthropicTransport(args.model))
            coordinator = VLMCoordinator(transport, specs, logger, budget)
        for index in range(args.max_calls):
            if adapter.ended:
                reason = "environment_success" if bool(scalar(adapter.last_info.get("success", False))) else "environment_ended"
                break
            if adapter.steps >= args.max_env_steps:
                reason = "experiment_step_limit"
                break
            if time.monotonic() - started >= args.max_wall_seconds:
                reason = "experiment_wall_clock_limit"
                break
            observation = adapter.observe()
            adapter.save_observation_images()
            if not observation.allowed_calls:
                reason = "no_admissible_skill"
                break
            if args.dry_run:
                admissible = observation.allowed_calls[0]
                spec = specs[admissible.skill]
                request = SkillRequest(f"oracle-{index}", admissible.skill, admissible.target_id,
                    observation.frame_id, (Requirement("benchmark-completion", "benchmark_success"),),
                    min(spec.max_steps, args.max_env_steps - adapter.steps), spec.timeout_seconds)
                logger.emit("oracle_protocol_decision", request=request, vlm=False)
            else:
                request = coordinator.decide(observation, history)
                api_calls = coordinator.calls_reserved
                remaining_steps = args.max_env_steps - adapter.steps
                if request.max_steps > remaining_steps:
                    # Never silently rewrite the VLM's request. Stop this bounded
                    # experiment rather than execute beyond the chosen limit.
                    reason = "request_exceeds_remaining_experiment_steps"
                    logger.emit("experiment_request_rejected", request=request, reason=reason)
                    break
            remaining_wall = args.max_wall_seconds - (time.monotonic() - started)
            if request.timeout_seconds > remaining_wall:
                reason = "request_exceeds_remaining_experiment_wall_budget"
                logger.emit("experiment_request_rejected", request=request, reason=reason)
                break
            decisions += 1
            result = runtime.execute(request)
            history.append({"call_id": request.call_id, "skill": request.skill,
                            "target_id": request.target_id, "requirements": request.requirements,
                            "feedback": result.feedback, "steps": result.steps,
                            "elapsed_seconds": result.elapsed_seconds})
            print(json.dumps({"call_id": request.call_id, "skill": request.skill,
                "status": result.feedback.status.value, "steps": result.steps,
                "reason": result.feedback.reason}), flush=True)
            if result.feedback.status in (SkillStatus.FAILED, SkillStatus.REJECTED):
                reason = result.feedback.reason or "skill_failed"
                break
            # An invocation timeout can be followed by a new bounded decision;
            # a benchmark fail is irreversible here and ends the episode.
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
        if navigation_client is not None:
            try:
                navigation_client.close()
            except Exception as exc:
                logger.emit("navigation_client_close_error", error_type=type(exc).__name__)
        if adapter is not None:
            if coordinator is not None:
                api_calls = coordinator.calls_reserved
            summary = {"benchmark_result": False, "reason": reason, "decisions": decisions,
                       "api_requests": api_calls, "vlm": not args.dry_run,
                       "vlm_feedback_loop_observed": not args.dry_run and len(history) >= 2,
                       "task_success": bool(scalar(adapter.last_info.get("success", False))),
                       "steps": adapter.steps, "wall_seconds": time.monotonic() - started,
                       "skill_results": jsonable(history),
                       "navigation_policy": args.navigation_policy,
                       "navigation_predictions": navigation_skill.total_predictions if navigation_skill else 0,
                       "api_cost_usd": None if not args.dry_run else 0,
                       "api_cost_status": "pending_reconciliation" if not args.dry_run else "no_api_calls"}
            try:
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                logger.emit("experiment_finished", **summary)
            finally:
                adapter.close()
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
