"""Run real MS-HAB skills through the constrained coordinator interface.

--dry-run uses an explicitly labelled oracle dispatcher with real simulation and
policies; it makes zero model API requests but still uses the host GPU. Live VLM
mode requires credentials and an explicit bounded API authorization. This
single-scene diagnostic run does not produce an official benchmark score.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

from bvi import (APIBudget, JsonlLogger, ProtocolError, Requirement, SerialRuntime,
                 SkillRequest, SkillStatus, VLMCoordinator)
from bvi.mshab_adapter import OfficialRLSkill, jsonable, make_mshab_adapter, scalar
from bvi.providers import AnthropicTransport, OpenAITransport


def oracle_skill_timeout(spec_timeout: float, remaining_wall: float) -> float | None:
    """Fit a new oracle request inside the experiment, reserving dispatch overhead."""
    if not math.isfinite(remaining_wall) or remaining_wall <= .5:
        return None
    return min(spec_timeout, remaining_wall - .5)


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
    parser.add_argument('--organizer', action='store_true', help='VLM execution organization with bounded skill slices, grasp-event yields and explicit abort')
    parser.add_argument('--tool-family-interface', action='store_true', help='Versioned GPT family/instruction delivery to both VLA backends')
    parser.add_argument('--benchmark-episode', action='store_true',
                        help='Strict TidyHouse val episode for the resumable LightNav/SAC baseline')
    parser.add_argument('--goal-tools', action='store_true', help='Full goal catalog and request-grounded PPO/SAC')
    parser.add_argument('--progress-feedback', action='store_true',
                        help='Use policy-observation progress; never expose native subtask success to GPT')
    parser.add_argument('--paired-ppo-episode', action='store_true',
                        help='Paired official PPO/SAC fixed-order versus GPT execution study')
    parser.add_argument('--expected-initial-state-sha256')
    parser.add_argument('--organizer-slice-steps', type=int, default=40)
    parser.add_argument('--stop-after-subtasks', type=int, help='Explicit partial-task diagnostic boundary; never full benchmark success')
    parser.add_argument('--inject-closure-fault',action='store_true',help='TEST ONLY: inject six initial Pick closure actions, labeled synthetic fault')
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--transport", choices=("direct", "bridge"), default="direct")
    parser.add_argument("--bridge-timeout-seconds", type=float, default=120)
    parser.add_argument('--bridge-dir', type=Path,
                        help='Persistent shared bridge spool; defaults to OUTPUT/bridge')
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
    parser.add_argument("--record-demonstrations",action='store_true',
                        help='Save synchronized Pick/Place training images, named state, applied action and feedback')
    parser.add_argument("--navigation-policy", choices=("official", "lightnav", "teleport"), default="official")
    parser.add_argument('--navigation-control',choices=('position_tracker','waypoint_velocity'),default='waypoint_velocity')
    parser.add_argument('--manipulation-policy',choices=('official','fetch-pi05'),default='official')
    parser.add_argument('--fetch-pi-url',default='ws://127.0.0.1:8051')
    parser.add_argument('--max-manipulation-predictions',type=int,default=100)
    parser.add_argument('--manipulation-chunk-steps',type=int,default=3)
    parser.add_argument('--manipulation-ensemble-samples',type=int,default=1)
    parser.add_argument('--manipulation-instructions',type=Path,
                        help='Explicit scene-bound manipulation descriptions, requiring expected-plan-uid')
    parser.add_argument('--workspace-camera',action='store_true',
                        help='Add a declared fixed oblique Fetch workspace camera (224 RGB)')
    parser.add_argument('--training-collection',action='store_true',
                        help='Label this run as training data, excluded from evaluation results')
    parser.add_argument('--collect-recovery-after',type=int,
                        help='Training collection only: execute N pi05 steps, then record official SAC recovery')
    parser.add_argument("--navigation-camera",choices=('fetch_head','fetch_nav'),default='fetch_head',
                        help='fetch_nav adds a declared forward-facing robot sensor; original RL sensors stay unchanged')
    parser.add_argument("--lightnav-url", default="ws://127.0.0.1:8050")
    parser.add_argument("--navigation-instructions", type=Path,
                        help="JSON mapping subtask indices to explicit visual language goals")
    parser.add_argument('--navigation-recovery-instructions',type=Path,
                        help='Optional rotation-only language goals when stopped close but misoriented')
    parser.add_argument("--expected-plan-uid", help="Require the exact first subtask UID before LightNav")
    parser.add_argument("--max-navigation-predictions", type=int, default=40)
    parser.add_argument("--lightnav-subtasks", type=int, nargs='+',
                        help="Explicit diagnostic routing: use LightNav only at these indices; official elsewhere")
    parser.add_argument("--video-debug-overlay", action="store_true",
                        help="Burn verbose simulator statistics into diagnostic video")
    parser.add_argument("--show-goal-markers", action="store_true",
                        help="Show benchmark debug goals in human-render video")
    args = parser.parse_args()
    if args.goal_tools and (not (args.paired_ppo_episode or args.benchmark_episode)
                            or not args.organizer or args.dry_run):
        parser.error('Goal tools require a live paired or benchmark organizer mode')
    if args.progress_feedback and not args.goal_tools:
        parser.error('Progress feedback requires goal-grounded tools')
    if args.paired_ppo_episode and (args.benchmark_episode or args.tool_family_interface
            or args.navigation_policy not in {'official','teleport'} or args.manipulation_policy != 'official'
            or args.policy_type != 'rl_per_obj' or not args.expected_plan_uid
            or args.max_env_steps != 7000 or args.max_wall_seconds != 900
            or args.skill_wall_seconds != 180 or args.organizer_slice_steps != 40
            or args.training_collection or args.record_demonstrations or args.inject_closure_fault
            or args.collect_recovery_after is not None or args.stop_after_subtasks is not None
            or (args.navigation_policy == 'teleport' and not args.dry_run)
            or (not args.dry_run and (not args.organizer or args.model != 'gpt-5.6-luna'
                or args.transport != 'bridge' or args.max_calls != 40 or args.max_output_tokens > 2048))):
        parser.error('Paired PPO study requires frozen official PPO/SAC and bounded fixed/GPT execution')
    if args.tool_family_interface and (args.dry_run or not args.organizer
            or args.navigation_policy != 'lightnav' or args.manipulation_policy not in {'official','fetch-pi05'}
            or args.collect_recovery_after is not None or args.lightnav_subtasks is not None):
        parser.error('Tool interface requires live organizer, LightNav, an approved manipulation backend, and no teacher takeover')
    if args.benchmark_episode and (not args.goal_tools or not args.tool_family_interface
            or args.provider != 'openai'
            or args.model != 'gpt-5.6-luna' or args.transport != 'bridge'
            or args.navigation_policy != 'lightnav' or args.manipulation_policy != 'official'
            or args.policy_type != 'rl_per_obj' or args.navigation_camera != 'fetch_nav'
            or not args.workspace_camera or args.organizer_slice_steps != 40
            or args.max_calls != 40 or args.max_env_steps != 7000
            or args.max_wall_seconds != 900 or args.skill_wall_seconds != 180
            or args.max_output_tokens > 2048 or args.stop_after_subtasks is not None
            or args.training_collection or args.record_demonstrations
            or args.inject_closure_fault or args.collect_recovery_after is not None
            or args.lightnav_subtasks is not None):
        parser.error('Benchmark episode requires goal-grounded GPT-5.6 Luna + LightNav + per-object SAC')
    if args.organizer and (args.dry_run or not 1<=args.organizer_slice_steps<=500):
        parser.error('Organizer requires live VLM and slice steps1..500')
    if args.inject_closure_fault and (not args.organizer or args.manipulation_policy!='official'):
        parser.error('Synthetic closure fault requires organizer and official manipulation')
    if args.stop_after_subtasks is not None and args.stop_after_subtasks<1:
        parser.error('Partial-task boundary must be positive')
    if args.collect_recovery_after is not None and (not args.dry_run or not args.record_demonstrations
            or args.manipulation_policy!='fetch-pi05' or not 0<=args.collect_recovery_after<=50):
        parser.error('Recovery collection requires --dry-run --record-demonstrations --manipulation-policy fetch-pi05 and prefix0..50')
    navigation_instructions = {}
    manipulation_instructions=None
    if args.manipulation_instructions:
        if args.manipulation_policy!='fetch-pi05' or not args.expected_plan_uid:
            parser.error('Manipulation instruction overrides require Fetch pi05 and expected-plan-uid')
        if args.record_demonstrations:
            parser.error('Manipulation instruction overrides are inference-only; the teacher exporter uses task-plan descriptions')
        manipulation_instructions=json.loads(args.manipulation_instructions.read_text(encoding='utf-8'))
        if not isinstance(manipulation_instructions,dict) or not manipulation_instructions or not all(
                isinstance(k,str) and k.isdigit() and isinstance(v,str) and v.strip() for k,v in manipulation_instructions.items()):
            parser.error('Manipulation instructions must map indices to nonempty strings')
    recovery_instructions={}
    if args.navigation_recovery_instructions:
        recovery_instructions=json.loads(args.navigation_recovery_instructions.read_text(encoding='utf-8'))
        if not isinstance(recovery_instructions,dict) or not all(isinstance(k,str) and k.isdigit() and isinstance(v,str) and v.strip() for k,v in recovery_instructions.items()):
            parser.error('Recovery instructions must map indices to nonempty strings')
    if args.navigation_policy == "lightnav":
        if (args.navigation_instructions is None and not args.tool_family_interface) or not args.expected_plan_uid or args.max_navigation_predictions < 1:
            parser.error("LightNav requires instructions, expected-plan-uid and a positive prediction cap")
        navigation_instructions = (json.loads(args.navigation_instructions.read_text(encoding="utf-8"))
                                   if args.navigation_instructions else {})
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
    if args.paired_ppo_episode or args.benchmark_episode:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    plan_path = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json"
    if not plan_path.is_file():
        parser.error(f"Missing task plans: {plan_path}")
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_base = checkpoint_root / 'rl/tidy_house'
    required_targets = {'navigate': ['all']}
    if args.policy_type == 'rl_all_obj':
        required_targets.update(pick=['all'], place=['all'])
    else:
        from mshab.evaluate import POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS
        mapping = POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS['rl']['tidy_house']
        required_targets.update(
            pick=[target for target in mapping['pick'] if target != 'all'],
            place=[target for target in mapping['place'] if target != 'all'])
    for name, targets in required_targets.items():
        if not targets:
            parser.error(f'No registered {name} targets for {args.policy_type}')
        for target in targets:
            directory = checkpoint_base / name / target
            for filename in ('policy.pt', 'config.yml'):
                if not (directory / filename).is_file():
                    parser.error(f'Missing {name}/{target}/{filename} under {checkpoint_root}')

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(args.output / "events.jsonl", args.output.name)
    env_id='SequentialTask-v0'
    if args.navigation_camera == 'fetch_nav':
        import bvi.nav_camera_env
        env_id='BVISequentialNavCamera-v0'
    if args.workspace_camera:
        import bvi.nav_camera_env
        env_id='BVISequentialWorkspaceCamera-v0'
    env_kwargs = {"require_build_configs_repeated_equally_across_envs": False,
                  "add_event_tracker_info": True,
                  "human_render_camera_configs": {"width": 512, "height": 512},
                  "task_cfgs": {"navigate": {"ignore_arm_checkers": True}}}
    from mani_skill.envs.sapien_env import BaseEnv
    if "invisible_goals_in_human_render" in inspect.signature(BaseEnv.__init__).parameters:
        env_kwargs["invisible_goals_in_human_render"] = not args.show_goal_markers
    cfg = EnvConfig(env_id=env_id, num_envs=1, max_episode_steps=7000,
        task_plan_fp=str(plan_path), obs_mode="rgbd", render_mode="rgb_array",
        record_video=not args.no_video, info_on_video=args.video_debug_overlay, continuous_task=True,
        frame_stack=3, stationary_base=False, stationary_torso=False, stationary_head=True,
        env_kwargs=env_kwargs)
    # Snapshot before official make_env expands task_plans inside env_kwargs.
    metadata = {"stage": "sac_interface_baseline_episode" if args.benchmark_episode else "G4_adapter_diagnostic",
                "benchmark_result": False, "benchmark_episode": args.benchmark_episode,
                "vlm": not args.dry_run,
                "dispatcher": "oracle_protocol_dry_run" if args.dry_run else "constrained_vlm",
                "coverage": "one validation scene / one sampled plan",
                "completion_source": ("policy_observation_progress" if args.progress_feedback
                                      else "oracle_benchmark"),
                "targets_source": "oracle_task_plan",
                "config": jsonable(cfg), "seed": args.seed, "policy_type": args.policy_type,
                "provider": None if args.dry_run else args.provider,
                "model": None if args.dry_run else args.model, "api_budget": jsonable(budget),
                "transport": args.transport,
                "image_detail": args.image_detail, "reasoning_effort": args.reasoning_effort,
                "max_env_steps": args.max_env_steps, "max_wall_seconds": args.max_wall_seconds}
    metadata.update(navigation_policy=args.navigation_policy,
                    navigation_control=args.navigation_control,
                    manipulation_policy=args.manipulation_policy,
                    manipulation_chunk_steps=args.manipulation_chunk_steps,
                    navigation_camera=args.navigation_camera,
                    lightnav_subtasks=args.lightnav_subtasks,
                    navigation_instructions=navigation_instructions,
                    navigation_recovery_instructions=recovery_instructions,
                    max_navigation_predictions=args.max_navigation_predictions)
    metadata['mixed_teacher_collection']=args.collect_recovery_after is not None
    metadata['tool_family_interface'] = {
        'enabled': args.tool_family_interface,
        'version': 'mshab-tool-family/1' if args.tool_family_interface else None,
        'families': ['navigate', 'pick', 'place'],
        'residual_family_adapters': False,
        'learned_progress_available': False,
        'feedback': ('policy_observation_progress' if args.progress_feedback
                     else 'benchmark_completion_and_disclosed_grasp_rules'),
        'scope': 'coarse_skills_heterogeneous_backends',
        'instruction_consumption': {
            'navigate': 'LightNav consumes the scene-grounded instruction',
            'pick': 'official SAC is selected by the native target and does not consume language',
            'place': 'official SAC is selected by the native target and does not consume language',
        } if args.tool_family_interface else None,
    }
    metadata['training_collection']=args.training_collection
    metadata['oracle_timeout_policy']='remaining_experiment_budget_minus_0.5s' if args.dry_run else None
    metadata['stop_after_subtasks']=args.stop_after_subtasks
    metadata['synthetic_closure_fault']=args.inject_closure_fault
    metadata['teacher_takeover_after']=args.collect_recovery_after
    metadata['manipulation_ensemble_samples']=args.manipulation_ensemble_samples
    metadata['manipulation_instructions']=manipulation_instructions
    if args.collect_recovery_after is not None:
        metadata['manipulation_policy']='fetch-pi05_then_sac_teacher'
    source_root=Path(__file__).resolve().parents[1]
    source_files=[Path(__file__).resolve(),*sorted((source_root/'src/bvi').glob('*.py'))]
    metadata['runtime_source_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in source_files}
    import mshab.envs.sequential_task as mshab_runtime
    runtime_path = Path(mshab_runtime.__file__).resolve()
    metadata['mshab_runtime'] = {'path': str(runtime_path),
                               'sha256': hashlib.sha256(runtime_path.read_bytes()).hexdigest()}
    (args.output / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    adapter = None
    navigation_client = None
    navigation_skill = None
    manipulation_client = None
    manipulation_skills = []
    started = time.monotonic()
    history = []
    decisions = 0
    api_calls = 0
    reason = "not_started"
    coordinator = None
    organizer = None
    try:
        adapter = make_mshab_adapter(cfg, logger, args.output, seed=args.seed)
        if args.expected_plan_uid and adapter.original_plan.subtasks[0].uid != args.expected_plan_uid:
            raise ProtocolError('Sampled plan does not match expected-plan-uid')
        if args.paired_ppo_episode or args.benchmark_episode:
            initial_state = jsonable(adapter.uenv.get_state_dict())
            initial_hash = hashlib.sha256(json.dumps(initial_state, sort_keys=True,
                allow_nan=False, separators=(',', ':')).encode()).hexdigest()
            initial = {'seed': args.seed, 'plan_uid': adapter.original_plan.subtasks[0].uid,
                'state_sha256': initial_hash, 'state': initial_state,
                'task_plan': jsonable(adapter.original_plan),
                'policy_state_shape': list(adapter.observe().policy['state'].shape),
                'native_horizons': {name: int(adapter.uenv.task_cfgs[name].horizon)
                                    for name in ('navigate', 'pick', 'place')}}
            (args.output / 'initial-state.json').write_text(json.dumps(initial), encoding='utf-8')
            if args.expected_initial_state_sha256 and initial_hash != args.expected_initial_state_sha256:
                raise ProtocolError('Paired initial simulator state hash mismatch')
            if initial['policy_state_shape'] != [1, 42] or initial['native_horizons'] != dict(navigate=500,pick=200,place=200):
                raise ProtocolError('Paired official runtime state/horizon mismatch')
        adapter.record_demonstrations = args.record_demonstrations
        specs = adapter.skill_specs(args.skill_wall_seconds)
        skills = {name: OfficialRLSkill(name, adapter, checkpoint_root, args.policy_type) for name in specs}
        if args.goal_tools:
            from bvi.goal_tools import GoalToolAdapter, GoalRLSkill, ProgressGoalRLSkill
            adapter = GoalToolAdapter(adapter)
            skill_class = ProgressGoalRLSkill if args.progress_feedback else GoalRLSkill
            skills = {name: skill_class(name, adapter, checkpoint_root, args.policy_type) for name in specs}
            metadata['planning_scope'] = 'whole_goals_request_grounded_tools_native_order_constraint'
            metadata['feedback_contract'] = ({
                'type': 'continuous_progress',
                'source': 'heuristic_policy_observation_relative_geometry_v1',
                'learned': False,
                'native_subtask_success_exposed': False,
                'native_evaluator_role': 'background_scoring_and_terminal_failure_only',
            } if args.progress_feedback else {
                'type': 'native_oracle_subtask_completion',
                'native_subtask_success_exposed': True,
            })
        if args.manipulation_policy=='fetch-pi05':
            from bvi.fetch_pi_skill import FetchPiSkill
            from bvi.vla_clients import OpenPiClient
            manipulation_client=OpenPiClient.connect(args.fetch_pi_url,timeout=120)
            for name in ('pick','place'):
                skill=FetchPiSkill(name,adapter,manipulation_client,args.max_manipulation_predictions,
                                   chunk_steps=args.manipulation_chunk_steps,
                                   ensemble_samples=args.manipulation_ensemble_samples,
                                   instructions=manipulation_instructions)
                if args.collect_recovery_after is not None:
                    from bvi.recovery_collection import RecoveryCollectionSkill
                    skills[name]=RecoveryCollectionSkill(skill,skills[name],adapter,args.collect_recovery_after)
                else:
                    skills[name]=skill
                manipulation_skills.append(skill)
        if args.navigation_policy == "lightnav":
            from bvi.lightnav_skill import LightNavSkill
            from bvi.vla_clients import LightNavClient
            navigation_client = LightNavClient.connect(args.lightnav_url)
            navigation_skill = LightNavSkill(adapter, navigation_client, navigation_instructions,
                                             args.max_navigation_predictions,camera=args.navigation_camera,
                                             control_mode=args.navigation_control,
                                             recovery_instructions=recovery_instructions,max_stop_replans=2)
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
        elif args.navigation_policy == "teleport":
            from bvi.teleport_skill import StandardizedTeleportSkill
            skills["navigate"] = StandardizedTeleportSkill(
                adapter, checkpoint_root, args.policy_type)
            metadata['teleport'] = {
                'implementation': 'paper_4729821_port_current_runtime',
                'upstream_commit': '4729821db3fc94a2470cfd625e6f8ab439f01478',
                'scope': 'one_env_tidy_house',
            }
            (args.output / 'run-metadata.json').write_text(
                json.dumps(metadata, indent=2), encoding='utf-8')
        if args.organizer:
            if args.dry_run:
                raise ProtocolError('Organizer requires actual VLM mode')
            from bvi.organizer import OrganizerView, GraspMonitor
            organizer = OrganizerView(adapter, specs, args.organizer_slice_steps, args.tool_family_interface)
            if 'pick' in skills and not args.goal_tools:
                if args.inject_closure_fault:
                    from bvi.organizer import InjectedClosureFault
                    skills['pick']=InjectedClosureFault(skills['pick'],logger)
                skills['pick'] = GraspMonitor(skills['pick'])
            metadata['organizer'] = {'enabled': True, 'scope': 'goal_decomposition' if args.goal_tools else 'benchmark_constrained_execution',
                'slice_steps': args.organizer_slice_steps,
                'grasp_source': ('policy_observation_progress' if args.progress_feedback
                                 else 'oracle_benchmark'),
                'reposition_skill': False, 'abort_is_success': False}
            (args.output / 'run-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        runtime = SerialRuntime(adapter, skills, specs, logger)
        if not args.dry_run:
            if args.transport == "bridge":
                from bvi.bridge import FileBridgeTransport
                bridge_dir = args.bridge_dir.resolve() if args.bridge_dir else args.output / 'bridge'
                transport = FileBridgeTransport(bridge_dir, args.provider, args.model,
                    args.authorization_id, args.bridge_timeout_seconds,
                    args.image_detail, args.reasoning_effort)
            else:
                transport = (OpenAITransport(args.model, image_detail=args.image_detail,
                                            reasoning_effort=args.reasoning_effort)
                             if args.provider == "openai" else AnthropicTransport(args.model))
            coordinator = VLMCoordinator(transport, organizer.specs if organizer else specs, logger, budget,
                                         tool_interface=args.tool_family_interface)
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
            observation = organizer.observe() if organizer else adapter.observe()
            if args.stop_after_subtasks is not None and observation.metadata.get('subtask_index',0)>=args.stop_after_subtasks:
                reason='declared_partial_task_boundary'
                break
            adapter.save_observation_images()
            if not observation.allowed_calls:
                reason = "no_admissible_skill"
                break
            if args.dry_run:
                admissible = observation.allowed_calls[0]
                spec = specs[admissible.skill]
                timeout = oracle_skill_timeout(spec.timeout_seconds,
                    args.max_wall_seconds - (time.monotonic() - started))
                if timeout is None:
                    reason = "experiment_wall_clock_limit"
                    break
                request = SkillRequest(f"oracle-{index}", admissible.skill, admissible.target_id,
                    observation.frame_id, (Requirement("benchmark-completion", "benchmark_success"),),
                    min(spec.max_steps, args.max_env_steps - adapter.steps,
                        40 if args.paired_ppo_episode else spec.max_steps), timeout)
                logger.emit("oracle_protocol_decision", request=request, vlm=False)
            else:
                request = coordinator.decide(observation, history)
                api_calls = coordinator.calls_reserved
                if organizer and request.skill == 'abort_task':
                    logger.emit('organizer_abort', request=request, task_success=False)
                    decisions += 1
                    reason = 'organizer_aborted_task'
                    break
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
            if organizer:
                organizer.note_result(result)
            history.append({"call_id": request.call_id, "skill": request.skill,
                            "target_id": request.target_id, "requirements": request.requirements,
                            "feedback": result.feedback, "steps": result.steps,
                            "elapsed_seconds": result.elapsed_seconds})
            print(json.dumps({"call_id": request.call_id, "skill": request.skill,
                "status": result.feedback.status.value, "steps": result.steps,
                "reason": result.feedback.reason}), flush=True)
            if args.goal_tools and result.feedback.status is SkillStatus.REJECTED:
                continue
            if result.feedback.status in (SkillStatus.FAILED, SkillStatus.REJECTED):
                reason = result.feedback.reason or "skill_failed"
                break
            if args.stop_after_subtasks is not None and result.observation.metadata.get('subtask_index',0)>=args.stop_after_subtasks:
                reason='declared_partial_task_boundary'
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
        if manipulation_client is not None:
            try:
                manipulation_client.close()
            except Exception as exc:
                logger.emit('manipulation_client_close_error',error_type=type(exc).__name__)
        if navigation_client is not None:
            try:
                navigation_client.close()
            except Exception as exc:
                logger.emit("navigation_client_close_error", error_type=type(exc).__name__)
        if adapter is not None:
            if coordinator is not None:
                api_calls = coordinator.calls_reserved
            completed_skills=[h['skill'] for h in history
                              if h['feedback'].status is SkillStatus.SUCCEEDED]
            completed_objects = sum(1 for item in history
                                    if item['skill'] == 'place'
                                    and item['feedback'].status is SkillStatus.SUCCEEDED)
            planned_objects = sum(1 for item in adapter.original_plan.subtasks if item.type == 'place')
            if args.paired_ppo_episode or args.benchmark_episode:
                native_index = int(scalar(adapter.uenv.subtask_pointer))
                completed_objects = sum(s.type == 'place' for s in adapter.original_plan.subtasks[:native_index])
            summary = {"benchmark_result": False, "benchmark_episode": args.benchmark_episode or args.paired_ppo_episode,
                       "evaluation_eligible": args.benchmark_episode or args.paired_ppo_episode,
                       "paired_ppo_episode": args.paired_ppo_episode,
                       "reason": reason, "decisions": decisions,
                       "first_object_chain_success": completed_objects > 0,
                       "api_requests": api_calls, "vlm": not args.dry_run,
                       "vlm_feedback_loop_observed": not args.dry_run and len(history) >= 2,
                       "task_success": bool(scalar(adapter.last_info.get("success", False))),
                       "completed_objects": min(completed_objects, planned_objects),
                       "planned_objects": planned_objects,
                       "steps": adapter.steps, "wall_seconds": time.monotonic() - started,
                       "skill_results": jsonable(history),
                       "navigation_policy": args.navigation_policy,
                       "manipulation_policy":metadata['manipulation_policy'],
                       "mixed_teacher_collection":metadata['mixed_teacher_collection'],
                       "manipulation_predictions":sum(s.total_predictions for s in manipulation_skills),
                       "manipulation_ensemble_samples":args.manipulation_ensemble_samples,
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
