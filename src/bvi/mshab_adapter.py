"""Single-environment MS-HAB integration using the pinned upstream wrappers.

The low-level policy and completion/target metadata use benchmark privileges.
VLM images are captured from the *same* raw reset/step observations before the
official depth wrapper discards RGB. No additional get_obs/get_info/evaluate
calls are made: SequentialTask.evaluate mutates the task state.

Policy constructors and preprocessing follow mshab/evaluate.py at
e9ff3d23496d38e4431c8d913e147ffa007f7f72. This is a diagnostic coordinator
runner, not a replacement for full-horizon official benchmark evaluation.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from .logging import JsonlLogger
from .protocol import (ActionBounds, AllowedCall, ImageFrame, Observation,
                       ProtocolError, RequirementResult, RequirementState,
                       SkillFeedback, SkillRequest, SkillSpec, SkillStatus,
                       Target, Transition)


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return str(value)


def scalar(value: Any) -> Any:
    value = jsonable(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, dict)):
        raise ProtocolError("This adapter supports exactly one environment")
    return value


def describe_target(plan: Any, index: int) -> Target:
    """Describe original semantic IDs, never merged obj_0 actor placeholders."""
    subtask = plan.subtasks[index]
    if subtask.type == "navigate" and index + 1 < len(plan.subtasks):
        next_subtask = plan.subtasks[index + 1]
        obj = getattr(next_subtask, "obj_id", None)
        if next_subtask.type == "place":
            description = f"Approach the task-plan placement region for held object {obj}."
        else:
            description = f"Approach {obj} to prepare for {next_subtask.type}."
    elif subtask.type == "place":
        description = f"Place object {subtask.obj_id} at its task-plan target region."
    elif subtask.type == "pick":
        description = f"Pick and stably hold object {subtask.obj_id}."
    elif subtask.type in ("open", "close"):
        description = f"{subtask.type.capitalize()} articulation {subtask.articulation_id}."
    else:
        description = "Navigate to the current task-plan destination."
    return Target(f"subtask-{index}-{subtask.uid}", description, "oracle_task_plan")


def make_mshab_adapter(env_cfg: Any, logger: JsonlLogger, output: str | Path,
                      task: str = "tidy_house", seed: int = 0) -> "MSHABAdapter":
    """Build through official make_env while retaining raw robot-camera RGB."""
    import gymnasium as gym
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mshab.envs.make import make_env

    if env_cfg.num_envs != 1 or env_cfg.obs_mode != "rgbd":
        raise ProtocolError("Coordinator needs num_envs=1 and obs_mode=rgbd")
    if env_cfg.stationary_base or env_cfg.stationary_torso:
        raise ProtocolError("Official whole-body skills require unlocked base and torso")
    if env_cfg.frame_stack != 3 or not env_cfg.cat_state or env_cfg.cat_pixels:
        raise ProtocolError("This RL adapter expects official separate depth cameras, stack=3")
    capture_ref: list[Any] = []

    class CaptureRawObservation(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.images: dict[str, Any] = {}
            self.info: dict[str, Any] = {}
            capture_ref.append(self)

        def capture(self, raw, info):
            self.images = {camera: raw["sensor_data"][camera]["rgb"].detach().clone()
                           for camera in ("fetch_head", "fetch_hand", "fetch_nav", "fetch_workspace")
                           if camera in raw['sensor_data']}
            # Info can hold references to mutable GPU state, e.g. subtask_pointer.
            self.info = jsonable(info)

        def reset(self, *args, **kwargs):
            raw, info = self.env.reset(*args, **kwargs)
            self.capture(raw, info)
            return raw, info

        def step(self, action):
            raw, reward, terminated, truncated, info = self.env.step(action)
            self.capture(raw, info)
            return raw, reward, terminated, truncated, info

    output = Path(output)
    env = make_env(env_cfg, video_path=output / "videos", wrappers=[CaptureRawObservation])
    try:
        # Pinned make_env returns statistics(ManiSkillVectorEnv(...)). Prevent a
        # terminal frame from being replaced by the next episode's reset frame.
        vector = env.env
        if not isinstance(vector, ManiSkillVectorEnv):
            raise ProtocolError("Pinned make_env wrapper layout changed")
        vector.auto_reset = False
        adapter = MSHABAdapter(env, capture_ref[0], logger, output, task)
        adapter.reset(seed)
        return adapter
    except Exception:
        env.close()
        raise


class MSHABAdapter:
    def __init__(self, env: Any, capture: Any, logger: JsonlLogger,
                 output: str | Path, task: str):
        self.env, self.capture, self.logger = env, capture, logger
        self.output, self.task = Path(output), task
        self.uenv = env.unwrapped
        if self.uenv.num_envs != 1:
            raise ProtocolError("MSHABAdapter supports exactly one environment")
        space = self.uenv.single_action_space
        if space.shape != (13,):
            raise ProtocolError("Expected pinned Fetch 13-dimensional action space")
        self.action_bounds = ActionBounds(tuple(float(x) for x in space.low),
                                          tuple(float(x) for x in space.high))
        controllers = self.uenv.agent.controller.controllers
        if not all(controller._normalize_action for controller in controllers.values()):
            raise ProtocolError("Explicit clipping requires the pinned normalized Fetch controllers")
        self.steps = 0
        self.episode_id = "unreset"
        self._observation: Observation | None = None
        self.last_info: dict[str, Any] = {}
        self.ended = False
        self.success_once = False
        self.record_demonstrations = False

    def reset(self, seed: int) -> Observation:
        from mshab.utils.array import to_tensor
        policy, _ = self.env.reset(seed=seed, options={"reconfigure": True})
        self.steps, self.ended, self.success_once = 0, False, False
        self.episode_id = f"seed-{seed}"
        bci = int(scalar(self.uenv.build_config_idxs))
        tpi = int(scalar(self.uenv.task_plan_idxs))
        self.original_plan = self.uenv.build_config_idx_to_task_plans[bci][tpi]
        if len(self.original_plan.subtasks) != len(self.uenv.task_plan):
            raise ProtocolError("Original/merged task plans have different lengths")
        self.last_info = dict(self.capture.info)
        policy = to_tensor(policy, device=self.uenv.device, dtype="float")
        self._observation = self._snapshot(policy)
        self.logger.emit("mshab_reset", seed=seed, build_config_index=bci,
                         task_plan_index=tpi, task_plan=jsonable(self.original_plan),
                         info=self.last_info, auto_reset=False,
                         targets_source="oracle_task_plan", completion_source="oracle_benchmark")
        return self.observe()

    def _snapshot(self, policy: Any) -> Observation:
        index = int(scalar(self.uenv.subtask_pointer))
        complete = index >= len(self.uenv.task_plan)
        if complete or self.ended:
            targets, allowed = (), ()
        else:
            target = describe_target(self.original_plan, index)
            skill = self.uenv.task_plan[index].type
            if skill != self.original_plan.subtasks[index].type:
                raise ProtocolError("Original/merged subtask types disagree")
            targets, allowed = (target,), (AllowedCall(skill, target.id),)
        return Observation(f"{self.episode_id}-step-{self.steps}", self.steps, policy=policy,
            targets=targets, allowed_calls=allowed,
            task=f"Execute the {self.task} mobile manipulation plan using admissible skills.",
            metadata={"subtask_index": index, "complete": complete,
                      "benchmark_info": self.last_info, "ended": self.ended})

    def _encode_images(self) -> tuple[ImageFrame, ...]:
        from PIL import Image
        images = []
        for camera, pixels in self.capture.images.items():
            array = pixels.detach().cpu().numpy()
            if array.shape[0] != 1 or array.ndim != 4 or array.shape[-1] != 3:
                raise ProtocolError("Expected one NHWC RGB camera image")
            buffer = io.BytesIO()
            Image.fromarray(array[0]).save(buffer, format="PNG")
            images.append(ImageFrame(camera, buffer.getvalue()))
        return tuple(images)

    def observe(self) -> Observation:
        if self._observation is None:
            raise ProtocolError("Reset the adapter before observing")
        if not self._observation.images:
            self._observation = replace(self._observation, images=self._encode_images())
        return self._observation

    def save_observation_images(self) -> list[str]:
        observation = self.observe()
        paths = []
        for image in observation.images:
            path = self.output / "frames" / f"{observation.frame_id}-{image.camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.data)
            paths.append(str(path))
        self.logger.emit("observation_images", frame_id=observation.frame_id,
                         paths=paths, cameras=[image.camera for image in observation.images])
        return paths

    def step(self, action: tuple[float, ...]) -> Transition:
        import torch
        from mshab.utils.array import to_tensor
        if self.ended:
            raise ProtocolError("Episode ended; this diagnostic runner does not auto-reset")
        action = self.action_bounds.validate(action)
        before = int(scalar(self.uenv.subtask_pointer))
        demonstration = None
        if self.record_demonstrations and self.original_plan.subtasks[before].type in ('pick','place'):
            observation = self.observe()
            paths = self.save_observation_images()
            demonstration = dict(frame_id=observation.frame_id,
                demonstration_source=getattr(self,'demonstration_source','unspecified'),
                subtask_index=before, skill=self.original_plan.subtasks[before].type,
                target_description=describe_target(self.original_plan,before).description,
                qpos=jsonable(self.uenv.agent.robot.qpos),
                qvel=jsonable(self.uenv.agent.robot.qvel),
                joint_names=[j.name for j in self.uenv.agent.robot.active_joints],
                base_pose=jsonable(self.uenv.agent.base_link.pose.raw_pose),
                tcp_pose=jsonable(self.uenv.agent.tcp_pose.raw_pose),
                action=action, images=paths,control_hz=20,
                action_convention='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity')
        tensor = torch.tensor([action], dtype=torch.float32, device=self.uenv.device)
        policy, reward, terminated, truncated, _ = self.env.step(tensor)
        self.steps += 1
        self.last_info = dict(self.capture.info)
        failed = bool(scalar(self.last_info.get("fail", False)))
        success = bool(scalar(self.last_info.get("success", False)))
        self.success_once |= success
        term, trunc = bool(scalar(terminated)), bool(scalar(truncated))
        self.ended = term or trunc or failed or success
        after = int(scalar(self.uenv.subtask_pointer))
        policy = to_tensor(policy, device=self.uenv.device, dtype="float")
        self._observation = self._snapshot(policy)
        info = {**self.last_info, "adapter_subtask_before": before,
                "adapter_subtask_after": after}
        self.logger.emit("mshab_step", frame_id=self._observation.frame_id,
                         subtask_before=before, subtask_after=after, info=info,
                         controller_action=jsonable(tensor), terminated=term, truncated=trunc)
        if demonstration is not None:
            demonstration['requested_action']=demonstration['action']
            # Official wrappers may zero stationary-head controls in-place.
            demonstration['action']=jsonable(tensor)[0]
            self.logger.emit('demonstration_step',**demonstration,
                             next_frame_id=self._observation.frame_id,feedback=info)
        return Transition(self._observation, float(scalar(reward)), term, trunc, info)

    def skill_specs(self, wall_timeout_seconds: float = 180.0) -> dict[str, SkillSpec]:
        return {name: SkillSpec(name, max_steps=int(self.uenv.task_cfgs[name].horizon),
                               timeout_seconds=wall_timeout_seconds)
                for name in {subtask.type for subtask in self.uenv.task_plan}}

    def close(self) -> None:
        self.env.close()


def load_checkpoint_config(path: Path, config_api: Any = None, _parents: tuple[Path, ...] = ()):
    """Load checkpoint inheritance without upstream parse_cfg's process CLI merge."""
    if config_api is None:
        from omegaconf import OmegaConf
        config_api = OmegaConf
    path = Path(path).resolve()
    if path in _parents:
        raise ProtocolError("Checkpoint base_config contains a cycle")
    current = config_api.load(path)
    parents = current.get("base_config", [])
    if isinstance(parents, str):
        parents = [parents]
    inherited = config_api.create()
    for parent in parents:
        if not isinstance(parent, str):
            raise ProtocolError("Checkpoint base_config paths must be strings")
        base = load_checkpoint_config(path.parent / parent, config_api, (*_parents, path))
        inherited = config_api.merge(inherited, base)
    return config_api.merge(inherited, current)


def load_rl_policy(config_path: Path, checkpoint_path: Path, adapter: MSHABAdapter):
    """Use upstream constructors and their native depth normalization unchanged."""
    import torch
    from gymnasium import spaces
    from mshab.agents.ppo import Agent as PPOAgent
    from mshab.agents.sac import Agent as SACAgent
    algo = load_checkpoint_config(config_path).algo
    sample = adapter.observe().policy
    obs_space = adapter.uenv.single_observation_space
    act_space = adapter.uenv.single_action_space
    device = adapter.uenv.device
    if algo.name == "ppo":
        policy = PPOAgent(sample, act_space.shape)
        action_fn = lambda obs: policy.get_action(obs, deterministic=True)
    elif algo.name == "sac":
        model_pixels = {}
        for key, space in obs_space["pixels"].items():
            shape, low, high = space.shape, space.low, space.high
            if len(shape) == 4:
                shape = (shape[0] * shape[1], shape[-2], shape[-1])
                low = low.reshape((-1, *low.shape[-2:]))
                high = high.reshape((-1, *high.shape[-2:]))
            model_pixels[key] = spaces.Box(low, high, shape, space.dtype)
        policy = SACAgent(spaces.Dict(model_pixels), obs_space["state"].shape,
            act_space.shape, actor_hidden_dims=list(algo.actor_hidden_dims),
            critic_hidden_dims=list(algo.critic_hidden_dims),
            critic_layer_norm=algo.critic_layer_norm, critic_dropout=algo.critic_dropout,
            encoder_pixels_feature_dim=algo.encoder_pixels_feature_dim,
            encoder_state_feature_dim=algo.encoder_state_feature_dim,
            cnn_features=list(algo.cnn_features), cnn_filters=list(algo.cnn_filters),
            cnn_strides=list(algo.cnn_strides), cnn_padding=algo.cnn_padding,
            log_std_min=algo.actor_log_std_min, log_std_max=algo.actor_log_std_max,
            device=device)
        action_fn = lambda obs: policy.actor(obs["pixels"], obs["state"],
                                             compute_pi=False, compute_log_pi=False)[0]
    else:
        raise ProtocolError(f"Only official PPO/SAC RL checkpoints supported; found {algo.name}")
    policy.eval()
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device)["agent"])
    policy.to(device)
    with torch.no_grad():
        action_fn(sample)
    adapter.logger.emit("policy_loaded", algorithm=algo.name,
                        config=str(config_path), checkpoint=str(checkpoint_path),
                        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    return action_fn


class OfficialRLSkill:
    def __init__(self, name: str, adapter: MSHABAdapter, checkpoint_root: str | Path,
                 policy_type: str = "rl_all_obj"):
        if policy_type not in ("rl_all_obj", "rl_per_obj"):
            raise ProtocolError("Only official RL policy variants are supported")
        self.name, self.adapter = name, adapter
        self.root, self.policy_type = Path(checkpoint_root), policy_type
        self._policies: dict[str, Any] = {}
        self._active_policy = None
        self._start_index: int | None = None
        self._call_id: str | None = None

    def start(self, request: SkillRequest, observation: Observation) -> None:
        from mshab.evaluate import POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS
        index = (self.adapter.resolve_request_index(request)
                 if hasattr(self.adapter, 'resolve_request_index')
                 else int(observation.metadata["subtask_index"]))
        if self.adapter.uenv.task_plan[index].type != self.name or request.skill != self.name:
            raise ProtocolError("Skill does not match the current benchmark subtask")
        self._start_index = index
        self._call_id = request.call_id
        target = "all"
        if self.name != "navigate" and (self.policy_type == "rl_per_obj"
                                         or self.name in ("open", "close")):
            collection = (self.adapter.uenv.subtask_articulations if self.name in ("open", "close")
                          else self.adapter.uenv.subtask_objs)
            actor_name = collection[index]._objs[0].name
            candidates = POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][self.adapter.task][self.name]
            matches = [name for name in candidates if name != "all" and name in actor_name]
            if len(matches) != 1:
                raise ProtocolError("Object-specific checkpoint target is missing or ambiguous")
            target = matches[0]
        if target not in self._policies:
            directory = self.root / "rl" / self.adapter.task / self.name / target
            self._policies[target] = load_rl_policy(directory / "config.yml", directory / "policy.pt",
                                                   self.adapter)
        self._active_policy = self._policies[target]

    def act(self, observation: Observation) -> tuple[float, ...]:
        import torch
        if self._active_policy is None:
            raise ProtocolError("Skill must be started before act")
        with torch.no_grad():
            raw = self._active_policy(observation.policy)
        if tuple(raw.shape) != (1, 13) or not torch.isfinite(raw).all():
            self.adapter.logger.emit("invalid_policy_action", shape=list(raw.shape),
                                     finite=bool(torch.isfinite(raw).all()))
            raise ProtocolError("Official policy emitted wrong-dimensional or nonfinite actions")
        values = tuple(float(v) for v in raw[0].detach().cpu().tolist())
        bounds = self.adapter.action_bounds
        clipped = tuple(max(lo, min(hi, value))
                        for value, lo, hi in zip(values, bounds.low, bounds.high))
        # PPO means are unbounded. Clipping is explicit at this adapter boundary,
        # matching the normalized controller's clipping before physical scaling.
        self.adapter.logger.emit("policy_action", skill=self.name, call_id=self._call_id, raw_action=values,
                                 raw_out_of_bounds=values != clipped, bounded_action=clipped)
        return clipped

    def feedback(self, request: SkillRequest, transition: Transition) -> SkillFeedback:
        return benchmark_feedback(request, transition, self._start_index)


def benchmark_feedback(request: SkillRequest, transition: Transition,
                       start_index: int | None) -> SkillFeedback:
    """Read the info already returned by env.step; never evaluate the env here."""
    if start_index is None:
        raise ProtocolError("Missing invocation start index")
    info = transition.info
    before = int(info["adapter_subtask_before"])
    after = int(info["adapter_subtask_after"])
    evidence = (f"events.jsonl:mshab_step:{transition.observation.frame_id}",)
    if bool(scalar(info.get("fail", False))):
        status, state, reason = SkillStatus.FAILED, RequirementState.UNSATISFIED, "benchmark_fail"
    elif before != start_index or after < before or after > before + 1:
        status, state, reason = SkillStatus.FAILED, RequirementState.UNKNOWN, "unexpected_pointer_change"
    elif after == start_index + 1:
        status, state, reason = SkillStatus.SUCCEEDED, RequirementState.SATISFIED, "oracle_subtask_advanced"
    elif transition.truncated:
        status, state, reason = SkillStatus.TIMED_OUT, RequirementState.UNKNOWN, "environment_horizon"
    else:
        status, state, reason = SkillStatus.EXECUTING, RequirementState.UNSATISFIED, None
    results = tuple(RequirementResult(item.id, state, evidence) for item in request.requirements)
    return SkillFeedback(status, results, reason, "oracle_benchmark")
