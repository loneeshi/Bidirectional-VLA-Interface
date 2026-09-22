"""Paper-style navigation teleport inside the current MS-HAB runtime.

STATUS: active — executors

This is intentionally restricted to one-environment TidyHouse evaluation.  The
state transition follows the teleport branch removed from upstream
``mshab/evaluate.py`` after commit 4729821.  Pick/place actions and all native
success/failure predicates remain owned by the current pinned runtime.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .mshab_adapter import OfficialRLSkill, describe_target, scalar
from .protocol import (Observation, ProtocolError, RequirementResult,
                       RequirementState, SkillFeedback, SkillRequest,
                       SkillStatus, Transition)


SPAWN_LOC_RADIUS = 1.8
SPAWN_XY_NOISE_STD = 0.1
SPAWN_XY_NOISE_MAX = 0.2
SPAWN_ROT_NOISE_MAX = 0.5
MAX_SPAWN_ATTEMPTS = 40


def _gpu_sync(uenv: Any, *, step: bool = False) -> None:
    import sapien.physx as physx

    if physx.is_gpu_enabled():
        uenv.scene._gpu_apply_all()
        uenv.scene.px.gpu_update_articulation_kinematics()
        if step:
            uenv.scene.step()
        uenv.scene._gpu_fetch_all()


def teleport_tidyhouse_navigation(adapter: Any, subtask_num: int) -> bool:
    """Apply the paper evaluator's noisy collision-checked teleport for one env."""
    import numpy as np
    import sapien
    import torch
    from mani_skill.utils.structs.pose import Pose
    from mshab.utils.array import recursive_deepcopy

    uenv = adapter.uenv
    if uenv.num_envs != 1 or adapter.task != "tidy_house":
        raise ProtocolError("Standardized teleport supports one-env TidyHouse only")
    if uenv.task_plan[subtask_num].type != "navigate":
        raise ProtocolError("Teleport can only start on a native navigate subtask")

    env_idx = torch.tensor([0], dtype=torch.long)
    device = uenv.device
    original_state = recursive_deepcopy(uenv.get_state_dict())
    original_robot_pose = uenv.agent.robot.pose.raw_pose.clone()
    original_qpos = uenv.agent.robot.qpos.clone()
    original_qvel = uenv.agent.robot.qvel.clone()
    subtask_obj = uenv.subtask_objs[subtask_num]
    object_pose = None if subtask_obj is None else subtask_obj.pose.raw_pose.clone()
    object_pose_wrt_tcp = (None if subtask_obj is None else
                           (uenv.agent.tcp.pose.inv() * subtask_obj.pose).raw_pose.clone())
    object_linear_velocity = (None if subtask_obj is None else
                              subtask_obj.linear_velocity.clone())
    object_angular_velocity = (None if subtask_obj is None else
                               subtask_obj.angular_velocity.clone())

    positions = torch.from_numpy(np.asarray(
        uenv.scene_builder.navigable_positions[0].vertices)).to(device)
    goal_xy = uenv.subtask_goals[subtask_num].pose.p[0, :2]
    positions = positions[torch.norm(positions - goal_xy, dim=-1) < SPAWN_LOC_RADIUS]
    if len(positions) == 0:
        raise ProtocolError("Official teleport found no navigable position near the goal")

    current_robot_pose = original_robot_pose.clone()
    current_qpos = original_qpos.clone()
    current_qvel = original_qvel.clone()
    current_object_pose = None if object_pose is None else object_pose.clone()
    current_object_linear_velocity = (None if object_linear_velocity is None else
                                      object_linear_velocity.clone())
    current_object_angular_velocity = (None if object_angular_velocity is None else
                                       object_angular_velocity.clone())
    accepted = False
    attempts = 0

    for attempts in range(1, MAX_SPAWN_ATTEMPTS + 1):
        uenv.set_state_dict(recursive_deepcopy(original_state))
        if subtask_obj is not None:
            subtask_obj.set_pose(sapien.Pose(p=[999, 999, 999]))

        sampled_index = int((torch.randint(2**63 - 1, size=(1,)) % len(positions)).item())
        robot_pose = Pose.create(current_robot_pose.clone())
        xy = positions[sampled_index].clone()
        xy += torch.clamp(torch.normal(0, SPAWN_XY_NOISE_STD, xy.shape),
                          -SPAWN_XY_NOISE_MAX, SPAWN_XY_NOISE_MAX).to(device)
        robot_pose.p[0, :2] = xy
        robot_pose.q[0] = Pose.create(sapien.Pose()).q
        current_robot_pose[0] = robot_pose.raw_pose[0].clone()
        current_qpos[0, 2] = 0
        current_qvel[0] = 0

        uenv.agent.robot.set_pose(Pose.create(current_robot_pose.clone()))
        uenv.agent.robot.set_qpos(current_qpos.clone())
        uenv.agent.robot.set_qvel(current_qvel.clone())
        _gpu_sync(uenv)

        goal_wrt_base = uenv.agent.base_link.pose.inv() * uenv.subtask_goals[subtask_num].pose
        target = goal_wrt_base.p[0, :2]
        unit_target = target / torch.norm(target)
        rotation = torch.sign(unit_target[1]) * torch.arccos(unit_target[0])
        # Preserve the paper evaluator, including its XY std for rotation noise.
        rotation += torch.clamp(torch.normal(0, SPAWN_XY_NOISE_STD, rotation.shape),
                                -SPAWN_ROT_NOISE_MAX, SPAWN_ROT_NOISE_MAX).to(device)
        current_qpos[0, 2] += rotation

        uenv.agent.robot.set_pose(Pose.create(current_robot_pose.clone()))
        uenv.agent.robot.set_qpos(current_qpos.clone())
        uenv.agent.robot.set_qvel(current_qvel.clone())
        _gpu_sync(uenv, step=True)
        robot_force = uenv.agent.robot.get_net_contact_forces(
            uenv.agent.robot_link_names)[env_idx].norm(dim=-1).sum(dim=-1)
        acceptable = bool((robot_force == 0).item())

        uenv.agent.robot.set_pose(Pose.create(current_robot_pose.clone()))
        uenv.agent.robot.set_qpos(current_qpos.clone())
        uenv.agent.robot.set_qvel(current_qvel.clone())
        _gpu_sync(uenv)
        current_robot_pose[0] = uenv.agent.robot.pose.raw_pose[0].clone()
        current_qpos[0] = uenv.agent.robot.qpos[0].clone()
        current_qvel[0] = uenv.agent.robot.qvel[0].clone()

        goal_wrt_base = uenv.agent.base_link.pose.inv() * uenv.subtask_goals[subtask_num].pose
        target = goal_wrt_base.p[0, :2]
        unit_target = target / torch.norm(target)
        rotation_from_goal = torch.sign(unit_target[1]) * torch.arccos(unit_target[0])
        distance_from_navmesh = torch.norm(
            positions - uenv.agent.robot.pose.p[0, :2], dim=-1).min()
        acceptable &= bool((rotation_from_goal <=
                            uenv.navigate_cfg.navigated_successfully_rot).item())
        acceptable &= bool((distance_from_navmesh <= 0.04).item())

        if subtask_obj is not None:
            current_object_linear_velocity[0] = 0
            current_object_angular_velocity[0] = 0
            teleported_pose = (uenv.agent.tcp.pose * Pose.create(object_pose_wrt_tcp)).raw_pose
            current_object_pose[0] = teleported_pose[0].clone()
            subtask_obj.set_pose(Pose.create(current_object_pose.clone()))
            subtask_obj.set_linear_velocity(current_object_linear_velocity.clone())
            subtask_obj.set_angular_velocity(current_object_angular_velocity.clone())
            uenv.agent.robot.set_pose(sapien.Pose(p=[999, 999, 999]))
            _gpu_sync(uenv, step=True)
            acceptable &= bool((subtask_obj.get_net_contact_forces()[env_idx]
                                .norm(dim=-1) == 0).item())
        if acceptable:
            accepted = True
            break

    original_robot_pose[0] = current_robot_pose[0].clone()
    original_qpos[0] = current_qpos[0].clone()
    original_qvel[0] = current_qvel[0].clone()
    if subtask_obj is not None:
        object_pose[0] = current_object_pose[0].clone()
        object_linear_velocity[0] = current_object_linear_velocity[0].clone()
        object_angular_velocity[0] = current_object_angular_velocity[0].clone()

    uenv.set_state_dict(recursive_deepcopy(original_state))
    uenv.agent.robot.set_pose(Pose.create(original_robot_pose.clone()))
    uenv.agent.robot.set_qpos(original_qpos.clone())
    uenv.agent.robot.set_qvel(original_qvel.clone())
    uenv.subtask_pointer[0] += 1
    if subtask_obj is not None:
        subtask_obj.set_pose(Pose.create(object_pose))
        subtask_obj.set_linear_velocity(object_linear_velocity)
        subtask_obj.set_angular_velocity(object_angular_velocity)
    _gpu_sync(uenv)
    adapter.logger.emit("official_paper_teleport", subtask_index=subtask_num,
                        attempts=attempts, accepted_spawn=accepted,
                        upstream_commit="4729821db3fc94a2470cfd625e6f8ab439f01478")
    return accepted


class StandardizedTeleportSkill:
    """Teleport navigation, then emit the next official manipulation action."""

    def __init__(self, adapter: Any, checkpoint_root: str | Path, policy_type: str):
        self.adapter = adapter
        self.checkpoint_root = checkpoint_root
        self.policy_type = policy_type
        self.start_index: int | None = None
        self.delegate: OfficialRLSkill | None = None

    def start(self, request: SkillRequest, observation: Observation) -> None:
        index = int(scalar(self.adapter.uenv.subtask_pointer))
        if request.skill != "navigate" or self.adapter.uenv.task_plan[index].type != "navigate":
            raise ProtocolError("Standardized teleport requires the current navigate call")
        self.start_index = index
        teleport_tidyhouse_navigation(self.adapter, index)
        next_index = int(scalar(self.adapter.uenv.subtask_pointer))
        if next_index != index + 1:
            raise ProtocolError("Teleport did not advance exactly one native subtask")
        next_skill = self.adapter.uenv.task_plan[next_index].type
        if next_skill not in {"pick", "place"}:
            raise ProtocolError("TidyHouse teleport must hand off to pick or place")
        target = describe_target(self.adapter.original_plan, next_index)
        synthetic = SkillRequest(f"{request.call_id}-handoff", next_skill, target.id,
            observation.frame_id, request.requirements, request.max_steps,
            request.timeout_seconds)
        policy_observation = replace(observation,
            metadata={**observation.metadata, "subtask_index": next_index})
        self.delegate = OfficialRLSkill(next_skill, self.adapter,
                                        self.checkpoint_root, self.policy_type)
        self.delegate.start(synthetic, policy_observation)

    def act(self, observation: Observation) -> tuple[float, ...]:
        if self.delegate is None:
            raise ProtocolError("Teleport skill must be started before act")
        # The paper evaluator queries the next policy with the pre-teleport obs
        # on the handoff step; preserve that behavior for the standardized arm.
        return self.delegate.act(observation)

    def feedback(self, request: SkillRequest, transition: Transition) -> SkillFeedback:
        if self.start_index is None:
            raise ProtocolError("Teleport feedback is missing its start index")
        info = transition.info
        evidence = (f"events.jsonl:mshab_step:{transition.observation.frame_id}",)
        if bool(scalar(info.get("fail", False))):
            status, state, reason = (SkillStatus.FAILED,
                RequirementState.UNSATISFIED, "benchmark_fail")
        elif int(info["adapter_subtask_after"]) >= self.start_index + 1:
            status, state, reason = (SkillStatus.SUCCEEDED,
                RequirementState.SATISFIED, "official_paper_teleport")
        elif transition.truncated:
            status, state, reason = (SkillStatus.TIMED_OUT,
                RequirementState.UNKNOWN, "environment_horizon")
        else:
            status, state, reason = (SkillStatus.FAILED,
                RequirementState.UNKNOWN, "teleport_pointer_not_advanced")
        results = tuple(RequirementResult(item.id, state, evidence)
                        for item in request.requirements)
        return SkillFeedback(status, results, reason, "oracle_benchmark")
