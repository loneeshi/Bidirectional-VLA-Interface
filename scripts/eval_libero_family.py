"""Matched GPT evaluation; learned events and simulator rules remain separate."""

import argparse, collections, dataclasses, hashlib, io, json, pathlib, sys, time, uuid
import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy
from bvi.bridge import FileBridgeTransport
from bvi.coordinator import VLMRequest
from bvi.protocol import ImageFrame
from bvi.tool_family import FamilyInvocation
from bvi.progress_monitor import ProgressMonitor, THRESHOLDS
from eval_libero_baseline import element

sys.path.insert(0, "/workspace/tapt/author/examples/libero/openvla_eval_port")
from libero_subtask import SubtaskTracker, evaluate_subtask, _get_object_pos

SCHEMA = {
    "type": "object",
    "properties": {
        "tool_family": {"type": "string", "enum": list(THRESHOLDS)},
        "instruction": {"type": "string", "minLength": 1, "maxLength": 160},
        "target": {"type": "string", "enum": ["cream_cheese_1", "butter_1"]},
        "max_steps": {"type": "integer", "minimum": 10, "maximum": 100},
    },
    "required": ["tool_family", "instruction", "target", "max_steps"],
    "additionalProperties": False,
}
SYSTEM = """Coordinate a robot to put BOTH the cream cheese box and butter in the basket. Inspect both real camera images. Choose one tool family and a specific scene-grounded English instruction. Available families: reach (approach the named object with open gripper), grasp (close and secure that object), move (transport held object above basket), release (open gripper to deposit object in basket). Do not claim success from a progress estimate alone. Use feedback to retry or return to reach/grasp after failed execution. Select a bounded 10-100 action-step budget. Usually reach/grasp/move/release in order per object, but all four choices remain available and you may revise based on images. Simulator rules, when supplied, are explicitly named and are imperfect checks. Learned progress is a model estimate, not proof. Your instruction is forwarded exactly to the robot policy. Output only the required JSON."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["standard", "tapt"], required=True)
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(1)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(7)
    np.random.seed(7)
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
    metadata = client.get_server_metadata()
    (out / "server-metadata.json").write_text(json.dumps(metadata, indent=2))
    if a.mode == "tapt":
        locked = json.loads((out.parent / "evaluation-lock.json").read_text())
        if metadata["checkpoint_sha256"] != locked["adapter_metadata"]["sha256"]:
            raise ValueError(
                "Server does not match the checkpoint locked before evaluation"
            )
    bridge = FileBridgeTransport(
        "/workspace/tapt/bridge", "openai", "gpt-5.6-luna", "TAPT007"
    )
    summaries = []
    try:
        for ep in range(5):
            env.reset()
            state = suite.get_task_init_states(1)[ep]
            obs = env.set_init_state(state)
            for _ in range(20):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            tracker = SubtaskTracker(env, task.name, task.language)
            frames = []
            done = False
            error = None
            step = 0
            calls = 0
            cooldown = 0
            history = []
            events = collections.Counter()
            start = time.monotonic()
            with (out / f"episode{ep:03d}.jsonl").open("w") as log:

                def emit(event, **kw):
                    log.write(json.dumps(dict(event=event, step=step, **kw)) + "\n")
                    log.flush()

                try:
                    while step < 520 and calls < 20 and not done:
                        inp = element(obs, task.language)
                        images = []
                        for camera, key in [
                            ("workspace", "observation/image"),
                            ("wrist", "observation/wrist_image"),
                        ]:
                            buf = io.BytesIO()
                            Image.fromarray(inp[key]).resize((128, 128)).save(
                                buf, format="JPEG", quality=75
                            )
                            images.append(
                                ImageFrame(camera, buf.getvalue(), "image/jpeg")
                            )
                        aid = uuid.uuid4().hex
                        calls += 1
                        request = VLMRequest(
                            SYSTEM,
                            json.dumps(
                                {
                                    "task": task.language,
                                    "step": step,
                                    "remaining_steps": 520 - step,
                                    "feedback_history": history[-5:],
                                    "feedback_source": "learned_progress"
                                    if a.mode == "tapt"
                                    else "simulator_rule",
                                }
                            ),
                            tuple(images),
                            SCHEMA,
                            600,
                            aid,
                        )
                        emit(
                            "vlm_request",
                            attempt_id=aid,
                            images=[
                                hashlib.sha256(im.data).hexdigest() for im in images
                            ],
                        )
                        response = bridge.generate(request)
                        decision = json.loads(response.text)
                        invocation = FamilyInvocation(
                            aid,
                            decision["tool_family"],
                            decision["instruction"],
                            decision["max_steps"],
                        )
                        if len(invocation.instruction.encode("utf-8")) > 160:
                            raise ValueError("Instruction too long")
                        if (
                            decision["target"] not in ("cream_cheese_1", "butter_1")
                            or not 10 <= invocation.max_steps <= 100
                        ):
                            raise ValueError("Invalid tool request")
                        emit(
                            "vlm_response",
                            attempt_id=aid,
                            request_id=response.request_id,
                            usage=response.usage,
                            raw_text=response.text,
                            invocation=dataclasses.asdict(invocation),
                        )
                        family = invocation.tool_family
                        target = decision["target"]
                        tracker.index = calls
                        tracker.active_object = target
                        subtask = {"primitive": family, "args": [target]}
                        if family == "release":
                            subtask["kwargs"] = {"release_full_open_threshold": 0.039}
                        if family == "move":
                            subtask = {
                                "primitive": family,
                                "args": ["basket_1"],
                                "object": target,
                                "kwargs": {"pos_offset": [0, 0, [0, 0.2]]},
                            }
                        # Initialize release displacement at invocation start.
                        if family == "release":
                            evaluate_subtask(tracker.env, obs, subtask, tracker)
                        monitor = ProgressMonitor(family, calls - 1, cooldown)
                        queue = collections.deque()
                        reason = "step_limit"
                        last_progress = None
                        rule = False
                        executed = 0
                        for _ in range(min(invocation.max_steps, 520 - step)):
                            inp = element(obs, invocation.instruction)
                            if not queue:
                                if a.mode == "tapt":
                                    inp.update(tool_family=family, call_id=aid)
                                pred = client.infer(inp)
                                actions = np.asarray(pred["actions"])
                                if (
                                    actions.ndim != 2
                                    or actions.shape[1] != 7
                                    or not np.isfinite(actions).all()
                                ):
                                    raise ValueError("Invalid actions")
                                if a.mode == "tapt":
                                    if (
                                        pred["instruction"] != invocation.instruction
                                        or pred["tool_family"] != family
                                        or pred["call_id"] != aid
                                        or pred["adapter_sha256"]
                                        != metadata["adapter_hashes"][family]
                                        or pred["checkpoint_sha256"]
                                        != metadata["checkpoint_sha256"]
                                        or pred["progress_head_sha256"]
                                        != metadata["progress_head_sha256"]
                                    ):
                                        raise ValueError("Routing mismatch")
                                    progress = np.asarray(pred["progress"]).reshape(-1)
                                    if not np.isfinite(progress).all():
                                        raise ValueError("Nonfinite learned progress")
                                    last_progress = float(progress[0])
                                    trigger = monitor.update(last_progress)
                                    emit(
                                        "learned_progress",
                                        call_id=aid,
                                        values=progress.tolist(),
                                        adapter_sha256=pred["adapter_sha256"],
                                        checkpoint_sha256=pred["checkpoint_sha256"],
                                        progress_head_sha256=pred[
                                            "progress_head_sha256"
                                        ],
                                        family=family,
                                        instruction=pred["instruction"],
                                    )
                                    if trigger:
                                        reason = trigger
                                        break
                                queue.extend(actions[:5])
                                emit(
                                    "prediction",
                                    call_id=aid,
                                    actions=actions.tolist(),
                                    instruction=invocation.instruction,
                                )
                            frames.append(inp["observation/image"])
                            action = queue.popleft()
                            obs, _, done, _ = env.step(action.tolist())
                            step += 1
                            executed += 1
                            rule = bool(
                                evaluate_subtask(tracker.env, obs, subtask, tracker)
                            )
                            contacts = [
                                obj
                                for obj in ("cream_cheese_1", "butter_1")
                                if evaluate_subtask(
                                    tracker.env,
                                    obs,
                                    {"primitive": "grasp", "args": [obj]},
                                    tracker,
                                )
                            ]
                            positions = {
                                obj: np.asarray(
                                    _get_object_pos(tracker.env, obj)
                                ).tolist()
                                for obj in ("cream_cheese_1", "butter_1", "basket_1")
                            }
                            emit(
                                "action",
                                call_id=aid,
                                action=action.tolist(),
                                native_success=bool(done),
                                simulator_rule=rule,
                                diagnostic_only={
                                    "requested_object": target,
                                    "dual_finger_contact_grasp_candidates": contacts,
                                    "object_positions": positions,
                                    "eef_position": obs["robot0_eef_pos"].tolist(),
                                    "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
                                },
                            )
                            if done:
                                reason = "native_success"
                                break
                            if a.mode == "standard" and rule:
                                reason = "simulator_rule"
                                break
                        discarded = len(queue)
                        queue.clear()
                        events[reason] += 1
                        cooldown = (
                            15
                            if reason in ("learned_drop", "learned_stagnation")
                            else monitor.replan_cooldown
                        )
                        feedback = {
                            "family": family,
                            "target": target,
                            "instruction": invocation.instruction,
                            "reason": reason,
                            "executed_steps": executed,
                            "progress": last_progress,
                            "source": "learned"
                            if a.mode == "tapt"
                            else "simulator_rule",
                        }
                        if a.mode == "standard":
                            feedback["rule_completed"] = rule
                        history.append(feedback)
                        emit(
                            "invocation_finished",
                            call_id=aid,
                            discarded_actions=discarded,
                            **feedback,
                        )
                except Exception as exc:
                    error = repr(exc)
                    emit("error", error=error)
            video = (
                f"vlm-{a.mode}-episode{ep:03d}" + ("" if done else "-failed") + ".mp4"
            )
            if frames:
                imageio.mimwrite(out / video, frames, fps=10)
            row = dict(
                termination_reason="error"
                if error
                else (
                    "native_success"
                    if done
                    else ("request_limit" if calls >= 20 else "step_limit")
                ),
                episode=ep,
                success=bool(done),
                steps=step,
                vlm_calls=calls,
                error=error,
                video=video if frames else None,
                switch_reasons=dict(events),
                initial_state_sha256=hashlib.sha256(
                    np.asarray(state).tobytes()
                ).hexdigest(),
                seed=7,
                max_steps=520,
                wait_steps=20,
                elapsed_seconds=time.monotonic() - start,
            )
            summaries.append(row)
            (out / "summary.json").write_text(json.dumps(summaries, indent=2))
            print(row, flush=True)
            if error:
                raise RuntimeError(error)
    finally:
        env.close()


if __name__ == "__main__":
    main()
