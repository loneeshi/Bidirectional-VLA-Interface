"""Bounded VLA-as-Tools dispatch without changing benchmark state."""
from dataclasses import replace

from .protocol import AllowedCall, SkillSpec, Target


ABORT = "abort_task"


class OrganizerView:
    def __init__(self, env, specs, max_slice_steps: int = 40, tool_interface: bool = False):
        if not 1 <= max_slice_steps <= 500:
            raise ValueError("Invalid organizer slice")
        self.env = env
        self.tool_interface = tool_interface
        self.specs = {
            key: replace(value, max_steps=min(value.max_steps, max_slice_steps))
            for key, value in specs.items()
        }
        self.specs[ABORT] = SkillSpec(ABORT, ("task_stopped",), ("task_stopped",), 1, 1.0, ())
        self.last_event = None

    def observe(self):
        observation = self.env.observe()
        if not observation.allowed_calls:
            return observation
        instruction = (
            " Choose the tool and grounded target yourself; the catalog is not a next-step hint. "
            "A step_limit yields control for another decision, not success. "
            "abort_task stops the episode unsuccessfully. No hidden monitor changes the tool outcome."
        )
        event = "" if self.last_event is None else f" Latest execution event: {self.last_event}."
        preferred = ("fetch_nav", "fetch_workspace")
        selected = tuple(image for image in observation.images if image.camera in preferred)
        images = selected if len(selected) == 2 else observation.images[:2]
        return replace(
            observation,
            images=images,
            task=observation.task + instruction + event,
            targets=observation.targets + (Target("episode", "Stop this episode unsuccessfully", "organizer"),),
            allowed_calls=observation.allowed_calls + (AllowedCall(ABORT, "episode"),),
        )

    def note_result(self, result) -> None:
        self.last_event = {
            "skill": result.request.skill,
            "status": result.feedback.status.value,
            "reason": result.feedback.reason,
            "steps": result.steps,
        }
