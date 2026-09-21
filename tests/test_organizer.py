from dataclasses import replace

import pytest

from bvi import AllowedCall, ImageFrame, Observation, SkillFeedback, SkillRequest, SkillResult, SkillSpec, SkillStatus, Target
from bvi.organizer import ABORT, OrganizerView


class Env:
    def __init__(self):
        self.observation = Observation(
            "frame-0",
            0,
            images=(ImageFrame("fetch_nav", b"a"), ImageFrame("fetch_workspace", b"b")),
            targets=(Target("apple", "apple", "task_plan"),),
            allowed_calls=(AllowedCall("pick", "apple"),),
            task="TidyHouse.",
            metadata={"interface": "goal-grounded-ppo-sac/1"},
        )

    def observe(self):
        return self.observation


def test_organizer_adds_abort_and_freezes_slice():
    view = OrganizerView(Env(), {"pick": SkillSpec("pick", max_steps=200)}, 40)
    observation = view.observe()
    assert view.specs["pick"].max_steps == 40
    assert view.specs[ABORT].max_steps == 1
    assert observation.allowed_calls[-1] == AllowedCall(ABORT, "episode")
    assert "not a next-step hint" in observation.task
    assert [image.camera for image in observation.images] == ["fetch_nav", "fetch_workspace"]


def test_organizer_reports_only_serialized_previous_result():
    view = OrganizerView(Env(), {"pick": SkillSpec("pick")}, 40)
    request = SkillRequest("call", "pick", "apple", "frame-0", ())
    result = SkillResult(request, SkillFeedback(SkillStatus.TIMED_OUT, reason="step_limit"), 40, 1.0, view.env.observe())
    view.note_result(result)
    assert view.last_event == {"skill": "pick", "status": "timed_out", "reason": "step_limit", "steps": 40}
    assert "step_limit" in view.observe().task


def test_invalid_slice_is_rejected():
    with pytest.raises(ValueError):
        OrganizerView(Env(), {"pick": SkillSpec("pick")}, 0)


def test_finished_observation_is_not_modified():
    env = Env()
    env.observation = replace(env.observation, allowed_calls=())
    assert OrganizerView(env, {"pick": SkillSpec("pick")}).observe() is env.observation
