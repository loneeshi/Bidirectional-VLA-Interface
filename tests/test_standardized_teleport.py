import importlib.util
from pathlib import Path
import sys

from bvi.protocol import (Observation, Requirement, SkillRequest, SkillStatus,
                          Transition)
from bvi.teleport_skill import StandardizedTeleportSkill


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'standardized_teleport_runner', ROOT / 'scripts/run_standardized_teleport16.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def request():
    return SkillRequest('call-1', 'navigate', 'target', 'frame-0',
                        (Requirement('done', 'benchmark_success'),), 40, 180)


def test_teleport_feedback_uses_native_pointer_advance():
    skill = StandardizedTeleportSkill.__new__(StandardizedTeleportSkill)
    skill.start_index = 0
    observation = Observation('frame-1', 1)
    transition = Transition(observation, info={
        'fail': False, 'adapter_subtask_before': 1, 'adapter_subtask_after': 1})
    feedback = skill.feedback(request(), transition)
    assert feedback.status is SkillStatus.SUCCEEDED
    assert feedback.reason == 'official_paper_teleport'


def test_teleport_feedback_keeps_native_failure_authoritative():
    skill = StandardizedTeleportSkill.__new__(StandardizedTeleportSkill)
    skill.start_index = 0
    observation = Observation('frame-1', 1)
    transition = Transition(observation, info={
        'fail': True, 'adapter_subtask_before': 1, 'adapter_subtask_after': 1})
    assert skill.feedback(request(), transition).status is SkillStatus.FAILED


def test_runner_freezes_same_rgbd_cameras_and_initial_state():
    argv = RUNNER.command({'seed': 2, 'plan_uid': 'uid'}, Path('/out'),
                          Path('/ckpt'), 'abc123')
    assert argv[argv.index('--navigation-policy') + 1] == 'teleport'
    assert argv[argv.index('--navigation-camera') + 1] == 'fetch_nav'
    assert '--workspace-camera' in argv
    assert argv[argv.index('--expected-initial-state-sha256') + 1] == 'abc123'
    assert '--dry-run' in argv
