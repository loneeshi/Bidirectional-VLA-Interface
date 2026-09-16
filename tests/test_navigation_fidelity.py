from types import SimpleNamespace
import pytest

from bvi.navigation_fidelity import NavigationHistoryAudit, first_waypoint_velocity
from bvi.lightnav_skill import waypoint_velocity
from bvi.protocol import ProtocolError


def test_zero_first_waypoint_is_preserved_and_legacy_is_unchanged():
    rows = ((0, 0, 0), (.2, 0, .1))
    assert first_waypoint_velocity(rows, .25) == (0, 0)
    assert waypoint_velocity(rows, .25) == (.6, .4)
    assert first_waypoint_velocity(((.1, .8, -.1),), .25) == (.4, -.4)


def test_goal_scope_keeps_history_but_exposes_changed_language():
    calls = []
    audit = NavigationHistoryAudit(SimpleNamespace(reset=lambda: calls.append('reset')), 'goal')
    assert audit.begin(episode_token='reset1', goal_token='target1', instruction='table').reset
    assert not audit.begin(episode_token='reset1', goal_token='target1', instruction='table').reset
    drift = audit.begin(episode_token='reset1', goal_token='target1', instruction='counter')
    assert not drift.reset and drift.instruction_changed
    assert audit.begin(episode_token='reset1', goal_token='target2', instruction='counter').reset
    assert audit.begin(episode_token='reset2', goal_token='target2', instruction='counter').reset
    assert len(calls) == 3


def test_call_scope_and_transport_failure_are_explicit():
    calls = []
    audit = NavigationHistoryAudit(SimpleNamespace(reset=lambda: calls.append(1)))
    for _ in range(2):
        assert audit.begin(episode_token='e', goal_token='g', instruction='apple').reset
    assert len(calls) == 2
    audit = NavigationHistoryAudit(SimpleNamespace(reset=lambda: (_ for _ in ()).throw(OSError('reset failed'))), 'goal')
    with pytest.raises(OSError):
        audit.begin(episode_token='e', goal_token='g', instruction='apple')
    assert audit.key is None


def test_bad_period_and_missing_episode_are_rejected():
    for dt in (0, -1, float('nan')):
        with pytest.raises(ProtocolError):
            first_waypoint_velocity(((0, 0, 0),), dt)
    with pytest.raises(ProtocolError):
        NavigationHistoryAudit(None).begin(episode_token='', goal_token='g', instruction='apple')
