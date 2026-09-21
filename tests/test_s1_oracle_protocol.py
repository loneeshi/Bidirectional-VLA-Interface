import pytest
from bvi.s1_oracle_protocol import PickOracle
from bvi.fetch_segments import segment_episode


@pytest.mark.parametrize('held,dist', [
    ([False,False,True,True,True,True,True], [.3,.08,.04,.03,.02,.02,.02]),
    ([False,True,False,True,True,True,True], [.3,.08,.04,.03,.02,.02,.02]),
    ([False,False,False,True,False,True,True,True,True], [.4,.2,.08,.03,.04,.03,.02,.02,.02]),
])
def test_causal_switches_match_offline_training_windows(held, dist):
    oracle = PickOracle(previous_held=held[0])
    transitions = []
    for h, d in zip(held[1:], dist[1:]):
        transition = oracle.after_action(d, h)
        if transition:
            transitions.append(transition)
    n = len(held) - 1
    windows = segment_episode('pick', held, dist, [1.] * len(held), [n])
    expected = [w['end'] for w in windows if w['family'] in ('reach', 'grasp')]
    assert [r['observation_index'] for r in transitions] == expected
    assert all(r['model_rng_reset'] is False for r in transitions)


def test_dropping_records_training_support_boundary_without_rewriting_native_score():
    oracle = PickOracle()
    oracle.after_action(.07, False)
    for _ in range(3):
        oracle.after_action(.03, True)
    assert oracle.family == 'move'
    assert oracle.after_action(.03, False) is None
    assert oracle.dropped_after_hold is True
    assert oracle.family == 'move'


def test_near_held_observation_does_not_count_toward_grasp_stability():
    oracle = PickOracle()
    oracle.after_action(.08, True)
    assert oracle.family == 'grasp'
    oracle.after_action(.03, True)
    oracle.after_action(.03, True)
    assert oracle.family == 'grasp'
    oracle.after_action(.03, True)
    assert oracle.family == 'move'


def test_invalid_distance_fails_closed():
    with pytest.raises(ValueError):
        PickOracle().after_action(float('nan'), False)
