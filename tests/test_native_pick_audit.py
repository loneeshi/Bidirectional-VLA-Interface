import pytest
from bvi.native_pick_audit import native_failure_causes, validate_episode, validate_seed_roster
from bvi.s1_capability_gate import SEEDS


@pytest.mark.parametrize('info,expected',[
    ({'success':[False],'fail':[True],'subtasks_steps_left':[0], 'cumulative_force_within_limit':[True]}, ['native_horizon_exhausted']),
    ({'success':[False],'fail':[True],'subtasks_steps_left':[175], 'cumulative_force_within_limit':[False]}, ['native_cumulative_force_limit']),
    ({'success':[True],'subtasks_steps_left':[0]}, []),
    ({'success':[False],'fail':[False],'subtasks_steps_left':[20]}, []),
    ({'fail':[True]}, ['other_native_failure']),
])
def test_native_cause_without_changing_termination(info,expected):
    assert native_failure_causes(info)==expected


def test_no_silent_multi_environment_cause():
    with pytest.raises(ValueError):native_failure_causes({'success':[False,True]})


@pytest.mark.parametrize('status,steps,events', [
    ('failed', 2, [{'step':1},{'step':2}]),
    ('episode_completed', 3, [{'step':1},{'step':2}]),
    ('episode_completed', 2, [{'step':1},{'step':1}]),
    ('episode_completed', 0, []),
])
def test_reject_incomplete_evidence(status,steps,events):
    with pytest.raises(ValueError):
        validate_episode({'status':status,'steps':steps},events)


def test_complete_failed_episode_is_valid_evidence():
    validate_episode({'status':'episode_completed','steps':2,'success':False},
                     [{'step':1},{'step':2}])


def test_roster_rejects_duplicate_even_when_ten_rows():
    validate_seed_roster([{'seed':s} for s in SEEDS])
    with pytest.raises(ValueError):
        validate_seed_roster([{'seed':SEEDS[0]} for _ in SEEDS])


def test_full_state_comparison_checks_non_robot_actors():
    from bvi.native_pick_audit import state_max_errors
    a={'simulator':{'actor':[1.,2.], 'robot':[0.,0.]},'controller':{}}
    b={'simulator':{'actor':[1.,3.], 'robot':[0.,0.]},'controller':{}}
    assert state_max_errors(a,b)['/simulator/actor']==1.
    with pytest.raises(ValueError): state_max_errors(a,{'simulator':{'robot':[0.,0.]},'controller':{}})
    with pytest.raises(ValueError): state_max_errors([1,2],[1])
    with pytest.raises(ValueError): state_max_errors([float('nan')],[0])
