import pytest
from bvi.baseline_v2 import RecoveryContext, validate_choice, summarize_panel
from bvi.protocol import ProtocolError


def request(g='pick', steps=40):
    return dict(g=g,z='抓起当前罐子。',target_id='can',max_steps=steps)


def test_no_silent_skill_or_target_substitution():
    ctx=RecoveryContext('pick','can')
    assert validate_choice(request(),ctx)==request()
    for r in [request('place'),dict(request(),target_id='other'),request('retry'),request('reposition')]:
        with pytest.raises(ProtocolError):validate_choice(r,ctx)


def test_retry_requires_same_target_and_failed_invocation():
    assert validate_choice(request('retry'),RecoveryContext('pick','can','pick','can','interrupted'))['g']=='retry'
    with pytest.raises(ProtocolError):
        validate_choice(request('retry'),RecoveryContext('pick','can','pick','other','failed'))


def test_reposition_limits_and_observe_are_explicit():
    ctx=RecoveryContext('pick','can',reposition_available=True)
    validate_choice(request('reposition'),ctx)
    validate_choice(request('observe',1),ctx)
    for r in [request('reposition',41),request('observe',2),request('pick',True)]:
        with pytest.raises(ProtocolError):validate_choice(r,ctx)


def test_infra_and_unrun_do_not_disappear_from_denominator():
    r=summarize_panel([0,1,2],[dict(seed=0,status='infrastructure_failure')])
    assert r['planned_n']==3 and r['attempted_n']==1 and r['completed_n']==0
    assert r['not_run_seeds']==[1,2] and r['infrastructure_failures']==1 and r['partial']
    with pytest.raises(ValueError):summarize_panel([0],[dict(seed=0,status='infrastructure_failure')]*2)
