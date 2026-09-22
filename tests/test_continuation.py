import pytest
from bvi.continuation import RetryLedger
from bvi.protocol import ProtocolError


def test_unsuccessful_full_horizon_repeat_is_retry():
    ledger=RetryLedger(3)
    assert ledger.begin(('pick','a')) == 'attempt_started'
    ledger.finish(('pick','a'),False)
    assert ledger.begin(('pick','a')) == 'retried'
    assert ledger.retries == {('pick','a'):1}


def test_switch_and_return_is_retry():
    ledger=RetryLedger(3)
    ledger.begin(('pick','a')); ledger.finish(('pick','a'),False)
    ledger.begin(('navigate','b')); ledger.finish(('navigate','b'),False)
    assert ledger.begin(('pick','a')) == 'retried'
    assert ledger.retries[('pick','a')] == 1
    ledger.finish(('pick','a'),True)
    assert not ledger.can_call(('pick','a'))


def test_c0_blocks_abandoned_goal_and_c1_limits_three():
    zero=RetryLedger(0); zero.begin(('pick','a')); zero.finish(('pick','a'),False)
    assert not zero.can_call(('pick','a'))
    three=RetryLedger(3); three.begin(('pick','a')); three.finish(('pick','a'),False)
    for _ in range(3):
        three.begin(('pick','a')); three.finish(('pick','a'),False)
    assert not three.can_call(('pick','a'))
    with pytest.raises(ProtocolError): three.begin(('pick','a'))
