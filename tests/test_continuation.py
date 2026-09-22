import pytest
from bvi.continuation import RetryLedger
from bvi.protocol import ProtocolError


def test_adjacent_slices_are_not_retries():
    ledger=RetryLedger(3)
    assert ledger.begin(('pick','a')) == 'attempt_started'
    assert ledger.begin(('pick','a')) == 'continued'
    assert ledger.retries == {}


def test_switch_and_return_is_retry():
    ledger=RetryLedger(3)
    ledger.begin(('pick','a')); ledger.begin(('navigate','b'))
    assert ledger.begin(('pick','a')) == 'retried'
    assert ledger.retries[('pick','a')] == 1
    ledger.finish(('pick','a'),True)
    assert not ledger.can_call(('pick','a'))


def test_c0_blocks_abandoned_goal_and_c1_limits_three():
    zero=RetryLedger(0); zero.begin(('pick','a')); zero.begin(('pick','b'))
    assert not zero.can_call(('pick','a'))
    three=RetryLedger(3); three.begin(('pick','a'))
    for other in ('b','c','d'):
        three.begin(('pick',other)); three.begin(('pick','a'))
    three.begin(('pick','z'))
    assert not three.can_call(('pick','a'))
    with pytest.raises(ProtocolError): three.begin(('pick','a'))
