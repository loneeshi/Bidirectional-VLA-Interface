import pytest
from bvi.ia_call_predicates import complete

def extra(held=False, d=.1):
    return dict(is_grasped=[held],tcp_pose_wrt_base=[[0,0,0,1,0,0,0]],obj_pose_wrt_base=[[d,0,0,1,0,0,0]])

def test_reach_near_not_held():
    assert complete('reach',extra(d=.08),{},0)
    assert not complete('reach',extra(d=.081),{},0)
    assert not complete('reach',extra(True,.01),{},3)

def test_grasp_requires_stability():
    assert not complete('grasp',extra(True),{},2)
    assert complete('grasp',extra(True),{},3)

def test_move_requires_native_completion_and_hold():
    assert not complete('move',extra(True),{'success':[False]},3)
    assert not complete('move',extra(False),{'success':[True]},0)
    assert complete('move',extra(True),{'success':[True]},3)

def test_unknown_family_rejected():
    with pytest.raises(ValueError):complete('release',extra(),{},0)
