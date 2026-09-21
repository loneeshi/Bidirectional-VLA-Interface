import numpy as np
import pytest
from bvi.pi05_recipe import state30,depth_image,mix_actions,channel_time_loss,ARM_GRIPPER


def test_current_state_order_and_no_base_index_confusion():
    extra=dict(tcp_pose_wrt_base=np.arange(7)+20,obj_pose_wrt_base=np.arange(7)+40,
               goal_pos_wrt_base=np.arange(3)+60,is_grasped=np.bool_(True))
    state=state30(np.arange(12),extra)
    assert state.shape==(30,)
    assert state[3]==3 and state[12]==20 and state[19]==40 and state[26]==60 and state[29]==1
    with pytest.raises(ValueError):state30(np.zeros(15),extra)


def test_depth_units_and_monotonicity():
    depth=np.zeros((128,128,1),np.int16);depth[1]=1000;depth[2]=2000
    rgb=depth_image(depth)
    assert rgb.shape==(128,128,3) and rgb.dtype==np.uint8
    assert int(rgb[0,0,0])>int(rgb[1,0,0])>int(rgb[2,0,0])
    assert np.array_equal(rgb[...,0],rgb[...,2])
    depth[0]=-1
    with pytest.raises(ValueError):depth_image(depth)


def test_explicit_ownership_and_bounds():
    pi=np.arange(13,dtype=np.float32)/10;sac=-pi
    mixed=mix_actions(pi,sac,'arm_gripper_sac_support')
    np.testing.assert_array_equal(mixed[:8],pi[:8])
    np.testing.assert_array_equal(mixed[10:],[-1,-1,-1])
    np.testing.assert_array_equal(mixed[8:10],[0,0])
    with pytest.raises(ValueError):mix_actions(pi,None,'arm_gripper_sac_support')
    np.testing.assert_array_equal(mix_actions(pi,None,'whole_body')[10:],[1,1,1])


def test_unsupervised_channels_and_padding_cannot_change_loss():
    error=np.ones((2,2,32));valid=np.array([[True,False],[True,True]])
    error[...,8:]=999999;error[0,1,:]=999999
    assert channel_time_loss(error,valid,ARM_GRIPPER)==1
    error[1,:,0]=9
    assert channel_time_loss(error,valid,ARM_GRIPPER)==1.5
