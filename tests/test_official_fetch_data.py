import numpy as np
import pytest
from bvi.official_fetch_data import parent_ids,inspect_episode,policy_frame,native_policy_observation


def fixture():
    n=8
    g={'actions':np.zeros((n,13),np.float32),'success':np.array([False]*6+[True,True]),'fail':np.zeros(n,bool),
       'obs/agent/qpos':np.ones((n+1,12),np.float32),'obs/agent/qvel':np.zeros((n+1,12),np.float32),
       'obs/extra/is_grasped':np.array([False]*3+[True]*6),
       'obs/extra/tcp_pose_wrt_base':np.zeros((n+1,7)),
       'obs/extra/obj_pose_wrt_base':np.zeros((n+1,7)),
       'obs/extra/goal_pos_wrt_base':np.zeros((n+1,3))}
    for camera in ['fetch_head','fetch_hand']:
        g[f'obs/sensor_data/{camera}/rgb']=np.zeros((n+1,128,128,3),np.uint8)
    return g


def test_first_success_and_no_endpoint_action_or_privileged_input():
    g=fixture();e=inspect_episode(g,'pick')
    assert e['exported_steps']==7 and e['state'].shape==(8,24)
    f=policy_frame(g,e,6,'Pick apple')
    assert set(f)=={'image','wrist_image','state','actions','task'}
    assert f['state'].shape==(24,)
    with pytest.raises(IndexError):policy_frame(g,e,7,'Pick apple')


def test_failure_never_becomes_native_completion():
    g=fixture();g['fail'][4]=True
    e=inspect_episode(g,'pick')
    assert e['exported_steps']==5 and not e['native_success']
    assert not any(w['family']=='move' for w in e['windows'])


def test_pilot_split_preserved_at_all_scales():
    small=parent_ids()
    for scale,n in [('medium',150),('full',300)]:
        ids=parent_ids(scale)
        assert len(ids['train'])==n and len(ids['validation'])==50
        assert set(small['validation'])<=set(ids['validation'])
        assert not set(ids['train'])&set(ids['validation'])


def test_reject_old_state_contract_and_do_not_silently_mask_actions():
    g=fixture();g['obs/agent/qpos']=np.zeros((9,15))
    with pytest.raises(ValueError):inspect_episode(g,'pick')
    g=fixture();g['actions'][0,8]=.1
    with pytest.raises(ValueError):inspect_episode(g,'pick')


def test_shared_contract_preserves_camera_order_instruction_and_owns_arrays():
    head=np.full((128,128,3),17,np.uint8);hand=np.full_like(head,91)
    qpos=np.arange(12,dtype=np.float32);qvel=-qpos
    z='Grasp the apple securely.'
    result=native_policy_observation(qpos,qvel,head,hand,z)
    np.testing.assert_array_equal(result['state'],np.r_[qpos,qvel])
    assert result['task']==z
    head[:]=0;hand[:]=0;qpos[:]=0
    assert result['image'].min()==17 and result['wrist_image'].min()==91
    assert result['state'][11]==11
    with pytest.raises(ValueError):native_policy_observation(np.zeros(15),qvel,head,hand,z)
    with pytest.raises(ValueError):native_policy_observation(qpos,qvel,head.astype(float),hand,z)
    with pytest.raises(ValueError):native_policy_observation(qpos,qvel,head,hand,' ')
