"""Explicit RGBD/state/action contract for the bounded pi05 recipe adaptation.

STATUS: frozen — historical training and diagnostics (retained)

This is not the AC-DiT architecture. Pure helpers are shared by data and deployment.
"""
import numpy as np

SCHEMA = 'pi05_rgbd_native_qpos12_privileged18_v1'
PROMPT = 'Go to the apple, grasp it, and return the arm to rest while holding it.'
ARM_GRIPPER = tuple(range(8))
ACTIVE_WHOLE_BODY = tuple(range(8)) + (10, 11, 12)
IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


def state30(qpos, extra):
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape != (12,):
        raise ValueError('Expected native qpos12; never full robot qpos15')
    parts = [qpos]
    for name, dim in [('tcp_pose_wrt_base',7),('obj_pose_wrt_base',7),('goal_pos_wrt_base',3)]:
        value=np.asarray(extra[name],dtype=np.float32)
        if value.shape!=(dim,): raise ValueError('Bad current observation field: '+name)
        parts.append(value)
    held=np.asarray(extra['is_grasped'])
    if held.shape not in [(),(1,)] or held.dtype!=np.bool_:
        raise ValueError('is_grasped must be a current boolean observation')
    parts.append(held.astype(np.float32).reshape(1))
    result=np.concatenate(parts)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite state')
    return result


def depth_image(depth_mm):
    """Fixed official-BC inverse-depth nonlinearity, quantized for a shared RGB encoder.

    A declared project adaptation, not AC-DiT pointcloud/LIFT3D or calibrated RGB.
    """
    depth=np.asarray(depth_mm)
    if depth.shape==(128,128,1): depth=depth[...,0]
    if depth.shape!=(128,128) or not np.isfinite(depth).all() or np.any(depth<0):
        raise ValueError('Expected nonnegative finite depth128 in millimeters')
    value=np.rint(255*(1-np.tanh(depth.astype(np.float32)/1000))).clip(0,255).astype(np.uint8)
    return np.repeat(value[...,None],3,axis=-1)


def observation(head_rgb,hand_rgb,head_depth_mm,qpos,extra):
    images=[]
    for raw in [head_rgb,hand_rgb]:
        x=np.asarray(raw)
        if x.shape!=(128,128,3) or x.dtype!=np.uint8: raise ValueError('Expected native uint8 RGB128')
        images.append(x)
    images.append(depth_image(head_depth_mm))
    return {'image':dict(zip(IMAGE_KEYS,images)), 'image_mask':{k:np.bool_(True) for k in IMAGE_KEYS},
            'state':state30(qpos,extra),'prompt':PROMPT}


def mix_actions(pi_action,sac_action,mode):
    pi=np.asarray(pi_action,dtype=np.float32)
    if pi.shape!=(13,) or not np.isfinite(pi).all(): raise ValueError('Bad pi05 controller action')
    if mode=='arm_gripper_sac_support':
        sac=np.asarray(sac_action,dtype=np.float32)
        if sac.shape!=(13,) or not np.isfinite(sac).all(): raise ValueError('Live SAC support action required')
        out=sac.copy();out[:8]=pi[:8]
    elif mode=='whole_body': out=pi.copy()
    else: raise ValueError('Unknown frozen action ownership')
    out[8:10]=0
    return np.clip(out,-1,1)


def channel_time_loss(squared_error,valid,channels,xp=np):
    """Mean over exactly supervised channels and valid chunk positions."""
    per_time=xp.mean(squared_error[...,list(channels)],axis=-1)
    valid=valid.astype(per_time.dtype)
    return xp.mean(xp.sum(per_time*valid,axis=-1)/xp.maximum(xp.sum(valid,axis=-1),1))
