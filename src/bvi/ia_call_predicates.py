"""Frozen diagnostic call predicates; privileged values score, never feed VLA.

STATUS: frozen — historical training and diagnostics (retained)
"""
import numpy as np

INSTRUCTIONS = {'reach':'Reach the apple with the gripper open.',
                'grasp':'Grasp the apple securely.',
                'move':'Move the held apple to the robot rest pose.'}
LIMITS = {'reach':120, 'grasp':60, 'move':120}


def distance(extra):
    tcp=np.asarray(extra['tcp_pose_wrt_base']).reshape(-1)
    obj=np.asarray(extra['obj_pose_wrt_base']).reshape(-1)
    return float(np.linalg.norm(tcp[:3]-obj[:3]))


def complete(family, extra, info, held_streak):
    held=bool(np.asarray(extra['is_grasped']).item())
    if family=='reach':return distance(extra)<=.08 and not held
    if family=='grasp':return held and held_streak>=3
    if family=='move':return held and bool(np.asarray(info['success']).item())
    raise ValueError('Unknown call family')
