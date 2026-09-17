"""Official Fetch H5 contract: native proprioception24, no privileged inputs."""
import numpy as np
from .fetch_segments import segment_episode


def native_policy_observation(qpos, qvel, head_rgb, hand_rgb, instruction):
    """Single native observation contract, shared by export and online adapters.

    Accept CPU arrays without batch dimensions. Callers must explicitly select
    an environment and transfer tensors to CPU; never infer/slice old state30.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    if qpos.shape != (12,) or qvel.shape != (12,):
        raise ValueError('Expected unbatched native qpos12/qvel12')
    state = np.concatenate([qpos, qvel])
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite native state')
    images = [np.asarray(head_rgb), np.asarray(hand_rgb)]
    if any(im.shape != (128, 128, 3) or im.dtype != np.uint8 for im in images):
        raise ValueError('Expected unbatched native uint8 RGB128 cameras')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Instruction must be nonempty text')
    return dict(image=images[0].copy(), wrist_image=images[1].copy(),
                state=state, task=instruction)


def parent_ids(scale='pilot'):
    train = list(range(20))
    validation = list(range(20,25))
    if scale in ('medium', 'full'):
        train += list(range(25,155 if scale == 'medium' else 305))
        validation += list(range(305,350))
    elif scale != 'pilot':
        raise ValueError('Unknown scale')
    assert not set(train) & set(validation)
    return {'train': train, 'validation': validation}


def inspect_episode(group, task):
    """Read small state/evidence arrays; leave images lazily in H5."""
    actions=np.asarray(group['actions'],dtype=np.float32)
    n=len(actions)
    if n<1 or actions.shape!=(n,13) or not np.isfinite(actions).all() or (abs(actions)>1.00001).any():
        raise ValueError('Invalid applied Fetch13 action contract')
    if np.any(actions[:,8:10]!=0):
        raise ValueError('Recorded stationary-head actions are not zero; do not silently mask')
    qpos=np.asarray(group['obs/agent/qpos'],dtype=np.float32)
    qvel=np.asarray(group['obs/agent/qvel'],dtype=np.float32)
    if qpos.shape!=(n+1,12) or qvel.shape!=(n+1,12):
        raise ValueError('Expected native qpos12/qvel12, not full robot state30')
    state=np.concatenate([qpos,qvel],axis=1)
    if not np.isfinite(state).all():raise ValueError('Nonfinite proprioception')
    for cam in ('fetch_head','fetch_hand'):
        image=group[f'obs/sensor_data/{cam}/rgb']
        if image.shape!=(n+1,128,128,3) or image.dtype!=np.uint8:
            raise ValueError('Expected native N+1 uint8 RGB cameras')
    success=np.asarray(group['success']); fail=np.asarray(group['fail'])
    if success.shape!=(n,) or fail.shape!=(n,) or success.dtype!=bool or fail.dtype!=bool:
        raise ValueError('Invalid transition success/failure evidence')
    stops=np.flatnonzero(success|fail)
    end=int(stops[0])+1 if len(stops) else n
    succeeded=bool(success[end-1] and not fail[end-1])
    held=np.asarray(group['obs/extra/is_grasped'])[:end+1]
    tcp=np.asarray(group['obs/extra/tcp_pose_wrt_base'])[:end+1,:3]
    obj=np.asarray(group['obs/extra/obj_pose_wrt_base'])[:end+1,:3]
    goal=np.asarray(group['obs/extra/goal_pos_wrt_base'])[:end+1]
    if held.shape!=(end+1,) or held.dtype!=bool or tcp.shape!=(end+1,3) or obj.shape!=tcp.shape or goal.shape!=tcp.shape:
        raise ValueError('Invalid annotation-only geometry')
    windows=segment_episode(task,held,np.linalg.norm(tcp-obj,axis=-1),np.linalg.norm(obj-goal,axis=-1),[end] if succeeded else [])
    return dict(state=state[:end+1],actions=actions[:end],windows=windows,
        original_steps=n,exported_steps=end,native_success=succeeded,
        failed_endpoint_is_completion=False,trim_policy='first_native_success_or_fail_transition_inclusive',
        supervision='pre_action_obs_t_to_applied_action_t; endpoint_obs_retained_in_source',
        privileged_fields_used_only_for_annotation=True)


def policy_frame(group, episode, t, instruction):
    if not 0 <= t < episode['exported_steps']:raise IndexError('No endpoint zero-action frame')
    frame = native_policy_observation(
        episode['state'][t, :12], episode['state'][t, 12:],
        group['obs/sensor_data/fetch_head/rgb'][t],
        group['obs/sensor_data/fetch_hand/rgb'][t], instruction)
    frame['actions'] = episode['actions'][t].copy()
    return frame
