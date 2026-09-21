"""Independent diagnostic arm: all recorded official H5 agent/extra state, never native RGB gate."""
import numpy as np

from . import official_fetch_data as native

SCHEMA = 'official_h5_fetch_agent24_extra18_v1'
EXTRA_FIELDS = (
    ('tcp_pose_wrt_base', 7), ('obj_pose_wrt_base', 7),
    ('goal_pos_wrt_base', 3), ('is_grasped', 1),
)
STATE_DIM = 42


def privileged_state(qpos, qvel, extra):
    """Preserve all 42 scalars in a frozen named order, including quaternions.

    The H5 omits simulator base angular/linear velocity and base position;
    this is recorded privileged42, not full environment state51.
    No action, next observation, reward, success, or future labels enter state.
    """
    arrays = [np.asarray(qpos, dtype=np.float32), np.asarray(qvel, dtype=np.float32)]
    if any(x.shape != (12,) for x in arrays):
        raise ValueError('Expected native unbatched qpos12/qvel12')
    for name, dimension in EXTRA_FIELDS:
        value = np.asarray(extra[name])
        if name == 'is_grasped':
            if value.shape not in ((), (1,)) or value.dtype != np.bool_:
                raise ValueError('Expected current-observation boolean is_grasped')
            value = value.reshape(1)
        if value.shape != (dimension,):
            raise ValueError(f'Wrong official privileged field shape: {name}')
        arrays.append(value.astype(np.float32))
    result = np.concatenate(arrays)
    if result.shape != (STATE_DIM,) or not np.isfinite(result).all():
        raise ValueError('Expected finite full privileged state42')
    return result


def inspect_episode(group, task):
    episode = native.inspect_episode(group, task)
    n = episode['exported_steps'] + 1
    extras = {name: np.asarray(group[f'obs/extra/{name}'])[:n] for name, _ in EXTRA_FIELDS}
    qpos, qvel = episode['state'][:, :12], episode['state'][:, 12:]
    episode['state'] = np.stack([
        privileged_state(qpos[t], qvel[t], {k: v[t] for k, v in extras.items()})
        for t in range(n)])
    episode['privileged_fields_used_only_for_annotation'] = False
    episode['policy_state_schema'] = SCHEMA
    return episode


def policy_frame(group, episode, t, instruction):
    if not 0 <= t < episode['exported_steps']:
        raise IndexError('No endpoint zero-action frame')
    frame = native.native_policy_observation(
        episode['state'][t, :12], episode['state'][t, 12:24],
        group['obs/sensor_data/fetch_head/rgb'][t],
        group['obs/sensor_data/fetch_hand/rgb'][t], instruction)
    frame['state'] = episode['state'][t].copy()
    frame['actions'] = episode['actions'][t].copy()
    return frame
