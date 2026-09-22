"""Invocation-aligned action-only data. No family routing or progress loss.

STATUS: frozen — historical training and diagnostics (retained)
"""
import numpy as np

from .official_fetch_data import current_progress_rows, native_policy_observation


def invocation_rows(manifest, horizon=10):
    """Reuse reviewed windows, but exclude endpoints from action sampling."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Positive integer horizon required')
    candidates = current_progress_rows(manifest, horizon)
    rows, seen = [], set()
    for row in candidates:
        if not row['action_valid'][0]:
            continue
        key = (row['parent_id'], row['observation_index'])
        if key in seen:
            raise ValueError('Overlapping action windows duplicate sample exposure')
        seen.add(key)
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('Invocation instruction missing')
        # Drop progress fields: this is IA SFT, not joint TAPT supervision.
        rows.append({k: row[k] for k in (
            'source_sha256', 'trajectory', 'parent_id', 'split', 'call_index',
            'tool_family', 'instruction', 'observation_index',
            'action_source_indices', 'action_valid')})
    if not rows:
        raise ValueError('No action-supervised invocation frames')
    return rows


def invocation_sample(group, row):
    """Read a pinned H5 row; caller verifies source hash before constructing us."""
    t = row['observation_index']
    indices = row['action_source_indices']
    valid = np.asarray(row['action_valid'], dtype=bool)
    if not len(valid) or not valid[0] or len(indices) != len(valid):
        raise ValueError('Invalid invocation action mask')
    n = int(valid.sum())
    if not np.array_equal(valid, np.arange(len(valid)) < n):
        raise ValueError('Invocation mask must be a contiguous valid prefix')
    if indices != list(range(t, t+n)) + [None]*(len(valid)-n):
        raise ValueError('Source indices do not match mask/current frame')
    real = np.asarray(group['actions'][t:t+n], dtype=np.float32)
    if real.shape != (n, 13) or not np.isfinite(real).all():
        raise ValueError('Invalid Fetch action source')
    if (np.abs(real) > 1.00001).any() or np.any(real[:, 8:10] != 0):
        raise ValueError('Source violates wrapper action contract')
    obs = native_policy_observation(
        group['obs/agent/qpos'][t], group['obs/agent/qvel'][t],
        group['obs/sensor_data/fetch_head/rgb'][t],
        group['obs/sensor_data/fetch_hand/rgb'][t], row['instruction'])
    obs['prompt'] = obs.pop('task')
    obs['actions'] = np.concatenate([real, np.repeat(real[-1:], len(valid)-n, axis=0)])
    obs['actions_is_pad'] = ~valid
    return obs


def transform_sample(sample, transform):
    """Keep time mask outside transforms that repack/drop auxiliary keys."""
    mask = np.asarray(sample['actions_is_pad'], dtype=bool).copy()
    transformed = dict(transform({k: v for k, v in sample.items() if k != 'actions_is_pad'}))
    if transformed['actions'].shape[0] != len(mask):
        raise ValueError('Transform changed action horizon')
    transformed['actions_is_pad'] = mask
    return transformed
