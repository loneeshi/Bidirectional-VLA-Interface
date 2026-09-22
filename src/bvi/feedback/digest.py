"""Presentation profiles for MS-HAB history; no task success inference.

STATUS: active — feedback
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass

PROFILES = ('raw_v0', 'object_v1', 'object_trajectory_v1', 'object_trajectory_prior_v1')


def summarize(history, profile='raw_v0', prior=None):
    if profile not in PROFILES:
        raise ValueError('Unknown feedback profile')
    if profile == 'raw_v0':
        return list(history)
    objects = {}
    trajectories = []
    for item in history:
        target, skill = item['target_id'], item['skill']
        if skill not in ('navigate', 'pick', 'place'):
            raise ValueError('Unsupported MS-HAB history skill')
        # Object and assigned destination share the catalog suffix.
        key = target.removeprefix('object-').removeprefix('destination-')
        obj = objects.setdefault(key, {'native_goal': 'unknown', 'last_by_family': {}})
        # Keep object-navigation and destination-navigation distinct.
        obj['last_by_family'][skill + ':' + target] = deepcopy({
            k: asdict(v) if is_dataclass(v) else v for k, v in item.items()
            if k != 'trajectory'})
        if 'trajectory' in item:
            trajectories.append(deepcopy(item['trajectory']))
    result = {'objects': objects, 'calls_total': len(history),
              'steps_total': sum(row.get('steps', 0) for row in history)}
    if profile in PROFILES[2:]:
        result['trajectory'] = trajectories
    if profile == PROFILES[3]:
        if prior is None:
            raise ValueError('Prior profile requires a frozen prior')
        result['spawn_prior'] = deepcopy(prior)
    return result
