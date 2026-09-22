"""Load a frozen independent-data prior; no online sampling.

STATUS: active — feedback
"""
import hashlib
import json
from pathlib import Path


def load_prior(path, evaluation_plan_uids):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    parents = data.get('training_plan_uids')
    if (data.get('schema') not in ('spawn-prior/1', 'spawn-prior/2') or not isinstance(parents, list)
            or not parents or not all(isinstance(x, str) and x for x in parents)):
        raise ValueError('Prior needs explicit independent training plan provenance')
    if set(parents) & set(evaluation_plan_uids):
        raise ValueError('Prior overlaps evaluation plans')
    if data['schema'] == 'spawn-prior/2':
        from bvi.pose_recovery import audit_pose_prior
        audit = audit_pose_prior(data)
        if not audit['ready']:
            raise ValueError('Incomplete executable pose prior: ' + ','.join(audit['reasons']))
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'data': data}
    if not isinstance(data.get('bins'), list) or not data['bins']:
        raise ValueError('Prior has no measured bins')
    for row in data['bins']:
        n, successes = row['samples'], row['successes']
        if type(n) is not int or type(successes) is not int or not 0 <= successes <= n or n == 0:
            raise ValueError('Invalid empirical prior counts')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'data': data}
