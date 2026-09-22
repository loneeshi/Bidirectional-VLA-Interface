"""Opt-in, request-scoped OpenPI noise for paired diagnostics only.

STATUS: frozen — historical training and diagnostics (retained)

No global RNG resets: other websocket calls cannot perturb a diagnostic stream.
The original policy stays unchanged when the diagnostic envelope is absent.
"""
import hashlib
import json

import numpy as np

PROTOCOL = 'bvi-paired-noise/1'
KEY = '_bvi_diagnostic_noise'


def request_noise(request, shape):
    if not isinstance(request, dict) or set(request) != {'seed', 'index'}:
        raise ValueError('Diagnostic noise requires exactly seed and index')
    if any(type(request[k]) is not int or not 0 <= request[k] < 2**32 for k in request):
        raise ValueError('Diagnostic seed/index must be uint32 integers')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([request['seed'], request['index']])))
    return rng.standard_normal(shape).astype('<f4')


def array_hash(array):
    a = np.asarray(array)
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError('Invalid diagnostic array')
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'), copy=False))
    header = json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + a.tobytes()).hexdigest()


def input_hash(obs):
    digest = {'prompt': obs['prompt']}
    for key in ('observation/state', 'observation/image', 'observation/wrist_image'):
        digest[key] = array_hash(obs[key])
    return hashlib.sha256(json.dumps(digest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DiagnosticNoisePolicy:
    """Wrap only a server explicitly launched with --diagnostic-noise."""
    def __init__(self, policy, horizon, model_action_dim):
        if type(horizon) is not int or type(model_action_dim) is not int or not 1 <= horizon <= 100 or not 1 <= model_action_dim <= 256:
            raise ValueError('Invalid internal noise shape')
        self.policy, self.shape = policy, (horizon, model_action_dim)
        self.metadata = {**policy.metadata, 'diagnostic_noise_protocol': PROTOCOL,
                         'diagnostic_noise_shape': list(self.shape)}

    def infer(self, observation):
        obs = dict(observation)
        request = obs.pop(KEY, None)
        if request is None:
            return self.policy.infer(obs)
        noise = request_noise(request, self.shape)
        fingerprint = input_hash(obs)
        result = dict(self.policy.infer(obs, noise=noise))
        result[KEY] = {'protocol': PROTOCOL, **request, 'shape': list(self.shape),
                       'noise_sha256': array_hash(noise), 'input_sha256': fingerprint}
        return result


class PairedNoiseClient:
    """Proxy with a private per-case counter and an actual repeat-inference gate."""
    def __init__(self, client, seed, logger):
        self.client, self.metadata, self.seed, self.logger = client, client.metadata, seed, logger
        if self.metadata.get('diagnostic_noise_protocol') != PROTOCOL:
            raise ValueError('Server does not acknowledge paired diagnostic noise')
        self.shape = tuple(self.metadata['diagnostic_noise_shape'])
        request_noise({'seed': seed, 'index': 0}, self.shape)
        self.index, self.records = 0, []

    def infer(self, observation, action_dim):
        request = {'seed': self.seed, 'index': self.index}
        expected = {'protocol': PROTOCOL, **request, 'shape': list(self.shape),
                    'noise_sha256': array_hash(request_noise(request, self.shape)),
                    'input_sha256': input_hash(observation)}
        def call():
            actions, audit = self.client.infer_diagnostic(observation, action_dim, request)
            if audit != expected:
                raise ValueError('Server diagnostic acknowledgement differs from request')
            return actions
        actions = call()
        repeat_error = None
        if self.index == 0:
            repeat = call()  # No simulation step occurs between these requests.
            if np.asarray(actions).shape != np.asarray(repeat).shape:
                raise ValueError('Repeat inference shape drift')
            repeat_error = float(np.max(np.abs(np.asarray(actions)-np.asarray(repeat))))
            if repeat_error > 1e-6:
                raise ValueError(f'Same-input/noise action drift: {repeat_error}')
        record = {**expected, 'actions': actions, 'repeat_action_max_error': repeat_error}
        self.records.append(record)
        self.logger.emit('paired_inference', **record)
        self.index += 1
        return actions


def check_paired_prediction(reference, current, atol=1e-6):
    for key in ('input_sha256', 'noise_sha256', 'seed', 'index'):
        if reference[key] != current[key]:
            raise ValueError(f'Paired prefix mismatch: {key}')
    a, b = np.asarray(reference['actions']), np.asarray(current['actions'])
    if a.shape != b.shape or not np.isfinite(b).all() or not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError('Paired prefix action mismatch')
    return float(np.max(np.abs(a-b)))
