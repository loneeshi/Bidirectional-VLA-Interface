import unittest
import numpy as np

from bvi.diagnostic_noise import (DiagnosticNoisePolicy, PairedNoiseClient,
    check_paired_prediction, request_noise, KEY)


class Policy:
    metadata = {'robot': 'fetch'}
    def __init__(self): self.calls = []
    def infer(self, obs, *, noise=None):
        self.calls.append((obs, noise))
        return {'actions': np.zeros((10, 13)) if noise is None else noise[:, :13]}


class Client:
    def __init__(self, server): self.server, self.metadata = server, server.metadata
    def infer_diagnostic(self, obs, action_dim, request):
        result = self.server.infer({**obs, KEY: request})
        return result['actions'], result[KEY]


class Logger:
    def emit(self, *args, **kwargs): pass


def obs():
    return {'prompt': 'pick can', 'observation/state': np.zeros(30, np.float32),
            'observation/image': np.zeros((224, 224, 3), np.uint8),
            'observation/wrist_image': np.zeros((128, 128, 3), np.uint8)}


class PairedNoiseTests(unittest.TestCase):
    def test_call_order_does_not_change_pair_and_internal_dim_is_32(self):
        policy = Policy(); server = DiagnosticNoisePolicy(policy, 10, 32)
        a = PairedNoiseClient(Client(server), 9, Logger())
        b = PairedNoiseClient(Client(server), 9, Logger())
        first = a.infer(obs(), 13)
        a.infer(obs(), 13)  # Another client/stream can run between paired calls.
        second = b.infer(obs(), 13)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(policy.calls[0][1].shape, (10, 32))
        self.assertNotIn(KEY, policy.calls[0][0])
        self.assertEqual(check_paired_prediction(a.records[0], b.records[0]), 0)

    def test_normal_calls_remain_unchanged(self):
        p = Policy(); server = DiagnosticNoisePolicy(p, 10, 32)
        server.infer(obs())
        self.assertIsNone(p.calls[0][1])

    def test_invalid_or_unacknowledged_request_fails(self):
        for r in ({'seed': True, 'index': 0}, {'seed': -1, 'index': 0}, {'seed': 0}, {'seed': 0, 'index': 0, 'other': 1}):
            with self.assertRaises(ValueError): request_noise(r, (10, 32))
        client = Client(DiagnosticNoisePolicy(Policy(), 10, 32))
        client.metadata = {}
        with self.assertRaises(ValueError): PairedNoiseClient(client, 0, Logger())

    def test_changed_image_is_not_a_paired_observation(self):
        server = DiagnosticNoisePolicy(Policy(), 10, 32)
        a = PairedNoiseClient(Client(server), 0, Logger())
        b = PairedNoiseClient(Client(server), 0, Logger())
        image_obs = obs(); image_obs['observation/image'][0, 0, 0] = 1
        a.infer(obs(), 13); b.infer(image_obs, 13)
        with self.assertRaisesRegex(ValueError, 'input_sha256'):
            check_paired_prediction(a.records[0], b.records[0])

    def test_different_index_changes_noise(self):
        self.assertFalse(np.array_equal(request_noise({'seed': 0, 'index': 0}, (10, 32)),
                                        request_noise({'seed': 0, 'index': 1}, (10, 32))))

    def test_no_ack_or_same_input_action_drift_is_rejected(self):
        class MissingAck(Client):
            def infer_diagnostic(self, *args):
                actions, _ = super().infer_diagnostic(*args)
                return actions, None
        server = DiagnosticNoisePolicy(Policy(), 10, 32)
        with self.assertRaisesRegex(ValueError, 'acknowledgement'):
            PairedNoiseClient(MissingAck(server), 0, Logger()).infer(obs(), 13)
        class DriftingPolicy(Policy):
            def infer(self, obs, *, noise=None):
                result = super().infer(obs, noise=noise)
                result['actions'] = result['actions'] + len(self.calls)
                return result
        with self.assertRaisesRegex(ValueError, 'action drift'):
            PairedNoiseClient(Client(DiagnosticNoisePolicy(DriftingPolicy(), 10, 32)), 0, Logger()).infer(obs(), 13)
