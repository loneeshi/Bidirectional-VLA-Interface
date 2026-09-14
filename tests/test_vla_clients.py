import json
import unittest

from bvi.protocol import ImageFrame, ProtocolError
from bvi.vla_clients import LightNavClient, OpenPiClient, finite_rows


class Connection:
    def __init__(self, replies):
        self.replies, self.sent, self.timeouts = iter(replies), [], []

    def send(self, value):
        self.sent.append(value)

    def recv(self, timeout):
        self.timeouts.append(timeout)
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def reply(**data):
    return json.dumps({"action": "next", "data": {"rc": 0, "seq": 0,
                      "stop": False, "actions": {"actions": [[1, .2, .1]]}, **data}})


class VLAClientTests(unittest.TestCase):
    image = ImageFrame("fetch_head", b"\x89PNG\r\n\x1a\nfixture")

    def test_navigation_session_and_cumulative_waypoints(self):
        conn = Connection([json.dumps({"action": "reset", "data": {"rc": 0}}), reply()])
        client = LightNavClient(conn, timeout=2)
        client.reset()
        result = client.infer(self.image, "Approach the counter")
        self.assertEqual(result.waypoints, ((1., .2, .1),))
        self.assertFalse(result.stop)
        self.assertEqual(json.loads(conn.sent[-1])["data"]["seq"], 0)
        self.assertEqual(conn.timeouts, [2, 2])

    def test_navigation_rejects_bad_sequence_and_nan(self):
        for response in [reply(seq=3), reply(actions={"actions": [[float("nan"), 0, 0]]}),
                         reply(stop="false"), reply(rc=1), reply(actions={"actions": [[1, 2]]})]:
            with self.subTest(response=response), self.assertRaises(ProtocolError):
                LightNavClient(Connection([response])).infer(self.image, "Go")

    def test_stop_is_prediction_not_skill_success(self):
        result = LightNavClient(Connection([reply(stop=True, actions={"actions": []})])).infer(self.image, "Go")
        self.assertTrue(result.stop)
        self.assertEqual(result.waypoints, ())

    def test_timeout_is_not_retried(self):
        conn = Connection([TimeoutError()])
        with self.assertRaises(TimeoutError):
            LightNavClient(conn).infer(self.image, "Go")
        self.assertEqual(len(conn.sent), 1)

    def test_openpi_handshake_and_shape_guard(self):
        # Codec fixture tests framing/control flow, not msgpack compatibility.
        class Codec:
            class Packer:
                def pack(self, value):
                    return json.dumps(value).encode()
            @staticmethod
            def unpackb(value):
                return json.loads(value)
        conn = Connection([b'{}', json.dumps({"actions": [[0] * 8]}).encode()])
        client = OpenPiClient(conn, Codec)
        self.assertEqual(len(client.infer({"prompt": "pick"}, 8)[0]), 8)
        with self.assertRaises(ProtocolError):
            finite_rows([[0] * 8], 13)  # Never pad DROID actions into Fetch.

    def test_openpi_server_error(self):
        class Codec:
            class Packer:
                pass
        with self.assertRaises(ProtocolError):
            OpenPiClient(Connection(["private traceback"]), Codec)


if __name__ == "__main__":
    unittest.main()
