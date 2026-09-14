"""Real localhost sockets + official openpi codec; synthetic policy outputs.

No model weights, GPU, network provider or simulator are used by these tests.
"""
import base64
import importlib.util
import json
import threading
import unittest

from bvi.protocol import ImageFrame
from bvi.vla_clients import LightNavClient, OpenPiClient


@unittest.skipUnless(importlib.util.find_spec("websockets"), "Optional websockets missing")
class WireTests(unittest.TestCase):
    def exchange(self, handler, client_fn):
        from websockets.sync.server import serve
        errors = []

        def checked(conn):
            try:
                handler(conn)
            except Exception as exc:
                errors.append(exc)

        with serve(checked, "127.0.0.1", 0, compression=None) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client_fn(f"ws://127.0.0.1:{server.socket.getsockname()[1]}")
            finally:
                server.shutdown()
                thread.join(timeout=5)
            self.assertFalse(errors, repr(errors))

    def test_lightnav_json_socket(self):
        frame = ImageFrame("head", b"\x89PNG\r\n\x1a\nfixture")

        def handler(conn):
            self.assertEqual(json.loads(conn.recv())["action"], "reset")
            conn.send(json.dumps({"action": "reset", "data": {"rc": 0}}))
            request = json.loads(conn.recv())
            self.assertEqual(base64.b64decode(request["data"]["image"]), frame.data)
            conn.send(json.dumps({"action": "next", "data": {
                "rc": 0, "seq": request["data"]["seq"], "stop": False,
                "actions": {"actions": [[.2, .1, .05]]}}}))

        def client_fn(url):
            client = LightNavClient.connect(url, timeout=3)
            try:
                client.reset()
                self.assertEqual(client.infer(frame, "Approach the can").waypoints,
                                 ((.2, .1, .05),))
            finally:
                client.close()
        self.exchange(handler, client_fn)

    @unittest.skipUnless(importlib.util.find_spec("openpi_client"), "Official codec missing")
    def test_openpi_numpy_socket(self):
        import numpy as np
        from openpi_client import msgpack_numpy

        def handler(conn):
            packer = msgpack_numpy.Packer()
            conn.send(packer.pack({"test_only": True}))
            request = msgpack_numpy.unpackb(conn.recv())
            self.assertEqual(request["image"].shape, (224, 224, 3))
            self.assertEqual(request["image"].dtype, np.uint8)
            conn.send(packer.pack({"actions": np.zeros((10, 8), dtype=np.float32)}))

        def client_fn(url):
            client = OpenPiClient.connect(url, timeout=3)
            try:
                self.assertTrue(client.metadata["test_only"])
                actions = client.infer({"image": np.zeros((224, 224, 3), dtype=np.uint8)}, 8)
                self.assertEqual((len(actions), len(actions[0])), (10, 8))
            finally:
                client.close()
        self.exchange(handler, client_fn)
