import sys

sys.path.insert(0, "/workspace/tapt/upstream/src")
import logging
from openpi.policies import policy_config
from openpi.training import config
from openpi.serving import websocket_policy_server

logging.basicConfig(level=logging.INFO)
policy = policy_config.create_trained_policy(
    config.get_config("pi05_libero"), "gs://openpi-assets/checkpoints/pi05_libero/"
)
websocket_policy_server.WebsocketPolicyServer(
    policy, host="127.0.0.1", port=8000
).serve_forever()
