import sys, argparse

sys.path.insert(0, "/workspace/tapt/upstream/src")
import logging
from openpi.policies import policy_config
from openpi.training import config
from openpi.serving import websocket_policy_server

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--skip-predictions", type=int, default=0)
args = parser.parse_args()
policy = policy_config.create_trained_policy(
    config.get_config("pi05_libero"), "gs://openpi-assets/checkpoints/pi05_libero/"
)
if args.skip_predictions:
    import jax

    for _ in range(args.skip_predictions):
        policy._rng, _ = jax.random.split(policy._rng)
    logging.info(
        "Restored policy RNG after %s logged predictions", args.skip_predictions
    )
websocket_policy_server.WebsocketPolicyServer(
    policy, host="127.0.0.1", port=8000
).serve_forever()
