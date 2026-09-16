"""Serve explicit family-conditioned residuals with a shared frozen backbone."""

import argparse, hashlib, json, pathlib, pickle
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
from openpi.models import model as models
from openpi.training import checkpoints
from openpi import transforms
from openpi.serving import websocket_policy_server
from train_libero_family import initialize, transform, FAMILIES, TRAIN


class FamilyPolicy:
    def __init__(self, checkpoint, adapters):
        root = pathlib.Path(adapters)
        p = root / "adapters.pkl"
        meta = json.loads((root / "metadata.json").read_text())
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError("Adapter checksum mismatch")
        self.checkpoint_sha256 = digest
        # Only load our locally trained artifact, whose manifest was checked.
        saved = pickle.loads(p.read_bytes())
        graph, state = initialize(checkpoint)
        self.frozen = state.filter(nnx.Not(TRAIN))
        self.banks = jax.device_put(saved["banks"])
        self.head = jax.device_put(saved["head"])
        self.head_sha256 = hashlib.sha256(pickle.dumps(saved["head"])).hexdigest()
        self.input = transform(checkpoint)
        self.output = transforms.Unnormalize(
            checkpoints.load_norm_stats(
                pathlib.Path(checkpoint) / "assets", "physical-intelligence/libero"
            ),
            use_quantiles=True,
        )
        self.hashes = {
            g: hashlib.sha256(
                pickle.dumps(jax.device_get(saved["banks"][g]))
            ).hexdigest()
            for g in FAMILIES
        }
        self.active = None
        self.call = None
        self.instruction = None
        self.rng = jax.random.key(0)

        def infer(params, frozen, key, obs):
            model = nnx.merge(graph, nnx.State.merge(frozen, params))
            return model.infer_actions_and_progress(key, obs)

        # Parameters remain explicit inputs: module_jit would capture old adapters.
        self.predict = jax.jit(infer)

    @property
    def metadata(self):
        return {
            "families": list(FAMILIES),
            "adapter_hashes": self.hashes,
            "adapter_hash_scope": "serialized single-family state; shared head separately hashed",
            "checkpoint_sha256": self.checkpoint_sha256,
            "progress_head_sha256": self.head_sha256,
            "progress_source": "learned",
        }

    def infer(self, obs):
        if obs.get("experiment_control") == "restore_prefix_rng":
            count = obs["prediction_count"]
            if not isinstance(count, int) or not 0 <= count <= 520:
                raise ValueError("Invalid prefix prediction count")
            self.rng = jax.random.key(0)
            for _ in range(count):
                self.rng, _ = jax.random.split(self.rng)
            self.active = self.call = self.instruction = None
            return {
                "rng": np.asarray(jax.random.key_data(self.rng)),
                "prediction_count": count,
                "checkpoint_sha256": self.checkpoint_sha256,
            }
        family = obs.get("tool_family")
        call = obs.get("call_id")
        text = obs.get("prompt")
        if (
            family not in FAMILIES
            or not isinstance(call, str)
            or not call
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Explicit family, call ID and instruction required")
        if call == self.call and (family != self.active or text != self.instruction):
            raise ValueError("Cannot change family/instruction inside a call")
        self.active, self.call, self.instruction = family, call, text
        x = self.input(
            {k: v for k, v in obs.items() if k not in ("tool_family", "call_id")}
        )
        x = jax.tree.map(lambda v: jnp.asarray(v)[None], x)
        self.rng, key = jax.random.split(self.rng)
        actions, progress = self.predict(
            nnx.State.merge(self.banks[family], self.head),
            self.frozen,
            key,
            models.Observation.from_dict(x),
        )
        values = self.output(
            {"actions": np.asarray(actions[0]), "state": np.asarray(x["state"][0])}
        )
        return {
            "actions": values["actions"][:, :7],
            "progress": np.asarray(progress[0]),
            "progress_source": "learned",
            "adapter_sha256": self.hashes[family],
            "checkpoint_sha256": self.checkpoint_sha256,
            "progress_head_sha256": self.head_sha256,
            "tool_family": family,
            "call_id": call,
            "instruction": text,
        }


def audit_routing(policy, data, output):
    """Hold image, instruction and diffusion noise fixed; vary only the bank."""
    import io
    import pandas as pd
    from PIL import Image

    manifest = json.loads((pathlib.Path(data) / "manifest.json").read_text())
    row = pd.read_parquet(manifest["train"][0]["path"]).iloc[0]
    observation = {
        "observation/image": np.asarray(
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        ),
        "observation/wrist_image": np.asarray(
            Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")
        ),
        "observation/state": np.asarray(row.state),
        "prompt": "reach the cream cheese box",
    }
    records = []
    for family in FAMILIES:
        policy.rng = jax.random.key(0)
        result = policy.infer(
            dict(observation, tool_family=family, call_id="audit-" + family)
        )
        records.append(
            {
                "family": family,
                "adapter_sha256": result["adapter_sha256"],
                "actions_sha256": hashlib.sha256(
                    result["actions"].tobytes()
                ).hexdigest(),
                "actions": result["actions"].tolist(),
                "progress": result["progress"].tolist(),
            }
        )
    assert len({r["adapter_sha256"] for r in records}) == 4
    assert len({r["actions_sha256"] for r in records}) == 4
    pathlib.Path(output).write_text(
        json.dumps(
            {
                "passed": True,
                "same_observation_instruction_noise": True,
                "sampling_seed_reset_to_zero_before_evaluation": True,
                "simulation_actions": 0,
                "api_calls": 0,
                "records": records,
            },
            indent=2,
        )
    )
    policy.rng = jax.random.key(0)
    policy.active = policy.call = policy.instruction = None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapters", required=True)
    p.add_argument("--audit-data")
    p.add_argument("--skip-predictions", type=int, default=0)
    p.add_argument(
        "--audit-output", default="/workspace/tapt/evidence/family-routing-audit.json"
    )
    a = p.parse_args()
    policy = FamilyPolicy(a.checkpoint, a.adapters)
    if a.audit_data:
        audit_routing(policy, a.audit_data, a.audit_output)
    if a.skip_predictions < 0:
        raise ValueError("Prediction count must be nonnegative")
    for _ in range(a.skip_predictions):
        policy.rng, _ = jax.random.split(policy.rng)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host="127.0.0.1", port=8000, metadata=policy.metadata
    ).serve_forever()
