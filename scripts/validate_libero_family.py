"""Select only on disjoint validation data, sampling early/middle/late progress."""

import argparse, json, pathlib, pickle, time, hashlib
import flax.nnx as nnx
import jax, jax.numpy as jnp
import numpy as np
from train_libero_family import initialize, Samples, TRAIN, FAMILIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--data", required=True)
    a = p.parse_args()
    root = pathlib.Path(a.training)
    graph, state = initialize(a.checkpoint)
    frozen = state.filter(nnx.Not(TRAIN))
    del state
    samples = Samples(a.data, a.checkpoint)

    def evaluate(params, frozen, key, batch):
        model = nnx.merge(graph, nnx.State.merge(frozen, params))
        obs, actions, target, mask = batch
        action_loss, progress = model.compute_action_and_progress_chunk_prefix(
            key, obs, actions, train=False
        )
        al = jnp.sum(action_loss * mask) / jnp.sum(mask)
        pl = jnp.sum(jnp.square(progress - target) * mask) / jnp.sum(mask)
        return al, pl, progress

    evaluate = jax.jit(evaluate)
    rows = []
    best = None
    for folder in sorted(root.glob("step-*")):
        payload = (folder / "adapters.pkl").read_bytes()
        assert (
            hashlib.sha256(payload).hexdigest()
            == json.loads((folder / "metadata.json").read_text())["sha256"]
        )
        saved = pickle.loads(payload)
        banks = jax.device_put(saved["banks"])
        head = jax.device_put(saved["head"])
        del saved
        records = []
        for family in FAMILIES:
            n = sum(
                w["split"] == "validation" and w["family"] == family
                for w in samples.windows
            )
            for wi in range(n):
                for position in [0.1, 0.5, 0.9]:
                    batch = samples.get("validation", family, True, wi, position)
                    al, pl, progress = evaluate(
                        nnx.State.merge(banks[family], head),
                        frozen,
                        jax.random.key(123 + wi),
                        batch,
                    )
                    target = np.asarray(batch[2])[0]
                    mask = np.asarray(batch[3])[0]
                    pred = np.asarray(progress)[0]
                    records.append(
                        dict(
                            family=family,
                            window=wi,
                            position=position,
                            action_loss=float(al),
                            progress_mse=float(pl),
                            target=target.tolist(),
                            prediction=pred.tolist(),
                            mask=mask.tolist(),
                            constant_half_mse=float(
                                np.sum((target - 0.5) ** 2 * mask) / mask.sum()
                            ),
                        )
                    )
        loss = float(
            np.mean([r["action_loss"] + 0.1 * r["progress_mse"] for r in records])
        )
        row = {
            "step": int(folder.name.split("-")[1]),
            "path": str(folder),
            "validation_joint_loss": loss,
            "progress_rmse": float(
                np.sqrt(np.mean([r["progress_mse"] for r in records]))
            ),
            "constant_half_rmse": float(
                np.sqrt(np.mean([r["constant_half_mse"] for r in records]))
            ),
            "validation_trajectories": 5,
            "windows": 40,
            "samples": 120,
        }
        (folder / "validation-full.json").write_text(
            json.dumps({"summary": row, "records": records}, indent=2)
        )
        rows.append(row)
        if best is None or loss < best["validation_joint_loss"]:
            best = row
        (root / "validation-selection.json").write_text(
            json.dumps(
                {
                    "criterion": "mean action loss + .1 progress MSE; fixed .1/.5/.9 positions, no test outcomes",
                    "all_checkpoints": rows,
                    "selected": best,
                },
                indent=2,
            )
        )
        print(row, flush=True)
    print("VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
