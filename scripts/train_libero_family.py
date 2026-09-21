"""Small invocation-aligned LoRA/progress pilot using the pinned author OpenPI.

One frozen backbone, four residual banks, one shared progress head. This omits
DROID intermediate post-training. Run inside the author fork's environment.
"""

import argparse, dataclasses, hashlib, io, json, pathlib, pickle, time
import flax.nnx as nnx
from flax import traverse_util
import jax, jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from PIL import Image
from openpi.models import model as models, pi0_config
from openpi.shared import nnx_utils
from openpi.training import config, checkpoints, weight_loaders
from openpi import transforms
from openpi.policies.libero_policy import LiberoInputs
from family_language import instruction_variants

FAMILIES = ("reach", "grasp", "move", "release")
LORA = nnx_utils.PathRegex(".*lora.*")
HEAD = nnx_utils.PathRegex(".*progress_chunk.*")
TRAIN = nnx.Any(LORA, HEAD)


def model_config():
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        enable_progress_head=True,
    )


def initialize(checkpoint):
    cfg = model_config()

    def init(key, loaded):
        model = cfg.create(key)
        graph, state = nnx.split(model)
        state.replace_by_pure_dict(loaded)
        frozen = state.filter(nnx.Not(TRAIN))
        frozen = jax.tree.map(lambda x: x.astype(jnp.bfloat16), frozen)
        return graph, nnx.State.merge(frozen, state.filter(TRAIN))

    shape = nnx.state(nnx.eval_shape(cfg.create, jax.random.key(7))).to_pure_dict()
    loaded = weight_loaders.CheckpointWeightLoader(
        str(pathlib.Path(checkpoint) / "params")
    ).load(shape)
    flat = {
        k: v
        for k, v in traverse_util.flatten_dict(loaded).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    }
    return jax.jit(init)(jax.random.key(7), traverse_util.unflatten_dict(flat))


def transform(checkpoint):
    stats = checkpoints.load_norm_stats(
        pathlib.Path(checkpoint) / "assets", "physical-intelligence/libero"
    )
    mc = model_config()
    return transforms.compose(
        [
            LiberoInputs(mc.model_type),
            transforms.Normalize(stats, use_quantiles=True),
            *config.ModelTransformFactory()(mc).inputs,
        ]
    )


class Samples:
    def __init__(self, data, checkpoint, language_augmentation=False):
        self.windows = json.loads((pathlib.Path(data) / "invocations.json").read_text())
        self.tables = {
            path: pd.read_parquet(path)
            for path in sorted({w["path"] for w in self.windows})
        }
        self.transform = transform(checkpoint)
        self.rng = np.random.default_rng(7)
        self.language_rng = np.random.default_rng(708)
        self.language_augmentation = language_augmentation

    def get(self, split, family, validation=False, window_index=0, position=None):
        choices = [
            w for w in self.windows if w["split"] == split and w["family"] == family
        ]
        w = (
            choices[window_index]
            if validation
            else choices[self.rng.integers(len(choices))]
        )
        if validation:
            i = (
                (w["start"] + w["end"]) // 2
                if position is None
                else w["start"] + round((w["end"] - w["start"] - 1) * position)
            )
        else:
            i = int(self.rng.integers(w["start"], w["end"]))
        d = self.tables[w["path"]]
        row = d.iloc[i]
        prompt = w["instruction"]
        if self.language_augmentation and split == "train":
            variants = instruction_variants(w["family"], w["target"], prompt)
            prompt = variants[int(self.language_rng.integers(len(variants)))]
        indices = np.minimum(np.arange(i, i + 10), w["end"] - 1)

        def image(k):
            return np.asarray(Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB"))

        x = self.transform(
            {
                "observation/image": image("image"),
                "observation/wrist_image": image("wrist_image"),
                "observation/state": np.asarray(row.state),
                "actions": np.stack(d.iloc[indices].actions),
                "prompt": prompt,
            }
        )
        action = x.pop("actions")
        x = jax.tree.map(lambda y: jnp.asarray(y)[None], x)
        progress = (indices - w["start"]) / max(w["end"] - w["start"] - 1, 1)
        mask = (np.arange(i, i + 10) < w["end"]).astype(np.float32)
        return (
            models.Observation.from_dict(x),
            jnp.asarray(action)[None],
            jnp.asarray(progress, dtype=jnp.float32)[None],
            jnp.asarray(mask)[None],
        )


def save(path, step, banks, head, opts, head_opt, metrics):
    path.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(
        dict(step=step, banks=banks, head=head, opts=opts, head_opt=head_opt)
    )
    file = path / "adapters.pkl"
    file.write_bytes(
        pickle.dumps(
            {k: v for k, v in payload.items() if k not in ("opts", "head_opt")}
        )
    )
    temporary = path.parent / "resume.tmp"
    temporary.write_bytes(pickle.dumps(payload))
    temporary.replace(path.parent / "resume.pkl")
    meta = dict(
        step=step,
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        metrics=metrics,
        families=FAMILIES,
        backbone_frozen=True,
        initialization="official pi05_libero",
        progress_weight=0.1,
        droid_post_training=False,
        label_source="reviewed temporal proxy",
        micro_batch=1,
        effective_batch=8,
    )
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seconds", type=int, default=7200)
    p.add_argument("--resume")
    p.add_argument("--warmstart-adapters")
    p.add_argument("--language-augmentation", action="store_true")
    a = p.parse_args()
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    assert 1 <= a.steps <= 2000 and 1 <= a.seconds <= 7200
    if a.resume and a.warmstart_adapters:
        raise ValueError("Choose optimizer resume or weights-only warmstart")
    graph, state = initialize(a.checkpoint)
    frozen = state.filter(nnx.Not(TRAIN))
    head = state.filter(HEAD)
    banks = {g: state.filter(LORA) for g in FAMILIES}
    del state
    tx = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(5e-5, weight_decay=0.0)
    )
    opts = {g: tx.init(banks[g]) for g in FAMILIES}
    head_opt = tx.init(head)
    start_step = 0
    if a.warmstart_adapters:
        source = pathlib.Path(a.warmstart_adapters)
        saved = pickle.loads(source.read_bytes())
        banks = jax.device_put(saved["banks"])
        head = jax.device_put(saved["head"])
        if set(banks) != set(FAMILIES):
            raise ValueError("Warmstart must contain all four families")
        opts = {g: tx.init(banks[g]) for g in FAMILIES}
        head_opt = tx.init(head)
    (out / "training-config.json").write_text(
        json.dumps(
            {
                "arguments": vars(a),
                "optimizer_reset": bool(a.warmstart_adapters),
                "parent_sha256": hashlib.sha256(
                    pathlib.Path(a.warmstart_adapters).read_bytes()
                ).hexdigest()
                if a.warmstart_adapters
                else None,
                "parent_step": saved["step"] if a.warmstart_adapters else None,
                "new_recovery_trajectories": 0,
                "validation_language": "unchanged original",
            },
            indent=2,
        )
    )
    if a.resume:
        rp = pathlib.Path(a.resume)
        saved = pickle.loads((rp if rp.is_file() else rp / "adapters.pkl").read_bytes())
        banks = jax.device_put(saved["banks"])
        head = jax.device_put(saved["head"])
        opts = jax.device_put(saved["opts"])
        head_opt = jax.device_put(saved["head_opt"])
        start_step = saved["step"]
    samples = Samples(a.data, a.checkpoint, a.language_augmentation)
    if a.language_augmentation:
        reviewed = {
            w["instruction"]: instruction_variants(
                w["family"], w["target"], w["instruction"]
            )
            for w in samples.windows
            if w["split"] == "train"
        }
        (out / "language-variants.json").write_text(json.dumps(reviewed, indent=2))

    def loss(params, frozen, key, batch):
        model = nnx.merge(graph, nnx.State.merge(frozen, params))
        obs, actions, target, mask = batch
        action_loss, progress = model.compute_action_and_progress_chunk_prefix(
            key, obs, actions, train=False
        )
        al = jnp.sum(action_loss * mask) / jnp.sum(mask)
        pl = jnp.sum(jnp.square(progress - target) * mask) / jnp.sum(mask)
        return al + 0.1 * pl, jnp.stack([al, pl])

    grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    evaluate = jax.jit(loss)
    started = time.monotonic()
    best = float("inf")
    last_metrics = {}
    step = 0
    if start_step or a.warmstart_adapters:
        vals = []
        for f in FAMILIES:
            n = sum(
                w["split"] == "validation" and w["family"] == f for w in samples.windows
            )
            vals.extend(
                float(
                    evaluate(
                        nnx.State.merge(banks[f], head),
                        frozen,
                        jax.random.key(123 + wi),
                        samples.get("validation", f, True, wi),
                    )[0]
                )
                for wi in range(n)
            )
        best = float(np.mean(vals))
        last_metrics = {
            "validation_joint_loss": best,
            "validation_trajectories": 5,
            "validation_windows": 40,
        }
        save(
            out / f"step-{start_step:04d}",
            start_step,
            banks,
            head,
            opts,
            head_opt,
            last_metrics,
        )
        (out / "best.json").write_text(
            json.dumps(
                {
                    "step": start_step,
                    "path": str(out / f"step-{start_step:04d}"),
                    "loss": best,
                }
            )
        )
    print("TRAIN READY", flush=True)
    with (out / "train.jsonl").open("a") as log:
        for step in range(start_step + 1, a.steps + 1):
            if time.monotonic() - started > a.seconds:
                step -= 1
                break
            g = FAMILIES[(step - 1) % 4]
            params = nnx.State.merge(banks[g], head)
            acc = None
            values = []
            if step <= start_step + 4:

                def digest(tree):
                    h = hashlib.sha256()
                    for leaf in jax.tree.leaves(jax.device_get(tree)):
                        h.update(np.asarray(leaf).tobytes())
                    return h.hexdigest()

                before = {f: digest(banks[f]) for f in FAMILIES}
            for micro in range(8):
                (value, aux), grads = grad(
                    params,
                    frozen,
                    jax.random.key(step * 8 + micro),
                    samples.get("train", g),
                )
                acc = (
                    grads
                    if acc is None
                    else jax.tree.map(lambda x, y: x + y, acc, grads)
                )
                values.append(float(value))
            acc = jax.tree.map(lambda x: x / 8, acc)
            update, opts[g] = tx.update(acc.filter(LORA), opts[g], banks[g])
            banks[g] = optax.apply_updates(banks[g], update)
            update, head_opt = tx.update(acc.filter(HEAD), head_opt, head)
            head = optax.apply_updates(head, update)
            if step <= start_step + 4:
                after = {f: digest(banks[f]) for f in FAMILIES}
                assert before[g] != after[g] and all(
                    before[f] == after[f] for f in FAMILIES if f != g
                )
                (out / f"update-audit-{g}.json").write_text(
                    json.dumps(
                        {
                            "selected": g,
                            "before": before,
                            "after": after,
                            "frozen_parameters_in_optimizer": False,
                        }
                    )
                )
            record = {
                "step": step,
                "family": g,
                "loss": float(np.mean(values)),
                "seconds": time.monotonic() - started,
            }
            if not np.isfinite(record["loss"]):
                raise ValueError("Nonfinite training loss")
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 5 == 0:
                print(record, flush=True)
            if step % 200 == 0 or step == 20 or step == a.steps:
                val = []
                for family in FAMILIES:
                    family_values = []
                    n = sum(
                        w["split"] == "validation" and w["family"] == family
                        for w in samples.windows
                    )
                    for wi in range(n):
                        result = evaluate(
                            nnx.State.merge(banks[family], head),
                            frozen,
                            jax.random.key(123 + wi),
                            samples.get("validation", family, True, wi),
                        )
                        family_values.append(float(result[0]))
                    val.append(float(np.mean(family_values)))
                last_metrics = {
                    "validation_joint_loss": float(np.mean(val)),
                    "family_validation": dict(zip(FAMILIES, val)),
                    "validation_trajectories": 5,
                    "validation_windows": 40,
                }
                save(
                    out / f"step-{step:04d}",
                    step,
                    banks,
                    head,
                    opts,
                    head_opt,
                    last_metrics,
                )
                if last_metrics["validation_joint_loss"] < best:
                    best = last_metrics["validation_joint_loss"]
                    (out / "best.json").write_text(
                        json.dumps(
                            {
                                "step": step,
                                "path": str(out / f"step-{step:04d}"),
                                "loss": best,
                            }
                        )
                    )
        save(out / "last", step, banks, head, opts, head_opt, last_metrics)
    print("TRAIN COMPLETE", step, flush=True)


if __name__ == "__main__":
    main()
