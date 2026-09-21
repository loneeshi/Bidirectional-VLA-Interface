"""Read-only native24 S2 deployment-parity and head x bank decomposition.

This entry point performs zero optimizer updates and never imports optax.  It
reuses the exact held-out rows and loss RNG contract from ``train_native_s2``;
then, for each H0/B0, H20/B0, H0/B20 and H20/B20 combination, it compares an
in-process validation path with a deployment-adapter path under identical
observations, explicit diffusion noise and RNG keys.

Run only on a clean, pinned author checkout with an external wall timeout.
No simulator is created or stepped by this script.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

from bvi.s2_offline_decomposition import (
    FAMILIES,
    SCHEMA,
    assert_path_parity,
    canonical_sha256,
    combination_spec,
    compare_historical_rows,
    first_action_diagnostics,
    fixed_validation_roster,
    summarize_rows,
)
from bvi.native_s2_deployment import NativeS2DeploymentAdapter
from train_native_s2 import action_context, chunk, load_dataset


AUTHOR_COMMIT = "f4eb160ba52b22c1e85fe432de59c24bbbac6187"
REPORT = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report(status: str, **details):
    if REPORT is not None:
        REPORT.write_text(
            json.dumps({"schema": SCHEMA, "status": status, "optimizer_updates": 0,
                        "simulator_steps": 0, **details}, indent=2) + "\n",
            encoding="utf-8",
        )


def validate_hash(value: str, label: str):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid {label} SHA256")


def main() -> int:
    global REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Frozen native24 family dataset; held-out validation parents only are evaluated")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Frozen native24 S1 OpenPI checkpoint root")
    parser.add_argument("--normalizer", type=Path, required=True,
                        help="Original train-only native24 normalizer directory")
    parser.add_argument("--step0", type=Path, required=True,
                        help="Immutable diagnostic step-0000.pkl")
    parser.add_argument("--step20", type=Path, required=True,
                        help="Immutable diagnostic step-0020.pkl")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--best-record", type=Path, required=True)
    parser.add_argument("--validation-step0", type=Path, required=True)
    parser.add_argument("--validation-step20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--repo-id", default="bvi/s1-official-pick-medium-train")
    parser.add_argument("--wall-seconds", type=int, default=2400)
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--parity-atol", type=float, default=0.0)
    parser.add_argument("--historical-atol", type=float, default=1e-6)
    parser.add_argument("--handoff-manifest", type=Path,
                        help="Frozen bvi.native24-handoff-sequence/1 source for the B-prime behavior gate")
    parser.add_argument("--input-manifest", type=Path,
                        help="Original verified bvi.native24-handoff/1 input gate")
    parser.add_argument("--progress-monitor-source", type=Path,
                        help="Frozen src/bvi/progress_monitor.py used by the behavior gate")
    parser.add_argument("--a-identity", type=Path,
                        help="A decomposition identity.json bound before B-prime model queries")
    parser.add_argument("--a-summary", type=Path,
                        help="A decomposition summary.json bound before B-prime model queries")
    args = parser.parse_args()
    behavior_mode = args.handoff_manifest is not None
    if behavior_mode and any(value is None for value in (
        args.input_manifest, args.progress_monitor_source, args.a_identity, args.a_summary
    )):
        parser.error(
            "--handoff-manifest requires --input-manifest, --progress-monitor-source, "
            "--a-identity and --a-summary"
        )
    if not 60 <= args.wall_seconds <= 3600:
        parser.error("--wall-seconds must be in [60,3600]")
    if not 1 <= args.max_rows <= 120:
        parser.error("--max-rows must be in [1,120]")
    if args.parity_atol < 0 or args.historical_atol < 0:
        parser.error("Tolerances must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    REPORT = args.output / "result.json"
    started = time.monotonic()
    save_report("preflight_no_model_loaded", native_success_evaluated=False,
                capability_admission=False, online_evaluation_authorized=False)

    try:
        records, tables, dataset_identity = load_dataset(args.data)
        roster = fixed_validation_roster(records, tables)
        # ``--max-rows`` binds the active evaluation roster.  The ordinary A
        # audit uses the 69-row training-validation roster; B-prime replaces
        # it with the separately frozen 40-frame handoff sequence roster.
        if not behavior_mode and len(roster) > args.max_rows:
            raise ValueError(f"Fixed held-out roster has {len(roster)} rows, exceeds --max-rows")
        data_manifest = args.data / "manifest.json"
        training_config = json.loads(args.training_config.read_text(encoding="utf-8"))
        checkpoint_record = json.loads(args.checkpoint_record.read_text(encoding="utf-8"))
        best_record = json.loads(args.best_record.read_text(encoding="utf-8"))
        historical0 = json.loads(args.validation_step0.read_text(encoding="utf-8"))
        historical20 = json.loads(args.validation_step20.read_text(encoding="utf-8"))
        evidence_hashes = {
            "data_manifest_sha256": sha256(data_manifest),
            "training_config_sha256": sha256(args.training_config),
            "checkpoint_record_sha256": sha256(args.checkpoint_record),
            "best_record_sha256": sha256(args.best_record),
            "validation_step0_sha256": sha256(args.validation_step0),
            "validation_step20_sha256": sha256(args.validation_step20),
            "step0_artifact_sha256": sha256(args.step0),
            "step20_artifact_sha256": sha256(args.step20),
        }
        if dataset_identity["manifest_sha256"] != evidence_hashes["data_manifest_sha256"]:
            raise ValueError("Dataset loader/manifest hash disagreement")
        if training_config.get("identity", {}).get("manifest_sha256") != dataset_identity["manifest_sha256"]:
            raise ValueError("Training config is not bound to this dataset manifest")
        if best_record.get("step") != 20 or best_record.get("sha256") != evidence_hashes["step20_artifact_sha256"]:
            raise ValueError("step20 artifact differs from immutable selected-best record")
        if checkpoint_record.get("step") != 20:
            raise ValueError("Final checkpoint record is not step20")
        for digest in evidence_hashes.values():
            validate_hash(digest, "evidence")
        behavior_adjudication_sha256 = None
        if behavior_mode:
            from bvi.native24_behavior_gate import (
                combination_specs as behavior_combination_specs,
                rng_contract as behavior_rng_contract,
                runtime_semantics as behavior_runtime_semantics,
            )
            from bvi.native24_handoff_sequence import load_manifest as load_sequence_manifest

            sequence_report, sequence_cases, sequence_tables, sequence_predicates = (
                load_sequence_manifest(
                    args.handoff_manifest, data_manifest, args.input_manifest,
                    args.progress_monitor_source,
                )
            )
            if sequence_report["frame_count"] != 40 or len(sequence_cases) != 4:
                raise ValueError("B-prime requires the exact four-by-ten frozen sequence roster")
            if sequence_report["frame_count"] > args.max_rows:
                raise ValueError("B-prime fixed frame roster exceeds --max-rows")
            a_identity = json.loads(args.a_identity.read_text(encoding="utf-8"))
            a_summary = json.loads(args.a_summary.read_text(encoding="utf-8"))
            if (
                a_identity.get("schema") != SCHEMA
                or a_identity.get("evidence_hashes") != evidence_hashes
                or a_identity.get("dataset_identity", {}).get("manifest_sha256")
                != dataset_identity["manifest_sha256"]
                or a_summary.get("schema") != SCHEMA
                or a_summary.get("deployment_adapter_evaluated") is not True
                or a_summary.get("native_success_evaluated") is not False
                or a_summary.get("capability_admission") is not False
            ):
                raise ValueError("A decomposition identity/summary differs from current artifacts")
            adjudication = {
                "schema": "bvi.native24-handoff-behavior-adjudication/1",
                "status": "frozen_before_model_load_or_query",
                "sequence_manifest": {
                    "sha256": sha256(args.handoff_manifest),
                    "preregistration": json.loads(args.handoff_manifest.read_text(encoding="utf-8"))[
                        "preregistration"
                    ],
                },
                "input_manifest_sha256": sha256(args.input_manifest),
                "training_manifest_sha256": evidence_hashes["data_manifest_sha256"],
                "a_identity": {
                    "identity_sha256": sha256(args.a_identity),
                    "summary_sha256": sha256(args.a_summary),
                    "summary_status": a_summary.get("status"),
                    "frozen_sha256": a_identity.get("frozen_sha256"),
                    "head_sha256": a_identity.get("head_sha256"),
                    "bank_sha256": a_identity.get("bank_sha256"),
                    "normalizer_sha256": a_identity.get("normalizer_sha256"),
                    "state_contract_sha256": a_identity.get("state_contract_sha256"),
                },
                "monitor_contract": sequence_report["monitor_contract"],
                "combinations": behavior_combination_specs(),
                "rng_contract": behavior_rng_contract(),
                "runtime_semantics": behavior_runtime_semantics(),
                "required_facets": [
                    "false_completion", "two_hit", "rollback", "stagnation",
                    "queue_routing",
                ],
                "scope": {
                    "optimizer_updates": 0, "simulator_steps": 0,
                    "transport_evaluated": False, "native_success_evaluated": False,
                    "deployment_cadence_evaluated": False,
                    "positive_two_hit_correctness_evaluable": False,
                    "physical_rollback_correctness_evaluable": False,
                    "physical_stagnation_correctness_evaluable": False,
                    "full_behavior_admission_evaluable": False,
                },
                "sequences": [{
                    **{key: case[key] for key in (
                        "sequence_id", "anchor_case_id", "classification", "family",
                        "instruction", "invocation_index", "parent_episode", "window",
                        "frame_observation_sha256", "sequence_observation_sha256",
                    )},
                    "replan_cooldown": 0,
                    "starts_at_deployed_family_invocation": False,
                    "ordered_physical_predicates": sequence_predicates[case["sequence_id"]],
                    "forbidden_events": ["learned_threshold"],
                } for case in sequence_cases],
            }
            adjudication["adjudication_sha256"] = canonical_sha256(adjudication)
            adjudication_path = args.output / "adjudication.json"
            with adjudication_path.open("x", encoding="utf-8") as stream:
                json.dump(adjudication, stream, indent=2, allow_nan=False); stream.write("\n")
            behavior_adjudication_sha256 = sha256(adjudication_path)
            (args.output / "roster.json").write_text(json.dumps({
                "schema": "bvi.native24-handoff-behavior-roster/1",
                "source": "frozen bvi.native24-handoff-sequence/1",
                "roster_sha256": canonical_sha256(adjudication["sequences"]),
                "frames": sequence_report["frame_count"], "sequences": adjudication["sequences"],
            }, indent=2) + "\n", encoding="utf-8")
            save_report("behavior_adjudication_frozen_before_model_load",
                        evidence_hashes=evidence_hashes,
                        adjudication_file_sha256=behavior_adjudication_sha256,
                        fixed_rows=sequence_report["frame_count"],
                        combinations=behavior_combination_specs())
        else:
            (args.output / "roster.json").write_text(json.dumps({
                "schema": SCHEMA, "source": "train_native_s2 fixed held-out traversal",
                "roster_sha256": canonical_sha256(roster), "rows": roster,
            }, indent=2) + "\n", encoding="utf-8")
            save_report("loading_read_only_artifacts", evidence_hashes=evidence_hashes,
                        fixed_rows=len(roster), combinations=combination_spec())

        # These are trusted, locally produced research checkpoints.  Bind bytes
        # before unpickling and retain no optimizer objects after extraction.
        saved0 = pickle.loads(args.step0.read_bytes())
        saved20 = pickle.loads(args.step20.read_bytes())
        if saved0.get("step") != 0 or saved20.get("step") != 20:
            raise ValueError("Expected exact step0 and step20 checkpoint payloads")
        # JSON object keys are strings, while the trusted pickle preserves the
        # integer seed keys in request_sha256_by_seed.  Compare the canonical
        # JSON identity rather than Python container key types; this preserves
        # every value while accepting that serialization-only conversion.
        config_identity_sha256 = canonical_sha256(training_config.get("identity"))
        if (canonical_sha256(saved0.get("identity")) != config_identity_sha256
                or canonical_sha256(saved20.get("identity")) != config_identity_sha256):
            raise ValueError("Checkpoint payload identity differs from training config")
        if set(saved0.get("banks", {})) != set(FAMILIES) or set(saved20.get("banks", {})) != set(FAMILIES):
            raise ValueError("Checkpoint payload does not contain exactly four family banks")
        banks_by_step = {0: saved0["banks"], 20: saved20["banks"]}
        heads_by_step = {0: saved0["head"], 20: saved20["head"]}
        del saved0, saved20

        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        gpu_map = dict(line.replace(" ", "").split(",") for line in gpu_rows)
        if gpu_map.get("1") != args.gpu_uuid:
            raise ValueError("Read-only audit is restricted to physical GPU1 exact UUID")
        used = int(subprocess.check_output([
            "nvidia-smi", "-i", args.gpu_uuid, "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ], text=True).strip())
        if used >= 1024:
            raise RuntimeError(f"Selected GPU occupied before audit: {used} MiB")
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                          XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2")

        import numpy as np
        import flax.nnx as nnx
        from flax import traverse_util
        import jax
        import jax.numpy as jnp
        import openpi
        from openpi import transforms
        from openpi.models import model as models
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared import nnx_utils
        from openpi.training import checkpoints, config as training_config_module
        from fetch_native_s1_config import config as fetch_config

        author_root = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "-C", str(author_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(author_root), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        if commit != AUTHOR_COMMIT or dirty.strip():
            raise ValueError("Author source must be the clean pinned f4 checkout")
        cfg = fetch_config(args.repo_id, str(args.output / "unused"), str(args.checkpoint), args.normalizer)
        cfg = dataclasses.replace(cfg, progress_loss_weight=0.1)
        mc = dataclasses.replace(cfg.model, enable_progress_head=True)
        if not (mc.action_dim == 32 and mc.action_horizon == 10 and mc.discrete_state_input):
            raise ValueError("Unexpected native24 model shape")
        assets = args.checkpoint / "assets" / args.repo_id
        normalizer_sha256 = sha256(assets / "norm_stats.json")
        state_contract_sha256 = sha256(assets / "bvi-state-contract.json")
        identity = training_config["identity"]
        if normalizer_sha256 != identity.get("normalizer_sha256") or state_contract_sha256 != identity.get("state_contract_sha256"):
            raise ValueError("Normalizer/state contract differs from diagnostic training identity")
        if sha256(args.normalizer / "norm_stats.json") != normalizer_sha256:
            raise ValueError("Original normalizer and checkpoint assets differ")

        stats = checkpoints.load_norm_stats(args.checkpoint / "assets", args.repo_id)
        transform = transforms.compose([
            LiberoInputs(mc.model_type), transforms.Normalize(stats, use_quantiles=True),
            *training_config_module.ModelTransformFactory()(mc).inputs,
        ])
        unnormalize = transforms.Unnormalize(stats, use_quantiles=True)
        lora = nnx_utils.PathRegex(".*lora.*")
        head_filter = nnx_utils.PathRegex(".*progress_chunk.*")
        train_filter = nnx.Any(lora, head_filter)
        reference = traverse_util.flatten_dict(
            nnx.state(nnx.eval_shape(mc.create, jax.random.key(7))).to_pure_dict()
        )
        loaded = models.restore_params(args.checkpoint / "params", restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        is_head = lambda key: any("progress_chunk" in str(part) for part in key)
        missing = set(reference) - set(flat)
        if (set(flat) - set(reference) or not missing or any(not is_head(key) for key in missing)
                or any(is_head(key) for key in flat)
                or any(reference[key].shape != value.shape for key, value in flat.items())):
            raise ValueError("Strict S1 checkpoint mismatch; only progress head may be absent")

        def state_digest(tree) -> str:
            digest = hashlib.sha256()
            pure = tree.to_pure_dict() if hasattr(tree, "to_pure_dict") else tree
            for key, value in sorted(traverse_util.flatten_dict(jax.device_get(pure)).items()):
                array = np.asarray(value)
                digest.update("/".join(map(str, key)).encode())
                digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode())
                digest.update(array.tobytes())
            return digest.hexdigest()

        def observation_digest(observation) -> str:
            digest = hashlib.sha256()
            leaves = jax.tree_util.tree_leaves(jax.device_get(observation))
            for index, value in enumerate(leaves):
                array = np.asarray(value)
                digest.update(str(index).encode()); digest.update(str(array.shape).encode())
                digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
            return digest.hexdigest()

        if state_digest(loaded) != identity.get("pretrained_sha256"):
            raise ValueError("Frozen S1 checkpoint parameter digest mismatch")

        @jax.jit
        def initialize(values):
            model = mc.create(jax.random.key(7))
            graph, variables = nnx.split(model)
            variables.replace_by_pure_dict(values)
            return graph, variables

        graph, variables = initialize(loaded)
        frozen = variables.filter(nnx.Not(train_filter))
        frozen_sha256 = state_digest(frozen)
        if frozen_sha256 != training_config.get("frozen_sha256") or frozen_sha256 != checkpoint_record.get("frozen_sha256"):
            raise ValueError("Frozen partition differs from both training records")
        del loaded, flat, reference, variables
        banks_by_step = {step: {family: jax.device_put(bank) for family, bank in banks.items()}
                         for step, banks in banks_by_step.items()}
        heads_by_step = {step: jax.device_put(head) for step, head in heads_by_step.items()}
        bank_hashes = {step: {family: state_digest(banks_by_step[step][family]) for family in FAMILIES}
                       for step in (0, 20)}
        head_hashes = {step: state_digest(heads_by_step[step]) for step in (0, 20)}
        if bank_hashes[0] != training_config.get("initial_bank_sha256"):
            raise ValueError("step0 banks differ from training config initial banks")
        if bank_hashes[20] != checkpoint_record.get("bank_sha256") or head_hashes[20] != checkpoint_record.get("head_sha256"):
            raise ValueError("step20 banks/head differ from final checkpoint record")
        if behavior_mode and (
            a_identity.get("frozen_sha256") != frozen_sha256
            or canonical_sha256(a_identity.get("head_sha256")) != canonical_sha256(head_hashes)
            or canonical_sha256(a_identity.get("bank_sha256")) != canonical_sha256(bank_hashes)
            or a_identity.get("normalizer_sha256") != normalizer_sha256
            or a_identity.get("state_contract_sha256") != state_contract_sha256
        ):
            raise ValueError("Loaded B-prime model identities differ from frozen A identity")

        def raw_input(record_index, index, include_actions):
            data, record = tables[record_index], records[record_index]
            value = {
                "observation/image": data["head_rgb"][index],
                "observation/wrist_image": data["wrist_rgb"][index],
                "observation/state": data["state"][index],
                "prompt": record["instruction"],
            }
            if include_actions:
                value["actions"] = action_context(data, index)
            return value

        def make_training_sample(record_index, index):
            data = tables[record_index]
            indices, action_valid, progress_valid = chunk(data, index)
            value = transform(raw_input(record_index, index, True))
            actions = value.pop("actions")
            observation = models.Observation.from_dict(
                jax.tree.map(lambda item: jnp.asarray(item)[None], value)
            )
            return (observation, jnp.asarray(actions)[None],
                    jnp.asarray(data["progress"][indices])[None],
                    jnp.asarray(action_valid)[None], jnp.asarray(progress_valid)[None], value)

        @jax.jit
        def evaluate(params, frozen_values, key, observation, actions, target, amask, pmask):
            model = nnx.merge(graph, nnx.State.merge(frozen_values, params))
            action_tokens, progress = model.compute_action_and_progress_chunk_prefix(
                key, observation, actions, train=False
            )
            action_loss = jnp.sum(action_tokens * amask) / jnp.maximum(jnp.sum(amask), 1)
            progress_loss = jnp.sum(jnp.square(progress - target) * pmask) / jnp.maximum(jnp.sum(pmask), 1)
            return action_loss, progress_loss, action_loss + 0.1 * progress_loss

        @jax.jit
        def validation_infer(params, frozen_values, key, observation, noise):
            model = nnx.merge(graph, nnx.State.merge(frozen_values, params))
            return model.infer_actions_and_progress(key, observation, noise=noise, num_steps=10)

        # Deliberately compiled as a separate call surface.  The transport-free
        # deployment adapter below owns request/routing/queue/response semantics
        # and is the implementation a future socket service must wrap.
        @jax.jit
        def deployment_infer(params, frozen_values, key, observation, noise):
            model = nnx.merge(graph, nnx.State.merge(frozen_values, params))
            return model.infer_actions_and_progress(key, observation, noise=noise, num_steps=10)

        if behavior_mode:
            from bvi.native24_behavior_gate import (
                adjudicate_behavior_rows,
                combination_specs as behavior_combination_specs,
                progress_scalar,
                roster_seeds,
                run_monitor_sequence,
            )
            from bvi.progress_monitor import ProgressMonitor

            behavior_rows, behavior_queue_events = [], []
            behavior_predictions = {}
            expected_model_calls = sequence_report["frame_count"] * len(
                behavior_combination_specs()
            )
            if expected_model_calls != 200:
                raise ValueError("B-prime fixed matrix must contain exactly 200 model calls")
            save_report(
                "evaluating_zero_training_behavior_matrix",
                schema="bvi.native24-handoff-behavior-result/1",
                adjudication_file_sha256=behavior_adjudication_sha256,
                expected_model_calls=expected_model_calls,
                optimizer_updates=0, simulator_steps=0,
                native_success_evaluated=False, capability_admission=False,
            )
            for combination in behavior_combination_specs():
                name = combination["name"]
                head_step = combination["head_step"]
                selected_bank_steps = combination["bank_steps"]
                selected_bank_hashes = {
                    family: bank_hashes[selected_bank_steps[family]][family]
                    for family in FAMILIES
                }
                params_by_family = {
                    family: nnx.State.merge(
                        banks_by_step[selected_bank_steps[family]][family],
                        heads_by_step[head_step],
                    ) for family in FAMILIES
                }
                combination_sha256 = canonical_sha256({
                    "frozen": frozen_sha256, "head": head_hashes[head_step],
                    "banks": selected_bank_hashes, "head_step": head_step,
                    "bank_steps": selected_bank_steps,
                })

                def adapter_predict(family, key, observation, noise):
                    return jax.device_get(deployment_infer(
                        params_by_family[family], frozen, key, observation, noise
                    ))

                adapter = NativeS2DeploymentAdapter(
                    transform=transform,
                    observation_factory=lambda value: models.Observation.from_dict(
                        jax.tree.map(lambda item: jnp.asarray(item)[None], value)
                    ),
                    predict=adapter_predict,
                    unnormalize=unnormalize,
                    observation_digest=observation_digest,
                    bank_sha256=selected_bank_hashes,
                    head_sha256=head_hashes[head_step],
                    checkpoint_sha256=combination_sha256,
                    normalizer_sha256=normalizer_sha256,
                    state_contract_sha256=state_contract_sha256,
                )
                combo_normalized, combo_external, combo_progress = [], [], []
                for sequence_index, case in enumerate(sequence_cases):
                    if time.monotonic() - started >= args.wall_seconds:
                        raise TimeoutError("Internal B-prime read-only audit wall limit reached")
                    sequence_id, family = case["sequence_id"], case["family"]
                    values = sequence_tables[sequence_id]
                    predicates = sequence_predicates[sequence_id]
                    call_id = f"{name}-{sequence_id}"
                    begin_event = adapter.begin(
                        call_id=call_id, family=family, instruction=case["instruction"]
                    )
                    if (
                        begin_event["discarded_actions"] != 0
                        or begin_event["queue_empty_before_inference"] is not True
                        or begin_event["selected_bank_sha256"] != selected_bank_hashes[family]
                    ):
                        raise ValueError("B-prime sequence begin/routing semantics differ")
                    monitor = ProgressMonitor(family, case["invocation_index"], 0)
                    terminal_event = None
                    progress_chunks, rng_rows = [], []
                    physical_completion = []
                    transformed_observation_sha256 = []
                    for frame_index in range(case["window"]["length"]):
                        if adapter.pending:
                            raise ValueError("B-prime queue was not empty before inference")
                        rng_row = roster_seeds(sequence_index, frame_index)
                        sample_key = jax.random.PRNGKey(rng_row["sample_rng_seed"])
                        noise = jax.random.normal(
                            jax.random.PRNGKey(rng_row["noise_rng_seed"]), (1, 10, 32)
                        )
                        noise_array = np.asarray(jax.device_get(noise))
                        noise_sha256 = hashlib.sha256(noise_array.tobytes()).hexdigest()
                        response = adapter.infer({
                            "head_rgb": values["head_rgb"][frame_index],
                            "wrist_rgb": values["wrist_rgb"][frame_index],
                            "state": values["state"][frame_index],
                            "prompt": case["instruction"],
                            "tool_family": family, "call_id": call_id,
                        }, rng_key=sample_key, noise=noise)
                        if (
                            response["adapter_sha256"] != selected_bank_hashes[family]
                            or response["progress_head_sha256"] != head_hashes[head_step]
                            or response["checkpoint_sha256"] != combination_sha256
                            or response["normalizer_sha256"] != normalizer_sha256
                            or response["state_contract_sha256"] != state_contract_sha256
                            or response["queued_actions"] != 10
                            or response["transport_evaluated"] is not False
                        ):
                            raise ValueError("B-prime response routing/identity differs")
                        scalar, source_dtype = progress_scalar(response["progress"])
                        if not 0.0 <= scalar <= 1.0:
                            raise ValueError("B-prime monitor scalar is outside [0,1]")
                        if terminal_event is None:
                            event = monitor.update(scalar)
                            reason = event or (
                                "shadow_sequence_end"
                                if frame_index == case["window"]["length"] - 1
                                else "shadow_advance"
                            )
                            if event is not None:
                                terminal_event = event
                        else:
                            event = None
                            reason = "post_terminal_diagnostic_only"
                        clear_event = adapter.clear(reason)
                        if clear_event["discarded_actions"] != 10 or clear_event[
                            "queue_empty"
                        ] is not True:
                            raise ValueError("B-prime queue clear did not discard one full chunk")
                        behavior_queue_events.append({
                            "combination": name, "sequence_id": sequence_id,
                            "prediction_index": frame_index, **clear_event,
                        })
                        rng_rows.append({**rng_row, "noise_sha256": noise_sha256})
                        progress_chunks.append(np.asarray(response["progress"]).copy())
                        physical_completion.append(predicates[frame_index]["physical_completion"])
                        transformed_observation_sha256.append(response["observation_sha256"])
                        combo_normalized.append(np.asarray(response["normalized_actions"][0]))
                        combo_external.append(np.asarray(response["actions"]))
                        combo_progress.append(np.asarray(response["progress"][0]))
                    monitor_trace = run_monitor_sequence(
                        family, case["invocation_index"], progress_chunks, cooldown=0
                    )
                    if monitor_trace["terminal_event"] != terminal_event:
                        raise ValueError("Live and replayed B-prime monitor decisions differ")
                    behavior_rows.append({
                        "combination": name, "sequence_id": sequence_id,
                        "sequence_index": sequence_index,
                        "classification": case["classification"], "family": family,
                        "head_step": head_step, "bank_step": selected_bank_steps[family],
                        "head_sha256": head_hashes[head_step],
                        "bank_sha256": selected_bank_hashes[family],
                        "combination_sha256": combination_sha256,
                        "monitor": monitor_trace,
                        "rng_seeds": [{key: value for key, value in row.items()
                                       if key != "noise_sha256"} for row in rng_rows],
                        "noise_sha256": [row["noise_sha256"] for row in rng_rows],
                        "physical_completion": physical_completion,
                        "ordered_observation_sha256": case["frame_observation_sha256"],
                        "transformed_observation_sha256": transformed_observation_sha256,
                    })
                behavior_predictions[f"{name}_normalized_actions"] = np.asarray(combo_normalized)
                behavior_predictions[f"{name}_external_actions"] = np.asarray(combo_external)
                behavior_predictions[f"{name}_progress"] = np.asarray(combo_progress)

            gate = adjudicate_behavior_rows(
                behavior_rows, behavior_queue_events,
                [case["sequence_id"] for case in sequence_cases],
            )
            # The same roster row must have the same preregistered RNG/noise in
            # all candidates; comparison happens after all calls, never by
            # selecting favorable frames or rerunning a candidate.
            noise_by_row = {}
            transformed_by_row = {}
            for row in behavior_rows:
                for frame_index, digest in enumerate(row["noise_sha256"]):
                    key = (row["sequence_id"], frame_index)
                    noise_by_row.setdefault(key, set()).add(digest)
                for frame_index, digest in enumerate(row["transformed_observation_sha256"]):
                    key = (row["sequence_id"], frame_index)
                    transformed_by_row.setdefault(key, set()).add(digest)
            if any(len(values) != 1 for values in noise_by_row.values()):
                raise ValueError("B-prime candidates did not share identical explicit noise")
            if any(len(values) != 1 for values in transformed_by_row.values()):
                raise ValueError("B-prime candidates did not share identical transformed observations")
            if sha256(adjudication_path) != behavior_adjudication_sha256:
                raise ValueError("B-prime adjudication changed after model queries")
            with (args.output / "rows.jsonl").open("w", encoding="utf-8") as stream:
                for row in behavior_rows:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            np.savez_compressed(args.output / "predictions.npz", **behavior_predictions)
            (args.output / "queue-audit.json").write_text(json.dumps({
                "passed": gate["checks"]["queue_clear"]["status"] == "passed",
                "deployment_adapter_evaluated": True,
                "deployment_transport_evaluated": False,
                "events": behavior_queue_events,
            }, indent=2) + "\n", encoding="utf-8")
            behavior_identity = {
                "schema": "bvi.native24-handoff-behavior-identity/1",
                "author_commit": commit,
                "audit_source_sha256": sha256(Path(__file__)),
                "behavior_core_source_sha256": sha256(
                    Path(inspect.getfile(__import__(
                        "bvi.native24_behavior_gate", fromlist=["native24_behavior_gate"]
                    )))
                ),
                "sequence_core_source_sha256": sha256(
                    Path(inspect.getfile(__import__(
                        "bvi.native24_handoff_sequence", fromlist=["native24_handoff_sequence"]
                    )))
                ),
                "deployment_adapter_source_sha256": sha256(
                    Path(inspect.getfile(__import__(
                        "bvi.native_s2_deployment", fromlist=["native_s2_deployment"]
                    )))
                ),
                "adjudication_file_sha256": behavior_adjudication_sha256,
                "sequence_manifest_sha256": sequence_report["manifest_sha256"],
                "a_identity_sha256": sha256(args.a_identity),
                "a_summary_sha256": sha256(args.a_summary),
                "frozen_sha256": frozen_sha256, "head_sha256": head_hashes,
                "bank_sha256": bank_hashes, "normalizer_sha256": normalizer_sha256,
                "state_contract_sha256": state_contract_sha256,
            }
            (args.output / "identity.json").write_text(
                json.dumps(behavior_identity, indent=2) + "\n", encoding="utf-8"
            )
            final_status = (
                "invalid_behavior_evidence"
                if gate["full_gate_status"] == "invalid"
                else "not_evaluable_full_behavior_admission"
            )
            summary = {
                **gate,
                "status": final_status,
                "learned_signal_mechanics_status": gate["offline_behavior_status"],
                "model_inference_calls": expected_model_calls,
                "optimizer_updates": 0, "simulator_steps": 0,
                "native_success_evaluated": False, "capability_admission": False,
                "online_evaluation_authorized": False,
                "interpretation_scope": (
                    "counterfactual wrong-family shadow queries on source-action causal tails; "
                    "not deployed cadence or physical recovery"
                ),
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            artifact_hashes = {name: sha256(args.output / name) for name in (
                "adjudication.json", "identity.json", "roster.json", "rows.jsonl",
                "predictions.npz", "queue-audit.json", "summary.json",
            )}
            save_report(
                final_status,
                schema="bvi.native24-handoff-behavior-result/1",
                wall_seconds=time.monotonic() - started,
                evaluated_model_calls=expected_model_calls,
                offline_behavior_status=gate["offline_behavior_status"],
                checks=gate["checks"], artifact_hashes=artifact_hashes,
                native_success_evaluated=False, capability_admission=False,
                online_evaluation_authorized=False,
            )
            print(json.dumps({
                "status": final_status,
                "offline_behavior_status": gate["offline_behavior_status"],
                "model_calls": expected_model_calls, "output": str(args.output),
            }))
            return 2 if gate["full_gate_status"] == "invalid" else 0

        queue_events = []
        rows = []
        predictions = {}
        save_report("evaluating_read_only", fixed_rows=len(roster), total_model_rows=len(roster) * 4,
                    evidence_hashes=evidence_hashes)
        for combination in combination_spec():
            name = combination["name"]
            head_step, bank_step = combination["head_step"], combination["bank_step"]
            params_by_family = {
                family: nnx.State.merge(banks_by_step[bank_step][family], heads_by_step[head_step])
                for family in FAMILIES
            }
            combination_sha256 = canonical_sha256({
                "frozen": frozen_sha256, "head": head_hashes[head_step],
                "banks": bank_hashes[bank_step], "head_step": head_step, "bank_step": bank_step,
            })

            def adapter_predict(family, key, observation, noise):
                return jax.device_get(deployment_infer(
                    params_by_family[family], frozen, key, observation, noise
                ))

            adapter = NativeS2DeploymentAdapter(
                transform=transform,
                observation_factory=lambda value: models.Observation.from_dict(
                    jax.tree.map(lambda item: jnp.asarray(item)[None], value)
                ),
                predict=adapter_predict,
                unnormalize=unnormalize,
                observation_digest=observation_digest,
                bank_sha256=bank_hashes[bank_step],
                head_sha256=head_hashes[head_step],
                checkpoint_sha256=combination_sha256,
                normalizer_sha256=normalizer_sha256,
                state_contract_sha256=state_contract_sha256,
            )
            combo_normalized, combo_external, combo_progress = [], [], []
            for roster_row in roster:
                if time.monotonic() - started >= args.wall_seconds:
                    raise TimeoutError("Internal read-only audit wall limit reached")
                window, index, family = roster_row["window"], roster_row["index"], roster_row["family"]
                params = nnx.State.merge(banks_by_step[bank_step][family], heads_by_step[head_step])
                observation, actions, target, amask, pmask, transformed = make_training_sample(window, index)
                loss_values = jax.device_get(evaluate(
                    params, frozen, jax.random.PRNGKey(roster_row["loss_rng_seed"]),
                    observation, actions, target, amask, pmask,
                ))
                noise = jax.random.normal(
                    jax.random.PRNGKey(roster_row["noise_rng_seed"]), (1, 10, 32)
                )
                sample_key = jax.random.PRNGKey(roster_row["sample_rng_seed"])
                normalized, progress = jax.device_get(
                    validation_infer(params, frozen, sample_key, observation, noise)
                )
                external = np.asarray(unnormalize({
                    "actions": np.asarray(normalized[0]), "state": np.asarray(transformed["state"]),
                })["actions"][:, :13])
                call_id = f"{name}-{roster_row['row_id']}"
                record = records[window]
                queue_event = adapter.begin(
                    call_id=call_id, family=family, instruction=record["instruction"]
                )
                response = adapter.infer({
                    "head_rgb": tables[window]["head_rgb"][index],
                    "wrist_rgb": tables[window]["wrist_rgb"][index],
                    "state": tables[window]["state"][index],
                    "prompt": record["instruction"],
                    "tool_family": family,
                    "call_id": call_id,
                }, rng_key=sample_key, noise=noise)
                deployed_normalized = response["normalized_actions"]
                deployed_progress = response["progress"]
                deployed_external = response["actions"]
                training_path = {
                    "observation_sha256": observation_digest(observation),
                    "normalized_actions": np.asarray(normalized),
                    "external_actions": external,
                    "progress": np.asarray(progress),
                }
                deployment_path = {
                    "observation_sha256": response["observation_sha256"],
                    "normalized_actions": np.asarray(deployed_normalized),
                    "external_actions": deployed_external,
                    "progress": np.asarray(deployed_progress),
                }
                parity = assert_path_parity(training_path, deployment_path, atol=args.parity_atol)
                queue_event.update(
                    call_id=call_id, combination=name,
                    queued_actions_after_inference=response["queued_actions"],
                    response_adapter_sha256=response["adapter_sha256"],
                    response_progress_head_sha256=response["progress_head_sha256"],
                    response_combination_sha256=response["checkpoint_sha256"],
                    transport_evaluated=response["transport_evaluated"],
                )
                queue_events.append(queue_event)
                diagnostic = first_action_diagnostics(
                    deployed_external[0], tables[window]["actions"][index],
                    eligible=roster_row["action_label_valid"],
                )
                row = {
                    **roster_row, "combination": name, "head_step": head_step,
                    "bank_step": bank_step, "head_sha256": head_hashes[head_step],
                    "bank_sha256": bank_hashes[bank_step][family],
                    "action_loss": float(loss_values[0]),
                    "progress_loss": float(loss_values[1]),
                    "joint_loss": float(loss_values[2]),
                    "deployment_parity": parity,
                    "first_action": diagnostic,
                }
                rows.append(row)
                combo_normalized.append(np.asarray(normalized[0]))
                combo_external.append(external)
                combo_progress.append(np.asarray(progress[0]))
            predictions[f"{name}_normalized_actions"] = np.asarray(combo_normalized)
            predictions[f"{name}_external_actions"] = np.asarray(combo_external)
            predictions[f"{name}_progress"] = np.asarray(combo_progress)

        historical_reproduction = {
            "H0_B0": compare_historical_rows(
                [row for row in rows if row["combination"] == "H0_B0"], historical0,
                atol=args.historical_atol,
            ),
            "H20_B20": compare_historical_rows(
                [row for row in rows if row["combination"] == "H20_B20"], historical20,
                atol=args.historical_atol,
            ),
        }
        summary = summarize_rows(rows)
        with (args.output / "rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        np.savez_compressed(args.output / "predictions.npz", **predictions)
        (args.output / "queue-audit.json").write_text(json.dumps({
            "passed": all(event["queue_empty_before_inference"] for event in queue_events),
            "deployment_adapter_evaluated": True,
            "deployment_transport_evaluated": False,
            "events": queue_events,
        }, indent=2) + "\n", encoding="utf-8")
        identity_report = {
            "schema": SCHEMA,
            "author_commit": commit,
            "audit_source_sha256": sha256(Path(__file__)),
            "core_source_sha256": sha256(Path(inspect.getfile(__import__(
                "bvi.s2_offline_decomposition", fromlist=["s2_offline_decomposition"]
            )))),
            "deployment_adapter_source_sha256": sha256(
                Path(inspect.getfile(__import__(
                    "bvi.native_s2_deployment", fromlist=["native_s2_deployment"]
                )))
            ),
            "evidence_hashes": evidence_hashes,
            "dataset_identity": dataset_identity,
            "frozen_sha256": frozen_sha256,
            "head_sha256": head_hashes,
            "bank_sha256": bank_hashes,
            "normalizer_sha256": normalizer_sha256,
            "state_contract_sha256": state_contract_sha256,
            "roster_sha256": canonical_sha256(roster),
            "rng_contract": "loss=PRNGKey(123+window*1000+index); sample=PRNGKey(700123+window*1000+index); explicit_noise=PRNGKey(900123+window*1000+index)",
        }
        (args.output / "identity.json").write_text(json.dumps(identity_report, indent=2) + "\n", encoding="utf-8")
        (args.output / "summary.json").write_text(json.dumps({
            "schema": SCHEMA,
            "status": "passed_in_process_deployment_adapter_parity_transport_not_evaluated",
            "historical_reproduction": historical_reproduction,
            "deployment_parity_rows": len(rows), "all_deployment_parity_passed": True,
            "deployment_adapter_evaluated": True,
            "deployment_transport_evaluated": False,
            "combination_summary": summary,
            "interpretation_scope": "offline fixed-row loss, sampled first-action diagnostics, and deployment adapter parity only",
            "native_success_evaluated": False, "capability_admission": False,
        }, indent=2) + "\n", encoding="utf-8")
        artifact_hashes = {
            name: sha256(args.output / name) for name in (
                "identity.json", "roster.json", "rows.jsonl", "predictions.npz",
                "queue-audit.json", "summary.json",
            )
        }
        save_report("passed_in_process_deployment_adapter_parity_transport_not_evaluated",
                    wall_seconds=time.monotonic() - started, fixed_rows=len(roster),
                    evaluated_model_rows=len(rows), evidence_hashes=evidence_hashes,
                    artifact_hashes=artifact_hashes, native_success_evaluated=False,
                    capability_admission=False, bounded_sft_expansion_authorized=False,
                    online_evaluation_authorized=False)
        print(json.dumps({"status": "passed_in_process_deployment_adapter_parity_transport_not_evaluated",
                          "rows": len(rows), "output": str(args.output)}))
        return 0
    except Exception as exc:
        save_report("failed", error=repr(exc), wall_seconds=time.monotonic() - started,
                    native_success_evaluated=False, capability_admission=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
