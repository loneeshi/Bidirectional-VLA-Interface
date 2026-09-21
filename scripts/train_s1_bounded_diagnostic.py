"""Path-B diagnostic only: frozen best/855 parameters, new optimizer, <=100 updates.

The Linux supervisor enforces <=3600 seconds INCLUDING imports, source hashing,
loading, compilation, evaluation and saving. No automatic resume or candidate.
Panel schema: schema_version=1, checkpoint_step=855, source_manifest_sha256,
source_sha256, normalizer_sha256, rows=[{parent_id,call_index,observation_index}].
Run --preflight for CPU-only provenance/panel validation (no model/GPU imports).
"""
import argparse
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GPU1_UUID = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'
REPO_ID = 'bvi/s1-official-pick-medium-train'
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_panel(panel, manifest, rows, provenance):
    if panel.get('schema_version') != 1 or panel.get('checkpoint_step') != 855:
        raise ValueError('Expected schema_version=1 and checkpoint_step=855')
    for field in ('source_manifest_sha256', 'source_sha256', 'normalizer_sha256'):
        if panel.get(field) != provenance[field]:
            raise ValueError('Panel provenance mismatch: ' + field)
    requested = panel.get('rows', [])
    if not 64 <= len(requested) <= 128:
        raise ValueError('Diagnostic requires 64-128 unique rows')
    def key(row):
        values = tuple(row[name] for name in ('parent_id', 'call_index', 'observation_index'))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError('Panel row indices must be nonnegative integers')
        return values
    keys = [key(row) for row in requested]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate panel row')
    lookup = {key(row): row for row in rows}
    if any(k not in lookup for k in keys):
        raise ValueError('Panel row absent from action-supervised source')
    selected = [lookup[k] for k in keys]
    parents = {row['parent_id'] for row in selected}
    if not 1 <= len(parents) <= 2:
        raise ValueError('At most two train parents permitted')
    episodes = {e['parent_id']: e for e in manifest['episodes']}
    for parent in parents:
        episode = episodes[parent]
        if (parent not in manifest['parent_split']['train'] or
                episode['split'] != 'train' or episode.get('native_success') is not True):
            raise ValueError('Panel requires successful train parents')
    if any(row['split'] != 'train' or not row['action_valid'][0] for row in selected):
        raise ValueError('Panel requires valid train actions')
    if {row['tool_family'] for row in selected} != {'reach', 'grasp', 'move'}:
        raise ValueError('Panel must cover exactly reach, grasp, move')
    return selected


@dataclasses.dataclass(frozen=True)
class ConstantSchedule:
    value: float = 1e-4

    def create(self):
        return lambda step: self.value


def budget_remaining(deadline, reserve=0, clock=time.monotonic):
    return clock() < deadline - reserve


def lora_only_freeze_filter():
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex('.*lora.*')))


def initialize_preserving_dtype(cfg, loaded):
    """New optimizer without the upstream init's frozen-parameter bf16 cast."""
    import flax.nnx as nnx
    import jax
    from openpi.training import optimizer, utils
    tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    @jax.jit
    def initialize(values):
        model = cfg.model.create(jax.random.key(7))
        graph, variables = nnx.split(model)
        variables.replace_by_pure_dict(values)
        return utils.TrainState(step=0, params=variables, model_def=graph, tx=tx,
                               opt_state=tx.init(variables.filter(cfg.trainable_filter)),
                               ema_decay=None, ema_params=None)
    state = initialize(loaded)
    return jax.block_until_ready(state)


def reconstruction_metrics(predicted, target, valid, families, zero_epsilon=1e-3):
    """Controller-normalized command errors, head excluded; no success/go claims."""
    import numpy as np
    predicted, target = np.asarray(predicted), np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    if (predicted.shape != target.shape or predicted.shape[-1] != 13 or
            valid.shape != target.shape[:2] or not valid[:, 0].all() or
            not np.isfinite(predicted).all() or not np.isfinite(target).all()):
        raise ValueError('Invalid reconstruction arrays')
    result = {}
    for phase in ('all', 'reach', 'grasp', 'move'):
        selected = np.ones(len(families), dtype=bool) if phase == 'all' else np.asarray(families) == phase
        if not selected.any():
            continue
        result[phase] = {}
        for scope in ('first_action', 'valid_chunk'):
            mask = valid[selected].copy()
            if scope == 'first_action':
                mask[:, 1:] = False
            p, t = predicted[selected][mask], target[selected][mask]
            error = p - t
            moving = np.abs(t[:, ACTIVE]) > zero_epsilon
            wrong = (np.sign(p[:, ACTIVE]) != np.sign(t[:, ACTIVE])) & moving
            result[phase][scope] = dict(
                valid_actions=int(mask.sum()), active_channels=ACTIVE,
                mae_active11=float(np.abs(error[:, ACTIVE]).mean()),
                rmse_active11=float(np.sqrt(np.mean(error[:, ACTIVE] ** 2))),
                per_channel_mae=np.abs(error).mean(axis=0).tolist(),
                per_channel_rmse=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                torso_mae=float(np.abs(error[:, 10]).mean()),
                yaw_mae=float(np.abs(error[:, 12]).mean()),
                torso_rmse=float(np.sqrt(np.mean(error[:, 10] ** 2))),
                yaw_rmse=float(np.sqrt(np.mean(error[:, 12] ** 2))),
                moving_sign_denominator=int(moving.sum()),
                moving_sign_errors=int(wrong.sum()), zero_epsilon=zero_epsilon)
    return result


def preflight(args):
    from audit_s1_teacher_actions import audit_provenance
    from bvi.ia_fetch_data import invocation_rows
    manifest, provenance = audit_provenance(args.source, args.dataset_manifest, args.normalizer)
    panel = json.loads(args.panel.read_text(encoding='utf-8'))
    rows = select_panel(panel, manifest, invocation_rows(manifest, 10), provenance)
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != '855' or checkpoint.parent.name != 'best' or not (checkpoint / 'params').is_dir():
        raise ValueError('Explicit frozen best/855 with full params directory required')
    assets = checkpoint / 'assets' / REPO_ID
    contract = json.loads((assets / 'bvi-state-contract.json').read_text())
    if (contract.get('normalizer_sha256') != provenance['normalizer_sha256'] or
            file_hash(assets / 'norm_stats.json') != provenance['normalizer_sha256'] or
            contract.get('training_stage') != 'S1_IA_single_bank_no_progress'):
        raise ValueError('Frozen checkpoint normalizer/stage mismatch')
    return rows, provenance, contract


def worker(args):
    # The parent monotonic deadline includes process startup and every stage here.
    deadline = args.deadline
    output = args.output
    report = dict(status='preflight', additional_updates=0, source_checkpoint_step=855,
                  optimizer='reset_AdamW', learning_rate=1e-4, lr_clock='reset_constant',
                  progress=False, family_bank=False, api_calls=0, native_success=None,
                  go_decision='not_assessed_requires_paired_behavior_review')
    def status(**changes):
        report.update(changes)
        report['remaining_seconds'] = max(0., deadline - time.monotonic())
        write_json(output / 'status.json', report)
    manager = dataset = None
    try:
        status()
        rows, provenance, contract = preflight(args)
        write_json(output / 'selected-rows.json', rows)
        (output / 'frozen-panel.json').write_bytes(args.panel.read_bytes())
        if not budget_remaining(deadline, args.finalize_reserve):
            raise TimeoutError('Budget exhausted during preflight')
        actual = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid,memory.used',
                     '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if actual[0].strip() != GPU1_UUID or int(actual[1]) >= 1024:
            raise RuntimeError('Physical GPU1 identity/idle gate failed')
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU1_UUID, JAX_PLATFORMS='cuda',
                          XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2')
        import numpy as np
        import jax
        import flax.nnx as nnx
        from flax import traverse_util
        import openpi
        from openpi import transforms
        from openpi.models import model as models
        from openpi.training import checkpoints, sharding
        from fetch_native_s1_config import config
        from fetch_ia_s1_dataset import InvocationDataset
        import ia_s1_steps
        author = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain',
                                        '--untracked-files=no'], text=True)
        if commit != 'f4eb160ba52b22c1e85fe432de59c24bbbac6187' or dirty.strip():
            raise ValueError('Expected clean pinned author f4 checkout')
        status(author_commit=commit)
        sys.path.insert(0, str(author))
        cfg = config(REPO_ID, str(output), str(args.checkpoint / 'params'), args.normalizer,
                     steps=args.max_updates, batch=1)
        cfg = dataclasses.replace(cfg, seed=7, exp_name='bounded-diagnostic', lr_schedule=ConstantSchedule(),
              freeze_filter=lora_only_freeze_filter(),
              policy_metadata=dict(cfg.policy_metadata, training_stage='S1_IA_single_bank_no_progress',
                  invocation_aligned=True, diagnostic_only=True, source_checkpoint_step=855,
                  optimizer_reset=True, lr_clock='reset_constant_1e-4',
                  trainable_scope='explicit_lora_only_narrower_than_legacy_default',
                  parameter_dtypes='preserved_from_source_checkpoint'))
        for name in ('robot', 'state_dim', 'state_components', 'action_dim', 'base_position_reference',
                     'base_camera', 'wrist_camera', 'state_conditioning', 'training_repo',
                     'action_convention', 'state_source', 'normalizer_sha256'):
            if contract.get(name) != cfg.policy_metadata[name]:
                raise ValueError('Checkpoint input contract differs: ' + name)
        if cfg.model.enable_progress_head or cfg.progress_loss_weight != 0 or cfg.ema_decay is not None:
            raise ValueError('Progress/EMA must remain disabled')
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset = InvocationDataset(args.source, args.dataset_manifest, args.normalizer, dc, cfg.model, 'train')
        dataset.rows = rows
        write_json(output / 'identity.json', dict(**provenance, panel_sha256=file_hash(args.panel),
                   checkpoint=str(args.checkpoint.resolve()), source_checkpoint_step=855,
                   max_updates=args.max_updates, total_seconds=args.seconds, accumulation=8,
                   optimizer_reset=True, learning_rate=1e-4, lr_clock='reset_constant',
                   sampler='seed7_permutation_repeated_to_update_budget', samples=len(rows),
                   finalization_reserve_seconds=args.finalize_reserve,
                   seed=7, inference_seed=19092026, denoising_steps=10,
                   trainer_sha256=file_hash(__file__), policy_metadata=cfg.policy_metadata))
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        ds_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        status(status='loading')
        # Match the serving initializer: every source leaf must exist, and its
        # value/dtype survives replacement. Do not randomly initialize a missing LoRA.
        ref = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(7))).to_pure_dict())
        loaded = models.restore_params(args.checkpoint / 'params', restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        if set(ref) != set(flat) or any(ref[k].shape != flat[k].shape for k in ref):
            raise ValueError('Warm start requires every pretrained and LoRA leaf')
        digest = hashlib.sha256()
        for key, value in sorted(flat.items()):
            array = np.asarray(value)
            digest.update('/'.join(map(str, key)).encode())
            digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        expected = json.loads(args.panel.read_text(encoding='utf-8')).get('checkpoint_parameters_sha256')
        if expected is not None and digest.hexdigest() != expected:
            raise ValueError('Frozen checkpoint parameter hash mismatch')
        status(source_parameters_sha256=digest.hexdigest())
        state = initialize_preserving_dtype(cfg, loaded)
        restored = traverse_util.flatten_dict(state.params.to_pure_dict())
        if any(restored[k].dtype != flat[k].dtype for k in flat):
            raise ValueError('Initialization changed source parameter dtype')
        del loaded, flat, ref, restored, array, value
        if int(state.step) != 0:
            raise ValueError('Warm start optimizer/update counter must start at zero')
        trainable = traverse_util.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict())
        if not trainable or any('lora' not in '/'.join(map(str, k)).lower() for k in trainable):
            raise ValueError('Only existing shared LoRA leaves may train')
        write_json(output / 'trainable-parameters.json', [
            dict(path='/'.join(map(str, k)), shape=list(v.shape), dtype=str(v.dtype))
            for k, v in sorted(trainable.items())])
        status(trainable_leaves=len(trainable))
        # Only actions are reconstructed here. Unnormalize defaults to strict
        # key matching, so including state stats would require a state payload.
        unnormalize = transforms.Unnormalize({'actions': dc.norm_stats['actions']}, use_quantiles=True)
        def batch(index):
            item = dataset[int(index)]
            valid = ~item.pop('actions_is_pad')
            x = jax.tree.map(lambda v: np.asarray(v)[None], item)
            return jax.device_put((models.Observation.from_dict(x), x['actions'], valid[None]), ds_sharding)
        @jax.jit
        def predict(params, key, obs):
            model = nnx.merge(state.model_def, params)
            model.eval()
            return model.sample_actions(key, obs, num_steps=10)
        families = [r['tool_family'] for r in rows]
        def evaluate(label):
            predictions, targets, masks, unclipped = [], [], [], []
            for index in range(len(rows)):
                if not budget_remaining(deadline, 10):
                    raise TimeoutError('Budget exhausted in paired evaluation')
                obs, actions, valid = batch(index)
                with sharding.set_mesh(mesh):
                    out = predict(state.params, jax.random.fold_in(jax.random.key(19092026), index), obs)
                raw = unnormalize({'actions': np.asarray(out)[0]})['actions'][:, :13]
                target = unnormalize({'actions': np.asarray(actions)[0]})['actions'][:, :13]
                unclipped.append(raw)
                predictions.append(np.clip(raw, -1, 1)); targets.append(np.clip(target, -1, 1))
                masks.append(np.asarray(valid)[0])
            p, t, m = np.asarray(predictions), np.asarray(targets), np.asarray(masks)
            p[:, :, 8:10] = 0; t[:, :, 8:10] = 0
            np.savez_compressed(output / (label + '.npz'), predicted=p, predicted_unclipped=np.asarray(unclipped),
                                target=t, valid=m, families=np.asarray(families),
                                row_keys=np.asarray([[r['parent_id'], r['call_index'], r['observation_index']] for r in rows]))
            metrics = reconstruction_metrics(p, t, m, families)
            write_json(output / (label + '-metrics.json'), metrics)
            return metrics
        status(status='before_evaluation')
        before = evaluate('before')
        grad = jax.jit(functools.partial(ia_s1_steps.micro_gradient, cfg))
        apply = jax.jit(functools.partial(ia_s1_steps.apply_average, cfg))
        order = np.random.default_rng(7).permutation(len(rows))
        exposure = 0
        status(status='training')
        for update in range(args.max_updates):
            summed = None
            losses = []
            for _ in range(8):
                if not budget_remaining(deadline, args.finalize_reserve):
                    break
                index = order[exposure % len(order)]
                with sharding.set_mesh(mesh):
                    loss, g = grad(state, jax.random.fold_in(jax.random.key(7001), exposure), batch(index))
                value = float(loss)
                if not np.isfinite(value):
                    raise ValueError('Nonfinite training loss')
                summed = g if summed is None else jax.tree.map(lambda x, y: x + y, summed, g)
                losses.append(value); exposure += 1
            # Discard interrupted accumulation, never apply after the stop boundary.
            if len(losses) != 8 or not budget_remaining(deadline, args.finalize_reserve):
                break
            with sharding.set_mesh(mesh):
                state, norm = apply(state, summed, 8)
            if not np.isfinite(float(norm)):
                raise ValueError('Nonfinite gradient')
            status(additional_updates=int(state.step), exposures=exposure, loss=float(np.mean(losses)))
            with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(update=int(state.step), loss=float(np.mean(losses)),
                                             gradient_norm=float(norm), lr=1e-4)) + '\n')
        if int(state.step) == 0:
            status(status='budget_exhausted_no_updates')
            return
        status(status='saving', stop_reason='update_limit' if int(state.step) == args.max_updates else 'finalization_reserve')
        manager, _ = checkpoints.initialize_checkpoint_dir(output / 'diagnostic', keep_period=None,
                                                            overwrite=False, resume=False)
        class Assets:
            def data_config(self):
                return dc
        checkpoints.save_state(manager, state, Assets(), int(state.step))
        manager.wait_until_finished()
        saved = output / 'diagnostic' / str(int(state.step))
        write_json(saved / 'assets' / REPO_ID / 'bvi-state-contract.json', cfg.policy_metadata)
        if (not (saved / 'params').is_dir() or
                file_hash(saved / 'assets' / REPO_ID / 'norm_stats.json') != provenance['normalizer_sha256'] or
                not budget_remaining(deadline)):
            raise TimeoutError('Checkpoint not complete inside budget')
        status(checkpoint=str(saved), checkpoint_complete=True, status='after_evaluation')
        after = evaluate('after')
        write_json(output / 'comparison.json', dict(before=before, after=after,
                   metric_space='controller_normalized_clipped_commands_active11',
                   rng='same_fold_in_seed19092026_per_frozen_row_before_after',
                   go_decision='requires_review_not_inferred_from_loss',
                   scope='training_mapping_reconstruction_not_generalization_or_native_success'))
        status(status='completed_diagnostic_requires_review')
    except Exception as exc:
        status(status='failed', error=repr(exc))
        raise
    finally:
        if dataset is not None:
            dataset.close()
        if manager is not None:
            manager.wait_until_finished()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset-manifest', 'normalizer', 'checkpoint', 'panel', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--max-updates', type=int, default=100)
    p.add_argument('--seconds', type=int, default=3600)
    p.add_argument('--finalize-reserve', type=int, default=900,
                   help='Stop updates this many seconds before hard deadline; saves and after evaluation share reserve')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 1 <= args.max_updates <= 100 or not 60 <= args.seconds <= 3600:
        p.error('Diagnostic bound: 1-100 added updates, 60-3600 total seconds')
    if not 30 <= args.finalize_reserve < args.seconds:
        p.error('Finalization reserve must be >=30 seconds and below total budget')
    return args


def main():
    started = time.monotonic()
    args = parse_args()
    if args.preflight:
        rows, provenance, _ = preflight(args)
        print(json.dumps(dict(status='cpu_preflight_passed', samples=len(rows), provenance=provenance)))
        return
    if args.worker:
        if args.deadline is None or os.environ.get('BVI_BOUNDED_PARENT') != str(os.getppid()):
            raise ValueError('Worker must run under this script\'s hard-deadline supervisor')
        return worker(args)
    if sys.platform != 'linux':
        raise RuntimeError('GPU execution requires lab Linux supervisor; CPU preflight works separately')
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = started + args.seconds
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--deadline', str(deadline)]
    environment = dict(os.environ, BVI_BOUNDED_PARENT=str(os.getpid()))
    write_json(args.output / 'supervisor.json', dict(status='starting', seconds=args.seconds))
    with (args.output / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    write_json(args.output / 'supervisor.json', dict(status='hard_timeout' if timed_out else 'exited',
               worker_exit_code=code, elapsed_seconds=time.monotonic() - started,
               budget_seconds=args.seconds, complete_artifacts_require_status_and_checkpoint_validation=True))
    if code:
        raise SystemExit(code if code > 0 else 124)


if __name__ == '__main__':
    main()
