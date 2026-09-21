"""New native24 S2 trainer. Historical native24 S1 trainer and holds remain untouched.

Full admission requires verified new S1 ten-episode evidence and native24
handoff validation.  A separate immutable adjudication can authorize only the
explicitly labelled, at-most-20-update diagnostic integration path.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

FAMILIES = ('reach', 'grasp', 'move', 'release')
AUTHOR_COMMIT = 'f4eb160ba52b22c1e85fe432de59c24bbbac6187'
REPORT_PATH = None
DIAGNOSTIC_SCOPE = 'diagnostic_20_update_only_not_s2_capability_admission'


def report(status, **details):
    if REPORT_PATH is not None:
        REPORT_PATH.write_text(json.dumps(dict(status=status, **details), indent=2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_status(entry_evidence, completed):
    if not completed:
        return 'wall_limit_before_requested_steps'
    if entry_evidence.get('scope') == DIAGNOSTIC_SCOPE:
        return DIAGNOSTIC_SCOPE
    return 'bounded_sft_complete_not_online_success'


def load_dataset(directory):
    """CPU validation only. Never import JAX or initialize a model here."""
    import numpy as np
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = manifest['invocations']
    if not records:
        raise ValueError('Empty invocation manifest')
    tables, parent_splits, hashes = {}, {}, {}
    for n, record in enumerate(records):
        if record['family'] not in FAMILIES or record['split'] not in ('train', 'validation'):
            raise ValueError('Unknown family or split')
        if (not record.get('instruction') or not record.get('completion_evidence')
                or record.get('completion_verified') is False):
            raise ValueError('Instruction and reviewed completion evidence required')
        parent = json.dumps(record['parent_episode'], sort_keys=True)
        if parent in parent_splits and parent_splits[parent] != record['split']:
            raise ValueError('Parent episode leaks across train/validation')
        parent_splits[parent] = record['split']
        path = Path(record['path'])
        path = path if path.is_absolute() else directory / path
        actual_sha = sha(path)
        if record.get('sha256') != actual_sha:
            raise ValueError(f'NPZ hash mismatch or absent hash: {path}')
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k].copy() for k in (
                'head_rgb', 'wrist_rgb', 'state', 'actions',
                'progress', 'action_valid', 'progress_valid')}
        length = len(data['state'])
        if length < 2:
            raise ValueError('Invocation must contain start and endpoint')
        shapes = dict(head_rgb=(length, 128, 128, 3),
                      wrist_rgb=(length, 128, 128, 3), state=(length, 24),
                      actions=(length, 13), progress=(length,),
                      action_valid=(length,), progress_valid=(length,))
        for key, shape in shapes.items():
            value = data[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key}: {value.shape}, expected {shape}')
            if key.endswith('rgb') and value.dtype != np.uint8:
                raise ValueError('Cameras must be uint8')
        for key in ('action_valid', 'progress_valid'):
            if not np.isin(data[key], [0, 1]).all():
                raise ValueError('Masks must be binary')
        if data['action_valid'][-1] or not data['progress_valid'][-1]:
            raise ValueError('Endpoint must have progress label but no action label')
        if not data['action_valid'].any() or not data['progress_valid'].any():
            raise ValueError('Invocation lacks valid supervision')
        if not np.array_equal(data['action_valid'].astype(bool), np.arange(length) < length - 1):
            raise ValueError('Reviewed invocation requires contiguous real actions before endpoint')
        if (abs(data['actions'][data['action_valid'].astype(bool)]) > 1.00001).any():
            raise ValueError('Raw Fetch actions exceed controller normalization')
        if ((data['progress'] < 0) | (data['progress'] > 1)).any():
            raise ValueError('Progress outside [0,1]')
        if not np.allclose(data['progress'], np.arange(length) / (length - 1), atol=1e-6, rtol=0):
            raise ValueError('Progress must equal current-observation invocation elapsed fraction')
        tables[n] = data
        hashes[str(n)] = actual_sha
    for split in ('train', 'validation'):
        for family in FAMILIES:
            if not any(r['split'] == split and r['family'] == family for r in records):
                raise ValueError(f'Missing {split}/{family}')
    return records, tables, dict(manifest_sha256=sha(manifest_path), npz_sha256=hashes)


def chunk(data, index):
    """Endpoint action invalidity and chunk padding are independent of progress."""
    import numpy as np
    positions = np.arange(index, index + 10)
    valid = positions < len(data['state'])
    clipped = np.minimum(positions, len(data['state']) - 1)
    return (clipped, valid * data['action_valid'][clipped],
            valid * data['progress_valid'][clipped])


def action_context(data, index):
    """Repeat the last real action, never the endpoint's dummy zero label.

    The author's suffix attends across action tokens, so invalid tail targets
    still affect valid-token context even when their direct loss is masked.
    """
    import numpy as np
    real = np.flatnonzero(data['action_valid'])
    if not len(real) or not np.array_equal(real, np.arange(real[-1] + 1)):
        raise ValueError('Action labels must form a contiguous invocation prefix')
    return data['actions'][np.minimum(np.arange(index, index + 10), real[-1])]


def main():
    global REPORT_PATH
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu-uuid')
    p.add_argument('--repo-id', default='bvi/s1-official-pick-medium-train')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--resume', type=Path)
    p.add_argument('--s1-panel', type=Path)
    p.add_argument('--s1-best-record', type=Path)
    p.add_argument('--normalizer', type=Path)
    p.add_argument('--handoff-validation-manifest', type=Path)
    p.add_argument('--diagnostic-adjudication-manifest', type=Path)
    p.add_argument('--diagnostic-adjudication-sha256')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 1 <= a.steps <= 20 or not 1 <= a.seconds <= 1800:
        p.error('Limits: initial gate only: 1..20 optimizer steps, 1..1800 wall seconds')
    entry_evidence = None
    diagnostic_requested = any((a.diagnostic_adjudication_manifest,
                                a.diagnostic_adjudication_sha256))
    full_requested = any((a.s1_panel, a.handoff_validation_manifest))
    if diagnostic_requested and full_requested:
        p.error('Diagnostic-only and full S2 admission evidence are mutually exclusive')
    if diagnostic_requested:
        if not all((a.diagnostic_adjudication_manifest,
                    a.diagnostic_adjudication_sha256, a.checkpoint,
                    a.normalizer, a.s1_best_record)):
            p.error('Diagnostic entry requires its manifest+SHA, checkpoint, normalizer and S1 best record')
        from bvi.s2_diagnostic_gate import validate_diagnostic_entry
        entry_evidence = validate_diagnostic_entry(
            a.diagnostic_adjudication_manifest,
            a.diagnostic_adjudication_sha256,
            a.checkpoint,
            a.normalizer,
            a.s1_best_record,
            a.steps,
        )
    elif not a.dry_run or full_requested:
        from native_s2_entry import validate_entry
        entry_evidence = validate_entry(
            a.s1_panel,
            a.s1_best_record,
            a.handoff_validation_manifest,
            a.data / 'manifest.json',
        )
    records, tables, identity = load_dataset(a.data)
    if entry_evidence is not None:
        train_parents = {json.dumps(r['parent_episode'], sort_keys=True) for r in records if r['split'] == 'train'}
        for parent in entry_evidence.get('handoff_parents', []):
            key = json.dumps(parent, sort_keys=True)
            if key in train_parents:
                raise ValueError('Wrong-handoff evidence must not belong to training parents')
    if a.dry_run:
        print(json.dumps(dict(status='cpu_dataset_validated_no_training', **identity,
              windows=len(records), families=FAMILIES, horizon=10,
              microbatch=1, accumulation=8,
              entry_scope=None if entry_evidence is None else entry_evidence.get('scope'),
              capability_admission=None if entry_evidence is None else
                  entry_evidence.get('capability_admission', True)), indent=2))
        return
    if not all((a.checkpoint, a.output, a.gpu_uuid)):
        p.error('Training requires --checkpoint, --output and --gpu-uuid')
    a.output.mkdir(parents=True, exist_ok=False)
    REPORT_PATH = a.output / 'result.json'
    report('preflight_no_updates', entry_scope=entry_evidence.get('scope'),
           capability_admission=entry_evidence.get('capability_admission', True),
           bounded_sft_expansion_authorized=entry_evidence.get('bounded_sft_expansion', False),
           online_evaluation_authorized=entry_evidence.get('online_evaluation', False))
    data_manifest = json.loads((a.data / 'manifest.json').read_text(encoding='utf-8'))
    if data_manifest.get('status') != 'built_all_families' or not all(data_manifest.get(k) is True for k in (
            'training_ready', 'source_collection_complete', 'fixed_collection_complete')):
        raise ValueError('Complete fixed collection and training-ready builder evidence required')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader'], text=True).splitlines()
    gpu_map = dict(line.replace(' ', '').split(',') for line in gpu_rows)
    if gpu_map.get('1') != a.gpu_uuid:
        raise ValueError('Training is restricted to physical GPU index 1 and its exact UUID')
    episodes = data_manifest.get('episodes', [])
    parents = {(r['split'], json.dumps(r['parent_episode'], sort_keys=True)): r['parent_episode'] for r in episodes}
    if len(parents) != len(episodes) or any((r['split'], json.dumps(r['parent_episode'], sort_keys=True)) not in parents for r in records):
        raise ValueError('Duplicate collection parents or invocation outside collection roster')
    # Source identity includes task: Pick/Place trajectory numbers may coincide.
    for split in ('train', 'validation'):
        if not any(s == split for s, _ in parents):
            raise ValueError('Both parent-disjoint splits required')
    used = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError(f'Selected GPU occupied: {used} MiB')
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2')
    import numpy as np
    import flax.nnx as nnx
    from flax import traverse_util
    import jax
    import jax.numpy as jnp
    import optax
    import openpi
    from openpi import transforms
    from openpi.models import model as models
    from openpi.shared import nnx_utils
    from openpi.training import checkpoints, config as training_config
    from openpi.policies.libero_policy import LiberoInputs
    from fetch_native_s1_config import config as fetch_config

    author_root = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(author_root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(author_root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if commit != AUTHOR_COMMIT or dirty.strip():
        raise ValueError('Author source must be pinned and tracked files unmodified')
    cfg = fetch_config(a.repo_id, str(a.output / 'unused'), str(a.checkpoint), a.normalizer)
    cfg = dataclasses.replace(cfg, progress_loss_weight=0.1)
    mc = dataclasses.replace(cfg.model, enable_progress_head=True)
    assert mc.action_dim == 32 and mc.action_horizon == 10 and mc.discrete_state_input
    assets = a.checkpoint / 'assets' / a.repo_id
    contract = json.loads((assets / 'bvi-state-contract.json').read_text())
    if (data_manifest.get('schema') != 'fetch_native24_family_v1'
            or data_manifest.get('label_contract') != 'current_observation_v2'
            or data_manifest.get('normalizer_sha256') != sha(assets / 'norm_stats.json')):
        raise ValueError('Dataset schema or native24 S1 normalizer provenance mismatch')
    for key in ('robot', 'state_dim', 'state_components', 'action_dim',
                'base_position_reference', 'base_camera', 'wrist_camera',
                'state_conditioning', 'training_repo', 'action_convention'):
        if contract.get(key) != cfg.policy_metadata[key]:
            raise ValueError(f'native24 S1 state contract mismatch: {key}')
        if key != 'training_repo' and data_manifest.get('state_contract', {}).get(key) != contract[key]:
            raise ValueError(f'Dataset state contract mismatch: {key}')
    stats = checkpoints.load_norm_stats(a.checkpoint / 'assets', a.repo_id)
    for key, dimension in [('state', 24), ('actions', 13)]:
        if any(np.asarray(getattr(stats[key], q)).shape != (dimension,) for q in ('q01', 'q99')):
            raise ValueError(f'Wrong quantile normalization dimensions: {key}')
    transform = transforms.compose([LiberoInputs(mc.model_type),
        transforms.Normalize(stats, use_quantiles=True),
        *training_config.ModelTransformFactory()(mc).inputs])
    lora = nnx_utils.PathRegex('.*lora.*')
    head_filter = nnx_utils.PathRegex('.*progress_chunk.*')
    train_filter = nnx.Any(lora, head_filter)
    reference = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(mc.create, jax.random.key(7))).to_pure_dict())
    loaded = models.restore_params(a.checkpoint / 'params', restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(loaded)
    is_head = lambda key: any('progress_chunk' in str(x) for x in key)
    missing = set(reference) - set(flat)
    if (set(flat) - set(reference) or not missing or any(not is_head(k) for k in missing)
        or any(is_head(k) for k in flat)
        or any(reference[k].shape != v.shape for k, v in flat.items())):
        raise ValueError('Strict native24 S1 checkpoint mismatch; only absent progress head is permitted')

    def digest(tree):
        h = hashlib.sha256()
        pure = tree.to_pure_dict() if hasattr(tree, 'to_pure_dict') else tree
        for key, value in sorted(traverse_util.flatten_dict(jax.device_get(pure)).items()):
            value = np.asarray(value)
            h.update('/'.join(map(str, key)).encode()); h.update(str(value.shape).encode())
            h.update(str(value.dtype).encode()); h.update(value.tobytes())
        return h.hexdigest()

    identity.update(pretrained_sha256=digest(loaded), author_commit=commit,
                    normalizer_sha256=sha(assets / 'norm_stats.json'),
                    state_contract_sha256=sha(assets / 'bvi-state-contract.json'),
                    trainer_sha256=sha(__file__), progress_weight=float(cfg.progress_loss_weight))
    if identity['pretrained_sha256'] != entry_evidence['native']['checkpoint_sha256']:
        raise ValueError('Training source checkpoint differs from frozen native capability gate')
    identity['entry_gate_evidence'] = entry_evidence
    if float(cfg.progress_loss_weight) != 0.1:
        raise ValueError('Pinned author TrainConfig progress weight changed from audited 0.1')
    report('strict_checkpoint_validated_initializing', identity=identity)
    @jax.jit
    def initialize(values):
        model = mc.create(jax.random.key(7))
        graph, variables = nnx.split(model)
        variables.replace_by_pure_dict(values)
        return graph, variables
    graph, variables = initialize(loaded)
    # No cast: preserve exact native24 S1 checkpoint values and dtypes, including its LoRA.
    frozen = variables.filter(nnx.Not(train_filter))
    head = variables.filter(head_filter)
    banks = {f: variables.filter(lora) for f in FAMILIES}
    if not jax.tree.leaves(head) or not jax.tree.leaves(banks[FAMILIES[0]]):
        raise ValueError('Empty trainable partition')
    frozen_hash = digest(frozen)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(5e-5, weight_decay=0.0))
    opts = {f: tx.init(banks[f]) for f in FAMILIES}
    head_opt = tx.init(head)
    rng = np.random.default_rng(7)
    key = jax.random.PRNGKey(7001)
    step = 0
    restored_best = None
    if a.resume:
        saved = pickle.loads(a.resume.read_bytes())  # Trusted local checkpoint only.
        if saved['identity'] != identity:
            raise ValueError('Resume source/checkpoint/dataset identity mismatch')
        banks, head, opts, head_opt = [jax.device_put(saved[k]) for k in ('banks', 'head', 'opts', 'head_opt')]
        rng.bit_generator.state = saved['numpy_rng']
        key = jax.device_put(saved['jax_rng'])
        step = saved['step']
        restored_best = saved.get('best_record')
        if set(banks) != set(FAMILIES) or step >= a.steps:
            raise ValueError('Invalid resume families or already reached total step limit')
    (a.output / 'config.json').write_text(json.dumps(dict(identity=identity,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        frozen_sha256=frozen_hash, initial_bank_sha256={f: digest(banks[f]) for f in FAMILIES},
        entry_scope=entry_evidence.get('scope'),
        capability_admission=entry_evidence.get('capability_admission', True),
        bounded_sft_expansion_authorized=entry_evidence.get('bounded_sft_expansion', False),
        online_evaluation_authorized=entry_evidence.get('online_evaluation', False),
        microbatch=1, accumulation=8, droid=False, grpo=False), indent=2))
    del loaded, flat, reference, variables

    def sample(record_index, index):
        data, record = tables[record_index], records[record_index]
        indices, action_valid, progress_valid = chunk(data, index)
        x = transform({'observation/image': data['head_rgb'][index],
            'observation/wrist_image': data['wrist_rgb'][index],
            'observation/state': data['state'][index], 'actions': action_context(data, index),
            'prompt': record['instruction']})
        action = x.pop('actions')
        obs = models.Observation.from_dict(jax.tree.map(lambda v: jnp.asarray(v)[None], x))
        return obs, jnp.asarray(action)[None], jnp.asarray(data['progress'][indices])[None], jnp.asarray(action_valid)[None], jnp.asarray(progress_valid)[None]

    def loss(params, frozen, key, batch):
        model = nnx.merge(graph, nnx.State.merge(frozen, params))
        obs, actions, target, amask, pmask = batch
        al, progress = model.compute_action_and_progress_chunk_prefix(key, obs, actions, train=False)
        action_loss = jnp.sum(al * amask) / jnp.maximum(jnp.sum(amask), 1)
        progress_loss = jnp.sum(jnp.square(progress - target) * pmask) / jnp.maximum(jnp.sum(pmask), 1)
        return action_loss + cfg.progress_loss_weight * progress_loss, jnp.stack([action_loss, progress_loss])
    grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    evaluate = jax.jit(loss)
    started = time.monotonic()
    best_record = restored_best
    best = float('inf') if best_record is None else best_record['joint_loss']
    if best_record is not None:
        best_path = Path(best_record['checkpoint'])
        if not best_path.is_file() or sha(best_path) != best_record['sha256']:
            raise ValueError('Resume prior best checkpoint unavailable or modified')
        (a.output / 'best.json').write_text(json.dumps(best_record))
    validation_cache = None

    def validate():
        # Fixed cap: three positions in up to ten deterministic windows per family.
        rows = []
        for family in FAMILIES:
            candidates = [n for n, r in enumerate(records) if r['split'] == 'validation' and r['family'] == family][:10]
            for n in candidates:
                for index in sorted({0, len(tables[n]['state']) // 2, len(tables[n]['state']) - 1}):
                    value, aux = evaluate(nnx.State.merge(banks[family], head), frozen,
                        jax.random.PRNGKey(123 + n * 1000 + index), sample(n, index))
                    row = dict(window=n, index=index, family=family, joint_loss=float(value),
                               action_loss=float(aux[0]), progress_loss=float(aux[1]))
                    if not all(np.isfinite(row[k]) for k in ('joint_loss', 'action_loss', 'progress_loss')):
                        raise ValueError('Nonfinite validation')
                    rows.append(row)
        family_means = {f: float(np.mean([r['joint_loss'] for r in rows if r['family'] == f])) for f in FAMILIES}
        return dict(step=step, joint_loss=float(np.mean(list(family_means.values()))),
                    family_joint_loss=family_means, rows=rows, max_samples=120,
                    selection='heldout_joint_loss_only_fixed_first10_windows_per_family')

    def save():
        target = a.output / 'resume.pkl'
        temporary = a.output / 'resume.tmp'
        payload = dict(step=step, banks=banks, head=head, opts=opts, head_opt=head_opt,
                       numpy_rng=rng.bit_generator.state, jax_rng=key, identity=identity,
                       best_record=best_record)
        temporary.write_bytes(pickle.dumps(jax.device_get(payload)))
        temporary.replace(target)
        step_path = a.output / f'step-{step:04d}.pkl'
        if not step_path.exists():
            # Hard link preserves immutable bytes when resume.pkl is later atomically replaced.
            os.link(target, step_path)
        (a.output / 'checkpoint.json').write_text(json.dumps(dict(step=step, sha256=sha(target),
            frozen_sha256=digest(frozen), bank_sha256={f: digest(banks[f]) for f in FAMILIES},
            head_sha256=digest(head)), indent=2))

    def validate_and_select():
        nonlocal best, best_record, validation_cache
        report('validation', steps=step)
        validation_cache = validate()
        (a.output / f'validation-{step:04d}.json').write_text(json.dumps(validation_cache, indent=2))
        save()
        if validation_cache['joint_loss'] < best:
            best = validation_cache['joint_loss']
            best_record = dict(step=step,
                checkpoint=str((a.output / f'step-{step:04d}.pkl').resolve()), joint_loss=best,
                sha256=sha(a.output / f'step-{step:04d}.pkl'))
            (a.output / 'best.json').write_text(json.dumps(best_record))
            # Resume points to the immutable best artifact without self-referential hashing.
            save()

    validate_and_select()

    with (a.output / 'train.jsonl').open('a') as log:
        while step < a.steps and time.monotonic() - started < a.seconds:
            report('training', completed_updates=step, requested_total_updates=a.steps)
            family = FAMILIES[step % 4]
            choices = [n for n, r in enumerate(records) if r['split'] == 'train' and r['family'] == family]
            params = nnx.State.merge(banks[family], head)
            before = {f: digest(banks[f]) for f in FAMILIES} if step < 4 else None
            head_before = digest(head) if before else None
            acc, values = None, []
            for micro in range(8):
                n = int(rng.choice(choices))
                index = int(rng.integers(len(tables[n]['state'])))
                key, subkey = jax.random.split(key)
                (value, aux), grads = grad(params, frozen, subkey, sample(n, index))
                if not np.isfinite(np.asarray(jax.device_get(aux))).all():
                    raise ValueError('Nonfinite loss; update rejected')
                acc = grads if acc is None else jax.tree.map(lambda x, y: x + y, acc, grads)
                values.append(np.asarray(jax.device_get(aux)))
            acc = jax.tree.map(lambda x: x / 8, acc)
            if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(jax.device_get(acc))):
                raise ValueError('Nonfinite gradients; update rejected')
            update, opts[family] = tx.update(acc.filter(lora), opts[family], banks[family])
            banks[family] = optax.apply_updates(banks[family], update)
            update, head_opt = tx.update(acc.filter(head_filter), head_opt, head)
            head = optax.apply_updates(head, update)
            step += 1
            if before:
                after = {f: digest(banks[f]) for f in FAMILIES}
                if before[family] == after[family] or any(before[f] != after[f] for f in FAMILIES if f != family) or digest(head) == head_before or digest(frozen) != frozen_hash:
                    raise ValueError('Selected bank/shared head/frozen isolation gate failed')
                (a.output / f'update-audit-{family}.json').write_text(json.dumps(dict(before=before, after=after, selected=family, frozen_unchanged=True, head_changed=True)))
            metrics = dict(step=step, family=family, action_loss=float(np.mean(values, axis=0)[0]),
                           progress_loss=float(np.mean(values, axis=0)[1]), seconds=time.monotonic()-started)
            log.write(json.dumps(metrics) + '\n'); log.flush()
            print(json.dumps(metrics), flush=True)
            if step == 1 or step % (20 if a.steps <= 20 else 200) == 0:
                save()
            if step % 200 == 0:
                validate_and_select()
        if validation_cache is None or validation_cache['step'] != step:
            validate_and_select()
        if digest(frozen) != frozen_hash:
            raise ValueError('Frozen backbone changed')
        save()
    status = completion_status(entry_evidence, step == a.steps)
    report(status, steps=step, requested_total_updates=a.steps,
           measured_training_validation_seconds=time.monotonic()-started,
           external_timeout_required=True, native_success_evaluated=False,
           entry_scope=entry_evidence.get('scope'),
           capability_admission=entry_evidence.get('capability_admission', True),
           bounded_sft_expansion_authorized=entry_evidence.get('bounded_sft_expansion', False),
           online_evaluation_authorized=entry_evidence.get('online_evaluation', False))
    print(json.dumps(dict(status=status, steps=step,
                          entry_scope=entry_evidence.get('scope'),
                          capability_admission=entry_evidence.get('capability_admission', True))))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        report('failed', error=repr(exc), checkpoint_may_exist=True)
        raise
