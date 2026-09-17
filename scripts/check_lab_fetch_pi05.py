"""Read-only Fetch V8 -> author pi0.5 prefix-head inference gate (no training).

NPZ must contain workspace_rgb (224,224,3 uint8), wrist_rgb (H,W,3 uint8),
state (30 float values: qpos+qvel, base XY relative to original skill start),
and optionally scalar prompt. A head-camera image is NOT a workspace substitute.
Run with an external wall timeout; output is a new directory, never overwritten.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-npz', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--repo-id', default='bvi/fetch-seed1-workspace-recovery-v8')
    p.add_argument('--prompt')
    p.add_argument('--denoising-steps', type=int, default=10)
    p.add_argument('--compute-dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                   help='float32 is a separate numerical diagnostic, never an automatic training change')
    p.add_argument('--expected-author-commit', default='f4eb160ba52b22c1e85fe432de59c24bbbac6187')
    a = p.parse_args()
    if not 1 <= a.denoising_steps <= 10:
        p.error('denoising steps must be in [1,10]')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='preflight', training_updates=0, api_calls=0,
                  compute_dtype=a.compute_dtype, precision_diagnostic=a.compute_dtype != 'bfloat16',
                  new_head_untrained=True, native_task_success_evaluated=False,
                  comparison_scope='same author graph: sample_actions vs infer_actions_and_progress; not upstream runtime parity',
                  progress_source='author pooled image+prompt+discrete-state prefix; chunk step embedding; newly initialized head')

    def save():
        report['wall_seconds'] = time.monotonic() - started
        (a.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        used = int(subprocess.check_output([
            'nvidia-smi', '-i', a.gpu_uuid, '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        if used >= 1024:
            raise RuntimeError(f'Selected GPU occupied: {used} MiB')
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,
                          XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2')
        import numpy as np
        import flax.nnx as nnx
        from flax import traverse_util
        import jax
        import jax.numpy as jnp
        import openpi
        from openpi import transforms
        from openpi.models import model as models
        from openpi.training import checkpoints, config as training_config
        from openpi.policies.libero_policy import LiberoInputs
        from fetch_openpi import config as fetch_config

        author_root = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(['git', '-C', str(author_root), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != a.expected_author_commit:
            raise ValueError(f'Unexpected author checkout: {commit}')
        dirty = subprocess.check_output(['git', '-C', str(author_root), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if dirty.strip():
            raise ValueError('Author tracked source is modified; pin and review before gate')
        report.update(author_commit=commit, author_root=str(author_root), gpu_uuid=a.gpu_uuid,
                      input_sha256=hashlib.sha256(a.input_npz.read_bytes()).hexdigest())
        contract_file = a.checkpoint / 'assets' / a.repo_id / 'bvi-state-contract.json'
        contract = json.loads(contract_file.read_text())
        expected = dict(robot='fetch', state_dim=30, state_components=['qpos', 'qvel'],
                        action_dim=13, base_position_reference='skill_start_xy',
                        base_camera='fetch_workspace', wrist_camera='fetch_hand',
                        state_conditioning=True, training_repo=a.repo_id,
                        action_convention='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity')
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(f'V8 state contract mismatch: {key}')
        with np.load(a.input_npz, allow_pickle=False) as data:
            workspace = data['workspace_rgb'].copy()
            wrist = data['wrist_rgb'].copy()
            state = np.asarray(data['state'], dtype=np.float32).copy()
            prompt = a.prompt or (str(data['prompt'].item()) if 'prompt' in data else None)
        if workspace.shape != (224, 224, 3) or workspace.dtype != np.uint8:
            raise ValueError('workspace_rgb must be the V8 workspace view: 224x224x3 uint8')
        if wrist.ndim != 3 or wrist.shape[-1] != 3 or wrist.dtype != np.uint8:
            raise ValueError('wrist_rgb must be HWC uint8')
        if state.shape != (30,) or not np.isfinite(state).all() or not prompt:
            raise ValueError('Finite raw state30 and a nonempty prompt are required')
        cfg = fetch_config(a.repo_id, str(a.output / 'unused-training-path'),
                           state_input=True, include_velocity=True,
                           relative_base=True, base_camera='fetch_workspace')
        mc = dataclasses.replace(cfg.model, enable_progress_head=True, dtype=a.compute_dtype)
        assert mc.action_dim == 32 and mc.action_horizon == 10 and mc.discrete_state_input
        shape = nnx.state(nnx.eval_shape(mc.create, jax.random.key(7))).to_pure_dict()
        ref = traverse_util.flatten_dict(shape)
        loaded = models.restore_params(a.checkpoint / 'params', restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        is_head = lambda key: any('progress_chunk' in str(part) for part in key)
        missing = set(ref) - set(flat)
        extra = set(flat) - set(ref)
        mismatch = [str(k) for k in set(ref) & set(flat) if ref[k].shape != flat[k].shape]
        if extra or mismatch or any(not is_head(k) for k in missing):
            raise ValueError(f'Strict checkpoint mismatch: extra={sorted(map(str, extra))}, shape={mismatch}, missing={sorted(map(str, missing))}')
        if not missing or any(is_head(k) for k in flat):
            raise ValueError('Expected V8 without any pre-existing progress head')

        def digest(values):
            h = hashlib.sha256()
            for key in sorted(values):
                value = np.asarray(values[key])
                h.update('/'.join(map(str, key)).encode())
                h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
                h.update(value.tobytes())
            return h.hexdigest()

        report.update(status='strict_checkpoint_loaded', pretrained_tensor_count=len(flat),
                      pretrained_parameters_sha256=digest(flat),
                      pretrained_lora_sha256=digest({k: v for k, v in flat.items() if any('lora' in str(t) for t in k)}),
                      initialized_head_keys=sorted(map(str, missing)), state_contract=contract,
                      state_contract_sha256=hashlib.sha256(contract_file.read_bytes()).hexdigest())
        save()
        # Preserve checkpoint dtype/values, including V8 LoRA tensors. Only head is random.
        @jax.jit
        def initialize(values):
            model = mc.create(jax.random.key(7))
            graph, variables = nnx.split(model)
            variables.replace_by_pure_dict(values)
            return graph, variables
        graph, variables = initialize(loaded)
        del flat, loaded, ref, shape
        head = {k: np.asarray(v) for k, v in traverse_util.flatten_dict(variables.to_pure_dict()).items() if is_head(k)}
        report['untrained_head_sha256'] = digest(head)
        stats = checkpoints.load_norm_stats(a.checkpoint / 'assets', a.repo_id)
        for key, dim in [('state', 30), ('actions', 13)]:
            if np.asarray(stats[key].q01).shape != (dim,):
                raise ValueError(f'Wrong Fetch normalizer dimension: {key}')
        norm_path = a.checkpoint / 'assets' / a.repo_id / 'norm_stats.json'
        report['normalizer_sha256'] = hashlib.sha256(norm_path.read_bytes()).hexdigest()
        transform = transforms.compose([LiberoInputs(mc.model_type),
            transforms.Normalize(stats, use_quantiles=True),
            *training_config.ModelTransformFactory()(mc).inputs])
        x = transform({'observation/image': workspace, 'observation/wrist_image': wrist,
                       'observation/state': state, 'prompt': prompt})
        obs = models.Observation.from_dict(jax.tree.map(lambda v: jnp.asarray(v)[None], x))
        noise = jax.random.normal(jax.random.key(8001), (1, 10, 32))
        @jax.jit
        def predict(v, n):
            model = nnx.merge(graph, v)
            return model.infer_actions_and_progress(jax.random.key(8002), obs,
                noise=n, num_steps=a.denoising_steps)
        @jax.jit
        def baseline(v, n):
            return nnx.merge(graph, v).sample_actions(jax.random.key(8002), obs,
                noise=n, num_steps=a.denoising_steps)
        @jax.jit
        def training_readout(v, target_actions, key):
            return nnx.merge(graph, v).compute_action_and_progress_chunk_prefix(
                key, obs, target_actions, train=False)
        report['status'] = 'inference'; save()
        actions, progress = jax.device_get(predict(variables, noise))
        plain = np.asarray(baseline(variables, noise))
        other_actions, other_progress = jax.device_get(predict(variables,
            jax.random.normal(jax.random.key(8003), (1, 10, 32))))
        # Training and inference must read the same current-observation prefix.
        # Change teacher actions and flow noise to detect accidental suffix access.
        _, train_progress = jax.device_get(training_readout(
            variables, jnp.zeros((1, 10, 32)), jax.random.key(8004)))
        _, other_train_progress = jax.device_get(training_readout(
            variables, noise, jax.random.key(8005)))
        output = transforms.Unnormalize(stats, use_quantiles=True)(
            {'actions': actions[0], 'state': np.asarray(x['state'])})['actions'][:, :13]
        finite = all(np.isfinite(v).all() for v in [actions, progress, plain, other_actions,
                     other_progress, train_progress, other_train_progress, output])
        report.update(action_shape=list(actions.shape), external_action_shape=list(output.shape),
                      progress_shape=list(progress.shape), all_finite=bool(finite),
                      paired_action_max_abs_diff=float(np.max(np.abs(actions - plain))),
                      paired_actions_exact=bool(np.array_equal(actions, plain)),
                      progress_noise_max_abs_diff=float(np.max(np.abs(progress - other_progress))),
                      progress_noise_independent=bool(np.allclose(progress, other_progress, atol=1e-6, rtol=0)),
                      train_inference_progress_max_abs_diff=float(np.max(np.abs(progress - train_progress))),
                      train_action_noise_progress_max_abs_diff=float(np.max(np.abs(train_progress - other_train_progress))),
                      train_inference_progress_agree=bool(np.allclose(progress, train_progress, atol=1e-4, rtol=0)),
                      train_progress_action_noise_independent=bool(np.allclose(train_progress, other_train_progress, atol=1e-6, rtol=0)),
                      normalized_action_max_abs=float(np.max(np.abs(actions[0, :, :13]))),
                      external_action_max_abs=float(np.max(np.abs(output))),
                      controller_clip_fraction=float(np.mean(np.abs(output) > 1)),
                      controller_clip_max_delta=float(np.max(np.abs(output - np.clip(output, -1, 1)))),
                      progress_min=float(np.min(progress)), progress_max=float(np.max(progress)))
        np.savez_compressed(a.output / 'predictions.npz', actions=actions, external_actions=output,
                            progress=progress, plain_actions=plain, other_noise_progress=other_progress,
                            train_progress=train_progress, other_train_progress=other_train_progress)
        if not finite or actions.shape != (1, 10, 32) or progress.shape != (1, 10):
            raise ValueError('Inference output contract failed')
        if (not np.allclose(actions, plain, atol=1e-5, rtol=1e-5)
            or not report['progress_noise_independent']
            or not report['train_inference_progress_agree']
            or not report['train_progress_action_noise_independent']):
            raise ValueError('Paired action or noise-independent prefix progress check failed')
        report['status'] = 'passed_interface_gate_untrained_head_not_task_success'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
