"""Native24 S1 inference only; independent of frozen historical S1-native24. GPU launch must be serialized."""
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
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--auth-file', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--repo-id', default='bvi/s1-official-pick-medium-train')
    p.add_argument('--normalizer', type=Path, required=True)
    p.add_argument('--max-inference-calls', type=int, default=2000)
    p.add_argument('--training-stage', choices=['S1_ordinary_target_domain_SFT_not_TAPT',
                    'S1_IA_single_bank_no_progress'], default='S1_ordinary_target_domain_SFT_not_TAPT')
    a = p.parse_args()
    if not 1 <= a.max_inference_calls <= 2000:
        p.error('--max-inference-calls must be in[1,2000]')
    output, socket_path = a.output.resolve(), a.socket.resolve()
    if not socket_path.is_relative_to(output):
        p.error('--socket must reside under --output')
    output.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or any((output / name).exists() for name in ['ready.json', 'result.json', 'metadata.json']):
        raise FileExistsError('Dedicated server output files already exist; use a fresh output')
    auth = a.auth_file.read_bytes()
    if len(auth) != 32:
        raise ValueError('Authentication file must contain exactly32 random bytes')
    report = dict(status='loading', training_updates=0, progress_head=False, tapt=False,
                  inference_calls=0, max_inference_calls=a.max_inference_calls,
                  client_disconnects=0, denoising_steps=10, action_horizon=10,
                  internal_action_dim=32, external_action_dim=13,
                  action_postprocessing='S1-native24 quantile unnormalization only; client owns clipping/headmask',
                  precision='bfloat16', gpu_idle_limit_mib=2048,
                  gpu_guard_scope='allows pre-existing SAPIEN allocation up to2GiB; no second model permitted')
    listener = connection = None
    started = time.monotonic()

    def save():
        report['wall_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        physical1 = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        if physical1 != a.gpu_uuid:
            raise ValueError('Server requires physical GPU1 exact UUID')
        used = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid, '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        if used > 2048:
            raise RuntimeError(f'GPU1 exceeds2GiB pre-server allocation: {used}MiB')
        report['gpu_memory_before_server_mib'] = used
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid, XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2')
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
        from fetch_native_s1_config import config as fetch_config
        from multiprocessing.connection import Listener

        author = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if commit != 'f4eb160ba52b22c1e85fe432de59c24bbbac6187' or dirty.strip():
            raise ValueError('Expected clean pinned authorf4 checkout')
        cfg = fetch_config(a.repo_id, str(output / 'unused'), str(a.checkpoint), a.normalizer)
        cfg = dataclasses.replace(cfg, policy_metadata=dict(cfg.policy_metadata, training_stage=a.training_stage))
        mc = dataclasses.replace(cfg.model, enable_progress_head=False)
        assert mc.action_dim == 32 and mc.action_horizon == 10 and mc.discrete_state_input and mc.dtype == 'bfloat16'
        assets = a.checkpoint / 'assets' / a.repo_id
        contract = json.loads((assets / 'bvi-state-contract.json').read_text())
        for key in ('robot', 'state_dim', 'state_components', 'action_dim', 'base_position_reference',
                    'base_camera', 'wrist_camera', 'state_conditioning', 'training_repo', 'action_convention',
                    'state_source', 'normalizer_sha256', 'training_stage'):
            if contract.get(key) != cfg.policy_metadata[key]:
                raise ValueError(f'S1-native24 state contract mismatch: {key}')
        ref = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(mc.create, jax.random.key(7))).to_pure_dict())
        loaded = models.restore_params(a.checkpoint / 'params', restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        if set(flat) != set(ref) or len(flat) != 71 or any(v.shape != ref[k].shape for k, v in flat.items()):
            raise ValueError('Strict originalS1-native24 checkpoint mismatch: expected71 exact key/shape matches')
        if any('progress_chunk' in str(k) for k in flat):
            raise ValueError('Original frozenS1-native24 must have no progress head')
        digest = hashlib.sha256()
        for key, value in sorted(flat.items()):
            array = np.asarray(value)
            digest.update('/'.join(map(str, key)).encode()); digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
        parameter_hash = digest.hexdigest()
        @jax.jit
        def initialize(values):
            model = mc.create(jax.random.key(7))
            graph, variables = nnx.split(model)
            variables.replace_by_pure_dict(values)
            return graph, variables
        graph, variables = initialize(loaded)
        jax.block_until_ready(variables)
        del loaded, flat, ref
        if hashlib.sha256((assets / 'norm_stats.json').read_bytes()).hexdigest() != cfg.policy_metadata['normalizer_sha256']:
            raise ValueError('Saved checkpoint normalizer differs from training provenance')
        stats = checkpoints.load_norm_stats(a.checkpoint / 'assets', a.repo_id)
        for key, dim in [('state', 24), ('actions', 13)]:
            for quantile in ['q01', 'q99']:
                values = np.asarray(getattr(stats[key], quantile))
                if values.shape != (dim,) or not np.isfinite(values).all():
                    raise ValueError('Invalid S1-native24 quantile normalization statistics')
        transform = transforms.compose([LiberoInputs(mc.model_type), transforms.Normalize(stats, use_quantiles=True),
                                        *training_config.ModelTransformFactory()(mc).inputs])
        unnormalize = transforms.Unnormalize(stats, use_quantiles=True)
        @jax.jit
        def predict(v, key, obs):
            return nnx.merge(graph, v).sample_actions(key, obs, num_steps=10)
        metadata = dict(**report, author_commit=commit, checkpoint=str(a.checkpoint.resolve()),
                        pretrained_parameters_sha256=parameter_hash, pretrained_tensor_count=71,
                        normalizer_sha256=hashlib.sha256((assets / 'norm_stats.json').read_bytes()).hexdigest(),
                        state_contract_sha256=hashlib.sha256((assets / 'bvi-state-contract.json').read_bytes()).hexdigest(),
                        server_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        state_contract=contract, gpu_uuid=a.gpu_uuid,
                        rng_contract='reset PRNGKey(episode_seed); key,subkey=split(key) per prediction',
                        ready_scope='weights and transforms loaded; first inference may compile')
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        listener = Listener(str(socket_path), family='AF_UNIX', authkey=auth)
        # Unix permissions supplement authentication and restrict local access.
        socket_path.chmod(0o600)
        report['status'] = 'ready'; save()
        (output / 'ready.json').write_text(json.dumps(dict(status='ready', socket=str(socket_path),
            metadata=str(output / 'metadata.json'), first_inference_compiled=False)), encoding='utf-8')
        connection = listener.accept()
        seed = key = None
        count = 0
        while True:
            try:
                request = connection.recv()
            except EOFError:
                connection.close(); connection = None
                seed = key = None; count = 0
                report['client_disconnects'] += 1
                report['status'] = 'waiting_for_next_client'; save()
                connection = listener.accept()  # Blocking; no busy polling or preserved episode state.
                report['status'] = 'ready'; save()
                continue
            try:
                if not isinstance(request, dict):
                    raise ValueError('Request must be a mapping')
                op = request.get('op')
                if op == 'shutdown':
                    connection.send(dict(status='shutdown', inference_calls=report['inference_calls']))
                    report['stop_reason'] = 'explicit_shutdown'; break
                if op == 'reset':
                    seed = request['seed']
                    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
                        seed = None; raise ValueError('Episode seed must be uint32 integer')
                    key = jax.random.PRNGKey(seed); count = 0
                    connection.send(dict(status='reset', seed=seed, prediction_count=count, metadata=metadata))
                    continue
                if op != 'predict' or seed is None or request.get('seed') != seed:
                    raise ValueError('Reset episode seed before predictions; explicit matching seed required')
                if report['inference_calls'] >= a.max_inference_calls:
                    raise ValueError('Frozen native evaluation inference-call budget exhausted')
                workspace, wrist = np.asarray(request['head_rgb']), np.asarray(request['wrist_rgb'])
                state = np.asarray(request['state'], np.float32)
                prompt = request['prompt']
                if workspace.shape != (128, 128, 3) or wrist.shape != (128, 128, 3) or workspace.dtype != np.uint8 or wrist.dtype != np.uint8:
                    raise ValueError('Invalid S1-native24 workspace/wrist RGB contract')
                if state.shape != (24,) or not np.isfinite(state).all() or not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError('Finite native state24 and nonempty prompt required')
                t0 = time.monotonic()
                x = transform({'observation/image': workspace, 'observation/wrist_image': wrist,
                               'observation/state': state, 'prompt': prompt})
                obs = models.Observation.from_dict(jax.tree.map(lambda v: jnp.asarray(v)[None], x))
                next_key, subkey = jax.random.split(key)
                report['inference_calls'] += 1  # Count a launched inference even if its output fails validation.
                actions = np.asarray(predict(variables, subkey, obs))[0]
                actions13 = unnormalize({'actions': actions, 'state': np.asarray(x['state'])})['actions'][:, :13]
                if actions13.shape != (10, 13) or not np.isfinite(actions13).all():
                    raise ValueError('Nonfinite/invalid S1-native24 action output')
                key = next_key; count += 1
                response = dict(status='ok', actions=actions13, seed=seed, prediction_count=count,
                    rng=np.asarray(jax.random.key_data(subkey)).tolist(),
                    inference_seconds=time.monotonic() - t0, pretrained_parameters_sha256=parameter_hash,
                    normalizer_sha256=metadata['normalizer_sha256'], progress_head=False,
                    controller_clip_fraction=float(np.mean(np.abs(actions13) > 1)))
                connection.send(response)
                report.update(last_seed=seed, last_prediction_count=count, last_inference_seconds=response['inference_seconds'])
                save()
            except Exception as exc:
                connection.send(dict(status='error', error=repr(exc)))
                report['last_request_error'] = repr(exc); save()
        report['status'] = 'stopped'
    except Exception as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        # Listener closes/removes its own Unix socket; never remove an unrelated path.
        save()


if __name__ == '__main__':
    main()
