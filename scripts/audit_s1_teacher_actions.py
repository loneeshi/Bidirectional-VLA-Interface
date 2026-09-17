"""Bounded frozen-checkpoint action audit on eight preregistered teacher frames.

No simulation, parameter updates, API calls or checkpoint selection. Errors use
controller-normalized Fetch13 actions after dataset unnormalization, clipping,
and stationary-head masking, complementing the padded flow training loss.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import numpy as np


def audit_provenance(source, dataset_manifest, normalizer):
    """CPU-only preflight; return (parsed_manifest, verified_provenance).

    Hash the H5 in bounded chunks and bind the exact manifest bytes to the
    train-only normalizer. No model, H5 runtime, or GPU dependency is loaded.
    """
    spec = importlib.util.spec_from_file_location(
        'native_s1_audit_config', Path(__file__).with_name('fetch_native_s1_config.py'))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    normalizer_metadata = native.normalizer_provenance(normalizer)
    manifest_bytes = Path(dataset_manifest).read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != normalizer_metadata['source_manifest_sha256']:
        raise ValueError('Audit manifest differs from normalizer source manifest')
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256()
    with Path(source).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    source_sha256 = digest.hexdigest()
    if source_sha256 != manifest.get('source_sha256'):
        raise ValueError('Audit source differs from manifest source SHA256')
    return manifest, dict(source_sha256=source_sha256,
        source_manifest_sha256=manifest_sha256,
        normalizer_sha256=normalizer_metadata['norm_stats_sha256'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset-manifest', 'checkpoint', 'normalizer', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest, provenance = audit_provenance(a.source, a.dataset_manifest, a.normalizer)
    from multiprocessing.connection import Client
    import h5py
    from bvi.official_fetch_data import inspect_episode
    parents = {e['parent_id']:e for e in manifest['episodes']}
    auth = a.output/'auth.local'; auth.write_bytes(secrets.token_bytes(32)); auth.chmod(0o600)
    sd = a.output/'server'; sd.mkdir()
    sock = sd/'s.sock'
    server = conn = None
    rows = []; started = time.monotonic()
    def save(status, **extra):
        (a.output/'summary.json').write_text(json.dumps(dict(status=status, stage='S1',
            question='Frozen action reconstruction on teacher observations', rows=rows,
            wall_seconds=time.monotonic()-started, training_updates=0, api_calls=0,
            max_seconds=300, frame_roster=[0,1,20,21], positions=['first','middle'],
            checkpoint_selection=False, provenance=provenance, **extra), indent=2))
    save('loading')
    try:
        with (sd/'stdout.log').open('w') as log:
            server = subprocess.Popen([os.sys.executable, str(Path(__file__).with_name('serve_native_s1.py')),
                '--checkpoint', str(a.checkpoint), '--normalizer', str(a.normalizer),
                '--output', str(sd), '--socket', str(sock), '--auth-file', str(auth),
                '--gpu-uuid', 'GPU-b7ebba23-7824-7601-df32-be55628936c3',
                '--max-inference-calls', '8'], stdout=log, stderr=subprocess.STDOUT)
            while not (sd/'ready.json').exists():
                if server.poll() is not None: raise RuntimeError('Model server loading failed')
                if time.monotonic()-started > 120: raise TimeoutError('Loading timeout')
                time.sleep(1)
            conn = Client(str(sock), family='AF_UNIX', authkey=auth.read_bytes())
            with h5py.File(a.source, 'r') as h:
                for parent in (0,1,20,21):
                    record = parents[parent]
                    g = h[f'traj_{parent}']; episode = inspect_episode(g, 'pick')
                    n = episode['exported_steps']
                    if n != record['exported_steps'] or record['split'] != ('train' if parent<20 else 'validation'):
                        raise ValueError('Teacher frame differs from exported training/validation interval')
                    for position in (0,n//2):
                        if time.monotonic()-started > 280: raise TimeoutError('Audit budget')
                        seed = 10000+parent*1000+position
                        conn.send(dict(op='reset', seed=seed))
                        if not conn.poll(10): raise TimeoutError('Reset timeout')
                        if conn.recv()['status'] != 'reset': raise RuntimeError('Reset failure')
                        request = dict(head_rgb=g['obs/sensor_data/fetch_head/rgb'][position],
                            wrist_rgb=g['obs/sensor_data/fetch_hand/rgb'][position],
                            state=np.r_[g['obs/agent/qpos'][position],g['obs/agent/qvel'][position]],
                            prompt='Pick and stably hold the apple.')
                        conn.send(dict(op='predict', seed=seed, **request))
                        if not conn.poll(min(120,300-(time.monotonic()-started))):
                            raise TimeoutError('Prediction timeout')
                        response = conn.recv()
                        if response['status'] != 'ok': raise RuntimeError(str(response))
                        predicted = np.asarray(response['actions'])
                        expert = np.asarray(g['actions'][position:min(position+10,n)])
                        valid = len(expert); applied = np.clip(predicted[:valid],-1,1)
                        applied[:,8:10] = 0
                        name=f'parent{parent}-frame{position}.npz'
                        np.savez_compressed(a.output/name, **request, predicted=predicted, expert=expert)
                        rows.append(dict(parent=parent, split='train' if parent<20 else 'validation',
                            frame=position, exported_steps=n, input_artifact=name, valid_horizon=valid,
                            first_action_mse=float(np.mean((applied[0]-expert[0])**2)),
                            valid_chunk_mse=float(np.mean((applied-expert)**2)),
                            zero_action_mse=float(np.mean(expert**2)),
                            clip_fraction=float(np.mean(np.abs(predicted[:valid])>1)),
                            predicted_gripper=float(applied[0,7]), expert_gripper=float(expert[0,7]),
                            pretrained_parameters_sha256=response['pretrained_parameters_sha256']))
                        save('running')
            save('completed')
    except Exception as error:
        save('failed',error=repr(error)); raise
    finally:
        if conn is not None: conn.close()
        if server is not None and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        auth.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
