"""Materialize native24 current-observation calls on CPU, retaining N+1 endpoints.

This is teacher-call data only. Wrong-handoff validation is a separate entry gate.
No labels are inferred from file endings; excluded trajectories stay in manifest.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from bvi.s1_evidence import sha
from bvi.official_fetch_data import inspect_episode, current_progress_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--manifests', type=Path, nargs='+', required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    contract = json.loads(a.contract.read_text())
    if contract.get('state_dim') != 24 or contract.get('state_source') != 'env_native_agent':
        raise ValueError('Expected native S1 observation contract')
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='fetch_native24_family_v1', label_contract='current_observation_v2',
        state_contract=contract, normalizer_sha256=contract['normalizer_sha256'],
        status='building', training_ready=False, source_collection_complete=False,
        fixed_collection_complete=False, episodes=[], invocations=[], sources=[], exclusions=[],
        wrong_handoff_validation_complete=False, training_updates=0, api_calls=0)
    def save():
        (a.output/'manifest.json').write_text(json.dumps(report, indent=2))
    save()
    try:
        for source_manifest in a.manifests:
            manifest = json.loads(source_manifest.read_text())
            current_progress_rows(manifest, 10)  # Checks complete parent split and window bounds.
            task = manifest['task']
            source = a.source/task/'013_apple.h5'
            if sha(source) != manifest['source_sha256']:
                raise ValueError('Official source bytes changed')
            report['sources'].append(dict(manifest_sha256=sha(source_manifest), source_sha256=manifest['source_sha256'],
                h5_sha256=manifest['source_sha256'], task=task))
            with h5py.File(source, 'r') as h:
                for row in manifest['episodes']:
                    group = h[row['trajectory']]
                    episode = inspect_episode(group, task)
                    if episode['windows'] != row['windows']:
                        raise ValueError('Call segmentation differs from frozen manifest')
                    parent = dict(task=task, source_sha256=manifest['source_sha256'],
                                  trajectory=row['trajectory'], parent_id=row['parent_id'])
                    report['episodes'].append(dict(split=row['split'], parent_episode=parent))
                    if not row['windows']:
                        report['exclusions'].append(dict(parent_episode=parent, reason='no_evidenced_call_windows'))
                    elif task == 'place' and not any(w['family'] == 'release' for w in row['windows']):
                        report['exclusions'].append(dict(parent_episode=parent, family='release',
                            reason='frozen_segmentation_predicate_not_met_no_threshold_relaxation'))
                    for call, window in enumerate(row['windows']):
                        start, end = window['start'], window['end']
                        n = end-start+1
                        real = episode['actions'][start:end]
                        if len(real) != n-1 or not len(real):
                            raise ValueError('Invalid real action span')
                        # Endpoint storage repeats context; its action loss is always masked.
                        actions = np.concatenate([real, real[-1:]], axis=0)
                        path = a.output/f'{task}-{row["trajectory"]}-call{call:02d}.npz'
                        np.savez_compressed(path,
                            head_rgb=group['obs/sensor_data/fetch_head/rgb'][start:end+1],
                            wrist_rgb=group['obs/sensor_data/fetch_hand/rgb'][start:end+1],
                            state=episode['state'][start:end+1], actions=actions,
                            progress=np.arange(n, dtype=np.float32)/(n-1),
                            action_valid=np.arange(n)<n-1, progress_valid=np.ones(n, bool))
                        report['invocations'].append(dict(path=path.name, sha256=sha(path),
                            split=row['split'], family=window['family'], instruction=window['instruction'],
                            parent_episode=parent, completion_evidence=window['completion_evidence'],
                            completion_verified=True, source_interval=[start,end]))
        from train_native_s2 import load_dataset
        save(); load_dataset(a.output)
        report.update(status='built_all_families', training_ready=True,
                      source_collection_complete=True, fixed_collection_complete=True)
    except Exception as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        save()
    print(json.dumps(dict(status=report['status'], calls=len(report['invocations']),
                          wrong_handoff_validation_complete=False)))


if __name__ == '__main__':
    main()
