"""CPU S1 normalization: training parents only, each recorded frame once.

Uses pinned OpenPI RunningStats; no action delta transform or chunk reweighting.
The resulting raw state24/action13 statistics are applied before model padding.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import pyarrow.parquet as pq
    from openpi.shared import normalize

    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'converted_not_training_validated' or manifest['state_dim'] != 24:
        raise ValueError('Expected completed native24 export')
    train_ids = manifest['parent_split']['train']
    if set(train_ids) & set(manifest['parent_split']['validation']):
        raise ValueError('Leaked parent split')
    parents = [e for e in manifest['episodes'] if e['split'] == 'train']
    if [e['parent_id'] for e in parents] != train_ids:
        raise ValueError('Incomplete/misordered training parents')
    files = sorted((args.dataset / 'train' / 'data').rglob('*.parquet'))
    if len(files) != len(parents):
        raise ValueError('Training episode count differs')
    stats = {key: normalize.RunningStats() for key in ('state', 'actions')}
    evidence = []
    frames = 0
    for episode_index, (path, parent) in enumerate(zip(files, parents, strict=True)):
        data = pq.read_table(path, columns=['state', 'actions', 'episode_index']).to_pydict()
        if set(data['episode_index']) != {episode_index}:
            raise ValueError('Parquet episode mapping differs')
        n = parent['exported_steps']
        for key, dimension in [('state', 24), ('actions', 13)]:
            values = np.asarray(data[key], dtype=np.float64)
            if values.shape != (n, dimension) or not np.isfinite(values).all():
                raise ValueError('Invalid normalization source')
            stats[key].update(values)
        frames += n
        evidence.append(dict(parent_id=parent['parent_id'], frames=n,
                             parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = {key: value.get_statistics() for key, value in stats.items()}
    for values in result.values():
        for field in ('mean', 'std', 'q01', 'q99'):
            if not np.isfinite(getattr(values, field)).all():
                raise ValueError('Nonfinite normalization output')
    args.output.mkdir(parents=True, exist_ok=False)
    normalize.save(args.output, result)
    report = dict(status='train_only_stats_ready_not_runtime_validated', frames=frames,
                  parents=evidence, validation_frames_used=0,
                  weighting='each_pre_action_training_frame_once_no_chunk_reweighting',
                  dimensions=dict(state=24, actions=13), delta_transform=False,
                  source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  implementation_sha256=hashlib.sha256(inspect.getsource(normalize).encode()).hexdigest(),
                  norm_stats_sha256=hashlib.sha256((args.output/'norm_stats.json').read_bytes()).hexdigest(),
                  training_updates=0, api_calls=0)
    (args.output/'provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'parents'}))


if __name__ == '__main__':
    main()
