"""CPU-only verification of downloaded evidence; never loads policy weights."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    copied = json.loads((ROOT / 'remote-copy-manifest.json').read_text())
    checked = 0
    for row in copied:
        if row.get('missing'):
            continue
        data = (ROOT / row['local']).read_bytes()
        assert len(data) == row['bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        assert row['stable_metadata_during_read']
        checked += 1
    roster = json.loads((ROOT / 'run/data/training-manifest.json').read_text())
    collection = json.loads((ROOT / 'run/data/collection.json').read_text())
    train, dev = set(roster['train_parents']), set(roster['validation_parents'])
    assert not train & dev
    episodes = {e['episode_id']: e for e in collection['episodes'] if e['success']}
    assert len(train | dev) == roster['unique_episodes'] == 743
    assert not ({episodes[i]['build_config_idx'] for i in train} &
                {episodes[i]['build_config_idx'] for i in dev})
    values = [json.loads(s) for s in (ROOT / 'run/stage1/validation.jsonl').read_text().splitlines()]
    assert all(v['identities'] == values[0]['identities'] for v in values)
    assert all(x['parent'] in dev for v in values for x in v['identities'])
    logs = [json.loads(s) for s in (ROOT / 'run/stage1/train.jsonl').read_text().splitlines()]
    assert [x['update'] for x in logs] == list(range(1, len(logs) + 1))
    assert all(math.isfinite(x[k]) for x in logs for k in ['loss', 'grad_norm'])
    syntax = 0
    for folder in ['remote-code', 'remote-source']:
        for p in (ROOT / folder).rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
            syntax += 1
    best = min(values[1:], key=lambda x: x['score'])
    out = {'status': 'passed', 'copied_files_sha256_verified': checked,
           'train_parents': len(train), 'dev_parents': len(dev),
           'scene_disjoint': True, 'fixed_dev_identities': True,
           'validation_records': len(values), 'finite_contiguous_train_updates': len(logs),
           'selected_stage1_dev_update_in_snapshot': best['update'],
           'selected_stage1_dev_score_in_snapshot': best['score'],
           'python_files_syntax_checked': syntax, 'gpu_used': False,
           'fresh_process_model_load': 'not_run', 'native_validation': 'not_run',
           'hdf5_hash': 'recorded_from_training_manifest_not_rehashed_this_turn',
           'checkpoint_tensor_integrity': 'not_independently_checked_this_turn'}
    (ROOT / 'static-verification.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
