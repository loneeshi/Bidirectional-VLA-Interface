"""CPU data-boundary checks; GPU gradients remain a separate acceptance gate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/train_fetch_pi05_family.py'
spec = importlib.util.spec_from_file_location('fetch_family_trainer', SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def dataset(tmp_path):
    records = []
    for split in ('train', 'validation'):
        for family in trainer.FAMILIES:
            path = tmp_path / f'{split}-{family}.npz'
            np.savez(path, workspace_rgb=np.zeros((3, 224, 224, 3), np.uint8),
                wrist_rgb=np.zeros((3, 128, 128, 3), np.uint8),
                state=np.zeros((3, 30), np.float32), actions=np.zeros((3, 13), np.float32),
                progress=np.array([0, .5, 1], np.float32), action_valid=np.array([1, 1, 0], bool),
                progress_valid=np.ones(3, bool))
            records.append(dict(path=path.name, sha256=trainer.sha(path), split=split,
                family=family, instruction=f'{family} object', completion_evidence='verified_test',
                parent_episode=dict(seed=1 if split == 'train' else 2, task=family)))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(invocations=records)))
    return records


def test_endpoint_has_progress_without_action(tmp_path):
    dataset(tmp_path)
    _, tables, _ = trainer.load_dataset(tmp_path)
    indices, action_mask, progress_mask = trainer.chunk(tables[0], 1)
    assert action_mask.tolist() == [1, 0] + [0] * 8
    assert progress_mask.tolist() == [1, 1] + [0] * 8
    _, action_mask, progress_mask = trainer.chunk(tables[0], 2)
    assert not action_mask.any()
    assert progress_mask.tolist() == [1] + [0] * 9


def test_suffix_context_repeats_real_action_not_dummy_endpoint(tmp_path):
    dataset(tmp_path)
    _, tables, _ = trainer.load_dataset(tmp_path)
    data = tables[0]
    data['actions'][0] = .2
    data['actions'][1] = -.7
    data['actions'][2] = 0  # Endpoint placeholder is deliberately unlike real action.
    assert np.allclose(trainer.action_context(data, 0)[0], .2)
    assert np.allclose(trainer.action_context(data, 0)[1:], -.7)
    assert np.allclose(trainer.action_context(data, 2), -.7)
    _, amask, pmask = trainer.chunk(data, 2)
    assert not amask.any() and pmask.sum() == 1


def test_explicit_unverified_completion_rejected(tmp_path):
    records = dataset(tmp_path)
    records[0]['completion_verified'] = False
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(invocations=records)))
    with pytest.raises(ValueError, match='completion evidence'):
        trainer.load_dataset(tmp_path)


def test_non_current_progress_rejected_even_inside_unit_interval(tmp_path):
    records = dataset(tmp_path)
    path = tmp_path / records[0]['path']
    with np.load(path) as source:
        data = {k: source[k].copy() for k in source.files}
    data['progress'] = np.array([.2, .5, .8], np.float32)
    np.savez(path, **data)
    records[0]['sha256'] = trainer.sha(path)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(invocations=records)))
    with pytest.raises(ValueError, match='current-observation'):
        trainer.load_dataset(tmp_path)


def test_parent_leakage_rejected_regardless_dict_order(tmp_path):
    records = dataset(tmp_path)
    records[4]['parent_episode'] = dict(task='reach', seed=1)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(invocations=records)))
    with pytest.raises(ValueError, match='leaks'):
        trainer.load_dataset(tmp_path)


def test_modified_npz_rejected(tmp_path):
    records = dataset(tmp_path)
    with (tmp_path / records[0]['path']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        trainer.load_dataset(tmp_path)


def test_dry_run_without_model_or_gpu(tmp_path):
    dataset(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), '--data', str(tmp_path), '--dry-run'],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['status'] == 'cpu_dataset_validated_no_training'
