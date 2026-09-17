"""CPU provenance checks: reject unrelated normalization before runtime loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'native_s1_config_audit', Path(__file__).parents[1] / 'scripts/fetch_native_s1_config.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_existing_pilot_normalizer_remains_valid():
    directory = Path(__file__).parents[1] / 'docs/results/s1-official-pilot-norm-2026-09-17-run01'
    provenance = native.normalizer_provenance(directory)
    assert provenance['delta_transform'] is False


@pytest.mark.parametrize('delta', [True, None, 0])
def test_reject_delta_or_missing_action_convention(tmp_path, delta):
    source = Path(__file__).parents[1] / 'docs/results/s1-official-pilot-norm-2026-09-17-run01'
    (tmp_path / 'norm_stats.json').write_bytes((source / 'norm_stats.json').read_bytes())
    provenance = json.loads((source / 'provenance.json').read_text())
    provenance['delta_transform'] = delta
    (tmp_path / 'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='normalizer provenance'):
        native.normalizer_provenance(tmp_path)


@pytest.mark.parametrize('split', ['train', 'validation'])
@pytest.mark.parametrize('binding', [None, '0' * 64])
def test_reject_wrong_or_unbound_dataset_before_loading(tmp_path, split, binding):
    (tmp_path / 'manifest.json').write_text('{}')
    config = SimpleNamespace(normalizer_source_manifest_sha256=binding)
    with pytest.raises(ValueError, match='normalizer source manifest'):
        native.native_dataset(config, None, tmp_path / split)


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_valid_binding_reaches_parent_validation(tmp_path, split):
    # Deliberate parent inconsistency verifies binding accepts both splits under
    # the same source manifest, before any LeRobot/OpenPI dependency is imported.
    manifest = {'parent_split': {'train': [1], 'validation': [2]}, 'episodes': []}
    content = json.dumps(manifest).encode()
    (tmp_path / 'manifest.json').write_bytes(content)
    config = SimpleNamespace(normalizer_source_manifest_sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match='Parent membership differs'):
        native.native_dataset(config, None, tmp_path / split)
