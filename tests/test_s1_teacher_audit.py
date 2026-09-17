"""CPU-only bindings for the frozen teacher-action audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    'teacher_action_audit', Path(__file__).parents[1] / 'scripts/audit_s1_teacher_actions.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'source.h5'
    source.write_bytes(b'fixture source bytes; no H5 runtime required')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                    'episodes': []}))
    normalizer = tmp_path / 'normalizer'
    normalizer.mkdir()
    stats = normalizer / 'norm_stats.json'
    stats.write_text('{}')
    (normalizer / 'provenance.json').write_text(json.dumps({
        'status': 'train_only_stats_ready_not_runtime_validated',
        'validation_frames_used': 0, 'dimensions': {'state': 24, 'actions': 13},
        'delta_transform': False,
        'norm_stats_sha256': hashlib.sha256(stats.read_bytes()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    return source, manifest, normalizer


def test_verified_hashes_and_streamed_source(inputs, monkeypatch):
    source, manifest, normalizer = inputs
    read_bytes = Path.read_bytes
    def reject_whole_source_read(path):
        assert path != source, 'H5 must be streamed'
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', reject_whole_source_read)
    parsed, provenance = audit.audit_provenance(*inputs)
    assert provenance['source_sha256'] == parsed['source_sha256']
    assert provenance['source_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert provenance['normalizer_sha256'] == hashlib.sha256((normalizer / 'norm_stats.json').read_bytes()).hexdigest()


def test_reject_different_source_even_with_same_length(inputs):
    source, _, _ = inputs
    original = source.read_bytes()
    source.write_bytes(b'x' * len(original))
    with pytest.raises(ValueError, match='manifest source SHA256'):
        audit.audit_provenance(*inputs)


def test_bind_exact_manifest_bytes(inputs):
    _, manifest, _ = inputs
    manifest.write_bytes(manifest.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='normalizer source manifest'):
        audit.audit_provenance(*inputs)


def test_reject_invalid_normalizer_provenance(inputs):
    _, _, normalizer = inputs
    path = normalizer / 'provenance.json'
    metadata = json.loads(path.read_text())
    metadata['delta_transform'] = True
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='normalizer provenance'):
        audit.audit_provenance(*inputs)


def test_main_rejects_before_gpu_launch(inputs, tmp_path, monkeypatch):
    source, manifest, normalizer = inputs
    source.write_bytes(b'wrong source')
    def forbidden_launch(*args, **kwargs):
        pytest.fail('GPU server must not launch before provenance passes')
    monkeypatch.setattr(audit.subprocess, 'Popen', forbidden_launch)
    monkeypatch.setattr(sys, 'argv', ['audit', '--source', str(source),
        '--dataset-manifest', str(manifest), '--normalizer', str(normalizer),
        '--checkpoint', str(tmp_path / 'unused-checkpoint'), '--output', str(tmp_path / 'audit')])
    with pytest.raises(ValueError, match='manifest source SHA256'):
        audit.main()
