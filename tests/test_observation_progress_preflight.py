"""Regression checks against the real, completed calibration metadata."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bvi.observation_progress_preflight import CLEAN_BATCH_KEYS, clean_batch_view, prepare_pairing

SOURCE = Path(__file__).resolve().parents[1] / 'docs/results/fetch-current-calibration-2026-09-17-run01'


@pytest.fixture
def evidence():
    return [json.loads((SOURCE / name).read_text(encoding='utf-8')) for name in (
        'split-manifest.json', 'result.json', 'archive-verification.json',
        'backup.json', 'selected-validation.json')]


def test_real_archive_keeps_validation_order_noise_and_no_test_leak(evidence):
    result = prepare_pairing(*evidence)
    samples = result['samples']
    assert len(samples) == 375
    held = [row for row in samples if row['split'] == 'validation']
    assert len(held) == 39
    assert [r['baseline_diffusion_seed'] for r in held] == list(range(8181, 8259, 2))
    assert not any(p['seed'] in (2025, 2030) for p in result['parents'])
    assert 'all four family LoRA banks' in result['fixed_policy']['freeze']
    assert 'not_implemented_or_evaluated' in result['status']


@pytest.mark.parametrize('bad_key', ['action_gt', 'latent_mobility_cond', 'timestep', 'actions', 'physical_diagnostics_only'])
def test_observation_boundary_refuses_action_or_label_leakage(bad_key):
    batch = dict.fromkeys(CLEAN_BATCH_KEYS, object())
    assert clean_batch_view(batch) == batch
    batch[bad_key] = object()
    with pytest.raises(ValueError, match='exact clean'):
        clean_batch_view(batch)


def test_missing_validity_mask_is_not_invented():
    batch = dict.fromkeys(CLEAN_BATCH_KEYS, object())
    del batch['lang_attn_mask']
    with pytest.raises(ValueError):
        clean_batch_view(batch)


def test_failed_invocation_cannot_gain_completion_or_future_targets(evidence):
    row = next(x for x in evidence[0] if x['kind'] == 'invocation_start_anchor')
    row['progress_target'][1] = 1.0
    row['progress_valid'][1] = True
    with pytest.raises(ValueError, match='Label/mask'):
        prepare_pairing(*evidence)


def test_teacher_endpoint_cannot_gain_fabricated_action(evidence):
    row = next(x for x in evidence[0] if x['kind'] == 'teacher' and x['sample_t'] == x['end'])
    row['action_valid'][0] = True
    with pytest.raises(ValueError, match='Label/mask'):
        prepare_pairing(*evidence)


def test_parent_leak_between_teacher_and_anchor_rejected(evidence):
    row = next(x for x in evidence[0] if x['kind'] == 'invocation_start_anchor' and x['seed'] == 3000)
    row['split'] = 'validation'
    with pytest.raises(ValueError, match='crosses splits'):
        prepare_pairing(*evidence)


def test_test_seed_cannot_be_used_for_development_model_selection(evidence):
    row = next(x for x in evidence[0] if x['split'] == 'validation')
    row['parent_episode'] = dict(scene_split='val', task='pick', seed=2030)
    with pytest.raises(ValueError, match='Diagnostic/test'):
        prepare_pairing(*evidence)


def test_wrong_cache_pairing_is_not_accepted(evidence):
    evidence[1]['cache_sha256']['00000.pt'] = 'a' * 64
    with pytest.raises(ValueError, match='Cache SHA'):
        prepare_pairing(*evidence)


def test_comparison_cannot_silently_change_action_checkpoint(evidence):
    evidence[4]['checkpoint_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='archived best'):
        prepare_pairing(*evidence)


def test_validation_reference_order_must_match(evidence):
    evidence[4]['selected']['rows'][0]['current_target'] = .765
    with pytest.raises(ValueError, match='ordering'):
        prepare_pairing(*evidence)


@pytest.mark.parametrize('bad_target', [True, float('nan'), float('inf')])
def test_non_numeric_progress_never_passes_equality_shortcuts(evidence, bad_target):
    evidence[0][0]['progress_target'][0] = bad_target
    with pytest.raises(ValueError, match='finite real'):
        prepare_pairing(*evidence)


def test_cli_binds_baseline_predictions_to_archive_not_only_matching_labels(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('split-manifest.json', 'result.json', 'archive-verification.json',
                 'backup.json', 'selected-validation.json'):
        shutil.copyfile(SOURCE / name, source / name)
    output = tmp_path / 'paired.json'
    script = SOURCE.parents[2] / 'scripts/prepare_fetch_observation_progress.py'
    command = [sys.executable, str(script), '--source', str(source), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original = output.read_bytes()
    # A baseline prediction can change while checkpoint SHA, targets and family
    # remain identical. Only the archived reference bytes catch this corruption.
    path = source / 'selected-validation.json'
    baseline = json.loads(path.read_text())
    baseline['selected']['rows'][0]['current_prediction'] = .123456
    path.write_text(json.dumps(baseline), encoding='utf-8')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'selected-validation.json does not match archived LF source bytes' in result.stderr
    assert output.read_bytes() == original
