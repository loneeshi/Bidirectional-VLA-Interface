"""CPU-only preparation for a paired progress-head comparison, not training.

STATUS: frozen — historical training and diagnostics (retained)

Checks archived metadata and binds cache rows. Tensor contents, clean feature
extraction, action identity and head performance require separate execution.
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping

from .fetch_current_labels import (
    ANCHOR_SUPERVISION, CONTRACT, first_call_anchor,
    validate_manifest_splits, verified_teacher_targets,
)

FAMILIES = ('reach', 'grasp', 'move', 'release')
CLEAN_BATCH_KEYS = (
    'lang_tokens', 'lang_attn_mask', 'img_tokens', 'pc_tokens',
    'state_tokens', 'action_mask', 'ctrl_freqs', 'in_context_conditions',
)


def clean_batch_view(batch):
    """Validate the original predict_action argument boundary without tensors.

    Values are not copied or numerically inspected here. Callers must not mutate
    them; this function does not itself implement observation feature extraction.
    """
    if not isinstance(batch, Mapping) or set(batch) != set(CLEAN_BATCH_KEYS):
        raise ValueError('Expected exact clean predict_action keys; no actions/latents/noise')
    return {key: batch[key] for key in CLEAN_BATCH_KEYS}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Missing or invalid SHA256')
    return value


def _canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_pairing(rows, report, archive, backup, selected):
    """Return a deterministic, cache-bound development manifest.

    Inputs are existing JSON artifacts. No model, archive extraction, tensor load,
    new hash of cache bytes, or resource launch occurs in this function.
    """
    if not rows or report.get('progress_label_contract') != CONTRACT:
        raise ValueError('Requires nonempty current-observation calibration')
    if report.get('frozen_native_unchanged') is not True:
        raise ValueError('Native freeze audit is missing')
    if archive.get('status') != 'verified_all_archived_files_match_frozen_source':
        raise ValueError('Source archive verification is missing')
    if backup.get('local_matches_remote_archive') is not True:
        raise ValueError('Local archive verification is missing')
    if _sha(archive['sha256']) != _sha(backup['sha256']) or archive['bytes'] != backup['bytes']:
        raise ValueError('Archive provenance disagrees')
    if selected['selected_step'] != report['best_step']:
        raise ValueError('Selected checkpoint disagrees')
    parents = validate_manifest_splits(rows)
    archived = {}
    for item in archive['files']:
        if item['path'] in archived:
            raise ValueError('Duplicate archive member')
        archived[item['path']] = item
    batch_id = backup['experiment_id']
    if archived.get(f'{batch_id}/best.pt', {}).get('sha256') != _sha(selected['checkpoint_sha256']):
        raise ValueError('Selected checkpoint does not match archived best.pt')
    expected_cache = {f'{index:05d}.pt' for index in range(len(rows))}
    if set(report['cache_sha256']) != expected_cache or report['cached_examples'] != len(rows):
        raise ValueError('Cache count/order binding disagrees')
    reference_rows = selected['selected']['rows']
    validation_index = 0
    output, identities = [], set()
    counts = Counter()
    for index, row in enumerate(rows):
        parent = row.get('parent_episode', row)
        if row['split'] not in ('train', 'validation') or parent['seed'] in (2025, 2030):
            raise ValueError('Diagnostic/test episodes cannot enter development data')
        if row.get('family') not in FAMILIES:
            raise ValueError('Unknown tool family')
        horizon = len(row['progress_target'])
        if horizon != 2:
            raise ValueError('This fixed cache comparison requires horizon2')
        parent_id = {key: parent[key] for key in ('scene_split', 'task', 'seed')}
        identity = dict(parent=parent_id, family=row['family'], kind=row['kind'])
        evidence = {}
        if row['kind'] == 'teacher':
            targets = verified_teacher_targets(row, row['sample_t'], horizon)
            identity.update(start=row['start'], end=row['end'], observation=row['sample_t'])
            evidence['trajectory_sha256'] = _sha(
                report['source_provenance']['trajectory_sha256'][row['trajectory']])
            evidence['completion_evidence'] = row['completion_evidence']
        elif row['kind'] == 'invocation_start_anchor':
            if row.get('supervision') != ANCHOR_SUPERVISION:
                raise ValueError('Unrecognized anchor supervision')
            if row['source_checkpoint_sha256'] != report['warmstart_sha256']:
                raise ValueError('Anchor collection checkpoint disagrees')
            targets = first_call_anchor(horizon)
            identity.update(call_id=row['call_id'], decision_id=row['decision_id'],
                            observation=row['prediction_step'])
            for key in ('inputs_sha256', 'raw_sha256', 'events_sha256', 'episode_result_sha256'):
                evidence[key] = _sha(row[key])
        else:
            raise ValueError('Unknown sample kind')
        for key, target in targets.items():
            actual = row[key]
            if key.endswith('_valid') and any(type(x) is not bool for x in actual):
                raise ValueError('Label masks must be boolean')
            if key == 'progress_target' and any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in actual):
                raise ValueError('Progress targets must be finite real numbers, not booleans')
            if actual != target:
                raise ValueError('Label/mask differs from current-observation contract')
        sample_id = _canonical_sha(identity)
        if sample_id in identities:
            raise ValueError('Duplicate semantic sample identity')
        identities.add(sample_id)
        cache_name = f'{index:05d}.pt'
        member_name = f'{batch_id}/cache/{cache_name}'
        digest = _sha(report['cache_sha256'][cache_name])
        if archived.get(member_name, {}).get('sha256') != digest:
            raise ValueError('Cache SHA differs between report and archive evidence')
        prepared = dict(sample_id=sample_id, source_row=index, split=row['split'],
                        identity=identity, cache_member=member_name, cache_sha256=digest,
                        labels=targets, source_evidence=evidence)
        if row['split'] == 'validation':
            if validation_index >= len(reference_rows):
                raise ValueError('Missing paired baseline validation row')
            reference = reference_rows[validation_index]
            if ((reference['family'], reference['kind']) != (row['family'], row['kind'])
                    or not math.isclose(reference['current_target'], row['progress_target'][0],
                                        rel_tol=0, abs_tol=1e-7)):
                raise ValueError('Baseline validation ordering disagrees')
            prepared['validation_index'] = validation_index
            prepared['baseline_diffusion_seed'] = 8181 + 2 * validation_index
            validation_index += 1
        counts[(row['split'], row['family'], row['kind'])] += 1
        output.append(prepared)
    if validation_index != len(reference_rows):
        raise ValueError('Unpaired baseline validation rows')
    for split in ('train', 'validation'):
        for family in FAMILIES:
            if not counts[(split, family, 'teacher')]:
                raise ValueError('Each family requires train and validation teacher support')
    return {
        'schema_version': 'observation_progress_pairing_v1',
        'status': 'metadata_prepared_model_not_implemented_or_evaluated',
        'progress_contract': CONTRACT,
        'source_batch': batch_id,
        'source_manifest_semantic_sha256': _canonical_sha(rows),
        'source_archive_sha256': archive['sha256'],
        'verification_scope': 'Metadata bindings against existing hash evidence only; cache tensors not loaded in this preflight.',
        'comparison_scope': f'Paired development validation; these {validation_index} rows previously selected the reference checkpoint. Not untouched test evidence.',
        'fixed_policy': {
            'checkpoint_sha256': _sha(selected['checkpoint_sha256']),
            'selected_additional_updates': selected['selected_step'],
            'native_parameters_and_buffers_sha256': _sha(report['frozen_native_sha256_after']),
            'freeze': ['native backbone', 'mobility expert', 'all four family LoRA banks', 'existing progress head', 'preprocessing/encoders'],
            'runtime_fingerprint_and_restored_rng_action_identity': 'required_not_yet_executed',
        },
        'observation_contract': {
            'clean_batch_keys': list(CLEAN_BATCH_KEYS),
            'privileged_simulator_context': True,
            'excluded': ['teacher actions', 'diffusion noise/timestep', 'mobility latent tokens', 'post-DiT hidden states', 'progress targets', 'physical diagnostic labels'],
            'language_validity': 'preserve lang_attn_mask',
            'image_pc_availability': 'not present in cached batches; substituted background/duplicated history retained as valid for proposed paired port; not inferred from embeddings',
            'action_mask_semantics': 'state/action channel validity, never image/PC availability',
            'candidate_architecture': 'AC-DiT observation-conditioned port; no author contextualized prefix and no inherited family LoRA in clean branch',
        },
        'runtime_gates_pending': [
            'Verify selected checkpoint and cache bytes before tensor loading',
            'Check tensor metadata, shapes, finite values and exact clean input keys',
            'Implement one shared clean feature path for train and inference',
            'Show identical progress under action-noise/timestep perturbation',
            'Show unchanged actions with restored RNG and frozen tensors/buffers',
            'Report current/chunk MSE by family and teacher/anchor separately',
            'Do not claim physical completion or task success from temporal-label MSE',
        ],
        'counts': {
            split: {family: {kind: counts[(split, family, kind)]
                            for kind in ('teacher', 'invocation_start_anchor')}
                    for family in FAMILIES} for split in ('train', 'validation')},
        'parents': [{'scene_split': key[0], 'task': key[1], 'seed': key[2], 'split': split}
                    for key, split in sorted(parents.items())],
        'samples': output,
    }
