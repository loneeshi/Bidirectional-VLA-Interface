"""Fail-closed CPU evidence gates for Fetch TAPT entry; never launches work."""
import hashlib
import json
from pathlib import Path

PRIMARY_SEEDS = [2024, 2025, 2026, 2027, 2028]
HANDOFF_CLASSES = {'far_grasp_diagnostic', 'unheld_move',
                   'near_grasp_candidate_not_completion', 'held_move'}
DEFAULT_HOLD = Path.home() / 'bvi-research/fetch-pi05-training-hold.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def scalar(value):
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError('Expected exactly one environment in native evidence')
        value = value[0]
    return value


def require_hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Missing/invalid {label} SHA256')
    return value


def check_holds(*paths):
    for path in dict.fromkeys(Path(p).resolve() for p in paths if p is not None):
        if path.exists() and read(path).get('training_allowed') is not True:
            raise ValueError(f'TAPT remains on hold: {path}')


def validate_native(path):
    report = read(path)
    if report.get('status') != 'completed' or report.get('primary_seeds') != PRIMARY_SEEDS:
        raise ValueError('Native primary five-seed protocol is incomplete or changed')
    rows = report.get('primary_episodes', [])
    if len(rows) != 5 or {r.get('seed') for r in rows} != set(PRIMARY_SEEDS):
        raise ValueError('Need five unique primary episodes; diagnostic overlap cannot count twice')
    for key in ['training_updates', 'api_calls']:
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError(f'Native capability batch must have {key}=0')
    metadata = report.get('server_metadata', report.get('model_metadata', {}))
    checkpoint_hash = require_hash(metadata.get('pretrained_parameters_sha256'), 'native source checkpoint')
    normalizer_hash = require_hash(metadata.get('normalizer_sha256'), 'native source normalizer')
    if metadata.get('progress_head') is not False or metadata.get('tapt') is not False:
        raise ValueError('Native gate must use original frozen V8 without TAPT/progress head')
    successes = 0
    episode_hashes = {}
    for row in rows:
        if not isinstance(row.get('result_path'), str) or not row['result_path'].strip():
            raise ValueError('Native gate requires actual episode result_path and result_sha256')
        result_path = Path(row['result_path'])
        result_path = result_path if result_path.is_absolute() else Path(path).resolve().parent / result_path
        expected_hash = require_hash(row.get('result_sha256'), 'native episode result')
        if sha(result_path) != expected_hash:
            raise ValueError('Native episode result bytes changed')
        episode_hashes[str(result_path.resolve())] = expected_hash
        actual = read(result_path)
        for key in ['seed', 'status', 'success', 'scene_split', 'max_actions', 'ee_rest_threshold_m', 'final_info']:
            if actual.get(key) != row.get(key):
                raise ValueError(f'Native batch summary contradicts saved episode: {key}')
        for key in ['training_updates', 'api_calls']:
            if type(actual.get(key)) is not int or actual[key] != 0:
                raise ValueError(f'Native episode must have {key}=0')
        actual_metadata = actual.get('model_metadata', {})
        for key in ['pretrained_parameters_sha256', 'normalizer_sha256']:
            if actual_metadata.get(key) != metadata[key]:
                raise ValueError(f'Native episode model identity differs from batch: {key}')
        if actual_metadata.get('progress_head') is not False or actual_metadata.get('tapt') is not False:
            raise ValueError('Native episode used TAPT or a progress head')
        if row.get('status') != 'episode_completed':
            raise ValueError('Infrastructure failures do not count as completed native failures')
        if row.get('scene_split') != 'val' or row.get('max_actions') != 200 or row.get('ee_rest_threshold_m') != .05:
            raise ValueError('Native validation/200-action/5cm protocol mismatch')
        if type(row.get('success')) is not bool:
            raise ValueError('Native success must be a boolean')
        if row['success']:
            info = row.get('final_info', row.get('native_info', {}))
            for key in ['success', 'is_grasped', 'ee_rest', 'robot_rest', 'is_static', 'cumulative_force_within_limit']:
                if scalar(info.get(key)) is not True:
                    raise ValueError(f'Claimed native success lacks actual predicate: {key}')
            if scalar(info.get('fail')) is not False:
                raise ValueError('Claimed native success has missing/true native fail flag')
            force = scalar(info.get('robot_cumulative_force'))
            if isinstance(force, bool) or not isinstance(force, (int, float)) or not 0 <= force < 5000:
                raise ValueError('Claimed native success exceeds native force limit')
            successes += 1
        elif 'final_info' in row and scalar(row['final_info'].get('success')) is not False:
            raise ValueError('Episode success contradicts native info')
    if type(report.get('primary_successes')) is not int or report['primary_successes'] != successes:
        raise ValueError('Native aggregate success count does not match episodes')
    if report.get('pi05_tapt_stop_this_week') is not (successes <= 1):
        raise ValueError('Native stop decision disagrees with <=1/5 rule')
    if successes <= 1:
        raise ValueError('Frozen V8 native success <=1/5: pi0.5 TAPT is stopped this week')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), primary_successes=successes,
                pretrained_parameters_sha256=checkpoint_hash, normalizer_sha256=normalizer_hash,
                episode_result_sha256=episode_hashes)


def validate_handoff(path):
    import numpy as np
    path = Path(path).resolve()
    manifest = read(path)
    if manifest.get('status') != 'verified_handoff_inputs':
        raise ValueError('Actual verified handoff inputs required; plan_only is not a validation gate')
    cases = manifest.get('cases', [])
    observed, case_ids, input_hashes = set(), set(), {}
    for case in cases:
        if case.get('role') != 'validation':
            raise ValueError('Only locked validation parents3020/3021 enter this handoff manifest')
        parent = case.get('original_parent', {})
        if parent.get('scene_split') != 'train' or type(parent.get('seed')) is not int or parent['seed'] not in (3020, 3021):
            raise ValueError('Handoff parent must be validation3020/3021 in scene train')
        kind = case.get('classification')
        if kind not in HANDOFF_CLASSES or case.get('workspace_replay_verified') is not True:
            raise ValueError('Wrong handoff classification or unverified workspace replay')
        case_id = case.get('case_id')
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError('Unique handoff case IDs required')
        case_ids.add(case_id); observed.add(kind)
        source = Path(case['input_path'])
        source = source if source.is_absolute() else path.parent / source
        expected = require_hash(case.get('npz_sha256'), 'handoff input')
        if sha(source) != expected:
            raise ValueError('Handoff input bytes changed or are not the verified artifact')
        with np.load(source, allow_pickle=False) as archive:
            for key, shape in [('workspace_rgb', (224, 224, 3)), ('wrist_rgb', (128, 128, 3)), ('state', (30,))]:
                value = archive[key]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid verified handoff input {key}')
                if key.endswith('rgb') and value.dtype != np.uint8:
                    raise ValueError('Handoff cameras must be actual uint8 RGB arrays')
        input_hashes[case_id] = expected
    if observed != HANDOFF_CLASSES:
        raise ValueError('Need far-grasp/unheld-move cases and near-grasp/held-move controls')
    return dict(path=str(path), sha256=sha(path), input_sha256=input_hashes,
                scope='input-readiness only; learned-head feedback/rollback acceptance remains separate')


def validate_entry(native_report, handoff_manifest, hold_file=None):
    check_holds(DEFAULT_HOLD, hold_file)
    if native_report is None or handoff_manifest is None:
        raise ValueError('TAPT requires --native-capability-report and --handoff-validation-manifest')
    return dict(native=validate_native(native_report), handoff=validate_handoff(handoff_manifest),
                authorizes_automatic_resume=False)
