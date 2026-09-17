"""Admission for NEW native24 S2 only; never edits historical training holds."""
import json
from pathlib import Path
import numpy as np
from bvi.s1_capability_gate import validate_s1_capability
from bvi.s1_evidence import build_report, sha


def validate_entry(panel, best_record, handoff_manifest):
    if any(x is None for x in (panel, best_record, handoff_manifest)):
        raise ValueError('New S1 panel, heldout best record and native24 handoff evidence required')
    native = build_report(panel, best_record)
    admission = validate_s1_capability(native, native['checkpoint_sha256'])
    path = Path(handoff_manifest)
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'bvi.native24-handoff/1'
            or manifest.get('status') != 'verified_handoff_inputs'
            or manifest.get('label_contract') != 'current_observation_v2'):
        raise ValueError('Native24 wrong-handoff validation not complete')
    required = {'far_grasp_diagnostic', 'unheld_move', 'near_grasp_candidate_not_completion', 'held_move'}
    seen, identities, parents = set(), set(), []
    for case in manifest.get('cases', []):
        kind = case.get('classification')
        if kind not in required or case.get('role') != 'validation':
            raise ValueError('Incorrect handoff classification/split')
        if case.get('policy_input_privileged') is not False or case.get('source_replay_verified') is not True:
            raise ValueError('Handoff input provenance not verified')
        ident = case['case_id']
        if ident in identities:
            raise ValueError('Duplicate handoff case')
        identities.add(ident); seen.add(kind)
        parent = case.get('parent_episode')
        if not isinstance(parent, dict) or not all(k in parent for k in ('task', 'source_sha256', 'trajectory', 'parent_id')):
            raise ValueError('Handoff must identify original parent trajectory')
        parents.append(parent)
        source = path.parent / case['path']
        if sha(source) != case.get('sha256'):
            raise ValueError('Changed handoff observation')
        with np.load(source, allow_pickle=False) as data:
            for key, shape in [('head_rgb', (128,128,3)), ('wrist_rgb', (128,128,3)), ('state', (24,))]:
                if data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError('Wrong handoff native observation')
                if key.endswith('rgb') and data[key].dtype != np.uint8:
                    raise ValueError('RGB dtype differs')
        if case.get('model_progress_used_as_ground_truth') is not False:
            raise ValueError('Ground truth cannot come from predicted progress')
    if seen != required:
        raise ValueError('Missing wrong-handoff validation classes')
    return dict(native=native, admission=admission, handoff_sha256=sha(path), handoff_parents=parents,
                budget_increase=False, historical_pipeline_resume=False)
