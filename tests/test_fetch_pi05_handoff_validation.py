import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('handoff_plan', ROOT / 'scripts/prepare_fetch_pi05_handoff_validation.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
RESULTS = ROOT / 'docs/results'


def test_actual_locked_boundaries_keep_physics_separate_from_temporal_labels():
    result = plan.build(RESULTS, True)
    cases = result['fixed_cases']
    assert {c['seed'] for c in cases} <= {3020, 3021, 2025, 2030}
    far = next(c for c in cases if c['seed'] == 3021 and c['family'] == 'grasp')
    assert far['boundary_step'] == 31
    assert far['tcp_object_distance_m'] == pytest.approx(.4221891285)
    assert far['temporal_start_label'] == 0
    assert far['physical_completion'] is None
    assert far['failed_endpoint_label'] is None
    bad_move = next(c for c in cases if c['seed'] == 3021 and c['family'] == 'move')
    good_move = next(c for c in cases if c['seed'] == 3020 and c['family'] == 'move')
    assert not bad_move['is_grasped'] and good_move['is_grasped']
    assert len(bad_move['applied_action_prefix']) == bad_move['boundary_step']
    assert not result['pairing']['same_parent_valid_wrong_pairs_available']
    assert all(not s['recorded_workspace_camera'] for s in result['sources'])
    assert all(j['fallback_to_head_rgb'] is False for j in result['required_replay_jobs'])
    legacy = next(c for c in cases if c['seed'] == 2025 and c['family'] == 'grasp' and c['boundary_step'] == 16)
    assert legacy['tcp_object_distance_m'] == pytest.approx(.24119285117)


def test_training_parent_rejected_before_any_case_added():
    with pytest.raises(ValueError, match='Training parent'):
        plan.audit_episode(RESULTS / plan.HANDOFF_BATCH / 'train-seed3003', 'validation')


def test_broken_prefix_cannot_be_called_replayable(tmp_path):
    source = RESULTS / plan.HANDOFF_BATCH / 'validation-seed3021'
    shutil.copy(source / 'result.json', tmp_path)
    events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
    events = [e for e in events if not (e['event'] == 'step' and e['step'] == 2)]
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match='not contiguous'):
        plan.audit_episode(tmp_path, 'validation')


def test_fixed_plan_does_not_silently_train_or_claim_exact_replay():
    result = plan.build(RESULTS)
    assert result['training_updates'] == result['api_calls'] == result['simulator_runs'] == 0
    assert all(not j['exact_state_restoration_proven'] for j in result['required_replay_jobs'])
    assert result['evaluation_contract']['threshold_changes'] is False
    assert result['evaluation_contract']['training_or_checkpoint_selection'] is False
