import json
import pytest
from bvi.s1_evidence import build_report, sha
from bvi.s1_capability_gate import SEEDS


def fixture_panel(tmp_path, successes=3):
    best = tmp_path/'best.json'
    best.write_text(json.dumps(dict(selection='heldout_action_loss_only', checkpoint='/best/1000')))
    for index, seed in enumerate(SEEDS):
        directory = tmp_path/f'seed{seed}'
        directory.mkdir()
        (directory/'events.jsonl').write_text(json.dumps({'info': {'success': [index < successes]}})+'\n')
        for name in ('initial-state.pt', 'reset-observation.npz'):
            (directory/name).write_bytes(b'test evidence only')
        result = dict(seed=seed, status='episode_completed', task='set_table/pick/013_apple',
                      scene_split='val', max_actions=200, control_frequency=20,
                      ee_rest_threshold_m=0.05, api_calls=0, steps=1, success=index < successes,
                      instruction='Pick and stably hold the apple.',
                      model_metadata=dict(checkpoint='/best/1000', pretrained_parameters_sha256='a'*64,
                          normalizer_sha256='b'*64, state_contract=dict(state_dim=24,
                              state_source='env_native_agent', base_camera='fetch_head', wrist_camera='fetch_hand',
                              training_stage='S1_ordinary_target_domain_SFT_not_TAPT')),
                      artifact_sha256={p.name: sha(p) for p in directory.iterdir()})
        (directory/'result.json').write_text(json.dumps(result))
    return best


def test_verified_panel_and_below_gate_are_both_preserved(tmp_path):
    best = fixture_panel(tmp_path, 2)
    result = build_report(tmp_path, best)
    assert len(result['episodes']) == 10
    assert not result['admission']['eligible']


def test_passed_panel(tmp_path):
    assert build_report(tmp_path, fixture_panel(tmp_path))['admission']['eligible']


@pytest.mark.parametrize('change', ['tamper', 'different_model', 'different_checkpoint', 'false_success', 'incomplete'])
def test_reject_invalid_evidence(tmp_path, change):
    best = fixture_panel(tmp_path)
    path = tmp_path/'seed2024'/'result.json'
    result = json.loads(path.read_text())
    if change == 'tamper':
        (path.parent/'events.jsonl').write_text('changed')
    elif change == 'different_model':
        result['model_metadata']['pretrained_parameters_sha256'] = 'c'*64
    elif change == 'different_checkpoint':
        result['model_metadata']['checkpoint'] = '/latest/2000'
    elif change == 'false_success':
        result['success'] = False
    else:
        result['status'] = 'infrastructure_failure'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        build_report(tmp_path, best)
