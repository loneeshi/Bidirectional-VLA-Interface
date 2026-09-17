import json
from pathlib import Path
import pytest
from bvi.baseline_v2 import validate_bound_panel
from bvi.protocol import ProtocolError


def bound():
    return dict(status='bound_ready_for_preflight',conditions=['fixed','vlm_rule','vlm_learned'],
        navigation_panels=['ppo','lightnav'],task='tidy_house',evaluation_split='val',
        evaluation_episodes=[dict(seed=i,plan_uid=f'test-only-{i}') for i in range(10)],
        max_env_steps=7000,max_vlm_calls=40,api_usd_per_episode=0.05)


def test_real_draft_cannot_start_evaluation():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/weekly-baseline-v2.json').read_text())
    with pytest.raises(ProtocolError):validate_bound_panel(cfg,'fixed','ppo')


def test_missing_uid_rejected_even_with_ready_status():
    cfg=bound();cfg['evaluation_episodes'][5]['plan_uid']=None
    with pytest.raises(ProtocolError):validate_bound_panel(cfg,'fixed','ppo')


def test_ppo_gate_does_not_admit_lightnav_learned_panel():
    cfg=bound();cfg['learned_progress_gates']={'ppo':dict(status='passed',feedback_source='learned_g_z_conditioned')}
    assert len(validate_bound_panel(cfg,'vlm_learned','ppo'))==10
    with pytest.raises(ProtocolError):validate_bound_panel(cfg,'vlm_learned','lightnav')


def test_changed_seed_or_budget_rejected():
    cfg=bound();cfg['evaluation_episodes'][9]['seed']=0
    with pytest.raises(ProtocolError):validate_bound_panel(cfg,'fixed','ppo')
    cfg=bound();cfg['max_vlm_calls']=41
    with pytest.raises(ProtocolError):validate_bound_panel(cfg,'vlm_rule','ppo')
