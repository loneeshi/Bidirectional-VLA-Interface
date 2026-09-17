import pytest
from bvi.s1_capability_gate import SEEDS, validate_s1_capability


def report():
    return dict(schema='bvi.s1-native-capability/1',stage='S1',arm='faithful_native24',
        task='set_table/pick/013_apple',scene_split='val',action_budget=200,
        native_success_threshold_metres=0.05,checkpoint_sha256='a'*64,state_dim=24,
        privileged_policy_inputs=False,state_source='env_native_agent',
        base_camera='fetch_head',wrist_camera='fetch_hand',
        success_source='native_environment_predicate',checkpoint_selection='heldout_loss',
        episodes=[dict(seed=s,status='completed',success=i<3,executed_actions=100,
                       evidence_sha256='b'*64) for i,s in enumerate(SEEDS)])


def test_only_new_s2_admission_not_budget_or_old_job_authorization():
    result=validate_s1_capability(report(),'a'*64)
    assert result['successes']==3 and result['scope']=='new_S2_only'
    assert not result['budget_increase'] and not result['historical_pipeline_resume']


@pytest.mark.parametrize('field,value',[
    ('arm','privileged_diagnostic'),('state_dim',30),('privileged_policy_inputs',True),
    ('checkpoint_sha256','c'*64),('action_budget',520),('native_success_threshold_metres',0.1),
    ('checkpoint_selection','test_success'),('scene_split','train'),
])
def test_reject_wrong_arm_checkpoint_or_protocol(field,value):
    r=report();r[field]=value
    with pytest.raises(ValueError):validate_s1_capability(r,'a'*64)


def test_reject_partial_panel_failure_and_duplicate_seed():
    r=report();r['episodes'].pop()
    with pytest.raises(ValueError):validate_s1_capability(r,'a'*64)
    r=report();r['episodes'][0]['success']=False
    with pytest.raises(ValueError):validate_s1_capability(r,'a'*64)
    r=report();r['episodes'][0]['status']='infrastructure_failed'
    with pytest.raises(ValueError):validate_s1_capability(r,'a'*64)
    r=report();r['episodes'][1]['seed']=2024
    with pytest.raises(ValueError):validate_s1_capability(r,'a'*64)
