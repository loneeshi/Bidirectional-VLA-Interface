import importlib.util
from pathlib import Path
import pytest
from bvi.protocol import ProtocolError

spec=importlib.util.spec_from_file_location('c1',Path(__file__).resolve().parents[1]/'scripts/run_c1_panel.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_c1_commands_preserve_uid_seed_limits_and_disable_api():
    cfg=dict(status='bound_ready_for_preflight',conditions=['fixed'],navigation_panels=['ppo'],
             task='tidy_house',evaluation_split='val',max_env_steps=7000,max_vlm_calls=40,
             api_usd_per_episode=0.05,evaluation_episodes=[dict(seed=i,plan_uid=f'fixture-{i}') for i in range(10)])
    jobs=module.commands(cfg,'python','models','runs/test')
    assert len(jobs)==10
    for i,job in enumerate(jobs):
        argv=job['argv']
        assert '--dry-run' in argv and '--organizer' not in argv
        assert argv[argv.index('--policy-type')+1]=='rl_per_obj'
        assert '--record-demonstrations' in argv
        assert argv[argv.index('--expected-plan-uid')+1]==f'fixture-{i}'
        assert argv[argv.index('--seed')+1]==str(i)
        assert argv[argv.index('--max-env-steps')+1]=='7000'
    cfg['evaluation_episodes'][0]['plan_uid']=None
    with pytest.raises(ProtocolError):module.commands(cfg,'python','models','runs/test')
