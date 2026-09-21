import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('paired',Path(__file__).parents[1]/'scripts/run_ppo_sac_paired16.py')
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_select_unique_plans_without_counting_repeats():
    rows=[dict(seed=i,plan_uid=f'uid-{i%16}',binding=dict(subtasks=[{}]*20)) for i in range(20)]
    assert [r['seed'] for r in m.select_plans(dict(episodes=rows))]==list(range(16))
    rows[15]['plan_uid']='uid-0'
    with pytest.raises(ValueError):m.select_plans(dict(episodes=rows))


def test_both_arms_use_same_physical_tools_and_gpt_requires_initial_state():
    row=dict(seed=5,plan_uid='uid')
    fixed=m.command(row,'fixed',Path('/f'),Path('/c'),Path('/b'),'auth')
    with pytest.raises(ValueError):m.command(row,'gpt',Path('/g'),Path('/c'),Path('/b'),'auth')
    gpt=m.command(row,'gpt',Path('/g'),Path('/c'),Path('/b'),'auth','hash')
    for flag in ['--navigation-policy','--manipulation-policy','--max-env-steps','--max-wall-seconds']:
        assert fixed[fixed.index(flag)+1]==gpt[gpt.index(flag)+1]
    assert '--dry-run' in fixed and '--organizer' in gpt
    assert gpt[gpt.index('--expected-initial-state-sha256')+1]=='hash'


def test_chunk_limit_is_exposed_by_runner():
    source=Path(m.__file__).read_text()
    assert "--max-new-per-arm" in source
    assert "started_by_arm[row['arm']]" in source
