import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('runner',Path(__file__).parents[1]/'scripts/run_continuation_c012.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_c2_only_receives_prior(tmp_path):
    row={'seed':1,'plan_uid':'uid'}
    base=dict(output=tmp_path/'out',checkpoints=tmp_path/'ckpt',bridge=tmp_path/'bridge',
              authorization='auth',manifest=tmp_path/'manifest.json',prior=tmp_path/'prior.json')
    c1=runner.command(row,'C1',**base)
    c2=runner.command(row,'C2',**base)
    assert '--spawn-prior' not in c1 and 'object_trajectory_v1' in c1
    assert '--spawn-prior' in c2 and 'object_trajectory_prior_v1' in c2


def test_all_conditions_have_identical_physical_budgets(tmp_path):
    row={'seed':1,'plan_uid':'uid'}
    kw=dict(output=tmp_path/'out',checkpoints=tmp_path/'ckpt',bridge=tmp_path/'bridge',
            authorization='auth',manifest=tmp_path/'manifest.json',prior=tmp_path/'prior.json')
    commands=[runner.command(row,c,**kw) for c in ('C0','C1','C2')]
    for flag in ('--max-env-steps','--max-calls','--organizer-slice-steps'):
        assert len({cmd[cmd.index(flag)+1] for cmd in commands})==1
