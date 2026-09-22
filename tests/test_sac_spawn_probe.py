import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'scripts/run_sac_spawn_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_probe_panel_is_balanced_and_unique():
    rows = probe.cases()
    assert len(rows) == 24
    assert len({tuple(r.values()) for r in rows}) == 24
    assert {r['seed'] for r in rows} == {100, 101, 102}
    assert sum(r['skill'] == 'pick' for r in rows) == 12


def test_atomic_record(tmp_path):
    import json
    target = tmp_path / 'manifest.json'
    probe.save(target, {'status': 'not_run', 'cost': None})
    assert json.loads(target.read_text())['cost'] is None
    assert not target.with_suffix('.tmp').exists()
