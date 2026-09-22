import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('native_summary',Path(__file__).parents[1]/'scripts/summarize_native_subtasks.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_native_counts_do_not_count_repeated_success_or_shrink_denominator(tmp_path):
    path=tmp_path/'events.jsonl'
    path.write_text('\n'.join(json.dumps(dict(event='mshab_step',subtask_before=a,subtask_after=b))
                              for a,b in [(0,1),(1,1),(1,2),(2,3),(3,4),(4,4)]))
    result=m.summarize_events(path)
    assert result['completed_objects']==1
    assert result['subtasks']['navigate']==dict(planned=10,reached=3,completed=2)
    assert result['subtasks']['place']==dict(planned=5,reached=1,completed=1)
    path.write_text(json.dumps(dict(event='mshab_step',subtask_before=8,subtask_after=9)))
    with pytest.raises(ValueError):m.summarize_events(path)
