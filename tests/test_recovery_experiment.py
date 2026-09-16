import copy
import json

import pytest

from bvi.recovery_experiment import extract_prefix, visible_history, progress_event
from bvi.progress_monitor import ProgressMonitor
from bvi.task_memory import object_memory


def test_hidden_progress_cannot_change_prompt_memory():
    a = [
        {
            "family": "reach",
            "target": "cream_cheese_1",
            "progress": 0.1,
            "reason": "learned_drop",
            "source": "learned",
            "shared_prefix": True,
        }
    ]
    b = copy.deepcopy(a)
    b[0].update(progress=0.99, reason="learned_threshold")
    assert visible_history(a, False) == visible_history(b, False)
    assert object_memory(visible_history(a, False)) == object_memory(
        visible_history(b, False)
    )
    assert a[0]["progress"] == 0.1
    assert visible_history(a, True) != visible_history(b, True)


def test_prefix_excludes_grasp_request_and_counts_trigger_only_prediction():
    rows = [
        {"event": "vlm_request"},
        {"event": "learned_progress"},
        {"event": "action", "action": [0] * 7, "diagnostic_only": {}},
        {"event": "learned_progress"},
        {"event": "invocation_finished", "family": "reach"},
        {"event": "vlm_request"},
        {
            "event": "vlm_raw_response",
            "raw_text": json.dumps(
                {"tool_family": "grasp", "target": "cream_cheese_1"}
            ),
        },
    ]
    result = extract_prefix(rows)
    assert result["calls"] == 1
    assert result["predictions"] == 2
    assert len(result["actions"]) == 1
    with pytest.raises(ValueError):
        extract_prefix(rows[:4])


def test_feedback_off_cannot_trigger_or_mutate_monitor():
    monitor = ProgressMonitor("reach", 1)
    before = copy.deepcopy(vars(monitor))
    for value in (0.99, 0.99, 0.1, 0.0):
        assert progress_event(monitor, value, False) is None
    assert vars(monitor) == before
    assert progress_event(monitor, 0.99, True) is None
    assert progress_event(monitor, 0.99, True) == "learned_threshold"
