import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "teleport16", Path(__file__).parents[1] / "scripts/run_official_teleport16.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selects_same_sixteen_unique_five_object_plans():
    episodes = []
    for seed, plan in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 13, 14, 7, 16, 3, 7, 19]):
        episodes.append({"seed": seed, "plan_uid": f"p-{plan}",
                         "binding": {"subtasks": [{}] * 20}})
    selected = module.select_plans({"episodes": episodes})
    assert len(selected) == 16
    assert [r["seed"] for r in selected] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 19]


def test_binds_one_exact_plan_and_parses_official_result(tmp_path):
    plans = {"dataset": "D", "plans": [
        {"subtasks": [{"uid": "p-1"}] + [{}] * 19},
        {"subtasks": [{"uid": "p-2"}] + [{}] * 19},
    ]}
    bound = module.bind_plan(plans, "p-2")
    assert bound["dataset"] == "D"
    assert bound["plans"][0]["subtasks"][0]["uid"] == "p-2"
    root = tmp_path / "official-logs" / "x"
    root.mkdir(parents=True)
    (root / "output.txt").write_text(
        "results\n{'success_once': tensor(0., device='cuda:0')}\n", encoding="utf-8")
    (root / "subtask_fail_counts.json").write_text(json.dumps({"7": 1}), encoding="utf-8")
    result = module.parse_result(tmp_path)
    assert result["task_success"] is False
    assert result["completed_objects"] == 1
    assert result["failure_subtask_index"] == 7
