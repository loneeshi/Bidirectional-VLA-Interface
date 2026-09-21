import json
from pathlib import Path

import pytest

from bvi.evaluation import (
    EXPECTED_TELEPORT_OBJECTS,
    PANEL_SCHEMA,
    PROFILES,
    acquire_panel_lock,
    attempt_destination,
    build_bindings,
    classify,
    panel_summary,
    render_markdown,
    runner_command,
    select_plans,
    summarize_panels,
    validate_resume_state,
)
from bvi.mshab_runner import oracle_step_budget


def manifest():
    return {
        "episodes": [
            {
                "seed": seed,
                "plan_uid": f"plan-{seed % 16}",
                "binding": {"subtasks": [{}] * 20},
            }
            for seed in range(20)
        ]
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_profiles_are_frozen_eval_only():
    assert set(PROFILES) == {"fixed", "gpt", "teleport"}
    assert all(profile.training_updates == 0 for profile in PROFILES.values())
    assert PROFILES["fixed"].navigation_policy == PROFILES["gpt"].navigation_policy == "official_ppo"
    assert PROFILES["fixed"].manipulation_policy == PROFILES["gpt"].manipulation_policy
    assert PROFILES["gpt"].goal_tools and PROFILES["gpt"].max_calls == 40
    assert PROFILES["teleport"].navigation_policy == "standardized_teleport"


def test_plan_roster_is_exactly_sixteen_unique_rows():
    rows = select_plans(manifest())
    assert [row["seed"] for row in rows] == list(range(16))
    broken = manifest()
    broken["episodes"][15]["plan_uid"] = "plan-0"
    with pytest.raises(ValueError, match="16 unique"):
        select_plans(broken)


def test_gpt_and_teleport_require_fixed_state_hashes(tmp_path):
    fixed = {
        "schema": PANEL_SCHEMA,
        "setting": "fixed",
        "episodes": [
            {
                "seed": seed,
                "plan_uid": f"plan-{seed}",
                "status": "completed",
                "initial_state_sha256": f"{seed:064x}",
            }
            for seed in range(16)
        ],
    }
    path = tmp_path / "fixed.json"
    write_json(path, fixed)
    for setting in ("gpt", "teleport"):
        rows = build_bindings(setting, manifest(), path)
        assert rows[7]["initial_state_sha256"] == f"{7:064x}"
    with pytest.raises(ValueError, match="reference-panel"):
        build_bindings("gpt", manifest(), None)


def test_runner_command_has_no_silent_policy_override():
    row = {"seed": 2, "plan_uid": "plan-2", "initial_state_sha256": "a" * 64}
    fixed = runner_command(PROFILES["fixed"], row, Path("out"), Path("ckpt"), None, None, True)
    assert fixed[:3] == [__import__("sys").executable, "-m", "bvi.mshab_runner"]
    assert fixed[fixed.index("--setting") + 1] == "fixed"
    assert "--execute" not in fixed
    with pytest.raises(ValueError, match="bridge-dir"):
        runner_command(PROFILES["gpt"], row, Path("out"), Path("ckpt"), None, None, True)


def test_fixed_and_teleport_keep_native_skill_horizons():
    assert oracle_step_budget(500, 7000) == 500
    assert oracle_step_budget(200, 6800) == 200
    assert oracle_step_budget(500, 12) == 12


def test_resume_summary_counts_all_attempt_api_claims():
    rows = [
        {
            "status": "completed",
            "result": {"task_success": False, "completed_objects": 2},
            "attempts": [
                {"result": {"api_requests": 3}},
                {"result": {"api_requests": 5}},
            ],
        }
    ] + [{"status": "not_run", "attempts": []} for _ in range(15)]
    summary = panel_summary(rows)
    assert summary["planned"] == 16
    assert summary["completed_objects"] == 2
    assert summary["api_requests"] == 8


def test_resume_binding_and_attempt_append_are_immutable(tmp_path):
    bindings = [{"seed": seed, "plan_uid": f"plan-{seed}"} for seed in range(16)]
    episodes = [{**row, "status": "not_run", "attempts": []} for row in bindings]
    state = {"schema": PANEL_SCHEMA, "setting": "fixed", "episodes": episodes}
    validate_resume_state(state, "fixed", bindings)
    episodes[0]["attempts"] = [{"attempt": 1}, {"attempt": 2}]
    assert attempt_destination(tmp_path, episodes[0]).name == "attempt-003"
    altered = [{**row} for row in bindings]
    altered[0]["plan_uid"] = "different-plan"
    with pytest.raises(ValueError, match="frozen plan/state bindings"):
        validate_resume_state(state, "fixed", altered)


def test_panel_lock_is_exclusive(tmp_path):
    lock = acquire_panel_lock(tmp_path)
    assert lock.is_file()
    with pytest.raises(FileExistsError):
        acquire_panel_lock(tmp_path)


def test_classification_separates_model_and_infrastructure_failures():
    base = {"evaluation_eligible": True, "benchmark_episode": True, "task_success": False}
    assert classify(1, {**base, "reason": "error:ModelResponseError"})[0] == "completed"
    assert classify(0, {**base, "reason": "adapter_error:RuntimeError"})[0] == "infrastructure_failure"
    assert classify(0, None)[1]["reason"] == "missing_summary"


def make_evidence(setting, completed_objects):
    rows = []
    remaining = completed_objects
    for seed in range(16):
        value = min(5, remaining)
        remaining -= value
        rows.append({
            "seed": seed,
            "plan_uid": f"plan-{seed}",
            "initial_state_sha256": f"{seed:064x}" if setting != "fixed" else None,
            "status": "completed",
            "result": {
                "task_success": False,
                "completed_objects": value,
                "planned_objects": 5,
                "reason": "benchmark_fail",
                "steps": 10,
                "api_requests": 1 if setting == "gpt" else 0,
            },
        })
    return rows


def test_public_summary_recomputes_14_13_21_and_contains_no_paths(tmp_path):
    paired = {
        "schema": "ppo-sac-paired16/1",
        "episodes": [
            *[{**row, "arm": "fixed"} for row in make_evidence("fixed", 14)],
            *[{**row, "arm": "gpt"} for row in make_evidence("gpt", 13)],
        ],
    }
    teleport = {
        "schema": "standardized-teleport16/1",
        "episodes": make_evidence("teleport", EXPECTED_TELEPORT_OBJECTS),
    }
    paired_path, teleport_path = tmp_path / "paired.json", tmp_path / "teleport.json"
    write_json(paired_path, paired)
    write_json(teleport_path, teleport)
    summary = summarize_panels(paired_path, teleport_path)
    assert [item["completed_objects"] for item in summary["settings"]] == [14, 13, 21]
    assert [item["mean_completed_objects"] for item in summary["settings"]] == [0.875, 0.8125, 1.3125]
    teleport_summary = summary["settings"][2]
    assert teleport_summary["completed_episodes"] == 16
    assert teleport_summary["api_requests_final_attempts"] == 0
    public = json.dumps(summary)
    assert "directory" not in public and "pid" not in public and "authorization" not in public
    table = render_markdown(summary)
    assert "21/80" in table and "1.313" in table


def test_public_summary_blocks_wrong_teleport_total(tmp_path):
    paired = {
        "schema": "ppo-sac-paired16/1",
        "episodes": [
            *[{**row, "arm": "fixed"} for row in make_evidence("fixed", 14)],
            *[{**row, "arm": "gpt"} for row in make_evidence("gpt", 13)],
        ],
    }
    teleport = {"schema": "standardized-teleport16/1", "episodes": make_evidence("teleport", 12)}
    paired_path, teleport_path = tmp_path / "paired.json", tmp_path / "teleport.json"
    write_json(paired_path, paired)
    write_json(teleport_path, teleport)
    with pytest.raises(ValueError, match="expected 21"):
        summarize_panels(paired_path, teleport_path)


def test_public_summary_blocks_seed_to_plan_mismatch(tmp_path):
    fixed = make_evidence("fixed", 14)
    gpt = make_evidence("gpt", 13)
    gpt[0]["plan_uid"], gpt[1]["plan_uid"] = gpt[1]["plan_uid"], gpt[0]["plan_uid"]
    paired = {
        "schema": "ppo-sac-paired16/1",
        "episodes": [
            *[{**row, "arm": "fixed"} for row in fixed],
            *[{**row, "arm": "gpt"} for row in gpt],
        ],
    }
    teleport = {"schema": "standardized-teleport16/1", "episodes": make_evidence("teleport", 21)}
    paired_path, teleport_path = tmp_path / "paired.json", tmp_path / "teleport.json"
    write_json(paired_path, paired)
    write_json(teleport_path, teleport)
    with pytest.raises(ValueError, match="same 16-plan roster"):
        summarize_panels(paired_path, teleport_path)


def test_public_summary_blocks_initial_state_hash_mismatch(tmp_path):
    paired = {
        "schema": "ppo-sac-paired16/1",
        "episodes": [
            *[{**row, "arm": "fixed"} for row in make_evidence("fixed", 14)],
            *[{**row, "arm": "gpt"} for row in make_evidence("gpt", 13)],
        ],
    }
    teleport_rows = make_evidence("teleport", 21)
    teleport_rows[0]["initial_state_sha256"] = "f" * 64
    teleport = {"schema": "standardized-teleport16/1", "episodes": teleport_rows}
    paired_path, teleport_path = tmp_path / "paired.json", tmp_path / "teleport.json"
    write_json(paired_path, paired)
    write_json(teleport_path, teleport)
    with pytest.raises(ValueError, match="initial-state hashes"):
        summarize_panels(paired_path, teleport_path)
