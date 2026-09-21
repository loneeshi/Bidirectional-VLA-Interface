"""Offline CI gate for the three-path official-BC T0a diagnostic.

The five fixed seeds are a deterministic regression panel, not a benchmark
success-rate estimate.  This check consumes the compact ``panel.json`` and
``comparison.json`` artifacts produced by the T0a scripts; it never launches a
simulator, allocates a GPU, or trains a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = (2024, 2025, 2026, 2027, 2028)
CHECKPOINT_REVISION = "91e96be85128df43728a7511355c3fa999bd2c94"
CHECKPOINT_SHA256 = "a3593b72dfc3c265fb8a294b6f3bb79c86472d4efada6437d50c66daee7cbd58"
PLAN_SHA256 = "135915353c8a56230fcbc68ada7f6a7d58d86119d8c8d1acbb15790d6da08c52"
OFFICIAL_SOURCE_REVISION = "e9ff3d23496d38e4431c8d913e147ffa007f7f72"
ACDIT_SOURCE_REVISION = "90ad00a926f34da04816ed9c3312aaf3bc845b7f"


class GateFailure(ValueError):
    """A checked diagnostic invariant was not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateFailure(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def load_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _load(root / "panel.json"), _load(root / "comparison.json")


def _episode_map(
    panel: dict[str, Any], *, label: str, expected_paths: tuple[str, ...],
    expected_environment: str, expected_source_revision: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    _require(panel.get("status") == "completed", f"{label}: panel is not completed")
    _require(panel.get("training_updates") == 0, f"{label}: training_updates must be zero")
    _require(panel.get("api_calls") == 0, f"{label}: api_calls must be zero")
    episodes = panel.get("episodes")
    _require(isinstance(episodes, list), f"{label}: episodes must be a list")
    expected = {(seed, path) for seed in SEEDS for path in expected_paths}
    found: dict[tuple[int, str], dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        _require(isinstance(episode, dict), f"{label}: episode {index} is not an object")
        key = (episode.get("seed"), episode.get("path"))
        _require(key in expected, f"{label}: unexpected seed/path {key}")
        _require(key not in found, f"{label}: duplicate seed/path {key}")
        # Early official artifacts predate the explicit environment field.  Its
        # absence means the official source; no other missing value is accepted.
        environment = episode.get("environment", "official")
        _require(environment == expected_environment,
                 f"{label}: {key} environment {environment!r} != {expected_environment!r}")
        _require(episode.get("status") == "completed", f"{label}: {key} is not completed")
        _require(episode.get("training_updates") == 0,
                 f"{label}: {key} training_updates must be zero")
        _require(episode.get("api_calls") == 0, f"{label}: {key} api_calls must be zero")
        _require(episode.get("checkpoint_revision") == CHECKPOINT_REVISION,
                 f"{label}: {key} checkpoint revision drift")
        _require(episode.get("checkpoint_sha256") == CHECKPOINT_SHA256,
                 f"{label}: {key} checkpoint bytes drift")
        _require(episode.get("plan_sha256") == PLAN_SHA256,
                 f"{label}: {key} task plan bytes drift")
        _require(episode.get("source_revision") == expected_source_revision,
                 f"{label}: {key} source revision drift")
        _require(episode.get("strict_load") is True, f"{label}: {key} did not strict-load")
        _require(episode.get("state_shape") == [1, 42], f"{label}: {key} BC state shape drift")
        _require("not continuous-task paper success-once" in episode.get("protocol", ""),
                 f"{label}: {key} lost protocol limitation")
        _require("not full VLA harness" in episode.get("scope", ""),
                 f"{label}: {key} lost scope limitation")
        _require(isinstance(episode.get("native_success"), bool),
                 f"{label}: {key} native_success is not boolean")
        _require(isinstance(episode.get("steps"), int) and episode["steps"] > 0,
                 f"{label}: {key} has invalid step count")
        for field in ("initial_state_sha256", "initial_policy_obs_sha256"):
            value = episode.get(field)
            _require(isinstance(value, str) and len(value) == 64,
                     f"{label}: {key} has invalid {field}")
        found[key] = episode
    _require(set(found) == expected,
             f"{label}: seed/path roster differs; missing={sorted(expected - set(found))}")
    return found


def _comparison_map(comparison: dict[str, Any], *, label: str) -> dict[int, dict[str, Any]]:
    _require(comparison.get("panel_status") == "completed",
             f"{label}: comparison panel_status is not completed")
    _require("not full VLA harness" in comparison.get("limitation", ""),
             f"{label}: comparison lost scope limitation")
    rows = comparison.get("pairs")
    _require(isinstance(rows, list), f"{label}: pairs must be a list")
    found: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{label}: pair {index} is not an object")
        seed = row.get("seed")
        _require(seed in SEEDS, f"{label}: unexpected seed {seed}")
        _require(seed not in found, f"{label}: duplicate seed {seed}")
        _require(row.get("status") == "paired", f"{label}: seed {seed} is not paired")
        _require(row.get("initial_exact") is True,
                 f"{label}: seed {seed} initial state/observation is not bitwise exact")
        _require(row.get("equal_steps") is True,
                 f"{label}: seed {seed} trajectory lengths differ")
        _require(row.get("all_policy_inputs_exact") is True,
                 f"{label}: seed {seed} policy inputs are not bitwise exact")
        _require(row.get("raw_action_max_abs_difference") == 0,
                 f"{label}: seed {seed} raw action delta is not zero")
        _require(row.get("all_info_and_qpos_exact") is True,
                 f"{label}: seed {seed} qpos/info trajectory is not bitwise exact")
        _require(row.get("official_success") == row.get("project_success"),
                 f"{label}: seed {seed} paired success outcome differs")
        _require(row.get("official_steps") == row.get("project_steps"),
                 f"{label}: seed {seed} paired step count differs")
        found[seed] = row
    _require(set(found) == set(SEEDS),
             f"{label}: comparison seed roster differs; missing={sorted(set(SEEDS) - set(found))}")
    return found


def validate_artifacts(
    official_panel: dict[str, Any], official_comparison: dict[str, Any],
    acdit_panel: dict[str, Any], acdit_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Validate all three paths and return a CI-serializable diagnostic report."""
    official = _episode_map(
        official_panel, label="official panel", expected_paths=("official", "project"),
        expected_environment="official", expected_source_revision=OFFICIAL_SOURCE_REVISION,
    )
    acdit = _episode_map(
        acdit_panel, label="AC-DiT panel", expected_paths=("project",),
        expected_environment="acdit", expected_source_revision=ACDIT_SOURCE_REVISION,
    )
    official_pairs = _comparison_map(official_comparison, label="official A/B comparison")
    acdit_pairs = _comparison_map(acdit_comparison, label="official A / AC-DiT C comparison")

    successes: list[int] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        path_a = official[(seed, "official")]
        path_b = official[(seed, "project")]
        path_c = acdit[(seed, "project")]
        for field in (
            "initial_state_sha256", "initial_policy_obs_sha256", "checkpoint_sha256",
            "plan_sha256", "native_success", "steps",
        ):
            _require(path_a.get(field) == path_b.get(field) == path_c.get(field),
                     f"three-path seed {seed}: {field} differs")
        for row, name in ((official_pairs[seed], "A/B"), (acdit_pairs[seed], "A/C")):
            _require(row["official_success"] == path_a["native_success"],
                     f"{name} seed {seed}: comparison/panel success differs")
            _require(row["official_steps"] == path_a["steps"],
                     f"{name} seed {seed}: comparison/panel steps differ")
        if path_a["native_success"]:
            successes.append(seed)
        fingerprints[str(seed)] = {
            "initial_state_sha256": path_a["initial_state_sha256"],
            "initial_policy_obs_sha256": path_a["initial_policy_obs_sha256"],
            "native_success": path_a["native_success"],
            "steps": path_a["steps"],
        }

    _require(successes, "known-good official BC produced no native success on the fixed panel")
    return {
        "status": "passed",
        "gate": "official_bc_t0a_three_path_diagnostic",
        "diagnostic_only": True,
        "benchmark_success_rate_estimate": False,
        "distinct_fixed_initial_conditions": len(SEEDS),
        "paths": {
            "A": "official wrappers on official environment",
            "B": "project-style BC assembly on official environment",
            "C": "project-style BC assembly on AC-DiT environment fork",
        },
        "invariants": {
            "initial_state_and_policy_observation": "bitwise exact",
            "policy_inputs_qpos_and_info": "bitwise exact across compared trajectories",
            "raw_action_max_abs_difference": 0,
            "three_path_outcomes_and_steps": "exact",
            "checkpoint_and_task_plan": "pinned SHA256",
        },
        "known_good_nonzero_diagnostic": {
            "passed": True,
            "successful_seeds": successes,
            "successful_seed_count": len(successes),
            "criterion": "at least one native success; not a statistical threshold",
        },
        "fixed_panel_fingerprint": fingerprints,
        "limitations": [
            "Five fixed initial conditions are a regression panel, not a benchmark sample.",
            "This clears only the measured depth/state BC assembly and environment-source paths.",
            "It does not clear RGB, point clouds, VLA normalization, precision, action chunks, or the full harness.",
        ],
    }


def run(official_root: Path, acdit_root: Path) -> dict[str, Any]:
    official_panel, official_comparison = load_artifacts(official_root)
    acdit_panel, acdit_comparison = load_artifacts(acdit_root)
    return validate_artifacts(official_panel, official_comparison, acdit_panel, acdit_comparison)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official", type=Path,
        default=repository / "docs/results/bc-t0a-2026-09-18-run01",
        help="A/B result directory containing panel.json and comparison.json",
    )
    parser.add_argument(
        "--acdit", type=Path,
        default=repository / "docs/results/bc-t0a-2026-09-18-acdit-fork-run01",
        help="C result directory containing panel.json and comparison.json",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    try:
        result = run(args.official, args.acdit)
        exit_code = 0
    except GateFailure as exc:
        result = {
            "status": "failed",
            "gate": "official_bc_t0a_three_path_diagnostic",
            "diagnostic_only": True,
            "benchmark_success_rate_estimate": False,
            "error": str(exc),
        }
        exit_code = 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
