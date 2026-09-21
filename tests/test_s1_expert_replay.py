import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "replay_s1_expert.py"
SPEC = importlib.util.spec_from_file_location("replay_s1_expert", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quaternion_angle_is_sign_invariant_and_reports_rotation():
    assert MODULE.quaternion_angle([0, 0, 0, 1], [0, 0, 0, -1]) == pytest.approx(0)
    half = math.sqrt(0.5)
    assert MODULE.quaternion_angle([0, 0, 0, 1], [0, 0, half, half]) == pytest.approx(
        math.pi / 2
    )


def test_level2_paths_do_not_mislabel_reconstructed_snapshot_as_original():
    assert MODULE.PATH_CONFIG["cpu_reset"]["uses_snapshot_restore"] is False
    assert MODULE.PATH_CONFIG["gpu_snapshot_restore"]["uses_snapshot_restore"] is True
    assert "original" in (MODULE.__doc__ or "").lower()
    assert MODULE.THRESHOLDS["qpos"] > 0
    assert MODULE.THRESHOLDS["qvel"] > MODULE.THRESHOLDS["qpos"]
