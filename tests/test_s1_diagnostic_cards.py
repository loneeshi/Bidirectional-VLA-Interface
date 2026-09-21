import numpy as np

from bvi.s1_diagnostic_cards import (
    d1_adjudication,
    d2_adjudication,
    fixed_action,
    normalized_gripper_target,
    perturbation_targets,
)


def test_fixed_actions_are_distinct_and_bounded():
    zero = fixed_action("zero", [0.04, 0.04])
    hold = fixed_action("hold", [0.04, 0.04])
    close = fixed_action("close", [0.04, 0.04])
    assert zero.shape == hold.shape == close.shape == (13,)
    assert np.count_nonzero(zero) == 0
    assert hold[7] == normalized_gripper_target(0.04)
    assert close[7] == -1
    assert np.all(np.abs(hold) <= 1)


def test_perturbation_targets_enforce_frozen_band():
    assert np.allclose(perturbation_targets(0.025), [0.055, 0.075])
    assert perturbation_targets(0.04) == []


def test_card_adjudication_is_conservative():
    assert d1_adjudication([23], [1000]) == "threshold_or_wrapper_layer"
    assert d1_adjudication([], [4500, 4300]) == "knife_edge_threshold_policy_precision_insufficient"
    assert d1_adjudication([], [1000, 1200]) == "ia_collision_requires_geometry_attribution"
    assert d1_adjudication([], [1000, 4500]) == "knife_edge_threshold_policy_precision_insufficient"
    assert d2_adjudication(3, 4) == "constant_close_not_distinguishable"
    assert d2_adjudication(0, 4, 2) == "weak_near_start_grasp_evidence"
    assert d2_adjudication(1, 4, 0) == "narrow_start_coupled_capability"
