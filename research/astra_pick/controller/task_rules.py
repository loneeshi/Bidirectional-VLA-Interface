"""Pure task-budget rules and evaluator-only annotations; no environment/API.

Never feed evaluate_state output to a controller. Official records stay intact.
"""
import math
import numpy as np

RULE_VERSION = 'fetch-task-rules-v1'
REST_CAP = 120
REST_SETTLE_STEPS = 20


def rest_budget(duration_s, dt, stage_cap, remaining_steps, *, settle_steps=REST_SETTLE_STEPS):
    """Reserve an endpoint command AND settling; never extend either hard cap.

    Commands are sampled at 0, dt, ...; ceil(duration/dt)+1 includes the endpoint.
    """
    if not math.isfinite(duration_s) or duration_s < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid rest duration/control period')
    if any(type(x) is not int or x < 0 for x in (stage_cap, remaining_steps, settle_steps)):
        raise ValueError('Invalid rest budget')
    reference_steps = math.ceil(duration_s / dt) + 1
    required = reference_steps + settle_steps
    return dict(version=RULE_VERSION, duration_s=float(duration_s), dt=float(dt),
                reference_steps=reference_steps, settle_steps=settle_steps,
                required_steps=required, stage_cap=stage_cap, remaining_steps=remaining_steps,
                feasible=required <= min(stage_cap, remaining_steps))


def _flag(info, name):
    value = info.get(name)
    if value is None:
        return None
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or a.item() not in (0, 1):
        raise ValueError('Invalid evaluation flag: ' + name)
    return bool(a.item())


def evaluate_state(qvel, official_info):
    """Additional score, not a replacement official score or grasp observation.

    Missing flags yield unknown; finger aperture alone never implies a grasp.
    Same body/base slices and thresholds as native Fetch, with absolute velocity.
    """
    v = np.asarray(qvel, float)
    if v.shape != (15,) or not np.isfinite(v).all():
        raise ValueError('Expected finite Fetch 15-joint velocity')
    absolute_static = bool(np.all(np.abs(v[3:-2]) <= .2) and np.all(np.abs(v[:3]) <= .05))
    grasp = _flag(official_info, 'is_grasped')
    flags = [grasp] + [_flag(official_info, k) for k in
                      ('ee_rest', 'robot_rest', 'cumulative_force_within_limit')]
    complete = all(x is not None for x in flags)
    return dict(version=RULE_VERSION, evaluation_only=True,
                official_success=_flag(official_info, 'success'),
                official_is_static=_flag(official_info, 'is_static'),
                absolute_velocity_static=absolute_static,
                body_abs_speed_max=float(np.max(np.abs(v[3:-2]))),
                base_abs_speed_max=float(np.max(np.abs(v[:3]))),
                grasp_established_eval=grasp,
                absolute_static_pick_eval=(bool(all(flags) and absolute_static) if complete else None))


def close_command_result(completed):
    """Command acknowledgement only; never manufactures a grasp success."""
    if type(completed) is not bool:
        raise ValueError('Expected command completion bool')
    return dict(closure_command_completed=completed,
                grasp_success_claim=False,
                grasp_evidence='separate evaluation_only.grasp_established_eval')


def close_progress(result, closed_steps, wait_cap=5):
    """An accepted but interrupted one-step command is not completed."""
    if type(closed_steps) is not int or closed_steps < 0 or type(wait_cap) is not int or wait_cap < 1:
        raise ValueError('Invalid close progress')
    accepted=result.get('accepted') is True
    completed=accepted and closed_steps > 0 and (result.get('stable') is True or closed_steps >= wait_cap)
    return dict(close_command_result(completed),closure_command_accepted=accepted,
                closed_action_steps=closed_steps)
