"""Reconstruct full controller dynamics by deterministic prefix replay, then audit."""

import hashlib
import json
import pathlib

import numpy as np

from bvi.recovery_experiment import canonical_hash, extract_prefix
from eval_libero_baseline import element


def numeric_state(value, depth=0):
    if depth > 4:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [numeric_state(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {
            str(k): numeric_state(v, depth + 1)
            for k, v in value.items()
            if k not in ("sim", "model")
        }
    if hasattr(value, "__dict__") and depth < 3:
        return numeric_state(vars(value), depth + 1)
    return type(value).__name__


def reconstruct(env, obs, prefix_path, displacement, out):
    path = pathlib.Path(prefix_path)
    prefix = extract_prefix(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    )
    frames = []
    for action in prefix["actions"]:
        frames.append(element(obs, "prefix replay")["observation/image"])
        obs, _, done, _ = env.step(action)
        if done:
            raise ValueError("Task completed before perturbation")
    base = env.env
    target = base.objects_dict["cream_cheese_1"]
    body = base.obj_body_id["cream_cheese_1"]
    sim = base.sim
    original = np.asarray(sim.data.body_xpos[body]).copy()
    expected = prefix["last_action_diagnostic"]["object_positions"]["cream_cheese_1"]
    if not np.allclose(original, expected, atol=1e-5, rtol=0):
        raise ValueError("Prefix replay target position differs from source trace")
    if np.linalg.norm(obs["robot0_eef_pos"] - original) > 0.15:
        raise ValueError("Not a pre-grasp approach state")
    if np.sum(np.abs(obs["robot0_gripper_qpos"])) < 0.05:
        raise ValueError("Gripper is not open at intervention")
    target_geoms = {sim.model.geom_name2id(g) for g in target.contact_geoms}

    def contacts():
        return [
            (
                sim.model.geom_id2name(c.geom2 if c.geom1 in target_geoms else c.geom1),
                float(c.dist),
            )
            for c in sim.data.contact[: sim.data.ncon]
            if c.geom1 in target_geoms or c.geom2 in target_geoms
        ]

    def validate_contact(rows):
        if not any(name and "table" in name for name, _ in rows):
            raise ValueError("Target has no table support contact")
        if any(not name or "table" not in name for name, _ in rows):
            raise ValueError("Target contacts another object or robot")
        if any(distance < -0.003 for _, distance in rows):
            raise ValueError("Target penetration exceeds tolerance")

    before_contacts = contacts()
    validate_contact(before_contacts)
    pre_state = np.asarray(env.get_sim_state()).copy()
    qpos = sim.data.get_joint_qpos(target.joints[0]).copy()
    qpos[:3] += np.asarray(displacement)
    sim.data.set_joint_qpos(target.joints[0], qpos)
    sim.forward()
    validate_contact(contacts())
    # Refresh sensors without stepping physics or resetting controller integrators.
    base._update_observables(force=True)
    obs = base._get_observations()
    state = {
        "simulator": np.asarray(env.get_sim_state()).tolist(),
        "qacc_warmstart": sim.data.qacc_warmstart.tolist(),
        "ctrl": sim.data.ctrl.tolist(),
        "controllers": [numeric_state(vars(robot.controller)) for robot in base.robots],
        "observations": {
            k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
            for k, v in obs.items()
        },
        "history": prefix["history"],
        "calls": prefix["calls"],
        "steps": len(prefix["actions"]),
        "prediction_count": prefix["predictions"],
        "action_queue": [],
        "monitor": "new_invocation",
    }
    digest = canonical_hash(state)
    output = pathlib.Path(out)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "simulator-snapshot.npz",
        before=pre_state,
        after=np.asarray(env.get_sim_state()),
        qacc_warmstart=sim.data.qacc_warmstart.copy(),
        ctrl=sim.data.ctrl.copy(),
    )
    report = {
        "state_sha256": digest,
        "state": state,
        "prefix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "restoration": "fresh_seeded_environment_and_exact_action_prefix_replay",
        "displacement_world_m": displacement,
        "before_position": original.tolist(),
        "after_position": sim.data.body_xpos[body].tolist(),
        "contacts_before": before_contacts,
        "contacts_after": contacts(),
        "prefix_api_calls_new": 0,
    }
    (output / "branch-state.json").write_text(json.dumps(report, indent=2))
    return obs, prefix, frames, digest
