"""Read-only, opt-in capture of one C2 geometry-precheck simulator state.

Call ``capture_reference`` before the frozen qpos edit, then ``export_bundle``
after its one settling action. This module never sets simulation state or runs a
policy. The result is observational evidence, not a reachability certificate:
no target-specific grasp frames are synthesized here. A missing or unsupported
physics collider is recorded as incomplete, never silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET


SCHEMA = "c2-cpu-reachability/1"
FROZEN_A1_SOURCES = {
    (8, "tidy_house-sequential-val-17-0", "call-1f6c9b2d", "object-03ff81c74789"),
    (4, "tidy_house-sequential-val-553-0", "call-b7c4d9e2f1a8", "object-7df706aa8cd2"),
    (10, "tidy_house-sequential-val-586-0", "call-13", "object-28e8c099370e"),
}
MAX_BODIES = 300
MAX_SHAPES = 5000
MAX_FACES = 10_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError("nonfinite_or_nonnumeric_state")
    return result


def _row(value, length):
    values = _array(value)
    if values.shape == (1, length):
        return values[0].astype(float).tolist()
    if values.shape == (length,):
        return values.astype(float).tolist()
    raise ValueError(f"expected_single_environment_vector_{length}: {values.shape}")


def _transform(pose):
    """Read a single ManiSkill Pose without setting or refreshing physics."""
    import numpy as np

    matrix = _array(pose.to_transformation_matrix())
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f"expected_single_environment_pose: {matrix.shape}")
    return matrix.astype(float)


def _velocity(body, getter):
    value = getattr(body, getter)()
    return _row(value, 3)


def _body_state(body, *, dynamic):
    state = {"pose_world": _transform(body.pose).tolist(), "dynamic": bool(dynamic),
             "body_type": getattr(body, "px_body_type", "articulation_link")}
    if dynamic:
        state["linear_velocity_world"] = _velocity(body, "get_linear_velocity")
        state["angular_velocity_world"] = _velocity(body, "get_angular_velocity")
    return state


def _other_bodies(uenv):
    robot = uenv.agent.robot
    bodies = [("actor", actor) for actor in uenv.scene.actors.values()]
    bodies += [("other_link", link)
               for art in uenv.scene.articulations.values()
               if art is not robot and art.name != robot.name
               for link in art.links]
    return [(f"{kind}:{index}:{body.name}", kind, body)
            for index, (kind, body) in enumerate(bodies)]


def _scene_registry(uenv):
    result = {}
    for key, kind, body in _other_bodies(uenv):
        dynamic = kind == "other_link" or getattr(body, "px_body_type", None) in (
            "dynamic", "kinematic")
        result[key] = _body_state(body, dynamic=dynamic)
    return result


def _moving_ids(states, *, threshold=1e-5):
    import numpy as np

    return sorted(key for key, state in states.items() if state.get("dynamic", False) and
                  (np.linalg.norm(state["linear_velocity_world"]) > threshold or
                   np.linalg.norm(state["angular_velocity_world"]) > threshold))


def _state_hash(uenv):
    from bvi.mshab_adapter import jsonable

    encoded = json.dumps(jsonable(uenv.get_state_dict()), sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_reference(uenv):
    """Capture handoff coordinates before any candidate qpos write."""
    robot = uenv.agent.robot
    return {
        "qpos": _row(robot.get_qpos(), robot.max_dof),
        "qvel": _row(robot.get_qvel(), robot.max_dof),
        "root_world": _transform(robot.root.pose).tolist(),
        "link_world": {link.name: _transform(link.pose).tolist()
                       for link in robot.links},
        "link_body_states": {link.name: _body_state(link, dynamic=True)
                             for link in robot.links},
        "scene_bodies": _scene_registry(uenv),
        "other_articulations": _other_articulations(uenv),
        "simulator_state_sha256": _state_hash(uenv),
    }


def _component_and_shapes(body):
    import sapien.physx as physx

    if len(body._objs) != 1:
        raise ValueError("body_wrapper_not_single_environment")
    obj = body._objs[0]
    components = ([obj] if isinstance(obj, physx.PhysxRigidBaseComponent)
                  else [comp for comp in obj.components
                        if isinstance(comp, physx.PhysxRigidBaseComponent)])
    if len(components) > 1:
        raise ValueError("multiple_rigid_components_in_body")
    component = components[0] if components else None
    return component, [] if component is None else list(component.get_collision_shapes())


def _mesh_of_body(body, body_id, *, world, coverage):
    import trimesh
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes

    component, shapes = _component_and_shapes(body)
    shape_info = []
    for index, shape in enumerate(shapes):
        kind = type(shape).__name__
        shape_info.append({"index": index, "type": kind,
                           "collision_groups": list(map(int, shape.get_collision_groups()))})
        if kind == "PhysxCollisionShapePlane":
            coverage["unexported_collision_shapes"].append(
                {"body": body_id, "shape": index, "type": kind,
                 "reason": "infinite_plane_has_no_finite_mesh"})
    coverage["shape_inventory"][body_id] = shape_info
    if not shapes:
        return None
    parts = get_component_meshes(component)
    expected = sum(row["type"] != "PhysxCollisionShapePlane" for row in shape_info)
    if len(parts) != expected:
        raise ValueError(f"shape_mesh_count_mismatch:{expected}:{len(parts)}")
    if not parts:
        return None
    mesh = trimesh.util.concatenate(parts)
    if world:
        mesh.apply_transform(_transform(body.pose))
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("empty_physics_collision_mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("nonfinite_collision_mesh")
    return mesh


def _save_mesh(mesh, directory, name, *, budget):
    faces = len(mesh.faces)
    if budget["faces"] + faces > MAX_FACES:
        raise ValueError("total_mesh_face_limit_exceeded")
    budget["faces"] += faces
    file = directory / f"{name}.obj"
    mesh.export(file, file_type="obj")
    return {"path": file.relative_to(directory.parent).as_posix(),
            "sha256": sha256(file), "vertices": len(mesh.vertices), "faces": faces}


def _urdf_info(agent, output):
    path = Path(agent.urdf_path).expanduser().resolve(strict=True)
    tree = ET.parse(path)
    links = {element.get("name") for element in tree.getroot().findall("link")}
    children = {joint.find("child").get("link")
                for joint in tree.getroot().findall("joint")}
    roots = links - children
    if len(roots) != 1:
        raise ValueError("urdf_root_ambiguous")
    destination = output / "fetch.urdf"
    shutil.copyfile(path, destination)
    return {"path": destination.name, "sha256": sha256(destination)}, next(iter(roots)), links


def _ignored_self_pairs(links, coverage):
    from c2_mesh_collisions import physx_pair_enabled

    by_link = {link.name: coverage["shape_inventory"].get(link.name, [])
               for link in links}
    ignored = []
    for index, left in enumerate(links):
        for right in links[index + 1:]:
            lhs, rhs = by_link[left.name], by_link[right.name]
            if lhs and rhs and not any(
                physx_pair_enabled(a["collision_groups"], b["collision_groups"])
                for a in lhs for b in rhs):
                ignored.append([left.name, right.name])
    return ignored


def _other_articulations(uenv):
    robot = uenv.agent.robot
    result = {}
    for index, art in enumerate(uenv.scene.articulations.values()):
        if art is robot or art.name == robot.name:
            continue
        key = f"articulation:{index}:{art.name}"
        result[key] = {"qpos": _row(art.get_qpos(), art.max_dof),
                       "qvel": _row(art.get_qvel(), art.max_dof),
                       "joint_order": [joint.name for joint in art.active_joints],
                       "link_world": {link.name: _transform(link.pose).tolist()
                                      for link in art.links}}
    return result


def _raw_camera_images(capture, output, issues):
    import numpy as np

    files = []
    for index, (camera, pixels) in enumerate(capture.images.items()):
        try:
            array = _array(pixels)
            if array.ndim != 4 or array.shape[0] != 1:
                raise ValueError(f"unexpected_camera_shape:{array.shape}")
            file = output / f"camera-{index:02d}.npy"
            np.save(file, array, allow_pickle=False)
            files.append({"camera": camera, "path": file.name,
                          "sha256": sha256(file), "shape": list(array.shape),
                          "dtype": str(array.dtype)})
        except Exception as exc:
            issues.append(f"camera_export:{camera}:{type(exc).__name__}:{exc}")
    return files


def export_bundle(uenv, capture, output, *, candidate, proposal, reference,
                  max_seconds=180, source_scope="frozen_A1"):
    """Export one post-settle bundle; raise only for a broken capture boundary.

    Per-body export failures stay in the manifest and make it ineligible for a
    CPU witness. ``output`` must not yet exist, so prior evidence is retained.
    """
    request = proposal.get("request", {})
    frozen_source = (proposal.get("seed"), proposal.get("plan_uid"),
                     request.get("call_id"), request.get("target_id"))
    if source_scope == "frozen_A1":
        if (candidate not in ("C0", "C1", "C2", "C3", "C4", "C5", "C6") or
                frozen_source not in FROZEN_A1_SOURCES):
            raise ValueError("not_frozen_A1_geometry")
    elif source_scope == "real_handoff_batch":
        if (candidate != "C0" or not proposal.get("plan_uid") or
                not proposal.get("handoff_snapshot_sha256") or
                not request.get("target_id")):
            raise ValueError("incomplete_real_handoff_geometry_identity")
    else:
        raise ValueError("unsupported_geometry_source_scope")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    exporter_source = output / "exporter-source.py"
    shutil.copyfile(Path(__file__), exporter_source)
    (output / "robot").mkdir()
    (output / "scene").mkdir()
    robot = uenv.agent.robot
    links = list(robot.links)
    others = _other_bodies(uenv)
    if len(links) + len(others) > MAX_BODIES:
        raise ValueError("body_limit_exceeded")
    coverage = {"robot_links_expected": len(links),
                "scene_bodies_expected": len(others),
                "robot_no_shape": [], "scene_no_shape": [],
                "unexported_collision_shapes": [], "mesh_errors": [],
                "shape_inventory": {}}
    issues = []
    budget = {"faces": 0, "shapes": 0}
    current_scene = _scene_registry(uenv)
    reference_scene = reference["scene_bodies"]
    moving_scene = _moving_ids(current_scene)
    scene_same = set(current_scene) == set(reference_scene) and all(
        current_scene[k]["pose_world"] == reference_scene[k]["pose_world"]
        for k in current_scene)
    scene_changed = sorted(key for key in set(current_scene) & set(reference_scene)
                           if current_scene[key]["pose_world"] !=
                           reference_scene[key]["pose_world"])
    data = {
        "schema": SCHEMA, "status": "incomplete_observation", "seed": proposal["seed"],
        "source_scope": source_scope,
        "handoff_snapshot_sha256": proposal.get("handoff_snapshot_sha256"),
        "candidate": candidate, "plan_uid": proposal["plan_uid"],
        "target_id": proposal["request"]["target_id"],
        "summary": {"path": "../summary.json", "sha256": None},
        "exporter_source": {"path": exporter_source.name,
                            "sha256": sha256(exporter_source)},
        "coverage": coverage, "reference_qpos": reference["qpos"],
        "reference_qvel": reference["qvel"],
        "reference_link_world": reference["link_world"],
        "reference_link_body_states": reference["link_body_states"],
        "reference_other_articulations": reference["other_articulations"],
        "reference_root_world": reference["root_world"],
        "reference_simulator_state_sha256": reference["simulator_state_sha256"],
        "candidate_qpos": _row(robot.get_qpos(), robot.max_dof),
        "candidate_qvel": _row(robot.get_qvel(), robot.max_dof),
        "candidate_simulator_state_sha256": _state_hash(uenv),
        "scene_body_states": current_scene,
        "reference_scene_body_states": reference_scene,
        "scene_pose_equal_reference_exact": scene_same,
        "scene_pose_changed_since_reference_ids": scene_changed,
        "moving_scene_body_ids": moving_scene,
        "other_articulations": _other_articulations(uenv),
        "scene": [], "grasps": [], "target_scene_id": None,
        "missing_for_witness": ["target_specific_pregrasp_and_grasp_tcp_frames",
                                "articulation_self_collision_filter_rules_unverified"],
        "camera_raw": _raw_camera_images(capture, output, issues),
        "limits": {"max_bodies": MAX_BODIES, "max_shapes": MAX_SHAPES,
                   "max_faces": MAX_FACES, "max_seconds": max_seconds},
    }
    try:
        urdf, urdf_root, urdf_links = _urdf_info(uenv.agent, output)
        joint_order = [joint.name for joint in robot.active_joints]
        qpos = data["candidate_qpos"]
        if len(joint_order) != len(qpos) or set(urdf_links) != {link.name for link in links}:
            issues.append("urdf_simulator_joint_or_link_coverage_mismatch")
        tcp_link = uenv.agent.tcp.name
        tcp_in_link = (_transform(uenv.agent.tcp.pose))
        import numpy as np
        tcp_in_link = (np.linalg.inv(tcp_in_link) @
                       _transform(uenv.agent.tcp_pose)).tolist()
        data["robot"] = {
            "urdf": urdf, "root_world": _transform(robot.root.pose).tolist(),
            "root_link": urdf_root, "anchor_qpos": qpos,
            "qpos_order": joint_order,
            "anchor_link_world": {link.name: _transform(link.pose).tolist()
                                  for link in links},
            "link_body_states": {link.name: _body_state(link, dynamic=True)
                                 for link in links},
            "tcp_link": tcp_link, "tcp_in_link": tcp_in_link,
            "ik_joints": list(getattr(uenv.agent, "arm_joint_names", [])),
            "finger_links": [uenv.agent.finger1_link.name,
                             uenv.agent.finger2_link.name],
            "meshes": [], "ignored_self_pairs": [],
            "self_filter_coverage": "shape_collision_groups_only",
        }
        for index, link in enumerate(links):
            try:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("export_time_limit_exceeded")
                mesh = _mesh_of_body(link, link.name, world=False, coverage=coverage)
                budget["shapes"] += len(coverage["shape_inventory"][link.name])
                if budget["shapes"] > MAX_SHAPES:
                    raise ValueError("shape_limit_exceeded")
                if mesh is None:
                    coverage["robot_no_shape"].append(link.name)
                else:
                    spec = _save_mesh(mesh, output / "robot", f"link-{index:03d}", budget=budget)
                    data["robot"]["meshes"].append({"link": link.name, **spec})
            except Exception as exc:
                coverage["mesh_errors"].append(
                    {"body": link.name, "error": f"{type(exc).__name__}: {exc}"})
        data["robot"]["ignored_self_pairs"] = _ignored_self_pairs(links, coverage)
        from bvi.mshab_adapter import scalar
        target = uenv.subtask_objs[int(scalar(uenv.subtask_pointer))]
        target_obj = target._objs[0]
        target_matches = []
        for index, (body_id, kind, body) in enumerate(others):
            if kind == "actor" and body._objs[0] is target_obj:
                target_matches.append(body_id)
            try:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("export_time_limit_exceeded")
                mesh = _mesh_of_body(body, body_id, world=True, coverage=coverage)
                budget["shapes"] += len(coverage["shape_inventory"][body_id])
                if budget["shapes"] > MAX_SHAPES:
                    raise ValueError("shape_limit_exceeded")
                if mesh is None:
                    coverage["scene_no_shape"].append(body_id)
                else:
                    spec = _save_mesh(mesh, output / "scene", f"body-{index:03d}", budget=budget)
                    data["scene"].append({"id": body_id, **spec})
            except Exception as exc:
                coverage["mesh_errors"].append(
                    {"body": body_id, "error": f"{type(exc).__name__}: {exc}"})
        if len(target_matches) != 1:
            issues.append(f"target_scene_actor_identity_matches:{len(target_matches)}")
        else:
            data["target_scene_id"] = target_matches[0]
            data["target_pose_world"] = current_scene[target_matches[0]]["pose_world"]
            if target_matches[0] not in {item["id"] for item in data["scene"]}:
                issues.append("target_collision_mesh_missing")
        if coverage["unexported_collision_shapes"] or coverage["mesh_errors"]:
            issues.append("physics_collision_mesh_coverage_incomplete")
        data["mesh_totals"] = budget
    except Exception as exc:
        issues.append(f"bundle_capture:{type(exc).__name__}:{exc}")
    data["issues"] = issues
    if moving_scene:
        data["missing_for_witness"].append("scene_dynamic_body_motion_requires_dynamic_collision_model")
    data["status"] = ("observed_missing_witness_inputs" if not issues
                      else "incomplete_observation")
    data["witness_ready"] = False
    manifest = output / "bundle.json"
    manifest.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return {"path": str(manifest), "status": data["status"], "issues": issues}


def finalize_summary_hash(bundle_directory, summary_path):
    """Bind the immutable export to the last summary save in the caller."""
    manifest = Path(bundle_directory) / "bundle.json"
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["summary"]["sha256"] = sha256(summary_path)
    manifest.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def compare_scene_poses(paths):
    """Offline exact cross-candidate scene pose and collision-mesh comparison."""
    def mesh_coverage_complete(data):
        coverage = data.get("coverage", {})
        return (not coverage.get("unexported_collision_shapes")
                and not coverage.get("mesh_errors")
                and len(data.get("scene", [])) + len(coverage.get("scene_no_shape", []))
                == coverage.get("scene_bodies_expected", -1))

    manifests = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    baseline = manifests[0]
    out = []
    for path, row in zip(paths[1:], manifests[1:]):
        left, right = baseline["scene_body_states"], row["scene_body_states"]
        common = set(left) & set(right)
        changed = sorted(key for key in common
                         if left[key]["pose_world"] != right[key]["pose_world"])
        mesh_left = {x["id"]: x["sha256"] for x in baseline["scene"]}
        mesh_right = {x["id"]: x["sha256"] for x in row["scene"]}
        baseline_moving, candidate_moving = _moving_ids(left), _moving_ids(right)
        covered = mesh_coverage_complete(baseline) and mesh_coverage_complete(row)
        same_target = (baseline.get("target_scene_id") is not None and
                       baseline.get("target_scene_id") == row.get("target_scene_id") and
                       baseline["target_scene_id"] in mesh_left and
                       row["target_scene_id"] in mesh_right)
        out.append({"path": str(path), "body_ids_equal": set(left) == set(right),
                    "changed_pose_ids": changed,
                    "baseline_moving_body_ids": baseline_moving,
                    "candidate_moving_body_ids": candidate_moving,
                    "mesh_coverage_complete": covered,
                    "target_scene_id_equal_and_meshed": same_target,
                    "scene_world_mesh_hashes_equal": mesh_left == mesh_right,
                    "exact_static_scene_equal": set(left) == set(right)
                    and not changed and mesh_left == mesh_right
                    and not baseline_moving and not candidate_moving
                    and covered and same_target})
    return {"baseline": str(paths[0]), "comparisons": out}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path,
                        help="C0 baseline bundle.json, then candidate bundle.json files")
    args = parser.parse_args()
    if len(args.manifests) < 2:
        parser.error("compare needs at least two bundles")
    print(json.dumps(compare_scene_poses(args.manifests), indent=2))
