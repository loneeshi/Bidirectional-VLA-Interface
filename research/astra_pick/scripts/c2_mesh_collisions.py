"""Read-only collision-mesh overlap inventory for a single C2 simulator state.

Requires python-fcl. ManiSkill exports the simulator's collision meshes, not
render meshes. This diagnoses overlaps at one instant; it does not certify a
motion path or unexported collider types such as infinite planes.
"""
from __future__ import annotations


def physx_pair_enabled(a, b):
    """SAPIEN's documented four-group collision filter for two shapes."""
    return bool(((a[0] & b[1]) or (a[1] & b[0])) and
                not ((a[2] & b[2]) and ((a[3] & 0xffff) == (b[3] & 0xffff))))


def audit_mesh_collisions(uenv, *, max_bodies=300):
    result = {"status": "incomplete", "scope": "instantaneous_exported_mesh_overlap",
              "robot_links": 0, "scene_bodies": 0, "unmeshed": [],
              "unmeshed_details": [],
              "mesh_errors": [],
              "overlaps": [], "self_overlaps": [],
              "self_overlap_filters": []}
    try:
        import trimesh
        import sapien.physx as physx
        from mani_skill.utils.geometry.trimesh_utils import (get_component_meshes,
                                                             merge_meshes)
        scene = uenv.scene
        robot = uenv.agent.robot
        links = list(robot.links)
        others = [("actor", body) for body in scene.actors.values()]
        others += [("other_link", link)
                   for art in scene.articulations.values()
                   if art is not robot and art.name != robot.name
                   for link in art.links]
        result["robot_links"] = len(links)
        result["scene_bodies"] = len(others)
        if not links or not others or len(links) + len(others) > max_bodies:
            result["reason"] = "empty_body_inventory_or_body_limit"
            return result
        manager = trimesh.collision.CollisionManager()
        self_manager = trimesh.collision.CollisionManager()
        robot_by_key = {}
        def get_mesh(body, key):
            try:
                if not hasattr(body, '_objs'):
                    mesh = body.get_first_collision_mesh(to_world_frame=True)
                    if mesh is None:
                        result["unmeshed"].append(key)
                    return mesh
                meshes = []
                shapes = []
                for obj in body._objs:
                    components = ([obj] if isinstance(obj, physx.PhysxRigidBaseComponent)
                                  else [c for c in obj.components
                                        if isinstance(c, physx.PhysxRigidBaseComponent)])
                    for component in components:
                        shapes.extend(type(shape).__name__
                                      for shape in component.get_collision_shapes())
                        mesh = merge_meshes(get_component_meshes(component))
                        if mesh is not None:
                            meshes.append(mesh)
                if not meshes:
                    result["unmeshed"].append(key)
                    result["unmeshed_details"].append({"body": key,
                                                       "collision_shapes": shapes})
                    return None
                mesh = trimesh.util.concatenate(meshes)
                # Follow ManiSkill's Actor/Articulation mesh exporter: the
                # wrapper Pose is authoritative for GPU-updated link poses.
                mesh.apply_transform(body.pose[0].sp.to_transformation_matrix())
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    result["unmeshed"].append(key)
                    return None
                return mesh
            except Exception as exc:
                result["mesh_errors"].append({"body": key,
                                              "error": f"{type(exc).__name__}: {exc}"})
                return None
        for index, (kind, body) in enumerate(others):
            key = f"{kind}:{index}:{body.name}"
            mesh = get_mesh(body, key)
            if mesh is None:
                continue
            manager.add_object(key, mesh)
        for index, link in enumerate(links):
            key = f"robot_link:{index}:{link.name}"
            mesh = get_mesh(link, key)
            if mesh is None:
                continue
            self_manager.add_object(key, mesh)
            robot_by_key[key] = link
            collides, names = manager.in_collision_single(mesh, return_names=True)
            if collides:
                result["overlaps"].extend({"robot_link": key, "other": name}
                                          for name in sorted(names))
        self_collides, self_pairs = self_manager.in_collision_internal(return_names=True)
        if self_collides:
            result["self_overlaps"] = [list(pair) for pair in sorted(self_pairs)]
            for pair in sorted(self_pairs):
                left, right = (robot_by_key[name] for name in pair)
                left_groups = [shape.get_collision_groups()
                               for shape in left._objs[0].get_collision_shapes()]
                right_groups = [shape.get_collision_groups()
                                for shape in right._objs[0].get_collision_shapes()]
                result["self_overlap_filters"].append({
                    "pair": list(pair),
                    "physx_collision_enabled": any(
                        physx_pair_enabled(a, b) for a in left_groups
                        for b in right_groups),
                    "left_shape_groups": left_groups,
                    "right_shape_groups": right_groups})
        result["status"] = ("measured_partial" if result["unmeshed"]
                            or result["mesh_errors"] else "measured")
        return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
