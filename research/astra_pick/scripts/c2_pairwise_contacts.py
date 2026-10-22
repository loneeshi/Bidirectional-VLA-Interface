"""Diagnostic contact-force inventory for the C2 stance pilot.

ManiSkill's GPU scene has pairwise contact-force queries, even though its
PhysxGpuSystem has no get_contacts(). A zero force is not a collision-distance
certificate: resting penetration, contact filtering, and transient contacts
still need separate checks. Query construction can also change GPU execution;
rerun the same C0 path before comparing any SAC outcome after enabling this.
"""
from __future__ import annotations

import math


def _name(body):
    return str(getattr(body, "name", type(body).__name__))


def audit_pairwise_contacts(uenv, *, selected_pairs=None, max_pairs=2000):
    """Measure robot link contacts with scene bodies after a physics step.

    Fail closed on incomplete enumeration or query errors. Return observations,
    never a `collision_free` verdict.
    """
    scene = uenv.scene
    robot = uenv.agent.robot
    links = list(robot.links)
    actors = list(scene.actors.values())
    other_links = [link for articulation in scene.articulations.values()
                   if articulation is not robot and _name(articulation) != _name(robot)
                   for link in articulation.links]
    targets = actors + other_links
    # Include robot self pairs. The engine's own collision filters decide which
    # pairs can produce impulses; zero is still only a measurement.
    if selected_pairs is None:
        pairs = [(left, right) for left in links for right in targets]
        pairs += [(links[i], links[j]) for i in range(len(links))
                  for j in range(i + 1, len(links))]
        coverage = "all_enumerated_pairs"
    else:
        link_map = {f"robot_link:{i}:{_name(link)}": link
                    for i, link in enumerate(links)}
        target_map = {f"actor:{i}:{_name(actor)}": actor
                      for i, actor in enumerate(actors)}
        target_map.update({f"other_link:{i + len(actors)}:{_name(link)}": link
                           for i, link in enumerate(other_links)})
        try:
            pairs = [(link_map[pair["robot_link"]], target_map[pair["other"]])
                     for pair in selected_pairs]
        except (KeyError, TypeError) as exc:
            return {"status": "incomplete", "reason": f"invalid_selected_pair: {exc}"}
        coverage = "mesh_overlap_pairs_only"
    result = {"status": "incomplete", "robot_links": len(links),
              "scene_actors": len(actors), "other_articulation_links": len(other_links),
              "planned_pairs": len(pairs), "queried_pairs": 0,
              "nonzero_pairs": [], "max_pairs": max_pairs,
              "scope": "post_step_contact_impulses_only_not_collision_clearance",
              "coverage": coverage}
    if not links or not targets or len(pairs) > max_pairs:
        result["reason"] = "empty_body_inventory_or_pair_limit"
        return result
    try:
        for left, right in pairs:
            vector = scene.get_pairwise_contact_forces(left, right)
            values = vector.detach().cpu().reshape(-1).tolist()
            if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("unexpected contact-force vector")
            magnitude = math.sqrt(sum(float(v) ** 2 for v in values))
            result["queried_pairs"] += 1
            if magnitude > 0:
                result["nonzero_pairs"].append({"link": _name(left),
                                                 "other": _name(right),
                                                 "force_n": magnitude})
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    result["status"] = "measured"
    return result
