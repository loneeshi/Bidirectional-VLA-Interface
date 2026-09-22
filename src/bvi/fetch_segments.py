"""Observable-boundary pilot labels, distinct from learned runtime progress.

STATUS: frozen — historical training and diagnostics (retained)

Actions are [start,end), observations are [0,N]. Local progress is supervised
elapsed fraction toward an evidenced endpoint, not a simulator progress oracle.
Failed trajectory ends never constitute completion. Thresholds are port choices.
"""
import numpy as np

def segment_episode(task, grasped, tcp_object_distance, object_goal_distance,
                    success_steps, *, reach_metres=.08, place_metres=.15, stable=3):
    grasped=np.asarray(grasped,dtype=bool)
    distances=np.asarray(tcp_object_distance,dtype=float)
    goals=np.asarray(object_goal_distance,dtype=float)
    n=len(grasped)-1
    if task not in ('pick','place') or n<1 or distances.shape!=(n+1,) or goals.shape!=(n+1,):
        raise ValueError('Invalid episode shapes/task')
    if not np.isfinite(distances).all() or not np.isfinite(goals).all() or stable<1:
        raise ValueError('Invalid annotation evidence')
    result=[]
    def add(family,start,end,predicate,instruction):
        if end>start:result.append(dict(family=family,start=start,end=end,
            completion_evidence=predicate,instruction=instruction,
            progress_label='local_elapsed_fraction_to_verified_boundary'))
    successes=sorted(int(i) for i in success_steps if 0<int(i)<=n)
    if task=='pick':
        near=next((i for i in range(1,n+1) if distances[i]<=reach_metres and not grasped[i-1]),None)
        if near is None:return []
        add('reach',0,near,'tcp_object_distance<=reach_metres','Reach the apple with the gripper open.')
        hold=next((i for i in range(near+stable,n+1) if grasped[i-stable+1:i+1].all()),None)
        if hold is not None:
            add('grasp',near,hold,'stable_grasp','Grasp the apple securely.')
            end=next((i for i in successes if i>hold and grasped[hold:i+1].all()),None)
            if end is not None:add('move',hold,end,'native_pick_success_while_grasped','Move the held apple to the robot rest pose.')
    else:
        # Reset has no physics contact yet. Start only at an evidenced held
        # interval; do not label the unverified reset frame as holding.
        held_start=next((i for i in range(n-stable+2) if grasped[i:i+stable].all()),None)
        if held_start is None:return []
        release=next((i for i in range(held_start+stable,n+1) if not grasped[i] and grasped[i-1]),None)
        if release is None:return []
        # Move ends while still holding close to the placement goal, before release.
        near=next((i for i in range(held_start+1,release) if goals[i]<=place_metres and grasped[held_start:i+1].all()),None)
        if near is None:return []
        add('move',held_start,near,'held_object_near_place_goal','Move the held apple to its placement goal.')
        end=next((i for i in successes if i>=release and not grasped[i]),None)
        if end is not None:add('release',near,end,'native_place_success_after_release','Release the apple at its placement goal and retract.')
    return result
