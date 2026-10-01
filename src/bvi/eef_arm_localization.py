"""Evaluator-only unsigned point-to-collision-triangle-surface distance.

Never returns this diagnostic as tool feedback. Interior points have positive
distance to the surface, unlike AABB containment distance used by the funnel.
"""
import numpy as np
from trimesh.triangles import closest_point


def diagnose_location(tool_result,triangles_object,object_world,base_world):
    point=tool_result.get('point_base_m')
    meta={'evaluation_only':True,'observation_id':tool_result.get('observation_id'),
          'surface_distance_m':None,'definition':'unsigned minimum Euclidean distance to all target collision-shape triangle surfaces at this call'}
    if not tool_result.get('valid') or point is None:
        return dict(meta,status='no_valid_returned_point')
    triangles=np.asarray(triangles_object,dtype=float)
    point=np.asarray(point,dtype=float)
    object_world=np.asarray(object_world,dtype=float);base_world=np.asarray(base_world,dtype=float)
    if (triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or len(triangles)==0
        or point.shape!=(3,) or object_world.shape!=(4,4) or base_world.shape!=(4,4)
        or not all(np.isfinite(x).all() for x in (triangles,point,object_world,base_world))):
        raise ValueError('invalid evaluator geometry')
    local=(np.linalg.inv(object_world)@base_world@np.r_[point,1])[:3]
    closest=closest_point(triangles,np.repeat(local[None,:],len(triangles),axis=0))
    distance=float(np.linalg.norm(closest-local,axis=1).min())
    return dict(meta,status='scored',surface_distance_m=distance,point_object_eval_only=local.tolist())
