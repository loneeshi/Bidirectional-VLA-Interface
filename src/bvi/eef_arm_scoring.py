"""Evaluation-only frozen funnel primitives. No evaluator signals enter requests."""
import math

FUNNEL = ('check_path_accepted','tcp_near_bbox','is_grasped','grasped_lift_5cm','strict_success')


def tcp_box_distance(tcp_object, bbox_min_object, bbox_max_object):
    # Transform TCP to target-object axes first; never use a world AABB expanded
    # by object rotation. Minimum distance is zero inside the object-local box.
    triples = list(zip(tcp_object,bbox_min_object,bbox_max_object))
    if len(triples)!=3 or any(not math.isfinite(float(x)) for values in triples for x in values):
        raise ValueError('three finite coordinates required')
    if any(lo>hi for _,lo,hi in triples): raise ValueError('invalid box')
    return math.sqrt(sum(max(lo-p,0,p-hi)**2 for p,lo,hi in triples))


def funnel(samples, accepted_checks, initial_object_z):
    """samples are per-step, including official final step; missing != false.

    object_z is the same target actor-origin world z at reset and every step.
    Distance uses that step's box and TCP. Lift and grasp must coincide.
    """
    if not samples: raise ValueError('missing trajectory; mark unscorable')
    required={'tcp_box_distance_m','is_grasped','object_z_m','strict_success'}
    for sample in samples:
        if not required <= set(sample): raise ValueError('incomplete evaluator sample')
        if any(type(sample[k]) is not bool for k in ('is_grasped','strict_success')): raise ValueError('official booleans required')
        if not all(math.isfinite(sample[k]) for k in ('tcp_box_distance_m','object_z_m')) or sample['tcp_box_distance_m']<0: raise ValueError('invalid measurements')
    raw=dict(zip(FUNNEL,[any(accepted_checks),any(s['tcp_box_distance_m']<=.05 for s in samples),
        any(s['is_grasped'] for s in samples),
        any(s['is_grasped'] and s['object_z_m']-initial_object_z>=.05 for s in samples),
        any(s['strict_success'] for s in samples)]))
    # Raw flags are authoritative. Cumulative intersections form a monotone
    # diagnostic funnel, but must not replace official strict-success counts.
    return {'raw':raw,'cumulative':{key:all(raw[k] for k in FUNNEL[:i+1]) for i,key in enumerate(FUNNEL)}}


def horizon_results(samples):
    successful=[s['step'] for s in samples if s['strict_success']]
    first=min(successful) if successful else None
    return {'first_strict_success_step':first,
            'success_within_200':first is not None and first<=200,
            'success_within_600':first is not None and first<=600,
            'horizon_description':'nonofficial 600-step horizon; 200-step prefix of same run'}


def wilson(k,n):
    if type(k) is not int or type(n) is not int or not 0<=k<=n: raise ValueError('invalid counts')
    if n==0:return None
    z=1.959963984540054; p=k/n; denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]
