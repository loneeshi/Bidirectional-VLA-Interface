"""Evaluation-only grasp heuristic, pending authorized development calibration.

No network, request construction, SAC outputs, collision checks or base actions.
The executor is supplied by the future runner and must be the Astra executor.
"""
import math
from .eef_arm_contract import SCHEMA, dispatch, validate_call
from .eef_tools import vector

VERSION = 'arm-reference-draft-v1'
APPROACH_ORDER = ('top', 'front', 'side')


def tool_call(tool, **kwargs):
    value = {key: None for key in SCHEMA['required']}
    value.update(tool=tool, poses=[], hindsight='', note='Evaluation-only scripted reference: execute the fixed first accepted approach.',
                 task_state={key: 'evaluation-only fixed heuristic' for key in ('belief','uncertainty','expected_change','next_check')})
    value.update(kwargs)
    return validate_call(value)


def candidates(lower, upper):
    lo, hi = vector(lower, 3), vector(upper, 3)
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError('invalid target AABB')
    center = [(a+b)/2 for a,b in zip(lo,hi)]
    root = math.sqrt(.5)
    for name, delta, quat in (
        ('top', [0,0,.12], [0,root,0,root]),
        ('front', [-.12,0,0], [0,0,0,1]),
        ('side', [0,-.12,0], [0,0,root,root])):
        pre = [a+b for a,b in zip(center,delta)]
        lift = [center[0],center[1],center[2]+.10]
        yield name, [{'position':p,'quaternion_xyzw':quat} for p in (pre,center,lift)]


def run_reference(executor, lower, upper, record, is_terminal):
    """Runner owns timeout, 600-step budget, recorder and official termination.

    record(call,result) must persist every check and command. Never fall through
    to another approach after execution starts. No automatic retry.
    """
    def invoke(tool, **kwargs):
        if is_terminal():
            return None
        call = tool_call(tool, **kwargs)
        _, result = dispatch(executor, call, None)
        record(call, result)
        return result
    selected = None
    for name, poses in candidates(lower, upper):
        result = invoke('check_path', poses=poses)
        if result is None:
            return {'status':'official_termination_before_selection'}
        if result.get('accepted'):
            selected = name, poses
            break
    if selected is None:
        return {'status':'no_accepted_path'}
    name, (pre, grasp, lift) = selected
    for tool, fields in (
        ('set_gripper', {'gripper':1.}), ('move_to', {'poses':[pre]}),
        ('move_to', {'poses':[grasp]}), ('set_gripper', {'gripper':0.}),
        ('move_to', {'poses':[lift]}), ('return_to_rest', {})):
        result = invoke(tool, **fields)
        if result is None:
            return {'status':'official_termination', 'approach':name}
        if result.get('accepted', True) is False or result.get('arrived') is False:
            return {'status':'execution_incomplete', 'approach':name, 'tool':tool}
    return {'status':'sequence_finished', 'approach':name,
            'strict_success':'read official evaluator; never infer from completion'}
