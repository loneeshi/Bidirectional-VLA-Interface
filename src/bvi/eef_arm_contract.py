"""V-only arm request contract. All evaluator geometry is forbidden."""
import copy
import json
import re

from . import eef_arm_visual_contract as compact
from .eef_guidance_v4 import PROMPT as BASE_PROMPT
from .eef_tools import vector

VERSION = 'guidance-v4-arm'
SCHEMA = copy.deepcopy(compact.SCHEMA)
SCHEMA['properties']['tool']['enum'].remove('move_base')
ARM_NOTE = '''
GUIDANCE OVERRIDE guidance-v4-arm: the robot starts at an official Pick spawn.
move_base is disabled; do not request base motion. Earlier guidance about choosing
base movements is inapplicable. Keep turn_rad and forward_m null. All other tools,
sensor inputs, reasoning effort and history remain unchanged. The extended
nonofficial episode budget is 600 steps, with at most 150 steps per move.
'''


def prompt(condition):
    if condition != 'V':
        raise ValueError('revision two permits only V')
    base = BASE_PROMPT.replace('200 official control steps', '600 control steps (nonofficial extended horizon)').replace('60 steps per call', '150 steps per call').replace('at most 60 steps', 'at most 150 steps').replace('Reserve up to 60', 'Reserve up to 150')
    return base + ARM_NOTE


def validate_call(call):
    compact.validate_call(call)
    if call['tool'] == 'move_base':
        raise ValueError('move_base disabled in arm capability evaluation')
    return call


def _clean(bundle, condition):
    value = copy.deepcopy(bundle)
    obs, turn = value['observation'], value['turn']
    prompt(condition)
    for entry in obs['history']:
        if entry['call'] is not None:
            validate_call(entry['call'])
    # The legacy audit has a frozen three-case position whitelist. Position is
    # bookkeeping, absent from its generated wire. Validate our real identity
    # separately, then use one fixed legacy validation slot without changing it.
    value['position'] = 5
    compact.audit_observation(obs, value['sensor_manifest'], turn)
    return value


def build_wire(bundle, archive, condition):
    prompt(condition)
    case_id = bundle['position']
    if not isinstance(case_id, str) or re.fullmatch(r'arm-(dev|test)-[0-9]{3}', case_id) is None:
        raise ValueError('arm case identity required')
    cleaned = _clean(bundle, condition)
    history = []
    for item in archive:
        if set(item) != {'bundle', 'call'} or item['bundle']['position'] != case_id:
            raise ValueError('cross-case archive')
        if item['call'] is not None:
            validate_call(item['call'])
        history.append({'bundle': _clean(item['bundle'], condition), 'call': item['call']})
    body = json.loads(compact.build_wire(cleaned, history))
    body['instructions'] = prompt(condition)
    body['text']['format'].update(name='arm_capability_tool_call', schema=SCHEMA)
    content = body['input'][0]['content']
    meta = json.loads(content[0]['text'])
    meta.update(context_version=VERSION, condition=condition)
    content[0]['text'] = compact.canonical(meta).decode()
    wire = compact.canonical(body)
    if compact.non_image_bytes(wire) > 262144:
        raise ValueError('arm text history cap')
    return wire


def audit_wire(wire, bundle, archive, condition):
    if wire != build_wire(bundle, archive, condition):
        raise ValueError('serialized arm request changed')


def dispatch(executor, call, depth, observation_ids=()):
    from .eef_compact_dispatch import dispatch as shared_dispatch
    from .eef_tools import pose
    validate_call(call)
    if call['tool'] not in ('move_to', 'move_eef_chunk'):
        return shared_dispatch(executor, call, depth, observation_ids)
    before=executor.steps
    goals=[pose(p['position'],p['quaternion_xyzw']) for p in call['poses']]
    result=(executor.move_to(goals[0],max_steps=150) if call['tool']=='move_to'
            else executor.move_eef_chunk(goals,max_steps=150))
    used=executor.steps-before
    settling=sum(t['kind']=='base_settle' for t in executor.trace if t['step']>before)
    return call,compact.public_result({**result,'steps_used':used,'settle_steps':settling,
        'motion_steps':used-settling,'steps_remaining':executor.max_steps-executor.steps})
