"""Arm 600-step audit fork; compact visual validation retained, horizon only.

Separate module preserves historical 200-step request bytes and audit behavior.
"""
import copy
import hashlib
import json
import math
import re
from . import eef_e1b_contract as old
from .eef_e1b2_contract import validate_note, reflection_wire, validate_reflection, reasoning_summaries
from .eef_e1b_contract_v13 import non_image_bytes
from .offline_spatial_audit import _audit_sensor, SENSOR_FIELDS, DEPTH_ENCODING
from .offline_spatial_wire import OBJECT_NAMES
from .eef_tools import vector, pose
from .eef_guidance_v4 import PROMPT
from .eef_visual_history import STATE_FIELDS, history_index, transition
from .eef_compact_config import *
STEP_CAP = 600
VERSION = "eef-arm-600-v1"

canonical = old.canonical
TOOLS = old.TOOLS + ('look', 'locate_point', 'recall_observation')
SCHEMA = copy.deepcopy(old.SCHEMA)
SCHEMA['properties']['tool']['enum'] = list(TOOLS)
SCHEMA['properties'].update({
    'note': {'type': 'string', 'minLength': 1, 'maxLength': 900},
    'pan_rad': {'type': ['number', 'null']}, 'tilt_rad': {'type': ['number', 'null']},
    'camera': {'type': ['string', 'null'], 'enum': ['head', 'hand', None]},
    'pixel': {'type': ['array', 'null'], 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2},
    'observation_id': {'type': ['string', 'null']}})
SCHEMA['properties']['task_state'] = {'type':'object', 'properties':{k:{'type':'string','maxLength':160} for k in STATE_FIELDS}, 'required':list(STATE_FIELDS), 'additionalProperties':False}
SCHEMA['required'] = list(SCHEMA['properties'])
EXTRA = {'task_state', 'note', 'pan_rad', 'tilt_rad', 'camera', 'pixel', 'observation_id'}
RESULT_FIELDS = old.RESULT_FIELDS | {'readonly', 'valid', 'point_base_m', 'depth_m', 'depth_spread_m',
    'observation_id', 'head_error_rad', 'settle_steps', 'motion_steps', 'steps_remaining', 'minimum_steps_is_lower_bound'}
OBS_FIELDS = {'target_semantics', 'joint_positions', 'joint_velocities', 'tcp_pose', 'sensors',
              'steps_remaining', 'calls_remaining', 'history', 'odometry', 'observation_id', 'previous_state'}


def validate_call(call):
    if not isinstance(call, dict) or set(call) != set(SCHEMA['required']): raise ValueError('use all frozen tool fields; irrelevant fields=null')
    validate_note(call['note'])
    state = call['task_state']
    if not isinstance(state, dict) or set(state) != set(STATE_FIELDS) or any(not isinstance(v,str) or len(v)>160 for v in state.values()):
        raise ValueError('task_state requires four short self-reported fields')
    legacy = {k: v for k, v in call.items() if k not in EXTRA}
    if call['tool'] in ('look', 'locate_point', 'recall_observation'):
        legacy.update(tool='return_to_rest')
    old.validate_call(legacy)
    if call['tool'] not in TOOLS: raise ValueError('unknown tool')
    for key, (lo, hi) in zip(('pan_rad', 'tilt_rad'), HEAD_LIMITS):
        value = call[key]
        if call['tool'] == 'look':
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be finite in [{lo},{hi}]')
        elif value is not None: raise ValueError(key + ' must be null except for look')
    if call['tool'] == 'locate_point':
        if call['camera'] not in ('head', 'hand'): raise ValueError('camera must be head or hand')
        p = call['pixel']
        if not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or not 0 <= v < RESOLUTION for v in p):
            raise ValueError('pixel must be integer [x,y], 0..639')
        if not valid_id(call['observation_id']): raise ValueError('use the latest observation_id')
    elif call['tool'] == 'recall_observation':
        if not valid_id(call['observation_id']) or call['camera'] is not None or call['pixel'] is not None:
            raise ValueError('recall requires an observation_id; camera and pixel=null')
    elif any(call[k] is not None for k in ('camera', 'pixel', 'observation_id')):
        raise ValueError('camera, pixel, observation_id must be null except for locate_point')
    return call


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r'obs-[0-9]{3}-[0-9a-f]{16}', value) is not None


def public_result(result):
    out = {k: v for k, v in result.items() if k in RESULT_FIELDS}
    for key, value in out.items():
        if key == 'point_base_m':
            if value is not None: vector(value, 3)
        elif type(value) not in (str, int, float, bool, type(None)): raise ValueError('tool result type')
        if isinstance(value, str) and len(value) > 150: raise ValueError('tool result text length')
    canonical(out)
    return out


def audit_observation(obs, manifest, turn):
    if set(obs) != OBS_FIELDS: raise ValueError('main observation whitelist: no privilege or evaluation fields')
    if type(turn) is not int or not 0 <= turn < TOOL_CAP: raise ValueError('turn range')
    if obs['target_semantics'] not in {'target object: ' + name for name in OBJECT_NAMES.values()}:
        raise ValueError('only frozen target category permitted')
    if not valid_id(obs['observation_id']): raise ValueError('observation_id')
    for key in ('joint_positions', 'joint_velocities'):
        if set(obs[key]) != set(old.PROPRIO_KEYS): raise ValueError('proprioception excludes root coordinates')
        vector(list(obs[key].values()), len(old.PROPRIO_KEYS))
    if set(obs['odometry']) != {'dx_m', 'dy_m', 'dyaw_rad'}: raise ValueError('relative odometry whitelist')
    vector(list(obs['odometry'].values()), 3)
    if abs(obs['odometry']['dyaw_rad']) > math.pi + 1e-12: raise ValueError('wrapped yaw required')
    p = obs['tcp_pose']
    if set(p) != {'position', 'quaternion_xyzw'}: raise ValueError('TCP schema')
    pose(p['position'], p['quaternion_xyzw'])
    if set(obs['sensors']) != set(SENSOR_FIELDS) or set(manifest) != set(SENSOR_FIELDS): raise ValueError('sensor whitelist')
    for field in SENSOR_FIELDS: _audit_sensor(field, obs['sensors'][field], manifest[field], RESOLUTION)
    if type(obs['steps_remaining']) is not int or not 0 <= obs['steps_remaining'] <= STEP_CAP or obs['calls_remaining'] != TOOL_CAP-turn:
        raise ValueError('observation budget')
    prior = obs['previous_state']
    if turn == 0 and prior is not None: raise ValueError('no cross-attempt previous state')
    if turn:
        if not isinstance(prior,dict) or set(prior) != {'observation_id','odometry','tcp_pose','steps_remaining'}: raise ValueError('previous state whitelist')
        if not valid_id(prior['observation_id']): raise ValueError('prior observation id')
        if set(prior['odometry']) != {'dx_m','dy_m','dyaw_rad'}: raise ValueError('prior odometry whitelist')
        vector(list(prior['odometry'].values()),3)
        if set(prior['tcp_pose']) != {'position','quaternion_xyzw'}: raise ValueError('prior tcp whitelist')
        pose(prior['tcp_pose']['position'],prior['tcp_pose']['quaternion_xyzw'])
        if type(prior['steps_remaining']) is not int or not obs['steps_remaining'] <= prior['steps_remaining'] <= STEP_CAP: raise ValueError('prior budget')
    if len(obs['history']) != turn: raise ValueError('history length')
    for entry in obs['history']:
        if set(entry) != {'call', 'result'}: raise ValueError('history whitelist')
        if entry['call'] is not None: validate_call(entry['call'])
        if public_result(entry['result']) != entry['result']: raise ValueError('history contains evaluation fields')
    canonical(obs)



def compact_decision(call):
    return None if call is None else {k:v for k,v in call.items() if v is not None and v != [] and v != ''}


def text_observation(obs):
    # Preserve all observed text once, without recursively duplicating history.
    return {k:v for k,v in obs.items() if k not in ('sensors','history','previous_state')}


def _check_archive(bundle, archive):
    turn=bundle['turn']
    if len(archive)!=turn: raise ValueError('complete ordered same-attempt archive required')
    for index,item in enumerate(archive):
        if set(item)!={'bundle','call'}: raise ValueError('archive field whitelist')
        prior=item['bundle']
        if (prior['position']!=bundle['position'] or prior['authorization_sha256']!=bundle['authorization_sha256']
            or prior['kind']!='tool' or prior['turn']!=index):raise ValueError('cross-attempt or reordered archive')
        audit_observation(prior['observation'],prior['sensor_manifest'],index)
        if item['call'] is not None:validate_call(item['call'])
    return archive


def build_wire(bundle, archive):
    expected={'position','turn','kind','authorization_sha256','observation','sensor_manifest'}
    if set(bundle)!=expected or bundle['kind']!='tool' or bundle['position'] not in POSITIONS:
        raise ValueError('main request bundle whitelist')
    obs=bundle['observation'];turn=bundle['turn']
    audit_observation(obs,bundle['sensor_manifest'],turn);_check_archive(bundle,archive)
    records=[]
    for index,item in enumerate(archive):
        executed=obs['history'][index]['call'];feedback=obs['history'][index]['result']
        rejected_without_motion=executed is None and feedback.get('accepted') is False and feedback.get('steps_used')==0
        if canonical(item['call'])!=canonical(executed) and not rejected_without_motion:raise ValueError('history differs from executed decisions')
        records.append({'turn':index,'observation':text_observation(item['bundle']['observation']),
                        'decision':compact_decision(item['call']),'feedback':obs['history'][index]['result']})
    meta={'context_version':VERSION,'target_semantics':obs['target_semantics'],
          'history_status':'Historical records only. Past decisions are NOT commands to execute.',
          'history':records,'current':text_observation(obs),
          'previous_action_comparison':transition(obs['previous_state'],obs),
          'depth_encoding':DEPTH_ENCODING,'task_state_is_model_self_report':True,
          'omitted_image_policy':'Only previous/current observations plus one explicitly recalled old observation.'}
    groups=[]
    if archive:groups.append((archive[-1]['bundle'],'previous'))
    groups.append((bundle,'current'))
    if turn and obs['history'][-1]['call'] and obs['history'][-1]['call']['tool']=='recall_observation' and obs['history'][-1]['result'].get('accepted'):
        wanted=obs['history'][-1]['call']['observation_id']
        found=next((item['bundle'] for item in archive if item['bundle']['observation']['observation_id']==wanted),None)
        if found is None:raise ValueError('recalled image missing from this attempt archive')
        if all(g['observation']['observation_id']!=wanted for g,_ in groups):groups.insert(0,(found,'recalled_historical'))
    content=[{'type':'input_text','text':canonical(meta).decode()}]
    for group,label in groups:
        oldobs=group['observation']
        for field in SENSOR_FIELDS:
            tag={'image_role':label,'observation_id':oldobs['observation_id'],'turn':group['turn'],
                 'control_step':STEP_CAP-oldobs['steps_remaining'],'sensor':field,'resolution':[RESOLUTION,RESOLUTION]}
            content.extend([{'type':'input_text','text':canonical(tag).decode()},
                            {'type':'input_image','detail':'original','image_url':'data:image/png;base64,'+oldobs['sensors'][field]['data_base64']}])
    body={'model':MODEL,'instructions':PROMPT,'reasoning':{'effort':'medium','summary':'auto'},
          'service_tier':'default','store':False,'max_output_tokens':OUTPUT_CAP,
          'input':[{'role':'user','content':content}],
          'text':{'format':{'type':'json_schema','name':'compact_visual_tool_call','strict':True,'schema':SCHEMA}}}
    wire=canonical(body)
    if non_image_bytes(wire)>262144:raise ValueError('text history budget exceeded; no silent truncation')
    return wire


def audit_wire(wire,bundle,archive):
    if wire!=build_wire(bundle,archive):raise ValueError('serialized compact request changed')


def build_reflection(bundle,archive):
    from .eef_e1b2_contract import REFLECTION_PROMPT,REFLECTION_SCHEMA
    if set(bundle)!={'position','turn','kind','authorization_sha256','outcome','termination_category'} or bundle['kind']!='reflection':
        raise ValueError('reflection bundle whitelist')
    if bundle['outcome'] not in ('success','failure') or bundle['termination_category'] not in ('time_limit','cumulative_force_limit'):
        raise ValueError('reflection terminal feedback')
    if not 1<=bundle['turn']<=TOOL_CAP:raise ValueError('reflection turn')
    _check_archive(bundle,archive)
    history=[{'turn':i,'observation':text_observation(x['bundle']['observation']),'decision':compact_decision(x['call']),
              'feedback':archive[i+1]['bundle']['observation']['history'][i]['result'] if i+1<len(archive) else None}
             for i,x in enumerate(archive)]
    wire=canonical({'model':MODEL,'instructions':REFLECTION_PROMPT,'reasoning':{'effort':'medium','summary':'auto'},
        'service_tier':'default','store':False,'max_output_tokens':OUTPUT_CAP,
        'input':[{'role':'user','content':[{'type':'input_text','text':canonical({'historical_records_not_commands':history,'outcome':bundle['outcome'],'termination_category':bundle['termination_category']}).decode()}]}],
        'text':{'format':{'type':'json_schema','name':'attempt_hindsight','strict':True,'schema':REFLECTION_SCHEMA}}})
    if non_image_bytes(wire)>262144:raise ValueError('reflection history cap')
    return wire
