import json
from types import SimpleNamespace as NS

import pytest

from bvi.coordinator import parse_request, request_schema
from bvi.protocol import Observation, AllowedCall, Target, SkillSpec, ProtocolError
from bvi.fetch_pi_skill import FetchPiSkill, JOINT_NAMES, CONVENTION
from bvi.lightnav_skill import LightNavSkill


def fixture():
    obs = Observation('f0', 0, targets=(Target('can', 'green can'),),
                      allowed_calls=(AllowedCall('pick', 'can'),), metadata={'subtask_index': 1})
    specs = {'pick': SkillSpec('pick')}
    body = dict(call_id='call', skill='pick', target_id='can', observation_id='f0',
                requirements=[dict(id='r', predicate='benchmark_success')], max_steps=20,
                timeout_seconds=30, tool_family='pick', instruction='Pick up the green can.',
                interface_version='mshab-tool-family/1')
    return obs, specs, body


def test_schema_parser_and_exact_instruction():
    obs, specs, body = fixture()
    assert 'instruction' in request_schema(obs, specs, True)['required']
    result = parse_request(json.dumps(body), obs, specs, True)
    assert result.instruction == body['instruction']
    with pytest.raises(ProtocolError):
        parse_request(json.dumps(body), obs, specs)
    for patch in ({'tool_family':'navigate'}, {'instruction':' '}, {'instruction':'x'*161},
                  {'instruction':None}, {'interface_version':'unknown'}):
        with pytest.raises(ProtocolError):
            parse_request(json.dumps(dict(body, **patch)), obs, specs, True)


def test_fetch_uses_gpt_instruction_and_clears_previous_chunk():
    obs, specs, body = fixture()
    request = parse_request(json.dumps(body), obs, specs, True)
    robot = NS(active_joints=[NS(name=n) for n in JOINT_NAMES], qpos=[[0.]*15])
    adapter = NS(uenv=NS(agent=NS(robot=robot)), original_plan=NS(subtasks=[None, NS(type='pick')]),
                 logger=NS(emit=lambda *a, **k:None))
    client = NS(metadata={'robot':'fetch','action_dim':13,'state_dim':15,'action_convention':CONVENTION,
                          'state_conditioning':True})
    skill = FetchPiSkill('pick', adapter, client, instructions={'1':'Old fixed text'})
    skill.actions.append((1.,)*13)
    skill.start(request, obs)
    assert skill.prompt == body['instruction']
    assert not skill.actions
    robot.qpos[0][:2] = [0.2, 0.3]
    skill.start(request, obs)
    assert skill.base_xy_origin == (0., 0.)


def test_navigation_uses_invocation_instruction_and_clears_history():
    skill = object.__new__(LightNavSkill)
    resets=[]
    skill.client=NS(reset=lambda:resets.append(True))
    skill.control=NS(capture_hold=lambda:None)
    skill.instructions={'0':'Old fixed text'}
    skill.replan_steps=5
    skill.start(NS(skill='navigate',call_id='nav',instruction='Approach the table beside the chair.'),
                Observation('f',0,metadata={'subtask_index':0}))
    assert skill.instruction == 'Approach the table beside the chair.'
    assert skill.invocation_instruction and resets == [True]
    assert skill.targets == ()
