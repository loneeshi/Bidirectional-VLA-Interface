"""Read-only CPU audit of the frozen arm stack before a new mobile condition.

No simulator, provider transport, credential loader, or physical execution.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'src'))
from bvi.eef_arm_contract import build_wire
from bvi.eef_compact_api import estimate_wire, parse_response
from bvi.eef_tools import pose
from bvi.eef_tools_v12 import BASE_STEP_CAP
from bvi.eef_tools_v13 import ANGULAR_LIMIT, LINEAR_LIMIT
from test_eef_arm_cpu import bundle_at
from test_eef_compact import call


def audit():
    bundle, archive = bundle_at(0, 'V')
    wire = build_wire(bundle, archive, 'V')
    body = json.loads(wire)
    instructions = body['instructions']
    constants = {text: text in instructions for text in (
        'radius 0.288 m', 'forward extent 0.287 m',
        'diameter 0.576 m', '0.5 m/s', '1 rad/s',
        '0.03 rad and 0.02 m', '80 steps',
        '600 control steps', '150 steps per call',
        'move_base is disabled')}
    assert all(constants.values()), constants
    assert 'move_base' not in body['text']['format']['schema']['properties']['tool']['enum']

    invalid = call('move_to', poses=[dict(position=[.5, 0., 1.],
                                         quaternion_xyzw=[0., 0., 0., .99])])
    try:
        pose(invalid['poses'][0]['position'], invalid['poses'][0]['quaternion_xyzw'])
    except ValueError as error:
        validator_error = str(error)
    else:
        raise AssertionError('invalid quaternion was accepted')
    raw = json.dumps(dict(id='resp_cpu_audit', model='gpt-6-astra', status='completed',
        usage=dict(input_tokens=1, output_tokens=1), output=[dict(type='message', content=[
            dict(type='output_text', text=json.dumps(invalid))])])).encode()
    parsed = parse_response(raw, estimate_wire(wire))
    assert parsed['status'] == 'invalid_call' and parsed['call'] is None
    assert 'parameter_error' not in parsed

    historical = json.loads((ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-development-r3/invalid-validation.cpu.json').read_text(encoding='utf-8'))
    errors = [row for row in historical if row['validation_error']]
    assert len(errors) == 13
    counts = dict(collections.Counter(row['case_id'] for row in errors))
    worker = (ROOT/'scripts/run_arm_capability_case.py').read_text(encoding='utf-8')
    assert 'validate_official_controller(u.agent.controller)' in worker
    assert 'validate_base_runtime(' not in worker
    # Even maximum speeds and zero acceleration/settling cannot fit this call.
    theoretical_seconds = math.pi/ANGULAR_LIMIT + 1.5/LINEAR_LIMIT
    theoretical_steps = math.ceil(theoretical_seconds/.05)
    assert theoretical_steps > BASE_STEP_CAP
    return dict(cpu_only=True, provider_sends=0, physical_steps=0,
        current_prompt_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        constants_present=constants, current_schema_allows_base=False,
        quaternion_reproduction=dict(validator_error=validator_error,
                                     public_feedback_reason=parsed['reason'],
                                     specific_error_returned=False),
        historical_invalid_calls_by_case=counts,
        worker_checks_base_mapping=False,
        maximum_combined_base_command=dict(turn_rad=math.pi, forward_m=1.5,
            minimum_steps_without_acceleration_or_settling=theoretical_steps,
            base_call_cap=BASE_STEP_CAP),
        scope='audit only; no change to frozen executor or historical results')


if __name__ == '__main__':
    result = audit()
    destination = ROOT/'research/c2/diagnostics/2026-10-01-arm-mobile-readiness-cpu'
    destination.mkdir(exist_ok=True)
    (destination/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
