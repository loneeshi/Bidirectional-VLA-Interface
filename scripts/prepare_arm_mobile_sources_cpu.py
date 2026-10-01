"""Create independent mobile operators from the audited frozen arm operators."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def change(text, old, new):
    if old not in text: raise ValueError('source drift: '+old[:80])
    return text.replace(old,new)


def main():
    text=(ROOT/'scripts/eef_arm_server_broker.py').read_text(encoding='utf-8')
    text=change(text,'from bvi.eef_arm_contract import build_wire, audit_wire, validate_call, _clean, compact',
        'from bvi.eef_mobile_contract import build_wire, audit_wire, validate_call, compact')
    text=change(text,'estimate_wire, parse_response, committed_cost','estimate_wire, committed_cost\nfrom bvi.eef_mobile_api import parse_response')
    text=change(text,"'revision':2","'revision':3,'condition':'V-mobile'")
    text=change(text,"'gpu1_process_seconds_cap':13500","'gpu1_process_seconds_cap':9000")
    text=change(text,"if not 0<Decimal(auth['usd_cap'])<=65", "if not Decimal(auth['usd_cap']).is_finite() or not 0<Decimal(auth['usd_cap'])<=65")
    start=text.index('def wire_for(');end=text.index('\n\ndef process(',start)
    text=text[:start]+'''def wire_for(bundle,archive,condition):
    if bundle['kind']!='tool':raise ValueError('mobile tool request only')
    return build_wire(bundle,archive,condition)
'''+text[end:]
    text=change(text,"condition != 'V'", "condition != 'V-mobile'")
    text=change(text,"            except ValueError:result.update(status='invalid_call',call=None,reason='invalid_arm_call')",
        "            except ValueError:raise AssertionError('mobile parser and validator disagree')")
    text=change(text,'Arm-specific durable broker','Mobile-specific durable broker')
    (ROOT/'scripts/eef_mobile_server_broker.py').write_text(text,encoding='utf-8')

    text=(ROOT/'scripts/run_arm_capability_case.py').read_text(encoding='utf-8')
    text=change(text,'from eef_arm_server_broker import validate_scope,save','from eef_mobile_server_broker import validate_scope,save')
    text=change(text,"args.condition not in ('V','SAC','script')", "args.condition != 'V-mobile'")
    text=change(text,"cap={'V':1800,'SAC':300,'script':600}[args.condition]", "cap=1800")
    text=change(text,'from bvi.eef_arm_contract import dispatch','from bvi.eef_mobile_contract import dispatch')
    text=change(text,'validate_official_controller(u.agent.controller)',
        'validate_official_controller(u.agent.controller)\n        from bvi.eef_tools_v12 import validate_base_runtime\n        validate_base_runtime(u.agent.controller,u.control_freq)')
    text=change(text,"        bind_initial(args.output.parent/'paired-initial-binding.json',report['initial_binding'])",
        "        reference=Path(auth['reference_result_root'])/args.case_id/'V/initial-binding.json'\n"
        "        if not reference.is_file():raise ValueError('missing frozen r3 initial state')\n"
        "        bind_initial(reference,report['initial_binding'])")
    text=change(text,'from eef_arm_server_broker import process,runtime_sender','from eef_mobile_server_broker import process,runtime_sender')
    text=change(text,"if call is None:actual=None;result={'accepted':False,'reason':'invalid_tool_call','steps_used':0}",
        "if call is None:\n                        actual=None;result={'accepted':False,'reason':'invalid_tool_call','steps_used':0,\n"
        "                            'parameter_error':response.get('parameter_error','invalid tool schema; check required fields'),\n"
        "                            'steps_remaining':600-ex.steps}")
    text=change(text,"choices=['V','P','SAC','script']", "choices=['V-mobile']")
    (ROOT/'scripts/run_arm_mobile_case.py').write_text(text,encoding='utf-8')

    text=(ROOT/'scripts/run_arm_capability_batch.py').read_text(encoding='utf-8')
    text=change(text,'from eef_arm_server_broker import validate_scope,save','from eef_mobile_server_broker import validate_scope,save')
    text=change(text,'from run_arm_capability_case import sha,GPU','from run_arm_mobile_case import sha,GPU')
    text=change(text,"CAPS={'V':1800,'SAC':300,'script':600}","CAPS={'V-mobile':1800}")
    text=change(text,"for condition in ('V','SAC','script')", "for condition in ('V-mobile',)")
    text=change(text,'len(jobs)!=15','len(jobs)!=5')
    text=change(text,"out.parent/'arm-capability.lock'", "out.parent/'arm-capability.lock'")
    text=change(text,'scripts/run_arm_capability_case.py','scripts/run_arm_mobile_case.py')
    text=change(text,'from summarize_arm_capability import summarize','from summarize_arm_mobile import summarize')
    text=change(text,"state.update(status='development_complete_no_test_authorization',test_gate_met=summary['test_gate_met'])",
        "state.update(status='mobile_development_complete_no_test_authorization',test_authorized=False)")
    (ROOT/'scripts/run_arm_mobile_batch.py').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
