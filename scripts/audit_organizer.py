"""Audit a VLM-organized first-object chain; does not establish planning benefit."""
import argparse,json
from pathlib import Path

def audit(path):
    es=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    meta=json.loads((path/'run-metadata.json').read_text());summary=json.loads((path/'summary.json').read_text())
    by=lambda name:[e for e in es if e['event']==name]
    starts=by('api_request_started');decisions=by('coordinator_decision');steps=by('mshab_step');finished=by('skill_finished')
    applied={e['call_id'] for e in by('skill_step')};chosen={e['request']['call_id'] for e in decisions}
    advances=[e for e in steps if e['subtask_after']>e['subtask_before']]
    carry=[e for e in steps if e['subtask_before']==2]
    checks=dict(live_vlm=meta.get('vlm') is True,organizer_enabled=meta.get('organizer',{}).get('enabled') is True,
        no_oracle_decisions=not by('oracle_protocol_decision'),single_reset=len(by('mshab_reset'))==1,
        real_images=bool(starts) and all(e.get('images') and all(x.get('bytes',0)>0 for x in e['images']) for e in starts),
        executed_calls_selected_by_vlm=bool(applied) and applied<=chosen,
        first_four_native_advances=[e['subtask_before'] for e in advances[:4]]==[0,1,2,3],
        stable_carry=bool(carry) and all(e['info'].get('is_grasped')==[True] for e in carry),
        declared_boundary=meta.get('stop_after_subtasks')==4 and bool(steps) and steps[-1]['subtask_after']==4,
        no_native_failure=not any(e['info'].get('fail')==[True] for e in steps),
        feedback_reaches_next_request=any(json.loads(e['prompt']).get('feedback_history') for e in starts[1:]),
        not_full_benchmark=meta.get('benchmark_result') is False)
    interrupted=[i for i,e in enumerate(finished) if e['status']=='interrupted']
    recovery=any(any(e['status']=='succeeded' and e['feedback'].get('reason')=='oracle_subtask_advanced' for e in finished[i+1:]) for i in interrupted)
    return dict(checks=checks,first_object_engineering_pass=all(checks.values()),api_requests=len(starts),decisions=len(decisions),physics_steps=len(steps),skill_invocations=len(finished),yield_reasons=[e['feedback'].get('reason') for e in finished if e['status'] in ('timed_out','interrupted')],abort_requested=bool(by('organizer_abort')),synthetic_fault=meta.get('synthetic_closure_fault',False),injected_steps=len(by('injected_closure_fault')),interruption_followed_by_native_success=recovery,visual_inspection_required=True,free_task_order_planning=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=audit(a.run);s=json.dumps(r,indent=2)
    if a.output:a.output.write_text(s)
    print(s)
