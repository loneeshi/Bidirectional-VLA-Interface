"""Audit saved first-object control evidence; does not execute simulation or APIs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def audit(directory):
    directory=Path(directory)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    summary=json.loads((directory/'summary.json').read_text())
    metadata=json.loads((directory/'run-metadata.json').read_text())
    counts=Counter(e['event'] for e in events)
    resets=[e for e in events if e['event']=='mshab_reset']
    steps=[e for e in events if e['event']=='mshab_step']
    carry=[e for e in steps if e['subtask_before']==2]
    def yes(info,key):return info.get(key)==[True]
    def valid_action(e):
        a=e.get('controller_action')
        return isinstance(a,list) and len(a)==1 and len(a[0])==13 and all(
            isinstance(v,(int,float)) and math.isfinite(v) and abs(v)<=1.00001 for v in a[0])
    skills=summary.get('skill_results',[])
    checks={
        'one_reset_no_auto_reset':len(resets)==1 and resets[0].get('auto_reset') is False,
        'continuous_step_sequence':bool(steps) and [e['info']['elapsed_steps'][0] for e in steps]==list(range(1,len(steps)+1)),
        'step_count_matches_summary':len(steps)==summary.get('steps'),
        'all_actions_finite_normalized_fetch13':bool(steps) and all(valid_action(e) for e in steps),
        'four_successful_skills':[s['skill'] for s in skills]==['navigate','pick','navigate','place'] and all(s['feedback']['status']=='succeeded' for s in skills),
        'grasp_maintained_through_carry':bool(carry) and all(yes(e['info'],'is_grasped') for e in carry),
        'place_goal_and_release':bool(steps) and steps[-1]['subtask_before']==3 and steps[-1]['subtask_after']==4 and yes(steps[-1]['info'],'obj_at_goal') and steps[-1]['info'].get('is_grasped')==[False],
        'no_reported_benchmark_failure':bool(steps) and all(e['info'].get('fail')==[False] for e in steps),
        # Historical A3 predates this summary field; compute success from events.
        'summary_does_not_contradict_diagnostic':summary.get('first_object_chain_success') in (None,True) and summary.get('benchmark_result') is False,
    }
    if metadata.get('navigation_policy')=='lightnav':
        checks['real_lightnav_predictions']=counts['navigation_prediction']>0
        checks['no_official_navigation_actions']=not any(e['event']=='policy_action' and e['skill']=='navigate' for e in events)
    if metadata.get('manipulation_policy')=='fetch-pi05':
        checks['real_pi_predictions']=counts['fetch_pi_prediction']>0
        checks['no_official_manipulation_actions']=not any(e['event']=='policy_action' and e['skill'] in ('pick','place') for e in events)
        starts=[e for e in events if e['event']=='fetch_pi_started']
        checks['state_conditioned_fetch_metadata']=len(starts)==2 and all(e['model_metadata'].get('state_conditioning') is True and e['model_metadata'].get('robot')=='fetch' for e in starts)
    reset=resets[0] if resets else {}
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'frames').glob('seed-*-step-0-*.png')}
    return dict(run=directory.name,automated_evidence_pass=all(checks.values()),checks=checks,
        visual_inspection_required=True,summary_first_object_field=summary.get('first_object_chain_success'),carry_steps=len(carry),event_counts=dict(counts),
        initial_images_sha256=fingerprints,seed=reset.get('seed'),
        scene=reset.get('task_plan',{}).get('build_config_name'),
        initial_config=reset.get('task_plan',{}).get('init_config_name'),
        events_sha256=hashlib.sha256((directory/'events.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=audit(args.run);text=json.dumps(result,indent=2)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)
    raise SystemExit(0 if result['automated_evidence_pass'] else 1)
