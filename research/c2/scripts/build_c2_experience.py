"""Freeze three development plans and source-isolated historical SAC cases, CPU-only.

Uses saved camera frames at actual invocation boundaries. Does not invent joint
states or derive them from rendered videos. Legacy 40-action slices are labelled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(panel_path, output):
    panel=json.loads(panel_path.read_text())
    rows=sorted([r for r in panel['episodes'] if r.get('arm')=='gpt' and r['status']=='completed'],
                key=lambda r:r['seed'])
    if len(rows)!=16:raise ValueError('Expected complete historical sixteen-plan GPT panel')
    chosen=rows[:3]; excluded={r['plan_uid'] for r in chosen}
    if len(excluded)!=3:raise ValueError('Nonunique development plans')
    output.mkdir(parents=True,exist_ok=False)
    episodes=[]
    for r in chosen:
        attempt=Path(r['attempts'][-1]['directory']);initial=json.loads((attempt/'initial-state.json').read_text())
        if initial['plan_uid']!=r['plan_uid'] or initial['seed']!=r['seed']:
            raise ValueError('Historical initial-state binding mismatch')
        episodes.append(dict(seed=r['seed'],plan_uid=r['plan_uid'],
            initial_state_sha256=initial['state_sha256'],source_initial_state_sha256=sha(attempt/'initial-state.json'),
            scene=initial['task_plan'].get('build_config_name'),
            selection_reason='first_three_complete_historical_plans_by_seed_not_selected_by_new_outcomes'))
    (output/'manifest.json').write_text(json.dumps({'schema':'c2-development/1','episodes':episodes,
        'source_panel_sha256':sha(panel_path)},indent=2))
    cases=[]; missing=[]
    for row in rows:
        if row['plan_uid'] in excluded:continue
        # Stable last completed attempt, not selected by success; other attempts remain archived.
        directory=Path(row['attempts'][-1]['directory']);events_path=directory/'events.jsonl'
        events=[json.loads(line) for line in events_path.read_text().splitlines()]
        reset=next(e for e in events if e['event']=='mshab_reset')
        if reset['task_plan']['subtasks'][0]['uid'] != row['plan_uid']:raise ValueError('Parent mismatch')
        categories={}
        for task in reset['task_plan']['subtasks']:
            if task['type']=='pick':
                key=hashlib.sha256(task['obj_id'].encode()).hexdigest()[:12]
                for prefix in ('object-','destination-'):categories[prefix+key]=task['obj_id'].split('-')[0]
        policies={}
        for e in events:
            if e['event']=='policy_loaded':
                cp=Path(e['checkpoint']); policies[(cp.parent.parent.name,cp.parent.name)]=e['checkpoint_sha256']
        starts={e['request']['call_id']:e['request'] for e in events if e['event']=='skill_started'}
        endings=[e for e in events if e['event']=='skill_finished']
        for end in endings:
            req=starts.get(end['call_id'])
            if req is None or req['skill'] not in ('pick','place') or end['steps']==0:continue
            category=categories[req['target_id']]
            checkpoint=policies.get((req['skill'],category))
            if not checkpoint:
                missing.append(dict(parent_uid=row['plan_uid'],call_id=end['call_id'],reason='missing_checkpoint_hash'));continue
            identity=hashlib.sha256((row['plan_uid']+str(directory)+end['call_id']).encode()).hexdigest()[:20]
            frames=[]
            for event,frame_id in [('start',req['observation_id']),('end',end['frame_id'])]:
                src=directory/'frames'/f'{frame_id}-fetch_workspace.png'
                if not src.is_file():continue
                match=re.fullmatch(r'seed-\d+-step-(\d+)',frame_id)
                if not match:raise ValueError('Unknown frame clock')
                dest=output/'frames'/identity/f'{event}.png';dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dest)
                frames.append(dict(step=int(match[1]),event=event,camera='fetch_workspace',
                                   path=dest.relative_to(output).as_posix(),sha256=sha(dest)))
            if not frames:
                missing.append(dict(parent_uid=row['plan_uid'],call_id=end['call_id'],reason='missing_source_frames'));continue
            cases.append(dict(parent_uid=row['plan_uid'],call_id=end['call_id'],skill=req['skill'],
                object_category=category,checkpoint_sha256=checkpoint,source_sha256=sha(events_path),
                source_directory=str(directory),source_protocol='historical_oracle_feedback_bounded_slice',
                invocation_horizon=req['max_steps'],
                start_state={'available':False,'missing_reason':'historical_log_has_no_measured_call_start_pose'},
                result={'status':end['status'],'feedback':end['feedback'],'steps':end['steps']},frames=frames))
    payload={'schema':'c2-experience/1','excluded_parent_uids':sorted(excluded),
             'selection':'same skill/category/checkpoint then distance if available; tie by parent/call; never outcome',
             'cases':cases,'missing':missing}
    (output/'cases.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(episodes=episodes,cases=len(cases),missing=len(missing))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.panel,a.output)
