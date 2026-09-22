"""Run a preserved paired 16-plan official PPO/SAC study, fixed then GPT per plan."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from bvi.coordinator import normalize_repeated_response
from run_sac_interface_baseline import classify, runtime_environment, OFFICIAL_MSHAB


def select_plans(manifest):
    seen, rows = set(), []
    for row in manifest['episodes'][:20]:
        if row['plan_uid'] in seen:
            continue
        if not row.get('plan_uid') or len(row['binding']['subtasks']) != 20:
            raise ValueError('Missing real five-object task binding')
        seen.add(row['plan_uid'])
        rows.append(dict(seed=row['seed'], plan_uid=row['plan_uid']))
    if len(rows) != 16:
        raise ValueError('Expected exactly 16 unique historical plans')
    return rows


def command(row, arm, output, checkpoints, bridge, authorization, initial_hash=None, goal_tools=False,
            feedback_mode='evaluator', feedback_profile='raw_v0', hide_feedback=()):
    argv = [sys.executable, str(Path(__file__).with_name('run_coordinator.py')),
        '--paired-ppo-episode', '--seed', str(row['seed']), '--expected-plan-uid', row['plan_uid'],
        '--checkpoint-root', str(checkpoints), '--policy-type', 'rl_per_obj',
        '--navigation-policy', 'official', '--manipulation-policy', 'official',
        '--navigation-camera', 'fetch_nav', '--workspace-camera',
        '--max-env-steps', '7000', '--max-wall-seconds', '900', '--skill-wall-seconds', '180',
        '--organizer-slice-steps', '40', '--output', str(output)]
    if arm == 'fixed':
        argv += ['--dry-run', '--max-calls', '175']
    elif arm == 'gpt':
        if not initial_hash:
            raise ValueError('GPT arm requires paired initial state hash')
        argv += ['--organizer', '--max-calls', '40', '--provider', 'openai', '--model', 'gpt-5.6-luna',
            '--transport', 'bridge', '--bridge-dir', str(bridge), '--bridge-timeout-seconds', '120',
            '--authorization-id', authorization, '--max-api-cost-usd', '0.05',
            '--request-cost-ceiling-usd', '0.00125', '--max-output-tokens', '2048',
            '--max-input-bytes', '512000', '--expected-initial-state-sha256', initial_hash]
        if goal_tools:
            argv += ['--goal-tools']
            if feedback_mode == 'continuous_progress':
                argv += ['--progress-feedback']
            elif feedback_mode != 'evaluator':
                raise ValueError('Unknown feedback mode')
            if feedback_profile != 'raw_v0':
                argv += ['--feedback-profile', feedback_profile]
            if hide_feedback:
                argv += ['--hide-feedback', *sorted(hide_feedback)]
    else:
        raise ValueError('Unknown arm')
    return argv


def summary(state):
    out = {}
    for arm in ('fixed','gpt'):
        rows = [r for r in state['episodes'] if r['arm'] == arm]
        completed = [r for r in rows if r['status'] == 'completed']
        out[arm] = dict(planned=16, completed=len(completed),
            infrastructure_failed=sum(r['status']=='infrastructure_failure' for r in rows),
            not_run=sum(r['status']=='not_run' for r in rows),
            running=sum(r['status']=='running' for r in rows),
            successes=sum(r.get('result',{}).get('task_success',False) for r in completed),
            completed_objects=sum(r.get('result',{}).get('completed_objects',0) for r in completed),
            api_requests=sum(a.get('result',{}).get('api_requests',0) for r in rows for a in r['attempts']))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--bridge-dir',type=Path,required=True)
    p.add_argument('--authorization-id',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--goal-tools',action='store_true')
    p.add_argument('--feedback-mode', choices=['evaluator','continuous_progress'], default='evaluator')
    p.add_argument('--feedback-profile', choices=['raw_v0','object_v1','object_trajectory_v1'], default='raw_v0')
    p.add_argument('--hide-feedback', nargs='+', default=[], choices=['images','trajectory','progress','structured_goals'])
    p.add_argument('--retry-infrastructure',action='store_true',help='Preserve failed attempts and retry them after a verified repair')
    p.add_argument('--repair-duplicate-seeds',type=int,nargs='+',
                   help='Retry only these GPT rows after verifying a duplicated-response failure; preserve prior attempts')
    p.add_argument('--max-new-per-arm',type=int,default=2,
                   help='Process at most this many new fixed and GPT rows, then checkpoint the chunk')
    a=p.parse_args()
    if not a.goal_tools and (a.feedback_mode != 'evaluator' or a.feedback_profile != 'raw_v0' or a.hide_feedback):
        p.error('Feedback variations require --goal-tools')
    selected=select_plans(json.loads(a.source_manifest.read_text()))
    if a.max_new_per_arm is not None and a.max_new_per_arm < 1:
        p.error('--max-new-per-arm must be positive')
    if not a.execute:
        print(json.dumps(dict(plans=selected,planned_per_arm=16,new_api_cap=640),indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'runner.lock'
    # A lock survives a crash: verify old PID and explicitly archive the lock before resuming.
    with lock.open('x') as f:f.write(str(os.getpid()))
    path=a.output/'panel-status.json'
    state=json.loads(path.read_text()) if path.exists() else dict(schema='ppo-sac-paired16/1',
        status='running', source_manifest=str(a.source_manifest),
        episodes=[dict(**row,arm=arm,status='not_run',attempts=[]) for row in selected for arm in ('fixed','gpt')])
    if path.exists() and state.get('goal_tools',False) != a.goal_tools:
        lock.unlink()
        raise ValueError('Changed planning interface requires a new batch directory')
    presentation = dict(mode=a.feedback_mode, profile=a.feedback_profile, hide=sorted(a.hide_feedback))
    legacy = dict(mode='evaluator', profile='raw_v0', hide=[])
    if any('--progress-feedback' in attempt.get('argv', [])
           for row in state['episodes'] for attempt in row.get('attempts', [])):
        legacy['mode'] = 'continuous_progress'
    if path.exists() and state.get('feedback_presentation', legacy) != presentation:
        lock.unlink()
        raise ValueError('Changed feedback setting requires a new batch directory')
    state['feedback_presentation'] = presentation
    env=runtime_environment(OFFICIAL_MSHAB)
    env['MS_ASSET_DIR']='/home/pshuai/bvi-research/assets'
    env['PYTHONHASHSEED']='0'
    env['BVI_DEBUG_ADAPTER_EXCEPTION']='1'
    started=time.monotonic()
    previous_wall=sum(max(0, x.get('finished_unix',x['started_unix'])-x['started_unix'])
                      for row in state['episodes'] for x in row['attempts'])
    state['status']='running'
    state['goal_tools']=a.goal_tools
    started_by_arm={'fixed':0,'gpt':0}
    def save():
        state['summary']=summary(state)
        atomic_json(path,state)
    save()
    try:
        for row in state['episodes']:
            repair = False
            if a.repair_duplicate_seeds is not None:
                if row['arm'] != 'gpt' or row['seed'] not in a.repair_duplicate_seeds:
                    continue
                if row.get('result',{}).get('reason') != 'error:ModelResponseError':
                    continue
                old_attempt = row['attempts'][-1]
                events = [json.loads(line) for line in
                          (Path(old_attempt['directory'])/'events.jsonl').read_text().splitlines()]
                usage = [e for e in events if e.get('event') == 'api_usage']
                if not usage or normalize_repeated_response(usage[-1].get('response_text',''))[1] < 2:
                    raise ValueError('Repair requires recorded identical duplicated response')
                repair = True
            if row['status']=='completed' and not repair:continue
            if row['status']=='infrastructure_failure' and not a.retry_infrastructure:continue
            if a.max_new_per_arm is not None and started_by_arm[row['arm']] >= a.max_new_per_arm:
                continue
            if previous_wall+time.monotonic()-started > 33000:
                state['status']='stopped_outer_wall_budget';break
            old=row['attempts']
            if repair:
                old[-1].setdefault('validity_adjudications',[]).append(dict(
                    classification='excluded_response_integrity',
                    reason='identical_json_repetition',time_unix=time.time(),
                    original_status=old[-1]['status']))
            for attempt in old:
                if attempt['status']=='running':attempt['status']='interrupted'
            dest=a.output/row['arm']/f"seed-{row['seed']:03d}"/f'attempt-{len(old)+1:03d}'
            dest.parent.mkdir(parents=True,exist_ok=True)
            initial_hash=None
            if row['arm']=='gpt':
                fixed=next(r for r in state['episodes'] if r['arm']=='fixed' and r['seed']==row['seed'])
                if fixed['status']!='completed':
                    state['status']='blocked_fixed_infrastructure';break
                initial=json.loads((Path(fixed['attempts'][-1]['directory'])/'initial-state.json').read_text())
                initial_hash=initial['state_sha256']
            argv=command(row,row['arm'],dest,a.checkpoint_root,a.bridge_dir,a.authorization_id,initial_hash,a.goal_tools,
                         a.feedback_mode,a.feedback_profile,a.hide_feedback)
            attempt=dict(directory=str(dest),argv=argv,status='running',started_unix=time.time())
            old.append(attempt);row['status']='running';save()
            started_by_arm[row['arm']]+=1
            with dest.with_suffix('.log').open('w') as log:
                child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                attempt['pid']=child.pid;save()
                try:code=child.wait(timeout=990)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            raw=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else None
            status,result=classify(code,raw)
            row.update(status=status,result=result)
            attempt.update(status=status,result=result,finished_unix=time.time())
            save()
            if status=='infrastructure_failure':
                state['status']='stopped_infrastructure';break
        else:
            state['status']=('finished' if all(r['status']=='completed' for r in state['episodes'])
                             else 'chunk_complete')
    except BaseException as exc:
        state.update(status='interrupted',error_type=type(exc).__name__)
        raise
    finally:
        save()
        lock.unlink()


if __name__=='__main__':main()
