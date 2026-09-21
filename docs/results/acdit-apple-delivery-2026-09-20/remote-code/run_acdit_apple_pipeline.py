"""One immediate, bounded training job. No scheduler, retries, or external APIs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--preflight-data',type=Path)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    root=Path.home()/'bvi-research'
    contract=json.loads((a.run/'contract.json').read_text())
    deadline=contract['deadline_unix']
    python=str(root/'envs/acdit/bin/python')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':contract['gpu_uuid'],'PYTHONHASHSEED':'20260919',
         'HF_HOME':str(root/'hf-cache'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'OMP_NUM_THREADS':'2','MS_ASSET_DIR':str(root/'assets')}
    status={'status':'starting','started_unix':time.time(),'deadline_unix':deadline,'pid':os.getpid(),
            'task':contract['task'],'scheduled_task':False,'auto_restart':False,'stages':[]}
    child=None
    def save():
        status['updated_unix']=time.time()
        target=a.run/('preflight-supervisor.json' if a.preflight_only else 'supervisor.json')
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(target)
    def terminate(signum,frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        raise RuntimeError('Supervisor received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def execute(name,args,limit):
        nonlocal child
        remaining=min(limit,deadline-time.time())
        if remaining<=0:raise TimeoutError('Overall wall budget exhausted')
        status['status']='running';status['current_stage']=name;save()
        began=time.time()
        with (a.run/f'{name}.log').open('x') as log:
            child=subprocess.Popen([python,'-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['child_pid']=child.pid;save()
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=min(20,max(0,deadline-time.time())))
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise TimeoutError(name+' exceeded frozen bound')
        status['stages'].append({'name':name,'exit_code':code,'wall_seconds':time.time()-began});save()
        if code!=0:raise RuntimeError(name+' exit '+str(code))
    trainer=str(root/'train_acdit_rdt_bounded.py')
    try:
        if a.preflight_only:
            if a.preflight_data is None:raise ValueError('Preflight data required')
            for stage in [1,2]:
                args=[trainer,'--stage',str(stage),'--data',str(a.preflight_data),
                      '--output',str(a.run/f'preflight-stage{stage}'),'--deadline-unix',str(min(deadline,time.time()+1500)),
                      '--preflight-only']
                if stage==2:args+=['--mobility',str(a.run/'preflight-stage1/preflight.pt')]
                execute(f'preflight-stage{stage}',args,1500)
            status['status']='preflight_passed';save();return
        for stage in [1,2]:
            d=json.loads((a.run/f'preflight-stage{stage}/status.json').read_text())
            if d['status']!='preflight_passed' or not d['checkpoint_roundtrip_verified']:
                raise RuntimeError('Both GPU preflights must pass')
        collection=json.loads((a.run/'data/collection.json').read_text())
        if collection['status']!='complete' and contract.get('resume_collection_seconds',0)>0:
            limit=contract['resume_collection_seconds']
            execute('collection-resume',[str(root/'collect_acdit_rdt.py'),'--resume',
                '--output',str(a.run/'data'),'--deadline-unix',str(min(deadline,time.time()+limit-30)),
                '--count',str(contract['collection']['target_successes']),
                '--min-successes',str(contract['collection']['minimum_successes']),
                '--max-attempts',str(contract['collection']['maximum_attempts']),
                '--num-envs',str(contract['collection']['num_envs'])],limit)
            collection=json.loads((a.run/'data/collection.json').read_text())
        if collection['status']!='complete':raise RuntimeError('Collection not complete')
        execute('finalize',[str(root/'finalize_acdit_rdt_data.py'),'--data',str(a.run/'data')],900)
        manifest=json.loads((a.run/'data/training-manifest.json').read_text())
        status['data']={k:v for k,v in manifest.items() if k not in ['train_parents','validation_parents','train_universal_state_mean']}
        for stage in [1,2]:
            end=min(deadline,time.time()+contract['stage1']['max_seconds']) if stage==1 else deadline
            args=[trainer,'--stage',str(stage),'--data',str(a.run/'data'),'--output',str(a.run/f'stage{stage}'),
                  '--deadline-unix',str(end)]
            if stage==2:
                prior=json.loads((a.run/'stage1/status.json').read_text())
                if prior['status']!='completed_bounded_stage' or prior['updates']<1:
                    raise RuntimeError('Stage1 did not finish')
                args+=['--mobility',prior['best_checkpoint']]
            execute(f'stage{stage}',args,end-time.time())
        status['status']='completed_training_native_evaluation_pending'
    except BaseException as exc:
        status.update(status='error',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        status['finished_or_snapshot_unix']=time.time()
        status['gpu_snapshot']=subprocess.check_output(['nvidia-smi','-i',contract['gpu_uuid'],
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
        save()


if __name__=='__main__':main()
