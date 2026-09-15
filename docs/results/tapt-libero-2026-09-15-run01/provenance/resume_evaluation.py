"""Continue untouched episodes after a recorded schema rejection; no retries."""
import datetime,json,os,pathlib,socket,subprocess,time
root=pathlib.Path('/workspace/tapt');e=root/'evidence';python=str(root/'author/.venv/bin/python');sim=str(root/'sim-env/bin/python');model='/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero'
env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONPATH='/workspace/tapt:/workspace/tapt/author/third_party/libero')
summary=json.loads((e/'vlm-standard/summary.json').read_text());assert len(summary)==3 and summary[2]['error']=="ValueError('Instruction too long')"
count=sum(json.loads(line)['event']=='prediction' for file in (e/'vlm-standard').glob('episode*.jsonl') for line in file.read_text().splitlines());assert count==158
(e/'evaluation-resumption.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retained_standard_episodes':[0,1,2],'next_standard_episode':3,'restored_policy_rng_splits':count,'failure':'160 Unicode characters consumed162 UTF8 bytes; request rejected, episode retained as failure','unchanged':'Prompts, byte limit, model selection, task, states, budgets, and progress thresholds','deviation':'Model service restarted; RNG advanced by exactly logged predictions; environment reset sequence advanced without policy replay. Controller continues after instruction-length failure, but not unknown infrastructure errors.'},indent=2))
def execute(mode,args,resume=False):
 label=mode+('-resume' if resume else '')
 print('START',label,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 with (e/('serve-'+label+'.log')).open('w') as log:
  proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   for _ in range(180):
    if proc.poll() is not None:raise RuntimeError('server exited')
    try:
     with socket.create_connection(('127.0.0.1',8000),timeout=1):break
    except OSError:time.sleep(2)
   else:raise TimeoutError('server readiness')
   cmd=[sim,str(root/'eval_libero_family.py'),'--mode',mode,'--output',str(e/('vlm-'+mode))]
   if resume:cmd.append('--resume')
   with (e/('eval-'+label+'.log')).open('w') as output:subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
  finally:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 print('DONE',label,flush=True)
execute('standard',[python,str(root/'serve_libero_baseline.py'),'--skip-predictions',str(count)],True)
selected=json.loads((e/'evaluation-lock.json').read_text())['selected']['path']
execute('tapt',[python,str(root/'serve_libero_family.py'),'--checkpoint',model,'--adapters',selected,'--audit-data',str(root/'data')])
print('PIPELINE COMPLETE',flush=True)
