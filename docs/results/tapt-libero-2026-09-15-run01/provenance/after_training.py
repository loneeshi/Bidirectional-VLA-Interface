"""Execute already authorized phases sequentially; never overlap GPU models."""
import pathlib,subprocess,os,time,json,socket,datetime
root=pathlib.Path('/workspace/tapt');e=root/'evidence';python=str(root/'author/.venv/bin/python');sim=str(root/'sim-env/bin/python');model='/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero'
env=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONPATH='/workspace/tapt:/workspace/tapt/author/third_party/libero')
def run(args,name,timeout):
 print('START',name,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 with (e/(name+'.log')).open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
 print('DONE',name,flush=True)
def serve_eval(server_args,mode):
 with (e/('serve-'+mode+'.log')).open('w') as log:
  proc=subprocess.Popen(server_args,stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   for _ in range(180):
    if proc.poll() is not None:raise RuntimeError(mode+' server exited')
    try:
     with socket.create_connection(('127.0.0.1',8000),timeout=1):break
    except OSError:time.sleep(2)
   else:raise TimeoutError(mode+' server readiness')
   run([sim,str(root/'eval_libero_family.py'),'--mode',mode,'--output',str(e/('vlm-'+mode))],'eval-'+mode,1800)
  finally:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
while True:
 text=(e/'training-v2.log').read_text()
 if 'TRAIN COMPLETE' in text:
  if pathlib.Path('/proc/10595').exists():time.sleep(2);continue
  break
 if not pathlib.Path('/proc/10595').exists():raise RuntimeError('Training stopped before completion; review checkpoints')
 time.sleep(30)
run([python,str(root/'validate_libero_family.py'),'--checkpoint',model,'--training',str(e/'training-v2'),'--data',str(root/'data')],'validation-selection',1800)
selection=json.loads((e/'training-v2/validation-selection.json').read_text());selected=pathlib.Path(selection['selected']['path'])
(e/'evaluation-lock.json').write_text(json.dumps({'selected':selection['selected'],'adapter_metadata':json.loads((selected/'metadata.json').read_text()),'locked_before_gpt_evaluation_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
serve_eval([python,str(root/'serve_baseline.py')],'standard')
serve_eval([python,str(root/'serve_libero_family.py'),'--checkpoint',model,'--adapters',str(selected),'--audit-data',str(root/'data')],'tapt')
print('PIPELINE COMPLETE',flush=True)
