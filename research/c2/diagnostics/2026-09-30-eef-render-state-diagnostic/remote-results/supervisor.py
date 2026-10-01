import subprocess,os,time,json,signal
from pathlib import Path
r=Path('/home/pshuai/bvi-research/runs/eef-render-diagnostic-20260930-r1')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-b7ebba23-7824-7601-df32-be55628936c3','MS_ASSET_DIR':'/home/pshuai/bvi-research/assets','PYTHONPATH':str(r/'code/src')+':'+str(r/'code/scripts')}
cmd=['/home/pshuai/bvi-research/envs/acdit/bin/python',str(r/'code/scripts/diagnose_eef_render_state.py'),'--authorization',str(r/'authorization.json'),'--manifest',str(r/'manifest.eval-only.json'),'--code',str(r/'code'),'--census-root','/home/pshuai/bvi-research/runs/real-handoff-pick-census-20260925','--urdf',str(r/'fetch.urdf'),'--position','5','--output',str(r/'result')]
start=time.monotonic()
with (r/'diagnostic.log').open('w') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:p.wait(timeout=114)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=2)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
usage={'pid':p.pid,'exit_code':p.returncode,'process_seconds':time.monotonic()-start,'gpu_final':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)}
(r/'usage.json').write_text(json.dumps(usage))
