"""Operator entry: only run after explicit approval of the new USD65 scope."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('--authorization-evidence',required=True)
args=parser.parse_args()
root=Path('/home/pshuai/bvi-research')
template=Path('/tmp/bvi-mobile-authorization.template.json')
auth=json.loads(template.read_text())
assert auth['status']=='not_authorized' and auth['condition']=='V-mobile'
assert auth['usd_cap']=='65' and auth['gpu1_process_seconds_cap']==9000
assert args.authorization_evidence.strip()
deploy=root/'deploy/arm-mobile-five-20261001-r1'
assert not deploy.exists(), 'Existing deployment: inspect; do not relaunch'
assert not Path(auth['output_root']).exists()
archive=Path('/tmp/bvi-mobile-frozen-runtime.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==auth['deployment_zip_sha256']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'GPU-b7ebba23-7824-7601-df32-be55628936c3' not in gpu
deploy.mkdir();code=deploy/'code';code.mkdir()
with zipfile.ZipFile(archive) as source:
    assert all((code/name).resolve().is_relative_to(code.resolve()) for name in source.namelist())
    source.extractall(code)
for mapping,base in ((auth['source_sha256'],code),(auth['asset_sha256'],root)):
    for name,digest in mapping.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
roster=deploy/'roster.eval-only.json';shutil.copyfile('/tmp/bvi-mobile-roster.eval-only.json',roster)
assert hashlib.sha256(roster.read_bytes()).hexdigest()==auth['roster_sha256']
for case,digest in auth['reference_binding_sha256'].items():
    assert hashlib.sha256((Path(auth['reference_result_root'])/case/'V/initial-binding.json').read_bytes()).hexdigest()==digest
auth.update(status='authorized',authorization_evidence=args.authorization_evidence,
            expires_at_epoch=time.time()+6*3600)
auth_path=deploy/'authorization.json';auth_path.write_text(json.dumps(auth,indent=2)+'\n')
sys.path[:0]=[str(code/'src'),str(code/'scripts')]
from eef_mobile_server_broker import validate_scope
validate_scope(auth)
command=[sys.executable,str(code/'scripts/run_arm_mobile_batch.py'),
    '--authorization',str(auth_path),'--roster',str(roster),'--code',str(code),'--lab-root',str(root)]
with (deploy/'batch-launch.log').open('x') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(code/'src')))
receipt=dict(pid=process.pid,launched_at_epoch=time.time(),command=command,gpu1_idle_preflight=True,
             authorization_sha256=hashlib.sha256(auth_path.read_bytes()).hexdigest())
(deploy/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
