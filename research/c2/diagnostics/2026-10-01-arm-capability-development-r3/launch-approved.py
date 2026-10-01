"""Operator launch receipt; does not alter the frozen experiment runtime."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

root = Path('/home/pshuai/bvi-research')
deploy = root / 'deploy/arm-revision2-20261001-r3'
auth_path = deploy / 'authorization.json'
assert not auth_path.exists(), 'Already staged; inspect instead of relaunching'
shutil.copyfile('/tmp/bvi-arm-approved-r3.json', auth_path)
auth = json.loads(auth_path.read_text())
archive = Path('/tmp/bvi-arm-frozen-runtime-r3.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == auth['deployment_zip_sha256']
code = deploy / 'code'
code.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert all((code / n).resolve().is_relative_to(code.resolve()) for n in z.namelist())
    z.extractall(code)
for mapping, base in ((auth['source_sha256'], code), (auth['asset_sha256'], root)):
    for name, digest in mapping.items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
roster = deploy / 'roster.eval-only.json'
assert hashlib.sha256(roster.read_bytes()).hexdigest() == auth['roster_sha256']
sys.path[:0] = [str(code / 'src'), str(code / 'scripts')]
from eef_arm_server_broker import validate_scope
validate_scope(auth)
assert not Path(auth['output_root']).exists()
gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert 'GPU-b7ebba23-7824-7601-df32-be55628936c3' not in gpu
receipt = {'preflight_epoch': time.time(), 'source_hashes_checked': len(auth['source_sha256']),
           'asset_hashes_checked': len(auth['asset_sha256']), 'gpu1_idle': True,
           'authorization_sha256': hashlib.sha256(auth_path.read_bytes()).hexdigest()}
(deploy / 'preflight.json').write_text(json.dumps(receipt, indent=2))
command = [sys.executable, str(code / 'scripts/run_arm_capability_batch.py'),
           '--authorization', str(auth_path), '--roster', str(roster),
           '--code', str(code), '--lab-root', str(root)]
with (deploy / 'batch-launch.log').open('x') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(code / 'src')),
                               start_new_session=True, cwd=root)
receipt.update(pid=process.pid, launched_at_epoch=time.time(), command=command)
(deploy / 'launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
