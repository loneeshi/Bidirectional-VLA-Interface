"""Lab CPU-only import, hash and permission gates; never constructs an env."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();auth=json.loads(args.authorization.read_text())
    digest=hashlib.sha256(args.archive.read_bytes()).hexdigest()
    assert digest==auth['deployment_zip_sha256']
    code=Path('/tmp')/('bvi-mobile-cpu-'+digest[:16]);code.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        assert all((code/name).resolve().is_relative_to(code.resolve()) for name in archive.namelist())
        archive.extractall(code)
    sys.path[:0]=[str(code/'src'),str(code/'scripts')]
    root=Path('/home/pshuai/bvi-research')
    for mapping,base in ((auth['source_sha256'],code),(auth['asset_sha256'],root)):
        for name,sha in mapping.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha,name
    import run_arm_mobile_case,run_arm_mobile_batch,summarize_arm_mobile
    from eef_mobile_server_broker import validate_scope
    try:validate_scope(auth)
    except ValueError:pass
    else:raise AssertionError('not-authorized template accepted')
    from arm_runtime_paths import configure_arm_runtime
    asset=configure_arm_runtime(root)
    import torch
    from mshab.envs.make import EnvConfig,make_env
    from bvi.eef_mobile_contract import ROBOT_SPEC,SCHEMA
    from bvi.eef_arm_executor import ArmCoordinatedV2
    from eef_server_transport import credential_metadata,validate_metadata
    meta=credential_metadata(Path.home()/'.config/bvi/openai.env');validate_metadata(meta)
    reference=Path(auth['reference_result_root'])
    for case in auth['case_ids']:
        binding=reference/case/'V/initial-binding.json'
        assert binding.is_file()
        assert hashlib.sha256(binding.read_bytes()).hexdigest()==auth['reference_binding_sha256'][case]
    assert not torch.cuda.is_initialized()
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    receipt=dict(archive_sha256=digest,source_hash_count=len(auth['source_sha256']),
        asset_hash_count=len(auth['asset_sha256']),imports_passed=True,unapproved_scope_rejected=True,
        effective_asset_directory=str(asset),cuda_initialized=False,provider_sends=0,physical_steps=0,
        credential_permissions_valid=True,credential_content_read=False,
        gpu1_idle='GPU-b7ebba23-7824-7601-df32-be55628936c3' not in gpu,
        gpu_processes=gpu.splitlines(),base_enabled='move_base' in SCHEMA['properties']['tool']['enum'],
        robot_specifications=ROBOT_SPEC,reference_binding_hashes_checked=5)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
