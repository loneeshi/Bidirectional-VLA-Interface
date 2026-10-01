"""Extract this task's source snapshot in /tmp and import it without physics."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
import argparse

os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=Path('/tmp/bvi-arm-frozen-runtime.zip'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/bvi-arm-deployment-cpu.json'))
    args=parser.parse_args()
    archive=args.archive
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    destination=Path('/tmp')/('bvi-arm-cpu-'+digest[:16]);destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination/name).resolve().is_relative_to(destination.resolve()):raise ValueError('unsafe archive path')
        z.extractall(destination)
    sys.path[:0]=[str(destination/'src'),str(destination/'scripts')]
    import run_arm_capability_case, run_arm_capability_batch, finalize_arm_capability_attempt
    from eef_arm_server_broker import validate_scope
    try:validate_scope({'status':'not_authorized'})
    except ValueError:pass
    else:raise AssertionError('unapproved batch accepted')
    from arm_runtime_paths import configure_arm_runtime,validate_asset_directory
    lab_root=Path('/home/pshuai/bvi-research')
    asset_dir=configure_arm_runtime(lab_root)
    try:validate_asset_directory(lab_root,asset_dir/'data')
    except ValueError:pass
    else:raise AssertionError('duplicate data path accepted')
    import torch
    from mshab.envs.make import EnvConfig,make_env
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    from export_c2_collision_bundle import _component_and_shapes
    from bvi.eef_arm_sensing import observe
    from bvi.eef_attempt_recording import AttemptRecording
    from bvi.eef_arm_executor import ArmCoordinatedV2
    from bvi.eef_arm_localization import diagnose_location
    assert diagnose_location({'valid':False},None,None,None)['surface_distance_m'] is None
    from bvi.mshab_adapter import load_rl_policy
    from dataclasses import fields
    assert {'obs_mode','stationary_base','stationary_torso','stationary_head'}<={f.name for f in fields(EnvConfig)}
    assert not torch.cuda.is_initialized()
    receipt={'archive_sha256':digest,'source_files':len(list(destination.rglob('*.py'))),
        'effective_asset_directory':str(asset_dir),'asset_directory_exists':asset_dir.is_dir(),
        'duplicate_data_path_rejected':True,
        'imports_passed':True,'unapproved_scope_rejected':True,'cuda_initialized':False,
        'simulator_steps':0,'provider_sends':0,'physical_runtime_checked':False,'python':sys.version}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
