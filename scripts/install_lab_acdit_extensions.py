"""Compile AC-DiT's PyTorch3D dependency for Turing, using two CPU workers."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

PYTORCH3D_REV = '33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'  # v0.7.9


def main():
    root = Path.home()/'bvi-research'
    python, uv = root/'envs/acdit/bin/python', root/'bootstrap/bin/uv'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-12.8',
               TORCH_CUDA_ARCH_LIST='7.5', FORCE_CUDA='1', MAX_JOBS='2',
               UV_CACHE_DIR=str(root/'cache/uv'))
    env['PATH'] = str(python.parent)+':/usr/local/cuda-12.8/bin:'+env['PATH']
    install = [str(uv),'pip','install','--python',str(python)]
    subprocess.run(install+['iopath==0.1.10','fvcore==0.1.5.post20221221'],env=env,check=True,timeout=180)
    subprocess.run(install+['--no-build-isolation','--no-deps',
                           'git+https://github.com/facebookresearch/pytorch3d.git@'+PYTORCH3D_REV],
                   env=env,check=True,timeout=900)
    result={'completed_at':datetime.now(timezone.utc).isoformat(),'pytorch3d_revision':PYTORCH3D_REV,
            'gpu_used':False,'arch':'7.5','max_cpu_build_workers':2,'training_started':False}
    (root/'extensions-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
