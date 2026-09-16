"""Install the pinned AC-DiT Python dependencies without using a GPU.

No sudo, system package changes, inference, data downloads or training. Native
extensions (PyTorch3D) and model import checks are a separate deployment gate.
"""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    root = Path.home()/'bvi-research'
    source = root/'src/AC-DiT'
    uv, python = root/'bootstrap/bin/uv', root/'envs/acdit/bin/python'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-12.8',
               UV_CACHE_DIR=str(root/'cache/uv'), MAX_JOBS='2', DS_BUILD_OPS='0',
               TORCH_CUDA_ARCH_LIST='7.5')
    env['PATH'] = str(python.parent)+':/usr/local/cuda-12.8/bin:'+env['PATH']
    def run(args):
        print(json.dumps({'command':[str(x) for x in args]}), flush=True)
        subprocess.run([str(x) for x in args], env=env, check=True, timeout=600)
    install = [uv, 'pip', 'install', '--python', python]
    run(install+['torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0',
                 '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run(install+['setuptools==69.5.1', 'wheel', 'pip', 'ninja'])
    run(install+['-r', source/'requirements.txt'])
    for path in ['third_party/ManiSkill', 'third_party/mshab', 'third_party/LIFT3D']:
        run(install+['--no-deps', '--no-build-isolation', '-e', source/path])
    packages = subprocess.check_output([str(uv),'pip','freeze','--python',str(python)],env=env,text=True)
    (root/'base-package-freeze.txt').write_text(packages)
    result = {'completed_at':datetime.now(timezone.utc).isoformat(),
              'python_dependencies_installed':True,'gpu_used':False,
              'pytorch3d_installed':False,'strict_model_load_verified':False,
              'training_started':False}
    (root/'base-install-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
