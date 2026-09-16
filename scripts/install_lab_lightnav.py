"""Prepare an isolated, pinned LightNav HF environment without touching GPUs."""
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path.home() / 'bvi-research'
    uv = root / 'bootstrap/bin/uv'
    source = root / 'src/LightNav-0'
    python = root / 'envs/lightnav/bin/python'
    commit = 'c6f40e3220edbf7011e4f17eaf2c865416737d4d'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'UV_CACHE_DIR': str(root / 'cache/uv'),
           'UV_NO_PROGRESS': '1', 'OMP_NUM_THREADS': '2'}
    def run(args):
        print(json.dumps({'command': [str(x) for x in args]}), flush=True)
        subprocess.run([str(x) for x in args], env=env, check=True, timeout=900)
    if not source.exists():
        run(['git', 'clone', 'https://github.com/lightorigins/LightNav-0.git', source])
        run(['git', '-C', source, 'checkout', '--detach', commit])
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
    if not python.exists():
        run([uv, 'venv', '--python', root / 'envs/acdit/bin/python', python.parent.parent])
    install = [uv, 'pip', 'install', '--python', python]
    run(install + ['torch==2.10.0', 'torchvision==0.25.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run(install + ['-e', source, 'pytest==8.3.5', 'pytest-asyncio==0.24.0'])
    (root / 'lightnav-package-freeze.txt').write_text(subprocess.check_output(
        [str(uv), 'pip', 'freeze', '--python', str(python)], env=env, text=True))
    run([python, '-c', "from huggingface_hub import snapshot_download; snapshot_download('LightOriginsHQ/LightNav-0', revision='826dc5fbfa37afa8293d2e336d329b6ffc0bfb64', local_dir='" + str(root / 'checkpoints/lightnav') + "', max_workers=2)"])
    (root / 'lightnav-install.json').write_text(json.dumps({'status': 'installed_not_inference_verified',
        'source_commit': commit, 'checkpoint_revision': '826dc5fbfa37afa8293d2e336d329b6ffc0bfb64',
        'gpu_used': False, 'backend': 'hf', 'native_bf16_unavailable': True}, indent=2)+'\n')


if __name__ == '__main__':
    main()
