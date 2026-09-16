"""User-local, CPU-only lab bootstrap. Run on Linux; no sudo or GPU work.

Creates a dedicated Python and source checkouts. Heavy dependencies, datasets,
model inference and training are separate gates. Existing checkouts must match.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ACDIT_REV = '90ad00a926f34da04816ed9c3312aaf3bc845b7f'
BVI_REV = '21cddca'


def run(args, env=None):
    print(json.dumps({'command': [str(x) for x in args]}), flush=True)
    subprocess.run([str(x) for x in args], check=True, env=env, timeout=300)


def main():
    root = Path.home() / 'bvi-research'
    if root.is_symlink():
        raise RuntimeError('Refusing symlinked workspace root')
    root.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', CUDA_HOME='/usr/local/cuda-12.8',
               UV_PYTHON_INSTALL_DIR=str(root/'python'), UV_CACHE_DIR=str(root/'cache/uv'))
    env['PATH'] = '/usr/local/cuda-12.8/bin:' + env['PATH']
    bootstrap = root/'bootstrap'
    uv = bootstrap/'bin/uv'
    if not uv.exists():
        run([sys.executable, '-m', 'pip', 'install', '--no-warn-script-location',
             '--target', bootstrap, 'uv==0.8.22'], env)
    run([uv, '--version'], env)
    run([uv, 'python', 'install', '3.11.13'], env)
    venv = root/'envs/acdit'
    if not venv.exists():
        run([uv, 'venv', '--python', '3.11.13', venv], env)
    for name, url, revision in [
        ('AC-DiT', 'https://github.com/PKU-HMI-Lab/AC-DiT.git', ACDIT_REV),
        ('Bidirectional-VLA-Interface', 'https://github.com/loneeshi/Bidirectional-VLA-Interface.git', BVI_REV),
    ]:
        dest = root/'src'/name
        if not dest.exists():
            dest.parent.mkdir(exist_ok=True)
            run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, dest], env)
            run(['git', '-C', dest, 'checkout', '--detach', revision], env)
        actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        if not actual.startswith(revision):
            raise RuntimeError('Existing checkout revision differs; refusing to reset it')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(),
              'python': subprocess.check_output([str(venv/'bin/python'), '--version'], text=True).strip(),
              'acdit_revision': ACDIT_REV, 'bvi_revision': BVI_REV,
              'cuda_home': env['CUDA_HOME'], 'gpu_used': False,
              'heavy_dependencies_installed': False, 'training_started': False}
    (root/'bootstrap-result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
