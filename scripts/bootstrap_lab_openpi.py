"""Bounded CPU-only author OpenPI installation in the user's lab workspace."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

REVISION = 'f4eb160ba52b22c1e85fe432de59c24bbbac6187'


def main():
    root = Path.home()/'bvi-research'
    source = root/'src/openpi-author'
    python = root/'envs/openpi/bin/python'
    uv = root/'bootstrap/bin/uv'
    if root.is_symlink() or source.is_symlink():
        raise RuntimeError('Refuse symlinked experiment directories')
    started = time.monotonic()
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'author_revision': REVISION, 'gpu_used': False,
              'api_calls': 0, 'training_started': False, 'status': 'installing'}
    path = root/'openpi-bootstrap-result.json'
    def save():
        path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               UV_PROJECT_ENVIRONMENT=str(python.parent.parent),
               UV_PYTHON_INSTALL_DIR=str(root/'python'),
               UV_CACHE_DIR=str(root/'cache/uv'), GIT_LFS_SKIP_SMUDGE='1',
               HF_HOME=str(root/'hf-cache'), UV_HTTP_TIMEOUT='90')
    def run(args, cwd=None, timeout=300):
        print(json.dumps({'command': [str(x) for x in args]}), flush=True)
        subprocess.run([str(x) for x in args], cwd=cwd, env=env, check=True, timeout=timeout)
    try:
        if not source.exists():
            run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                 'https://github.com/cxliu0314/openpi.git', source])
            run(['git', '-C', source, 'checkout', '--detach', REVISION])
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != REVISION:
            raise RuntimeError('Existing author checkout differs; no automatic reset')
        run([uv, 'sync', '--frozen', '--no-dev', '--python', '3.11.13'], cwd=source, timeout=1800)
        run([python, '-c', 'import jax,flax,openpi; print(jax.__version__,flax.__version__,jax.devices())'])
        freeze = subprocess.check_output([str(uv), 'pip', 'freeze', '--python', str(python)], text=True, env=env)
        (root/'openpi-package-freeze.txt').write_text(freeze)
        report.update(status='cpu_import_gate_passed', environment=str(python.parent.parent))
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        save()


if __name__ == '__main__':
    main()
