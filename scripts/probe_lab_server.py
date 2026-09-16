"""Read-only SSH preflight. Credentials never enter CLI args or output.

First connections use trust-on-first-use and persist the host key locally;
changed known host keys are rejected. Does not install packages or start GPUs.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys


REMOTE_PROBE = """python3 - <<'BVI_READ_ONLY_PROBE'
import json, os, pathlib, shutil, subprocess
def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=result.returncode, stdout=result.stdout[:12000], stderr=result.stderr[:1000])
    except Exception as exc:
        return dict(error=type(exc).__name__)
memory = {}
mem = pathlib.Path('/proc/meminfo')
if mem.exists():
    memory = {line.split(':')[0]:line.split(':')[1].strip() for line in mem.read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
paths = [p for p in [str(pathlib.Path.home()), '/tmp', '/scratch', '/data', '/workspace'] if pathlib.Path(p).exists()]
report = {
    'uname': command(['uname', '-srm']),
    'user': command(['id', '-un']),
    'cpu_count': os.cpu_count(),
    'memory': memory,
    'disk': command(['df', '-h', *paths]),
    'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader']),
    'tools': {tool:shutil.which(tool) for tool in ['python3','conda','uv','git','srun','sbatch','sinfo','qsub','docker','apptainer','nvcc']},
    'scheduler_allocation': {key:os.environ.get(key) for key in ['SLURM_JOB_ID','PBS_JOBID','CUDA_VISIBLE_DEVICES']},
    'home_exists': pathlib.Path.home().is_dir(),
    'probe_is_read_only': True,
}
print(json.dumps(report))
BVI_READ_ONLY_PROBE
"""


def config(path):
    result = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('\"', "'"):
                value = value[1:-1]
            result[key.strip()] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    args = parser.parse_args()
    import paramiko
    cfg = config(args.config)
    if not cfg.get('LAB_SSH_HOST') or not cfg.get('LAB_SSH_USER'):
        raise ValueError('Missing SSH host/user in local configuration')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    args.known_hosts.parent.mkdir(parents=True, exist_ok=True)
    if args.known_hosts.exists():
        client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    report = {'authenticated': False, 'remote_probe_completed': False}
    try:
        client.connect(cfg['LAB_SSH_HOST'].strip(), port=int(cfg.get('LAB_SSH_PORT') or 22),
                       username=cfg['LAB_SSH_USER'].strip(), password=cfg.get('LAB_SSH_PASSWORD') or None,
                       key_filename=cfg.get('LAB_SSH_KEY_PATH') or None,
                       look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=15, banner_timeout=15)
        client.save_host_keys(str(args.known_hosts))
        key = client.get_transport().get_remote_server_key()
        fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        report.update(authenticated=True, host_key_type=key.get_name(), host_key_sha256=fingerprint)
        _, stdout, stderr = client.exec_command(REMOTE_PROBE, timeout=60)
        raw, error = stdout.read().decode(), stderr.read().decode()
        status = stdout.channel.recv_exit_status()
        if status == 0:
            report['server'] = json.loads(raw)
            report['remote_probe_completed'] = True
        else:
            # Do not echo arbitrary remote output into the public transcript.
            report.update(remote_exit_status=status, remote_error_present=bool(error))
    except Exception as exc:
        # Classify without printing exception arguments or credential data.
        report['error_type'] = type(exc).__name__
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['remote_probe_completed'] else 1


if __name__ == '__main__':
    sys.exit(main())
