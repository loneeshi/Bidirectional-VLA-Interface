"""One read-only SSH snapshot; no GPU allocation, retries, scheduler or mutations.

Run from the repository root; output must be a NEW filename.
"""
import argparse
import json
from pathlib import Path
import sys

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from probe_lab_server import config

COMMAND = """python3 - <<'BVI_SNAPSHOT'
import datetime,json,pathlib,subprocess,os
p=pathlib.Path('/home/pshuai/bvi-research/runs/acdit-apple-bounded-20260919-run01')
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for n in ['supervisor.json','stage1/status.json','stage2/status.json','stage1/latest-checkpoint.json','stage2/latest-checkpoint.json']:
 f=p/n;r['files'][n]=json.loads(f.read_text()) if f.exists() else None
r['checkpoint_inventory']=[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in p.glob('stage*/*') if f.suffix in ['.pt','.tmp']]
for name,args in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader']),('compute_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'])]:
 x=subprocess.run(args,capture_output=True,text=True,timeout=15);r[name]={'exit_code':x.returncode,'stdout':x.stdout}
x=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,ppid,etimes,args'],capture_output=True,text=True,timeout=10)
r['owned_training_processes']=[line for line in x.stdout.splitlines() if ('train_acdit_rdt_bounded.py --stage' in line or 'python -u /home/pshuai/bvi-research/run_acdit_apple_pipeline.py --run' in line) and not 'bash -c' in line]
print(json.dumps(r))
BVI_SNAPSHOT
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve old snapshots; choose a new output name')
    cfg = config(Path('.env.server.local'))
    client = paramiko.SSHClient()
    client.load_host_keys('.runtime/lab-server/known_hosts')
    try:
        client.connect(cfg['LAB_SSH_HOST'], port=int(cfg['LAB_SSH_PORT']),
                       username=cfg['LAB_SSH_USER'], password=cfg.get('LAB_SSH_PASSWORD') or None,
                       look_for_keys=False, allow_agent=False, timeout=12,
                       auth_timeout=12, banner_timeout=12)
        _, out, err = client.exec_command(COMMAND, timeout=50)
        raw = out.read()
        code = out.channel.recv_exit_status()
        report = json.loads(raw) if code == 0 else {'remote_exit_code': code}
    except Exception as exc:
        report = {'error_type': type(exc).__name__}
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
