"""Local operator SSH utility; credentials never enter remote commands."""
import argparse
from pathlib import Path
import paramiko

p=argparse.ArgumentParser()
p.add_argument('--command', required=True)
p.add_argument('--upload', nargs=2)
p.add_argument('--download', nargs=2)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
cfg={}
for line in (root/'.env.server.local').read_text(encoding='utf-8').splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        k,v=line.split('=',1);cfg[k.strip()]=v.strip().strip('"').strip("'")
c=paramiko.SSHClient()
c.load_host_keys(str(root/'.runtime/lab-server/known_hosts'))
options=dict(hostname=cfg['LAB_SSH_HOST'],port=int(cfg['LAB_SSH_PORT']),
             username=cfg['LAB_SSH_USER'],password=cfg.get('LAB_SSH_PASSWORD') or None,
             timeout=20,banner_timeout=20,auth_timeout=20)
if cfg.get('LAB_SSH_KEY_PATH'):options['key_filename']=cfg['LAB_SSH_KEY_PATH']
try:
    c.connect(**options)
    if a.upload:
        with c.open_sftp() as s:s.put(*a.upload)
    if a.download:
        local = Path(a.download[1])
        local.parent.mkdir(parents=True, exist_ok=True)
        with c.open_sftp() as s:s.get(a.download[0], str(local))
    _,out,err=c.exec_command(a.command,timeout=120)
    print(out.read().decode('utf-8','replace'))
    print(err.read().decode('utf-8','replace'))
    code=out.channel.recv_exit_status()
finally:c.close()
raise SystemExit(code)
