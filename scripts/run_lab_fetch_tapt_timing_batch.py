"""Two bounded lab-only timing controls, same seed/checkpoint, no training/API."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True)
args = parser.parse_args()
root = Path.home()/'bvi-research'
out = root/args.out
out.mkdir(parents=True, exist_ok=False)
report = dict(status='running', seed=2025, started_at=datetime.now(timezone.utc).isoformat(),
              api_calls=0, training_updates=0, max_seconds_per_episode=300, runs=[])
def save():
    (out/'batch.json').write_text(json.dumps(report,indent=2))
save()
started = time.monotonic()
try:
    for label, timing in [('legacy-pre-action','legacy_pre_action'), ('post-action','post_action_v1')]:
        target = out/label
        command = ['timeout','-k','15s','300s',str(root/'envs/acdit/bin/python'),
                   str(root/'run_lab_fetch_tapt_timing.py'),'--gpu-index','1','--seed','2025',
                   '--output',str(target),'--progress-timing',timing,'--record-decisions']
        with (out/f'{label}.log').open('w') as log:
            result = subprocess.run(command,cwd=root,env={**os.environ,'PYTHONHASHSEED':'2025',
                'PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
        report['runs'].append(dict(label=label,returncode=result.returncode,command=command))
        save()
        if result.returncode:
            raise RuntimeError(f'{label} process failed: {result.returncode}')
        episode = json.loads((target/'result.json').read_text())
        if episode['status'] != 'episode_completed':
            raise RuntimeError(f'{label}: unexpected {episode["status"]}')
    report['status']='complete'
except Exception as exc:
    report.update(status='failed',error=repr(exc))
    raise
finally:
    report.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    save()
