"""Bounded serial BC diagnostic; halt on errors or initial-state pairing failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(2024,2029)))
    p.add_argument('--environment',choices=['official','acdit'],default='official')
    p.add_argument('--paths',choices=['official','project'],nargs='+',default=['official','project'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',episodes=[],max_seconds=1800,training_updates=0,api_calls=0)
    start=time.monotonic()
    def save():
        report['wall_seconds']=time.monotonic()-start
        (a.output/'panel.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for seed in a.seeds:
            pair=[]
            for path in a.paths:
                remaining=1780-(time.monotonic()-start)
                if remaining<1: raise TimeoutError('Panel time limit')
                out=a.output/f'{seed}-{path}'
                with (a.output/f'{seed}-{path}.log').open('w') as log:
                    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('eval_bc_t0a.py')),
                        '--seed',str(seed),'--path',path,'--environment',a.environment,'--output',str(out)],
                        env=dict(os.environ,PYTHONHASHSEED=str(seed)),stdout=log,stderr=subprocess.STDOUT,
                        timeout=min(210,remaining))
                result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {'status':'missing_result'}
                report['episodes'].append(result);pair.append(result);save()
                if proc.returncode or result['status']!='completed': raise RuntimeError(f'{seed} {path} failed')
            keys=['initial_state_sha256','initial_policy_obs_sha256','checkpoint_sha256']
            if len(pair)==2 and not all(pair[0][key]==pair[1][key] for key in keys):
                raise RuntimeError(f'{seed}: initial pairing mismatch; results are not paired')
        report['status']='completed'
    except Exception as exc:
        report.update(status='stopped',error=repr(exc));raise
    finally:
        report['gpu_after']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader'],text=True).strip()
        save()


if __name__=='__main__': main()
