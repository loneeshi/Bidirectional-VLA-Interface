"""Fixed 25 train-scene starts per Pick/Place; retain every failure.

New SAC-generated demonstrations, not claimed as original published data or
actual LightNav handoff coverage. Whole-trajectory split fixed before collection.
"""
import json,os,subprocess,time
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path.home()/'bvi-research'
out=root/'runs/fetch-tapt-teacher-2026-09-16'
out.mkdir(exist_ok=True,parents=True)
report={'status':'running','source':'official frozen SAC teacher in AC-DiT fork',
        'train_seeds':list(range(3000,3020)),'validation_seeds':list(range(3020,3025)),
        'handoff_coverage':'native starts only; actual navigation handoff supplementation pending',
        'episodes':[], 'api_calls':0,'training_updates':0}
def save():(out/'collection.json').write_text(json.dumps(report,indent=2))
save()
try:
    snapshot_download('arth-shukla/mshab_checkpoints',revision='91e96be85128df43728a7511355c3fa999bd2c94',
        allow_patterns=['rl/set_table/place/013_apple/*'],local_dir=root/'checkpoints/mshab',max_workers=2)
    start=time.monotonic()
    for task in ['pick','place']:
        for seed in range(3000,3025):
            if time.monotonic()-start>3300:raise TimeoutError('Collection wall budget reached')
            p=out/task/f'seed{seed}/result.json'
            if not p.exists():
                with (out/f'{task}-{seed}.log').open('w') as log:
                    result=subprocess.run(['timeout','--kill-after=10','240',str(root/'envs/acdit/bin/python'),
                        str(root/'run_lab_sac_reference.py'),str(seed),'--collect','--task',task],
                        env={**os.environ,'PYTHONHASHSEED':str(seed)},stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'{task}/{seed} infrastructure exit {result.returncode}; preserve failure, no retry')
            d=json.loads(p.read_text())
            if d['status']!='completed':raise RuntimeError(f'{task}/{seed}: incomplete existing run')
            report['episodes'].append({'task':task,'seed':seed,'split':'train' if seed<3020 else 'validation',
                                       'success':d['success'],'steps':d['steps']})
            save()
    report['status']='collected_not_segmented_not_training_ready'
except Exception as exc:
    report.update(status='stopped',error=repr(exc));raise
finally:save()
