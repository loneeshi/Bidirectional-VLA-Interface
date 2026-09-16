"""Pin official apple Pick/Place source; inspect before labeling or training.

Official demonstrations are not assumed to cover actual LightNav handoffs.
No synthetic progress labels or random resampling of missing trajectories.
"""
from pathlib import Path
import json
import shutil
import hashlib
from huggingface_hub import snapshot_download
import h5py

root = Path.home() / 'bvi-research'
out = root / 'data/fetch-tapt-source'
out.mkdir(parents=True, exist_ok=True)
revision = '3e58f01aa0fd8484de9b913fa8a1c6a099884fcf'
assert shutil.disk_usage(root).free > 50 * 1024**3, 'Require 50 GiB free before download'
report = {'status':'downloading', 'repository':'arth-shukla/MS-HAB-SetTable',
          'revision':revision, 'api_calls':0, 'training_updates':0,
          'handoff_coverage':'unverified; official data alone does not establish coverage'}
def save():
    (out/'source-audit.json').write_text(json.dumps(report,indent=2))
save()
try:
    snapshot_download(report['repository'],repo_type='dataset',revision=revision,
                      allow_patterns=['pick/013_apple.h5','pick/013_apple.json',
                                      'place/013_apple.h5','place/013_apple.json'],
                      local_dir=out,max_workers=2)
    report['datasets']={}
    for task in ['pick','place']:
        path=out/task/'013_apple.h5'
        with h5py.File(path) as h:
            ids=sorted([k for k in h if k.startswith('traj_')],key=lambda x:int(x.split('_')[-1]))
            if len(ids)<25:raise ValueError(f'{task}: fewer than25 trajectories')
            schema={}
            h[ids[0]].visititems(lambda name,obj: schema.update({name:{'shape':list(obj.shape),'dtype':str(obj.dtype)}}) if isinstance(obj,h5py.Dataset) else None)
            report['datasets'][task]={'trajectories':len(ids),'train_ids':ids[:20],
                'validation_ids':ids[20:25], 'first_trajectory_schema':schema,
                'split_scope':'whole trajectory before segmentation, per source file'}
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
        report['datasets'][task]['sha256']=digest.hexdigest()
        save()
    report['status']='source_downloaded_and_inspected_not_training_ready'
except Exception as exc:
    report.update(status='failed',error=repr(exc))
    raise
finally:
    save()
