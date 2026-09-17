"""CPU deserialization of the acquired base, not model inference or training."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
import hashlib
import json
from pathlib import Path
import time


def main():
    import numpy as np
    from flax import traverse_util
    from openpi.models import model
    root=Path('checkpoints/s1-pi05-base-2026-09-17')
    output=root/'restore-audit.json'
    report=dict(status='checking_hashes',training_updates=0,gpu_runs=0)
    started=time.monotonic()
    def save():
        report['seconds']=time.monotonic()-started
        output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        acquisition=json.loads((root/'acquisition.json').read_text())
        if acquisition['status']!='downloaded_not_loaded':raise ValueError('Incomplete download')
        for item in acquisition['files']:
            p=root/'params'/item['path'];h=hashlib.sha256()
            with p.open('rb') as f:
                while b:=f.read(8*1024*1024):h.update(b)
            if p.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
                raise ValueError('Downloaded checkpoint bytes changed')
        report['status']='deserializing';save()
        params=model.restore_params(root/'params',restore_type=np.ndarray)
        flat=traverse_util.flatten_dict(params)
        metadata=json.loads((root/'params/_METADATA').read_text())['tree_metadata']
        expected={tuple(k['key'] for k in v['key_metadata'])[1:]:tuple(v['value_metadata']['write_shape']) for v in metadata.values()}
        if set(flat)!=set(expected):raise ValueError('Restored parameter paths differ')
        for key,value in flat.items():
            if value.shape!=expected[key] or not np.isfinite(value).all():
                raise ValueError(f'Invalid tensor {key}')
        report.update(status='passed',restored_tensors=len(flat),all_finite=True,
                      checkpoint_bytes_verified=True,model_forward_executed=False)
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
