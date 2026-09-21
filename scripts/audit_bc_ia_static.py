"""CPU-only saved-state and normalizer provenance audit. No simulator or model."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch


def main():
    root=Path.home()/'bvi-research'
    out=root/'audits/bc-ia-static-2026-09-18.json'
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def leaves(x,prefix=''):
        if isinstance(x,dict):
            return {k:v for key,value in x.items() for k,v in leaves(value,prefix+'/'+str(key)).items()}
        return {prefix:np.asarray(x.cpu() if isinstance(x,torch.Tensor) else x)}
    rows=[]
    for seed in range(2024,2029):
        bc=root/f'runs/bc-t0a-2026-09-18-run01/{seed}-official/initial-state.pt'
        ia=root/f'runs/s1-native-render-aligned-2026-09-18-run01/seed{seed}/initial-state.pt'
        reach=root/f'runs/s1-ia-calls-2026-09-18-run01/starts/seed{seed}/reach-state.pt'
        def compare(other):
            a=torch.load(bc,map_location='cpu',weights_only=False)
            b=torch.load(other,map_location='cpu',weights_only=False)
            a=leaves({k:a[k] for k in ['simulator','controller']})
            b=leaves({k:b[k] for k in ['simulator','controller']})
            differences={}
            for k in sorted(a.keys()|b.keys()):
                if k not in a or k not in b or a[k].shape!=b[k].shape:
                    differences[k]={'shape_or_key_mismatch':True};continue
                if not np.array_equal(a[k],b[k]):
                    differences[k]={'max_abs':float(np.max(np.abs(a[k].astype(float)-b[k].astype(float))))}
            return dict(exact=not differences,differences=differences,reference=str(other),reference_sha256=sha(other))
        rows.append(dict(seed=seed,bc_sha256=sha(bc),ia_panel=compare(ia),ia_reach=compare(reach)))
    run=root/'runs/s1-ia-epoch-2026-09-18-run01'
    ident=json.loads((run/'identity.json').read_text())
    best=Path(json.loads((run/'best.json').read_text())['checkpoint'])
    asset=best/'assets/bvi/s1-official-pick-medium-train/norm_stats.json'
    stats=json.loads(asset.read_text())
    metadata=json.loads((root/'runs/s1-ia-calls-2026-09-18-run01/server/metadata.json').read_text())
    table=stats.get('norm_stats',stats)['actions']
    q01=np.asarray(table['q01']);q99=np.asarray(table['q99'])
    x=np.random.default_rng(1).uniform(-1,1,(1000,13))
    normalized=(x-q01)/(q99-q01+1e-6)*2-1
    roundtrip=(normalized+1)/2*(q99-q01+1e-6)+q01
    report=dict(states=rows,normalizer=dict(training_hash=ident['norm_sha256'],
        actual_checkpoint_hash=sha(asset),inference_metadata_hash=metadata['normalizer_sha256'],
        hashes_match=ident['norm_sha256']==sha(asset)==metadata['normalizer_sha256'],
        q01=q01.tolist(),q99=q99.tolist(),roundtrip_max_abs=float(np.max(np.abs(roundtrip-x))),
        test_scope='formula reproduced from pinned source; no model inference',
        zero_width_channels=np.flatnonzero(q99==q01).tolist()),gpu_used=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
