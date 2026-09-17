"""Verify exact learned outputs from trusted recorded inputs and saved RNG.

No environment rollout, optimization, GPT or teacher/oracle substitution.
Use an external <=180s timeout and only the lab GPU1 when idle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch',required=True)
args=parser.parse_args()
root=Path.home()/'bvi-research'
directory=root/args.batch
assert json.loads((directory/'batch.json').read_text())['status']=='complete'
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used',
    '--format=csv,noheader,nounits'],text=True).strip())
assert used<1024, 'GPU1 occupied'
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
    HF_HOME=str(root/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
sys.path[:0]=[str(root/'src/AC-DiT'),str(root/'src/Bidirectional-VLA-Interface/src')]
import torch
import yaml
from model_wrappers.mshab_model import create_model
from bvi.acdit_tapt import ACDiTFamilyTool
from bvi.decision_trace import restore_rng_state
torch.set_num_threads(2)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
source=root/'src/AC-DiT'
os.chdir(source)
started=time.monotonic()
report=dict(status='loading',api_calls=0,training_updates=0,decisions=[])
def save():(directory/'replay.json').write_text(json.dumps(report,indent=2))
save()
try:
    encoder=json.loads((root/'encoder-lock.json').read_text())
    wrapper=create_model(args=yaml.safe_load((source/'configs/config.yaml').read_text()),
        pretrained=str(root/'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
        device='cuda:0',dtype=torch.float32,method_name='AC-DiT',combine_flag=False,
        pretrained_vision_encoder_name_or_path=encoder['snapshot'],
        mobility_head_ckpt_path=str(root/'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'))
    checkpoint_path=root/'runs/fetch-tapt-sft-2026-09-16-run01/best.pt'
    checkpoint_sha=sha(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    tool=ACDiTFamilyTool(wrapper.policy,checkpoint['report']['projection_paths'])
    params=dict(tool.named_parameters())
    assert set(checkpoint['trainable'])=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n,p in checkpoint['trainable'].items():
            assert params[n].shape==p.shape
            params[n].copy_(p)
    tool.eval()
    # Three diagnostically important decisions per arm: initial state, final
    # reach decision, and first grasp. No selecting by replay error/success.
    for arm in ['legacy-pre-action','post-action']:
        traces=directory/arm/'decisions'
        manifests=[json.loads(p.read_text()) for p in sorted(traces.glob('*-manifest.json'))]
        chosen=[manifests[0], [m for m in manifests if m['metadata']['family']=='reach'][-1]]
        grasp=[m for m in manifests if m['metadata']['family']=='grasp']
        if grasp:chosen.append(grasp[0])
        for manifest in {m['decision_id']:m for m in chosen}.values():
            assert manifest['metadata']['checkpoint_sha256']==checkpoint_sha
            for artifact in manifest['artifacts'].values():
                assert sha(traces/artifact['path'])==artifact['sha256']
            inputs=torch.load(traces/manifest['artifacts']['inputs']['path'],map_location='cpu',weights_only=False)
            outputs=torch.load(traces/manifest['artifacts']['outputs']['path'],map_location='cpu',weights_only=False)
            batch={k:v.cuda() for k,v in inputs['batch'].items()}
            restore_rng_state(inputs['rng_before_model'])
            actions,progress=tool.predict(manifest['metadata']['family'],batch)
            actions=actions.cpu();progress=progress.cpu()
            exact_actions=torch.equal(actions,outputs['actions'])
            exact_progress=torch.equal(progress,outputs['progress'])
            report['decisions'].append(dict(arm=arm,decision_id=manifest['decision_id'],metadata=manifest['metadata'],
                exact_actions=exact_actions,exact_progress=exact_progress,
                action_max_abs_error=float((actions-outputs['actions']).abs().max()),
                progress_max_abs_error=float((progress-outputs['progress']).abs().max())))
            save()
    report.update(status='complete',all_bitwise_equal=all(r['exact_actions'] and r['exact_progress'] for r in report['decisions']))
except Exception as exc:
    report.update(status='failed',error=repr(exc))
    raise
finally:
    report['elapsed_seconds']=time.monotonic()-started
    save()
