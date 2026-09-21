"""Audit original AC-DiT initialization and dependencies without CUDA access."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    root = Path.home()/'bvi-research'
    os.environ['HF_HOME'] = str(root/'hf-cache')
    source = root/'src/AC-DiT-rdt-20260919'
    sys.path.insert(0, str(source)); os.chdir(source)
    out = root/'runs/acdit-rdt-preparation-20260919'
    import torch
    import yaml
    torch.set_num_threads(2)
    from models.dit_runner import DiTRunner
    from models.acdit_runner import ACDiTRunner
    from models.multimodal_encoder.siglip_lang_encoder import SiglipLangEncoder
    started = time.monotonic()
    report = {'status':'started','cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
              'gpu_operations':0,'api_calls':0,'models':[]}
    try:
        for stage, cls, file, base in [(1,DiTRunner,'config_s.yaml','rdt-170m'),(2,ACDiTRunner,'config.yaml','rdt-1b')]:
            config = yaml.safe_load((source/'configs'/file).read_text())
            m = config['model']; c = config['common']
            kwargs = dict(action_dim=128,pred_horizon=c['action_chunk_size'],config=m,
                lang_token_dim=m['lang_token_dim'],img_token_dim=m['img_token_dim'],pc_token_dim=m['pc_token_dim'],
                state_token_dim=128,max_lang_cond_len=config['dataset']['tokenizer_max_length'],
                img_cond_len=4374,pc_cond_len=c['img_history_size']*config['dataset']['pointcloud_feature_length'],
                in_context_cond_dim=18,dtype=torch.float32)
            if stage == 2:
                # Only shape/load audit. Actual stage 2 must use the newly trained stage 1.
                kwargs['mobility_head_ckpt_path']=str(root/'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt')
            # Upstream PCViews creates constant camera tensors with .cuda() in __init__.
            # For this shape/loading-only audit, keep those constants on CPU. Never run
            # a forward pass under this override; the real training process is unmodified.
            original_cuda = torch.Tensor.cuda
            torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor
            try:
                model = cls(**kwargs)
            finally:
                torch.Tensor.cuda = original_cuda
            weights = torch.load(root/'checkpoints/acdit-rdt-init-20260919'/base/'pytorch_model.bin',map_location='cpu',weights_only=True)
            removed = [k for k in weights if k.startswith('lang_adaptor')]
            for k in removed: del weights[k]  # Upstream explicitly replaces T5 4096 with SigLIP 1152.
            result = model.load_state_dict(weights,strict=False)
            row = {'stage':stage,'base':base,'parameters':sum(p.numel() for p in model.parameters()),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'removed_language_keys':removed,'missing_keys':result.missing_keys,'unexpected_keys':result.unexpected_keys,
                'base_tensor_count':len(weights),'finite_initialization':all(torch.isfinite(p).all().item() for p in model.parameters()),
                'cpu_audit_only_override':'Tensor.cuda returns self during construction of constant LIFT3D camera tensors; no forward or training'}
            # Architecture additions may be missing; every original transformer block must load.
            if any(k.startswith('model.blocks.') for k in result.missing_keys):
                raise ValueError('Pretrained transformer blocks missing')
            report['models'].append(row)
            (out/'cpu-preflight.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ['missing_keys','unexpected_keys']}),flush=True)
            del model,weights; gc.collect()
        encoder = SiglipLangEncoder(from_pretrained='google/siglip-so400m-patch14-384',
            model_max_length=1024,device='cpu',torch_dtype=torch.float32)
        instructions=next(x['instructions'] for x in json.loads((source/'configs/mshab_languages.json').read_text())
                          if (x['task'],x['subtask'],x['object'])==('set_table','pick','013_apple'))
        langdir=out/'instructions';langdir.mkdir(exist_ok=True)
        tokens=encoder.tokenizer(instructions,return_tensors='pt',padding='longest',truncation=False)
        with torch.no_grad(): embeddings=encoder.model(input_ids=tokens['input_ids'])['last_hidden_state'].detach().cpu()
        for i,(instruction,embedding) in enumerate(zip(instructions,embeddings)):
            torch.save({'task_name':'set_table_pick_013_apple','instruction':instruction,'text_embed':embedding},langdir/f'lang_embed_{i}.pt')
        report.update(status='passed',instruction_count=len(instructions),language_shape=list(embeddings.shape),
            limitation='CPU checks do not establish GPU backward compatibility, throughput, or native task capability')
    except Exception as exc:
        report.update(status='error',error=repr(exc)); raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        (out/'cpu-preflight.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
