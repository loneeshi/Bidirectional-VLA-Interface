"""Strict full AC-DiT checkpoint load on an explicitly selected idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, required=True)
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    output = root / 'full-model-check.json'
    report = {'status': 'started', 'training_started': False, 'api_calls': 0,
              'precision': 'float32', 'physical_gpu_index': args.gpu_index}
    start = time.monotonic()
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        row = next(row.split(',') for row in rows.splitlines() if int(row.split(',')[0]) == args.gpu_index)
        if int(row[2]) > 1024:
            raise RuntimeError('GPU occupied; idle-memory guard refused model load')
        os.environ.update(CUDA_VISIBLE_DEVICES=row[1].strip(), OMP_NUM_THREADS='2',
                          HF_HOME=str(root / 'hf-cache'), HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1')
        import torch
        import yaml
        torch.set_num_threads(2)
        torch.manual_seed(2024)
        source = root / 'src/AC-DiT'
        os.chdir(source)
        sys.path.insert(0, str(source))
        from model_wrappers.mshab_model import create_model
        encoder = json.loads((root / 'encoder-lock.json').read_text())
        config = yaml.safe_load((source / 'configs/config.yaml').read_text())
        model = create_model(
            args=config, pretrained=str(root / 'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
            method_name='AC-DiT', device='cuda:0', dtype=torch.float32,
            pretrained_vision_encoder_name_or_path=encoder['snapshot'],
            mobility_head_ckpt_path=str(root / 'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'),
            combine_flag=False)
        torch.cuda.synchronize()
        report.update(status='passed', strict_load=True,
                      policy_parameters=sum(p.numel() for p in model.policy.parameters()),
                      policy_state_tensors=len(model.policy.state_dict()),
                      max_allocated_bytes=torch.cuda.max_memory_allocated(),
                      encoder_revision=encoder['revision'],
                      parameter_dtypes=sorted({str(p.dtype) for p in model.policy.parameters()}))
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
