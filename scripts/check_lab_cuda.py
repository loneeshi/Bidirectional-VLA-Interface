"""Bounded small CUDA arithmetic check on one explicitly selected idle GPU.

No model loading, simulator, training, or API usage. Never selects GPU zero by
default. Uses the UUID so CUDA numbering does not silently select another card.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    selected = [row.split(',') for row in raw.splitlines()
                if int(row.split(',')[0]) == args.gpu_index]
    if len(selected) != 1 or int(selected[0][2]) > 1024:
        raise RuntimeError('Selected GPU unavailable or exceeds 1 GiB occupied-memory guard')
    os.environ['CUDA_VISIBLE_DEVICES'] = selected[0][1].strip()
    os.environ['OMP_NUM_THREADS'] = '2'
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    report = {'physical_gpu_index': args.gpu_index, 'torch': torch.__version__,
              'cuda_build': torch.version.cuda, 'device_name': torch.cuda.get_device_name(0),
              'capability': list(torch.cuda.get_device_capability(0)),
              'native_bfloat16_supported': torch.cuda.is_bf16_supported(including_emulation=False),
              'checks': {}, 'training_started': False}
    reference_input = torch.randn(256, 256)
    reference = reference_input @ reference_input
    for dtype in (torch.float32, torch.float16):
        tensor = reference_input.to(device='cuda:0', dtype=dtype)
        actual = tensor @ tensor
        torch.cuda.synchronize()
        report['checks'][str(dtype)] = {
            'finite': bool(torch.isfinite(actual).all()),
            'max_abs_error_vs_cpu_float32': float((actual.float().cpu() - reference).abs().max())}
        del actual, tensor
    report['max_allocated_bytes'] = torch.cuda.max_memory_allocated()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not all(x['finite'] for x in report['checks'].values()):
        raise RuntimeError('Nonfinite CUDA arithmetic')


if __name__ == '__main__':
    main()
