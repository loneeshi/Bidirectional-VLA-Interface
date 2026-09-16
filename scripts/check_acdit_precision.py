"""CPU comparison of upstream and patched perception weighting only.

This checks that the default BF16 path is unchanged and the explicit FP32 path
works; it does not certify the full AC-DiT model or task success.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import types

REVISION = '90ad00a926f34da04816ed9c3312aaf3bc845b7f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    original = types.ModuleType('upstream_weighting')
    patched = types.ModuleType('patched_weighting')
    code = subprocess.check_output(['git','-C',str(args.source),'show',
                                    REVISION+':models/weighting.py'],text=True)
    exec(code, original.__dict__)
    exec((args.source/'models/weighting.py').read_text(), patched.__dict__)
    old = original.PerceptionAwareMultimodalAdaptor().eval()
    new = patched.PerceptionAwareMultimodalAdaptor().eval()
    new.load_state_dict(old.state_dict(), strict=True)
    inputs = [torch.randn(1,6*729,1152),torch.randn(1,2*128,1152),torch.randn(1,8,1152)]
    with torch.no_grad():
        expected, observed = old(*inputs), new(*inputs)
        identical = all(torch.equal(a,b) for a,b in zip(expected, observed))
        fp32 = new.float()(*inputs)
        finite = all(bool(torch.isfinite(x).all()) and x.dtype == torch.float32 for x in fp32)
    report = {'scope':'perception_weighting_component_only',
              'bf16_default_bitwise_equal_to_original':identical,
              'fp32_outputs_finite':finite,'gpu_used':False,
              'full_model_verified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not identical or not finite:
        raise RuntimeError('Precision component validation failed')


if __name__ == '__main__':
    main()
