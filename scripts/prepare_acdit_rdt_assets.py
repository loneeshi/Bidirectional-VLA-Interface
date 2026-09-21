"""CPU-only, revision-pinned public RDT initialization download; no inference."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DISABLE_XET'] = '1'


def main():
    from huggingface_hub import HfApi, hf_hub_download
    root = Path.home() / 'bvi-research'
    out = root / 'runs/acdit-rdt-preparation-20260919'
    plan = json.loads((out / 'asset-plan.json').read_text())
    started = time.monotonic()
    report = {'status': 'downloading', 'files': [], 'gpu_operations': 0, 'api_calls': 0,
              'lab_charge_usd': None, 'public_hub_download_not_model_api': True}
    try:
        for model, revision in plan['models'].items():
            repo = 'robotics-diffusion-transformer/' + model
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            assert info.sha == revision
            for name in ('config.json', 'pytorch_model.bin'):
                entry = next(x for x in info.siblings if x.rfilename == name)
                if entry.size > plan['max_download_bytes']:
                    raise ValueError('Asset exceeds byte ceiling')
                path = Path(hf_hub_download(repo, name, revision=revision,
                    local_dir=root/'checkpoints/acdit-rdt-init-20260919'/model))
                h = hashlib.sha256()
                with path.open('rb') as f:
                    for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
                assert path.stat().st_size == entry.size
                expected = getattr(entry.lfs, 'sha256', None)
                if expected and expected != h.hexdigest(): raise ValueError('LFS SHA mismatch')
                report['files'].append({'path': str(path), 'repository': repo, 'revision': revision,
                    'bytes': path.stat().st_size, 'sha256': h.hexdigest(), 'lfs_sha256': expected})
                (out/'asset-result.json').write_text(json.dumps(report, indent=2))
                print(json.dumps(report['files'][-1]), flush=True)
        assert sum(x['bytes'] for x in report['files']) <= plan['max_download_bytes']
        report['status'] = 'verified'
    except Exception as exc:
        report.update(status='error', error=repr(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (out/'asset-result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
