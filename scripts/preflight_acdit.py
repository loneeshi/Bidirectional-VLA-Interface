"""Download/hash pinned community candidates without renting compute.

Passing this check proves file identity, NOT checkpoint compatibility or success.
No torch.load/pickle execution, model API calls, or simulator creation occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, entry):
    if not path.is_file():
        return {'status': 'missing'}
    size = path.stat().st_size
    if size != entry['bytes']:
        return {'status': 'size_mismatch', 'bytes': size}
    digest = sha256_file(path)
    return {'status': 'verified' if digest == entry['sha256'] else 'hash_mismatch',
            'bytes': size, 'sha256': digest}


def fetch(url, path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    # Never overwrite a complete but different checkpoint.
    if path.exists():
        raise ValueError(f'Existing checkpoint failed verification: {path}')
    if shutil.disk_usage(path.parent).free < entry['bytes'] + 1024**3:
        raise ValueError('Insufficient local free space (including 1 GiB reserve)')
    request = urllib.request.Request(url, headers={'User-Agent': 'bvi-acdit-preflight/1'})
    with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as out:
        total = 0
        last_notice = 0
        while block := response.read(8 * 1024 * 1024):
            total += len(block)
            if total > entry['bytes']:
                raise ValueError('Download exceeds pinned size')
            out.write(block)
            if total - last_notice >= 256 * 1024 * 1024:
                print(f'{path.name}: {total}/{entry["bytes"]} bytes', flush=True)
                last_notice = total
    checked = verify(part, entry)
    if checked['status'] != 'verified':
        raise ValueError(f'Download failed verification: {checked}')
    part.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, default=root/'configs/acdit_mshab_reference.json')
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fetch-weights', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    results = []
    weights_root = args.weights_dir.resolve()
    for entry in cfg['checkpoints']:
        path = (weights_root/entry['file']).resolve()
        if not path.is_relative_to(weights_root):
            raise ValueError('Checkpoint path escapes destination')
        checked = verify(path, entry)
        if checked['status'] == 'missing' and args.fetch_weights:
            url = (f'https://huggingface.co/{cfg["weights_repository"]}/resolve/'
                   f'{cfg["weights_revision"]}/{entry["file"]}')
            try:
                print(f'Downloading pinned {entry["role"]}', flush=True)
                fetch(url, path, entry)
                checked = verify(path, entry)
            except Exception as exc:
                checked = {'status': 'download_failed', 'error': str(exc)}
        results.append({'role': entry['role'], 'file': entry['file'], **checked})
        print(json.dumps(results[-1]), flush=True)
    report = {
        'at': datetime.now(timezone.utc).isoformat(),
        'code_commit': cfg['code_commit'],
        'weights_repository': cfg['weights_repository'],
        'weights_revision': cfg['weights_revision'],
        'weights_provenance': cfg['weights_provenance'],
        'checkpoint_files': results,
        'all_file_hashes_verified': all(r['status'] == 'verified' for r in results),
        'runtime_verified': False,
        'training_started': False,
        'unresolved_runtime_gates': cfg['unresolved_runtime_gates'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['all_file_hashes_verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
