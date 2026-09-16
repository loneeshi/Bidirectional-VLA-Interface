"""Pin and download the upstream SigLIP encoder; no GPU or model API use."""
import hashlib
import json
import os
from pathlib import Path


def main():
    root = Path.home() / 'bvi-research'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HOME'] = str(root / 'hf-cache')
    from huggingface_hub import HfApi, snapshot_download
    repo = 'google/siglip-so400m-patch14-384'
    lock = root / 'encoder-lock.json'
    revision = json.loads(lock.read_text())['revision'] if lock.exists() else HfApi().model_info(repo).sha
    record = {'repository': repo, 'revision': revision, 'status': 'downloading'}
    lock.write_text(json.dumps(record, indent=2) + '\n')
    path = Path(snapshot_download(repo, revision=revision, max_workers=2,
                                 allow_patterns=['*.json', '*.model', '*.safetensors', 'tokenizer*']))
    record.update(status='downloaded', snapshot=str(path), files=[])
    for file in sorted(path.iterdir()):
        if file.is_file():
            h = hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            record['files'].append({'name': file.name, 'bytes': file.stat().st_size, 'sha256': h.hexdigest()})
    # LIFT3D names this repository internally. Pin its offline main reference to
    # this same immutable snapshot so both encoder constructors use identical files.
    ref = path.parent.parent / 'refs' / 'main'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)
    lock.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
