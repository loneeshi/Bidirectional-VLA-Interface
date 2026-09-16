"""Explicit opt-in HF precision on Turing; preserve upstream BF16 by default."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path.home() / 'bvi-research'
    source = root / 'src/LightNav-0'
    commit = 'c6f40e3220edbf7011e4f17eaf2c865416737d4d'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
    changes = []
    for name, old, new, count in [
        ('src/lightnav/inference/model.py', 'torch_dtype=torch.bfloat16,',
         "torch_dtype=getattr(torch, os.environ.get('LIGHTNAV_HF_DTYPE', 'bfloat16')),", 2),
        ('src/lightnav/inference/engine.py', 'v.to(device=device, dtype=torch.bfloat16)',
         "v.to(device=device, dtype=getattr(torch, __import__('os').environ.get('LIGHTNAV_HF_DTYPE', 'bfloat16')))", 1),
    ]:
        before = subprocess.check_output(['git', '-C', str(source), 'show', f'{commit}:{name}']).decode()
        current = (source / name).read_text()
        assert before.count(old) == count
        after = before.replace(old, new)
        assert current in (before, after), f'Unknown changes to {name}'
        changes.append((name, before, after))
    records, diff = [], []
    for name, before, after in changes:
        records.append({'path': name, 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(after.encode()).hexdigest()})
        diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile='a/'+name, tofile='b/'+name))
        (source / name).write_text(after)
    out = root / 'patches'
    out.mkdir(exist_ok=True)
    (out / 'lightnav-precision.patch').write_text(''.join(diff))
    (out / 'lightnav-precision.json').write_text(json.dumps({'source_commit': commit,
        'default': 'bfloat16', 'diagnostic_env': 'LIGHTNAV_HF_DTYPE=float32', 'files': records,
        'policy_or_prompt_modified': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
