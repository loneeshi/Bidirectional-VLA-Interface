"""Explicit device-precision patch; keeps the upstream BF16 default.

Run against a dedicated pinned AC-DiT checkout. Writes the exact diff and input/
output hashes before changing files. No policy, reward, action or tool logic is
changed. Native smoke tests are still required before reporting compatibility.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


REPLACEMENTS = {
    'models/weighting.py': [
        ('image_feature = image_feature.to(dtype=torch.bfloat16)',
         'image_feature = image_feature.to(dtype=self.image_proj[0].weight.dtype)'),
        ('pc_feature = pc_feature.to(dtype=torch.bfloat16)',
         'pc_feature = pc_feature.to(dtype=self.pc_proj[0].weight.dtype)'),
        ('lang_tokens = lang_tokens.to(dtype=torch.bfloat16)',
         'lang_tokens = lang_tokens.to(dtype=self.lang_proj[0].weight.dtype)'),
    ],
    'model_wrappers/mshab_model.py': [
        ('self.policy.encode_pointcloud(pointcloud, weight_dtype=torch.bfloat16)',
         'self.policy.encode_pointcloud(pointcloud, weight_dtype=dtype)'),
    ],
    'scripts/eval_mshab.py': [
        ('    parser.add_argument("--device", type=str, default="cuda")',
         '    parser.add_argument("--device", type=str, default="cuda")\n'
         '    parser.add_argument("--precision", choices=["bf16", "fp32", "fp16"], default="bf16")'),
        ('        dtype=torch.bfloat16,',
         '        dtype={"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.precision],'),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.source_lock.read_text())
    changes = []
    # Validate all files before mutating any. Never overwrite unknown edits.
    for relative, replacements in REPLACEMENTS.items():
        path = args.source/relative
        raw = path.read_bytes()
        before_hash = hashlib.sha256(raw).hexdigest()
        if before_hash != lock['files'].get(relative):
            raise ValueError('Source differs from pinned input: '+relative)
        before = raw.decode('utf-8')
        after = before
        for old, new in replacements:
            if after.count(old) != 1:
                raise ValueError('Expected exactly one replacement in '+relative)
            after = after.replace(old, new)
        changes.append((relative, before, after, before_hash))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    diff = ''.join(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                              fromfile='a/'+name, tofile='b/'+name))
                   for name, before, after, _ in changes)
    (args.report_dir/'device-precision.patch').write_text(diff, encoding='utf-8')
    manifest = {name:{'before_sha256':digest,
                      'after_sha256':hashlib.sha256(after.encode()).hexdigest()}
                for name, _, after, digest in changes}
    (args.report_dir/'device-precision.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name, _, after, _ in changes:
        (args.source/name).write_bytes(after.encode('utf-8'))
    print(json.dumps({'files_patched':list(manifest),'default_precision':'bf16',
                      'training_started':False}))


if __name__ == '__main__':
    main()
