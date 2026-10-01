"""Snapshot the local runtime dependency closure; no deployment or experiment."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu'


def sources():
    pending=[ROOT/'scripts'/name for name in ('run_arm_capability_batch.py','run_arm_capability_case.py',
        'finalize_arm_capability_attempt.py','eef_arm_server_broker.py')]
    found=set()
    def add(module):
        for base in (ROOT/'src',ROOT/'scripts'):
            path=base.joinpath(*module.split('.')).with_suffix('.py')
            if path.exists():pending.append(path)
            init=base.joinpath(*module.split('.'))/'__init__.py'
            if init.exists():pending.append(init)
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path);text=path.read_text(encoding='utf-8');compile(text,str(path),'exec')
        package='.'.join(path.relative_to(ROOT/'src').parts[:-1]) if path.is_relative_to(ROOT/'src') else ''
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Import):
                for alias in node.names:add(alias.name)
            if isinstance(node,ast.ImportFrom):
                parts=package.split('.')
                prefix='.'.join(parts[:len(parts)-node.level+1]) if node.level else ''
                module='.'.join(x for x in (prefix,node.module) if x)
                add(module)
                for alias in node.names:
                    if alias.name!='*':add(module+'.'+alias.name)
        if package:add(package)
    found.update(ROOT/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup'/name
        for name in ('fetch.urdf','response-model.json'))
    return sorted(found)


def main():
    files=sources();mapping={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    archive=D/'frozen-runtime.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in mapping:
            info=zipfile.ZipInfo(name,date_time=(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
    receipt={'status':'CPU source snapshot; no GPU smoke test','source_sha256':mapping,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(mapping),
        'external_assets_receipt':'runtime-assets.cpu.json','physical_steps':0,'provider_sends':0}
    (D/'deployment-freeze.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(mapping),'archive_bytes':archive.stat().st_size}))


if __name__=='__main__':main()
