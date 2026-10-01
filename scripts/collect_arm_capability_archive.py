"""Collect an arm batch archive and register exact online media locally."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def collect(archive, diagnostics, media_root):
    diagnostics=Path(diagnostics);media_root=Path(media_root)
    manifest_path=media_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    registered=[]
    with tarfile.open(archive) as tar:
        members={m.name:m for m in tar.getmembers()}
        def read(member):
            with tar.extractfile(member) as stream:return stream.read()
        root=next(n.rsplit('/',1)[0] for n in members if n.endswith('/batch.json'))
        for name in ('batch.json','api-ledger.json','summary.json','media-index.json'):
            if root+'/'+name in members:(diagnostics/name).write_bytes(read(members[root+'/'+name]))
        for name in sorted(members):
            if not name.endswith('/result.json') or '/media-source/' in name:continue
            result=json.loads(read(members[name]));case=result['case_id'];condition=result['condition']
            raw=read(members[name]);digest=hashlib.sha256(raw).hexdigest()
            local=diagnostics/'results'/case/condition;local.mkdir(parents=True,exist_ok=True)
            (local/'result.json').write_bytes(raw)
            prefix=name.rsplit('/',1)[0]
            receipt_name=prefix+'/media-receipt.json'
            if receipt_name in members:(local/'media-receipt.json').write_bytes(read(members[receipt_name]))
            if not result.get('frames'):continue
            date=result['recorded_at_utc'][:10]
            folder=media_root/f'c2-pick-arm-{date}-{case}-{condition}-{digest[:8]}'
            folder.mkdir(exist_ok=True)
            for source, target_name in ((prefix+'/media-source/result.json','rendering-result.json'),
                                        (receipt_name,'media-receipt.json')):
                if source in members:(folder/target_name).write_bytes(read(members[source]))
            if (folder/'result.json').exists():
                if (folder/'result.json').read_bytes()!=raw:raise ValueError('existing media identity changed')
                registered.append(folder.relative_to(media_root).as_posix())
                continue
            (folder/'result.json').write_bytes(raw)
            delivery=prefix+'/media-source/delivery/'
            sensors=prefix+'/sensors/'
            for n,m in members.items():
                if n.startswith(delivery):relative='delivery/'+n[len(delivery):]
                elif n.startswith(sensors) and n.endswith('.png'):relative='sensors/'+n[len(sensors):]
                else:continue
                if m.isdir():continue
                target=folder/relative
                if not target.resolve().is_relative_to(folder.resolve()):raise ValueError('unsafe media path')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(read(m))
                if target.suffix=='.mp4':
                    manifest['videos'].append(dict(filename=target.name,path=target.relative_to(media_root).as_posix(),
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
                        case_id=case,condition=condition,outcome=result.get('termination_category'),
                        status=result['status'],strict_pick_success=result.get('strict_pick_success'),
                        recorded_at_utc=result['recorded_at_utc'],kind='online official-spawn arm attempt; evaluation-only overlay'))
            registered.append(folder.relative_to(media_root).as_posix())
            with (media_root/'README.md').open('a',encoding='utf-8') as stream:
                stream.write(f"\n- [{folder.name}]({folder.name}/delivery/README.md) — {case}/{condition}; {result['status']}; strict Pick={result.get('strict_pick_success')}; online recording, analysis only.\n")
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return registered


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('archive','diagnostics','media-root'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(collect(args.archive,args.diagnostics,args.media_root)))
