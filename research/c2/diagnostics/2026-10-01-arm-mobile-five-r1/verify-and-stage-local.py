"""Verify this batch and stage only its own media/index additions."""
import hashlib
import json
from pathlib import Path
import subprocess

diag=Path(__file__).resolve().parent;repo=diag.parents[3];media=repo/'docs/media'
folders=sorted(media.glob('c2-pick-arm-2026-10-01-arm-dev-*-V-mobile-*'))
assert len(folders)==5
names=[p.name for p in folders]
manifest=json.loads((media/'manifest.json').read_text(encoding='utf-8'))
videos=[v for v in manifest['videos'] if v['path'].split('/')[0] in names]
assert len(videos)==15
for v in videos:
    path=media/v['path']
    assert path.stat().st_size==v['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==v['sha256']
for folder in folders:
    for name in ('result.json','rendering-result.json','media-receipt.json',
                 'delivery/trajectory.svg','delivery/trajectory.html','delivery/history.html'):
        assert (folder/name).is_file()
close=json.loads((diag/'closeout.json').read_text())
h=hashlib.sha256()
with (diag/'raw-run.tar.gz').open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
assert h.hexdigest()==close['raw_archive_sha256']
receipt=dict(physical_processes=5,online_video_count=15,analytical_trajectory_count=5,
    all_video_hashes_verified=True,raw_archive_sha256=h.hexdigest(),videos=videos)
(diag/'media-verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def git(*args,**kwargs):return subprocess.check_output(['git',*args],cwd=repo,**kwargs)
def stage(path,data):
    oid=git('hash-object','-w','--stdin',input=data).decode().strip()
    git('update-index','--cacheinfo',f'100644,{oid},{path}')
base=git('show','HEAD:docs/media/manifest.json').decode('utf-8')
start=base.index('[',base.index('"videos"'));_,length=json.JSONDecoder().raw_decode(base[start:])
end=start+length-1
assert not any(v['path'].split('/')[0] in names for v in json.loads(base)['videos'])
entries=',\n'.join('    '+json.dumps(v,ensure_ascii=False,indent=2).replace('\n','\n    ') for v in videos)
staged=base[:end].rstrip()+',\n'+entries+'\n  '+base[end:]
assert json.loads(staged)['videos']==json.loads(base)['videos']+videos
stage('docs/media/manifest.json',staged.encode())
lines=[line for line in (media/'README.md').read_text(encoding='utf-8').splitlines()
       if any(line.startswith('- ['+name+']') for name in names)]
assert len(lines)==5
readme=git('show','HEAD:docs/media/README.md')
stage('docs/media/README.md',readme.rstrip()+b'\n\n'+('\n\n'.join(lines)+'\n').encode())
exclude=repo/'.git/info/exclude'
rule='/research/c2/diagnostics/2026-10-01-arm-mobile-five-r1/*.tar.gz'
if rule not in exclude.read_text(encoding='utf-8'):
    with exclude.open('a',encoding='utf-8') as stream:stream.write('\n'+rule+'\n')
git('add','--',diag.relative_to(repo).as_posix(),*[p.relative_to(repo).as_posix() for p in folders])
git('add','-f','--',*[str((media/v['path']).relative_to(repo)) for v in videos])
print(json.dumps(dict(videos_verified=15,folders=names,archive_hash=h.hexdigest())))
