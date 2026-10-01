"""Verify retained media and stage only this batch, preserving unrelated edits."""
import hashlib
import json
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[4]
diag = Path(__file__).resolve().parent
media = repo/'docs/media'
prefix = 'c2-pick-arm-2026-10-01-arm-dev-'
working = json.loads((media/'manifest.json').read_text(encoding='utf-8'))
videos = [v for v in working['videos'] if v['path'].startswith(prefix)]
assert len(videos) == 36
for video in videos:
    path = media/video['path']
    assert path.stat().st_size == video['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == video['sha256']
folders = sorted(media.glob(prefix+'*'))
assert len(folders) == 12
for folder in folders:
    for filename in ('result.json','rendering-result.json','media-receipt.json',
                     'delivery/trajectory.html','delivery/trajectory.svg','delivery/history.html'):
        assert (folder/filename).is_file(), (folder,filename)
archive = diag/'raw-run.tar.gz'
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
assert h.hexdigest() == 'bd6140866038a11feb642804664da1de0784d663503622e0a6b54b4142d0c88a'
(diag/'media-verification.json').write_text(json.dumps(dict(
    physical_processes=12, zero_step_script_processes=3, online_video_count=36,
    analytical_trajectory_count=12, all_video_hashes_verified=True,
    raw_archive_sha256=h.hexdigest(), raw_archive_bytes=archive.stat().st_size,
    videos=videos),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def git(*args, **kwargs):
    return subprocess.check_output(['git',*args],cwd=repo,**kwargs)
def stage_bytes(path, data):
    oid=git('hash-object','-w','--stdin',input=data).decode().strip()
    git('update-index','--cacheinfo',f'100644,{oid},{path}')

# Preserve the current working files; build the index version from HEAD + our entries.
base_text=git('show','HEAD:docs/media/manifest.json').decode('utf-8')
base=json.loads(base_text)
assert not any(v['path'].startswith(prefix) for v in base['videos'])
array_start=base_text.index('[',base_text.index('"videos"'))
_, array_length=json.JSONDecoder().raw_decode(base_text[array_start:])
array_end=array_start+array_length-1
new_entries=',\n'.join('    '+json.dumps(v,ensure_ascii=False,indent=2).replace('\n','\n    ') for v in videos)
staged_manifest=base_text[:array_end].rstrip()+',\n'+new_entries+'\n  '+base_text[array_end:]
assert json.loads(staged_manifest)['videos']==base['videos']+videos
stage_bytes('docs/media/manifest.json',staged_manifest.encode())
readme=git('show','HEAD:docs/media/README.md')
lines=[line for line in (media/'README.md').read_text(encoding='utf-8').splitlines()
       if line.startswith('- ['+prefix)]
assert len(lines)==12
stage_bytes('docs/media/README.md',readme.rstrip()+b'\n\n'+('\n\n'.join(lines)+'\n').encode())

exclude=repo/'.git/info/exclude'
ignore='/research/c2/diagnostics/2026-10-01-arm-capability-development-r3/*.tar.gz'
if ignore not in exclude.read_text(encoding='utf-8'):
    with exclude.open('a',encoding='utf-8') as stream: stream.write('\n'+ignore+'\n')
attributes=repo/'.gitattributes'
for rule in ('research/c2/diagnostics/2026-10-01-arm-capability-development-r3/** -text whitespace=cr-at-eol',
             'docs/media/c2-pick-arm-2026-10-01-arm-dev-*/** -text whitespace=cr-at-eol'):
    if rule not in attributes.read_text(encoding='utf-8'):
        with attributes.open('a',encoding='utf-8',newline='\n') as stream: stream.write('\n'+rule+'\n')
git('add','--','.gitattributes','scripts/run_arm_capability_batch.py',
    'scripts/collect_arm_capability_archive.py','tests/test_eef_arm_runtime_cpu.py',
    diag.relative_to(repo).as_posix(),*[f.relative_to(repo).as_posix() for f in folders])
git('add','-f','--',*[str((media/v['path']).relative_to(repo)) for v in videos])
print(json.dumps(dict(videos=36,physical_processes=12,archive_sha256=h.hexdigest(),staged=True)))
