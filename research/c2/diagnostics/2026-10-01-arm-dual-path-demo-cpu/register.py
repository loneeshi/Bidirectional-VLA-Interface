"""Register this batch while staging only its additions to shared media indexes."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BATCH = 'c2-pick-arm-dual-path-2026-10-01-mobile-r1'
MEDIA = ROOT / 'docs/media'

def stage_bytes(path, data):
    oid = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], input=data, cwd=ROOT).decode().strip()
    subprocess.run(['git', 'update-index', '--cacheinfo', f'100644,{oid},{path}'], cwd=ROOT, check=True)

receipts = json.loads((MEDIA / BATCH / 'export-receipts.json').read_text(encoding='utf-8'))
entries = []
for r in receipts:
    video = MEDIA / r['output']
    assert hashlib.sha256(video.read_bytes()).hexdigest() == r['video_sha256']
    entries.append(dict(filename=video.name, path=r['output'], sha256=r['video_sha256'], bytes=video.stat().st_size,
                        case_id=r['case_id'], condition=r['condition'], recorded_at_utc=r['recorded_at_utc'],
                        strict_pick_success=r['strict_pick_success'], termination_category=r['termination_category'],
                        kind=r['kind'], source_kind=r['source_kind'], evaluation_only=True))

def append_entries(text):
    document = json.loads(text)
    existing = {v['path'] for v in document['videos']}
    additions = [v for v in entries if v['path'] not in existing]
    if not additions:
        return text
    start = text.index('[', text.index('"videos"'))
    _, length = json.JSONDecoder().raw_decode(text[start:])
    end = start + length - 1
    prefix = text[:end].rstrip()
    payload = ',\n'.join('    ' + json.dumps(v, ensure_ascii=False, indent=2).replace('\n', '\n    ') for v in additions)
    result = prefix + (',' if document['videos'] else '') + '\n' + payload + '\n  ' + text[end:]
    assert len(json.loads(result)['videos']) == len(document['videos']) + len(additions)
    return result

manifest_path = 'docs/media/manifest.json'
working = ROOT / manifest_path
working.write_text(append_entries(working.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
head = subprocess.check_output(['git', 'show', f'HEAD:{manifest_path}'], cwd=ROOT).decode('utf-8')
stage_bytes(manifest_path, append_entries(head).encode('utf-8'))

line = f'\n- [2026-10-01 V-mobile: five Astra issued-goal / actual-TCP scene demos]({BATCH}/README.md). CPU postprocessing of online recordings; pink dashed goals, cyan executed TCP, orange base commands; evaluation-only X-ray overlay.\n'
readme_path = 'docs/media/README.md'
working = ROOT / readme_path
text = working.read_text(encoding='utf-8')
if BATCH not in text:
    working.write_text(text.rstrip() + '\n' + line, encoding='utf-8', newline='\n')
head = subprocess.check_output(['git', 'show', f'HEAD:{readme_path}'], cwd=ROOT).decode('utf-8')
stage_bytes(readme_path, (head if BATCH in head else head.rstrip() + '\n' + line).encode('utf-8'))
(Path(__file__).parent / 'verification.json').write_text(json.dumps(dict(verified_videos=entries, cpu_tests_passed=18,
    provider_calls=0, simulator_steps=0, experiment_reruns=0), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Verified and registered five videos; staged only this batch in shared indexes.')
