"""Acquire official pi05_base params, pin GCS generations and local SHA256.

CPU only. Caller must impose an outer timeout. No checkpoint loading/training.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path('checkpoints/s1-pi05-base-2026-09-17')
    root.mkdir(parents=True, exist_ok=True)
    report = dict(status='listing', source='gs://openpi-assets/checkpoints/pi05_base/params',
                  initialization='official_pi05_base_not_V8', training_updates=0, files=[])
    def save():
        (root/'acquisition.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        prefix = 'checkpoints/pi05_base/params/'
        items, token = [], None
        while True:
            params = {'prefix':prefix}
            if token: params['pageToken'] = token
            url = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?' + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=60) as response:
                page = json.load(response)
            items.extend(page.get('items', []))
            token = page.get('nextPageToken')
            if not token: break
        size = sum(int(x['size']) for x in items)
        if not items or size > 20_000_000_000 or shutil.disk_usage(root).free < size*2:
            raise ValueError('Empty source, >20GB cap or insufficient free storage')
        report.update(status='downloading', source_bytes=size, object_count=len(items))
        save()
        for item in items:
            relative = Path(item['name'][len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe source object name')
            target = root/'params'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            url = ('https://storage.googleapis.com/storage/v1/b/openpi-assets/o/'
                   + urllib.parse.quote(item['name'], safe='') + '?alt=media&generation=' + item['generation'])
            partial = target.with_name(target.name+'.download-partial')
            h = hashlib.sha256(); count = 0
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
                while chunk := response.read(8*1024*1024):
                    out.write(chunk);h.update(chunk);count += len(chunk)
            if count != int(item['size']):
                raise ValueError('Downloaded length differs from pinned object')
            partial.replace(target)
            report['files'].append(dict(path=str(relative),generation=item['generation'],
                                        bytes=count,sha256=h.hexdigest(),gcs_crc32c=item.get('crc32c')))
            save()
        report['status'] = 'downloaded_not_loaded'
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
