"""CPU-only verified assembly/extraction of an existing Fetch V8 checkpoint.

Waits for the SFTP completion marker; never considers file length alone evidence
of completion. Does not load a model or start any GPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-seconds', type=int, default=3600)
    a = p.parse_args()
    root = Path.home()/'bvi-research'
    prefix = root/'checkpoints/fetch-v8-parallel'
    target = root/'checkpoints/fetch-v8'
    staging = root/'checkpoints/fetch-v8-staging'
    archive = root/'checkpoints/fetch-v8-assembled.tar.gz'
    marker = Path(str(prefix)+'.parts-complete.json')
    expected_size = 6337444039
    expected_sha = 'c78c75642e2b23b07d4ae56dcdf3682ac00590916ba718133917f33d31f9300d'
    receipt = root/'fetch-v8-stage-result.json'
    report = {'status':'waiting_for_transfer', 'gpu_used':False,
              'expected_bytes':expected_size,'expected_sha256':expected_sha}
    def save():
        receipt.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        if target.exists() or staging.exists() or archive.exists():
            raise FileExistsError('Refuse to overwrite staged checkpoint/archive')
        until = time.monotonic()+a.wait_seconds
        while not marker.exists():
            if time.monotonic()>until: raise TimeoutError('Transfer marker not received')
            time.sleep(10)
        completion = json.loads(marker.read_text())
        if completion['parts'] != 4 or completion['bytes'] != expected_size:
            raise ValueError('Unexpected transfer manifest')
        if shutil.disk_usage(root).free < 30*1024**3:
            raise RuntimeError('Require30GiB free before assembling checkpoint')
        report['status']='assembling';save()
        digest = hashlib.sha256()
        with archive.open('xb') as out:
            for index in range(4):
                part = Path(str(prefix)+f'.part{index:02d}')
                part_size = expected_size*(index+1)//4-expected_size*index//4
                if part.stat().st_size != part_size: raise ValueError('Incomplete part')
                with part.open('rb') as source:
                    for data in iter(lambda: source.read(8*1024*1024), b''):
                        digest.update(data);out.write(data)
        if digest.hexdigest()!=expected_sha or archive.stat().st_size!=expected_size:
            raise ValueError('Assembled archive SHA/size mismatch')
        report.update(status='extracting', archive_sha256=digest.hexdigest());save()
        staging.mkdir()
        with tarfile.open(archive, 'r:gz') as handle:
            for item in handle:
                path = PurePosixPath(item.name)
                if path.is_absolute() or '..' in path.parts or item.issym() or item.islnk():
                    raise ValueError('Unsafe checkpoint archive member')
                if path.parts and path.parts[0] not in ('params','assets'):
                    raise ValueError('Unexpected checkpoint archive root')
                handle.extract(item, path=staging, filter='data')
        contract = staging/'assets/bvi/fetch-seed1-workspace-recovery-v8/bvi-state-contract.json'
        norm = contract.with_name('norm_stats.json')
        if not contract.is_file() or not norm.is_file() or not (staging/'params').is_dir():
            raise ValueError('Checkpoint required assets missing')
        report['state_contract_sha256']=hashlib.sha256(contract.read_bytes()).hexdigest()
        report['normalizer_sha256']=hashlib.sha256(norm.read_bytes()).hexdigest()
        staging.rename(target)
        report.update(status='verified_checkpoint_staged', checkpoint=str(target))
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        save()
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
