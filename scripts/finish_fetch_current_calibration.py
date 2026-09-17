"""Finish an already-running bounded lab calibration; never starts training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    report_path = args.training / 'result.json'
    report = json.loads(report_path.read_text())
    deadline = datetime.fromisoformat(report['started_at']).timestamp() + 2400 + 30
    while report['status'] not in ('failed', 'bounded_calibration_complete_not_online_evaluated'):
        if time.time() >= deadline:
            raise TimeoutError('Existing training did not complete within its original bound; no evaluation')
        time.sleep(10)
        report = json.loads(report_path.read_text())
    if report['status'] == 'failed':
        raise RuntimeError('Calibration failed; no evaluation')
    # Report is written just before CUDA teardown. Wait at most30s for release.
    for _ in range(15):
        used = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'], text=True).strip())
        if used < 1024:
            break
        time.sleep(2)
    else:
        raise RuntimeError('GPU1 remains occupied; no overlapping evaluation')
    subprocess.run([str(root / 'envs/lightnav/bin/python'), str(root / 'summarize_fetch_current_calibration.py'),
                    str(args.training)], cwd=root, check=True,
                   env={**__import__('os').environ, 'CUDA_VISIBLE_DEVICES': ''})
    subprocess.run(['timeout', '-k', '15s', '630s', str(root / 'envs/acdit/bin/python'),
                    str(root / 'evaluate_fetch_current_calibration.py'), '--checkpoint',
                    str(args.training / 'best.pt'), '--output', str(args.evaluation)], cwd=root, check=True)
    (args.training / 'pipeline-complete.json').write_text(json.dumps(dict(
        completed_at=datetime.now(timezone.utc).isoformat(), evaluation=str(args.evaluation),
        note='Evaluation process completed; consult native per-episode success flags.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
