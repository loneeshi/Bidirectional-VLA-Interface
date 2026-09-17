"""CPU-only audit of the physical states behind local temporal progress labels."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np

root = Path.home()/'bvi-research'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True, help='Existing audit directory relative to ~/bvi-research')
args = parser.parse_args()
sys.path.insert(0, str(root/'src/Bidirectional-VLA-Interface/src'))
from bvi.progress_monitor import ProgressMonitor, THRESHOLDS

data = root/'runs/fetch-tapt-teacher-2026-09-16'
windows = json.loads((data/'segments.json').read_text())['windows']
rows = []
for w in windows:
    with h5py.File(w['trajectory']) as h:
        def state(t):
            extra = h[f'observations/{t:04d}/extra']
            tcp = extra['tcp_pose_wrt_base'][0,:3]
            obj = extra['obj_pose_wrt_base'][0,:3]
            return dict(t=t, tcp_object_distance=float(np.linalg.norm(tcp-obj)),
                        is_grasped=bool(extra['is_grasped'][:].reshape(-1)[0]))
        sampled = []
        for t in sorted({w['start'],(w['start']+w['end']-1)//2,w['end']-1}):
            sampled.append(dict(**state(t), target_first=(t+1-w['start'])/(w['end']-w['start'])))
        # Counterfactual ideal temporal predictor, not a learned-policy rollout.
        # Two-frame chunk, decisions before actions, same two-hit threshold logic.
        monitor = ProgressMonitor(w['family'], 0)
        ideal_trigger = None
        for t in range(w['start'],w['end'],2):
            value = (t+1-w['start'])/(w['end']-w['start'])
            event = monitor.update(value)
            if event:
                ideal_trigger = dict(**state(t), target_first=value, trigger=event)
                break
        rows.append(dict(window=w, samples=sampled, start_state=state(w['start']),
                         end_state=state(w['end']), ideal_temporal_monitor_trigger=ideal_trigger))
out = root/args.out/'labels.json'
assert out.parent.is_dir(), 'Run the paired audit first'
out.write_text(json.dumps(dict(scope='label/support audit; no learned-policy actions',
    thresholds=THRESHOLDS,windows=rows),indent=2))
print(json.dumps({'windows':len(rows),'output':str(out)}))
