"""Inspect completed teacher episodes; never train until fixed collection complete."""
import json
from pathlib import Path
import h5py
import numpy as np
from bvi.fetch_segments import segment_episode

root=Path.home()/'bvi-research/runs/fetch-tapt-teacher-2026-09-16'
collection=json.loads((root/'collection.json').read_text())
report={'collection_status':collection['status'],'label_version':'observable-pilot-v1',
        'thresholds':{'reach_metres':.08,'place_metres':.15,'stable_grasp_observations':3},
        'windows':[],'episodes':[],'training_ready':False,
        'handoff_coverage':'native-start only; actual navigation handoff pending'}
for ep in collection['episodes']:
    directory=root/ep['task']/f"seed{ep['seed']}"
    events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
    success=[x['step'] for x in events if x['info']['success'][0]]
    with h5py.File(directory/'trajectory.h5') as h:
        groups=[h['observations'][k]['extra'] for k in sorted(h['observations'])]
        grasp=[bool(g['is_grasped'][0]) for g in groups]
        dist=[float(np.linalg.norm(g['tcp_pose_wrt_base'][0,:3]-g['obj_pose_wrt_base'][0,:3])) for g in groups]
        goal=[float(np.linalg.norm(g['goal_pos_wrt_base'][0]-g['obj_pose_wrt_base'][0,:3])) for g in groups]
        windows=segment_episode(ep['task'],grasp,dist,goal,success)
    for w in windows:w.update(task=ep['task'],seed=ep['seed'],split=ep['split'],trajectory=str(directory/'trajectory.h5'))
    report['windows']+=windows
    report['episodes'].append({**ep,'labeled_families':[w['family'] for w in windows]})
report['counts']={split:{f:sum(w['split']==split and w['family']==f for w in report['windows'])
                           for f in ['reach','grasp','move','release']} for split in ['train','validation']}
report['all_families_present']=all(n>0 for counts in report['counts'].values() for n in counts.values())
(root/'segments.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['windows','episodes']},indent=2))
