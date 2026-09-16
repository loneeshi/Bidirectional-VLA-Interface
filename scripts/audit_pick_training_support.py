"""Compare archived v8 training segment starts with the diagnostic start.

Recovery segments begin partway through a policy rollout and must not be
interpreted as independent navigation endpoints. No model inference occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--norm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    selection_path=a.backup/'v8-data/probe/evidence/v8-teacher-selection.json'
    selection=json.loads(selection_path.read_text())
    ref=json.loads(a.reference.read_text())
    q=np.asarray(ref['qpos'][0]); v=np.asarray(ref['qvel'][0])
    norm=json.loads(a.norm.read_text())['norm_stats']['state']
    state=np.concatenate([q,v]); state[:2]=0
    lower=np.asarray(norm['q01']); upper=np.asarray(norm['q99'])
    rows=[]
    for source in selection['sources']:
        suffix=source['run'].split('/workspace/bvi/',1)[1]
        path=a.backup/suffix
        if not path.exists(): path=a.backup/'v8-data'/suffix
        events_path=path/'events.jsonl'
        events=[json.loads(line) for line in events_path.read_text().splitlines()]
        for segment in source['segments']:
            if segment['skill']!='pick' or segment['subtask_index']!=1: continue
            demo=next(e for e in events if e['event']=='demonstration_step' and e['subtask_index']==1)
            pq=np.asarray(demo['qpos'][0]); pv=np.asarray(demo['qvel'][0])
            rows.append(dict(source=suffix,recovery='workspace-recovery-teachers' in suffix,
                events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
                frames=segment['frames'],base_xy=pq[:2].tolist(),yaw=float(pq[2]),
                yaw_velocity=float(pv[2]),base_xy_distance_m=float(np.linalg.norm(q[:2]-pq[:2])),
                wrapped_yaw_difference_rad=float(np.arctan2(np.sin(q[2]-pq[2]),np.cos(q[2]-pq[2])))))
    outside=np.where((state<lower-1e-6)|(state>upper+1e-6))[0]
    result=dict(selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        norm_sha256=hashlib.sha256(a.norm.read_bytes()).hexdigest(),
        total_selected_frames=selection['frames'],selected_sources=len(selection['sources']),
        selected_recovery_segments=selection['recovery_segments'],
        full_pick_starts=sum(not r['recovery'] for r in rows),
        initial_base_xy=q[:2].tolist(),initial_yaw=float(q[2]),initial_yaw_velocity=float(v[2]),
        initial_state_outside_training_quantiles=[dict(index=int(i),value=float(state[i]),
            q01=float(lower[i]),q99=float(upper[i])) for i in outside],
        pick_segment_starts=rows,
        limitations=['Marginal quantile inclusion does not establish joint visual/state distribution support.',
                     'Only first-object Pick starts are compared; recovery starts are mid-rollout.',
                     'No causal policy intervention or online normalization test was performed.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Saved',a.output,'with',len(rows),'Pick segment starts')


if __name__=='__main__': main()
