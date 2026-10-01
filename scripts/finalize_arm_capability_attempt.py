"""CPU delivery adapter for official spawns; raw robot qpos are never rewritten."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from eef_server_broker import save


def rendering_report(raw):
    result=copy.deepcopy(raw)
    if not result.get('frames'):return result
    states=[result['initial_robot_state']]+result['frames']
    z0=np.asarray(states[0]['base_world_eval_only'])[2,3]
    for state in states:
        pose=np.asarray(state['base_world_eval_only'],dtype=float)
        yaw=math.atan2(pose[1,0],pose[0,0]);c,s=math.cos(yaw),math.sin(yaw)
        if not np.allclose(pose[:3,:3],[[c,-s,0],[s,c,0],[0,0,1]],atol=1e-6,rtol=0) or abs(pose[2,3]-z0)>1e-6:
            raise ValueError('legacy camera adapter requires planar base and constant base z')
        state['qpos_raw']=state['qpos'][:]
        state['qpos'][:3]=[float(pose[0,3]),float(pose[1,3]),yaw]
    result['mode']=result['condition']
    result['rendering_adapter']={'evaluation_only':True,
        'description':'derived qpos[:3] holds measured base-world XY/yaw; constant base z removed from BOTH camera and TCP; raw qpos retained',
        'removed_common_z_m':float(z0)}
    if result.get('pending_command'):
        pending=result['pending_command']
        result['commands'].append({**pending,'result':{'steps_used':len(result['frames'])-pending['start_step'],
            'reason':'censored during command; completion unknown'}})
    return result


def finalize(folder):
    from finalize_eef_visual_attempt import finalize as legacy_finalize
    folder=Path(folder);source=folder/'result.json'
    raw=json.loads(source.read_text());derived=folder/'media-source';derived.mkdir(exist_ok=False)
    report=rendering_report(raw);report['raw_result_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    save(derived/'result.json',report)
    for source_dir in ('recording-frames','sensors'):
        for path in (folder/source_dir).rglob('*'):
            if path.is_file():
                target=derived/path.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);os.link(path,target)
    for request in (folder/'provider').glob('turn-*'):
        for name,subdir in [('request-bundle.json','requests'),('delivery.json','responses')]:
            if (request/name).exists():
                target=derived/subdir/(request.name+'.json');target.parent.mkdir(exist_ok=True);os.link(request/name,target)
    receipt=legacy_finalize(derived)
    receipt.update(case_id=raw['case_id'],condition=raw['condition'],raw_result_sha256=report['raw_result_sha256'],
        recorded_at_utc=raw['recorded_at_utc'],outcome=raw.get('termination_category'),
        raw_qpos_preserved=True,media_directory='media-source/delivery')
    save(folder/'media-receipt.json',receipt)
    return receipt
