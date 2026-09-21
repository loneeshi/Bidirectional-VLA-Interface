"""Validate newly collected AC-DiT HDF5 and freeze a scene-disjoint training roster."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
os.environ['CUDA_VISIBLE_DEVICES']=''


def scene_split(episodes, seed=20260919):
    import numpy as np
    scenes=sorted({int(e['build_config_idx']) for e in episodes})
    if len(scenes)<10: raise ValueError('Insufficient scene diversity')
    validation=sorted(np.random.default_rng(seed).choice(scenes,len(scenes)//5,replace=False).tolist())
    train=[e['episode_id'] for e in episodes if e['build_config_idx'] not in validation]
    dev=[e['episode_id'] for e in episodes if e['build_config_idx'] in validation]
    if len(train)<200 or len(dev)<30: raise ValueError('Insufficient train/validation coverage')
    return train,dev,validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    import h5py
    import numpy as np
    root=Path.home()/'bvi-research'
    report=json.loads((a.data/'collection.json').read_text())
    assert report['status']=='complete' and report['successes']>=300
    accepted=[e for e in report['episodes'] if e['success']]
    # Resumed infrastructure collection may revisit a deterministic reset.
    # Keep all raw evidence but never double-count identical reset identities.
    seen=set();episodes=[];duplicates=[]
    for e in accepted:
        identity=(e['seed'],e['env_index'],e['build_config_idx'],e['init_config_idx'],e['task_plan_idx'])
        if identity in seen:duplicates.append(e['episode_id']);continue
        seen.add(identity);episodes.append(e)
    if len(episodes)<300:raise ValueError('Fewer than 300 unique accepted starts')
    train,dev,scenes=scene_split(episodes)
    path=a.data/'set_table/pick/train/013_apple/demonstrations.h5'
    sums=np.zeros(13,dtype=np.float64);count=0;frames={'train':0,'validation':0}
    with h5py.File(path) as h:
        assert len(h)==report['successes']
        for e in episodes:
            g=h[f'traj_{e["episode_id"]}'];n=len(g['actions']);actions=g['actions'][:]
            assert actions.shape==(n,13) and np.isfinite(actions).all() and np.max(np.abs(actions))<=1
            assert g['success'][-1] and not g['fail'][:].any() and not g['success'][:-1].any()
            for name,width in [('base_linear_vel',3),('base_angular_vel',3)]:
                assert g[f'obs/extra/{name}'].shape==(n+1,width)
            assert g['obs/pointcloud/xyzrgb'].shape==(n+1,1024,6)
            for key in ['base_linear_vel','base_angular_vel','goal_pos_wrt_base','tcp_pose_wrt_base','obj_pose_wrt_base','is_grasped']:
                assert np.isfinite(g['obs/extra/'+key][:]).all(),key
            pc=g['obs/pointcloud/xyzrgb'][:]
            assert np.isfinite(pc).all() and (pc[:,:,3:]>=0).all() and (pc[:,:,3:]<=1).all()
            assert np.any(np.abs(pc[:,:,:3])>0)
            q=g['obs/agent/qpos'][:n]
            state=np.column_stack([q[:,[2,4,5,6,7,8,9,10,1,3,0]],
                g['obs/extra/base_linear_vel'][:n,0],g['obs/extra/base_angular_vel'][:n,2]])
            assert state.shape==(n,13) and np.isfinite(state).all()
            role='train' if e['episode_id'] in train else 'validation';frames[role]+=n
            if role=='train':sums+=state.sum(0);count+=n
    universal=np.zeros(128);universal[[0,1,2,3,4,5,6,10,125,126,127,100,102]]=sums/count
    instructions=root/'runs/acdit-rdt-preparation-20260919/instructions'
    shutil.copytree(instructions,path.parent/'instructions')
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
    manifest={'status':'frozen','source':str(path),'source_sha256':sha.hexdigest(),'episodes':report['successes'],
        'train_parents':train,'validation_parents':dev,'validation_scenes':scenes,
        'scene_count':len({e['build_config_idx'] for e in episodes}),'frames':frames,
        'train_universal_state_mean':universal.tolist(),'source_kind':'newly_collected_ACDiT_native_SAC',
        'not_the_original_official_H5':True,'split_seed':20260919,'selection':'before any model optimization',
        'unique_episodes':len(episodes),'excluded_duplicate_parents':duplicates}
    (a.data/'training-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['train_parents','validation_parents','train_universal_state_mean']}))


if __name__=='__main__':main()
