"""Bounded CPU audit of the pinned official Pick corpus. No model or simulator imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import h5py
    import numpy as np
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((a.source/'source-audit.json').read_text())
    h5 = a.source/'pick/013_apple.h5'
    meta_path = h5.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    source_sha = sha(h5)
    if (audit['revision'] != '3e58f01aa0fd8484de9b913fa8a1c6a099884fcf'
            or source_sha != audit['datasets']['pick']['sha256']):
        raise ValueError('Pinned source differs')
    episodes = sorted(meta['episodes'], key=lambda e: e['episode_id'])
    scenes = sorted({e['build_config_idx'] for e in episodes})
    dev_scenes = sorted(np.random.default_rng(20260919).choice(scenes, max(1,len(scenes)//5), replace=False).tolist())
    rows, arrays, state_arrays, keys, phases, init_hashes = [], [], [], [], [], set()
    with h5py.File(h5, 'r') as handle:
        for parent in episodes:
            if time.monotonic()-started > 850:
                raise TimeoutError('CPU audit exceeded bound')
            ident = parent['episode_id']; group = handle[f'traj_{ident}']
            actions = np.asarray(group['actions'], dtype=np.float32)
            success = np.asarray(group['success']); fail = np.asarray(group['fail'])
            stops = np.flatnonzero(success|fail)
            end = int(stops[0])+1 if len(stops) else len(actions)
            passed = bool(success[end-1] and not fail[end-1])
            if actions.shape[1:] != (13,) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001:
                raise ValueError('Invalid controller actions')
            if not passed:
                rows.append({'parent_id':ident,'excluded':'no_success_before_native_failure','first_stop':end})
                continue
            qpos = np.asarray(group['obs/agent/qpos'][:end+1], dtype=np.float32)
            qvel = np.asarray(group['obs/agent/qvel'][:end+1], dtype=np.float32)
            extra = group['obs/extra']
            pieces = [qpos[:end],np.asarray(extra['tcp_pose_wrt_base'][:end]),
                      np.asarray(extra['obj_pose_wrt_base'][:end]),np.asarray(extra['goal_pos_wrt_base'][:end]),
                      np.asarray(extra['is_grasped'][:end],dtype=np.float32)[:,None]]
            state = np.concatenate(pieces,axis=1).astype(np.float32)
            if state.shape != (end,30) or not np.isfinite(state).all():
                raise ValueError('Invalid qpos12+privileged18 state')
            tcp=np.asarray(extra['tcp_pose_wrt_base'][:end,:3]); obj=np.asarray(extra['obj_pose_wrt_base'][:end,:3])
            held=np.asarray(extra['is_grasped'][:end],dtype=bool)
            phase=np.where(held,'move',np.where(np.linalg.norm(tcp-obj,axis=-1)<0.08,'grasp','reach'))
            split='validation' if parent['build_config_idx'] in dev_scenes else 'train'
            initial=hashlib.sha256(qpos[0].tobytes()+qvel[0].tobytes()+state[0,12:].tobytes()).hexdigest()
            init_hashes.add(initial)
            rows.append({'parent_id':ident,'trajectory':f'traj_{ident}','split':split,'scene_index':parent['build_config_idx'],
                         'task_plan_idx':parent['task_plan_idx'],'subtask_uid':parent['subtask_uid'],'init_config_idx':parent['init_config_idx'],
                         'spawn_selection_idx':parent['spawn_selection_idx'],'initial_state_subset_sha256':initial,
                         'original_steps':len(actions),'exported_steps':end,'native_success':True})
            arrays.append(actions[:end]); state_arrays.append(state)
            keys.extend([(ident,t) for t in range(end)]); phases.extend(phase.tolist())
        schema={}
        handle['traj_0'].visititems(lambda name,obj: schema.update({name:{'shape':list(obj.shape),'dtype':str(obj.dtype)}}) if isinstance(obj,h5py.Dataset) else None)
    action=np.concatenate(arrays); states=np.concatenate(state_arrays); keys=np.asarray(keys)
    assignments={r['parent_id']:r['split'] for r in rows if 'split' in r}
    train=np.asarray([assignments[int(k[0])]=='train' for k in keys]); phases=np.asarray(phases)
    channel_stats={}
    for name,index in [('torso',10),('base_forward',11),('base_yaw',12)]:
        channel_stats[name]={}
        for phase in ['all','reach','grasp','move']:
            mask=train & (True if phase=='all' else phases==phase)
            v=action[mask,index]; hist,bins=np.histogram(v,bins=np.linspace(-1,1,41))
            channel_stats[name][phase]={'count':len(v),'mean':float(v.mean()),'std':float(v.std()),
                'zero_rmse':float(np.sqrt(np.mean(v*v))),'negative_fraction':float(np.mean(v<-.05)),
                'near_zero_fraction':float(np.mean(np.abs(v)<=.05)),'positive_fraction':float(np.mean(v>.05)),
                'histogram':hist.tolist(),'bins':bins.tolist()}
    norms={}
    for name,values in [('state',states[train]),('actions',action[train])]:
        norms[name]={'mean':values.mean(0).tolist(),'std':values.std(0).tolist(),
                     'q01':np.quantile(values,.01,axis=0).tolist(),'q99':np.quantile(values,.99,axis=0).tolist()}
    np.savez_compressed(a.output/'audit-arrays.npz',state=states,actions=action,row_keys=keys,train=train,phase=phases)
    (a.output/'norm_stats.json').write_text(json.dumps({'norm_stats':norms},indent=2))
    report={'schema':'bvi.pi05-recipe-data-audit/1','status':'cpu_audited_not_training_started','source':str(h5),
            'source_sha256':source_sha,'source_metadata_sha256':sha(meta_path),'source_revision':audit['revision'],
            'source_episodes':len(episodes),'accepted_episodes':sum('split' in r for r in rows),
            'scene_count':len(scenes),'train_scenes':sorted(set(scenes)-set(dev_scenes)),'validation_scenes':dev_scenes,
            'unique_initial_state_subset_hashes':len(init_hashes),'unique_subtask_uids':len({e['subtask_uid'] for e in episodes}),
            'split_policy':'deterministic scene-disjoint 80/20 over build_config_idx, seed20260919; no online checkpoint selection',
            'split_counts':{s:{'parents':sum(r.get('split')==s for r in rows),'frames':sum(r['exported_steps'] for r in rows if r.get('split')==s)} for s in ['train','validation']},
            'episodes':rows,'source_schema':schema,'channel_statistics':channel_stats,'normalizer_file':'norm_stats.json',
            'normalizer_sha256':sha(a.output/'norm_stats.json'),'arrays_sha256':sha(a.output/'audit-arrays.npz'),
            'normalizer_scope':'train scenes only; current obs_t to action_t; cutoff first native success/failure; no endpoint zero action',
            'candidate_state_schema':'native_qpos12_tcp7_object7_goal3_grasp1_v1',
            'native_qpos3_semantics':'head_tilt_joint, not a mobile-base state channel; corroborate named joint mapping',
            'train_qpos3_range':[float(states[train,3].min()),float(states[train,3].max())],
            'missing_acdit_fields':['base_linear_vel','base_angular_vel','precomputed_pointcloud'],
            'limitations':['Marginal yaw histogram cannot establish conditional multimodality or flow-matching mode collapse.',
                           'Small-model failure cannot rule out dataset learnability.',
                           'Initial-state subset hashes do not prove full physics-state independence.',
                           'Fresh pi05_base initialization is required to avoid previously trained adapters seeing held-out scenes.'],
            'wall_seconds':time.monotonic()-started,'gpu_operations':0,'model_updates':0,'api_calls':0}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['status','source_episodes','accepted_episodes','scene_count','split_counts','unique_initial_state_subset_hashes','wall_seconds']}))


if __name__=='__main__':
    main()
