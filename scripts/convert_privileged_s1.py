"""CPU full privileged42 diagnostic export; ordinary S1 only, no S2 admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scale',choices=['medium'],default='medium')
    p.add_argument('--task',choices=['pick'],default='pick')
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='2'
    import h5py
    from bvi.official_fetch_data import parent_ids
    from bvi.privileged_fetch_data import inspect_episode,policy_frame,SCHEMA,EXTRA_FIELDS
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='checking_source',question='Independent privileged-state S1 diagnostic; not faithful native gate',
        task=a.task,scale=a.scale,gpu_runs=0,api_calls=0,training_updates=0,episodes=[],
        state_dim=42,state_components=['native_qpos12','native_qvel12']+[name for name,_ in EXTRA_FIELDS],
        policy_state_schema=SCHEMA,privileged_policy_inputs=True,
        state_mapping='SequentialTaskEnv._get_obs_agent removes robot base indices0:3 from qpos/qvel',
        source_state30_compatibility=False,progress_contract='current_observation_v2',
        annotation_only_keys=[],
        policy_input_keys=['image','wrist_image','state','task'],parent_split=parent_ids(a.scale))
    def save():
        report['seconds']=time.monotonic()-started
        (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        source=a.source/a.task/'013_apple.h5';meta_path=source.with_suffix('.json')
        audit=json.loads((a.source/'source-audit.json').read_text())
        if audit['revision']!='3e58f01aa0fd8484de9b913fa8a1c6a099884fcf':raise ValueError('Unpinned dataset revision')
        actual=sha(source)
        if actual!=audit['datasets'][a.task]['sha256']:raise ValueError('Source file checksum differs')
        metadata=json.loads(meta_path.read_text());parents={e['episode_id']:e for e in metadata['episodes']}
        report.update(source_sha256=actual,source_json_sha256=sha(meta_path),source_env=metadata['env_info'],status='converting')
        with h5py.File(source) as h:
            for split,ids in report['parent_split'].items():
                dataset=None
                if not a.audit_only:
                    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
                    dataset=LeRobotDataset.create(repo_id=f'bvi/s1-privileged42-{a.task}-{a.scale}-{split}',root=a.output/split,
                        fps=20,robot_type='fetch',use_videos=False,image_writer_threads=2,image_writer_processes=0,
                        features={'image':{'dtype':'image','shape':(128,128,3),'names':['height','width','channel']},
                            'wrist_image':{'dtype':'image','shape':(128,128,3),'names':['height','width','channel']},
                            'state':{'dtype':'float32','shape':(42,),'names':['state']},
                            'actions':{'dtype':'float32','shape':(13,),'names':['actions']}})
                for ident in ids:
                    parent=parents[ident]
                    if f'set_table-{a.task}-train-' not in parent['subtask_uid'] or parent['control_mode']!='pd_joint_delta_pos':
                        raise ValueError('Parent task/split/control differs from declared official train source')
                    group=h[f'traj_{ident}'];ep=inspect_episode(group,a.task)
                    instruction='Pick and stably hold the apple.' if a.task=='pick' else 'Place the held apple at its placement goal and retract.'
                    if dataset is not None:
                        for t in range(ep['exported_steps']):dataset.add_frame(policy_frame(group,ep,t,instruction))
                        dataset.save_episode()
                    row={k:v for k,v in ep.items() if k not in ('state','actions')}
                    row.update(trajectory=f'traj_{ident}',parent_id=ident,split=split,subtask_uid=parent['subtask_uid'],instruction=instruction)
                    report['episodes'].append(row);save()
        report['status']='audited_no_export' if a.audit_only else 'converted_not_training_validated'
        report['exported_frames']=sum(x['exported_steps'] for x in report['episodes'])
        report['family_counts']={f:sum(w['family']==f for e in report['episodes'] for w in e['windows']) for f in ['reach','grasp','move','release']}
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
