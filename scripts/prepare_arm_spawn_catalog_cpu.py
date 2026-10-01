"""Read official Pick assets on CPU; never import an environment or run a policy.

Height is an explicitly labelled initial collision-bottom estimate, not a measured
support contact. The output is a candidate catalog, not execution authorization.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

PIN = 'e9ff3d23496d38e4431c8d913e147ffa007f7f72'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    import trimesh
    runtime = root / 'src/official-mshab-runtime'
    commit = subprocess.check_output(['git', '-C', str(runtime/'mshab'),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN:
        raise ValueError('MS-HAB commit changed')
    assets = root / 'assets/data'
    data = assets / 'scene_datasets/replica_cad_dataset/rearrange'
    split = 'val' if (data/'spawn_data/tidy_house/pick/val/spawn_data.pt').is_file() else 'train'
    spawn_path = data / f'spawn_data/tidy_house/pick/{split}/spawn_data.pt'
    spawn = torch.load(spawn_path, map_location='cpu', weights_only=True)
    model_root = assets / 'assets/mani_skill2_ycb'
    info = json.loads((model_root/'info_pick_v0.json').read_text())
    sources = {str(spawn_path.relative_to(root)): sha(spawn_path),
               str((model_root/'info_pick_v0.json').relative_to(root)): sha(model_root/'info_pick_v0.json')}
    for path in [runtime/'mshab/mshab/envs/subtask.py',
                 runtime/'ManiSkill/mani_skill/utils/scene_builder/replicacad/rearrange/scene_builder.py',
                 runtime/'ManiSkill/mani_skill/utils/building/actors/ycb.py']:
        sources[str(path.relative_to(root))] = sha(path)
    # Official scene builder: Rx(90deg) * episode pose * Rx(-90deg), +.01m z.
    rx = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    rows, episodes, seen = [], {}, set()
    for plan_path in sorted((data/f'task_plans/tidy_house/pick/{split}').glob('*.json')):
        if plan_path.stem == 'all':
            continue
        category = plan_path.stem
        mesh_path = model_root/'models'/category/'collision.ply'
        vertices = np.asarray(trimesh.load(mesh_path, force='mesh', process=False).vertices)
        vertices = vertices * info[category].get('scales', [1.0])[0]
        sources[str(mesh_path.relative_to(root))] = sha(mesh_path)
        sources[str(plan_path.relative_to(root))] = sha(plan_path)
        for plan in json.loads(plan_path.read_text())['plans']:
            sub = plan['subtasks'][0]
            uid = sub['composite_subtask_uids'][0]
            if sub['type'] != 'pick' or uid in seen:
                raise ValueError('unexpected or duplicate Pick UID')
            seen.add(uid)
            tensors = spawn[uid]
            if set(tensors) != {'robot_pos', 'robot_qpos'}:
                raise ValueError('object override in spawn: height reconstruction needs revision')
            n = len(tensors['robot_pos'])
            if tensors['robot_pos'].shape != (n, 3) or tensors['robot_qpos'].shape != (n, 15):
                raise ValueError('spawn shape changed')
            if not all(torch.isfinite(x).all().item() and x.device.type == 'cpu' for x in tensors.values()):
                raise ValueError('invalid spawn tensor')
            episode_name = plan['init_config_name']
            if episode_name not in episodes:
                episode_path = data/episode_name
                episodes[episode_name] = json.loads(episode_path.read_text())
                sources[str(episode_path.relative_to(root))] = sha(episode_path)
            episode = episodes[episode_name]
            cat, number = sub['obj_id'].rsplit('-', 1)
            matches = [matrix for name, matrix in episode['rigid_objs'] if name.split('.')[0] == cat]
            transform = np.asarray(matches[int(number)], dtype=float)
            rotation = rx @ transform[:3,:3] @ rx.T
            position = rx @ transform[:3,3] + [0,0,.01]
            bottom = float((vertices @ rotation.T + position)[:,2].min())
            label = f'{cat}_:{int(number):04d}'
            target_index = list(episode['targets']).index(label)
            receptacle = episode['target_receptacles'][target_index]
            rows.append({'uid':uid, 'object_category':category, 'object_id':sub['obj_id'],
                         'scene':plan['build_config_name'], 'init_config_name':episode_name,
                         'spawn_index_range':[0,n], 'spawn_count':n,
                         'support_receptacle':receptacle,
                         'support_height_estimate_m':bottom,
                         'height_band': 'low' if bottom < .35 else 'middle' if bottom < .70 else 'high'})
    if seen != set(spawn):
        raise ValueError('task plans and spawn UIDs differ')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU-only invariant violated')
    rows.sort(key=lambda x:x['uid'])
    return {'schema':'arm-spawn-catalog-v1', 'mshab_commit':commit, 'split':split,
            'height_definition':'initial collision mesh minimum world z; includes official +0.01m placement offset; support-height estimate, not measured contact',
            'height_band_edges_m':[.35,.70], 'uid_count':len(rows),
            'candidate_spawn_count':sum(r['spawn_count'] for r in rows),
            'category_uid_counts':dict(sorted(Counter(r['object_category'] for r in rows).items())),
            'stratum_uid_counts':dict(sorted(Counter(r['object_category']+'/'+r['height_band'] for r in rows).items())),
            'scene_count':len({r['scene'] for r in rows}), 'rows':rows,
            'source_sha256':sources, 'cuda_initialized':False,
            'environment_created':False, 'api_calls':0, 'gpu_runs':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    result = build(args.lab_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))
