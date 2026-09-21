"""Create/resume the 1000-row manifest and bind the next real val resets.

This performs no policy action and no model request. It only records the exact
task-plan identity sampled by each deterministic environment seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

from bvi import JsonlLogger
from bvi.bridge import atomic_json
from bvi.mshab_adapter import jsonable, make_mshab_adapter
from bvi.sac_interface_baseline import DEFAULT_BATCH_SIZE, bind_row, new_manifest, validate_manifest


GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if not 1 <= args.count <= DEFAULT_BATCH_SIZE:
        parser.error(f'count must be within 1..{DEFAULT_BATCH_SIZE}')
    used = int(subprocess.check_output(
        ['nvidia-smi', '-i', GPU, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True).strip())
    if used >= 1024:
        raise RuntimeError('GPU1 occupied; binding refused')
    manifest = (json.loads(args.manifest.read_text(encoding='utf-8'))
                if args.manifest.is_file() else new_manifest())
    rows = validate_manifest(manifest)
    targets = [row for row in rows if row['status'] == 'unbound'][:args.count]
    if not targets:
        print(json.dumps({'status': 'nothing_to_bind'}))
        return
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    from mani_skill import ASSET_DIR
    from mani_skill.envs.sapien_env import BaseEnv
    from mshab.envs.make import EnvConfig

    plan_path = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json'
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    env_kwargs = {'require_build_configs_repeated_equally_across_envs': False,
                  'add_event_tracker_info': True,
                  'human_render_camera_configs': {'width': 512, 'height': 512},
                  'task_cfgs': {'navigate': {'ignore_arm_checkers': True}}}
    if 'invisible_goals_in_human_render' in inspect.signature(BaseEnv.__init__).parameters:
        env_kwargs['invisible_goals_in_human_render'] = True
    config = EnvConfig(env_id='BVISequentialWorkspaceCamera-v0', num_envs=1,
        max_episode_steps=7000, task_plan_fp=str(plan_path), obs_mode='rgbd',
        render_mode='rgb_array', record_video=False, info_on_video=False,
        continuous_task=True, frame_stack=3, stationary_base=False,
        stationary_torso=False, stationary_head=True, env_kwargs=env_kwargs)
    # make_mshab_adapter expands task-plan data into this object in place. Keep
    # one pre-mutation identity snapshot instead of duplicating the full val
    # dataset in every manifest row.
    config_identity = jsonable(config)
    import bvi.nav_camera_env  # noqa: F401 - registers the environment
    for row in targets:
        seed = row['seed']
        directory = args.evidence_dir / f'seed-{seed:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        logger = JsonlLogger(directory / 'events.jsonl', f'sac-bind-seed-{seed}')
        # The factory mutates EnvConfig. Each reset must receive a fresh copy.
        row_config = copy.deepcopy(config)
        row_identity = jsonable(row_config)
        if row_identity != config_identity:
            raise RuntimeError('Per-seed environment configuration identity changed')
        adapter = make_mshab_adapter(row_config, logger, directory, seed=seed)
        try:
            subtasks = [{'uid': item.uid, 'type': item.type,
                         'obj_id': getattr(item, 'obj_id', None)}
                        for item in adapter.original_plan.subtasks]
            if len(subtasks) != 20 or sum(item['type'] == 'place' for item in subtasks) != 5:
                raise RuntimeError('TidyHouse plan is not the expected five-object/20-subtask chain')
            evidence = {'method': 'real_environment_reset_no_policy_actions',
                        'plan_sha256': sha256(plan_path), 'environment': row_identity,
                        'environment_identity_stage': 'before_factory_mutation',
                        'environment_config_isolated_per_seed': True,
                        'subtasks': subtasks, 'api_calls': 0, 'training_updates': 0}
            bind_row(manifest, seed, subtasks[0]['uid'], evidence)
            atomic_json(directory / 'binding.json', {'seed': seed,
                        'plan_uid': subtasks[0]['uid'], **evidence})
            atomic_json(args.manifest, manifest)
        finally:
            adapter.close()
    print(json.dumps({'status': 'bound', 'seeds': [row['seed'] for row in targets],
                      'manifest': str(args.manifest)}))


if __name__ == '__main__':
    main()
