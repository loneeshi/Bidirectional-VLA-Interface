"""Official train-spawn SAC diagnostic; no GPT, no training, no early fail stop.

Every case is a separate process. A completed case is never rerun on resume.
Native success is recorded, not conflated with collision-free chain success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def cases():
    return [dict(skill=skill, category=category, seed=seed, spawn_index=spawn)
            for skill in ('pick', 'place')
            for category in ('024_bowl', '009_gelatin_box')
            for seed in (100, 101, 102) for spawn in (0, 1)]


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(a):
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    from dacite import from_dict
    from omegaconf import OmegaConf
    from mshab.envs.make import EnvConfig, make_env
    from bvi.logging import JsonlLogger
    from bvi.mshab_adapter import load_rl_policy, jsonable

    case = cases()[a.case]
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out / 'events.jsonl', out.name)
    seed = case['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = a.checkpoints / 'rl/tidy_house' / case['skill'] / case['category']
    cfg_path = checkpoint / 'config.yml'
    raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    cfg = dict(raw['eval_env'])
    data = a.assets / 'data/scene_datasets/replica_cad_dataset/rearrange'
    plan = data / 'task_plans/tidy_house' / case['skill'] / 'train' / (case['category'] + '.json')
    spawn = data / 'spawn_data/tidy_house' / case['skill'] / 'train/spawn_data.pt'
    cfg.update(num_envs=1, max_episode_steps=a.steps, continuous_task=True,
               task_plan_fp=str(plan), spawn_data_fp=str(spawn), record_video=True)
    # Use native checkpoint depth input. This is an isolated diagnostic, not
    # the RGB-D matched long-horizon comparison.
    env_cfg = from_dict(EnvConfig, cfg)
    env = make_env(env_cfg, video_path=out / 'videos')
    env.env.auto_reset = False
    started = time.monotonic()
    try:
        obs, info = env.reset(seed=seed, options={'spawn_selection_idxs': [case['spawn_index']]})
        uenv = env.unwrapped
        uid = str(uenv.task_plan[0].composite_subtask_uids[0])
        torch.save({'state': uenv.get_state_dict(), 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all(), 'observation': obs}, out / 'initial.pt')
        binding = dict(case, uid=uid, scene=jsonable(getattr(uenv, 'build_config_idxs', None)),
                       config_sha=sha(cfg_path), checkpoint_sha=sha(checkpoint / 'policy.pt'),
                       plan_sha=sha(plan), spawn_sha=sha(spawn), initial_sha=sha(out / 'initial.pt'),
                       env_config=cfg, restore_claim='archive only; wrapper replay not validated')
        save(out / 'binding.json', binding)
        adapter = SimpleNamespace(uenv=uenv, logger=logger, observe=lambda: SimpleNamespace(policy=obs))
        policy = load_rl_policy(cfg_path, checkpoint / 'policy.pt', adapter)
        ever_success = False
        force_violation = False
        first_success = None
        for step in range(1, a.steps + 1):
            with torch.no_grad():
                action = policy(obs)
            if not torch.isfinite(action).all():
                raise RuntimeError('nonfinite_action')
            obs, reward, terminated, truncated, info = env.step(action)
            data_info = jsonable(info)
            success = bool(torch.as_tensor(info.get('success', False)).any())
            force_ok = bool(torch.as_tensor(info.get('cumulative_force_within_limit', True)).all())
            force_violation |= not force_ok
            ever_success |= success
            if success and first_success is None:
                first_success = step
            logger.emit('step', step=step, info=data_info,
                        terminated=jsonable(terminated), truncated=jsonable(truncated))
            # Fail, force and native horizon are measurements, not cutoffs.
        save(out / 'summary.json', dict(status='completed', case=case, steps=a.steps,
             ever_native_success=ever_success, final_native_success=success,
             first_success_step=first_success, force_violation=force_violation,
             final_info=data_info, wall_seconds=time.monotonic()-started, api_requests=0,
             lab_cost_usd=None, scope='isolated_train_spawn_not_long_horizon'))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--wall-seconds', type=int, default=3600)
    p.add_argument('--case', type=int)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        p.error('GPU1 UUID must be explicitly pinned')
    if a.steps != 200 or not 0 < a.wall_seconds <= 3600:
        p.error('pilot frozen to 200 actions/case and at most one GPU hour')
    if a.case is not None:
        worker(a); return
    a.output.mkdir(parents=True, exist_ok=True)
    lock = a.output / 'runner.lock'
    # Exclusive lock also prevents accidental concurrent resume. After an
    # unclean shutdown, operator must verify PID before removing stale lock.
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    manifest = a.output / 'manifest.json'
    specification = dict(schema='sac-spawn-probe/1', cases=cases(), steps=a.steps,
                         max_wall_seconds=a.wall_seconds, api_requests=0, gpu_uuid=GPU)
    start = time.monotonic()
    try:
        if manifest.exists():
            state = json.loads(manifest.read_text())
            if state['specification'] != specification:
                raise ValueError('resume configuration mismatch')
            if any(r['status'] == 'running' for r in state['rows']):
                raise ValueError('unclean exit: reconcile process and elapsed budget before resume')
        else:
            state = dict(specification=specification, consumed_wall_seconds=0,
                         rows=[dict(case=c, status='not_run', attempts=[]) for c in cases()])
            save(manifest, state)
        consumed = state['consumed_wall_seconds']
        for index, row in enumerate(state['rows']):
            if row['status'] != 'not_run':
                continue
            remaining = a.wall_seconds - consumed - (time.monotonic()-start)
            if remaining < 30:
                break
            out = a.output / f'case-{index:03d}' / 'attempt-001'
            out.mkdir(parents=True, exist_ok=False)
            row.update(status='running', attempts=[str(out)])
            save(manifest, state)
            argv = [sys.executable, __file__, '--case', str(index), '--output', str(out),
                    '--checkpoints', str(a.checkpoints), '--assets', str(a.assets)]
            with (out / 'process.log').open('w') as log:
                try:
                    result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=min(remaining, 300))
                    summary = out / 'summary.json'
                    row['status'] = 'completed' if result.returncode == 0 and summary.exists() else 'infrastructure_failed'
                    row['returncode'] = result.returncode
                except subprocess.TimeoutExpired:
                    row['status'] = 'timeout'
            state['consumed_wall_seconds'] = consumed + time.monotonic()-start
            save(manifest, state)
            if row['status'] != 'completed':
                break  # fail fast on compatibility issues, preserve all rows
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
