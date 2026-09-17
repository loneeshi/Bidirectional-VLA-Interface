"""Frozen V8 Pick capability episode, matching the 09/16 native environment.

The model lives in an isolated OpenPI process; this client owns only simulation.
No tool-family routing, progress-based stop, GPT, training, retry or SAC takeover.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from run_lab_fetch_pi05_teacher import jsonable, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--socket', required=True)
    p.add_argument('--auth-file', type=Path, required=True)
    p.add_argument('--historical-reset', type=Path, required=True)
    a = p.parse_args()
    if a.seed not in (2024, 2025, 2026, 2027, 2028, 2030):
        p.error('Only the predeclared native and auxiliary validation seeds are allowed')
    if os.environ.get('PYTHONHASHSEED') != str(a.seed):
        raise ValueError('PYTHONHASHSEED must match the seed before process start')
    root = Path.home() / 'bvi-research'
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='preflight', seed=a.seed, scene_split='val',
        task='set_table/pick/013_apple', max_actions=200, control_frequency=20,
        api_calls=0, training_updates=0, success=False, ever_grasped=False,
        stable_grasp_3_observations=False, longest_grasp_streak=0, steps=0,
        policy='frozen FetchV8; no new prefix head or tool-family banks',
        native_predicate='grasped + ee_rest<=0.05m + robot_rest + is_static + cumulative_force<5000',
        precision='bfloat16', predicted_action_horizon=10, executed_per_prediction=1,
        head_mask_indices=[8, 9], historical_pairing='qpos/head/hand RGB+depth only; original full state unavailable',
        started_utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__))
    env = writer = conn = None
    recording = out / f'fetch-v8-native-pick-seed{a.seed}-episode000-incomplete.mp4'

    def save():
        report['wall_seconds'] = time.monotonic() - started
        (out / 'result.json').write_text(json.dumps(jsonable(report), indent=2))

    save()
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
            MS_ASSET_DIR=str(root / 'assets'), OMP_NUM_THREADS='2')
        sys.path[:0] = [str(root / 'src/AC-DiT'), str(root / 'src/Bidirectional-VLA-Interface/src')]
        import numpy as np
        import torch
        import imageio.v2 as imageio
        from PIL import Image
        from multiprocessing.connection import Client
        import gymnasium as gym
        import mshab.envs
        from mshab.envs.planner import plan_data_from_file
        from mani_skill.utils.registration import REGISTERED_ENVS, register_env
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils.structs.pose import Pose
        from build_fetch_pi05_family_data import V8_CAMERA, V8_JOINT_NAMES
        torch.set_num_threads(2)
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        os.chdir(root / 'src/AC-DiT')
        spec = REGISTERED_ENVS['PickSubtaskTrain-v0']

        class WorkspacePick(spec.cls):
            @property
            def _default_sensor_configs(self):
                return [*super()._default_sensor_configs, CameraConfig(uid='fetch_workspace',
                    pose=Pose.create_from_pq(p=V8_CAMERA['p'], q=V8_CAMERA['q']),
                    width=224, height=224, fov=V8_CAMERA['fov'], near=.01, far=100,
                    mount=self.agent.base_link)]

        register_env('BVIV8NativePick-v0', max_episode_steps=spec.max_episode_steps,
            asset_download_ids=spec.asset_download_ids, **spec.default_kwargs)(WorkspacePick)
        rearrange = root / 'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        plan_path = rearrange / 'task_plans/set_table/pick/val/013_apple.json'
        plans = plan_data_from_file(plan_path)
        env = gym.make('BVIV8NativePick-v0', num_envs=1, robot_uids='fetch', obs_mode='rgbdp',
            control_mode='pd_joint_delta_pos', render_mode='rgb_array', reward_mode='dense',
            sensor_configs={'shader_pack': 'default'}, human_render_camera_configs={'shader_pack': 'default'},
            viewer_camera_configs={'shader_pack': 'default'}, sim_backend='cpu', max_episode_steps=200,
            task_plans=plans.plans, scene_builder_cls=plans.dataset,
            spawn_data_fp=rearrange / 'spawn_data/set_table/pick/val/spawn_data.pt',
            require_build_configs_repeated_equally_across_envs=False)
        obs, info = env.reset(seed=a.seed)
        u = env.unwrapped
        joints = [j.name for j in u.agent.robot.active_joints]
        if joints != V8_JOINT_NAMES or u.control_freq != 20:
            raise ValueError('Native joint/control contract changed')
        report['task_plan_sha256'] = sha(plan_path)
        report['spawn_data_sha256'] = sha(rearrange / 'spawn_data/set_table/pick/val/spawn_data.pt')
        report['renderer_pci'] = u._render_device.pci_string
        expected_pci = subprocess.check_output(['nvidia-smi', '-i',
            'GPU-b7ebba23-7824-7601-df32-be55628936c3', '--query-gpu=pci.bus_id',
            '--format=csv,noheader'], text=True).strip()
        if str(report['renderer_pci']).lower()[-7:] != expected_pci.lower()[-7:]:
            raise ValueError('Renderer is not on physical GPU1')
        if u.pick_cfg.ee_rest_thresh != .05 or u.pick_cfg.robot_cumulative_force_limit != 5000:
            raise ValueError('Native Pick rest/force predicates changed')
        report['ee_rest_threshold_m'] = .05
        report['source_sha256'] = {str(Path(inspect.getfile(c)).resolve()): sha(inspect.getfile(c))
            for c in spec.cls.__mro__ if c.__module__.startswith(('mshab.', 'mani_skill.'))}
        report['task_cfg'] = str(u.task_cfgs['pick'])
        origin = u.agent.robot.qpos[0, :2].cpu().numpy().copy()
        snapshot = dict(simulator=u.get_state_dict(), controller=u.agent.controller.get_state(),
            reset_info=info, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            original_subtask_start_base_xy=origin, joint_names=joints)
        torch.save(snapshot, out / 'initial-state.pt')
        (out / 'initial-state.json').write_text(json.dumps(jsonable(snapshot), indent=2))
        reset = dict(qpos=u.agent.robot.qpos.cpu().numpy(),
            **{f'{cam}_{k}': v.cpu().numpy() for cam, vals in obs['sensor_data'].items()
               for k, v in vals.items()})
        np.savez_compressed(out / 'reset-observation.npz', **reset)
        with np.load(a.historical_reset, allow_pickle=False) as historical:
            keys = ['qpos', 'fetch_head_rgb', 'fetch_head_depth', 'fetch_hand_rgb', 'fetch_hand_depth']
            comparison = {k: bool(np.array_equal(historical[k], reset[k])) for k in keys}
        report['historical_reset_exact_matches'] = comparison
        report['historical_reset_sha256'] = sha(a.historical_reset)
        if not all(comparison.values()):
            raise RuntimeError('Historical reset observations differ; no comparable score or control launched')
        original_plan = u.build_config_idx_to_task_plans[int(u.build_config_idxs[0])][int(u.task_plan_idxs[0])]
        target = original_plan.subtasks[0].obj_id
        report['original_plan'] = jsonable(original_plan)
        # This is the checkpoint's existing task-plan language template, not a prompt sweep.
        prompt = f'Pick and stably hold object {target}.'
        report.update(instruction=prompt, instruction_source='existing_V8_task_plan_template',
                      original_subtask_start_base_xy=origin.tolist(), joint_names=joints)
        conn = Client(a.socket, family='AF_UNIX', authkey=a.auth_file.read_bytes())
        conn.send({'op': 'reset', 'seed': a.seed})
        response = conn.recv()
        if response.get('status') != 'reset':
            raise RuntimeError(f'Model reset failed: {response}')
        report['model_metadata'] = response['metadata']

        def frame():
            return np.concatenate([obs['sensor_data']['fetch_workspace']['rgb'][0].cpu().numpy(),
                np.asarray(Image.fromarray(obs['sensor_data']['fetch_hand']['rgb'][0].cpu().numpy()).resize((224, 224)))], axis=1)

        Image.fromarray(frame()).save(out / 'reset-cameras.png')
        writer = imageio.get_writer(recording, fps=20, codec='libx264', quality=8)
        writer.append_data(frame())
        (out / 'requests').mkdir()
        grasp_streak = 0
        report['status'] = 'episode_running'; save()
        with (out / 'events.jsonl').open('x') as log:
            for step in range(200):
                qpos = u.agent.robot.qpos[0].cpu().numpy().copy(); qpos[:2] -= origin
                state = np.r_[qpos, u.agent.robot.qvel[0].cpu().numpy()].astype(np.float32)
                request = dict(workspace_rgb=obs['sensor_data']['fetch_workspace']['rgb'][0].cpu().numpy(),
                    wrist_rgb=obs['sensor_data']['fetch_hand']['rgb'][0].cpu().numpy(), state=state, prompt=prompt)
                np.savez_compressed(out / 'requests' / f'step{step:03d}.npz', **request)
                before = time.monotonic(); conn.send({'op': 'predict', 'seed': a.seed, **request})
                request_limit = 120 if step == 0 else 60
                if not conn.poll(request_limit):
                    raise TimeoutError(f'Model request exceeded{request_limit}seconds')
                response = conn.recv()
                if response.get('status') != 'ok':
                    raise RuntimeError(f'Model inference failed: {response}')
                actions = np.asarray(response['actions'], np.float32)
                if actions.shape != (10, 13) or not np.isfinite(actions).all():
                    raise ValueError('Invalid V8 action chunk')
                inference_seconds = time.monotonic() - before
                raw = actions[0].copy(); applied = np.clip(raw, -1, 1); applied[8:10] = 0
                obs, reward, terminated, truncated, info = env.step(torch.from_numpy(applied[None]))
                held = bool(obs['extra']['is_grasped'].item())
                grasp_streak = grasp_streak + 1 if held else 0
                report.update(steps=step + 1, success=bool(info['success'].item()),
                    ever_grasped=report['ever_grasped'] or held,
                    longest_grasp_streak=max(report['longest_grasp_streak'], grasp_streak),
                    stable_grasp_3_observations=report['stable_grasp_3_observations'] or grasp_streak >= 3,
                    final_info=jsonable(info))
                log.write(json.dumps(jsonable(dict(step=step + 1, raw_actions=actions, action=applied,
                    clipped_indices=np.flatnonzero(raw != np.clip(raw, -1, 1)),
                    inference_seconds=inference_seconds, model_rng=response.get('rng'),
                    terminated=terminated, truncated=truncated, info=info, extra=obs['extra'],
                    qpos=u.agent.robot.qpos, qvel=u.agent.robot.qvel))) + '\n'); log.flush()
                writer.append_data(frame()); save()
                if bool(terminated.item()) or bool(truncated.item()):
                    report['stop_reason'] = 'native_termination' if bool(terminated.item()) else 'native_truncation'
                    break
            else:
                report['stop_reason'] = '200_action_budget'
        Image.fromarray(frame()).save(out / 'final-cameras.png')
        report['status'] = 'episode_completed'
    except Exception as exc:
        report.update(status='infrastructure_failure', error=repr(exc))
        raise
    finally:
        if conn is not None: conn.close()
        if writer is not None: writer.close()
        if env is not None: env.close()
        if report['status'] == 'episode_completed' and recording.exists():
            name = f'fetch-v8-native-pick-seed{a.seed}-episode000' + ('' if report['success'] else '-failed') + '.mp4'
            recording.rename(out / name); report['video'] = name
        report['artifact_sha256'] = {x.name: sha(x) for x in out.iterdir() if x.is_file() and x.name != 'result.json'}
        save()


if __name__ == '__main__':
    main()
