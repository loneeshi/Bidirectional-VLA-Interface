"""Bounded native-start SAC collection with the declared V8 workspace camera.

No pi0.5 inference/training, online TAPT claim, recovery takeover, or video.
All failed/incomplete episodes remain in their new output directory. Launch with
PYTHONHASHSEED equal to --seed and an external wall timeout.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['pick', 'place'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data-split', choices=['train', 'validation'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--gpu-uuid', default='GPU-b7ebba23-7824-7601-df32-be55628936c3')
    p.add_argument('--root', type=Path, default=Path.home() / 'bvi-research')
    a = p.parse_args()
    if not 1 <= a.max_steps <= 200:
        p.error('--max-steps must be in [1,200]')
    root, out = a.root.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='preflight', started_utc=datetime.now(timezone.utc).isoformat(),
                  task=a.task, seed=a.seed, split=a.data_split, scene_split='train', max_steps=a.max_steps,
                  steps=0, success=False, api_calls=0, training_updates=0,
                  source='official frozen deterministic per-object SAC in AC-DiT fork',
                  scope='native-start teacher data; not pi0.5 or online TAPT evaluation',
                  navigation_handoff_coverage=False, policy_inputs='original head/hand depth history + official42 state',
                  action_convention='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity',
                  head_mask_indices=[8, 9], clip_range=[-1, 1], base_position_reference='original_subtask_start_xy',
                  collector_sha256=sha(__file__))
    env = dataset = None

    def save():
        report['wall_seconds'] = time.monotonic() - start
        (out / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    try:
        if os.environ.get('PYTHONHASHSEED') != str(a.seed):
            raise ValueError('Launch with PYTHONHASHSEED matching --seed')
        physical1 = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
                                             '--format=csv,noheader'], text=True).strip()
        if a.gpu_uuid != physical1:
            raise ValueError('Selected UUID must identify physical GPU1')
        used = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid,
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used >= 1024:
            raise RuntimeError(f'GPU1 occupied: {used} MiB')
        os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid, MS_ASSET_DIR=str(root / 'assets'), OMP_NUM_THREADS='2')
        import torch
        import numpy as np
        import yaml
        import h5py
        import gymnasium as gym
        from gymnasium import spaces
        from PIL import Image
        import mshab.envs
        from mshab.envs.planner import plan_data_from_file
        from mshab.agents.sac import Agent
        from mani_skill.utils.common import flatten_state_dict
        from mani_skill.utils.registration import REGISTERED_ENVS, register_env
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils.structs.pose import Pose
        torch.set_num_threads(2)
        torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
        os.chdir(root / 'src/AC-DiT')
        native_id = f'{a.task.capitalize()}SubtaskTrain-v0'
        native_spec = REGISTERED_ENVS[native_id]
        native_cls = native_spec.cls
        pitch, yaw = .65, -.35
        cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
        cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
        camera_contract = dict(uid='fetch_workspace', p=[.35, .35, 1.5],
            q=[cz * cy, -sz * sy, cz * sy, sz * cy], width=224, height=224,
            fov=math.pi / 2, near=.01, far=100, mount='agent.base_link')

        class WorkspaceTeacherEnv(native_cls):
            @property
            def _default_sensor_configs(self):
                original = super()._default_sensor_configs
                if not isinstance(original, (list, tuple)):
                    raise TypeError('Unexpected original sensor config container')
                if any(c.uid == 'fetch_workspace' for c in original):
                    raise ValueError('Workspace sensor already exists')
                return [*original, CameraConfig(uid='fetch_workspace',
                    pose=Pose.create_from_pq(p=camera_contract['p'], q=camera_contract['q']),
                    width=224, height=224, fov=math.pi / 2, near=.01, far=100,
                    mount=self.agent.base_link)]

        augmented_id = f'BVIFetchPi05Teacher{a.task.capitalize()}-v0'
        register_env(augmented_id, max_episode_steps=native_spec.max_episode_steps,
                     asset_download_ids=native_spec.asset_download_ids,
                     **native_spec.default_kwargs)(WorkspaceTeacherEnv)
        camera_reference = root / 'src/Bidirectional-VLA-Interface/src/bvi/nav_camera_env.py'
        # Bind the copied camera definition and all inherited environment sources.
        source_paths = {str(Path(inspect.getfile(cls)).resolve()) for cls in native_cls.__mro__
                        if cls.__module__.startswith(('mshab.', 'mani_skill.'))}
        source_hashes = {path: sha(path) for path in sorted(source_paths)}
        report.update(camera=camera_contract, camera_reference_sha256=sha(camera_reference),
                      native_environment_id=native_id, augmented_environment_id=augmented_id,
                      environment_source_sha256=source_hashes, gpu_uuid=a.gpu_uuid,
                      overridden_environment_methods=['_default_sensor_configs'])
        rearrange = root / 'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        plan_path = rearrange / f'task_plans/set_table/{a.task}/train/013_apple.json'
        spawn_path = rearrange / f'spawn_data/set_table/{a.task}/train/spawn_data.pt'
        plans = plan_data_from_file(plan_path)
        report['task_plan_sha256'] = sha(plan_path)
        env = gym.make(augmented_id, num_envs=1, robot_uids='fetch', obs_mode='rgbdp',
            control_mode='pd_joint_delta_pos', render_mode='rgb_array', reward_mode='dense',
            sensor_configs={'shader_pack': 'default'}, human_render_camera_configs={'shader_pack': 'default'},
            viewer_camera_configs={'shader_pack': 'default'}, sim_backend='cpu', max_episode_steps=200,
            task_plans=plans.plans, scene_builder_cls=plans.dataset, spawn_data_fp=spawn_path,
            require_build_configs_repeated_equally_across_envs=False)
        obs, info = env.reset(seed=a.seed)
        u = env.unwrapped
        joint_names = [j.name for j in u.agent.robot.active_joints]
        base_xy_origin = u.agent.robot.qpos[0, :2].detach().cpu().numpy().copy()
        if len(joint_names) != 15 or tuple(u.agent.robot.qvel.shape) != (1, 15):
            raise ValueError('Expected Fetch qpos/qvel15')
        controller_paths = {inspect.getfile(type(u.agent.controller))}
        for controller in getattr(u.agent.controller, 'controllers', {}).values():
            controller_paths.add(inspect.getfile(type(controller)))
        report.update(joint_names=joint_names, original_subtask_start_base_xy=base_xy_origin.tolist(),
                      controller_source_sha256={str(Path(path).resolve()): sha(path) for path in sorted(controller_paths)},
                      robot_source_sha256=sha(inspect.getfile(type(u.agent))))
        initial = dict(simulator=u.get_state_dict(), controller=u.agent.controller.get_state(),
                       reset_info=info, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       original_subtask_start_base_xy=base_xy_origin, joint_names=joint_names)
        torch.save(initial, out / 'initial-state.pt')
        (out / 'initial-state.json').write_text(json.dumps(jsonable(initial), indent=2), encoding='utf-8')
        dataset = h5py.File(out / 'trajectory.h5', 'x')
        dataset.attrs['joint_names'] = json.dumps(joint_names)
        dataset.attrs['original_subtask_start_base_xy'] = base_xy_origin
        dataset.attrs['camera_contract'] = json.dumps(camera_contract)
        dataset.attrs['provenance'] = report['source']
        dataset.attrs['controller_action_convention'] = report['action_convention']

        def record_observation(index, observation):
            def write(group, values):
                for key, value in values.items():
                    if isinstance(value, dict):
                        write(group.create_group(key), value)
                    elif hasattr(value, 'detach') or isinstance(value, np.ndarray):
                        arr = value.detach().cpu().numpy() if hasattr(value, 'detach') else value
                        group.create_dataset(key, data=arr, **({'compression': 'gzip'} if arr.ndim else {}))
            g = dataset.create_group(f'observations/{index:04d}')
            write(g, observation)
            qpos = u.agent.robot.qpos.detach().cpu().numpy().copy()
            qvel = u.agent.robot.qvel.detach().cpu().numpy().copy()
            relative = qpos[0].copy(); relative[:2] -= base_xy_origin
            state30 = np.concatenate([relative, qvel[0]]).astype(np.float32)
            if state30.shape != (30,) or not np.isfinite(state30).all():
                raise ValueError('Invalid V8 state30')
            robot = g.create_group('recorded_robot')
            robot.create_dataset('qpos', data=qpos); robot.create_dataset('qvel', data=qvel)
            robot.create_dataset('v8_state30', data=state30)
            for camera in ['fetch_workspace', 'fetch_hand', 'fetch_head']:
                rgb = observation['sensor_data'][camera]['rgb'][0].detach().cpu().numpy()
                if camera == 'fetch_workspace' and rgb.shape != (224, 224, 3):
                    raise ValueError('Workspace image contract failed')
                if index == 0:
                    Image.fromarray(rgb).save(out / f'reset-{camera}.png')
            dataset.flush()

        record_observation(0, obs)
        frames = {camera: deque(maxlen=3) for camera in ['fetch_head', 'fetch_hand']}

        def encode(observation, first=False):
            pixels = {}
            for camera, queue in frames.items():
                depth = observation['sensor_data'][camera]['depth'].permute(0, 3, 1, 2)
                for _ in range(3 if first else 1):
                    queue.append(depth)
                pixels[camera + '_depth'] = torch.cat(list(queue), dim=1).to(device='cuda', dtype=torch.float32).contiguous()
            official_extra = {k: observation['extra'][k] for k in
                ['tcp_pose_wrt_base', 'obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped']}
            state = torch.cat([flatten_state_dict(observation['agent'], use_torch=True),
                               flatten_state_dict(official_extra, use_torch=True)], dim=1).to('cuda')
            if state.shape != (1, 42):
                raise ValueError(f'Official SAC state changed: {state.shape}')
            return pixels, state

        pixels, state = encode(obs, True)
        ck = root / f'checkpoints/mshab/rl/set_table/{a.task}/013_apple'
        cfg = yaml.safe_load((ck / 'config.yml').read_text())['algo']
        keys = ['actor_hidden_dims', 'critic_hidden_dims', 'critic_layer_norm', 'critic_dropout',
                'encoder_pixels_feature_dim', 'encoder_state_feature_dim', 'cnn_features',
                'cnn_filters', 'cnn_strides', 'cnn_padding']
        policy = Agent(spaces.Dict({k: spaces.Box(0, 32767, tuple(v.shape[1:]), np.int16) for k, v in pixels.items()}),
            tuple(state.shape[1:]), (13,), **{k: cfg[k] for k in keys},
            log_std_min=cfg['actor_log_std_min'], log_std_max=cfg['actor_log_std_max'], device='cuda')
        policy.load_state_dict(torch.load(ck / 'policy.pt', map_location='cuda', weights_only=False)['agent'], strict=True)
        policy.to('cuda').eval()
        report.update(checkpoint_sha256=sha(ck / 'policy.pt'), teacher_config_sha256=sha(ck / 'config.yml'),
                      teacher_source_sha256=sha(inspect.getfile(Agent)), status='collecting')
        (out / 'run-metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        dataset.attrs['header_metadata'] = json.dumps(report)
        save()
        with (out / 'events.jsonl').open('x', encoding='utf-8') as log:
            for step in range(a.max_steps):
                with torch.no_grad():
                    action = policy.actor(pixels, state, compute_pi=False, compute_log_pi=False)[0].cpu()
                if action.shape != (1, 13) or not torch.isfinite(action).all():
                    raise ValueError('Invalid SAC action')
                action[..., 8:10] = 0
                action = action.clamp(-1, 1)
                obs, reward, terminated, truncated, info = env.step(action)
                dataset.create_dataset(f'actions/{step:04d}', data=action.numpy())
                report.update(steps=step + 1, success=bool(info['success'].item()), final_info=jsonable(info))
                record_observation(step + 1, obs)
                log.write(json.dumps(jsonable(dict(step=step + 1, action=action, info=info,
                    reward=reward, terminated=terminated, truncated=truncated,
                    qpos=u.agent.robot.qpos, qvel=u.agent.robot.qvel))) + '\n')
                log.flush(); save()
                if bool(terminated.item()) or bool(truncated.item()):
                    report['stop_reason'] = 'native_termination' if bool(terminated.item()) else 'native_truncation'
                    break
                pixels, state = encode(obs)
            else:
                report['stop_reason'] = 'collection_step_limit'
        for camera in ['fetch_workspace', 'fetch_hand', 'fetch_head']:
            Image.fromarray(obs['sensor_data'][camera]['rgb'][0].cpu().numpy()).save(out / f'final-{camera}.png')
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='error', error=repr(exc))
        raise
    finally:
        try:
            if dataset is not None:
                dataset.attrs['native_success'] = report['success']
                dataset.attrs['status'] = report['status']
                dataset.close()
            if env is not None:
                env.close()
        finally:
            save()
    report['artifacts_sha256'] = {path.name: sha(path) for path in sorted(out.iterdir())
                                 if path.is_file() and path.name != 'result.json'}
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
