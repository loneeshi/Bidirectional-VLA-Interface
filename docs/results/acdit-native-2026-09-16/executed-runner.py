"""One bounded native AC-DiT MS-HAB Pick episode; no GPT or TAPT training.

Uses pinned upstream observations, both camera histories, pointcloud sampling,
whole-body normalized control and privileged context. Stops at every native
termination, keeps failed recordings, and binds one explicit training instruction
instead of sampling an unrelated stored embedding.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reset-only', action='store_true')
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'seed': args.seed, 'task': 'set_table/pick/013_apple', 'split': 'val',
              'max_steps': 200, 'api_calls': 0, 'training_updates': 0,
              'precision': 'float32', 'success': False, 'steps': 0,
              'privileged_context': True, 'reset_only': args.reset_only}
    def save():
        (args.output / 'result.json').write_text(json.dumps(jsonable(report), indent=2)+'\n')
    def event(name, **data):
        row = {'event': name, **jsonable(data)}
        with (args.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    env, writer = None, None
    recording = args.output / 'acdit-native-pick-episode000-incomplete.mp4'
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        row = next(r.split(',') for r in rows.splitlines() if int(r.split(',')[0]) == args.gpu_index)
        if int(row[2]) > 1024:
            raise RuntimeError('Selected GPU exceeds idle-memory guard')
        os.environ.update(CUDA_VISIBLE_DEVICES=row[1].strip(), OMP_NUM_THREADS='2',
                          HF_HOME=str(root / 'hf-cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          MS_ASSET_DIR=str(root / 'assets'))
        manifest = json.loads((root / 'assets/download-manifest.json').read_text())
        if 'completed_at_unix' not in manifest:
            raise RuntimeError('Asset extraction has not completed; do not start the environment yet')
        # PYTHONHASHSEED must be set by the launcher, before Python starts.
        if os.environ.get('PYTHONHASHSEED') != str(args.seed):
            raise RuntimeError('Launch with matching PYTHONHASHSEED')
        import numpy as np
        import torch
        import yaml
        import imageio.v2 as imageio
        from PIL import Image
        torch.set_num_threads(2)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        source = root / 'src/AC-DiT'
        sys.path[:0] = [str(source), str(root / 'src/Bidirectional-VLA-Interface/src')]
        os.chdir(source)
        import gymnasium as gym
        import mshab.envs
        from mshab.envs.planner import plan_data_from_file
        from scripts.eval_mshab import get_proprio, get_pointcloud
        from bvi.acdit_contract import fetch_proprio, privileged_context, validate_pointcloud, NativeCallBoundary
        rearrange = root / 'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        plan_path = rearrange / 'task_plans/set_table/pick/val/013_apple.json'
        plans = plan_data_from_file(plan_path)
        env = gym.make('PickSubtaskTrain-v0', num_envs=1, robot_uids='fetch', obs_mode='rgbdp',
                       control_mode='pd_joint_delta_pos', render_mode='rgb_array', reward_mode='dense',
                       sensor_configs={'shader_pack': 'default'}, human_render_camera_configs={'shader_pack': 'default'},
                       viewer_camera_configs={'shader_pack': 'default'}, sim_backend='cpu',
                       max_episode_steps=200, task_plans=plans.plans, scene_builder_cls=plans.dataset,
                       spawn_data_fp=rearrange / 'spawn_data/set_table/pick/val/spawn_data.pt',
                       require_build_configs_repeated_equally_across_envs=False)
        obs, info = env.reset(seed=args.seed)
        raw_env = env.unwrapped
        robot = raw_env.agent.robot
        joint_names = [j.name for j in robot.get_active_joints()]
        def proprio():
            value = fetch_proprio(robot.qpos[0].cpu().numpy(), joint_names,
                                  obs['extra']['base_linear_vel'][0].cpu().numpy(),
                                  obs['extra']['base_angular_vel'][0].cpu().numpy())
            np.testing.assert_allclose(value, get_proprio(obs)[0].cpu().numpy(), atol=1e-6, rtol=0)
            return torch.tensor(value).unsqueeze(0)
        proprio()
        privileged_context({k: v.cpu().numpy() if hasattr(v, 'cpu') else v
                            for k, v in obs['extra'].items()}, allow_privileged=True)
        validate_pointcloud(get_pointcloud(obs))
        low, high = env.action_space.low.reshape(-1), env.action_space.high.reshape(-1)
        report['interface'] = {'joint_names': joint_names, 'full_qpos_shape': list(robot.qpos.shape),
            'observation_qpos_shape': list(obs['agent']['qpos'].shape), 'action_low': low,
            'action_high': high, 'control_freq': raw_env.control_freq,
            'renderer_pci': raw_env._render_device.pci_string,
            'pointcloud_shape': list(obs['pointcloud']['xyzrgb'].shape),
            'state_mapping_matches_upstream': True,
            'controller_action_mapping': raw_env.agent.controller.action_mapping,
            'controllers': {key: {'type': type(value).__name__, 'config': str(value.config)}
                            for key, value in raw_env.agent.controller.controllers.items()}}
        history, clouds = deque(maxlen=2), deque(maxlen=2)
        history.append([None, None, None])
        clouds.append(None)
        def capture():
            head = obs['sensor_data']['fetch_head']['rgb'][0].cpu().numpy()
            hand = obs['sensor_data']['fetch_hand']['rgb'][0].cpu().numpy()
            history.append([head, hand, None])
            clouds.append(validate_pointcloud(get_pointcloud(obs)))
            return np.concatenate([head, hand], axis=1)
        frame = capture()
        Image.fromarray(frame).save(args.output / 'reset-cameras.png')
        report['reset_info'] = jsonable(info)
        report['observation_schema'] = {name: {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
                                              for k, v in data.items()}
                                         for name, data in obs['sensor_data'].items()}
        np.savez_compressed(args.output / 'reset-observation.npz',
                            qpos=robot.qpos.cpu().numpy(), pointcloud=get_pointcloud(obs),
                            **{f'{cam}_{k}': v.cpu().numpy() for cam, data in obs['sensor_data'].items()
                               for k, v in data.items()})
        report['status'] = 'reset_passed'
        save()
        event('reset_passed', interface=report['interface'])
        if args.reset_only:
            return
        # Reproduce the training tokenizer's padded batch; explicitly bind the
        # first matching apple instruction and retain its actual token IDs.
        from transformers import AutoTokenizer, SiglipTextModel
        encoder = json.loads((root / 'encoder-lock.json').read_text())
        languages = json.loads((source / 'configs/mshab_languages.json').read_text())
        entry = next(x for x in languages if (x['task'], x['subtask'], x['object']) ==
                     ('set_table', 'pick', '013_apple'))
        instruction = entry['instructions'][0]
        tokenizer = AutoTokenizer.from_pretrained(encoder['snapshot'], model_max_length=1024)
        tokens = tokenizer(entry['instructions'], return_tensors='pt', padding='longest', truncation=True)
        text_model = SiglipTextModel.from_pretrained(encoder['snapshot'], torch_dtype=torch.float32).eval()
        with torch.no_grad():
            embedding = text_model(input_ids=tokens['input_ids'][:1]).last_hidden_state[0].detach()
        del text_model
        binding = NativeCallBoundary(lambda text: embedding if text == instruction else None)
        binding.begin('native-pick-000', instruction, control_owner='acdit_whole_body')
        report.update(instruction=instruction, instruction_sha256=binding.instruction_sha256,
                      instruction_token_ids=tokens['input_ids'][0].tolist(),
                      embedding_sha256=hashlib.sha256(embedding.numpy().tobytes()).hexdigest(),
                      embedding_shape=list(embedding.shape), encoder_revision=encoder['revision'])
        from model_wrappers.mshab_model import create_model
        policy = create_model(args=yaml.safe_load((source / 'configs/config.yaml').read_text()),
            pretrained=str(root / 'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
            device='cuda:0', dtype=torch.float32, method_name='AC-DiT', combine_flag=False,
            pretrained_vision_encoder_name_or_path=encoder['snapshot'],
            mobility_head_ckpt_path=str(root / 'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'))
        report['strict_load'] = True
        writer = imageio.get_writer(recording, fps=raw_env.control_freq, codec='libx264', quality=8)
        writer.append_data(frame)
        event('model_ready', instruction=instruction, embedding_shape=list(embedding.shape))
        ever_grasped, clipping = False, 0
        while report['steps'] < 200:
            images = [Image.fromarray(a) if a is not None else None for window in history for a in window]
            started = time.monotonic()
            actions = policy.step(proprio(), images, binding.embedding, list(clouds), obs['extra'])[0].cpu().numpy()
            chunk = binding.enqueue('native-pick-000', actions, low, high)
            clipping += len(chunk.clipped_indices)
            event('prediction', step=report['steps'], seconds=time.monotonic()-started,
                  raw_actions=chunk.raw, clipped_indices=chunk.clipped_indices)
            stop = False
            while binding.pending and report['steps'] < 200:
                action = binding.pop()
                obs, reward, terminated, truncated, info = env.step(action)
                report['steps'] += 1
                writer.append_data(capture())
                grasped = bool(obs['extra']['is_grasped'].item())
                ever_grasped |= grasped
                report.update(success=bool(info['success'].item()), ever_grasped=ever_grasped,
                              clipped_channels=clipping, final_info=jsonable(info))
                event('step', step=report['steps'], action=action, reward=reward,
                      terminated=terminated, truncated=truncated, info=info,
                      extra=obs['extra'], qpos=robot.qpos, qvel=robot.qvel)
                save()
                if bool(terminated.item()) or bool(truncated.item()):
                    binding.interrupt()
                    report['stop_reason'] = 'native_terminated' if bool(terminated.item()) else 'native_truncated'
                    stop = True
                    break
            if stop:
                break
        report.setdefault('stop_reason', '200_step_budget')
        report.update(status='episode_completed', max_allocated_bytes=torch.cuda.max_memory_allocated())
        Image.fromarray(frame if report['steps'] == 0 else np.concatenate(history[-1][:2], axis=1)).save(args.output / 'final-cameras.png')
    except Exception as exc:
        report.update(status='error', error_type=type(exc).__name__, error=str(exc))
        (args.output / 'traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
            suffix = '' if report['success'] else '-failed' if report['status'] == 'episode_completed' else '-incomplete'
            final = args.output / f'acdit-native-pick-episode000{suffix}.mp4'
            if recording != final:
                recording.rename(final)
            report['video'] = final.name
            report['video_sha256'] = hashlib.sha256(final.read_bytes()).hexdigest()
        if env is not None:
            env.close()
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(jsonable(report)), flush=True)
    if report['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
