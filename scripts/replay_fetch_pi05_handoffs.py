"""Replay locked historical actions with CPU physics and physical-GPU1 rendering.

No policy, API, threshold changes or training. Each source runs in its own
300-second subprocess; two independent seeded reconstructions must match.
Trusted local historical Torch snapshots are loaded only after manifest hashes.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ATOL = 1e-5
ALLOWED = {
    'fetch-current-handoffs-2026-09-17-run01/validation-seed3020': (3020, 'train'),
    'fetch-current-handoffs-2026-09-17-run01/validation-seed3021': (3021, 'train'),
    'fetch-current-progress-eval-2026-09-17-run01/seed2025': (2025, 'val'),
    'fetch-current-progress-eval-2026-09-17-run01/seed2030': (2030, 'val'),
    'fetch-tapt-timing-2026-09-17-run01/legacy-pre-action': (2025, 'val'),
}
BOUNDARIES = {(3020, 'grasp', 17), (3020, 'move', 20),
              (3021, 'grasp', 31), (3021, 'move', 34),
              (2025, 'grasp', 26), (2030, 'grasp', 18),
              (2025, 'grasp', 16), (2025, 'move', 18)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(value):
    import numpy as np
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)


def compare(actual, expected, path='', exact=False):
    """Compare every expected physical tensor; booleans/integers remain exact."""
    import numpy as np
    if expected is None:
        if actual is not None:
            raise ValueError(f'Replay mismatch at {path}: expected None')
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected) <= set(actual):
            raise ValueError(f'Missing recorded fields: {path}')
        for name, value in expected.items():
            compare(actual[name], value, f'{path}/{name}', exact)
        return
    a, b = array(actual), array(expected)
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch at {path}: {a.shape}/{b.shape}')
    if b.dtype.kind == 'b' and a.dtype.kind != 'b':
        raise ValueError(f'Boolean dtype mismatch at {path}')
    if b.dtype.kind in 'biuUS' or exact:
        passed = np.array_equal(a, b)
    else:
        passed = np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=ATOL, rtol=0)
    if not passed:
        raise ValueError(f'Replay mismatch at {path}; exact={exact}, atol={ATOL}')


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item, prefix):
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], prefix + '/' + str(key))
        elif isinstance(item, (list, tuple)):
            for i, child in enumerate(item):
                visit(child, prefix + '/' + str(i))
        elif item is None or isinstance(item, (str, bool, int, float)):
            digest.update(prefix.encode())
            digest.update(json.dumps(item, allow_nan=False).encode())
        else:
            data = array(item)
            digest.update(prefix.encode()); digest.update(str(data.shape).encode())
            digest.update(str(data.dtype).encode()); digest.update(data.tobytes())
    visit(value, '')
    return digest.hexdigest()


def source_suffix(path):
    normalized = str(path).replace('\\', '/')
    matches = [suffix for suffix in ALLOWED if normalized.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError('Source is outside five locked histories; training parents forbidden')
    return matches[0]


def prepare_jobs(plan, root):
    if plan.get('schema') != 'fetch_pi05_wrong_handoff_validation_plan_v1':
        raise ValueError('Unknown plan schema')
    cases = plan['fixed_cases']
    if len(cases) != 8 or len({c['case_id'] for c in cases}) != 8:
        raise ValueError('Exactly eight locked historical cases are required')
    if {(c['seed'], c['family'], c['boundary_step']) for c in cases} != BOUNDARIES:
        raise ValueError('Locked historical boundary set changed')
    jobs = {}
    for case in cases:
        suffix = source_suffix(case['source_directory'])
        seed, split = ALLOWED[suffix]
        if (case['seed'], case['scene_split']) != (seed, split):
            raise ValueError('Source identity mismatch')
        if not 1 <= case['boundary_step'] <= 200:
            raise ValueError('Action prefix exceeds 200-step bound')
        jobs.setdefault(suffix, []).append(case)
    if set(jobs) != set(ALLOWED):
        raise ValueError('Missing locked source; no case substitution')
    return [(suffix, Path(root) / 'runs' / suffix, values) for suffix, values in jobs.items()]


def verify_initial_manifest(manifest, result, source):
    metadata = manifest['metadata']
    if (manifest.get('decision_id') != '000000' or metadata.get('prediction_step') != 0
            or metadata.get('seed') != result['seed'] or metadata.get('family') != 'reach'
            or metadata.get('checkpoint_sha256') != source['source_checkpoint_sha256']):
        raise ValueError('Initial decision snapshot identity differs from locked source')


def replay_horizon(suffix, result):
    # The original runner hard-coded env200 while max_steps bounded its loop.
    # Commit6baddc5 later changed env200 to args.max_steps during the collection
    # period; today's deployed source is not proof of the older process config.
    # All locked prefixes end before35, safely before either possible TimeLimit.
    expected_action_cap = 80 if suffix.startswith('fetch-current-handoffs-') else 200
    if result['max_steps'] != expected_action_cap:
        raise ValueError('Historical action cap differs from locked source')
    return 200


def worker(args, plan, suffix):
    import numpy as np
    root = args.lab_root
    directory = root / 'runs' / suffix
    cases = next(c for s, _, c in prepare_jobs(plan, root) if s == suffix)
    source = next(s for s in plan['sources'] if s['seed'] == cases[0]['seed'] and
                  s['events_sha256'] == sha(directory / 'events.jsonl'))
    if sha(directory / 'result.json') != source['result_sha256']:
        raise ValueError('Historical result hash mismatch')
    result = json.loads((directory / 'result.json').read_text())
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    steps = {e['step']: e for e in events if e['event'] == 'step'}
    max_step = max(c['boundary_step'] for c in cases)
    if not set(range(1, max_step + 1)) <= set(steps):
        raise ValueError('Historical applied-action prefix incomplete')
    for case in cases:
        actual_prefix = [steps[i]['action'] for i in range(1, case['boundary_step'] + 1)]
        if actual_prefix != case['applied_action_prefix']:
            raise ValueError('Plan prefix differs from source applied actions')
    manifest_path = directory / 'decisions/000000-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    verify_initial_manifest(manifest, result, source)
    raw_record = manifest['artifacts']['raw']
    if Path(raw_record['path']).name != raw_record['path']:
        raise ValueError('Invalid raw artifact path')
    raw_path = manifest_path.parent / raw_record['path']
    if sha(raw_path) != raw_record['sha256']:
        raise ValueError('Initial raw snapshot hash mismatch')
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    initial = torch.load(raw_path, map_location='cpu', weights_only=False)['raw']
    if not all(k in initial for k in ('simulator', 'controller', 'observation')):
        raise ValueError('Full initial simulator/controller/observation evidence missing')
    sys.path[:0] = [str(root / 'src/AC-DiT'), str(root / 'src/Bidirectional-VLA-Interface/src')]
    os.chdir(root / 'src/AC-DiT')
    import gymnasium as gym
    import mshab.envs
    from mshab.envs.planner import plan_data_from_file
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils.structs.pose import Pose
    native = REGISTERED_ENVS['PickSubtaskTrain-v0']
    environment_sources = {str(Path(inspect.getfile(cls)).resolve()) for cls in native.cls.__mro__
                           if cls.__module__.startswith(('mshab.', 'mani_skill.'))}
    environment_source_sha256 = {path: sha(path) for path in sorted(environment_sources)}
    cy, sy, cz, sz = math.cos(.65 / 2), math.sin(.65 / 2), math.cos(-.35 / 2), math.sin(-.35 / 2)
    camera = dict(uid='fetch_workspace', p=[.35, .35, 1.5], q=[cz*cy, -sz*sy, cz*sy, sz*cy],
                  width=224, height=224, fov=math.pi/2, near=.01, far=100, mount='agent.base_link')
    class WorkspaceReplay(native.cls):
        @property
        def _default_sensor_configs(self):
            original = super()._default_sensor_configs
            if any(c.uid == 'fetch_workspace' for c in original):
                raise ValueError('Workspace already exists')
            return [*original, CameraConfig(uid='fetch_workspace',
                pose=Pose.create_from_pq(p=camera['p'], q=camera['q']), width=224, height=224,
                fov=math.pi/2, near=.01, far=100, mount=self.agent.base_link)]
    env_id = 'BVIFetchPi05HandoffReplay-v0'
    register_env(env_id, max_episode_steps=native.max_episode_steps,
                 asset_download_ids=native.asset_download_ids, **native.default_kwargs)(WorkspaceReplay)
    seed, split = ALLOWED[suffix]
    rearrange = root / 'assets/data/scene_datasets/replica_cad_dataset/rearrange'
    task_path = rearrange / f'task_plans/set_table/pick/{split}/013_apple.json'
    spawn_path = rearrange / f'spawn_data/set_table/pick/{split}/spawn_data.pt'
    task = plan_data_from_file(task_path)
    expected_pci = subprocess.check_output(['nvidia-smi', '-i', args.gpu_uuid,
        '--query-gpu=pci.bus_id', '--format=csv,noheader'], text=True).strip().lower().split(':', 1)[1]
    captures = {}
    started = time.monotonic()
    for repeat in range(2):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        env = gym.make(env_id, num_envs=1, robot_uids='fetch', obs_mode='rgbdp',
            control_mode='pd_joint_delta_pos', render_mode='rgb_array', reward_mode='dense',
            sensor_configs={'shader_pack': 'default'}, human_render_camera_configs={'shader_pack': 'default'},
            viewer_camera_configs={'shader_pack': 'default'}, sim_backend='cpu',
            max_episode_steps=replay_horizon(suffix, result), task_plans=task.plans, scene_builder_cls=task.dataset,
            spawn_data_fp=spawn_path, require_build_configs_repeated_equally_across_envs=False)
        try:
            obs, info = env.reset(seed=seed)
            raw = env.unwrapped
            renderer_pci = raw._render_device.pci_string.lower().split(':', 1)[1]
            if renderer_pci != expected_pci:
                raise ValueError('Renderer is not physicalGPU1; stop before replay')
            # Seed equivalence is proved, not assumed: full reset physics/controller,
            # robot observation and original RGB/depth must exactly match archive.
            compare(raw.get_state_dict(), initial['simulator'], 'reset/simulator', exact=True)
            compare(raw.agent.controller.get_state(), initial['controller'], 'reset/controller', exact=True)
            compare(obs['agent'], initial['observation']['agent'], 'reset/agent', exact=True)
            for camera_name in ('fetch_head', 'fetch_hand'):
                for field in ('rgb', 'depth'):
                    compare(obs['sensor_data'][camera_name][field], initial['observation']['sensor_data'][camera_name][field], f'reset/{camera_name}/{field}', exact=True)
            compare(info, result['reset_info'], 'reset/info')
            origin = array(raw.agent.robot.qpos)[0, :2].copy()
            for step in range(1, max_step + 1):
                if time.monotonic() - started > 285:
                    raise TimeoutError('Origin replay reached internal 285-second guard')
                action = np.asarray(steps[step]['action'], np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                compare(raw.agent.robot.qpos, steps[step]['qpos'], f'{step}/qpos')
                compare(raw.agent.robot.qvel, steps[step]['qvel'], f'{step}/qvel')
                compare(obs['extra'], steps[step]['extra'], f'{step}/extra')
                compare(info, steps[step]['info'], f'{step}/info')
                compare(reward, steps[step]['reward'], f'{step}/reward')
                compare(terminated, steps[step]['terminated'], f'{step}/terminated')
                compare(truncated, steps[step]['truncated'], f'{step}/truncated')
                for case in (c for c in cases if c['boundary_step'] == step):
                    state = np.concatenate([array(raw.agent.robot.qpos)[0], array(raw.agent.robot.qvel)[0]]).astype(np.float32)
                    state[:2] -= origin
                    workspace = array(obs['sensor_data']['fetch_workspace']['rgb'])[0]
                    wrist = array(obs['sensor_data']['fetch_hand']['rgb'])[0]
                    if workspace.shape != (224,224,3) or wrist.shape != (128,128,3) or state.shape != (30,):
                        raise ValueError('V8 boundary input shape mismatch')
                    if workspace.dtype != np.uint8 or wrist.dtype != np.uint8 or not np.isfinite(state).all():
                        raise ValueError('V8 boundary input dtype/finiteness mismatch')
                    hashes = dict(observation=tree_hash(obs), simulator=tree_hash(raw.get_state_dict()),
                        controller=tree_hash(raw.agent.controller.get_state()), state30=tree_hash(state),
                        origin=tree_hash(origin))
                    name = case['case_id']
                    if repeat == 0:
                        captures[name] = dict(hashes=hashes, case=case)
                        np.savez_compressed(args.output / f'{name}.npz', workspace_rgb=workspace,
                            wrist_rgb=wrist, state=state, original_base_xy=origin,
                            prompt=np.asarray(case['instruction']))
                    elif captures[name]['hashes'] != hashes:
                        raise ValueError(f'Double reconstruction mismatch: {name}')
                if (bool(array(terminated).item()) or bool(array(truncated).item())) and step < max_step:
                    raise ValueError('Replay terminated before requested boundary')
        finally:
            env.close()
    return dict(status='accepted_double_replay', cases=[dict(case_id=name, hashes=value['hashes'],
        role=value['case']['role'], classification=value['case']['classification'],
        family=value['case']['family'], original_parent=dict(seed=value['case']['seed'], scene_split=value['case']['scene_split']),
        workspace_replay_verified=True,
        npz_sha256=sha(args.output / f'{name}.npz'), input_path=str(args.output / f'{name}.npz'),
        physical_completion=None, current_local_first_progress=0.0,
        temporal_zero_is_physical_reward=False) for name, value in captures.items()],
        initial_raw_sha256=raw_record['sha256'], initial_manifest_sha256=sha(manifest_path),
        task_plan_sha256=sha(task_path), spawn_sha256=sha(spawn_path), camera=camera,
        environment_source_sha256=environment_source_sha256,
        renderer_pci=renderer_pci, historical_time_limit=replay_horizon(suffix, result),
        historical_action_cap=result['max_steps'],
        horizon_provenance='Original online/timing runner env200; 6baddc5 later changed online env limit. Historical process source not archived here; all requested prefixes<35 before both80/200limits.',
        physical_atol=ATOL, integer_boolean_exact=True, initial_state_method='proved_exact_seeded_reset',
        api_calls=0, training_updates=0, policy_calls=0, seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--lab-root', type=Path, default=Path.home() / 'bvi-research')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--worker-source', choices=list(ALLOWED), help=argparse.SUPPRESS)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = prepare_jobs(plan, args.lab_root)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    row = next(line.replace(' ', '').split(',') for line in rows.splitlines() if line.split(',')[0].strip() == '1')
    if row[1] != args.gpu_uuid or int(row[2]) >= 1024:
        raise ValueError('Requires idle physicalGPU1 exactUUID; GPU0 forbidden')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, MS_ASSET_DIR=str(args.lab_root / 'assets'), OMP_NUM_THREADS='2')
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', accepted_cases=[], training_ready=False, api_calls=0, training_updates=0,
                  plan_sha256=sha(args.plan), script_sha256=sha(__file__), gpu_uuid=args.gpu_uuid)
    result_path = args.output / 'result.json'
    try:
        if args.worker_source:
            if os.environ.get('PYTHONHASHSEED') != str(ALLOWED[args.worker_source][0]):
                raise ValueError('Worker PYTHONHASHSEED differs from source seed')
            status.update(worker(args, plan, args.worker_source))
        else:
            for suffix, directory, cases in jobs:
                target = args.output / suffix.replace('/', '--')
                command = [sys.executable, str(Path(__file__).resolve()), '--plan', str(args.plan.resolve()),
                    '--lab-root', str(args.lab_root.resolve()), '--output', str(target.resolve()),
                    '--gpu-uuid', args.gpu_uuid, '--worker-source', suffix]
                with (args.output / (suffix.replace('/', '--') + '.log')).open('w') as stream:
                    subprocess.run(command, env={**os.environ, 'PYTHONHASHSEED': str(ALLOWED[suffix][0])},
                        stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
                child = json.loads((target / 'result.json').read_text())
                if child['status'] != 'accepted_double_replay':
                    raise ValueError('Worker did not accept both reconstructions')
                status['accepted_cases'].extend(child['cases'])
                result_path.write_text(json.dumps(status, indent=2))
            if len(status['accepted_cases']) != 8:
                raise ValueError('Incomplete fixed case coverage')
            status.update(status='verified_handoff_inputs', handoff_input_ready=True,
                          trained_progress_validated=False, training_ready=False)
            status['cases'] = [c for c in status['accepted_cases'] if c['role'] == 'validation']
            status['diagnostic_cases'] = [c for c in status['accepted_cases'] if c['role'] == 'locked_diagnostic']
    except Exception as exc:
        status.update(status='failed', error=repr(exc), handoff_input_ready=False)
        raise
    finally:
        result_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
