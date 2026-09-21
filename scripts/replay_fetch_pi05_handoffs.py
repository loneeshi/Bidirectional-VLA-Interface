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
NATIVE24_VALIDATION_ALLOWED = {
    key: value for key, value in ALLOWED.items()
    if key.startswith('fetch-current-handoffs-2026-09-17-run01/validation-seed')
}
NATIVE24_VALIDATION_BOUNDARIES = {
    (3020, 'grasp', 17), (3020, 'move', 20),
    (3021, 'grasp', 31), (3021, 'move', 34),
}


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


def source_suffix(path, allowed=ALLOWED):
    normalized = str(path).replace('\\', '/')
    matches = [suffix for suffix in allowed if normalized.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError('Source is outside five locked histories; training parents forbidden')
    return matches[0]


def prepare_jobs(plan, root, native24_validation_only=False):
    if plan.get('schema') != 'fetch_pi05_wrong_handoff_validation_plan_v1':
        raise ValueError('Unknown plan schema')
    cases = plan['fixed_cases']
    allowed = ALLOWED
    boundaries = BOUNDARIES
    expected_count = 8
    if native24_validation_only:
        cases = [case for case in cases if case.get('role') == 'validation']
        allowed = NATIVE24_VALIDATION_ALLOWED
        boundaries = NATIVE24_VALIDATION_BOUNDARIES
        expected_count = 4
    if len(cases) != expected_count or len({c['case_id'] for c in cases}) != expected_count:
        raise ValueError(f'Exactly {expected_count} locked historical cases are required')
    if {(c['seed'], c['family'], c['boundary_step']) for c in cases} != boundaries:
        raise ValueError('Locked historical boundary set changed')
    jobs = {}
    for case in cases:
        suffix = source_suffix(case['source_directory'], allowed)
        seed, split = allowed[suffix]
        if (case['seed'], case['scene_split']) != (seed, split):
            raise ValueError('Source identity mismatch')
        if not 1 <= case['boundary_step'] <= 200:
            raise ValueError('Action prefix exceeds 200-step bound')
        jobs.setdefault(suffix, []).append(case)
    if set(jobs) != set(allowed):
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


def build_native24_parent_registry(root, output):
    """Bind the fixed 3000..3003/3020..3021 source partition before model use."""
    from bvi.native24_handoff import PARENT_REGISTRY_SCHEMA
    source_root = root / 'runs/fetch-current-handoffs-2026-09-17-run01'
    index_path, batch_path = source_root / 'collection-index.json', source_root / 'batch.json'
    index = json.loads(index_path.read_text()); batch = json.loads(batch_path.read_text())
    expected = {('train', seed) for seed in range(3000, 3004)} | {
        ('validation', seed) for seed in (3020, 3021)}
    if batch.get('status') != 'complete' or {
            (row.get('split'), row.get('seed')) for row in index} != expected:
        raise ValueError('Frozen native24 replay parent roster changed')
    if {(row.get('split'), row.get('seed')) for row in batch.get('runs', [])} != expected:
        raise ValueError('Source batch and collection index disagree')
    parents = []
    for row in index:
        seed, split = row['seed'], row['split']
        directory = source_root / f'{split}-seed{seed}'
        result_path, events_path = directory / 'result.json', directory / 'events.jsonl'
        result = json.loads(result_path.read_text())
        if (result.get('status') != 'episode_completed' or result.get('seed') != seed
                or result.get('task') != 'set_table/pick/013_apple'):
            raise ValueError('Source parent result identity differs')
        events_sha = sha(events_path)
        parent = dict(source_kind='archived_online_replay', task='pick',
            source_sha256=events_sha, trajectory=f'{split}-seed{seed}', parent_id=seed)
        parents.append(dict(split=split, parent_episode=parent,
            events_sha256=events_sha, result_sha256=sha(result_path)))
    registry = dict(schema=PARENT_REGISTRY_SCHEMA, status='verified_parent_partitions',
        source_batch='fetch-current-handoffs-2026-09-17-run01',
        collection_index_sha256=sha(index_path), batch_sha256=sha(batch_path), parents=parents)
    path = output / 'parent-registry.json'
    path.write_text(json.dumps(registry, indent=2))
    return path


def build_sequence_preregistration(plan, training_manifest, input_manifest, monitor_source):
    """Freeze source windows and monitor semantics before rendering or inference."""
    from bvi.native24_handoff import LABEL_CONTRACT, canonical_sha256
    from bvi.native24_handoff_sequence import (
        FIXED_WINDOWS, INVOCATION_INDEX, monitor_contract,
    )
    jobs = prepare_jobs(plan, Path('/unused'), native24_validation_only=True)
    planned = {case['case_id']: case for _, _, cases in jobs for case in cases}
    gate = json.loads(Path(input_manifest).read_text(encoding='utf-8'))
    if gate.get('schema') != 'bvi.native24-handoff/1' or gate.get('status') != 'verified_handoff_inputs':
        raise ValueError('B input manifest is not verified')
    anchors = {case.get('case_id'): case for case in gate.get('cases', [])}
    if set(anchors) != set(FIXED_WINDOWS) or set(planned) != set(FIXED_WINDOWS):
        raise ValueError('Sequence anchors differ from the fixed B input roster')
    sequences = []
    for case_id in sorted(FIXED_WINDOWS):
        case, anchor = planned[case_id], anchors[case_id]
        if any(anchor.get(name) != case.get(source) for name, source in (
            ('classification', 'classification'), ('family', 'family'),
            ('instruction', 'instruction'), ('observation_index', 'boundary_step'),
        )):
            raise ValueError('Sequence anchor metadata differs from the B input gate')
        start, end = FIXED_WINDOWS[case_id]
        if end != case['boundary_step'] or end - start + 1 != 10:
            raise ValueError('Fixed sequence window is not the causal ten-frame tail')
        sequences.append(dict(
            sequence_id=f'{case_id}-causal-tail10', anchor_case_id=case_id,
            classification=case['classification'], family=case['family'],
            instruction=case['instruction'], parent_episode=anchor['parent_episode'],
            anchor_observation_index=end, invocation_index=INVOCATION_INDEX[case['family']],
            replan_cooldown=0,
            window=dict(start=start, end=end, length=10,
                        selection_rule='previous_9_through_anchor_source_observations'),
            starts_at_deployed_family_invocation=False,
            expected_first_event=None,
            forbidden_events=['learned_threshold'],
        ))
    contract = monitor_contract(monitor_source)
    document = dict(
        schema='bvi.native24-handoff-sequence-preregistration/1',
        status='frozen_before_render_or_model_query', label_contract=LABEL_CONTRACT,
        training_manifest_sha256=sha(training_manifest),
        input_manifest_sha256=sha(input_manifest), monitor_contract=contract,
        source_replay=dict(parents=[3020, 3021], repetitions=2,
                           exact_simulator_actions=108, hard_simulator_action_cap=136,
                           per_parent_double_replay_actions={'3020': 40, '3021': 68}),
        query_scope=dict(
            cadence='counterfactual_shadow_one_query_per_contiguous_source_observation',
            source_actions_cause_observations=True, s2_actions_cause_observations=False,
            deployment_cadence_evaluated=False, physical_stagnation_correctness_evaluable=False,
            physical_rollback_correctness_evaluable=False, native_success_evaluated=False,
            full_behavior_admission_evaluable=False,
        ),
        sequences=sequences,
    )
    document['preregistration_sha256'] = canonical_sha256(document)
    return document


def worker(args, plan, suffix):
    import numpy as np
    root = args.lab_root
    directory = root / 'runs' / suffix
    validation_mode = args.native24_validation_only or args.native24_sequence_gate
    cases = next(c for s, _, c in prepare_jobs(plan, root, validation_mode) if s == suffix)
    source = next(s for s in plan['sources'] if s['seed'] == cases[0]['seed'] and
                  s['events_sha256'] == sha(directory / 'events.jsonl'))
    if sha(directory / 'result.json') != source['result_sha256']:
        raise ValueError('Historical result hash mismatch')
    result = json.loads((directory / 'result.json').read_text())
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    steps = {e['step']: e for e in events if e['event'] == 'step'}
    if args.native24_sequence_gate:
        from bvi.native24_handoff_sequence import FIXED_WINDOWS
        max_step = max(FIXED_WINDOWS[c['case_id']][1] for c in cases)
    else:
        max_step = max(c['boundary_step'] for c in cases)
    if not set(range(1, max_step + 1)) <= set(steps):
        if args.native24_sequence_gate:
            from bvi.native24_handoff import NotEvaluableError
            raise NotEvaluableError('Fixed archived source is shorter than the causal sequence window')
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
                if args.native24_sequence_gate:
                    from bvi.native24_handoff import observation_sha256
                    from bvi.native24_handoff_sequence import FIXED_WINDOWS
                    for case in cases:
                        start, end = FIXED_WINDOWS[case['case_id']]
                        if not start <= step <= end:
                            continue
                        name = case['case_id']
                        head = array(obs['sensor_data']['fetch_head']['rgb'])[0]
                        wrist = array(obs['sensor_data']['fetch_hand']['rgb'])[0]
                        qpos = array(obs['agent']['qpos'])[0].astype(np.float32)
                        qvel = array(obs['agent']['qvel'])[0].astype(np.float32)
                        state = np.concatenate([qpos, qvel]).astype(np.float32)
                        values = dict(head_rgb=head, wrist_rgb=wrist, state=state)
                        if (head.shape != (128,128,3) or wrist.shape != (128,128,3)
                                or state.shape != (24,) or head.dtype != np.uint8
                                or wrist.dtype != np.uint8 or not np.isfinite(state).all()):
                            raise ValueError('Native24 sequence input contract mismatch')
                        observation_digest = observation_sha256(values)
                        tcp = array(obs['extra']['tcp_pose_wrt_base'])[0, :3]
                        obj = array(obs['extra']['obj_pose_wrt_base'])[0, :3]
                        predicate = dict(observation_index=step,
                            tcp_object_distance_m=float(np.linalg.norm(tcp - obj)),
                            is_grasped=bool(array(obs['extra']['is_grasped']).item()),
                            physical_completion=bool(array(info['success']).item()),
                            terminated=bool(array(terminated).item()),
                            truncated=bool(array(truncated).item()))
                        if (predicate['physical_completion'] or predicate['terminated']
                                or predicate['truncated']):
                            from bvi.native24_handoff import NotEvaluableError
                            raise NotEvaluableError(
                                'Fixed causal sequence crosses completion or termination'
                            )
                        if step == case['boundary_step'] and (
                            abs(predicate['tcp_object_distance_m'] - case['tcp_object_distance_m']) > ATOL
                            or predicate['is_grasped'] is not case['is_grasped']
                        ):
                            raise ValueError('Sequence anchor predicates differ from frozen B case')
                        if repeat == 0:
                            capture = captures.setdefault(name, dict(
                                case=case, frames=[], frame_observation_sha256=[]))
                            capture['frames'].append({key: value.copy() for key, value in values.items()})
                            capture['frame_observation_sha256'].append(observation_digest)
                            capture.setdefault('predicates', []).append(predicate)
                        else:
                            capture = captures.get(name)
                            offset = step - start
                            if capture is None or capture['frame_observation_sha256'][offset] != observation_digest:
                                raise ValueError(f'Double reconstruction observation mismatch: {name}/{step}')
                            expected_predicate = capture['predicates'][offset]
                            if (expected_predicate['observation_index'] != predicate['observation_index']
                                    or expected_predicate['is_grasped'] is not predicate['is_grasped']
                                    or expected_predicate['physical_completion'] is not predicate['physical_completion']
                                    or expected_predicate['terminated'] is not predicate['terminated']
                                    or expected_predicate['truncated'] is not predicate['truncated']
                                    or abs(expected_predicate['tcp_object_distance_m']
                                           - predicate['tcp_object_distance_m']) > ATOL):
                                raise ValueError(f'Double reconstruction predicate mismatch: {name}/{step}')
                for case in (() if args.native24_sequence_gate else
                             (c for c in cases if c['boundary_step'] == step)):
                    name = case['case_id']
                    if args.native24_validation_only:
                        from bvi.native24_handoff import observation_sha256
                        head = array(obs['sensor_data']['fetch_head']['rgb'])[0]
                        wrist = array(obs['sensor_data']['fetch_hand']['rgb'])[0]
                        qpos = array(obs['agent']['qpos'])[0].astype(np.float32)
                        qvel = array(obs['agent']['qvel'])[0].astype(np.float32)
                        state = np.concatenate([qpos, qvel]).astype(np.float32)
                        values = dict(head_rgb=head, wrist_rgb=wrist, state=state)
                        if (head.shape != (128,128,3) or wrist.shape != (128,128,3)
                                or state.shape != (24,) or head.dtype != np.uint8
                                or wrist.dtype != np.uint8 or not np.isfinite(state).all()):
                            raise ValueError('Native24 boundary input contract mismatch')
                        observation_digest = observation_sha256(values)
                        tcp = array(obs['extra']['tcp_pose_wrt_base'])[0, :3]
                        obj = array(obs['extra']['obj_pose_wrt_base'])[0, :3]
                        distance = float(np.linalg.norm(tcp - obj))
                        held = bool(array(obs['extra']['is_grasped']).item())
                        completed = bool(array(info['success']).item())
                        if (abs(distance - case['tcp_object_distance_m']) > ATOL
                                or held is not case['is_grasped'] or completed):
                            raise ValueError('Native24 boundary physical predicates differ from frozen case')
                        hashes = dict(observation=tree_hash(obs), simulator=tree_hash(raw.get_state_dict()),
                            controller=tree_hash(raw.agent.controller.get_state()),
                            native24=observation_digest)
                    else:
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
                    if repeat == 0:
                        captures[name] = dict(hashes=hashes, case=case)
                        if args.native24_validation_only:
                            captures[name].update(observation_sha256=observation_digest,
                                distance=distance, held=held, completed=completed)
                            np.savez_compressed(args.output / f'{name}.npz', **values)
                        else:
                            np.savez_compressed(args.output / f'{name}.npz', workspace_rgb=workspace,
                                wrist_rgb=wrist, state=state, original_base_xy=origin,
                                prompt=np.asarray(case['instruction']))
                    elif captures[name]['hashes'] != hashes:
                        raise ValueError(f'Double reconstruction mismatch: {name}')
                if (bool(array(terminated).item()) or bool(array(truncated).item())) and step < max_step:
                    raise ValueError('Replay terminated before requested boundary')
        finally:
            env.close()
    if args.native24_validation_only or args.native24_sequence_gate:
        from bvi.native24_handoff import REPLAY_SCHEMA, TRAJECTORY_SCHEMA
        trajectory_name = Path(suffix).name
        parent = dict(source_kind='archived_online_replay', task='pick',
            source_sha256=source['events_sha256'], trajectory=trajectory_name, parent_id=seed)
        trajectory_path = args.output / f'{trajectory_name}-source-trajectory.json'
        trajectory_record = dict(schema=TRAJECTORY_SCHEMA, status='verified_source_trajectory',
            parent_episode=parent, source_events_sha256=source['events_sha256'],
            source_result_sha256=source['result_sha256'],
            digest_contract='historical-events-jsonl-v1',
            trajectory_sha256=source['events_sha256'])
        trajectory_path.write_text(json.dumps(trajectory_record, indent=2))
        output_cases = []
        for name, value in captures.items():
            case = value['case']
            if args.native24_sequence_gate:
                from bvi.native24_handoff import canonical_sha256
                from bvi.native24_handoff_sequence import (
                    FIXED_WINDOWS, INVOCATION_INDEX, PREDICATE_SCHEMA,
                    REPLAY_SCHEMA as SEQUENCE_REPLAY_SCHEMA,
                )
                start, end = FIXED_WINDOWS[name]
                if (len(value['frames']) != 10
                        or [frame['observation_index'] for frame in value['predicates']]
                        != list(range(start, end + 1))):
                    from bvi.native24_handoff import NotEvaluableError
                    raise NotEvaluableError(f'Fixed sequence is incomplete: {name}')
                sequence_id = f'{name}-causal-tail10'
                npz_path = args.output / f'{sequence_id}.npz'
                np.savez_compressed(npz_path,
                    head_rgb=np.stack([frame['head_rgb'] for frame in value['frames']]),
                    wrist_rgb=np.stack([frame['wrist_rgb'] for frame in value['frames']]),
                    state=np.stack([frame['state'] for frame in value['frames']]))
                predicate_path = args.output / f'{sequence_id}-physical-predicates.json'
                predicate_record = dict(schema=PREDICATE_SCHEMA,
                    status='source_replay_physical_predicates', sequence_id=sequence_id,
                    endpoint_only_classification=case['classification'],
                    class_label_propagated_to_prior_frames=False,
                    frames=value['predicates'])
                predicate_path.write_text(json.dumps(predicate_record, indent=2))
                replay_path = args.output / f'{sequence_id}-source-replay.json'
                window = dict(start=start, end=end, length=10,
                    selection_rule='previous_9_through_anchor_source_observations')
                replay_record = dict(schema=SEQUENCE_REPLAY_SCHEMA,
                    status='verified_source_sequence_replay', sequence_id=sequence_id,
                    instruction=case['instruction'], parent_episode=parent,
                    trajectory_sha256=source['events_sha256'], window=window,
                    npz_sha256=sha(npz_path), physical_predicates_sha256=sha(predicate_path),
                    frame_observation_sha256=value['frame_observation_sha256'],
                    sequence_observation_sha256=canonical_sha256(value['frame_observation_sha256']),
                    reconstruction_frame_observation_sha256=[value['frame_observation_sha256']]*2,
                    physical_predicate_sequence_sha256=canonical_sha256(value['predicates']),
                    reconstruction_physical_predicate_sha256=[
                        canonical_sha256(value['predicates'])
                    ] * 2,
                    physical_atol=ATOL, initial_state_verified=True,
                    action_prefix_verified=True, boundary_observation_verified=True,
                    physical_predicates_verified=True, policy_calls=0, training_updates=0)
                replay_path.write_text(json.dumps(replay_record, indent=2))
                output_cases.append(dict(sequence_id=sequence_id, anchor_case_id=name,
                    role='validation', classification=case['classification'], family=case['family'],
                    instruction=case['instruction'], invocation_index=INVOCATION_INDEX[case['family']],
                    anchor_observation_index=end, parent_episode=parent, window=window,
                    native24_sequence_replay_verified=True, npz_sha256=sha(npz_path),
                    input_path=str(npz_path), physical_predicates_path=str(predicate_path),
                    source_trajectory_record_path=str(trajectory_path),
                    source_replay_record_path=str(replay_path),
                    frame_observation_sha256=value['frame_observation_sha256']))
                continue
            npz_path = args.output / f'{name}.npz'
            replay_path = args.output / f'{name}-source-replay.json'
            ground_truth = dict(source='source_replay_physical_predicates',
                tcp_object_distance_m=value['distance'], is_grasped=value['held'],
                physical_completion=value['completed'])
            replay_record = dict(schema=REPLAY_SCHEMA, status='verified_source_replay',
                case_id=name, instruction=case['instruction'], parent_episode=parent,
                trajectory_sha256=source['events_sha256'], observation_index=case['boundary_step'],
                npz_sha256=sha(npz_path), boundary_observation_sha256=value['observation_sha256'],
                reconstruction_observation_sha256=[value['observation_sha256']]*2,
                physical_atol=ATOL, initial_state_verified=True, action_prefix_verified=True,
                boundary_observation_verified=True, physical_predicates_verified=True,
                policy_calls=0, training_updates=0, ground_truth=ground_truth)
            replay_path.write_text(json.dumps(replay_record, indent=2))
            output_cases.append(dict(case_id=name, hashes=value['hashes'], role='validation',
                classification=case['classification'], family=case['family'],
                instruction=case['instruction'], parent_episode=parent,
                native24_replay_verified=True, npz_sha256=sha(npz_path),
                input_path=str(npz_path), source_trajectory_record_path=str(trajectory_path),
                source_replay_record_path=str(replay_path), ground_truth=ground_truth))
    else:
        output_cases = [dict(case_id=name, hashes=value['hashes'],
            role=value['case']['role'], classification=value['case']['classification'],
            family=value['case']['family'], original_parent=dict(seed=value['case']['seed'], scene_split=value['case']['scene_split']),
            workspace_replay_verified=True,
            npz_sha256=sha(args.output / f'{name}.npz'), input_path=str(args.output / f'{name}.npz'),
            physical_completion=None, current_local_first_progress=0.0,
            temporal_zero_is_physical_reward=False) for name, value in captures.items()]
    return dict(status='accepted_double_replay', cases=output_cases,
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
    p.add_argument('--native24-validation-only', action='store_true',
                   help='Replay only fixed validation3020/3021 and emit native head/wrist/state24 evidence')
    p.add_argument('--native24-sequence-gate', action='store_true',
                   help='Replay fixed validation3020/3021 causal tails for the zero-training B-prime gate')
    p.add_argument('--training-manifest', type=Path,
                   help='Required with either native24 mode; binds the S2 cache roster/hash')
    p.add_argument('--input-manifest', type=Path,
                   help='Verified bvi.native24-handoff/1 manifest; required by --native24-sequence-gate')
    p.add_argument('--progress-monitor-source', type=Path,
                   help='Frozen src/bvi/progress_monitor.py; required by --native24-sequence-gate')
    p.add_argument('--worker-source', choices=list(ALLOWED), help=argparse.SUPPRESS)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.native24_validation_only and args.native24_sequence_gate:
        p.error('Choose one native24 acquisition mode')
    native24_mode = args.native24_validation_only or args.native24_sequence_gate
    if native24_mode and args.training_manifest is None:
        p.error('--training-manifest is required with either native24 mode')
    if args.native24_sequence_gate and (
        args.input_manifest is None or args.progress_monitor_source is None
    ):
        p.error('--input-manifest and --progress-monitor-source are required by --native24-sequence-gate')
    try:
        jobs = prepare_jobs(plan, args.lab_root, native24_mode)
    except ValueError as exc:
        if native24_mode and str(exc) in {
                'Exactly 4 locked historical cases are required',
                'Locked historical boundary set changed',
                'Missing locked source; no case substitution'}:
            args.output.mkdir(parents=True, exist_ok=False)
            result = dict(status='not_evaluable_missing_fixed_native24_class_or_boundary',
                error=str(exc), handoff_input_ready=False, input_gate_only=True,
                training_updates=0, policy_calls=0, api_calls=0)
            (args.output / 'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
            return
        raise
    args.output.mkdir(parents=True, exist_ok=False)
    if args.native24_sequence_gate:
        try:
            preregistration = build_sequence_preregistration(
                plan, args.training_manifest, args.input_manifest, args.progress_monitor_source
            )
            with (args.output / 'sequence-preregistration.json').open('x', encoding='utf-8') as stream:
                json.dump(preregistration, stream, indent=2, allow_nan=False); stream.write('\n')
        except Exception as exc:
            result = dict(status='invalid_sequence_preregistration', error=repr(exc),
                handoff_input_ready=False, learned_feedback_behavior_evaluated=False,
                training_updates=0, policy_calls=0, api_calls=0)
            (args.output / 'result.json').write_text(json.dumps(result, indent=2))
            raise
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    row = next(line.replace(' ', '').split(',') for line in rows.splitlines() if line.split(',')[0].strip() == '1')
    if row[1] != args.gpu_uuid or int(row[2]) >= 1024:
        raise ValueError('Requires idle physicalGPU1 exactUUID; GPU0 forbidden')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, MS_ASSET_DIR=str(args.lab_root / 'assets'), OMP_NUM_THREADS='2')
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
                if args.native24_validation_only:
                    command += ['--native24-validation-only', '--training-manifest',
                                str(args.training_manifest.resolve())]
                elif args.native24_sequence_gate:
                    command += [
                        '--native24-sequence-gate', '--training-manifest',
                        str(args.training_manifest.resolve()), '--input-manifest',
                        str(args.input_manifest.resolve()), '--progress-monitor-source',
                        str(args.progress_monitor_source.resolve()),
                    ]
                with (args.output / (suffix.replace('/', '--') + '.log')).open('w') as stream:
                    subprocess.run(command, env={**os.environ, 'PYTHONHASHSEED': str(ALLOWED[suffix][0])},
                        stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=True)
                child = json.loads((target / 'result.json').read_text())
                if child['status'].startswith('not_evaluable'):
                    from bvi.native24_handoff import NotEvaluableError
                    raise NotEvaluableError(child.get('error', child['status']))
                if child['status'] != 'accepted_double_replay':
                    raise ValueError('Worker did not accept both reconstructions')
                status['accepted_cases'].extend(child['cases'])
                result_path.write_text(json.dumps(status, indent=2))
            expected_cases = 4 if native24_mode else 8
            if len(status['accepted_cases']) != expected_cases:
                raise ValueError('Incomplete fixed case coverage')
            status.update(status='verified_handoff_inputs', handoff_input_ready=True,
                          trained_progress_validated=False, training_ready=False)
            status['cases'] = [c for c in status['accepted_cases'] if c['role'] == 'validation']
            status['diagnostic_cases'] = [c for c in status['accepted_cases'] if c['role'] == 'locked_diagnostic']
            if args.native24_validation_only:
                from bvi.native24_handoff import (ACQUISITION_SCHEMA, LABEL_CONTRACT,
                                                   build_manifest)
                registry_path = build_native24_parent_registry(args.lab_root, args.output)
                acquisition = dict(schema=ACQUISITION_SCHEMA, status='capture_complete',
                    label_contract=LABEL_CONTRACT,
                    training_manifest_sha256=sha(args.training_manifest),
                    parent_registry_path=registry_path.relative_to(args.output).as_posix(), cases=[])
                for case in status['cases']:
                    acquisition['cases'].append(dict(case_id=case['case_id'],
                        classification=case['classification'], family=case['family'],
                        instruction=case['instruction'], parent_episode=case['parent_episode'],
                        policy_input_privileged=False, model_progress_used_as_ground_truth=False,
                        npz_path=Path(case['input_path']).relative_to(args.output).as_posix(),
                        source_trajectory_record_path=Path(case['source_trajectory_record_path']).relative_to(args.output).as_posix(),
                        source_replay_record_path=Path(case['source_replay_record_path']).relative_to(args.output).as_posix()))
                acquisition_path = args.output / 'acquisition.json'
                acquisition_path.write_text(json.dumps(acquisition, indent=2))
                manifest_path = args.output / 'native24-handoff-manifest.json'
                validation = build_manifest(acquisition_path, args.training_manifest, manifest_path)
                status.update(status='verified_native24_handoff_inputs',
                    native24_handoff_manifest=str(manifest_path),
                    native24_handoff_manifest_sha256=validation['manifest_sha256'],
                    input_gate_only=True, learned_feedback_behavior_evaluated=False,
                    stagnation_evaluable=False, rollback_evaluable=False)
            elif args.native24_sequence_gate:
                from bvi.native24_handoff import LABEL_CONTRACT
                from bvi.native24_handoff_sequence import (
                    ACQUISITION_SCHEMA as SEQUENCE_ACQUISITION_SCHEMA,
                    build_manifest as build_sequence_manifest,
                    monitor_contract,
                )
                registry_path = build_native24_parent_registry(args.lab_root, args.output)
                preregistration_path = args.output / 'sequence-preregistration.json'
                acquisition = dict(
                    schema=SEQUENCE_ACQUISITION_SCHEMA, status='capture_complete',
                    label_contract=LABEL_CONTRACT,
                    training_manifest_sha256=sha(args.training_manifest),
                    input_manifest_sha256=sha(args.input_manifest),
                    preregistration_path=preregistration_path.relative_to(args.output).as_posix(),
                    preregistration_sha256=sha(preregistration_path),
                    monitor_contract=monitor_contract(args.progress_monitor_source),
                    parent_registry_path=registry_path.relative_to(args.output).as_posix(),
                    exact_simulator_actions=108, hard_simulator_action_cap=136,
                    policy_calls=0, training_updates=0, cases=[])
                for case in status['cases']:
                    acquisition['cases'].append(dict(
                        sequence_id=case['sequence_id'], anchor_case_id=case['anchor_case_id'],
                        classification=case['classification'], family=case['family'],
                        instruction=case['instruction'], invocation_index=case['invocation_index'],
                        anchor_observation_index=case['anchor_observation_index'],
                        parent_episode=case['parent_episode'], window=case['window'],
                        npz_path=Path(case['input_path']).relative_to(args.output).as_posix(),
                        physical_predicates_path=Path(case['physical_predicates_path']).relative_to(args.output).as_posix(),
                        source_trajectory_record_path=Path(case['source_trajectory_record_path']).relative_to(args.output).as_posix(),
                        source_replay_record_path=Path(case['source_replay_record_path']).relative_to(args.output).as_posix()))
                acquisition_path = args.output / 'sequence-acquisition.json'
                acquisition_path.write_text(json.dumps(acquisition, indent=2))
                manifest_path = args.output / 'native24-handoff-sequence-manifest.json'
                validation = build_sequence_manifest(
                    acquisition_path, args.training_manifest, args.input_manifest,
                    args.progress_monitor_source, manifest_path,
                )
                status.update(
                    status='verified_native24_handoff_sequence_inputs',
                    native24_handoff_sequence_manifest=str(manifest_path),
                    native24_handoff_sequence_manifest_sha256=validation['manifest_sha256'],
                    input_gate_only=False, learned_feedback_behavior_evaluated=False,
                    shadow_query_cadence='one_query_per_contiguous_source_replay_observation',
                    deployment_cadence_evaluated=False,
                    physical_stagnation_correctness_evaluable=False,
                    physical_rollback_correctness_evaluable=False,
                    native_success_evaluated=False)
    except Exception as exc:
        from bvi.native24_handoff import NotEvaluableError
        if isinstance(exc, NotEvaluableError):
            status.update(status='not_evaluable_missing_required_classes',
                          error=repr(exc), handoff_input_ready=False)
            return
        else:
            status.update(status=(
                'invalid_sequence_evidence' if args.native24_sequence_gate else 'failed'
            ), error=repr(exc), handoff_input_ready=False)
            raise
    finally:
        result_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
