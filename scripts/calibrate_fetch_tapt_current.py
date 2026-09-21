"""Bounded current-observation TAPT calibration, with no API or native updates.

Teacher endpoints require recorded completion evidence. On-policy first-call
anchors reconstruct only invocation-local time zero, not physical failure labels.
The shared head still reads AC-DiT action-token features, not OpenPI prefixes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-index', type=int, default=1)
    parser.add_argument('--output', required=True, help='New directory relative to ~/bvi-research')
    parser.add_argument('--handoff-dir', required=True, help='Collection directory relative to ~/bvi-research')
    parser.add_argument('--max-updates', type=int, default=200)
    parser.add_argument('--max-seconds', type=float, default=1200)
    args = parser.parse_args(argv)
    if args.gpu_index != 1:
        parser.error('Only the authorized laboratory GPU1 is allowed')
    if not 20 <= args.max_updates <= 200 or not 0 < args.max_seconds <= 1200:
        parser.error('Require 20..200 updates and >0..1200 training/validation seconds')
    return args


def decision_manifests(directory):
    """Accept the recorder's individual manifests or an exported JSONL index."""
    directory = Path(directory)
    jsonl = directory / 'manifest.jsonl'
    if jsonl.is_file():
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    else:
        rows = [json.loads(path.read_text()) for path in sorted(directory.glob('*-manifest.json'))]
    if not rows:
        raise ValueError(f'No complete decision manifests in {directory}')
    rows.sort(key=lambda row: int(row['decision_id']))
    if len({row['decision_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate decision IDs')
    return rows


def first_call_events(events):
    """Prove first predictions coincide with recorded invocation boundaries."""
    first = {}
    boundaries = {}
    last_step = 0
    last_clear = None
    active_call = None
    for event in events:
        kind = event['event']
        if kind == 'step':
            last_step = event['step']
        elif kind == 'queue_cleared':
            last_clear = event['call_id']
        elif kind == 'family_switch':
            if last_clear is None or (active_call is not None and last_clear != active_call):
                raise ValueError('Family switch lacks the preceding invocation queue-clear evidence')
            boundaries[event['call_id']] = (last_step, event['family'])
            active_call = event['call_id']
            last_clear = None
        elif kind == 'progress_prediction':
            call = event['call_id']
            if call in first:
                continue
            if not first:
                if event['step'] != 0 or event['family'] != 'reach':
                    raise ValueError('Initial invocation must begin with reach at step zero')
                active_call = call
            elif boundaries.get(call) != (event['step'], event['family']) or active_call != call:
                raise ValueError('First prediction does not match the recorded family-switch boundary')
            first[call] = event
    if not first:
        raise ValueError('No complete invocation events')
    return first


def main(argv=None):
    args = parse_args(argv)
    root = Path.home() / 'bvi-research'
    source = root / 'src/AC-DiT'
    sys.path[:0] = [str(source), str(root / 'src/Bidirectional-VLA-Interface/src')]
    from bvi.fetch_current_labels import (
        ANCHOR_SUPERVISION, CONTRACT, first_call_anchor, sample_times,
        validate_manifest_splits, verified_teacher_targets,
    )

    data = root / 'runs/fetch-tapt-teacher-2026-09-16'
    segments_path = data / 'segments.json'
    segments = json.loads(segments_path.read_text())
    handoff = root / args.handoff_dir
    index_path = handoff / 'collection-index.json'
    collection = json.loads(index_path.read_text())
    collection = collection if isinstance(collection, list) else collection['rows']
    if not collection:
        raise ValueError('A nonempty completed on-policy collection is required')
    batch_path = handoff / 'batch.json'
    batch_report = json.loads(batch_path.read_text())
    if batch_report['status'] != 'complete' or len(batch_report['runs']) != len(collection):
        raise ValueError('Entire predeclared collection must be complete before calibration')
    identity = lambda row: (row['path'], row['split'], row['seed'], row['scene_split'], row['task'])
    if len({identity(row) for row in collection}) != len(collection):
        raise ValueError('Duplicate collection parents')
    if {identity(row) for row in collection} != {identity(row) for row in batch_report['runs']}:
        raise ValueError('Collection index and completed batch runs disagree')
    if any(row['returncode'] != 0 for row in batch_report['runs']):
        raise ValueError('Collection contains an unsuccessful process')
    windows = [{**window, 'scene_split': 'train'} for window in segments['windows']]
    # Validate parents before loading torch, allocating a GPU, or constructing data.
    validate_manifest_splits(windows + collection)
    if any(row['split'] not in ('train', 'validation') for row in windows + collection):
        raise ValueError('Calibration accepts only training and validation parents')
    examples = []
    for window in windows:
        times = sample_times(window['start'], window['end'])
        if window['split'] == 'validation':
            # Fixed bounded validation support: current start, midpoint, endpoint.
            times = sorted({times[0], window['start'] + (window['end'] - window['start']) // 2, times[-1]})
        for t in times:
            examples.append({**window, 'kind': 'teacher', 'sample_t': t,
                             **verified_teacher_targets(window, t)})
    for parent in collection:
        path = Path(parent['path'])
        if not path.is_absolute():
            path = root / path if (root / path).exists() else handoff / path
        directory = path if path.name == 'decisions' else path / 'decisions'
        episode_path = directory.parent / 'result.json'
        episode = json.loads(episode_path.read_text())
        if (episode['status'] != 'episode_completed' or episode['seed'] != parent['seed']
                or episode['split'] != parent['scene_split']
                or episode['task'] != f"set_table/{parent['task']}/013_apple"):
            raise ValueError('Completed episode identity disagrees with collection parent')
        events_path = directory.parent / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        first_events = first_call_events(events)
        seen = set()
        for manifest in decision_manifests(directory):
            meta = manifest['metadata']
            if meta['seed'] != parent['seed']:
                raise ValueError('Collection parent seed disagrees with decision provenance')
            if meta.get('scene_split', parent['scene_split']) != parent['scene_split']:
                raise ValueError('Decision scene split disagrees with parent')
            if meta.get('task', parent['task']) not in (parent['task'], episode['task']):
                raise ValueError('Decision task disagrees with parent')
            call_id = meta['call_id']
            if call_id in seen:
                continue
            seen.add(call_id)
            first = first_events.get(call_id)
            if first is None or (first['decision_id'], first['step'], first['family']) != (
                    manifest['decision_id'], meta['prediction_step'], meta['family']):
                raise ValueError('Earliest recorded input is not the proven invocation-start decision')
            artifact = manifest['artifacts']['inputs']
            inputs = directory / artifact['path']
            if sha256(inputs) != artifact['sha256']:
                raise ValueError(f'Input artifact hash mismatch: {inputs}')
            raw_artifact = manifest['artifacts']['raw']
            raw_path = directory / raw_artifact['path']
            if sha256(raw_path) != raw_artifact['sha256']:
                raise ValueError(f'Raw artifact hash mismatch: {raw_path}')
            examples.append({**parent, 'kind': 'invocation_start_anchor',
                             'supervision': ANCHOR_SUPERVISION,
                             'family': meta['family'], 'call_id': call_id,
                             'decision_id': manifest['decision_id'],
                             'inputs_path': str(inputs), 'inputs_sha256': artifact['sha256'],
                             'raw_path': str(raw_path), 'raw_sha256': raw_artifact['sha256'],
                             'episode_result_sha256': sha256(episode_path), 'events_sha256': sha256(events_path),
                             'source_checkpoint_sha256': meta['checkpoint_sha256'],
                             'prediction_step': meta['prediction_step'], **first_call_anchor()})
        if seen != set(first_events):
            raise ValueError('Invocation event log has a missing first-call decision artifact')
    validate_manifest_splits(examples)
    validation_specs = [row for row in examples if row['split'] == 'validation']
    val_calls = sum(1 + int(all(row['action_valid'])) for row in validation_specs)
    if not validation_specs or val_calls > 80:
        raise ValueError(f'Fixed validation requires {val_calls} calls; authorized cap is 80')
    if not any(row['kind'] == 'invocation_start_anchor' and row['split'] == 'train' for row in examples):
        raise ValueError('No training invocation-start anchors collected')
    if not any(row['kind'] == 'invocation_start_anchor' and row['split'] == 'validation' for row in examples):
        raise ValueError('No held-out invocation-start anchors collected')
    used = int(subprocess.check_output(
        ['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=memory.used',
         '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 1024:
        raise RuntimeError('GPU1 occupied; do not overlap workloads')
    gpu_uuid = subprocess.check_output(
        ['nvidia-smi', '-i', str(args.gpu_index), '--query-gpu=uuid', '--format=csv,noheader'],
        text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, HF_HOME=str(root / 'hf-cache'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
    out = root / args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / 'cache').mkdir()
    report = dict(status='loading', started_at=datetime.now(timezone.utc).isoformat(), updates=0, api_calls=0, progress_label_contract=CONTRACT,
                  scope='bounded current-observation TAPT calibration; not native task success',
                  rank=8, alpha=8, learning_rate=1e-4, progress_weight=.1,
                  micro_batch=1, effective_batch=8, max_updates=args.max_updates,
                  max_training_validation_seconds=args.max_seconds,
                  time_scope='Hard admission deadline from first update; includes periodic validation and saves. '
                             'Loading, preprocessing, zero-update control and final hash audit are setup/teardown. '
                             'An in-flight GPU call may finish beyond the deadline.',
                  head_caveat='AC-DiT action-token head retained; label/consumption alignment only, not author prefix head',
                  anchor_supervision=ANCHOR_SUPERVISION,
                  source_scope='SAC teacher/native starts and frozen-policy on-policy starts; not LightNav coverage',
                  checkpoint_selection='fixed-seed held-out native action MSE + 0.1 generated-inference progress MSE',
                  validation_teacher_support='start, floor midpoint, verified observation endpoint',
                  validation_calls=val_calls,
                  release_coverage_limitation='Historical teacher coverage is only 2 train / 1 validation release intervals; '
                                              'Pick on-policy collection does not add release anchors.',
                  source_provenance=dict(segments_path=str(segments_path), segments_sha256=sha256(segments_path),
                                         collection_index_path=str(index_path), collection_index_sha256=sha256(index_path),
                                         collection_batch_sha256=sha256(batch_path)),
                  gpu_uuid=gpu_uuid)

    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    save()
    (out / 'split-manifest.json').write_text(json.dumps(examples, indent=2))
    try:
        import torch
        import numpy as np
        import h5py
        import yaml
        from PIL import Image
        from transformers import AutoTokenizer, SiglipTextModel
        from model_wrappers.mshab_model import create_model
        from bvi.acdit_contract import to_unified
        from bvi.acdit_tapt import ACDiTFamilyTool, FAMILIES, masked_progress_loss

        if any(row['family'] not in FAMILIES for row in examples):
            raise ValueError('Unknown family in calibration input')
        torch.set_num_threads(2)
        torch.manual_seed(4017)
        np.random.seed(4017)
        os.chdir(source)
        encoder = json.loads((root / 'encoder-lock.json').read_text())
        wrapper = create_model(
            args=yaml.safe_load((source / 'configs/config.yaml').read_text()),
            pretrained=str(root / 'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
            device='cuda:0', dtype=torch.float32, method_name='AC-DiT', combine_flag=False,
            pretrained_vision_encoder_name_or_path=encoder['snapshot'],
            mobility_head_ckpt_path=str(root / 'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'))
        tokenizer = AutoTokenizer.from_pretrained(encoder['snapshot'])
        text_model = SiglipTextModel.from_pretrained(encoder['snapshot'], torch_dtype=torch.float32).eval()

        class CapturedBatch(Exception):
            pass

        def cpu_clone(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu_clone(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu_clone(item) for item in value)
            return value

        cache = []
        trajectory_hashes = {}
        for row_index, row in enumerate(examples):
            item = dict(metadata=row)
            if row['kind'] == 'invocation_start_anchor':
                payload = torch.load(row['inputs_path'], map_location='cpu', weights_only=False)
                if payload['decision_id'] != row['decision_id'] or payload['metadata']['call_id'] != row['call_id']:
                    raise ValueError('Input payload and first-call manifest disagree')
                item['batch'] = payload['batch']
                if 'action_gt' in item['batch']:
                    raise ValueError('Inference anchor unexpectedly contains action supervision')
                raw = torch.load(row['raw_path'], map_location='cpu', weights_only=False)
                extra = raw['raw']['observation']['extra']
                tcp = np.asarray(extra['tcp_pose_wrt_base']).reshape(-1, 7)[0, :3]
                obj = np.asarray(extra['obj_pose_wrt_base']).reshape(-1, 7)[0, :3]
                row['physical_diagnostics_only'] = dict(
                    tcp_object_distance=float(np.linalg.norm(tcp - obj)),
                    is_grasped=bool(np.asarray(extra['is_grasped']).reshape(-1)[0]))
                del raw
            else:
                trajectory = Path(row['trajectory'])
                if str(trajectory) not in trajectory_hashes:
                    trajectory_hashes[str(trajectory)] = sha256(trajectory)
                tokens = tokenizer(row['instruction'], return_tensors='pt', padding=False)
                with torch.no_grad():
                    embedding = text_model(input_ids=tokens['input_ids']).last_hidden_state[0]
                t = row['sample_t']
                with h5py.File(trajectory) as handle:
                    def obs(i):
                        group = handle[f'observations/{i:04d}']
                        return group, {key: torch.from_numpy(group['extra'][key][:]) for key in group['extra']}
                    group, extra = obs(t)
                    qpos = group['agent/qpos'][0]
                    native = np.r_[qpos[[2, 4, 5, 6, 7, 8, 9, 10, 1, 3, 0]],
                                   extra['base_linear_vel'][0, 0], extra['base_angular_vel'][0, 2]].astype('float32')
                    images, clouds = [], []
                    for i in (t - 1, t):
                        if i < 0:
                            images.extend([None, None, None])
                            clouds.append(None)
                        else:
                            previous, _ = obs(i)
                            images.extend([Image.fromarray(previous[f'sensor_data/{camera}/rgb'][0])
                                           for camera in ('fetch_head', 'fetch_hand')] + [None])
                            clouds.append(previous['pointcloud/xyzrgb'][:].reshape(-1, 6))
                    captured = {}
                    original = wrapper.policy.predict_action
                    def capture(**kwargs):
                        captured.update(cpu_clone(kwargs))
                        # Exit after native preprocessing; fabricate no action tensor.
                        raise CapturedBatch()
                    wrapper.policy.predict_action = capture
                    try:
                        with torch.no_grad():
                            wrapper.step(torch.from_numpy(native)[None], images, embedding, clouds, extra)
                    except CapturedBatch:
                        pass
                    finally:
                        wrapper.policy.predict_action = original
                    if not captured:
                        raise RuntimeError('Native preprocessing produced no batch')
                    item['batch'] = captured
                    if all(row['action_valid']):
                        actions = np.stack([to_unified(handle[f'actions/{t+j:04d}'][0]) for j in range(2)])
                        if not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                            raise ValueError('Invalid real normalized teacher action')
                        item['actions'] = torch.tensor(actions)[None]
            item['target'] = torch.tensor(row['progress_target'], dtype=torch.float32)[None]
            item['valid'] = torch.tensor(row['progress_valid'], dtype=torch.bool)[None]
            path = out / 'cache' / f'{row_index:05d}.pt'
            torch.save(item, path)
            cache.append(item)
            report.update(status='caching_real_features', cached_examples=len(cache))
            save()
        del text_model
        report['source_provenance']['trajectory_sha256'] = trajectory_hashes
        (out / 'split-manifest.json').write_text(json.dumps(examples, indent=2))
        report['cache_sha256'] = {path.name: sha256(path) for path in sorted((out / 'cache').glob('*.pt'))}
        warm_path = root / 'runs/fetch-tapt-sft-2026-09-16-run01/best.pt'
        warm = torch.load(warm_path, map_location='cpu', weights_only=False)
        if warm['updates'] != 1199:
            raise ValueError('Expected the frozen 1199-update warm-start checkpoint')
        report.update(warmstart_path=str(warm_path), warmstart_sha256=sha256(warm_path),
                      warmstart_updates=warm['updates'], projection_paths=warm['report']['projection_paths'])
        if any(row['source_checkpoint_sha256'] != report['warmstart_sha256']
               for row in examples if row['kind'] == 'invocation_start_anchor'):
            raise ValueError('Handoff collector did not use the frozen warm-start checkpoint')
        tool = ACDiTFamilyTool(wrapper.policy, report['projection_paths'], rank=8, alpha=8)
        # eval leaves gradients enabled while preserving native inference/dropout behavior.
        tool.eval()
        parameters = dict(tool.named_parameters())
        trainable = {name: value for name, value in parameters.items() if value.requires_grad}
        if set(trainable) != set(warm['trainable']):
            raise ValueError('Warm-start trainable keys differ')
        with torch.no_grad():
            for name, value in warm['trainable'].items():
                if value.shape != trainable[name].shape:
                    raise ValueError(f'Warm-start shape differs for {name}')
                trainable[name].copy_(value)
        initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
        del warm

        def frozen_digest():
            digest = hashlib.sha256()
            for name, value in tool.named_parameters():
                if not value.requires_grad:
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            for name, value in tool.named_buffers():
                digest.update(('buffer:' + name).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        report['frozen_native_sha256_before'] = frozen_digest()
        params = list(trainable.values())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        rng = random.Random(4017)
        training = {family: [item for item in cache if item['metadata']['split'] == 'train'
                            and item['metadata']['family'] == family] for family in FAMILIES}
        if not all(training.values()):
            raise ValueError('Every family needs training support')
        validation = [item for item in cache if item['metadata']['split'] == 'validation']
        if {item['metadata']['family'] for item in validation} != set(FAMILIES):
            raise ValueError('Every family needs held-out support')
        report['anchor_coverage'] = {
            split: {family: sum(item['metadata']['split'] == split and item['metadata']['family'] == family
                               and item['metadata']['kind'] == 'invocation_start_anchor' for item in cache)
                    for family in FAMILIES} for split in ('train', 'validation')}
        report['sample_counts'] = {
            split: {family: {kind: sum(item['metadata']['split'] == split
                                      and item['metadata']['family'] == family
                                      and item['metadata']['kind'] == kind for item in cache)
                             for kind in ('teacher', 'invocation_start_anchor')}
                    for family in FAMILIES} for split in ('train', 'validation')}
        report['anchor_physical_states'] = [
            {key: item['metadata'][key] for key in ('split', 'seed', 'family', 'call_id', 'physical_diagnostics_only')}
            for item in cache if item['metadata']['kind'] == 'invocation_start_anchor']
        report['incorrect_handoff_coverage'] = {
            split: dict(
                grasp_outside_teacher_reach_boundary=sum(
                    row['split'] == split and row['family'] == 'grasp'
                    and row['physical_diagnostics_only']['tcp_object_distance'] > .08
                    for row in report['anchor_physical_states']),
                move_without_hold=sum(
                    row['split'] == split and row['family'] == 'move'
                    and not row['physical_diagnostics_only']['is_grasped']
                    for row in report['anchor_physical_states']))
            for split in ('train', 'validation')}
        report['handoff_coverage_scope'] = 'Physical predicates are audit-only; labels remain invocation-local clock zero.'
        if not any(report['incorrect_handoff_coverage']['train'].values()):
            raise ValueError('Training collection lacks any erroneous grasp or unheld move invocation start')
        teacher_pools = {family: [item for item in training[family] if item['metadata']['kind'] == 'teacher']
                         for family in FAMILIES}
        anchor_pools = {family: [item for item in training[family] if item['metadata']['kind'] == 'invocation_start_anchor']
                        for family in FAMILIES}
        if not all(teacher_pools.values()):
            raise ValueError('Every family needs real teacher support')
        report['training_mixture'] = {
            family: dict(anchors=1 if anchor_pools[family] else 0,
                         teacher=7 if anchor_pools[family] else 8)
            for family in FAMILIES}
        report['sampling'] = 'Round-robin families; exactly one anchor plus seven uniform teacher rows if anchors exist, '
        report['sampling'] += 'otherwise eight uniform teacher rows; deterministic random.Random(4017).'

        def device_batch(item):
            def move(value):
                if isinstance(value, torch.Tensor):
                    return value.to('cuda')
                if isinstance(value, dict):
                    return {key: move(v) for key, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return type(value)(move(v) for v in value)
                return value
            return move(item['batch'])

        def run_loss(item):
            batch = device_batch(item)
            family = item['metadata']['family']
            target, valid = item['target'].to('cuda'), item['valid'].to('cuda')
            if all(item['metadata']['action_valid']):
                if 'actions' not in item:
                    raise ValueError('Missing real action supervision')
                return tool.training_loss(family, {**batch, 'action_gt': item['actions'].to('cuda')}, target, valid)
            return tool.progress_only_loss(family, batch, target, valid)

        def checkpoint(name, zero_control=False):
            checkpoint_report = dict(report)
            if zero_control:
                checkpoint_report.update(progress_label_contract='post_action_position_v1',
                                         calibration_target_contract=CONTRACT,
                                         current_contract_checkpoint_eligible=False)
            torch.save(dict(trainable={name: value.detach().cpu() for name, value in trainable.items()},
                            optimizer=optimizer.state_dict(), updates=report['updates'], report=checkpoint_report,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            sampler_rng=rng.getstate()), out / name)

        deadline = None
        def expired():
            return deadline is not None and time.monotonic() >= deadline

        def validate():
            rows, action_values, progress_values = [], [], []
            tool.eval()
            try:
                with torch.random.fork_rng(devices=[0]), torch.no_grad():
                    for index, item in enumerate(validation):
                        if expired():
                            return None
                        # Reset per row/path so action-loss availability cannot change inference noise.
                        torch.manual_seed(8181 + index * 2)
                        _, prediction = tool.predict(item['metadata']['family'], device_batch(item))
                        pl = float(masked_progress_loss(prediction, item['target'].to('cuda'), item['valid'].to('cuda')))
                        progress_values.append(pl)
                        value = dict(family=item['metadata']['family'], kind=item['metadata']['kind'],
                                     progress_loss=pl, current_prediction=float(prediction[0, 0]),
                                     current_target=float(item['target'][0, 0]))
                        if all(item['metadata']['action_valid']):
                            if expired():
                                return None
                            torch.manual_seed(8182 + index * 2)
                            al = float(run_loss(item)['action_loss'])
                            action_values.append(al)
                            value['action_loss'] = al
                        rows.append(value)
                if not action_values or not np.isfinite(action_values + progress_values).all():
                    raise ValueError('Nonfinite or missing held-out loss')
                action_mean, progress_mean = float(np.mean(action_values)), float(np.mean(progress_values))
                return dict(action_loss=action_mean, generated_progress_mse=progress_mean,
                            score=action_mean + .1 * progress_mean, rows=rows)
            finally:
                tool.eval()

        baseline = validate()
        report.update(zero_update_control=baseline, validation=baseline, validation_step=0,
                      best_step=None, best_score=None, status='training',
                      zero_update_checkpoint_eligible=False)
        checkpoint('zero-update.pt', zero_control=True)
        save()
        start = time.monotonic()
        deadline = start + args.max_seconds
        last_validated = 0
        with (out / 'updates.jsonl').open('w') as log:
            for step in range(args.max_updates):
                if expired():
                    break
                family = FAMILIES[step % len(FAMILIES)]
                optimizer.zero_grad(set_to_none=True)
                totals = dict(loss=0., action_loss=0., progress_loss=0.)
                completed = 0
                for micro in range(8):
                    if expired():
                        break
                    pool = anchor_pools[family] if micro == 0 and anchor_pools[family] else teacher_pools[family]
                    result = run_loss(rng.choice(pool))
                    if not torch.isfinite(result['loss']):
                        raise ValueError('Nonfinite training loss')
                    (result['loss'] / 8).backward()
                    for key in totals:
                        totals[key] += float(result[key].detach()) / 8
                    completed += 1
                if completed != 8 or expired():
                    optimizer.zero_grad(set_to_none=True)
                    break
                if not any(value.grad is not None for value in params):
                    raise ValueError('No training gradients')
                if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in params):
                    raise ValueError('Nonfinite gradients')
                for layer in tool.layers:
                    if any(layer.a[f].grad is not None or layer.b[f].grad is not None for f in FAMILIES if f != family):
                        raise ValueError('Unselected family received gradients')
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                report['updates'] = step + 1
                if step + 1 == 20:
                    changed_at_gate = [name for name, value in trainable.items()
                                       if not torch.equal(initial[name], value.detach().cpu())]
                    gate_family_changes = {
                        f: sum(name.endswith((f'.a.{f}', f'.b.{f}')) for name in changed_at_gate)
                        for f in FAMILIES}
                    gate_head_changed = any(name.startswith('progress.') for name in changed_at_gate)
                    gate_native_unchanged = frozen_digest() == report['frozen_native_sha256_before']
                    report['twenty_update_gate'] = dict(
                        finite_losses_gradients=True, nonselected_banks_grad_none=True,
                        changed_parameters_by_family=gate_family_changes,
                        progress_head_changed=gate_head_changed, native_unchanged=gate_native_unchanged)
                    if not all(gate_family_changes.values()) or not gate_head_changed or not gate_native_unchanged:
                        raise ValueError('Twenty-update parameter isolation/change gate failed')
                log.write(json.dumps(dict(step=step+1, family=family,
                                          mixture=report['training_mixture'][family], **totals)) + '\n')
                log.flush()
                if step + 1 == 20 or (step + 1) % 50 == 0:
                    checkpoint('latest.pt')
                    checkpoint(f'checkpoint-{step+1}.pt')
                    values = validate()
                    if values is not None:
                        report.update(validation=values, validation_step=step+1)
                        last_validated = step + 1
                        if report['best_score'] is None or values['score'] < report['best_score']:
                            report.update(best_step=step+1, best_score=values['score'])
                            checkpoint('best.pt')
                save()
        if report['updates'] != last_validated and not expired():
            values = validate()
            if values is not None:
                report.update(validation=values, validation_step=report['updates'])
                if report['best_score'] is None or values['score'] < report['best_score']:
                    report.update(best_step=report['updates'], best_score=values['score'])
                    checkpoint('best.pt')
        report['training_validation_elapsed_seconds'] = time.monotonic() - start
        report['deadline_reached'] = expired()
        report['frozen_native_sha256_after'] = frozen_digest()
        report['frozen_native_unchanged'] = report['frozen_native_sha256_before'] == report['frozen_native_sha256_after']
        changed = [name for name, value in trainable.items() if not torch.equal(initial[name], value.detach().cpu())]
        report['changed_parameters_by_family'] = {
            family: sum(name.endswith((f'.a.{family}', f'.b.{family}')) for name in changed) for family in FAMILIES}
        report['progress_head_changed'] = any(name.startswith('progress.') for name in changed)
        report['all_four_banks_changed'] = all(report['changed_parameters_by_family'].values())
        if not report['frozen_native_unchanged']:
            raise ValueError('Frozen native tensors changed')
        if report['updates'] < 20 or not report['all_four_banks_changed'] or not report['progress_head_changed']:
            raise ValueError('Calibration did not pass the twenty-update/component-change gate')
        if report['best_step'] is None:
            raise ValueError('No updated candidate finished held-out validation; zero control is never eligible')
        report['all_updated_candidates_regressed_vs_zero'] = report['best_score'] > baseline['score']
        report.update(status='bounded_calibration_complete_not_online_evaluated',
                      max_allocated_bytes=torch.cuda.max_memory_allocated())
        checkpoint('final.pt')
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
