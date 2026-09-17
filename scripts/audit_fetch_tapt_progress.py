"""Read-only paired audit of the trained Fetch TAPT progress paths.

Uses the original held-out start/middle/end observations and frozen checkpoint.
Diagnostic interventions are process-local; no source/model/threshold is changed.
Run with an external timeout (900 seconds); no simulator, optimizer or API.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path.home() / 'bvi-research'
    source = root / 'src/AC-DiT'
    sys.path[:0] = [str(source), str(root / 'src/Bidirectional-VLA-Interface/src')]
    used = int(subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    assert used < 1024, 'GPU1 occupied; do not overlap workloads'
    os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
        HF_HOME=str(root / 'hf-cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
    out = root / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checkpoint = root / 'runs/fetch-tapt-sft-2026-09-16-run01/best.pt'
    report = dict(status='loading', started_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=sha(checkpoint), runner_sha256=sha(source / 'models/acdit_runner.py'),
        script_sha256=sha(__file__), seed=8181, api_calls=0, optimizer_updates=0,
        scope='paired held-out offline progress audit; no task-success evaluation',
        limitations=['one inference seed per observation', 'elapsed-time labels are not physical completion predicates',
                    'held-out SAC states, not the failed online rollout states'])
    def save():
        (out / 'result.json').write_text(json.dumps(report, indent=2))
    save()
    try:
        import h5py
        import numpy as np
        import torch
        import yaml
        from PIL import Image
        from transformers import AutoTokenizer, SiglipTextModel
        from model_wrappers.mshab_model import create_model
        from bvi.acdit_contract import to_unified
        from bvi.acdit_tapt import ACDiTFamilyTool
        torch.set_num_threads(2)
        torch.manual_seed(4017)
        np.random.seed(4017)
        os.chdir(source)
        encoder = json.loads((root / 'encoder-lock.json').read_text())
        wrapper = create_model(args=yaml.safe_load((source / 'configs/config.yaml').read_text()),
            pretrained=str(root / 'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
            device='cuda:0', dtype=torch.float32, method_name='AC-DiT', combine_flag=False,
            pretrained_vision_encoder_name_or_path=encoder['snapshot'],
            mobility_head_ckpt_path=str(root / 'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'))
        tokenizer = AutoTokenizer.from_pretrained(encoder['snapshot'])
        text_model = SiglipTextModel.from_pretrained(encoder['snapshot'], torch_dtype=torch.float32).eval()
        segments = json.loads((root / 'runs/fetch-tapt-teacher-2026-09-16/segments.json').read_text())
        selected = [{**w, 'sample_t': t} for w in segments['windows'] if w['split'] == 'validation'
            for t in sorted({w['start'], (w['start'] + w['end'] - 1) // 2, w['end'] - 1})]
        assert len(selected) == 33, 'Audit requires the original 33 held-out examples'
        cache = []
        # Exactly the original gate script's native observation preprocessing.
        for w in selected:
            t, end = w['sample_t'], w['end']
            tokens = tokenizer(w['instruction'], return_tensors='pt', padding=False)
            with torch.no_grad():
                embedding = text_model(input_ids=tokens['input_ids']).last_hidden_state[0]
            with h5py.File(w['trajectory']) as h:
                g = h[f'observations/{t:04d}']
                extra = {k: torch.from_numpy(g['extra'][k][:]) for k in g['extra']}
                qpos = g['agent/qpos'][0]
                native = np.r_[qpos[[2,4,5,6,7,8,9,10,1,3,0]],
                    extra['base_linear_vel'][0,0], extra['base_angular_vel'][0,2]].astype('float32')
                images, clouds = [], []
                for i in [t - 1, t]:
                    if i < 0:
                        images.extend([None, None, None]); clouds.append(None)
                    else:
                        gi = h[f'observations/{i:04d}']
                        images.extend([Image.fromarray(gi[f'sensor_data/{c}/rgb'][0])
                            for c in ['fetch_head', 'fetch_hand']] + [None])
                        clouds.append(gi['pointcloud/xyzrgb'][:].reshape(-1, 6))
                captured = {}
                original = wrapper.policy.predict_action
                def capture(**kwargs):
                    captured.update({k: v.detach().cpu().clone() for k, v in kwargs.items()})
                    return torch.zeros(1, 2, 128, device='cuda')
                wrapper.policy.predict_action = capture
                try:
                    with torch.no_grad():
                        wrapper.step(torch.from_numpy(native)[None], images, embedding, clouds, extra)
                finally:
                    wrapper.policy.predict_action = original
                actions = np.stack([to_unified(h[f'actions/{min(t+j,end-1):04d}'][0]) for j in range(2)])
                target = torch.tensor([[min((t+j+1-w['start'])/(end-w['start']), 1) for j in range(2)]])
                valid = torch.tensor([[t+j < end for j in range(2)]])
                cache.append((w, captured, torch.tensor(actions)[None], target, valid))
            report.update(status='caching', cached_examples=len(cache)); save()
        del text_model
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        tool = ACDiTFamilyTool(wrapper.policy, saved['report']['projection_paths'])
        params = dict(tool.named_parameters())
        assert set(saved['trainable']) == {n for n, p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n, v in saved['trainable'].items():
                assert params[n].shape == v.shape
                params[n].copy_(v)
        tool.eval()
        def trainable_hash():
            h = hashlib.sha256()
            for n, p in tool.named_parameters():
                if p.requires_grad:
                    h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before = trainable_hash()
        runner = wrapper.policy
        report.update(status='auditing', checkpoint_updates=saved['updates'], examples=33,
                      model_training=runner.training, prediction_type=runner.prediction_type)
        records = []
        with torch.no_grad(), (out / 'paired.jsonl').open('w') as log:
            for index, (w, cpu_batch, cpu_gt, cpu_target, cpu_valid) in enumerate(cache):
                batch = {k: v.cuda() for k, v in cpu_batch.items()}
                gt, target, valid = cpu_gt.cuda(), cpu_target.cuda(), cpu_valid.cuda()
                weighted = runner.perception_aware_multimodal_adaptor(
                    batch['img_tokens'], batch['pc_tokens'], batch['lang_tokens'])
                row = dict(index=index, window=w, target=target.cpu().tolist()[0], valid=valid.cpu().tolist()[0],
                    observation_progress=(w['sample_t']-w['start'])/(w['end']-w['start']), paths={})
                raw_adapt = runner.adapt_conditions_mobility_head
                @contextmanager
                def mobility(mode):
                    def adapt(**kwargs):
                        if mode != 'native':
                            kwargs['img_tokens'], kwargs['pc_tokens'] = (
                                (batch['img_tokens'], batch['pc_tokens']) if mode == 'raw' else weighted)
                        return raw_adapt(**kwargs)
                    runner.adapt_conditions_mobility_head = adapt
                    try:
                        yield
                    finally:
                        runner.adapt_conditions_mobility_head = raw_adapt
                def evaluate(name, *, inference=False, timestep=None, mobility_mode='native', action_input=None):
                    trace = []
                    current_t = []
                    def time_hook(module, args, kwargs):
                        current_t[:] = kwargs['t'].detach().cpu().reshape(-1).tolist()
                    def progress_hook(module, inputs):
                        trace.append(dict(timestep=current_t[:], progress=tool.progress(
                            inputs[0][:,-runner.pred_horizon:]).cpu().tolist()[0]))
                    h1 = runner.model.register_forward_pre_hook(time_hook, with_kwargs=True)
                    h2 = runner.model.final_layer.register_forward_pre_hook(progress_hook)
                    raw_randint = torch.randint
                    def fixed_randint(*args, **kwargs):
                        value = raw_randint(*args, **kwargs)
                        return value if timestep is None else torch.full_like(value, timestep)
                    torch.manual_seed(8181)  # reset before every path: no path-order RNG drift
                    try:
                        with mobility(mobility_mode), patch('torch.randint', fixed_randint):
                            if inference:
                                actions, progress = tool.predict(w['family'], batch)
                                loss = None
                            else:
                                loss = tool.training_loss(w['family'], {**batch,
                                    'action_gt': gt if action_input is None else action_input}, target, valid)
                                progress = loss['progress']
                                actions = None
                    finally:
                        h1.remove(); h2.remove()
                    pl = float((progress[valid]-target[valid]).square().mean())
                    result = dict(progress=progress.cpu().tolist()[0], progress_mse=pl,
                        trace=trace, mobility=mobility_mode, timestep_override=timestep)
                    if loss is not None:
                        result.update(action_loss=float(loss['action_loss']), joint_loss=float(loss['loss']))
                    if actions is not None:
                        mask = batch['action_mask'].expand_as(actions).bool() & valid[...,None]
                        result['active_action_mse_to_teacher'] = float((actions[mask]-gt[mask]).square().mean())
                    row['paths'][name] = result
                    return actions
                evaluate('train_random_gt')
                for timestep in [0, 200, 500, 999]:
                    evaluate(f'train_t{timestep}_gt', timestep=timestep)
                evaluate('train_t200_gt_weighted_mobility', timestep=200, mobility_mode='weighted')
                generated = evaluate('inference_native', inference=True)
                evaluate('inference_raw_mobility', inference=True, mobility_mode='raw')
                evaluate('train_t200_generated_actions', timestep=200, action_input=generated)
                records.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
                report.update(completed_examples=len(records), elapsed_seconds=time.monotonic()-started); save()
        report.update(status='complete', trainable_unchanged=before==trainable_hash(),
            finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            max_allocated_bytes=torch.cuda.max_memory_allocated())
        assert report['trainable_unchanged']
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
