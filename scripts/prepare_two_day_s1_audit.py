"""CPU-only frozen sample panel and input evidence for the two-day S1 diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset', 'normalizer', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from PIL import Image, ImageDraw
    import h5py
    from audit_s1_teacher_actions import audit_provenance
    from bvi.ia_fetch_data import invocation_rows, invocation_sample
    from fetch_native_s1_config import config
    from openpi import transforms

    manifest, provenance = audit_provenance(a.source, a.dataset / 'manifest.json', a.normalizer)
    rows = invocation_rows(manifest)
    # Selection uses source order only, before viewing predictions or outcomes.
    selected = [r for r in rows if r['parent_id'] in (0, 1)]
    parents = {e['parent_id']: e for e in manifest['episodes']}
    assert all(parents[n]['split'] == 'train' and parents[n]['native_success'] for n in (0, 1))
    assert 64 <= len(selected) <= 128
    assert set(r['tool_family'] for r in selected) == {'reach', 'grasp', 'move'}
    panel = dict(schema_version=1, **provenance, checkpoint_step=855,
                 selection='all valid invocation rows from first two successful training parents 0 and 1',
                 rows=[{k: r[k] for k in ('parent_id', 'call_index', 'observation_index')} for r in selected])
    (a.output / 'diagnostic-panel.json').write_text(json.dumps(panel, indent=2))
    cfg = config('bvi/s1-official-pick-medium-train', str(a.output / 'unused'),
                 str(a.checkpoint / 'params'), a.normalizer, steps=100, batch=1)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    front = transforms.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                               transforms.Normalize(dc.norm_stats, use_quantiles=True)])
    model_transform = transforms.compose(dc.model_transforms.inputs)
    audit = []
    for split, ids in [('train', (0, 1)), ('validation', (20, 21))]:
        for parent in ids:
            assert parents[parent]['split'] == split
            for family, count in [('reach', 2), ('grasp', 1), ('move', 1)]:
                available = [r for r in rows if r['parent_id'] == parent and r['tool_family'] == family]
                assert len(available) >= count
                indices = [0, len(available) // 2] if count == 2 else [len(available) // 2]
                audit.extend(available[i] for i in indices)
    (a.output / 'audit-panel.json').write_text(json.dumps(audit, indent=2))
    records = []
    with h5py.File(a.source, 'r') as h:
        for index, row in enumerate(audit):
            raw = invocation_sample(h[row['trajectory']], row)
            normalized = front({k: v for k, v in raw.items() if k != 'actions_is_pad'})
            # Preserve before tokenization, since TokenizePrompt consumes prompt.
            state = np.asarray(normalized['state']).copy()
            prompt = normalized['prompt']
            clean_prompt = prompt.strip().replace('_', ' ').replace('\n', ' ')
            bins = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            tokenizer_text = f'Task: {clean_prompt}, State: ' + ' '.join(map(str, bins)) + ';\nAction: '
            actual = model_transform(normalized)
            d = a.output / f'frame-{index:02d}'
            d.mkdir()
            frames = []
            for key in ('image', 'wrist_image'):
                Image.fromarray(raw[key]).save(d / f'raw-{key}.png')
                frames.append((key, Image.fromarray(raw[key]).resize((224, 224))))
            for key, value in actual['image'].items():
                value = np.asarray(value)
                # Keep lossless actual tensor as well as a display rendering.
                np.save(d / f'model-{key}.npy', value)
                pixels = value if value.dtype == np.uint8 else np.clip((value + 1) * 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(d / f'model-{key}.png')
                if 'right' not in key:
                    frames.append((key, Image.fromarray(pixels)))
            canvas = Image.new('RGB', (224 * len(frames), 250), 'white')
            draw = ImageDraw.Draw(canvas)
            for j, (label, im) in enumerate(frames):
                canvas.paste(im, (224 * j, 26)); draw.text((224 * j + 2, 5), label, fill='black')
            canvas.save(d / 'comparison.png')
            record = dict(row=row, camera_mapping={'image': 'fetch_head', 'wrist_image': 'fetch_hand'},
                          raw_state=raw['state'].tolist(), normalized_state=state.tolist(), discrete_state=bins.tolist(),
                          state_outside_quantiles=np.flatnonzero((state < -1) | (state > 1)).tolist(),
                          tokenizer_input=tokenizer_text, prompt=prompt,
                          token_ids=np.asarray(actual['tokenized_prompt']).tolist(),
                          token_mask=np.asarray(actual['tokenized_prompt_mask']).tolist(),
                          valid_tokens=int(np.asarray(actual['tokenized_prompt_mask']).sum()),
                          token_capacity=len(actual['tokenized_prompt']),
                          input_hashes={f.name: digest(f) for f in d.iterdir()},
                          note='Model input before train=True image augmentation; no policy forward.')
            (d / 'record.json').write_text(json.dumps(record, indent=2))
            records.append(record)
    summary = dict(status='completed_cpu_input_export', provenance=provenance,
                   audit_count=len(audit), diagnostic_rows=len(selected), train_parents=[0, 1], dev_parents=[20, 21],
                   audit_family_counts={'train': {'reach':4,'grasp':2,'move':2},'validation': {'reach':4,'grasp':2,'move':2}},
                   token_counts=[r['valid_tokens'] for r in records],
                   quantile_ood_rows=sum(bool(r['state_outside_quantiles']) for r in records),
                   seconds=time.monotonic()-started, model_forwards=0, optimizer_updates=0, simulator_steps=0, api_calls=0)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
