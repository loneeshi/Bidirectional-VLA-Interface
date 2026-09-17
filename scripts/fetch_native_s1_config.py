"""S1-only native Fetch config; distinct from old V8 and TAPT family training.

Requires an explicit initialization and train-only normalization directory.
This module does not launch training or claim a checkpoint capability gate.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np


def config(repo_id, work, init_checkpoint, norm_directory, *, steps=100, batch=1):
    from fetch_openpi import config as legacy_config
    from openpi import transforms
    from openpi.shared import normalize

    if not init_checkpoint or not repo_id or steps < 1 or batch < 1:
        raise ValueError('Explicit initialization, training repo and positive bounds required')
    norm_directory = Path(norm_directory)
    provenance = json.loads((norm_directory/'provenance.json').read_text())
    norm_path = norm_directory/'norm_stats.json'
    digest = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    if (provenance.get('status') != 'train_only_stats_ready_not_runtime_validated'
            or provenance.get('validation_frames_used') != 0
            or provenance.get('dimensions') != {'state':24, 'actions':13}
            or provenance.get('norm_stats_sha256') != digest):
        raise ValueError('Missing or incompatible train-only normalizer provenance')
    stats = normalize.load(norm_directory)
    for key, dimension in [('state',24),('actions',13)]:
        for field in ('mean','std','q01','q99'):
            values = np.asarray(getattr(stats[key],field))
            if values.shape != (dimension,) or not np.isfinite(values).all():
                raise ValueError('Wrong native normalization dimensions/values')
    base = legacy_config(repo_id, work, steps=steps, batch=batch,
                         include_velocity=True, init_checkpoint=init_checkpoint)
    # Reuse action13 outputs/model conventions, but never legacy state30 metadata
    # or the author's LIBERO progress-specific required repack fields.
    @dataclasses.dataclass(frozen=True)
    class NativeData(type(base.data)):
        def create(self, assets_dirs, model_config):
            result = super().create(assets_dirs, model_config)
            repack = transforms.Group(inputs=[transforms.RepackTransform({
                'observation/image':'image', 'observation/wrist_image':'wrist_image',
                'observation/state':'state', 'actions':'actions', 'prompt':'prompt'})])
            return dataclasses.replace(result, norm_stats=stats, use_quantile_norm=True,
                                       repack_transforms=repack)
    metadata = dict(base.policy_metadata)
    metadata.update(state_dim=24,state_components=['native_qpos12','native_qvel12'],
                    state_source='env_native_agent',base_position_reference='world',
                    base_camera='fetch_head',wrist_camera='fetch_hand',
                    evaluation_scope='S1_official_train_fixed_heldout_native_gate',
                    normalizer_sha256=digest,initialization=str(init_checkpoint),
                    training_stage='S1_ordinary_target_domain_SFT_not_TAPT')
    model = base.model
    if hasattr(model,'enable_progress_head'):
        model = dataclasses.replace(model,enable_progress_head=False)
    return dataclasses.replace(base,name='pi05_fetch_native24_s1',exp_name='official-pilot',
                               model=model,data=NativeData(repo_id=repo_id,base_config=base.data.base_config,
                                                        extra_delta_transform=False),
                               use_val_set=False,progress_loss_weight=0.0,
                               policy_metadata=metadata)


def native_dataset(data_config, model_config, dataset_root):
    """Explicit-root dataset for either pre-registered parent split.

    Does not apply the author's episode-hash train/val repartition or synthesize
    progress labels. This S1 adapter returns action learning inputs only.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.training import data_loader
    from openpi import transforms
    root=Path(dataset_root)
    manifest=json.loads((root.parent/'manifest.json').read_text())
    split=root.name
    if split not in ('train','validation'):
        raise ValueError('Explicit train/validation root required')
    expected=manifest['parent_split'][split]
    rows=[e for e in manifest['episodes'] if e['split']==split]
    if [e['parent_id'] for e in rows]!=expected:
        raise ValueError('Parent membership differs')
    if set(manifest['parent_split']['train']) & set(manifest['parent_split']['validation']):
        raise ValueError('Leaked parent split')
    dataset=LeRobotDataset(data_config.repo_id,root=root,
        delta_timestamps={'actions':[t/20 for t in range(model_config.action_horizon)]})
    if dataset.meta.fps!=20 or len(dataset)!=sum(e['exported_steps'] for e in rows):
        raise ValueError('Dataset frame rate/count differs')
    dataset=data_loader.TransformedDataset(dataset,
        [transforms.PromptFromLeRobotTask(dataset.meta.tasks)])
    return data_loader.transform_dataset(dataset,data_config)
