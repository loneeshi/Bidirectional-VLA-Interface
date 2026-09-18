"""Verified H5 IA reader using unchanged native24 OpenPI preprocessing."""
import hashlib
import json
from pathlib import Path

from bvi.ia_fetch_data import invocation_rows, invocation_sample, transform_sample


class InvocationDataset:
    def __init__(self, source, manifest, normalizer, data_config, model_config, split):
        from audit_s1_teacher_actions import audit_provenance
        from openpi import transforms
        import h5py
        if split not in ('train', 'validation'):
            raise ValueError('Explicit heldout split required')
        parsed, self.provenance = audit_provenance(source, manifest, normalizer)
        if data_config.normalizer_source_manifest_sha256 != self.provenance['source_manifest_sha256']:
            raise ValueError('Runtime data configuration provenance differs')
        self.rows = [r for r in invocation_rows(parsed, model_config.action_horizon) if r['split'] == split]
        if not self.rows:
            raise ValueError('Empty split')
        self.transform = transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])
        self.handle = h5py.File(source, 'r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return transform_sample(invocation_sample(self.handle[row['trajectory']], row), self.transform)

    def close(self):
        self.handle.close()
