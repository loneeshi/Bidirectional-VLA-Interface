"""Resolve the fixed ManiSkill asset root before constructing any environment."""
import os
from pathlib import Path


def configure_arm_runtime(lab_root):
    from check_eef_g0_runtime_cpu import configure_runtime
    lab_root = Path(lab_root)
    configure_runtime(lab_root)
    # This pinned ManiSkill version appends /data to MS_ASSET_DIR itself.
    os.environ['MS_ASSET_DIR'] = str(lab_root / 'assets')
    from mani_skill import ASSET_DIR
    return validate_asset_directory(lab_root, ASSET_DIR)


def validate_asset_directory(lab_root, actual):
    expected = Path(lab_root) / 'assets/data'
    actual = Path(actual)
    if actual.resolve() != expected.resolve():
        raise ValueError('fixed ManiSkill asset directory mismatch')
    if not actual.is_dir():
        raise FileNotFoundError('fixed ManiSkill asset directory missing')
    return actual
