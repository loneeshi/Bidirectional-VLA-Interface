"""Read-only config resolution; no environment construction or provider access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import traceback
root = Path('/home/pshuai/bvi-research')
deploy = root / 'deploy/arm-revision2-20261001'
auth = json.loads((deploy/'authorization.json').read_text())
row = json.loads((deploy/'roster.eval-only.json').read_text())['dev'][0]
checkpoint = Path(auth['checkpoint_root'])/'rl/tidy_house/pick'/row['object_category']
receipt = {'checkpoint_config': str(checkpoint/'config.yml'), 'checkpoint_config_exists': (checkpoint/'config.yml').exists(), 'simulator_actions': 0, 'api_sends': 0}
try:
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(OmegaConf.load(checkpoint/'config.yml'), resolve=True)
    receipt['config_loaded'] = True
    receipt['env_config'] = raw['eval_env']
except Exception:
    receipt['traceback'] = traceback.format_exc()
import sys
sys.path[:0] = [str(root/'src/official-mshab-runtime/mshab'), str(root/'src/official-mshab-runtime/ManiSkill')]
os.environ['MS_ASSET_DIR'] = str(root/'assets/data')
from mani_skill import ASSET_DIR
receipt['worker_MS_ASSET_DIR'] = os.environ['MS_ASSET_DIR']
receipt['effective_ASSET_DIR'] = str(ASSET_DIR)
receipt['effective_asset_dir_exists'] = ASSET_DIR.exists()
receipt['expected_asset_dir'] = str(root/'assets/data')
receipt['expected_asset_dir_exists'] = (root/'assets/data').exists()
receipt['diagnosis'] = 'Frozen worker sets assets/data, but fixed ManiSkill appends data, resolving assets/data/data. Original worker suppresses traceback; exact first failing asset read is unavailable.'
import torch
receipt['cuda_initialized'] = torch.cuda.is_initialized()
(deploy/'startup-cpu-diagnosis.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
