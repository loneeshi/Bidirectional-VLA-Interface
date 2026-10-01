"""Read-only server checks of the nine SAC assets and CPU importability."""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/pshuai/bvi-research')
CATEGORIES=('002_master_chef_can','003_cracker_box','004_sugar_box','005_tomato_soup_can',
    '007_tuna_fish_can','008_pudding_box','009_gelatin_box','010_potted_meat_can','024_bowl')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def main():
    from omegaconf import OmegaConf
    import torch
    hashes={};configs={}
    for category in CATEGORIES:
        base=ROOT/'checkpoints/mshab/rl/tidy_house/pick'/category
        cfg=OmegaConf.to_container(OmegaConf.load(base/'config.yml'),resolve=True)
        assert cfg['eval_env']['env_id']=='PickSubtaskTrain-v0'
        for name in ('config.yml','policy.pt'):hashes[str((base/name).relative_to(ROOT))]=sha(base/name)
        configs[category]=cfg['eval_env']
    assert not torch.cuda.is_initialized()
    result={'source_sha256':hashes,'checkpoint_root':str(ROOT/'checkpoints/mshab'),'eval_env':configs,
        'cuda_initialized':False,'simulator_steps':0,'provider_sends':0}
    Path('/tmp/bvi-arm-runtime-assets.cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'categories':len(configs),'hashed_files':len(hashes),'cuda_initialized':False}))


if __name__=='__main__':main()
