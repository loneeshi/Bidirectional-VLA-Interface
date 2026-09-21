"""Check every paired action, input and outcome; do not equate missing runs with failure."""
import argparse
import json
from pathlib import Path


def summarize(root, reference=None):
    panel=json.loads((root/'panel.json').read_text())
    pairs=[]
    seeds=sorted({e['seed'] for e in panel['episodes']})
    for seed in seeds:
        results=[];events=[]
        for path in ['official','project']:
            folder=(reference if reference is not None and path=='official' else root)/f'{seed}-{path}'
            if not (folder/'result.json').exists(): break
            results.append(json.loads((folder/'result.json').read_text()))
            events.append([json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()])
        if len(results)!=2 or any(r['status']!='completed' for r in results):
            pairs.append(dict(seed=seed,status='incomplete'));continue
        keys=['initial_state_sha256','initial_policy_obs_sha256','checkpoint_sha256']
        initial=all(results[0][k]==results[1][k] for k in keys)
        equal_length=len(events[0])==len(events[1])
        input_equal=equal_length and all(x['input_sha256']==y['input_sha256'] for x,y in zip(*events))
        delta=max((abs(xv-yv) for x,y in zip(*events)
                   for xv,yv in zip(x['raw_action'][0],y['raw_action'][0])),default=0)
        outcome_equal=equal_length and all(x['info']==y['info'] and x['qpos']==y['qpos']
                                          for x,y in zip(*events))
        pairs.append(dict(seed=seed,status='paired' if initial else 'unpaired',
            initial_exact=initial,equal_steps=equal_length,all_policy_inputs_exact=input_equal,
            raw_action_max_abs_difference=delta,all_info_and_qpos_exact=outcome_equal,
            official_success=results[0]['native_success'],project_success=results[1]['native_success'],
            official_steps=results[0]['steps'],project_steps=results[1]['steps']))
    return dict(panel_status=panel['status'],pairs=pairs,
        reference=str(reference) if reference else None,
        limitation='BC diagnostic paths only; not full VLA harness or paper benchmark',
        gpu_after=panel.get('gpu_after'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--reference',type=Path);a=p.parse_args()
    result=summarize(a.root,a.reference)
    (a.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
