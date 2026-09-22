"""Build a frozen empirical C2 prior from completed independent train-spawn probes."""
import argparse, glob, hashlib, json, math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    cache={}
    records=[]
    for summary_path in sorted(a.probe_root.glob('case-*/attempt-001/summary.json')):
        binding_path=summary_path.with_name('binding.json')
        summary=json.loads(summary_path.read_text()); binding=json.loads(binding_path.read_text())
        initial=torch.load(summary_path.with_name('initial.pt'),map_location='cpu',weights_only=False)
        state=initial['observation']['state'].reshape(-1).float()
        if state.numel()!=42:
            raise ValueError('Expected official 42D policy state')
        # Upstream state = agent24 + tcp7 + object7 + goal3 + grasp1.
        target_xy=state[31:33] if binding['skill']=='pick' else state[38:40]
        binding['relative_target_distance_xy']=float(torch.linalg.vector_norm(target_xy))
        records.append((binding,summary))
    if len(records)!=24:
        raise ValueError('Frozen pilot requires all 24 completed rows')
    parents=sorted({b['uid'] for b,_ in records})
    bins=[]
    for skill in ('pick','place'):
        for category in ('024_bowl','009_gelatin_box'):
            rows=[s for b,s in records if b['skill']==skill and b['category']==category]
            source=[b for b,s in records if b['skill']==skill and b['category']==category]
            distances=[b['relative_target_distance_xy'] for b in source
                       if b['relative_target_distance_xy'] is not None]
            strict_distances=[b['relative_target_distance_xy'] for b,s in records
                if b['skill']==skill and b['category']==category and
                s['ever_native_success'] and not s['force_violation'] and
                b['relative_target_distance_xy'] is not None]
            bins.append({'skill':skill,'object_category':category,'samples':len(rows),
                'successes':sum(x['ever_native_success'] for x in rows),
                'strict_successes':sum(x['ever_native_success'] and not x['force_violation'] for x in rows),
                'force_violations':sum(x['force_violation'] for x in rows),
                'first_success_step_range':[
                    min(x['first_success_step'] for x in rows if x['first_success_step'] is not None),
                    max(x['first_success_step'] for x in rows if x['first_success_step'] is not None)],
                'guidance':('navigate close to the object before Pick; use short SAC slices and verify grasp'
                    if skill=='pick' else
                    'navigate while holding to a clear destination-side approach; minimize prolonged contact during Place'),
                'official_spawn_distance_xy_m_range':([min(distances),max(distances)] if distances else None),
                'strict_success_distance_xy_m_range':([min(strict_distances),max(strict_distances)]
                                                       if strict_distances else None),
                'pose_range_status':'relative distance support only; orientation and collision geometry unmeasured'})
    data={'schema':'spawn-prior/1','training_plan_uids':parents,'bins':bins,
          'scope':'two_categories_three_train_scenes_two_spawns_each',
          'limitations':['Ranges are relative target distance at official spawns, not an optimum.',
                         'Orientation, arm, torso and collision geometry were not isolated.',
                         'Place guidance is qualitative and force-aware.',
                         'Use only as empirical tool guidance, never as completion evidence.']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(hashlib.sha256(a.output.read_bytes()).hexdigest())

if __name__=='__main__':main()
