"""Mobile five-case diagnostic; old fixed-base script gate is never overridden."""
import json
from pathlib import Path
from bvi.eef_arm_scoring import wilson


def summarize(root,roster):
    rows=roster['dev'];records=[]
    for row in rows:
        path=Path(root)/row['case_id']/'V-mobile/result.json'
        records.append(json.loads(path.read_text()) if path.exists() else None)
    valid=[r for r in records if r and r['status']=='completed']
    counts={}
    for horizon in (200,600):
        k=sum(r[f'success_within_{horizon}'] for r in valid)
        counts[str(horizon)]=dict(planned_n=len(rows),valid_n=len(valid),
            censored_n=sum(bool(r) and r['status']!='completed' for r in records),
            unrun_n=sum(r is None for r in records),success_k=k if valid else None,
            valid_wilson95=wilson(k,len(valid)))
    return dict(condition='V-mobile',results=counts,test_authorized=False,
        prior_fixed_base_script_gate='failed 0/5; unchanged',
        cases={row['case_id']:None if r is None else {k:r.get(k) for k in (
            'status','termination_category','first_strict_success_step','funnel','localization_eval_only')}
            for row,r in zip(rows,records)},
        strata={row['case_id']:dict(category=row['object_category'],height=row['height_band'],scene=row['scene']) for row in rows},
        correlation_note='same five exposed development scenes; paired with prior V; not independent holdout evidence')
