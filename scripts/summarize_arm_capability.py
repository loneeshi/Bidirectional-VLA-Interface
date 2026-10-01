"""Frozen CPU-only summary; missing/censored tasks never silently become failures."""
import json
from pathlib import Path
from collections import defaultdict
from bvi.eef_arm_scoring import wilson


def summarize(root,roster):
    root=Path(root);rows=roster['dev'];results={}
    for row in rows:
        for condition in ('V','SAC','script'):
            path=root/row['case_id']/condition/'result.json'
            results[row['case_id'],condition]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    def counts(selected,condition,horizon):
        records=[results[r['case_id'],condition] for r in selected]
        valid=[r for r in records if r and r['status']=='completed']
        k=sum(r[f'success_within_{horizon}'] for r in valid)
        return dict(planned_n=len(records),valid_n=len(valid),censored_n=sum(bool(r) and r['status']!='completed' for r in records),
            unrun_n=sum(r is None for r in records),success_k=k if valid else None,
            valid_wilson95=wilson(k,len(valid)),fixed_denominator_wilson95=wilson(k,len(records)) if len(valid)==len(records) else None)
    def valid(row,condition,horizon):
        r=results[row['case_id'],condition]
        return r[f'success_within_{horizon}'] if r and r['status']=='completed' else None
    strata=defaultdict(list);scenes=defaultdict(list)
    for row in rows:strata[row['object_category']+'/'+row['height_band']].append(row);scenes[row['scene']].append(row)
    subset=[r for r in rows if valid(r,'script',600) is True]
    conditions=('V','SAC','script')
    report={'conditions':{c:{str(h):counts(rows,c,h) for h in (200,600)} for c in conditions},
        'script_success_subset':{c:{str(h):counts(subset,c,h) for h in (200,600)} for c in ('V',)},
        'strata':{s:{c:counts(group,c,600) for c in conditions} for s,group in strata.items()},
        'scenes':{s:{c:counts(group,c,600) for c in conditions} for s,group in scenes.items()},
        'correlation_note':'scene-clustered observations; Wilson intervals are unadjusted descriptive binomial intervals',
        'sac_600_note':'SAC is observed only within its official 200-step configuration; no 600-step SAC extrapolation',
        'localization_eval_only':{r['case_id']:(results[r['case_id'],'V'] or {}).get('localization_eval_only',[]) for r in rows},
        'test_authorized':False}
    script=report['conditions']['script']['600']
    report['test_gate_met']=len(rows)==5 and script['valid_n']==5 and script['success_k']>=4
    report['next_action']='separate test proposal required' if report['test_gate_met'] else 'stop and report; no automatic repair or test run'
    return report
