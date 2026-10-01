"""One bounded CPU candidate audit on the 15 archived standard pose moves."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from bvi.eef_tools import RobotFK
from bvi.eef_arm_executor import ArmCoordinatedV2, ArmJointApproach

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research/c2/diagnostics'
OUT=D/'2026-10-01-arm-capability-cpu'


def no_step(action):raise AssertionError('CPU audit must never step')


def main():
    OUT.mkdir(exist_ok=True)
    path=OUT/'executor-cpu.json'
    if path.exists():raise ValueError('preserve candidate audit; no retuning/retry')
    fk=RobotFK(D/'2026-09-30-eef-r3-offline-followup/fetch.urdf')
    model=json.loads((D/'2026-09-30-eef-r3-offline-followup/response-model.json').read_text())['model']
    movements=json.loads((D/'2026-10-01-eef-systematic-foundations-cpu/movements.json').read_text())['movements']
    results={p.parent.name:json.loads(p.read_text()) for p in (D/'2026-10-01-eef-systematic-entries/results').glob('*/result.json')}
    rows=[]
    for move in movements:
        if move['tool']=='return_to_rest':continue
        entry=results[move['entry']];q=np.array(entry['settling']['qpos']);velocity=np.array(entry['settling']['qvel'])
        row={'id':move['id'],'kind':move['kind'],'entry':move['entry'],'original_clock_steps_left':entry['settling']['steps_left'],
             'cpu_new_attempt_budget':600,'conditions':{}}
        for cls in (ArmCoordinatedV2,ArmJointApproach):
            ex=cls(fk,lambda:q,no_step,lambda:float(q[-2:].sum()),lambda:{},model,max_steps=600,
                   read_qvel=lambda:velocity,control_dt=entry['control_dt'],initial_delta=np.asarray(entry['frames'][-1]['qpos'])-np.asarray(entry['frames'][-2]['qpos']) if len(entry['frames'])>1 else velocity*entry['control_dt'])
            started=time.perf_counter();goals=[np.array(x) for x in move['targets']]
            plan=ex._plan(goals)
            receipt={k:plan[k] for k in ('accepted','reason','minimum_steps','tails') if k in plan}
            receipt['last_records']=plan.get('records',[])[-2:]
            if plan['accepted']:
                try:
                    points,_=ex.coupled_schedule(plan)
                    receipt.update(reference_count=len(points),reference_steps=len(points)-1)
                except ValueError as error:receipt.update(accepted=False,reason='reference_timing_infeasible',detail=str(error))
            if cls is ArmJointApproach and (not plan['accepted'] or 'small' in move['id'] or move['kind']=='small'):
                _,records=ex.endpoint_candidates(goals[-1],q)
                receipt['independent_endpoint_search']=records
                receipt['unreachable_proven']=False
            receipt['cpu_seconds']=time.perf_counter()-started
            row['conditions'][cls.VERSION]=receipt
        rows.append(row)
        print(json.dumps({'id':row['id'],'conditions':{k:{f:v.get(f) for f in ('accepted','reason','reference_count')} for k,v in row['conditions'].items()}}),flush=True)
    assert len(rows)==15
    candidate=ArmJointApproach.VERSION
    all_planned=all(r['conditions'][candidate]['accepted'] for r in rows)
    table_under60=all(r['conditions'][candidate].get('reference_steps',float('inf'))<=60 for r in rows if r['kind']=='table')
    result={'rows':rows,'candidate_all_15_planned':all_planned,'candidate_table_at_most60':table_under60,
            'planning_gate_passed':all_planned and table_under60,
            'selected_executor':candidate if all_planned and table_under60 else ArmCoordinatedV2.VERSION,
            'selection_final_pending_regression':True,'gpu_runs':0,'api_calls':0,'simulator_steps':0,
            'unreachable_exemptions':[],'note':'A bounded IK search failure is not proof of unreachable target. No exemptions invented.'}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
