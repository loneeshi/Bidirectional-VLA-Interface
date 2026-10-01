"""Audit frozen A1 one-step predictions against archived close-alignment actions.

No fitting, simulator steps, API requests or counterfactual physical execution.
Run from repo root with PYTHONPATH=src;scripts and .venv/Scripts/python.exe.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from bvi.eef_response_model import QIND,AIND

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
MODEL=ROOT/'research/c2/diagnostics/2026-09-28-eef-g0-streaming-stage-a/response-model.json'


def main():
    case=json.loads((OUT/'plan-005.json').read_text(encoding='utf-8'))
    model=json.loads(MODEL.read_text(encoding='utf-8'))['model']
    rho=np.array([p['rho'] for p in model['joint_parameters']])
    beta=np.array([p['beta'] for p in model['joint_parameters']])
    traces=case['executor_trace']; rows=[]
    for i,trace in enumerate(traces):
        if trace['kind']!='close_align':
            continue
        previous=np.array(traces[i-1]['delta'])[QIND]
        action=np.array(trace['action'])[AIND]
        actual=np.array(trace['delta'])[QIND]
        predicted=rho*previous+.1*beta*action
        rows.append({'step':trace['step'],'torso_q_m':trace['qpos'][3],
            'torso_action':float(action[-1]),
            'torso_predicted_delta_m':float(predicted[-1]),
            'torso_observed_delta_m':float(actual[-1]),
            'wrist_action':float(action[5]),
            'wrist_predicted_delta_rad':float(predicted[5]),
            'wrist_observed_delta_rad':float(actual[5]),
            'predicted_joint_delta':predicted.tolist(),
            'observed_joint_delta':actual.tolist(),
            'residual':(actual-predicted).tolist()})
    assert len(rows)==10
    assert max(abs(r['torso_predicted_delta_m']) for r in rows)<1e-12
    report={'method':'Frozen A1 one-step prediction conditioned on actual previous measured displacement and applied action. No refit or simulated counterfactual.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (OUT/'plan-005.json',MODEL,Path(__file__))},
        'rows':rows,'all_alignment_ik_accepted':all(x['accepted'] for x in case['ik_records'][-10:]),
        'wrist_saturated_steps':sum(abs(r['wrist_action'])>=.999999 for r in rows),
        'torso_alignment_start_m':traces[14]['qpos'][3],
        'torso_alignment_end_m':traces[24]['qpos'][3],
        'gpu_process_seconds':0,'api_requests':0,'simulator_actions':0}
    (OUT/'control-model-residuals.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'alignment_steps':len(rows),
        'ik_all_accepted':report['all_alignment_ik_accepted'],
        'wrist_saturated_steps':report['wrist_saturated_steps'],
        'max_abs_torso_residual_m':max(abs(r['torso_observed_delta_m']) for r in rows)}))


if __name__=='__main__':
    main()
