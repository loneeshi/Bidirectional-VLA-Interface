"""CPU FK decomposition; does not execute a counterfactual or change a gate.

Run from repository root with PYTHONPATH=src;scripts and .venv/Scripts/python.exe.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from bvi.eef_tools import RobotFK

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
URDF=ROOT/'runs/real-handoff-eef-g0-20260928-run02/fetch.urdf'
POSES=ROOT/'research/c2/diagnostics/2026-09-29-eef-tool-smoke-cpu/resume-r2/smoke-poses.json'


def main():
    p=json.loads((OUT/'plan-005.json').read_text())
    goal=np.asarray(json.loads(POSES.read_text())['5']['poses'][-1])
    fk=RobotFK(URDF); records=p['ik_records']; index=0; output=[]
    for trace in p['executor_trace']:
        if trace['kind'] not in ('move','endpoint','close_align'):
            continue
        solution=records[index];index+=1
        if trace['kind']!='close_align':
            continue
        measured=np.asarray(trace['qpos']);solved=np.asarray(solution['qpos'])
        substituted=solved.copy();substituted[3]=measured[3]
        error=fk.tcp(measured)[:3,3]-goal[:3,3]
        bias=fk.tcp(solved)[:3,3]-goal[:3,3]
        torso=fk.tcp(substituted)[:3,3]-fk.tcp(solved)[:3,3]
        arm=fk.tcp(measured)[:3,3]-fk.tcp(substituted)[:3,3]
        assert np.max(abs(error-bias-torso-arm))<1e-12
        output.append({'step':trace['step'],'total_error_xyz_m':error.tolist(),
            'total_error_m':float(np.linalg.norm(error)),
            'ik_position_bias_xyz_m':bias.tolist(),
            'torso_after_solve_translation_xyz_m':torso.tolist(),
            'arm_after_solve_tracking_xyz_m':arm.tolist(),
            'residual_without_post_solve_torso_motion_m':float(np.linalg.norm(error-torso)),
            'torso_delta_m':trace['delta'][3],
            'normalized_arm_action_peak':max(abs(a) for a in trace['action'][:7]),
            'arm_joint_error_rad':(measured[[5,7,8,9,10,11,12]]-solved[[5,7,8,9,10,11,12]]).tolist()})
    assert index==len(records) and len(output)==10
    report={'scope':'CPU kinematic decomposition of archived physical measurements; not an executed counterfactual or new controller test',
        'method':'Replace only solved torso with measured after-step torso. Vectors add exactly. Causal dominance and a locked-torso outcome are not proved.',
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in (OUT/'plan-005.json',URDF,POSES,Path(__file__))},
        'records':output,'requires_user_torso_decision':True}
    (OUT/'closure-alignment-attribution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(output[-1]))


if __name__=='__main__':
    main()
