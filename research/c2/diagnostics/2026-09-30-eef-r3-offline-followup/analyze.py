from pathlib import Path
import json,numpy as np
from bvi.eef_tools import RobotFK,pose
from bvi.eef_tools_v6 import FetchToolsV6
from bvi.eef_tools_v3 import PreferredIK
out=Path(__file__).parent
r=json.loads(Path('research/c2/diagnostics/2026-09-30-eef-compact-640-pilot-r3/remote-results/results/plan-005-visual/result.json').read_text());fk=RobotFK(out/'fetch.urdf');model=json.loads((out/'response-model.json').read_text())['model'];tr=r['executor_trace'];rows=[]
for c in r['commands']:
 if c['call']['tool']!='move_to':continue
 ts=[x for x in tr if c['start_step']<x['step']<=c['start_step']+c['result']['steps_used']];first=next(x for x in ts if x['kind']=='move');i=first['step']-1
 q=np.array(tr[i-1]['qpos']);prev=np.array(tr[i-1]['delta']);e=FetchToolsV6(fk,lambda:q,lambda a:(_ for _ in ()).throw(Exception('No environment actions allowed')),lambda:0,lambda:{},model,read_qvel=lambda:np.zeros(15),control_dt=.05,initial_delta=prev)
 targets=[pose(p['position'],p['quaternion_xyzw']) for p in c['call']['poses']];plan=e._plan(targets);print('plan',c['turn'],plan['accepted'],flush=True)
 points,z=e.coupled_schedule(plan);end=ts[-1];idx=end['reference_index'];point=points[idx];row={'turn':c['turn'],'cursor':idx,'reference_count':len(points),'recorded_height':end['torso_reference_m'],'reconstructed_height':float(z[idx]),'samples':[]}
 for label,h in [('scheduled',z[idx]),('measured',end['qpos'][3])]:
  seed=np.array(end['qpos']);seed[3]=h;sol=PreferredIK(fk,torso=False).solve(point,seed);row['samples'].append({'mode':label,**sol.summary()})
 sol=PreferredIK(fk).solve(point,np.array(end['qpos']));row['ik8']=sol.summary();rows.append(row);print('finished',c['turn'],flush=True)
(out/'ik-reconstruction.json').write_text(json.dumps(rows,indent=2))
