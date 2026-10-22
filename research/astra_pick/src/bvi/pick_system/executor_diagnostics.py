"""EX phase diagnostics: immutable states and commands, never Pick scores.

Preparation is CPU-only. The lab worker reuses the existing v13/coordinated-v2
executor and exact snapshot restoration helper; it cannot synthesize held states.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .perception_capture import bound_path, canonical, digest, reference, save, sha

SCHEMA='bvi-pick-ex-diagnostics-v1'
CONDITION='pick-ex-executor-diagnostics-v1'
LIMITS={'units':26,'physical_steps':3760,'api_requests':0,'processes':26,
        'concurrent_processes':1,'seconds_per_process':600,'process_seconds':15600,
        'output_bytes':10*1024**3,'automatic_retries':0}
CASES=('arm-dev-000','arm-dev-004')
BASE_COMMANDS={
    'positive_turn':[{'tool':'move_base','turn_rad':1.,'forward_m':0.,'max_steps':80}],
    'negative_turn':[{'tool':'move_base','turn_rad':-1.,'forward_m':0.,'max_steps':80}],
    'turn_then_forward':[{'tool':'move_base','turn_rad':.5,'forward_m':0.,'max_steps':60},
                         {'tool':'move_base','turn_rad':0.,'forward_m':.2,'max_steps':60}],
    'reverse':[{'tool':'move_base','turn_rad':0.,'forward_m':-.2,'max_steps':80}],
}
SOURCE_FILES=(
    'src/bvi/pick_system/perception_capture.py','src/bvi/pick_system/diagnostics.py',
    'src/bvi/pick_system/broker.py','src/bvi/logging.py','src/bvi/protocol.py',
    'src/bvi/pick_system/executor_diagnostics.py','scripts/prepare_pick_system_executor_diagnostics.py',
    'scripts/finalize_pick_system_executor_release.py',
    'scripts/run_pick_system_executor_diagnostics.py','scripts/finalize_pick_system_executor_diagnostics.py','scripts/collect_pick_system_e1_replay_gt.py',
    'scripts/check_c2_snapshot_restore.py','scripts/arm_runtime_paths.py',
    'src/bvi/eef_arm_executor.py','src/bvi/eef_tools_v13.py','src/bvi/eef_tools_v12.py',
    'src/bvi/eef_tools_v3.py','src/bvi/eef_tools_v4.py','src/bvi/eef_tools_v8.py',
    'src/bvi/eef_coordinated_tools.py','src/bvi/eef_coordinated_tools_v2.py',
    'src/bvi/eef_attempt_recording.py','src/bvi/mshab_adapter.py',
    'scripts/finalize_arm_capability_attempt.py','scripts/finalize_eef_visual_attempt.py',
    'scripts/export_c2_collision_bundle.py','scripts/eef_server_broker.py',
    'scripts/visualize_eef_trajectory.py','scripts/render_astra_dual_path_demo.py',
    'scripts/project_eef_trajectory_into_demo.py','scripts/eef_commanded_trajectory.py')
MODEL='research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/response-model.json'
FK_MODEL='research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf'
GPU_BUFFERS=('cuda_rigid_body_data','cuda_articulation_qpos','cuda_articulation_qvel',
             'cuda_articulation_qf','cuda_articulation_target_qpos','cuda_articulation_target_qvel')


def scalar(value):
    while isinstance(value,list) and len(value)==1:value=value[0]
    return value


def source_dependencies(code,entries=None):
    """Pin the local import closure, including package initializers and renderers."""
    import ast
    code=Path(code);pending=list(SOURCE_FILES if entries is None else entries);seen=set()
    def module_files(name):
        parts=name.split('.')
        for root in (code/'src',code/'scripts'):
            leaf=root.joinpath(*parts)
            for candidate in (leaf.with_suffix('.py'),leaf/'__init__.py'):
                if candidate.is_file():yield candidate.relative_to(code).as_posix()
            for i in range(1,len(parts)):
                initializer=root.joinpath(*parts[:i])/'__init__.py'
                if initializer.is_file():yield initializer.relative_to(code).as_posix()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=code/name
        if path.suffix!='.py':continue
        package=path.relative_to(code/'src').parts[:-1] if path.is_relative_to(code/'src') else ()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                base='.'.join((*package[:len(package)-node.level+1],node.module or '')).rstrip('.') if node.level else node.module or ''
                names=[base]+[base+'.'+a.name for a in node.names if a.name!='*']
            for module in names:pending.extend(module_files(module))
    return seen|{MODEL,FK_MODEL}


def arm_commands(phase):
    if phase in ('empty_hold','loaded_hold'):
        return [{'tool':'hold_current','max_steps':150,'gripper':0. if phase=='loaded_hold' else None}]
    if phase=='lift':return [{'tool':'lift_relative','delta_base_m':[0.,0.,.08],'max_steps':150,'gripper':0.}]
    if phase=='return_to_rest':return [{'tool':'return_to_rest','max_steps':150,'gripper':0.}]
    raise ValueError('unknown arm phase')


def validate_sac_binding(binding,collection_row,assets):
    if set(binding)!={'config','policy'}:raise ValueError('SAC config/policy binding schema')
    for ref in binding.values():
        if (not isinstance(ref.get('path'),str) or not isinstance(ref.get('sha256'),str)
                or len(ref['sha256'])!=64 or any(c not in '0123456789abcdef' for c in ref['sha256'])):
            raise ValueError('SAC file receipt required')
    config=binding['config'];expected=collection_row.get('runtime_config')
    if config['path']!=expected or assets.get(expected)!=config['sha256']:
        raise ValueError('SAC config must be the D0-bound target category config')
    if binding['policy']['path']!=Path(expected).with_name('policy.pt').as_posix():
        raise ValueError('SAC policy must belong to the same official target checkpoint directory')


def units():
    rows=[]
    for name,commands in BASE_COMMANDS.items():
        for replicate in (1,2):rows.append(dict(unit_id=f'base-{name}-r{replicate}',kind='base',case_id=CASES[0],
            phase=name,replicate=replicate,step_cap=120,commands=commands))
    for case in CASES:
        for phase in ('empty_hold','loaded_hold','lift','return_to_rest'):
            for replicate in (1,2):rows.append(dict(unit_id=f'{case}-{phase}-r{replicate}',kind='arm',
                case_id=case,phase=phase,replicate=replicate,step_cap=150,commands=arm_commands(phase)))
    for case in CASES:rows.append(dict(unit_id=f'{case}-sac',kind='sac_reference',case_id=case,
        phase='official_sac_native',replicate=1,step_cap=200,commands=[{'tool':'official_sac','max_steps':200}]))
    return rows


def validate_snapshot(snapshot):
    """Require the full existing capture format; no qpos-only restore fallback."""
    required={'simulator','gpu_buffers','attributes','skipped','batched_rng_states',
              'python_rng','numpy_rng','torch_rng','cuda_rng'}
    if set(snapshot)!=required or set(snapshot['gpu_buffers'])!=set(GPU_BUFFERS):raise ValueError('complete simulator/controller/RNG snapshot required')
    owners=set(snapshot['attributes'])
    if 'adapter' not in owners or 'controller' not in owners or not any(o.startswith('wrapper') for o in owners):raise ValueError('wrapper and controller state missing')
    if not any(o.startswith('controller:') for o in owners):raise ValueError('controller child state missing')
    if set(snapshot['skipped'])!=owners:raise ValueError('unsupported-attribute coverage missing')
    # Runtime objects are restored by the native state and GPU-buffer paths;
    # silently skipping numeric target/counter/RNG state is not acceptable.
    for owner,fields in snapshot['skipped'].items():
        for name,kind in fields.items():
            if kind in ('ndarray','Tensor','float','int','bool','RandomState','Generator','deque'):
                raise ValueError('unrestored numerical state: '+owner+'/'+name)
    return True


def state_plain(value):
    """Value comparison for captured RNG/arrays/dataclasses; no memory addresses."""
    import dataclasses
    from collections import deque
    from enum import Enum
    import numpy as np
    if hasattr(value,'detach'):return {'array':state_plain(value.detach().cpu().numpy().tolist()),'dtype':str(value.dtype)}
    if isinstance(value,np.ndarray):return {'array':state_plain(value.tolist()),'dtype':str(value.dtype)}
    if isinstance(value,np.generic):return state_plain(value.item())
    if isinstance(value,np.random.RandomState):return {'RandomState':state_plain(value.get_state())}
    if isinstance(value,np.random.Generator):return {'Generator':state_plain(value.bit_generator.state)}
    if type(value).__name__=='Generator' and hasattr(value,'get_state'):return {'TorchGenerator':state_plain(value.get_state())}
    if isinstance(value,Enum):return {'enum':type(value).__name__,'value':value.value}
    if dataclasses.is_dataclass(value):return {'dataclass':type(value).__name__,'fields':state_plain(dataclasses.asdict(value))}
    if type(value).__name__=='Pose':
        if hasattr(value,'raw_pose'):return {'Pose':state_plain(value.raw_pose)}
        if hasattr(value,'p') and hasattr(value,'q'):return {'Pose':state_plain([value.p,value.q])}
    if type(value).__name__=='BatchedRNG' and hasattr(value,'rngs'):
        return {'BatchedRNG':state_plain([r.get_state() for r in value.rngs])}
    if isinstance(value,dict):return {str(k):state_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,deque)):return [state_plain(v) for v in value]
    if isinstance(value,set):return sorted([state_plain(v) for v in value],key=lambda x:canonical(x))
    if isinstance(value,bytes):return {'bytes_hex':value.hex()}
    if type(value) is float and not math.isfinite(value):return {'nonfinite_float':repr(value)}
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('unverified captured state type: '+type(value).__name__)


def state_difference_paths(before,after,*,limit=32):
    """Bounded evaluator-only mismatch locations; never dump state payloads."""
    rows=[]
    def visit(a,b,path):
        if len(rows)>=limit:return
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):
                rows.append({'path':path,'kind':'keys','missing':sorted(map(str,set(a)-set(b)))[:limit],
                             'extra':sorted(map(str,set(b)-set(a)))[:limit]})
            for key in sorted(set(a)&set(b),key=str):visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
            if len(a)!=len(b):rows.append({'path':path,'kind':'length','before':len(a),'after':len(b)})
            else:
                for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif digest(state_plain(a))!=digest(state_plain(b)):
            rows.append({'path':path,'kind':'value_or_type','before_type':type(a).__name__,'after_type':type(b).__name__})
    visit(before,after,'')
    return rows[:limit]


def snapshot_comparison(before,after):
    validate_snapshot(before);validate_snapshot(after)
    # Numeric GPU and simulator state has the established 1e-6 restore tolerance;
    # controller targets/task counters/cache/RNG and skipped coverage are exact.
    errors={name:numeric_error(before[name],after[name]) for name in ('simulator','gpu_buffers')}
    exact={name:digest(state_plain(before[name]))==digest(state_plain(after[name])) for name in before if name not in errors}
    failed=[name for name,e in errors.items() if e>1e-6]+[name for name,ok in exact.items() if not ok]
    return {'passed':all(e<=1e-6 for e in errors.values()) and all(exact.values()),
            'max_abs_error':{k:v if math.isfinite(v) else None for k,v in errors.items()},
            'exact_sections':exact,'physical_actions':0,
            'difference_paths':{name:state_difference_paths(before[name],after[name]) for name in failed}}


def load_snapshot(path,device,torch_api=None,receipt=None):
    """Preserve CPU RNG tensors; remap only native CUDA state to this device.

    Call only after verifying the checkpoint's frozen SHA. A global GPU
    map_location also moves torch_rng/CUDA-generator RNG ByteTensors, although
    the native restore API requires those generator states on CPU.
    """
    if torch_api is None:
        import torch as torch_api
    if not str(device).startswith('cuda'):raise ValueError('native EX CUDA device required')
    def location(storage,source):
        if source=='cpu':return storage
        if source.startswith('cuda:'):return torch_api.serialization.default_restore_location(storage,str(device))
        raise ValueError('unsupported checkpoint tensor device: '+source)
    source=path
    if receipt and receipt.get('compression')=='gzip':
        import gzip,hashlib,io
        expected_bytes=receipt.get('payload_bytes');expected_sha=receipt.get('payload_sha256')
        if (type(expected_bytes) is not int or not 0<expected_bytes<=512*1024**2 or
                not isinstance(expected_sha,str) or len(expected_sha)!=64):raise ValueError('compressed checkpoint payload receipt required')
        # Compressed bytes were pinned by bound_path before entry. Decompress
        # without a temporary .pt that could double the disk allocation.
        with gzip.open(path,'rb') as stream:payload=stream.read(expected_bytes+1)
        if len(payload)!=expected_bytes or hashlib.sha256(payload).hexdigest()!=expected_sha:
            raise ValueError('compressed checkpoint payload differs')
        source=io.BytesIO(payload)
    elif (receipt and receipt.get('compression') not in (None,'none')) or str(path).endswith('.gz'):
        raise ValueError('checkpoint compression receipt missing or unsupported')
    return torch_api.load(source,map_location=location,weights_only=False)


def stage_annotations(report):
    """Label frozen script intent; no stage inference from SAC motion."""
    if report.get('kind')=='sac_reference' or not report.get('frames'):return []
    phase=report['phase']
    stage={'lift':'lifting','return_to_rest':'return_to_rest','empty_hold':'empty_hold',
           'loaded_hold':'loaded_hold'}.get(phase,'navigation' if report['kind']=='base' else None)
    if stage is None:raise ValueError('unknown EX phase')
    return [{'id':report['unit_id']+'-script-phase','stage':stage,'start_step':0,
             'end_step':report['frames'][-1]['step'],
             'evidence_source':'frozen script intent: command_sha256='+report['command_sha256'],
             'scope':'issued phase, not proof that intended motion or grasp succeeded'}]


def diagnostic_metrics(report):
    """Reuse motion analysis; keep aggregate native forces distinct from contacts."""
    from .diagnostics import analyze
    result=analyze(report)
    result['native_aggregate_force_peaks_eval_only']={}
    for key in ('robot_force','robot_target_pairwise_force','robot_cumulative_force'):
        values=[scalar(f.get('official_info_eval_only',{}).get(key)) for f in report.get('frames',[])]
        values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
        result['native_aggregate_force_peaks_eval_only'][key]=max(values,default=None)
    result['diagnostic_only']=True;result['strict_pick_score']=False
    result['SAC_stage_scope']='native policy has no script stage labels; unlabelled motion remains unknown'
    return result


def numeric_error(a,b):
    import numpy as np
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):return float('inf')
        return max((numeric_error(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        if not isinstance(b,(tuple,list)) or len(a)!=len(b):return float('inf')
        return max((numeric_error(x,y) for x,y in zip(a,b)),default=0.)
    if hasattr(a,'detach'):a=a.detach().cpu().numpy()
    if hasattr(b,'detach'):b=b.detach().cpu().numpy()
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:return float('inf')
    if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':return float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.))
    return 0. if np.array_equal(a,b) else float('inf')


def select_phase_evidence(index_path,collection_row,collection_root,evidence_root):
    """Choose from newly captured states only, preserving missing held states."""
    root=Path(evidence_root);index_path=Path(index_path);run_root=index_path.parent
    index=json.loads(index_path.read_text(encoding='utf-8'))
    if index.get('capture_origin')!='simulator' or index['case_id']!=collection_row['case_id'] or index['source_family']!='official-state':
        raise ValueError('EX requires real D1 official-state source capture')
    if index['source_action_sha256']!=collection_row['source_action_sha256'] or index.get('status') not in ('completed','normal_terminal'):
        raise ValueError('D1 source mismatch or unresolved collection')
    states=[json.loads(s) for s in bound_path(run_root,index['state_trajectory']).read_text(encoding='utf-8').splitlines() if s]
    if [s['control_step'] for s in states]!=list(range(len(states))):raise ValueError('complete D1 phase trajectory required')
    checkpoints={c['control_step']:c for c in index.get('checkpoint_coverage',[])}
    old=json.loads(bound_path(collection_root,collection_row['source_result']).read_text(encoding='utf-8'))
    returns=[c['start_step'] for c in old['commands'] if (c.get('call') or {}).get('tool')=='return_to_rest']
    held=[s['control_step'] for s in states if bool(scalar(s['official_info_eval_only'].get('is_grasped',False)))
          and not bool(scalar(s['official_info_eval_only'].get('success',False))) and s['control_step'] in checkpoints]
    choices={'initial':0,'empty_hold':0,'loaded_hold':min(held) if held else None,
             'lift':min(held) if held else None,'return_to_rest':returns[0] if returns else None}
    output={}
    for phase,step in choices.items():
        if step is None or step not in checkpoints or step>=len(states):
            output[phase]={'status':'not_ready','reason':'required synchronized checkpoint absent'};continue
        state=states[step];info=state['official_info_eval_only'];grasped=bool(scalar(info.get('is_grasped',False)))
        if (phase in ('loaded_hold','lift','return_to_rest'))!=grasped:
            output[phase]={'status':'not_ready','reason':'actual D1 grasp state does not match phase'};continue
        if bool(scalar(info.get('fail',False))) or bool(scalar(info.get('success',False))):
            output[phase]={'status':'not_ready','reason':'checkpoint is already terminal'};continue
        checkpoint=checkpoints[step];binary=bound_path(run_root,checkpoint['checkpoint'])
        coverage=checkpoint.get('skipped')
        if not isinstance(coverage,dict) or 'controller' not in coverage:
            output[phase]={'status':'not_ready','reason':'controller snapshot coverage absent'};continue
        checkpoint_ref=reference(binary,root)
        checkpoint_ref.update({k:checkpoint['checkpoint'][k] for k in ('compression','payload_sha256','payload_bytes') if k in checkpoint['checkpoint']})
        output[phase]={'status':'bound_restore_unverified','control_step':step,
            'capture_index':reference(index_path,root),'checkpoint':checkpoint_ref,
            'state_trajectory':reference(bound_path(run_root,index['state_trajectory']),root),
            'state_sha256':digest(state),'skipped_sha256':digest(coverage),'grasped_eval_only':grasped,
            'restoration':'existing buffers restore; zero-action full-state equality required before command',
            'source_run_id':index['run_id']}
        if 'snapshot_contract' in checkpoint:
            from .snapshot_transport import CONTRACT
            if checkpoint['snapshot_contract']!=CONTRACT:raise ValueError('unknown checkpoint transport contract')
            output[phase]['snapshot_contract']=CONTRACT
            output[phase]['restoration']='v2 identity-remapped buffers; zero-action complete equality required before command'
    return output


def prepare(collection_path,index_paths,evidence_root,output,code,policy_bindings=None):
    """Freeze commands and real state files if available, otherwise finite gaps."""
    collection_path=Path(collection_path).resolve();root=Path(evidence_root).resolve();code=Path(code).resolve();out=Path(output)
    collection=json.loads(collection_path.read_text(encoding='utf-8'))
    if collection.get('schema')!='bvi-e1-collection-package-v1':raise ValueError('frozen D1 package required')
    indexes={};gaps=[];bindings={}
    if not collection.get('runtime_asset_sha256'):gaps.append({'reason':'D0 runtime assets not frozen'})
    for p in map(Path,index_paths):
        idx=json.loads(p.read_text(encoding='utf-8'))
        if idx['case_id'] in CASES and idx['source_family']=='official-state':
            if idx['case_id'] in indexes:raise ValueError('ambiguous D1 run for phase binding')
            indexes[idx['case_id']]=p
    selected={case:next(r for r in collection['rows'] if r['case_id']==case and r['source_family']=='official-state') for case in CASES}
    for case,row in selected.items():
        if case not in indexes:gaps.append({'case_id':case,'reason':'new D1 capture index unavailable'});continue
        try:bindings[case]=select_phase_evidence(indexes[case],row,collection_path.parent,root)
        except (ValueError,KeyError,OSError) as exc:gaps.append({'case_id':case,'reason':str(exc)})
    rows=units();policy_bindings=policy_bindings or {}
    for row in rows:
        phase='initial' if row['kind'] in ('base','sac_reference') else row['phase']
        row['state_binding']=bindings.get(row['case_id'],{}).get(phase,{'status':'not_ready','reason':'source state unavailable'})
        row['collection_row']=selected[row['case_id']]
        if row['kind']=='sac_reference':
            row['policy_binding']=policy_bindings.get(row['case_id'])
            if not row['policy_binding']:gaps.append({'unit_id':row['unit_id'],'reason':'official SAC config and policy hash binding missing'})
            else:validate_sac_binding(row['policy_binding'],selected[row['case_id']],collection['runtime_asset_sha256'])
        if row['state_binding']['status']=='not_ready':gaps.append({'unit_id':row['unit_id'],'reason':row['state_binding']['reason']})
        row['command_sha256']=digest(row['commands']);row['state_binding_sha256']=digest(row['state_binding'])
    source_pins={name:sha(code/name) for name in sorted(source_dependencies(code))}
    package={'schema':SCHEMA,'status':'frozen' if not gaps else 'not_ready','limits':LIMITS,
        'collection':reference(collection_path,root),'rows':rows,'gaps':gaps,'source_sha256':source_pins,
        'executor':'arm-coordinated-v2-600','runtime_commit':collection['runtime_commit'],
        'runtime_asset_sha256':collection.get('runtime_asset_sha256',{}),
        'diagnostic_only':True,'strict_pick_score':False,'api_requests':0,
        'restore_status':'unverified until lab zero-action equality','policy_inputs':'SAC official native; scripted EX has no model input'}
    out.mkdir(parents=True,exist_ok=False);save(out/'manifest.json',package)
    return package


def validate_package(manifest,root,code):
    if manifest.get('schema')!=SCHEMA or manifest.get('status')!='frozen' or manifest.get('limits')!=LIMITS:raise ValueError('EX frozen complete package required')
    if manifest.get('gaps') or not manifest.get('diagnostic_only') or manifest.get('strict_pick_score') is not False:raise ValueError('unresolved EX state or wrong score scope')
    collection_path=bound_path(root,manifest['collection'])
    collection=json.loads(collection_path.read_text(encoding='utf-8'))
    expected=units();rows=manifest['rows'];phase_cache={}
    if len(rows)!=26 or sum(r['step_cap'] for r in rows)!=3760:raise ValueError('EX unit/step allocation changed')
    for row,pattern in zip(rows,expected):
        if any(row[k]!=pattern[k] for k in ('unit_id','kind','case_id','phase','replicate','step_cap')):raise ValueError('EX order changed')
        if row['command_sha256']!=digest(row['commands']) or row['state_binding_sha256']!=digest(row['state_binding']):raise ValueError('EX command/state binding changed')
        if row['commands']!=pattern['commands']:raise ValueError('frozen regression commands changed')
        b=row['state_binding']
        if b['status']!='bound_restore_unverified':raise ValueError('EX checkpoint not bound')
        for key in ('capture_index','checkpoint','state_trajectory'):bound_path(root,b[key])
        original=next((r for r in collection['rows'] if r['case_id']==row['case_id'] and r['source_family']=='official-state'),None)
        if original!=row['collection_row']:raise ValueError('EX source collection row changed')
        if row['kind']=='sac_reference':validate_sac_binding(row['policy_binding'],original,manifest['runtime_asset_sha256'])
        cache_key=(row['case_id'],b['capture_index']['sha256'])
        if cache_key not in phase_cache:
            phase_cache[cache_key]=select_phase_evidence(bound_path(root,b['capture_index']),original,collection_path.parent,root)
        phase='initial' if row['kind'] in ('base','sac_reference') else row['phase']
        if b!=phase_cache[cache_key][phase]:raise ValueError('EX phase/checkpoint pairing changed')
    for name,expected_hash in manifest['source_sha256'].items():bound_path(code,{'path':name,'sha256':expected_hash})
    if not source_dependencies(code)<=set(manifest['source_sha256']):raise ValueError('EX executable source pins missing')
    bound_path(root,manifest['collection'])


def build_diagnostics_release(output):
    """Certify complete recorded diagnostics, not controller acceptance.

    The subsequent freeze decision is a separate evidence-bound record. It may
    document unresolved stability; no new jitter threshold is invented here.
    """
    output=Path(output);manifest_path=output/'manifest.json';manifest=json.loads(manifest_path.read_text())
    ledger_path=output/'ex-ledger.json';ledger=json.loads(ledger_path.read_text())
    authorization=output/'authorization.json'
    if (manifest.get('schema')!=SCHEMA or manifest.get('status')!='frozen' or manifest.get('limits')!=LIMITS
        or ledger.get('status')!='completed' or ledger.get('manifest_sha256')!=sha(manifest_path)
        or ledger.get('authorization_sha256')!=sha(authorization)):
        raise ValueError('complete bound EX ledger required')
    patterns=units()
    if [r['unit_id'] for r in manifest['rows']]!=[r['unit_id'] for r in patterns]:raise ValueError('EX release requires all26 frozen units')
    if [r['unit_id'] for r in ledger['attempts']]!=[r['unit_id'] for r in patterns]:raise ValueError('EX release requires all26 attempts')
    rows=[];base_ok=True
    for frozen,entry in zip(manifest['rows'],ledger['attempts']):
        folder=output/frozen['unit_id'];result_path=folder/'result.json';raw=json.loads(result_path.read_text())
        count=raw.get('simulator_actions')
        if (entry.get('status')!='completed' or entry.get('returncode')!=0 or not entry.get('media_complete')
            or raw.get('status')!='completed' or raw.get('execution_origin')!='simulator'
            or raw.get('diagnostic_only') is not True or raw.get('strict_pick_success') is not False
            or raw.get('unit_id')!=frozen['unit_id'] or raw.get('state_binding')!=frozen['state_binding']
            or raw.get('command_sha256')!=frozen['command_sha256'] or raw.get('api_requests')!=0
            or type(count) is not int or not 0<=count<=frozen['step_cap'] or (count==0 and not raw.get('commands')) or entry.get('actual_steps')!=count
            or [f['step'] for f in raw['frames']]!=list(range(1,count+1))):
            raise ValueError('EX incomplete or mismatched unit: '+frozen['unit_id'])
        restore_path=folder/'restore-receipt.json';restore=json.loads(restore_path.read_text())
        if restore.get('physical_actions')!=0:raise ValueError('EX restore consumed unrecorded actions')
        if frozen['kind']=='sac_reference':
            if restore.get('kind')!='native_SAC_same_initial_physical_state' or not restore.get('max_error') or any(
                    not math.isfinite(v) or v>1e-6 for v in restore['max_error'].values()):raise ValueError('SAC state equivalence missing')
        elif restore.get('passed') is not True:raise ValueError('EX state restoration unverified')
        media_path=next((folder/name/'media-receipt.json' for name in ('diagnostic-media-v2','diagnostic-media-v1')
                         if (folder/name/'media-receipt.json').is_file()),folder/'diagnostic-media-v2/media-receipt.json') if count else folder/'media-receipt.json'
        media=json.loads(media_path.read_text())
        if media.get('complete') is not True or media.get('raw_result_sha256')!=sha(result_path):raise ValueError('EX media source changed')
        if count:
            for ref in media['artifacts'].values():bound_path(folder/media['media_directory'],ref)
        elif media.get('no_physical_actions') is not True:raise ValueError('zero-action receipt required')
        analysis_path=folder/'phase-analysis.json';analysis=json.loads(analysis_path.read_text())
        if analysis!=diagnostic_metrics(raw):raise ValueError('EX analysis no longer matches raw trajectory')
        if frozen['kind']=='base':
            if len(raw['commands'])!=len(frozen['commands']):base_ok=False
            for command in raw['commands']:
                feedback=command.get('result',{})
                base_ok &= feedback.get('arrived') is True and feedback.get('accepted') is True and \
                    0<=feedback.get('position_error_m',float('inf'))<=.02 and \
                    0<=feedback.get('rotation_error_rad',float('inf'))<=.03
        rows.append({'unit_id':frozen['unit_id'],'case_id':frozen['case_id'],'kind':frozen['kind'],'phase':frozen['phase'],
            'actual_steps':count,'result':reference(result_path,output),'restore':reference(restore_path,output),
            'media':reference(media_path,output),'analysis':reference(analysis_path,output),
            'registration':reference(folder/'media-registration.json',output) if count else None})
    return {'schema':'bvi-pick-ex-release-v1','status':'diagnostics_complete_acceptance_pending',
        'diagnostic_only':True,'strict_pick_score':False,'executor':manifest['executor'],
        'manifest':reference(manifest_path,output),'ledger':reference(ledger_path,output),'authorization':reference(authorization,output),
        'source_sha256':manifest['source_sha256'],'asset_sha256':manifest['runtime_asset_sha256'],
        'rows':rows,'actual_physical_steps':sum(r['actual_steps'] for r in rows),'api_requests':0,
        'commands_within_existing_tolerances':bool(base_ok),
        'base_tolerances_source':'eef_tools_v13.move_base: position_error<=0.02m,rotation_error<=0.03rad,arrived=True',
        'freeze_decision_interface':{'diagnostics_release_sha256':'SHA of this release',
            'status':'passed or repair_required','executor':manifest['executor'],
            'required_evidence':'source_sha256,asset_sha256,acceptance_protocol receipt,findings and unresolved stability limitations'},
        'limitations':['Complete diagnostics alone do not validate a controller or strict Pick policy.',
            'Zero-action equality covers captured simulator/buffer/controller/RNG state, not unexposed PhysX solver caches.',
            'Official SAC stages are unlabelled without explicit stage evidence; no automatically paired stage claim.']}


def validate_diagnostics_release(path,code=None,evidence_root=None):
    path=Path(path);release=json.loads(path.read_text(encoding='utf-8'))
    if release!=build_diagnostics_release(path.parent):raise ValueError('EX release does not match complete diagnostics')
    if code is not None:
        for name,expected in release['source_sha256'].items():bound_path(code,{'path':name,'sha256':expected})
    if evidence_root is not None:
        if code is None:raise ValueError('full EX package validation requires code root')
        validate_package(json.loads(bound_path(path.parent,release['manifest']).read_text()),evidence_root,code)
    return release


def validate_authorization(auth,manifest_path,output,lab_root,now=None):
    now=now or datetime.now(timezone.utc)
    expiry=datetime.fromisoformat(auth.get('valid_until_utc','').replace('Z','+00:00'))
    if (auth.get('schema')!='bvi-pick-ex-authorization-v1' or auth.get('status')!='approved'
        or auth.get('manifest_sha256')!=sha(manifest_path) or auth.get('limits')!=LIMITS
        or expiry.tzinfo is None or expiry<=now or auth.get('scheduling_clearance_confirmed') is not True
        or not auth.get('cuda_visible_devices')):raise ValueError('independent EX resource authorization required')
    if Path(auth['output_root']).resolve()!=Path(output).resolve() or Path(auth['lab_root']).resolve()!=Path(lab_root).resolve():raise ValueError('EX authorized resource root changed')


def reserve(ledger,row):
    if ledger['status'] not in ('prepared','running'):raise ValueError('EX ledger closed')
    prior=ledger['attempts']
    if any(p['status'] not in ('completed','normal_failure') or p.get('media_complete') is not True for p in prior):raise ValueError('unresolved EX run/media blocks dispatch')
    if any(p['unit_id']==row['unit_id'] for p in prior):raise ValueError('EX is at most once, no retries')
    if len(prior)>=26 or sum(p['reserved_steps'] for p in prior)+row['step_cap']>3760:raise ValueError('EX allocation exhausted')
    expected=units()[len(prior)]['unit_id']
    if row['unit_id']!=expected:raise ValueError('EX frozen order required')
    entry={'unit_id':row['unit_id'],'case_id':row['case_id'],'status':'reserved','reserved_steps':row['step_cap'],
           'reserved_seconds':600,'actual_steps':None,'media_complete':False}
    prior.append(entry);ledger['status']='running';return entry


def execute_commands(executor,row,event=None):
    """Reuse existing executor control laws, including the hold inverse controller."""
    import numpy as np
    replies=[]
    for command in row['commands']:
        if executor.stopped or executor.steps>=row['step_cap']:break
        start=executor.steps;tool=command['tool'];remaining=min(command['max_steps'],row['step_cap']-start)
        call={**command,'note':'Frozen EX phase command; no model decision'}
        goal=None
        if tool=='lift_relative':
            from scipy.spatial.transform import Rotation
            goal=executor.fk.tcp(executor.read_qpos()).copy();goal[:3,3]+=np.asarray(command['delta_base_m'])
            call.update(tool='move_to',diagnostic_tool='lift_relative',poses=[{'position':goal[:3,3].tolist(),
                'quaternion_xyzw':Rotation.from_matrix(goal[:3,:3]).as_quat().tolist()}])
        pending={'turn':len(replies),'start_step':start,'call':call}
        if event:event('before',pending)
        if tool=='move_base':reply=executor.move_base(command['turn_rad'],command['forward_m'],max_steps=remaining)
        elif tool=='return_to_rest':executor.gripper=command['gripper'];reply=executor.return_to_rest()
        elif tool=='lift_relative':
            executor.gripper=command['gripper'];reply=executor.move_to(goal,max_steps=remaining)
        elif tool=='hold_current':
            if command['gripper'] is not None:executor.gripper=command['gripper']
            hold=np.asarray(executor.read_qpos()).copy()
            executor.deadline=executor._guard_deadline=min(executor.max_steps,start+remaining)
            while executor._available():executor._step(executor.inverse_action(np.zeros(15),hold),'diagnostic_hold',reference_qpos=hold.tolist())
            reply={'steps_used':executor.steps-start,'reason':'terminal' if executor.stopped else 'fixed_hold_completed'}
        else:raise ValueError('unknown frozen diagnostic command')
        replies.append({**pending,'result':reply})
        if event:event('after',replies[-1])
        if reply.get('arrived') is False or reply.get('accepted') is False:break
    return replies
