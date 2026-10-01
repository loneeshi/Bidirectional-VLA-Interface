"""Layer mobile files over exact r3 source bytes; template grants no permission."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIAG=ROOT/'research/c2/diagnostics/2026-10-01-arm-mobile-readiness-cpu'
OLD=ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-development-r3'


def main():
    with zipfile.ZipFile(OLD/'frozen-runtime.zip') as archive:
        sources={name:archive.read(name) for name in archive.namelist()}
    added=['src/bvi/eef_mobile_contract.py','src/bvi/eef_mobile_api.py',
        'scripts/eef_mobile_server_broker.py','scripts/run_arm_mobile_case.py',
        'scripts/run_arm_mobile_batch.py','scripts/summarize_arm_mobile.py',
        'scripts/check_arm_mobile_deployment_cpu.py',
        'scripts/finalize_arm_capability_attempt.py','scripts/finalize_eef_visual_attempt.py',
        'scripts/render_astra_dual_path_demo.py']
    for name in added:sources[name]=(ROOT/name).read_bytes()
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(sources.items())}
    with zipfile.ZipFile(DIAG/'frozen-mobile-runtime.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(sources.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,raw)
    zip_sha=hashlib.sha256((DIAG/'frozen-mobile-runtime.zip').read_bytes()).hexdigest()
    old=json.loads((OLD/'authorization.json').read_text(encoding='utf-8'))
    auth={k:v for k,v in old.items() if not k.startswith('consumed_')}
    auth.update(revision=3,condition='V-mobile',status='not_authorized',authorization_evidence=None,
        expires_at_epoch=0,gpu1_process_seconds_cap=9000,provider_requests_cap=125,usd_cap='65',
        output_root='/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1',
        source_sha256=hashes,deployment_zip_sha256=zip_sha,
        consumed_api_entries=[],consumed_gpu1_process_seconds=0,prior_budget_transferred=0,
        reference_result_root=old['output_root'])
    # Bind exact r3 measured initial states, independently from model inputs.
    bindings={}
    for case in auth['case_ids']:
        report=json.loads((OLD/'results'/case/'V/result.json').read_text(encoding='utf-8'))
        bindings[case]=report['initial_binding']
    (DIAG/'reference-bindings.eval-only.json').write_text(json.dumps(bindings,indent=2)+'\n',encoding='utf-8')
    # Remote initial-binding files were written with the existing canonical save helper.
    auth['reference_binding_sha256']={}
    for case,binding in bindings.items():
        path=DIAG/(case+'-binding.eval-only.json')
        path.write_bytes((json.dumps(binding,indent=2,allow_nan=False)+'\n').encode('utf-8'))
        auth['reference_binding_sha256'][case]=hashlib.sha256(path.read_bytes()).hexdigest()
    (DIAG/'authorization.template.json').write_text(json.dumps(auth,indent=2)+'\n',encoding='utf-8')
    roster=ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu/roster.eval-only.json'
    assert hashlib.sha256(roster.read_bytes()).hexdigest()==auth['roster_sha256']
    (DIAG/'roster.eval-only.json').write_bytes(roster.read_bytes())
    entries=json.loads((OLD/'api-ledger.json').read_text())['entries']
    current=[e for e in entries if not e['request_id'].startswith('prior:')]
    inp=[e['usage']['input_tokens'] for e in current];out=[e['usage']['output_tokens'] for e in current]
    from decimal import Decimal
    history=dict(r3_new_requests=len(current),input_median=statistics.median(inp),input_max=max(inp),
        output_median=statistics.median(out),output_max=max(out),
        estimated_usd=str(sum((Decimal(e['estimated_usd']) for e in current),Decimal(0))),
        actual_provider_bill_usd=None,old_scope_transferred=0)
    receipt=dict(status='CPU frozen; new run needs budget authorization',source_sha256=hashes,
        deployment_zip_sha256=zip_sha,base_archive_sha256=hashlib.sha256((OLD/'frozen-runtime.zip').read_bytes()).hexdigest(),
        added_mobile_files=added,history=history)
    (DIAG/'mobile-freeze.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archive_sha256=zip_sha,source_files=len(hashes),history=history)))


if __name__=='__main__':main()
