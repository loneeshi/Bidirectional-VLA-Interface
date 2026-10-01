"""Arm-specific durable broker; real transport is constructed only on lab Linux."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from bvi.eef_arm_contract import build_wire, audit_wire, validate_call, _clean, compact
from bvi.eef_compact_api import estimate_wire, parse_response, committed_cost
from eef_server_broker import save


def validate_scope(auth):
    expected={'revision':2,'status':'authorized','model':'gpt-6-astra','reasoning':'medium',
              'executor':'arm-coordinated-v2-600','provider_requests_cap':125,
              'gpu1_process_seconds_cap':13500,'prior_budget_transferred':0}
    if any(auth.get(k)!=v for k,v in expected.items()) or not auth.get('authorization_evidence'):
        raise ValueError('arm authorization missing or mismatched')
    if not 0<Decimal(auth['usd_cap'])<=65 or time.time()>=auth['expires_at_epoch']:
        raise ValueError('arm authorization cap or lease')


def wire_for(bundle,archive,condition):
    if bundle['kind']=='tool':return build_wire(bundle,archive,condition)
    if bundle['position'] not in {x['bundle']['position'] for x in archive}:
        raise ValueError('reflection case identity')
    clean_archive=[{'bundle':_clean(x['bundle'],condition),'call':x['call']} for x in archive]
    clean=dict(bundle,position=5)
    return compact.build_reflection(clean,clean_archive)


def process(bundle,archive,condition,folder,ledger_path,auth,auth_sha,send):
    """Batch holds an exclusive server lock. At most one send, never auto retry."""
    validate_scope(auth)
    if condition != 'V' or bundle['position'] not in auth['case_ids'] or bundle['authorization_sha256']!=auth_sha:
        raise ValueError('request scope binding')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    wire=wire_for(bundle,archive,condition)
    estimate=estimate_wire(wire);wire_sha=hashlib.sha256(wire).hexdigest()
    key=f"{bundle['position']}-{condition}-{bundle['kind']}-{bundle['turn']:03d}"
    ledger=json.loads(Path(ledger_path).read_text())
    if ledger['authorization_sha256']!=auth_sha or ledger['status']!='running':raise ValueError('ledger stopped or identity changed')
    entries=ledger['entries']
    prior=[e for e in entries if e['request_id']==key]
    if prior:
        if prior[0]['wire_sha256']!=wire_sha:raise ValueError('same request changed')
        if (folder/'delivery.json').exists():return json.loads((folder/'delivery.json').read_text())
        raise ValueError('prior send without durable delivery; reconcile, do not resend')
    if (len(entries)>=125 or sum(e['case_id']==bundle['position'] and e['condition']==condition for e in entries)>=25
        or committed_cost(entries)+Decimal(estimate['reserve_usd'])>Decimal(auth['usd_cap'])):
        ledger['status']='stopped_budget_exhausted';save(ledger_path,ledger)
        raise ValueError('arm budget exhausted before network')
    (folder/'request-wire.json').write_bytes(wire)
    save(folder/'request-bundle.json',bundle)
    entry={'request_id':key,'case_id':bundle['position'],'condition':condition,'wire_sha256':wire_sha,
           'status':'reserved','reserve_usd':estimate['reserve_usd'],'estimated_usd':None,
           'reserved_at_epoch':time.time()}
    entries.append(entry);save(ledger_path,ledger)
    try:
        raw=send(wire,key)
        (folder/'response-raw.json').write_bytes(raw)
        result=parse_response(raw,estimate,reflection=bundle['kind']=='reflection')
        if result['status']=='valid' and bundle['kind']=='tool':
            try:validate_call(result['call'])
            except ValueError:result.update(status='invalid_call',call=None,reason='invalid_arm_call')
        entry.update(result)
        if result['status']=='censored':ledger['status']='stopped_provider_contract'
    except Exception as exc:
        entry.update(status='unknown',error_type=type(exc).__name__)
        ledger['status']='stopped_unknown_outcome';save(ledger_path,ledger)
        raise RuntimeError('provider outcome unknown; reserved charge retained') from None
    save(ledger_path,ledger);save(folder/'delivery.json',result)
    return result


def runtime_sender(folder,remaining):
    if os.name!='posix' or not Path('/home/pshuai/bvi-research').is_dir():
        raise RuntimeError('API transport is lab-server-only')
    from eef_server_transport import load_runtime_key,PhaseTransport
    path=Path(folder)/'transport-phases.jsonl'
    def emit(value):
        with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
    return PhaseTransport(load_runtime_key(Path.home()/'.config/bvi/openai.env'),emit,remaining,timeout=180.)
