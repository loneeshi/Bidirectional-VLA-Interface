"""Derive a separate audit from an immutable batch archive; no simulator/API calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

from bvi.bridge import atomic_json
from bvi.sac_interface_baseline import summarize
from run_sac_interface_baseline import classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--bridge-events', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Audit output must be empty; preserve previous reports')
    with tarfile.open(args.archive) as archive:
        def read(name):
            return archive.extractfile(name).read().decode('utf-8')
        prefix = 'runs/sac-interface-baseline-20260920/'
        original = json.loads(read(prefix + 'manifest.json'))
        derived = copy.deepcopy(original)
        rows = []
        for row in derived['episodes'][:20]:
            for attempt in row['attempts']:
                directory = '/'.join(attempt['directory'].split('/')[-2:])
                summary = json.loads(read(prefix + directory + '/summary.json'))
                events = [json.loads(line) for line in read(prefix + directory + '/events.jsonl').splitlines()]
                rejection = any(e.get('event') == 'coordinator_rejected' for e in events)
                status, result = classify(attempt['result']['returncode'], summary, model_rejection=rejection)
                attempt['original_status'] = attempt['status']
                attempt.update(status=status, result=result)
                if attempt is row['attempts'][-1]:
                    row['status'] = status
                    steps = [e for e in events if e.get('event') == 'mshab_step']
                    rows.append(dict(seed=row['seed'], plan_uid=row['plan_uid'],
                        original_status=attempt['original_status'], audited_status=status,
                        **result, last_native_info=steps[-1]['info'] if steps else None))
        remote_ids = set()
        for member in archive.getnames():
            if member.endswith('/events.jsonl') and '/seed-' in member:
                for line in read(member).splitlines():
                    e = json.loads(line)
                    if e.get('event') == 'api_request_started': remote_ids.add(e['attempt_id'])
    events = [json.loads(line) for line in args.bridge_events.read_text(encoding='utf-8').splitlines()]
    starts = {e['bridge_id']: e for e in events if e.get('event') == 'bridge_attempt_started'}
    usage = {e['bridge_id']: e for e in events if e.get('event') == 'bridge_api_usage'}
    assert remote_ids == starts.keys() == usage.keys(), 'Bridge and remote request IDs must reconcile'
    total = len(starts)
    report = dict(source_archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        **summarize(derived, derived['batches'][0]),
        physical_actions_all_latest_rows=sum(row['steps'] or 0 for row in rows),
        episode_wall_seconds_all_latest_rows=sum(row['wall_seconds'] or 0 for row in rows),
        global_requests_used=total, global_requests_cap=800, global_requests_remaining=800-total,
        new_twenty_worst_case_requests=800, request_shortfall_for_twenty=total,
        reserved_cost_usd=sum(e['reserved_cost_usd'] for e in starts.values()),
        provider_actual_cost_usd=None, actual_remaining_usd=None,
        cost_status='unreconciled; reservation is not verified billing',
        tokens={key:sum(e.get('usage', {}).get(key, 0) for e in usage.values())
                for key in ('input_tokens', 'output_tokens', 'total_tokens')},
        rerun_status='blocked_budget_gate', rerun_planned=20, rerun_not_run=list(range(20)),
        rows=rows)
    report['schema'] = 'bvi-sac-followup-audit/2'
    # The archive and source manifest remain unchanged. Derived copies are marked.
    derived['audit_source'] = str(args.archive)
    derived['audit_only_not_resume_manifest'] = True
    atomic_json(args.output / 'audited-manifest.json', derived)
    atomic_json(args.output / 'audit.json', report)
    atomic_json(args.output / 'api-reconciliation.json', {
        'accounting_role':'reconciliation_of_existing_usage_not_additional_calls',
        'requests':list(starts.values()), 'usage':list(usage.values()),
        'remote_ids_equal_local_ids':True, 'actual_cost_usd':None})
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
