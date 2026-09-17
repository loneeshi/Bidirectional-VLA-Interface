"""CPU-only historical handoff/replay plan. Does not run policy, simulator or training.

Locks validation3020/3021 and optional historical2025/2030 diagnostics. Training
origins3000..3003 are inventoried but never admitted to validation. Historical
AC observations are not relabeled as V8 workspace inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HANDOFF_BATCH = 'fetch-current-handoffs-2026-09-17-run01'
DIAGNOSTIC_BATCH = 'fetch-current-progress-eval-2026-09-17-run01'
LEGACY_BATCH = 'fetch-tapt-timing-2026-09-17-run01'


def read(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('CPU evidence file exceeds 32MiB bound')
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def scalar(value):
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError('Expected single-environment scalar')
        value = value[0]
    return value


def audit_episode(directory, role):
    directory = Path(directory)
    result = json.loads(read(directory / 'result.json'))
    seed = result['seed']
    allowed = {'validation': {3020, 3021}, 'locked_diagnostic': {2025, 2030}}
    if role not in allowed or seed not in allowed[role]:
        raise ValueError('Training parent or unlocked seed cannot enter handoff validation')
    if result['status'] != 'episode_completed':
        raise ValueError('Incomplete historical source')
    events = [json.loads(line) for line in read(directory / 'events.jsonl').splitlines()]
    steps, cases = {}, []
    latest_step = 0
    for n, event in enumerate(events):
        if event['event'] == 'step':
            latest_step = event['step']
            if latest_step in steps or latest_step != len(steps) + 1:
                raise ValueError('Action prefix is not contiguous')
            action = event['action']
            if len(action) != 13 or not all(math.isfinite(x) and abs(x) <= 1.00001 for x in action):
                raise ValueError('Historical applied action is not Fetch13 normalized')
            steps[latest_step] = event
        if event['event'] != 'family_switch' or event['family'] not in ('grasp', 'move'):
            continue
        physical = steps.get(latest_step)
        if physical is None:
            raise ValueError('Handoff lacks recorded physical boundary')
        extra = physical['extra']
        tcp = extra['tcp_pose_wrt_base'][0][:3]
        obj = extra['obj_pose_wrt_base'][0][:3]
        distance = math.dist(tcp, obj)
        held = bool(scalar(extra['is_grasped']))
        family = event['family']
        # Diagnostic classification only. These do not change control thresholds.
        classification = ('unheld_move' if not held else 'held_move') if family == 'move' else (
            'far_grasp_diagnostic' if distance >= .10 else 'near_grasp_candidate_not_completion')
        next_prediction = next((e for e in events[n+1:] if e['event'] == 'progress_prediction'
                                and e.get('call_id') == event['call_id']), None)
        prefix = [steps[s]['action'] for s in range(1, latest_step + 1)]
        prefix_bytes = json.dumps(prefix, separators=(',', ':'), allow_nan=False).encode()
        qpos, qvel = physical['qpos'][0], physical['qvel'][0]
        if len(qpos) != 15 or len(qvel) != 15:
            raise ValueError('Full robot qpos/qvel unavailable')
        cases.append(dict(case_id=f'{role}-seed{seed}-{family}-step{latest_step}',
            role=role, seed=seed, scene_split=result['split'], family=family,
            instruction=event['instruction'], boundary_step=latest_step,
            classification=classification, tcp_object_distance_m=distance, is_grasped=held,
            physical_completion=None, temporal_start_label=0.0,
            temporal_label_is_physical_reward=False, failed_endpoint_label=None,
            call_id=event['call_id'], decision_id=None if next_prediction is None else next_prediction.get('decision_id'),
            historical_first_progress=None if next_prediction is None else next_prediction['values'][0],
            historical_progress_contract=result.get('progress_label_contract'),
            applied_action_prefix=prefix, action_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
            boundary_qpos=qpos, boundary_qvel=qvel,
            boundary_extra=extra, boundary_native_info=physical['info'],
            input_readiness='workspace_replay_required', source_directory=str(directory.resolve())))
    schema = result.get('observation_schema', {})
    return dict(seed=seed, role=role, scene_split=result['split'], native_success=result['success'],
        result_sha256=sha(directory / 'result.json'), events_sha256=sha(directory / 'events.jsonl'),
        source_checkpoint_sha256=result.get('trained_checkpoint_sha256'),
        source_cameras=sorted(schema), recorded_workspace_camera='fetch_workspace' in schema,
        exact_reset_replay_verified=False, cases=cases)


def build(results, include_diagnostics=False):
    results = Path(results)
    sources = [audit_episode(results / HANDOFF_BATCH / f'validation-seed{s}', 'validation') for s in (3020, 3021)]
    if include_diagnostics:
        sources += [audit_episode(results / DIAGNOSTIC_BATCH / f'seed{s}', 'locked_diagnostic') for s in (2025, 2030)]
        # Preserve the specific historical 24.1cm regression, distinct from
        # current-observation2025 (which switches at roughly15cm).
        legacy = audit_episode(results / LEGACY_BATCH / 'legacy-pre-action', 'locked_diagnostic')
        historical_hashes = json.loads(read(results / LEGACY_BATCH / 'manifest.json'))
        for name in ('events.jsonl', 'result.json'):
            if sha(results / LEGACY_BATCH / 'legacy-pre-action' / name) != historical_hashes['legacy-pre-action/' + name]:
                raise ValueError('Historical 24cm source differs from archived source manifest')
        legacy['historical_manifest_sha256'] = sha(results / LEGACY_BATCH / 'manifest.json')
        legacy['scope'] = 'locked original24cm timing regression; not another independent episode seed'
        sources.append(legacy)
    archives = []
    for batch in [HANDOFF_BATCH] + ([DIAGNOSTIC_BATCH] if include_diagnostics else []):
        path = results / batch / 'backup.json'
        archives.append(dict(batch=batch, backup_record_sha256=sha(path), recorded_archive=json.loads(read(path)),
                             archive_hash_reverified_this_run=False))
    cases = [case for source in sources for case in source['cases']]
    jobs = []
    for case in cases:
        jobs.append(dict(case_id=case['case_id'], status='required_not_executed',
            replay_seed=case['seed'], scene_split=case['scene_split'], max_prefix_actions=case['boundary_step'],
            source_action_prefix_sha256=case['action_prefix_sha256'],
            required_inputs=['pinned historical simulator+assets+control config',
                'archive SHA verification and decision manifest/raw tensor hashes',
                'decision000000 simulator/controller snapshot or proven seeded-reset equivalence',
                'original subtask-start base XY from reset state (never handoff XY)'],
            procedure=['Restore exact reset and controller state; attach audited V8 workspace camera.',
                'Replay recorded applied Fetch13 actions exactly, with no model/GPT/SAC takeover.',
                'Compare every recorded qpos/qvel/extra boundary and native predicates; reject divergent replay.',
                'Independently reconstruct twice and hash simulator/controller/observation/origin/call state.',
                'Capture real workspace224 RGB and hand128 RGB plus original-origin-relative state30.',
                'Use identical accepted boundary snapshots for each frozen-model comparison arm.'],
            stop_on_mismatch=True, fallback_to_head_rgb=False,
            exact_state_restoration_proven=False))
    return dict(schema='fetch_pi05_wrong_handoff_validation_plan_v1', status='plan_only_replay_unverified',
        training_updates=0, api_calls=0, simulator_runs=0, sources=sources, archives=archives,
        excluded_training_origins=[3000, 3001, 3002, 3003], fixed_cases=cases, required_replay_jobs=jobs,
        pairing=dict(validation_near_or_held_reference_seed=3020, validation_wrong_reference_seed=3021,
            reference_comparison='same-family cross-parent diagnostic; not matched simulator states',
            within_case_comparison='all model arms must share twice-verified same boundary snapshot',
            same_parent_valid_wrong_pairs_available=False,
            missing_pair_job='If same-parent valid/wrong pairs are required, lock source/boundary acquisition protocol before new execution; do not select favorable substitutes.'),
        evaluation_contract=dict(threshold_changes=False, training_or_checkpoint_selection=False,
            progress_target_for_failed_unverified_tail=None,
            progress_false_completion='Count learned completion events whose separately logged physical completion predicate is false; report prediction and predicate independently.',
            rollback='Record frozen monitor regression/stagnation trigger, requested recovery and subsequent physical recovery separately; a trigger alone is not recovery.',
            physical_predicates='Use existing locked family completion predicates. Being near at grasp entry is not secure grasp; unheld move violates its precondition.',
            first_call_zero='Invocation-local temporal zero is not a zero physical reward or proof the scene is valid.',
            native_gate='Frozen V8 native Pick seeds2024..2028 scene_split val first; <=1/5 blocks TAPT and triggers read-only diagnosis.',
            overlap='Historical diagnostic2025 overlaps native five-seed gate; never treat it as an independent test or use it for checkpoint/threshold selection.'),
        acceptance='Plan does not authorize simulator execution or training. Native gate and user-required wrong-handoff validation remain pending.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/results')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-locked-diagnostics', action='store_true')
    args = p.parse_args()
    plan = build(args.results, args.include_locked_diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(plan, output, indent=2, allow_nan=False)
    print(json.dumps(dict(status=plan['status'], cases=len(plan['fixed_cases']), output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
