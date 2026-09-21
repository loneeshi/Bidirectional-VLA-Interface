"""Fresh-process CPU verification of one bounded candidate; never launches work.

Validate completed budgets, frozen sampling, every dev selection point, and the
winning checkpoint. --verify-checkpoint restores original best855 then selected
parameters sequentially on CPU, checking unchanged non-LoRA leaf bytes/dtypes.
No model forward, GPU, optimizer initialization, retries, or online selection.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
from verify_s1_bounded_diagnostic import read_json, write_json, sha256, require, parameter_digest

SOURCE_HASH = '7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e'
TRAINER_HASH = 'bbe60d0ab7bf985b385cd90f68213a8c8e9aa6b32b50e1ee5005066d83af2e6d'
HELPER_HASH = '3dd94984a854528520eb69856f168136ac1417e3db3fd2e94fa3493cd2751221'
AUTHOR_COMMIT = 'f4eb160ba52b22c1e85fe432de59c24bbbac6187'
SCHEMA = 'bvi.s1-bounded-candidate/1'
PROTOCOL = 'best855_lora_only_weighted_train_dev8_v1'
REPO_ID = 'bvi/s1-official-pick-medium-train'
ACTIVE = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12)


def row_keys(rows):
    keys = [[r[k] for k in ('parent_id', 'call_index', 'observation_index')] for r in rows]
    require(all(type(v) is int and v >= 0 for key in keys for v in key), 'Invalid row identifiers')
    require(len({tuple(k) for k in keys}) == len(keys), 'Duplicate row identifiers')
    return np.asarray(keys, dtype=np.int64)


def validate_bounds(status, supervisor, identity):
    require(status.get('schema') == supervisor.get('schema') == identity.get('schema') == SCHEMA,
            'Candidate schema mismatch')
    count = status.get('additional_updates')
    require(type(count) is int and 0 <= count <= 500, 'Invalid added-update count')
    require(identity.get('max_updates') == 500 and identity.get('max_seconds') == 9000
            and identity.get('finalize_reserve') == 900 and identity.get('accumulation') == 8,
            'Frozen candidate budget changed')
    elapsed = supervisor.get('elapsed_seconds', float('nan'))
    require(type(elapsed) in (int, float) and np.isfinite(elapsed)
            and 0 < elapsed <= supervisor.get('budget_seconds', 0) == 9000,
            'Whole-process time budget exceeded or invalid')
    require(status.get('checkpoint_selection_complete') is True, 'Selection incomplete')
    require(status.get('source_checkpoint_step') == 855
            and status.get('source_checkpoint_parameters_sha256') == SOURCE_HASH
            and identity.get('source_checkpoint_parameters_sha256') == SOURCE_HASH,
            'Wrong source best855 checkpoint')
    require(status.get('diagnostic_only') is False and status.get('candidate_protocol') == PROTOCOL
            and status.get('optimizer') == 'reset_AdamW' and status.get('learning_rate') == 1e-4
            and status.get('progress') is False and status.get('family_bank') is False
            and status.get('api_calls') == 0, 'Candidate runtime contract changed')
    require(identity.get('trainer_sha256') == TRAINER_HASH and identity.get('helper_sha256') == HELPER_HASH,
            'Executed source hash differs from frozen candidate')
    require(identity.get('checkpoints') == [100, 250, 500, 'budget_stop']
            and identity.get('rng_seed') == 19092026 and identity.get('denoising_steps') == 10
            and identity.get('selection') == 'dev8_relative_rmse_mean_with_reach_yaw_torso_guard_tie_earlier',
            'Development selection protocol changed')
    expected = sorted({s for s in (100, 250, 500) if s <= count} | ({count} if count else set()))
    require(status.get('evaluated_steps') == expected, 'Skipped or added development selection points')
    require(status.get('stop_reason') == ('update_limit' if count == 500 else 'finalization_reserve'),
            'Unexpected stop reason')
    return expected


def verify_sampling(roster, archive, summary, true_actions):
    require(set(archive) >= {'weights', 'indices', 'current_actions'}, 'Sampling arrays missing')
    actions = np.asarray(archive['current_actions'])
    require(actions.shape == (len(roster), 13) and np.isfinite(actions).all()
            and np.array_equal(actions, true_actions), 'Sampler current actions differ from frozen H5')
    require(all(r['split'] == 'train' and r['parent_id'] not in (20, 21) for r in roster),
            'Sampler contains development rows')
    weights = np.ones(len(roster), dtype=np.float64)
    for i, row in enumerate(roster):
        if row['tool_family'] == 'reach' and max(abs(actions[i, 12]), abs(actions[i, 10])) >= .1:
            weights[i] = 3
        elif row['tool_family'] == 'move' and np.max(abs(actions[i, ACTIVE])) >= .1:
            weights[i] = 2
    require(np.array_equal(weights, archive['weights']), 'Sampling weights differ from preregistration')
    indices = np.asarray(archive['indices'])
    require(indices.shape == (4000,) and np.issubdtype(indices.dtype, np.integer), 'Invalid 4000-draw roster')
    expected = np.random.default_rng(19092026).choice(len(roster), size=4000, replace=True, p=weights / weights.sum())
    require(np.array_equal(indices, expected), 'Sampling draws differ from fixed seed19092026')
    require(summary.get('seed') == 19092026 and summary.get('draws') == 4000
            and summary.get('replacement') is True, 'Sampling summary protocol changed')
    require(summary.get('rules') == {'reach_yaw_or_torso_abs_ge_0.1': 3,
                'move_any_active11_abs_ge_0.1': 2, 'otherwise': 1}, 'Sampling rule description changed')
    expected_counts = dict(eligible_parent_counts=dict(Counter(str(r['parent_id']) for r in roster)),
        sampled_parent_counts=dict(Counter(str(roster[i]['parent_id']) for i in indices)),
        sampled_family_counts=dict(Counter(roster[i]['tool_family'] for i in indices)))
    require(all(summary.get(k) == v for k, v in expected_counts.items()), 'Sampling exposure counts differ')
    return dict(draws=4000, train_rows=len(roster), **expected_counts)


def read_dev_arrays(path, dev_rows):
    with np.load(path, allow_pickle=False) as archive:
        names = {'predicted', 'predicted_unclipped', 'target', 'valid', 'families', 'row_keys'}
        require(names <= set(archive.files), 'Missing development arrays')
        data = {name: archive[name].copy() for name in names}
    for key in ('predicted', 'predicted_unclipped', 'target'):
        require(data[key].shape == (8, 10, 13) and np.issubdtype(data[key].dtype, np.floating)
                and np.isfinite(data[key]).all(), 'Invalid development action arrays')
    valid = data['valid']
    require(valid.dtype == np.bool_ and valid.shape == (8, 10) and valid[:, 0].all()
            and np.array_equal(valid, np.arange(10)[None, :] < valid.sum(axis=1)[:, None]),
            'Invalid development action mask')
    require(np.array_equal(valid, np.asarray([r['action_valid'] for r in dev_rows], dtype=bool)),
            'Development masks differ from frozen roster')
    require(np.array_equal(data['row_keys'], row_keys(dev_rows))
            and np.array_equal(data['families'], [r['tool_family'] for r in dev_rows]),
            'Development row/family identity differs')
    applied = np.clip(data['predicted_unclipped'], -1, 1); applied[:, :, 8:10] = 0
    require(np.array_equal(applied, data['predicted']), 'Prediction clipping/head-mask differs')
    require(np.all(np.abs(data['target']) <= 1) and np.all(data['target'][:, :, 8:10] == 0),
            'Development targets violate Fetch13 contract')
    return data


def dev_metrics(data):
    error = data['predicted'].astype(np.float64) - data['target'].astype(np.float64)
    reach = data['families'] == 'reach'
    require(reach.sum() == 4, 'Expected four development reach samples')
    return dict(first=float(np.sqrt(np.mean(error[:, 0, ACTIVE] ** 2))),
                chunk=float(np.sqrt(np.mean(error[data['valid']][:, ACTIVE] ** 2))),
                yaw=float(np.sqrt(np.mean(error[reach, 0, 12] ** 2))),
                torso=float(np.sqrt(np.mean(error[reach, 0, 10] ** 2))))


def score_metrics(current, baseline):
    require(all(np.isfinite(v) and v >= 0 for group in (current, baseline) for v in group.values()),
            'Nonfinite or negative development RMSE')
    require(baseline['first'] > 0 and baseline['chunk'] > 0, 'Zero baseline denominator')
    first, chunk = current['first'] / baseline['first'], current['chunk'] / baseline['chunk']
    guards = dict(yaw_rmse=current['yaw'] <= baseline['yaw'] * 1.1,
                  torso_rmse=current['torso'] <= baseline['torso'] * 1.1)
    score = (first + chunk) / 2
    return dict(score=score, first_rmse_ratio=first, chunk_rmse_ratio=chunk,
                guards=guards, eligible=score < 1 and all(guards.values()))


def check_reported_metrics(document, actual):
    extracted = dict(first=document['all']['first_action']['rmse_active11'],
                     chunk=document['all']['valid_chunk']['rmse_active11'],
                     yaw=document['reach']['first_action']['yaw_rmse'],
                     torso=document['reach']['first_action']['torso_rmse'])
    require(all(np.isclose(extracted[k], actual[k], rtol=1e-6, atol=1e-9) for k in actual),
            'Saved metrics differ from float64 NPZ reconstruction')


def check_selection(record, actual):
    require(record.get('eligible') is actual['eligible'] and record.get('guards') == actual['guards'],
            'Eligibility/guard differs from reconstructed development arrays')
    require(all(np.isclose(record.get(k, float('nan')), actual[k], rtol=1e-6, atol=1e-9)
                for k in ('score', 'first_rmse_ratio', 'chunk_rmse_ratio')),
            'Selection score differs from reconstructed development arrays')


def summarize_leaves(flat):
    result = {}
    for key, value in flat.items():
        array = np.asarray(value)
        require(str(array.dtype) in ('float32', 'bfloat16') and np.isfinite(array).all(),
                'Invalid/nonfinite restored parameter dtype/value')
        result[key] = dict(shape=list(array.shape), dtype=str(array.dtype),
                           sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return result


def compare_frozen_leaves(source, selected):
    require(set(source) == set(selected), 'Source/selected parameter keys differ')
    frozen = lora = changed_lora = 0
    for key, old in source.items():
        new = selected[key]
        require(old['shape'] == new['shape'] and old['dtype'] == new['dtype'],
                'Source parameter shape/dtype changed: ' + '/'.join(map(str, key)))
        if 'lora' in '/'.join(map(str, key)).lower():
            lora += 1; changed_lora += old['sha256'] != new['sha256']
        else:
            frozen += 1
            require(old['sha256'] == new['sha256'], 'Frozen non-LoRA parameter bytes changed: ' + '/'.join(map(str, key)))
    require(lora == 20 and frozen > 0, 'Expected exactly 20 existing shared-LoRA leaves')
    return dict(non_lora_leaves=frozen, lora_leaves=lora, changed_lora_leaves=changed_lora,
                all_non_lora_byte_shape_dtype_equal=True)


def verify_checkpoint(source_path, selected_path, expected_hash):
    import dataclasses
    import jax
    import flax.nnx as nnx
    from flax import traverse_util
    import openpi
    from openpi.models import model as models, pi0_config
    require(all(d.platform == 'cpu' for d in jax.devices()), 'CPU-only checkpoint restore required')
    author = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True)
    require(commit == AUTHOR_COMMIT and not dirty.strip(), 'Expected clean pinned OpenPI checkout')
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True,
                              paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    cfg = dataclasses.replace(cfg, enable_progress_head=False)
    reference = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(cfg.create, jax.random.key(7))).to_pure_dict())
    summaries = []
    for checkpoint, expected in ((source_path, SOURCE_HASH), (selected_path, expected_hash)):
        loaded = models.restore_params(checkpoint / 'params', restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        require(set(flat) == set(reference) and all(value.shape == reference[key].shape for key, value in flat.items()),
                'Restored full parameter tree differs from pinned no-progress architecture')
        require(parameter_digest(flat) == expected, 'Restored checkpoint parameter hash mismatch')
        summaries.append(summarize_leaves(flat))
        del loaded, flat
        gc.collect()  # Keep only small per-leaf hashes before restoring the second full model.
    unchanged = compare_frozen_leaves(*summaries)
    return dict(status='passed_cpu_restore_and_frozen_parameter_comparison', author_commit=commit,
                source_parameters_sha256=SOURCE_HASH, selected_parameters_sha256=expected_hash,
                **unchanged, sequential_full_restores=2, forward_calls=0, gpu_used=False)


def verify_run(args):
    run = args.run_dir.resolve()
    report = dict(schema='bvi.s1-bounded-candidate-decision/1', run_dir=str(run),
        verified_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), verifier_sha256=sha256(__file__),
        status='verifying', decision='not_assessed', candidate_ready_for_oracle=False,
        policy_failure=False, native_success_established=False, gpu_used=False, forward_calls=0,
        optimizer_updates=0, api_calls=0, checkpoint_verification={'status': 'not_requested'}, evidence_sha256={})
    try:
        def document(name):
            report['evidence_sha256'][name] = sha256(run / name)
            return read_json(run / name)
        supervisor, status = document('supervisor.json'), document('status.json')
        report.update(run_status=status.get('status'), supervisor_status=supervisor.get('status'))
        if (supervisor.get('status') != 'exited' or supervisor.get('worker_exit_code') != 0 or
                status.get('status') not in ('completed_candidate', 'completed_no_eligible_candidate')):
            report.update(status='run_not_evaluable', decision='no_go_incomplete_or_infrastructure')
            return report
        identity = document('identity.json')
        points = validate_bounds(status, supervisor, identity)
        report['additional_updates'] = status['additional_updates']
        source_checkpoint = args.source_checkpoint.resolve()
        require(source_checkpoint.name == '855' and source_checkpoint.parent.name == 'best'
                and Path(identity['checkpoint']).parts[-2:] == ('best', '855'), 'Source must be frozen best855')
        from audit_s1_teacher_actions import audit_provenance
        from bvi.ia_fetch_data import invocation_rows
        import h5py
        manifest, provenance = audit_provenance(args.source, args.dataset_manifest, args.normalizer)
        require(all(identity.get(k) == v for k, v in provenance.items()), 'Candidate source/normalizer provenance differs')
        all_rows = invocation_rows(manifest, 10)
        train_rows, dev_rows = document('train-roster.json'), document('dev-roster.json')
        require(train_rows == [r for r in all_rows if r['split'] == 'train'], 'Training roster is not the complete frozen train split')
        row_keys(train_rows); row_keys(dev_rows)
        audit = read_json(args.audit_panel)
        require(sha256(args.audit_panel) == identity['audit_panel_sha256'], 'Original audit panel hash differs')
        require(dev_rows == [r for r in audit if r['split'] == 'validation'] and len(dev_rows) == 8
                and {r['parent_id'] for r in dev_rows} == {20, 21}
                and Counter(r['tool_family'] for r in dev_rows) == Counter(reach=4, grasp=2, move=2),
                'Development8 membership/family counts changed')
        source_lookup = {tuple(k): r for k, r in zip(row_keys(all_rows), all_rows)}
        require(all(source_lookup.get(tuple(k)) == r for k, r in zip(row_keys(dev_rows), dev_rows)),
                'Development row differs from source contract')
        source_assets = source_checkpoint / 'assets' / REPO_ID
        require(sha256(source_assets / 'norm_stats.json') == identity['normalizer_sha256'], 'Original best855 normalizer differs')
        source_contract = read_json(source_assets / 'bvi-state-contract.json')
        metadata = identity['policy_metadata']
        required_metadata = dict(robot='fetch', state_dim=24, action_dim=13,
            state_components=['native_qpos12', 'native_qvel12'], state_conditioning=True,
            state_source='env_native_agent', base_position_reference='world',
            base_camera='fetch_head', wrist_camera='fetch_hand', training_repo=REPO_ID,
            action_convention='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity',
            training_stage='S1_IA_single_bank_no_progress')
        require(all(metadata.get(k) == v for k, v in required_metadata.items()), 'Native24 input contract changed')
        for name in ('robot', 'state_dim', 'state_components', 'action_dim', 'base_position_reference',
                     'base_camera', 'wrist_camera', 'state_conditioning', 'training_repo',
                     'action_convention', 'state_source', 'normalizer_sha256', 'training_stage'):
            require(metadata.get(name) == source_contract.get(name), 'Input/model contract changed: ' + name)
        require(metadata.get('diagnostic_only') is False and metadata.get('candidate_protocol') == PROTOCOL
                and metadata.get('optimizer_reset') is True and metadata.get('source_checkpoint_step') == 855
                and metadata.get('lr_clock') == 'reset_constant_1e-4', 'Candidate metadata changed')
        report['evidence_sha256']['sampling.npz'] = sha256(run / 'sampling.npz')
        require(report['evidence_sha256']['sampling.npz'] == identity['sampling_sha256'], 'Frozen sampling artifact changed')
        with np.load(run / 'sampling.npz', allow_pickle=False) as archive:
            sampling = {k: archive[k].copy() for k in archive.files}
        with h5py.File(args.source, 'r') as handle:
            actions = np.asarray([handle[r['trajectory']]['actions'][r['observation_index']] for r in train_rows])
            true_targets = []
            for row in dev_rows:
                valid = sum(row['action_valid']); t = row['observation_index']
                real = np.asarray(handle[row['trajectory']]['actions'][t:t + valid], dtype=np.float32)
                true_targets.append(np.concatenate([real, np.repeat(real[-1:], 10 - valid, axis=0)]))
        report['sampling'] = verify_sampling(train_rows, sampling, document('sampling-summary.json'), actions)
        if status['additional_updates']:
            report['evidence_sha256']['train.jsonl'] = sha256(run / 'train.jsonl')
            logs = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
            require([r.get('step') for r in logs] == list(range(1, status['additional_updates'] + 1)), 'Optimizer log gaps or extra updates')
            require(all(r.get('lr') == 1e-4 and np.isfinite([r['loss'], r['grad_norm']]).all() for r in logs),
                    'Nonfinite or changed optimizer log')
        def arrays(step):
            name = f'dev-step{step:03d}.npz'; report['evidence_sha256'][name] = sha256(run / name)
            result = read_dev_arrays(run / name, dev_rows)
            require(np.allclose(result['target'], np.asarray(true_targets), rtol=0, atol=1e-5),
                    'Development targets differ from recorded expert actions')
            return result
        baseline = arrays(0); baseline_metrics = dev_metrics(baseline)
        check_reported_metrics(document('dev-step000.json'), baseline_metrics)
        selection = []
        if points:
            report['evidence_sha256']['selection.jsonl'] = sha256(run / 'selection.jsonl')
            selection = [json.loads(line) for line in (run / 'selection.jsonl').read_text().splitlines()]
        require([r.get('step') for r in selection] == points, 'Selection log differs from fixed schedule')
        eligible = []
        for record in selection:
            current = arrays(record['step'])
            for key in ('target', 'valid', 'families', 'row_keys'):
                require(current[key].dtype == baseline[key].dtype and np.array_equal(current[key], baseline[key]),
                        'Baseline/current development pairing changed: ' + key)
            actual = dev_metrics(current)
            check_reported_metrics(document(f"dev-step{record['step']:03d}.json"), actual)
            scored = score_metrics(actual, baseline_metrics); check_selection(record, scored)
            if scored['eligible']:
                eligible.append(dict(record))
        best = document('best.json')
        if not eligible:
            require(status['status'] == 'completed_no_eligible_candidate' and status.get('best') is None
                    and best.get('eligible') is False and best.get('checkpoint') is None,
                    'No eligible dev checkpoint but published winner')
            report.update(status='verified_no_eligible_candidate', decision='no_go_development_selection',
                          dev_selection_points=points, no_go_scope='frozen_development_rule_not_native_policy_failure')
            return report
        winner = min(eligible, key=lambda r: (r['score'], r['step']))
        require(status['status'] == 'completed_candidate' and status.get('best') == best, 'Published winner/status mismatch')
        require(best.get('schema') == SCHEMA and best.get('candidate_protocol') == PROTOCOL
                and best.get('selection') == 'dev8_only_no_online_selection'
                and best.get('diagnostic_only') is False and best.get('checkpoint_complete') is True
                and best.get('source_checkpoint_parameters_sha256') == SOURCE_HASH,
                'Winner contract is not non-diagnostic dev8 selection')
        require(best.get('step') == winner['step'], 'Winner is not lowest dev score with earlier tie break')
        check_selection(best, winner)
        checkpoint = run / 'best' / str(winner['step'])
        require(Path(best['checkpoint']).parts[-2:] == ('best', str(winner['step']))
                and (checkpoint / 'params').is_dir() and (checkpoint / 'train_state').is_dir(), 'Winner checkpoint incomplete')
        assets = checkpoint / 'assets' / REPO_ID
        require(read_json(assets / 'bvi-state-contract.json') == metadata
                and sha256(assets / 'norm_stats.json') == identity['normalizer_sha256'], 'Saved winner assets changed')
        evidence = best.get('dev_selection', {})
        require(evidence.get('split') == 'development' and evidence.get('used_online_success') is False,
                'Winner used non-development selection')
        for file_key, hash_key, expected_name in (
                ('evidence_file', 'sha256', f"dev-step{winner['step']:03d}.json"),
                ('raw_evidence_file', 'raw_sha256', f"dev-step{winner['step']:03d}.npz"),
                ('baseline_evidence_file', 'baseline_sha256', 'dev-step000.json'),
                ('baseline_raw_evidence_file', 'baseline_raw_sha256', 'dev-step000.npz')):
            require(evidence.get(file_key) == expected_name and evidence.get(hash_key) == sha256(run / expected_name),
                    'Winner evidence file/hash mismatch: ' + file_key)
        expected_hash = best.get('checkpoint_parameters_sha256')
        require(isinstance(expected_hash, str) and len(expected_hash) == 64
                and all(c in '0123456789abcdef' for c in expected_hash), 'Missing saved parameter digest')
        report.update(selected_step=winner['step'], score=winner['score'], checkpoint=str(checkpoint),
                      expected_checkpoint_parameters_sha256=expected_hash, dev_selection_points=points,
                      dev_selection_recomputed_from_raw_arrays=True)
        if not args.verify_checkpoint:
            report.update(status='development_selection_passed_restore_pending', decision='no_go_until_fresh_cpu_restore')
            return report
        report['checkpoint_verification'] = verify_checkpoint(source_checkpoint, checkpoint, expected_hash)
        report.update(status='verified_candidate_ready', decision='eligible_for_frozen_oracle_panel',
                      candidate_ready_for_oracle=True)
        return report
    except Exception as exc:
        report.update(status='evidence_invalid', decision='no_go_insufficient_or_infrastructure_evidence', error=repr(exc))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'source-checkpoint', 'source', 'dataset-manifest', 'normalizer', 'audit-panel'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verify-checkpoint', action='store_true', help='Sequential full CPU restores; zero forwards')
    parser.add_argument('--output', type=Path, help='Default <run-dir>/candidate-decision.json; never overwrite')
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / 'candidate-decision.json'
    if output.exists():
        parser.error('Output exists; preserve previous verification and choose another path')
    report = verify_run(args); write_json(output, report)
    print(json.dumps({k: report[k] for k in ('status', 'decision', 'candidate_ready_for_oracle', 'policy_failure')}))
    return 0 if report['status'] in ('verified_candidate_ready', 'verified_no_eligible_candidate',
                                    'development_selection_passed_restore_pending') else 2


if __name__ == '__main__':
    raise SystemExit(main())
