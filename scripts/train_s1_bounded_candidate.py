"""One gated candidate: original best855, new LoRA optimizer, dev8 selection.

No resume, grid, online checkpoint selection, or diagnostic-checkpoint reuse.
--preflight reads evidence/data only. GPU execution additionally requires passed
diagnostic and CPU restore evidence. Linux supervisor includes all work in 9000s.
"""
import argparse
from collections import Counter
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import train_s1_bounded_diagnostic as base

SOURCE_HASH = '7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e'
SCHEMA = 'bvi.s1-bounded-candidate/1'
PROTOCOL = 'best855_lora_only_weighted_train_dev8_v1'
SEED = 19092026


def row_key(row):
    return tuple(row[k] for k in ('parent_id', 'call_index', 'observation_index'))


def select_dev(panel, source_rows):
    selected = [r for r in panel if r['split'] == 'validation']
    lookup = {row_key(r): r for r in source_rows if r['split'] == 'validation'}
    keys = [row_key(r) for r in selected]
    if (len(selected) != 8 or len(set(keys)) != 8 or
            {r['parent_id'] for r in selected} != {20, 21} or
            Counter(r['tool_family'] for r in selected) != Counter(reach=4, grasp=2, move=2)):
        raise ValueError('Dev panel must contain frozen validation8, parents20/21, 4reach2grasp2move')
    for row in selected:
        if row_key(row) not in lookup or row != lookup[row_key(row)]:
            raise ValueError('Dev row differs from source IA contract')
    return selected


def sample_weights(rows, actions):
    import numpy as np
    actions = np.asarray(actions)
    if actions.shape != (len(rows), 13) or not np.isfinite(actions).all() or (abs(actions) > 1.00001).any():
        raise ValueError('Expected finite current controller action13 per train row')
    if any(r['split'] != 'train' or r['parent_id'] in (20, 21) for r in rows):
        raise ValueError('Sampler cannot contain dev/validation rows')
    weights = np.ones(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        if row['tool_family'] == 'reach' and max(abs(actions[i, 12]), abs(actions[i, 10])) >= .1:
            weights[i] = 3
        elif row['tool_family'] == 'move' and np.max(abs(actions[i, base.ACTIVE])) >= .1:
            weights[i] = 2
    return weights


def fixed_sampling(rows, actions):
    import numpy as np
    weights = sample_weights(rows, actions)
    indices = np.random.default_rng(SEED).choice(len(rows), size=4000, replace=True, p=weights / weights.sum())
    return weights, indices


def score_dev(current, baseline):
    import math
    ratios = []
    for scope in ('first_action', 'valid_chunk'):
        denominator = baseline['all'][scope]['rmse_active11']
        numerator = current['all'][scope]['rmse_active11']
        if not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(numerator) or numerator < 0:
            raise ValueError('Dev selection requires finite positive baseline RMSE')
        ratios.append(numerator / denominator)
    guard = {}
    for channel in ('yaw_rmse', 'torso_rmse'):
        old = baseline['reach']['first_action'][channel]
        new = current['reach']['first_action'][channel]
        if not math.isfinite(old) or not math.isfinite(new) or min(old, new) < 0:
            raise ValueError('Invalid dev reach guard metric')
        guard[channel] = new <= 1.1 * old
    score = sum(ratios) / 2
    return dict(score=score, eligible=score < 1 and all(guard.values()), guards=guard,
                first_rmse_ratio=ratios[0], chunk_rmse_ratio=ratios[1])


def prefer(new, old):
    return new['eligible'] and (old is None or (new['score'], new['step']) < (old['score'], old['step']))


def best_record(output, checkpoint, result, parameter_hash):
    """Publish selection only after all metrics/raw files and checkpoint exist."""
    if (not result.get('eligible') or not 1 <= result.get('step', 0) <= 500 or
            not 0 <= result.get('score', float('inf')) < 1 or
            not all(result.get('guards', {}).get(k) is True for k in ('yaw_rmse', 'torso_rmse'))):
        raise ValueError('Cannot publish ineligible development selection')
    if (len(parameter_hash) != 64 or any(c not in '0123456789abcdef' for c in parameter_hash) or
            not (checkpoint / 'params').is_dir() or not (checkpoint / 'train_state').is_dir()):
        raise ValueError('Cannot publish incomplete checkpoint identity')
    step = result['step']
    evidence = dict(split='development', used_online_success=False)
    for name, filename, hash_name in (
            ('evidence_file', f'dev-step{step:03d}.json', 'sha256'),
            ('raw_evidence_file', f'dev-step{step:03d}.npz', 'raw_sha256'),
            ('baseline_evidence_file', 'dev-step000.json', 'baseline_sha256'),
            ('baseline_raw_evidence_file', 'dev-step000.npz', 'baseline_raw_sha256')):
        evidence[name] = filename
        evidence[hash_name] = base.file_hash(output / filename)
    selected = json.loads((output / evidence['evidence_file']).read_text())
    baseline = json.loads((output / evidence['baseline_evidence_file']).read_text())
    recomputed = score_dev(selected, baseline)
    if any(result.get(key) != recomputed[key] for key in recomputed):
        raise ValueError('Selected development score differs from actual saved metrics')
    return dict(result, schema=SCHEMA, candidate_protocol=PROTOCOL, diagnostic_only=False,
                checkpoint=str(checkpoint.resolve()), checkpoint_complete=True,
                checkpoint_parameters_sha256=parameter_hash, fresh_checkpoint_restore_verified=False,
                selection='dev8_only_no_online_selection', source_checkpoint_parameters_sha256=SOURCE_HASH,
                dev_selection=evidence)


def verify_launch_gate(decision, status, supervisor):
    if (decision.get('schema') != 'bvi.s1-bounded-diagnostic-decision/1' or
            decision.get('status') != 'verified_metric_go' or decision.get('candidate_eligible') is not True):
        raise ValueError('Real diagnostic decision has not authorized candidate')
    if (status.get('status') != 'completed_diagnostic_requires_review' or
            status.get('source_parameters_sha256') != SOURCE_HASH or
            supervisor.get('status') != 'exited' or supervisor.get('worker_exit_code') != 0):
        raise ValueError('Diagnostic incomplete, timed out, or wrong source checkpoint')
    gate = decision.get('metric_gate', {})
    expected = {'first_action_active11_rmse', 'valid_chunk_active11_rmse',
                'reach_yaw_first_rmse', 'reach_torso_first_rmse'}
    if (gate.get('passed') is not True or set(gate.get('checks', {})) != expected or
            any(gate['checks'][name].get('passed') is not True for name in expected)):
        raise ValueError('Verified diagnostic metric gate missing or failed')
    proof = decision.get('checkpoint_verification', {})
    if (proof.get('status') != 'passed_cpu_restore' or proof.get('forward_calls') != 0 or
            proof.get('gpu_used') is not False or not isinstance(proof.get('parameters_sha256'), str) or
            len(proof['parameters_sha256']) != 64):
        raise ValueError('Missing actual diagnostic CPU checkpoint restore verification')


def read_gates(args):
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    evidence = dict(decision=read(args.decision), status=read(args.diagnostic_run / 'status.json'),
                    supervisor=read(args.diagnostic_run / 'supervisor.json'))
    verify_launch_gate(**evidence)
    decision = evidence['decision']
    if decision.get('execution_contract_sha256') != base.file_hash(args.execution_contract):
        raise ValueError('Decision execution-contract hash differs')
    hashes = decision.get('evidence_sha256', {})
    if not {'status.json', 'supervisor.json', 'identity.json', 'selected-rows.json', 'before.npz', 'after.npz'} <= set(hashes):
        raise ValueError('Decision lacks complete bound diagnostic evidence')
    for name, digest in hashes.items():
        path = (args.diagnostic_run / name).resolve()
        if not path.is_relative_to(args.diagnostic_run.resolve()) or base.file_hash(path) != digest:
            raise ValueError('Diagnostic evidence changed since verification: ' + name)
    return dict(decision_sha256=base.file_hash(args.decision),
                execution_contract_sha256=base.file_hash(args.execution_contract), diagnostic_evidence=hashes)


def preflight(args):
    from audit_s1_teacher_actions import audit_provenance
    from bvi.ia_fetch_data import invocation_rows
    manifest, provenance = audit_provenance(args.source, args.dataset_manifest, args.normalizer)
    rows = invocation_rows(manifest, 10)
    train = [r for r in rows if r['split'] == 'train']
    if not train or {r['parent_id'] for r in train} & {20, 21}:
        raise ValueError('Invalid frozen train split')
    dev = select_dev(json.loads(args.audit_panel.read_text(encoding='utf-8')), rows)
    checkpoint = args.checkpoint.resolve()
    if checkpoint.name != '855' or checkpoint.parent.name != 'best' or not (checkpoint / 'params').is_dir():
        raise ValueError('Candidate must initialize original best/855, never diagnostic output')
    assets = checkpoint / 'assets' / base.REPO_ID
    contract = json.loads((assets / 'bvi-state-contract.json').read_text())
    if (contract.get('training_stage') != 'S1_IA_single_bank_no_progress' or
            contract.get('normalizer_sha256') != provenance['normalizer_sha256'] or
            base.file_hash(assets / 'norm_stats.json') != provenance['normalizer_sha256']):
        raise ValueError('Checkpoint stage/normalizer differs')
    return train, dev, provenance, contract


def worker(args):
    output, deadline = args.output, args.deadline
    report = dict(schema=SCHEMA, candidate_protocol=PROTOCOL, diagnostic_only=False,
                  status='preflight', additional_updates=0, source_checkpoint_step=855,
                  source_checkpoint_parameters_sha256=SOURCE_HASH, best=None, native_success=None,
                  optimizer='reset_AdamW', learning_rate=1e-4, progress=False, family_bank=False, api_calls=0)
    def status(**changes):
        report.update(changes); report['remaining_seconds'] = max(0., deadline - time.monotonic())
        base.write_json(output / 'status.json', report)
    datasets = []; manager = None
    try:
        status()
        gates = read_gates(args)
        train_rows, dev_rows, provenance, contract = preflight(args)
        diagnostic_identity = json.loads((args.diagnostic_run / 'identity.json').read_text())
        if any(diagnostic_identity.get(k) != provenance[k] for k in provenance):
            raise ValueError('Candidate data/normalizer differs from verified diagnostic source')
        if not base.budget_remaining(deadline, 900):
            raise TimeoutError('Preflight consumed training budget')
        actual = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid,memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if actual[0].strip() != base.GPU1_UUID or int(actual[1]) >= 1024:
            raise RuntimeError('Physical GPU1 identity/idle gate failed')
        os.environ.update(CUDA_VISIBLE_DEVICES=base.GPU1_UUID, JAX_PLATFORMS='cuda',
                          XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2')
        import numpy as np
        import jax
        import flax.nnx as nnx
        from flax import traverse_util
        import openpi
        from openpi import transforms
        from openpi.models import model as models
        from openpi.training import checkpoints, sharding
        from fetch_native_s1_config import config
        from fetch_ia_s1_dataset import InvocationDataset
        import ia_s1_steps
        author = Path(openpi.__file__).resolve().parents[2]
        commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if commit != 'f4eb160ba52b22c1e85fe432de59c24bbbac6187' or dirty.strip():
            raise ValueError('Expected clean pinned OpenPI author checkout')
        sys.path.insert(0, str(author))
        cfg = config(base.REPO_ID, str(output), str(args.checkpoint / 'params'), args.normalizer, steps=500, batch=1)
        cfg = dataclasses.replace(cfg, seed=7, exp_name='bounded-candidate', lr_schedule=base.ConstantSchedule(),
              freeze_filter=base.lora_only_freeze_filter(), policy_metadata=dict(cfg.policy_metadata,
              training_stage='S1_IA_single_bank_no_progress', invocation_aligned=True, candidate_schema=SCHEMA,
              candidate_protocol=PROTOCOL, diagnostic_only=False,
              source_checkpoint_step=855, optimizer_reset=True, lr_clock='reset_constant_1e-4',
              trainable_scope='explicit_lora_only_narrower_than_legacy_default', parameter_dtypes='preserved'))
        for name in ('robot', 'state_dim', 'state_components', 'action_dim', 'base_position_reference',
                     'base_camera', 'wrist_camera', 'state_conditioning', 'training_repo',
                     'action_convention', 'state_source', 'normalizer_sha256'):
            if contract.get(name) != cfg.policy_metadata[name]:
                raise ValueError('Input contract changed: ' + name)
        if cfg.model.enable_progress_head or cfg.progress_loss_weight != 0 or cfg.ema_decay is not None:
            raise ValueError('Progress/EMA must be off')
        dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        for split, roster in (('train', train_rows), ('validation', dev_rows)):
            dataset = InvocationDataset(args.source, args.dataset_manifest, args.normalizer, dc, cfg.model, split)
            dataset.rows = roster; datasets.append(dataset)
        actions = np.asarray([datasets[0].handle[r['trajectory']]['actions'][r['observation_index']] for r in train_rows])
        weights, indices = fixed_sampling(train_rows, actions)
        np.savez_compressed(output / 'sampling.npz', weights=weights, indices=indices, current_actions=actions)
        base.write_json(output / 'train-roster.json', train_rows)
        base.write_json(output / 'dev-roster.json', dev_rows)
        base.write_json(output / 'sampling-summary.json', dict(seed=SEED, draws=4000, replacement=True,
            rules={'reach_yaw_or_torso_abs_ge_0.1': 3, 'move_any_active11_abs_ge_0.1': 2, 'otherwise': 1},
            eligible_parent_counts=dict(Counter(str(r['parent_id']) for r in train_rows)),
            sampled_parent_counts=dict(Counter(str(train_rows[i]['parent_id']) for i in indices)),
            sampled_family_counts=dict(Counter(train_rows[i]['tool_family'] for i in indices))))
        base.write_json(output / 'identity.json', dict(schema=SCHEMA, **provenance, gates=gates,
            source_checkpoint_parameters_sha256=SOURCE_HASH, checkpoint=str(args.checkpoint.resolve()),
            audit_panel_sha256=base.file_hash(args.audit_panel), sampling_sha256=base.file_hash(output / 'sampling.npz'),
            trainer_sha256=base.file_hash(__file__), helper_sha256=base.file_hash(base.__file__),
            max_updates=500, max_seconds=9000, finalize_reserve=900, accumulation=8,
            checkpoints=[100, 250, 500, 'budget_stop'], rng_seed=SEED, denoising_steps=10,
            selection='dev8_relative_rmse_mean_with_reach_yaw_torso_guard_tie_earlier', policy_metadata=cfg.policy_metadata))
        status(status='loading')
        ref = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(7))).to_pure_dict())
        loaded = models.restore_params(args.checkpoint / 'params', restore_type=np.ndarray)
        flat = traverse_util.flatten_dict(loaded)
        if set(ref) != set(flat) or any(ref[k].shape != flat[k].shape for k in ref):
            raise ValueError('Full source checkpoint leaf mismatch')
        digest = hashlib.sha256()
        for key, value in sorted(flat.items()):
            array = np.asarray(value); digest.update('/'.join(map(str, key)).encode())
            digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
        if digest.hexdigest() != SOURCE_HASH:
            raise ValueError('Actual checkpoint parameters are not frozen best855')
        state = base.initialize_preserving_dtype(cfg, loaded)
        restored = traverse_util.flatten_dict(state.params.to_pure_dict())
        if int(state.step) != 0 or any(restored[k].dtype != flat[k].dtype for k in flat):
            raise ValueError('Initialization changed dtype or failed optimizer step reset')
        trainable = traverse_util.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict())
        if len(trainable) != 20 or any('lora' not in '/'.join(map(str, k)).lower() for k in trainable):
            raise ValueError('Expected exactly 20 shared LoRA trainable leaves')
        del loaded, flat, ref, restored, array, value
        mesh = sharding.make_mesh(cfg.fsdp_devices)
        ds_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        def batch(split, index):
            item = datasets[split][int(index)]; valid = ~item.pop('actions_is_pad')
            x = jax.tree.map(lambda v: np.asarray(v)[None], item)
            return jax.device_put((models.Observation.from_dict(x), x['actions'], valid[None]), ds_sharding)
        @jax.jit
        def predict(params, key, obs):
            model = nnx.merge(state.model_def, params); model.eval()
            return model.sample_actions(key, obs, num_steps=10)
        unnormalize = transforms.Unnormalize({'actions': dc.norm_stats['actions']}, use_quantiles=True)
        def evaluate(step):
            predicted, targets, masks, raw_predictions = [], [], [], []
            for i in range(8):
                if not base.budget_remaining(deadline, 30):
                    raise TimeoutError('Budget exhausted during dev evaluation')
                obs, target, valid = batch(1, i)
                with sharding.set_mesh(mesh):
                    out = predict(state.params, jax.random.fold_in(jax.random.key(SEED), i), obs)
                raw = unnormalize({'actions': np.asarray(out)[0]})['actions'][:, :13]
                target = unnormalize({'actions': np.asarray(target)[0]})['actions'][:, :13]
                raw_predictions.append(raw); predicted.append(np.clip(raw, -1, 1))
                targets.append(np.clip(target, -1, 1)); masks.append(np.asarray(valid)[0])
            p, t, m = np.asarray(predicted), np.asarray(targets), np.asarray(masks)
            p[:, :, 8:10] = 0; t[:, :, 8:10] = 0
            families = [r['tool_family'] for r in dev_rows]
            np.savez_compressed(output / f'dev-step{step:03d}.npz', predicted=p, target=t, valid=m,
                                predicted_unclipped=np.asarray(raw_predictions), families=np.asarray(families),
                                row_keys=np.asarray([row_key(r) for r in dev_rows]))
            metrics = base.reconstruction_metrics(p, t, m, families)
            base.write_json(output / f'dev-step{step:03d}.json', metrics)
            return metrics
        status(status='baseline_dev')
        baseline = evaluate(0)
        score_dev(baseline, baseline)  # Reject zero/nonfinite reference denominators.
        best = None; evaluated = set()
        # Retain earlier complete winners if a later write is interrupted. There
        # are at most four selection points; never delete an archived winner.
        manager, _ = checkpoints.initialize_checkpoint_dir(output / 'best', keep_period=1, overwrite=False, resume=False)
        base.write_json(output / 'best.json', dict(schema=SCHEMA, eligible=False, checkpoint=None,
                         diagnostic_only=False, candidate_protocol=PROTOCOL,
                         selection='dev8_only_no_online_selection'))
        class Assets:
            def data_config(self):
                return dc
        def assess():
            nonlocal best
            step = int(state.step)
            if step in evaluated or step == 0:
                return
            status(status='dev_selection')
            result = dict(step=step, **score_dev(evaluate(step), baseline))
            evaluated.add(step)
            with (output / 'selection.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result) + '\n')
            if prefer(result, best):
                if not base.budget_remaining(deadline, 30):
                    raise TimeoutError('No time left for eligible checkpoint save')
                status(status='saving_eligible')
                checkpoints.save_state(manager, state, Assets(), step); manager.wait_until_finished()
                path = output / 'best' / str(step)
                base.write_json(path / 'assets' / base.REPO_ID / 'bvi-state-contract.json', cfg.policy_metadata)
                if (step not in manager.all_steps() or not (path / 'params').is_dir() or
                        not (path / 'train_state').is_dir() or
                        base.file_hash(path / 'assets' / base.REPO_ID / 'norm_stats.json') != provenance['normalizer_sha256'] or
                        not base.budget_remaining(deadline)):
                    raise TimeoutError('Eligible checkpoint not complete within deadline')
                # Hash source arrays of this completed save for the independent
                # fresh-process restore check. This transfer also counts in budget.
                saved_digest = hashlib.sha256()
                for key, value in sorted(traverse_util.flatten_dict(state.params.to_pure_dict()).items()):
                    array = np.asarray(value)
                    saved_digest.update('/'.join(map(str, key)).encode())
                    saved_digest.update(str(array.shape).encode()); saved_digest.update(str(array.dtype).encode())
                    saved_digest.update(array.tobytes())
                completed = best_record(output, path, result, saved_digest.hexdigest())
                if not base.budget_remaining(deadline):
                    raise TimeoutError('Selection evidence finalization exceeded deadline')
                base.write_json(output / 'best.json', completed)
                best = completed
            status(best=best)
        grad = jax.jit(functools.partial(ia_s1_steps.micro_gradient, cfg))
        apply = jax.jit(functools.partial(ia_s1_steps.apply_average, cfg))
        for update in range(500):
            status(status='training')
            summed = None; losses = []
            for micro in range(8):
                if not base.budget_remaining(deadline, 900):
                    break
                exposure = update * 8 + micro
                with sharding.set_mesh(mesh):
                    loss, g = grad(state, jax.random.fold_in(jax.random.key(7001), exposure), batch(0, indices[exposure]))
                value = float(loss)
                if not np.isfinite(value):
                    raise ValueError('Nonfinite training loss')
                summed = g if summed is None else jax.tree.map(lambda x, y: x + y, summed, g)
                losses.append(value)
            if len(losses) != 8 or not base.budget_remaining(deadline, 900):
                break
            with sharding.set_mesh(mesh):
                state, norm = apply(state, summed, 8)
            if not np.isfinite(float(norm)):
                raise ValueError('Nonfinite gradient')
            step = int(state.step)
            status(additional_updates=step, loss=float(np.mean(losses)))
            with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(step=step, loss=float(np.mean(losses)), grad_norm=float(norm), lr=1e-4)) + '\n')
            if step in (100, 250, 500):
                assess()
        assess()  # Budget-stop point; does not reevaluate a scheduled checkpoint.
        status(status='completed_candidate' if best else 'completed_no_eligible_candidate', best=best,
               stop_reason='update_limit' if int(state.step) == 500 else 'finalization_reserve',
               checkpoint_selection_complete=True, evaluated_steps=sorted(evaluated))
    except Exception as exc:
        status(status='failed', error=repr(exc))
        raise
    finally:
        for dataset in datasets:
            dataset.close()
        if manager is not None:
            manager.wait_until_finished()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset-manifest', 'normalizer', 'checkpoint', 'audit-panel', 'output',
                 'decision', 'diagnostic-run', 'execution-contract'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main():
    started = time.monotonic(); args = parse_args()
    if args.preflight:
        train, dev, provenance, _ = preflight(args)
        print(json.dumps(dict(schema=SCHEMA, status='data_preflight_only_not_launch_authorization',
                              train_rows=len(train), dev_rows=len(dev), provenance=provenance)))
        return
    if args.worker:
        if args.deadline is None or os.environ.get('BVI_CANDIDATE_PARENT') != str(os.getppid()):
            raise ValueError('Worker requires hard-deadline supervisor')
        return worker(args)
    read_gates(args)  # Fail before GPU/worker creation; recheck inside worker.
    if sys.platform != 'linux':
        raise RuntimeError('Execution requires lab Linux supervisor')
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = started + 9000
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--deadline', str(deadline)]
    base.write_json(args.output / 'supervisor.json', dict(schema=SCHEMA, status='starting', budget_seconds=9000))
    with (args.output / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, BVI_CANDIDATE_PARENT=str(os.getpid())), start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True; os.killpg(process.pid, signal.SIGKILL); code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL); process.wait(); raise
    base.write_json(args.output / 'supervisor.json', dict(schema=SCHEMA, status='hard_timeout' if timed_out else 'exited',
                    worker_exit_code=code, elapsed_seconds=time.monotonic() - started, budget_seconds=9000))
    if code:
        raise SystemExit(code if code > 0 else 124)


if __name__ == '__main__':
    main()
