"""Bounded live Pick matrix after a recorded navigation prefix; no paid VLM.

Each case reconstructs the prefix from reset. Start-state equality is a gate,
not an assumption. Privileged contact diagnostics never enter policy inputs.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from diagnose_grasp_replay import diagnostic


def check_start(reference, current, atol=1e-5):
    errors = {}
    for key in ('qpos', 'qvel', 'tcp_pose', 'object_pose'):
        a, b = np.asarray(reference[key]), np.asarray(current[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'Invalid start state: {key}')
        errors[key] = float(np.max(np.abs(a-b)))
    return errors, all(value <= atol for value in errors.values())


def component_action(action, teacher, component, pick_step):
    """Explicit diagnostic intervention; never counted as pure VLA success."""
    windows = {'base': (30, (11, 12)), 'arm-torso': (30, (*range(7), 10)),
               'gripper': (40, (7,))}
    end, channels = windows[component]
    selected = channels if 1 <= pick_step <= end else ()
    mixed = list(action)
    for i in selected:
        mixed[i] = teacher[i]
    return tuple(mixed), list(selected)


def compare_state_tree(reference, current, atol=1e-6):
    """Keep every field; tolerate only bounded floating-point drift, not IDs/counters."""
    differences = []
    def walk(a, b, path):
        if type(a) is not type(b):
            differences.append(dict(path=path, kind='type', accepted=False)); return
        if isinstance(a, dict):
            if a.keys() != b.keys():
                differences.append(dict(path=path, kind='keys', accepted=False)); return
            for key in a: walk(a[key], b[key], f'{path}/{key}')
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(dict(path=path, kind='length', accepted=False)); return
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f'{path}/{i}')
        elif isinstance(a, float):
            delta = abs(a-b)
            if not np.isfinite(a) or not np.isfinite(b) or delta != 0:
                differences.append(dict(path=path, kind='float', reference=a, current=b,
                    abs_error=delta if np.isfinite(delta) else None,
                    accepted=bool(np.isfinite(a) and np.isfinite(b) and delta <= atol)))
        elif a != b:
            differences.append(dict(path=path, kind='value', reference=a, current=b, accepted=False))
    walk(reference, current, '')
    return differences, all(d['accepted'] for d in differences)


def controller_snapshot(controller, jsonable):
    # Pinned ManiSkill PDJointPos.get_state() returns {} when use_target=False.
    # Also audit the actual command targets and interpolation bookkeeping.
    data = {'public': jsonable(controller.get_state())}
    for name in ('_target_qpos', '_start_qpos', '_step', '_step_size'):
        if hasattr(controller, name): data[name] = jsonable(getattr(controller, name))
    if hasattr(controller, 'controllers'):
        data['components'] = {k: controller_snapshot(v, jsonable) for k,v in controller.controllers.items()}
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--url', default='ws://127.0.0.1:8051')
    p.add_argument('--max-steps', type=int, default=120)
    p.add_argument('--wall-seconds', type=int, default=900)
    p.add_argument('--noise-seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--component-probes', action='store_true', help='Add three labelled live SAC channel interventions for the first seed')
    p.add_argument('--only-case', help='Run one case in a fresh simulation process; share audited references on disk')
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        p.error('Start this process with PYTHONHASHSEED=0; Python set iteration affects MS-HAB scene construction')
    if not 1 <= args.max_steps <= 120 or not 1 <= args.wall_seconds <= 2400:
        p.error('Diagnostic limits exceeded')
    if not 1 <= len(args.noise_seeds) <= 3 or len(set(args.noise_seeds)) != len(args.noise_seeds) or any(not 0 <= s < 2**32 for s in args.noise_seeds):
        p.error('Provide one to three distinct uint32 noise seeds')
    from bvi import JsonlLogger
    from bvi.protocol import SkillRequest, Requirement, SkillStatus
    from bvi.mshab_adapter import make_mshab_adapter, OfficialRLSkill, describe_target, jsonable
    from bvi.fetch_pi_skill import FetchPiSkill
    from bvi.organizer import GraspMonitor
    from bvi.vla_clients import OpenPiClient
    from bvi.diagnostic_noise import PairedNoiseClient, check_paired_prediction
    from mshab.envs.make import EnvConfig
    import bvi.nav_camera_env  # register custom camera environments

    events = [json.loads(line) for line in (args.source/'events.jsonl').read_text().splitlines()]
    meta = json.loads((args.source/'run-metadata.json').read_text())
    prefix = []
    for event in events:
        if event['event'] == 'mshab_step':
            if event['subtask_before'] != 0:
                break
            prefix.append(event)
    original = next(e for e in events if e['event'] == 'fetch_pi_started')
    if len(prefix) != 330:
        raise ValueError('Expected audited MSHAB011 330-step prefix')
    args.output.mkdir(parents=True, exist_ok=bool(args.only_case))
    start = time.monotonic()
    reference_path = args.output/'reference-start.json'
    state_path = args.output/'reference-simulator-state.json'
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    state_reference = json.loads(state_path.read_text()) if state_path.exists() else None
    paired_reference = {}
    results = json.loads((args.output/'summary.json').read_text()) if args.only_case and (args.output/'summary.json').exists() else []
    previous_count = len(results)
    cases = [(f'{mode}-{label}-seed{seed}', mode, interrupt, seed, None)
             for seed in args.noise_seeds
             for mode, label, interrupt in [('template', 'stop', True), ('gpt', 'stop', True),
                                             ('template', 'observe', False), ('gpt', 'observe', False)]]
    cases += [('sac-reference', 'sac', False, None, None)]
    if args.component_probes:
        cases += [(f'template-teacher-{component}-seed{args.noise_seeds[0]}', 'template', False,
                   args.noise_seeds[0], component) for component in ('base', 'arm-torso', 'gripper')]
    if args.only_case:
        cases = [c for c in cases if c[0] == args.only_case]
        if len(cases) != 1: p.error('Unknown case name')
    for name, prompt_mode, interrupt, noise_seed, component in cases:
        if time.monotonic()-start >= args.wall_seconds:
            break
        folder = args.output/name
        folder.mkdir()
        if args.only_case and prompt_mode != 'sac' and not interrupt and component is None:
            paired_path = args.output/f'{prompt_mode}-stop-seed{noise_seed}'/'events.jsonl'
            paired_reference[(noise_seed,prompt_mode)] = [json.loads(line) for line in paired_path.read_text().splitlines()
                                                         if json.loads(line)['event']=='paired_inference']
        logger = JsonlLogger(folder/'events.jsonl', name)
        cfg = EnvConfig(**meta['config'])
        cfg.record_video = True
        cfg.info_on_video = False
        adapter = client = proxy = teacher = None
        result = {'case': name, 'api_requests': 0, 'benchmark_result': False,
                  'prefix_replayed': True, 'teacher_reference': prompt_mode == 'sac',
                  'teacher_assisted': component is not None, 'intervention': component,
                  'noise_seed': noise_seed, 'paired_prefix_steps_verified': 0,
                  'completed': False, 'steps': 0, 'ever_grasped': False}
        try:
            adapter = make_mshab_adapter(cfg, logger, folder, seed=meta['seed'])
            source_reset = next(e for e in events if e['event'] == 'mshab_reset')
            if adapter.original_plan.subtasks[0].uid != source_reset['task_plan']['subtasks'][0]['uid']:
                raise ValueError('Task plan differs')
            for row in prefix:
                if time.monotonic()-start >= args.wall_seconds:
                    raise TimeoutError('Total diagnostic deadline')
                if int(adapter.observe().metadata['subtask_index']) != row['subtask_before']:
                    raise ValueError('Prefix boundary drift')
                transition = adapter.step(tuple(row['controller_action'][0]))
                if transition.info['adapter_subtask_after'] != row['subtask_after']:
                    raise ValueError('Prefix boundary drift')
            obj = adapter.uenv.subtask_objs[1]
            measured = diagnostic(adapter.uenv, obj, jsonable)
            if reference is None:
                reference = measured
                (args.output/'reference-start.json').write_text(json.dumps(reference, indent=2))
            errors, equal = check_start(reference, measured)
            result.update(start_errors=errors, start_equivalent=equal)
            logger.emit('start_gate', measured=measured, errors=errors, equivalent=equal)
            if not equal:
                raise ValueError('Reconstructed starts differ; no paired causal comparison')
            observation = adapter.observe()
            controller = adapter.uenv.agent.controller
            state = {'simulator': jsonable(adapter.uenv.get_state_dict()),
                     'controller': controller_snapshot(controller, jsonable)}
            (folder/'start-simulator-state.json').write_text(json.dumps(state, indent=2))
            state_digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            if state_reference is None:
                state_reference = state
                (args.output/'reference-simulator-state.json').write_text(json.dumps(state, indent=2))
            differences, state_equal = compare_state_tree(state_reference, state)
            logger.emit('extended_start_gate', state_sha256=state_digest,
                        controller_state_available=state['controller'] is not None,
                        numeric_atol=1e-6, differences=differences, equivalent=state_equal)
            if not state_equal:
                raise ValueError('Serialized simulator/controller start state differs')
            target = describe_target(adapter.original_plan, 1)
            prompt = original['prompt'] if prompt_mode == 'gpt' else target.description
            request = SkillRequest(name, 'pick', target.id, observation.frame_id,
                (Requirement('benchmark-completion', 'benchmark_success'),),
                args.max_steps, args.wall_seconds, instruction=prompt,
                tool_family='pick', interface_version='mshab-tool-family/1')
            if prompt_mode == 'sac':
                skill = OfficialRLSkill('pick', adapter, args.checkpoint_root, 'rl_per_obj')
            else:
                client = OpenPiClient.connect(args.url, timeout=120)
                proxy = PairedNoiseClient(client, noise_seed, logger)
                skill = FetchPiSkill('pick', adapter, proxy, max_predictions=args.max_steps, chunk_steps=1)
            if component:
                teacher = OfficialRLSkill('pick', adapter, args.checkpoint_root, 'rl_per_obj')
                teacher.start(request, observation)
            monitor = GraspMonitor(skill)
            monitor.start(request, observation)
            logger.emit('diagnostic_config', prompt=prompt, interrupt=interrupt, noise_seed=noise_seed,
                        python_hash_seed=os.environ['PYTHONHASHSEED'],
                        intervention=component, teacher_assisted=component is not None,
                        source_sha256=hashlib.sha256((args.source/'events.jsonl').read_bytes()).hexdigest())
            reason = 'step_limit'
            for _ in range(args.max_steps):
                if time.monotonic()-start >= args.wall_seconds:
                    reason = 'total_wall_limit'
                    break
                before = diagnostic(adapter.uenv, obj, jsonable)
                action = tuple(monitor.act(adapter.observe()))
                pick_step = result['steps'] + 1
                if proxy is not None and component is None:
                    key = (noise_seed, prompt_mode)
                    if not interrupt and pick_step <= len(paired_reference.get(key, [])):
                        error = check_paired_prediction(paired_reference[key][pick_step-1], proxy.records[-1])
                        result['paired_prefix_steps_verified'] += 1
                        logger.emit('paired_prefix_gate', pick_step=pick_step, action_max_error=error)
                    elif not interrupt and key not in paired_reference:
                        raise ValueError('Missing matched interrupt reference')
                if component:
                    # Query at the actual learner state, not a recorded teacher state.
                    teacher_action = teacher.act(adapter.observe())
                    mixed, channels = component_action(action, teacher_action, component, pick_step)
                    logger.emit('teacher_channel_intervention', pick_step=pick_step, channels=channels,
                                learner_action=action, teacher_action=teacher_action, executed_action=mixed)
                    action = mixed
                transition = adapter.step(action)
                native = skill.feedback(request, transition)
                monitored = monitor.feedback(request, transition)
                after = diagnostic(adapter.uenv, obj, jsonable)
                result['steps'] += 1
                result['ever_grasped'] |= bool(after['grasp30'][0])
                logger.emit('contact_diagnostic', step=adapter.steps, action=action,
                            before=before, after=after, native=native, monitored=monitored)
                if native.status is not SkillStatus.EXECUTING or adapter.ended:
                    reason = native.reason
                    result['pick_success'] = native.status is SkillStatus.SUCCEEDED
                    break
                if interrupt and monitored.status is not SkillStatus.EXECUTING:
                    reason = monitored.reason
                    break
            result.update(completed=True, reason=reason)
            if proxy is not None and component is None and interrupt:
                paired_reference[(noise_seed, prompt_mode)] = list(proxy.records)
        except Exception as exc:
            result['error'] = repr(exc)
        finally:
            if client is not None:
                client.close()
            if adapter is not None:
                adapter.close()
            (folder/'summary.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            (args.output/'summary.json').write_text(json.dumps(results, indent=2))
        if not result['completed']:
            break  # stop on environment/interface gate failure
    print(json.dumps(results))
    if len(results)-previous_count != len(cases) or any(not r['completed'] for r in results):
        raise SystemExit(2)  # A failed diagnostic gate must not report pipeline success.


if __name__ == '__main__':
    main()
