"""Bounded live Pick matrix after a recorded navigation prefix; no paid VLM.

Each case reconstructs the prefix from reset. Start-state equality is a gate,
not an assumption. Privileged contact diagnostics never enter policy inputs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from diagnose_grasp_replay import diagnostic


def check_start(reference, current, atol=1e-5):
    errors = {}
    for key in ('qpos', 'qvel', 'tcp_pose', 'object_pose'):
        a, b = np.asarray(reference[key]), np.asarray(current[key])
        if a.shape != b.shape or not np.isfinite(b).all():
            raise ValueError(f'Invalid start state: {key}')
        errors[key] = float(np.max(np.abs(a-b)))
    return errors, all(value <= atol for value in errors.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--url', default='ws://127.0.0.1:8051')
    p.add_argument('--max-steps', type=int, default=120)
    p.add_argument('--wall-seconds', type=int, default=900)
    args = p.parse_args()
    if not 1 <= args.max_steps <= 120 or not 1 <= args.wall_seconds <= 1200:
        p.error('Diagnostic limits exceeded')
    from bvi import JsonlLogger
    from bvi.protocol import SkillRequest, Requirement, SkillStatus
    from bvi.mshab_adapter import make_mshab_adapter, OfficialRLSkill, describe_target, jsonable
    from bvi.fetch_pi_skill import FetchPiSkill
    from bvi.organizer import GraspMonitor
    from bvi.vla_clients import OpenPiClient
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
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    reference = None
    results = []
    cases = [('template-stop', 'template', True), ('gpt-stop', 'gpt', True),
             ('template-observe', 'template', False), ('gpt-observe', 'gpt', False),
             ('sac-reference', 'sac', False)]
    for name, prompt_mode, interrupt in cases:
        if time.monotonic()-start >= args.wall_seconds:
            break
        folder = args.output/name
        folder.mkdir()
        logger = JsonlLogger(folder/'events.jsonl', name)
        cfg = EnvConfig(**meta['config'])
        cfg.record_video = True
        cfg.info_on_video = False
        adapter = client = None
        result = {'case': name, 'api_requests': 0, 'benchmark_result': False,
                  'prefix_replayed': True, 'teacher_reference': prompt_mode == 'sac',
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
                skill = FetchPiSkill('pick', adapter, client, max_predictions=args.max_steps, chunk_steps=1)
            monitor = GraspMonitor(skill)
            monitor.start(request, observation)
            logger.emit('diagnostic_config', prompt=prompt, interrupt=interrupt,
                        source_sha256=hashlib.sha256((args.source/'events.jsonl').read_bytes()).hexdigest())
            reason = 'step_limit'
            for _ in range(args.max_steps):
                if time.monotonic()-start >= args.wall_seconds:
                    reason = 'total_wall_limit'
                    break
                before = diagnostic(adapter.uenv, obj, jsonable)
                action = tuple(monitor.act(adapter.observe()))
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


if __name__ == '__main__':
    main()
