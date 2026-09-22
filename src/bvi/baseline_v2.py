"""CPU contracts for v2 recovery choices and fixed evaluation accounting.

STATUS: frozen — historical training and diagnostics (retained)

These validators never execute a policy or mutate a benchmark task pointer.
Runtime integration must advertise only physically implemented recovery tools.
"""
from dataclasses import dataclass
import math

from .protocol import ProtocolError


def validate_bound_panel(config: dict, condition: str, navigation: str) -> list[dict]:
    """Pre-execution C panel guard; never fills missing UIDs from a guessed seed.

    Binding evidence must come from real environment resets before this guard.
    Passing this check is not budget approval or proof a backend is available.
    """
    if config.get('status') != 'bound_ready_for_preflight':
        raise ProtocolError('Evaluation plan is not explicitly bound')
    if condition not in {'fixed','vlm_rule','vlm_learned'} or condition not in config.get('conditions',[]):
        raise ProtocolError('Unknown/unregistered C condition')
    if navigation not in {'ppo','lightnav'} or navigation not in config.get('navigation_panels',[]):
        raise ProtocolError('Unknown/unregistered navigation panel')
    if config.get('evaluation_split') != 'val' or config.get('task') != 'tidy_house':
        raise ProtocolError('Wrong C task/split')
    rows = config.get('evaluation_episodes')
    if not isinstance(rows,list) or len(rows)!=10:
        raise ProtocolError('Fixed ten-episode panel required')
    if any(type(row.get('seed')) is not int for row in rows) or [row['seed'] for row in rows]!=list(range(10)):
        raise ProtocolError('Seeds must be the ordered fixed panel 0..9')
    for row in rows:
        if not isinstance(row.get('plan_uid'),str) or not row['plan_uid'].strip():
            raise ProtocolError('Missing real plan UID')
    for field,expected in [('max_env_steps',7000),('max_vlm_calls',40),('api_usd_per_episode',0.05)]:
        actual=config.get(field)
        if type(actual) is not type(expected) or actual!=expected:
            raise ProtocolError(f'Changed panel limit: {field}')
    if condition=='vlm_learned':
        # Separate PPO/LightNav distributions require separately recorded gates.
        gate=config.get('learned_progress_gates',{}).get(navigation,{})
        if gate.get('status')!='passed' or gate.get('feedback_source')!='learned_g_z_conditioned':
            raise ProtocolError('Learned progress not admitted for this navigation distribution')
    return [dict(row) for row in rows]


@dataclass(frozen=True)
class RecoveryContext:
    current_skill: str
    target_id: str
    last_skill: str | None = None
    last_target_id: str | None = None
    last_status: str | None = None
    reposition_available: bool = False
    terminal: bool = False


def validate_choice(body: dict, context: RecoveryContext) -> dict:
    """Validate explicit (g,z); preserve text, never silently replace a choice."""
    if context.terminal:
        raise ProtocolError('Episode is terminal')
    if not isinstance(body, dict) or set(body) != {'g', 'z', 'target_id', 'max_steps'}:
        raise ProtocolError('Expected exactly g,z,target_id,max_steps')
    g, z, steps = body['g'], body['z'], body['max_steps']
    if not isinstance(g, str) or not isinstance(z, str) or not z.strip() or len(z.encode('utf-8')) > 640:
        raise ProtocolError('A grounded instruction of at most640 UTF8 bytes is required')
    if type(steps) is not int or not 1 <= steps <= 7000:
        raise ProtocolError('Invalid action budget')
    if g == 'abort':
        if body['target_id'] != 'episode' or steps != 1:
            raise ProtocolError('Abort must target episode with a unit sentinel budget')
    else:
        if body['target_id'] != context.target_id:
            raise ProtocolError('Target is incompatible with current native subtask')
        if g == 'retry':
            if (context.last_skill != context.current_skill or context.last_target_id != context.target_id
                    or context.last_status not in {'failed', 'timeout', 'interrupted'}):
                raise ProtocolError('No compatible unsuccessful invocation to retry')
        elif g == 'reposition':
            if not context.reposition_available or steps > 40:
                raise ProtocolError('Reposition executor unavailable or budget exceeds40')
        elif g == 'observe':
            if steps != 1:
                raise ProtocolError('Observe refreshes once without a physical action')
        elif g not in {'navigate', 'pick', 'place'} or g != context.current_skill:
            raise ProtocolError('Skill is incompatible with current native subtask')
    return dict(body)


def summarize_panel(planned_seeds: list[int], episodes: list[dict]) -> dict:
    """Keep planned N and infra failures; unfinished entries are never failures/successes."""
    if not planned_seeds or len(set(planned_seeds)) != len(planned_seeds):
        raise ValueError('Unique planned seeds required')
    seen, valid, infra = set(), [], 0
    for row in episodes:
        seed = row['seed']
        if seed not in planned_seeds or seed in seen:
            raise ValueError('Unexpected or duplicate episode')
        seen.add(seed)
        if row['status'] == 'infrastructure_failure':
            infra += 1
            continue
        if row['status'] != 'completed':
            raise ValueError('Only finalized records belong in results')
        n = row['completed_objects']
        if type(n) is not int or not 0 <= n <= 5 or type(row['native_task_success']) is not bool:
            raise ValueError('Invalid native outcome')
        if row['native_task_success'] and n != 5:
            raise ValueError('Native task success contradicts object count')
        for key in ['steps', 'vlm_calls', 'invalid_requests', 'recovery_attempts', 'recovery_successes']:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f'Invalid count {key}')
        if row['recovery_successes'] > row['recovery_attempts']:
            raise ValueError('Recovery success exceeds attempts')
        if row['steps'] > 7000 or row['vlm_calls'] > 40:
            raise ValueError('Episode exceeds preregistered limits')
        cost = row['api_cost_usd']
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or not 0 <= cost <= .05:
            raise ValueError('Missing, unknown or over-budget API consumption')
        valid.append(row)
    successes = sum(r['native_task_success'] for r in valid)
    return dict(planned_n=len(planned_seeds), attempted_n=len(episodes), completed_n=len(valid),
        infrastructure_failures=infra, not_run_seeds=[s for s in planned_seeds if s not in seen],
        native_task_successes=successes, task_sr_planned=successes / len(planned_seeds),
        completed_object_counts=[r['completed_objects'] for r in valid],
        partial=len(seen) != len(planned_seeds),
        completed_episode_api_cost_usd=sum(r['api_cost_usd'] for r in valid),
        cost_scope='completed episodes only; infrastructure attempts remain in call ledger')
