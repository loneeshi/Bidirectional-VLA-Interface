"""New S1 -> S2 admission contract; never authorizes the historical V8 pipeline.

STATUS: frozen — historical training and diagnostics (retained)

The evaluator must produce this report from archived native episode evidence.
The launcher must separately verify the supplied checkpoint digest against disk.
"""
import re

SEEDS = (2024, 2025, 2026, 2027, 2028, 2030, 2031, 2032, 2033, 2034)


def validate_s1_capability(report, checkpoint_sha256):
    """Fail closed on partial panels, changed protocol, or diagnostic inputs."""
    if not isinstance(report, dict) or report.get('schema') != 'bvi.s1-native-capability/1':
        raise ValueError('A new S1 native capability report is required')
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256):
        raise ValueError('Verified checkpoint SHA256 required')
    required = dict(stage='S1', arm='faithful_native24', task='set_table/pick/013_apple',
                    scene_split='val', action_budget=200, native_success_threshold_metres=0.05,
                    checkpoint_sha256=checkpoint_sha256, state_dim=24,
                    privileged_policy_inputs=False, state_source='env_native_agent',
                    base_camera='fetch_head', wrist_camera='fetch_hand',
                    success_source='native_environment_predicate', checkpoint_selection='heldout_loss')
    for key, expected in required.items():
        actual = report.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'Wrong S1 admission contract: {key}')
    episodes = report.get('episodes')
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise ValueError('All ten fixed episodes, including failures, are required')
    seeds, successes = set(), 0
    for episode in episodes:
        seed = episode.get('seed')
        if type(seed) is not int or seed not in SEEDS or seed in seeds:
            raise ValueError('Wrong/duplicate native evaluation seed')
        seeds.add(seed)
        if episode.get('status') != 'completed' or type(episode.get('success')) is not bool:
            raise ValueError('Incomplete or infrastructure-failed episode')
        steps = episode.get('executed_actions')
        if type(steps) is not int or not 1 <= steps <= 200:
            raise ValueError('Invalid action count')
        digest = episode.get('evidence_sha256')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Archived episode evidence hash required')
        successes += int(episode['success'])
    if successes < 3:
        raise ValueError(f'S1 native capability below 3/10: {successes}/10')
    return dict(eligible=True, successes=successes, planned_episodes=10,
                checkpoint_sha256=checkpoint_sha256, scope='new_S2_only',
                budget_increase=False, historical_pipeline_resume=False)
