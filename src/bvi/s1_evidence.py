"""Build native S1 admission from on-disk episode evidence, not claimed scores."""
import hashlib
import json
from pathlib import Path

from .s1_capability_gate import SEEDS, validate_s1_capability


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(panel, best_record):
    panel = Path(panel)
    best = json.loads(Path(best_record).read_text())
    if best.get('selection') != 'heldout_action_loss_only':
        raise ValueError('Checkpoint must be selected by heldout action loss')
    episodes, identity = [], None
    for seed in SEEDS:
        path = panel / f'seed{seed}' / 'result.json'
        result = json.loads(path.read_text())
        for key, expected in dict(seed=seed, status='episode_completed',
                                  task='set_table/pick/013_apple', scene_split='val',
                                  max_actions=200, control_frequency=20,
                                  ee_rest_threshold_m=0.05, api_calls=0,
                                  instruction='Pick and stably hold the apple.').items():
            if type(result.get(key)) is not type(expected) or result[key] != expected:
                raise ValueError(f'Incomplete or changed episode {seed}: {key}')
        metadata = result['model_metadata']
        contract = metadata['state_contract']
        for key, value in dict(state_dim=24, state_source='env_native_agent',
                               base_camera='fetch_head', wrist_camera='fetch_hand',
                               training_stage='S1_ordinary_target_domain_SFT_not_TAPT').items():
            if contract.get(key) != value:
                raise ValueError(f'Wrong model contract: {key}')
        if metadata['checkpoint'] != best['checkpoint']:
            raise ValueError('Episode did not use heldout-selected checkpoint')
        current = (metadata['pretrained_parameters_sha256'], metadata['normalizer_sha256'])
        if identity is not None and identity != current:
            raise ValueError('Mixed models/normalizers in native panel')
        identity = current
        artifacts = result['artifact_sha256']
        if not {'events.jsonl', 'initial-state.pt', 'reset-observation.npz'} <= artifacts.keys():
            raise ValueError('Missing native evidence')
        for name, digest in artifacts.items():
            if Path(name).name != name or sha(path.parent / name) != digest:
                raise ValueError(f'Episode artifact hash mismatch: {name}')
        events = [json.loads(line) for line in (path.parent/'events.jsonl').read_text().splitlines()]
        if len(events) != result['steps'] or not events:
            raise ValueError('Action log length differs from report')
        native_success = events[-1]['info']['success']
        while isinstance(native_success, list) and len(native_success) == 1:
            native_success = native_success[0]
        if type(native_success) is not bool or native_success != result['success']:
            raise ValueError('Success differs from native event')
        episodes.append(dict(seed=seed, status='completed', success=result['success'],
                             executed_actions=result['steps'], evidence_sha256=sha(path)))
    report = dict(schema='bvi.s1-native-capability/1', stage='S1', arm='faithful_native24',
                  task='set_table/pick/013_apple', scene_split='val', action_budget=200,
                  native_success_threshold_metres=0.05, checkpoint_sha256=identity[0],
                  normalizer_sha256=identity[1], state_dim=24, privileged_policy_inputs=False,
                  state_source='env_native_agent', base_camera='fetch_head', wrist_camera='fetch_hand',
                  success_source='native_environment_predicate', checkpoint_selection='heldout_loss',
                  episodes=episodes, best_record_sha256=sha(best_record))
    try:
        report['admission'] = validate_s1_capability(report, identity[0])
    except ValueError as error:
        report['admission'] = dict(eligible=False, reason=str(error))
    return report
