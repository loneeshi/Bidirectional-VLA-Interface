"""Summarize paired read-only progress diagnostics without task-success claims."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

THRESHOLDS = {'reach': .9, 'grasp': .6, 'move': .9, 'release': .6}


def summarize(rows):
    assert len(rows) == 33 and len({r['index'] for r in rows}) == 33
    names = list(rows[0]['paths'])
    assert len(names) == 9 and all(set(r['paths']) == set(names) for r in rows)
    for row in rows:
        for name, path in row['paths'].items():
            assert len(path['trace']) == (5 if name.startswith('inference_') else 1)
            assert path['progress'] == path['trace'][-1]['progress']
            assert all(0 <= p <= 1 for p in path['progress'])
            if path['timestep_override'] is not None:
                assert path['trace'][0]['timestep'] == [path['timestep_override']]
    def metrics(group, name):
        paths = [r['paths'][name] for r in group]
        negatives = [r for r in group if r['target'][0] < THRESHOLDS[r['window']['family']]]
        positives = [r for r in group if r['target'][0] >= THRESHOLDS[r['window']['family']]]
        early = [r for r in negatives if r['paths'][name]['progress'][0] >= THRESHOLDS[r['window']['family']]]
        missed = [r for r in positives if r['paths'][name]['progress'][0] < THRESHOLDS[r['window']['family']]]
        result = dict(examples=len(group), masked_progress_mse=mean(p['progress_mse'] for p in paths),
            first_token_mae=mean(abs(r['paths'][name]['progress'][0]-r['target'][0]) for r in group),
            first_token_mean_bias=mean(r['paths'][name]['progress'][0]-r['target'][0] for r in group),
            label_threshold_false_positives=len(early), label_threshold_negative_examples=len(negatives),
            label_threshold_false_negatives=len(missed), label_threshold_positive_examples=len(positives),
            false_positive_indices=[r['index'] for r in early])
        if 'action_loss' in paths[0]:
            result['mean_action_loss'] = mean(p['action_loss'] for p in paths)
        if 'active_action_mse_to_teacher' in paths[0]:
            result['mean_active_action_mse_to_teacher'] = mean(p['active_action_mse_to_teacher'] for p in paths)
        return result
    output = dict(examples=len(rows), family_examples=dict(Counter(r['window']['family'] for r in rows)),
        held_out_trajectories=len({r['window']['trajectory'] for r in rows}),
        held_out_call_segments=len({(r['window']['trajectory'], r['window']['family'],
                                    r['window']['start'], r['window']['end']) for r in rows}),
        inference_timesteps=sorted({tuple(x['timestep'][0] for x in r['paths']['inference_native']['trace']) for r in rows}),
        random_training_timesteps=sorted({r['paths']['train_random_gt']['trace'][0]['timestep'][0] for r in rows}),
        metrics={name: metrics(rows, name) for name in names},
        by_family={f: {name: metrics([r for r in rows if r['window']['family']==f], name) for name in names}
                   for f in THRESHOLDS},
        warnings=['Threshold errors are single-frame comparisons to elapsed-fraction targets, not physical completion errors or two-hit monitor events.',
                  '33 correlated observations from 11 held-out call segments; one diffusion seed, reset on every observation/path.',
                  'train_random_gt is a paired single-random-t probe, not the original validation RNG stream or an expectation over training noise.',
                  'Generated-action-input action loss uses generated actions as its target and is not teacher action error.',
                  'No intervention here measures recovery of the failed online trajectory.'])
    pairs = [('inference_native','inference_raw_mobility'),
             ('train_t200_gt','train_t200_gt_weighted_mobility'),
             ('train_t200_gt','train_t200_generated_actions'),
             ('train_t200_gt','inference_raw_mobility')]
    output['paired_first_progress_changes'] = {}
    for before, after in pairs:
        delta = [r['paths'][after]['progress'][0]-r['paths'][before]['progress'][0] for r in rows]
        output['paired_first_progress_changes'][f'{before} -> {after}'] = dict(
            mean_delta=mean(delta), mean_absolute_delta=mean(abs(x) for x in delta), max_absolute_delta=max(abs(x) for x in delta))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--training-report', type=Path)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.directory/'paired.jsonl').read_text().splitlines()]
    output = summarize(rows)
    if args.training_report:
        original = json.loads(args.training_report.read_text())
        expected = [w for w in original['gate_examples'] if w['split']=='validation']
        assert [r['window'] for r in rows] == expected
        output['identities_equal_original_validation'] = True
    labels = json.loads((args.directory/'labels.json').read_text())['windows']
    output['label_support'] = {}
    for split in ['train', 'validation']:
        output['label_support'][split] = {}
        for family in THRESHOLDS:
            ws = [w for w in labels if w['window']['split']==split and w['window']['family']==family]
            samples = [s for w in ws for s in w['samples']]
            output['label_support'][split][family] = dict(segments=len(ws), samples=len(samples),
                distance_min=min(s['tcp_object_distance'] for s in samples),
                distance_max=max(s['tcp_object_distance'] for s in samples),
                held_samples=sum(s['is_grasped'] for s in samples),
                final_action_distances=[w['samples'][-1]['tcp_object_distance'] for w in ws])
    physical = []
    for row in rows:
        if row['window']['family'] != 'reach' or row['paths']['inference_native']['progress'][0] < .9:
            continue
        w = next(w for w in labels if all(w['window'][k]==row['window'][k]
                    for k in ['trajectory','family','start','end']))
        sample = next(s for s in w['samples'] if s['t']==row['window']['sample_t'])
        physical.append(dict(index=row['index'], target=row['target'][0],
            prediction=row['paths']['inference_native']['progress'][0], **sample))
    output['reach_positive_predictions_current_state'] = physical
    (args.directory/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['examples','family_examples','inference_timesteps','random_training_timesteps','metrics','paired_first_progress_changes']},indent=2))
