"""CPU-only extraction of selected validation evidence, not checkpoint selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['family']}/{row['kind']}"].append(row)
    result = {}
    for name, items in groups.items():
        endpoint = [row for row in items if row['current_target'] == 1]
        start = [row for row in items if row['current_target'] == 0]
        result[name] = dict(
            count=len(items),
            current_mse=sum((row['current_prediction']-row['current_target'])**2 for row in items)/len(items),
            mean_start_prediction=sum(row['current_prediction'] for row in start)/len(start) if start else None,
            mean_endpoint_prediction=sum(row['current_prediction'] for row in endpoint)/len(endpoint) if endpoint else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    import torch
    report = json.loads((args.directory / 'result.json').read_text())
    if report['status'] != 'bounded_calibration_complete_not_online_evaluated':
        raise ValueError('Training did not complete its gates')
    checkpoint_path = args.directory / 'best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['updates'] != report['best_step']:
        raise ValueError('Selected checkpoint and final report disagree')
    selected = checkpoint['report']['validation']
    baseline = report['zero_update_control']
    result = dict(selected_step=checkpoint['updates'], final_updates=report['updates'],
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  baseline=baseline, selected=selected,
                  baseline_by_family=summarize(baseline['rows']),
                  selected_by_family=summarize(selected['rows']),
                  interpretation='Held-out loss calibration only; task success requires native online outcomes.',
                  architecture_caveat=report['head_caveat'])
    (args.directory / 'selected-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('selected_step','final_updates','checkpoint_sha256')}))


if __name__ == '__main__':
    main()
