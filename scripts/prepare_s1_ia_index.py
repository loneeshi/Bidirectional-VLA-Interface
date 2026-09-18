"""CPU-only IA action index from frozen S1 parent/windows manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bvi.ia_fetch_data import invocation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    source = json.loads(raw)
    rows = invocation_rows(source)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/'actions.jsonl'
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows), encoding='utf-8')
    summary = dict(stage='S1-IA', status='indexed_not_training_validated',
        source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=source['source_sha256'], parent_split=source['parent_split'],
        counts=dict(Counter(r['split']+'/'+r['tool_family'] for r in rows)),
        samples=len(rows), source_action_frames=sum(e['exported_steps'] for e in source['episodes']),
        endpoint_training_samples=0, training_updates=0, gpu_runs=0, api_calls=0,
        family_routing=False, progress_loss=False, runtime_validated=False)
    summary['unannotated_action_frames'] = summary['source_action_frames']-len(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
