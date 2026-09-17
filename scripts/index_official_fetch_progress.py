"""CPU-only current-observation supervision index; does not train or read images."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bvi.official_fetch_data import current_progress_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=2)
    args = parser.parse_args()
    rows, sources = [], []
    seen = set()
    for path in args.manifests:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        digest = manifest['source_sha256']
        if digest in seen:
            raise ValueError('Duplicate source manifest')
        seen.add(digest)
        rows.extend(current_progress_rows(manifest, args.horizon))
        sources.append(dict(path=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            source_sha256=digest))
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'progress-index.jsonl'
    data.write_text(''.join(json.dumps(row, ensure_ascii=True)+'\n' for row in rows), encoding='utf-8')
    report = dict(status='supervision_index_only', contract='current_observation_v2',
                  horizon=args.horizon, sources=sources, rows=len(rows),
                  row_counts=dict(Counter(f"{r['split']}/{r['tool_family']}" for r in rows)),
                  endpoint_rows=sum(not any(r['action_valid']) for r in rows),
                  index_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  training_updates=0, image_state_cache_validated=False,
                  wrong_handoff_validation_complete=False)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    main()
