"""Build a CPU-only paired manifest from completed calibration JSON evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from bvi.observation_progress_preflight import prepare_pairing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = ['split-manifest.json', 'result.json', 'archive-verification.json',
             'backup.json', 'selected-validation.json']
    inputs = [args.source.joinpath(name).read_bytes() for name in names]
    audit = json.loads(inputs[2])
    batch_id = json.loads(inputs[3])['experiment_id']
    archived = {item['path']: item for item in audit['files']}
    # Git may check out text with CRLF; source artifacts were LF on Linux.
    # Bind report, sample order and reference predictions to the verified archive.
    for index in (0, 1, 4):
        name, data = names[index], inputs[index]
        digest = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
        if archived.get(f'{batch_id}/{name}', {}).get('sha256') != digest:
            raise ValueError(f'{name} does not match archived LF source bytes')
    result = prepare_pairing(*(json.loads(data) for data in inputs))
    result['source_json_sha256'] = {
        name: hashlib.sha256(data).hexdigest() for name, data in zip(names, inputs)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No implicit overwrite of a previously reviewed pairing.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'samples': len(result['samples']),
                      'status': result['status']}))


if __name__ == '__main__':
    main()
