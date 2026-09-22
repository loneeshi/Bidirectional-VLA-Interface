"""Read native pointer transitions; preserve missing traces and episode clustering."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize_events(path):
    raw = Path(path).read_bytes()
    steps = [row for line in raw.splitlines() if (row := json.loads(line)).get('event') in ('mshab_step','official_paper_teleport')]
    if not steps:
        return dict(validity='missing_native_steps', completed_objects=None, subtasks=None)
    pointer = 0
    completed, reached, teleported = set(), set(), set()
    for row in steps:
        if row['event'] == 'official_paper_teleport':
            index = row['subtask_index']
            if index != pointer or index % 4 not in (0,2):
                raise ValueError('Invalid teleport pointer transition')
            teleported.add(index)
            completed.add(index)
            reached.add(index)
            pointer += 1
            continue
        before, after = row['subtask_before'], row['subtask_after']
        if type(before) is not int or type(after) is not int or before != pointer or not 0 <= before <= after <= 20:
            raise ValueError('Discontinuous native trace or unsupported plan')
        if before < 20:
            reached.add(before)
        completed.update(range(before, after))
        pointer = after
    types = ['navigate','pick','navigate','place'] * 5
    counts = {kind: dict(planned=types.count(kind), reached=sum(types[i] == kind for i in reached),
                        completed=sum(types[i] == kind for i in completed))
              for kind in ('navigate','pick','place')}
    return dict(validity='native_trace', sha256=hashlib.sha256(raw).hexdigest(),
                completed_objects=counts['place']['completed'], subtasks=counts,
                navigation_bypassed_by_teleport=len(teleported))


def audit_panel(panel, trace_root):
    state = json.loads(Path(panel).read_text())
    rows = []
    for episode in state['episodes']:
        attempts = episode.get('attempts', [])
        attempt = attempts[-1] if attempts else None
        relative = (Path(episode.get('arm', '')) / f"seed-{episode['seed']:03d}" /
                    Path(attempt['directory']).name / 'events.jsonl') if attempt else None
        path = Path(trace_root)/relative if relative else None
        result = summarize_events(path) if path and path.exists() else dict(
            validity='missing_trace', completed_objects=None, subtasks=None)
        expected = episode.get('result',{}).get('completed_objects')
        result.update(seed=episode['seed'], plan_uid=episode.get('plan_uid'),
            arm=episode.get('arm','teleport'), status=episode['status'],
            expected_objects=expected, matches_archive=(None if result['completed_objects'] is None
                                                       else result['completed_objects'] == expected))
        rows.append(result)
    return dict(planned=len(rows), independent_unit='episode', rows=rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--trace-root',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit_panel(a.panel,a.trace_root),indent=2))
