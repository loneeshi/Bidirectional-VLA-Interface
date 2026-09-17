"""CPU-only attribution of archived native S1 episodes; preserves raw evidence."""
import argparse
import json
from pathlib import Path
from bvi.native_pick_audit import summarize_actions, validate_episode, validate_seed_roster

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in sorted(a.panel.glob('seed*/events.jsonl')):
        result=json.loads((path.parent/'result.json').read_text())
        events=[json.loads(line) for line in path.read_text().splitlines()]
        validate_episode(result, events)
        rows.append(dict(seed=result['seed'],native_success=result['success'],
                         ever_grasped=result['ever_grasped'],**summarize_actions(events)))
    validate_seed_roster(rows)
    with a.output.open('x',encoding='utf-8') as out:
        json.dump(dict(rows=rows,policy_modified=False,geometry_is_diagnostic_only=True),out,indent=2)
