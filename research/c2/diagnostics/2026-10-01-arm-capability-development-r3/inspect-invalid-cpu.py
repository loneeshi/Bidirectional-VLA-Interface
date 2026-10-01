import json
from pathlib import Path
import sys
root=Path('/home/pshuai/bvi-research')
code=root/'deploy/arm-revision2-20261001-r3/code'
sys.path[:0]=[str(code/'src'),str(code/'scripts')]
from bvi.eef_arm_contract import validate_call
folder=root/'runs/arm-capability-development-revision2-20261001-r3'
rows=[]
for p in sorted(folder.glob('arm-dev-*/V/provider/turn-*/response-raw.json')):
    raw=json.loads(p.read_text())
    texts=[part['text'] for item in raw.get('output',[]) if item.get('type')=='message'
           for part in item.get('content',[]) if part.get('type')=='output_text']
    call={}
    try:
        call=json.loads(''.join(texts));validate_call(call)
        error=None
    except (ValueError,TypeError,KeyError) as exc:error=str(exc)
    rows.append({'case_id':p.parents[3].name,'turn':p.parent.name,'tool':call.get('tool'),'note':call.get('note'),
                 'validation_error':error})
Path('/tmp/bvi-arm-r3-invalid-validation.json').write_text(json.dumps(rows,indent=2))
from collections import Counter
print(json.dumps(dict(Counter(row['validation_error'] for row in rows if row['validation_error']))))
