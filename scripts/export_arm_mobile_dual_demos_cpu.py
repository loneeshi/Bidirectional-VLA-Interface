"""Export all five archived online attempts without simulator/API reruns."""
import json
from pathlib import Path
from render_astra_dual_path_demo import render,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/media/c2-pick-arm-dual-path-2026-10-01-mobile-r1'


def main():
    OUT.mkdir(exist_ok=True);receipts=[]
    for source in sorted((ROOT/'docs/media').glob('c2-pick-arm-2026-10-01-arm-dev-*-V-mobile-*')):
        result=json.loads((source/'rendering-result.json').read_text(encoding='utf-8'))
        case=result['case_id'];folder=OUT/case;folder.mkdir(exist_ok=True)
        output=folder/(case+'-astra-commanded-vs-executed-3d-demo.mp4')
        receipt_path=folder/'receipt.json'
        if output.exists():
            receipt=json.loads(receipt_path.read_text());assert receipt['video_sha256']==sha(output)
        else:receipt=render(source/'delivery/online-raw.mp4',source/'rendering-result.json',output,receipt_path)
        receipt.update(output=output.relative_to(ROOT/'docs/media').as_posix(),
            source_video=(source/'delivery/online-raw.mp4').relative_to(ROOT/'docs/media').as_posix(),
            source_result=(source/'rendering-result.json').relative_to(ROOT/'docs/media').as_posix())
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        receipts.append(receipt)
    assert len(receipts)==5
    (OUT/'export-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    lines=['# Astra goals and real execution in the same 3D scene',
        'CPU postprocessing of the original online recordings; no simulator or API rerun.',
        'Pink dashed: issued Astra TCP waypoint connectors. Cyan: measured TCP history. Orange: base metric commands.',
        'Waypoint connectors are not the controller interpolation or a physical prediction. X-ray overlay, without depth occlusion.',
        *[f"- [{r['case_id']} — {'success' if r['strict_pick_success'] else 'failure'}]({r['case_id']}/{Path(r['output']).name})" for r in receipts]]
    (OUT/'README.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([r['output'] for r in receipts]))


if __name__=='__main__':main()
