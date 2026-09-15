import io, json, pathlib
import pandas as pd, numpy as np
from PIL import Image, ImageDraw

r = pathlib.Path("/workspace/tapt/data")
m = json.loads((r / "manifest.json").read_text())
rows = []
tiles = []
for split in ("train", "validation"):
    for e in m[split]:
        d = pd.read_parquet(e["path"])
        a = np.stack(d.actions)
        s = np.stack(d.state)
        closed = a[:, -1] > 0.5
        starts = np.where(closed & ~np.r_[False, closed[:-1]])[0]
        ends = np.where(closed & ~np.r_[closed[1:], False])[0] + 1
        for start, end in zip(starts, ends):
            width = np.abs(s[start:end, -2:]).mean(1)
            lift = s[start:end, 2] - s[start, 2]
            held = np.where((width > 0.012) & (width < 0.035) & (lift > 0.035))[0]
            accepted = bool(len(held))
            mid = int(start + held[0]) if accepted else int((start + end) // 2)
            rec = {
                "episode": e["episode_index"],
                "split": split,
                "start": int(start),
                "end": int(end),
                "grasp_end": mid,
                "candidate_success_proxy": accepted,
                "grasp_xyz": s[start, :3].tolist(),
            }
            rows.append(rec)
            im = (
                Image.open(io.BytesIO(d.iloc[mid]["image"]["bytes"]))
                .convert("RGB")
                .resize((160, 160))
            )
            ImageDraw.Draw(im).text(
                (3, 3), f"e{e['episode_index']} {start}:{end} {accepted}", fill="red"
            )
            tiles.append(im)
sheet = Image.new("RGB", (160 * 8, 180 * ((len(tiles) + 7) // 8)), "white")
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % 8) * 160, (i // 8) * 180))
sheet.save(r / "segmentation-contact.png")
(r / "segments-candidates.json").write_text(json.dumps(rows, indent=2))
print("counts", len(rows), sum(x["candidate_success_proxy"] for x in rows))
