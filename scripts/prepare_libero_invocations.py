"""Convert reviewed gripper/motion boundaries into invocation windows.

Segmentation is a local proxy, not the authors' released DROID-split dataset.
The reviewed source task executes cream cheese before butter. Simulator-contact
truth is unavailable in this parquet export and is not claimed here.
"""

import argparse, json, pathlib
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    root = pathlib.Path(args.data)
    m = json.loads((root / "manifest.json").read_text())
    candidates = json.loads((root / "segments-candidates.json").read_text())
    windows = []
    for split in ("train", "validation"):
        for episode in m[split]:
            eid = episode["episode_index"]
            d = pd.read_parquet(episode["path"])
            s = np.stack(d.state)
            spans = [x for x in candidates if x["episode"] == eid]
            good = [
                x
                for x in spans
                if x["candidate_success_proxy"]
                and not (eid == 166 and x["start"] == 201)
            ]
            # e166:201 is a reviewed empty/unsustained attempt, despite proxy lift.
            if len(good) != 2:
                raise ValueError(
                    f"Episode {eid} requires manual review: {len(good)} grasps"
                )
            cursor = 0
            for n, g in enumerate(good):
                target = ("cream cheese box", "butter")[n]
                # Never include an earlier failed grasp inside the next reach window.
                failures = [
                    x
                    for x in spans
                    if cursor <= x["start"] < g["start"] and x not in good
                ]
                if failures:
                    cursor = failures[-1]["end"]
                opening = np.where(np.abs(s[g["end"] :, -2:]).mean(1) > 0.035)[0]
                if not len(opening):
                    raise ValueError(f"Missing release end in {eid}")
                release_end = min(g["end"] + int(opening[0]) + 2, len(d))
                bounds = [
                    ("reach", cursor, g["start"], f"reach the {target}"),
                    ("grasp", g["start"], g["grasp_end"], f"grasp the {target}"),
                    (
                        "move",
                        g["grasp_end"],
                        g["end"],
                        f"move the {target} into the basket",
                    ),
                    (
                        "release",
                        g["end"],
                        release_end,
                        f"release the {target} in the basket",
                    ),
                ]
                for family, start, end, text in bounds:
                    if end - start < 2:
                        raise ValueError(f"Degenerate window: {eid} {family}")
                    windows.append(
                        dict(
                            episode=eid,
                            split=split,
                            path=episode["path"],
                            family=family,
                            start=start,
                            end=end,
                            instruction=text,
                            target=target,
                            progress_source="reviewed_temporal_proxy",
                        )
                    )
                cursor = release_end
    train = {w["episode"] for w in windows if w["split"] == "train"}
    val = {w["episode"] for w in windows if w["split"] == "validation"}
    assert len(train) == 20 and len(val) == 5 and not train & val
    (root / "invocations.json").write_text(json.dumps(windows, indent=2))
    print("invocations", len(windows), flush=True)


if __name__ == "__main__":
    main()
