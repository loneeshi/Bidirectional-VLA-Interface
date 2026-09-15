"""Plot real logs; full-position validation is distinct from midpoint monitoring."""

import argparse, json, pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
root = pathlib.Path(a.root)
rows = [
    r
    for r in map(json.loads, (root / "training/train.jsonl").read_text().splitlines())
    if r["step"] <= 200
]
rows += list(
    map(json.loads, (root / "training-v2/train.jsonl").read_text().splitlines())
)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
for family in ["reach", "grasp", "move", "release"]:
    r = [x for x in rows if x["family"] == family]
    v = np.array([x["loss"] for x in r])
    smooth = [np.median(v[max(0, i - 19) : i + 1]) for i in range(len(v))]
    ax[0].plot([x["step"] for x in r], smooth, label=family, lw=1.5)
ax[0].set(
    title="Train joint loss (20-update rolling median)",
    xlabel="Total optimizer updates",
    ylabel="Action loss + 0.1 progress MSE",
    yscale="log",
)
ax[0].legend(fontsize=8)
meta = [
    json.loads(p.read_text())
    for p in sorted((root / "training-v2").glob("step-*/metadata.json"))
]
ax[1].plot(
    [r["step"] for r in meta],
    [r["metrics"]["validation_joint_loss"] for r in meta],
    marker="o",
    label="Midpoint monitoring",
)
select = root / "training-v2/validation-selection.json"
if select.exists():
    s = json.loads(select.read_text())
    v = s["all_checkpoints"]
    x = [r["step"] for r in v]
    ax[1].plot(
        x,
        [r["validation_joint_loss"] for r in v],
        marker="s",
        label="Early/mid/late selection",
    )
    ax[2].plot(x, [r["progress_rmse"] for r in v], marker="o", label="Learned progress")
    ax[2].plot(x, [r["constant_half_rmse"] for r in v], ls="--", label="Constant 50%")
    ax[2].legend(fontsize=8)
else:
    ax[2].text(
        0.5,
        0.5,
        "Full-position validation pending",
        ha="center",
        va="center",
        transform=ax[2].transAxes,
    )
ax[1].set(title="Held-out joint loss", xlabel="Checkpoint step", ylabel="Joint loss")
ax[1].legend(fontsize=8)
ax[2].set(
    title="Held-out progress error",
    xlabel="Checkpoint step",
    ylabel="RMSE (progress range 0–1)",
)
for panel in ax:
    panel.grid(alpha=0.2)
fig.suptitle(
    "LIBERO invocation-aligned adaptation; not benchmark task success", fontsize=12
)
fig.savefig(a.output, dpi=180)
