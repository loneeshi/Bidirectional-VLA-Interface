import json, pathlib, numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

root = pathlib.Path("/workspace/tapt/evidence")
suite = benchmark.get_benchmark_dict()["libero_10"]()
idx = next(
    i
    for i in range(suite.n_tasks)
    if "cream_cheese_box_and_the_butter" in suite.get_task(i).name
)
task = suite.get_task(idx)
env = OffScreenRenderEnv(
    bddl_file_name=str(
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    ),
    camera_heights=256,
    camera_widths=256,
)
env.seed(7)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(idx)[0])
for _ in range(20):
    obs, _, done, info = env.step([0.0] * 6 + [-1.0])
for name in ("agentview_image", "robot0_eye_in_hand_image"):
    imageio.imwrite(root / (name + ".png"), obs[name])
(root / "smoke.json").write_text(
    json.dumps(
        dict(
            task_id=idx,
            task=task.name,
            instruction=task.language,
            shapes={k: list(v.shape) for k, v in obs.items() if hasattr(v, "shape")},
            done=bool(done),
            action_dim=env.env.action_dim,
        ),
        indent=2,
    )
)
env.close()
print("SMOKE PASSED", task.name)
