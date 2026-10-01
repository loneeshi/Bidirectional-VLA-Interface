# Default export validation

User decision on 2026-10-01: automatically use the scene demo showing Astra issued paths and actual execution for future experiments.

The shared EEF finalizer now creates `astra-commanded-vs-executed-3d-demo.mp4`, sets `preferred_demo`, records hashes, and includes the link in delivery README. Fixed-base and mobile arm adapters delegate to it; visual and compact EEF batch entrypoints already call it directly. A missing dual-path demo makes physical delivery incomplete, even if the legacy video exists. No-motion attempts remain explicitly labelled. Original recordings and analytical trajectories remain preserved.

Removed duplicate rendering from the arm adapter. Future mobile freezes include the updated shared finalizer. Existing source-closure-based EEF packaging discovers the renderer through its static import. `scripts/AGENTS.md` records the default and acceptance rule for future work. No historical frozen archive was regenerated.

CPU validation command:

```powershell
$env:PYTHONPATH='src;scripts'
.venv/Scripts/python.exe -m pytest tests/test_astra_demo_default_delivery.py tests/test_astra_dual_path_demo.py tests/test_eef_commanded_trajectory.py tests/test_project_eef_trajectory_into_demo.py tests/test_eef_arm_runtime_cpu.py -q
```

Result: 20 passed in 3.64 seconds. The integration test encodes a real two-frame CPU scene video through the shared finalizer; the failure test confirms renderer errors cannot count as complete delivery. No API calls, simulator actions, or GPU experiment runs.

Mainline remains stage 2. This changes media delivery only, not success criteria or recorded experimental results. Next gate: package/import checks at the next experiment freeze, followed by that experiment's approved run.
