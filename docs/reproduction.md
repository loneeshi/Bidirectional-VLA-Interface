# Setup and reproduction

The simulator was tested on Linux with an NVIDIA RTX A6000. G0 passed with a 60-step Fetch `Empty-v1` rollout; G1 then passed with 200 zero-action steps in a real ReplicaCAD TidyHouse validation scene. Pretrained skill execution and the VLM loop have separate acceptance gates and must not be inferred from either smoke test.

## Tested versions

| Component | Observed version or revision |
|---|---|
| OS | Ubuntu 22.04.5 |
| GPU | RTX A6000, 48 GB |
| NVIDIA driver | 580.159.04 |
| Python | 3.11.10 |
| PyTorch | 2.4.1+cu124 |
| ManiSkill | 3.0.0b18 |
| ManiSkill commit | `17121e3f96e3ee3ed0c03610b17f8bc2864617af` |
| MS-HAB commit | `e9ff3d23496d38e4431c8d913e147ffa007f7f72` |
| SAPIEN | 3.0.0b1 |
| Optional local OpenAI SDK | 3.13.0; tested with six live `gpt-5.6-luna` image requests |

Use a dedicated Python environment. Preserve the complete resolved package list and both upstream commits for every experiment. Upstream branch names are moving references; the recorded revisions establish the tested baseline.

## Simulator installation

The official [MS-HAB installation guide](https://github.com/arth-shukla/mshab#setup-and-installation) selects the `mshab` branch of ManiSkill. Start from that compatible branch, then use the pinned commits above. The simulator requires working CUDA and NVIDIA Vulkan access; a successful `nvidia-smi` alone is insufficient.

For headless operation, expose graphics as well as compute capabilities to the container. `libEGL.so.1` was initially missing in the tested image and was supplied by Ubuntu's `libegl1` package. Record this system dependency in the environment build rather than relying on a manual change that disappears when a container restarts.

Place source checkouts, the Python environment, simulator caches, assets, and results on persistent storage when the host is ephemeral. Persisting source files does not by itself preserve system packages installed outside that storage.

Repository-specific installation and simulation entrypoints are listed below. G2–G4 have passed in a constrained single-scene diagnostic, described in [evaluation](evaluation.md#live-vlm-protocol-diagnostic). The presence of these scripts is not a claim of a fully reproduced MS-HAB evaluation.

## External assets and checkpoints

MS-HAB requires YCB objects, the MS-HAB-compatible ReplicaCAD assets, and ReplicaCADRearrange task plans. This repository's downloader fetches them from their official hosts at pinned revisions. The [upstream asset instructions](https://github.com/arth-shukla/mshab#setup-and-installation) explain their role. Choose an explicit asset directory and check the download and extracted sizes before installation.

The large demonstration datasets are not required just to run released RL checkpoints. Download only the assets and policy files required for the intended evaluation. Obtain policies from [arth-shukla/mshab_checkpoints](https://huggingface.co/arth-shukla/mshab_checkpoints), retaining their model-card and license information.

The unmodified upstream evaluator loads additional object-specific checkpoints even when `rl_all_obj` is selected. The documented command therefore includes `--full-tidy` to fetch the complete `rl/tidy_house/**` checkpoint subtree. Omitting that flag downloads only the Navigate/Pick/Place all-object policies, which is insufficient for the unmodified evaluator.

The downloader pins four upstream revisions:

| Artifact source | Revision |
|---|---|
| YCB archive in `haosulab/ManiSkill2` | `a1988e0918a9a1f2a9cb46ad7984efc3d422ac63` |
| `haosulab/ReplicaCAD` | `88bdca74dd4ec1f8c994904f7985392fc3f2b4c4` |
| `haosulab/ReplicaCADRearrange` | `c5921276ddece53607cd7dd55e7ed54385c9e49d` |
| `arth-shukla/mshab_checkpoints` | `91e96be85128df43728a7511355c3fa999bd2c94` |

ZIP archives are checked against pinned sizes and SHA-256 digests before extraction. Snapshot files are checked against the pinned repository metadata using SHA-256 or Git blob hashes. The downloader records `download-manifest.json` under the asset root and supports resuming interrupted downloads. These checks establish file identity; they do not establish policy success.

## Execution sequence

Run commands from the repository root on the Linux simulation host. Read the script help and setup configuration before downloading assets or allocating a GPU. Use writable storage with adequate space; paths below are examples and must be set for the host.

Start with a Linux Python 3.11 environment containing CUDA-enabled PyTorch matching the tested manifest. The bootstrap script inherits system site packages; it does not install or replace the GPU driver or PyTorch. With that prerequisite met, install the pinned simulator dependencies:

```bash
export BVI_WORKSPACE="/workspace/bvi"
bash scripts/bootstrap.sh
source "$BVI_WORKSPACE/activate.sh"
```

Choose a different `BVI_WORKSPACE` if needed. Keep the same value when sourcing `activate.sh`; that file selects the Python environment and sets `MS_ASSET_DIR`, `MSHAB_CHECKPOINT_DIR`, and headless rendering variables. Run the remaining repository commands from the repository root. A lightweight developer environment can separately install the standard-library `bvi` core and run its tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The current suite passed 56 tests on CPU. It exercises protocol validation, runtime control flow, injected provider behavior, bridge accounting/recovery, and download checks; no simulator rollout or paid model request is performed by these tests.

Then prepare external assets and inspect a real task:

```bash
python scripts/download_assets.py \
  --root "$MS_ASSET_DIR" \
  --checkpoint-root "$MSHAB_CHECKPOINT_DIR" \
  --full-tidy
python scripts/smoke_mshab.py --steps 200 --output runs/g1
```

`--root` is `MS_ASSET_DIR`; the downloader appends `/data` itself. Do not pass the `data` subdirectory as the root. The command above downloads assets and checkpoints. Use `--only assets` or `--only checkpoints` to perform either part separately, retaining `--full-tidy` for the complete checkpoint layout. A download needs no GPU, although a rented host can still incur infrastructure charges while it runs.

Once the G1 smoke test passes and the official policies are available, run the diagnostic official-policy chain:

```bash
python scripts/run_official.py \
  --max-steps 7000 \
  --checkpoint-root "$MSHAB_CHECKPOINT_DIR" \
  --output runs/g3
```

The chain runner does not by itself establish a VLM result or full validation coverage. Its checkpoint loading, scene selection, and trajectory outcome must be included in the run manifest.

The first recorded seed-0 run completed 7,000 steps but did not complete the task. The [diagnostic result](evaluation.md#first-official-policy-diagnostic) reports the official metrics and the first failure; running to the full horizon does not by itself pass G2 or G3.

## Coordinator diagnostic

The coordinator entrypoint exercises official skills through the `bvi` protocol and serial runtime. First use the oracle mode to diagnose the adapter:

```bash
python scripts/run_coordinator.py \
  --dry-run \
  --checkpoint-root "$MSHAB_CHECKPOINT_DIR" \
  --max-env-steps 500 \
  --max-calls 3 \
  --output runs/coordinator-oracle
```

Here `--dry-run` means **no model API requests**, not no execution. It loads the real simulator and checkpoints, uses the GPU, and selects skills from oracle task metadata. This bounded single-scene diagnostic is not an official benchmark score or a VLM result. The default policy mode is `rl_all_obj`.

A seed-0 oracle run using `--policy-type rl_per_obj` completed Navigate, Pick, and navigation while holding before failing during Place at total step 470. To diagnose all four invocations, use at least `--max-calls 4` with adequate step and wall-clock budgets. The three-call example above intentionally stops earlier and cannot verify the final Place invocation. See the [per-object result](evaluation.md#per-object-oracle-protocol-diagnostic).

The runner writes `events.jsonl`, `frames/`, `videos/`, `run-metadata.json`, and `summary.json` in the output directory. Use a fresh output directory for each attempt. Metadata labels the dispatcher, oracle target/completion sources, scene coverage, and whether the run used a VLM.

Live VLM mode passed the constrained G4 interface gate with `gpt-5.6-luna`, `--seed 1`, `--policy-type rl_per_obj`, and `--max-calls 6`. The run completed the first object's four-skill chain, then stopped at the request limit with the full task incomplete. Every decision had one allowed oracle skill/target pair. Live mode omits `--dry-run` and requires all of the following:

| Argument or prerequisite | Purpose |
|---|---|
| `--provider openai` or `--provider anthropic` | Select a transport |
| `--model MODEL_ID` | Select an explicit, account-accessible image-input model |
| `--authorization-id APPROVED_ID` | Reference the approved experiment scope |
| `--max-api-cost-usd LIMIT` | Bound the total configured cost reservations |
| `--request-cost-ceiling-usd LIMIT` | Reserve a conservative amount before each request |
| `--max-calls N`, `--max-output-tokens N` | Bound attempts and response size |
| Matching optional SDK and local credentials | Enable the selected provider |

Install an optional SDK with `python -m pip install -e '.[openai]'` or `python -m pip install -e '.[anthropic]'`. Provide `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` through the process environment or the ignored `.env.local` file. The runner reads only those known names and does not print the key; `--credentials-file` can select another local path.

OpenAI defaults are `--image-detail low` and `--reasoning-effort none`; the chosen model must support those values. Both providers default to a 2,048-token output limit. `--max-input-bytes` defaults to 128,000 and limits serialized content size, not precise billable input tokens. See [API configuration](#api-configuration) before enabling real requests.

With `--transport bridge`, the simulation host writes requests to its output directory and a local helper performs the API call over an SSH file bridge. The key stays on the local computer. Both sides validate the same model and spending scope; matching attempt IDs link their logs without counting one request twice. See [local inference with remote simulation](bridge.md) for the two-terminal commands and the tested run configuration. Both offline recovery tests and the six-request live run have been verified.

## Verification order

1. Verify CUDA arithmetic and that Vulkan selects the NVIDIA hardware device.
2. Record a Fetch empty-scene rollout, action bounds, camera outputs, and simulator/control rates.
3. Load a genuine ReplicaCAD MS-HAB task; record reset/step and observation metadata.
4. Load each official skill checkpoint; verify raw outputs are finite and have the expected shape, then verify the actions delivered through the controller/adapter are within bounds. Record raw out-of-range events and confirm at least one successful rollout per skill.
5. Execute a continuous chain, preserving the physical state between skills.
6. Enable the VLM provider only after the previous checks pass and the request budget is configured.

## Recorded G1 result

The real-scene smoke test used TidyHouse `val`, seed 0, build configuration index 69 and task-plan index 23 under the pinned assets. These are implementation indices, not a claim of validation-set coverage.

| Field | Recorded value |
|---|---|
| Environment | `SequentialTask-v0`, Fetch, one environment |
| Actions | 200 zero vectors; `Box(-1, 1, (13,), float32)` |
| Controller | `pd_joint_delta_pos` |
| Rates | Control 20 Hz; simulation 100 Hz |
| Wrapped state | Shape `[1, 42]`, `float32` |
| Wrapped head depth | Shape `[1, 3, 1, 128, 128]`, `int16` |
| Wrapped hand depth | Shape `[1, 3, 1, 128, 128]`, `int16` |
| Visualization | Nonblank 512 × 512 RGB frames; exported video |
| Timed loop | 13.15 s for stepping, rendering, and video encoding; excludes environment initialization/reset |

The script emits `metadata.json`, `scene-smoke.png`, and `scene-smoke.mp4` in its output directory. `passed=true` refers to the smoke-test assertions. `benchmark_result=false` is intentional: no learned navigation/manipulation policy was evaluated, and the script does not assert task success. The timing is a single recorded diagnostic, not a simulator performance benchmark.

The earlier empty-scene test also recorded raw head/hand RGBD at 128 × 128, with RGB `uint8` and depth `int16`. The G1 policy observation above is the wrapped depth/state representation; VLM RGB is captured separately by the coordinator adapter.

PPO's raw mean action can exceed the environment's `Box` bounds. This alone is not a malformed model output: the original controller path clips actions, and an adapter feeding the strict `bvi` runtime must apply the corresponding clipping before runtime validation. Preserve raw action values and clipping/out-of-range counts for diagnosis. The official runner records `raw_outside_unit_box` per step; aggregate those events when reporting the rate. NaN/Inf values or a wrong action dimension remain errors and must not be hidden by clipping.

## API configuration

Environment and official policy checks do not need paid model calls. The implemented coordinator accepts an injected transport, explicit provider/model, and an `APIBudget` with request count, output-token, input-byte, and cost-reservation limits. Optional OpenAI/Anthropic transports load their SDKs lazily and use environment credentials. Do not put API keys in configuration files committed to the repository.

An input-byte cap is not a precise input-token cap. Cost reservations are conservative operator-supplied amounts, not a provider-enforced billing ceiling. Unknown actual costs remain pending reconciliation. Before enabling a real provider, validate the expected image/text input size and current pricing, and configure an authorized request budget.

An API client or a mocked response is not a validated VLM integration. A successful VLM run must save the actual image references, validated requests, feedback, model identity, provider usage, and decision transitions. Provider errors and budget exhaustion are terminal or recoverable runtime events with explicit logging.

## Reproducibility artifacts

Every run should retain:

- A resolved configuration, source/checkpoint revisions, and package/system manifest.
- Scene ID, task-plan ID, seed, initial-state identity, and navigation mode.
- Observation/action metadata, policy transitions, and per-step or per-event outcomes.
- Coordinator requests/results, evidence references, and provider usage if applicable.
- A video, official metrics where applicable, and a factual failure report.

Keep local machine addresses, account identifiers, credentials, and billing records outside public artifacts. Publish only deliberately reviewed results; raw logs and videos are ignored by default.
