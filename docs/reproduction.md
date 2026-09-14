# Setup and reproduction

The simulator was tested on Linux with an NVIDIA RTX A6000. The validated result is a 60-step Fetch `Empty-v1` rollout. ReplicaCAD loading, pretrained skill execution, and the VLM loop have separate acceptance gates and must not be inferred from this smoke test.

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

Use a dedicated Python environment. Preserve the complete resolved package list and both upstream commits for every experiment. Upstream branch names are moving references; the recorded revisions establish the tested baseline.

## Simulator installation

The official [MS-HAB installation guide](https://github.com/arth-shukla/mshab#setup-and-installation) selects the `mshab` branch of ManiSkill. Start from that compatible branch, then use the pinned commits above. The simulator requires working CUDA and NVIDIA Vulkan access; a successful `nvidia-smi` alone is insufficient.

For headless operation, expose graphics as well as compute capabilities to the container. `libEGL.so.1` was initially missing in the tested image and was supplied by Ubuntu's `libegl1` package. Record this system dependency in the environment build rather than relying on a manual change that disappears when a container restarts.

Place source checkouts, the Python environment, simulator caches, assets, and results on persistent storage when the host is ephemeral. Persisting source files does not by itself preserve system packages installed outside that storage.

Repository-specific installation and simulation entrypoints are listed below. Their availability is separate from each acceptance gate: G1–G4 still need successful recorded runs. The presence of these scripts is not a claim of a fully reproduced MS-HAB evaluation.

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

The initial suite passed 28 tests on CPU. It exercises protocol validation, runtime control flow, and injected provider behavior; no simulator rollout or paid model request is performed by these tests.

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

The chain runner does not by itself establish a VLM result or full validation coverage. Its checkpoint loading, scene selection, and trajectory outcome must be included in the run manifest. VLM runner instructions will be added after the image/request/feedback path is exercised; there is currently no validated VLM launch command.

## Verification order

1. Verify CUDA arithmetic and that Vulkan selects the NVIDIA hardware device.
2. Record a Fetch empty-scene rollout, action bounds, camera outputs, and simulator/control rates.
3. Load a genuine ReplicaCAD MS-HAB task; record reset/step and observation metadata.
4. Load each official skill checkpoint; verify raw outputs are finite and have the expected shape, then verify the actions delivered through the controller/adapter are within bounds. Record raw out-of-range events and confirm at least one successful rollout per skill.
5. Execute a continuous chain, preserving the physical state between skills.
6. Enable the VLM provider only after the previous checks pass and the request budget is configured.

The observed empty-scene interface used a 13-dimensional normalized action, a 20 Hz control rate, a 100 Hz physics rate, and head/hand RGBD cameras at 128 × 128. RGB was `uint8`; depth was `int16`. Record the actual interface again in a task environment, particularly after applying the policy observation wrappers.

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
