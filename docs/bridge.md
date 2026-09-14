# Local API inference with remote simulation

The file bridge lets an NVIDIA Linux host run MS-HAB while an API key stays on the local development computer. The simulator writes image/request envelopes; a local helper retrieves them over SSH, calls the selected provider, and transfers the result back. It has passed offline recovery tests and a six-request live `gpt-5.6-luna` diagnostic. See the [verified result and limitations](evaluation.md#live-vlm-protocol-diagnostic).

## Before starting

Install this repository on both machines. The remote machine needs the [simulator and assets](reproduction.md); the local machine needs Python, `ssh`/`scp`, this package, and the selected optional SDK. For OpenAI:

```bash
python -m pip install -e '.[openai]'
```

Create and verify a local SSH alias for the simulation host. Keep API credentials only in the local process environment or the ignored `.env.local` file. The remote bridge path must be absolute and contain no spaces or parent-directory traversal. It must match the remote runner's output directory plus `/bridge`.

The examples below use a **$1 total reservation, $0.10 per attempt, and at most 10 attempts** to illustrate a bounded experiment. Choose your own approved scope and verify the reservation against the current provider price and input/output sizes. These values do not create a provider-enforced billing ceiling or report any user's funds.

## Local helper

Start the local helper first in one terminal. It waits for a validated remote request and makes no API request just by starting. Replace the SSH config path, alias, model ID, and authorization reference with your configuration. This example uses PowerShell:

```powershell
python scripts/serve_vlm_bridge.py `
  --ssh-config "C:/path/to/ssh_config" `
  --ssh-alias simulation-host `
  --remote-bridge-dir /workspace/bvi/runs/coordinator-vlm/bridge `
  --provider openai `
  --model MODEL_ID `
  --authorization-id APPROVED_EXPERIMENT_ID `
  --max-calls 10 `
  --max-output-tokens 2048 `
  --max-input-bytes 128000 `
  --max-api-cost-usd 1 `
  --request-cost-ceiling-usd 0.1 `
  --image-detail low `
  --reasoning-effort none `
  --idle-timeout-seconds 300 `
  --max-wall-seconds 1200 `
  --output runs/bridge-local
```

Linux/macOS users can run the same arguments with Bash line continuations and their own SSH config path. The model must support the requested image detail and reasoning settings. For a different provider, use the matching SDK and transport settings rather than assuming identical provider options.

## Remote runner

In a second terminal, connect to the Linux host, activate the simulator environment, and run the following from its repository root. Use the **same** model, authorization reference, and API limits as the local helper:

```bash
export BVI_WORKSPACE="/workspace/bvi"
source "$BVI_WORKSPACE/activate.sh"
python scripts/run_coordinator.py \
  --transport bridge \
  --provider openai \
  --model MODEL_ID \
  --authorization-id APPROVED_EXPERIMENT_ID \
  --max-calls 10 \
  --max-output-tokens 2048 \
  --max-input-bytes 128000 \
  --max-api-cost-usd 1 \
  --request-cost-ceiling-usd 0.1 \
  --image-detail low \
  --reasoning-effort none \
  --max-env-steps 7000 \
  --max-wall-seconds 1200 \
  --checkpoint-root "$MSHAB_CHECKPOINT_DIR" \
  --output /workspace/bvi/runs/coordinator-vlm
```

Use a fresh remote output directory for a new experiment. The remote process loads the real simulator and policies; a rented GPU can incur charges while it initializes or waits. The simulation remains bounded independently of the API reservations. This runner is a single-scene diagnostic with oracle target and completion metadata, not a full validation evaluator.

The recorded successful diagnostic used `--model gpt-5.6-luna`, `--seed 1`, `--policy-type rl_per_obj`, `--max-calls 6`, and `--max-env-steps 1800`; use six calls on both sides when reproducing that configuration. Other settings matched the image/output and wall-clock limits above. Six skills completed in 336 steps; the first object's Navigate → Pick → Navigate → Place chain ended at step 233. The full task did not finish. Each decision had one allowed skill, so this is a transport and execution demonstration rather than a test of free skill selection.

## Logs, recovery, and cost accounting

- Remote `events.jsonl` and the local helper's mirror use the same attempt ID. They are two records of one API request, not two charges.
- The local helper persists an attempt claim before calling the provider. Reuse its output directory when recovering the same experiment so previous claims and reservations remain visible.
- A cached successful response can be resent if transfer failed, without a second API call. An unresolved prior claim or unknown provider outcome stops automatic replay and needs reconciliation.
- Responses are transferred with an atomic rename. Keys and authorization headers are not part of the bridge envelope.
- SDK automatic retries are disabled. Errors, rejected outputs, and uncertain requests must remain in the usage record; an unsuccessful task is not proof of zero API cost.

Input bytes are a guard on request size, not exact billable tokens. Provider usage and billing evidence determine the final cost. Ending the bridge does not stop a remote GPU or remove retained storage; manage those resources separately.

The remote output retains `run-metadata.json`, `summary.json`, frame references, video, and the request/response spool. G4 requires actual images, a validated skill request, real execution feedback, and a subsequent VLM decision. Offline bridge tests and a startup message do not meet that requirement.
