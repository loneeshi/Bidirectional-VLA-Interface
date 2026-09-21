# GPT bridge

The GPT setting keeps provider credentials on a local trusted computer while
MS-HAB runs on a Linux simulation host. The simulator and bridge exchange
validated files over one explicitly configured SSH destination.

## Security boundary

- Credentials come from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` or an ignored
  local credential file; they are never accepted in the remote request.
- The provider, model, authorization reference, call count, byte/token limits,
  wall time and cost reservations are fixed when the bridge starts.
- Responses are cached by attempt ID. An uncertain provider result stops the
  bridge rather than silently issuing a duplicate paid request.
- Remote paths, host keys and credential files are validated before serving.
- Bridge logs record status and accounting metadata, not credential values.

## Start the bridge

Install the required extras, then use either an SSH config alias or a separate
server-credentials file with pinned known hosts.

```bash
bvi-eval bridge \
  --ssh-config /path/to/ssh_config \
  --ssh-alias simulation-host \
  --remote-bridge-dir /remote/run/bridge \
  --provider openai \
  --model gpt-5.6-luna \
  --authorization-id APPROVED_SCOPE \
  --max-calls 40 \
  --max-api-cost-usd 0.05 \
  --request-cost-ceiling-usd 0.00125 \
  --max-output-tokens 2048 \
  --max-input-bytes 512000 \
  --idle-timeout-seconds 300 \
  --max-wall-seconds 3600 \
  --output /local/audit/bridge
```

Starting the bridge does not itself make an API request. Start the GPT panel
only after the bridge prints `BRIDGE_READY`. Reuse the same output directory
when resuming so prior claims remain visible.
