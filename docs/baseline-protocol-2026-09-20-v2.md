# Official-model baseline protocol, revision 2

## Comparison

The main paired comparison has 16 unique TidyHouse validation plans per arm:
fixed official subtask order with published PPO navigation/per-object SAC;
GPT-5.6 Luna whole-goal tool selection with the same PPO/SAC checkpoints.
This is a baseline using official models with continuous navigation. It is a
valid research setting, not an exact reproduction of the paper's teleport setup.

Freeze plan UID, scene, object/destination bindings, seed, initial simulator-state
hash, checkpoint hashes, native scoring, force thresholds, observation inputs,
action limits, source archive, prompt, and response-normalization version before
the next formal attempt. GPT retains the disclosed native object-order constraint.
Both arms retain real navigation and unchanged native scoring. The existing
navigation ignore_arm_checkers=True setting must be disclosed in comparisons.
The fixed arm has up to 175 slices; GPT has up to 40 API decisions. Report this
budget asymmetry and budget-terminated episodes explicitly.

Report full-task success, native objects completed per episode, reached-subtask
success/attempt counts, actions, wall time, API physical attempts and costs.
Each arm retains planned=16 with completed, infrastructure-failed, timeout,
running and not-run counts; partial observed SR includes its observed denominator.
Compare matched plans separately while one arm has additional completed rows.
Reached-subtask rates are conditional on surviving earlier subtasks, not standalone
skill performance estimates. A failure during manipulation does not establish
that navigation handoff state played no causal role.

## Response integrity

Revision identical_json_repetition/1 collapses only byte-identical complete JSON
objects (allowing whitespace between copies). The resulting single request must
pass the existing complete schema, grounded target, budget and observation checks.
Different objects, partial JSON, prose, duplicate fields and invalid targets are
rejected. This is one decision and one tool execution, without an extra API call.
api_usage retains raw text; coordinator_output_normalized records copy count and
raw/normalized SHA256. Never execute both copies or choose between different calls.

The four historical GPT evaluation-terminal attempts ended on duplicated JSON:
seed0 attempt002, seed1 attempt001, seed2 attempt003, seed3 attempt001. Their
response shape is verified; the originating layer (service/model/SDK aggregation)
is not established from the flattened stored response. Their physical traces and
costs remain evidence. Treat model-capability validity as unresolved/excluded
pending an additive adjudication, rather than claiming GPT planning SR=0/4.
Do not overwrite their raw summaries, statuses or videos, or silently change a
completed row to runnable. Repair reruns require explicit linked new attempts.
Retain the five fixed results if frozen physical configuration remains identical.

## Small batches and resumption

Runner default is at most 2 new fixed and 2 new GPT attempts per invocation.
These are caps, not a requirement to rerun completed fixed rows to balance a repair
batch. An infrastructure retry consumes one slot. Every attempt has an independent
directory and is atomically recorded before and after execution. Preserve locks,
source versions, all results and global API claims across invocations.

Preflight verifies no duplicate runner, idle/authorized GPU1, API bridge availability,
remaining request/resource budget and paired reset hashes. After each small batch,
check terminal rows, error categories, completed artifacts and provider accounting
before continuing. Transport retries are bounded at 10 and individually charged
against existing global claims. True model/physical failures remain scored outcomes.
Infrastructure faults checkpoint and stop the chunk for diagnosis; never continue
burning the remaining panel on a repeated known interface defect. External outages
cannot be guaranteed absent; successful rows must survive every interruption.

## Teleport reference

The paper reports RL-Per versus RL-All full long-horizon results in Appendix A.4.5,
Figure 10: https://arxiv.org/html/2412.13211#A4.SS5 . Its cross-apartment navigation
uses teleport. Published aggregate results are an external reference, not paired
observations on our manifest; numeric values must be verified before transcription.

A local matched teleport reference needs 16 fixed-order episodes on the same plans,
not another 16+16 panel unless a teleport-GPT comparison is separately requested.
Use the upstream teleport implementation, verify its reset/randomization semantics,
and identify its intentional handoff-state difference. Do not implement an ad hoc
pose change and call it official. Keep its result table separate from the continuous
navigation panel. Small batches, native scoring and the existing resource ceiling
apply. No new simulation or paid API was launched as part of this local repair.

## Acceptance before resuming formal GPT episodes

Offline replay of the four stored duplicated responses must yield exactly one
valid request each, with no API or environment actions. Conflicting, malformed and
unsafe calls must remain rejected. CPU suite and a train/development real bridge
smoke must pass before freezing the next source/prompt/manifest snapshot. Local CPU
repair alone does not certify live service or native-task performance.

Local validation: 640 CPU tests passed, 6 skipped. Offline replay against the
original stored prompt targets, allowed calls and skill contracts passed for all
four affected responses: 72f896daa2cd418c8e4ace15667257ec (navigate, seed0 step0),
9120282f716243b19093c6fe8a082ad6 (pick, seed1 step69),
8d847004930a4c118564269ca0c6a936 (navigate, seed2 step0), and
5ed797ffd2144bcf90b6302a980ea8d4 (navigate, seed3 step40). Each had exactly two
identical copies. No provider calls or physical actions were made by the replay.
This validates deterministic normalization, not the upstream cause of duplication.
