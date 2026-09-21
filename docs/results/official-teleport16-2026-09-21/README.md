# Official MS-HAB teleport reference — matched 16 plans

Status: finished at 2026-09-21T03:47:44Z (2026-09-20 23:47:44 EDT).

This reference uses the exact paper-release upstream MS-HAB commit
`4729821db3fc94a2470cfd625e6f8ab439f01478`. Its official evaluator teleports
each Navigate subtask using the paper's spawn randomization and executes the
official per-object SAC policy for Pick/Place. It is separate from the later
upstream commit `8b2f934`, which added learned PPO navigation.

The plan roster is the same 16 unique TidyHouse validation plans used by the
continuous-navigation Fixed/GPT panel: seeds 0–11, 13, 14, 16 and 19. Each
episode binds one exact plan UID. Teleport intentionally changes the navigation
handoff distribution, so initial physical states are not claimed identical to
the PPO runs.

## Result

- Planned/completed: 16/16
- Infrastructure failures: 0
- Full-task successes: 0/16 (SR 0%)
- Completed objects: 12/80 (15.0%), mean 0.75 per episode
- Per-seed completed objects: `0:0, 1:0, 2:1, 3:3, 4:0, 5:2, 6:0, 7:1,
  8:0, 9:0, 10:1, 11:1, 13:1, 14:2, 16:0, 19:0`

For the same plan roster, the continuous PPO Fixed arm completed 14/80 objects
(17.5%) and the GPT tool-calling arm completed 13/80 (16.25%); all three arms
had full-task SR 0/16. With only 16 plans and zero full successes, this is a
matched diagnostic comparison, not a precise estimate of the published
1000-episode benchmark score.

Machine-readable panel: [panel-status.json](panel-status.json), SHA-256
`93ad15b74a26a5eeabad2757369beb86a6c59733e4e837edb08904ddf266bb50`.
Raw remote run: `/home/pshuai/bvi-research/runs/official-teleport16-20260921`.

The first seed had two preserved attempts. Attempt 1 stopped before model or
environment execution because the upstream config asserts that the task-plan
path contains `tidy_house`. Attempt 2 executed normally; a local result parser
initially rejected PyTorch's `device='cuda:0'` suffix, then recovered the same
physical result from its existing official artifacts without rerunning it.

GPU1 was 15 MiB/0% after completion; GPU0's unrelated process was untouched.
No training, fine-tuning, external model API request, or RunPod resource was
used. New rental and API cost are USD 0; laboratory service cost remains unknown.
