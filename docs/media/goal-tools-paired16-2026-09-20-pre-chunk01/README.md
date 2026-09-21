# Goal-tools paired panel: pre-chunk01 snapshot

This snapshot contains the five evaluation-terminal rows that existed before source-v4 chunk01: fixed seeds0–2 and GPT seeds0–1. They are five different plan UIDs, not repeated rollouts. All full-task success values are false, so every available video is explicitly a failed/incomplete rollout, not a successful demo.

| Arm | Seed | Plan UID | Outcome | Objects | Steps | API calls | Video |
|---|---:|---|---|---:|---:|---:|---|
| fixed |0|tidy_house-sequential-val-557-0|native benchmark failure|0/5|373|0|[fixed-seed-000-attempt-001.mp4](fixed-seed-000-attempt-001.mp4) · [summary](fixed-seed-000-attempt-001-summary.json)|
| fixed |1|tidy_house-sequential-val-90-0|native benchmark failure|0/5|266|0|[fixed-seed-001-attempt-001.mp4](fixed-seed-001-attempt-001.mp4) · [summary](fixed-seed-001-attempt-001-summary.json)|
| fixed |2|tidy_house-sequential-val-147-0|native benchmark failure|2/5|675|0|[fixed-seed-002-attempt-001.mp4](fixed-seed-002-attempt-001.mp4) · [summary](fixed-seed-002-attempt-001-summary.json)|
| GPT |0|tidy_house-sequential-val-557-0|invalid structured model output|0/5|0|1|No video: no physical step occurred. [summary](gpt-seed-000-attempt-002-summary.json)|
| GPT |1|tidy_house-sequential-val-90-0|invalid structured model output after execution|0/5|69|3|[gpt-seed-001-attempt-001.mp4](gpt-seed-001-attempt-001.mp4) · [summary](gpt-seed-001-attempt-001-summary.json)|

This is not the completed16+16 result. Fixed full-task SR is0/3 and GPT full-task SR is0/2 only for this partial snapshot; denominators must not replace the planned16 per arm. Fixed completed-object mean is2/3; GPT is0/2. Invalid model output is an observed planner failure, not infrastructure failure or native task success.

The adjacent `*-summary.json` files are byte-identical downloads of the corresponding remote attempt summaries. Raw source remains under `/home/pshuai/bvi-research/runs/goal-tools-paired16-20260920/`.
