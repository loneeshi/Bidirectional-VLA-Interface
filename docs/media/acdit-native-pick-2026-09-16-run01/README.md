# Native AC-DiT Pick — 2026-09-16, run01

[acdit-native-pick-episode000-failed.mp4](acdit-native-pick-episode000-failed.mp4)
records the first policy episode, seed 2024, `set_table/pick/013_apple` validation
split. Left: Fetch head camera; right: hand camera, each 128×128. Playback is the
simulated 20 Hz control frequency; inference actually takes longer than real time.
There is no overlay covering the observations.

Result: **failed**, native termination at 199 actions. Grasp first appears at
step 16; after a false grasp flag at step 21 it persists from step 22 through 199.
Steps 31–38 meet the rest-position threshold but not the static criterion. The
final held pose is 10.84 cm from the required rest location, outside 5 cm tolerance.
No additional hold controller or relaxed success criterion was used.

See [full report](../../acdit-native-validation.md) and the parent
[SHA256 manifest](../manifest.json). This is a native policy diagnostic, not
GPT/LightNav collaboration, trained TAPT, or benchmark success.
