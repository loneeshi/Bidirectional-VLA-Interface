# C12: locate the first failed grasp before another training run

This is an offline comparison with successful training trajectory `workspace-ppo-02`, not a causal intervention or held-out evaluation.

## Confirmed observations

- C12 executes 29 PPO navigation steps and 153 real π manipulation predictions. The existing audit confirms executed actions match bounded predictions and the declared stationary-head mask.
- Initial relative qpos/qvel differs from this teacher by at most 0.08746 across the 30 components. This mixed-unit maximum is a diagnostic, not a physical distance. Initial maximum absolute qvel is 0.81937, so handoff is not at rest.
- First action MSE against the teacher is 0.003851. A wholesale action permutation is not supported by this observation, but controller semantics still require direct verification.
- Both trajectories first show substantial contact at Pick step20. C12 forces at20–22 are695.1/475.6/409.1; teacher701.7/520.3/462.8. Contact alone therefore does not distinguish failure from success.
- Both close the gripper at step23. Teacher obtains native `is_grasped=true`; C12 does not. C12 opens again at27 and continues switching without obtaining a grasp. All153 learner steps remain ungrasped.
- 61/153 raw action vectors exceed normalized bounds in at least one channel. Clipping alone does not establish causality: the first, near-teacher prediction also clips slightly.
- By step41 the trajectory is far from this teacher's proprioceptive states. Native cumulative-force failure is the final termination mechanism, not yet the established initiating cause.

## Current hypothesis and decisive next measurements

The leading hypothesis is insufficient grasp alignment at the step23 closure, followed by failure to recover after a missed grasp. RGB contact sheets show a missed object, but do not measure millimeter alignment or identify collision pairs. The target geometry, TCP-to-object transform and finger contacts must be logged before assigning a root cause. Ground-truth poses may be used for diagnostics only, without feeding them to the evaluated policy.

Before more training: reproduce C with identical checkpoint/config; log TCP/object poses, both finger contacts, qpos/qvel, controller targets and raw actions at every step15–35. Compare a fresh successful SAC trajectory from the same initialization. Test short, labeled diagnostic branches from the same saved state: π vs SAC one-step action, and π temporal chunk length with repeated seeds. Teacher interventions are diagnostic, never C success. Separate handoff settling from grasp-control changes so each intervention has an interpretable effect.

No additional GPU or external API was started for this offline analysis. The C13/D10 language ablations remain unexecuted. Do not infer that more LoRA steps or a different prompt will repair the grasp.
