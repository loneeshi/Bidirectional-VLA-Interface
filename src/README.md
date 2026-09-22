# Source map

Current week: evaluator feedback with GPT + official PPO/SAC; continuous rule
progress remains opt-in. Status describes research scope, not test coverage.
Frozen modules retain their paths and imports. Legacy feedback exports are
available for compatibility; MS-HAB summaries use feedback/digest.py.

```text
scripts/run_ppo_sac_paired16.py (paired batches and resume)
  -> scripts/run_coordinator.py (episode loop)
     -> coordinator.py -> feedback/digest.py + views.py -> bridge -> GPT
     -> runtime.py -> goal_tools.py -> mshab_adapter.py -> PPO / SAC -> env.step
     -> optional feedback/trajectory.py -> next decision history
```

| Module | Status | Subsystem |
|---|---|---|
| `bvi/__init__.py` | active | core |
| `bvi/acdit_contract.py` | frozen | historical training and diagnostics (retained) |
| `bvi/acdit_tapt.py` | frozen | historical training and diagnostics (retained) |
| `bvi/baseline_v2.py` | frozen | historical training and diagnostics (retained) |
| `bvi/bridge.py` | active | organizer |
| `bvi/closed_loop_divergence.py` | frozen | historical training and diagnostics (retained) |
| `bvi/coordinator.py` | active | organizer |
| `bvi/continuation.py` | active | evaluation |
| `bvi/decision_trace.py` | frozen | historical training and diagnostics (retained) |
| `bvi/diagnostic_noise.py` | frozen | historical training and diagnostics (retained) |
| `bvi/feedback/__init__.py` | active | feedback |
| `bvi/feedback/digest.py` | active | feedback |
| `bvi/feedback/spawn_prior.py` | active | feedback |
| `bvi/feedback/trajectory.py` | active | feedback |
| `bvi/feedback/views.py` | active | feedback |
| `bvi/fetch_current_labels.py` | frozen | historical training and diagnostics (retained) |
| `bvi/fetch_pi_skill.py` | frozen | historical training and diagnostics (retained) |
| `bvi/fetch_segments.py` | frozen | historical training and diagnostics (retained) |
| `bvi/first_action_probe.py` | frozen | historical training and diagnostics (retained) |
| `bvi/goal_tools.py` | active | executors |
| `bvi/ia_call_predicates.py` | frozen | historical training and diagnostics (retained) |
| `bvi/ia_fetch_data.py` | frozen | historical training and diagnostics (retained) |
| `bvi/lightnav_skill.py` | active | executors |
| `bvi/logging.py` | active | core |
| `bvi/mshab_adapter.py` | active | environment |
| `bvi/native24_behavior_gate.py` | frozen | historical training and diagnostics (retained) |
| `bvi/native24_handoff.py` | frozen | historical training and diagnostics (retained) |
| `bvi/native24_handoff_sequence.py` | frozen | historical training and diagnostics (retained) |
| `bvi/native_pick_audit.py` | active | evaluation |
| `bvi/native_s2_deployment.py` | frozen | historical training and diagnostics (retained) |
| `bvi/nav_camera_env.py` | active | environment |
| `bvi/navigation_fidelity.py` | active | evaluation |
| `bvi/null_drift_calibration.py` | frozen | historical training and diagnostics (retained) |
| `bvi/observation_progress_preflight.py` | frozen | historical training and diagnostics (retained) |
| `bvi/official_fetch_data.py` | frozen | historical training and diagnostics (retained) |
| `bvi/organizer.py` | active | organizer |
| `bvi/pi05_recipe.py` | frozen | historical training and diagnostics (retained) |
| `bvi/privileged_fetch_data.py` | frozen | historical training and diagnostics (retained) |
| `bvi/progress_monitor.py` | active | feedback legacy facade |
| `bvi/progress_timing.py` | frozen | historical training and diagnostics (retained) |
| `bvi/protocol.py` | active | core |
| `bvi/providers.py` | active | organizer |
| `bvi/recovery_collection.py` | frozen | historical training and diagnostics (retained) |
| `bvi/recovery_experiment.py` | frozen | historical training and diagnostics (retained) |
| `bvi/runtime.py` | active | core |
| `bvi/s1_capability_gate.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s1_diagnostic_cards.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s1_evidence.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s1_oracle_protocol.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s1_preprocessing_parity.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s2_diagnostic_gate.py` | frozen | historical training and diagnostics (retained) |
| `bvi/s2_offline_decomposition.py` | frozen | historical training and diagnostics (retained) |
| `bvi/sac_interface_baseline.py` | active | evaluation |
| `bvi/same_key_confirmation.py` | frozen | historical training and diagnostics (retained) |
| `bvi/task_memory.py` | active | feedback legacy facade |
| `bvi/teleport_skill.py` | active | executors |
| `bvi/tool_family.py` | frozen | historical training and diagnostics (retained) |
| `bvi/transform_parity.py` | active | evaluation |
| `bvi/vla_clients.py` | active | executors |

Registry coverage is checked by tests/test_module_registry.py.
