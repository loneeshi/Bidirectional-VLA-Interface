"""One Astra decision loop, grouped by function. No environment or provider client.

Provenance: frozen C0/C1 deployments and the byte-verified pick_base extraction.
Robot defaults preserve the frozen algorithms; optional CPU changes are explicit.
"""
from __future__ import annotations
import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Protocol
import numpy as np
from PIL import Image
from .robot.eef_tools import vector, pose, FETCH_ORDER

# Configuration and explicit cross-turn state; no module-owned attempt state.
FROZEN = json.loads(Path(__file__).with_name('profiles.json').read_text(encoding='utf8'))
PROMPT = Path(__file__).with_name('prompt.md').read_bytes().decode('utf8')
MODEL = 'gpt-6-astra'
OUTPUT_CAP = 4000
CONTEXT_LIMIT = 1050000
RESOLUTION = 640
STEP_CAP = 600
TOOL_CAP = 25
HEAD_LIMITS = ((-1.57, 1.57), (-.76, 1.45))
@dataclass(frozen=True)
class Profile:
    name: str
    model: str = MODEL
    response_session: bool = False
    active_memory: bool = False
    task_handoff: bool = False
    transport_task: bool = False
    measure: bool = False
    modules: tuple = ('detector', 'navigate', 'manipulation')
    verification: bool = False
    step_cap: int = 600
    call_cap: int = 25
    close_uses_endpoint_tolerance: bool = False
    stall_feedback: bool = False
    stall_guard_mode: str = "legacy"
    use_located_semantics: bool = True
    verification_call_cap: int = 5
    verification_errors_continue: bool = False
    identity_revision: int = 1
    model_target_boxes: bool = False
    staged_metrics: bool = False
    automatic_supervision: bool = False
    history_first: bool = False
    base_motion_abort: bool = False
    schedule_precheck: bool = False
    robot_diagnostics: bool = False
    base_heading_hold: bool = False
    continuous_base_speed: bool = False
    target_association: bool = False
    target_evidence_diagnostics: bool = False
    base_progress_monitor: bool = False
    base_response_corroboration: bool = False
    base_response_monitor: bool = False
    base_response_debt_monitor: bool = False
    base_progress_abort: bool = False
    execution_lifecycle: bool = False
    grasp_alignment: bool = False
    weekly_budget: bool = False
    active_confirmation: bool = False
    session_usage_anchor: bool = False
    visual_identity_confirmation: bool = False
    multiview_confirmation: bool = False
    partial_target_marker: bool = False
    gripper_geometry_feedback: bool = False
    post_localization_review_recovery: bool = False
    smooth_grasp_localization: bool = False
    persistent_target_confirmation: bool = False
    motion_guard: bool = False
    self_collision_guard: bool = False
    base_tracking_guard: bool = False
    incremental_recording: bool = False
    fast_base_approach: bool = False
    primary_steps: int = 800
    search_reconsideration: bool = False
    identity_evidence_quality: bool = False
    search_progress: bool = False
    hd_textures: bool = False
    efficient_io: bool = False
    efficient_search: bool = False
    hot_continuation: bool = False
    cold_checkpoint: bool = False
    episode_memory: bool = False
    local_clearance: bool = False
    clearance_unknown_advisory: bool = False
    carried_envelope: bool = False
    local_support_floor: bool = False
    rest_progress_feedback: bool = False
    parallel_case_workers: bool = False
    r15_protocol: bool = False
    r15_incremental: bool = False
    r2a_identity: bool = False
    r2a_close: bool = False
    r2a_feedback: bool = False
    r2b_clearance: bool = False
    r2b_grip_monitor: bool = False
    motion_progress_only: bool = False
    compound_feedback: bool = False
    r2d_extended_calls: bool = False
    r2d_base_clearance: bool = False
    base_tracking_exit: bool = False
    arm_base_push_exit: bool = False
    g1_dispatch_semantics: bool = False
    error_progress_guard: bool = False
    dual_reachability: bool = False
    clearance_verification: bool = False
    blocked_grasp_direction: bool = False
    g1_insertion_guard: bool = False
    search_recessed_view: bool = False
    search_candidate_priority: bool = False
    identity_attempt_recovery: bool = False
    identity_confirmation: bool = False
    identity_multiview_evidence: bool = False
    structured_close_evidence: bool = False
    identity_confirmed_persistence: bool = False
    rest_joint_target: bool = False
    arm_tools_lock_base: bool = False
    arm_base_explicit: bool = False
    r2b_test_lift: bool = False
    r15_completion: bool = False
    r15_localization_eval: bool = False
    r15_standoff: bool = False
    r15_pregrasp: bool = False
    r15_carry: bool = False
    r1_extended_budget: bool = False
    strict_tool_errors: bool = False
    view_occlusion_guidance: bool = False
    lean_feedback: bool = False
    cache_aware_budget: bool = False
    # S2 is an opt-in protocol, not a change to any frozen single-tool profile.
    stage_plan: bool = False
    short_plan_queue: bool = False
    image_window: bool = False
    segmented_exit_guidance: bool = False
    rest_base_push_report_only: bool = False

    approach_point: bool = False

    multi_locate: bool = False


C0 = Profile(**FROZEN['profiles']['C0'])
C1 = Profile(**FROZEN['profiles']['C1'])
C0_CLOSE_ALIGNED = Profile(**FROZEN['profiles']['C0-close-aligned'])
C2 = Profile(**FROZEN['profiles']['C2'])
PROFILES = {'C0': C0, 'C1': C1, 'C0-close-aligned': C0_CLOSE_ALIGNED, 'C2': C2}
from dataclasses import replace
C2_R2 = replace(C2, name='C2-r2-shared-target-evidence', identity_revision=2)
PROFILES['C2-r2'] = C2_R2
C2_BASELINE_PREP = replace(C2_R2, name='C2-r2-baseline-preparation', stall_feedback=True, stall_guard_mode='reference_progress')
PROFILES['C2-baseline-prep'] = C2_BASELINE_PREP
C2_BASELINE_FAR = replace(C2_BASELINE_PREP, name='C2-r2-baseline-far-preparation', step_cap=1000, call_cap=40)
PROFILES['C2-baseline-far-prep'] = C2_BASELINE_FAR
C2_R25 = replace(C2_BASELINE_PREP, name='C2-r2.5-astra-boxes', model_target_boxes=True)
C2_R25_FAR = replace(C2_R25, name='C2-r2.5-astra-boxes-far', step_cap=1000, call_cap=40)
PROFILES.update({'C2-r2.5': C2_R25, 'C2-r2.5-far': C2_R25_FAR})
C2_R3 = replace(C2_R25, name='C2-r3', staged_metrics=True,
                automatic_supervision=True, history_first=True, base_motion_abort=True)
C2_R3_FAR = replace(C2_R3, name='C2-r3-far', step_cap=1000, call_cap=40)
PROFILES.update({'C2-r3': C2_R3, 'C2-r3-far': C2_R3_FAR})
C2_R31 = replace(C2_R3, name='C2-r3.1', schedule_precheck=True,
                 robot_diagnostics=True, base_heading_hold=True, continuous_base_speed=True)
C2_R31_FAR = replace(C2_R31, name='C2-r3.1-far', step_cap=1000, call_cap=40)
PROFILES.update({'C2-r3.1': C2_R31, 'C2-r3.1-far': C2_R31_FAR})
C2_R32 = replace(C2_R3, name='C2-r3.2', schedule_precheck=True,
    robot_diagnostics=True, target_association=True, target_evidence_diagnostics=True,
    base_progress_monitor=True, base_response_monitor=True, base_response_debt_monitor=True,
    base_progress_abort=True, execution_lifecycle=True)
C2_R32_FAR = replace(C2_R32, name='C2-r3.2-far', step_cap=1000, call_cap=40)
PROFILES.update({'C2-r3.2': C2_R32, 'C2-r3.2-far': C2_R32_FAR})
C2_R33 = replace(C2_R32, name='C2-r3.3', base_response_corroboration=True)
C2_R33_FAR = replace(C2_R33, name='C2-r3.3-far', step_cap=1000, call_cap=40)
PROFILES.update({'C2-r3.3': C2_R33, 'C2-r3.3-far': C2_R33_FAR})
SOL_SESSION = replace(C0, **FROZEN['sol_session']['profile'])
PROFILES['Sol-session-v1'] = SOL_SESSION
SOL_PICK20 = replace(SOL_SESSION, name='Sol-session-pick20-v1')
PROFILES[SOL_PICK20.name] = SOL_PICK20
# CPU-ready revision; no existing paid scope authorizes this profile.
SOL_PICK20_ALIGNED = replace(SOL_PICK20,name='Sol-session-pick20-aligned-v1',
                             call_cap=40,grasp_alignment=True)
PROFILES[SOL_PICK20_ALIGNED.name] = SOL_PICK20_ALIGNED
SOL_DEV_R1 = replace(SOL_PICK20_ALIGNED, name='Sol-pick-dev-r1-interface-v1',

    view_occlusion_guidance=True, approach_point=True, multi_locate=True, session_usage_anchor=True, strict_tool_errors=True)

SOL_DEV_R1_1600 = replace(SOL_PICK20_ALIGNED, name='Sol-pick-dev-r1-budget1600-v1',
    step_cap=1600, call_cap=50, r1_extended_budget=True,
    view_occlusion_guidance=True, approach_point=True, multi_locate=True,
    session_usage_anchor=True, strict_tool_errors=True)
SOL_DEV_R1_PARALLEL = replace(SOL_PICK20_ALIGNED, name='Sol-pick-dev-r1-budget1600-parallel-v1',
    step_cap=1600, call_cap=50, r1_extended_budget=True, parallel_case_workers=True,
    view_occlusion_guidance=True, approach_point=True, multi_locate=True,
    session_usage_anchor=True, strict_tool_errors=True)
PROFILES[SOL_DEV_R1_PARALLEL.name] = SOL_DEV_R1_PARALLEL
DEV_PROFILES = (SOL_DEV_R1, SOL_DEV_R1_1600, SOL_DEV_R1_PARALLEL)
PROFILES[SOL_DEV_R1_1600.name] = SOL_DEV_R1_1600

PROFILES[SOL_DEV_R1.name] = SOL_DEV_R1
# Both arms derive directly from the unchanged aligned baseline. Recording and
# scoring are common infrastructure; E2/S1-S3 belong only to the candidate.
SOL_R15_BASELINE = replace(SOL_PICK20_ALIGNED, name='Sol-aligned-v1-new-limits',
    step_cap=1600, call_cap=50, r1_extended_budget=True, parallel_case_workers=True,
    r15_protocol=True, r15_incremental=True, r15_localization_eval=True,
    strict_tool_errors=True, session_usage_anchor=True)
SOL_R15 = replace(SOL_PICK20_ALIGNED, name='Sol-pick-dev-r15',
    step_cap=1600, call_cap=50, r1_extended_budget=True, parallel_case_workers=True,
    r15_protocol=True, r15_incremental=True, r15_completion=True,
    r15_localization_eval=True, r15_standoff=True, r15_pregrasp=True, r15_carry=True,
    view_occlusion_guidance=True, approach_point=True, multi_locate=True,
    strict_tool_errors=True, session_usage_anchor=True)
R15_PROFILES = (SOL_R15_BASELINE, SOL_R15)
DEV_PROFILES += R15_PROFILES
PROFILES.update({p.name:p for p in R15_PROFILES})
SOL_R2A = replace(SOL_R15_BASELINE, name='Sol-pick-dev-r2a', r2a_identity=True, r2a_close=True, r2a_feedback=True)
R2A_PROFILES = (SOL_R15_BASELINE, SOL_R2A)
PAIRED_PROFILES = R15_PROFILES + (SOL_R2A,)
DEV_PROFILES += (SOL_R2A,)
PROFILES[SOL_R2A.name] = SOL_R2A
SOL_R2B = replace(SOL_R2A, name='Sol-pick-dev-r2b', r2b_clearance=True,
                  r2b_grip_monitor=True, r2b_test_lift=True, motion_progress_only=True, arm_base_explicit=True)
PAIRED_PROFILES += (SOL_R2B,)
DEV_PROFILES += (SOL_R2B,)
PROFILES[SOL_R2B.name] = SOL_R2B
R2B_PROFILES = (SOL_R2A,SOL_R2B)
# R2c derives from the evaluated R2b; the separate stop-on-proposal draft is off.
SOL_R2C = replace(SOL_R2B,name='Sol-pick-dev-r2c',arm_base_explicit=False,arm_tools_lock_base=True)
PROFILES[SOL_R2C.name]=SOL_R2C
DEV_PROFILES += (SOL_R2C,)
PAIRED_PROFILES += (SOL_R2C,)  # Shared executor protocol; R2c scope creates only candidate jobs.
SOL_R2D = replace(SOL_R2C,name='Sol-pick-dev-r2d',compound_feedback=True,
    r2d_extended_calls=True,call_cap=80,r2d_base_clearance=True,base_tracking_exit=True)
PROFILES[SOL_R2D.name]=SOL_R2D
DEV_PROFILES += (SOL_R2D,)
PAIRED_PROFILES += (SOL_R2D,)
SOL_R2E = replace(SOL_R2D,name='Sol-pick-dev-r2e',error_progress_guard=True,
    dual_reachability=True,clearance_verification=True,blocked_grasp_direction=True,
    rest_progress_feedback=True,search_recessed_view=True)
# G1 insertion remains uncalibrated and disabled by the user's explicit decision.
PROFILES[SOL_R2E.name]=SOL_R2E
DEV_PROFILES += (SOL_R2E,)
PAIRED_PROFILES += (SOL_R2E,)
SOL_R2E_RECOVERY = replace(SOL_R2E, name='Sol-pick-dev-r2e-identity-recovery',
    identity_attempt_recovery=True)
PROFILES[SOL_R2E_RECOVERY.name] = SOL_R2E_RECOVERY
DEV_PROFILES += (SOL_R2E_RECOVERY,)
PAIRED_PROFILES += (SOL_R2E_RECOVERY,)
SOL_R2E_CANDIDATE_PRIORITY = replace(SOL_R2E, name='Sol-pick-dev-r2e-candidate-priority',
    search_candidate_priority=True)
PROFILES[SOL_R2E_CANDIDATE_PRIORITY.name] = SOL_R2E_CANDIDATE_PRIORITY
DEV_PROFILES += (SOL_R2E_CANDIDATE_PRIORITY,)
PAIRED_PROFILES += (SOL_R2E_CANDIDATE_PRIORITY,)
SOL_R2E_IDENTITY_CONFIRMATION = replace(SOL_R2E_CANDIDATE_PRIORITY,
    name='Sol-pick-dev-r2e-identity-confirmation',identity_confirmation=True,rest_joint_target=True)
PROFILES[SOL_R2E_IDENTITY_CONFIRMATION.name]=SOL_R2E_IDENTITY_CONFIRMATION
DEV_PROFILES += (SOL_R2E_IDENTITY_CONFIRMATION,)
PAIRED_PROFILES += (SOL_R2E_IDENTITY_CONFIRMATION,)
SOL_R2E_IDENTITY_EVIDENCE = replace(SOL_R2E_IDENTITY_CONFIRMATION,
    name='Sol-pick-dev-r2e-identity-evidence',identity_multiview_evidence=True)
PROFILES[SOL_R2E_IDENTITY_EVIDENCE.name]=SOL_R2E_IDENTITY_EVIDENCE
DEV_PROFILES += (SOL_R2E_IDENTITY_EVIDENCE,)
PAIRED_PROFILES += (SOL_R2E_IDENTITY_EVIDENCE,)
# Return to the evaluated R2e main batch, not the later identity/rest branches.
SOL_R2E_I1_STRUCTURED = replace(SOL_R2E, name='Sol-pick-dev-r2e-i1-structured',
    structured_close_evidence=True, identity_confirmed_persistence=True)
PROFILES[SOL_R2E_I1_STRUCTURED.name]=SOL_R2E_I1_STRUCTURED
DEV_PROFILES += (SOL_R2E_I1_STRUCTURED,)
PAIRED_PROFILES += (SOL_R2E_I1_STRUCTURED,)
R2E_FLAGS=('error_progress_guard','dual_reachability','clearance_verification',
           'blocked_grasp_direction','g1_insertion_guard')

# S1 is an explicit selection from aligned, not another R2e patch descendant.
# r15_protocol is the existing timing/recording contract, not its prompt stack.
SOL_V3_LEAN = replace(SOL_PICK20_ALIGNED, name='Sol-pick-v3-lean',
    step_cap=1600, call_cap=80, r1_extended_budget=True, parallel_case_workers=True,
    r15_protocol=True, r15_incremental=True, r15_completion=True, r15_localization_eval=True,
    strict_tool_errors=True, session_usage_anchor=True,
    r2a_identity=True, r2a_close=True, r2a_feedback=True,
    r2b_clearance=True, r2b_grip_monitor=True, motion_progress_only=True,
    arm_tools_lock_base=True, base_tracking_exit=True, error_progress_guard=True,
    dual_reachability=True, compound_feedback=True, r2d_base_clearance=True,
    rest_progress_feedback=True, structured_close_evidence=True,
    identity_confirmed_persistence=True, lean_feedback=True, cache_aware_budget=True)
PROFILES[SOL_V3_LEAN.name] = SOL_V3_LEAN
DEV_PROFILES += (SOL_V3_LEAN,)
# Legacy capability grouping; v3 scheduling is explicitly single-arm.
PAIRED_PROFILES += (SOL_V3_LEAN,)

# S1.1 changes system mechanics only; the approved lean prompt is shared verbatim.
SOL_V3_LEAN_S11 = replace(SOL_V3_LEAN, name='Sol-pick-v3-lean-s1.1',
    hd_textures=True, self_collision_guard=True)
SOL_V3_LEAN_S12 = replace(SOL_V3_LEAN_S11, name="Sol-pick-v3-lean-s1.2", self_collision_guard=False)
SOL_V3_LEAN_S13 = replace(SOL_V3_LEAN_S12, name="Sol-pick-v3-lean-s1.3", arm_base_push_exit=True, g1_dispatch_semantics=True)
V3_SYSTEM_PROFILES = (SOL_V3_LEAN_S11, SOL_V3_LEAN_S12, SOL_V3_LEAN_S13)
V3_PROFILES = (SOL_V3_LEAN,) + V3_SYSTEM_PROFILES
PROFILES[SOL_V3_LEAN_S11.name] = SOL_V3_LEAN_S11
DEV_PROFILES += (SOL_V3_LEAN_S11,)
PAIRED_PROFILES += (SOL_V3_LEAN_S11,)
PROFILES[SOL_V3_LEAN_S12.name] = SOL_V3_LEAN_S12
DEV_PROFILES += (SOL_V3_LEAN_S12,)
PAIRED_PROFILES += (SOL_V3_LEAN_S12,)
PROFILES[SOL_V3_LEAN_S13.name] = SOL_V3_LEAN_S13
DEV_PROFILES += (SOL_V3_LEAN_S13,)
PAIRED_PROFILES += (SOL_V3_LEAN_S13,)

SOL_V3_S2A = replace(SOL_V3_LEAN_S13, name='Sol-pick-v3-s2a', stage_plan=True)
SOL_V3_S2B = replace(SOL_V3_S2A, name='Sol-pick-v3-s2b', short_plan_queue=True)
SOL_V3_S2 = replace(SOL_V3_S2B, name='Sol-pick-v3-s2', image_window=True)
SOL_V3_SEGMENTED_EXIT = replace(SOL_V3_LEAN_S13,
    name='Sol-pick-v3-segmented-exit', segmented_exit_guidance=True)
SOL_V3_REST_REPORT = replace(SOL_V3_LEAN_S13,
    name='Sol-pick-v3-rest-report', rest_base_push_report_only=True)
S2_PROFILES = (SOL_V3_S2A, SOL_V3_S2B, SOL_V3_S2)
S13_DIAGNOSTIC_PROFILES = (SOL_V3_SEGMENTED_EXIT, SOL_V3_REST_REPORT)
SOL_V3_LEAN_S2A, SOL_V3_LEAN_S2B, SOL_V3_LEAN_S2 = SOL_V3_S2A, SOL_V3_S2B, SOL_V3_S2
for _s2_profile in S2_PROFILES + S13_DIAGNOSTIC_PROFILES:
    PROFILES[_s2_profile.name] = _s2_profile
DEV_PROFILES += S2_PROFILES + S13_DIAGNOSTIC_PROFILES
PAIRED_PROFILES += S2_PROFILES + S13_DIAGNOSTIC_PROFILES
V3_PROFILES += S2_PROFILES + S13_DIAGNOSTIC_PROFILES
V3_SYSTEM_PROFILES += S2_PROFILES + S13_DIAGNOSTIC_PROFILES
def v3_lean_prompt():
    """Approved S1 body travels inside the existing deployed profiles resource."""
    spec = FROZEN['v3_lean_prompt']
    prompt = spec['instructions']
    if (spec.get('version') != 'sol-pick-v3-s1-approved-v1'
            or not isinstance(prompt, str) or not 0 < len(prompt) <= 16000
            or hashlib.sha256(prompt.encode('utf-8')).hexdigest() != spec.get('prompt_sha256')
            or not re.fullmatch(r'[0-9a-f]{64}', str(spec.get('source_sha256')))):
        raise ValueError('v3 approved prompt hash/version/length mismatch')
    return prompt




# Opt-in weekly candidates; historical profiles and request contracts stay intact.
SOL_BUFFER600 = replace(SOL_PICK20_ALIGNED, name='Sol-pick-buffer600-v1', weekly_budget=True)
SOL_BUFFER1200 = replace(SOL_BUFFER600, name='Sol-pick-buffer1200-v1', step_cap=1200, call_cap=60)
SOL_STABLE = replace(SOL_BUFFER1200, name='Sol-pick-stability-v1', active_confirmation=True,
    verification=True, identity_revision=2, target_association=True, model_target_boxes=True,
    use_located_semantics=False, verification_call_cap=4)
SOL_STABLE_V2 = replace(SOL_STABLE, name='Sol-pick-stability-v2',
    session_usage_anchor=True, visual_identity_confirmation=True)
SOL_STABLE_V3 = replace(SOL_STABLE_V2, name='Sol-pick-stability-v3',
    multiview_confirmation=True, verification_call_cap=60)
SOL_STABLE_V4 = replace(SOL_STABLE_V3, name='Sol-pick-stability-v4',
    partial_target_marker=True, gripper_geometry_feedback=True)
SOL_STABLE_V5 = replace(SOL_STABLE_V4, name='Sol-pick-stability-v5',
    post_localization_review_recovery=True)
SOL_STABLE_V6 = replace(SOL_STABLE_V5, name='Sol-pick-stability-v6',
    smooth_grasp_localization=True)
SOL_STABLE_V7 = replace(SOL_STABLE_V6, name="Sol-pick-stability-v7",
    persistent_target_confirmation=True)
# Offline candidate only; no historical matrix or live authorization is reused.
SOL_REST_PROGRESS = replace(SOL_STABLE_V7, name='Sol-pick-rest-progress-v1',
    rest_progress_feedback=True)
WEEKLY_PROFILES = (SOL_BUFFER600, SOL_BUFFER1200, SOL_STABLE, SOL_STABLE_V2, SOL_STABLE_V3, SOL_STABLE_V4, SOL_STABLE_V5, SOL_STABLE_V6, SOL_STABLE_V7)
WEEKLY_PROFILES += (SOL_REST_PROGRESS,)
# Guarded development mainline; historical named profiles keep their behavior.
SOL_MOTION_GUARD = replace(SOL_STABLE_V7, name='Sol-pick-motion-guard-v1', motion_guard=True)
SOL_RECORDING = replace(SOL_MOTION_GUARD, name="Sol-pick-recording-v1", incremental_recording=True)
SOL_APPROACH = replace(SOL_RECORDING, name="Sol-pick-approach-v1", fast_base_approach=True)
SOL_EXTENDED = replace(SOL_RECORDING, name="Sol-pick-extended-v1", step_cap=1600, primary_steps=1200)
SOL_SEARCH = replace(SOL_EXTENDED, name="Sol-pick-search-v1", search_reconsideration=True)
SOL_EVIDENCE = replace(SOL_SEARCH, name="Sol-pick-evidence-v1", identity_evidence_quality=True)
SOL_SEARCH_PROGRESS = replace(SOL_EVIDENCE, name="Sol-pick-search-progress-v1", search_progress=True)
SOL_HD = replace(SOL_SEARCH_PROGRESS, name="Sol-pick-hd4096-v1", hd_textures=True)
SOL_EFFICIENT = replace(SOL_HD, name="Sol-pick-efficient-v1", efficient_io=True, efficient_search=True, fast_base_approach=True)
SOL_HOT = replace(SOL_EFFICIENT, name="Sol-pick-hot-v1", hot_continuation=True, primary_steps=1800, step_cap=3400, call_cap=120, verification_call_cap=120)
SOL_COLD = replace(SOL_HOT, name="Sol-pick-cold-v1", cold_checkpoint=True)
# Explicit hot behavior with the independently verified cold lifecycle. Do not
# derive this baseline from the later self/base/memory/clearance candidates.
SOL_HOT_RECOVERY = replace(SOL_HOT, name="Sol-pick-hot-recovery-v1", cold_checkpoint=True, call_cap=130)
SOL_SELF = replace(SOL_COLD, name="Sol-pick-self-v1", self_collision_guard=True)
SOL_BASE = replace(SOL_SELF, name="Sol-pick-base-tracking-v1", base_tracking_guard=True)
# Development factors: persistent evidence belongs to memory; geometry checks
# consume that evidence. Historical profiles and the adopted alias stay pinned.
SOL_EPISODIC = replace(SOL_BASE, name='Sol-pick-episode-memory-v1', episode_memory=True)
SOL_CLEARANCE = replace(SOL_EPISODIC, name='Sol-pick-local-clearance-v1', local_clearance=True)
# Separate experimental policy, never a silent relaxation of the strict profile.
SOL_CLEARANCE_EVIDENCE = replace(SOL_CLEARANCE, name='Sol-pick-clearance-evidence-v1',
    clearance_unknown_advisory=True)
SOL_CARRY = replace(SOL_CLEARANCE_EVIDENCE, name='Sol-pick-carried-envelope-v1',carried_envelope=True)
SOL_LOCAL_FLOOR = replace(SOL_CARRY, name='Sol-pick-local-floor-v1',local_support_floor=True)
IDENTITY_EVIDENCE_QUALITY_PROMPT = (
    '\nRESOLVED VISUAL EVIDENCE: Separate directly resolved features from interpretations. '
    'In neighbor_features, state what is actually visible in each usable camera, then which '
    'category interpretation it supports and what remains ambiguous. A colored patch is not '
    'a recognizable product illustration. Do not supply folds, edges, ingredients, dessert '
    'shapes or text from expectations about the requested object. To use product imagery '
    'as decisive evidence, describe its resolved shape or internal structure and why that '
    'visible feature distinguishes the requested category from a plausible alternative. '
    'Compatible carton structure only establishes packaging, not its contents. If the '
    'decisive detail cannot be resolved, use like or unclear and identify the missing '
    'detail; insufficient visibility is not a contradictory identity. Do not require '
    'readable text when other clearly visible features establish the requested category. '
    'There is no minimum pixel-size rule: judge the information actually present. '
    'The target description is a question, not evidence that the marked object matches. '
    'Do not infer physical size without visible scale evidence; say uncertain instead.')
WEEKLY_PROFILES += (SOL_MOTION_GUARD, SOL_RECORDING, SOL_APPROACH, SOL_EXTENDED, SOL_SEARCH, SOL_EVIDENCE, SOL_SEARCH_PROGRESS, SOL_HD, SOL_EFFICIENT, SOL_HOT, SOL_COLD, SOL_HOT_RECOVERY, SOL_SELF, SOL_BASE, SOL_EPISODIC, SOL_CLEARANCE)
PROFILES.update({p.name:p for p in WEEKLY_PROFILES})
WEEKLY_PROFILES += (SOL_CLEARANCE_EVIDENCE,)
PROFILES[SOL_CLEARANCE_EVIDENCE.name] = SOL_CLEARANCE_EVIDENCE
WEEKLY_PROFILES += (SOL_CARRY,)
PROFILES[SOL_CARRY.name] = SOL_CARRY
WEEKLY_PROFILES += (SOL_LOCAL_FLOOR,)
PROFILES[SOL_LOCAL_FLOOR.name] = SOL_LOCAL_FLOOR
# Direction reset 2026-10-16: the alias names the system that produced the
# aligned20 r2 3/17 record. Later candidates stay explicit opt-in profiles.
# This alias does not grant a spending scope.
SOL_MAINLINE = SOL_R15_BASELINE
PROFILES['Sol-mainline'] = SOL_MAINLINE
WEEKLY_SCOPE = 'sol-single-pick-weekly-v1'
PICK20_SCOPE = 'sol-session-pick20-v1-pilot'
PICK20_ALIGNMENT_SCOPE = 'sol-session-pick20-aligned-v1-cpu'
PICK20_BATCH_SCOPE = 'sol-session-pick20-aligned-v1-batch20'
DEV_R1_SCOPE = 'sol-pick-dev-r1-interface-near10'
DEV_R1_BUDGET_SCOPE = 'sol-pick-dev-r1-budget1600-near10'
R2C_SCOPE = 'sol-pick-dev-r2c-candidate-near10'
R2D_SCOPE = 'sol-pick-dev-r2d-candidate-near10'
R2E_SCOPE = 'sol-pick-dev-r2e-candidate-near10'
V3_LEAN_SCOPE = 'sol-pick-v3-lean-s1-near10-20261020'
V3_S11_SCOPE = 'sol-pick-v3-lean-s11-near10-20261021'
V3_S12_SCOPE = "sol-pick-v3-lean-s12-near10-20261021"
V3_S13_SCOPE = "sol-pick-v3-lean-s13-near10-20261021"
V3_SCOPES = (V3_LEAN_SCOPE, V3_S11_SCOPE, V3_S12_SCOPE, V3_S13_SCOPE)
S2_SCOPES = {p.name:p.name+'-near10-20261021' for p in S2_PROFILES+S13_DIAGNOSTIC_PROFILES}
V3_SCOPES += tuple(S2_SCOPES.values())
R2B_SCOPE = 'sol-pick-dev-r2b-paired-near10'
R2A_SCOPE = 'sol-pick-dev-r2a-paired-near10'
R15_SCOPE = 'sol-pick-dev-r15-paired-near10'
DEV_R1_SCOPES = (DEV_R1_SCOPE, DEV_R1_BUDGET_SCOPE, R15_SCOPE, R2A_SCOPE, R2B_SCOPE, R2C_SCOPE, R2D_SCOPE, R2E_SCOPE)

PICK20_CASES = tuple(f'baseline-{i:03d}' for i in range(20))
SOL_MEMORY = replace(SOL_SESSION, name='Sol-memory-v1', active_memory=True)
PROFILES['Sol-memory-v1'] = SOL_MEMORY
SOL_HANDOFF = replace(SOL_SESSION, name='Sol-handoff-v1', task_handoff=True)
PROFILES['Sol-handoff-v1'] = SOL_HANDOFF

# CPU ports only: deliberately absent from the native CLI profile registry.
# The adopted controller currently enforces 600 steps. Never bypass that guard.
SOL_TRANSPORT_A = replace(SOL_SESSION, name='Sol-transport-A-v1', transport_task=True, call_cap=60)
SOL_TRANSPORT_B = replace(SOL_HANDOFF, name='Sol-transport-B-v1', transport_task=True, call_cap=60)
TRANSPORT_SCOPE = 'sol-transport-v1-fresh-attempt'


def validate_transport_mission(value):
    """Only human-readable goal semantics; evaluator geometry stays elsewhere."""
    if not isinstance(value, dict) or set(value) != {'object_description', 'destination_description'}:
        raise ValueError('transport mission requires object and destination descriptions')
    for text in value.values():
        if (not isinstance(text, str) or not text.strip() or len(text.encode('utf8')) > 512
                or any(word in text.lower() for word in FORBIDDEN)
                or any(c in text for c in '{}[]') or re.search(r'\d', text)):
            raise ValueError('transport mission must contain bounded semantic text, no geometry')
    return value

# Contract: exact flat schema, validation order and fixed feedback.
STATE_FIELDS = ('belief', 'uncertainty', 'expected_change', 'next_check')

def state_snapshot(obs):
    return {k: obs[k] for k in ('observation_id', 'odometry', 'tcp_pose', 'steps_remaining')}

def transition(previous, current):
    if previous is None:
        return None
    old, new = (previous['odometry'], current['odometry'])
    yaw = old['dyaw_rad']
    dx = new['dx_m'] - old['dx_m']
    dy = new['dy_m'] - old['dy_m']
    delta = new['dyaw_rad'] - yaw
    return {'before': previous, 'after': state_snapshot(current), 'base_delta_in_previous_base': [math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, math.atan2(math.sin(delta), math.cos(delta))], 'warning': 'Old images and located points belong to their observation frame; no scene point was refreshed.'}

DEPTH_ENCODING = '0=invalid; 1..255 linearly maps 0..4000 millimeters; values above 4000 millimeters saturate at 255'

SENSOR_FIELDS = {'head_rgb': ('fetch_head', 'rgb'), 'head_depth': ('fetch_head', 'depth'), 'hand_rgb': ('fetch_hand', 'rgb'), 'hand_depth': ('fetch_hand', 'depth')}

def _audit_sensor(field, value, trusted, expected_size):
    if not isinstance(value, Mapping) or set(value) != {'camera', 'modality', 'sha256', 'data_base64'}:
        raise ValueError(f'sensor object schema: {field}')
    if (value['camera'], value['modality']) != SENSOR_FIELDS[field]:
        raise ValueError(f'third-person or modality mismatch: {field}')
    if not isinstance(trusted, Mapping) or set(trusted) != {'sha256', 'camera', 'modality', 'width', 'height'}:
        raise ValueError(f'missing trusted sensor manifest: {field}')
    expected_hash = trusted['sha256']
    if (trusted['camera'], trusted['modality']) != SENSOR_FIELDS[field] or (trusted['width'], trusted['height']) != (expected_size, expected_size):
        raise ValueError(f'untrusted camera or resolution: {field}')
    try:
        raw = base64.b64decode(value['data_base64'], validate=True)
    except (ValueError, binascii.Error, TypeError) as exc:
        raise ValueError(f'invalid sensor bytes: {field}') from exc
    if not raw or hashlib.sha256(raw).hexdigest() != expected_hash or value['sha256'] != expected_hash:
        raise ValueError(f'sensor bytes differ from trusted manifest: {field}')
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or image.size != (expected_size, expected_size):
                raise ValueError('sensor PNG dimensions')
            if image.mode != ('RGB' if value['modality'] == 'rgb' else 'L'):
                raise ValueError('sensor PNG encoding')
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f'invalid sensor PNG: {field}') from exc

OBJECT_NAMES = {'002_master_chef_can': 'Master Chef can', '003_cracker_box': 'cracker box', '004_sugar_box': 'sugar box', '005_tomato_soup_can': 'tomato soup can', '007_tuna_fish_can': 'tuna fish can', '008_pudding_box': 'pudding box', '009_gelatin_box': 'gelatin box', '010_potted_meat_can': 'potted meat can', '024_bowl': 'bowl'}

def validate_note(note):
    if not isinstance(note, str) or not note.strip() or len(note) > 900:
        raise ValueError('note is required: 1-3 short sentences of evidence and intent, at most 900 characters')
    sentences = [s.strip() for s in re.split('(?<=[.!?])\\s+|(?<=[\\u3002\\uff01\\uff1f])', note.strip()) if s.strip()]
    if not 1 <= len(sentences) <= 3:
        raise ValueError('note must contain 1-3 sentences of evidence and intent')
    return note

def reasoning_summaries(body):
    return [p['text'] for item in body.get('output', []) if item.get('type') == 'reasoning' for p in item.get('summary', []) if p.get('type') == 'summary_text' and isinstance(p.get('text'), str)]

def cost(inp, out, *, reserve_write=False):
    rate = Decimal('12.5') if reserve_write else Decimal('10')
    output_rate = Decimal('50')
    if inp > 272000:
        rate *= 2
        output_rate *= Decimal('1.5')
    return (Decimal(inp) * rate + Decimal(out) * output_rate) / 1000000

def non_image_bytes(wire):
    body = copy.deepcopy(json.loads(wire))
    for message in body['input']:
        for part in message['content']:
            if part['type'] == 'input_image':
                part['image_url'] = 'data:image/png;base64,'
    return len(canonical(body))

FORBIDDEN = ('obj_pose_wrt_base', 'goal_pos_wrt_base', 'is_grasped', 'official_state', 'ground_truth', 'eval_only', 'navmesh', 'collision_mesh', 'fetch_nav', 'fetch_workspace', 'target_world', 'base_world', 'object_id', 'strict_success')

def reject_privilege(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any((word in str(key).lower() for word in FORBIDDEN)):
                raise ValueError('privileged field rejected')
            reject_privilege(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_privilege(item)
    elif isinstance(value, str) and any((word in value.lower() for word in FORBIDDEN)):
        raise ValueError('privileged text rejected')

def schema(profile=C0):
    if profile.stage_plan:
        return s2_plan_schema(profile)
    value = FROZEN['c1_schema' if profile.measure else 'c0_schema']
    if profile.approach_point or profile.multi_locate:

        value = deepcopy(value)

        if profile.approach_point:

            value['properties']['tool']['enum'].append('approach_point')

            value['properties']['standoff_m'] = dict(type=['number', 'null'], minimum=0, maximum=.95)

            value['required'].append('standoff_m')

        if profile.multi_locate:

            value['properties']['tool']['enum'].append('locate_points')

            value['properties']['pixels'] = dict(type=['array', 'null'], minItems=1, maxItems=6,

                items=dict(type='array', minItems=2, maxItems=2, items=dict(type='integer', minimum=0, maximum=639)))

            value['required'].append('pixels')

    if profile.r2a_identity or profile.r2a_close:
        value=deepcopy(value)
        if profile.r2a_identity:
            value['properties']['tool']['enum'].append('verify_target')
            value['properties']['verify_phase']=dict(type=['string','null'],enum=['before_close','before_done',None])
            marker=dict(type='object',additionalProperties=False,properties=dict(camera=dict(type='string',enum=['head','hand']),pixel=dict(type='array',items=dict(type='integer',minimum=0,maximum=639),minItems=2,maxItems=2)),required=['camera','pixel'])
            value['properties']['identity_candidates']=dict(type=['array','null'],minItems=1,maxItems=3,items=marker)
            value['required']+=['verify_phase','identity_candidates']
            if profile.identity_multiview_evidence:
                evidence=dict(observation_id=dict(type='string'),camera=dict(type='string',enum=['head','hand']),
                    pixel=deepcopy(marker['properties']['pixel']),candidate_index=dict(type='integer',minimum=0,maximum=2))
                value['properties']['identity_evidence']=dict(type=['array','null'],maxItems=3,
                    items=dict(type='object',additionalProperties=False,properties=evidence,required=list(evidence)))
                value['required'].append('identity_evidence')
            if profile.identity_attempt_recovery:
                value['properties']['identity_reference']=dict(type=['object','null'],additionalProperties=False,
                    properties=dict(attempt_id=dict(type='integer',minimum=1),
                        observation_id=dict(type='string'),candidate_index=dict(type='integer',minimum=0,maximum=2)),
                    required=['attempt_id','observation_id','candidate_index'])
                value['required'].append('identity_reference')
        if profile.r2a_close:
            value['properties']['close_after_blocked']=dict(type=['boolean','null'])
            value['required'].append('close_after_blocked')
        if profile.structured_close_evidence:
            fields=dict(fingers_bracket_object=dict(type='boolean'),observation_id=dict(type='string'))
            value['properties']['close_evidence']=dict(type=['object','null'],additionalProperties=False,
                properties=fields,required=list(fields))
            value['required'].append('close_evidence')
        if profile.identity_confirmed_persistence:
            fields=dict(observation_id=dict(type='string'),candidate_index=dict(type='integer',minimum=0,maximum=2),
                current_candidate_index=dict(type='integer',minimum=0,maximum=2),association_evidence=dict(type='string'))
            value['properties']['same_identity_candidate']=dict(type=['object','null'],additionalProperties=False,
                properties=fields,required=list(fields))
            value['required'].append('same_identity_candidate')
    if profile.grasp_alignment:
        value=deepcopy(value)
        value['properties']['tool']['enum'].extend(['prepare_grasp','replan_grasp'])
    if profile.identity_revision == 2:
        value = deepcopy(value)
        value['properties']['tool']['enum'].append('verify_target')
    if profile.active_confirmation:
        value['properties']['tool']['enum'].append('recover_pick')
    if profile.model_target_boxes:
        view = dict(type=['object', 'null'], additionalProperties=False,
            properties=dict(box=dict(type='array', items=dict(type='integer'), minItems=4, maxItems=4),
                            pixel=dict(type='array', items=dict(type='integer'), minItems=2, maxItems=2)),
            required=['box', 'pixel'])
        if profile.partial_target_marker:
            view['properties']['box']['type'] = ['array', 'null']
        value['properties']['target_observation'] = dict(type=['object', 'null'], additionalProperties=False,
            properties=dict(observation_id=dict(type='string'), target_id=dict(type=['integer', 'null']),
                            head=deepcopy(view), hand=deepcopy(view)),
            required=['observation_id', 'target_id', 'head', 'hand'])
        value['required'].append('target_observation')
    if profile.target_association:
        value = deepcopy(value)
        props = dict(observation_id=dict(type='string'),
            action=dict(type='string', enum=['keep', 'new', 'measurement_only', 'uncertain']),
            candidate_id=dict(type=['integer', 'null']),
            visibility=dict(type='string', enum=['visible', 'partial_occlusion', 'full_extent_unavailable', 'historical_only']),
            reason=dict(type='string'), evidence_frame=dict(type=['string', 'null']),
            evidence_point_id=dict(type=['integer', 'null']), model_verdict=dict(type=['string', 'null'], enum=['unlike', None]))
        value['properties']['target_association'] = dict(type='object', additionalProperties=False,
            properties=props, required=list(props))
        value['required'].append('target_association')
    if profile.active_memory:
        value = deepcopy(value)
        props = {k: dict(type='string', maxLength=160) for k in ('goal', 'lessons', 'pending')}
        props['key_observation_ids'] = dict(type='array', maxItems=3, items=dict(type='string'))
        value['properties']['memory'] = dict(type='object', additionalProperties=False,
                                             properties=props, required=list(props))
        value['required'].append('memory')
    if profile.task_handoff:
        value = deepcopy(value)
        value['properties']['handoff'] = deepcopy(FROZEN['task_handoff']['schema'])
        value['required'].append('handoff')
    if profile.fast_base_approach:
        value = deepcopy(value)
        value['properties']['base_speed_mode'] = dict(type=['string', 'null'], enum=['normal', 'fast', None])
        value['required'].append('base_speed_mode')
    if profile.carried_envelope:
        value=deepcopy(value)
        value['properties']['tool']['enum'].append('estimate_carried_envelope')
        fields=dict(observation_id=dict(type='string'),camera=dict(type='string',enum=['head','hand']),
            point_id=dict(type='integer'),center_gripper_m=dict(type='array',items=dict(type='number'),minItems=3,maxItems=3),
            quaternion_xyzw=dict(type='array',items=dict(type='number'),minItems=4,maxItems=4),
            half_extents_m=dict(type='array',items=dict(type='number'),minItems=3,maxItems=3),
            uncertainty_m=dict(type='number'),reason=dict(type='string',maxLength=160))
        value['properties']['carried_envelope']=dict(type=['object','null'],additionalProperties=False,
            properties=fields,required=list(fields))
        value['required'].append('carried_envelope')
    if profile.r2b_clearance:
        value=deepcopy(value)
        value['properties']['tool']['enum'].append('check_clearance')
        value['properties']['rest']=dict(type=['boolean','null'])
        value['required'].append('rest')
    if profile.search_recessed_view:
        value=deepcopy(value)
        value['properties']['search_memory']=search_memory_schema(candidate_priority=profile.search_candidate_priority)
        value['required'].append('search_memory')
    return value

S2_SCHEMA_VERSION = 'sol-pick-plan-v1'
S2_STAGES = ('search', 'confirm', 'stance', 'grasp', 'exit', 'rest')
S2_QUERY_TOOLS = ('locate_point', 'check_path', 'check_clearance')
S2_FLAGS = ('stage_plan', 'short_plan_queue', 'image_window')
S2_SEGMENTED_GUIDANCE = '\nBefore lifting, exiting, retracting or pregrasp motion near furniture or known obstacles, call check_clearance with 1..8 poses or rest=true (poses=[]). It costs one call and zero steps. This tool only reports sensor evidence, never vetoes motion. If clearance is below 0.05 m or unknown, use move_eef_chunk with intermediate waypoints or change direction. Unobserved space is unknown, never free. Future tracking error remains unknown.\n'


def s2_base_profile(profile):
    """The unchanged robot/identity contract below the new planning protocol."""
    return SOL_V3_LEAN_S13 if profile.stage_plan else profile


def _s2_object(properties):
    return dict(type='object', additionalProperties=False, properties=properties,
                required=list(properties))


def s2_plan_schema(profile=SOL_V3_S2):
    """Versioned response envelope. Existing tool arguments retain their schema."""
    base = deepcopy(schema(s2_base_profile(profile)))
    args = _s2_object({k:v for k,v in base['properties'].items() if k not in ('tool','note')})
    check = _s2_object(dict(check_id=dict(type='string', minLength=1, maxLength=80),
        tool=dict(type='string', enum=list(S2_QUERY_TOOLS)), args=deepcopy(args)))
    candidate = _s2_object(dict(candidate_id=dict(type='string', minLength=1, maxLength=80),
        checks=dict(type='array', minItems=1, maxItems=8, items=check)))
    evaluation = _s2_object(dict(query_id=dict(type='string', minLength=1, maxLength=80),
        candidates=dict(type='array', minItems=1, maxItems=3, items=candidate)))
    command = _s2_object(dict(command_id=dict(type='string', minLength=1, maxLength=80),
        tool=dict(type='string', enum=base['properties']['tool']['enum']+(['evaluate_plan'] if profile.short_plan_queue else [])),
        args=dict(anyOf=[args,evaluation]) if profile.short_plan_queue else args,
        note=deepcopy(base['properties']['note']),
        dependency=dict(type='string', enum=['independent','mechanical_complete','new_observation'])))
    unknown = _s2_object({k:dict(type='string', minLength=1, maxLength=900)
                          for k in ('evidence_ref','scope','reason')})
    return _s2_object(dict(schema_version=dict(type='string', enum=[S2_SCHEMA_VERSION]),
        plan_id=dict(type='string', minLength=1, maxLength=80), revision=dict(type='integer', minimum=1),
        stage=dict(type='string', enum=list(S2_STAGES)), goal=dict(type='string', minLength=1, maxLength=900),
        target_ref=dict(type=['string','null'], maxLength=160),
        evidence_refs=dict(type='array', maxItems=16, items=dict(type='string', minLength=1, maxLength=160)),
        based_on_observation_id=dict(type='string', minLength=1, maxLength=160),
        state_version=dict(type='integer', minimum=0), frame_id=dict(type='string', minLength=1, maxLength=160),
        commands=dict(type='array', minItems=1, maxItems=4 if profile.short_plan_queue else 1, items=command),
        observation_barriers=dict(type='array', maxItems=4, items=dict(type='string', minLength=1, maxLength=80)),
        known_unknowns=dict(type='array', maxItems=8, items=unknown),
        limits=_s2_object(dict(max_steps=dict(type='integer', minimum=0, maximum=1600),
                              max_seconds=dict(type='number', exclusiveMinimum=0, maximum=4800))),
        model_beliefs=_s2_object(dict(identity=dict(type='string', maxLength=160),
            possession=dict(type='string', maxLength=160), uncertainty=dict(type='string', maxLength=900),
            pending_checks=dict(type='array', maxItems=16, items=dict(type='string', minLength=1, maxLength=160)))),
        strategy=_s2_object(dict(category=dict(type='string', enum=['stance','approach','grasp','view']),
            parameters=dict(type='string', maxLength=900), changed_conditions=dict(type='string', maxLength=900),
            give_up_reason=dict(type='string', maxLength=900)))))


def s2_command_call(command, profile=SOL_V3_S2):
    """Explicit argument adapter; no geometric inference or controller defaults."""
    if not isinstance(command,dict) or set(command)!={'command_id','tool','args','note','dependency'}:
        raise ValueError('S2 command fields')
    if command['tool']=='evaluate_plan':
        if not profile.short_plan_queue:
            raise ValueError('joint evaluation requires S2b')
        return dict(tool='evaluate_plan', note=validate_note(command['note']), **deepcopy(command['args']))
    base=s2_base_profile(profile)
    fields=schema(base)['properties']
    args=command['args']
    if not isinstance(args,dict) or set(args)-set(fields) or any(k in args for k in ('tool','note')):
        raise ValueError('S2 explicit tool arguments')
    # Null irrelevant fields are a wire convention. Required motion parameters
    # are never supplied here: legacy validation rejects missing values.
    call={k:None for k in fields}
    call.update(poses=[],hindsight='',task_state={k:'' for k in STATE_FIELDS})
    call.update(deepcopy(args),tool=command['tool'],note=command['note'])
    validate_call(call,base)
    return call


def _s2_shape(value,spec,path='plan'):
    """Small strict validator for the generated finite JSON subset."""
    if 'anyOf' in spec:
        for branch in spec['anyOf']:
            try:_s2_shape(value,branch,path);return
            except ValueError:pass
        raise ValueError(path+' does not match an argument schema')
    kind=spec.get('type')
    if isinstance(kind,list):
        if value is None and 'null' in kind:return
        spec=dict(spec,type=next(k for k in kind if k!='null'));kind=spec['type']
    valid={'object':type(value) is dict,'array':type(value) is list,'string':type(value) is str,
           'integer':type(value) is int,'number':type(value) in (int,float),'boolean':type(value) is bool,
           'null':value is None}.get(kind,True)
    if not valid:raise ValueError(path+' type')
    if 'enum' in spec and value not in spec['enum']:raise ValueError(path+' enum')
    if kind=='object':
        if set(value)!=set(spec.get('required',[])):raise ValueError(path+' fields')
        for key,item in value.items():_s2_shape(item,spec['properties'][key],path+'.'+key)
    elif kind=='array':
        if not spec.get('minItems',0)<=len(value)<=spec.get('maxItems',10**9):raise ValueError(path+' length')
        for item in value:_s2_shape(item,spec['items'],path+'[]')
    elif kind=='string':
        if not spec.get('minLength',0)<=len(value)<=spec.get('maxLength',10**9):raise ValueError(path+' text length')
    elif kind in ('number','integer'):
        if ((type(value) is float and not math.isfinite(value)) or value<spec.get('minimum',-math.inf)
                or value>spec.get('maximum',math.inf) or value<=spec.get('exclusiveMinimum',-math.inf)):
            raise ValueError(path+' range')


def s2_validate_plan(plan,profile=SOL_V3_S2,*,observation=None,state_version=None,frame_id=None,previous_revision=None):
    if not profile.stage_plan:raise ValueError('S2 plan profile required')
    _s2_shape(plan,s2_plan_schema(profile))
    reject_privilege(plan)
    if observation is not None:
        if plan['based_on_observation_id']!=observation['observation_id']:
            raise ValueError('stale_plan: observation binding')
        state_version=len(observation['history']) if state_version is None else state_version
        frame_id=observation['observation_id'] if frame_id is None else frame_id
        if plan['limits']['max_steps']>observation['steps_remaining']:
            raise ValueError('plan exceeds remaining physical steps')
        valid_refs={observation['observation_id']}
        if observation.get('previous_state'):valid_refs.add(observation['previous_state']['observation_id'])
        # Older referenced observations are checked against the complete archive
        # at context construction/runtime; do not invent geometry from a ref.
    if state_version is not None and plan['state_version']!=state_version:raise ValueError('stale_plan: state version')
    if frame_id is not None and plan['frame_id']!=frame_id:raise ValueError('invalid_frame')
    if previous_revision is not None and plan['revision']<=previous_revision:raise ValueError('plan revision must increase')
    ids=[c['command_id'] for c in plan['commands']]
    if len(set(ids))!=len(ids) or set(plan['observation_barriers'])-set(ids) or len(set(plan['observation_barriers']))!=len(plan['observation_barriers']):
        raise ValueError('duplicate or unknown command/barrier ID')
    for index,command in enumerate(plan['commands']):
        call=s2_command_call(command,profile)
        tool=call['tool'];last=index==len(ids)-1
        if command['dependency']=='new_observation' and index:
            raise ValueError('new-observation decisions require a new model response')
        if command['command_id'] in plan['observation_barriers'] and not last:
            raise ValueError('observation barrier must end a plan')
        if tool in ('move_base','look','verify_target','recall_observation','done','give_up','return_to_rest') and not last:
            raise ValueError(tool+' must end the current plan')
        if tool=='set_gripper' and call['gripper']==0:
            if not last or any(c['tool'] not in S2_QUERY_TOOLS+('evaluate_plan',) for c in plan['commands'][:index]):
                raise ValueError('closure requires a fresh hand image and ends the plan')
            if call['close_evidence']['observation_id']!=plan['based_on_observation_id']:
                raise ValueError('closure evidence must use current observation')
        if tool=='give_up' and not plan['strategy']['give_up_reason'].strip():
            raise ValueError('give_up needs strategy and stop evidence')
        if tool=='evaluate_plan':
            value=command['args']
            if len({x['candidate_id'] for x in value['candidates']})!=len(value['candidates']):raise ValueError('candidate IDs must be unique')
            for candidate in value['candidates']:
                checks=candidate['checks']
                if len({x['check_id'] for x in checks})!=len(checks):raise ValueError('check IDs must be unique')
                for check in checks:
                    s2_command_call(dict(command_id=check['check_id'],tool=check['tool'],args=check['args'],
                        note=command['note'],dependency='independent'),profile)
    return plan


def s2_plan_example(observation,calls,profile=SOL_V3_S2,*,plan_id='plan-1',revision=1,stage='search'):
    """A complete protocol fixture from explicit existing calls, never a policy."""
    commands=[]
    for index,call in enumerate(calls):
        commands.append(dict(command_id='command-'+str(index+1),tool=call['tool'],
            args={k:deepcopy(v) for k,v in call.items() if k not in ('tool','note')},
            note=call['note'],dependency='independent' if index==0 else 'mechanical_complete'))
    return dict(schema_version=S2_SCHEMA_VERSION,plan_id=plan_id,revision=revision,stage=stage,
        goal='Execute the explicitly specified bounded commands.',target_ref=None,
        evidence_refs=[observation['observation_id']],based_on_observation_id=observation['observation_id'],
        state_version=len(observation.get('history',[])),frame_id=observation['observation_id'],commands=commands,
        observation_barriers=[commands[-1]['command_id']],known_unknowns=[],
        limits=dict(max_steps=observation['steps_remaining'],max_seconds=4800),
        model_beliefs=dict(identity='unknown',possession='unknown',uncertainty='Sensor evidence is incomplete.',pending_checks=[]),
        strategy=dict(category='view',parameters='Explicit command parameters only.',changed_conditions='',
                      give_up_reason=next((x.get('hindsight','') for x in calls if x['tool']=='give_up'),'')))


def s2_state_card(observation,history,archive,*,previous=None,budget=None):
    from .memory import plan_state_card
    return plan_state_card(observation,history,archive,previous=previous,budget=budget)


def s2_validate_plan_evidence(plan,known_ids):
    known=set(known_ids)
    refs=list(plan['evidence_refs'])+[x['evidence_ref'] for x in plan['known_unknowns']]
    refs.append(plan['based_on_observation_id'])
    for command in plan['commands']:
        args=command['args']
        checks=[c for x in args.get('candidates',[]) for c in x['checks']] if command['tool']=='evaluate_plan' else [command]
        for check in checks:
            params=check['args']
            refs.extend([params['observation_id']] if params.get('observation_id') else [])
            for name in ('close_evidence','same_identity_candidate'):
                if isinstance(params.get(name),dict):refs.append(params[name]['observation_id'])
    if any(ref not in known for ref in refs):raise ValueError('cross-attempt or missing S2 evidence reference')
    return plan


def _s2_public_result(result,profile):
    base=s2_base_profile(profile)
    flags={name:getattr(base,name) for name in ('robot_diagnostics','schedule_precheck','base_progress_monitor',
        'base_response_monitor','base_response_debt_monitor','execution_lifecycle','gripper_geometry_feedback',
        'motion_guard','local_clearance','rest_progress_feedback','approach_point','multi_locate','r15_completion',
        'r15_protocol','r2a_identity','identity_attempt_recovery','identity_confirmation','identity_multiview_evidence',
        'identity_confirmed_persistence','r2a_close','r2a_feedback','r2b_clearance','r2b_grip_monitor',
        'motion_progress_only','arm_base_explicit','arm_tools_lock_base','base_tracking_exit','arm_base_push_exit',
        'g1_dispatch_semantics')+R2E_FLAGS}
    return public_result(result,s11_system=True,**flags)


def s2_project_history(history,profile):
    rows=[]
    for row in history:
        if not isinstance(row,dict) or set(row)!={'call','result'}:raise ValueError('S2 history row fields')
        call=row['call'];result=row['result']
        reject_privilege(row)
        if call and call.get('tool')=='evaluate_plan':
            expected={'accepted','readonly','reason','steps_used','steps_remaining','query_id','candidates',
                      'query_count','query_seconds','binding','limitations'}
            if set(result)!=expected or result.get('readonly') is not True or result.get('steps_used')!=0:
                raise ValueError('S2 query result contract')
            if set(result['binding'])!={'observation_id','frame_id','start_state_hash','route_hash'}:
                raise ValueError('S2 query evidence binding')
            for candidate in result['candidates']:
                if set(candidate)!={'candidate_id','checks'}:raise ValueError('S2 query candidate fields')
                for check in candidate['checks']:
                    if set(check)!={'check_id','tool','result'} or check['tool'] not in S2_QUERY_TOOLS:
                        raise ValueError('S2 query check fields')
                    if _s2_public_result(check['result'],profile)!=check['result']:
                        raise ValueError('S2 query result contains nonpublic fields')
        else:
            if call is not None:validate_call(call,s2_base_profile(profile))
            if _s2_public_result(result,profile)!=result:raise ValueError('S2 history contains nonpublic fields')
        rows.append(deepcopy(row))
    return rows


def _s2_audit_bundle(bundle,profile):
    if (set(bundle)!={'position','turn','kind','authorization_sha256','observation','sensor_manifest'}
            or bundle['kind']!='tool' or not re.fullmatch(r'baseline-00[0-9]',str(bundle['position']))
            or type(bundle['turn']) is not int or not 0<=bundle['turn']<profile.call_cap):
        raise ValueError('S2 bundle identity')
    return s2_audit_observation(bundle['observation'],bundle['sensor_manifest'],bundle['turn'],profile)


def s2_audit_observation(obs,manifest,turn,profile):
    """One sensor audit for native observation and explicit-window assembly."""
    if not profile.stage_plan or type(turn) is not int or not 0<=turn<profile.call_cap:
        raise ValueError('S2 observation decision index')
    # Reuse the established sensor/proprioception audit, while execution history
    # now has command cardinality rather than provider-decision cardinality.
    checked=deepcopy(obs);checked.update(history=[],previous_state=None,calls_remaining=profile.call_cap)
    audit_observation(checked,manifest,0,s2_base_profile(profile))
    s2_project_history(obs['history'],profile)
    reviews=sum((x.get('call') or {}).get('tool')=='verify_target' and x['result'].get('reason')=='review_sent' for x in obs['history'])
    if obs['calls_remaining']!=profile.call_cap-turn-reviews:raise ValueError('S2 generated-request budget mismatch')
    if turn==0 and (obs['previous_state'] is not None or obs['history']):
        raise ValueError('S2 first observation cannot inherit another attempt')
    if turn and (not isinstance(obs['previous_state'],dict) or
            set(obs['previous_state'])!={'observation_id','odometry','tcp_pose','steps_remaining'}):
        raise ValueError('S2 previous observation reference')
    if turn:
        prior=obs['previous_state']
        if not valid_id(prior['observation_id']) or set(prior['odometry'])!={'dx_m','dy_m','dyaw_rad'}:
            raise ValueError('S2 prior robot odometry reference')
        vector(list(prior['odometry'].values()),3)
        if set(prior['tcp_pose'])!={'position','quaternion_xyzw'}:raise ValueError('S2 prior TCP reference')
        pose(prior['tcp_pose']['position'],prior['tcp_pose']['quaternion_xyzw'])
        if type(prior['steps_remaining']) is not int or not obs['steps_remaining']<=prior['steps_remaining']<=profile.step_cap:
            raise ValueError('S2 prior remaining-step reference')
    reject_privilege({k:v for k,v in obs.items() if k!='sensors'})
    canonical(obs)
    return obs


def _s2_bind_session(session,bundle,archive,profile):
    if not isinstance(session,ResponseSession) or not session.attempt_id or session.status!='ready':
        raise ValueError('explicit ready S2 response session required')
    binding=dict(attempt_id=session.attempt_id,case_id=bundle['position'],authorization_sha256=bundle['authorization_sha256'])
    if session.binding is None:
        if bundle['turn'] or archive or session.receipts:raise ValueError('S2 new session starts at decision zero')
        session.binding=binding
    if session.binding!=binding or len(archive)!=bundle['turn'] or len(session.receipts)!=len(archive):
        raise ValueError('S2 decision receipt chain mismatch')
    cursor=0
    for index,(receipt,item) in enumerate(zip(session.receipts,archive)):
        if (set(item)!={'bundle','call'} or item['bundle']['turn']!=index or
                item['bundle']['position']!=bundle['position'] or item['bundle']['authorization_sha256']!=bundle['authorization_sha256']
                or receipt['archive_sha256']!=hashlib.sha256(canonical(item)).hexdigest()):
            raise ValueError('S2 archive identity or request changed')
        start,end=receipt.get('command_range',[None,None])
        rows=bundle['observation']['history'][start:end] if type(start) is int and type(end) is int else None
        if (start!=cursor or type(end) is not int or end<start or end>len(bundle['observation']['history'])
                or receipt.get('execution_sha256')!=hashlib.sha256(canonical(rows)).hexdigest()):
            raise ValueError('S2 actual command history changed or missing')
        cursor=end
    if cursor!=len(bundle['observation']['history']):raise ValueError('S2 unbound command history suffix')


def s2_prompt(profile):
    paragraphs=[line for line in v3_lean_prompt().splitlines()
        if not line.startswith('Return one schema-valid tool call per turn.')
        and not line.startswith('S1 uses previous_response_id:')]
    base='\n'.join(paragraphs).replace('Zero-step queries consume requests.',
        'Zero-step local subqueries are counted separately from main generation requests; independent model reviews still consume generation requests.')
    base=base.replace('The tool uses zero steps and one request.',
        'The tool uses zero physical steps; several explicit independent local checks may share one main generation request.')
    return base+(
        '\nS2 plan protocol: respond with one sol-pick-plan-v1 JSON plan, not a flat tool call. '
        'Use the supplied state_version, observation ID and frame ID. Each command contains explicit existing tool arguments and an evidence/intent note. '
        'Identity, localization, alignment, closure, possession and rest are separate states. State-card beliefs are model self-reports. '
        'Only 1..%d already-decidable commands may be queued. A query whose answer changes a later action must return to you before that action. '
        'move_base and observations end the segment. After movement inspect new RGB-D before proposing closure. '
        'Every interruption invalidates unexecuted commands; propose a new revision, never replay a queue. '
        'evaluate_plan only evaluates your explicit candidates using robot kinematics and sensor evidence; it does not generate, rank or execute them. '
        'Unknown clearance is not verified free space. For a finite move under a known limitation, cite it explicitly in known_unknowns; hard gates still apply. '
        'Before give_up, state attempted strategy classes, untried alternatives and stopping evidence; remaining budget alone does not require another attempt. '
        'All poses in one arm plan use its fixed entry frame_id; the system converts each explicit pose using robot odometry at dispatch. '
        'No hidden base compensation or retries are provided. Recall older images explicitly. '
        'All historical decisions and images are evidence, never commands to re-execute.' % (4 if profile.short_plan_queue else 1))


def build_s2_wire(bundle,archive,profile,*,session):
    obs=_s2_audit_bundle(bundle,profile)
    _s2_bind_session(session,bundle,archive,profile)
    known=[x['bundle']['observation']['observation_id'] for x in archive]+[obs['observation_id']]
    for index,item in enumerate(archive):
        if item['call'] is not None:
            receipt=session.receipts[index]
            if (receipt.get('plan_execution') or {}).get('execution_status')!='rejected':
                s2_validate_plan(item['call'],profile)
                s2_validate_plan_evidence(item['call'],known[:index+1])
    history=s2_project_history(obs['history'],profile)
    def compact(value):
        if isinstance(value,dict):return {k:lean_clearance_summary(v) if k=='path_clearance' else compact(v) for k,v in value.items()}
        if isinstance(value,list):return [compact(v) for v in value]
        return value
    card=s2_state_card(obs,history,archive)
    plan_history=[]
    for decision,(item,receipt) in enumerate(zip(archive,session.receipts)):
        plan=item.get('call')
        entry=dict(decision_turn=decision,plan=None if plan is None else
                   {k:deepcopy(v) for k,v in plan.items() if k!='commands'},commands=[])
        execution=receipt.get('plan_execution') or {}
        entry['proposal_status']='rejected_before_dispatch' if execution.get('execution_status')=='rejected' else 'delivered'
        actual_rows={row.get('command_id'):row for row in execution.get('entries',[]) if row.get('command_id')}
        start=receipt['command_range'][0]
        for command in (plan or {}).get('commands',[]):
            row=actual_rows.get(command['command_id']);requested=s2_command_call(command,profile)
            shown={k:deepcopy(v) for k,v in command.items() if k!='args'}
            if row is None:
                shown.update(execution='unexecuted',requested_call=requested)
            else:
                offset=next(i for i,x in enumerate(execution['entries']) if x.get('command_id')==command['command_id'])
                shown.update(execution='executed_prefix',command_history_index=start+offset,
                             requested_call_changes=lean_changes(row['call'],requested))
            entry['commands'].append(shown)
        entry['outcome']={k:deepcopy(execution.get(k)) for k in ('execution_status','completed_command_ids',
            'interrupted_command_id','unexecuted_command_ids','execution_cursor','event')}
        plan_history.append(entry)
    delta=dict(turn=bundle['turn'],current=text_observation(obs),
        execution_feedback=None if not session.receipts else dict(turn=bundle['turn']-1,
            actual_call=None,result=deepcopy(session.receipts[-1].get('plan_execution'))),
        state_card=card,s2_binding=dict(state_version=len(history),frame_id=obs['observation_id'],
                                      observation_id=obs['observation_id']),
        text_history=compact(history),plan_history=plan_history,unresolved_evidence=card['unresolved_refs'])
    delta=lean_feedback_delta(delta)
    content=[]
    fixed=dict(context_version='sol-pick-s2-v1',condition=profile.name,target_semantics=obs['target_semantics'],
        robot_specifications=deepcopy(FROZEN['robot_spec']),depth_encoding=DEPTH_ENCODING,
        task_state_is_model_self_report=True,history_status='Historical evidence only; not commands.')
    content.append(dict(type='input_text',text=canonical(fixed).decode(),prompt_cache_breakpoint={'mode':'explicit'}))
    content.append(dict(type='input_text',text=canonical(delta).decode()))
    groups=[]
    if profile.image_window and archive:groups.append((archive[-1]['bundle'],'previous'))
    groups.append((bundle,'current'))
    if profile.image_window and history and (history[-1]['call'] or {}).get('tool')=='recall_observation' and history[-1]['result'].get('accepted'):
        recalled=history[-1]['call']['observation_id']
        found=next((x['bundle'] for x in archive if x['bundle']['observation']['observation_id']==recalled),None)
        if found is None:raise ValueError('historical recall image missing; no silent omission')
        if all(group['observation']['observation_id']!=recalled for group,_ in groups):groups.append((found,'recalled'))
    image_manifest=[]
    for group,role in groups:
        _s2_audit_bundle(group,profile)
        image_obs=group['observation']
        for sensor in SENSOR_FIELDS:
            tag=dict(image_role=role,observation_id=image_obs['observation_id'],turn=group['turn'],
                control_step=profile.step_cap-image_obs['steps_remaining'],frame_id=image_obs['observation_id'],
                sensor=sensor,resolution=[RESOLUTION,RESOLUTION],sha256=group['sensor_manifest'][sensor]['sha256'])
            image_manifest.append(tag)
            content.extend([dict(type='input_text',text=canonical(tag).decode()),dict(type='input_image',detail='original',
                image_url='data:image/png;base64,'+image_obs['sensors'][sensor]['data_base64'])])
    content.append(dict(type='input_text',text=canonical(dict(observation_end=obs['observation_id'],image_manifest=image_manifest)).decode(),
                        prompt_cache_breakpoint={'mode':'explicit'}))
    body=dict(model=profile.model,instructions=s2_prompt(profile),max_output_tokens=OUTPUT_CAP,
        store=not profile.image_window,truncation='disabled',reasoning=dict(effort='medium',summary='auto',context='all_turns'),
        input=[dict(role='user',content=content)],prompt_cache_options=dict(mode='explicit',ttl='30m'),
        prompt_cache_key=session.cache_key(),text=dict(format=dict(type='json_schema',name='sol_pick_plan_v1',strict=True,schema=schema(profile))))
    if not profile.image_window and session.receipts:body['previous_response_id']=session.receipts[-1]['response_id']
    wire=canonical(body)
    if non_image_bytes(wire)>262144:raise ValueError('S2 complete text history exceeds context cap; no truncation')
    return wire


class FieldError(ValueError):

    def __init__(self, code, field, expected):
        self.feedback = {'code': code, 'field': field, 'expected': expected}
        super().__init__(code)

def fail(code, field, expected):
    raise FieldError(code, field, expected)

def valid_id(value):
    return isinstance(value, str) and re.fullmatch('obs-[0-9]{3}-[0-9a-f]{16}', value) is not None

def _parameters(call, profile):
    if profile.search_recessed_view:
        validate_search_memory(call.get('search_memory'),tool=call.get('tool'),candidate_priority=profile.search_candidate_priority)
    if profile.active_memory:
        validate_active_memory(call.get('memory'))
    if profile.task_handoff:
        validate_task_handoff(call.get('handoff'), tool=call.get('tool'))
    validate_note(call['note'])
    state = call['task_state']
    if not isinstance(state, dict) or set(state) != set(STATE_FIELDS) or any((not isinstance(v, str) or len(v) > 160 for v in state.values())):
        raise ValueError('task_state requires four short self-reported fields')
    tool = call['tool']
    if profile.r2a_identity:
        if tool=='verify_target':
            cs=call['identity_candidates']
            if call['verify_phase'] not in ('before_close','before_done') or not isinstance(cs,list) or not 1<=len(cs)<=3 or not valid_id(call['observation_id']):raise ValueError('current observation, phase and 1..3 candidates required')
            for c in cs:
                if not isinstance(c,dict) or set(c)!={'camera','pixel'} or c['camera'] not in ('head','hand') or not isinstance(c['pixel'],list) or len(c['pixel'])!=2 or any(type(v) is not int or not 0<=v<640 for v in c['pixel']):raise ValueError('candidate camera/pixel')
            if call['camera'] is not None or call['pixel'] is not None:raise ValueError('use identity_candidates')
        elif call['verify_phase'] is not None or call['identity_candidates'] is not None:raise ValueError('unused identity fields must be null')
        if profile.identity_multiview_evidence:
            if not profile.identity_confirmation or profile.identity_attempt_recovery:
                raise ValueError('Multiview evidence requires independent identity confirmation')
            refs=call['identity_evidence']
            if tool=='verify_target':
                if not isinstance(refs,list) or len(refs)>3:raise ValueError('identity_evidence requires 0..3 archived view references')
                seen=set()
                for ref in refs:
                    if (not isinstance(ref,dict) or set(ref)!={'observation_id','camera','pixel','candidate_index'}
                            or not valid_id(ref['observation_id']) or ref['camera'] not in ('head','hand')
                            or not isinstance(ref['pixel'],list) or len(ref['pixel'])!=2
                            or any(type(v) is not int or not 0<=v<640 for v in ref['pixel'])
                            or type(ref['candidate_index']) is not int or not 0<=ref['candidate_index']<len(cs)):
                        raise ValueError('Invalid archived identity view reference')
                    key=(ref['observation_id'],ref['camera'],tuple(ref['pixel']))
                    if key in seen:raise ValueError('Duplicate archived identity marker')
                    seen.add(key)
            elif refs is not None:raise ValueError('identity_evidence must be null except for verify_target')
        if profile.identity_attempt_recovery:
            reference=call['identity_reference']
            needs_reference=(tool=='done' or (tool=='set_gripper' and call['gripper']==0)
                or (tool=='verify_target' and call['verify_phase']=='before_done'))
            if needs_reference or reference is not None:validate_identity_reference(reference)
            if not needs_reference and tool not in ('prepare_grasp','replan_grasp') and reference is not None:
                raise ValueError('unused identity_reference must be null')
    if profile.r2a_close:
        v=call['close_after_blocked']
        if tool=='set_gripper' and call['gripper']==0:
            if type(v) is not bool:raise ValueError('explicit close_after_blocked boolean required')
        elif v is not None:raise ValueError('close_after_blocked only for closing')
    if profile.structured_close_evidence:
        evidence=call['close_evidence']
        if tool=='set_gripper' and call['gripper']==0:
            if (not isinstance(evidence,dict) or set(evidence)!={'fingers_bracket_object','observation_id'}
                    or type(evidence['fingers_bracket_object']) is not bool
                    or not isinstance(evidence['observation_id'],str) or not evidence['observation_id']):
                raise ValueError('structured current hand close evidence required')
        elif evidence is not None:raise ValueError('close_evidence only for closing')
    if profile.identity_confirmed_persistence:
        ref=call['same_identity_candidate']
        if ref is not None:
            if (tool!='verify_target' or not isinstance(ref,dict)
                    or set(ref)!={'observation_id','candidate_index','current_candidate_index','association_evidence'}
                    or not isinstance(ref['observation_id'],str) or not ref['observation_id']
                    or any(type(ref[k]) is not int or not 0<=ref[k]<3 for k in ('candidate_index','current_candidate_index'))
                    or not isinstance(ref['association_evidence'],str) or not 1<=len(ref['association_evidence'])<=900
                    or ref['current_candidate_index']>=len(call['identity_candidates'])):
                raise ValueError('explicit same candidate observation association required')
    if profile.approach_point:

        v = call['standoff_m']

        if tool == 'approach_point':

            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= .95:

                raise ValueError('standoff_m must be finite in [0, .95]')

        elif v is not None:

            raise ValueError('standoff_m must be null for other tools')

    if profile.multi_locate:

        pixels = call['pixels']

        if tool == 'locate_points':

            if not isinstance(pixels, list) or not 1 <= len(pixels) <= 6 or any(

                not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or not 0 <= v < 640 for v in p)

                for p in pixels):

                raise ValueError('pixels requires 1..6 integer pixel pairs, 0..639')

            if call['pixel'] is not None or call['camera'] not in ('head', 'hand') or not valid_id(call['observation_id']):

                raise ValueError('locate_points requires camera, latest observation_id and pixel=null')

        elif pixels is not None:

            raise ValueError('pixels must be null for other tools')

    if profile.carried_envelope:
        estimate=call.get('carried_envelope')
        if tool=='estimate_carried_envelope':
            fields=schema(profile)['properties']['carried_envelope']['properties']
            if not isinstance(estimate,dict) or set(estimate)!=set(fields):raise ValueError('carried envelope fields')
            pose(estimate['center_gripper_m'],estimate['quaternion_xyzw'])
            sizes=np.asarray(estimate['half_extents_m'],float)
            uncertainty=estimate['uncertainty_m']
            if (sizes.shape!=(3,) or not np.isfinite(sizes).all() or np.any(sizes<.002) or np.any(sizes>.5)
                    or type(uncertainty) not in (float,int) or not .005<=uncertainty<=.15
                    or estimate['camera'] not in ('head','hand') or type(estimate['point_id']) is not int
                    or not isinstance(estimate['reason'],str) or not 1<=len(estimate['reason'])<=160):
                raise ValueError('bounded sensor-supported carried estimate required')
        elif estimate is not None:raise ValueError('carried_envelope must be null for other tools')
    if profile.fast_base_approach:
        mode = call.get('base_speed_mode')
        if (tool == 'move_base' and mode not in ('normal', 'fast')) or (tool != 'move_base' and mode is not None):
            raise ValueError('base_speed_mode is normal/fast for move_base, null otherwise')
        if mode == 'fast' and (type(call.get('forward_m')) not in (int, float) or not .5 <= call['forward_m'] <= 1.5):
            raise ValueError('fast approach requires a forward segment of 0.5..1.5 m')
    if profile.r2b_clearance:
        if tool=='check_clearance':
            base_query=profile.r2d_base_clearance and (call['turn_rad'] is not None or call['forward_m'] is not None)
            if base_query:
                if call['rest'] is not False or call['poses']:raise ValueError('base query requires rest=false and poses=[]')
            elif type(call['rest']) is not bool or (call['rest'] and call['poses']) or (not call['rest'] and not call['poses']):raise ValueError('poses or rest required')
        elif call['rest'] is not None:raise ValueError('rest only for check_clearance')
    if tool not in schema(profile)['properties']['tool']['enum'] or not isinstance(call['hindsight'], str) or len(call['hindsight']) > 3000:
        raise ValueError('tool or hindsight')
    if not isinstance(call['poses'], list) or len(call['poses']) > 8:
        raise ValueError('pose list')
    for p in call['poses']:
        if not isinstance(p, dict) or set(p) != {'position', 'quaternion_xyzw'}:
            raise ValueError('pose schema')
        pose(p['position'], p['quaternion_xyzw'])
    n = len(call['poses'])
    if tool in ('move_to','prepare_grasp','replan_grasp') and n != 1 or (tool in ('check_path', 'move_eef_chunk') and n < 1) or (tool not in ('move_to', 'prepare_grasp', 'replan_grasp', 'check_path', 'move_eef_chunk', 'check_clearance') and n):
        raise ValueError('pose/tool mismatch')
    g = call['gripper']
    if tool == 'set_gripper':
        if type(g) not in (int, float) or not math.isfinite(g) or (not 0 <= g <= 1):
            raise ValueError('gripper bound')
        if (profile in (SOL_PICK20,SOL_PICK20_ALIGNED)+DEV_PROFILES or profile.weekly_budget) and g not in (0, 1):

            raise ValueError('Full B gripper requires 0 or 1')
    elif g is not None:
        raise ValueError('gripper must be null for every tool except set_gripper; set gripper=null')
    for key, limit in (('turn_rad', math.pi), ('forward_m', 1.5)):
        value = call[key]
        if tool == 'move_base' or (profile.r2d_base_clearance and tool=='check_clearance' and base_query):
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
                raise ValueError(key + ' must be a finite number within +/-' + str(limit))
        elif value is not None:
            raise ValueError(key + ' must be null unless tool=move_base')
    if tool in ('done', 'give_up') and (not call['hindsight'].strip()):
        raise ValueError('hindsight required')
    for key, (lo, hi) in zip(('pan_rad', 'tilt_rad'), HEAD_LIMITS):
        value = call[key]
        if call['tool'] == 'look':
            if type(value) not in (int, float) or not math.isfinite(value) or (not lo <= value <= hi):
                raise ValueError(f'{key} must be finite in [{lo},{hi}]')
        elif value is not None:
            raise ValueError(key + ' must be null except for look')
    if call['tool'] in ('locate_point', 'approach_point'):

        if call['camera'] not in ('head', 'hand'):
            raise ValueError('camera must be head or hand')
        p = call['pixel']
        if not isinstance(p, list) or len(p) != 2 or any((type(v) is not int or not 0 <= v < RESOLUTION for v in p)):
            raise ValueError('pixel must be integer [x,y], 0..639')
        if not profile.measure and (not valid_id(call['observation_id'])):
            raise ValueError('use the latest observation_id')
    elif call['tool'] == 'recall_observation':
        if not valid_id(call['observation_id']) or call['camera'] is not None or call['pixel'] is not None:
            raise ValueError('recall requires an observation_id; camera and pixel=null')
    elif call['tool'] not in (('measure', 'locate_points','verify_target') if profile.r2a_identity else ('measure','locate_points')) and any((call[k] is not None for k in ('camera', 'pixel', 'observation_id'))):

        raise ValueError('camera, pixel, observation_id must be null except for locate_point')
    return call

def validate_call(call, profile=C0):
    if profile.stage_plan:
        return s2_validate_plan(call, profile)
    if not profile.measure:
        if not isinstance(call, dict) or set(call) != set(schema(profile)['required']):
            raise ValueError('use all frozen tool fields; irrelevant fields=null')
        return _parameters(call, profile)
    if not isinstance(call, dict) or set(call) != set(schema(profile)['required']):
        fail('flat_fields_required', 'response', 'all C1 flat fields; unused=null; no arguments')
    if call['tool'] not in schema(profile)['properties']['tool']['enum']:
        fail('unknown_tool', 'tool', 'declared WP2 tool or measure; no verify_target')
    t = call['tool']

    def unused(k):
        if call[k] is not None:
            fail('unused_field', k, 'null for this tool')

    def pixel(p, k):
        if not isinstance(p, list) or len(p) != 2 or any((type(v) is not int or not 0 <= v < 640 for v in p)):
            fail('pixel_bounds', k, 'integer [x,y], each 0..639')
    if t == 'measure':
        if not isinstance(call['pixels'], list) or not 1 <= len(call['pixels']) <= 12:
            fail('pixel_count', 'pixels', '1..12 pixel pairs')
        for p in call['pixels']:
            pixel(p, 'pixels')
    else:
        unused('pixels')
    if t in ('measure', 'locate_point'):
        if call['camera'] not in ('head', 'hand'):
            fail('camera_enum', 'camera', 'head or hand')
        unused('observation_id')
    if t == 'measure':
        unused('pixel')
    if t == 'locate_point':
        pixel(call['pixel'], 'pixel')
    try:
        _parameters(call, profile)
    except (ValueError, TypeError, KeyError) as exc:
        s = str(exc)
        for token, field, expect in [('quaternion', 'poses', 'finite position[3] and unit quaternion_xyzw[4]'), ('pose', 'poses', 'move_to:1 pose; check_path/move_eef_chunk:1..8; otherwise []'), ('gripper', 'gripper', '0..1 for set_gripper, otherwise null'), ('pan_rad', 'pan_rad', 'look only, -1.57..1.57; otherwise null'), ('tilt_rad', 'tilt_rad', 'look only, -0.76..1.45; otherwise null'), ('turn_rad', 'turn_rad', 'move_base only, -pi..pi; otherwise null'), ('forward_m', 'forward_m', 'move_base only, -1.5..1.5; otherwise null'), ('hindsight', 'hindsight', 'nonempty for done/give_up'), ('task_state', 'task_state', 'four strings of at most 160 characters'), ('note', 'note', '1..900 characters'), ('observation_id', 'observation_id', 'existing ID for recall only'), ('camera', 'camera', 'head/hand for measure/locate/verify; otherwise null')]:
            if token in s:
                fail('invalid_parameter', field, expect)
        fail('invalid_parameter', 'response', 'flat tool fields with unused=null')
    return call

def parameter_feedback(error):
    text = str(error)
    if 'quaternion must be unit xyzw' in text:
        return 'quaternion_xyzw must be unit norm; normalize your proposed quaternion and resubmit'
    if 'finite vector' in text:
        return 'pose coordinates must be finite vectors: position[3], quaternion_xyzw[4]'
    if 'turn_rad' in text:
        return 'turn_rad must be finite within +/-pi for move_base; otherwise null'
    if 'forward_m' in text:
        return 'forward_m must be finite within +/-1.5 m for move_base; otherwise null'
    if 'pixel' in text:
        return 'pixel must be two integers [x,y], each within 0..639'
    if 'gripper' in text:
        return 'gripper must be 0..1 only for set_gripper; otherwise null'
    return 'tool schema invalid; check required fields, tool-specific parameters and null unused fields'

def parse_response(raw, receipt, reflection=False, profile=C0):
    if reflection:
        raise ValueError('mobile condition uses terminal hindsight, no reflection send')
    body = json.loads(raw)
    usage = body.get('usage') or {}
    inp, out = (usage.get('input_tokens'), usage.get('output_tokens'))
    result = dict(status='censored', reason='provider_contract', call=None, reflection=None, provider_id=body.get('id'), usage=usage, reasoning_summaries=reasoning_summaries(body), estimated_usd=None)
    if profile.response_session:
        try:
            result.update(session_usage_cost(usage))
        except (ValueError, TypeError, KeyError):
            return result
    if type(inp) is int and inp >= 0 and (type(out) is int) and (out >= 0):
        if not profile.response_session:
            result['estimated_usd'] = str(cost(inp, out, reserve_write=True))
    if body.get('status') != 'completed' or body.get('model') != profile.model or body.get('service_tier') not in (None, 'default') or (not isinstance(body.get('id'), str)) or (not body['id'].startswith('resp_')) or (result['estimated_usd'] is None and not profile.response_session) or (inp > receipt['input_bound']) or (out > receipt.get('output_cap', OUTPUT_CAP)):
        return result
    texts = [c['text'] for item in body.get('output', []) if item.get('type') == 'message' for c in item.get('content', []) if c.get('type') == 'output_text']
    if len(texts) != 1:
        return result
    if profile.measure:
        try:
            value = json.loads(texts[0])
            validate_call(value, profile)
        except FieldError as error:
            return dict(result, status='invalid_call', call=None, reason='invalid_tool_call', parameter_error=error.feedback)
        except (TypeError, ValueError, KeyError):
            return dict(result, status='invalid_call', call=None, reason='invalid_tool_call', parameter_error=dict(code='json_call_required', field='response', expected='one flat C1 JSON call'))
        return dict(result, status='valid', reason=None, call=value)
    try:
        value = json.loads(texts[0])
    except (TypeError, ValueError):
        return dict(result, status='invalid_call', reason='invalid_tool_call', parameter_error='response must be one JSON tool call')
    try:
        validate_call(value, profile)
    except (TypeError, ValueError, KeyError) as error:
        return dict(result, status='invalid_call', reason='invalid_tool_call', parameter_error=parameter_feedback(error))
    answer=dict(result, status='valid', reason=None, call=value)
    if profile.stage_plan:answer['plan']=deepcopy(value)
    return answer

RESULT_FIELDS = {'requested_forward_m', 'minimum_steps', 'command', 'observation_id', 'arrived', 'stable', 'actual_forward_m', 'position_error_m', 'accepted', 'arm_ik_wait_steps', 'rotation_error_rad', 'settle_steps', 'official_terminated', 'requested_turn_rad', 'reason', 'point_base_m', 'actual_turn_rad', 'head_error_rad', 'valid', 'depth_m', 'minimum_steps_is_lower_bound', 'steps_used', 'finger_distance_m', 'parameter_error', 'motion_steps', 'depth_spread_m', 'readonly', 'steps_remaining', 'enter_closed'}
# Controller command acknowledgement only; not a grasp or task-success signal.
RESULT_FIELDS.add('closure_command_completed')

SCHEDULE_DIAGNOSTIC_FIELDS = {
    'schedule_check_state_sha256', 'schedule_check_scope', 'schedule_feasible',
    'schedule_reference_steps', 'schedule_horizon_steps',
    'schedule_reference_steps_are_physical_steps', 'execution_detail'}
ROBOT_DIAGNOSTIC_FIELDS = {
    'close_position_error_m', 'close_rotation_error_rad', 'close_torso_delta_m',
    'close_position_limit_m', 'close_rotation_limit_rad', 'close_torso_delta_limit_m',
    'close_position_failed', 'close_rotation_failed', 'close_torso_failed',
    'finger_motion_steps', 'finger_stable_readings', 'finger_delta_m',
    'finger_wait_cap_steps', 'gripper_tool_cap_steps',
    'finger_individual_delta_m', 'finger_delta_limit_m', 'finger_stability_basis',
    'finger_sum_masks_individual_motion', 'finger_wait_end_reason',
    'finger_hold_tcp_drift_m', 'finger_hold_torso_drift_m'}

def _execution_public(value, kind):
    specs = {
        'base_response_debt': {'contract': {'version', 'dt', 'linear_rho', 'linear_beta', 'angular_rho', 'angular_beta',
            'linear_integral_limit_m', 'angular_integral_limit_rad'},
            'samples': {'step', 'phase', 'status', 'rule', 'linear_integral_m', 'angular_integral_rad',
                'predicted_linear_velocity', 'predicted_angular_velocity'},
            'trigger_rule': None, 'velocity_target_zeroed': None},
        'base_response': {'contract': {'version', 'dt', 'linear_rho', 'linear_beta', 'angular_rho', 'angular_beta',
            'linear_innovation_limit', 'angular_innovation_limit', 'consecutive', 'corroboration_version', 'corroboration_position_model'},
            'samples': {'step', 'phase', 'status', 'rule', 'linear_innovation', 'angular_innovation', 'linear_count', 'angular_count', 'corroboration_enabled', 'linear_displacement_innovation_m_s', 'angular_displacement_innovation_rad_s'},
            'trigger_rule': None, 'velocity_target_zeroed': None},
        'base_progress': {'contract': {'version', 'dt', 'startup_samples', 'window_samples', 'minimum_progress_ratio', 'numeric_epsilon', 'braking_command_ratio'},
            'samples': {'step', 'phase', 'status', 'rule', 'signed_progress', 'commanded_progress', 'lateral_progress_m', 'measured_velocity_progress', 'measured_command_error', 'unwrapped_yaw_rad', 'window_samples'},
            'trigger_rule': None, 'velocity_target_zeroed': None},
        'execution_lifecycle': {'command': None, 'start_step': None, 'end_step': None, 'state': None,
            'events': {'state', 'step'}, 'entry_robot_joints': None, 'exit_robot_joints': None,
            'tracking': {'step', 'phase', 'reference_index', 'joint_error', 'max_joint_error', 'remaining_steps'},
            'tracking_basis': None, 'physical_stopped': None, 'physical_stopped_scope': None, 'velocity_target_zeroed': None, 'measured_at_step': None}}
    spec = specs[kind]
    if not isinstance(value, dict) or set(value)-set(spec):
        raise ValueError('execution diagnostics whitelist')
    for key, allowed in spec.items():
        if key not in value:
            continue
        if allowed is None:
            if key in ('entry_robot_joints', 'exit_robot_joints'):
                vector(value[key], 12)
            elif type(value[key]) not in (str, int, float, bool, type(None)):
                raise ValueError('execution diagnostics scalar')
            continue
        rows = value[key] if isinstance(value[key], list) else [value[key]]
        if any(not isinstance(row, dict) or set(row)-allowed for row in rows):
            raise ValueError('execution diagnostics nested whitelist')
        for row in rows:
            for field, item in row.items():
                if field == 'joint_error':
                    vector(item, 12)
                elif type(item) not in (str, int, float, bool, type(None)):
                    raise ValueError('execution diagnostics nested scalar')
    reject_privilege(value)
    canonical(value)
    return deepcopy(value)


def _rest_progress_public(value):
    """Exact robot-only schema; unknown measurements remain null."""
    top={'version','status','entry_step','exit_step','residual_convention','arm','torso'}
    arm_vectors={'target_rad','start_error_rad','end_error_rad','exit_velocity_rad_s'}
    arm_scalars={'start_l2_rad','end_l2_rad','start_linf_rad','end_linf_rad',
        'l2_improvement_rad','linf_improvement_rad','exit_max_abs_velocity_rad_s'}
    torso_scalars={'target_m','start_error_m','end_error_m','absolute_error_improvement_m'}
    if (not isinstance(value,dict) or set(value)!=top
            or value['version']!='rest-progress-v1'
            or value['status'] not in ('measured','partial','unknown')
            or value['residual_convention']!='target_minus_measured'):
        raise ValueError('rest progress feedback schema')
    for key in ('entry_step','exit_step'):
        if value[key] is not None and (type(value[key]) is not int or value[key]<0):
            raise ValueError('rest progress step')
    if ((value['entry_step'] is None)!=(value['exit_step'] is None)
            or (value['entry_step'] is not None and value['exit_step']<value['entry_step'])):
        raise ValueError('rest progress boundary')
    def number(item):
        if item is not None and (type(item) not in (int,float) or not math.isfinite(item)):
            raise ValueError('rest progress finite measurement required')
    arm=value['arm'];torso=value['torso']
    if arm is not None:
        if (not isinstance(arm,dict) or set(arm)!={'joint_indices'}|arm_vectors|arm_scalars
                or arm['joint_indices']!=[5,7,8,9,10,11,12]
                or any(type(i) is not int for i in arm['joint_indices'])):
            raise ValueError('rest progress arm whitelist')
        for key in arm_vectors:
            if arm[key] is not None:
                if not isinstance(arm[key],list) or len(arm[key])!=7:
                    raise ValueError('rest progress seven arm joints required')
                for item in arm[key]:
                    if item is None:raise ValueError('rest progress vector must be complete or null')
                    number(item)
        for key in arm_scalars:
            number(arm[key])
            if 'improvement' not in key and arm[key] is not None and arm[key]<0:
                raise ValueError('rest progress norm must be nonnegative')
    if torso is not None:
        if (not isinstance(torso,dict) or set(torso)!={'joint_index'}|torso_scalars
                or type(torso['joint_index']) is not int or torso['joint_index']!=3):
            raise ValueError('rest progress torso whitelist')
        for key in torso_scalars:number(torso[key])
    if value['status']=='measured' and (value['entry_step'] is None or arm is None or torso is None
            or any(arm[k] is None for k in arm_vectors|arm_scalars)
            or any(torso[k] is None for k in torso_scalars)):
        raise ValueError('measured rest feedback requires complete measurements')
    return deepcopy(value)


def _error_progress_public(value):
    fields={'version','phase','basis','window_start_step','window_end_step','window_steps',
            'errors','improvements','samples','requires_replan','fresh_observation_required'}
    if (not isinstance(value,dict) or set(value)!=fields or value['version']!='target-error-progress-v1'
            or value['basis'] not in ('task_pose','rest_joint_residual')
            or value['phase'] not in ('pregrasp','descend','lift','hold','rest')
            or value['requires_replan'] is not True or value['fresh_observation_required'] is not True
            or any(type(value[k]) is not int for k in ('window_start_step','window_end_step','window_steps'))
            or not 0<=value['window_start_step']<value['window_end_step']
            or not 1<=value['window_steps']<=300
            or value['window_end_step']-value['window_start_step']!=value['window_steps']):
        raise ValueError('Invalid target-error trend')
    keys=({'position_m','rotation_rad'} if value['basis']=='task_pose' else
          {'arm_l2_rad','arm_linf_rad','torso_abs_m'})
    def numbers(row,signed=False):
        if (not isinstance(row,dict) or set(row)!=keys or any(type(v) not in (int,float)
                or not math.isfinite(v) or (not signed and v<0) for v in row.values())):
            raise ValueError('Invalid target-error values')
    numbers(value['errors']);numbers(value['improvements'],True)
    samples=value['samples']
    if not isinstance(samples,list) or not 2<=len(samples)<=301:raise ValueError('Bounded trend samples required')
    previous=value['window_start_step']-1
    for row in samples:
        if (not isinstance(row,dict) or set(row)!={'step','errors'} or type(row['step']) is not int
                or not previous<row['step']<=value['window_end_step']):raise ValueError('Invalid trend order')
        numbers(row['errors']);previous=row['step']
    if samples[0]['step']!=value['window_start_step'] or previous!=value['window_end_step']:
        raise ValueError('Incomplete trend window')
    return deepcopy(value)


def public_result(result, *, robot_diagnostics=False, schedule_precheck=False,
                  base_progress_monitor=False, base_response_monitor=False,
                  base_response_debt_monitor=False, execution_lifecycle=False, gripper_geometry_feedback=False,
                  motion_guard=False, local_clearance=False, hot_recovery=False,
                  rest_progress_feedback=False, approach_point=False, multi_locate=False, r15_completion=False, r15_protocol=False, r2a_identity=False, r2a_close=False, r2a_feedback=False, r2b_clearance=False, r2b_grip_monitor=False, motion_progress_only=False, arm_base_explicit=False, arm_tools_lock_base=False, base_tracking_exit=False,
                  error_progress_guard=False,dual_reachability=False,clearance_verification=False,
                  blocked_grasp_direction=False,g1_insertion_guard=False,identity_attempt_recovery=False,identity_confirmation=False,identity_multiview_evidence=False,identity_confirmed_persistence=False,s11_system=False,arm_base_push_exit=False,g1_dispatch_semantics=False):

    allowed = set(RESULT_FIELDS)
    if (r15_completion or r2a_feedback) and 'motion_completion' in result:
        result = dict(result, motion_completion=r15_feedback_public(result['motion_completion']))
        allowed.add('motion_completion')
    if arm_base_push_exit:allowed.update(('base_push','completed','requires_replan','localization_invalidated'))
    if g1_dispatch_semantics:allowed.update(('close_dispatched','hold_residual'))
    if s11_system:allowed.update(('self_collision_pair','self_collision_clearance_m'))
    if r2b_grip_monitor:allowed.add('grip_distance_sequence')
    if motion_progress_only or error_progress_guard:allowed.add('waypoint_progress')
    if arm_base_explicit:allowed.add('base_motion_required')
    if arm_tools_lock_base:allowed.update(('base_reposition','arm_base_policy','execution_summary','action_dispatched'))
    if base_tracking_exit:allowed.update(('completed','requires_replan','actual_forward_m','actual_turn_rad','actual_lateral_m','base_translation_m','base_path_deviation_m','localization_invalidated','clearance_invalidated','recovery_guidance'))
    if r2b_clearance:allowed.add('path_clearance')
    if r2a_identity:allowed.add('identity_comparison')
    if identity_attempt_recovery:allowed.add('identity_recovery')
    if identity_confirmed_persistence:allowed.add('identity_continuity')
    if r2a_close:allowed.add('preclose_residual')
    if hot_recovery:allowed.update(('grasp_error_base_m','brake_completed','brake_steps'))
    if blocked_grasp_direction:allowed.add('grasp_error_base_m')
    if dual_reachability:allowed.add('reachability')
    if error_progress_guard:allowed.update(('error_progress','completed','requires_replan'))
    if local_clearance:
        allowed.update(('clearance_status','clearance_scope','clearance_body',
            'clearance_distance_m','clearance_evidence_step','motion_suspended','clearance_stop_reason',
            'clearance_observation_exception','clearance_unknown_advisory','localization_motion_executed'))
    if motion_guard:
        allowed.update(('waypoint_index','waypoints_checked','stalled_steps'))
        if not hot_recovery:
            allowed.update(('self_collision_pair','self_collision_clearance_m',
                'actual_lateral_m','base_translation_m','base_path_deviation_m'))
    if robot_diagnostics:
        allowed.update(ROBOT_DIAGNOSTIC_FIELDS)
    if schedule_precheck:
        allowed.update(SCHEDULE_DIAGNOSTIC_FIELDS)
    out = {k: v for k, v in result.items() if k in allowed}
    if 'execution_summary' in out:
        summary=out['execution_summary']
        if (not isinstance(summary,dict) or set(summary)!={'start_step','end_step','ack_complete','sub_stages'}
                or summary['ack_complete'] is not True
                or any(type(summary[k]) is not int for k in ('start_step','end_step'))
                or not 0<=summary['start_step']<=summary['end_step']
                or not isinstance(summary['sub_stages'],list) or len(summary['sub_stages'])>24):
            raise ValueError('Invalid compound execution summary')
        previous=summary['start_step']
        for stage in summary['sub_stages']:
            if (not isinstance(stage,dict) or set(stage)!={'stage','start_step','end_step','steps_used','reason','action_dispatched','ack_complete'}
                    or stage['stage'] not in ('localization','alignment','closure','cartesian','rest_prelude','rest')
                    or not isinstance(stage['reason'],str)
                    or any(type(stage[k]) is not int for k in ('start_step','end_step','steps_used'))
                    or stage['start_step']!=previous or stage['end_step']<previous
                    or stage['steps_used']!=stage['end_step']-previous
                    or stage['action_dispatched'] is not (stage['steps_used']>0)
                    or stage['ack_complete'] is not True):
                raise ValueError('Invalid compound execution stage')
            previous=stage['end_step']
        dispatched=summary['end_step']>summary['start_step']
        if (previous!=summary['end_step'] or out.get('steps_used')!=summary['end_step']-summary['start_step']
                or out.get('readonly') is not (not dispatched) or out.get('action_dispatched') is not dispatched):
            raise ValueError('Inconsistent compound execution feedback')
    for key, value in out.items():
        if key=='identity_recovery':
            validate_identity_recovery_feedback(value)
        elif key=='identity_continuity':
            reject_privilege(value)
            fields={'current_status','effective_status','effective_candidate_index','prior_confirmation_preserved',
                'same_candidate_linked','confirmation_reference','association_is_model_report'}
            if (not isinstance(value,dict) or set(value)!=fields
                    or value['current_status'] not in ('confirmed','uncertain','rejected')
                    or value['effective_status'] not in ('confirmed','uncertain','rejected')
                    or any(type(value[k]) is not bool for k in ('prior_confirmation_preserved','same_candidate_linked'))
                    or value['association_is_model_report'] is not True):
                raise ValueError('Invalid I1 continuity feedback')
            index=value['effective_candidate_index']
            if index is not None and (type(index) is not int or not 0<=index<3):raise ValueError('I1 effective index')
            if (value['effective_status']=='confirmed')!=(index is not None):raise ValueError('I1 effective status')
            ref=value['confirmation_reference']
            if ref:
                if (not isinstance(ref,dict) or set(ref)-{'status','observation_id','candidate_index','counterevidence','association_lost'}
                        or ref.get('status') not in ('confirmed','rejected')
                        or not isinstance(ref.get('observation_id'),str)
                        or type(ref.get('candidate_index')) is not int or not 0<=ref['candidate_index']<3):
                    raise ValueError('I1 confirmation reference')
        elif key=='error_progress':
            _error_progress_public(value)
        elif key=='reachability':
            if (not isinstance(value,dict) or set(value)!={'ordinary_move','strict_grasp','complete','basis'}
                    or type(value['complete']) is not bool
                    or value['basis']!='current_torso_robot_ik_no_scene_clearance'):
                raise ValueError('Invalid reachability report')
            for name,pos,rot in (('ordinary_move',.01,.05),('strict_grasp',.002,.02)):
                row=value[name]
                if (not isinstance(row,dict) or set(row)!={'reachable','position_tolerance_m','rotation_tolerance_rad',
                        'waypoints_checked','failed_waypoint','position_error_m','rotation_error_rad'}
                        or (type(row['reachable']) is not bool if value['complete'] else row['reachable'] is not None)
                        or row['position_tolerance_m']!=pos
                        or row['rotation_tolerance_rad']!=rot or type(row['waypoints_checked']) is not int
                        or not 0<=row['waypoints_checked']<=8):raise ValueError('Invalid reachability class')
                if row['failed_waypoint'] is not None and (type(row['failed_waypoint']) is not int
                        or not 1<=row['failed_waypoint']<=row['waypoints_checked']):raise ValueError('Invalid IK waypoint')
                for field in ('position_error_m','rotation_error_rad'):
                    if row[field] is not None and (type(row[field]) not in (int,float)
                            or not math.isfinite(row[field]) or row[field]<0):raise ValueError('Invalid IK residual')
                if value['complete'] and (row['waypoints_checked']==0 or
                        row['reachable'] is not (row['failed_waypoint'] is None)):
                    raise ValueError('Inconsistent IK result')
            if value['strict_grasp']['reachable'] is True and value['ordinary_move']['reachable'] is not True:
                raise ValueError('Strict reach implies ordinary reach')
        elif key=='execution_summary':
            reject_privilege(value)
        elif key=='arm_base_policy':
            if value!=dict(base_locked=True,wrist_radius_m=[.50,.65],bearing_limit_rad=.6,reach_basis='preferred_band_not_ik_guarantee',clearance_validity='invalidate_after_pose_route_change_or_unexpected_base_motion'):raise ValueError('Invalid arm base policy')
        elif key=='base_reposition':
            reject_privilege(value)
            required={'forward_m','turn_rad','waypoint_index','position_error_m','rotation_error_rad','basis','scene_clearance','action_dispatched'}
            dispatched=out.get('action_dispatched',False) if 'execution_summary' in out else False
            if not isinstance(value,dict) or set(value)!=required or value['action_dispatched'] is not dispatched or value['scene_clearance']!='unknown':raise ValueError('Invalid reposition feedback')
            if any(type(value[k]) not in (int,float) or not math.isfinite(value[k]) for k in ('forward_m','turn_rad','position_error_m','rotation_error_rad')):raise ValueError('Invalid reposition numbers')
        elif key=='base_motion_required':
            reject_privilege(value)
            if (not isinstance(value,dict) or set(value)!={'proposed_action','phase','waypoint_index','action_dispatched'}
                    or value['action_dispatched'] is not False or type(value['waypoint_index']) is not int
                    or not 0<=value['waypoint_index']<=8 or value['phase'] not in ('settling','pregrasp','descend','lift','rest','hold','open','close')
                    or not isinstance(value['proposed_action'],list) or len(value['proposed_action'])!=2
                    or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1 for x in value['proposed_action'])):
                raise ValueError('Invalid base motion feedback')
        elif key=='base_push':
            if (not arm_base_push_exit or not isinstance(value,dict)
                    or set(value)!={'entry_step','first_exceeded_step','translation_m','turn_rad','delta_xy_m','translation_limit_m','rotation_limit_rad'}
                    or any(type(value[k]) is not int or value[k]<0 for k in ('entry_step','first_exceeded_step'))
                    or value['first_exceeded_step']<value['entry_step']
                    or not isinstance(value['delta_xy_m'],list) or len(value['delta_xy_m'])!=2
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in value['delta_xy_m']+[value[k] for k in ('translation_m','turn_rad','translation_limit_m','rotation_limit_rad')])):
                raise ValueError('Invalid base push feedback')
        elif key=='close_dispatched':
            if not g1_dispatch_semantics or type(value) is not bool:raise ValueError('Invalid close dispatch')
        elif key=='hold_residual':
            if (not g1_dispatch_semantics or not isinstance(value,dict)
                    or set(value)!={'position_error_m','rotation_error_rad','torso_delta_m'}
                    or any(x is None and k!='torso_delta_m' or x is not None and (type(x) not in (int,float) or not math.isfinite(x) or x<0) for k,x in value.items())):
                raise ValueError('Invalid hold residual')
        elif key in ('identity_comparison','preclose_residual'):
            reject_privilege(value)
            if key=='identity_comparison':validate_r2a_reply(value,identity_confirmation=identity_confirmation,identity_multiview_evidence=identity_multiview_evidence,identity_confirmed_persistence=identity_confirmed_persistence)
            elif set(value)!={'position_error_m','rotation_error_rad'} or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value.values()):raise ValueError('preclose residual whitelist')
        elif key == 'path_clearance':
            reject_privilege(value)
            if not isinstance(value,dict) or value.get('frame')!='command_entry_base_link' or not isinstance(value.get('links'),list):raise ValueError('invalid clearance report')
            required={'frame','links','path_samples','tracking_error','stale_evidence'}
            optional={'coverage_basis','path_model','preview_steps','path_complete',
                      'preview_status','acknowledged_steps','pending_ack','completion_basis','stages_completed'}
            if s11_system:optional.add('low_surface')
            if clearance_verification:optional.add('verification')
            if not required<=set(value) or set(value)-required-optional:raise ValueError('clearance report whitelist')
            if 'low_surface' in value:
                low=value['low_surface']
                if (set(low)!={'count','height_limit_base_m','source','source_sha256','quantization_rule','excluded_links','not_a_traversability_certificate'}
                        or type(low['count']) is not int or low['count']<0
                        or not math.isfinite(low['height_limit_base_m'])
                        or low['source']!='pinned_base_link_collision_bounds_min_z'
                        or not re.fullmatch('[0-9a-f]{64}',low['source_sha256'])
                        or low['quantization_rule']!='near_voxel_lower_grid_face_at_or_below_chassis_underside'
                        or low['excluded_links']!=['base_link','r_wheel_link','l_wheel_link']
                        or low['not_a_traversability_certificate'] is not True):raise ValueError('S1.1 low surface contract')
            if 'verification' in value:
                v=value['verification']
                if (not isinstance(v,dict) or set(v)!={'status','reasons','high_unobserved_links','unobserved_threshold','guidance'}
                        or v['status'] not in ('unverified','sensor_report_only') or v['unobserved_threshold']!=.5
                        or not isinstance(v['reasons'],list) or len(v['reasons'])>4
                        or any(r not in ('missing_relevant_links','majority_unobserved','stale_evidence','incomplete_preview') for r in v['reasons'])
                        or (v['status']=='unverified')!=bool(v['reasons'])
                        or not isinstance(v['high_unobserved_links'],list) or len(v['high_unobserved_links'])>64
                        or any(not isinstance(link,str) or len(link)>80 for link in v['high_unobserved_links'])
                        or not isinstance(v['guidance'],str) or len(v['guidance'])>150):raise ValueError('Invalid clearance verification')
            if 'preview_status' in value:
                if (value['preview_status'] not in ('plan_complete','episode_step_limit','stage_step_limit','rest_budget_infeasible','base_motion_required')
                        or type(value.get('pending_ack')) is not bool or value['pending_ack']
                        or type(value.get('acknowledged_steps')) is not int
                        or not 0<=value['acknowledged_steps']<=300
                        or value['acknowledged_steps']!=value.get('preview_steps')
                        or value.get('path_complete') is not (value['preview_status']=='plan_complete')
                        or type(value.get('stages_completed')) is not int or not 0<=value['stages_completed']<=9
                        or value.get('completion_basis')!='nominal_robot_kinematics_not_official_success'):
                    raise ValueError('clearance preview acknowledgement')
            if value['tracking_error']!='unknown' or type(value['stale_evidence']) is not bool:raise ValueError('clearance uncertainty')
            if type(value['path_samples']) is not int or not 1<=value['path_samples']<=100000 or len(value['links'])>64:raise ValueError('bounded clearance path')
            for row in value['links']:
                if set(row)!={'link','min_clearance_m','segment_index','nearest_point_base_m','unobserved_fraction','status'}:raise ValueError('clearance link fields')
                if not isinstance(row['link'],str) or not 1<=len(row['link'])<=80:raise ValueError('clearance link')
                if row['segment_index'] is not None and (type(row['segment_index']) is not int or not 0<=row['segment_index']<value['path_samples']):raise ValueError('clearance segment')
                if not 0<=row['unobserved_fraction']<=1 or row['status'] not in ('observed','unknown'):raise ValueError('clearance coverage')
                if row['min_clearance_m'] is not None and not math.isfinite(row['min_clearance_m']):raise ValueError('finite clearance')
                if row['nearest_point_base_m'] is not None:vector(row['nearest_point_base_m'],3)
        elif key == 'grip_distance_sequence':
            if not isinstance(value,list) or not 1<=len(value)<=1601:raise ValueError('bounded grip samples required')
            previous=-1
            for row in value:
                if set(row)!={'step','distance_m'} or type(row['step']) is not int or row['step']<=previous or type(row['distance_m']) not in (float,int) or not math.isfinite(row['distance_m']) or not -.01<=row['distance_m']<=.12:raise ValueError('invalid grip sample')
                previous=row['step']
        elif key == 'waypoint_progress':
            _waypoint_progress_public(value)
        elif key == 'motion_completion':
            r15_feedback_public(value)
        elif (base_tracking_exit or error_progress_guard) and key in ('completed','requires_replan','localization_invalidated','clearance_invalidated'):
            if type(value) is not bool:raise ValueError('Boolean base exit feedback')
        elif base_tracking_exit and key in ('actual_forward_m','actual_turn_rad','actual_lateral_m','base_translation_m','base_path_deviation_m'):
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite base exit measurement')
        elif key == 'brake_completed':
            if type(value) is not bool:raise ValueError('Boolean brake result required')
        elif key == 'brake_steps':
            if type(value) is not int or not 0<=value<=60:raise ValueError('Bounded brake steps required')
        elif key == 'clearance_status':
            if value not in ('observed_conflict','coverage_insufficient','stale_evidence','no_observed_conflict'):
                raise ValueError('clearance status')
        elif key == 'clearance_distance_m':
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('clearance finite distance')
        elif key == 'clearance_evidence_step':
            if type(value) is not int or value < 0:raise ValueError('clearance evidence step')
        elif key in ('clearance_scope','clearance_body','clearance_stop_reason'):
            if type(value) is not str or len(value)>100:raise ValueError('clearance bounded text')
            reject_privilege(value)
        elif key in ('closure_command_completed','motion_suspended','clearance_observation_exception','clearance_unknown_advisory','localization_motion_executed'):
            if type(value) is not bool:
                raise ValueError(key+' must be bool')
        elif key == 'grasp_error_base_m':
            vector(value, 3)
        elif key == 'point_base_m':
            if value is not None:
                vector(value, 3)
        elif key == 'self_collision_pair':
            if (type(value) is not list or len(value) != 2
                    or value[0] not in ('head_pan_link', 'head_tilt_link')
                    or value[1] not in ('elbow_flex_link', 'forearm_roll_link', 'wrist_flex_link', 'wrist_roll_link')):
                raise ValueError('invalid robot self collision pair')
        elif type(value) not in (str, int, float, bool, type(None)):
            raise ValueError('tool result type')
        if isinstance(value, str) and len(value) > 150:
            raise ValueError('tool result text length')
    for key, enabled in (('base_progress', base_progress_monitor), ('base_response', base_response_monitor),
                         ('base_response_debt', base_response_debt_monitor), ('execution_lifecycle', execution_lifecycle)):
        if enabled and key in result:
            out[key] = _execution_public(result[key], key)
    if gripper_geometry_feedback and 'robot_gripper_envelopes' in result:
        validate_gripper_envelopes(result['robot_gripper_envelopes'])
        out['robot_gripper_envelopes']=deepcopy(result['robot_gripper_envelopes'])
    if (rest_progress_feedback or error_progress_guard) and 'rest_progress_feedback' in result:
        out['rest_progress_feedback']=_rest_progress_public(result['rest_progress_feedback'])
    if multi_locate and 'located_points' in result:

        rows = result['located_points']

        if not isinstance(rows, list) or not 1 <= len(rows) <= 6:

            raise ValueError('bounded located_points required')

        out['located_points'] = []

        for row in rows:

            if set(row) != {'pixel', 'result'} or not isinstance(row['pixel'], list) or len(row['pixel']) != 2 or any(

                    type(v) is not int or not 0 <= v < 640 for v in row['pixel']):

                raise ValueError('located point schema')

            projected = public_result(row['result'])

            if projected != row['result']:raise ValueError('located point contains nonpublic fields')

            out['located_points'].append(dict(pixel=list(row['pixel']), result=projected))

    if approach_point and 'approach_feedback' in result:

        out['approach_feedback'] = approach_feedback_public(result['approach_feedback'], r15_protocol=r15_protocol)

    canonical(out)
    return out

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


# Context: the ONLY model-input assembly site.
RESOLUTION = 640

STEP_CAP = 600

TOOL_CAP = 25

OBS_FIELDS = {'observation_id', 'joint_positions', 'sensors', 'steps_remaining', 'calls_remaining', 'target_semantics', 'history', 'odometry', 'previous_state', 'tcp_pose', 'joint_velocities'}

PROPRIO_KEYS = FETCH_ORDER[3:]

def audit_observation(obs, manifest, turn, profile=C0, *, hot_recovery=False):
    if profile.stage_plan:
        return s2_audit_observation(obs,manifest,turn,profile)
    if set(obs) != OBS_FIELDS:
        raise ValueError('main observation whitelist: no privilege or evaluation fields')
    if type(turn) is not int or not 0 <= turn < profile.call_cap:
        raise ValueError('turn range')
    if obs['target_semantics'] not in {'target object: ' + name for name in OBJECT_NAMES.values()}:
        raise ValueError('only frozen target category permitted')
    if not valid_id(obs['observation_id']):
        raise ValueError('observation_id')
    for key in ('joint_positions', 'joint_velocities'):
        if set(obs[key]) != set(PROPRIO_KEYS):
            raise ValueError('proprioception excludes root coordinates')
        vector(list(obs[key].values()), len(PROPRIO_KEYS))
    if set(obs['odometry']) != {'dx_m', 'dy_m', 'dyaw_rad'}:
        raise ValueError('relative odometry whitelist')
    vector(list(obs['odometry'].values()), 3)
    if abs(obs['odometry']['dyaw_rad']) > math.pi + 1e-12:
        raise ValueError('wrapped yaw required')
    p = obs['tcp_pose']
    if set(p) != {'position', 'quaternion_xyzw'}:
        raise ValueError('TCP schema')
    pose(p['position'], p['quaternion_xyzw'])
    if set(obs['sensors']) != set(SENSOR_FIELDS) or set(manifest) != set(SENSOR_FIELDS):
        raise ValueError('sensor whitelist')
    for field in SENSOR_FIELDS:
        _audit_sensor(field, obs['sensors'][field], manifest[field], RESOLUTION)
    if type(obs['steps_remaining']) is not int or not 0 <= obs['steps_remaining'] <= profile.step_cap or obs['calls_remaining'] != profile.call_cap - turn - (sum((row.get('call') or {}).get('tool')=='verify_target' and row.get('result',{}).get('reason')=='review_sent' for row in obs['history']) if profile.r2a_identity else 0):
        raise ValueError('observation budget')
    prior = obs['previous_state']
    if turn == 0 and prior is not None:
        raise ValueError('no cross-attempt previous state')
    if turn:
        if not isinstance(prior, dict) or set(prior) != {'observation_id', 'odometry', 'tcp_pose', 'steps_remaining'}:
            raise ValueError('previous state whitelist')
        if not valid_id(prior['observation_id']):
            raise ValueError('prior observation id')
        if set(prior['odometry']) != {'dx_m', 'dy_m', 'dyaw_rad'}:
            raise ValueError('prior odometry whitelist')
        vector(list(prior['odometry'].values()), 3)
        if set(prior['tcp_pose']) != {'position', 'quaternion_xyzw'}:
            raise ValueError('prior tcp whitelist')
        pose(prior['tcp_pose']['position'], prior['tcp_pose']['quaternion_xyzw'])
        if type(prior['steps_remaining']) is not int or not obs['steps_remaining'] <= prior['steps_remaining'] <= profile.step_cap:
            raise ValueError('prior budget')
    if len(obs['history']) != turn:
        raise ValueError('history length')
    for entry in obs['history']:
        if set(entry) != {'call', 'result'}:
            raise ValueError('history whitelist')
        if entry['call'] is not None:
            validate_call(entry['call'], profile if profile.grasp_alignment else C0)
        if public_result(entry['result'], robot_diagnostics=profile.robot_diagnostics,
schedule_precheck=profile.schedule_precheck, base_progress_monitor=profile.base_progress_monitor, base_response_monitor=profile.base_response_monitor, base_response_debt_monitor=profile.base_response_debt_monitor, execution_lifecycle=profile.execution_lifecycle, gripper_geometry_feedback=profile.gripper_geometry_feedback, motion_guard=profile.motion_guard, local_clearance=profile.local_clearance, rest_progress_feedback=profile.rest_progress_feedback, approach_point=profile.approach_point, multi_locate=profile.multi_locate, r15_completion=profile.r15_completion, r15_protocol=profile.r15_protocol, r2a_identity=profile.r2a_identity, identity_attempt_recovery=profile.identity_attempt_recovery, identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence, r2a_close=profile.r2a_close, r2a_feedback=profile.r2a_feedback, r2b_clearance=profile.r2b_clearance, r2b_grip_monitor=profile.r2b_grip_monitor, motion_progress_only=profile.motion_progress_only, arm_base_explicit=profile.arm_base_explicit, arm_tools_lock_base=profile.arm_tools_lock_base, base_tracking_exit=profile.base_tracking_exit, arm_base_push_exit=profile.arm_base_push_exit, g1_dispatch_semantics=profile.g1_dispatch_semantics, **{f:getattr(profile,f) for f in R2E_FLAGS}, hot_recovery=hot_recovery or profile==SOL_HOT_RECOVERY, s11_system=profile in V3_SYSTEM_PROFILES) != entry['result']:

            raise ValueError('history contains evaluation fields')
    canonical(obs)

def compact_decision(call):
    return None if call is None else {k: v for k, v in call.items() if v is not None and v != [] and (v != '')}

def text_observation(obs):
    return {k: v for k, v in obs.items() if k not in ('sensors', 'history', 'previous_state')}

def history_call(call, result, profile=C0):
    """Historical display format, not an executor/validator adapter."""
    if call is None:
        return None
    if 'pixels' not in call or profile.multi_locate:

        validate_call(call, profile if profile.grasp_alignment else C0)
        return deepcopy(call)
    validate_call(call, C1)
    if call['tool'] == 'measure':
        return None
    displayed = {k: v for k, v in call.items() if k != 'pixels'}
    if call['tool'] == 'locate_point':
        displayed['observation_id'] = result['observation_id']
    return displayed

def project_history(history, profile=C0):
    reject_privilege(history)
    return [dict(call=history_call(h['call'], h['result'], profile), result=public_result({k: v for k, v in h['result'].items() if k != 'parameter_error' or not isinstance(v, dict)}, robot_diagnostics=profile.robot_diagnostics, schedule_precheck=profile.schedule_precheck, base_progress_monitor=profile.base_progress_monitor, base_response_monitor=profile.base_response_monitor, base_response_debt_monitor=profile.base_response_debt_monitor, execution_lifecycle=profile.execution_lifecycle, gripper_geometry_feedback=profile.gripper_geometry_feedback, motion_guard=profile.motion_guard, local_clearance=profile.local_clearance, rest_progress_feedback=profile.rest_progress_feedback, approach_point=profile.approach_point, multi_locate=profile.multi_locate, r15_completion=profile.r15_completion, r15_protocol=profile.r15_protocol, r2a_identity=profile.r2a_identity, identity_attempt_recovery=profile.identity_attempt_recovery, identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence, r2a_close=profile.r2a_close, r2a_feedback=profile.r2a_feedback, r2b_clearance=profile.r2b_clearance, r2b_grip_monitor=profile.r2b_grip_monitor, motion_progress_only=profile.motion_progress_only, arm_base_explicit=profile.arm_base_explicit, arm_tools_lock_base=profile.arm_tools_lock_base, base_tracking_exit=profile.base_tracking_exit, arm_base_push_exit=profile.arm_base_push_exit, g1_dispatch_semantics=profile.g1_dispatch_semantics, **{f:getattr(profile,f) for f in R2E_FLAGS}, hot_recovery=profile==SOL_HOT_RECOVERY,s11_system=profile in V3_SYSTEM_PROFILES)) for h in history]


def _feedback(history):
    result = []
    for i, h in enumerate(history):
        c = h['call']
        r = h['result']
        if c and c['tool'] in ('measure', 'locate_point') and ('samples' in r):
            allowed = {'accepted', 'readonly', 'reason', 'observation_id', 'camera', 'samples', 'geometry', 'steps_used', 'steps_remaining', 'valid', 'point_base_m', 'depth_m', 'depth_spread_m'}
            if set(r) - allowed:
                raise ValueError('measurement feedback fields')
            for sample in r['samples']:
                if set(sample) - {'pixel', 'accepted', 'readonly', 'reason', 'valid', 'point_base_m', 'depth_m', 'depth_spread_m', 'observation_id'}:
                    raise ValueError('measurement sample fields')
            if set(r['geometry']) != {'reference_definition', 'centroid_base_m', 'plane_normal_base', 'principal_axis_base', 'secondary_axis_base', 'width_major_m', 'width_minor_m', 'plane_residual_m', 'diagnostics'}:
                raise ValueError('measurement geometry fields')
            result.append(dict(turn=i, call=deepcopy(c), result=deepcopy(r)))
        if isinstance(r.get('parameter_error'), dict):
            if set(r['parameter_error']) != {'code', 'field', 'expected'}:
                raise ValueError('error fields')
            result.append(dict(turn=i, parameter_error=deepcopy(r['parameter_error'])))
    reject_privilege(result)
    canonical(result)
    return result

RESUME_VERIFY_PRIORITY = 'verify_before_further_inspection_v1'
RESUME_VERIFY_GUIDANCE = (
    'User continuation priority: request independent verify_target on the current candidate '
    'before further inspection moves. Full framing is not category confirmation. '
    'If confirmed, proceed with localization and grasp using the existing gates. '
    'If review is unclear, use its specific missing evidence to choose a targeted adjustment; '
    'avoid repeated viewing changes without new evidence. Never bypass identity or grasp checks.')


RESUME_GRASP_PRIORITY = 'confirmed_target_grasp_v1'
RESUME_GRASP_GUIDANCE = (
    'User continuation priority: preserve the existing independently confirmed current target. '
    'With that confirmation still valid, next localize its grasp point and support, then attempt '
    'grasp alignment and gripper closure using the existing gates. Do not restart search or '
    'repeat identity inspection merely because the view becomes occluded. If a grasp check fails, '
    'use its feedback for local pose adjustments at the current target. Additional observation '
    'must resolve specific missing grasp geometry, not repeat completed category review. '
    'Explicit contradictory identity evidence still requires review; this instruction never '
    'creates confirmation when absent and never bypasses IK, arrival, or grasp checks.')
RESUME_REPLAN_PRIORITY = 'confirmed_target_arm_base_replan_v1'
RESUME_REPLAN_GUIDANCE = RESUME_GRASP_GUIDANCE + (
    ' Local replanning includes the arm AND the mobile base; it does not mean keeping the base fixed. '
    'After IK failure, compare wrist/grasp rotation, base rotation, and a small base translation '
    'using current RGB-D, odometry, joint state, and previous check_path feedback. These options '
    'are complementary: choose one or a sequence and explain why it changes reach or clearance. '
    'ik_unresolved does not prove excessive distance, scene collision, or impossibility. '
    'Do not repeat nearby grasp angles without explaining what the next test resolves. '
    'If reach appears limiting, consider a small approach rather than only rotating the wrist. '
    'Before base motion, estimate clearance to the cabinet front and corners, not the target center; '
    'include chassis front/width, rotational swept space, and protruding arm/gripper throughout '
    'the proposed motion. Keep a positive uncertainty allowance; unknown or occluded space is '
    'not free space. If needed, obtain a targeted depth view or reposition the arm before moving. '
    'Do not automatically advance or rotate on IK failure. After a short supported base segment '
    'check actual odometry and fresh RGB-D, update target coordinates, and test the revised grasp. '
    'Retain target confirmation unless explicit contrary evidence arises; remeasurement after '
    'base motion is geometry updating, not a requirement to restart identity review.')
RESUME_PRIORITIES = {RESUME_VERIFY_PRIORITY: RESUME_VERIFY_GUIDANCE,
                     RESUME_GRASP_PRIORITY: RESUME_GRASP_GUIDANCE,
                     RESUME_REPLAN_PRIORITY: RESUME_REPLAN_GUIDANCE}
RESUME_CONTACT_REPLAN_PRIORITY = 'confirmed_target_contact_and_clearance_replan_v2'
RESUME_PRIORITIES[RESUME_CONTACT_REPLAN_PRIORITY] = RESUME_REPLAN_GUIDANCE + (
    ' grasp_error_base_m is commanded TCP minus measured robot FK in the current base frame '
    '(x forward, y left, z up); it is not target error or contact identity. '
    'compliance_blocked means sustained controller deflection with an unmet arrival gate, '
    'not proof of table contact. Target contact during finger closure is expected, but an open '
    'finger striking a box during approach can prevent the commanded grasp pose from arriving. '
    'Use the residual direction and fresh hand RGB-D to distinguish approach obstruction from '
    'height or reach error. Check box edges, finger opening and approach orientation; do not '
    'interpret a visible surface point as the box center or repeatedly raise the pose without '
    'geometric evidence. Replan an entry that places the open fingers on opposite sides before '
    'closure. Preserve the strict arrival gate; do not close at an unarrived pose. '
    'For base motion, a single depth point cannot establish a cabinet plane, corner, or corridor '
    'clearance. A high or lateral point does not bound lower doors or protrusions. Inspect the '
    'full swept chassis width and height, and arm extent, with robot RGB-D; use multiple relevant '
    'surface/edge measurements and account for uncertainty. Do not subtract chassis radius from '
    'an unrelated point and call the remainder normal clearance. When the limiting lower/side '
    'surface is occluded, obtain a targeted view before translation or rotation. A completed '
    'move means command execution, not collision-free clearance; compare odometry and new images.')


RESUME_CONTACT_PROGRESS_PRIORITY = 'confirmed_target_contact_and_clearance_replan_v3'
RESUME_PRIORITIES[RESUME_CONTACT_PROGRESS_PRIORITY] = RESUME_PRIORITIES[RESUME_CONTACT_REPLAN_PRIORITY] + (
    ' CONFIRMED-TARGET PROGRESS: do not require enough requests for a guaranteed complete pick '
    'before taking one useful sensor-supported action. After IK failure explicitly compare '
    'wrist rotation, base rotation, and small translation; state the current evidence and '
    'missing clearance observation for each, rather than assume they all require a full retry. '
    'Request an informative check or short supported adjustment when admitted. Never force '
    'movement into unknown space just to spend remaining requests. Preserve confirmation and '
    'used recovery flags. return_to_rest is not a collision-free escape: it raises and rotates '
    'the arm toward a fixed robot rest posture and can sweep wrist/forearm into overhead or '
    'side cabinetry. First inspect that swept region and use short observed retreat waypoints '
    'to leave furniture when necessary. Do not assume moving upward alone clears cabinets. '
    'If clearance cannot be established within current limits, report the obstruction and '
    'give_up with the current scene preserved rather than force an unverified rest movement. '
    'A reserved rest request is an option, not an obligation to execute an unsafe rest path.')


def validate_resumed_execution(value):
    """Public run lifecycle only; no saved simulator state or target truth."""
    if (not isinstance(value,dict) or set(value)-{'user_priority'}!=
            {'status','restored_control_step','segment_wall_seconds_remaining','history_policy'}
            or value['status']!='resumed_with_new_wall_window'
            or value['history_policy']!='prior_limits_are_historical; current_limits_still_apply'
            or type(value['restored_control_step']) is not int or value['restored_control_step']<0
            or type(value['segment_wall_seconds_remaining']) is not int
            or value['segment_wall_seconds_remaining']<0):
        raise ValueError('Invalid resumed execution feedback')
    if 'user_priority' in value and value['user_priority'] not in RESUME_PRIORITIES:
        raise ValueError('Invalid resumed user priority')


def _view(bundle, profile, *, historical=False):
    hot_scope=profile==SOL_HOT_RECOVERY
    if historical and hot_scope and bundle['observation'].get('single_pick',{}).get('total_request_cap')==120:
        profile=replace(profile,call_cap=120)
    value = deepcopy(bundle)
    if profile.cold_checkpoint:
        resumed = value['observation'].pop('resumed_execution', None)
        if resumed is not None:
            validate_resumed_execution(resumed)
    if profile.episode_memory:
        from .memory import validate_summary
        memory = value['observation'].pop('episode_memory', None)
        if memory is not None:
            validate_summary(memory)
            reject_privilege(memory)
    if profile.weekly_budget:
        budget = value['observation'].pop('single_pick', None)
        if budget is not None:
            validate_single_pick_snapshot(budget, profile)
    if profile.target_evidence_diagnostics:
        accounting = value['observation'].pop('accounting_remaining', None)
        if accounting is not None:
            validate_accounting_remaining(accounting)
    if profile.base_motion_abort or (profile.base_progress_abort and (profile.base_progress_monitor or profile.base_response_monitor or profile.base_response_debt_monitor)):
        for entry in value['observation']['history']:
            rule = entry['result'].pop('base_abort_rule', None)
            allowed_rules = (None,) + (('A_spin', 'B_slip') if profile.base_motion_abort else ())
            if profile.base_progress_monitor and profile.base_progress_abort:
                allowed_rules += ('progress_stuck', 'progress_reverse', 'progress_lateral')
            if profile.base_response_monitor and profile.base_progress_abort:
                allowed_rules += ('response_linear', 'response_angular')
            if profile.base_response_debt_monitor and profile.base_progress_abort:
                allowed_rules += ('response_debt_linear', 'response_debt_angular')
            if rule not in allowed_rules:
                raise ValueError('invalid base abort rule')
    if profile.identity_revision == 2:
        summary = value['observation'].pop('target_record', None)
        reject_privilege(summary)
        for entry in value['observation']['history']:
            for key in ('target_record', 'grasp_sensor_feedback', 'target_observation_feedback', 'target_evidence_diagnostics'):
                if key == 'target_evidence_diagnostics' and key in entry['result']:
                    if not profile.target_evidence_diagnostics:
                        raise ValueError('target diagnostics disabled')
                    validate_target_diagnostics(entry['result'][key])
                reject_privilege(entry['result'].pop(key, None))
            if profile.model_target_boxes and entry['call']:
                entry['call'].pop('target_observation', None)
            if profile.target_association and entry['call']:
                entry['call'].pop('target_association', None)
            if entry['call'] and entry['call']['tool'] in ('verify_target', 'recover_pick'):
                validate_call(entry['call'], replace(profile, model_target_boxes=False, target_association=False))
                entry['call'] = None
    if profile.stall_feedback:
        for entry in value['observation']['history']:
            for key in ('execution_phase', 'consecutive_ik_failures'):
                entry['result'].pop(key, None)
    if profile.verification:
        for entry in value['observation']['history']:
            feedback = entry['result'].pop('identity_verification', None)
            if feedback is not None:
                validate_identity_feedback(feedback)
    if profile.measure:
        value['observation']['history'] = project_history(value['observation']['history'], profile)
    audit_profile = replace(profile, model_target_boxes=False, target_association=False) if profile.active_confirmation else profile
    # The audit copy disables target annotations, so dataclass equality no longer
    # identifies the original execution profile. Preserve its feedback scope.
    audit_observation(value['observation'], value['sensor_manifest'], value['turn'], audit_profile,
                      hot_recovery=hot_scope)
    return value

def build_wire(bundle, archive, profile=C0, *, verification=False, session=None):
    if profile.stage_plan:
        if verification:
            raise ValueError('S2 main plans are not independent verification requests')
        return build_s2_wire(bundle, archive, profile, session=session)
    if profile.response_session and verification:
        raise ValueError('Sol-session-v1 uses the C0 controller without a verifier')
    if verification:
        if archive:
            raise ValueError('independent verification accepts no history')
        return canonical(verification_body(bundle))
    expected = {'position', 'turn', 'kind', 'authorization_sha256', 'observation', 'sensor_manifest'}
    case_pattern = r'(?:arm-dev-00[0-4]|c2-smoke-00[01]|c2-base-00[0-4]|c2-expand-00[0-9]|c2-expand-01[0-2])' if profile.verification else r'arm-dev-00[0-4]'
    if profile.stall_guard_mode == 'reference_progress':
        case_pattern = r'(?:baseline-0(?:0[0-9]|1[0-9])|arm-dev-00[0-4]|c2-expand-00[0-9]|c2-expand-01[0-2])'
    if profile in (SOL_SESSION, SOL_MEMORY, SOL_HANDOFF):
        case_pattern = r'(?:arm-dev-00[0-4]|baseline-(?:000|001|004|005|008|016))'
    if profile in (SOL_TRANSPORT_A, SOL_TRANSPORT_B):
        case_pattern = r'handoff-transport-val000-object0'
    if profile in (SOL_PICK20,SOL_PICK20_ALIGNED)+DEV_PROFILES or profile.weekly_budget:

        case_pattern = r'baseline-0[01][0-9]'
    if set(bundle) != expected or bundle['kind'] != 'tool' or (not isinstance(bundle['position'], str)) or (not re.fullmatch(case_pattern, bundle['position'])):
        raise ValueError('mobile development identity required')
    if len(archive) != bundle['turn']:
        raise ValueError('complete ordered same-attempt archive required')
    current = _view(bundle, profile)
    obs = current['observation']
    turn = bundle['turn']
    if profile.response_session:
        if not isinstance(session, ResponseSession):
            raise ValueError('explicit response session required')
        if getattr(session, 'authorization_lineage', None) and not profile.cold_checkpoint:
            raise ValueError('checkpoint authorization lineage requires a cold-checkpoint profile')
        # A restored session may retain immutable requests from older grants.
        # Check the complete receipt chain and its authorized prefix before
        # permitting any historical authorization to differ from this request.
        session.bind(bundle, archive)
    history = []
    for index, item in enumerate(archive):
        historical = item.get('source_profile')
        allowed_historical = {C2_R25.name:C2_R25,C2_R25_FAR.name:C2_R25_FAR}
        if set(item) != {'bundle','call'}:
            if (profile not in (C2_R33,C2_R33_FAR) or set(item)!={'bundle','call','source_profile'}
                    or historical not in allowed_historical):
                raise ValueError('archive field whitelist')
        prior = item['bundle']
        prior_authorization = (session.archive_authorization(index, bundle['authorization_sha256'])
                               if profile.response_session else bundle['authorization_sha256'])
        if prior['position'] != bundle['position'] or prior['authorization_sha256'] != prior_authorization or prior['kind'] != 'tool' or (prior['turn'] != index):
            raise ValueError('cross-attempt or reordered archive')
        call = (history_call(item['call'], bundle['observation']['history'][index]['result'],profile)
                if profile.search_recessed_view and profile.measure else
                history_call(item['call'], bundle['observation']['history'][index]['result']) if profile.measure else item['call'])
        if call is not None:
            call_profile = allowed_historical[historical] if historical is not None else profile if profile.identity_revision == 2 or profile.active_memory or profile.task_handoff or profile.grasp_alignment else C0
            validate_call(call,call_profile)
        record=dict(bundle=_view(prior,profile,historical=True),call=call)
        if historical is not None:record['source_profile']=historical
        history.append(record)
    records = []
    for index, item in enumerate(history):
        executed = obs['history'][index]['call']
        if profile.identity_revision == 2:
            executed = bundle['observation']['history'][index]['call']
        feedback = obs['history'][index]['result']
        if (profile.base_motion_abort or profile.base_progress_abort) and 'base_abort_rule' in bundle['observation']['history'][index]['result']:
            feedback = dict(feedback, base_abort_rule=bundle['observation']['history'][index]['result']['base_abort_rule'])
        rejected_without_motion = executed is None and feedback.get('accepted') is False and (feedback.get('steps_used') == 0)
        expected_call = ({k:v for k,v in item['call'].items() if k not in ('memory', 'handoff')}
                         if (profile.active_memory or profile.task_handoff) and item['call'] is not None else item['call'])
        if canonical(expected_call) != canonical(executed) and (not rejected_without_motion):
            raise ValueError('history differs from executed decisions')
        if profile.verification and 'identity_verification' in bundle['observation']['history'][index]['result']:
            feedback = dict(feedback, identity_verification=deepcopy(bundle['observation']['history'][index]['result']['identity_verification']))
        if profile.identity_revision == 2:
            for key in ('target_record', 'grasp_sensor_feedback', 'target_observation_feedback', 'target_evidence_diagnostics'):
                if key in bundle['observation']['history'][index]['result']:
                    feedback = dict(feedback, **{key: deepcopy(bundle['observation']['history'][index]['result'][key])})
        record=dict(turn=index, observation=text_observation(item['bundle']['observation']), decision=compact_decision(item['call']), feedback=feedback)
        if 'source_profile' in item:record['source_profile']=item['source_profile']
        records.append(record)
    meta = {'context_version': 'guidance-v4-mobile', 'target_semantics': obs['target_semantics'], 'history_status': 'Historical records only. Past decisions are NOT commands to execute.', 'history': records, 'current': text_observation(obs), 'previous_action_comparison': transition(obs['previous_state'], obs), 'depth_encoding': DEPTH_ENCODING, 'task_state_is_model_self_report': True, 'omitted_image_policy': 'Only previous/current observations plus one explicitly recalled old observation.', 'condition': profile.name, 'robot_specifications': deepcopy(FROZEN['robot_spec'])}
    if profile.r15_protocol:
        meta['robot_specifications'].update(look_step_cap=150,base_call_step_cap=250,arm_call_step_cap=300,rest_step_cap=300,rest_reserved_steps=300,episode_step_cap=1600,tool_call_cap=profile.call_cap if profile.lean_feedback else 80 if profile.r2d_extended_calls else 50)
    if profile.identity_revision == 2:
        meta['current']['target_record'] = deepcopy(bundle['observation'].get('target_record'))
    if profile.episode_memory:
        from .memory import validate_summary
        memory = bundle['observation'].get('episode_memory')
        if memory is None:raise ValueError('episode memory must be bound before model input')
        validate_summary(memory)
        reject_privilege(memory)
        meta['current']['episode_memory'] = deepcopy(memory)
    if profile.weekly_budget:
        meta['current']['single_pick'] = deepcopy(bundle['observation'].get('single_pick'))
    if profile.cold_checkpoint and 'resumed_execution' in bundle['observation']:
        meta['current']['resumed_execution'] = deepcopy(bundle['observation']['resumed_execution'])
    if profile.target_evidence_diagnostics:
        meta['current']['accounting_remaining'] = deepcopy(bundle['observation'].get('accounting_remaining'))

    def semantics(value):
        if isinstance(value, dict):
            for key, v in value.items():
                if key == 'target_semantics':
                    value[key] = (obs['target_semantics'] if profile.transport_task or (profile.response_session and bundle['position'].startswith('baseline-'))
                                  else FROZEN['semantics'][bundle['position']])
                else:
                    semantics(v)
        elif isinstance(value, list):
            for v in value:
                semantics(v)
    if profile.use_located_semantics:
        if profile.verification:
            raise ValueError("verification requires original observation target description")
        semantics(meta)
    if profile.stall_feedback:
        meta['execution_feedback_condition'] = 'stall-feedback-cpu-v1'
        meta['execution_diagnostics_history'] = [execution_diagnostics(h['result']) for h in bundle['observation']['history']]
    if profile.measure:
        meta['measure_feedback'] = _feedback(bundle['observation']['history'])
    groups = []
    if history:
        groups.append((history[-1]['bundle'], 'previous'))
    groups.append((current, 'current'))
    if turn and obs['history'][-1]['call'] and (obs['history'][-1]['call']['tool'] == 'recall_observation') and obs['history'][-1]['result'].get('accepted'):
        wanted = obs['history'][-1]['call']['observation_id']
        found = next((item['bundle'] for item in history if item['bundle']['observation']['observation_id'] == wanted), None)
        if found is None:
            raise ValueError('recalled image missing from this attempt archive')
        if all((g['observation']['observation_id'] != wanted for g, _ in groups)):
            groups.insert(0, (found, 'recalled_historical'))
    content = history_first_content(meta) if profile.history_first else [{'type': 'input_text', 'text': canonical(meta).decode()}]
    for group, label in groups:
        oldobs = group['observation']
        for field in SENSOR_FIELDS:
            tag = {'image_role': label, 'observation_id': oldobs['observation_id'], 'turn': group['turn'], 'control_step': profile.step_cap - oldobs['steps_remaining'], 'sensor': field, 'resolution': [RESOLUTION, RESOLUTION]}
            content.extend([{'type': 'input_text', 'text': canonical(tag).decode()}, {'type': 'input_image', 'detail': 'original', 'image_url': 'data:image/png;base64,' + oldobs['sensors'][field]['data_base64']}])
    body = {'model': 'gpt-6-astra', 'instructions': PROMPT + (FROZEN['c1_addition'] if profile.measure else ''), 'reasoning': {'effort': 'medium', 'summary': 'auto'}, 'service_tier': 'default', 'store': False, 'max_output_tokens': 4000, 'input': [{'role': 'user', 'content': content}], 'text': {'format': {'type': 'json_schema', 'name': 'wp2_measure_flat_call' if profile.measure else 'mobile_capability_tool_call', 'strict': True, 'schema': schema(profile)}}}
    if profile.verification and profile.identity_revision != 2:
        body['instructions'] += FROZEN['experimental']['identity_verification']['main_prompt_addition'].format(selection_distance_m=FROZEN['experimental']['identity_verification']['selection_distance_m'])
    if profile.identity_revision == 2 and not profile.active_confirmation:
        body['instructions'] += FROZEN['experimental']['identity_revision_2']['main_prompt']
    if profile == C2:
        body['instructions'] += FROZEN['experimental']['identity_verification']['c2_prompt_addition']
    if profile.stall_feedback:
        body['instructions'] += FROZEN['experimental']['stall_feedback']['prompt_addition']
    if profile.model_target_boxes:
        body['instructions'] += FROZEN['experimental']['astra_target_boxes']['main_prompt']
    if profile.base_motion_abort:
        body['instructions'] += FROZEN['experimental']['base_motion_abort']['prompt_addition']
    if any((profile.schedule_precheck, profile.robot_diagnostics,
            profile.base_heading_hold, profile.continuous_base_speed)):
        additions = FROZEN['experimental']['cpu_repair_r31']['prompt_additions']
        for flag in ('schedule_precheck', 'robot_diagnostics', 'base_heading_hold', 'continuous_base_speed'):
            if getattr(profile, flag):
                body['instructions'] += additions[flag]
    for flag in ('target_association', 'target_evidence_diagnostics', 'base_progress_monitor', 'base_response_monitor', 'base_response_debt_monitor', 'base_response_corroboration', 'base_progress_abort', 'execution_lifecycle'):
        if getattr(profile, flag):
            body['instructions'] += FROZEN['experimental']['cpu_repair_r32']['prompt_additions'][flag]
    if profile.history_first:
        body['prompt_cache_options'] = {'mode': 'explicit'}
    if profile.name in (C2_BASELINE_FAR.name, C2_R25_FAR.name, C2_R3_FAR.name, C2_R31_FAR.name, C2_R32_FAR.name, C2_R33_FAR.name):
        body['instructions'] = body['instructions'].replace('at most 25 tool calls and 600 control steps', 'at most 40 tool calls and 1000 control steps').replace('arm calls are capped at 150, total 600', 'arm calls are capped at 150, total 1000')
    if profile.response_session:
        return build_session_wire(body, meta, current, bundle, archive, profile, session)
    wire = canonical(body)
    if non_image_bytes(wire) > 262144:
        raise ValueError('mobile text history cap')
    return wire


def history_first_content(meta):
    """Lossless path/value blocks; append history without rewriting old blocks."""
    rolling = ('history', 'execution_diagnostics_history', 'measure_feedback')
    current = ('current', 'previous_action_comparison')
    fixed = {k: v for k, v in meta.items() if k not in rolling + current}
    def block(value):
        return {'type': 'input_text', 'text': canonical(value).decode()}
    blocks = [block({'block': 'fixed', 'fields': fixed,
                     'history_arrays': [k for k in rolling if k in meta]})]
    # Diagnostics are interleaved at their own turn, never appended before history.
    for i, record in enumerate(meta['history']):
        fields = [{'path': ['history', i], 'value': record}]
        for key in rolling[1:]:
            for j, value in enumerate(meta.get(key, [])):
                turn = value.get('turn', j) if key == 'measure_feedback' else j
                if turn == i:
                    fields.append({'path': [key, j], 'value': value})
        blocks.append(block({'block': 'history', 'fields': fields}))
    for value in blocks:
        value['prompt_cache_breakpoint'] = {'mode': 'explicit'}
    blocks.append(block({'block': 'current', 'fields': {k: meta[k] for k in current if k in meta}}))
    return blocks


@dataclass
class ResponseSession:
    """Attempt-local chain journal. No implicit restart, recovery or action replay."""
    attempt_id: str
    path: Path | None = None
    binding: dict | None = None
    receipts: list = field(default_factory=list)
    status: str = 'ready'
    handoff_plan: dict | None = None
    handoff_journal: dict | None = None
    transport_mission: dict | None = None
    single_pick: dict | None = None
    authorization_lineage: list = field(default_factory=list)
    cache_binding: dict | None = None

    def persist(self):
        if self.path is not None:
            value = dict(attempt_id=self.attempt_id, binding=self.binding,
                         receipts=self.receipts, status=self.status)
            if self.handoff_plan is not None:
                value.update(handoff_plan=self.handoff_plan, handoff_journal=self.handoff_journal)
            if self.transport_mission is not None:
                value['transport_mission'] = self.transport_mission
            if self.single_pick is not None:
                value['single_pick'] = deepcopy(self.single_pick)
            if getattr(self, 'authorization_lineage', None):
                value['authorization_lineage'] = deepcopy(self.authorization_lineage)
            if getattr(self, 'cache_binding', None) is not None:
                value['cache_binding'] = deepcopy(self.cache_binding)
            save(self.path, value)

    def cache_key(self):
        binding = getattr(self, 'cache_binding', None) or self.binding
        grants = {self.binding['authorization_sha256']}
        for event in getattr(self, 'authorization_lineage', []):
            grants.update((event['from_authorization_sha256'], event['to_authorization_sha256']))
        if (set(binding) != set(self.binding) or any(binding[k] != self.binding[k]
                for k in self.binding if k != 'authorization_sha256')
                or binding['authorization_sha256'] not in grants):
            raise ValueError('cache routing binding differs from authorized session')
        return 'sol-session-'+hashlib.sha256(canonical(binding)).hexdigest()[:52]

    def archive_authorization(self, index, current_authorization):
        """Return the grant bound to this receipt, never a general allowlist."""
        lineage = getattr(self, 'authorization_lineage', [])
        if not lineage:
            return current_authorization
        expected = lineage[0]['from_authorization_sha256']
        for item in lineage:
            if index >= item['prefix_main_requests']:
                expected = item['to_authorization_sha256']
        return expected

    def _validate_authorization_lineage(self, archive, history):
        lineage = getattr(self, 'authorization_lineage', [])
        if not lineage:
            return
        keys = {'contract','attempt_id','case_id','from_authorization_sha256',
                'to_authorization_sha256','prefix_main_requests','prefix_archive_sha256',
                'prefix_execution_sha256','prefix_receipts_sha256','checkpoint_receipt_sha256'}
        if not isinstance(lineage, list):
            raise ValueError('checkpoint authorization lineage structure')
        self._validate_checkpoint_receipt_chain()
        previous = None
        boundary = 0
        for item in lineage:
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError('checkpoint authorization lineage fields')
            count = item['prefix_main_requests']
            if (item['contract'] != 'sol-checkpoint-authorization-lineage-v1'
                    or item['attempt_id'] != self.attempt_id or item['case_id'] != self.binding['case_id']
                    or type(count) is not int or not boundary <= count <= len(archive)
                    or any(not isinstance(item[key], str) or not re.fullmatch(r'[0-9a-f]{64}', item[key])
                           for key in keys if key.endswith('_sha256'))
                    or (previous is not None and item['from_authorization_sha256'] != previous)
                    or item['from_authorization_sha256'] == item['to_authorization_sha256']):
                raise ValueError('checkpoint authorization lineage binding')
            for name, value in (('archive', archive), ('execution', history), ('receipts', self.receipts)):
                if item['prefix_'+name+'_sha256'] != hashlib.sha256(canonical(value[:count])).hexdigest():
                    raise ValueError('checkpoint authorization lineage prefix changed')
            previous = item['to_authorization_sha256']
            boundary = count
        if previous != self.binding['authorization_sha256']:
            raise ValueError('checkpoint authorization lineage current grant differs')
        for index, item in enumerate(archive):
            prior = item['bundle']
            if (prior.get('position') != self.binding['case_id'] or prior.get('turn') != index
                    or prior.get('authorization_sha256') != self.archive_authorization(index, previous)):
                raise ValueError('checkpoint authorization lineage archive differs')

    def _validate_checkpoint_receipt_chain(self):
        previous = None
        seen = set()
        for receipt in self.receipts:
            response = receipt.get('response_id')
            if (not isinstance(response, str) or not response or response in seen
                    or 'previous_response_id' not in receipt or receipt['previous_response_id'] != previous):
                raise ValueError('checkpoint provider response chain differs')
            seen.add(response)
            previous = response

    def rebind_checkpoint_authorization(self, authorization_sha256, archive, history, *, checkpoint_receipt_sha256):
        """Called only after the runner validates the closed parent and checkpoint.

        Keep every old bundle/receipt unchanged. A later cold resume appends a
        second exact prefix, so earlier grants cannot authorize later requests.
        """
        if (not isinstance(self.binding, dict)
                or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                       for value in (authorization_sha256, checkpoint_receipt_sha256,
                                     self.binding.get('authorization_sha256')))):
            raise ValueError('checkpoint authorization hashes required')
        old = self.binding['authorization_sha256']
        self.bind(dict(position=self.binding['case_id'], authorization_sha256=old,
                       turn=len(archive), observation={'history':history}), archive)
        self._validate_checkpoint_receipt_chain()
        for index, item in enumerate(archive):
            prior = item['bundle']
            if (prior.get('position') != self.binding['case_id'] or prior.get('turn') != index
                    or prior.get('authorization_sha256') != self.archive_authorization(index, old)):
                raise ValueError('checkpoint authorization prefix is not the bound session')
        if authorization_sha256 == old:
            return
        event = dict(contract='sol-checkpoint-authorization-lineage-v1',
            attempt_id=self.attempt_id, case_id=self.binding['case_id'],
            from_authorization_sha256=old, to_authorization_sha256=authorization_sha256,
            prefix_main_requests=len(archive), checkpoint_receipt_sha256=checkpoint_receipt_sha256,
            prefix_archive_sha256=hashlib.sha256(canonical(archive)).hexdigest(),
            prefix_execution_sha256=hashlib.sha256(canonical(history)).hexdigest(),
            prefix_receipts_sha256=hashlib.sha256(canonical(self.receipts)).hexdigest())
        # Pin the exact last-segment routing identity, not the new spending grant.
        if getattr(self, 'cache_binding', None) is None:
            self.cache_binding = deepcopy(self.binding)
        self.cache_key()
        self.authorization_lineage = deepcopy(getattr(self, 'authorization_lineage', []))+[event]
        self.binding = dict(self.binding, authorization_sha256=authorization_sha256)
        self._validate_authorization_lineage(archive, history)
        self.persist()

    def bind(self, bundle, archive):
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError('explicit session attempt_id required')
        binding = dict(attempt_id=self.attempt_id, case_id=bundle['position'],
                       authorization_sha256=bundle['authorization_sha256'])
        if self.handoff_plan is not None:
            binding['handoff_plan_sha256'] = hashlib.sha256(canonical(validate_handoff_plan(self.handoff_plan))).hexdigest()
        if self.transport_mission is not None:
            binding['transport_mission_sha256'] = hashlib.sha256(canonical(validate_transport_mission(self.transport_mission))).hexdigest()
        if self.binding is None:
            if bundle['turn'] or archive or self.receipts:
                raise ValueError('new session must begin at turn zero')
            self.binding = binding
        if self.binding != binding or self.status != 'ready':
            raise ValueError('session binding or execution state mismatch; no automatic recovery')
        if len(self.receipts) != bundle['turn'] or len(archive) != len(self.receipts):
            raise ValueError('session chain is missing or reordered')
        for i, (receipt, item) in enumerate(zip(self.receipts, archive)):
            if (receipt['archive_sha256'] != hashlib.sha256(canonical(item)).hexdigest()
                    or receipt.get('execution_sha256') != hashlib.sha256(canonical(bundle['observation']['history'][i])).hexdigest()):
                raise ValueError('session archive or actual execution feedback changed')
        self._validate_authorization_lineage(archive, bundle['observation']['history'])


def memory_ab_matrix():
    """Expand a local proposal; this is not a launch or authorization API."""
    cfg = FROZEN['memory_ab']
    if ([c['case_id'] for c in cfg['cases']] != [f'baseline-{n:03d}' for n in (0,1,4,5,8)]
            or cfg['arms'] != {'A': SOL_SESSION.name, 'B': SOL_MEMORY.name}
            or cfg['orders'] != [['A','B']] or cfg['repeats'] != 1
            or cfg['episode_calls_cap'] != 25 or cfg['episode_steps_cap'] != 600
            or cfg['total_calls_cap'] != 250 or cfg['total_steps_cap'] != 6000
            or cfg['automatic_retries'] != 0 or cfg['shared_initial_state_required'] is not True):
        raise ValueError('memory AB frozen matrix mismatch')
    rows = []
    for case in cfg['cases']:
        for repeat, order in enumerate(cfg['orders'],1):
            for arm in order:
                rows.append(dict(case_id=case['case_id'], position=case['position'], repeat=repeat,
                    arm=arm, execution_profile=cfg['arms'][arm],
                    attempt_id=f"memory-ab-{case['position']:03d}-r{repeat}-{arm}",
                    pair_id=f"memory-ab-{case['position']:03d}-r{repeat}",
                    per_case_decisions_cap=25, per_case_steps_cap=600,
                    initial_state_status='requires_paired_start_verification',
                    selection_source=case['selection_source'], selection_sha256=case['selection_sha256']))
    return rows


def validate_memory_ab_episode(auth, profile, *, require_start=True):
    cfg = FROZEN['memory_ab']
    if cfg['status'] != 'authorized':
        if (auth.get('approved_pilot')!='sol-memory-ab001-pilot-approved-40usd'
                or auth.get('case_ids')!=['baseline-001']):
            raise ValueError('memory AB is CPU-only proposed configuration, not execution authorization')
        receipt_path=Path(auth['full_b_preflight_receipt'])
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=auth['full_b_preflight_sha256']:
            raise ValueError('Pilot preflight receipt changed')
        receipt=json.loads(receipt_path.read_bytes())
        if (receipt.get('status')!='preflight_passed' or receipt.get('case_id')!='baseline-001'
                or receipt.get('api_requests')!=0 or not 0<receipt.get('simulator_actions',-1)<=160
                or receipt.get('preflight',{}).get('passed') is not True
                or receipt.get('source_fingerprint')!=ab_source_fingerprint()
                or receipt.get('wall_seconds',601)>600
                or any(k in receipt for k in ('close_error_type','full_b_recording_error'))):
            raise ValueError('Pilot needs passed same-source physical preflight')
    matches = [j for j in memory_ab_matrix() if j['attempt_id'] == auth.get('ab_attempt_id')]
    if len(matches) != 1:
        raise ValueError('memory AB episode not in matrix')
    job = matches[0]
    if (profile.name != job['execution_profile'] or auth.get('case_ids') != [job['case_id']]
            or auth.get('attempt_ids') != {job['case_id']:job['attempt_id']}
            or auth.get('ab_pair_id') != job['pair_id']):
        raise ValueError('memory AB profile, pair or initial-state binding mismatch')
    if (Decimal(str(auth.get('usd_cap','NaN'))) > Decimal(cfg['proposed_episode_usd_cap'])
            or auth.get('provider_requests_cap') != 25 or auth.get('physical_steps_cap') != 600):
        raise ValueError('memory AB episode cap mismatch')
    batch=MemoryABBatch.from_auth(auth)
    if require_start:batch.verify_start(job['attempt_id'])
    else:batch.preflight(job['attempt_id'])


def ab_source_fingerprint():
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*.py') if '__pycache__' not in p.parts}
    config = deepcopy(FROZEN)
    config['memory_ab'].pop('status', None)
    return hashlib.sha256(canonical(dict(files=files, config=config))).hexdigest()


def ab_initial_check(reference, actual):
    """Evaluation-only full-state comparison; never normalizes or restores a reset."""
    for snapshot in (reference, actual):
        if set(snapshot) != {'metadata','state'}:
            raise ValueError('paired snapshot fields')
        meta, state = snapshot['metadata'], snapshot['state']
        if set(meta) != {'case_id','uid','seed','spawn_index','source_sha256','runtime_sha256'}:
            raise ValueError('paired metadata fields')
        if (type(meta['seed']) is not int or type(meta['spawn_index']) is not int
                or not isinstance(meta['uid'],str) or not meta['uid']
                or any(not re.fullmatch('[0-9a-f]{64}',str(meta[k])) for k in ('source_sha256','runtime_sha256'))):
            raise ValueError('paired metadata types')
        if set(state) != {'qpos','qvel','root_pose','environment'}:
            raise ValueError('paired state requires robot and full environment')
        q, v, root = (np.asarray(state[k],dtype=float) for k in ('qpos','qvel','root_pose'))
        if q.ndim != 1 or not q.size or v.shape != q.shape or root.shape != (4,4):
            raise ValueError('paired robot state shape')
        env = state['environment']
        if not isinstance(env,dict) or not env.get('actors') or not env.get('articulations'):
            raise ValueError('paired full actor/articulation state required')
        canonical(snapshot)  # Reject NaN/infinity, including nested values.
    if reference['metadata'] != actual['metadata']:
        raise ValueError('paired metadata mismatch')
    differences = []
    def compare(a,b,path):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a)!=set(b):
                differences.append(path+':keys'); return
            for k in a:compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):
                differences.append(path+':shape'); return
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}[{i}]')
        elif type(a) is float:
            if type(b) not in (float,int) or not math.isfinite(b) or abs(a-b)>1e-6:
                differences.append(path)
        elif type(a)!=type(b) or a!=b:
            differences.append(path)
    compare(reference['state'],actual['state'],'state')
    if differences:
        raise ValueError('paired initial mismatch: '+', '.join(differences[:12]))
    return dict(passed=True, atol=1e-6, rtol=0, all_state_fields_checked=True)


class MemoryABBatch:
    """One durable ledger for all configured attempts. A crash lock is never auto-cleared."""
    call_cap = 25
    def __init__(self, root):
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, root, batch_id, references, *, usd_cap=None, episode_usd_cap='20', case_ids=None):
        cfg = FROZEN['memory_ab']; jobs = memory_ab_matrix()
        if case_ids is not None:
            if case_ids != ['baseline-001']:raise ValueError('Only the approved001 pilot subset is supported')
            jobs=[job for job in jobs if job['case_id'] in case_ids]
        limit='40' if case_ids is not None else cfg['proposed_usd_cap']
        if usd_cap is None:usd_cap=limit
        if set(references) != {j['case_id'] for j in jobs}:
            raise ValueError('all selected initial references required')
        for case, ref in references.items():
            ab_initial_check(ref,ref)
            if ref['metadata']['case_id']!=case or ref['metadata']['source_sha256']!=ab_source_fingerprint():
                raise ValueError('initial reference case/source mismatch')
        def ceiling(value, limit):
            amount = Decimal(str(value))
            if not amount.is_finite() or not 0 < amount <= Decimal(limit):
                raise ValueError('AB monetary ceiling')
            return str(amount)
        total = ceiling(usd_cap,limit)
        episode = ceiling(episode_usd_cap,min(Decimal(total),Decimal(cfg['proposed_episode_usd_cap'])))
        obj = cls(root); obj.root.mkdir(parents=True,exist_ok=False)
        save(obj.root/'manifest.json',dict(batch_id=batch_id,source_sha256=ab_source_fingerprint(),
            jobs=jobs,references=references,usd_cap=total,episode_usd_cap=episode,
            requests_cap=25*len(jobs),physical_steps_cap=600*len(jobs),currency='USD',case_ids=case_ids))
        save(obj.root/'ledger.json',dict(batch_id=batch_id,status='running',entries=[],starts={},closed={}))
        return obj

    @property
    def manifest_sha256(self):
        return hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()

    @classmethod
    def from_auth(cls, auth):
        if not isinstance(auth.get('ab_batch_path'),str):
            raise ValueError('AB shared ledger path required')
        obj=cls(auth['ab_batch_path']);manifest=obj.manifest()
        if obj.manifest_sha256!=auth.get('ab_manifest_sha256') or manifest['batch_id']!=auth['batch_id']:
            raise ValueError('AB shared manifest authorization mismatch')
        return obj

    def manifest(self):
        value=json.loads((self.root/'manifest.json').read_bytes())
        jobs=memory_ab_matrix();pilot=value.get('case_ids')
        if pilot is not None:
            if pilot!=['baseline-001']:raise ValueError('Unknown AB pilot subset')
            jobs=[job for job in jobs if job['case_id'] in pilot]
        limit=Decimal('40' if pilot is not None else FROZEN['memory_ab']['proposed_usd_cap'])
        if value['source_sha256']!=ab_source_fingerprint() or value['jobs']!=jobs:
            raise ValueError('AB source/matrix changed')
        total,episode=(Decimal(str(value[k])) for k in ('usd_cap','episode_usd_cap'))
        if (not total.is_finite() or not episode.is_finite() or not 0<episode<=total<=limit or episode>20
                or value['requests_cap']!=25*len(jobs) or value['physical_steps_cap']!=600*len(jobs) or value['currency']!='USD'
                or set(value['references'])!={j['case_id'] for j in value['jobs']}):
            raise ValueError('AB shared resource contract changed')
        return value

    @contextmanager
    def locked(self):
        lock=self.root/'ledger.lock'
        with lock.open('x',encoding='utf8') as stream:
            stream.write(str(os.getpid()));stream.flush();os.fsync(stream.fileno())
        try:yield
        finally:lock.unlink()  # Process death leaves the claim; never recover implicitly.

    def ledger(self):
        value=json.loads((self.root/'ledger.json').read_bytes())
        if value['batch_id']!=self.manifest()['batch_id'] or value['status']!='running':
            raise ValueError('AB batch stopped or identity mismatch')
        entries=value['entries'];seen=set()
        for e in entries:
            if e['request_id'] in seen or e['status'] not in ('valid','invalid_call'):
                raise ValueError('AB unresolved/duplicate request; reconcile without resend')
            seen.add(e['request_id'])
            for k in ('reserve_usd','estimated_usd'):
                if e.get(k) is None and k=='estimated_usd':continue
                a=Decimal(str(e.get(k)))
                if not a.is_finite() or a<0:raise ValueError('AB malformed accounting')
        return value

    def register_start(self, attempt_id, snapshot, initial_bundle):
        with self.locked():
            manifest=self.manifest();ledger=self.ledger()
            jobs=manifest['jobs']; job=next((j for j in jobs if j['attempt_id']==attempt_id),None)
            if job is None or attempt_id in ledger['starts'] or attempt_id in ledger['closed']:
                raise ValueError('AB start missing/duplicate; no restart')
            if any(j['attempt_id'] not in ledger['closed'] for j in jobs[:jobs.index(job)]):
                raise ValueError('AB order requires previous attempts closed')
            if (initial_bundle['turn']!=0 or initial_bundle['position']!=job['case_id']
                    or initial_bundle['observation']['steps_remaining']!=600
                    or initial_bundle['observation']['calls_remaining']!=self.call_cap):
                raise ValueError('AB initial observation binding')
            record=dict(attempt_id=attempt_id,pair_id=job['pair_id'],snapshot=snapshot,
                        initial_bundle_sha256=hashlib.sha256(canonical(initial_bundle)).hexdigest())
            # Preserve the failed actual state as evidence; do not retry its reset.
            save(self.root/(attempt_id+'.initial.json'),record)
            try:
                ab_initial_check(manifest['references'][job['case_id']],snapshot)
                for other in jobs:
                    if other['pair_id']==job['pair_id'] and other['attempt_id'] in ledger['starts']:
                        ab_initial_check(self._start(other['attempt_id'],ledger)['snapshot'],snapshot)
            except Exception:
                ledger['status']='stopped_initial_mismatch';save(self.root/'ledger.json',ledger);raise
            ledger['starts'][attempt_id]=hashlib.sha256(canonical(record)).hexdigest()
            save(self.root/'ledger.json',ledger)

    def preflight(self, attempt_id):
        if (self.root/'ledger.lock').exists():
            raise FileExistsError('AB outstanding batch lock; no automatic reset or recovery')
        ledger=self.ledger();jobs=self.manifest()['jobs']
        if len(ledger['entries'])>=self.manifest()['requests_cap'] or committed_cost(ledger['entries'])>=Decimal(self.manifest()['usd_cap']):
            raise ValueError('AB batch budget exhausted before reset')
        matches=[j for j in jobs if j['attempt_id']==attempt_id]
        if len(matches)!=1 or attempt_id in ledger['starts'] or attempt_id in ledger['closed']:
            raise ValueError('AB duplicate or unknown attempt before reset')
        if any(j['attempt_id'] not in ledger['closed'] for j in jobs[:jobs.index(matches[0])]):
            raise ValueError('AB order requires previous attempts closed')

    def abort(self, attempt_id, steps):
        with self.locked():
            ledger=json.loads((self.root/'ledger.json').read_bytes())
            if ledger['status']=='running':ledger['status']='stopped_execution_unknown'
            ledger['execution_failure']=dict(attempt_id=attempt_id,observed_steps=steps)
            save(self.root/'ledger.json',ledger)

    def _start(self, attempt_id, ledger):
        record=json.loads((self.root/(attempt_id+'.initial.json')).read_bytes())
        if hashlib.sha256(canonical(record)).hexdigest()!=ledger['starts'].get(attempt_id):
            raise ValueError('AB initial receipt changed')
        return record

    def verify_start(self, attempt_id, initial_bundle=None):
        manifest=self.manifest();ledger=self.ledger()
        if attempt_id in ledger['closed']:raise ValueError('AB attempt already closed')
        record=self._start(attempt_id,ledger)
        job=next(j for j in manifest['jobs'] if j['attempt_id']==attempt_id)
        if record['attempt_id']!=attempt_id or record['pair_id']!=job['pair_id']:
            raise ValueError('AB start receipt attempt/pair mismatch')
        ab_initial_check(manifest['references'][job['case_id']],record['snapshot'])
        for other in manifest['jobs']:
            if other['pair_id']==job['pair_id'] and other['attempt_id']!=attempt_id and other['attempt_id'] in ledger['starts']:
                ab_initial_check(self._start(other['attempt_id'],ledger)['snapshot'],record['snapshot'])
        if initial_bundle is not None and hashlib.sha256(canonical(initial_bundle)).hexdigest()!=record['initial_bundle_sha256']:
            raise ValueError('AB initial model observation changed')
        return record

    def finish(self, attempt_id, steps, outcome):
        with self.locked():
            self.verify_start(attempt_id);ledger=self.ledger();manifest=self.manifest()
            if type(steps) is not int or not 0<=steps<=600:
                raise ValueError('AB physical step cap')
            if sum(v['steps'] for v in ledger['closed'].values())+steps>manifest['physical_steps_cap']:
                raise ValueError('AB total physical step cap')
            if outcome not in ('done','give_up','call_limit','step_limit','official_termination','failed'):
                raise ValueError('AB explicit terminal outcome required')
            entries=[e for e in ledger['entries'] if e['attempt_id']==attempt_id]
            if ((outcome in ('done','give_up') and not entries) or
                    (outcome=='call_limit' and len(entries)!=self.call_cap) or (outcome=='step_limit' and steps!=600)):
                raise ValueError('AB terminal counters do not support outcome')
            ledger['closed'][attempt_id]=dict(steps=steps,outcome=outcome)
            if outcome=='failed':ledger['status']='stopped_execution_unknown'
            elif len(ledger['closed'])==len(self.manifest()['jobs']):ledger['status']='complete'
            save(self.root/'ledger.json',ledger)

    def deliver(self, bundle, archive, auth, send, invoke):
        with self.locked():
            attempt=auth['ab_attempt_id'];self.verify_start(attempt,archive[0]['bundle'] if archive else bundle)
            manifest=self.manifest();ledger=self.ledger();reserved=None
            def guarded_send(wire,key):
                nonlocal reserved
                local=json.loads(Path(auth['ab_episode_ledger_path']).read_bytes())
                row=local['entries'][-1]
                reserve=Decimal(row['reserve_usd'])
                prior=[e for e in ledger['entries'] if e['attempt_id']==attempt]
                if (row['request_id']!=key or row['wire_sha256']!=hashlib.sha256(wire).hexdigest()
                        or len(prior)!=bundle['turn'] or not reserve.is_finite() or reserve<0
                        or any(e['request_id']==key for e in ledger['entries'])
                        or len(ledger['entries'])>=manifest['requests_cap'] or len(prior)>=self.call_cap
                        or committed_cost(ledger['entries'])+reserve>Decimal(manifest['usd_cap'])
                        or committed_cost(prior)+reserve>Decimal(manifest['episode_usd_cap'])):
                    ledger['status']='stopped_budget_exhausted';save(self.root/'ledger.json',ledger)
                    raise ValueError('AB shared budget exhausted before provider send')
                reserved=dict(request_id=key,attempt_id=attempt,status='reserved',reserve_usd=str(reserve),estimated_usd=None,
                    wire_sha256=row['wire_sha256'],sequence=bundle['turn'],
                    authorization_sha256=bundle['authorization_sha256'],episode_ledger_path=auth['ab_episode_ledger_path'],
                    provider='OpenAI',model=SOL_SESSION.model)
                ledger['entries'].append(reserved);save(self.root/'ledger.json',ledger)
                return send(wire,key)
            try:
                result=invoke(guarded_send)
                reserved.update(status=result['status'],estimated_usd=result['estimated_usd'],usage=result['usage'],provider_id=result['provider_id'])
                # Provider usage can exceed the estimate: count it, then stop the batch.
                if result['estimated_usd'] is not None and Decimal(result['estimated_usd'])>Decimal(reserved['reserve_usd']):
                    ledger['status']='stopped_usage_exceeded_reservation'
                save(self.root/'ledger.json',ledger)
                return result
            except Exception:
                if reserved is not None:
                    if reserved.get('estimated_usd') is not None:
                        reserved['reserve_usd']=str(max(Decimal(reserved['reserve_usd']),Decimal(reserved['estimated_usd'])))
                    reserved['status']='unknown';ledger['status']='stopped_unknown_outcome'
                    save(self.root/'ledger.json',ledger)
                raise


class TransportABBatch(MemoryABBatch):
    """Two ordered arms with one ceiling, one reference and no implicit retries."""
    call_cap = 60
    case_id = 'handoff-transport-val000-object0'

    @classmethod
    def create(cls, root, batch_id, reference, *, mission, usd_cap, episode_usd_cap):
        validate_transport_mission(mission)
        ab_initial_check(reference, reference)
        if (not isinstance(batch_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,65}',batch_id)
                or reference['metadata']['case_id'] != cls.case_id
                or reference['metadata']['uid'] != 'tidy_house-sequential-val-0-0'
                or reference['metadata']['source_sha256'] != ab_source_fingerprint()
                or 'transport_context' not in reference['state']['environment']):
            raise ValueError('transport reference/source binding')
        total,episode = Decimal(str(usd_cap)),Decimal(str(episode_usd_cap))
        if not total.is_finite() or not episode.is_finite() or not 0 < episode <= total:
            raise ValueError('transport monetary ceilings')
        obj=cls(root);obj.root.mkdir(parents=True,exist_ok=False)
        save(obj.root/'manifest.json',dict(batch_id=batch_id,source_sha256=ab_source_fingerprint(),
            jobs=[dict(arm=arm,attempt_id=batch_id+'-'+arm,case_id=cls.case_id,pair_id=batch_id+'-pair',
                       profile=profile.name) for arm,profile in [('A',SOL_TRANSPORT_A),('B',SOL_TRANSPORT_B)]],
            references={cls.case_id:reference},transport_mission=mission,
            usd_cap=str(total),episode_usd_cap=str(episode),requests_cap=120,physical_steps_cap=1200,currency='USD'))
        save(obj.root/'ledger.json',dict(batch_id=batch_id,status='running',entries=[],starts={},closed={}))
        return obj

    def manifest(self):
        value=json.loads((self.root/'manifest.json').read_bytes())
        batch=value['batch_id']
        expected=[dict(arm=a,attempt_id=batch+'-'+a,case_id=self.case_id,pair_id=batch+'-pair',profile=p.name)
                  for a,p in [('A',SOL_TRANSPORT_A),('B',SOL_TRANSPORT_B)]]
        total,episode=(Decimal(str(value[k])) for k in ('usd_cap','episode_usd_cap'))
        if (value['source_sha256'] != ab_source_fingerprint() or value['jobs'] != expected
                or value['requests_cap'] != 120 or value['physical_steps_cap'] != 1200
                or value['currency'] != 'USD' or set(value['references']) != {self.case_id}
                or not total.is_finite() or not episode.is_finite() or not 0 < episode <= total):
            raise ValueError('transport shared contract changed')
        validate_transport_mission(value['transport_mission'])
        return value

    def validate_episode(self, auth, profile):
        value=self.manifest()
        job=next((j for j in value['jobs'] if j['attempt_id']==auth.get('ab_attempt_id')),None)
        if (job is None or job['profile'] != profile.name or auth.get('transport_mission') != value['transport_mission']
                or auth.get('attempt_ids') != {self.case_id:job['attempt_id']}
                or auth.get('case_ids') != [self.case_id]
                or Decimal(str(auth['usd_cap'])) != Decimal(value['usd_cap'])
                or Decimal(str(auth['per_case_usd_cap'])) != Decimal(value['episode_usd_cap'])
                or Path(auth.get('ab_episode_ledger_path','')).resolve() != self.root/(job['arm']+'-provider-ledger.json')):
            raise ValueError('transport episode differs from shared manifest')

    def _media(self, arm, steps):
        path=self.root/(arm+'-media.json')
        value=json.loads(path.read_bytes())
        if (value.get('case_id')!=self.case_id or value.get('arm')!=arm or value.get('physical_steps')!=steps
                or value.get('status') not in (('complete','no_physical_steps') if steps==0 else ('complete',))
                or (steps and not value.get('artifacts'))):
            raise ValueError('transport media receipt incomplete or misbound')
        for artifact in value['artifacts'].values():
            if hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()!=artifact['sha256']:
                raise ValueError('transport media artifact changed')
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def preflight(self, attempt_id):
        super().preflight(attempt_id)
        for prior,receipt in self.ledger()['closed'].items():
            arm=next(j['arm'] for j in self.manifest()['jobs'] if j['attempt_id']==prior)
            if self._media(arm,receipt['steps'])!=receipt['media_sha256']:
                raise ValueError('transport prior media receipt changed')

    def finish(self, attempt_id, steps, outcome):
        with self.locked():
            self.verify_start(attempt_id);ledger=self.ledger();manifest=self.manifest()
            arm=next(j['arm'] for j in manifest['jobs'] if j['attempt_id']==attempt_id)
            if type(steps) is not int or not 0<=steps<=600:
                raise ValueError('transport physical ceiling')
            if outcome not in ('done','give_up','call_limit','step_limit','official_termination'):
                raise ValueError('transport terminal outcome')
            entries=[e for e in ledger['entries'] if e['attempt_id']==attempt_id]
            if ((outcome in ('done','give_up') and not entries) or
                    (outcome=='call_limit' and len(entries)!=60) or (outcome=='step_limit' and steps!=600)):
                raise ValueError('transport terminal counters')
            if sum(v['steps'] for v in ledger['closed'].values())+steps>1200:
                raise ValueError('transport shared physical ceiling')
            ledger['closed'][attempt_id]=dict(steps=steps,outcome=outcome,media_sha256=self._media(arm,steps))
            if len(ledger['closed'])==2:ledger['status']='complete'
            save(self.root/'ledger.json',ledger)


SEARCH_PHASES=('search','approach','grasp','lift_rest')
SEARCH_REGION_LIMIT=12


def search_memory_schema(*, candidate_priority=False):
    """Opt-in model notes, not scene geometry or an executor command."""
    text=lambda limit:dict(type='string',maxLength=limit)
    region=dict(region_id=text(40),description=text(160),nearby=dict(type='boolean'),
        overhead_status=dict(type='string',enum=['unseen','partial','observed','uncertain']),
        observation_id=text(40),camera=dict(type='string',enum=['head','hand']))
    fields=dict(phase=dict(type='string',enum=list(SEARCH_PHASES)),
        region_updates=dict(type='array',maxItems=SEARCH_REGION_LIMIT,items=dict(
            type='object',additionalProperties=False,properties=region,required=list(region))),
        next_search_scope=dict(type='string',enum=['local','distant','not_search']),
        unviewed_nearby_region_ids=dict(type='array',maxItems=SEARCH_REGION_LIMIT,items=text(40)),
        budget_assessment=dict(type='string',enum=['sufficient','insufficient','uncertain']),
        budget_reason=text(240),cannot_observe_reason=text(240))
    if candidate_priority:
        candidate=dict(candidate_id=text(40),description=text(160),
            status=dict(type='string',enum=['unverified','excluded','deferred']),
            reason_kind=dict(type='string',enum=['appearance_support','identity_mismatch','access_blocked']),
            reason=text(240),observation_id=text(40),camera=dict(type='string',enum=['head','hand']))
        fields['next_search_scope']['enum'].append('candidate')
        fields.update(candidate_updates=dict(type='array',maxItems=SEARCH_REGION_LIMIT,items=dict(
                type='object',additionalProperties=False,properties=candidate,required=list(candidate))),
            focus_candidate_id=dict(type=['string','null'],maxLength=40),
            unresolved_candidate_ids=dict(type='array',maxItems=SEARCH_REGION_LIMIT,items=text(40)))
    return dict(type='object',additionalProperties=False,properties=fields,required=list(fields))


def _merge_search_regions(regions, updates):
    merged=deepcopy(regions)
    for row in updates:merged[row['region_id']]=deepcopy(row)
    if len(merged)>SEARCH_REGION_LIMIT:
        raise ValueError('search_memory region capacity; retain known IDs instead of dropping pending regions')
    return merged


def _merge_search_candidates(candidates, updates):
    merged=deepcopy(candidates)
    for row in updates:merged[row['candidate_id']]=deepcopy(row)
    if len(merged)>SEARCH_REGION_LIMIT:
        raise ValueError('search_memory candidate capacity; retain unresolved candidates')
    return merged


def validate_search_candidates(value, *, tool=None, known_ids=None, candidates=None, active_candidate_id=None):
    """Validate model evidence labels, never decide identity or reachability from scene truth."""
    updates=value['candidate_updates'];pending=value['unresolved_candidate_ids'];focus=value['focus_candidate_id']
    def candidate_id(x):return isinstance(x,str) and re.fullmatch(r'candidate-[A-Za-z0-9_-]{1,30}',x) is not None
    if not isinstance(updates,list) or len(updates)>SEARCH_REGION_LIMIT:
        raise ValueError('search_memory at most 12 candidate updates')
    if (not isinstance(pending,list) or len(pending)>SEARCH_REGION_LIMIT
            or any(not candidate_id(x) for x in pending) or len(pending)!=len(set(pending))
            or (focus is not None and not candidate_id(focus))):
        raise ValueError('search_memory distinct candidate IDs required')
    seen=set();kinds={'unverified':'appearance_support','excluded':'identity_mismatch','deferred':'access_blocked'}
    for row in updates:
        if (not isinstance(row,dict) or set(row)!={'candidate_id','description','status','reason_kind','reason','observation_id','camera'}
                or not candidate_id(row['candidate_id']) or row['candidate_id'] in seen
                or not isinstance(row['status'],str) or row['status'] not in kinds or row['reason_kind']!=kinds[row['status']]
                or any(not isinstance(row[k],str) or not row[k].strip() or len(row[k])>limit
                    for k,limit in (('description',160),('reason',240)))
                or not valid_id(row['observation_id']) or row['camera'] not in ('head','hand')):
            raise ValueError('search_memory candidate needs sensor evidence; access_blocked is deferred, not identity_mismatch')
        seen.add(row['candidate_id'])
        if known_ids is not None:
            if row['observation_id'] not in known_ids:
                raise ValueError('search_memory candidate observation must exist in this session')
            previous=(candidates or {}).get(row['candidate_id'])
            if previous is not None:
                order={oid:i for i,oid in enumerate(known_ids)}
                if order[row['observation_id']]<order[previous['observation_id']]:
                    raise ValueError('search_memory older evidence cannot replace newer candidate evidence')
    if value['next_search_scope']=='candidate' and focus is None:
        raise ValueError('search_memory candidate verification requires focus_candidate_id')
    if value['next_search_scope'] in ('local','distant') and focus is not None:
        raise ValueError('search_memory blind region search cannot claim a candidate focus')
    if candidates is None:return
    merged=_merge_search_candidates(candidates,updates)
    unresolved=sorted(k for k,row in merged.items() if row['status']!='excluded')
    if sorted(pending)!=unresolved:
        raise ValueError('search_memory list every unresolved candidate, including deferred candidates: '+','.join(unresolved))
    if focus is not None and (focus not in merged or merged[focus]['status']!='unverified'):
        raise ValueError('search_memory focus requires a sensor-supported unverified candidate')
    actionable=[k for k,row in merged.items() if row['status']=='unverified']
    if (active_candidate_id is not None and focus is not None and focus!=active_candidate_id
            and merged[active_candidate_id]['status']=='unverified'):
        raise ValueError('search_memory before switching candidate record identity mismatch or temporary access obstruction')
    if value['next_search_scope'] in ('local','distant') and actionable:
        raise ValueError('search_memory verify supported candidates before blind region search; record exclusion or temporary access obstruction first')
    if tool=='give_up' and actionable and value['budget_assessment']=='sufficient':
        raise ValueError('search_memory unresolved observable candidates remain and budget is sufficient; verify a candidate first')


def validate_search_memory(value, *, tool=None, known_ids=None, regions=None, observation=None,
                           candidate_priority=False, candidates=None, active_candidate_id=None):
    """Validate self-reports and provenance; never infer a view from a planned action."""
    fields=set(search_memory_schema(candidate_priority=candidate_priority)['required'])
    if not isinstance(value,dict) or set(value)!=fields:
        raise ValueError('search_memory requires phase, region updates, pending list and budget explanation')
    if value['phase'] not in SEARCH_PHASES:
        raise ValueError('search_memory phase must be search/approach/grasp/lift_rest')
    if value['next_search_scope'] not in (('local','distant','not_search','candidate') if candidate_priority else ('local','distant','not_search')):
        raise ValueError('search_memory next_search_scope')
    if value['budget_assessment'] not in ('sufficient','insufficient','uncertain'):
        raise ValueError('search_memory budget_assessment')
    for key in ('budget_reason','cannot_observe_reason'):
        if not isinstance(value[key],str) or len(value[key])>240:
            raise ValueError('search_memory short budget and robot-evidence reasons required')
    updates=value['region_updates'];pending=value['unviewed_nearby_region_ids']
    if not isinstance(updates,list) or len(updates)>SEARCH_REGION_LIMIT:
        raise ValueError('search_memory at most 12 region updates')
    def region_id(x):return isinstance(x,str) and re.fullmatch(r'region-[A-Za-z0-9_-]{1,32}',x) is not None
    if (not isinstance(pending,list) or len(pending)>SEARCH_REGION_LIMIT
            or any(not region_id(x) for x in pending) or len(set(pending))!=len(pending)):
        raise ValueError('search_memory pending region IDs must be distinct region-* IDs')
    seen=set()
    for row in updates:
        if (not isinstance(row,dict) or set(row)!={'region_id','description','nearby','overhead_status','observation_id','camera'}
                or not region_id(row['region_id']) or row['region_id'] in seen
                or not isinstance(row['description'],str) or not row['description'].strip() or len(row['description'])>160
                or type(row['nearby']) is not bool or row['overhead_status'] not in ('unseen','partial','observed','uncertain')
                or not valid_id(row['observation_id']) or row['camera'] not in ('head','hand')):
            raise ValueError('search_memory region needs a distinct ID, sensor description, view status and observation source')
        seen.add(row['region_id'])
        if known_ids is not None and row['observation_id'] not in known_ids:
            raise ValueError('search_memory observation must exist in this session before the proposed action')
        previous=(regions or {}).get(row['region_id'])
        if previous is not None and known_ids is not None:
            order={oid:i for i,oid in enumerate(known_ids)}
            if order[row['observation_id']]<order[previous['observation_id']]:
                raise ValueError('search_memory older evidence cannot replace a newer region observation')
    if candidate_priority:
        validate_search_candidates(value,tool=tool,known_ids=known_ids,candidates=candidates,active_candidate_id=active_candidate_id)
    reject_privilege(value)
    if len(canonical(value))>(16384 if candidate_priority else 8192):raise ValueError('search_memory byte limit')
    boundary=tool=='give_up' or value['next_search_scope']=='distant'
    if boundary and not value['budget_reason'].strip():
        raise ValueError('search_memory before give_up/distant search explain the current request, step and time budget')
    if regions is not None:
        merged=_merge_search_regions(regions,updates)
        expected=sorted(k for k,row in merged.items() if row['nearby'] and row['overhead_status']!='observed')
        if (boundary or candidate_priority) and sorted(pending)!=expected:
            raise ValueError('search_memory before give_up/distant search list every nearby region lacking an observed overhead view: '+','.join(expected))
        if any(k not in merged for k in pending):
            raise ValueError('search_memory pending ID has no sensor-supported region record')
        if (boundary and expected and value['budget_assessment']=='sufficient'
                and not value['cannot_observe_reason'].strip()):
            raise ValueError('search_memory nearby overhead views remain and budget is sufficient; choose a local observation or cite the robot evidence preventing it')
    if (observation is not None and value['budget_assessment']=='sufficient'
            and (observation['calls_remaining']<=0 or observation['steps_remaining']<=0)):
        raise ValueError('search_memory sufficient budget contradicts the observed remaining limits')
    return value


def search_memory_context(archive, *, candidate_priority=False):
    """Replay the same session's durable model transcript, without a second scene map."""
    known=[];regions={};latest=None;binding=None;candidates={};active=None
    for index,item in enumerate(archive):
        bundle=item['bundle'];obs=bundle['observation'];known.append(obs['observation_id'])
        current_binding=(bundle['position'],bundle['authorization_sha256'])
        if binding is None:binding=current_binding
        if (current_binding!=binding or bundle['turn']!=index or not valid_id(obs['observation_id'])
                or len(known)!=len(set(known))):
            raise ValueError('search_memory cross-session, reordered or duplicate observation history')
        call=item.get('call')
        if call is None:continue
        value=validate_search_memory(call.get('search_memory'),tool=call.get('tool'),
            known_ids=known,regions=regions,observation=obs,candidate_priority=candidate_priority,
            candidates=candidates if candidate_priority else None,active_candidate_id=active)
        regions=_merge_search_regions(regions,value['region_updates'])
        if candidate_priority:
            candidates=_merge_search_candidates(candidates,value['candidate_updates'])
            focus=value['focus_candidate_id']
            if focus is not None:active=focus
            elif value['next_search_scope'] in ('local','distant') or (active and candidates[active]['status']!='unverified'):
                active=None
        latest=dict(source_turn=bundle['turn'],phase=value['phase'],
            next_search_scope=value['next_search_scope'],budget_assessment=value['budget_assessment'],
            budget_reason=value['budget_reason'],cannot_observe_reason=value['cannot_observe_reason'])
    ordered=[regions[k] for k in sorted(regions)]
    context=dict(model_self_report=True,meaning='Sensor-cited model search notes, not verified coverage, object absence or clearance. A planned camera action never marks a region observed.',
        regions=ordered,overhead_observed_region_ids=[r['region_id'] for r in ordered if r['overhead_status']=='observed'],
        unviewed_nearby_region_ids=[r['region_id'] for r in ordered if r['nearby'] and r['overhead_status']!='observed'],
        latest_decision=latest)
    if candidate_priority:
        context.update(candidates=[candidates[k] for k in sorted(candidates)],active_candidate_id=active,
            unresolved_candidate_ids=sorted(k for k,row in candidates.items() if row['status']!='excluded'),
            candidate_meaning='Model evidence labels only. Deferred means identity unresolved and temporarily inaccessible, not excluded. Verify a supported candidate before blind region search; retain pending overhead views.')
    return context


def validate_search_decision(call, archive, observation, *, candidate_priority=False):
    context=search_memory_context(archive,candidate_priority=candidate_priority)
    return validate_search_memory(call.get('search_memory'),tool=call.get('tool'),
        known_ids=[x['bundle']['observation']['observation_id'] for x in archive]+[observation['observation_id']],
        regions={row['region_id']:row for row in context['regions']},observation=observation,candidate_priority=candidate_priority,
        candidates={row['candidate_id']:row for row in context['candidates']} if candidate_priority else None,
        active_candidate_id=context.get('active_candidate_id'))


def validate_active_memory(memory, known_ids=None):
    """Model-written notes only. Byte bound conservatively stays below 512 tokens."""
    keys = {'goal', 'lessons', 'pending', 'key_observation_ids'}
    if not isinstance(memory, dict) or set(memory) != keys:
        raise ValueError('memory requires goal, lessons, pending, key_observation_ids')
    if any(not isinstance(memory[k], str) or len(memory[k]) > 160 for k in keys-{'key_observation_ids'}):
        raise ValueError('memory text must be short strings')
    ids = memory['key_observation_ids']
    if (not isinstance(ids, list) or len(ids) > 3 or any(not valid_id(i) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('memory requires at most three distinct observation IDs')
    reject_privilege(memory)
    if len(canonical(memory)) > 512:
        raise ValueError('memory exceeds 512 UTF-8 bytes; shorten without automatic truncation')
    if known_ids is not None and not set(ids) <= set(known_ids):
        raise ValueError('memory refers outside the current attempt or into the future')
    return memory


def active_memory_context(archive):
    # Do not manufacture lessons from evaluation files or silently accept new IDs.
    known = []
    latest = None
    for item in archive:
        known.append(item['bundle']['observation']['observation_id'])
        call = item.get('call')
        if call is not None:
            validate_active_memory(call.get('memory'), known)
            latest = dict(source_turn=item['bundle']['turn'], model_self_report=True,
                          memory=deepcopy(call['memory']))
    return latest


def validate_handoff_plan(plan):
    """Trusted task specification, never a model-generated target registry."""
    cfg = FROZEN['task_handoff']
    if not isinstance(plan, dict) or set(plan) != {'initial_stage', 'initial_target', 'targets'}:
        raise ValueError('handoff plan fields')
    targets = plan['targets']
    if (plan['initial_stage'] not in cfg['stages'] or not isinstance(targets, dict) or not targets
            or len(targets) > 16 or plan['initial_target'] not in targets
            or any(not isinstance(k, str) or re.fullmatch(r'target-[0-9]{1,3}', k) is None
                   or not isinstance(v, str) or not v.strip() or len(v) > 160 for k, v in targets.items())):
        raise ValueError('handoff requires bounded task-declared targets and initial stage')
    reject_privilege(plan)
    return plan


def validate_task_handoff(value, *, tool=None, known_ids=None, context=None, plan=None):
    """Validate a proposal before dispatch. It is not an executed boundary yet."""
    if value is None:
        return
    cfg = FROZEN['task_handoff']
    keys = {'from_stage', 'to_stage', 'from_visit', 'target_id', 'reason',
            'depends_on_action', 'completion_claim', 'card'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('handoff proposal fields')
    if tool in ('done', 'give_up'):
        raise ValueError('terminal calls cannot propose another stage')
    if (value['from_stage'] not in cfg['stages'] or value['to_stage'] not in cfg['stages']
            or type(value['from_visit']) is not int or value['from_visit'] < 0
            or type(value['depends_on_action']) is not bool
            or value['reason'] not in ('completed', 'replan', 'target_change')
            or value['completion_claim'] not in ('unconfirmed', 'action_completed', 'action_failed')
            or not isinstance(value['target_id'], str)
            or re.fullmatch(r'target-[0-9]{1,3}', value['target_id']) is None):
        raise ValueError('handoff transition fields')
    if value['reason'] == 'completed' and not value['depends_on_action']:
        raise ValueError('completed transition must wait for execution')
    card = value['card']
    fields = {'goal', 'progress', 'lessons', 'pending', 'uncertainty', 'key_observation_ids'}
    if (not isinstance(card, dict) or set(card) != fields
            or any(not isinstance(card[k], str) or len(card[k]) > 160
                   for k in fields - {'key_observation_ids'})):
        raise ValueError('handoff card fields')
    ids = card['key_observation_ids']
    if (not isinstance(ids, list) or len(ids) > 3 or any(not valid_id(i) for i in ids)
            or len(set(ids)) != len(ids) or len(canonical(card)) > cfg['card_bytes_cap']):
        raise ValueError('handoff card exceeds bounds or has invalid observations')
    reject_privilege(value)
    if known_ids is not None and not set(ids) <= set(known_ids):
        raise ValueError('handoff references a foreign or future observation')
    if context is not None:
        if value['from_stage'] != context['stage'] or value['from_visit'] != context['stage_visit_id']:
            raise ValueError('handoff stale or repeated boundary')
        if plan is None or value['target_id'] not in validate_handoff_plan(plan)['targets']:
            raise ValueError('handoff target is not declared by the task')
        changed = value['target_id'] != context['target_id']
        if changed != (value['reason'] == 'target_change'):
            raise ValueError('handoff target change must be explicit; identity correction is not a new target')
        if not changed and value['to_stage'] == value['from_stage']:
            raise ValueError('same-stage actions do not trigger memory')


def task_handoff_context(archive, history, session):
    """Replay only bound public decisions/feedback. Pure and repeatable on wire rebuild.

    No evaluator fields, extra calls, or counters that can reset spending. The
    response-chain receipt is the exactly-once delivery gate; events are derived.
    """
    plan = validate_handoff_plan(session.handoff_plan)
    if (session.binding is None or session.binding.get('handoff_plan_sha256')
            != hashlib.sha256(canonical(plan)).hexdigest()):
        raise ValueError('handoff plan changed after session binding')
    if len(archive) != len(history):
        raise ValueError('handoff requires complete feedback before a boundary')
    context = dict(stage=plan['initial_stage'], stage_visit_id=0, target_id=plan['initial_target'])
    events, known = [], []
    for index, (item, executed) in enumerate(zip(archive, history)):
        b = item['bundle']
        if (b['turn'] != index or session.binding is None or b['position'] != session.binding['case_id']
                or b['authorization_sha256'] != session.binding['authorization_sha256']):
            raise ValueError('handoff cross-task or unordered archive')
        known.append(b['observation']['observation_id'])
        call = item.get('call')
        candidate = call.get('handoff') if call else None
        if candidate is None:
            continue
        validate_task_handoff(candidate, tool=call['tool'], known_ids=known, context=context, plan=plan)
        feedback = public_result(executed['result'])
        completed = (feedback.get('accepted') is True
                     and feedback.get('reason') in ('completed', 'ok', 'robot_ik_only')
                     and feedback.get('closure_command_completed') is not False)
        reject = None
        if executed.get('call') is None:
            reject = 'action_not_executed'
        elif candidate['depends_on_action'] and not completed:
            reject = 'dependent_action_not_completed'
        elif ((candidate['completion_claim'] == 'action_completed' and not completed)
              or (candidate['completion_claim'] == 'action_failed' and completed)):
            reject = 'completion_claim_contradicts_feedback'
        event = dict(episode_id=session.attempt_id, source_turn=index,
            feedback_ref=hashlib.sha256(canonical(executed)).hexdigest(),
            from_stage=context['stage'], to_stage=candidate['to_stage'],
            target_id=candidate['target_id'], stage_visit_id=context['stage_visit_id'] + 1,
            previous_visit_id=context['stage_visit_id'], reason=candidate['reason'],
            status='rejected' if reject else 'committed', rejection=reject,
            action_completed=completed, semantic_completion='unconfirmed', model_self_report=True,
            coordinate_policy='historical_only; reobserve before metric actions',
            card=deepcopy(candidate['card']))
        event['card_bytes'] = len(canonical(candidate['card']))
        event['handoff_id'] = hashlib.sha256(canonical(event)).hexdigest()
        events.append(event)
        if not reject:
            context.update(stage=event['to_stage'], stage_visit_id=event['stage_visit_id'],
                           target_id=event['target_id'])
    # A boundary is fresh only for the immediately following request, not every turn.
    fresh = events[-1] if events and events[-1]['source_turn'] == len(archive)-1 and events[-1]['status'] == 'committed' else None
    return dict(current=context, events=events, handoff=deepcopy(fresh))


def dev_instructions(instructions, profile):

    if profile.view_occlusion_guidance:

        instructions += ('\nROBOT VIEW OCCLUSION: At the official rest arm posture, head pan about -0.9 to +0.4 rad '

            'has 20-60% arm occlusion. Pan <= -1.0 or >= +0.75 is largely clear. To see ahead, first turn '

            'the base with move_base then observe using a clear pan, or look sideways first. '

            'Do not move the arm just to clear the view: free arm motion near furniture risks contact.')

    if profile.approach_point:

        instructions = instructions.replace('Near obstacles, use |forward_m| <= 0.3 m per move_base call; ', '')

        instructions += ('\napproach_point(camera,pixel,observation_id,standoff_m): use the latest observation. '

            'Reads one surface point, turns toward it, then drives horizontal distance minus standoff, clipped '

            'to [-0.5,1.5] m. Standoff is 0..0.95 m measured from base center to the selected surface. '

            'The chassis front extends 0.287 m; furniture edges may be closer than the target. Historical '

            'successful grasp distances were 0.60-0.95 m, median about 0.77 m. For floor/doorway points '

            'standoff may be zero. Check the whole travel corridor using current robot RGB-D first. '

            'Forward >0.75 m splits into two equal drives, each capped at 80 steps; only the first turns. '

            'There is no observation between segments and no automatic look afterward. A failed segment '

            'stops the tool. All steps share the 600-step budget and preserve 122 rest steps. '

            'odometry_prediction is a propagated old surface point, not a new observation. One tool slot.')

    if profile.approach_point:
        instructions = instructions.replace(
            'First use check_path on both pregrasp and grasp poses to test robot-only reach, before deciding to approach further.',
            'Use check_path on your chosen pregrasp and grasp poses before arm execution. When the target is distant, plan a sensor-supported base approach before repeatedly guessing distant arm poses.')
        instructions += ('\nR1 APPROACH POLICY: After identifying a plausible target, use locate_points to query the target surface and the furniture/corridor points needed for the next decision together. '
            'For a direct approach supported by current RGB-D, prefer approach_point over a series of small move_base calls. '
            'Choose the pixel and standoff yourself from current evidence; the tool does not identify the target, choose a safe stopping point, or certify reachability. '
            'Furniture clearance can require a larger standoff than reach alone suggests. Do not move just to satisfy a step target. '
            'If a straight approach is unsupported, use move_base for a side route or look for missing evidence, and state why. '
            'The 80-step segment cap may end before arrival, especially after a large turn; read actual displacement and residuals before deciding the next action. '
            'After motion, localize from fresh RGB-D, check your proposed poses, and proceed when evidence is sufficient. '
            'Repeat a query or adjust the base when new evidence or a specific unresolved question justifies it, not merely to repeat a failed reach test. '
            'This policy does not remove corridor checks, authorize blind motion, skip pregrasp, or weaken grasp/rest gates.\n')
    if profile.multi_locate:

        instructions += ('\nlocate_points(camera,pixels,observation_id): 1..6 integer [x,y] pixels in the latest '

            'observation, returned in order with the same fields as locate_point. Zero motion steps and one '

            'tool slot. locate_point remains available. Set pixels=null for other tools.')

    if profile.approach_point:

        instructions += '\nSet standoff_m=null for every tool except approach_point.'

    return instructions





def pick20_instructions(instructions):
    """New profile only: describe the frozen Full B public execution contract."""
    replacements = {
        'move_base(turn_rad,forward_m)': 'move_base(turn_rad,forward_m) turns then drives in the command-entry frame; negative distance reverses. Public limits are +/-pi rad and +/-1.5 m. Full B uses at most 150 control steps, speed targets up to 0.6 rad/s and 0.25 m/s, with completion tolerances 0.02 rad and 0.015 m. No scene collision filtering is performed.',
        'check_path(poses)': 'check_path(poses) solves robot-only IK at the current torso height. It does not test scene collisions or certify the executed trajectory. Both check_path and locate_point are read-only, consume zero environment steps and one tool slot. IK failure means no solution found, not proof of unreachability.',
        'move_to uses one pose;': 'move_to uses one pose; move_eef_chunk uses 1..8 poses sharing at most 150 steps including settling. set_gripper accepts only 1=open or 0=closed. Closing first aligns to the preceding motion target and can fail even after a completed move: the close gate is stricter. closure_command_completed reports completion of closure, not object possession. return_to_rest retains the gripper and uses at most 122 steps including settling. Lifting or a completed command alone is not strict success.',
    }
    lines = instructions.splitlines()
    for prefix, replacement in replacements.items():
        matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError('pick20 prompt source changed: '+prefix)
        lines[matches[0]] = replacement
    instructions = '\n'.join(lines).replace('2048 by 2048', '640 by 640')
    instructions = instructions.replace('Base calls remain capped at 80 steps', 'Base calls remain capped at 150 steps')
    instructions = instructions.replace('All actual motion starts with measured base settling, consuming the same budget;', 'Motion consumes the shared step budget;')
    instructions = instructions.replace('Observe actual residuals and step breakdown before planning further.',
        'Inspect command reason, steps_used, current sensors and odometry before planning further.')
    instructions = instructions.replace('Read actual movement,\nresiduals and remaining steps, especially when arrived=false.',
        'Read current sensors, odometry, command reason and remaining steps after every move.')
    return instructions + '\nFULL B CONTRACT: accepted=true means a command was accepted, not that its goal or the task succeeded. Check reason and closure_command_completed where present; no missing arrived or residual field is evidence of success.\n'


def build_session_wire(body, meta, current, bundle, archive, profile, session):
    if profile not in (SOL_SESSION, SOL_MEMORY, SOL_HANDOFF, SOL_TRANSPORT_A, SOL_TRANSPORT_B, SOL_PICK20, SOL_PICK20_ALIGNED)+DEV_PROFILES+WEEKLY_PROFILES or not isinstance(session, ResponseSession):

        raise ValueError('Sol session requires its explicit profile and attempt state')
    if profile.transport_task != (session.transport_mission is not None):
        raise ValueError('transport mission requires the explicit transport profile')
    session.bind(bundle, archive)
    settings = FROZEN['sol_session']
    for old, new in settings['prompt_replacements'].items():
        if old not in body['instructions']:
            raise ValueError('session prompt replacement source changed')
        body['instructions'] = body['instructions'].replace(old, new)
    body['instructions'] += settings['prompt_addition']
    if profile in (SOL_PICK20,SOL_PICK20_ALIGNED)+DEV_PROFILES or profile.weekly_budget:

        body['instructions'] = pick20_instructions(body['instructions'])
    body['instructions'] = dev_instructions(body['instructions'], profile)

    if profile.grasp_alignment:
        body['instructions']=body['instructions'].replace('at most 25 tool calls','at most 40 tool calls').replace(
            'Closing first aligns to the preceding motion target and can fail even after a completed move: the close gate is stricter.',
            'Closing aligns to the locked prepare_grasp/replan_grasp goal, never a later generic movement goal.')
        body['instructions'] += ('\nGRASP ALIGNMENT V1: Use prepare_grasp with one pose for the final grasp localization. '
            'Use replan_grasp with one pose and an evidence-based note to explicitly revise that goal; '
            'the previous and new goals are recorded. Generic move_to/move_eef_chunk cannot change the locked grasp goal. '
            'Do not replace a grasp point with current TCP merely to satisfy the gate. '
            'Grasp localization and closing both require position error <=0.002 m, orientation error <=0.02 rad '
            'and torso change <0.001 m per control tick. Localization allows up to 150 movement steps '
            'plus 40 additional convergence steps (early exit); closing rechecks the locked goal. '
            'A timeout is not arrival. These internal steps do not use extra model calls. '
            'compliance_blocked means the controller estimates sustained contact-related target deflection '
            'while the original goal is not reached; it does not identify a contacted object or prove contact. '
            'Do not close or retarget to the current TCP to bypass it. Use fresh RGB-D to decide whether '
            'to retreat and explicitly replan, or return_to_rest. No recovery motion is automatic. '
            'All non-rest motions preserve 122 of the 600 steps for return_to_rest; '
            'this reserves budget, not guaranteed physical rest. Preserve a final tool slot for return_to_rest. '
            'tool_motion_limit means only this movement allowance ended; rest_motion_limit means only '
            'this rest attempt allowance ended. Neither means the official task has ended. '
            'Read steps_remaining. After tool_motion_limit use fresh sensors to replan if budget allows '
            'or request return_to_rest; never assume arrival or close merely because motion stopped. '
            'On rest_budget_reserved or insufficient_grasp_and_rest_budget, request return_to_rest. '
            'No gate certifies object possession.\n')
    if profile.weekly_budget:
        body['instructions'] = weekly_instructions(body['instructions'], profile)
    if profile.r1_extended_budget:
        body['instructions'] = re.sub(r'\b600\b', str(profile.step_cap), body['instructions'])
        body['instructions'] = body['instructions'].replace('at most 40 tool calls', f'at most {profile.call_cap} tool calls')
        body['instructions'] = body['instructions'].replace('Reserve up to 150 steps for rest', 'Reserve 122 steps for rest').replace(
            'Reserve up to 150 of the official steps for return_to_rest', 'Reserve 122 of the shared steps for return_to_rest')
        body['instructions'] += ('\nEXTENDED R1 BUDGET: Search, approach, grasp, recovery and rest share '
            f'{profile.step_cap} control steps and {profile.call_cap} model calls. '
            'One control step is 0.05 simulated seconds. API waiting consumes wall time, not control steps. '
            'Use the remaining counts in each observation; the extended horizon does not change success or force limits.\n')
    if profile.r15_protocol:
        text=body['instructions']
        text=text.replace('122','300')
        text=text.replace('at most 150 steps','at most 300 steps')
        text=text.replace('Full B uses at most 150 control steps','Full B uses at most 250 control steps')
        text=text.replace('Base calls remain capped at 150 steps','Base calls remain capped at 250 steps')
        text=text.replace('at most 40 steps including settling','at most 150 steps including settling')
        text=text.replace('Reserve up to 150 steps for rest','Reserve 300 steps for rest')
        text=text.replace('Reserve up to 150 of the official steps for return_to_rest','Reserve 300 of the shared steps for return_to_rest')
        if profile.r15_standoff:
            # Remove obsolete D2 splitting/approach guidance as a whole.
            text=re.sub(r'approach_point\(camera,pixel,observation_id,standoff_m\):.*?(?=\n|$)','',text)
            text=re.sub(r'For a direct approach supported by current RGB-D, prefer approach_point[^.]*\.','',text)
        body['instructions']=text + ('\nR1.5 PROTOCOL: look timeout 150 steps; move_base/approach_point '
            '250 steps per segment; move_to/move_eef_chunk share 300 steps per call; return_to_rest '
            '300 steps total; reserve 300 shared steps for rest. These are timeout ceilings, not '
            'planning constraints or safety margins. Do not reject useful plans solely because '
            'their estimated motion duration approaches a per-tool ceiling. Shared 1600-step and '
            '50-call limits still apply; no automatic retry. Furniture safety uses short observed '
            'distance segments, not artificially short action durations.\n')
    if profile.r2b_clearance:
        body['instructions'] += ('\nBefore lifting, exiting, retracting or pregrasp motion near furniture or known obstacles, call check_clearance with 1..8 poses or rest=true (poses=[]). It costs one call and zero steps. This tool only reports sensor evidence, never vetoes motion. If clearance is below 0.05 m or unknown, use move_eef_chunk with intermediate waypoints or change direction. Unobserved space is unknown, never free. Future tracking error remains unknown.\n')
    if profile.r2d_extended_calls:
        body['instructions']=body['instructions'].replace('50-call limits','80-call limits')
    if profile.r2d_base_clearance:
        body['instructions'] += '\nBefore move_base near furniture, call check_clearance with poses=[], rest=false, and explicit turn_rad/forward_m. This is a zero-step, one-call advisory sensor sweep; unknown is not free and no motion is rejected. needs_base_reposition supplies turn_rad and forward_m suggestions, not a cleared route.'
    if profile.base_tracking_exit:
        body['instructions'] += '\nA move_base path deviation or stalled progress stops only that command, reports completed=false and requires_replan=true with actual forward/lateral displacement. Old localization and clearance are invalid: reassess RGB-D and first restore furniture spacing using an explicitly chosen move_base if needed. There is no automatic retreat or lockout of later tools.'
    if profile.arm_tools_lock_base:
        body['instructions'] += '\nArm tools and test lifts use fixed-base planning and zero base drive. needs_base_reposition means private fixed-base IK did not resolve the target; it is not proof of unreachability. Suggested move_base displacement/turn is robot geometry only, not an obstacle-free route; inspect RGB-D before explicitly moving. The legacy planner prefers wrist-to-shoulder horizontal radius 0.50–0.65 m and bearing within 0.6 rad. Within this band radial/bearing limits alone require no base movement. This band alone does not guarantee fixed-base reach for every orientation or joint state. check_path tests fixed-base robot IK. Any changed pose or route invalidates the prior clearance report. Unexpected base displacement requires reassessment before continuing.'
    if profile.arm_base_explicit:
        body['instructions'] += '\nArm and grasp tools do not implicitly drive the base. base_motion_required means the proposed drive was not dispatched and the arm target was not reached. Use current RGB-D and odometry to explicitly plan move_base if needed, or choose another arm target. A changed route needs a new check_clearance; never treat an incomplete preview as clear.'
    if profile.error_progress_guard:
        body['instructions'] += ('\nMotion may stop with motion_stalled when current-target error does not decrease over its measured window, even if the arm is moving. This includes return_to_rest using measured joint residuals. error_progress reports the error trend; waypoint_progress refers to the current waypoint. This is incomplete motion, not arrival or a contact diagnosis. Inspect the new observation and explicitly replan; there is no automatic retreat or retry.\n')
    elif profile.motion_progress_only:
        body['instructions'] += ('\nCartesian motion_stalled means the current waypoint made no measurable progress after its nominal trajectory. It is not arrival. waypoint_progress uses one-based waypoint_index (null only when all completed); residual and requested pose refer to that current waypoint, not the final requested waypoint. Inspect current sensor evidence and explicitly replan position, orientation or approach; no waypoint is skipped and no automatic retry occurs.\n')
    if profile.dual_reachability:
        body['instructions'] += '\ncheck_path reports ordinary_move reachability (0.010 m / 0.05 rad) separately from strict_grasp reachability (0.002 m / 0.02 rad). Ordinary reachability does not certify grasp alignment or scene clearance. IK unresolved is not proof of impossibility.'
    if profile.clearance_verification:
        body['instructions'] += '\nA clearance report marked unverified has mostly unobserved relevant links, stale evidence or an incomplete preview. Turn the head to observe the swept region, then refresh the report. A positive distance in such a report is not a verified free route; the report remains advisory.'
    if profile.blocked_grasp_direction:
        body['instructions'] += '\nOn blocked grasp feedback, grasp_error_base_m is commanded TCP minus measured TCP in the current base frame. If a support surface blocks approach, do not repeat millimetre adjustments in place: change approach direction, reposition to a wrist-to-shoulder horizontal radius of 0.50–0.65 m, or change yaw. Use current RGB-D to choose, not a fixed recovery sequence.'
    if profile.r2b_test_lift:
        body['instructions'] += ('\nAfter closing, first test-lift 0.02–0.03 m. Check the hand image and finger distance for possession before transport. grip_lost stops the current action; use the new observation to decide the next action.\n')
    if profile.r2a_identity and not profile.identity_attempt_recovery:
        body['instructions'] += ('\nR2a identity: appearance cards below describe categories only, never locations. '
            +('Call verify_target before closing and before done; after a fresh physical observation you may explicitly review again, using the CURRENT observation_id. ' if profile.identity_confirmation else 'Call verify_target exactly once before closing and once before done, using the CURRENT observation_id. ')+
            'Supply 1..3 candidate camera/pixel markers from original robot images. Candidate 0 is the object '
            'you intend to close on (before_close), or the held object (before_done); include visible alternatives. '
            'The independent comparison may select any offered candidate, or none. Before closing, use only its selected candidate; cite its review observation ID and literal candidate N in the close note. You may reposition to that candidate after review. A before_done review must select candidate 0, the held object. '
            +('Verification and its decision both count inside '+str(profile.call_cap)+' total requests. The before_done permit expires after physical movement; verify after completing rest. A null result grants no close/done permit. No automatic review or retry.\n')+json.dumps(r2a_appearance_cards(profile),ensure_ascii=False))
    if profile.identity_confirmation:
        body['instructions'] += ('\nIdentity confirmation: selected_index ranks candidates; it does not confirm identity. '
            'done requires a current before_done review selecting held candidate 0 with identity_status=confirmed. '
            'uncertain or rejected cannot permit done, even after successful grasping and rest. '
            +('When uncertain, obtain the specific missing category feature or correspondence evidence, then explicitly verify again. '
              if profile.identity_multiview_evidence else 'When uncertain, first obtain an unobstructed side view that reveals the missing label/body proportions, then explicitly verify again. ')+
            'Fresh-view reviews may repeat within the shared total call budget; no automatic request or motion. '
            'If inspection cannot resolve identity or contradicts the target, use robot depth and clearance to choose a supported placement, '
            'lower the held object onto it, explicitly open, check release, and then verify another candidate. '
            'Do not drop an unsupported object or automatically retreat. Opening revokes old close/done permits. '
            'Closing also requires identity_status=confirmed; uncertain is not a close permit. Obtain the missing view with head/hand observation before closing.\n')
    if profile.identity_multiview_evidence:
        body['instructions'] += ('\nMultiview identity evidence: verify_target still uses CURRENT identity_candidates. '
            'Set identity_evidence to [] or at most three earlier robot view references: observation_id, camera, pixel and candidate_index. '
            'Each historical pixel marks the object you propose is the SAME as that current candidate; this is only a proposal for independent visual checking. '
            'Choose your best complementary face/side/detail views from this session, including earlier non-review observations; keep their exact observation IDs and pixels in your notes. '
            'Do not map a different candidate, an untracked object, or a historical item to the held object merely by color, location or ranking. '
            'The reviewer receives the original images together, not your identity conclusions. Other tools use identity_evidence=null. '
            'Before spending another review, check whether the new view actually resolves its last missing feature; a new observation ID alone adds no evidence. '
            'Clear category-specific text OR a distinctive category illustration/structure with compatible shape may suffice; you need not collect text, side and thickness together. '
            'If the current view or cross-view identity remains uncertain, change the inspection route or preserve the unresolved candidate. '
            'Do not take an uncertain object solely for inspection: close and done still require confirmed, and before_done still needs fresh held-object evidence.\n')
    if profile.rest_joint_target:
        body['instructions'] += ('\nRest uses the exact robot joint posture as its reference, with the existing compliance and arrival gates. The swept path differs from a task-space rest pose: request a fresh check_clearance(rest) before retracting near obstacles. A preview does not certify future tracking or unseen geometry.\n')
    if profile.identity_attempt_recovery:
        body['instructions'] += ('\nIdentity recovery: verification belongs to a candidate and a grasp attempt, not to the whole episode. '
            'Use verify_target on the CURRENT observation with 1..3 original robot camera/pixel candidates. '
            'Before_close candidate 0 is your intended object; before_done candidate 0 is the held object. Include visible alternatives. '
            'A selected before_close candidate grants identity_recovery.current_reference. Copy that exact identity_reference '
            'to close, before_done, done, and any prepare_grasp/replan_grasp for that same candidate; other tools use null. '
            'Each new before_close review requires an explicit prepare_grasp/replan_grasp with that reference '
            'before closing, even if an old grasp pose remains locked; keep the unchanged arrival gates. '
            'A prepare/replan with null abandons an unused close permit and requires a new before_close review. '
            'A review may select another offered candidate or none; null grants no permission. Never reuse a reference for another object. '
            'After a test lift, if identity is uncertain, use robot views to inspect the held object; a before_done review may be used '
            'for this check, but physical movement invalidates its done permit, so review again after final rest. '
            'If the held object is wrong or unsupported, budget permitting, choose an observed support surface, plan the placement '
            'with current RGB-D and clearance, lower it onto that support and explicitly open the gripper. Confirm release in images; '
            'an open-command ACK alone does not prove stable placement. Then select another candidate, refresh localization using '
            'replan_grasp as needed, and request a new before_close review. Opening invalidates all prior identity permits. '
            'Do not switch candidates while an object may still be held. Partial or unknown opening is not confirmed release. '
            'You may explicitly request another review after obtaining a new physical observation; do not repeat a review at the same step. '
            'Review disagreement is evidence to resolve, not an instruction to give_up. There is no automatic placement, motion or API retry. '
            'Every main and review request shares the unchanged '+str(profile.call_cap)+' total-request budget, step/wall and dollar limits. '
            'Only a current before_done review selecting the held candidate 0 permits done. Appearance cards describe categories only:\n'
            +json.dumps(r2a_appearance_cards(profile),ensure_ascii=False))
    if profile.search_recessed_view:
        body['instructions'] += ('\nRECESSED VIEW SEARCH: Nearby sinks, basins, cabinet compartments and rimmed or occluded regions are uncertain candidates when robot RGB-D supports them. Before abandoning a local area, approach its visible edge only along an inspected route, then choose a larger downward head view or a reachable hand-camera view above the opening looking down. Choose the region and viewpoint from current sensors; no fixed scene position or scripted sequence is supplied. Before raising the arm near furniture call check_clearance; an unverified report needs a better view. Existing motion_stalled exits stop only the current motion; they do not prove a viewpoint safe. No zoom tool is added.\n'
            'SEARCH MEMORY: Every response supplies search_memory and a self-reported phase search/approach/grasp/lift_rest. Region updates cite an existing observation_id and head/hand image showing the evidence. Use stable region-* IDs, sensor descriptions, nearby and overhead_status. observed means the interior was actually visible from above in the cited image, not that a look/move is planned or merely completed. partial/uncertain views remain pending; looking at a rim does not clear the hidden interior. Omitted updates retain older records. Current search_memory repeats the durable seen and pending lists; all are model self-reports, not verified scene coverage.\n'
            'Before give_up or next_search_scope=distant, list every nearby region not overhead-observed in unviewed_nearby_region_ids. Explain the actual remaining call/step budget and known time constraint in budget_reason; unknown time is unknown. If a useful local view remains and budget is sufficient, continue observing. If it cannot be observed, cite the current robot-sensor, reachability or execution evidence in cannot_observe_reason. Do not mark a region distant or observed merely to avoid this checkpoint. No automatic motion, retry, added budget or changed arrival threshold is authorized.\n'
            'CONTAINER GRASP: From head/hand depth, estimate the visible container bottom or supporting surface and leave explicit fingertip/grasp clearance above that measured surface while obtaining enough object insertion between the fingers. If the bottom is occluded or depth ownership uncertain, obtain another view; do not invent a bottom height, use a scene coordinate, or force a grasp through the support.\n')
        if profile.search_candidate_priority:
            body['instructions']=body['instructions'].replace(
                'Before abandoning a local area, approach its visible edge only along an inspected route, then choose a larger downward head view or a reachable hand-camera view above the opening looking down.',
                'When no supported visible candidate can be further checked, inspect unresolved rimmed regions: approach the visible edge only along an inspected route, then choose a larger downward head view or a reachable hand-camera view above the opening looking down.')
            body['instructions'] += ('\nCANDIDATE PRIORITY: First verify a candidate already supported by appearance evidence in robot images; '
                'do not interrupt it merely because a nearby sink, basin or cabinet region lacks an overhead view. '
                'Use a view, measurement or identity comparison that distinguishes its shape, color, proportions or visible text; '
                'one shared color alone does not establish identity. Candidate verification and approach use next_search_scope=candidate '
                'and focus_candidate_id, even when the candidate is farther away. local/distant mean blind region search without a candidate clue. '
                'Candidate scope bypasses the pending-container search boundary, while every pending region remains listed. '
                'Record candidate_updates with stable candidate-* IDs, a sensor description, observation_id, head/hand camera and concrete reason. '
                'unverified/appearance_support keeps a candidate under consideration; excluded/identity_mismatch requires visible contradictory '
                'identity evidence. deferred/access_blocked means still unverified but currently unable to observe or approach, based on robot '
                'reachability, clearance or execution evidence; it never means not the target. New observations may restore deferred candidates '
                'to unverified. These are model self-reports, not verified identity or safety. '
                'Before switching focus or leaving for blind region search, explain the prior candidate by updating it to excluded or deferred; '
                'do not invent a mismatch to escape this priority. If no candidate can currently be checked, retain the original edge-and-overhead '
                'method for the unresolved regions. Do not stare indefinitely at the first guess: obtain discriminating evidence and exclude '
                'a wrong candidate when supported. On every response list all unresolved_candidate_ids, including deferred ones, and all '
                'unviewed_nearby_region_ids. focus_candidate_id=null with not_search keeps the current focus; manipulation need not change it. '
                'Before give_up review both unresolved candidates and unobserved regions against remaining budget. When budget is sufficient '
                'and a candidate remains unverified, continue its verification; insufficient budget may justify giving up but never silently '
                'erases unresolved candidates. No new automatic action, verification allowance, threshold or target location is introduced.\n')
    if profile.r2a_close:
        body['instructions'] += ('\nR2a G1: after prepare_grasp/replan_grasp reports compliance_blocked, you may explicitly '
            'set close_after_blocked=true on set_gripper=0 only when current locked-goal error is <=0.020 m and <=0.1 rad. '
            'The note must cite the current observation ID and HAND image evidence that the object is between both fingers. '
            'This is an explicit experimental close, not strict arrival. Other close requests set false; other tools null. '
            'The feedback preserves baseline reason words and reports arrived plus measured residuals; closure is not possession.\n')
    if profile.structured_close_evidence:
        body['instructions']=body['instructions'].replace(
            'The note must cite the current observation ID and HAND image evidence that the object is between both fingers. ',
            'For every close request, supply close_evidence={fingers_bracket_object: true, observation_id: CURRENT observation ID} only if the current HAND image shows the object bracketed by both fingers. False or stale evidence cannot close. The note explains the intent but is not parsed for this visual claim. Other tools use null. ')
    if profile.identity_confirmed_persistence:
        body['instructions']=body['instructions'].replace('A null result grants no close/done permit. ',
            'A null result creates no new identity confirmation. ')
        body['instructions'] += ('\nI1 identity continuity: keep the original supported-candidate standard and exactly two reviews (before_close and before_done). '
            'Confirmed identity for the SAME candidate survives a later uncertain view. Explicit rejected with visible counterevidence revokes it. '
            'For a later review of that same object, supply same_identity_candidate with the original confirmed observation_id/candidate_index, '
            'current_candidate_index and concrete robot-view association_evidence. Matching index, color or location alone is not an association. '
            'Use null when association is uncertain or the candidate differs. Before_done requires current hand candidate 0, maintained grasp and a fresh review; '
            'historical identity alone is not possession. Released or lost-object associations cannot be reused. No extra review allowance, automatic re-review, or retry.\n')
    if profile.r15_completion:
        body['instructions'] += ('Motion feedback status is arrived, step_cap_reached, blocked or '
            'official_terminated, with requested/actual/residual robot measurements. Arrival is '
            'tool completion, never proof of target identity, possession or task success. '
            'After step_cap_reached or blocked, inspect fresh sensors and replan; no automatic retry.\n')
    if profile.r15_standoff:
        body['instructions'] += ('approach_point executes ONE segment toward standoff_m+0.30 m '
            'when farther away; once within that margin a NEW call after fresh observation may '
            'cover the last <=0.30 m to standoff_m. Never chain a blind final segment. Near furniture '
            'use <=0.30 m per final move_base segment and leave at least 0.10 m clearance beyond '
            'the robot body. Historical no-contact segments undershot by about 0.013 m and '
            'drifted laterally up to 0.024 m; these are observations, not guaranteed bounds. '
            'Account for chassis, arm and uncertainty using current RGB-D.\n')
    if profile.r15_pregrasp:
        body['instructions'] += ('Open-gripper move_to/move_eef_chunk uses a Cartesian pregrasp '
            'transition for each proposed pose: rise at current xy to max(current z,target z+0.15 m), '
            'move horizontally at that height, then descend. It uses your proposed pose, no scene '
            'candidate generation; inspect overhead space first. prepare_grasp itself retains its '
            'strict 0.002 m/0.02 rad final localization gate.\n')
    if profile.r15_carry:
        body['instructions'] += ('With the gripper closed, return_to_rest first lifts 0.10 m from '
            'the current pose, retracts 0.15 m along robot -x, then rests, sharing its 300-step '
            'timeout. Do not duplicate that prelude in anticipation of this tool. If rest fails '
            'to arrive, first use fresh images and an explicit arm pose change; an unchanged '
            'rest retry is rejected. No pose change or retry is automatic after failure.\n')
    if profile.rest_progress_feedback:
        body['instructions'] += ('\nREST PROGRESS FEEDBACK: When return_to_rest reports '
            'rest_progress_feedback, compare its measured entry and last acknowledged state, '
            'remaining rest error and exit velocity with fresh robot RGB-D. '
            'Arm start_error_rad/end_error_rad and torso start_error_m/end_error_m use '
            'target minus measured joint position. Positive l2_improvement_rad or '
            'absolute_error_improvement_m means less rest error; negative means more. '
            'These are rest-target errors, not servo tracking errors. '
            'Use arm.exit_velocity_rad_s only as measured joint velocity. '
            'rest_motion_limit ends this attempt allowance; it does not establish rest, possession, '
            'a safe stop or a blocked scene path. Exit velocity is the last measured velocity, '
            'not proof that later residual motion has stopped. If progress is poor, use current '
            'sensor evidence to choose an intermediate retreat or a changed arm pose before '
            'requesting rest again. Continuing the same rest path requires evidence of useful '
            'progress and adequate remaining budget. Missing or unavailable feedback is unknown. '
            'Do not infer object identity, contact identity or safe clearance from these robot '
            'measurements. No retreat, retry, rejection or braking is automatic. Preserve the '
            'locked grasp goal, strict arrival gates, target confirmation and recovery accounting.\n')
    if profile.transport_task:
        original = "Pick the specified object and carry it to the robot's rest posture."
        if original not in body['instructions']:
            raise ValueError('transport prompt source changed')
        body['instructions'] = body['instructions'].replace(original,
            'Complete the entire transport mission: approach the described object, pick it up, '
            'carry it to the described destination, release it there, then return to rest and stop. '
            'Picking alone is not mission success. The place stage includes navigation while carrying. '
            'A closed-gripper command does not prove possession; use current sensors. '
            'Do not call done at an intermediate stage.')
        body['instructions'] = body['instructions'].replace('at most 25 tool calls', 'at most 60 tool calls')
        body['instructions'] = body['instructions'].replace('only a target category', 'only semantic task descriptions')
        body['instructions'] = body['instructions'].replace('same official Pick spawn', 'native sequential task start')
    if profile.active_memory:
        body['instructions'] += FROZEN['memory_ab']['prompt_addition']
    if profile.task_handoff:
        body['instructions'] += FROZEN['task_handoff']['prompt_addition']
    if profile.lean_feedback:
        if profile not in V3_PROFILES:
            raise ValueError('approved lean prompt requires the exact S1 profile')
        # Replace historical patch paragraphs only for this explicit profile.
        body['instructions'] = v3_lean_prompt()
    if profile.segmented_exit_guidance:
        body['instructions'] += S2_SEGMENTED_GUIDANCE
    delta = dict(turn=bundle['turn'], current=meta['current'],
                 previous_action_comparison=meta['previous_action_comparison'],
                 execution_feedback=None)
    priority = meta['current'].get('resumed_execution', {}).get('user_priority')
    if priority in RESUME_PRIORITIES:
        delta['user_continuation_instruction'] = RESUME_PRIORITIES[priority]
    if meta['history']:
        last = meta['history'][-1]
        delta['execution_feedback'] = dict(turn=last['turn'],
            actual_call=(deepcopy(bundle['observation']['history'][-1]['call']) if profile.active_confirmation
                         else project_history(bundle['observation']['history'][-1:], profile)[0]['call']),
            result=last['feedback'])
        call = last['decision']
        if call and call['tool'] == 'recall_observation' and last['feedback'].get('accepted'):
            delta['recalled_observation_id'] = call['observation_id']
            delta['recall_delivery'] = 'reference_only; original images remain in this response chain'
    content = []
    if profile.active_memory:
        delta['active_memory'] = active_memory_context(archive)
    if profile.search_recessed_view:
        delta['search_memory']=search_memory_context(archive,candidate_priority=profile.search_candidate_priority)
        delta['search_memory']['observed_budget']=dict(calls_remaining=bundle['observation']['calls_remaining'],
            steps_remaining=bundle['observation']['steps_remaining'],wall_seconds_remaining=None,
            wall_time_status='not_observed_here; do not invent remaining time')
    if profile.task_handoff:
        handoff = task_handoff_context(archive, bundle['observation']['history'], session)
        delta['task_stage'] = handoff['current']
        if handoff['handoff'] is not None:
            delta['task_handoff'] = handoff['handoff']
    if profile.lean_feedback:
        previous_delta = None
        if archive:
            previous_delta = dict(current=text_observation(archive[-1]['bundle']['observation']))
            if profile.active_memory:
                previous_delta['active_memory'] = active_memory_context(archive[:-1])
            if profile.search_recessed_view:
                previous_delta['search_memory'] = search_memory_context(archive[:-1],candidate_priority=profile.search_candidate_priority)
        delta = lean_feedback_delta(delta, previous_delta=previous_delta,
            proposed_call=archive[-1]['call'] if archive else None)
    if not archive:
        fixed = dict(context_version='sol-session-v1', target_semantics=meta['target_semantics'],
                     robot_specifications=meta['robot_specifications'], depth_encoding=DEPTH_ENCODING,
                     history_status=meta['history_status'], task_state_is_model_self_report=True)
        if profile.task_handoff:
            fixed['handoff_plan'] = deepcopy(session.handoff_plan)
        if profile.transport_task:
            fixed['transport_mission'] = deepcopy(validate_transport_mission(session.transport_mission))
        content.append(dict(type='input_text', text=canonical(fixed).decode(),
                            prompt_cache_breakpoint={'mode': 'explicit'}))
    content.append(dict(type='input_text', text=canonical(delta).decode()))
    obs = current['observation']
    for sensor in SENSOR_FIELDS:
        tag = dict(image_role='current', observation_id=obs['observation_id'], turn=bundle['turn'],
                   control_step=profile.step_cap-obs['steps_remaining'], sensor=sensor,
                   resolution=[RESOLUTION, RESOLUTION])
        content.extend([dict(type='input_text', text=canonical(tag).decode()),
                        dict(type='input_image', detail='original',
                             image_url='data:image/png;base64,'+obs['sensors'][sensor]['data_base64'])])
    # A text boundary after all four images caches the complete immutable turn.
    content.append(dict(type='input_text', text=canonical(dict(observation_end=obs['observation_id'])).decode(),
                        prompt_cache_breakpoint={'mode': 'explicit'}))
    body.update(model=profile.model, store=True, truncation='disabled',
                reasoning=dict(effort='medium', summary='auto', context='all_turns'),
                input=[dict(role='user', content=content)],
                prompt_cache_options=dict(mode='explicit', ttl='30m'),
                # The provider caps this routing/accounting key at 64 characters.
                # Keep the attempt binding stable and isolated within that limit.
                prompt_cache_key=session.cache_key())
    if session.receipts:
        body['previous_response_id'] = session.receipts[-1]['response_id']
    wire = canonical(body)
    if non_image_bytes(wire) > 262144:
        raise ValueError('session delta text cap')
    return wire


def lean_changes(previous, current, path=()):
    """Exact structural changes, including deletions; lists use stable indices."""
    if type(previous) is type(current) and isinstance(current, dict):
        edits = []
        for key in sorted(set(previous)-set(current)):
            edits.append(dict(path=list(path+(key,)), remove=True))
        for key in sorted(current):
            if key not in previous:
                edits.append(dict(path=list(path+(key,)), value=deepcopy(current[key])))
            else:
                edits.extend(lean_changes(previous[key],current[key],path+(key,)))
        return edits
    if isinstance(previous,list) and isinstance(current,list) and len(previous)==len(current):
        return [edit for i,(old,new) in enumerate(zip(previous,current))
                for edit in lean_changes(old,new,path+(i,))]
    return [] if type(previous) is type(current) and previous == current else [dict(path=list(path),value=deepcopy(current))]


def lean_clearance_summary(report):
    """Decision summary, not lossless geometry: raw per-link evidence stays archived.

    Distances round down and unknown fractions round up. A positive minimum is
    never a safety certificate. Unfamiliar extensions survive instead of vanishing.
    Group entries are [minimum metres, worst segment, max unknown fraction].
    """
    groups={}
    for row in report['links']:
        name=row['link'].lower()
        group=('fingers' if 'finger' in name or 'gripper' in name else
               'head' if 'head' in name else 'torso' if 'torso' in name else
               'base' if 'base' in name or 'wheel' in name else 'arm')
        item=groups.setdefault(group,[None,None,0.])
        distance=row['min_clearance_m']
        if distance is not None and (item[0] is None or distance<item[0]):
            item[0],item[1]=distance,row['segment_index']
        item[2]=max(item[2],row['unobserved_fraction'])
    for item in groups.values():
        if item[0] is not None:item[0]=math.floor(item[0]*1000)/1000
        item[2]=math.ceil(item[2]*1000)/1000
    result={k:deepcopy(v) for k,v in report.items()
            if k not in ('links','coverage_basis','verification','completion_basis')}
    result['groups']=groups
    if 'low_surface' in report:
        result['nearest_points_base_m']={g:next((row.get('nearest_point_base_m') for row in report['links']
            if ('base' if 'base' in row['link'] or 'wheel' in row['link'] else 'other')==g
            and row.get('min_clearance_m') is not None
            and row['min_clearance_m']==min(x['min_clearance_m'] for x in report['links']
                if ('base' if 'base' in x['link'] or 'wheel' in x['link'] else 'other')==g
                and x.get('min_clearance_m') is not None)),None) for g in ('base',)}
        result['nearest_height_base_m']={g:None if point is None else point[2]
            for g,point in result['nearest_points_base_m'].items()}
    result['group_units']='min_m,worst_segment,max_unknown'
    unknown=not report['links'] or any(r['status']=='unknown' or r['unobserved_fraction']>0 or r['min_clearance_m'] is None
                for r in report['links'])
    verification=report.get('verification')
    result['conclusion']=('unverified' if unknown or report.get('stale_evidence')
        or report.get('path_complete') is False or (verification or {}).get('status')=='unverified'
        else 'sensor_report_only')
    if verification:
        result['verification_reasons']=deepcopy(verification['reasons'])
    return result


def lean_feedback_delta(delta, *, previous_delta=None, proposed_call=None):
    """Wire-only T1 projection. Never alters execution, gates or raw observations.

    Only joint display loses sub-.0005 precision; residual/gate flags stay exact.
    Stationary means every measured velocity rounds to zero at three decimals.
    Memory edits reconstruct its full value from this same response chain.
    """
    out=deepcopy(delta)
    current=out['current']
    for key in ('joint_positions','joint_velocities'):
        if key in current:
            current[key]={k:round(float(v),3) for k,v in current[key].items()}
    if current.get('joint_velocities') and all(v==0 for v in current['joint_velocities'].values()):
        current['joint_velocities']='静止'
    feedback=out.get('execution_feedback')
    if feedback is not None:
        actual=feedback.pop('actual_call',None)
        changes=lean_changes(proposed_call,actual)
        if changes:feedback['execution_call_changes']=changes
        def compact(value):
            if isinstance(value,dict):
                return {k:lean_clearance_summary(v) if k=='path_clearance' else compact(v) for k,v in value.items()}
            if isinstance(value,list):return [compact(v) for v in value]
            return value
        feedback['result']=compact(feedback['result'])
    comparison=out.get('previous_action_comparison')
    if comparison is not None:
        before,after=comparison['before'],comparison['after']
        # Both poses are coordinates in their respective base frames. This is
        # a reversible coordinate delta, not an asserted world-frame TCP motion.
        change=dict(from_observation_id=before['observation_id'],
            base_delta_in_previous_base=comparison['base_delta_in_previous_base'],
            tcp_position_coordinate_delta=[b-a for a,b in zip(before['tcp_pose']['position'],after['tcp_pose']['position'])],
            tcp_quaternion_coordinate_delta=[b-a for a,b in zip(before['tcp_pose']['quaternion_xyzw'],after['tcp_pose']['quaternion_xyzw'])],
            steps_used=before['steps_remaining']-after['steps_remaining'])
        out['previous_action_comparison']=change
    for container,previous in ((out,previous_delta or {}),
            (current,(previous_delta or {}).get('current',{}))):
        for key in tuple(container):
            if key not in ('active_memory','episode_memory','search_memory'):continue
            value=container.pop(key)
            edits=lean_changes(previous.get(key),value)
            if edits:container[key+'_changes']=edits
    return out


def session_cost(inp, out, *, cached=0, written=0, reserve=False):
    """Standard-tier service estimate, not provider invoice or cash payment."""
    if any(type(n) is not int or n < 0 for n in (inp, out, cached, written)) or cached+written > inp:
        raise ValueError('invalid session token accounting')
    p = FROZEN['sol_session']['pricing']
    input_multiplier = Decimal(p['long_input_multiplier']) if inp > p['long_input_threshold'] else Decimal(1)
    output_multiplier = Decimal(p['long_output_multiplier']) if inp > p['long_input_threshold'] else Decimal(1)
    ordinary = inp-cached-written
    input_cost = (Decimal(inp)*Decimal(p['cache_write']) if reserve else
                  ordinary*Decimal(p['input'])+cached*Decimal(p['cached_input'])+written*Decimal(p['cache_write']))
    return (input_cost*input_multiplier+out*Decimal(p['output'])*output_multiplier)/1000000


def session_usage_cost(usage):
    inp, out = usage.get('input_tokens'), usage.get('output_tokens')
    details = usage.get('input_tokens_details') or {}
    cached, written = details.get('cached_tokens'), details.get('cache_write_tokens')
    if any(type(n) is not int or n < 0 for n in (inp, out)):
        raise ValueError('missing session usage totals')
    if cached is None or written is None:
        # Do not fabricate cache misses or release a reservation on incomplete usage.
        return dict(estimated_usd=None, usage_cost_status='cache_details_unknown')
    amount = session_cost(inp, out, cached=cached, written=written)
    return dict(estimated_usd=str(amount), usage_cost_status='usage_estimate',
                ordinary_input_tokens=inp-cached-written, cached_input_tokens=cached,
                cache_write_tokens=written, pricing=deepcopy(FROZEN['sol_session']['pricing']))


def session_count_payload(wire, *, independent=False):
    """Count exactly the model-visible request, including the stored response chain."""
    body = json.loads(wire)
    fields = {'model', 'input', 'instructions', 'previous_response_id', 'reasoning',
              'text', 'tools', 'tool_choice', 'parallel_tool_calls', 'truncation'}
    non_input = {'store', 'max_output_tokens', 'prompt_cache_key', 'prompt_cache_options',
                 'service_tier'}
    if (set(body)-fields-non_input
            or (body.get('truncation') != 'disabled' and not (independent and 'truncation' not in body))
            or (independent and (body.get('store') is not False or 'previous_response_id' in body))):
        raise ValueError('unsupported exact input count fields')
    return canonical({k: v for k, v in body.items() if k in fields})


def count_session_input(wire, count_send, folder, *, independent=False):
    """One counting request; no generation, history mutation, or optimistic fallback."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    payload = session_count_payload(wire,independent=independent)
    receipt = dict(wire_sha256=hashlib.sha256(wire).hexdigest(),
                   payload_sha256=hashlib.sha256(payload).hexdigest(),
                   endpoint='/v1/responses/input_tokens', status='pending')
    save(folder/'receipt.json', receipt)
    (folder/'request.json').write_bytes(payload)
    try:
        raw = count_send(payload, 'count-'+receipt['wire_sha256'])
        (folder/'response.json').write_bytes(raw)
        response = json.loads(raw)
        count = response.get('input_tokens')
        if response.get('object') != 'response.input_tokens' or type(count) is not int or count <= 0:
            raise ValueError('invalid provider input token count')
        receipt.update(status='completed', input_tokens=count,
                       response_sha256=hashlib.sha256(raw).hexdigest())
        save(folder/'receipt.json', receipt)
        return receipt
    except Exception as error:
        receipt.update(status='failed_no_generation', error_type=type(error).__name__)
        save(folder/'receipt.json', receipt)
        raise


def estimate_session_wire(wire, session, profile=SOL_SESSION, *, input_count=None):
    if profile.cache_aware_budget and input_count is None:
        raise ValueError('v3 requires exact provider input count; no estimated count fallback')
    body = json.loads(wire)
    images = [c for m in body['input'] for c in m['content'] if c['type']=='input_image']
    expected_previous = session.receipts[-1]['response_id'] if session.receipts and not profile.image_window else None
    valid_images=(len(images) in (4,8,12) if profile.image_window else len(images)==4)
    if (body.get('model') != SOL_SESSION.model or body.get('store') is not (not profile.image_window)
            or body.get('previous_response_id') != expected_previous or not valid_images
            or (profile.image_window and 'previous_response_id' in body)
            or body.get('truncation') != 'disabled' or 'context_management' in body
            or body.get('max_output_tokens') != OUTPUT_CAP):
        raise ValueError('session request contract')
    for image in images:
        with Image.open(BytesIO(base64.b64decode(image['image_url'].split(',', 1)[1], validate=True))) as im:
            if im.size != (640,640) or im.format != 'PNG' or image['detail'] != 'original':
                raise ValueError('session requires original 640 PNG')
    image_tokens = len(images)*FROZEN['sol_session']['image_token_bound_640']
    delta = non_image_bytes(wire)+1024+image_tokens
    # Counting repeated instructions again is deliberately conservative. Include
    # all prior output (including reasoning), not merely the newly uploaded images.
    prior = session.receipts[-1] if session.receipts else None
    anchor = prior['input_bound'] if prior and not profile.image_window else 0
    if prior and profile.session_usage_anchor and not profile.image_window:
        anchor = prior.get('input_tokens')
        if type(anchor) is not int or not 0 <= anchor <= prior['input_bound']:
            raise ValueError('completed provider input usage required for context anchor')
    cumulative = delta+anchor+(prior['output_tokens'] if prior and not profile.image_window else 0)
    conservative = cumulative
    if input_count is not None:
        if (input_count.get('status') != 'completed'
                or input_count.get('wire_sha256') != hashlib.sha256(wire).hexdigest()
                or input_count.get('payload_sha256') != hashlib.sha256(session_count_payload(wire)).hexdigest()
                or input_count.get('endpoint') != '/v1/responses/input_tokens'
                or type(input_count.get('input_tokens')) is not int or input_count['input_tokens'] <= 0):
            raise ValueError('exact input count binding mismatch')
        # Provider count already includes instructions, schema, images and history.
        # Never add the historical anchor again or infer any cache discount.
        cumulative = input_count['input_tokens']
    output_cap = body['max_output_tokens']
    if cumulative > FROZEN['sol_session']['input_bound_cap'] or cumulative+output_cap > CONTEXT_LIMIT:
        if profile.session_usage_anchor:
            error = C2BudgetExhausted('session cumulative context cap; no truncation or compaction')
            error.scope = 'case' if profile.cache_aware_budget else 'context'
            raise error
        raise ValueError('session cumulative context cap; no truncation or compaction')
    result = dict(image_count=len(images), cumulative_image_count=len(images) if profile.image_window else 4*(len(session.receipts)+1),
                image_tokens=image_tokens, text_bytes=non_image_bytes(wire), delta_input_bound=delta,
                input_bound=cumulative, output_cap=output_cap,
                reserve_usd=str(session_cost(cumulative, output_cap, reserve=True)),
                estimate_basis=('completed provider input usage plus prior full output and conservative current delta'
                    if profile.session_usage_anchor else 'conservative bytes plus 1024/image and prior full output; not actual usage'))
    if input_count is not None:
        result.update(estimate_basis='provider exact input count; all input reserved at cold cache-write rate',
                      conservative_input_bound=conservative, input_count=deepcopy(input_count))
    if profile.cache_aware_budget:
        assumption=lean_cache_assumption(wire,session,cumulative)
        cached=assumption['assumed_cached_tokens']
        cold=Decimal(result['reserve_usd'])
        forecast=session_cost(cumulative,output_cap,cached=cached,written=cumulative-cached)
        margin=max(Decimal('.02'),forecast*Decimal('.25'))
        reserve=min(cold,forecast+margin).quantize(Decimal('.01'),rounding='ROUND_CEILING')
        result.update(cold_reserve_usd=str(cold),cache_assumption=assumption,
            cold_exposure_usd=str(cold.quantize(Decimal('.01'),rounding='ROUND_CEILING')),
            cache_assumed_reserve_usd=str(forecast),reserve_usd=str(reserve),
            safety_margin_usd=str(margin),
            reservation_formula='ceil_cent(min(cold_full_output,cache_forecast_full_output+max(0.02,0.25*cache_forecast_full_output)))',
            estimate_basis='exact input count; observed-prefix reserve plus margin; separate cold-miss exposure protects hard caps')
    return result


def lean_cache_identity(wire):
    body=json.loads(wire)
    return hashlib.sha256(canonical({k:body.get(k) for k in
        ('model','instructions','text','prompt_cache_key','prompt_cache_options')})).hexdigest()


def lean_cache_assumption(wire, session, input_tokens, *, now=None):
    """Discount only a conservatively reduced prefix observed in two recent reads.

    This is a forecast, never a guaranteed hit or a worst-case spend ceiling.
    Unsettled sends, changed schema/instructions/key and expired evidence get zero.
    """
    now=time.time() if now is None else now
    recent=session.receipts[-2:]
    identity=lean_cache_identity(wire)
    eligible=len(recent)==2 and all(
        r.get('cache_identity')==identity and type(r.get('cached_input_tokens')) is int
        and r['cached_input_tokens']>0 and type(r.get('settled_at_epoch')) in (int,float)
        and 0<=now-r['settled_at_epoch']<1800 for r in recent)
    floor=min(input_tokens,math.floor(.9*min(r['cached_input_tokens'] for r in recent))) if eligible else 0
    return dict(assumed_cached_tokens=floor,
        basis='90_percent_of_minimum_of_two_recent_settled_cache_reads' if eligible else 'cold_no_eligible_cache_evidence',
        source_response_ids=[r['response_id'] for r in recent] if eligible else [],
        cache_hit_guaranteed=False,ttl_seconds=1800)


def lean_budget_wire(wire, *, case_remaining, batch_remaining, reserve, assumption,
                     forecast=None, cold_exposure=None, snapshot_epoch=None):
    """Add the real ledger balance to the model-visible current observation."""
    body=json.loads(wire)
    for message in body['input']:
        for part in message['content']:
            if part['type']!='input_text':continue
            value=json.loads(part['text'])
            if 'execution_feedback' not in value:continue
            value['current']['budget_usd']=dict(
                case_remaining_usd=str(case_remaining),batch_remaining_usd=str(batch_remaining),
                next_request_reserve_usd=str(reserve),
                next_request_cache_forecast_usd=None if forecast is None else str(forecast),
                next_request_cold_exposure_usd=None if cold_exposure is None else str(cold_exposure),
                case_after_reserve_usd=str(case_remaining-reserve),
                balance_basis='authorization_minus_settled_usage_and_pending_reservations_not_account_cash',
                balance_as_of_epoch=snapshot_epoch,
                snapshot_may_change_during_parallel_admission=True,
                reservation_basis='exact_count_cache_forecast_plus_max_2c_or_25pct_margin_capped_at_cold',
                hard_cap_basis='generation_usage_plus_pending_cold_exposures_plus_retained_count_risk_plus_next_cold_exposure',
                counting_fee_policy='usd-0.01-per-count-risk-reserve-v1; not a confirmed provider price',
                counting_actual_charge_usd=None,
                forecast_basis=assumption['basis'],cache_hit_guaranteed=False)
            if 'state_card' in value:
                value['state_card']['budget']['usd']=deepcopy(value['current']['budget_usd'])
            part['text']=canonical(value).decode()
            return canonical(body)
    raise ValueError('missing current observation for v3 budget')


def prepare_lean_request(wire,session,profile,count_send,folder,*,case_remaining,batch_remaining):
    """Count the final budget-bearing bytes, not an earlier request draft.

    Counting revisions are saved separately; no generation is retried. A one-cent
    upward reservation grid bounds feedback/count self-reference. No fixed point
    within four counts means no generation send. API-free tests inject a counter.
    """
    if profile not in V3_PROFILES:raise ValueError('explicit lean profile required')
    assumption=lean_cache_assumption(wire,session,CONTEXT_LIMIT)
    if profile in V3_SYSTEM_PROFILES:
        final=lean_budget_wire(wire,case_remaining=case_remaining-Decimal('.01'),
            batch_remaining=batch_remaining-Decimal('.01'),reserve=Decimal('0'),
            assumption=assumption,snapshot_epoch=time.time())
        body=json.loads(final)
        for message in body['input']:
            for part in message['content']:
                if part['type']!='input_text':continue
                value=json.loads(part['text'])
                if 'execution_feedback' not in value:continue
                b=value['current']['budget_usd']
                b.update(next_request_reserve_usd=None,case_after_reserve_usd=None,
                    reservation_status='pending_exact_count_before_send',
                    reservation_record='request-estimate.json and admission ledger; exact reserve is not written back into counted input')
                if 'state_card' in value:value['state_card']['budget']['usd']=deepcopy(b)
                part['text']=canonical(value).decode()
        final=canonical(body)
        counted=count_session_input(final,count_send,Path(folder)/'count-00')
        estimate=estimate_session_wire(final,session,profile,input_count=counted)
        estimate.update(count_revisions=1,budget_snapshot_contract='immutable_precount_balance_exact_reserve_in_admission_ledger')
        return final,estimate
    reserve=Decimal('0.00')
    forecast=None
    cold_exposure=None
    snapshot_epoch=time.time()
    for revision in range(4):
        final=lean_budget_wire(wire,case_remaining=case_remaining,batch_remaining=batch_remaining,
            reserve=reserve,assumption=assumption,forecast=forecast,cold_exposure=cold_exposure,snapshot_epoch=snapshot_epoch)
        counted=count_session_input(final,count_send,Path(folder)/('count-%02d'%revision))
        estimate=estimate_session_wire(final,session,profile,input_count=counted)
        needed=Decimal(estimate['reserve_usd']).quantize(Decimal('.01'),rounding='ROUND_CEILING')
        predicted=Decimal(estimate['cache_assumed_reserve_usd']).quantize(Decimal('.01'),rounding='ROUND_CEILING')
        exposure=Decimal(estimate['cold_exposure_usd'])
        if needed<=reserve and predicted==forecast and exposure==cold_exposure:
            estimate.update(calculated_reserve_usd=estimate['reserve_usd'],reserve_usd=str(reserve),
                count_revisions=revision+1)
            return final,estimate
        reserve=needed
        forecast=predicted
        cold_exposure=exposure
    raise ValueError('v3 final budget/count did not converge; no generation sent')


def lean_budget_snapshot(entries, prior, auth, *, reserve=None, cold_exposure=None, reason):
    """Capture the same ledger used for admission; usage estimates are not invoices."""
    settled=[e for e in prior if e.get('request_kind')!='input_count'
             and e['status'] in ('valid','invalid_call','invalid_verification','not_sent')
             and e.get('estimated_usd') is not None]
    count_rows=[e for e in prior if e.get('request_kind')=='input_count']
    return dict(case_spent_usd=str(committed_cost(settled)),
        case_count_risk_reserved_usd=str(committed_cost(count_rows)),count_actual_usd=None,
        case_committed_usd=str(committed_cost(prior)),
        request_reserve_usd=None if reserve is None else str(reserve),
        request_cold_exposure_usd=None if cold_exposure is None else str(cold_exposure),
        case_remaining_usd=str(Decimal(str(auth['per_case_usd_cap']))-committed_cost(prior)),
        batch_remaining_usd=str(Decimal(str(auth['usd_cap']))-committed_cost(entries)),
        per_case_usd_cap=str(auth['per_case_usd_cap']),batch_usd_cap=str(auth['usd_cap']),
        reason=reason,balance_as_of_epoch=time.time(),
        basis='generation_usage_estimate_plus_pending_cold_exposure_and_retained_count_risk_not_invoice_or_cash')


def lean_precount_admission(entries, prior, auth, profile, *, verification=False, count_entries=()):
    """Known exhausted limits reject before even the non-generating count API."""
    # Peer requests may release most of their cold exposure on settlement.
    # Leave temporary reservation pressure to admit_budget's locked wait path.
    all_entries=list(entries)+list(count_entries)
    case_ids={e['case_id'] for e in prior}
    cost_prior=list(prior)+[e for e in count_entries if e['case_id'] in case_ids]
    settled_commitment=committed_cost([e for e in all_entries if e['status']!='reserved'])
    if (branch_request_slots_exhausted(auth,entries) or len(entries)>=auth['provider_requests_cap']
            or settled_commitment>=Decimal(str(auth['usd_cap']))):
        error=C2BudgetExhausted('v3 global limit exhausted before token counting')
        error.scope='global'
        error.budget_snapshot=lean_budget_snapshot(all_entries,cost_prior,auth,reason='global_limit_before_count')
        raise error
    if len(prior)+inherited_request_debit(auth)>=profile.call_cap:
        raise CaseLimitReached('tool_call_limit')
    if verification and sum(e.get('request_kind')=='verification' for e in prior)>=2:
        raise CaseLimitReached('identity_review_limit')
    if committed_cost(cost_prior)>=Decimal(str(auth['per_case_usd_cap'])):
        error=C2BudgetExhausted('v3 case limit exhausted before token counting')
        error.scope='case'
        error.budget_snapshot=lean_budget_snapshot(all_entries,cost_prior,auth,reason='case_usd_cap_before_count')
        raise error


def weekly_instructions(instructions, profile):
    instructions = instructions.replace('at most 40 tool calls', f'at most {profile.call_cap} tool calls')
    instructions = re.sub(r'\b600\b', str(profile.step_cap), instructions)
    instructions += ('\nSTEP ACCOUNTING: One environment control step is 0.05 simulated seconds, '
        'containing five physics substeps. Read-only tools use zero control steps. '
        'API waiting uses wall time, not simulated steps. steps_used includes settling. '
        'single_pick reports the authoritative shared request and usable step budget. '
        'Legacy calls_remaining counts main decision slots only; use single_pick.requests_remaining for admission. '
        'The horizon is an extended evaluation horizon, not a changed success or force criterion.\n')
    if profile.active_confirmation:
        instructions=instructions.replace(
            'After your response the system checks growth and may independently review before executing, except closing reads cached verdict only. ', '')
        instructions=instructions.replace(
            'Historical like can be used; confirmed does not prove current identity. Closing sends no new review. Explicit verify_target may refresh confirmed once on a new frame, within the original five-attempt limit; no retry. ', '')
        instructions += ('\nACTIVE SINGLE PICK: Identify a candidate explicitly using target_association and '
            'current target_observation. Same-object measurement changes use keep; new objects use new '
            'and cannot inherit confirmation. Furniture depth queries use measurement_only. '
            'Choose views to resolve identity (labels, shape and surrounding context); do not assume rest '
            'clears the head camera. Verify from the next actual image. Use verify_target deliberately '
            'before prepare_grasp/replan_grasp and again before closing after motion. This is an independent '
            'Sol review of a current robot image. After initial candidate selection, take a physical observation action '
            '(for example look) before the first review; merely repeating the same image is insufficient. '
            'confirmed is required, like/unclear is insufficient. '
            'Closing also requires grasp localization for this same candidate; a newly reviewed '
            'candidate cannot reuse the previous candidate\'s locked grasp. '
            'Any physical motion makes that review stale for the next grasp/close gate. At most four '
            'reviews, counted inside the total 60 requests; each verify_target main decision also counts. '
            'Keep new image evidence on every action. check_path is robot IK only, not collision clearance. '
            'Use sparse chunks only for visually clear transit; use separate short-distance move_to goals '
            'near furniture, on descent and retreat, then inspect fresh RGB-D. '
            f'Primary allocation is 40 total requests and {profile.primary_steps} steps, including a 122-step rest reserve. '
            'One additional 20 requests/400 steps unlocks only after real confirmed-target grasp localization '
            'and a failed grasp action. No resets. On such controller feedback the runtime releases it once. '
            'If a completed close is followed by fresh RGB-D showing no retained object, use recover_pick; '
            'cite the current observation ID and the empty-grasp evidence in note. This remains your '
            'visual judgment, not measured grasp truth. A rejected/zero-step prepare does not enter Pick. '
            'Do not repeat a failed pose without new evidence: retreat when supported, open, remeasure, '
            'and explicitly replan. Recovery cannot bypass official termination or force limits. '
            'Retain a request for rest. done is a self-report, never proof of possession.\n')
    if profile.multiview_confirmation:
        instructions = instructions.replace('At most four reviews, counted inside the total 60 requests; each verify_target main decision also counts.',
            'There is no separate four-review quota. Reviews and main decisions share the current '
            'request allocation (40 initially, 60 only after eligible recovery); each verify_target '
            'decision and its independent review use two requests. Do not give up because three or '
            'four reviews were used. Keep enough shared requests for review, localization, fresh '
            'review, close and rest. Review uses available current head and hand views of the same '
            'candidate. Seek complementary views of material, closure/seams, shape, package graphics '
            'and contents. If like/unclear persists, change the view to resolve the stated ambiguity '
            'instead of repeatedly submitting the same face. Readable category text is useful but '
            'not mandatory when distinctive visible evidence establishes the category. No review '
            'quota is replenished and shared request, cost, step and terminal limits still apply.')
    if profile.partial_target_marker:
        instructions = instructions.replace(
            'Box the entire selected object; give null for a camera if its full extent is unclear, occluded or outside view.',
            'Box the entire selected object when its full extent is visible. If its extent is partly '
            'occluded but a surface belonging to the candidate is visible, provide '
            '{box:null,pixel:[x,y]} on that CURRENT visible surface and declare partial_occlusion or '
            'full_extent_unavailable. Use a null camera only when no current surface can be marked. '
            'Never extrapolate hidden box edges or reuse an old pixel. A current pixel-only marker '
            'can request independent review without estimating full-object width. The reviewer may '
            'still reject ambiguous ownership or insufficient identity evidence; partial visibility '
            'never automatically confirms identity. After localization, first review the visible '
            'part in the current image; do not move merely to recover a complete outline. New targets '
            'still clear confirmation and all grasp/close freshness gates remain unchanged.')
    if profile.gripper_geometry_feedback:
        instructions += ('\nROBOT GEOMETRY FEEDBACK: check_path additionally reports conservative finger '
            'mesh bounds in the current command base frame at each proposed pose, for measured and fully '
            'open finger joints. These are robot-only geometry, not a scene collision test or a planned '
            'trajectory. Compare their lowest extent with support-surface points you measure from current '
            'RGB-D. A vertical table edge is not automatically the horizontal support surface. The '
            'gripper_link origin and visible object top are not the lowest fingertip. Account for the '
            'whole rotated finger volume and sensing uncertainty when choosing a grasp. Do not simply '
            'subtract a fixed offset from an object surface. You must propose any change and check it; '
            'the tool never moves or adjusts your pose. After a contact-related drift, do not assume '
            'a look action or an earlier arrived result preserves grasp alignment.\n')
    if profile.post_localization_review_recovery and not profile.persistent_target_confirmation:
        instructions = instructions.replace(
            'and a failed grasp action. No resets. On such controller feedback the runtime releases it once.',
            'and either a failed grasp action or a completed current independent review returning like/unclear '
            'after localization of that same candidate. No resets. The runtime releases this one shared '
            'reserve automatically; read the next single_pick budget. A negative identity, unsent/error '
            'review, changed target or terminal environment does not qualify.')
        instructions += ('\nCLOSING VIEW PLAN: Treat a usable closing-view as part of your grasp plan. '
            'Before descent, inspect which face carries distinguishing graphics/material/shape and '
            'whether the palm or fingers will hide it at your proposed closing pose. Prefer a feasible '
            'grasp orientation that leaves that face in the hand image, or position the head for a '
            'complementary view while the arm is still high. Do not assume a head turn can see through '
            'the arm. Inspect the actual new images; IK does not validate visibility. Keep the existing '
            'support-clearance checks. After localization, mark all currently visible candidate views '
            'and request fresh independent review, including partial markers when necessary. '
            'If review is like/unclear, read its specific missing evidence and the released budget: '
            'do not stop merely because the original 40-request allocation was nearly spent. Use a '
            'short supported retreat and a modest viewing-angle change to recover that evidence; '
            're-measure after motion, check a revised pose that preserves a usable closing view, '
            'obtain current confirmed, and explicitly replan_grasp for the same candidate. Review '
            'again at the final pose before closing. Do not descend back to the same obscuring pose '
            'without a reason the view will improve. like/unclear never authorizes close; releasing '
            'recovery does not confirm the target or prove possession. No hidden tracking, historical '
            'image substitution or automatic pose adjustment is performed.\n')
    if profile.smooth_grasp_localization:
        instructions += ('\nGRASP LOCALIZATION TIMING: prepare_grasp/replan_grasp use a smooth '
            'start and stop along your unchanged pose path. Nominal duration is 1.875 times '
            'the linear reference to preserve the same peak speed limits. Actual steps still '
            'count, with unchanged motion/convergence caps and rest reserve. A timing-budget '
            'rejection performs no localization; it is not Pick entry or permission to close. '
            'Approach with fresh observations before proposing a shorter localization if needed.\n')
    if profile.motion_guard:
        instructions += ('\nMOTION GUARD: every proposed arm waypoint receives robot-only IK '
            'checking before motion. waypoint_ik_unresolved means no motion occurred and does not '
            'prove global unreachability or scene clearance. Revise the indicated waypoint or '
            'your robot posture. motion_stalled means motion stopped without arriving: inspect '
            'the current sensors and residual errors, then replan instead of repeating the same '
            'pose. No waypoint is skipped and no arrival or grasp tolerance is relaxed.\n')
    if profile.self_collision_guard:
        instructions += ('\nSELF CLEARANCE: robot-only arm/head envelope checks may return self_collision_risk. This is not target rejection. Keep target identity and replan locally. Do not swing a raised elbow across the head to clear the camera. Move outward/forward away from the head first, using short inspected segments or another camera/head angle; do not repeat a rejected cross-side pose. IK alone does not prove whole-arm path clearance. The guard covers arm/head only, not furniture.\n')
    if profile.base_tracking_guard:
        instructions += ('\nFURNITURE APPROACH AND DEPARTURE: A locate_point is one visible surface sample, '
            'not the nearest boundary of a cabinet or proof of a clear corridor. Inspect its near corner '
            'and side at chassis height, not just a farther face or tabletop. Account for the full '
            '0.288 m chassis radius on BOTH sides and rear, plus the projecting arm and uncertainty. '
            'Before every turn, assess the swept chassis disk and arm, not only the forward floor. '
            'After rejecting a candidate at close range, first create departure clearance using a '
            'short normal retreat along a recently observed clear approach, if still supported by '
            'current RGB-D and odometry; otherwise observe the intended escape region first. Then '
            'turn and resume search. Do not slide along a cabinet or rotate against its near edge '
            'merely because the next target lies elsewhere. This departure rule concerns leaving '
            'a rejected candidate, not restarting a local grasp on a still-confirmed target. '
            'State near-side, far-side and rear clearance evidence in note before departing. '
            'Actual base motion is reported relative to command entry: actual_forward_m and '
            'actual_lateral_m use the requested final heading; actual_turn_rad is measured yaw. '
            'base_path_deviation means lateral travel or translation during a requested in-place '
            'turn exceeded 0.02 m. base_motion_stalled means odometry made no useful progress. '
            'Neither diagnoses collision. Stop treating the prior path as clear: inspect new views, '
            're-measure near edges and plan a supported escape, without repeating the same push. '
            'No contact identities, scene mesh checks, automatic detours or target resets are used.\n')
    if profile.fast_base_approach:
        instructions = instructions.replace('speed targets up to 0.6 rad/s and 0.25 m/s',
            'yaw targets up to 0.6 rad/s; normal translation up to 0.25 m/s, explicitly selected fast translation up to 0.4 m/s')
        instructions += ('\nBASE APPROACH: set base_speed_mode=normal or fast for move_base, null for other tools. '
            'Fast is allowed only for forward segments 0.5..1.5 m that YOU judge clear from current robot RGB-D, '
            'including the swept space of the held arm. Distance to the target alone is not clearance evidence. '
            'State that evidence in note. The controller does not check scene meshes or certify clearance. '
            'Fast uses bounded 0.4 m/s2 ramps, distance braking and slow final approach; it does not shorten '
            'observation intervals automatically. Stop each segment for fresh observation. Near furniture, '
            'clutter or uncertain clearance use short normal segments; never use fast for grasp descent or local corrections.\n')
    if profile.persistent_target_confirmation:
        instructions = instructions.replace(
            'confirmed is required, like/unclear is insufficient.',
            'An initial confirmed identity is required. For the continuously associated same candidate, '
            'later like/unclear is current-view uncertainty, not counterevidence to that confirmed identity.')
        instructions = instructions.replace(
            'Any physical motion makes that review stale for the next grasp/close gate.',
            'Physical motion requires a current review before grasp/close; it does not revoke identity. '
            'A current inconclusive review may preserve a prior confirmed identity for the same tracked target.')
        instructions = instructions.replace(
            'Do not repeat a failed pose without new evidence: retreat when supported, open, remeasure, and explicitly replan.',
            'A failed grasp check is feedback for local correction at the current target, not a default retreat or restart.')
        instructions += ('\nIDENTITY CONTINUITY AND LOCAL GRASP LOOP: Keep the same confirmed target unless '
            'current evidence explicitly contradicts its identity. unlike must cite the contradictory visible '
            'feature; blur, occlusion, distance or alignment error alone is not unlike. A new target never '
            'inherits confirmation. If association is uncertain, resolve association before closing; do not '
            'pretend certainty. Current review remains required after motion and its raw verdict is retained. '
            'For the same continuously tracked confirmed candidate, like/unclear alone does not require '
            're-reading packaging or changing grasp orientation. At the closing pose inspect whether the '
            'same object lies between the fingers and whether alignment and support clearance permit closing. '
            'If these checks pass, issue set_gripper close; an arrived pose alone is not possession. '
            'If a grasp check fails, use its specific residual and fresh RGB-D to propose sequential local '
            'actions: millimeter-scale position corrections, small orientation/opening changes, or head '
            'observation while preserving the grasp approach. Execute a short segment, observe again, then '
            'replan_grasp and close when ready. Do not abandon a valid approach merely to expose a label; '
            'do not lay the arm horizontally over the table to improve visibility. Account for wrist and '
            'forearm swept space, not just fingertip clearance; IK is not collision clearance. A small '
            'contact-relieving move may be needed, but justify it with actual feedback and continue on the '
            'same target. No automatic retreat, scene reset or blind trajectory sequence. Official terminal, '
            'cost and step limits still stop execution. After closing inspect retention and lift/rest.\n')
    if profile.search_reconsideration:
        instructions += ('\nUNCONFIRMED CANDIDATE SEARCH: A candidate that has never been confirmed '
            'is a hypothesis, not the task target. Shared branding and like are not confirmation. '
            'After two completed inconclusive reviews of the same never-confirmed candidate, '
            'explicitly reconsider it in task_state and note before spending further observations. '
            'This is a planning checkpoint, not a review quota or an automatic rejection. '
            "Compare the reviewers' specific unresolved alternatives with current RGB-D. "
            'If another view can distinguish them, name the distinguishing feature and why your '
            'next reachable view should reveal it; acquire that evidence before another review. '
            'Do not repeatedly magnify the same generic branding without new discriminating evidence. '
            'If no informative view is currently supported, retain this candidate and its uncertainty '
            'in the task history, then search another uninspected visible region with robot sensing. '
            'Choose search direction from current sensors and observed history, never a hidden map. '
            'A different object requires target_association=new and fresh independent confirmation; '
            "do not transfer the previous candidate's identity or locked grasp. Until another object "
            'is selected, use uncertain when correspondence is lost; do not invent a new candidate '
            'to reset counters. A same-object measurement change still uses keep/measurement_only. '
            'Use single_pick.requests_remaining to reserve enough shared requests for another '
            'search observation, localization, independent review, grasp and rest before repeated '
            'inspection consumes them. A verify_target decision and its review both cost requests. '
            'Failure to confirm this candidate alone does not show the target is absent or unreachable. '
            'Before give_up explain remaining search options and the actual budget, sensor, '
            'kinematic or termination constraint that prevents continuing. No forced motion, '
            'extra requests, search recovery allowance or automatic retry is added. '
            'This branch applies only to never-confirmed candidates. Preserve the confirmed-target '
            'continuity and local grasp correction rules; blur or alignment failure alone cannot '
            'revoke confirmed identity. Official termination and all hard limits still stop.\n')
    if profile.hot_continuation:
        instructions += ('\nLIVE CONTINUATION: Initial allocation is 60 shared requests/1800 steps. '
            'Before its boundary the host checkpoints this SAME live environment, controller, '
            'search history and response chain, then releases ONE extra 40 requests/1200 steps. '
            'The first give_up while extension remains is a soft segment pause: host checkpoints and resumes once, preserving your hindsight. '
            'Do not give_up merely to reserve a whole future sequence when this extension is pending. '
            'The fresh single_pick snapshot states currently usable limits; all counts are cumulative. '
            'Pick recovery is separate: once eligible, +20 requests/+400 steps, never search recovery. '
            'Absolute cap120 requests/3400 steps, with external dollar and wall limits unchanged by '
            'continuation. Official termination and force limits cannot be cleared. After continuation '
            'use fresh RGB-D, retain rejected candidates/inspected regions, and remeasure old coordinates.\n')
    if profile.efficient_search:
        instructions += ('\nSEARCH EFFICIENCY: Keep in task_state.belief a concise inventory of inspected '
            'regions and rejected candidates with their observed category evidence; in next_check '
            'name the next uninspected region and the information the next view should reveal. '
            'A red package or familiar brand is only a candidate, not a reason to repeatedly '
            'return to an excluded object. After ruling one out, inspect a new region. '
            'Maintain an observation posture that current head RGB-D confirms clears the forward '
            'corridor. return_to_rest is not a visibility tool: if it obscures the camera, do not '
            'repeat rest to clear the view. Propose a robot-kinematically reachable arm pose from '
            'current sensor evidence, clear furniture with short observed adjustments, then keep '
            'that posture during transit when suitable. Do not assume one fixed pose fits all scenes. '
            'Use the longest travel segment supported by the CURRENT visible depth corridor and '
            'stopping clearance: in a clear distant corridor consider 0.6-1.0 m with fast base_speed_mode; '
            'near furniture, occluded boundaries, turns or uncertain depth use shorter normal segments. '
            'Do not default to repeated 0.30 m segments when the same view supports farther travel. '
            'Reobserve after each command; shorten immediately if measured progress or visibility differs. '
            'Do not cross unseen space or infer clearance from target location. Near a candidate, '
            'move the camera to resolve a named discriminating face/detail, not just its brand. '
            'No scene truth, automatic candidate route or relaxed identity/grasp gates is provided.\n')
    if profile.search_progress:
        instructions += ('\nSEARCH PROGRESS AND REQUEST ALLOCATION: While no target is confirmed, '
            'plan the next useful observation or approach, not a guaranteed entire pick sequence. '
            'The earlier instruction to reserve requests for all remaining stages must not cause '
            'early abandonment merely because their total cost is uncertain. Keep one main request '
            'for return_to_rest when needed; the physical rest reserve is already enforced. '
            'With additional usable requests, time and motion budget, continue if a sensor-supported '
            'action can expose an uninspected region, improve discriminating detail, or resolve a '
            'specific route uncertainty. In task_state.next_check name the evidence the next action '
            'should obtain; evaluate its actual result before choosing the following action. '
            'Do not stop just because the full search-and-pick sequence may exceed the remaining '
            'requests. For give_up, identify the next concrete useful action considered and the '
            'observed obstruction, kinematic limit, actual cost/time/step admission limit, or exhausted '
            'informative views preventing it. Do not invent an obstruction to justify stopping. '
            'Avoid spending a two-request independent review on distant generic color patches: '
            'first acquire a view that could resolve the requested category. Close enough is judged '
            'from current visible detail, not a fixed distance or pixel threshold. This is scheduling, '
            'not permission to skip independent confirmation before grasping. Scan unexplored visible '
            'regions before repeatedly inspecting the same unresolved face; retain ambiguous '
            'candidates as hypotheses and revisit only for a named new discriminating view. '
            'During search, use check_path only when its result changes the next reachable action; '
            'do not test a speculative final grasp while the candidate is still far and unconfirmed. '
            'Recheck current depth after movement when needed; never reuse stale coordinates. '
            'Before sending, check field consistency: new requires both candidate_id and '
            'target_observation.target_id null; keep uses the current ID. A pixel-only marker in '
            'either camera requires partial_occlusion/full_extent_unavailable even if the other '
            'camera sees a complete outline. All observation IDs must match the current frame. '
            'Never transfer confirmation, weaken grasp gates, force unsafe movement, or claim '
            'search recovery. Eligible Pick recovery, official termination and all hard limits '
            'remain unchanged. This policy does not require exhausting budgets when no useful '
            'supported action remains.\n')
    if profile.episode_memory:
        instructions += ('\nEPISODE EVIDENCE MEMORY: current.episode_memory is a bounded summary of this '
            'episode only. Use its target identities, last observed positions, inspected views and action '
            'outcomes to avoid repeated uninformative inspection. Position is in the fixed episode odometry '
            'frame, not current base coordinates; transform it using current odometry before planning. '
            'A last-seen position is a surface observation with uncertainty, not a fresh grasp pose. '
            'A viewed region is not exhausted, empty or free; an excluded candidate excludes that object, '
            'not the whole region. Stale or occluded evidence needs observation. Confirmation belongs to '
            'its candidate and never transfers to another object. Memory cannot replace current association '
            'and grasp gates. Explain remaining uninspected areas in task_state and note.\n')
    if profile.local_clearance:
        instructions += ('\nSENSOR CLEARANCE: Physical tools now check the robot body and arm swept space '
            'against head/hand depth evidence stored in episode memory. This is sensor-assisted control. '
            'no_observed_conflict is not proof of safety. coverage_insufficient/stale_evidence means '
            'observe the affected side or shorten the proposed segment; observed_conflict means change '
            'the path or posture. Consider the protruding gripper, elbow and turn sweep even when the '
            'base center has room. On departure from furniture, first check the exit sweep. If it intersects '
            'the furniture, acquire rear/side evidence and separate from it before turning or moving along '
            'it. Retraction itself also needs a clear path. motion_suspended retains intent but holds the '
            'measured pose; a look command must not resume the rejected arm motion. Choose the next local '
            'action using fresh evidence, not repeated identical blocked commands.\n')
        instructions += ('A clearance_observation_exception allows only a bounded small head-only '
            'observation despite incomplete rear-side coverage. Its status remains coverage_insufficient; '
            'it proves neither the blind area nor a later base/arm path safe. No such exception applies '
            'to known conflicts or base/arm motion.\n')
    if profile.clearance_unknown_advisory:
        instructions += ('\nEVIDENCE-GUARD EXPERIMENT: Fresh coverage_insufficient is advisory, not an '
            'automatic veto. clearance_unknown_advisory means the proposed bare-robot motion was '
            'allowed despite missing visibility, NOT that the blind area is free or the path safe. '
            'Observed conflicts, stale evidence, self-collision, carried-geometry unavailability and '
            'official termination still stop motion. Choose the next observation or short local '
            'adjustment using RGB-D, body dimensions and remembered obstacles; do not assume that '
            'every unobserved shoulder surface makes all movement impossible. Do not repeat a '
            'known-conflict command. Near furniture consider the entire arm and base exit sweep.\n')
    if profile.carried_envelope:
        instructions += ('\nTARGET CONTACT AND CARRIED ENVELOPE: Before prepare_grasp/replan_grasp, measure a '
            'surface of the SAME candidate with locate_point at the current physical step, then use '
            'estimate_carried_envelope. Supply that point_id, source observation_id/camera, the estimated '
            'whole object center and quaternion in CURRENT gripper_link coordinates, half_extents_m, '
            'uncertainty_m (0.005..0.15m), and a short RGB-D-based reason. Include the hidden side using '
            'an explicit conservative estimate; one visible point does not prove the outer shape. '
            'This read-only estimate permits finger contact ONLY with source-labelled target points during localization; it does not enter Pick or prove attachment. After localization, remeasure and refresh the estimate before closing. Until closing, the object stays fixed in episode odometry; after closing it is only possibly attached. carried_envelope=null for other tools. '
            'Changing candidates requires a new estimate. Closure marks possibly attached; opening '
            'releases it. Subsequent arm/base motion checks the estimated object as well as the robot. '
            'After closing, use fresh images and short lift/retract actions to assess actual holding. '
            'Never infer grasp success from closure completion or an accepted envelope.\n')
    return instructions



def weekly_request_cap(session, profile):
    if profile.hot_continuation:
        saved=(session.single_pick or {}) if session else {}
        return 60+40*bool(saved.get('search_continuation'))+20*bool(saved.get('recovery_event'))+(10 if profile==SOL_HOT_RECOVERY and saved.get('search_continuation') and saved.get('recovery_event') else 0)
    released = bool(session and session.single_pick and session.single_pick.get('recovery_event'))
    return profile.call_cap if not profile.active_confirmation or released else 40


def single_pick_snapshot(state, executor, profile):
    cap = weekly_request_cap(state.response_session, profile)
    used = len(state.archive)+state.verification_requests
    return dict(requests_used=used, requests_remaining=max(0, cap-used), request_cap=cap,
        total_request_cap=profile.call_cap, reviews_used=state.verification_requests,
        reviews_cap=profile.verification_call_cap if profile.active_confirmation else 0, control_steps_used=executor.steps,
        usable_steps_remaining=max(0, executor.usable_step_cap()-executor.steps),
        total_step_cap=profile.step_cap, rest_reserved_steps=122,
        recovery_released=state.recovery_event is not None,
        pick_entered=state.pick_entry is not None)


def validate_single_pick_snapshot(value, profile):
    integer_keys = {'requests_used','requests_remaining','request_cap','total_request_cap',
        'reviews_used','reviews_cap','control_steps_used','usable_steps_remaining',
        'total_step_cap','rest_reserved_steps'}
    if (not isinstance(value, dict) or set(value) != integer_keys|{'recovery_released','pick_entered'}
            or any(type(value[k]) is not int or value[k]<0 for k in integer_keys)
            or any(type(value[k]) is not bool for k in ('recovery_released','pick_entered'))
            or value['total_step_cap'] != profile.step_cap or value['total_request_cap'] != profile.call_cap
            or value['requests_used']+value['requests_remaining'] != value['request_cap']
            or value['request_cap']>profile.call_cap or value['control_steps_used']>profile.step_cap
            or value['reviews_used']>value['reviews_cap']):
        raise ValueError('single-pick resource snapshot')


def persist_weekly_state(state):
    if state.response_session is None:
        raise ValueError('weekly policy requires the durable task session')
    continuation=(state.response_session.single_pick or {}).get("search_continuation")
    state.response_session.single_pick = deepcopy(dict(pick_entry=state.pick_entry,
        pick_closure=state.pick_closure,recovery_event=state.recovery_event))
    if continuation is not None:state.response_session.single_pick["search_continuation"]=continuation
    state.response_session.persist()


def weekly_target_fresh(state, bundle, executor, call, profile=None):
    target = state.identity_target
    reviews = [r for r in state.target_reviews if r.get('candidate_id') == target['id']
               and r.get('status') == 'completed' and (r.get('verdict') == 'confirmed'
               or (profile is not None and profile.persistent_target_confirmation
                   and target.get('verdict') == 'confirmed' and r.get('verdict') in ('like','unclear')))]
    if not reviews:
        return False
    review = reviews[-1]
    # Observation IDs may advance through read-only calls; physical steps may not.
    return review['source_snapshot']['steps_remaining'] == executor.max_steps-executor.steps


def weekly_release(state, executor, bundle, cause, *, model_self_report=False):
    if state.recovery_event is not None or state.pick_entry is None or not executor._available():
        return False
    executor.release_recovery()
    state.recovery_event = dict(step=executor.steps, observation_id=bundle['observation']['observation_id'],
        candidate_id=state.pick_entry['candidate_id'], cause=cause,
        model_self_report=model_self_report, added_requests=20, added_steps=400,
        counters_reset=False, environment_reset=False)
    persist_weekly_state(state)
    return True


def weekly_after_action(state, executor, call, actual, result, bundle):
    target = state.identity_target
    cid = target['id'] if target else None
    local = bool(getattr(executor,'local_clearance_guard',False))
    if state.pick_entry and state.pick_entry['candidate_id'] != cid:
        state.pick_entry = state.pick_closure = None
    entered = (result.get('localization_motion_executed') is True and target
        and target.get('verdict')=='confirmed') if local else result.get('steps_used',0)>0
    if actual and actual['tool'] in ('prepare_grasp','replan_grasp') and entered:
        state.pick_entry = dict(candidate_id=cid, step=executor.steps-result['steps_used'],
            observation_id=bundle['observation']['observation_id'])
        state.pick_closure = None
    if actual and actual['tool']=='set_gripper':
        if actual['gripper']==1:
            state.pick_closure = None
        elif result.get('closure_command_completed') is True and state.pick_entry:
            state.pick_closure = dict(candidate_id=cid, step=executor.steps,
                observation_id=bundle['observation']['observation_id'])
    reasons=('compliance_blocked','grasp_alignment_timeout','tool_motion_limit')
    if local:reasons+=('local_clearance_observed_conflict','local_clearance_coverage_insufficient',
        'local_clearance_stale_evidence')
    if (state.pick_entry and actual and actual['tool'] in ('prepare_grasp','replan_grasp','set_gripper')
            and result.get('reason') in reasons and (local or result.get('steps_used',0)>0)):
        weekly_release(state,executor,bundle,result['reason'])
    persist_weekly_state(state)


def weekly_visual_recovery(state, executor, call, bundle, association):
    oid=bundle['observation']['observation_id']; target=state.identity_target
    permitted = (state.pick_entry and state.pick_closure and target
        and state.pick_entry['candidate_id']==state.pick_closure['candidate_id']==target['id']
        and oid != state.pick_closure['observation_id'] and oid in call['note']
        and association['status']=='accepted' and not state.target_association_unresolved
        and state.recovery_event is None and executor._available())
    released = bool(permitted and weekly_release(state,executor,bundle,'visual_empty_grasp',model_self_report=True))
    return (call if released else None), dict(accepted=released,
        reason='recovery_released_visual_claim' if released else 'recovery_not_eligible',
        steps_used=0,steps_remaining=executor.max_steps-executor.steps)


def weekly_review_recovery(state, executor, bundle, profile):
    """Release once on actual current inconclusive review, never infer identity."""
    if not profile.post_localization_review_recovery or not state.target_reviews:
        return False
    target = state.identity_target
    entry = state.pick_entry
    review = state.target_reviews[-1]
    oid = bundle['observation']['observation_id']
    if (not target or not entry or state.target_association_unresolved
            or entry['candidate_id'] != target['id'] or review.get('candidate_id') != target['id']
            or review.get('status') != 'completed' or review.get('request_sent') is not True
            or review.get('verdict') not in ('like', 'unclear')
            or (target.get('verdict') != review['verdict'] and not
                (profile.persistent_target_confirmation and target.get('verdict') == 'confirmed'))
            or review.get('frame') != oid or oid == entry.get('observation_id')
            or executor.steps <= entry['step']
            or review.get('source_snapshot', {}).get('steps_remaining') != executor.max_steps-executor.steps):
        return False
    if not weekly_release(state, executor, bundle, 'post_localization_review_insufficient'):
        return False
    state.recovery_event.update(review_request_index=review['request_index'],
        review_verdict=review['verdict'], review_reason=review.get('reply', {}).get('reason'),
        identity_confirmed=target.get("verdict")=="confirmed")
    persist_weekly_state(state)
    return True


def weekly_matrix(stage):
    if stage=='diagnostic-hot-recovery':
        return [dict(row,stage=stage,profile=SOL_HOT_RECOVERY.name,
            attempt_id=row['attempt_id'].replace('diagnostic-hot',stage))
            for row in weekly_matrix('diagnostic-hot')]
    if stage=='diagnostic-local-floor':
        return [dict(row,stage=stage,profile=SOL_LOCAL_FLOOR.name,
            attempt_id=row['attempt_id'].replace('diagnostic-carried-envelope',stage))
            for row in weekly_matrix('diagnostic-carried-envelope')]
    if stage=='diagnostic-carried-envelope':
        return [dict(row,stage=stage,profile=SOL_CARRY.name,
            attempt_id=row['attempt_id'].replace('diagnostic-clearance-evidence',stage))
            for row in weekly_matrix('diagnostic-clearance-evidence')]
    """Deterministic proposed attempts; contains no authorization or hidden retries."""
    cfg=FROZEN['weekly_pick']; rows=[]
    if stage in ('diagnostic', 'diagnostic-v2', 'diagnostic-v3', 'diagnostic-v4', 'diagnostic-v5', 'diagnostic-v6', 'diagnostic-v7', 'diagnostic-motion-guard', 'diagnostic-recording', 'diagnostic-approach', 'diagnostic-extended', 'diagnostic-search', 'diagnostic-evidence', 'diagnostic-search-progress', 'diagnostic-hd', 'diagnostic-efficient', 'diagnostic-hot', 'diagnostic-cold', 'diagnostic-self', 'diagnostic-base-tracking', 'diagnostic-episode-memory', 'diagnostic-local-clearance', 'diagnostic-clearance-evidence'):
        # Each numbered release is a single case. The supervisor closes its
        # ledger before the next release, allowing evidence review between cases.
        candidate = {'diagnostic': SOL_STABLE, 'diagnostic-v2': SOL_STABLE_V2,
                     'diagnostic-v3': SOL_STABLE_V3, 'diagnostic-v4': SOL_STABLE_V4,
                     'diagnostic-v5': SOL_STABLE_V5, 'diagnostic-v6': SOL_STABLE_V6, 'diagnostic-v7': SOL_STABLE_V7, 'diagnostic-motion-guard': SOL_MOTION_GUARD, 'diagnostic-recording': SOL_RECORDING, 'diagnostic-approach': SOL_APPROACH, 'diagnostic-extended': SOL_EXTENDED, 'diagnostic-search': SOL_SEARCH, 'diagnostic-evidence': SOL_EVIDENCE, 'diagnostic-search-progress': SOL_SEARCH_PROGRESS, 'diagnostic-hd': SOL_HD, 'diagnostic-efficient': SOL_EFFICIENT, 'diagnostic-hot': SOL_HOT, 'diagnostic-cold': SOL_COLD, 'diagnostic-self': SOL_SELF, 'diagnostic-base-tracking': SOL_BASE, 'diagnostic-episode-memory': SOL_EPISODIC, 'diagnostic-local-clearance': SOL_CLEARANCE, 'diagnostic-clearance-evidence': SOL_CLEARANCE_EVIDENCE}[stage]
        return [dict(stage=stage,repeat=i,case_id=case,profile=candidate.name,
            attempt_id=f'weekly-{stage}-r{i}-{candidate.step_cap}-{case}',steps_cap=candidate.step_cap,
            requests_cap=candidate.call_cap,status='proposed') for i,case in enumerate(
                ('baseline-005','baseline-009','baseline-013','baseline-015','baseline-014'),1)]
    if stage=='buffer':
        cases=cfg['buffer_cases']; profiles=(SOL_BUFFER600,SOL_BUFFER1200); repeats=3
    elif stage=='representative':
        cases=cfg['representative_cases']; profiles=(SOL_STABLE,); repeats=3
    elif stage=='regression':
        cases=list(PICK20_CASES); profiles=(SOL_STABLE,); repeats=1
    else:raise ValueError('weekly stage')
    for repeat in range(1,repeats+1):
        # Counterbalance the paired budget condition order across repetitions.
        ordered=profiles if repeat%2 else tuple(reversed(profiles))
        for profile in ordered:
            for case in cases:
                rows.append(dict(stage=stage,repeat=repeat,case_id=case,profile=profile.name,
                    attempt_id=f'weekly-{stage}-r{repeat}-{profile.step_cap}-{case}',
                    steps_cap=profile.step_cap,requests_cap=profile.call_cap,status='proposed'))
    return rows


def validate_weekly_scope(auth, profile):
    if profile not in WEEKLY_PROFILES:
        raise ValueError('weekly profile required')
    required=dict(scope=WEEKLY_SCOPE,status='authorized',execution_profile=profile.name,
        model=profile.model,reasoning='medium',service_tier='default',currency='USD',
        controller_version='full-B-r4-plus-task-rules-v1',store=True,response_session=True,
        automatic_retries=0,prior_budget_transferred=0,per_case_steps_cap=profile.step_cap,
        per_case_decisions_cap=profile.call_cap,per_case_verification_cap=profile.verification_call_cap if profile.active_confirmation else 0)
    if any(type(auth.get(k)) is not type(v) or auth[k]!=v for k,v in required.items()):
        raise ValueError('fresh weekly authorization required; historical budgets cannot transfer')
    rows=[r for r in weekly_matrix(auth.get('stage')) if r['repeat']==auth.get('repeat') and r['profile']==profile.name]
    cases=[r['case_id'] for r in rows]
    expected_attempts={r['case_id']:r['attempt_id'] for r in rows}
    branch=auth.get('checkpoint_branch');resume=auth.get('cold_resume')
    if branch is not None:
        # The new batch names the diagnostic fork; its provider session retains
        # the original logical ID so archived receipts/ledger remain immutable.
        # The runner checks parent files, source delta, native state and hashes
        # before seeding this ledger or creating a worker.
        b=branch.get('binding',{})
        brake_branch=(profile==SOL_HOT_RECOVERY and branch.get('migration_contract')=='rest-brake-branch-v1')
        if (resume is not None or (profile!=SOL_BASE and not brake_branch) or cases!=['baseline-014']
                or branch.get('label')!='diagnostic-checkpoint-branch'
                or b.get('profile')!=(SOL_HOT_RECOVERY.name if brake_branch else SOL_SELF.name) or b.get('case_id')!=cases[0]
                or branch.get('logical_parent_attempt_id')!=b.get('attempt_id')
                or not isinstance(b.get('attempt_id'),str) or not b['attempt_id']
                or branch.get('parent_authorization_sha256')!=b.get('authorization_sha256')
                or branch.get('parent_batch_id')==auth.get('batch_id')
                or type(branch.get('prefix_requests')) is not int
                or not 0<branch['prefix_requests']<profile.call_cap
                or not re.fullmatch('[0-9a-f]{64}',branch.get('prefix_ledger_sha256',''))):
            raise ValueError('Explicit checkpoint diagnostic branch binding required')
        expected_attempts={cases[0]:b['attempt_id']}
    elif resume is not None:
        # A later same-source cold resume may retain a diagnostic branch's
        # logical session ID. Full closed-parent checks remain in the runner.
        b=resume.get('binding',{})
        floor_migration=(profile==SOL_LOCAL_FLOOR and b.get('profile')==SOL_CARRY.name
            and resume.get('source_migration',{}).get('contract')=='local-support-floor-resume-v1')
        if ((b.get('profile')==profile.name or floor_migration) and cases==[b.get('case_id')]
                and (b.get('source_sha256')==auth.get('source_sha256') or
                     floor_migration or (profile==SOL_CARRY and resume.get('source_migration',{}).get('contract') in ('carry-memory-copy-v1','carry-resume-lifecycle-v1')))
                and b.get('asset_manifest_sha256')==auth.get('asset_manifest_sha256')
                and isinstance(b.get('attempt_id'),str) and b['attempt_id']):
            expected_attempts={cases[0]:b['attempt_id']}
    if not rows or auth.get('case_ids')!=cases or auth.get('attempt_ids')!=expected_attempts:
        raise ValueError('weekly frozen matrix mismatch')
    for k,v in dict(provider_requests_cap=profile.call_cap*len(cases),physical_steps_cap=profile.step_cap*len(cases)).items():
        if type(auth.get(k)) is not int or auth[k]!=v:raise ValueError('weekly resource ceiling mismatch')
    for key in ('usd_cap','per_case_usd_cap'):
        try:value=Decimal(str(auth[key]))
        except (KeyError,ArithmeticError):raise ValueError('explicit weekly monetary caps required') from None
        if not value.is_finite() or value<=0:raise ValueError('explicit weekly monetary caps required')
    if Decimal(str(auth['per_case_usd_cap']))>Decimal(str(auth['usd_cap'])):
        raise ValueError('case cap exceeds batch cap')
    if (not isinstance(auth.get('batch_id'),str) or not auth['batch_id']
            or type(auth.get('expires_at_epoch')) not in (int,float)
            or not math.isfinite(auth['expires_at_epoch']) or auth['expires_at_epoch']<=time.time()
            or type(auth.get('gpu1_process_seconds_cap')) is not int or auth['gpu1_process_seconds_cap']<=0
            or type(auth.get('batch_wall_seconds_cap')) is not int or auth['batch_wall_seconds_cap']<=0
            or auth['gpu1_process_seconds_cap']>auth['batch_wall_seconds_cap']
            or auth.get('pricing')!=FROZEN['sol_session']['pricing']):
        raise ValueError('weekly batch, wall limit, lease or price binding missing')
    if any(k in auth for k in ('infrastructure_retry','recovery_attempt_id','replay_initial_anchor','pick20_continuation','pick20_remaining')):
        raise ValueError('weekly fresh attempts only')


def authorized_pick_cases(auth, default):
    """Explicit fresh subset, never an implicit continuation or default reduction."""
    if 'fresh_case_subset' not in auth:return list(default)
    subset=auth['fresh_case_subset']
    if (auth.get('scope')!=R2E_SCOPE or not isinstance(subset,list) or not subset
            or any(not isinstance(c,str) or c not in PICK20_CASES[:10] for c in subset)
            or subset!=sorted(set(subset)) or auth.get('case_ids')!=subset
            or auth.get('prior_budget_transferred')!=0):
        raise ValueError('Explicit ordered fresh R2e case subset required')
    return subset


def r15_case_isolation(auth):
    """Scheduling-only opt-in; historical grants keep their stop semantics."""
    return auth.get('scope') in (R15_SCOPE,R2A_SCOPE,R2B_SCOPE,R2C_SCOPE,R2D_SCOPE,R2E_SCOPE,*V3_SCOPES) and auth.get('isolate_case_outcomes') is True


def r15_job_id(case,profile,scope=None):
    if profile in V3_PROFILES:
        if case not in PICK20_CASES[:10]:raise ValueError('v3 near case required')
        return case
    if profile not in PAIRED_PROFILES:return case
    if case not in PICK20_CASES[:10]:raise ValueError('R1.5 near case required')
    control=SOL_R2A if scope==R2B_SCOPE else SOL_R15_BASELINE
    return ('control-' if profile==control else 'candidate-')+case


def session_attempt_id(auth,case,profile):
    value=auth.get('attempt_ids',{}).get(case)
    if profile in V3_PROFILES:return value
    return value+('-control' if profile==(SOL_R2A if auth.get('scope')==R2B_SCOPE else SOL_R15_BASELINE) else '-candidate') if profile in PAIRED_PROFILES and value else value


def validate_v3_lean_scope(auth, profile, *, require_start=True):
    """Fresh single-arm S1 grant, with deployed prompt and final CPU source binding."""
    v3_lean_prompt()
    spec = FROZEN['v3_lean_prompt']
    is_s2=profile in S2_PROFILES
    diagnostic=profile in S13_DIAGNOSTIC_PROFILES
    required = dict(scope=S2_SCOPES[profile.name] if is_s2 or diagnostic else V3_S13_SCOPE if profile==SOL_V3_LEAN_S13 else V3_S12_SCOPE if profile==SOL_V3_LEAN_S12 else V3_S11_SCOPE if profile==SOL_V3_LEAN_S11 else V3_LEAN_SCOPE, execution_profile=profile.name,
        status='authorized', model=SOL_V3_LEAN.model, reasoning='medium',
        service_tier='default', currency='USD', store=not profile.image_window, response_session=True,
        automatic_retries=0, prior_budget_transferred=0, isolate_case_outcomes=True,
        per_case_decisions_cap=80, per_case_steps_cap=1600,
        provider_requests_cap=800, physical_steps_cap=16000,
        gpu1_process_seconds_cap=4800, batch_wall_seconds_cap=48000,
        controller_version='full-B-r4-plus-task-rules-v1',
        exact_input_count='responses-input-count-v1',
        prompt_sha256=spec['prompt_sha256'], prompt_source_sha256=spec['source_sha256'],
        schema_sha256=hashlib.sha256(canonical(schema(profile))).hexdigest())
    if is_s2:
        required.update(prompt_sha256=hashlib.sha256(s2_prompt(profile).encode()).hexdigest(),
            stage_plan=True,short_plan_queue=profile.short_plan_queue,image_window=profile.image_window,
            context_mode='window' if profile.image_window else 'chain',schema_version=S2_SCHEMA_VERSION,
            segmented_exit_guidance=False,rest_base_push_report_only=False)
        if not isinstance(auth.get('authorization_evidence'),str) or not auth['authorization_evidence'].strip():
            raise ValueError('new S2 run requires explicit authorization evidence')
    if diagnostic:
        required.update(segmented_exit_guidance=profile.segmented_exit_guidance,
                        rest_base_push_report_only=profile.rest_base_push_report_only)
        if profile.segmented_exit_guidance:
            required['prompt_sha256']=hashlib.sha256((v3_lean_prompt()+S2_SEGMENTED_GUIDANCE).encode()).hexdigest()
    if (profile not in V3_PROFILES or any(type(auth.get(k)) is not type(v) or auth[k] != v
                                    for k, v in required.items())):
        raise ValueError('fresh v3 S1 scope/profile/prompt/schema contract required')
    forbidden = ('infrastructure_retry','recovery_attempt_id','replay_initial_anchor',
        'cold_resume','checkpoint_branch','pick20_remaining','pick20_continuation',
        'fresh_case_subset','paired_profiles','r2e_cpu_acceptance','r2a_cpu_acceptance',
        'baseline_executor_replay_waiver','active_memory','task_handoff',
        'identity_confirmation','identity_multiview_evidence','search_candidate_priority')
    if any(k in auth for k in forbidden):
        raise ValueError('v3 S1 forbids legacy grants, continuation and S2/S3 overrides')
    cases = list(PICK20_CASES[:10])
    if auth.get('case_ids') != cases:
        raise ValueError('v3 requires the ten fresh baseline cases exactly once')
    attempts = auth.get('attempt_ids')
    if (not isinstance(attempts, dict) or set(attempts) != set(cases)
            or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', v)
                   for v in attempts.values()) or len(set(attempts.values())) != 10):
        raise ValueError('unique explicit v3 attempt IDs required')
    if any(not isinstance(auth.get(k), str) or not auth[k] for k in ('batch_id','output_root')):
        raise ValueError('v3 batch identity and output root required')
    expires = auth.get('expires_at_epoch')
    if type(expires) not in (int,float) or not math.isfinite(expires) or expires <= time.time():
        raise ValueError('v3 authorization lease expired')
    try:
        if Decimal(str(auth['usd_cap'])) != 40 or Decimal(str(auth['per_case_usd_cap'])) != 4:
            raise ValueError('v3 fixed generation caps are USD4/case and USD40/batch')
    except (ArithmeticError, KeyError, TypeError):
        raise ValueError('v3 explicit generation caps required') from None
    if auth.get('pricing') != FROZEN['sol_session']['pricing']:
        raise ValueError('v3 model price contract mismatch')
    parallel = dict(max_workers=8, initialization_stagger_seconds=30, gpu_min_free_mib=8192,
        gpu_admission_utilization_max=75, request_serialization=False, api_concurrency=8,
        worker_seconds_cap=4800, total_worker_seconds_cap=48000)
    if auth.get('parallel_execution') != parallel or api_concurrency(auth) != 8:
        raise ValueError('v3 requires the fixed eight-worker/eight-API resource contract')
    counting = auth.get('input_count_authorization')
    count_required = dict(status='authorized', endpoint='/v1/responses/input_tokens',
        max_per_main=4, max_per_review=1, max_per_case=320, total_cap=3200,
        pricing_status='unconfirmed_budgeted', accounting_policy='usd-0.01-per-count-risk-reserve-v1',
        risk_reserve_per_request_usd='0.01', actual_charge_usd=None)
    if profile in V3_SYSTEM_PROFILES:count_required.update(max_per_main=1,max_per_case=80,total_cap=800)
    # Risk allowance is not a provider price or a proven billing upper bound.
    if (not isinstance(counting, dict) or any(type(counting.get(k)) is not type(v)
            or counting[k] != v for k,v in count_required.items())
            or counting.get('included_in_case_and_batch_caps') is not True):
        raise ValueError('v3 requires explicit shared-cap counting risk accounting')
    if profile in V3_SYSTEM_PROFILES:
        revision=('s2-plan-execute-window-v1' if is_s2 else 's13-independent-candidate-v1' if diagnostic else 's13-arm-base-g1' if profile==SOL_V3_LEAN_S13 else 's12-look-only' if profile==SOL_V3_LEAN_S12 else 's11-f1-f4')
        if auth.get('system_revision')!=revision or auth.get('count_once_per_wire') is not True:
            raise ValueError('S1.1 system and count contract required')
        if not all(auth.get(k) for k in ('asset_root','asset_manifest_path','asset_manifest_sha256')):
            raise ValueError('S1.1 approved HD assets required')
    proof = auth.get('v3_cpu_acceptance')
    if not isinstance(proof,dict) or set(proof) != {'path','sha256'}:
        raise ValueError('v3 final CPU receipt required')
    raw = Path(proof['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof['sha256']:
        raise ValueError('v3 CPU receipt hash mismatch')
    receipt = json.loads(raw)
    expected = dict(status='passed',kind='v3_s2_cpu_regression' if is_s2 else 'v3_s13_candidate_cpu_regression' if diagnostic else 'v3_s1_cpu_regression',failures=0,skipped=0,
        api_requests=0,physical_steps=0,gpu_experiment_jobs=0,
        prompt_sha256=required['prompt_sha256'],schema_sha256=required['schema_sha256'])
    if (any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k,v in expected.items())
            or type(receipt.get('tests')) is not int or receipt['tests'] <= 0):
        raise ValueError('v3 final CPU evidence incomplete')
    bound_sources=('pick.py','run_case.py','profiles.json','robot/executor.py','robot/clearance.py')+ (('memory.py',) if is_s2 else ())
    for relative in bound_sources:
        digest = hashlib.sha256((Path(__file__).parent/relative).read_bytes()).hexdigest()
        if receipt.get('candidate_sha256',{}).get(relative) != digest:
            raise ValueError('v3 CPU candidate source changed: '+relative)


def validate_session_scope(auth, profile, *, require_start=True):
    """A new contract; old Astra grants and closed/recovery budgets cannot match."""
    if profile.lean_feedback or profile.cache_aware_budget or auth.get('scope') in V3_SCOPES:
        return validate_v3_lean_scope(auth, profile, require_start=require_start)
    if profile.weekly_budget or auth.get('scope') == WEEKLY_SCOPE:
        return validate_weekly_scope(auth, profile)
    required = dict(scope=FROZEN['sol_session']['scope'], execution_profile='Sol-session-v1',
                    status='authorized', model=SOL_SESSION.model, reasoning='medium',
                    service_tier='default', currency='USD', store=True, response_session=True,
                    automatic_retries=0, prior_budget_transferred=0,
                    per_case_decisions_cap=25, per_case_steps_cap=600)
    ab = auth.get('scope') == FROZEN['memory_ab']['scope']
    handoff = auth.get('scope') == FROZEN['task_handoff']['scope']
    transport = auth.get('scope') == TRANSPORT_SCOPE
    aligned20 = auth.get('scope') == PICK20_BATCH_SCOPE
    dev = auth.get('scope') in DEV_R1_SCOPES
    r15=auth.get('scope') in (R15_SCOPE,R2A_SCOPE,R2B_SCOPE)
    r2e=auth.get('scope')==R2E_SCOPE
    identity_recovery=getattr(profile,'identity_attempt_recovery',False)
    candidate_priority=getattr(profile,'search_candidate_priority',False)
    confirmation=getattr(profile,'identity_confirmation',False)
    multiview=getattr(profile,'identity_multiview_evidence',False)
    i1_structured=profile.structured_close_evidence or profile.identity_confirmed_persistence
    if i1_structured:
        if (profile!=SOL_R2E_I1_STRUCTURED or not r2e
                or any(auth.get(k) is not True for k in ('structured_close_evidence','identity_confirmed_persistence','isolate_case_outcomes'))):
            raise ValueError('Fresh R2e I1 structured profile and isolated grant required')
    elif any(k in auth for k in ('structured_close_evidence','identity_confirmed_persistence','i1_structured_cpu_acceptance')):
        raise ValueError('I1 structured grant cannot authorize another profile')
    confirmation_profile=SOL_R2E_IDENTITY_EVIDENCE if multiview else SOL_R2E_IDENTITY_CONFIRMATION
    if multiview and (not confirmation or profile!=SOL_R2E_IDENTITY_EVIDENCE or auth.get('identity_multiview_evidence') is not True):
        raise ValueError('Fresh multiview identity profile and grant required')
    if not multiview and any(k in auth for k in ('identity_multiview_evidence','identity_multiview_cpu_acceptance')):
        raise ValueError('Multiview identity grant cannot authorize legacy profile')
    if confirmation and (profile!=confirmation_profile or auth.get('identity_confirmation') is not True or auth.get('rest_joint_target') is not True):
        raise ValueError('Fresh identity confirmation profile and grant required')
    if not confirmation and any(k in auth for k in ('identity_confirmation','identity_confirmation_cpu_acceptance','rest_joint_target')):
        raise ValueError('Identity confirmation grant cannot authorize legacy profile')
    if identity_recovery and candidate_priority:
        raise ValueError('Identity recovery and candidate priority require independent experiments')
    if identity_recovery:
        if not r2e or profile!=SOL_R2E_RECOVERY or auth.get('identity_attempt_recovery') is not True:
            raise ValueError('Fresh identity recovery profile and explicit grant required')
    elif any(k in auth for k in ('identity_attempt_recovery','identity_recovery_cpu_acceptance')):
        raise ValueError('Identity recovery grant cannot authorize a legacy profile')
    if candidate_priority:
        if not r2e or profile!=(confirmation_profile if confirmation else SOL_R2E_CANDIDATE_PRIORITY) or auth.get('search_candidate_priority') is not True:
            raise ValueError('Fresh candidate priority profile and explicit grant required')
    elif any(k in auth for k in ('search_candidate_priority','candidate_priority_cpu_acceptance')):
        raise ValueError('Candidate priority grant cannot authorize another profile')
    r2e_profile=SOL_R2E_I1_STRUCTURED if i1_structured else confirmation_profile if confirmation else SOL_R2E_RECOVERY if identity_recovery else SOL_R2E_CANDIDATE_PRIORITY if candidate_priority else SOL_R2E
    r2d=auth.get('scope') in (R2D_SCOPE,R2E_SCOPE)
    r2c=auth.get('scope') in (R2C_SCOPE,R2D_SCOPE,R2E_SCOPE)
    if 'isolate_case_outcomes' in auth and (not (r15 or r2c) or type(auth['isolate_case_outcomes']) is not bool):
        raise ValueError('Case isolation requires explicit boolean R1.5 grant')
    extended_dev = auth.get('scope') in (DEV_R1_BUDGET_SCOPE,R15_SCOPE,R2A_SCOPE,R2B_SCOPE,R2C_SCOPE,R2D_SCOPE,R2E_SCOPE)

    pick20 = auth.get('scope') in (PICK20_SCOPE,PICK20_BATCH_SCOPE,*DEV_R1_SCOPES)

    if pick20:
        required.update(scope=PICK20_BATCH_SCOPE if aligned20 else PICK20_SCOPE, execution_profile=SOL_PICK20_ALIGNED.name if aligned20 else SOL_PICK20.name,
            controller_version='full-B-r4-plus-task-rules-v1',
            per_case_usd_cap='8', usd_cap='160' if aligned20 else '40', gpu1_process_seconds_cap=1800,
            per_case_decisions_cap=40 if aligned20 else 25)
        if aligned20:
            required.pop('usd_cap')  # A fresh grant may be lower than the campaign ceiling.
        if dev:

            required.update(scope=DEV_R1_SCOPE, execution_profile=SOL_DEV_R1.name,

                per_case_usd_cap='6', usd_cap='6', per_case_decisions_cap=40,

                batch_wall_seconds_cap=18000)

        if extended_dev:
            required.update(scope=DEV_R1_BUDGET_SCOPE, execution_profile=SOL_DEV_R1_PARALLEL.name if profile.parallel_case_workers else SOL_DEV_R1_1600.name,
                per_case_usd_cap='15', usd_cap='15', per_case_decisions_cap=50,
                per_case_steps_cap=1600, gpu1_process_seconds_cap=4800,
                batch_wall_seconds_cap=48000)
        if r2c:
            required.update(scope=R2E_SCOPE if r2e else R2D_SCOPE if r2d else R2C_SCOPE,execution_profile=(r2e_profile if r2e else SOL_R2D if r2d else SOL_R2C).name,isolate_case_outcomes=True)
            if r2d:required.update(per_case_usd_cap='24',usd_cap='24',per_case_decisions_cap=80)
            if i1_structured:
                cap=auth.get('per_case_usd_cap')
                try:valid=isinstance(cap,str) and Decimal(cap).is_finite() and 0<Decimal(cap)<=Decimal('24')
                except (ValueError,ArithmeticError):valid=False
                if not valid:raise ValueError('Explicit positive I1 case USD cap at most 24 required')
                required['per_case_usd_cap']=cap
            if 'paired_profiles' in auth:raise ValueError('R2c is candidate only')
        if r15:
            if auth['scope'] in (R2A_SCOPE,R2B_SCOPE) and auth.get('isolate_case_outcomes') is not True:raise ValueError('R2a requires case outcome isolation')
            required.update(scope=auth['scope'],execution_profile='Sol-r2b-paired' if auth['scope']==R2B_SCOPE else 'Sol-r2a-paired' if auth['scope']==R2A_SCOPE else 'Sol-r15-paired',per_case_usd_cap='30',usd_cap='30',batch_wall_seconds_cap=96000)
            if auth.get('paired_profiles')!=[p.name for p in (R2B_PROFILES if auth['scope']==R2B_SCOPE else R2A_PROFILES if auth['scope']==R2A_SCOPE else R15_PROFILES)]:raise ValueError('Both R1.5 arms required')
        if auth.get('case_ids') != authorized_pick_cases(auth,PICK20_CASES[:10] if dev else PICK20_CASES if aligned20 else PICK20_CASES[:5]):

            raise ValueError('pick20 pilot authorizes exactly baseline-000 through 004')
    if transport:
        required.update(scope=TRANSPORT_SCOPE, execution_profile=profile.name, per_case_decisions_cap=60)
        validate_transport_mission(auth.get('transport_mission'))
        if profile.task_handoff:
            validate_handoff_plan(auth.get('handoff_plan'))
        TransportABBatch.from_auth(auth).validate_episode(auth, profile)
    if handoff:
        required.update(scope=FROZEN['task_handoff']['scope'], execution_profile=SOL_HANDOFF.name)
        validate_handoff_plan(auth.get('handoff_plan'))
    if ab:
        required.update(scope=FROZEN['memory_ab']['scope'], execution_profile=profile.name)
        validate_memory_ab_episode(auth, profile, require_start=require_start)
    allowed = (r2e_profile if r2e else SOL_R2D if r2d else SOL_R2C,) if r2c else (R2B_PROFILES if auth['scope']==R2B_SCOPE else R2A_PROFILES if auth['scope']==R2A_SCOPE else R15_PROFILES) if r15 else (SOL_DEV_R1_1600,SOL_DEV_R1_PARALLEL) if extended_dev else (SOL_DEV_R1,) if dev else (SOL_PICK20_ALIGNED,) if aligned20 else (SOL_PICK20,) if pick20 else (SOL_TRANSPORT_A, SOL_TRANSPORT_B) if transport else (SOL_HANDOFF,) if handoff else (SOL_SESSION, SOL_MEMORY) if ab else (SOL_SESSION,)

    if profile not in allowed or any(type(auth.get(k)) is not type(v) or auth[k] != v for k,v in required.items()):
        raise ValueError('fresh Sol-session-v1 scope required; old authorization is not transferable')
    if profile.parallel_case_workers:
        worker_cap = 4800*len(authorized_pick_cases(auth,PICK20_CASES[:10])) if 'fresh_case_subset' in auth else 96000 if r15 else 48000
        if auth['scope'] == R2B_SCOPE:
            worker_cap = auth.get('parallel_execution', {}).get('total_worker_seconds_cap')
            if type(worker_cap) is not int or not 0 < worker_cap <= 96000:
                raise ValueError('R2b worker grant must be a positive integer at most 96000')
        max_workers=auth.get('parallel_execution',{}).get('max_workers') if r2c else 10
        if r2c and (type(max_workers) is not int or not 4<=max_workers<=8):raise ValueError('R2c requires 4..8 workers')
        expected_parallel = dict(max_workers=max_workers, initialization_stagger_seconds=30,
                gpu_min_free_mib=8192, gpu_admission_utilization_max=75,
                request_serialization=True, worker_seconds_cap=4800, total_worker_seconds_cap=worker_cap)
        concurrency=api_concurrency(auth)
        if 'api_concurrency' in auth.get('parallel_execution',{}):
            expected_parallel.update(api_concurrency=concurrency,request_serialization=concurrency==1)
        if auth.get('parallel_execution') != expected_parallel:
            raise ValueError('Explicit bounded parallel execution grant required')
    elif 'parallel_execution' in auth:
        raise ValueError('Parallel grant requires explicit parallel profile')
    cases = auth.get('case_ids')
    if (not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)
            or any(not isinstance(c, str) or (c not in {j['case_id'] for j in memory_ab_matrix()} if ab
                   else c not in PICK20_CASES if pick20
                   else c != 'handoff-transport-val000-object0' if transport
                   else not re.fullmatch(r'arm-dev-00[0-4]', c)) for c in cases)):
        raise ValueError('session development cases required')
    attempts = auth.get('attempt_ids')
    if (not isinstance(attempts, dict) or set(attempts) != set(cases)
            or any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', v) for v in attempts.values())
            or len(set(attempts.values())) != len(cases)):
        raise ValueError('unique explicit session attempt IDs required')
    if not isinstance(auth.get('batch_id'), str) or not auth['batch_id']:
        raise ValueError('session batch identity required')
    logical_count=len(cases)*(2 if r15 else 1)
    for field, maximum in (('provider_requests_cap',(80 if r2d else 50 if extended_dev else 40 if aligned20 or dev else 60 if transport else 25)*logical_count), ('physical_steps_cap',(1600 if extended_dev else 600)*logical_count)):

        if type(auth.get(field)) is not int or not 0 < auth[field] <= maximum:
            raise ValueError('session resource ceiling required')
    if auth['physical_steps_cap'] != (1600 if extended_dev else 600)*logical_count:
        raise ValueError('session physical cap must cover the unchanged 600-step controller per case')
    expires = auth.get('expires_at_epoch')
    if type(expires) not in (int,float) or not math.isfinite(expires) or expires <= time.time():
        raise ValueError('session authorization lease expired')
    try:
        cap, per_case = (Decimal(str(auth[k])) for k in ('usd_cap','per_case_usd_cap'))
        if not cap.is_finite() or not per_case.is_finite() or not 0 < per_case <= cap or (aligned20 and cap > 160):
            raise ValueError('session monetary ceilings required')
    except (ArithmeticError, KeyError, TypeError):
        raise ValueError('session monetary ceilings required') from None
    if auth.get('pricing') != FROZEN['sol_session']['pricing']:
        raise ValueError('session model price contract mismatch')
    if any(k in auth for k in ('infrastructure_retry','recovery_attempt_id','replay_initial_anchor')):
        raise ValueError('session scope does not permit recovery or borrowed replay state')
    if dev:

        if any(k in auth for k in ('cold_resume','checkpoint_branch','pick20_remaining','pick20_continuation')):

            raise ValueError('R1 requires ten independent fresh attempts')

        # R2c was approved after archived CPU regression, without a new physical replay.
        # Keep this distinct from the older baseline-005 native replay evidence.
        if r2e:
            proof=auth.get('r2e_cpu_acceptance')
            if (not isinstance(proof,dict) or set(proof)!={'path','sha256'} or
                    any(k in auth for k in ('r2d_cpu_acceptance','r2c_cpu_acceptance','baseline_executor_replay','baseline_executor_replay_waiver'))
                    or auth.get('g1_insertion_gate')!='deferred_by_user'):
                raise ValueError('R2e requires its own CPU receipt and explicit G1 deferral')
            raw=Path(proof['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('R2e CPU receipt hash')
            receipt=json.loads(raw)
            expected=dict(status='passed_with_g1_deferred',kind='r2e_archived_cpu_regression',api_requests=0,
                physical_steps=0,failures=0,skipped=0,normal_motion_false_positives=0,
                legacy_request_parity=True,legacy_executor_parity=True,next_request_feedback=True,
                dual_reachability=True,clearance_unverified=True,g1_insertion_gate='deferred_by_user',
                search_recessed_view=True,search_memory_next_request=True,
                search_memory_observation_binding=True,search_boundary_budget_feedback=True)
            if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('R2e CPU evidence incomplete')
            force=receipt.get('r2d006_trigger_evaluation_cumulative_force')
            if type(force) not in (int,float) or not math.isfinite(force) or not 0<=force<=2500:
                raise ValueError('R2e 006 evaluation gate missing')
            if identity_recovery:
                recovery_proof=auth.get('identity_recovery_cpu_acceptance')
                if not isinstance(recovery_proof,dict) or set(recovery_proof)!={'path','sha256'}:
                    raise ValueError('Identity recovery requires its own CPU receipt')
                recovery_raw=Path(recovery_proof['path']).read_bytes()
                if hashlib.sha256(recovery_raw).hexdigest()!=recovery_proof['sha256']:
                    raise ValueError('Identity recovery CPU receipt hash')
                recovery_receipt=json.loads(recovery_raw)
                recovery_expected=dict(status='passed',kind='r2e_identity_recovery_cpu_regression',
                    api_requests=0,physical_steps=0,gpu_experiment_jobs=0,failures=0,skipped=0,
                    legacy_request_parity=True,identity_recovery=True,
                    baseline_r2e_cpu_acceptance_sha256=proof['sha256'])
                if (any(type(recovery_receipt.get(k)) is not type(v) or recovery_receipt[k]!=v
                        for k,v in recovery_expected.items())
                        or type(recovery_receipt.get('tests')) is not int or recovery_receipt['tests']<=0):
                    raise ValueError('Identity recovery CPU evidence incomplete')
                # The historical R2e receipt remains immutable baseline evidence.
                # Only the new CPU receipt can attest the modified candidate bytes.
                receipt=recovery_receipt
            if candidate_priority:
                priority_proof=auth.get('candidate_priority_cpu_acceptance')
                if not isinstance(priority_proof,dict) or set(priority_proof)!={'path','sha256'}:
                    raise ValueError('Candidate priority requires its own CPU receipt')
                priority_raw=Path(priority_proof['path']).read_bytes()
                if hashlib.sha256(priority_raw).hexdigest()!=priority_proof['sha256']:
                    raise ValueError('Candidate priority CPU receipt hash')
                priority_receipt=json.loads(priority_raw)
                priority_expected=dict(status='passed',kind='r2e_candidate_priority_cpu_regression',
                    api_requests=0,physical_steps=0,gpu_experiment_jobs=0,failures=0,skipped=0,
                    legacy_request_parity=True,candidate_priority=True,next_request_feedback=True,
                    baseline_r2e_cpu_acceptance_sha256=proof['sha256'])
                if (any(type(priority_receipt.get(k)) is not type(v) or priority_receipt[k]!=v
                        for k,v in priority_expected.items())
                        or type(priority_receipt.get('tests')) is not int or priority_receipt['tests']<=0):
                    raise ValueError('Candidate priority CPU evidence incomplete')
                receipt=priority_receipt
            if confirmation:
                confirm_proof=auth.get('identity_confirmation_cpu_acceptance')
                if not isinstance(confirm_proof,dict) or set(confirm_proof)!={'path','sha256'}:raise ValueError('Identity confirmation CPU receipt required')
                confirm_raw=Path(confirm_proof['path']).read_bytes()
                if hashlib.sha256(confirm_raw).hexdigest()!=confirm_proof['sha256']:raise ValueError('Identity confirmation CPU receipt hash')
                receipt=json.loads(confirm_raw)
                expected_confirm=dict(status='passed',kind='identity_confirmation_cpu_regression',api_requests=0,
                    physical_steps=0,gpu_experiment_jobs=0,failures=0,skipped=0,legacy_request_parity=True,
                    next_request_feedback=True,uncertain_done_blocked=True,uncertain_close_blocked=True,
                    rest_joint_target=True,rest_preview_same_reference=True)
                if any(type(receipt.get(k)) is not type(v) or receipt[k]!=v for k,v in expected_confirm.items()):
                    raise ValueError('Identity confirmation CPU evidence incomplete')
                if type(receipt.get('tests')) is not int or receipt['tests']<=0:raise ValueError('Identity confirmation tests missing')
            if multiview:
                proof=auth.get('identity_multiview_cpu_acceptance')
                if not isinstance(proof,dict) or set(proof)!={'path','sha256'}:raise ValueError('Multiview identity CPU receipt required')
                raw=Path(proof['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('Multiview identity CPU receipt hash')
                receipt=json.loads(raw)
                expected=dict(status='passed',kind='identity_multiview_cpu_regression',api_requests=0,
                    physical_steps=0,gpu_experiment_jobs=0,failures=0,skipped=0,legacy_request_parity=True,
                    next_request_feedback=True,uncertain_done_blocked=True,uncertain_close_blocked=True,
                    original_sensor_images=True,unresolved_association_blocked=True)
                if any(type(receipt.get(k)) is not type(v) or receipt[k]!=v for k,v in expected.items()):
                    raise ValueError('Multiview identity CPU evidence incomplete')
                if type(receipt.get('tests')) is not int or receipt['tests']<=0:raise ValueError('Multiview identity tests missing')
            if i1_structured:
                new_proof=auth.get('i1_structured_cpu_acceptance')
                if not isinstance(new_proof,dict) or set(new_proof)!={'path','sha256'}:
                    raise ValueError('I1 structured CPU receipt required')
                raw=Path(new_proof['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=new_proof['sha256']:raise ValueError('I1 structured CPU receipt hash')
                receipt=json.loads(raw)
                expected=dict(status='passed',kind='r2e_i1_structured_cpu_regression',api_requests=0,
                    physical_steps=0,gpu_experiment_jobs=0,failures=0,skipped=0,legacy_request_parity=True,
                    legacy_executor_parity=True,next_request_feedback=True,structured_close_evidence=True,
                    confirmed_persistence=True,case_budget_isolation=True)
                if any(type(receipt.get(k)) is not type(v) or receipt[k]!=v for k,v in expected.items()):
                    raise ValueError('I1 structured CPU evidence incomplete')
                if type(receipt.get('tests')) is not int or receipt['tests']<=0:raise ValueError('I1 structured tests missing')
            for relative in ('pick.py','run_case.py','robot/executor.py','robot/clearance.py','profiles.json'):
                if receipt.get('candidate_sha256',{}).get(relative)!=hashlib.sha256((Path(__file__).parent/relative).read_bytes()).hexdigest():
                    raise ValueError('R2e CPU candidate source changed')
            return
        if r2d:
            proof=auth.get('r2d_cpu_acceptance')
            if (not isinstance(proof,dict) or set(proof)!={'path','sha256'} or
                    any(k in auth for k in ('r2c_cpu_acceptance','baseline_executor_replay','baseline_executor_replay_waiver'))):
                raise ValueError('R2d requires its own CPU receipt')
            raw=Path(proof['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('R2d CPU receipt hash')
            receipt=json.loads(raw)
            expected=dict(status='passed',kind='r2d_archived_cpu_regression',api_requests=0,physical_steps=0,
                failures=0,skipped=0,normal_base_false_positives=0,base_003_trigger_step=146,
                compound_cases=['000','006'],legacy_request_parity=True,legacy_executor_parity=True,
                base_clearance_003=True,next_request_feedback=True)
            if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('R2d CPU evidence incomplete')
            for relative in ('pick.py','run_case.py','robot/executor.py','robot/clearance.py','profiles.json'):
                if receipt.get('candidate_sha256',{}).get(relative)!=hashlib.sha256((Path(__file__).parent/relative).read_bytes()).hexdigest():
                    raise ValueError('R2d CPU candidate source changed')
            return
        cpu_proof = auth.get('r2c_cpu_acceptance')
        if cpu_proof is not None:
            if not r2c or any(k in auth for k in ('baseline_executor_replay','baseline_executor_replay_waiver')):
                raise ValueError('R2c CPU proof cannot replace another scope or masquerade as replay')
            if not isinstance(cpu_proof,dict) or set(cpu_proof)!={'path','sha256'}:
                raise ValueError('R2c CPU receipt required')
            raw=Path(cpu_proof['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=cpu_proof['sha256']:raise ValueError('R2c CPU receipt hash')
            receipt=json.loads(raw)
            expected=dict(status='passed',kind='r2c_archived_cpu_regression',api_requests=0,physical_steps=0,
                          failures=0,skipped=0,r2a_main_requests_equal=321,r2a_verification_requests_equal=11,
                          r2a_executor_commands_equal=16,r2a_executor_steps_equal=547)
            if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('R2c CPU evidence incomplete')
            for relative in ('pick.py','run_case.py','robot/executor.py','profiles.json'):
                if receipt.get('candidate_sha256',{}).get(relative)!=hashlib.sha256((Path(__file__).parent/relative).read_bytes()).hexdigest():
                    raise ValueError('R2c CPU candidate source changed')
            return
        waiver = auth.get('baseline_executor_replay_waiver')
        if waiver is not None:
            if (not r15 or waiver != dict(status='user_explicitly_waived',
                    user_message='跳过你直接跑正式的，现在能做到多少并发？',
                    replay_performed=False) or 'baseline_executor_replay' in auth):
                raise ValueError('Invalid explicit R1.5 replay waiver')
            return

        proof = auth.get('baseline_executor_replay')

        if not isinstance(proof, dict) or set(proof) != {'path','sha256'}:

            raise ValueError('R1 requires baseline-005 executor replay receipt')

        raw = Path(proof['path']).read_bytes()

        if hashlib.sha256(raw).hexdigest() != proof['sha256']:raise ValueError('executor replay receipt hash')

        receipt = json.loads(raw)

        if any(receipt.get(k) != v for k,v in dict(case_id='baseline-005',baseline='Sol-session-pick20-aligned-v1',

                reference='sol-aligned20-r2-20261012',status='passed',api_requests=0,

                executor_commands_match=True,per_step_state_match=True).items()):

            raise ValueError('baseline executor replay did not pass')

        sources = receipt.get('candidate_sha256', {})

        for relative in ('pick.py','run_case.py','robot/executor.py','profiles.json'):

            source = Path(__file__).parent/relative

            if sources.get(relative) != hashlib.sha256(source.read_bytes()).hexdigest():

                raise ValueError('executor replay candidate source changed')



def inherited_request_debit(auth, _seen=None):
    """Paid failed suffixes consume slots without entering the restored conversation."""
    seen=set() if _seen is None else _seen
    grant=auth.get('checkpoint_branch') or auth.get('cold_resume') or {}
    debit=0
    if grant.get('migration_contract')=='rest-brake-branch-v1':
        prefix=grant['prefix_requests']; parent=grant['parent_requests']
        if type(prefix) is not int or type(parent) is not int or parent<prefix:
            raise ValueError('invalid inherited request debit')
        debit=parent-prefix
    ancestry=grant.get('source_migration') or grant
    path=ancestry.get('parent_authorization_path')
    if path:
        path=str(Path(path).resolve())
        if path in seen:raise ValueError('cyclic request ancestry')
        seen.add(path);raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ancestry.get('parent_authorization_sha256'):
            raise ValueError('request ancestry authorization hash')
        debit+=inherited_request_debit(json.loads(raw),seen)
    return debit


def continuation_bundle(bundle, auth, entries, profile):
    if profile!=SOL_HOT_RECOVERY:return bundle
    result=deepcopy(bundle);obs=result['observation']
    if 'single_pick' in obs:
        snap=obs['single_pick'];used=len([e for e in entries if e['case_id']==bundle['position']])+inherited_request_debit(auth)
        if used>snap['request_cap']:raise ValueError('inherited request cap exhausted')
        snap.update(requests_used=used,requests_remaining=snap['request_cap']-used)
        validate_single_pick_snapshot(snap,profile)
    resumed=obs.get('resumed_execution')
    if resumed and resumed.get('user_priority')=='confirmed_target_contact_and_clearance_replan_v2':
        resumed['user_priority']=RESUME_CONTACT_PROGRESS_PRIORITY
        validate_resumed_execution(resumed)
    return result


def continuation_window(estimate, available, remaining, *, first=False):
    """Rolling two-request cold reserve; future growth is bounded, never a cache promise."""
    current=Decimal(estimate['reserve_usd'])
    slots=2 if first or remaining>=2 else 1
    future_input=estimate['input_bound']+estimate['output_cap']+estimate['delta_input_bound']
    future=session_cost(future_input,estimate['output_cap'],reserve=True) if slots==2 else Decimal(0)
    return dict(admitted=remaining>=slots and Decimal(str(available))>=current+future,
        required_usd=str(current+future),available_usd=str(available),requests=slots,
        next_input_bound=future_input,cache_credit_usd='0',
        basis='current exact/conservative input plus full output and current delta for next request; recheck every send')


def branch_request_slots_exhausted(auth, entries):
    grant=auth.get('checkpoint_branch',{})
    if grant.get('migration_contract')!='rest-brake-branch-v1':return False
    # The failed suffix remains consumed although its response is not replayed.
    return len(entries)-grant['prefix_requests']>=grant['new_requests_cap']


def process_weekly_verification(wire, case, sequence, folder, ledger_path, auth, auth_sha, send, session):
    """Independent review, same durable dollar/request ledger, no response-chain parent."""
    profile=PROFILES[auth['execution_profile']]
    validate_weekly_scope(auth,profile)
    body=json.loads(wire)
    if (not profile.active_confirmation or body.get('model')!=profile.model or body.get('store') is not False
            or 'previous_response_id' in body or case not in auth['case_ids']
            or session.attempt_id!=auth['attempt_ids'][case] or session.status!='executing'):
        raise ValueError('independent weekly verifier binding')
    if Path(ledger_path).resolve()!=(Path(auth['output_root'])/'ledger.json').resolve():
        raise ValueError('weekly verifier requires shared ledger')
    ledger=json.loads(Path(ledger_path).read_bytes());entries=ledger['entries']
    if (ledger.get('authorization_sha256')!=auth_sha or ledger.get('batch_id')!=auth['batch_id']
            or ledger.get('status')!='running' or any(e['status'] in ('reserved','unknown','censored') for e in entries)):
        raise VerificationOutcomeUnknown('unresolved weekly ledger; never resend')
    prior=[e for e in entries if e['case_id']==case]
    reviews=[e for e in prior if e['request_kind']=='verification']
    if sequence!=len(reviews) or any(e.get('attempt_id')!=session.attempt_id for e in prior):
        raise ValueError('weekly verifier sequence/attempt mismatch')
    images=[p for m in body['input'] for p in m['content'] if p['type']=='input_image']
    if len(images) not in ((1,2) if profile.multiview_confirmation else (1,)):
        raise ValueError('weekly verifier robot image count')
    bound=non_image_bytes(wire)+2048*len(images)
    reserve=session_cost(bound,2000,reserve=True)
    if (branch_request_slots_exhausted(auth,entries) or len(reviews)>=profile.verification_call_cap or len(prior)+inherited_request_debit(auth)>=weekly_request_cap(session,profile)
            or len(entries)>=auth['provider_requests_cap']
            or committed_cost(prior)+reserve>Decimal(str(auth['per_case_usd_cap']))
            or committed_cost(entries)+reserve>Decimal(str(auth['usd_cap']))):
        error=C2BudgetExhausted('weekly review budget exhausted before network')
        error.scope='global' if (branch_request_slots_exhausted(auth,entries) or len(entries)>=auth['provider_requests_cap'] or committed_cost(entries)+reserve>Decimal(str(auth['usd_cap']))) else 'case'
        raise error
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=f"{auth['batch_id']}-{case}-{session.attempt_id}-review-{sequence:03d}"
    digest=hashlib.sha256(wire).hexdigest()
    with (folder/'send-claim.json').open('x',encoding='utf8') as f:
        json.dump(dict(request_id=key,wire_sha256=digest),f);f.flush();os.fsync(f.fileno())
    (folder/'request-wire.json').write_bytes(wire)
    entry=dict(request_id=key,case_id=case,attempt_id=session.attempt_id,request_kind='verification',
        sequence=sequence,wire_sha256=digest,status='reserved',reserve_usd=str(reserve),estimated_usd=None)
    entries.append(entry);save(ledger_path,ledger)
    start=time.monotonic()
    try:
        raw=send(wire,key)
        (folder/'response-raw.json').write_bytes(raw)
        response=json.loads(raw);usage=response.get('usage') or {}
        values=session_usage_cost(usage)
        if (response.get('status')!='completed' or response.get('model')!=profile.model
                or response.get('service_tier') not in (None,'default')
                or not isinstance(response.get('id'),str) or not response['id'].startswith('resp_')
                or any(e.get('provider_id')==response['id'] for e in entries[:-1])
                or usage['input_tokens']>bound or usage['output_tokens']>2000):
            raise ValueError('weekly verifier provider contract')
        entry.update(values,provider_id=response['id'],usage=usage,
                     reasoning_summaries=reasoning_summaries(response))
        texts=[p['text'] for item in response.get('output',[]) if item.get('type')=='message'
               for p in item.get('content',[]) if p.get('type')=='output_text']
        result=dict(status='invalid_verification',reply=None,reason='invalid_verification_reply')
        try:
            if len(texts)!=1:raise ValueError('one verifier message required')
            reply=json.loads(texts[0]);validate_evidence_reply(reply)
            result.update(status='valid',reply=reply,reason=None)
        except (ValueError,TypeError,KeyError):
            pass
        entry.update(result,latency_seconds=time.monotonic()-start)
        save(ledger_path,ledger);save(folder/'delivery.json',entry)
        return entry
    except Exception as error:
        entry.update(status='unknown',error_type=type(error).__name__,latency_seconds=time.monotonic()-start)
        ledger['status']='stopped_unknown_outcome';session.status='stopped_unknown_outcome'
        save(ledger_path,ledger);session.persist()
        raise VerificationOutcomeUnknown('weekly verification outcome unknown; no retry') from error


def process_r2a_verification(wire,case,sequence,folder,ledger_path,auth,auth_sha,send,session,profile=SOL_R2A):
    if api_concurrency(auth)>1:
        with ConcurrentDelivery(ledger_path,auth,r15_job_id(case,profile,auth.get('scope'))) as delivery:
            return _process_r2a_verification(wire,case,sequence,folder,ledger_path,auth,auth_sha,send,session,profile,delivery)
    with parallel_request_lock(ledger_path,auth['output_root'],r15_job_id(case,profile,auth.get('scope')),auth['expires_at_epoch'],isolate_case=True):
        return _process_r2a_verification(wire,case,sequence,folder,ledger_path,auth,auth_sha,send,session,profile)

def _process_r2a_verification(wire, case, sequence, folder, ledger_path, auth, auth_sha, send, session,profile=SOL_R2A,delivery=None):
    """Independent review, same durable dollar/request ledger, no response-chain parent."""
    validate_session_scope(auth,profile)
    body=json.loads(wire)
    if (not profile.r2a_identity or body.get('model')!=profile.model or body.get('store') is not False
            or 'previous_response_id' in body or case not in auth['case_ids']
            or session.attempt_id!=session_attempt_id(auth,case,profile) or session.status!='executing'):
        raise ValueError('independent weekly verifier binding')
    if Path(ledger_path).resolve()!=(Path(auth['output_root'])/'ledger.json').resolve():
        raise ValueError('weekly verifier requires shared ledger')
    ledger=json.loads(Path(ledger_path).read_bytes());entries=ledger['entries']
    if (ledger.get('authorization_sha256')!=auth_sha or ledger.get('batch_id')!=auth['batch_id']
            or ledger.get('status')!='running' or unresolved_delivery(entries,delivery)):
        raise VerificationOutcomeUnknown('unresolved weekly ledger; never resend')
    prior=[e for e in entries if e['case_id']==case and e.get('attempt_id')==session.attempt_id]
    reviews=[e for e in prior if e['request_kind']=='verification']
    if sequence!=len(reviews) or any(e.get('attempt_id')!=session.attempt_id for e in prior):
        raise ValueError('weekly verifier sequence/attempt mismatch')
    images=[p for m in body['input'] for p in m['content'] if p['type']=='input_image']
    if len(images) not in (range(1,6) if profile.identity_multiview_evidence else (1,2)):
        raise ValueError('weekly verifier robot image count')
    bound=non_image_bytes(wire)+2048*len(images)
    if profile.cache_aware_budget:
        lean_precount_admission(entries,prior,auth,profile,verification=True,count_entries=ledger.get('count_entries',[]))
        if auth.get('exact_input_count')!='responses-input-count-v1':
            raise ValueError('v3 verifier exact input count authorization required')
        counter=getattr(send,'count_input',None)
        if not callable(counter):raise ValueError('v3 verifier exact input count transport required')
        # The count transport logs phases in the review folder, not its sibling receipt folder.
        Path(folder).mkdir(parents=True,exist_ok=True)
        counter=budgeted_input_counter(counter,ledger,ledger_path,auth,case,session.attempt_id,
            'verification',sequence,delivery=delivery)
        counted=count_session_input(wire,counter,Path(folder).with_name(Path(folder).name+'-input-count'),independent=True)
        bound=counted['input_tokens']
    reserve=session_cost(bound,2000,reserve=True)
    if profile.cache_aware_budget:
        reserve=reserve.quantize(Decimal('.01'),rounding='ROUND_CEILING')
        Path(folder).mkdir(parents=True,exist_ok=True)
        (Path(folder)/'request-wire.json').write_bytes(wire)
        save(Path(folder)/'request-estimate.json',dict(input_bound=bound,output_cap=2000,
            input_count=counted,reserve_usd=str(reserve),cold_exposure_usd=str(reserve),
            estimate_basis='independent review has no eligible cache prefix; exact count and full cold output bound'))
    if len(prior)>=profile.call_cap:raise CaseLimitReached('tool_call_limit')
    if delivery:
        extra={'cold_exposure':reserve} if profile.cache_aware_budget else {}
        try:
            delivery.admit_budget(ledger,case,reserve,wire,profile.call_cap,case_entries=session.attempt_id,**extra)
        except C2BudgetExhausted as error:
            if profile.cache_aware_budget:save(Path(folder)/'budget-rejection.json',error.budget_snapshot)
            raise
        prior=[e for e in entries if e['case_id']==case and e.get('attempt_id')==session.attempt_id]
    if (branch_request_slots_exhausted(auth,entries) or (not (profile.identity_attempt_recovery or profile.identity_confirmation) and len(reviews)>=2) or len(prior)+inherited_request_debit(auth)>=profile.call_cap
            or len(entries)>=auth['provider_requests_cap']
            or budget_cost(ledger,case_id=case,attempt_id=session.attempt_id)+reserve>Decimal(str(auth['per_case_usd_cap']))
            or budget_cost(ledger)+reserve>Decimal(str(auth['usd_cap']))):
        error=C2BudgetExhausted('weekly review budget exhausted before network')
        error.scope='global' if (branch_request_slots_exhausted(auth,entries) or len(entries)>=auth['provider_requests_cap'] or budget_cost(ledger)+reserve>Decimal(str(auth['usd_cap']))) else 'case'
        if profile.cache_aware_budget:
            error.budget_snapshot=lean_budget_snapshot(budget_entries(ledger),[e for e in budget_entries(ledger) if e['case_id']==case and e.get('attempt_id')==session.attempt_id],auth,reserve=reserve,
                cold_exposure=reserve,reason='independent_review_'+error.scope+'_budget_cap')
            save(Path(folder)/'budget-rejection.json',error.budget_snapshot)
        raise error
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=f"{auth['batch_id']}-{case}-{session.attempt_id}-review-{sequence:03d}"
    digest=hashlib.sha256(wire).hexdigest()
    with (folder/'send-claim.json').open('x',encoding='utf8') as f:
        json.dump(dict(request_id=key,wire_sha256=digest),f);f.flush();os.fsync(f.fileno())
    (folder/'request-wire.json').write_bytes(wire)
    entry=dict(request_id=key,case_id=case,attempt_id=session.attempt_id,request_kind='verification',
        sequence=sequence,wire_sha256=digest,status='reserved',reserve_usd=str(reserve),estimated_usd=None)
    if profile.cache_aware_budget:entry['worst_case_usd']=str(reserve)
    entries.append(entry)
    if delivery:delivery.reserve(ledger,entry)
    else:atomic_ledger_save(ledger_path,ledger)
    start=time.monotonic()
    try:
        raw=delivery.send(send,wire,key,ledger,entry) if delivery else send(wire,key)
        (folder/'response-raw.json').write_bytes(raw)
        response=json.loads(raw);usage=response.get('usage') or {}
        values=session_usage_cost(usage)
        if (response.get('status')!='completed' or response.get('model')!=profile.model
                or response.get('service_tier') not in (None,'default')
                or not isinstance(response.get('id'),str) or not response['id'].startswith('resp_')
                or any(e.get('provider_id')==response['id'] for e in entries[:-1])
                or usage['input_tokens']>bound or usage['output_tokens']>2000):
            raise ValueError('weekly verifier provider contract')
        entry.update(values,provider_id=response['id'],usage=usage,
                     reasoning_summaries=reasoning_summaries(response))
        texts=[p['text'] for item in response.get('output',[]) if item.get('type')=='message'
               for p in item.get('content',[]) if p.get('type')=='output_text']
        result=dict(status='invalid_verification',reply=None,reason='invalid_verification_reply')
        try:
            if len(texts)!=1:raise ValueError('one verifier message required')
            reply=json.loads(texts[0]);validate_r2a_reply(reply,identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence)
            validate_identity_evidence_reply(reply,wire,profile)
            result.update(status='valid',reply=reply,reason=None)
        except (ValueError,TypeError,KeyError):
            pass
        entry.update(result,latency_seconds=time.monotonic()-start)
        atomic_ledger_save(ledger_path,ledger);save(folder/'delivery.json',entry)
        return entry
    except Exception as error:
        entry.update(status='unknown',error_type=type(error).__name__,latency_seconds=time.monotonic()-start)
        ledger['status']='stopped_unknown_outcome';session.status='stopped_unknown_outcome'
        atomic_ledger_save(ledger_path,ledger);session.persist()
        raise VerificationOutcomeUnknown('weekly verification outcome unknown; no retry') from error


def atomic_ledger_save(path, value):
    """Parallel opt-in only: readers see a whole prior or whole next ledger."""
    path=Path(path);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def api_concurrency(auth):
    """Absent field retains the historical one-request path and wire bytes."""
    value=auth.get('parallel_execution',{}).get('api_concurrency',1)
    if type(value) is not int or not 1<=value<=8:
        raise ValueError('API concurrency must be an integer from 1 to 8')
    if value>1 and auth.get('scope') not in (R2B_SCOPE,R2C_SCOPE,R2D_SCOPE,R2E_SCOPE,*V3_SCOPES):
        raise ValueError('Concurrent API delivery requires the paired R2b scope')
    return value


def unresolved_delivery(entries, delivery=None):
    return any(e['status'] in (('unknown','censored') if delivery else ('reserved','unknown','censored')) for e in entries)


class ConcurrentDelivery:
    """FIFO cross-session slots; ledger lock only covers reservation/settlement.

    Linux process identity and durable reservations fail closed after a crash.
    Already sent requests settle even after STOP; they never restore running status.
    The session flock covers the complete delivery, including receipt persistence.
    """
    def __init__(self,ledger_path,auth,job):
        self.path=Path(ledger_path);self.auth=auth;self.root=Path(auth['output_root']);self.job=job
        self.limit=api_concurrency(auth);self.queue_path=Path(str(self.path)+'.admission.json')
        self.mutex=None;self.session_lock=None;self.locked=False;self.token=None

    @staticmethod
    def identity(pid):
        try:
            fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
            return None if fields[0]=='Z' else fields[19]
        except (OSError,IndexError):return None

    def lock(self):
        import fcntl
        fcntl.flock(self.mutex,fcntl.LOCK_EX);self.locked=True

    def unlock(self):
        import fcntl
        if self.locked:fcntl.flock(self.mutex,fcntl.LOCK_UN);self.locked=False

    def stopped(self):
        if (self.root/self.job/'CASE_WALL_LIMIT').exists():raise CaseLimitReached('wall_time_limit')
        if (self.root/'STOP').exists() or (self.root/self.job/'STOP').exists() or time.time()>=self.auth['expires_at_epoch']:
            raise RuntimeError('Concurrent admission stopped before send')

    def sweep(self,state):
        ledger=json.loads(self.path.read_bytes());orphan=False
        active_tokens={t['token'] for t in state['tickets'] if t['active']}
        for e in budget_entries(ledger):
            if e['status']=='reserved' and (not e.get('delivery_owner') or
                    e['delivery_owner']['token'] not in active_tokens or
                    self.identity(e['delivery_owner']['pid'])!=e['delivery_owner']['start']):
                e.update(status='unknown',error_type='DeliveryOwnerExited');orphan=True
        if orphan:
            ledger['status']='stopped_unknown_outcome';atomic_ledger_save(self.path,ledger)
            (self.root/'STOP').touch(exist_ok=True)
        state['tickets']=[t for t in state['tickets'] if self.identity(t['pid'])==t['start']]
        return ledger

    def __enter__(self):
        import fcntl,uuid
        try:
            self.session_lock=(self.root/(self.job+'.session.lock')).open('a+')
            while True:
                self.stopped()
                try:fcntl.flock(self.session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.05)
            self.mutex=Path(str(self.path)+'.request.lock').open('a+')
            self.token=uuid.uuid4().hex;started=time.monotonic()
            atomic_ledger_save(self.root/(self.job+'.request-wait.json'),dict(waiting=True,started_epoch=time.time()))
            while True:
                self.lock();self.stopped()
                state=json.loads(self.queue_path.read_bytes()) if self.queue_path.exists() else dict(tickets=[])
                ledger=self.sweep(state)
                if ledger['status']!='running' or unresolved_delivery(budget_entries(ledger),self):
                    raise RuntimeError('Concurrent ledger stopped; no resend')
                if not any(t['token']==self.token for t in state['tickets']):
                    state['tickets'].append(dict(token=self.token,pid=os.getpid(),start=self.identity(os.getpid()),job=self.job,active=False))
                waiting=[t for t in state['tickets'] if not t['active']]
                active=[t for t in state['tickets'] if t['active']]
                admitted=waiting[0]['token']==self.token and len(active)<self.limit
                if admitted:
                    next(t for t in state['tickets'] if t['token']==self.token)['active']=True
                atomic_ledger_save(self.queue_path,state)
                if admitted:
                    atomic_ledger_save(self.root/(self.job+'.request-wait.json'),dict(waiting=False,wait_seconds=time.monotonic()-started))
                    return self # Keep mutex through validation and reservation.
                self.unlock();time.sleep(.05)
        except BaseException:
            self.__exit__(None,None,None);raise

    def admit_budget(self, ledger, case, reserve, wire, request_cap, *, case_entries=None, cold_exposure=None):
        """Wait for peers to settle before reserving; never resend or discount a bound."""
        assert self.locked
        exposure=reserve if cold_exposure is None else max(reserve,cold_exposure)
        started=time.monotonic();last=None
        while True:
            state=json.loads(self.queue_path.read_bytes())
            fresh=self.sweep(state)
            atomic_ledger_save(self.queue_path,state)
            ledger['entries'][:]=fresh['entries']
            for k,v in fresh.items():
                if k!='entries':ledger[k]=v
            entries=ledger['entries']
            prior=[e for e in entries if e['case_id']==case and
                   (case_entries is None or e.get('attempt_id')==case_entries)]
            cost_rows=budget_entries(ledger)
            cost_prior=[e for e in cost_rows if e['case_id']==case and
                        (case_entries is None or e.get('attempt_id')==case_entries)]
            pending=[e for e in cost_rows if e['status']=='reserved']
            committed=committed_cost(cost_rows)
            settled=committed_cost([e for e in cost_rows if e['status']!='reserved'])
            reason=None
            if ledger['status']!='running' or unresolved_delivery(cost_rows,self):reason='ledger_stopped_or_unknown'
            elif branch_request_slots_exhausted(self.auth,entries) or len(entries)>=self.auth['provider_requests_cap']:reason='global_request_cap'
            elif len(prior)+inherited_request_debit(self.auth)>=request_cap:reason='case_request_cap'
            elif committed_cost(cost_prior)+exposure>Decimal(str(self.auth['per_case_usd_cap'])):reason='case_usd_cap'
            elif committed+exposure>Decimal(str(self.auth['usd_cap'])):
                reason=('waiting_settlement' if pending and settled+exposure<=Decimal(str(self.auth['usd_cap']))
                        else 'global_usd_cap')
            ticket=next(t for t in state['tickets'] if t['token']==self.token)
            waiting=[t for t in state['tickets'] if not t['active']]
            active=[t for t in state['tickets'] if t['active']]
            if reason is None and not ticket['active'] and (waiting[0]['token']!=self.token or len(active)>=self.limit):
                reason='waiting_slot'
            row=dict(case_id=case,job=self.job,wire_sha256=hashlib.sha256(wire).hexdigest(),
                decision=reason or 'admitted',settled_usd=str(settled),committed_usd=str(committed),
                new_reserve_usd=str(reserve),usd_cap=str(self.auth['usd_cap']),
                case_committed_usd=str(committed_cost(cost_prior)),per_case_usd_cap=str(self.auth['per_case_usd_cap']),
                requests=len(entries),request_cap=self.auth['provider_requests_cap'],
                case_requests=len(prior),case_request_cap=request_cap,
                pending=[dict(request_id=e['request_id'],reserve_usd=e['reserve_usd']) for e in pending])
            if cold_exposure is not None:
                row.update(new_cold_exposure_usd=str(exposure),
                    pending_cold_exposures=[dict(request_id=e['request_id'],
                        worst_case_usd=e.get('worst_case_usd',e['reserve_usd'])) for e in pending])
            signature=json.dumps(row,sort_keys=True)
            if signature!=last:
                with (self.root/'budget-admission.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(row,epoch=time.time(),wait_seconds=time.monotonic()-started))+'\n')
                    stream.flush();os.fsync(stream.fileno())
                last=signature
            self.stopped()
            if reason=='ledger_stopped_or_unknown':
                raise RuntimeError('Concurrent ledger stopped; no resend')
            if reason not in (None,'waiting_settlement','waiting_slot'):
                if not ((r15_case_isolation(self.auth) or cold_exposure is not None) and reason.startswith('case_')):
                    ledger['status']='stopped_budget_exhausted';atomic_ledger_save(self.path,ledger)
                error=C2BudgetExhausted('Concurrent budget exhausted before network: '+reason)
                error.scope='case' if reason.startswith('case_') else 'global'
                if cold_exposure is not None:
                    error.budget_snapshot=lean_budget_snapshot(cost_rows,cost_prior,self.auth,
                        reserve=reserve,cold_exposure=exposure,reason=reason)
                raise error
            ticket['active']=reason is None
            atomic_ledger_save(self.queue_path,state)
            if reason is None:return
            # Keep the session lock and FIFO ticket, but release API slot and ledger lock.
            self.unlock();time.sleep(.05);self.lock()

    def reserve(self,ledger,entry):
        assert self.locked
        self.stopped()
        entry['delivery_owner']=dict(pid=os.getpid(),start=self.identity(os.getpid()),token=self.token,job=self.job)
        atomic_ledger_save(self.path,ledger)

    def send(self,transport,wire,key,ledger,entry):
        assert self.locked and entry['status']=='reserved'
        self.stopped()
        self.unlock()
        try:return transport(wire,key) # Exactly one send, no retries.
        finally:
            self.lock()
            fresh=json.loads(self.path.read_bytes())
            book='count_entries' if entry.get('request_kind')=='input_count' else 'entries'
            matches=[e for e in fresh.get(book,[]) if e['request_id']==entry['request_id']]
            if len(matches)!=1 or matches[0].get('delivery_owner')!=entry['delivery_owner']:
                raise RuntimeError('Concurrent reservation identity changed')
            # Preserve the caller's list/entry references but refresh every peer
            # and the global stop state before any caller mutation or save.
            for name in ('entries','count_entries'):
                if name in fresh:
                    merged=[entry if name==book and e['request_id']==entry['request_id'] else e
                            for e in fresh[name]]
                    ledger.setdefault(name,[])[:]=merged
            for k,v in fresh.items():
                if k not in ('entries','count_entries'):ledger[k]=v

    def __exit__(self,*exc):
        try:
            if self.mutex is not None:
                if not self.locked:self.lock()
                if self.queue_path.exists():
                    state=json.loads(self.queue_path.read_bytes())
                    state['tickets']=[t for t in state['tickets'] if t['token']!=self.token]
                    atomic_ledger_save(self.queue_path,state)
                ledger=json.loads(self.path.read_bytes())
                if ledger.get('status')!='running':(self.root/'STOP').touch(exist_ok=True)
        finally:
            self.unlock()
            if self.mutex is not None:self.mutex.close()
            if self.session_lock is not None:self.session_lock.close()


@contextmanager
def parallel_request_lock(ledger_path, root, case_id, expires_at, *, isolate_case=False):
    """Serialize the entire durable delivery, including unknown-outcome handling.

    Physics and recording remain concurrent. Death releases flock, but the durable
    reserved/unknown ledger entry still prevents another request from being sent.
    """
    import fcntl
    root=Path(root);started=time.monotonic()
    wait_path=root/(case_id+'.request-wait.json')
    atomic_ledger_save(wait_path,dict(waiting=True,started_epoch=time.time()))
    with Path(str(ledger_path)+'.request.lock').open('a+') as lock:
        while True:
            if isolate_case and (root/case_id/'CASE_WALL_LIMIT').exists():raise CaseLimitReached('wall_time_limit')
            if (root/'STOP').exists() or (root/case_id/'STOP').exists() or time.time()>=expires_at:
                raise RuntimeError('Parallel request admission stopped before send')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.05)
        try:
            atomic_ledger_save(wait_path,dict(waiting=False,wait_seconds=time.monotonic()-started))
            if (root/'STOP').exists() or (root/case_id/'STOP').exists():
                raise RuntimeError('Parallel request admission stopped before send')
            if isolate_case and (root/case_id/'CASE_WALL_LIMIT').exists():raise CaseLimitReached('wall_time_limit')
            yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def process_session(bundle, archive, folder, ledger_path, auth, auth_sha, send, profile, session):
    if (profile in (SOL_PICK20,SOL_PICK20_ALIGNED)+DEV_PROFILES or profile.weekly_budget) and 'output_root' in auth:

        if Path(ledger_path).resolve() != (Path(auth['output_root'])/'ledger.json').resolve():
            raise ValueError('pick20 requires the single shared batch ledger')
    if auth.get('scope') == TRANSPORT_SCOPE:
        if Path(auth.get('ab_episode_ledger_path','')).resolve()!=Path(ledger_path).resolve():
            raise ValueError('transport episode ledger path mismatch')
        batch=TransportABBatch.from_auth(auth)
        return batch.deliver(bundle,archive,auth,send,lambda guarded:
            _process_session(bundle,archive,folder,ledger_path,auth,auth_sha,guarded,profile,session))
    if auth.get('scope') == FROZEN['memory_ab']['scope']:
        if Path(auth.get('ab_episode_ledger_path','')).resolve()!=Path(ledger_path).resolve():
            raise ValueError('AB episode ledger path mismatch')
        batch=MemoryABBatch.from_auth(auth)
        return batch.deliver(bundle,archive,auth,send,lambda guarded:
            _process_session(bundle,archive,folder,ledger_path,auth,auth_sha,guarded,profile,session))
    if profile.parallel_case_workers:
        if api_concurrency(auth)>1:
            with ConcurrentDelivery(ledger_path,auth,r15_job_id(bundle['position'],profile,auth.get('scope'))) as delivery:
                return _process_session(bundle,archive,folder,ledger_path,auth,auth_sha,send,profile,session,delivery)
        with parallel_request_lock(ledger_path, auth['output_root'], r15_job_id(bundle['position'],profile,auth.get('scope')), auth['expires_at_epoch'], isolate_case=r15_case_isolation(auth)):
            return _process_session(bundle,archive,folder,ledger_path,auth,auth_sha,send,profile,session)
    return _process_session(bundle,archive,folder,ledger_path,auth,auth_sha,send,profile,session)


def _process_session(bundle, archive, folder, ledger_path, auth, auth_sha, send, profile, session,delivery=None):
    """Durable single delivery through an injected transport; CPU tests supply fakes."""
    validate_session_scope(auth, profile)
    save_ledger = lambda value: (atomic_ledger_save if profile.parallel_case_workers else save)(ledger_path,value)
    if (bundle['authorization_sha256'] != auth_sha or bundle['position'] not in auth['case_ids']
            or session is None or session.attempt_id != session_attempt_id(auth,bundle['position'],profile)):
        raise ValueError('session authorization/attempt binding')
    if profile.task_handoff and session.handoff_plan != auth['handoff_plan']:
        raise ValueError('session task plan differs from authorized task')
    if profile.transport_task and session.transport_mission != auth['transport_mission']:
        raise ValueError('session transport mission differs from authorized task')
    folder, ledger_path = Path(folder), Path(ledger_path)
    ledger = json.loads(ledger_path.read_bytes())
    if (ledger.get('authorization_sha256') != auth_sha or ledger.get('status') != 'running'
            or ledger.get('batch_id') != auth['batch_id']):
        raise ValueError('session ledger identity or status')
    entries = ledger['entries']
    if (len({e['request_id'] for e in entries}) != len(entries)
            or unresolved_delivery(entries,delivery)):
        raise ValueError('session unresolved delivery; reconcile without resend')
    case = bundle['position']
    if profile == SOL_PICK20 and auth.get('pick20_continuation') and case == 'baseline-000':
        raise ValueError('completed baseline-000 must not resend')
    prior = [e for e in entries if e['case_id'] == case and (profile not in PAIRED_PROFILES or e.get('attempt_id')==session.attempt_id)]
    if profile.weekly_budget and any(e.get('attempt_id')!=session.attempt_id for e in prior):
        raise ValueError('weekly ledger attempt mismatch')
    main_prior = [e for e in prior if e.get('request_kind') == 'main'] if (profile.weekly_budget or profile.r2a_identity) else prior
    if (len(main_prior) != len(session.receipts) or
            any(e.get('attempt_id') != session.attempt_id or e.get('provider_id') != r['response_id']
                or e.get('wire_sha256') != r['wire_sha256'] for e,r in zip(main_prior,session.receipts))):
        raise ValueError('session ledger and chain disagree; no automatic restart')
    if profile.session_usage_anchor and any(
            e.get('usage',{}).get('input_tokens') != r.get('input_tokens')
            or e.get('usage',{}).get('output_tokens') != r.get('output_tokens')
            for e,r in zip(main_prior,session.receipts)):
        raise ValueError('context anchor differs from durable provider usage')
    effective_bundle=continuation_bundle(bundle,auth,entries,profile)
    # The caller passes this same current bundle to advance(). Bind execution
    # to exactly the request we sent; never rewrite an archived bundle.
    if effective_bundle is not bundle:
        bundle.clear();bundle.update(effective_bundle)
    wire=build_wire(bundle,archive,profile,session=session)
    counted = None
    if profile.cache_aware_budget:
        lean_precount_admission(entries,prior,auth,profile,count_entries=ledger.get('count_entries',[]))
        if auth.get('exact_input_count')!='responses-input-count-v1':
            raise ValueError('v3 exact input count authorization required')
        counter=getattr(send,'count_input',None)
        if not callable(counter):raise ValueError('v3 exact input count transport required')
        counter=budgeted_input_counter(counter,ledger,ledger_path,auth,case,session.attempt_id,
            'main',len(session.receipts),delivery=delivery)
        try:
            wire,estimate=prepare_lean_request(wire,session,profile,counter,
                folder.with_name(folder.name+'-input-count'),
                case_remaining=Decimal(str(auth['per_case_usd_cap']))-budget_cost(ledger,case_id=case,attempt_id=session.attempt_id if profile in PAIRED_PROFILES else None),
                batch_remaining=Decimal(str(auth['usd_cap']))-budget_cost(ledger))
        except C2BudgetExhausted as error:
            if hasattr(error,'budget_snapshot'):raise
            error.scope='case'
            error.budget_snapshot=lean_budget_snapshot(budget_entries(ledger),[e for e in budget_entries(ledger) if e['case_id']==case and e.get('attempt_id')==session.attempt_id],auth,reason='context_capacity_before_generation')
            raise
    elif 'exact_input_count' in auth:
        if profile != SOL_HOT_RECOVERY or auth['exact_input_count'] != 'responses-input-count-v1':
            raise ValueError('exact input count authorization')
        counter = getattr(send, 'count_input', None)
        if not callable(counter):
            raise ValueError('exact input count transport required; no estimate fallback')
        counted = count_session_input(wire, counter, folder.with_name(folder.name+'-input-count'))
    if not profile.cache_aware_budget:
        estimate = estimate_session_wire(wire, session, profile, input_count=counted)
    reserve = Decimal(estimate['reserve_usd'])
    exposure=Decimal(estimate['cold_exposure_usd']) if profile.cache_aware_budget else reserve
    if profile.cache_aware_budget:
        # Preserve denied requests as well as sent requests. No send claim yet.
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'request-wire.json').write_bytes(wire)
        save(folder/'request-bundle.json',bundle)
        save(folder/'request-estimate.json',estimate)
    request_cap = weekly_request_cap(session, profile) if profile.weekly_budget else profile.call_cap
    if profile==SOL_HOT_RECOVERY and (auth.get('cold_resume') or auth.get('checkpoint_branch')):
        available=min(Decimal(str(auth['usd_cap']))-budget_cost(ledger),
                      Decimal(str(auth['per_case_usd_cap']))-budget_cost(ledger,case_id=case,attempt_id=session.attempt_id if profile in PAIRED_PROFILES else None))
        remaining=request_cap-len(prior)-inherited_request_debit(auth)
        first=not any(e.get('request_kind')=='main' and e['request_id'].startswith(auth['batch_id']+'-') for e in entries)
        window=continuation_window(estimate,available,remaining,first=first)
        save(folder.with_name(folder.name+'-continuation-window.json'),window)
        if not window['admitted']:
            ledger['status']='stopped_budget_exhausted';save_ledger(ledger)
            raise ValueError('session budget exhausted before network')
    if r15_case_isolation(auth) and len(prior)>=request_cap:
        raise CaseLimitReached('tool_call_limit')
    if delivery:
        extra={'cold_exposure':exposure} if profile.cache_aware_budget else {}
        try:
            delivery.admit_budget(ledger,case,reserve,wire,request_cap,
                case_entries=session.attempt_id if profile in PAIRED_PROFILES else None,**extra)
        except C2BudgetExhausted as error:
            if profile.cache_aware_budget:save(folder/'budget-rejection.json',error.budget_snapshot)
            raise
        prior=[e for e in entries if e['case_id']==case and (profile not in PAIRED_PROFILES or e.get('attempt_id')==session.attempt_id)]
    if (branch_request_slots_exhausted(auth,entries) or len(prior)+inherited_request_debit(auth) >= request_cap or len(entries) >= auth['provider_requests_cap']
            or budget_cost(ledger)+exposure > Decimal(str(auth['usd_cap']))
            or budget_cost(ledger,case_id=case,attempt_id=session.attempt_id if profile in PAIRED_PROFILES else None)+exposure > Decimal(str(auth['per_case_usd_cap']))):
        global_exhausted=(branch_request_slots_exhausted(auth,entries) or len(entries)>=auth['provider_requests_cap'] or budget_cost(ledger)+exposure>Decimal(str(auth['usd_cap'])))
        if global_exhausted or not (r15_case_isolation(auth) or profile.cache_aware_budget):
            ledger['status'] = 'stopped_budget_exhausted'
            save_ledger(ledger)
        if profile in (SOL_PICK20,SOL_PICK20_ALIGNED)+DEV_PROFILES:

            error=C2BudgetExhausted('session budget exhausted before network')
            error.scope='global' if global_exhausted else 'case'
            if profile.cache_aware_budget:
                error.budget_snapshot=lean_budget_snapshot(budget_entries(ledger),[e for e in budget_entries(ledger) if e['case_id']==case and e.get('attempt_id')==session.attempt_id],auth,reserve=reserve,
                    cold_exposure=exposure,reason='global_usd_or_request_cap' if global_exhausted else 'case_usd_cap')
                save(folder/'budget-rejection.json',error.budget_snapshot)
            raise error
        raise ValueError('session budget exhausted before network')
    folder.mkdir(parents=True, exist_ok=True)
    key = f"{auth['batch_id']}-{case}-{session.attempt_id}-{bundle['turn']:03d}"
    wire_sha = hashlib.sha256(wire).hexdigest()
    # Exclusive claim survives crashes even before the ledger reservation is saved.
    with (folder/'send-claim.json').open('x', encoding='utf8') as stream:
        json.dump(dict(request_id=key, wire_sha256=wire_sha), stream)
        stream.flush(); os.fsync(stream.fileno())
    (folder/'request-wire.json').write_bytes(wire)
    save(folder/'request-bundle.json', bundle)
    save(folder/'request-estimate.json', estimate)
    entry = dict(request_id=key, case_id=case, attempt_id=session.attempt_id, request_kind='main',
                 sequence=bundle['turn'], wire_sha256=wire_sha,
                 previous_response_id=session.receipts[-1]['response_id'] if session.receipts and not profile.image_window else None,
                 status='reserved', reserve_usd=estimate['reserve_usd'], estimated_usd=None,
                 reserved_at_epoch=time.time())
    if profile.cache_aware_budget:
        entry.update(worst_case_usd=str(exposure),reservation_formula=estimate['reservation_formula'],
            cache_assumption=deepcopy(estimate['cache_assumption']))
    entries.append(entry)
    if delivery:delivery.reserve(ledger,entry)
    else:save_ledger(ledger)
    session.status = 'delivering'
    session.persist()
    started = time.monotonic()
    try:
        raw = delivery.send(send,wire,key,ledger,entry) if delivery else send(wire, key)
        (folder/'response-raw.json').write_bytes(raw)
        result = parse_response(raw, estimate, profile=profile)
        if profile.search_recessed_view and result.get('call') is not None:
            try:
                validate_search_decision(result['call'],archive,bundle['observation'],candidate_priority=profile.search_candidate_priority)
            except ValueError as exc:
                result.update(status='invalid_call',call=None,reason='invalid_tool_call',parameter_error=str(exc))
        if profile.active_memory and result.get('call') is not None:
            try:
                validate_active_memory(result['call']['memory'],
                    [r['observation_id'] for r in session.receipts]+[bundle['observation']['observation_id']])
            except ValueError as exc:
                result.update(status='invalid_call', call=None, reason='invalid_tool_call', parameter_error=str(exc))
        if profile.task_handoff and result.get('call') is not None:
            try:
                context = task_handoff_context(archive, bundle['observation']['history'], session)
                validate_task_handoff(result['call']['handoff'], tool=result['call']['tool'],
                    known_ids=[r['observation_id'] for r in session.receipts]+[bundle['observation']['observation_id']],
                    context=context['current'], plan=session.handoff_plan)
            except ValueError as exc:
                result.update(status='invalid_call', call=None, reason='invalid_tool_call', parameter_error=str(exc))
        prior_ids=({e.get('provider_id') for e in entries[:-1]} if profile.weekly_budget
                   else {r['response_id'] for r in session.receipts})
        if result['status'] == 'censored' or result['provider_id'] in prior_ids:
            raise ValueError('session provider contract or duplicate response ID')
        result['latency_seconds'] = time.monotonic()-started
        receipt = dict(turn=bundle['turn'], observation_id=bundle['observation']['observation_id'],
                       response_id=result['provider_id'], previous_response_id=entry['previous_response_id'],
                       wire_sha256=wire_sha, response_sha256=hashlib.sha256(raw).hexdigest(),
                       archive_sha256=hashlib.sha256(canonical(dict(bundle=bundle, call=result.get('call')))).hexdigest(),
                       input_bound=estimate['input_bound'], output_tokens=result['usage']['output_tokens'])
        if profile.session_usage_anchor:
            receipt['input_tokens'] = result['usage']['input_tokens']
        if profile.cache_aware_budget:
            receipt.update(cached_input_tokens=(result['usage'].get('input_tokens_details') or {}).get('cached_tokens'),
                cache_identity=lean_cache_identity(wire),settled_at_epoch=time.time())
        if profile.task_handoff:
            context = task_handoff_context(archive, bundle['observation']['history'], session)
            receipt['handoff_injected_id'] = (context['handoff'] or {}).get('handoff_id')
            receipt['handoff_stage'] = deepcopy(context['current'])
        entry.update(result)
        save_ledger(ledger)
        save(folder/'delivery.json', result)
        session.receipts.append(receipt)
        session.status = 'awaiting_execution'
        session.persist()
        return result
    except Exception as exc:
        entry.update(status='unknown', error_type=type(exc).__name__)
        ledger['status'] = 'stopped_unknown_outcome'
        session.status = 'stopped_unknown_outcome'
        save_ledger(ledger)
        session.persist()
        raise RuntimeError('session delivery uncertain; retain reservation, never resend or reset chain') from exc


# Detector: current robot depth only; no target registry or action gates.
def fit(points, kind, condition_max):
    """Fit only the primitive explicitly requested by the policy. No center of mass."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or (not np.isfinite(p).all()):
        raise ValueError('at least three finite surface points required')
    origin = p.mean(axis=0)
    _, singular, vt = np.linalg.svd(p - origin, full_matrices=False)
    if singular[0] < 1e-08:
        raise ValueError('coincident points')
    if kind == 'line':
        residual = np.linalg.norm(np.cross(p - origin, vt[0]), axis=1)
        return {'reference': origin.tolist(), 'axis': vt[0].tolist(), 'residual_m': float(np.max(residual)), 'reference_definition': 'observed_line_centroid'}
    if singular[1] < 1e-08 or singular[0] / singular[1] > condition_max:
        raise ValueError('degenerate surface fit')
    if kind == 'plane':
        return {'reference': origin.tolist(), 'axis': vt[2].tolist(), 'residual_m': float(np.max(np.abs((p - origin) @ vt[2]))), 'reference_definition': 'observed_plane_centroid'}
    if kind != 'circle':
        raise ValueError('unsupported primitive')
    xy = (p - origin) @ vt[:2].T
    design = np.column_stack([2 * xy, np.ones(len(p))])
    if np.linalg.cond(design) > condition_max:
        raise ValueError('ill-conditioned circle')
    solution = np.linalg.lstsq(design, (xy * xy).sum(axis=1), rcond=None)[0]
    radius_sq = solution[2] + solution[:2] @ solution[:2]
    if radius_sq <= 0:
        raise ValueError('invalid radius')
    center = origin + solution[:2] @ vt[:2]
    residual = np.sqrt(((p - center) @ vt[2]) ** 2 + (np.linalg.norm(xy - solution[:2], axis=1) - np.sqrt(radius_sq)) ** 2)
    return {'reference': center.tolist(), 'axis': vt[2].tolist(), 'radius_m': float(np.sqrt(radius_sq)), 'residual_m': float(np.max(residual)), 'reference_definition': 'visible_circle_center'}

def measure(depth, camera, pixels, measured):
    if depth.frame is None:
        raise FieldError('missing_depth', 'camera', 'a current robot RGB-D observation')
    oid = depth.frame[0]
    samples = []
    for pixel in pixels:
        try:
            s = depth.locate(camera, pixel, oid, measured)
        except ValueError:
            raise FieldError('stale_sensor', 'camera', 'refresh robot RGB-D after movement') from None
        samples.append({'pixel': list(pixel), **s})
    p = np.asarray([s['point_base_m'] for s in samples if s['valid']], float)
    geometry = {'reference_definition': 'sampled_visible_surface_not_object_center', 'centroid_base_m': None, 'plane_normal_base': None, 'principal_axis_base': None, 'secondary_axis_base': None, 'width_major_m': None, 'width_minor_m': None, 'plane_residual_m': None, 'diagnostics': []}
    if len(p) >= 3:
        geometry['centroid_base_m'] = p.mean(axis=0).tolist()
        _, sing, vt = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
        if sing[0] > 1e-08:
            geometry.update(principal_axis_base=vt[0].tolist(), secondary_axis_base=vt[1].tolist(), width_major_m=float(np.ptp(p @ vt[0])), width_minor_m=float(np.ptp(p @ vt[1])))
            if sing[0] - sing[1] < 0.05 * sing[0]:
                geometry['diagnostics'].append('in_plane_axes_ambiguous')
        try:
            plane = fit(p, 'plane', 1000.0)
            geometry.update(plane_normal_base=plane['axis'], plane_residual_m=plane['residual_m'])
        except ValueError:
            geometry['diagnostics'].append('degenerate_surface_fit')
    else:
        geometry['diagnostics'].append('fewer_than_three_valid_samples')
    return dict(accepted=True, readonly=True, reason='measured', observation_id=oid, camera=camera, samples=samples, geometry=geometry, steps_used=0)

def locate_point(depth, camera, pixel, observation_id, measured):
    return depth.locate(camera, pixel, observation_id, measured)




def approach_segment_step_cap(r15_protocol=False):
    """One protocol bound shared by execution and public feedback validation."""
    return 250 if r15_protocol else 80


def approach_feedback_public(value, *, r15_protocol=False):

    fields = {'surface_point_base_m', 'requested_turn_rad', 'requested_forward_m', 'segments', 'odometry_prediction'}

    if not isinstance(value, dict) or set(value) != fields:raise ValueError('approach feedback schema')

    vector(value['surface_point_base_m'], 3)

    vector(value['odometry_prediction'], 3)

    vector([value['requested_turn_rad'], value['requested_forward_m']], 2)

    if abs(value['requested_turn_rad']) > math.pi or not -.5 <= value['requested_forward_m'] <= 1.5:

        raise ValueError('bounded approach command')

    rows = value['segments']

    keys = {'requested_turn_rad', 'requested_forward_m', 'actual_turn_rad', 'actual_forward_m',

            'actual_lateral_m', 'turn_residual_rad', 'forward_residual_m', 'steps_used', 'reason', 'accepted'}

    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:raise ValueError('approach segments')

    for row in rows:

        if set(row) != keys:raise ValueError('approach segment schema')

        vector([row[k] for k in keys - {'steps_used', 'reason', 'accepted'}], 7)

        if type(row['steps_used']) is not int or not 0 <= row['steps_used'] <= approach_segment_step_cap(r15_protocol):

            raise ValueError('approach segment steps')

        if type(row['accepted']) is not bool or not isinstance(row['reason'], str) or len(row['reason']) > 150:

            raise ValueError('approach segment outcome')

    reject_privilege(value)

    return deepcopy(value)





def executor_odometry(executor):
    """Name the production executor's relative [dx, dy, yaw] sensor vector."""
    return dict(zip(('dx_m', 'dy_m', 'dyaw_rad'), map(float, vector(executor.odometry(), 3))))


class ToolExecutionError(RuntimeError):
    """An internal tool failure must terminate the attempt, never request replanning."""


def dispatch_locate(depth, camera, pixel, observation_id, measured, profile):
    # Expected stale input is recoverable; failures inside the kernel are not.
    if profile.strict_tool_errors and hasattr(depth, 'frame'):
        frame = depth.frame
        if frame is None or observation_id != frame[0]:
            raise FieldError('stale_observation', 'observation_id', 'latest observation ID')
        if not np.array_equal(vector(measured, 15), frame[1]):
            raise FieldError('stale_observation', 'observation_id', 'fresh observation after movement')
    return locate_point(depth, camera, pixel, observation_id, measured)


def approach_from_point(executor, located, standoff, profile=C0):

    if not located.get('valid') or not located.get('accepted'):

        result=dict(located, accepted=False, reason='invalid_approach_depth')
        if profile.r15_completion:
            result.update(reason='blocked',arrived=False,motion_completion=dict(tool='approach_point',status='blocked',
                requested={'base_delta':None},actual={'base_delta':[0.,0.,0.]},residual={'base_delta':None},
                detail='invalid_approach_depth',requires_replan=False))
        return result

    point = vector(located['point_base_m'], 3)

    turn = math.atan2(point[1], point[0])

    remaining = math.hypot(point[0], point[1]) - standoff
    # One observed segment only. The last <=.30 m requires another model call
    # with a fresh observation; it is never executed inside the same call.
    if profile.r15_standoff and remaining > .30 + 1e-9:remaining -= .30
    forward = float(np.clip(remaining, -.5, 1.5))

    origin = executor_odometry(executor)

    local = point_in_odometry(point, origin)

    rows = []

    result = {}

    for angle, distance in ([(turn, forward)] if profile.r15_standoff else ([(turn, forward / 2), (0., forward / 2)] if forward > .75 else [(turn, forward)])):

        previous = executor_odometry(executor); before = executor.steps

        operation=lambda:executor.move_base(angle, distance, max_steps=approach_segment_step_cap(profile.r15_protocol))
        result = (executor.r15_execute('move_base', dict(turn_rad=angle,forward_m=distance),operation)
                  if profile.r15_completion else operation())

        current = executor_odometry(executor)

        yaw = current['dyaw_rad'] - previous['dyaw_rad']

        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        dx = current['dx_m'] - previous['dx_m']; dy = current['dy_m'] - previous['dy_m']

        heading = previous['dyaw_rad'] + angle

        actual = math.cos(heading)*dx + math.sin(heading)*dy

        rows.append(dict(requested_turn_rad=angle, requested_forward_m=distance, actual_turn_rad=yaw,

            actual_forward_m=actual, actual_lateral_m=-math.sin(heading)*dx + math.cos(heading)*dy,

            turn_residual_rad=math.atan2(math.sin(angle-yaw), math.cos(angle-yaw)),

            forward_residual_m=distance-actual, steps_used=executor.steps-before,

            reason=result['reason'], accepted=result['accepted']))

        if not result.get('accepted') or result.get('reason') not in ('completed','arrived'):break

    current = executor_odometry(executor); yaw = current['dyaw_rad']

    dx = local[0]-current['dx_m']; dy = local[1]-current['dy_m']

    prediction = [math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy, float(local[2])]

    if profile.r15_completion:result['motion_completion']['tool']='approach_point'
    return dict(result, approach_feedback=dict(surface_point_base_m=list(map(float, point)),

        requested_turn_rad=turn, requested_forward_m=forward, segments=rows, odometry_prediction=prediction))



# Goal verification: independent CPU ports; disabled in C0/C1 and all live scopes.
def execution_diagnostics(result):
    """Reuse official-state diagnostics whitelist, excluding all official state."""
    out = {k: result[k] for k in ('execution_phase', 'consecutive_ik_failures') if k in result}
    if 'execution_phase' in out and out['execution_phase'] not in ('planning', 'tracking', 'endpoint'):
        raise ValueError('invalid execution phase')
    if 'consecutive_ik_failures' in out and (type(out['consecutive_ik_failures']) is not int or out['consecutive_ik_failures'] < 0):
        raise ValueError('invalid IK failure count')
    return out


def verification_body(value, *, revision=1):
    """One unchanged sensor image plus a textual pixel marker; no history or boxes."""
    if set(value) != {'target_description', 'image', 'pixel'}:
        raise ValueError('verification input whitelist')
    if not isinstance(value['target_description'], str) or not 1 <= len(value['target_description']) <= 300:
        raise ValueError('target description')
    reject_privilege(value['target_description'])
    image = value['image']
    if set(image) != {'camera', 'sha256', 'data_base64'} or image['camera'] not in ('head', 'hand'):
        raise ValueError('original robot RGB required')
    raw = base64.b64decode(image['data_base64'], validate=True)
    if hashlib.sha256(raw).hexdigest() != image['sha256']:
        raise ValueError('image hash mismatch')
    with Image.open(BytesIO(raw)) as im:
        if im.format != 'PNG' or im.mode != 'RGB' or im.size != (640, 640):
            raise ValueError('original 640 RGB PNG required')
        im.verify()
    pixel = value['pixel']
    if not isinstance(pixel, list) or len(pixel) != 2 or any(type(x) is not int or not 0 <= x < 640 for x in pixel):
        raise ValueError('selected pixel bounds')
    spec = FROZEN['experimental']['identity_revision_2' if revision == 2 else 'identity_verification']
    marker = dict(target_description=value['target_description'], selected_pixel=pixel,
                  pixel_convention='[x,y], origin top-left, x right, y down; identifies the object under this point')
    return dict(model=MODEL, instructions=spec['prompt'], reasoning={'effort':'medium','summary':'auto'},
        service_tier='default', store=False, max_output_tokens=2000 if revision == 2 else 1000,
        input=[dict(role='user',content=[dict(type='input_text',text=canonical(marker).decode()),
            dict(type='input_image',detail='original',image_url='data:image/png;base64,'+image['data_base64'])])],
        text={'format':{'type':'json_schema','name':'independent_target_comparison','strict':True,'schema':spec['schema']}})


# R2a: category descriptions only. No scene locations or metric dimensions.
R2A_CARDS = {
 'Master Chef can': 'Medium cylindrical coffee can; blue body with a light label/illustration and metal round lid; look for MASTER CHEF / COFFEE. Do not confuse with a red-and-white soup can.',
 'cracker box': 'Large tall rectangular carton; mainly red with yellow/light panels and cracker pictures; CHEEZ-IT lettering.',
 'sugar box': 'Medium rectangular carton; yellow and white packaging with blue accents; DOMINO and SUGAR lettering. Do not choose a red cracker carton merely because it is a box.',
 'tomato soup can': 'Medium cylindrical can; red upper label and light cream/white lower label; Campbell\'s and TOMATO SOUP lettering.',
 'tuna fish can': 'Small short wide round metal can; silver/gray circular lid and a printed side label; look for TUNA wording if readable; shape alone is not enough to confirm identity.',
 'pudding box': 'Small rectangular carton; brown dessert picture with red/white JELL-O branding; look for PUDDING if readable.',
 'gelatin box': 'Small rectangular carton; red/light panels; JELL-O and GELATIN wording with a red gelatin dessert picture. Distinguish from pudding by text and dessert shape.',
 'potted meat can': 'Small squat rounded rectangular metal can; blue label with large yellow SPAM wording and sliced meat picture.',
 'bowl': 'Medium round open bowl; red inner and outer surface with an open curved rim; no required packaging text.'
}


def r2a_appearance_cards(profile):
    if not profile.search_recessed_view:return R2A_CARDS
    shapes={
        'Master Chef can':'A cylindrical body with height comparable to its round-face width; distinguish its body proportions from a shallow tuna tin.',
        'cracker box':'A tall flat carton with a long upright side; distinguish a broad front face from the narrow side face.',
        'sugar box':'An upright rectangular carton; compare the full front-face height/width with nearby cartons and readable package evidence.',
        'tomato soup can':'An upright cylinder, taller than a shallow disk-shaped tuna can; round ends do not make the two categories equivalent.',
        'tuna fish can':'A shallow wide cylinder: side-wall height is much smaller than circular-lid width.',
        'pudding box':'A small rectangular carton; distinguish height/width and thickness of the same visible face from gelatin, using dessert/text evidence too.',
        'gelatin box':'A small rectangular carton; compare the same visible face and side thickness against pudding rather than relying on common JELL-O branding.',
        'potted meat can':'A squat rounded rectangular can, with a rounded rectangular top rather than a circular lid.',
        'bowl':'An open curved vessel, with a visible rim and concave interior when viewed from above; a rim-only view does not establish the whole shape.'}
    caution=(' Shape and height/width are category clues, not metric scene facts. Compare the same object face; '
        'perspective, pose and occlusion can change the image ratio. With incomplete extent, report ratio uncertainty '
        'and obtain another robot view. Do not infer identity from one ratio or color alone.')
    return {name:card+' '+shapes[name]+caution for name,card in R2A_CARDS.items()}

def validate_r2a_reply(reply,*,identity_confirmation=False,identity_multiview_evidence=False,identity_confirmed_persistence=False):
    if identity_confirmed_persistence:
        if not isinstance(reply,dict) or set(reply)!={'selected_index','reason','identity_status','counterevidence'}:
            raise ValueError('I1 continuity reply fields')
        validate_r2a_reply({k:reply[k] for k in ('selected_index','reason')})
        status=reply['identity_status']; evidence=reply['counterevidence']
        if status not in ('confirmed','uncertain','rejected') or not isinstance(evidence,str) or len(evidence)>900:
            raise ValueError('I1 continuity status/evidence')
        if (status=='confirmed') != (reply['selected_index'] is not None):raise ValueError('I1 selected candidate is supported')
        if status=='rejected' and not evidence.strip():raise ValueError('rejection requires visible counterevidence')
        if status!='rejected' and evidence:raise ValueError('counterevidence requires explicit rejection')
        reject_privilege(reply)
        return reply
    if identity_confirmation:
        fields={'selected_index','reason','identity_status','supporting_evidence','conflicting_evidence','missing_evidence'}
        if identity_multiview_evidence:fields|={'confirmation_basis','evidence_basis'}
        if not isinstance(reply,dict) or set(reply)!=fields:raise ValueError('identity confirmation reply fields')
        if identity_multiview_evidence:
            if reply['confirmation_basis'] not in ('category_text','distinctive_visual_features','unresolved'):
                raise ValueError('identity confirmation basis')
            if reply['evidence_basis'] not in ('current_only','associated_views','unresolved_association'):
                raise ValueError('identity evidence basis')
            if reply['identity_status']=='confirmed' and (reply['confirmation_basis']=='unresolved' or reply['evidence_basis']=='unresolved_association'):
                raise ValueError('Unresolved identity or view association cannot confirm')
        if reply['identity_status'] not in ('confirmed','uncertain','rejected'):raise ValueError('identity confirmation status')
        for key in ('supporting_evidence','conflicting_evidence','missing_evidence'):
            if (not isinstance(reply[key],list) or len(reply[key])>6 or
                    any(not isinstance(x,str) or not 1<=len(x)<=500 for x in reply[key])):
                raise ValueError('identity evidence list')
        if reply['identity_status']=='confirmed' and (reply['selected_index'] is None or not reply['supporting_evidence'] or reply['conflicting_evidence'] or reply['missing_evidence']):
            raise ValueError('confirmation requires discriminating evidence without unresolved evidence')
        if reply['identity_status']=='uncertain' and not (reply['missing_evidence'] or reply['conflicting_evidence']):
            raise ValueError('uncertain identity requires missing or conflicting evidence')
        if reply['identity_status']=='rejected' and not reply['conflicting_evidence']:raise ValueError('rejected identity requires conflicting evidence')
        validate_r2a_reply({k:reply[k] for k in ('selected_index','reason')})
        reject_privilege(reply)
        return reply
    if not isinstance(reply,dict) or set(reply)!={'selected_index','reason'}:raise ValueError('comparison reply fields')
    i=reply['selected_index']
    if i is not None and (type(i) is not int or not 0<=i<3):raise ValueError('comparison index')
    if not isinstance(reply['reason'],str) or not 1<=len(reply['reason'])<=900:raise ValueError('comparison reason')
    reject_privilege(reply)
    return reply

class IdentityEvidenceReferenceError(ValueError):
    """Correctable model reference; no provider request or action was sent."""


def validate_identity_evidence_reply(reply,wire,profile):
    """Check provenance claims against supplied images, not visual identity."""
    if not profile.identity_multiview_evidence:return
    if reply['evidence_basis']!='associated_views':return
    index=reply['selected_index'];linked=[]
    for message in json.loads(wire)['input']:
        for part in message['content']:
            if part['type']!='input_text':continue
            try:meta=json.loads(part['text'])
            except (ValueError,TypeError):continue
            if not isinstance(meta,dict) or 'historical_robot_view' not in meta:continue
            if any(r['candidate_index']==index for r in meta['proposed_correspondences']):
                linked.append(meta['historical_robot_view'])
    if not linked or not any(oid in ' '.join(reply['supporting_evidence']) for oid in linked):
        raise ValueError('Associated identity evidence must cite a supplied view for the selected candidate')


def r2a_comparison_wire(bundle,call,profile,*,archive=()):
    validate_call(call,profile)
    obs=bundle['observation']
    if call['observation_id']!=obs['observation_id']:raise ValueError('current comparison observation required')
    name=obs['target_semantics'][len('target object: '):]
    if name not in R2A_CARDS:raise ValueError('category card missing')
    markers=call['identity_candidates'];content=[]
    for camera in dict.fromkeys(c['camera'] for c in markers):
        field=camera+'_rgb';im=obs['sensors'][field]
        _audit_sensor(field,im,bundle['sensor_manifest'][field],RESOLUTION)
        content.extend([dict(type='input_text',text='Robot camera: '+camera),dict(type='input_image',detail='original',image_url='data:image/png;base64,'+im['data_base64'])])
    content.insert(0,dict(type='input_text',text=canonical(dict(target=obs['target_semantics'],appearance_card=r2a_appearance_cards(profile)[name],phase=call['verify_phase'],candidates=markers,pixel_convention='[x,y], top-left origin; candidate index starts at zero')).decode()))
    if profile.identity_multiview_evidence:
        # Reuse same-session sensor bundles, never planner verdicts, evaluation
        # images, pixel crops, generated object positions or automatic tracking.
        grouped={}
        for ref in call['identity_evidence']:
            matches=[row['bundle'] for row in archive
                if row['bundle']['observation']['observation_id']==ref['observation_id']]
            if len(matches)!=1:raise IdentityEvidenceReferenceError('Historical identity observation missing or ambiguous')
            past=matches[0];old=past['observation']
            if (past['position']!=bundle['position'] or old['target_semantics']!=obs['target_semantics']
                    or past['turn']>=bundle['turn']):
                raise IdentityEvidenceReferenceError('Historical identity view must belong to this target and earlier session turn')
            grouped.setdefault((ref['observation_id'],ref['camera']),[past,[]])[1].append(ref)
        for (oid,camera),(past,refs) in grouped.items():
            field=camera+'_rgb';im=past['observation']['sensors'][field]
            _audit_sensor(field,im,past['sensor_manifest'][field],RESOLUTION)
            meta=dict(historical_robot_view=oid,camera=camera,turn=past['turn'],
                proposed_correspondences=refs,association_is_model_proposal=True)
            content.extend([dict(type='input_text',text=canonical(meta).decode()),
                dict(type='input_image',detail='original',image_url='data:image/png;base64,'+im['data_base64'])])
    spec=dict(type='object',additionalProperties=False,properties=dict(selected_index=dict(type=['integer','null'],minimum=0,maximum=len(markers)-1),reason=dict(type='string')),required=['selected_index','reason'])
    instructions='Independently compare the marked objects in the original robot camera images with the target appearance card. Choose the best supported candidate index, or null if none is supported. Visible packaging text, color and shape matter; do not infer identity from the planner, location or grasp success. Explain visible evidence and uncertainty.'
    if profile.identity_confirmed_persistence:
        spec['properties'].update(identity_status=dict(type='string',enum=['confirmed','uncertain','rejected']),counterevidence=dict(type='string'))
        spec['required'] += ['identity_status','counterevidence']
        instructions += (' Keep the I1 supported-candidate standard above: a selected candidate is confirmed; '
            'null is uncertain unless visible contrary evidence explicitly rejects the marked linked candidate. '
            'Do not require extra ideal views or readable text when the visual evidence already supports the choice. '
            'For rejected, set selected_index=null and give concrete visible counterevidence; otherwise counterevidence is empty. '
            'Uncertainty or occlusion alone is not counterevidence.')
        if call.get('same_identity_candidate'):
            content.append(dict(type='input_text',text=canonical(dict(current_candidate_for_counterevidence=call['same_identity_candidate']['current_candidate_index'])).decode()))
    if profile.identity_confirmation:
        spec['properties'].update(identity_status=dict(type='string',enum=['confirmed','uncertain','rejected']),
            **{k:dict(type='array',items=dict(type='string')) for k in ('supporting_evidence','conflicting_evidence','missing_evidence')})
        spec['required'] += ['identity_status','supporting_evidence','conflicting_evidence','missing_evidence']
        instructions += (' selected_index is only a relative candidate ranking, never an identity confirmation. '
            'Separately report identity_status for that candidate. confirmed requires visible discriminating evidence '
            'consistent with the target category and no unresolved identity conflict or missing distinguishing evidence. '
            'A best guess, unreadable distinguishing features, or occluded body proportions stays uncertain even when one candidate ranks first. '
            'Use rejected for visible contradictory identity evidence. List supporting_evidence, conflicting_evidence and missing_evidence. '
            'A readable target word is not mandatory if other visible evidence distinguishes the category. '
            'Do not treat successful grasping, rest arrival or the planner choice as identity evidence. '
            'If uncertain, describe the missing side view or other robot observation needed; do not upgrade a guess to confirmed.')
    if profile.identity_multiview_evidence:
        spec['properties'].update(
            confirmation_basis=dict(type='string',enum=['category_text','distinctive_visual_features','unresolved']),
            evidence_basis=dict(type='string',enum=['current_only','associated_views','unresolved_association']))
        spec['required']+=['confirmation_basis','evidence_basis']
        instructions=('Independently identify the CURRENT marked candidate using original robot images. '
            'Rank candidates separately from identity_status; first place, repeated selection, planner belief, location, grasping and rest never confirm identity. '
            'Confirm only when visible evidence discriminates the target category from plausible alternatives with no unresolved identity conflict. '
            'There are two sufficient evidence routes: (1) readable category-specific text; or (2) a distinctive category-specific illustration or physical structure, '
            'supported by compatible visible packaging/body shape. State the distinguishing features and why they exclude the closest alternative. '
            'Shared brand, general color or one aspect ratio alone is insufficient. The appearance card describes useful clues, not an all-features checklist. '
            'Readable text, a separate side view and measured thickness are NOT all mandatory when another evidence route already discriminates the category. '
            'missing_evidence lists only information still NEEDED to decide identity, not optional ideal views or unreadable features already made unnecessary by other decisive evidence. '
            'Uncertain evidence must remain uncertain; do not treat a vaguely matching blurred patch as a distinctive illustration. '
            'Historical view correspondences are untrusted proposals. Independently check the marked object in each old view against the CURRENT candidate. '
            'Do not combine features from different objects or associate by candidate number, brand, color or table location alone. '
            'If an old image cannot reliably be linked, exclude it and decide from the current image alone if sufficient; otherwise remain uncertain with unresolved_association. '
            'Historical identity alone never proves which object is currently held; before_done must identify CURRENT held candidate 0. '
            'With associated_views cite the observation IDs and visible correspondence and category evidence in supporting_evidence. '
            'With current_only use only the current RGB evidence. Use rejected for visible contradictory identity evidence. '
            'When uncertain, specify the single most useful missing feature/view and how it would distinguish alternatives; avoid repeating a generic front-and-side checklist. '
            'confirmation_basis is category_text, distinctive_visual_features, or unresolved; evidence_basis is current_only, associated_views, or unresolved_association. '
            'A confirmed result requires a resolved basis, supporting evidence, no conflicts and no remaining decision-critical missing evidence.')
    return canonical(dict(model=profile.model,reasoning=dict(effort='medium',summary='auto'),service_tier='default',store=False,max_output_tokens=2000,
        instructions=instructions,
        input=[dict(role='user',content=content)],text=dict(format=dict(type='json_schema',name='independent_target_comparison',strict=True,schema=spec))))

def validate_identity_reference(reference):
    if (not isinstance(reference,dict) or set(reference)!={'attempt_id','observation_id','candidate_index'}
            or type(reference['attempt_id']) is not int or reference['attempt_id']<1
            or not valid_id(reference['observation_id'])
            or type(reference['candidate_index']) is not int or not 0<=reference['candidate_index']<=2):
        raise ValueError('identity_reference requires an attempt, review observation and candidate index')
    return reference


def validate_identity_recovery_feedback(value):
    if (not isinstance(value,dict) or set(value)!={'attempt_id','current_reference','current_candidate',
            'holding_possible','close_permitted','done_permitted','grasp_prepared','release_pending','event','review_requests'}
            or type(value['attempt_id']) is not int or value['attempt_id']<0
            or type(value['review_requests']) is not int or value['review_requests']<0
            or any(type(value[k]) is not bool for k in ('holding_possible','close_permitted','done_permitted','grasp_prepared','release_pending'))
            or not isinstance(value['event'],str) or len(value['event'])>96):
        raise ValueError('Invalid identity recovery feedback')
    reference=value['current_reference'];candidate=value['current_candidate']
    if reference is not None:
        validate_identity_reference(reference)
        if reference['attempt_id']!=value['attempt_id']:raise ValueError('Identity attempt mismatch')
        if (not isinstance(candidate,dict) or set(candidate)!={'camera','pixel'}
                or candidate['camera'] not in ('head','hand') or not isinstance(candidate['pixel'],list)
                or len(candidate['pixel'])!=2 or any(type(x) is not int or not 0<=x<=639 for x in candidate['pixel'])):
            raise ValueError('Invalid identity candidate marker')
    elif candidate is not None or value['close_permitted'] or value['done_permitted']:
        raise ValueError('Identity permission without candidate')
    return value


def identity_recovery_state(state):
    # Old serialized States do not acquire a new default_factory field on load.
    # Never silently turn legacy live permits into recovery permissions.
    if not getattr(state,'identity_attempt',None):
        if state.r2a_reviews:raise ValueError('Legacy identity state requires a fresh recovery session')
        state.identity_attempt=dict(generation=0,reference=None,candidate=None,holding_possible=False,
            close_permitted=False,grasp_reference=None,done_review=None,release_pending=False,last_review_step={},reviews=[],event='idle')
    return state.identity_attempt


def identity_recovery_feedback(state,executor):
    attempt=identity_recovery_state(state);done=attempt['done_review']
    value=dict(attempt_id=attempt['generation'],current_reference=deepcopy(attempt['reference']),
        current_candidate=deepcopy(attempt['candidate']),holding_possible=attempt['holding_possible'],
        close_permitted=bool(attempt['close_permitted'] and attempt['grasp_reference']==attempt['reference']),
        grasp_prepared=bool(attempt['reference'] and attempt['grasp_reference']==attempt['reference']),
        release_pending=attempt['release_pending'],
        done_permitted=bool(done and done['step']==executor.steps and done['reply']['selected_index']==0
            and done['reply'].get('identity_status','confirmed')=='confirmed'
            and attempt['holding_possible'] and not attempt['release_pending']),
        event=attempt['event'],review_requests=state.verification_requests)
    return validate_identity_recovery_feedback(value)


def dispatch_identity_recovery(state,executor,call,depth,bundle,deliver,profile,*,sink=None):
    """Explicit same-session recovery. No automatic motion, semantic truth or API retry."""
    validate_call(call,profile)
    attempt=identity_recovery_state(state);tool=call['tool'];start=executor.steps
    oid=bundle['observation']['observation_id'];reference=call['identity_reference']
    if any(row['status']=='pending' for row in attempt['reviews']):
        raise VerificationOutcomeUnknown('Identity review outcome unknown; no retry')
    def answer(actual,result):
        result=dict(result,identity_recovery=identity_recovery_feedback(state,executor))
        record(sink,tool,call,(actual,result),start,executor.steps,oid)
        return actual,result
    def reject(reason):
        return answer(None,dict(accepted=False,reason=reason,steps_used=0,steps_remaining=executor.max_steps-start))
    def matches():
        return reference is not None and reference==attempt['reference']
    def revoke(event):
        attempt.update(reference=None,candidate=None,close_permitted=False,grasp_reference=None,done_review=None,event=event)
    if tool=='verify_target':
        phase=call['verify_phase']
        if call['observation_id']!=oid:return reject('stale_observation')
        if attempt['release_pending']:return reject('identity_release_incomplete')
        if phase=='before_close':
            if attempt['holding_possible']:return reject('identity_release_required')
        elif not matches() or not attempt['holding_possible']:
            return reject('identity_attempt_reference_required')
        if start<=attempt['last_review_step'].get(phase,-1):return reject('identity_fresh_observation_required')
        if len(state.archive)+state.verification_requests+1>=profile.call_cap:return reject('identity_review_budget_exhausted')
        if deliver is None:raise RuntimeError('Independent verification delivery required')
        wire=r2a_comparison_wire(bundle,call,profile,archive=state.archive)
        if phase=='before_close':
            revoke('review_pending');attempt['generation']+=1
        else:
            attempt.update(done_review=None,event='review_pending')
        attempt['last_review_step'][phase]=start
        row=dict(phase=phase,attempt_id=attempt['generation'],status='pending',step=start,
            observation_id=oid,candidates=deepcopy(call['identity_candidates']))
        attempt['reviews'].append(row)
        state.verification_requests+=1
        # A raised/unknown delivery remains pending and propagates; never try again.
        reply=validate_r2a_reply(deliver(wire),identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence)
        if reply['selected_index'] is not None and reply['selected_index']>=len(call['identity_candidates']):
            raise ValueError('comparison index outside offered candidates')
        row.update(status='completed',reply=deepcopy(reply))
        state.r2a_reviews[phase]=deepcopy(row)
        if phase=='before_close' and reply['selected_index'] is not None:
            attempt.update(reference=dict(attempt_id=attempt['generation'],observation_id=oid,
                candidate_index=reply['selected_index']),candidate=deepcopy(call['identity_candidates'][reply['selected_index']]),
                close_permitted=True,event='candidate_reviewed')
        elif phase=='before_done':
            attempt.update(done_review=deepcopy(row),event='held_identity_supported' if reply['selected_index']==0 else 'held_identity_unconfirmed')
        else:attempt['event']='candidate_unconfirmed'
        return answer(call,dict(accepted=True,reason='review_sent',steps_used=0,
            steps_remaining=executor.max_steps-start,identity_comparison=reply))
    closing=tool=='set_gripper' and call['gripper']==0
    opening=tool=='set_gripper' and call['gripper']==1
    if closing:
        if attempt['release_pending']:return reject('identity_release_incomplete')
        if not matches() or not attempt['close_permitted']:return reject('identity_attempt_reference_required')
        if attempt['grasp_reference']!=reference:return reject('identity_grasp_replan_required')
    if tool=='done':
        if not matches() or not identity_recovery_feedback(state,executor)['done_permitted']:
            return reject('identity_match_required')
    if tool in ('prepare_grasp','replan_grasp'):
        if reference is not None and not matches():return reject('identity_attempt_reference_required')
        if reference is None and attempt['reference'] is not None:
            if attempt['holding_possible'] or attempt['release_pending']:return reject('identity_release_required')
            revoke('candidate_changed_requires_review')
    if opening:
        # Revoke before dispatch so an interrupted opening cannot retain a valid token.
        revoke('release_pending');attempt['release_pending']=True
    revisions_before=len(getattr(executor,'grasp_revisions',[]))
    try:
        actual,result=dispatch(executor,call,depth,state.observation_ids,profile,sink=None,observation_id=oid)
    except BaseException:
        if closing:
            attempt.update(close_permitted=False,holding_possible=True,done_review=None,event='close_outcome_unknown')
        raise
    if opening:
        complete=(actual is not None and result.get('accepted') is True and result.get('reason')=='completed'
            and result.get('arrived') is True
            and getattr(executor,'pending',None) is None and not getattr(executor,'failed',False))
        if complete:
            attempt.update(holding_possible=False,release_pending=False,event='open_acknowledged_check_placement')
        else:attempt['event']='release_incomplete'
    elif closing:
        if result.get('closure_command_completed') is True or (executor.steps>start and getattr(executor,'closed',False)):
            attempt.update(holding_possible=True,done_review=None,event='close_acknowledged_identity_unconfirmed')
    elif tool in ('prepare_grasp','replan_grasp') and matches():
        # Acknowledge a newly installed robot-local goal, including a blocked
        # approach whose existing G1/arrival rules still decide closure. A
        # rejected command cannot associate a leftover goal with a new object.
        if actual is not None and len(getattr(executor,'grasp_revisions',[]))>revisions_before:
            attempt.update(grasp_reference=deepcopy(reference),event='grasp_goal_bound')
    if executor.steps>start and attempt['done_review'] is not None:
        attempt.update(done_review=None,event='movement_requires_done_review')
    return answer(actual,result)


def update_i1_confirmation(state,call,reply,oid):
    """I1 identity memory; references are explicit robot-view associations, not tracking truth."""
    prior=state.r2a_confirmed_candidate
    ref=call.get('same_identity_candidate')
    linked=bool(prior and ref and prior.get('status')=='confirmed' and not prior.get('association_lost')
        and ref['observation_id']==prior['observation_id']
        and ref['candidate_index']==prior['candidate_index'])
    status=reply['identity_status']
    if status=='confirmed':
        state.r2a_confirmed_candidate=dict(status='confirmed',observation_id=oid,
            candidate_index=reply['selected_index'])
        effective=reply['selected_index']
    elif status=='rejected' and linked:
        state.r2a_confirmed_candidate=dict(prior,status='rejected',counterevidence=reply['counterevidence'])
        effective=None
    elif status=='uncertain' and linked:
        effective=ref['current_candidate_index']
    else:
        effective=None
    return dict(current_status=status,effective_status='confirmed' if effective is not None else status,
        effective_candidate_index=effective,prior_confirmation_preserved=status=='uncertain' and linked,
        same_candidate_linked=linked,confirmation_reference=deepcopy(state.r2a_confirmed_candidate),
        association_is_model_report=True)


def dispatch_r2a_identity(state,executor,call,depth,bundle,deliver,profile,*,sink=None):
    if profile.identity_attempt_recovery:
        return dispatch_identity_recovery(state,executor,call,depth,bundle,deliver,profile,sink=sink)
    validate_call(call,profile);oid=bundle['observation']['observation_id'];tool=call['tool'];start=executor.steps
    def answer(actual,result):
        record(sink,tool,call,(actual,result),start,executor.steps,oid)
        return actual,result
    def reject(reason,reply=None):
        result=dict(accepted=False,reason=reason,steps_used=0,steps_remaining=executor.max_steps-executor.steps)
        if reply is not None:result['identity_comparison']=reply
        return answer(None,result)
    if tool=='verify_target':
        phase=call['verify_phase']
        if call['observation_id']!=oid:return reject('stale_observation')
        if profile.identity_confirmed_persistence and call.get('same_identity_candidate'):
            ref=call['same_identity_candidate'];prior=state.r2a_confirmed_candidate
            if (prior.get('status')!='confirmed' or prior.get('association_lost') or ref['observation_id']!=prior.get('observation_id')
                    or ref['candidate_index']!=prior.get('candidate_index')):
                return reject('same_candidate_reference_invalid')
            if phase=='before_done' and (ref['current_candidate_index']!=0
                    or call['identity_candidates'][0]['camera']!='hand' or not executor.closed):
                return reject('current_held_candidate_evidence_required')
        previous=state.r2a_reviews.get(phase)
        if previous:
            if not profile.identity_confirmation:return reject('identity_review_already_used')
            if previous['status']=='pending':raise VerificationOutcomeUnknown('Identity review outcome unknown; no retry')
            if start<=previous['step']:return reject('identity_fresh_view_required')
        if len(state.archive)+state.verification_requests+1>=profile.call_cap:return reject('identity_review_budget_exhausted')
        if deliver is None:raise RuntimeError('R2a independent verification delivery required')
        try:wire=r2a_comparison_wire(bundle,call,profile,archive=state.archive)
        except IdentityEvidenceReferenceError:return reject('identity_evidence_reference_invalid')
        if profile.identity_confirmation and phase=='before_close' and 'before_done' in state.r2a_reviews:
            prior_done=state.r2a_reviews['before_done']
            if prior_done.get('status')=='pending':raise VerificationOutcomeUnknown('Identity review outcome unknown; no retry')
            prior_done['status']='revoked_for_candidate_review'
        state.r2a_reviews[phase]=dict(status='pending',step=start)
        state.verification_requests+=1
        reply=validate_r2a_reply(deliver(wire),identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence)
        validate_identity_evidence_reply(reply,wire,profile)
        if reply['selected_index'] is not None and reply['selected_index']>=len(call['identity_candidates']):raise ValueError('comparison index outside offered candidates')
        state.r2a_reviews[phase]=dict(status='completed',step=start,observation_id=oid,reply=deepcopy(reply))
        feedback={}
        if profile.identity_confirmed_persistence:
            continuity=update_i1_confirmation(state,call,reply,oid)
            state.r2a_reviews[phase]['identity_continuity']=continuity
            feedback['identity_continuity']=continuity
        return answer(call,dict(accepted=True,reason='review_sent',steps_used=0,steps_remaining=executor.max_steps-executor.steps,identity_comparison=reply,**feedback))
    phase='before_close' if tool=='set_gripper' and call['gripper']==0 else 'before_done' if tool=='done' else None
    if phase:
        review=state.r2a_reviews.get(phase,{})
        selected=(review.get('identity_continuity',{}).get('effective_candidate_index')
            if profile.identity_confirmed_persistence else review.get('reply',{}).get('selected_index'))
        if review.get('status')!='completed' or selected is None:return reject('identity_match_required',review.get('reply'))
        if phase=='before_done' and (review.get('step')!=start or selected!=0):return reject('identity_match_required',review.get('reply'))
        if phase=='before_done' and profile.identity_confirmation and review['reply'].get('identity_status')!='confirmed':
            return reject('identity_unconfirmed_obtain_missing_evidence_then_review' if profile.identity_multiview_evidence else 'identity_unconfirmed_obtain_side_view_then_review',review['reply'])
        if phase=='before_close' and profile.identity_confirmation and review['reply'].get('identity_status')!='confirmed':
            return reject('identity_unconfirmed_obtain_missing_evidence_then_review' if profile.identity_multiview_evidence else 'identity_unconfirmed_obtain_side_view_then_review',review['reply'])
        if phase=='before_close' and (review['observation_id'] not in call['note'] or 'candidate '+str(review['reply']['selected_index']) not in call['note']):return reject('selected_candidate_reference_required',review.get('reply'))
    if profile.identity_confirmation and tool=='set_gripper' and call['gripper']==1:
        for row in state.r2a_reviews.values():
            if row.get('status')=='pending':raise VerificationOutcomeUnknown('Identity review outcome unknown; no retry')
            row['status']='revoked_after_release'
    result=dispatch(executor,call,depth,state.observation_ids,profile,sink=sink,observation_id=oid)
    if profile.identity_confirmed_persistence and (result[1].get('reason')=='grip_lost'
            or (tool=='set_gripper' and call['gripper']==1 and result[1].get('steps_used',0)>0)):
        # Identity knowledge is retained, but a released/lost item has no current
        # held-object association. It must not lend its permit to another item.
        if state.r2a_confirmed_candidate:state.r2a_confirmed_candidate['association_lost']=True
        for row in state.r2a_reviews.values():row['status']='association_lost'
    return result


class C2BudgetExhausted(RuntimeError):
    """A spending ceiling is a batch stop, not a soft identity error."""


class CaseLimitReached(C2BudgetExhausted):
    """A local worker limit; no shared ledger stop or automatic retry."""
    def __init__(self, category):
        super().__init__(category)
        self.category=category
        self.scope='case'


class RequestNotSent(RuntimeError):
    """Transport proves no HTTP request bytes were sent."""


class VerificationOutcomeUnknown(RuntimeError):
    """Transport cannot prove the request outcome: always stop, never retry."""


def identity_review(value, deliver):
    """Exactly one independent delivery; errors/unknown outcomes propagate, never retry."""
    wire = build_wire(value, [], verification=True)
    reply = deliver(wire)
    if not isinstance(reply, dict) or set(reply) != {'verdict', 'reason'}:
        raise ValueError('verification reply schema')
    validate_identity_feedback(dict(target_id=1, **reply))
    return dict(reply=deepcopy(reply), wire_sha256=hashlib.sha256(wire).hexdigest())


def validate_identity_feedback(feedback):
    if not isinstance(feedback, dict) or set(feedback) != {'target_id', 'verdict', 'reason'}:
        raise ValueError('identity feedback whitelist')
    if type(feedback['target_id']) is not int or feedback['target_id'] < 0:
        raise ValueError('identity target binding')
    if feedback['verdict'] not in ('match', 'mismatch', 'uncertain'):
        raise ValueError('identity verdict')
    if not isinstance(feedback['reason'], str) or not 1 <= len(feedback['reason'].strip()) <= 240 or '\n' in feedback['reason']:
        raise ValueError('identity reason must be one short line')
    reject_privilege(feedback)


def point_in_odometry(point, odometry):
    """Base-frame depth point to reset-relative robot odometry; never world truth."""
    xyz = vector(point, 3)
    if set(odometry) != {'dx_m', 'dy_m', 'dyaw_rad'}:
        raise ValueError('relative odometry whitelist')
    dx, dy, yaw = vector([odometry[k] for k in ('dx_m', 'dy_m', 'dyaw_rad')], 3)
    c, t = math.cos(yaw), math.sin(yaw)
    return [c*xyz[0]-t*xyz[1]+dx, t*xyz[0]+c*xyz[1]+dy, float(xyz[2])]


def identity_feedback(state):
    target = state.identity_target
    if target is None:
        return dict(target_id=0, verdict='uncertain', reason='No valid target point has been selected; identity is unverified.')
    if target['status'] != 'completed':
        raise RuntimeError('identity delivery unresolved; do not retry or continue this attempt')
    return dict(target_id=target['id'], **deepcopy(target['reply']))


def dispatch_with_identity(state, executor, call, depth, bundle, deliver, profile, *, sink=None):
    """Selection follows successful current locate_point; all state lives in State.

    Fixed selection anchor prevents repeated small displacements from drifting a match.
    Every valid locate_point is a selection proxy, including furniture measurements.
    measure is not a selection event. No note parsing or model-declared change signal.
    """
    if profile.identity_revision == 2:
        return dispatch_shared_target(state, executor, call, depth, bundle, deliver, profile, sink=sink)
    oid = bundle['observation']['observation_id']
    if profile.r2a_identity:
        return dispatch_r2a_identity(state,executor,call,depth,bundle,deliver,profile,sink=sink)
    if not profile.verification:
        return dispatch(executor, call, depth, state.observation_ids, profile, sink=sink, observation_id=oid)
    validate_call(call, profile)
    if profile.use_located_semantics:
        raise ValueError('verification requires original observation target description')
    if state.identity_target is not None and state.identity_target['status'] != 'completed':
        raise RuntimeError('identity delivery unresolved; do not retry or continue this attempt')
    closing = call['tool'] == 'set_gripper' and call['gripper'] <= .1
    if closing:
        feedback = identity_feedback(state)
        if feedback['verdict'] == 'mismatch':
            result = dict(accepted=False, reason='identity_mismatch', steps_used=0,
                          steps_remaining=executor.max_steps-executor.steps, identity_verification=feedback)
            record(sink, 'identity_close_gate', call, (None, result), executor.steps, executor.steps, oid)
            return None, result
    actual, result = dispatch(executor, call, depth, state.observation_ids, profile, sink=sink, observation_id=oid)
    if call['tool'] == 'locate_point' and result.get('accepted') and result.get('valid'):
        if call['observation_id'] != oid and not profile.measure:
            raise ValueError('selection requires the current observation')
        obs = bundle['observation']
        point = point_in_odometry(result['point_base_m'], obs['odometry'])
        previous = state.identity_target
        threshold = FROZEN['experimental']['identity_verification']['selection_distance_m']
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('selection distance threshold')
        distance = None if previous is None else float(np.linalg.norm(np.asarray(point)-previous['anchor_odometry_m']))
        if previous is None or distance > threshold:
            state.identity_sequence += 1
            state.identity_target = dict(id=state.identity_sequence, anchor_odometry_m=point,
                selected_observation_id=oid, pixel=deepcopy(call['pixel']), camera=call['camera'],
                distance_from_previous_m=distance, status='pending', reply=None)
            # Reserve the new target before delivery, so a lost reply cannot reuse an old match.
            field = call['camera'] + '_rgb'
            _audit_sensor(field, obs['sensors'][field], bundle['sensor_manifest'][field], RESOLUTION)
            image = obs['sensors'][field]
            value = dict(target_description=obs['target_semantics'], pixel=deepcopy(call['pixel']),
                         image=dict(camera=call['camera'], sha256=image['sha256'], data_base64=image['data_base64']))
            if profile.verification_errors_continue and state.verification_requests >= profile.verification_call_cap:
                review = dict(reply=dict(verdict='uncertain', reason='Identity review quota exhausted; this target is unverified.'),
                              wire_sha256=None, status='quota_exhausted', request_sent=False)
            else:
                try:
                    if deliver is None:
                        raise RuntimeError('independent verification delivery port required')
                    state.verification_requests += 1
                    review = identity_review(value, deliver)
                    review.update(status='completed', request_sent=True)
                except (VerificationOutcomeUnknown, C2BudgetExhausted):
                    raise
                except Exception as error:
                    if not profile.verification_errors_continue:
                        raise
                    review = dict(reply=dict(verdict='uncertain', reason='Identity review error; this target is unverified.'),
                                  wire_sha256=None, status='error', error_type=type(error).__name__, request_sent=None)
            state.identity_target.update(status='completed', reply=review['reply'], wire_sha256=review['wire_sha256'])
            record(sink, 'identity_review', dict(target_id=state.identity_sequence, observation_id=oid,
                image_sha256=image['sha256'], pixel=call['pixel'], distance_from_previous_m=distance),
                review, executor.steps, executor.steps, oid)
        result = dict(result, identity_verification=identity_feedback(state))
    elif closing:
        result = dict(result, identity_verification=feedback)
    if 'identity_verification' in result:
        record(sink, 'identity_feedback', dict(tool=call['tool']), result, executor.steps, executor.steps, oid)
    return actual, result


def robot_finger_pixels(points, depth):
    """Reject measured points inside own fingers; no scene geometry is read."""
    q = depth.frame[1]
    hand = depth.frame[2].get('hand')
    if q is None or hand is None:
        return np.zeros(points.shape[:-1], dtype=bool)
    config = FROZEN['experimental']['identity_revision_2']['self_finger_filter']
    # The calibrated hand camera is rigidly mounted on gripper_link. Both
    # prismatic fingers have identity rotation relative to that link.
    camera = np.asarray(hand['camera_in_base'])
    local = (points-camera[:3, 3]) @ camera[:3, :3]
    local += np.asarray(config['camera_mount_translation_m'])
    excluded = np.zeros(points.shape[:-1], dtype=bool)
    for side, sign in (('l', -1), ('r', 1)):
        joint = float(q[FETCH_ORDER.index(side+'_gripper_finger_joint')])
        finger = local - np.array([0., sign*(config['joint_origin_y_m']+joint), 0.])
        lo, hi = np.asarray(config['bounds_m'][side])
        margin = config['depth_margin_m']
        excluded |= ((finger >= lo-margin) & (finger <= hi+margin)).all(axis=-1)
    return excluded


def target_view(camera, pixel, bundle, depth):
    """Seeded depth extent above a sensor support plane, never a semantic mask."""
    from scipy.ndimage import label
    spec = FROZEN['experimental']['identity_revision_2']
    if depth.frame is None or depth.frame[0] != bundle['observation']['observation_id']:
        return None
    raw = depth.frame[2][camera]['depth_mm'].astype(float)
    x, y = pixel
    if not (0 <= x < 640 and 0 <= y < 640) or raw[y, x] <= 0:
        return None
    # Estimate a horizontal support height from calibrated sensor points. Unlike
    # per-pixel depth derivatives, a height histogram tolerates mm quantization.
    frame = depth.frame[2][camera]
    yy_grid, xx_grid = np.indices(raw.shape)
    rays = np.stack((xx_grid, yy_grid, np.ones_like(raw)), axis=-1) @ np.linalg.inv(frame['intrinsics']).T
    optical = rays * (raw / 1000)[..., None]
    points = np.stack((optical[..., 2], -optical[..., 0], -optical[..., 1]), axis=-1)
    transform = np.asarray(frame['camera_in_base'])
    points = points @ transform[:3, :3].T + transform[:3, 3]
    heights = points[..., 2]
    config = spec['support_plane']
    seed_height = heights[y, x]
    bins = np.arange(seed_height-config['search_below_m'],
                     seed_height-config['minimum_separation_m'], config['bin_m'])
    counts, edges = np.histogram(heights[raw > 0], bins=bins)
    peak = int(np.argmax(counts))
    support_height = None
    if counts[peak] >= config['minimum_pixels']:
        # Refine the modal bin without scene labels, object poses or text matching.
        modal = (raw > 0) & (heights >= edges[peak]) & (heights < edges[peak+1])
        support_height = float(np.median(heights[modal]))
        mask = (raw > 0) & (np.abs(raw-raw[y,x]) <= spec['depth_component_tolerance_mm'])
        mask &= heights > support_height + config['clearance_m']
    else:
        # No supported plane: do not call a seed-colored surface an object extent.
        return None
    mask &= ~robot_finger_pixels(points, depth)
    if not mask[y, x]:
        return None
    components, _ = label(mask)
    selected = components == components[y, x]
    yy, xx = np.where(selected)
    if len(xx) < 4 or len(xx) > spec['component_max_fraction']*raw.size:
        return None
    if xx.min() == 0 or xx.max() == 639 or yy.min() == 0 or yy.max() == 639:
        return None
    return dict(camera=camera, frame=bundle['observation']['observation_id'], pixel=list(pixel),
                width_px=int(xx.max()-xx.min()+1), height_px=int(yy.max()-yy.min()+1),
                method='sensor_support_separated_depth_component', support_height_base_m=support_height, touches_border=bool(xx.min()==0 or xx.max()==639 or yy.min()==0 or yy.max()==639))


def target_summary(target):
    """One target record owns tracking data; expose its summary, not image patches."""
    return None if target is None else deepcopy({k: v for k, v in target.items() if k != '_tracking'})


def track_target_pixel(target, camera, pixel, bundle, depth):
    """Conservative local RGB correspondence, checked against metric sensor depth.

    A failed correspondence is not permission to treat the old projection as an
    observation. Explicit selections remain authoritative sensor measurements.
    """
    from scipy.signal import correlate2d
    config = FROZEN['experimental']['identity_revision_2']['tracking']
    half = config['patch_half_width_px']; side = 2*half+1
    obs = bundle['observation']; oid = obs['observation_id']
    rgb = np.asarray(Image.open(BytesIO(base64.b64decode(obs['sensors'][camera+'_rgb']['data_base64']))), dtype=float)
    gray = rgb.mean(axis=2)
    references = target.setdefault('_tracking', {})
    ref = references.get(camera)
    x, y = pixel
    explicit = target.get('source_frame') == oid and target.get('selected_camera') == camera
    if explicit:
        x, y = target['selected_pixel']
    elif ref is not None and ref['frame'] == oid:
        return ref['pixel']
    elif ref is not None:
        template = np.asarray(ref['patch']); centered = template-template.mean()
        energy = float(np.sum(centered**2))
        radius = config['search_radius_px']
        left, top = max(0, x-radius-half), max(0, y-radius-half)
        right, bottom = min(640, x+radius+half+1), min(640, y+radius+half+1)
        search = gray[top:bottom, left:right]
        if energy < side**2*config['minimum_texture_variance'] or min(search.shape) < side:
            return None
        ones = np.ones((side, side)); count = side**2
        sums = correlate2d(search, ones, mode='valid')
        variance = np.maximum(correlate2d(search**2, ones, mode='valid')-sums**2/count, 0)
        scores = correlate2d(search, centered, mode='valid') / np.sqrt(np.maximum(energy*variance, 1e-12))
        iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
        best = scores[iy, ix]
        competitors = scores.copy()
        competitors[max(0,iy-half):iy+half+1, max(0,ix-half):ix+half+1] = -1
        if best < config['minimum_correlation'] or best-float(competitors.max()) < config['uniqueness_margin']:
            return None
        x, y = left+ix+half, top+iy+half
        frame = depth.frame[2][camera]; z = float(frame['depth_mm'][y,x])/1000
        if z <= 0:
            return None
        optical = np.linalg.solve(frame['intrinsics'], [x,y,1])*z
        base = np.asarray(frame['camera_in_base']) @ [optical[2], -optical[0], -optical[1], 1]
        position = point_in_odometry(base[:3], obs['odometry'])
        if np.linalg.norm(np.asarray(position)-target['position_odometry_m']) > config['max_sensor_motion_m']:
            return None
        target.update(position_odometry_m=position, tracking_frame=oid)
    elif not explicit and target_view(camera, pixel, bundle, depth) is None:
        # A new camera requires a bounded, non-support sensor component.
        return None
    if half <= x < 640-half and half <= y < 640-half:
        references[camera] = dict(frame=oid, pixel=[int(x),int(y)], patch=gray[y-half:y+half+1,x-half:x+half+1].tolist())
    return [int(x), int(y)]


def refresh_target_views(state, bundle, depth):
    target = state.identity_target
    if target is None:
        return
    target['current_views'] = {}
    if getattr(depth, 'frame', None) is None:
        return
    if depth.frame[0] != bundle['observation']['observation_id']:
        target['tracking_status'] = 'stale_sensor_frame'
        return
    obs = bundle['observation']; od = obs['odometry']; yaw = od['dyaw_rad']
    c, s = math.cos(yaw), math.sin(yaw)
    for camera, frame in depth.frame[2].items():
        p = np.asarray(target['position_odometry_m'])-np.array([od['dx_m'], od['dy_m'], 0.])
        point = np.array([c*p[0]+s*p[1], -s*p[0]+c*p[1], p[2], 1.])
        local = np.linalg.inv(frame['camera_in_base']) @ point
        if local[0] <= 0:
            continue
        uv = frame['intrinsics'] @ np.array([-local[1], -local[2], local[0]])
        pixel = np.rint(uv[:2]/uv[2]).astype(int).tolist()
        x, y = pixel
        if not (0 <= x < 640 and 0 <= y < 640):
            continue
        # A correspondence proposal must not mutate the sole target record before
        # the depth and extent checks have accepted it.
        candidate = deepcopy(target)
        pixel = track_target_pixel(candidate, camera, pixel, bundle, depth)
        if pixel is None:
            continue
        x, y = pixel
        residual = float(frame['depth_mm'][y, x])/1000-local[0]
        spec = FROZEN['experimental']['identity_revision_2']
        if residual < -spec['occlusion_tolerance_m'] or residual > spec['projection_depth_tolerance_m']:
            continue
        view = target_view(camera, pixel, bundle, depth)
        if view:
            for key in ('position_odometry_m', 'tracking_frame', '_tracking'):
                if key in candidate:
                    target[key] = candidate[key]
            target['current_views'][camera] = view
    target['tracking_status'] = 'sensor_correspondence' if target['current_views'] else 'uncertain'


def validate_evidence_reply(reply):
    spec = FROZEN['experimental']['identity_revision_2']['schema']
    if not isinstance(reply, dict) or set(reply) != set(spec['required']):
        raise ValueError('evidence reply fields')
    if reply['verdict'] not in ('like', 'confirmed', 'unclear', 'unlike'):
        raise ValueError('evidence verdict')
    for key in ('label_text', 'ownership_evidence', 'neighbor_features', 'size_evidence', 'reason'):
        if not isinstance(reply[key], str) or not reply[key].strip():
            raise ValueError('nonempty evidence required: '+key)
    reject_privilege(reply)


def review_shared_target(state, bundle, deliver, profile=C2_R2, *, selected=False, sink=None, explicit=False):
    if profile.active_confirmation and not explicit:
        return False  # Deliberate active observation/review, never opportunistic growth review.
    target = state.identity_target
    if target is None or (target['verdict'] == 'confirmed' and not (profile.target_association and explicit)) or state.verification_requests >= profile.verification_call_cap:
        return False
    views = [v for v in target['current_views'].values() if v.get('width_px') is not None
             or (profile.partial_target_marker and explicit and v.get('frame') == bundle['observation']['observation_id'])]
    view = max(views, key=lambda v: v.get('width_px') or 0) if views else None
    if profile.target_association and explicit:
        if view is None:
            return False
    elif selected:
        # Full original frame retains surrounding objects; no crop or isolation.
        view = target['current_views'].get(target['selected_camera'], dict(camera=target['selected_camera'],
            pixel=target['selected_pixel'], frame=target['source_frame'], width_px=None, height_px=None))
    else:
        if view is None or target['last_review'] is None:
            return False
        baseline = target['last_review']
        if baseline['width_px'] is None:
            baseline = target.get('growth_reference')
            if baseline is None:
                target['growth_reference'] = dict(deepcopy(view), reason='first_reliable_width_after_unmeasurable_review')
                record(sink, 'target_growth_reference', dict(target_id=target['id']),
                       deepcopy(target['growth_reference']), 0, 0, bundle['observation']['observation_id'])
                return False
        if view['width_px'] < FROZEN['experimental']['identity_revision_2']['growth_ratio']*baseline['width_px']:
            return False
    if target.get('last_attempt_frame') == view['frame']:
        return False
    target['last_attempt_frame'] = view['frame']
    target['last_review'] = deepcopy(view)
    target.pop('growth_reference', None)
    obs = bundle['observation']; camera = view['camera']; field = camera+'_rgb'
    state.verification_requests += 1  # every attempt, including errors; no retry
    review = dict(status='error', trigger='selection' if selected else 'growth', request_index=state.verification_requests, camera=camera, frame=view['frame'])
    if profile.target_association or profile.target_evidence_diagnostics:
        review.update(candidate_id=target['id'], marker=deepcopy(view), request_sent=False,
                      trigger='explicit' if explicit else review['trigger'], verdict=None,
                      source_snapshot=deepcopy(state_snapshot(obs)))
        # Append before delivery, so unknown delivery and budget exceptions retain provenance.
        state.target_reviews.append(review)
    try:
        _audit_sensor(field, obs['sensors'][field], bundle['sensor_manifest'][field], RESOLUTION)
        image = obs['sensors'][field]
        value = dict(target_description=obs['target_semantics'], pixel=view['pixel'], image=dict(camera=camera, sha256=image['sha256'], data_base64=image['data_base64']))
        body = verification_body(value, revision=2)
        if profile.active_confirmation:
            body['model'] = profile.model
        if profile.visual_identity_confirmation:
            body['instructions'] = FROZEN['weekly_visual_identity_prompt']
        if profile.multiview_confirmation:
            body['instructions'] = FROZEN['weekly_multiview_identity_prompt']
            marker = json.loads(body['input'][0]['content'][0]['text'])
            body['input'][0]['content'][0]['text'] = canonical(dict(marker, camera=camera)).decode()
            review['source_views'] = [deepcopy(view)]
            for other in views:
                if other['camera'] == camera or other['frame'] != obs['observation_id']:
                    continue
                other_field = other['camera']+'_rgb'
                _audit_sensor(other_field, obs['sensors'][other_field], bundle['sensor_manifest'][other_field], RESOLUTION)
                extra = verification_body(dict(target_description=obs['target_semantics'],
                    pixel=other['pixel'], image=dict(camera=other['camera'],
                    sha256=obs['sensors'][other_field]['sha256'],
                    data_base64=obs['sensors'][other_field]['data_base64'])), revision=2)
                marker = json.loads(extra['input'][0]['content'][0]['text'])
                extra['input'][0]['content'][0]['text'] = canonical(dict(marker, camera=other['camera'])).decode()
                body['input'][0]['content'].extend(extra['input'][0]['content'])
                review['source_views'].append(deepcopy(other))
        if profile.persistent_target_confirmation:
            body['instructions'] += (' A verdict unlike requires an explicit contradictory visible feature '
                'or evidence that the marked object is a different object/category. State that evidence '
                'in reason. Blur, occlusion, unreadable text, distance, or inability to decide is unclear '
                'or like, never by itself unlike. Assess only the supplied current sensor images.')
        if profile.identity_evidence_quality:
            body['instructions'] += IDENTITY_EVIDENCE_QUALITY_PROMPT
        wire = canonical(body)
        review['wire_sha256'] = hashlib.sha256(wire).hexdigest()
        if profile.target_association or profile.target_evidence_diagnostics:
            review['request_sent'] = True
        reply = deliver(wire)
        validate_evidence_reply(reply)
        if (profile.persistent_target_confirmation and target.get('verdict') == 'confirmed'
                and reply['verdict'] in ('like', 'unclear')):
            # Keep the original identity evidence; archive this raw inconclusive review separately.
            # Dispatch still requires current association, a completed fresh review and alignment.
            pass
        else:
            target.update(deepcopy(reply))
        review.update(status='completed', reply=deepcopy(reply))
        if profile.target_association or profile.target_evidence_diagnostics:
            review['verdict'] = reply['verdict']
            if reply['verdict'] == 'unlike':
                state.negative_targets.append(dict(candidate_id=target['id'], source='verifier',
                    frame=view['frame'], request_index=state.verification_requests, reason=reply['reason']))
    except Exception as error:
        # No retry, no task interruption, and no inference from a failed request.
        if profile.target_association or profile.target_evidence_diagnostics:
            if isinstance(error, RequestNotSent):
                review.update(status='not_sent', request_sent=False)
            elif isinstance(error, VerificationOutcomeUnknown):
                review.update(status='unknown_delivery', request_sent=None)
        if (profile.automatic_supervision or profile.active_confirmation) and isinstance(error, VerificationOutcomeUnknown):
            review.update(status='unknown_delivery', error_type=type(error).__name__)
            record(sink, 'target_review', dict(target_id=target['id']), review, 0, 0, obs['observation_id'])
            raise
        if profile.model_target_boxes and isinstance(error, C2BudgetExhausted):
            review.update(status='budget_exhausted',scope=getattr(error,'scope','unknown'))
            if profile.target_association or profile.target_evidence_diagnostics:
                review['request_sent'] = False
            record(sink, 'target_review', dict(target_id=target['id']), review, 0, 0, obs['observation_id'])
            raise
        review.update(error_type=type(error).__name__)
    target['last_review_status'] = review['status']
    target['requests_used'] = state.verification_requests
    record(sink, 'target_review', dict(target_id=target['id']), review, 0, 0, obs['observation_id'])
    return True


def apply_target_observation(state, bundle, value, *, selection_call=None, previous_target_id=None, allow_partial=False):
    """Use only current model-reported boxes. Never infer a mask from depth."""
    target = state.identity_target
    feedback = dict(status='unmeasurable', cameras={})
    if target is None:
        return dict(status='no_target', cameras={})
    target['current_views'] = {}
    target['tracking_status'] = 'astra_boxes_unmeasurable'
    oid = bundle['observation']['observation_id']
    if value is None:
        return feedback
    if not isinstance(value, dict) or set(value) != {'observation_id', 'target_id', 'head', 'hand'}:
        return dict(status='invalid_fields', cameras={})
    if value['observation_id'] != oid:
        return dict(status='stale_frame', cameras={})
    expected_id = previous_target_id if selection_call is not None else target['id']
    if value['target_id'] != expected_id or (value['target_id'] is not None and type(value['target_id']) is not int):
        return dict(status='target_id_mismatch', cameras={})
    for camera in ('head', 'hand'):
        view = value[camera]
        if view is None:
            feedback['cameras'][camera] = 'unmeasurable'
            continue
        valid = isinstance(view, dict) and set(view) == {'box', 'pixel'}
        box = view.get('box') if valid else None
        pixel = view.get('pixel') if valid else None
        valid = valid and isinstance(pixel, list) and len(pixel) == 2
        valid = valid and all(type(v) is int and 0 <= v < RESOLUTION for v in pixel)
        if not (allow_partial and box is None):
            valid = valid and isinstance(box, list) and len(box) == 4
            valid = valid and all(type(v) is int and 0 <= v < RESOLUTION for v in box)
            valid = valid and box[0] < box[2] and box[1] < box[3] and box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]
        if valid and selection_call is not None and camera == selection_call['camera']:
            valid = pixel == selection_call['pixel']
        if not valid:
            feedback['cameras'][camera] = 'invalid_box_or_marker'
            continue
        target['current_views'][camera] = dict(camera=camera, frame=oid, pixel=deepcopy(pixel),
            box=deepcopy(box), width_px=None if box is None else box[2]-box[0],
            height_px=None if box is None else box[3]-box[1], source='astra')
        feedback['cameras'][camera] = 'accepted'
    if target['current_views']:
        feedback['status'] = 'accepted'
        target['tracking_status'] = 'astra_boxes'
    return feedback


def apply_candidate_association(state, bundle, call, profile=C2_R32):
    """Validate model declarations, never infer semantic identity from distance."""
    value = call.get('target_association'); oid = bundle['observation']['observation_id']
    keys = {'observation_id', 'action', 'candidate_id', 'visibility', 'reason',
            'evidence_frame', 'evidence_point_id', 'model_verdict'}
    if not isinstance(value, dict) or set(value) != keys:
        return dict(status='invalid_association')
    reject_privilege(value)
    if value['observation_id'] != oid:
        return dict(status='stale_association')
    action = value['action']; target = state.identity_target
    if action not in ('keep', 'new', 'measurement_only', 'uncertain') or value['visibility'] not in ('visible', 'partial_occlusion', 'full_extent_unavailable', 'historical_only'):
        return dict(status='invalid_association')
    if not isinstance(value['reason'], str) or not value['reason'].strip() or len(value['reason']) > 900:
        return dict(status='invalid_association')
    if value['model_verdict'] not in (None, 'unlike'):
        return dict(status='invalid_association')
    cid = value['candidate_id']
    if cid is not None and type(cid) is not int:
        return dict(status='invalid_candidate')
    if action == 'new' and (cid is not None or value['evidence_frame'] is not None or value['evidence_point_id'] is not None):
        return dict(status='new_must_not_inherit')
    if action == 'keep' and (target is None or cid != target['id']):
        return dict(status='missing_or_wrong_candidate')
    if action in ('measurement_only', 'uncertain') and cid is not None and (target is None or cid != target['id']):
        return dict(status='missing_or_wrong_candidate')
    ref_frame, ref_point = value['evidence_frame'], value['evidence_point_id']
    if ref_frame is not None or ref_point is not None:
        if not isinstance(ref_frame, str) or (ref_point is not None and type(ref_point) is not int):
            return dict(status='invalid_evidence_reference')
        matches = [e for e in state.target_evidence if e['candidate_id'] == cid and e['frame'] == ref_frame]
        if ref_point is not None:
            matches = [e for e in matches if e.get('point_id') == ref_point]
        if not matches:
            return dict(status='invalid_evidence_reference')
    boxes = deepcopy(call.get('target_observation'))
    allow_partial = profile.partial_target_marker and value['visibility'] in ('partial_occlusion', 'full_extent_unavailable')
    if action in ('keep', 'new') and boxes is not None:
        if not isinstance(boxes, dict) or set(boxes) != {'observation_id', 'target_id', 'head', 'hand'}:
            return dict(status='invalid_fields')
        if boxes['target_id'] != cid or boxes['observation_id'] != oid:
            return dict(status='target_id_or_frame_mismatch')
        # Check boxes on a temporary target before committing a new identity.
        temporary = State(identity_target=dict(id=cid, current_views={}))
        checked = apply_target_observation(temporary, bundle, boxes,
            selection_call=call if call['tool']=='locate_point' else None, previous_target_id=cid, allow_partial=allow_partial)
        if any(v == 'invalid_box_or_marker' for v in checked['cameras'].values()):
            return dict(status='invalid_box_or_marker')
    if action == 'new':
        state.identity_sequence += 1
        state.identity_target = dict(id=state.identity_sequence, verdict='unchecked',
            label_text='无', reason='Not reviewed yet.', last_review=None, current_views={})
        target = state.identity_target
    if action == 'uncertain':
        state.target_association_unresolved = True
    if value['model_verdict'] == 'unlike' and target is not None and action in ('keep', 'new'):
        state.negative_targets.append(dict(candidate_id=target['id'], source='model', frame=oid, reason=value['reason']))
    if action not in ('keep', 'new'):
        return dict(status=action, visibility=value['visibility'], association_basis='model_assertion')
    # Existing boxes refer to the candidate before selection; new receives its ID here.
    expected = cid
    if boxes is not None and boxes.get('target_id') != expected:
        return dict(status='target_id_mismatch')
    if boxes is not None:
        boxes['target_id'] = target['id']
    feedback = apply_target_observation(state, bundle, boxes, allow_partial=allow_partial)
    marked = feedback['status'] == 'accepted' and all(v != 'invalid_box_or_marker' for v in feedback['cameras'].values())
    continuity = (action == 'keep' and value['visibility'] in ('partial_occlusion', 'full_extent_unavailable')
                  and ref_frame is not None and feedback['status'] == 'unmeasurable')
    valid = (value['visibility'] == 'visible' and marked) or (value['visibility'] in ('partial_occlusion', 'full_extent_unavailable') and (marked or continuity))
    if valid:
        state.target_association_unresolved = False
    if marked:
        state.target_evidence.append(dict(candidate_id=target['id'], frame=oid, point_id=None,
            views=deepcopy(target['current_views']), source_snapshot=deepcopy(state_snapshot(bundle['observation']))))
    return dict(status='accepted' if valid else 'no_current_visual_support',
        visibility=value['visibility'], association_basis='model_asserted_continuity' if continuity else 'model_assertion',
        evidence_frame=ref_frame, evidence_point_id=ref_point, boxes=feedback)


def validate_accounting_remaining(value):
    keys = {'status', 'currency', 'campaign_remaining_usd', 'case_remaining_usd',
            'provider_bill_status', 'basis', 'ledger_sha256'}
    if not isinstance(value, dict) or set(value) != keys or value['status'] not in ('unknown', 'estimated_ledger_headroom') or value['currency'] != 'USD' or value['provider_bill_status'] != 'unknown':
        raise ValueError('accounting snapshot whitelist')
    for key in ('campaign_remaining_usd', 'case_remaining_usd'):
        if value[key] is not None and (not isinstance(value[key], str) or not Decimal(value[key]).is_finite() or Decimal(value[key]) < 0):
            raise ValueError('accounting amount')
        if value['status']=='unknown' and value[key] is not None:
            raise ValueError('unknown accounting is not zero')
    if not isinstance(value['basis'], str) or len(value['basis']) > 400:
        raise ValueError('accounting basis')
    if value['ledger_sha256'] is not None and (not isinstance(value['ledger_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['ledger_sha256'])):
        raise ValueError('accounting ledger hash')
    reject_privilege(value)


def validate_target_diagnostics(value):
    # Additional provenance for the opt-in current-frame multi-camera verifier.
    allowed = set('association unresolved points evidence reviews review_freshness negative_memory resources status visibility association_basis evidence_frame evidence_point_id boxes cameras head hand point_id candidate_id frame camera pixel valid point_base_m quality source_snapshot distance_from_anchor_m views box width_px height_px source observation_id odometry tcp_pose steps_remaining dx_m dy_m dyaw_rad position quaternion_xyzw method frame_status qpos_status center_valid touches_border neighborhood_valid_fraction neighborhood_depth_range_m trigger request_index marker request_sent verdict wire_sha256 reply label_text ownership_evidence neighbor_features size_evidence reason error_type scope source_frame physical_steps_since as_of_observation_id motion_excludes_current_command current_command_steps_after_observation robot_motion_since before after base_delta_in_previous_base warning decisions_remaining physical_steps_remaining reviews_remaining ledger_budget_status ledger_remaining_usd accounting_remaining currency campaign_remaining_usd case_remaining_usd provider_bill_status basis ledger_sha256'.split())
    allowed.add('source_views')
    def check(item):
        if isinstance(item, dict):
            if set(item)-allowed:
                raise ValueError('target diagnostics whitelist')
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError('target diagnostics type')
    check(value)
    reject_privilege(value)
    canonical(value)


def point_quality(depth, bundle, call):
    """Read-only risk report; never replace the requested center depth."""
    out = dict(method='center_pixel_only_no_fill', frame_status='unavailable', qpos_status='unavailable', center_valid=None,
               touches_border=None, neighborhood_valid_fraction=None, neighborhood_depth_range_m=None)
    frame = getattr(depth, 'frame', None)
    if frame is None:
        return out
    if frame[0] != bundle['observation']['observation_id']:
        out['frame_status'] = 'stale'
        return out
    out['frame_status'] = 'current'
    observed = bundle['observation'].get('joint_positions', {})
    if frame[1] is not None and all(k in observed for k in FETCH_ORDER[3:]):
        measured = np.asarray(frame[1], dtype=float)
        out['qpos_status'] = 'current' if measured.shape == (15,) and np.array_equal(measured[3:], [observed[k] for k in FETCH_ORDER[3:]]) else 'stale'
    x, y = call['pixel']; raw = np.asarray(frame[2][call['camera']]['depth_mm'])
    out['touches_border'] = x < 1 or y < 1 or x >= raw.shape[1]-1 or y >= raw.shape[0]-1
    neighborhood = raw[max(0,y-1):y+2,max(0,x-1):x+2]
    valid = neighborhood[np.isfinite(neighborhood) & (neighborhood>0)]
    out.update(center_valid=bool(np.isfinite(raw[y,x]) and raw[y,x]>0),
        neighborhood_valid_fraction=float(len(valid)/neighborhood.size),
        neighborhood_depth_range_m=float((valid.max()-valid.min())/1000) if len(valid) else None)
    return out


def validate_gripper_envelopes(value):
    if value == dict(status='joint_measurement_unavailable',poses=[]):
        return
    if (not isinstance(value,dict) or set(value)!={'status','observation_id','basis','scene_collision_checked','pose_adjusted','poses'}
            or value['status']!='computed' or not valid_id(value['observation_id'])
            or value['basis']!='own_robot_mesh_AABB_transformed_to_command_base; conservative, not exact occupied vertices'
            or value['scene_collision_checked'] is not False or value['pose_adjusted'] is not False
            or not isinstance(value['poses'],list) or not 1<=len(value['poses'])<=8):
        raise ValueError('robot gripper envelope whitelist')
    for index,row in enumerate(value['poses']):
        if set(row)!={'pose_index','measured_opening','fully_open'} or type(row['pose_index']) is not int or row['pose_index']!=index:
            raise ValueError('robot gripper envelope pose')
        for mode in ('measured_opening','fully_open'):
            if set(row[mode])!={'l','r'}:raise ValueError('robot gripper envelope fingers')
            for box in row[mode].values():
                if set(box)!={'min_base_m','max_base_m'}:raise ValueError('robot gripper envelope bounds')
                lo=vector(box['min_base_m'],3);hi=vector(box['max_base_m'],3)
                if np.any(lo>hi):raise ValueError('robot gripper envelope ordering')


def robot_gripper_envelopes(poses, observation):
    """Conservative own-robot mesh bounds; no scene inputs or pose correction."""
    from itertools import product
    from scipy.spatial.transform import Rotation
    spec = FROZEN['experimental']['identity_revision_2']['self_finger_filter']
    joints = observation.get('joint_positions', {})
    names = {side: side+'_gripper_finger_joint' for side in ('l','r')}
    if any(type(joints.get(name)) not in (int,float) or not math.isfinite(joints[name]) for name in names.values()):
        return dict(status='joint_measurement_unavailable', poses=[])
    records=[]
    for index, proposed in enumerate(poses):
        rotation=Rotation.from_quat(proposed['quaternion_xyzw']).as_matrix()
        position=np.asarray(proposed['position']); modes={}
        for mode in ('measured_opening','fully_open'):
            fingers={}
            for side,sign in (('l',-1),('r',1)):
                opening=joints[names[side]] if mode=='measured_opening' else .05
                bounds=np.asarray(spec['bounds_m'][side])
                corners=np.asarray(list(product(*zip(bounds[0],bounds[1]))))
                corners=corners+np.array([0,sign*(spec['joint_origin_y_m']+opening),0])
                base=corners@rotation.T+position
                fingers[side]=dict(min_base_m=base.min(axis=0).tolist(),max_base_m=base.max(axis=0).tolist())
            modes[mode]=fingers
        records.append(dict(pose_index=index,**modes))
    return dict(status='computed',observation_id=observation['observation_id'],
        basis='own_robot_mesh_AABB_transformed_to_command_base; conservative, not exact occupied vertices',
        scene_collision_checked=False,pose_adjusted=False,poses=records)


def estimate_carried_envelope(state,executor,call,depth=None):
    """Bind a model estimate to a measured surface and current gripper frame.

    This checks consistency with one sensor point, not hidden object geometry.
    It never asserts that closure has attached the object.
    """
    from fetch_episode import ARM
    value=call['carried_envelope'];target=state.identity_target;memory=state.episode_memory
    if (memory is None or target is None or target.get('verdict')!='confirmed'
            or state.target_association_unresolved or executor.closed):
        raise ValueError('confirmed current candidate required')
    point=next((p for p in state.target_points if p['point_id']==value['point_id']),None)
    if (point is None or not point.get('valid') or point.get('candidate_id')!=target['id']
            or point.get('frame')!=value['observation_id'] or point.get('camera')!=value['camera']):
        raise ValueError('candidate-bound measured surface required')
    evidence=memory.observation_evidence(value['observation_id'],value['camera'])
    if evidence['control_step']!=executor.steps:raise ValueError('new surface measurement required after motion')
    measured=executor.ports.observe();q=np.asarray(measured['qpos'],float)
    p,R=executor.session.model.fk(q[ARM],q[3],jac=False)
    from scipy.spatial.transform import Rotation
    surface=(np.asarray(point['point_base_m'],float)-p)@R
    estimate=pose(value['center_gripper_m'],value['quaternion_xyzw'])
    local=(surface-estimate[:3,3])@estimate[:3,:3]
    if np.any(np.abs(local)>np.asarray(value['half_extents_m'])+value['uncertainty_m']):
        raise ValueError('estimated envelope excludes its measured surface')
    memory.record_attachment(dict(candidate_id=target['id'],evidence=evidence,
        center_gripper_m=value['center_gripper_m'],rotation_gripper=estimate[:3,:3].tolist(),
        half_extents_m=value['half_extents_m'],uncertainty_m=value['uncertainty_m'],
        basis='sensor_supported_estimate',reason=value['reason'],surface_seed_gripper_m=surface.tolist(),
        estimate_gripper_odom=(measured['local_from_base']@pose(p,Rotation.from_matrix(R).as_quat())).tolist()),step=executor.steps)
    if depth is not None:
        if depth.frame is None:raise ValueError('current depth frame required')
        oid,source_q,cameras=depth.frame
        count=executor.clearance.associate_carried_surface(oid,source_q,
            memory.observation_binding(oid)['odometry'],cameras,control_step=executor.steps)
        # Ingestion may already have associated this exact frame before the
        # read-only estimate is refreshed. A zero NEW-label count is not a
        # missing-current-evidence result. Historical labels alone still fail.
        current_rows=(memory.attachment() or {}).get('surface_memberships',[])
        current_owned=any(row['evidence']['observation_id']==oid and row['indices'] for row in current_rows)
        if not current_owned:return dict(accepted=False,reason='carried_surface_unresolved',steps_used=0,
                               steps_remaining=executor.max_steps-executor.steps)
    return dict(accepted=True,reason='carried_envelope_estimated',steps_used=0,
                steps_remaining=executor.max_steps-executor.steps)


def dispatch_associated_target(state, executor, call, depth, bundle, deliver, profile, *, sink=None):
    obs = bundle['observation']; oid = obs['observation_id']
    value = call.get('target_association')
    action = value.get('action') if isinstance(value, dict) else None
    before = deepcopy((state.identity_target, state.identity_sequence, state.target_association_unresolved,
                       state.target_evidence, state.negative_targets))
    association = apply_candidate_association(state, bundle, call, profile)
    if profile.active_confirmation and association.get('status') == 'accepted' and value.get('model_verdict') == 'unlike':
        state.identity_target.update(verdict='unlike',last_review_status='invalidated_by_current_model_evidence')
    if association['status'] not in ('accepted', 'no_current_visual_support', 'measurement_only', 'uncertain'):
        (state.identity_target, state.identity_sequence, state.target_association_unresolved,
         state.target_evidence, state.negative_targets) = before
    target = state.identity_target
    closing = call['tool'] == 'set_gripper' and call['gripper'] <= .1
    remaining = executor.max_steps-executor.steps
    valid = association['status'] in ('accepted', 'no_current_visual_support', 'measurement_only', 'uncertain')
    reason = None
    if not valid:
        reason = 'target_association_invalid'
    elif profile.active_confirmation and call['tool'] in ('prepare_grasp', 'replan_grasp', 'set_gripper') and (call['tool'] != 'set_gripper' or closing):
        if (target is None or state.target_association_unresolved or association['status'] != 'accepted'
                or action != 'keep' or target.get('verdict') != 'confirmed'
                or target.get('last_review_status') != 'completed'):
            reason = 'confirmed_current_target_required'
        elif not weekly_target_fresh(state, bundle, executor, call, profile):
            reason = 'fresh_target_review_required'
        elif closing and (state.pick_entry is None or state.pick_entry.get('candidate_id') != target['id']):
            reason = 'current_target_grasp_localization_required'
    elif closing and (target is None or state.target_association_unresolved or association['status'] != 'accepted' or action not in ('keep', 'new')):
        reason = 'current_target_evidence_required'
    elif closing and target['verdict'] == 'unlike':
        reason = 'identity_unlike'
    elif closing and target['verdict'] not in ('like', 'confirmed'):
        reason = 'target_review_required'
    if reason is None and profile.carried_envelope and closing:
        attachment=state.episode_memory.attachment() if state.episode_memory is not None else None
        if (attachment is None or attachment['candidate_id']!=target['id'] or attachment['status']!='estimated'
                or attachment['control_step']!=executor.steps or not attachment.get('surface_memberships')):
            reason='current_candidate_carried_envelope_required'
    if reason is None and profile.carried_envelope and call['tool'] in ('prepare_grasp','replan_grasp'):
        contact=state.episode_memory.contact_attachment() if state.episode_memory is not None else None
        if contact is None or contact['candidate_id']!=target['id'] or contact['control_step']!=executor.steps:
            reason='current_candidate_surface_estimate_required'
    if reason:
        actual, result = None, dict(accepted=False, reason=reason, steps_used=0, steps_remaining=remaining)
    elif profile.active_confirmation and call['tool'] == 'recover_pick':
        actual, result = weekly_visual_recovery(state, executor, call, bundle, association)
    elif profile.carried_envelope and call['tool']=='estimate_carried_envelope':
        try:
            if association['status']!='accepted' or action!='keep':raise ValueError('keep current candidate required')
            result=estimate_carried_envelope(state,executor,call,depth);actual=call
        except ValueError as error:
            actual,result=None,dict(accepted=False,reason='carried_envelope_invalid',
                parameter_error=dict(code='carried_envelope_invalid',field='carried_envelope',expected=str(error)),
                steps_used=0,steps_remaining=remaining)
    elif call['tool'] == 'verify_target':
        sent = False
        available = not profile.weekly_budget or len(state.archive)+1+state.verification_requests < weekly_request_cap(state.response_session, profile)
        observed = not profile.active_confirmation or (target is not None and any(
            e.get('candidate_id')==target['id'] and e.get('source_snapshot',{}).get('steps_remaining',0)>remaining
            for e in state.target_evidence))
        if available and observed and association['status'] == 'accepted' and not state.target_association_unresolved and action in ('keep', 'new'):
            sent = review_shared_target(state, bundle, deliver, profile, explicit=True, sink=sink)
        actual, result = call, dict(accepted=True, reason='review_sent' if sent else 'active_observation_required' if not observed else 'cached_target_record', steps_used=0, steps_remaining=remaining)
        if profile.active_confirmation:
            # A tool returning normally is not proof of an independent review.
            if sent:
                reason = 'review_completed' if target.get('last_review_status') == 'completed' else 'review_failed'
            elif not available:
                reason = 'review_request_budget_exhausted'
            elif state.verification_requests >= profile.verification_call_cap:
                reason = 'review_limit_exhausted'
            elif target is None or association['status'] != 'accepted' or state.target_association_unresolved:
                reason = 'current_target_evidence_required'
            elif not observed:
                reason = 'active_observation_required'
            else:
                reason = 'review_not_sent'
            result.update(accepted=bool(sent and target.get('last_review_status') == 'completed'), reason=reason)
            if sent and result['accepted']:
                weekly_review_recovery(state, executor, bundle, profile)
    else:
        if not closing and call['tool'] != 'locate_point' and association['status'] == 'accepted':
            review_shared_target(state, bundle, deliver, profile, sink=sink)
        actual, result = dispatch(executor, call, depth, state.observation_ids, profile, sink=sink, observation_id=oid)
        if profile.gripper_geometry_feedback and call['tool']=='check_path':
            result=dict(result,robot_gripper_envelopes=robot_gripper_envelopes(call['poses'],obs))
        if call['tool'] == 'locate_point':
            point_id = len(state.target_points)+1
            bound = target is not None and action in ('keep', 'new') and not state.target_association_unresolved
            point = dict(point_id=point_id, candidate_id=target['id'] if bound else None,
                frame=oid, camera=call['camera'], pixel=deepcopy(call['pixel']),
                valid=bool(result.get('valid')), point_base_m=deepcopy(result.get('point_base_m')),
                quality=point_quality(depth, bundle, call), source_snapshot=deepcopy(state_snapshot(obs)))
            state.target_points.append(point)
            if bound and point['valid']:
                position = point_in_odometry(point['point_base_m'], obs['odometry'])
                old_anchor = target.get('selection_anchor_odometry_m')
                point['distance_from_anchor_m'] = None if old_anchor is None else float(np.linalg.norm(np.asarray(position)-old_anchor))
                target.setdefault('selection_anchor_odometry_m', position)
                target.update(position_odometry_m=position, source_frame=oid, selected_camera=call['camera'],
                    selected_pixel=deepcopy(call['pixel']), point_id=point_id)
                state.target_evidence.append(dict(candidate_id=target['id'], frame=oid, point_id=point_id,
                    source_snapshot=deepcopy(state_snapshot(obs))))
            if bound and action == 'new' and point['valid']:
                review_shared_target(state, bundle, deliver, profile, selected=True, sink=sink)
        if closing or call['tool'] in ('move_to', 'move_eef_chunk', 'return_to_rest'):
            sensor_port = getattr(executor, 'grasp_sensor_feedback', None)
            if sensor_port:
                result = dict(result, grasp_sensor_feedback=sensor_port())
    if profile.active_confirmation:
        weekly_after_action(state, executor, call, actual, result, bundle)
    result = dict(result, target_record=target_summary(target), target_observation_feedback=association)
    if profile.target_evidence_diagnostics:
        result['target_evidence_diagnostics'] = target_diagnostics(state, executor, obs, profile, association)
    return actual, result


def target_diagnostics(state, executor, obs, profile, association):
    accounting = obs.get('accounting_remaining')
    if accounting is not None:
        validate_accounting_remaining(accounting)
    return dict(association=deepcopy(association),
        unresolved=state.target_association_unresolved, points=deepcopy(state.target_points),
        evidence=deepcopy(state.target_evidence), reviews=deepcopy(state.target_reviews),
        review_freshness=[dict(request_index=r['request_index'], candidate_id=r['candidate_id'],
            source_frame=r['frame'], physical_steps_since=max(0,r['source_snapshot']['steps_remaining']-obs['steps_remaining']),
            as_of_observation_id=obs['observation_id'], motion_excludes_current_command=True,
            current_command_steps_after_observation=max(0,obs['steps_remaining']-(executor.max_steps-executor.steps)),
            robot_motion_since=transition(r['source_snapshot'], obs)) for r in state.target_reviews],
        negative_memory=deepcopy(state.negative_targets),
        resources=dict(decisions_remaining=max(0, profile.call_cap-len(state.archive)-1),
            physical_steps_remaining=executor.max_steps-executor.steps,
            reviews_remaining=max(0, profile.verification_call_cap-state.verification_requests),
            ledger_budget_status='unknown' if accounting is None else accounting['status'],
            ledger_remaining_usd=None if accounting is None else accounting['case_remaining_usd'],
            accounting_remaining=deepcopy(accounting)))


def dispatch_shared_target(state, executor, call, depth, bundle, deliver, profile, *, sink=None):
    validate_call(call, profile)
    if profile.target_association:
        return dispatch_associated_target(state, executor, call, depth, bundle, deliver, profile, sink=sink)
    actual, result = _dispatch_legacy_shared_target(state, executor, call, depth, bundle, deliver, profile, sink=sink)
    if profile.target_evidence_diagnostics:
        obs = bundle['observation']; target = state.identity_target
        if call['tool'] == 'locate_point':
            entry = dict(point_id=len(state.target_points)+1,
                candidate_id=target['id'] if target is not None and result.get('valid') else None,
                frame=obs['observation_id'], camera=call['camera'], pixel=deepcopy(call['pixel']),
                valid=bool(result.get('valid')), point_base_m=deepcopy(result.get('point_base_m')),
                quality=point_quality(depth,bundle,call), source_snapshot=deepcopy(state_snapshot(obs)))
            state.target_points.append(entry)
        if target and target.get('current_views'):
            state.target_evidence.append(dict(candidate_id=target['id'], frame=obs['observation_id'],
                point_id=None, views=deepcopy(target['current_views']), source_snapshot=deepcopy(state_snapshot(obs))))
        result = dict(result, target_evidence_diagnostics=target_diagnostics(state,executor,obs,profile,
            dict(status='legacy_policy',association_basis='legacy_distance_rule')))
    return actual, result


def _dispatch_legacy_shared_target(state, executor, call, depth, bundle, deliver, profile, *, sink=None):
    obs = bundle['observation']; oid = obs['observation_id']
    closing = call['tool'] == 'set_gripper' and call['gripper'] <= .1
    box_feedback = None
    if profile.model_target_boxes and call['tool'] != 'locate_point':
        box_feedback = apply_target_observation(state, bundle, call.get('target_observation'))
        if not closing and call['tool'] != 'verify_target':
            review_shared_target(state, bundle, deliver, profile, sink=sink)
    if call['tool'] == 'verify_target':
        if not profile.model_target_boxes:
            refresh_target_views(state, bundle, depth)
        sent = review_shared_target(state, bundle, deliver, profile, sink=sink)
        return call, dict(accepted=True, reason='review_sent' if sent else 'cached_target_record', steps_used=0,
            steps_remaining=executor.max_steps-executor.steps, target_record=target_summary(state.identity_target),
            **({'target_observation_feedback': box_feedback} if profile.model_target_boxes else {}))
    if closing and state.identity_target and state.identity_target['verdict'] == 'unlike':
        return None, dict(accepted=False, reason='identity_unlike', steps_used=0,
            steps_remaining=executor.max_steps-executor.steps, target_record=target_summary(state.identity_target),
            **({'target_observation_feedback': box_feedback} if profile.model_target_boxes else {}))
    actual, result = dispatch(executor, call, depth, state.observation_ids, profile, sink=sink, observation_id=oid)
    if call['tool'] == 'locate_point' and result.get('accepted') and result.get('valid'):
        point = point_in_odometry(result['point_base_m'], obs['odometry'])
        old = state.identity_target
        changed = old is None or np.linalg.norm(np.asarray(point)-old['selection_anchor_odometry_m']) > FROZEN['experimental']['identity_revision_2']['selection_distance_m']
        if changed:
            state.identity_sequence += 1
            state.identity_target = dict(id=state.identity_sequence, selection_anchor_odometry_m=point, position_odometry_m=point, source_frame=oid,
                selected_camera=call['camera'], selected_pixel=deepcopy(call['pixel']), verdict='unchecked',
                label_text='无', reason='Not reviewed yet.', last_review=None, current_views={})
        else:
            state.identity_target.update(position_odometry_m=point, source_frame=oid,
                selected_camera=call['camera'], selected_pixel=deepcopy(call['pixel']))
        state.identity_target.pop('_tracking', None)  # explicit reselection reanchors sensor correspondence
        if profile.model_target_boxes:
            box_feedback = apply_target_observation(state, bundle, call.get('target_observation'),
                selection_call=call, previous_target_id=old['id'] if old else None)
        else:
            refresh_target_views(state, bundle, depth)
        review_shared_target(state, bundle, deliver, profile, selected=changed, sink=sink)
    if profile.model_target_boxes:
        result = dict(result, target_observation_feedback=box_feedback, target_record=target_summary(state.identity_target))
    if closing or call['tool'] == 'locate_point':
        result = dict(result, target_record=target_summary(state.identity_target))
    # Manipulation measurements belong to its sensor port, never to the verifier.
    if closing or call['tool'] in ('move_to', 'move_eef_chunk', 'return_to_rest'):
        sensor_port = getattr(executor, 'grasp_sensor_feedback', None)
        if sensor_port:
            result = dict(result, grasp_sensor_feedback=sensor_port())
    return actual, result


class Verification(Protocol):

    def __call__(self, observation: Mapping, candidate: Mapping) -> Mapping:
        ...


# Navigate: direct robot executor calls.
def look(executor, pan_rad, tilt_rad):
    return executor.look(pan_rad, tilt_rad)

def move_base(executor, turn_rad, forward_m):
    return executor.move_base(turn_rad, forward_m, **({'max_steps':250} if getattr(executor,'r15_protocol',False) else {}))


# Manipulation: existing executor refusals and per-call limits unchanged.
def check_path(executor, goals):
    return {**executor.check_path(goals), 'readonly': True, 'minimum_steps_is_lower_bound': True}

def move_to(executor, goal):
    return executor.move_to(goal, max_steps=300 if getattr(executor,'r15_protocol',False) else 150)

def move_eef_chunk(executor, goals):
    return executor.move_eef_chunk(goals, max_steps=300 if getattr(executor,'r15_protocol',False) else 150)

def set_gripper(executor, gripper):
    return executor.set_gripper(gripper)

def return_to_rest(executor):
    return executor.return_to_rest()


# Trace: output-only; never appended to model history.
def record(sink, function, inputs, output, start_step, end_step, observation_id):
    if sink is not None:
        sink(deepcopy(dict(function=function, inputs=inputs, output=output, start_step=start_step, end_step=end_step, steps_used=end_step - start_step, observation_id=observation_id)))


# Main loop and durable accounting: only Astra chooses the next tool.
def estimate_wire(wire):
    body = json.loads(wire)
    if 'previous_response_id' in body or body.get('store') is not False:
        raise ValueError('hidden history forbidden')
    images = [c for message in body['input'] for c in message['content'] if c['type'] == 'input_image']
    if len(images) not in (0, 4, 8, 12):
        raise ValueError('bounded image groups')
    image_tokens = len(images) * 480
    input_bound = non_image_bytes(wire) + 1024 + image_tokens
    if input_bound + OUTPUT_CAP > CONTEXT_LIMIT:
        raise ValueError('context cap')
    return {'image_count': len(images), 'image_tokens': image_tokens, 'text_bytes': non_image_bytes(wire), 'input_bound': input_bound, 'output_cap': OUTPUT_CAP, 'reserve_usd': str(cost(input_bound, OUTPUT_CAP, reserve_write=True))}

def budget_entries(ledger):
    """All monetary obligations; generation request counters still use entries only."""
    return list(ledger['entries'])+list(ledger.get('count_entries',[]))


def budget_cost(ledger, case_id=None, attempt_id=None):
    return committed_cost([e for e in budget_entries(ledger)
        if (case_id is None or e['case_id']==case_id)
        and (attempt_id is None or e.get('attempt_id')==attempt_id)])


def budgeted_input_counter(transport, ledger, ledger_path, auth, case, attempt_id,
                           parent_kind, sequence, *, delivery=None):
    """Durable USD0.01 risk allowance per count, not verified provider billing.

    The session/admission lock is held by the caller. ConcurrentDelivery releases
    the ledger mutex during network delivery and refreshes both journals on return.
    """
    policy=auth.get('input_count_authorization',{})
    required=dict(accounting_policy='usd-0.01-per-count-risk-reserve-v1',
        pricing_status='unconfirmed_budgeted',risk_reserve_per_request_usd='0.01',
        actual_charge_usd=None,included_in_case_and_batch_caps=True)
    if any(type(policy.get(k)) is not type(v) or policy.get(k)!=v for k,v in required.items()):
        raise ValueError('count risk allowance must be explicitly included in shared caps')
    if parent_kind not in ('main','verification'):raise ValueError('count parent kind')
    group=f"{auth['batch_id']}-{case}-{attempt_id}-{parent_kind}-{sequence}"
    allowance=Decimal('0.01')
    def send(payload,key):
        if delivery:assert delivery.locked
        if ledger.get('status')!='running':raise RuntimeError('count ledger stopped')
        rows=ledger.setdefault('count_entries',[])
        prior=[e for e in rows if e['case_id']==case]
        group_rows=[e for e in prior if e['count_group']==group]
        if auth.get('count_once_per_wire') is True:
            digest=hashlib.sha256(payload).hexdigest()
            matches=[e for e in group_rows if e.get('payload_sha256')==digest]
            if matches:
                if len(matches)!=1 or matches[0]['status']!='valid':
                    raise ValueError('unresolved duplicate count; no automatic retry')
                return canonical(dict(object='response.input_tokens',input_tokens=matches[0]['input_tokens']))
        scope=None;reason=None
        if len(rows)>=policy.get('total_cap',3200):scope,reason='global','count_batch_cap'
        elif len(prior)>=policy.get('max_per_case',320):scope,reason='case','count_case_cap'
        elif len(group_rows)>=(policy.get('max_per_main',4) if parent_kind=='main' else policy.get('max_per_review',1)):
            raise ValueError('count per-generation cap exceeded; no retry')
        if scope is None and delivery:
            delivery.admit_budget(ledger,case,allowance,payload,80,
                case_entries=attempt_id,cold_exposure=allowance)
            rows=ledger.setdefault('count_entries',[])
            # Session lock excludes same-case races; peers can consume global slots.
            if len(rows)>=policy.get('total_cap',3200):scope,reason='global','count_batch_cap'
        if scope is None:
            if budget_cost(ledger,case,attempt_id)+allowance>Decimal(str(auth['per_case_usd_cap'])):
                scope,reason='case','count_case_usd_cap'
            elif budget_cost(ledger)+allowance>Decimal(str(auth['usd_cap'])):
                scope,reason='global','count_batch_usd_cap'
        if scope is not None:
            if scope=='global':ledger['status']='stopped_budget_exhausted';atomic_ledger_save(ledger_path,ledger)
            error=C2BudgetExhausted('input counting budget exhausted before network: '+reason)
            error.scope=scope
            error.budget_snapshot=lean_budget_snapshot(budget_entries(ledger),
                [e for e in budget_entries(ledger) if e['case_id']==case],auth,
                reserve=allowance,cold_exposure=allowance,reason=reason)
            raise error
        row=dict(request_id=group+'-count-'+str(len(group_rows)),case_id=case,attempt_id=attempt_id,
            request_kind='input_count',parent_request_kind=parent_kind,count_group=group,
            endpoint='/v1/responses/input_tokens',status='reserved',reserve_usd='0.01',estimated_usd=None,
            actual_charge_usd=None,pricing_status='unconfirmed_budgeted',
            accounting_policy=required['accounting_policy'],
            basis='risk_allowance_not_provider_price_or_proven_upper_bound; retained_until_billing_reconciliation',
            payload_sha256=hashlib.sha256(payload).hexdigest(),created_at_epoch=time.time())
        rows.append(row)
        if delivery:delivery.reserve(ledger,row)
        else:atomic_ledger_save(ledger_path,ledger)
        try:
            raw=delivery.send(transport,payload,key,ledger,row) if delivery else transport(payload,key)
            row['response_raw_base64']=base64.b64encode(raw).decode('ascii')
            response=json.loads(raw)
            if (response.get('object')!='response.input_tokens' or type(response.get('input_tokens')) is not int
                    or response['input_tokens']<=0):raise ValueError('invalid count response')
            row.update(status='valid',estimated_usd='0.01',input_tokens=response['input_tokens'],
                response_sha256=hashlib.sha256(raw).hexdigest(),settled_at_epoch=time.time())
            atomic_ledger_save(ledger_path,ledger)
            return raw
        except BaseException as error:
            row.update(status='unknown',error_type=type(error).__name__)
            ledger['status']='stopped_unknown_outcome';atomic_ledger_save(ledger_path,ledger)
            raise
    return send


def committed_cost(entries):
    return sum((Decimal(e['estimated_usd']) if e['status'] in ('valid', 'invalid_call', 'invalid_verification', 'not_sent') and e.get('estimated_usd') is not None else Decimal(e.get('worst_case_usd',e['reserve_usd'])) for e in entries), Decimal(0))

def _waypoint_progress_public(value):
    keys={'waypoint_index','waypoints_total','waypoints_completed','stalled_steps'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Waypoint progress whitelist')
    n=value['waypoints_total'];done=value['waypoints_completed'];i=value['waypoint_index'];steps=value['stalled_steps']
    if (type(n) is not int or not 1<=n<=8 or type(done) is not int or not 0<=done<=n
        or type(steps) is not int or not 0<=steps<=300
        or (i is None and done!=n) or (i is not None and (type(i) is not int or i!=done+1 or i>n))):
        raise ValueError('Waypoint progress range')
    return deepcopy(value)


def r15_feedback_public(value):
    keys={'tool','status','requested','actual','residual','detail','requires_replan',
          'head_rad','base_delta','poses','joints','gripper_command','finger_distance_m',
          'position_error_m','rotation_error_rad','max_joint_error_rad','rest_flags',
          'ee_rest','robot_rest','is_static','closure_command_completed',
          'waypoint_progress','waypoint_index','waypoints_total','waypoints_completed','stalled_steps'}
    def check(x):
        if isinstance(x,dict):
            if set(x)-keys:raise ValueError('Non-robot completion feedback field')
            for y in x.values():check(y)
        elif isinstance(x,(list,tuple)):
            for y in x:check(y)
        elif isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite feedback')
        elif x is not None and not isinstance(x,(str,int,float,bool)):raise ValueError('Invalid feedback')
    check(value)
    if 'waypoint_progress' in value:_waypoint_progress_public(value['waypoint_progress'])
    if value.get('status') not in ('arrived','step_cap_reached','blocked','official_terminated'):
        raise ValueError('Unknown motion status')
    return deepcopy(value)


def dispatch(executor, call, depth, observation_ids=(), profile=C0, *, sink=None, observation_id=None):
    motions=('look','move_base','move_to','move_eef_chunk','prepare_grasp','replan_grasp','set_gripper','return_to_rest')
    if not (profile.r15_completion or profile.r2a_feedback) or call['tool'] not in motions:
        return _dispatch_original(executor,call,depth,observation_ids,profile,sink=sink,observation_id=observation_id)
    validate_call(call,profile)
    before=executor.steps;actual=[None]
    def operation():
        actual[0],result=_dispatch_original(executor,call,depth,observation_ids,profile,observation_id=observation_id)
        return result
    result=executor.r15_execute(call['tool'],call,operation)
    if profile.r2a_feedback:result['reason']=result['motion_completion']['detail']
    result=public_result(result, s11_system=profile in V3_SYSTEM_PROFILES, r15_completion=True, rest_progress_feedback=profile.rest_progress_feedback, r2a_close=profile.r2a_close, r2b_grip_monitor=profile.r2b_grip_monitor, motion_progress_only=profile.motion_progress_only, arm_base_explicit=profile.arm_base_explicit, arm_tools_lock_base=profile.arm_tools_lock_base, base_tracking_exit=profile.base_tracking_exit, arm_base_push_exit=profile.arm_base_push_exit, g1_dispatch_semantics=profile.g1_dispatch_semantics, **{f:getattr(profile,f) for f in R2E_FLAGS})
    output=(actual[0],result)
    record(sink,call['tool'],call,output,before,executor.steps,observation_id)
    return output


def _dispatch_original(executor, call, depth, observation_ids=(), profile=C0, *, sink=None, observation_id=None):
    validate_call(call, profile)
    if profile.rest_base_push_report_only and getattr(executor,'rest_base_push_report_only',False) is not True:
        raise ValueError('rest report candidate requires its explicitly bound executor')
    if profile.g1_insertion_guard:
        raise ValueError('G1 insertion threshold is uncalibrated; no execution is authorized')
    for flag in ('compound_feedback','r2d_base_clearance','base_tracking_exit','arm_base_push_exit','g1_dispatch_semantics')+R2E_FLAGS:
        if getattr(profile,flag) and getattr(executor,flag,False) is not True:
            raise ValueError('R2d executor flag not bound: '+flag)
    if profile.local_clearance and (not profile.episode_memory or
            getattr(executor,'local_clearance_guard',False) is not True or getattr(executor,'clearance',None) is None):
        raise ValueError('Local clearance requires bound sensor checker and episode memory')
    if bool(getattr(executor,'clearance_unknown_advisory',False))!=profile.clearance_unknown_advisory:
        raise ValueError('Clearance advisory policy does not match profile')
    if bool(getattr(executor,'carried_envelope_enabled',False))!=profile.carried_envelope:
        raise ValueError('Carried envelope policy does not match profile')
    if profile.motion_guard and getattr(executor,'motion_guard',False) is not True:
        raise ValueError('Motion-guard profile requires a guarded executor')
    if profile.base_tracking_guard and (not profile.motion_guard or
            getattr(executor,'base_tracking_guard',False) is not True):
        raise ValueError('Base-tracking profile requires guarded executor and motion feedback')
    if profile.grasp_alignment and getattr(executor,'version',None)!='full-B-r4-grasp-alignment-v1':
        raise ValueError('Explicit aligned Full B executor required')
    if profile in PAIRED_PROFILES:
        for flag in ('r15_protocol','r15_completion','r15_pregrasp','r15_carry'):
            if getattr(executor,flag,False)!=getattr(profile,flag):raise ValueError('R1.5 executor/profile mismatch: '+flag)
    before = executor.steps
    tool = call['tool']
    if profile.measure and tool in ('measure', 'locate_point'):
        try:
            result = measure(depth, call['camera'], call['pixels'] if tool == 'measure' else [call['pixel']], executor.read_qpos())
        except FieldError as error:
            output = (None, dict(accepted=False, reason='invalid_tool_call', parameter_error=error.feedback, steps_used=0, steps_remaining=executor.max_steps - executor.steps))
        else:
            if tool == 'locate_point':
                result.update({k: v for k, v in result['samples'][0].items() if k != 'pixel'})
            result['steps_remaining'] = executor.max_steps - executor.steps
            output = (call, result)
    else:
        try:
            goals = [pose(p['position'], p['quaternion_xyzw']) for p in call['poses']]
            if tool in ('done', 'give_up'):
                result = dict(accepted=True, reason=tool)
            elif tool == 'recall_observation':
                if call['observation_id'] not in observation_ids:
                    raise FieldError('invalid_observation_id', 'observation_id', 'archived ID from this attempt') if profile.strict_tool_errors else ValueError('recall requires an archived ID from this attempt')
                result = dict(accepted=True, readonly=True, observation_id=call['observation_id'], reason='archived_images_next_request')
            elif tool == 'locate_point':
                result = dispatch_locate(depth, call['camera'], call['pixel'], call['observation_id'], executor.read_qpos(), profile)
            elif tool in ('locate_points', 'approach_point'):

                if call['observation_id'] != observation_id:

                    raise FieldError('stale_observation', 'observation_id', 'latest observation ID') if profile.strict_tool_errors else ValueError('latest observation_id required')

                if tool == 'locate_points':

                    measured = executor.read_qpos()

                    rows = [dict(pixel=list(pixel), result=dispatch_locate(depth, call['camera'], pixel,

                        call['observation_id'], measured, profile)) for pixel in call['pixels']]

                    result = dict(accepted=True, readonly=True, reason='located_points', located_points=rows)

                else:

                    located = dispatch_locate(depth, call['camera'], call['pixel'], call['observation_id'], executor.read_qpos(), profile)

                    result = approach_from_point(executor, located, call['standoff_m'], profile)

            elif tool == 'look':
                result = look(executor, call['pan_rad'], call['tilt_rad'])
            elif tool == 'move_base':
                if profile.fast_base_approach:
                    if not getattr(executor, 'fast_base_approach', False):
                        raise ValueError('Explicit fast approach executor required')
                    result = executor.move_base(call['turn_rad'], call['forward_m'], speed_mode=call['base_speed_mode'])
                else:
                    result = move_base(executor, call['turn_rad'], call['forward_m'])
            elif tool == 'check_clearance':
                result = executor.check_clearance(goals,rest=call['rest'],**(dict(turn_rad=call['turn_rad'],forward_m=call['forward_m']) if profile.r2d_base_clearance else {}))
            elif tool == 'check_path':
                result = check_path(executor, goals)
            elif tool == 'move_to':
                result = move_to(executor, goals[0])
            elif tool in ('prepare_grasp','replan_grasp'):
                if not profile.grasp_alignment or getattr(executor,'version',None)!='full-B-r4-grasp-alignment-v1':
                    raise ValueError('Explicit aligned Full B executor required')
                result = executor.prepare_grasp(goals[0],reason=call['note'],replan=tool=='replan_grasp')
            elif tool == 'move_eef_chunk':
                result = move_eef_chunk(executor, goals)
            elif tool == 'set_gripper':
                if profile.structured_close_evidence and call['gripper']==0 and (
                        call['close_evidence']['fingers_bracket_object'] is not True
                        or call['close_evidence']['observation_id']!=observation_id):
                    result=dict(accepted=False,reason='hand_image_evidence_required',closure_command_completed=False)
                elif profile.structured_close_evidence and profile.r2a_close and call.get('close_after_blocked') is True:
                    result=executor.close_after_blocked(call['note'],observation_id,close_evidence=call['close_evidence'],structured_evidence=True)
                else:
                    result = (executor.close_after_blocked(call['note'],observation_id) if profile.r2a_close and call.get('close_after_blocked') is True else set_gripper(executor, call['gripper']))
            else:
                result = return_to_rest(executor)
        except (ValueError, TypeError, KeyError) as error:
            if profile.strict_tool_errors and not isinstance(error, FieldError):
                raise ToolExecutionError(f'{tool}: {type(error).__name__}: {error}') from error
            if tool in ('move_to', 'move_eef_chunk') or executor.steps != before:
                raise
            feedback = str(error)[:150]
            if profile.measure:
                field_name, expected = ('observation_id', 'an archived observation ID from this attempt') if tool == 'recall_observation' else ('poses', 'finite robot-frame poses accepted by the original WP2 executor')
                feedback = dict(code='execution_parameter_rejected', field=field_name, expected=expected)
            output = (None, dict(accepted=False, reason='invalid_tool_call', parameter_error=feedback, steps_used=0, settle_steps=0, motion_steps=0, steps_remaining=executor.max_steps - executor.steps))
        else:
            used = executor.steps - before
            settling = sum((t['kind'] == 'base_settle' for t in executor.trace if t['step'] > before))
            if profile.weekly_budget:
                settling = sum(t['kind'] in ('base_settle', 'settling') for t in executor.trace if t['step'] > before)
            if getattr(executor,'version',None)=='full-B-r4-task-rules-v1-online-v1' and tool=='set_gripper':
                result=dict(result,closure_command_completed=result.get('closure_command_completed',False))
            output = (call, public_result(dict(result, steps_used=used, settle_steps=settling, motion_steps=used - settling, steps_remaining=executor.max_steps - executor.steps), robot_diagnostics=profile.robot_diagnostics, schedule_precheck=profile.schedule_precheck, base_progress_monitor=profile.base_progress_monitor, base_response_monitor=profile.base_response_monitor, base_response_debt_monitor=profile.base_response_debt_monitor, execution_lifecycle=profile.execution_lifecycle, gripper_geometry_feedback=profile.gripper_geometry_feedback, motion_guard=profile.motion_guard, local_clearance=profile.local_clearance, rest_progress_feedback=profile.rest_progress_feedback, approach_point=profile.approach_point, multi_locate=profile.multi_locate, r15_completion=profile.r15_completion, r15_protocol=profile.r15_protocol, r2a_identity=profile.r2a_identity, identity_attempt_recovery=profile.identity_attempt_recovery, identity_confirmation=profile.identity_confirmation, identity_multiview_evidence=profile.identity_multiview_evidence, identity_confirmed_persistence=profile.identity_confirmed_persistence, r2a_close=profile.r2a_close, r2a_feedback=profile.r2a_feedback, r2b_clearance=profile.r2b_clearance, r2b_grip_monitor=profile.r2b_grip_monitor, motion_progress_only=profile.motion_progress_only, arm_base_explicit=profile.arm_base_explicit, arm_tools_lock_base=profile.arm_tools_lock_base, base_tracking_exit=profile.base_tracking_exit, arm_base_push_exit=profile.arm_base_push_exit, g1_dispatch_semantics=profile.g1_dispatch_semantics, **{f:getattr(profile,f) for f in R2E_FLAGS}, hot_recovery=profile==SOL_HOT_RECOVERY,s11_system=profile in V3_SYSTEM_PROFILES))

            if (profile.base_motion_abort or profile.base_progress_abort) and tool == 'move_base' and 'base_abort_rule' in result:
                output[1]['base_abort_rule'] = result['base_abort_rule']
            if profile.stall_feedback and tool in ('move_to', 'move_eef_chunk'):
                result.setdefault('execution_phase', 'planning')
                result.setdefault('consecutive_ik_failures', 0)
                output[1].update(execution_diagnostics(result))
        if profile.measure and isinstance(output[1].get('parameter_error'), str):
            field_name, expected = ('observation_id', 'an archived observation ID from this attempt') if tool == 'recall_observation' else ('poses', 'finite robot-frame poses accepted by the original WP2 executor')
            output = (output[0], dict(output[1], parameter_error=dict(code='execution_parameter_rejected', field=field_name, expected=expected)))
    if profile.r15_completion and tool=='approach_point' and 'motion_completion' not in output[1]:
        result=output[1]
        feedback=dict(tool=tool,status='blocked',requested={'base_delta':None},actual={'base_delta':[0.,0.,0.]},
                      residual={'base_delta':None},detail=result['reason'],requires_replan=False)
        output=(output[0],dict(result,reason='blocked',arrived=False,motion_completion=r15_feedback_public(feedback)))
    record(sink, tool, call, output, before, executor.steps, observation_id)
    return output

@dataclass
class State:
    archive: list = field(default_factory=list)
    history: list = field(default_factory=list)
    previous: dict | None = None
    observation_ids: list = field(default_factory=list)
    terminal_reason: str | None = None
    identity_target: dict | None = None
    identity_sequence: int = 0
    verification_requests: int = 0
    r2a_reviews: dict = field(default_factory=dict)
    r2a_confirmed_candidate: dict = field(default_factory=dict)
    identity_attempt: dict = field(default_factory=dict)
    target_points: list = field(default_factory=list)
    target_evidence: list = field(default_factory=list)
    target_reviews: list = field(default_factory=list)
    negative_targets: list = field(default_factory=list)
    target_association_unresolved: bool = False
    response_session: ResponseSession | None = None
    pick_entry: dict | None = None
    pick_closure: dict | None = None
    recovery_event: dict | None = None
    episode_memory: object | None = None


def update_episode_memory(state, bundle, call, actual, result, executor):
    """Copy public evidence into memory; memory never grants target permission."""
    def bounded_text(value):
        return str(value)[:160]
    memory=state.episode_memory
    if memory is None:raise ValueError('Episode memory is not bound')
    obs=bundle['observation'];oid=obs['observation_id']
    target=state.identity_target
    camera=(call or {}).get('camera')
    if camera not in ('head','hand'):camera='head'
    evidence=memory.observation_evidence(oid,camera)
    if target is not None:
        current={name:view for name,view in (target.get('current_views') or {}).items()
            if name in ('head','hand') and isinstance(view,dict) and view.get('frame')==oid}
        visible=bool(current) and not state.target_association_unresolved
        review=next((r for r in reversed(state.target_reviews)
            if r.get('candidate_id')==target['id'] and r.get('frame')==oid
            and r.get('status')=='completed'),None)
        # Motion tools have no camera argument. Attribute identity to the view
        # which actually contains this candidate, not the action's default head
        # source. An independent review remains the authoritative identity view.
        candidate_camera=camera if camera in current else next(iter(current),camera)
        identity_camera=review.get('camera',candidate_camera) if review else candidate_camera
        identity_evidence=memory.observation_evidence(oid,identity_camera)
        verdict=review.get('verdict') if review else None
        status='confirmed' if verdict=='confirmed' else 'rejected' if verdict=='unlike' else 'unknown'
        record=dict(id=target['id'],status=status,visible=visible,evidence=identity_evidence,
            label='candidate '+str(target['id']))
        # The independent review supplies contradiction. Planner uncertainty,
        # occlusion and a changed pixel alone cannot withdraw confirmation.
        if status=='rejected':
            reason=bounded_text((review.get('reply') or {}).get('reason','visual identity mismatch'))
            record['contradiction']=dict(kind='identity_mismatch',reason=reason,evidence=deepcopy(identity_evidence))
        association=(call or {}).get('target_association') or {}
        if association.get('model_verdict')=='unlike' and target.get('last_review_status')=='invalidated_by_current_model_evidence':
            record['status']='rejected'
            record['evidence']=identity_evidence
            record['contradiction']=dict(kind='identity_mismatch',
                reason=bounded_text('planner: '+association.get('reason','visible contradiction')),evidence=deepcopy(identity_evidence))
        point=next((p for p in reversed(state.target_points) if p.get('candidate_id')==target['id']
            and p.get('frame')==oid and p.get('valid')),None)
        if point is not None and visible:
            record['last_visible_position_odom_m']=list(map(float,point_in_odometry(point['point_base_m'],obs['odometry'])))
            record['position_evidence']=memory.observation_evidence(oid,point.get('camera',camera))
        memory.record_target(record,step=executor.steps)
    tool=(call or {}).get('tool','invalid_call');reason=bounded_text(result.get('reason','unknown'))
    if not result.get('accepted'):outcome='rejected'
    elif 'stall' in reason:outcome='stalled'
    elif 'deviat' in reason:outcome='deviated'
    elif result.get('arrived') is False:outcome='failed'
    else:outcome='completed'
    exit_odom=dict(zip(('dx_m','dy_m','dyaw_rad'),map(float,executor.odometry())))
    memory.record_action(dict(id='decision-'+str(len(state.archive)),tool=tool,outcome=outcome,
        reason=reason,entry_odometry=deepcopy(obs['odometry']),exit_odometry=exit_odom,
        steps_used=int(result.get('steps_used',0)),evidence=evidence,
        **({'target_id':target['id']} if target else {})),step=executor.steps)

def s2_advance(state,bundle,response,executor,depth,profile,*,sink=None,verification_deliver=None):
    """Bind one provider decision to a durable list of actual command prefixes."""
    if response['status']=='censored':raise RuntimeError('provider censored')
    plan=response.get('call');session=state.response_session
    expected=hashlib.sha256(canonical(dict(bundle=bundle,call=plan))).hexdigest()
    if (session is None or session.status!='awaiting_execution' or not session.receipts
            or session.receipts[-1]['response_id']!=response.get('provider_id')
            or session.receipts[-1]['archive_sha256']!=expected or len(session.receipts)!=len(state.archive)+1):
        raise ValueError('S2 response not durably delivered or already executed')
    start=len(state.history);invalid=None
    if plan is not None:
        try:
            previous=max((item['call']['revision'] for item in state.archive if item.get('call')
                and item['call']['plan_id']==plan['plan_id']),default=0)
            s2_validate_plan(plan,profile,observation=bundle['observation'],previous_revision=previous)
            s2_validate_plan_evidence(plan,state.observation_ids+[bundle['observation']['observation_id']])
        except (ValueError,TypeError,KeyError) as exc:invalid=str(exc)
    else:invalid=response.get('parameter_error','invalid plan schema')
    session.status='executing';session.persist()
    if invalid is not None:
        result=dict(accepted=False,reason='invalid_tool_call',steps_used=0,settle_steps=0,motion_steps=0,
            steps_remaining=profile.step_cap-executor.steps,parameter_error=str(invalid)[:150])
        aggregate=dict(plan_id=plan.get('plan_id') if isinstance(plan,dict) else None,
            revision=plan.get('revision') if isinstance(plan,dict) else None,decision_turn=bundle['turn'],
            execution_status='rejected',entries=[dict(call=None,result=result)],completed_command_ids=[],
            interrupted_command_id=None,unexecuted_command_ids=[x['command_id'] for x in plan['commands']] if plan else [],
            execution_cursor=dict(command_index=0,control_step=executor.steps,steps_used=0),
            event=dict(code='invalid_contract',scope='plan',decision_needed=True),
            observation_id=bundle['observation']['observation_id'],requires_new_observation=True,
            steps_used=0,steps_remaining=profile.step_cap-executor.steps)
        record(sink,'invalid_call',response,(None,result),executor.steps,executor.steps,bundle['observation']['observation_id'])
        record(sink,'decision_result',plan,(None,result),executor.steps,executor.steps,bundle['observation']['observation_id'])
        record(sink,'s2_plan_result',plan,(None,aggregate),executor.steps,executor.steps,bundle['observation']['observation_id'])
    else:
        from .run_case import execute_s2_plan
        aggregate=execute_s2_plan(state,bundle,plan,executor,depth,profile,sink=sink,
                                 verification_deliver=verification_deliver)
    rows=[deepcopy(dict(call=row['call'],result=row['result'])) for row in aggregate['entries']]
    state.archive.append(deepcopy(dict(bundle=bundle,call=plan)))
    state.history.extend(rows)
    state.previous=deepcopy(state_snapshot(bundle['observation']))
    state.observation_ids.append(bundle['observation']['observation_id'])
    receipt=session.receipts[-1]
    receipt.update(command_range=[start,len(state.history)],execution_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        execution=deepcopy(rows),plan_execution=deepcopy(aggregate),context_mode='window' if profile.image_window else 'chain')
    session.status='stopped' if state.terminal_reason else 'ready';session.persist()
    return plan,aggregate


def advance(state, bundle, response, executor, depth, profile=C0, *, sink=None, verification_deliver=None):
    """Apply exactly one delivered response. No retry or extra decision."""
    if profile.stage_plan:
        return s2_advance(state,bundle,response,executor,depth,profile,sink=sink,verification_deliver=verification_deliver)
    if response['status'] == 'censored':
        raise RuntimeError('provider censored')
    call = response.get('call')
    if profile.search_recessed_view and call is not None:
        validate_search_decision(call,state.archive,bundle['observation'],candidate_priority=profile.search_candidate_priority)
    if profile.response_session:
        session = state.response_session
        expected = hashlib.sha256(canonical(dict(bundle=bundle, call=call))).hexdigest()
        if (session is None or session.status != 'awaiting_execution' or not session.receipts
                or session.receipts[-1]['response_id'] != response.get('provider_id')
                or session.receipts[-1]['archive_sha256'] != expected
                or len(session.receipts) != len(state.archive)+1):
            raise ValueError('session response not durably delivered or already executed')
        session.status = 'executing'
        session.persist()  # Before any physical dispatch; uncertainty must never replay it.
    if call is None:
        actual = None
        result = dict(accepted=False, reason='invalid_tool_call', steps_used=0, parameter_error=response.get('parameter_error', 'invalid tool schema; check required fields'), steps_remaining=profile.step_cap - executor.steps)
        record(sink, 'invalid_call', response, (actual, result), executor.steps, executor.steps, bundle['observation']['observation_id'])
    else:
        if profile.active_memory or profile.task_handoff:
            # Memory belongs to the model transcript, never the controller command.
            if profile.task_handoff:
                context = task_handoff_context(state.archive, state.history, state.response_session)
                validate_task_handoff(call.get('handoff'), tool=call['tool'],
                    known_ids=state.observation_ids+[bundle['observation']['observation_id']],
                    context=context['current'], plan=state.response_session.handoff_plan)
            command = {k:v for k,v in call.items() if k not in ('memory', 'handoff')}
            actual, result = dispatch_with_identity(state, executor, command, depth, bundle, verification_deliver, SOL_SESSION, sink=sink)
        else:
            actual, result = dispatch_with_identity(state, executor, call, depth, bundle, verification_deliver, profile, sink=sink)
    if (profile.hot_continuation and actual and actual['tool']=='give_up'
            and not (state.response_session.single_pick or {}).get('search_continuation')):
        # Keep history in the public result contract. The reason already carries
        # the soft boundary; extra private fields break the next observation.
        result.update(reason='continuation_requested')
    if profile.verification:
        record(sink, 'decision_result', call, (actual, result), executor.steps-result.get('steps_used',0), executor.steps, bundle['observation']['observation_id'])
    if profile.episode_memory:
        update_episode_memory(state,bundle,call,actual,result,executor)
    state.archive.append(deepcopy(dict(bundle=bundle, call=call)))
    state.history.append(deepcopy(dict(call=actual, result=result)))
    state.previous = deepcopy(state_snapshot(bundle['observation']))
    state.observation_ids.append(bundle['observation']['observation_id'])
    if actual and actual['tool'] in ('done', 'give_up') and result.get('reason')!='continuation_requested':
        state.terminal_reason = actual['tool']
    if profile.response_session:
        if profile.task_handoff:
            state.response_session.handoff_journal = task_handoff_context(state.archive, state.history, state.response_session)
        session.receipts[-1]['execution_sha256'] = hashlib.sha256(canonical(state.history[-1])).hexdigest()
        session.receipts[-1]['execution'] = deepcopy(state.history[-1])
        session.status = 'stopped' if state.terminal_reason else 'ready'
        session.persist()
    return (actual, result)

def maybe_hot_continue(state, executor, stopped, profile):
    if not profile.hot_continuation or stopped() or state.terminal_reason:
        return False
    session=state.response_session
    if session is None:raise ValueError('Live continuation requires the same response session')
    if (session.single_pick or {}).get('search_continuation'):return False
    used=len(state.archive)+state.verification_requests
    requested=bool(getattr(state,'history',[]) and state.history[-1]['result'].get('reason')=='continuation_requested')
    if not requested and used<weekly_request_cap(session,profile)-3 and executor.usable_step_cap()-executor.steps>272:
        return False
    if executor.pending is not None or executor.failed or executor.ended:
        raise ValueError('Cannot continue an unknown or terminated action')
    checkpoint=executor.hot_checkpoint(state)
    if stopped():raise ValueError('Official stop during checkpoint')
    executor.release_search_continuation()
    persist_weekly_state(state)
    session.single_pick['search_continuation']=dict(checkpoint=checkpoint,
        requests_used=used,steps_used=executor.steps,additional_requests=40,additional_steps=1200)
    session.persist()
    return True


def run(state, observe, deliver, executor, depth, stopped, profile=C0, *, sink=None, verification_deliver=None):
    """Ports supply sensor bundles, durable delivery and the original stop signal.

    deliver(wire, bundle, archive) must retain the frozen broker's spending,
    lease, reservation, no-retry and durable-output checks. No live transport
    is shipped here. Official outcome classification stays in the evaluator.
    """
    while len(state.archive) < profile.call_cap:
        if stopped() or state.terminal_reason:
            break
        if profile.cold_checkpoint and getattr(executor,'action_budget_expired',lambda:False)():
            executor.cold_checkpoint(state)
            raise TimeoutError('authorized worker wall limit at completed boundary')
        continued=maybe_hot_continue(state,executor,stopped,profile)
        if continued and profile.cold_checkpoint:executor.cold_checkpoint(state)
        if ((profile.r2a_identity and len(state.archive)+state.verification_requests >= profile.call_cap) or (profile.weekly_budget and len(state.archive)+state.verification_requests >= weekly_request_cap(state.response_session, profile))):
            state.terminal_reason = 'request_limit'
            break
        bundle = observe(len(state.archive), state.history, state.previous)
        if profile.weekly_budget:
            bundle['observation']['single_pick'] = single_pick_snapshot(state, executor, profile)
        if profile.identity_revision == 2:
            if profile.model_target_boxes:
                if state.identity_target:
                    state.identity_target['current_views'] = {}
                    state.identity_target['tracking_status'] = 'awaiting_current_astra_boxes'
            else:
                refresh_target_views(state, bundle, depth)
                review_shared_target(state, bundle, verification_deliver, profile, sink=sink)
            bundle['observation']['target_record'] = target_summary(state.identity_target)
        if profile.episode_memory:
            if state.episode_memory is None:raise ValueError('Episode memory missing at observation')
            bundle['observation']['episode_memory']=state.episode_memory.summary(step=executor.steps)
        wire = (build_wire(bundle, state.archive, profile, session=state.response_session)
                if profile.response_session else build_wire(bundle, state.archive, profile))
        response = deliver(wire, bundle, state.archive)
        advance(state, bundle, response, executor, depth, profile, sink=sink, verification_deliver=verification_deliver)
        if profile.cold_checkpoint and not stopped() and state.terminal_reason is None:
            executor.cold_checkpoint(state)
            if getattr(executor,'action_budget_expired',lambda:False)():
                raise TimeoutError('authorized worker wall limit at completed boundary')
    if profile.response_session and state.response_session.status == 'ready':
        state.response_session.status = 'stopped'
        state.response_session.persist()
    return state

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)

R3_SCOPE = 'c2-r3-staged-20261008'
R31_REPLAY_SCOPE = 'c2-r31-l1-replay-20261008'
R31_REPLAY_VARIANT = 'r31-heading-envelope-abort'
R31_REPLAY_FLAGS = {
    'control': dict(base_motion_abort=False, base_heading_hold=False,
                    continuous_base_speed=False, schedule_precheck=False, robot_diagnostics=False),
    'abort': dict(base_motion_abort=True, base_heading_hold=True,
                  continuous_base_speed=True, schedule_precheck=False, robot_diagnostics=False)}

R32_REPLAY_SCOPE = 'c2-r32-l1-replay-20261008'
R32_REPLAY_VARIANT = 'r32-progress-response-debt-abort'
R32_REPLAY_FLAGS = {
    arm: {flag: arm == 'candidate' and flag not in ('base_heading_hold', 'continuous_base_speed')
          for flag in ('base_motion_abort', 'base_progress_monitor', 'base_response_monitor',
                       'base_response_debt_monitor', 'base_progress_abort', 'execution_lifecycle',
                       'schedule_precheck', 'robot_diagnostics', 'base_heading_hold', 'continuous_base_speed')}
    for arm in ('control', 'candidate')}
R32_MATRIX_CONTRACT_KEYS = ('arms', 'matrices', 'resources', 'stopping', 'media',
                          'old_campaigns_unchanged', 'matrices_do_not_share_denominators')
R32_MATRIX_CONTRACT_SHA256 = 'dc3bc8387783f65bb3ba8c09ec3f99112d559191325d2e1c459e4ec397f7ea4f'
R32_L1B_EVIDENCE_SHA256 = '1c2e2ba0b8dc55a159c72df8b88c55c39d4ca468547747190d6e9f9f7818369d'
R33_REPLAY_SCOPE = 'c2-r33-l1-replay-20261008'
R33_REPLAY_VARIANT = 'r33-corroborated-response-abort'
R33_REPLAY_FLAGS = {arm:dict(values,base_response_corroboration=arm=='candidate') for arm,values in R32_REPLAY_FLAGS.items()}
R33_MATRIX_CONTRACT_SHA256 = '5f76c73f24148a3cf6abcce89c8ccc98f502b5bca11a8983740fc558e8358c1a'
R33_CORROBORATION_CONTRACT_SHA256 = '65d43c6857607927f2a15d394e93172c83d443b70ebda92824ed0cfa82246f97'


REPLAY_ACTUAL_ANCHOR_PROTOCOL = 'final-attempt-actual-anchor-comparison-v1'
REPLAY_ACTUAL_ANCHOR_MANIFEST_SHA256 = '6ef1f52c71f52188e8292d00551cfd93d8f12f4a1ff7104ecc438d6b24a8859f'


INITIALIZATION_ENV_CONFIG_FIELDS = ('env_id', 'num_envs', 'max_episode_steps', 'make_env', 'obs_mode', 'render_mode', 'shader_dir', 'sim_backend', 'continuous_task', 'cat_state', 'cat_pixels', 'frame_stack', 'stack', 'stationary_base', 'stationary_torso', 'stationary_head', 'task_plan_fp', 'spawn_data_fp', 'record_video', 'debug_video', 'debug_video_gen', 'save_video_freq', 'info_on_video', 'extra_stat_keys', 'env_kwargs')
INITIALIZATION_CONFIG_INPUT_FIELDS = ('env_id', 'num_envs', 'max_episode_steps', 'continuous_task', 'record_video', 'task_plan_fp', 'spawn_data_fp', 'env_kwargs', 'stationary_base', 'stationary_torso', 'stationary_head', 'obs_mode')


INITIALIZATION_OBSERVABILITY_STAGES = (
    'before_construction_metadata_only',
    'construction_returned_before_explicit_reset_single_snapshot',
    'explicit_reset_returned_snapshot_and_existing_returned_info',
)


R33_INIT_ONLY_SCOPE = 'c2-r33-init-only-v1'
INIT_ONLY_CONTRACT = dict(protocol='init-only-two-observations-v1', case_id='baseline-017',
    profile='C2-r3.3-far', observations=['init-a', 'init-b'], fresh_process=True,
    explicit_reset_calls=1, control_actions=0, api_requests=0, retries=0,
    strict_rtol=0, strict_atol=1e-6, reference='same-final-attempt-actual',
    continue_after_complete_strict_mismatch=True)
INIT_ONLY_CAPS = dict(worker_count=2, worker_wall_seconds=600, total_gpu_process_seconds=1200,
    cpu_postprocess_seconds=600, output_bytes=8*1024**3, api_requests=0, retries=0,
    explicit_control_steps=0)


def validate_init_only_scope(auth, profile=C2_R33_FAR, *, case_id, observation):
    """A new, separately approved diagnostic contract; no full-L1 authority transfers."""
    if (auth.get('status') != 'approved' or auth.get('scope') != R33_INIT_ONLY_SCOPE or auth.get('stage') != 'INIT'
            or auth.get('mode') != 'init-only' or profile != C2_R33_FAR
            or case_id != 'baseline-017' or auth.get('case_ids') != ['baseline-017']
            or observation not in ('init-a', 'init-b')
            or canonical(auth.get('initialization_only')) != canonical(INIT_ONLY_CONTRACT)
            or canonical(auth.get('resource_caps')) != canonical(INIT_ONLY_CAPS)):
        raise ValueError('dedicated initialization-only scope/roster/caps required')
    source = auth.get('source_sha256')
    if (not isinstance(source, dict) or len(source) != 21
            or hashlib.sha256(canonical(source)).hexdigest() != auth.get('source_manifest_sha256')):
        raise ValueError('initialization-only source21 manifest mismatch')
    option = validate_initialization_observability(auth)
    l0=r3_bound_json(next(item for item in auth['upstream_receipts'] if item.get('stage')=='L0'))
    validate_r32_l0_evidence(l0,auth)
    review=r3_bound_json(l0['evidence_bindings']['monitor_review'])
    if review.get('source_sha256')!=source:
        raise ValueError('initialization-only complete current source review required')
    if option is None or replay_initial_anchor(auth, case_id) is None:
        raise ValueError('initialization-only requires full boundary and original actual anchor evidence')
    matrix = r3_bound_json(auth['matrix_plan'])
    if matrix.get('status') != 'approved' or matrix.get('scope') != R33_INIT_ONLY_SCOPE or matrix.get('initialization_only') != INIT_ONLY_CONTRACT or matrix.get('resource_caps') != INIT_ONLY_CAPS:
        raise ValueError('initialization-only matrix contract mismatch')
    approval = r3_bound_json(auth['human_approval'])
    if (approval.get('status') != 'explicit_human_approval' or approval.get('source_kind') != 'human_message'
            or approval.get('source_role') != 'user' or approval.get('human_source_verified_by_root') is not True
            or not isinstance(approval.get('actual_user_reply'), str) or not approval['actual_user_reply'].strip()
            or approval['actual_user_reply'] != approval.get('user_text') or not approval.get('source')):
        raise ValueError('specific human initialization-only approval required')
    for evidence in (approval, r3_bound_json(auth['resource_lease']), r3_bound_json(auth['resource_ledger'])):
        if (evidence.get('scope') != R33_INIT_ONLY_SCOPE or evidence.get('batch_id') != auth.get('batch_id')
                or not auth.get('batch_id') or evidence.get('source_manifest_sha256') != auth['source_manifest_sha256']
                or evidence.get('matrix_plan_sha256') != auth['matrix_plan']['sha256']
                or evidence.get('initialization_only') != INIT_ONLY_CONTRACT or evidence.get('resource_caps') != INIT_ONLY_CAPS):
            raise ValueError('initialization-only approval/resource binding mismatch')
    lease=r3_bound_json(auth['resource_lease']);ledger=r3_bound_json(auth['resource_ledger'])
    if (lease.get('status') != 'active' or type(lease.get('expires_at_epoch')) not in (int,float)
            or not time.time() < lease['expires_at_epoch'] <= time.time()+6*3600
            or ledger.get('status') != 'open' or ledger.get('prior_allowance_transferred') != 0
            or ledger.get('jobs_started') != 0 or ledger.get('gpu_process_seconds') != 0
            or ledger.get('api_requests') != 0 or ledger.get('explicit_control_steps') != 0):
        raise ValueError('initialization-only fresh lease/ledger required')
    rows=r3_bound_json(dict(path=auth['roster_path'],sha256=auth['roster_sha256']))['dev']
    selected=[row for row in rows if row.get('case_id')==case_id]
    if len(selected)!=1 or selected[0].get('seed')!=20261006 or selected[0].get('spawn_index')!=0 or selected[0].get('spawn_source')!='official_navigate':
        raise ValueError('initialization-only original017 seed/spawn binding')
    return dict(INIT_ONLY_CONTRACT)


def validate_init_only_sequence(auth, completed, *, authorization_path=None):
    """Consume closed workers in fixed order; unknown/partial evidence never permits another."""
    if not isinstance(completed, list) or len(completed)>2:
        raise ValueError('initialization-only sequence length')
    if auth.get('scope') != R33_INIT_ONLY_SCOPE or auth.get('initialization_only') != INIT_ONLY_CONTRACT:
        raise ValueError('initialization-only sequence scope')
    if completed and (authorization_path is None or json.loads(Path(authorization_path).read_bytes())!=auth):
        raise ValueError('initialization-only current authorization file required')
    pids=set();ended=None
    for index, bindings in enumerate(completed):
        result=r3_bound_json(bindings['result']);process=r3_bound_json(bindings['process'])
        expected=INIT_ONLY_CONTRACT['observations'][index]
        if result.get('authorization_sha256')!=hashlib.sha256(Path(authorization_path).read_bytes()).hexdigest():
            raise ValueError('initialization-only actual authorization hash mismatch')
        if (result.get('scope') != R33_INIT_ONLY_SCOPE or result.get('observation') != expected
                or result.get('case_id') != 'baseline-017' or result.get('batch_id') != auth.get('batch_id')
                or result.get('status') not in ('initialization_match','initialization_mismatch')
                or result.get('explicit_reset_calls') != 1 or result.get('environment_closed') is not True
                or result.get('simulator_actions') != 0 or result.get('api_requests') != 0
                or result.get('commands') != [] or result.get('frames') != []
                or result.get('initialization_checks_eval_only') != dict(identity=True,horizon=True,official_criteria=True,controller=True,base_runtime=True,FK=True)
                or any(type(result.get(key)) is not int for key in ('explicit_reset_calls','simulator_actions','api_requests'))
                or result.get('initialization_secondary_errors_eval_only') or result.get('initialization_result_write_error_eval_only')):
            raise ValueError('initialization-only incomplete/infra/action evidence')
        primary=result.get('initialization_error_eval_only')
        if result['status']=='initialization_mismatch':
            if not isinstance(primary,dict) or primary.get('error_type')!='StrictInitializationMismatch' or primary.get('primary') is not True or primary.get('recording_secondary_error'):
                raise ValueError('initialization-only mismatch first-cause evidence')
        elif primary is not None or result.get('error_type'):
            raise ValueError('initialization-only match has failure evidence')
        if (process.get('status') != 'exited' or process.get('exit_code') != 0
                or process.get('process_group') != [] or process.get('pid') != result.get('pid')
                or type(result.get('pid')) is not int or result['pid'] in pids
                or process.get('result_sha256') != bindings['result']['sha256']):
            raise ValueError('initialization-only fresh exited process evidence required')
        if (type(result.get('started_epoch')) not in (int,float) or type(result.get('ended_epoch')) not in (int,float)
                or not 0 <= result['ended_epoch']-result['started_epoch'] <=600
                or (ended is not None and result['started_epoch'] < ended)):
            raise ValueError('initialization-only serial time binding')
        pids.add(result['pid']);ended=result['ended_epoch']
        attempt=Path(bindings['result']['path']).parent
        validate_initialization_observability_result(auth,result,attempt)
        first=json.loads((attempt/result['initialization_boundaries_eval_only'][0]['path']).read_bytes())
        metadata=json.loads((attempt/first['files']['metadata']['path']).read_bytes())
        parsed=metadata['parsed_args'];argv=metadata['actual_sys_argv']
        flag='--initialization-observation'
        if (parsed.get('mode')!='init-only' or parsed.get('initialization_observation')!=expected
                or argv.count(flag)!=1 or argv.index(flag)+1>=len(argv) or argv[argv.index(flag)+1]!=expected):
            raise ValueError('initialization-only actual mode/observation mismatch')
        binding=result.get('initialization_runtime_checks_eval_only',{})
        if binding.get('path')!='initialization-runtime-checks.eval-only.json':
            raise ValueError('initialization-only runtime numeric evidence missing')
        runtime=r3_bound_json(dict(path=str(attempt/binding['path']),sha256=binding.get('sha256')))
        from types import SimpleNamespace
        from .robot.eef_tools import RobotFK, validate_official_controller
        from .robot.motion import validate_base_runtime
        controller=SimpleNamespace(action_mapping=runtime['action_mapping'],controllers={name:SimpleNamespace(config=SimpleNamespace(**cfg)) for name,cfg in runtime['controller_configs'].items()})
        validate_official_controller(controller);validate_base_runtime(controller,runtime['control_freq'])
        q=np.asarray(runtime['qpos']);base=np.asarray(runtime['base_world']);tcp=np.asarray(runtime['tcp_world']);fk_saved=np.asarray(runtime['fk_tcp'])
        if q.shape!=(15,) or not np.isfinite(q).all() or any(x.shape!=(4,4) or not np.isfinite(x).all() for x in (base,tcp,fk_saved)):
            raise ValueError('initialization-only runtime finite shapes')
        fk=RobotFK(Path(auth['code_root'])/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',runtime['joint_names'])
        if not np.array_equal(fk.tcp(q),fk_saved) or np.max(abs(fk_saved-np.linalg.inv(base)@tcp))>1e-5:
            raise ValueError('initialization-only runtime FK mismatch')
        identity=metadata['worker_identity'];initial=result['initial_binding']
        if (runtime['uid']!=identity['uid'] or runtime['spawn_index']!=metadata['reset_options']['spawn_selection_idxs'][0]
                or runtime['steps_left']!=C2_R33_FAR.step_cap or runtime['horizon']!=C2_R33_FAR.step_cap
                or runtime['internal_pick_horizon']!=C2_R33_FAR.step_cap+1
                or runtime['robot_cumulative_force_limit']!=5000 or runtime['ee_rest_thresh']!=.05
                or runtime['qpos']!=initial['qpos'] or runtime['base_world']!=initial['base_world']):
            raise ValueError('initialization-only runtime identity/horizon/criteria')
        checks=json.loads((attempt/'baseline-reset-check.eval-only.json').read_bytes())
        diag=json.loads((attempt/'baseline-reset-diagnostics.eval-only.json').read_bytes())
        anchor=replay_initial_anchor(auth,'baseline-017')
        snapshot=diag.get('snapshot',{})
        if snapshot.get('status')!='saved' or snapshot.get('path')!='reset-state.eval-only.pt':
            raise ValueError('initialization-only strict snapshot missing')
        path=attempt/snapshot['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot.get('sha256'):
            raise ValueError('initialization-only strict snapshot changed')
        state=replay_anchor_snapshot(path);robot=np.asarray(state['articulations']['fetch'])
        if robot.shape!=(1,43) or not np.isfinite(robot).all() or not np.array_equal(robot[0,13:28],q):
            raise ValueError('initialization-only runtime qpos/state mismatch')
        reference=anchor['reference'];actual=dict(qpos=robot[0,13:28],root_pose=robot[0,:7])
        expected_values=dict(qpos=reference['qpos'],root_pose=reference['root_pose'])
        for name,value in reference['objects'].items():
            actual[name]=np.asarray(state['actors']['env-0_'+name])[0,:7];expected_values[name]=value
        if any(actual[name].shape!=np.asarray(value).shape or not np.isfinite(actual[name]).all() for name,value in expected_values.items()):
            raise ValueError('initialization-only malformed actual values')
        exact={name:bool(np.allclose(actual[name],value,rtol=0,atol=1e-6)) for name,value in expected_values.items()}
        for name,value in expected_values.items():
            field=diag.get('fields',{}).get(name,{})
            if field.get('reference')!=value or field.get('actual')!=actual[name].tolist() or field.get('residual')!=(actual[name]-value).tolist() or field.get('passed') is not exact[name]:
                raise ValueError('initialization-only diagnostic values changed')
        if (checks!=exact or set(diag.get('fields',{}))!=set(exact) or diag.get('rtol')!=0 or diag.get('atol')!=1e-6
                or result.get('strict_initialization_match') is not all(exact.values())
                or result['status']!=('initialization_match' if all(exact.values()) else 'initialization_mismatch')):
            raise ValueError('initialization-only strict classification mismatch')
    return None if len(completed)==2 else INIT_ONLY_CONTRACT['observations'][len(completed)]


def validate_initialization_observability(auth):
    """Optional evaluator-only evidence, bound to a fresh source and its L0/matrix."""
    if 'initialization_observability' not in auth:
        return None
    option = auth['initialization_observability']
    if (not isinstance(option, dict)
            or set(option) != {'protocol', 'stages', 'source_manifest_sha256'}
            or option.get('protocol') != 'init-boundary-eval-v1'
            or option.get('stages') != list(INITIALIZATION_OBSERVABILITY_STAGES)
            or not re.fullmatch(r'[0-9a-f]{64}', str(option.get('source_manifest_sha256', '')))
            or option['source_manifest_sha256'] != auth.get('source_manifest_sha256')
            or (auth.get('scope'), auth.get('stage')) not in
                ((R33_INIT_ONLY_SCOPE, 'INIT'), (R33_REPLAY_SCOPE, 'L1'), (R33_STAGED_SCOPE, 'L2'),
                 (R33_STAGED_SCOPE, 'L3'), (R33_STAGED_SCOPE, 'L4'))):
        raise ValueError('initialization observability protocol/source/stage mismatch')
    matrix = r3_bound_json(auth['matrix_plan'])
    upstream = auth.get('upstream_receipts', [])
    l0s = [b for b in upstream if isinstance(b, dict) and b.get('stage') == 'L0']
    if len(l0s) != 1:
        raise ValueError('initialization observability requires exactly one L0')
    l0 = r3_bound_json(l0s[0])
    if (l0.get('stage') != 'L0' or l0.get('status') != 'passed' or l0.get('cpu_only') is not True
            or l0.get('matrix_plan_sha256') != auth['matrix_plan']['sha256']):
        raise ValueError('initialization observability requires fresh passed CPU L0')
    for evidence in (matrix, l0):
        if (evidence.get('initialization_observability') != option
                or evidence.get('source_manifest_sha256') != auth['source_manifest_sha256']):
            raise ValueError('initialization observability missing from matrix/L0 source')
    if auth.get('scope') == R33_STAGED_SCOPE:
        l1s = [b for b in upstream if isinstance(b, dict) and b.get('stage') == 'L1']
        if len(l1s) != 1:
            raise ValueError('initialization observability requires its own L1 evidence')
        l1 = r3_bound_json(l1s[0])
        if (l1.get('initialization_observability') != option
                or l1.get('source_manifest_sha256') != auth['source_manifest_sha256']):
            raise ValueError('initialization observability L1 source/protocol mismatch')
    return dict(option)


def validate_initialization_observability_result(auth, result, attempt):
    """Verify actual worker boundary files; a top-level option alone is not evidence."""
    if 'initialization_observability' not in auth:
        if any(k in result for k in ('initialization_observability', 'initialization_capture_active', 'initialization_boundaries_eval_only')):
            raise ValueError('initialization evidence without authorization option')
        return
    if (result.get('initialization_capture_active') is not True
            or result.get('initialization_observability') != auth['initialization_observability']
            or result.get('source_manifest_sha256') != auth['source_manifest_sha256']):
        raise ValueError('initialization worker option/source mismatch')
    bindings = result.get('initialization_boundaries_eval_only')
    if (not isinstance(bindings, list) or len(bindings) != 3
            or [b.get('stage') for b in bindings if isinstance(b, dict)] != list(INITIALIZATION_OBSERVABILITY_STAGES)):
        raise ValueError('initialization worker boundaries incomplete')
    root = Path(attempt).resolve()
    def local(binding):
        if (not isinstance(binding, dict) or not isinstance(binding.get('path'), str)
                or Path(binding['path']).name != binding['path'] or '\\' in binding['path']):
            raise ValueError('initialization boundary path must be local filename')
        path = (root/binding['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != binding.get('sha256'):
            raise ValueError('initialization boundary file changed')
        return path
    for index, binding in enumerate(bindings):
        receipt = json.loads(local(binding).read_bytes())
        stage = INITIALIZATION_OBSERVABILITY_STAGES[index]
        if (receipt.get('stage') != stage or receipt.get('status') != 'capture_complete'
                or receipt.get('evaluation_only') is not True):
            raise ValueError('initialization boundary not complete')
        marker = json.loads(local(receipt.get('marker')).read_bytes())
        if marker != {'stage': stage, 'status': 'capture_started', 'evaluation_only': True}:
            raise ValueError('initialization boundary start marker mismatch')
        expected = ({'metadata'}, {'state'}, {'state', 'info'})[index]
        if not isinstance(receipt.get('files'), dict) or set(receipt['files']) != expected:
            raise ValueError('initialization boundary contents incomplete')
        for file_binding in receipt['files'].values():
            local(file_binding)
        if 'info' in receipt['files']:
            info = json.loads(local(receipt['files']['info']).read_bytes())
            if not isinstance(info, dict) or not {'subtasks_steps_left', 'robot_cumulative_force'} <= set(info):
                raise ValueError('initialization existing reset info incomplete')
            def finite_info(value):
                if isinstance(value, dict):
                    return all(finite_info(v) for v in value.values())
                if isinstance(value, list):
                    return all(finite_info(v) for v in value)
                return not isinstance(value, float) or math.isfinite(value)
            if not finite_info(info):
                raise ValueError('initialization existing reset info nonfinite')
            for key in ('subtasks_steps_left', 'robot_cumulative_force'):
                value = np.asarray(info[key])
                if value.size == 0 or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
                    raise ValueError('initialization existing reset info field invalid')
        if 'state' in receipt['files']:
            state = replay_anchor_snapshot(local(receipt['files']['state']))
            required_state = {'actors', 'articulations', 'task_plan_idxs', 'build_config_idxs',
                              'init_config_idxs', 'subtask_pointer', 'subtask_steps_left', 'robot_cumulative_force'}
            if (not isinstance(state, dict) or not required_state <= set(state)
                    or any(not isinstance(state[k], dict) or not state[k] for k in ('actors', 'articulations'))):
                raise ValueError('initialization snapshot exported state incomplete')
            def numeric_tree(value):
                if isinstance(value, dict):
                    return bool(value) and all(isinstance(k,str) and numeric_tree(v) for k,v in value.items())
                if isinstance(value, (list, tuple)):
                    return bool(value) and all(numeric_tree(v) for v in value)
                return isinstance(value, np.ndarray) and value.size > 0 and value.dtype.kind in 'biuf' and bool(np.isfinite(value).all())
            if not numeric_tree(state):
                raise ValueError('initialization snapshot nonnumeric or nonfinite export')
            if any(v.ndim != 2 or v.shape != (1,13) for v in state['actors'].values()):
                raise ValueError('initialization actor snapshot shape')
            if any(v.ndim != 2 or v.shape[0] != 1 or v.shape[1] < 13 for v in state['articulations'].values()):
                raise ValueError('initialization articulation snapshot shape')

        if index == 0:
            metadata = json.loads(local(receipt['files']['metadata']).read_bytes())
            if (metadata.get('source_manifest_sha256') != auth['source_manifest_sha256']
                    or metadata.get('protocol') != auth['initialization_observability']['protocol']):
                raise ValueError('initialization metadata source/protocol mismatch')
            args = metadata.get('parsed_args')
            cfg, final = metadata.get('final_config_input'), metadata.get('env_config')
            identity = metadata.get('worker_identity', {})
            initial = result.get('initial_binding', {})
            argv = metadata.get('actual_sys_argv')
            if (metadata.get('evaluation_only') is not True or not isinstance(args, dict)
                    or not {'lab_root', 'code', 'output', 'authorization', 'case_id', 'profile', 'condition', 'mode'} <= set(args)
                    or not isinstance(cfg, dict) or not cfg or not isinstance(final, dict) or not final
                    or metadata.get('env_config_dataclass') != 'mshab.envs.make.EnvConfig'
                    or set(final) != set(INITIALIZATION_ENV_CONFIG_FIELDS)
                    or not set(INITIALIZATION_CONFIG_INPUT_FIELDS) <= set(cfg)
                    or not isinstance(cfg.get('env_kwargs'), dict)
                    or hashlib.sha256(canonical(final)).hexdigest() != metadata.get('env_config_sha256')
                    or any(k not in final or final[k] != v for k,v in cfg.items())
                    or not isinstance(argv, list) or not argv or not all(isinstance(v,str) for v in argv) or not argv[0]
                    or not initial or metadata.get('seed') != initial.get('seed')
                    or metadata.get('reset_options') != {'reconfigure': True, 'spawn_selection_idxs': [initial.get('spawn_index')]}
                    or identity.get('case_id') != result.get('case_id') or identity.get('uid') != initial.get('uid')
                    or args.get('case_id') != result.get('case_id') or args.get('profile') != result.get('information_condition')
                    or args.get('condition') != result.get('condition')
                    or str(args.get('output','')).replace('\\', '/').rstrip('/').rsplit('/',1)[-1] != root.name
                    or not isinstance(metadata.get('authorization'), dict)
                    or metadata['authorization'].get('sha256') != result.get('authorization_sha256')
                    or metadata['authorization'].get('path') != args.get('authorization')):
                raise ValueError('initialization metadata configuration/identity incomplete')
            for key in ('case_id', 'profile', 'authorization', 'output', 'mode'):
                flag = '--'+key.replace('_', '-')
                if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv) or argv[argv.index(flag)+1] != str(args.get(key)):
                    raise ValueError('initialization metadata actual argv mismatch')
            if args['mode'] == 'l1-replay' and (args.get('replay_arm') != result.get('replay_arm') or args.get('replay_turn') != result.get('replay_turn')):
                raise ValueError('initialization metadata replay identity mismatch')
            required_modules = {
                'mshab.envs.make': 'src/official-mshab-runtime/mshab/mshab/envs/make.py',
                'mshab.envs.pick': 'src/official-mshab-runtime/mshab/mshab/envs/pick.py',
                'mshab.envs.subtask': 'src/official-mshab-runtime/mshab/mshab/envs/subtask.py',
                'mshab.envs.sequential_task': 'src/official-mshab-runtime/mshab/mshab/envs/sequential_task.py',
                'mani_skill.envs.sapien_env': 'src/official-mshab-runtime/ManiSkill/mani_skill/envs/sapien_env.py',
                'mani_skill.utils.scene_builder.replicacad.rearrange.scene_builder': 'src/official-mshab-runtime/ManiSkill/mani_skill/utils/scene_builder/replicacad/rearrange/scene_builder.py',
            }
            installed = metadata.get('installed_modules', {})
            if not isinstance(installed, dict) or set(installed) != set(required_modules):
                raise ValueError('initialization metadata installed module set incomplete')
            def portable(path):
                return str(path).replace('\\', '/').rstrip('/')
            def joined(base, name):
                return portable(base)+'/'+name
            def verify_bound(item, required_key=None):
                if not isinstance(item, dict) or set(item) != {'path', 'key', 'sha256'}:
                    raise ValueError('initialization metadata asset binding incomplete')
                key = item['key']
                if required_key is not None and key != required_key:
                    raise ValueError('initialization metadata module asset key mismatch')
                for root_key, manifest in (('lab_root','asset_sha256'), ('code','source_sha256')):
                    if (required_key is not None and manifest != 'asset_sha256'):
                        continue
                    if (auth.get(manifest, {}).get(key) == item['sha256']
                            and portable(item['path']) == joined(args[root_key], key)):
                        return
                raise ValueError('initialization metadata asset not frozen')
            for name,key in required_modules.items():
                verify_bound(installed[name], key)
            files = metadata.get('files', {})
            required_files = {'plan', 'spawn', 'checkpoint_config'}
            if identity.get('spawn_source') == 'official_navigate':
                required_files |= {'navigate_plan', 'navigate_spawn'}
            elif identity.get('spawn_source') != 'official_pick':
                raise ValueError('initialization metadata spawn source invalid')
            if not isinstance(files,dict) or set(files) != required_files:
                raise ValueError('initialization metadata source files incomplete')
            for item in files.values():
                verify_bound(item)
            split, category, plan_name = identity.get('dataset_split'), identity.get('object_category'), identity.get('plan_filename')
            if split not in ('train', 'val') or not isinstance(category,str) or not category or '/' in category or '\\' in category or plan_name not in ('all.json', category+'.json'):
                raise ValueError('initialization metadata task source identity incomplete')
            roles = {'plan': f'/task_plans/tidy_house/pick/{split}/{plan_name}',
                     'spawn': f'/spawn_data/tidy_house/pick/{split}/spawn_data.pt',
                     'checkpoint_config': f'/rl/tidy_house/pick/{category}/config.yml'}
            if identity['spawn_source'] == 'official_navigate':
                roles.update(navigate_plan=f'/task_plans/tidy_house/navigate/{split}/all.json',
                             navigate_spawn=f'/spawn_data/tidy_house/navigate/{split}/spawn_data.pt')
            if any(not portable(files[label]['path']).endswith(suffix) for label,suffix in roles.items()):
                raise ValueError('initialization metadata source file roles mismatch')

            derived = metadata.get('derived_files', {})
            expected_derived = {'single_plan': 'single-plan.json'}
            if identity['spawn_source'] == 'official_navigate':
                expected_derived['adapted_spawn'] = 'official-navigate-pick-spawn.pt'
            if not isinstance(derived,dict) or set(derived) != set(expected_derived):
                raise ValueError('initialization metadata derived inputs incomplete')
            for name,filename in expected_derived.items():
                item = derived[name]
                if (not isinstance(item,dict) or portable(item.get('path')) != joined(args['output'], filename)
                        or not item.get('provenance') or hashlib.sha256((root/filename).read_bytes()).hexdigest() != item.get('sha256')):
                    raise ValueError('initialization metadata derived input changed')
            expected_spawn = derived.get('adapted_spawn', files['spawn'])['path']
            if cfg.get('task_plan_fp') != derived['single_plan']['path'] or cfg.get('spawn_data_fp') != expected_spawn:
                raise ValueError('initialization metadata config source paths changed')



def replay_anchor_snapshot(path):
    """Decode archived numpy arrays only; never construct an environment or restore state."""
    import pickle
    import zipfile
    import codecs
    import importlib
    modules = {}
    def array_module(suffix):
        if suffix not in modules:
            modern = 'numpy._core.'+suffix
            try:
                modules[suffix] = importlib.import_module(modern)
            except ModuleNotFoundError as error:
                # NumPy 1.26 may expose _core without every compatibility submodule.
                # Only that missing module permits fallback; dependency/import failures propagate.
                if error.name not in ('numpy._core', modern):
                    raise
                modules[suffix] = importlib.import_module('numpy.core.'+suffix)
        return modules[suffix]
    class ArraysOnly(pickle.Unpickler):
        def find_class(self, module, name):
            allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype, ('_codecs', 'encode'): codecs.encode}
            for prefix in ('numpy.core.multiarray', 'numpy._core.multiarray'):
                allowed[(prefix, '_reconstruct')] = ('multiarray', '_reconstruct')
                allowed[(prefix, 'scalar')] = ('multiarray', 'scalar')
            for prefix in ('numpy.core.numeric', 'numpy._core.numeric'):
                allowed[(prefix, '_frombuffer')] = ('numeric', '_frombuffer')
            if (module, name) not in allowed:
                raise ValueError('anchor snapshot contains a non-array pickle global')
            value = allowed[(module, name)]
            return getattr(array_module(value[0]), value[1]) if isinstance(value, tuple) else value
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(members) != 1:
            raise ValueError('anchor snapshot archive layout')
        state = ArraysOnly(BytesIO(archive.read(members[0]))).load()
    return state


def replay_initial_anchor(auth, case_id=None):
    """Validate evaluator-only final-attempt provenance; absent option preserves legacy."""
    if 'replay_initial_anchor' not in auth:
        return None
    option = auth['replay_initial_anchor']
    if (auth.get('scope') not in (R33_REPLAY_SCOPE, R33_STAGED_SCOPE, R33_INIT_ONLY_SCOPE)
            or not isinstance(option, dict) or set(option) != {'protocol', 'manifest'}
            or option['protocol'] != REPLAY_ACTUAL_ANCHOR_PROTOCOL):
        raise ValueError('actual anchor requires explicit new r33 protocol')
    root = Path(auth.get('evidence_root', auth['code_root'])).resolve()
    def bound(binding):
        if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
            raise ValueError('anchor file binding schema')
        raw = binding['path']
        if not isinstance(raw, str) or '\\' in raw or Path(raw).is_absolute() or '..' in Path(raw).parts:
            raise ValueError('anchor file path escapes evidence root')
        path = (root/raw).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('anchor file changed or missing')
        return path
    # The top-level manifest may be absolute for deployment, but must remain inside evidence_root.
    manifest_binding = option.get('manifest')
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {'path', 'sha256'}:
        raise ValueError('anchor manifest binding schema')
    if manifest_binding['sha256'] != REPLAY_ACTUAL_ANCHOR_MANIFEST_SHA256:
        raise ValueError('anchor immutable final20/32 manifest changed')
    mp = Path(manifest_binding['path'])
    mp = mp if mp.is_absolute() else root/mp
    if not mp.resolve().is_relative_to(root):
        raise ValueError('anchor manifest outside evidence root')
    manifest = r3_bound_json(dict(path=str(mp), sha256=manifest_binding['sha256']))
    if (set(manifest) != {'protocol','mode','rtol','atol','original20_evidence_sha256','cases','command_links'}
            or manifest['protocol'] != option['protocol'] or manifest['mode'] != 'natural_reset_comparison_only'
            or type(manifest['rtol']) is not int or manifest['rtol'] != 0
            or type(manifest['atol']) is not float or manifest['atol'] != 1e-6
            or manifest['original20_evidence_sha256'] != R32_L1B_EVIDENCE_SHA256):
        raise ValueError('anchor manifest contract changed')
    if auth['l1b_sources_and_scores']['sha256'] != R32_L1B_EVIDENCE_SHA256:
        raise ValueError('anchor original20 provenance changed')
    evidence = r3_bound_json(auth['l1b_sources_and_scores'])
    original = evidence['original20_final_attempts']
    rows = manifest['cases']
    if (not isinstance(rows, list) or len(rows) != 20 or
            [r.get('case_id') for r in rows] != ['baseline-'+r['case'] for r in original]):
        raise ValueError('anchor requires ordered complete final20')
    sources = {}; selected = None
    def finite_vector(value, length):
        if not isinstance(value, list) or len(value) != length or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):
            raise ValueError('anchor nonfinite or invalid vector shape')
        return value
    for row, orig in zip(rows, original):
        if (set(row) != {'case_id','attempt_index','result','state','nominal','reference'}
                or type(row['attempt_index']) is not int or row['attempt_index'] != orig['attempt_index']
                or row['result'] != {'path':orig['path'], 'sha256':orig['sha256']}):
            raise ValueError('anchor final attempt/result mismatch')
        source = json.loads(bound(row['result']).read_text(encoding='utf8'))
        expected_state = str(Path(orig['path']).parent/'trajectory/step-0000/state.eval-only.pt').replace('\\','/')
        if row['state']['path'] != expected_state:
            raise ValueError('anchor must use same final attempt step0000')
        state = replay_anchor_snapshot(bound(row['state']))
        nominal = json.loads(bound(row['nominal']).read_text(encoding='utf8'))
        case = row['case_id']; original_nominal = auth.get('initial_binding_sha256', {})
        # Nominal provenance is the originally authorized binding, not a caller-chosen key set.
        hashes = [h for path,h in original_nominal.items() if Path(path).name == case.removeprefix('baseline-')+'.json']
        if len(hashes) != 1 or row['nominal']['sha256'] != hashes[0]:
            raise ValueError('anchor nominal source binding mismatch')
        reference_path = bound(row['reference'])
        reference = json.loads(reference_path.read_text(encoding='utf8'))
        if set(reference) != {'qpos','root_pose','objects'} or not isinstance(reference['objects'],dict) or set(reference['objects']) != set(nominal['objects']):
            raise ValueError('anchor actor key set changed')
        robot = np.asarray(state['articulations']['fetch'])
        if robot.shape != (1,43) or not np.isfinite(robot).all():
            raise ValueError('anchor fetch state layout/nonfinite')
        expected = dict(qpos=robot[0,13:28].tolist(), root_pose=robot[0,:7].tolist(), objects={})
        finite_vector(reference['qpos'],15); finite_vector(reference['root_pose'],7)
        for name in nominal['objects']:
            actor = np.asarray(state['actors'].get('env-0_'+name, []))
            if actor.shape != (1,13) or not np.isfinite(actor).all():
                raise ValueError('anchor actor missing/shape/nonfinite')
            finite_vector(reference['objects'][name],7)
            expected['objects'][name] = actor[0,:7].tolist()
        if reference != expected or reference['qpos'] != source['initial_robot_state']['qpos']:
            raise ValueError('anchor reference is not exact actual step0 extraction')
        sources[case] = source
        if case == case_id:
            selected = dict(reference_path=str(reference_path), reference=reference, nominal=nominal,
                binding=dict(protocol=option['protocol'], manifest_sha256=manifest_binding['sha256'],
                    case_id=case, attempt_index=row['attempt_index'], result_sha256=row['result']['sha256'],
                    state_sha256=row['state']['sha256'], reference_sha256=row['reference']['sha256'], nominal_sha256=row['nominal']['sha256']))
    a = r3_bound_json(dict(path=auth['replay_roster_path'],sha256=auth['replay_roster_sha256']))['roster']
    b = [r for r in evidence['rows'] if r['scope']=='L1-B_proposed_pair']
    links = manifest['command_links']
    expected_ids = [('L1-A','baseline-'+r['case'],r['turn']) for r in a]+[('L1-B','baseline-'+r['case'],r['turn']) for r in b]
    if not isinstance(links,list) or [(x.get('group'),x.get('case_id'),x.get('turn')) for x in links] != expected_ids:
        raise ValueError('anchor original32 command mapping changed')
    for link,old in zip(links,a+b):
        case = link['case_id']; source = sources[case]; row = rows[int(case.removeprefix('baseline-'))]
        commands = [c for c in source['commands'] if c['turn']==link['turn']]
        if len(commands)!=1: raise ValueError('anchor command ambiguous')
        command = commands[0]; start=command['start_step']; end=start+command['result']['steps_used']
        frames={f['step']:f for f in source['frames']}
        def segment(first,last):
            payload=[dict(step=i,action=frames[i]['action']) for i in range(first,last+1)]
            return dict(first_step=first,last_step=last,count=len(payload),canonical_step_action_sha256=hashlib.sha256(canonical(payload)).hexdigest())
        snap = dict(path=old['precommand_snapshot'],sha256=old['snapshot_sha256']) if link['group']=='L1-A' else {k:old['snapshots']['precommand'][k] for k in ('path','sha256')}
        expected_link=dict(group=link['group'],case_id=case,turn=command['turn'],attempt_index=row['attempt_index'],result_sha256=row['result']['sha256'],
            command_sha256=hashlib.sha256(canonical(command)).hexdigest(),prefix=segment(1,start),suffix=segment(end+1,max(frames)),precommand_snapshot=snap)
        if link != expected_link or command['call'] != (old['command'] if link['group']=='L1-A' else old['original_command']['call']):
            raise ValueError('anchor original command/prefix/suffix mismatch')
        bound(snap)
    if case_id is not None and selected is None:raise ValueError('anchor case missing')
    return selected if case_id is not None else manifest


def validate_replay_anchor_receipts(auth, matrix, l0):
    """Old receipts cannot acquire the new protocol merely by changing authorization."""
    if not (auth.get('initialization_observability') == matrix.get('initialization_observability') == l0.get('initialization_observability')):
        raise ValueError('initialization observability requires symmetric auth/matrix/L0 markers')
    option=auth.get('replay_initial_anchor')
    if 'replay_initial_anchor' in auth:
        replay_initial_anchor(auth)
        if matrix.get('replay_initial_anchor') != option or l0.get('replay_initial_anchor') != option:
            raise ValueError('anchor requires new same-protocol matrix and L0')
    elif 'replay_initial_anchor' in matrix or 'replay_initial_anchor' in l0:
        raise ValueError('nominal auth cannot borrow actual-anchor evidence')


def validate_r32_replay_scope(auth, profile=None, *, case_id=None):
    """Validate a separately approved 28+4, two-arm L1; never provider permission."""
    if 'initialization_observability' in auth:
        validate_initialization_observability(auth)
    r33=auth.get('scope')==R33_REPLAY_SCOPE
    scope,revision,variant,expected_flags,expected_profiles,matrix_hash = ((R33_REPLAY_SCOPE,'C2-r3.3',R33_REPLAY_VARIANT,R33_REPLAY_FLAGS,(C2_R33,C2_R33_FAR),R33_MATRIX_CONTRACT_SHA256) if r33 else (R32_REPLAY_SCOPE,'C2-r3.2',R32_REPLAY_VARIANT,R32_REPLAY_FLAGS,(C2_R32,C2_R32_FAR),R32_MATRIX_CONTRACT_SHA256))
    if profile is None:profile=expected_profiles[0]
    if (auth.get('scope') != scope or auth.get('stage') != 'L1'
            or auth.get('system_revision') != revision
            or auth.get('replay_variant') != variant
            or profile not in expected_profiles):
        raise ValueError('r32 requires its fresh L1 replay-only scope')
    if (auth.get('status') != 'authorized' or auth.get('approval_status') != 'APPROVED'
            or auth.get('explicit_user_approval') is not True
            or not isinstance(auth.get('authorization_evidence'), str)
            or not auth['authorization_evidence'].strip()
            or type(auth.get('expires_at_epoch')) not in (int, float)
            or not math.isfinite(auth['expires_at_epoch']) or time.time() >= auth['expires_at_epoch']):
        raise ValueError('r32 explicit fresh approval and lease required')
    if not re.fullmatch(('c2-r33' if r33 else 'c2-r32')+r'-l1-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id', ''))):
        raise ValueError('r32 new batch namespace required')
    for key in ('automatic_retries', 'automatic_recoveries', 'prior_budget_transferred', 'provider_requests_cap'):
        if type(auth.get(key)) is not int or auth[key] != 0:
            raise ValueError('r32 forbids requests, retries and closed-budget transfer')
    if any(auth.get(key) is not None for key in ('infrastructure_retry', 'recovery_budget', 'recovery_attempt_id', 'resume_batch_id')):
        raise ValueError('r32 forbids recovery or closed-batch continuation')
    fixed = dict(usd_cap='0', currency='USD', campaign_usd_cap='364', condition='V-mobile',
                 cumulative_force_to_model=False, recorded_action_replay=True,
                 model=MODEL, reasoning='medium', per_case_verification_cap=5, detailed_trajectory=True)
    if any(auth.get(k) != v or type(auth.get(k)) is not type(v) for k,v in fixed.items()):
        raise ValueError('r32 frozen replay-only contract changed')
    limits = dict(resets_max=64, physical_steps_cap=70400, worker_seconds_cap=600,
                  gpu1_process_seconds_cap=12000, cpu_media_seconds_cap=7200, output_bytes_cap=64*1024**3)
    if any(type(auth.get(k)) is not int or auth[k] != v for k,v in limits.items()):
        raise ValueError('r32 combined resource limits changed')
    flags = auth.get('effective_controller_flags_exact')
    if (flags != expected_flags or any(type(v) is not bool for row in flags.values() for v in row.values())
            or auth.get('replay_variant_sha256') != hashlib.sha256(canonical(flags)).hexdigest()):
        raise ValueError('r32 effective arm flags changed')
    cases = [f'baseline-{i:03d}' for i in range(20)]
    profiles = {case: revision+'-far' if i >= 10 else revision for i,case in enumerate(cases)}
    if auth.get('case_ids') != cases or auth.get('case_profiles') != profiles:
        raise ValueError('r32 frozen near/far profiles changed')
    if case_id is not None and (case_id not in cases or profiles[case_id] != profile.name):
        raise ValueError('r32 per-case profile mismatch')
    source = auth.get('source_sha256')
    package = Path(__file__).parent
    required = {'src/astra_pick/'+p.relative_to(package).as_posix() for p in package.rglob('*')
                if p.is_file() and p.suffix in ('.py', '.md', '.json')}
    required |= {'tests/test_astra_pick_'+label+'.py' for label in ('contract','parity','measure')}
    required |= {'.runtime/r32_l1_supervisor.py', '.runtime/r3_l1_supervisor.py', '.runtime/r3_finalize_replay.py',
                 'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',
                 'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/response-model.json'}
    if not isinstance(source, dict) or not required <= set(source) or hashlib.sha256(canonical(source)).hexdigest() != auth.get('source_manifest_sha256'):
        raise ValueError('r32 complete source20 binding required')
    code_root = Path(auth['code_root']).resolve()
    for name,digest in source.items():
        path = (code_root/name).resolve()
        if not path.is_relative_to(code_root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('r32 frozen source changed')
    for path in package.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.md', '.json'):
            if hashlib.sha256(path.read_bytes()).hexdigest() != source.get('src/astra_pick/'+path.relative_to(package).as_posix()):
                raise ValueError('r32 running package differs from frozen source')
    frozen = {'replay_roster':'09e1d119565bee3fd54c58735d0079ec07b8f404c0499414077fe110b1b640ea',
        'roster':'33dbc4dbad4a270c23eda2dec642736fe7c132d653e66f2c49aefda8e4c6fcdc',
        'controller_reconstruction':'d5a9c6b68f757a79e18c93efa1d67cb85b0261253eb7094226ebcada10a54fc3',
        'asset_manifest':'2bbb2f9d9e5d9ead7284de8664e0dbd80ff51acf8f6a4e98bf68fc26c9f4c07a'}
    bound = {}
    for key,digest in frozen.items():
        if auth.get(key+'_sha256') != digest:
            raise ValueError('r32 original '+key+' changed')
        bound[key] = r3_bound_json(dict(path=auth[key+'_path'], sha256=digest))
    if (not isinstance(auth.get('asset_sha256'), dict)
            or hashlib.sha256(canonical(auth['asset_sha256'])).hexdigest() != '90c009efad671635f28413f54dffedc8dbcfc0b5abb5ae2f23ec4aa06db6a783'
            or Path(auth['asset_root']).resolve() != Path(bound['asset_manifest']['asset_root']).resolve()):
        raise ValueError('r32 frozen asset binding changed')
    matrix = r3_bound_json(auth['matrix_plan'])
    if hashlib.sha256(canonical({k:matrix.get(k) for k in R32_MATRIX_CONTRACT_KEYS})).hexdigest() != matrix_hash:
        raise ValueError('r32 original A28 and separate B4 matrix/score contracts changed')
    if matrix.get('source_manifest_sha256') != auth['source_manifest_sha256']:
        raise ValueError('r32 matrix is not bound to this frozen source')
    evidence_binding = auth['l1b_sources_and_scores']
    if evidence_binding.get('sha256') != R32_L1B_EVIDENCE_SHA256:
        raise ValueError('r32 original20 and L1-B source/score evidence changed')
    evidence = r3_bound_json(evidence_binding)
    evidence_root = Path(auth.get('evidence_root', auth['code_root'])).resolve()
    for item in evidence['original20_final_attempts']:
        path = (evidence_root/item['path']).resolve()
        if not path.is_relative_to(evidence_root) or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('r32 original20 source bytes changed')
    upstream = auth.get('upstream_receipts')
    if not isinstance(upstream, list) or len(upstream) != 1 or upstream[0].get('stage') != 'L0':
        raise ValueError('r32 fresh L0 evidence required')
    l0 = r3_bound_json(upstream[0])
    for key in ('scope', 'system_revision', 'replay_variant', 'replay_variant_sha256', 'source_manifest_sha256'):
        if l0.get(key) != auth.get(key):
            raise ValueError('r32 L0 source or revision mismatch')
    if (l0.get('stage') != 'L0' or l0.get('status') != 'passed' or l0.get('cpu_only') is not True
            or any(type(l0.get(k)) is not int or l0[k] != 0 for k in ('api_requests', 'physics_steps'))
            or l0.get('matrix_plan_sha256') != auth['matrix_plan']['sha256']
            or l0.get('l1b_sources_and_scores_sha256') != R32_L1B_EVIDENCE_SHA256
            or not isinstance(l0.get('gates'), dict) or not l0['gates'] or any(v is not True for v in l0['gates'].values())):
        raise ValueError('r32 L0 gates or concrete matrix binding incomplete')
    validate_replay_anchor_receipts(auth, matrix, l0)
    validate_r32_l0_evidence(l0, auth)


def validate_r32_l0_evidence(l0, auth):
    """Read actual CPU evidence; a dictionary of passing booleans is insufficient."""
    bindings = l0.get('evidence_bindings')
    if not isinstance(bindings, dict) or set(bindings) != {'controller', 'wire', 'negative', 'monitor_review', 'tests'}:
        raise ValueError('r32 L0 actual evidence bindings required')
    controller = r3_bound_json(bindings['controller'])
    if (controller.get('status') != 'passed' or controller.get('source_manifest_sha256') != auth['source_manifest_sha256']
            or any(controller.get(k) != 28 for k in ('commands','historical_actions_exact',
                'historical_public_feedback_exact','same_host_actions_exact','same_host_native_feedback_exact'))):
        raise ValueError('r32 L0 original-host 28 exact receipt required')
    actions = r3_bound_json(bindings['controller']['actions'])
    if (controller.get('report_sha256') != bindings['controller']['actions']['sha256']
            or actions.get('test_source_sha256') != auth['source_sha256']['tests/test_astra_pick_measure.py']
            or len(actions.get('rows', [])) != 28
            or any(row.get('same_host_actions_exact') is not True or row.get('same_host_full_native_feedback_exact') is not True
                or len(row.get('checks', [])) != 2
                or any(check.get('actions_exact') is not True or check.get('public_feedback_exact') is not True for check in row['checks'])
                for row in actions['rows'])):
        raise ValueError('r32 L0 controller rows or source incomplete')
    l1b = r3_bound_json(bindings['controller']['l1b'])
    if (controller.get('l1b_report_sha256') != bindings['controller']['l1b']['sha256']
            or l1b.get('status') != 'passed' or l1b.get('proposal_sha256') != R32_L1B_EVIDENCE_SHA256
            or l1b.get('source_before') != l1b.get('source_after')
            or any(auth['source_sha256'].get(k) != v for k,v in l1b.get('source_before', {}).items())
            or len(l1b.get('source_before', {})) != 12):
        raise ValueError('r32 L0 L1-B source receipt mismatch')
    expected = [('000',19),('005',16),('006',21),('009',20),('009',21)]
    if ([(r.get('case'),r.get('turn')) for r in l1b.get('rows', [])] != expected
            or any(row.get('same_host_actions_exact') is not True or row.get('same_host_full_native_feedback_exact') is not True
                or len(row.get('checks', [])) != 2
                or any(c.get('historical_actions_exact') is not True or c.get('historical_public_feedback_exact') is not True or c.get('error') is not None for c in row['checks'])
                for row in l1b['rows'])):
        raise ValueError('r32 L0 B4 and separate009/21 exact rows required')
    for name,count in (('L1-B',4),('009-21-separate',1)):
        if any(l1b.get('summary', {}).get(name, {}).get(k) != count for k in ('commands','same_host_actions_exact',
                'same_host_full_native_feedback_exact','historical_actions_exact','historical_public_feedback_exact')):
            raise ValueError('r32 L0 L1-B separate denominator mismatch')
    wire = r3_bound_json(bindings['wire'])
    if (bindings['wire'].get('source_manifest_sha256') != auth['source_manifest_sha256']
            or wire.get('count') != 427 or len(wire.get('rows', [])) != 427
            or len({(r.get('case'),r.get('turn')) for r in wire['rows']}) != 427
            or any(r.get(k) is not True for r in wire['rows'] for k in ('old_wire_exact','frozen_wire_exact',
                'same_host_frozen_candidate_exact','content_exact','images_exact','prefix_exact','non_layout_request_fields_exact'))):
        raise ValueError('r32 L0 427 exact request rows required')
    from .robot.motion import BaseProgressMonitor, BaseResponseMonitor, BaseResponseDebtMonitor
    for name,model in (('response',BaseResponseMonitor),('debt',BaseResponseDebtMonitor)):
        negative = r3_bound_json(bindings['negative'][name])
        contract = negative.get('contracts', {}).get('response') if name == 'response' else negative.get('contract')
        if (negative.get('negative_commands') != 107 or negative.get('negative_triggers') != 0
                or contract != model.CONTRACT
                or (name == 'response' and negative.get('contracts', {}).get('progress') != BaseProgressMonitor.CONTRACT)):
            raise ValueError('r32 L0 negative contract or coverage mismatch')
    if auth.get('scope') in (R33_REPLAY_SCOPE,R33_STAGED_SCOPE,R33_INIT_ONLY_SCOPE):
        corroboration=r3_bound_json(bindings['negative']['corroboration'])
        if (corroboration.get('negative_commands')!=107 or corroboration.get('negative_triggers')!=0
                or corroboration.get('contract_sha256')!=R33_CORROBORATION_CONTRACT_SHA256
                or corroboration.get('motion_source_sha256')!=auth['source_sha256']['src/astra_pick/robot/motion.py']):
            raise ValueError('r33 current corroboration negative evidence required')
    review = r3_bound_json(bindings['monitor_review'])
    if (review.get('status') != 'no_blocking_implementation_findings'
            or review.get('findings') != []
            or review.get('source_sha256', {}).get('src/astra_pick/robot/motion.py') != auth['source_sha256']['src/astra_pick/robot/motion.py']):
        raise ValueError('r32 L0 current monitor review required')
    tests = r3_bound_json(bindings['tests'])
    if tests.get('source_manifest_sha256') != auth['source_manifest_sha256']:
        raise ValueError('r32 L0 test source mismatch')
    collected = tests.get('collected_tests')
    if (not isinstance(collected, list) or not collected or any(not isinstance(x,list) or len(x)!=2 or any(not isinstance(y,str) or not y for y in x) for x in collected)
            or len({tuple(x) for x in collected}) != len(collected)
            or tests.get('collected_tests_sha256') != hashlib.sha256(canonical(collected)).hexdigest()):
        raise ValueError('r32 L0 unique collected test inventory required')
    import xml.etree.ElementTree as ET
    passed = set()
    for binding in tests.get('runs', []):
        raw = Path(binding['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding.get('sha256'):
            raise ValueError('r32 L0 test XML changed')
        for node in ET.fromstring(raw).iter('testcase'):
            if node.find('failure') is None and node.find('error') is None and node.find('skipped') is None:
                passed.add((node.get('classname'),node.get('name')))
    if (not set(map(tuple,collected)) <= passed or any(not any('test_astra_pick_'+label in item[0] for item in collected) for label in ('contract','parity','measure'))):
        raise ValueError('r32 L0 unresolved tests or missing candidate module')
    import ast
    for label in ('contract','parity','measure'):
        module = 'test_astra_pick_'+label
        source = (Path(auth['code_root'])/'tests'/(module+'.py')).read_text(encoding='utf8')
        required = {node.name for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test_')}
        observed = {name.split('[',1)[0] for classname,name in collected if module in classname}
        if not required <= observed:
            raise ValueError('r32 L0 collection omits source test functions')


def validate_r31_replay_scope(auth, profile=C2_R31, *, case_id=None):
    """Fresh L1 component permission only; never inherits the failed r3 run."""
    if (auth.get('scope') != R31_REPLAY_SCOPE or auth.get('stage') != 'L1'
            or auth.get('system_revision') != 'C2-r3.1'
            or auth.get('replay_variant') != R31_REPLAY_VARIANT
            or profile not in (C2_R31, C2_R31_FAR)):
        raise ValueError('r31 is CPU-only without its fresh L1 component scope')
    if any(type(auth.get(k)) is not int or auth[k] != 0 for k in
           ('automatic_retries','automatic_recoveries','prior_budget_transferred')):
        raise ValueError('r31 forbids automatic retries, recovery and prior budget transfer')
    flags = auth.get('effective_controller_flags_exact')
    if (flags != R31_REPLAY_FLAGS or not isinstance(flags, dict)
            or any(type(v) is not bool for row in flags.values() for v in row.values())
            or auth.get('replay_variant_sha256') != hashlib.sha256(canonical(flags)).hexdigest()):
        raise ValueError('r31 replay effective flags or variant binding changed')
    if not re.fullmatch(r'c2-r31-l1-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id', ''))):
        raise ValueError('r31 replay requires a new batch; old campaign cannot resume')
    cases = [f'baseline-{i:03d}' for i in range(20)]
    profiles = {case: ('C2-r3.1-far' if i >= 10 else 'C2-r3.1')
                for i,case in enumerate(cases)}
    if auth.get('case_ids') != cases or auth.get('case_profiles') != profiles:
        raise ValueError('r31 must preserve the frozen near/far case profiles')
    if auth.get('replay_roster_sha256') != '09e1d119565bee3fd54c58735d0079ec07b8f404c0499414077fe110b1b640ea':
        raise ValueError('r31 must retain the original frozen 28-command roster')
    # Bind the original task/environment evidence, not caller-rehashed replacements.
    frozen_bindings = (
        ('roster', '33dbc4dbad4a270c23eda2dec642736fe7c132d653e66f2c49aefda8e4c6fcdc'),
        ('controller_reconstruction', 'd5a9c6b68f757a79e18c93efa1d67cb85b0261253eb7094226ebcada10a54fc3'),
        ('asset_manifest', '2bbb2f9d9e5d9ead7284de8664e0dbd80ff51acf8f6a4e98bf68fc26c9f4c07a'))
    bound = {}
    for field, digest in frozen_bindings:
        if auth.get(field+'_sha256') != digest:
            raise ValueError('r31 frozen '+field+' binding changed')
        bound[field] = r3_bound_json(dict(path=auth[field+'_path'], sha256=digest))
    assets = auth.get('asset_sha256')
    if (not isinstance(assets, dict) or hashlib.sha256(canonical(assets)).hexdigest()
            != '90c009efad671635f28413f54dffedc8dbcfc0b5abb5ae2f23ec4aa06db6a783'):
        raise ValueError('r31 complete frozen asset mapping changed')
    if (not isinstance(auth.get('asset_root'), str)
            or Path(bound['asset_manifest']['asset_root']).resolve() != Path(auth['asset_root']).resolve()):
        raise ValueError('r31 asset root differs from frozen manifest')
    source = auth.get('source_sha256', {})
    required = {'tests/test_astra_pick_'+label+'.py' for label in ('contract','parity','measure')}
    if not isinstance(source, dict) or not required <= set(source):
        raise ValueError('r31 source manifest must bind all three candidate tests')
    upstream = auth.get('upstream_receipts', [])
    if len(upstream) != 1 or upstream[0].get('stage') != 'L0':
        raise ValueError('r31 requires its new L0 receipt')
    l0 = r3_bound_json(upstream[0])
    for key in ('scope','system_revision','replay_variant','replay_variant_sha256'):
        if l0.get(key) != auth.get(key):
            raise ValueError('r31 L0 belongs to another revision or replay variant')
    required_gates = ('historical_request_bytes', 'historical_controller_actions',
        'historical_controller_feedback', 'old_default_frozen_ast', 'replay_effective_flags',
        'termination_and_post_observation', 'evaluation_isolation', 'replay_command_roster')
    if (l0.get('cpu_only') is not True
            or any(type(l0.get(k)) is not int or l0[k] != 0 for k in ('api_requests','physics_steps'))
            or any(l0.get('gates', {}).get(key) is not True for key in required_gates)):
        raise ValueError('r31 L0 exact CPU gates incomplete')
    # Reuse every existing resource/source/lease/upstream/no-recovery check.
    # This validation-only projection neither grants permission nor mutates auth.
    projected = deepcopy(auth)
    projected['scope'] = R3_SCOPE
    projected['batch_id'] = 'c2-r3-l1-'+auth['batch_id'][len('c2-r31-l1-'):]
    if 'case_profiles' in projected:
        projected['case_profiles'] = {k: {'C2-r3.1':'C2-r3', 'C2-r3.1-far':'C2-r3-far'}.get(v,v)
                                      for k,v in projected['case_profiles'].items()}
    validate_r3_scope(projected, C2_R3_FAR if profile == C2_R31_FAR else C2_R3,
                      replay=True, case_id=case_id)


R31_STAGED_SCOPE = 'c2-r31-staged-20261008'
R33_STAGED_SCOPE = 'c2-r33-staged-20261008'


def validate_r31_paid_scope(auth, profile=C2_R31, *, case_id=None):
    """Validate new-revision paid permission without changing request identity."""
    stage = auth.get('stage')
    if (auth.get('scope') != R31_STAGED_SCOPE or stage not in ('L2','L3','L4')
            or auth.get('system_revision') != 'C2-r3.1' or profile not in (C2_R31,C2_R31_FAR)):
        raise ValueError('r31 paid scope or revision mismatch')
    if (auth.get('l2_gate_definition_status') != 'approved_original_criteria'
            or not isinstance(auth.get('l2_gate_definition_authorization_evidence'), str)
            or not auth['l2_gate_definition_authorization_evidence'].strip()):
        raise ValueError('r31 original L2 gate definitions require explicit approval evidence')
    flags = auth.get('full_profile_flags_exact')
    if flags != {k:True for k in R31_REPLAY_FLAGS['control']} or any(type(v) is not bool for v in flags.values()):
        raise ValueError('r31 paid full-profile flags changed')
    variant_hash = hashlib.sha256(canonical(R31_REPLAY_FLAGS)).hexdigest()
    if auth.get('replay_variant') != R31_REPLAY_VARIANT or auth.get('replay_variant_sha256') != variant_hash:
        raise ValueError('r31 paid upstream replay variant changed')
    if any(type(auth.get(k)) is not int or auth[k] != 0 for k in
           ('automatic_retries','automatic_recoveries','prior_budget_transferred')):
        raise ValueError('r31 paid retries/recovery/transfer forbidden')
    if not re.fullmatch(r'c2-r31-'+stage.lower()+r'-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id',''))):
        raise ValueError('r31 paid fresh stage batch required')
    for field,digest in (
            ('original_task_roster','33dbc4dbad4a270c23eda2dec642736fe7c132d653e66f2c49aefda8e4c6fcdc'),
            ('replay_roster','09e1d119565bee3fd54c58735d0079ec07b8f404c0499414077fe110b1b640ea'),
            ('controller_reconstruction','d5a9c6b68f757a79e18c93efa1d67cb85b0261253eb7094226ebcada10a54fc3'),
            ('asset_manifest','2bbb2f9d9e5d9ead7284de8664e0dbd80ff51acf8f6a4e98bf68fc26c9f4c07a')):
        if auth.get(field+'_sha256') != digest:
            raise ValueError('r31 paid frozen '+field+' changed')
        value = r3_bound_json(dict(path=auth[field+'_path'],sha256=digest))
        if field == 'asset_manifest' and (not isinstance(auth.get('asset_root'),str)
                or Path(value['asset_root']).resolve()!=Path(auth['asset_root']).resolve()):
            raise ValueError('r31 paid asset root changed')
    if not isinstance(auth.get('asset_sha256'),dict) or hashlib.sha256(canonical(auth['asset_sha256'])).hexdigest() != '90c009efad671635f28413f54dffedc8dbcfc0b5abb5ae2f23ec4aa06db6a783':
        raise ValueError('r31 paid full asset mapping changed')
    if not {f'tests/test_astra_pick_{s}.py' for s in ('contract','parity','measure')} <= set(auth.get('source_sha256',{})):
        raise ValueError('r31 paid source must include three candidate tests')
    for binding in auth.get('upstream_receipts',[]):
        receipt = r3_bound_json(binding)
        expected_scope = R31_REPLAY_SCOPE if binding.get('stage') in ('L0','L1') else R31_STAGED_SCOPE
        if (receipt.get('scope') != expected_scope or receipt.get('system_revision') != 'C2-r3.1'
                or receipt.get('replay_variant') != R31_REPLAY_VARIANT
                or receipt.get('replay_variant_sha256') != variant_hash):
            raise ValueError('r31 paid upstream revision or variant mismatch')
        if binding.get('stage') in ('L0','L1') and (receipt.get('roster_sha256') != auth['replay_roster_sha256']
                or binding.get('roster_sha256') != auth['replay_roster_sha256']):
            raise ValueError('r31 paid upstream replay roster changed')
        if binding.get('stage') == 'L0':
            gates = ('historical_request_bytes','historical_controller_actions','historical_controller_feedback',
                'old_default_frozen_ast','replay_effective_flags','termination_and_post_observation',
                'evaluation_isolation','replay_command_roster')
            if (receipt.get('cpu_only') is not True
                    or any(type(receipt.get(k)) is not int or receipt[k]!=0 for k in ('api_requests','physics_steps'))
                    or any(receipt.get('gates',{}).get(k) is not True for k in gates)):
                raise ValueError('r31 paid upstream L0 exact CPU gates incomplete')
        required_gates = {
            'L1': ('control_reproduction','force_reduction','negative_controls','media_complete'),
            'L2': ('cache_coverage','reaction_three_quarters','cost_projection_verified'),
            'L3': ('infrastructure_complete','media_complete','ledger_complete','per_case_cost_within_30pct')
        }.get(binding.get('stage'), ())
        if any(receipt.get('gates',{}).get(key) is not True for key in required_gates):
            raise ValueError('r31 paid upstream required acceptance gates incomplete')
    if stage == 'L2':
        for field, cap in (('l2_wall_seconds_cap',3600),('output_bytes_cap',64*1024**3)):
            if type(auth.get(field)) is not int or not 0 < auth[field] <= cap:
                raise ValueError('r31 L2 bounded wall/output scope required')
        probes = auth.get('probe_requests')
        if not isinstance(probes,list) or not probes:
            raise ValueError('r31 L2 frozen probe inputs required')
        for probe in probes:
            if not isinstance(probe,dict):
                raise ValueError('r31 L2 probe input binding incomplete')
            for field in ('wire','bundle'):
                path, digest = probe.get(field+'_path'), probe.get(field+'_sha256')
                if (not isinstance(path,str) or not path or not isinstance(digest,str)
                        or not re.fullmatch(r'[0-9a-f]{64}',digest)):
                    raise ValueError('r31 L2 '+field+' binding incomplete')
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError('r31 L2 '+field+' bytes changed')
    profiles = auth.get('case_profiles')
    if (not isinstance(profiles,dict) or set(profiles)!=set(auth.get('case_ids',[]))
            or any(v not in ('C2-r3.1','C2-r3.1-far') for v in profiles.values())
            or profile.name not in profiles.values()):
        raise ValueError('r31 paid case profiles changed')
    projected = deepcopy(auth)
    projected['scope'] = R3_SCOPE
    projected['batch_id'] = 'c2-r3-'+auth['batch_id'][len('c2-r31-'):]
    projected['case_profiles'] = {k:{'C2-r3.1':'C2-r3','C2-r3.1-far':'C2-r3-far'}.get(v,v)
                                  for k,v in auth.get('case_profiles',{}).items()}
    validate_r3_scope(projected,C2_R3_FAR if profile==C2_R31_FAR else C2_R3,case_id=case_id)

def validate_r33_paid_scope(auth, profile=C2_R33, *, case_id=None):
    """Validate new-revision paid permission without changing request identity."""
    if 'initialization_observability' in auth:
        validate_initialization_observability(auth)
    _validate_staged_scope(auth,profile,case_id=case_id,scope=R33_STAGED_SCOPE,
        batch_prefix='c2-r33-',profiles=(C2_R33,C2_R33_FAR))
    stage = auth.get('stage')
    if (auth.get('scope') != R33_STAGED_SCOPE or stage not in ('L2','L3','L4')
            or auth.get('system_revision') != 'C2-r3.3' or profile not in (C2_R33,C2_R33_FAR)):
        raise ValueError('r33 paid scope or revision mismatch')
    if (auth.get('l2_gate_definition_status') != 'approved_original_criteria'
            or not isinstance(auth.get('l2_gate_definition_authorization_evidence'), str)
            or not auth['l2_gate_definition_authorization_evidence'].strip()):
        raise ValueError('r33 original L2 gate definitions require explicit approval evidence')
    if (auth.get('approval_status') != 'APPROVED' or auth.get('explicit_user_approval') is not True
            or not isinstance(auth.get('authorization_evidence'),str)
            or not auth['authorization_evidence'].strip()
            or auth.get('resume_batch_id') is not None):
        raise ValueError('r33 fresh approved stage scope required')
    flags = auth.get('full_profile_flags_exact')
    expected_flags = {k:getattr(C2_R33,k) for k in R33_REPLAY_FLAGS['control']}
    expected_flags.update(target_association=True,target_evidence_diagnostics=True,history_first=True,
                          automatic_supervision=True,staged_metrics=True)
    if flags != expected_flags or any(type(v) is not bool for v in flags.values()):
        raise ValueError('r33 paid full-profile flags changed')
    variant_hash = hashlib.sha256(canonical(R33_REPLAY_FLAGS)).hexdigest()
    if auth.get('replay_variant') != R33_REPLAY_VARIANT or auth.get('replay_variant_sha256') != variant_hash:
        raise ValueError('r33 paid upstream replay variant changed')
    if any(type(auth.get(k)) is not int or auth[k] != 0 for k in
           ('automatic_retries','automatic_recoveries','prior_budget_transferred')):
        raise ValueError('r33 paid retries/recovery/transfer forbidden')
    if not re.fullmatch(r'c2-r33-'+stage.lower()+r'-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id',''))):
        raise ValueError('r33 paid fresh stage batch required')
    for field,digest in (
            ('original_task_roster','33dbc4dbad4a270c23eda2dec642736fe7c132d653e66f2c49aefda8e4c6fcdc'),
            ('replay_roster','09e1d119565bee3fd54c58735d0079ec07b8f404c0499414077fe110b1b640ea'),
            ('controller_reconstruction','d5a9c6b68f757a79e18c93efa1d67cb85b0261253eb7094226ebcada10a54fc3'),
            ('asset_manifest','2bbb2f9d9e5d9ead7284de8664e0dbd80ff51acf8f6a4e98bf68fc26c9f4c07a')):
        if auth.get(field+'_sha256') != digest:
            raise ValueError('r33 paid frozen '+field+' changed')
        value = r3_bound_json(dict(path=auth[field+'_path'],sha256=digest))
        if field == 'asset_manifest' and (not isinstance(auth.get('asset_root'),str)
                or Path(value['asset_root']).resolve()!=Path(auth['asset_root']).resolve()):
            raise ValueError('r33 paid asset root changed')
    if not isinstance(auth.get('asset_sha256'),dict) or hashlib.sha256(canonical(auth['asset_sha256'])).hexdigest() != '90c009efad671635f28413f54dffedc8dbcfc0b5abb5ae2f23ec4aa06db6a783':
        raise ValueError('r33 paid full asset mapping changed')
    if not {f'tests/test_astra_pick_{s}.py' for s in ('contract','parity','measure')} <= set(auth.get('source_sha256',{})):
        raise ValueError('r33 paid source must include three candidate tests')
    required = {'.runtime/r32_l1_supervisor.py','.runtime/r32_l1b_cpu_ports.py','.runtime/r3_l1_supervisor.py','.runtime/r3_finalize_replay.py',
        'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',
        'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/response-model.json'}
    if not required <= set(auth.get('source_sha256',{})) or len(auth['source_sha256']) != 21:
        raise ValueError('r33 paid complete source21 required')
    package = Path(__file__).parent
    for path in package.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.json'):
            if hashlib.sha256(path.read_bytes()).hexdigest() != auth['source_sha256'].get('src/astra_pick/'+path.relative_to(package).as_posix()):
                raise ValueError('r33 running source differs from paid freeze')
    matrix = r3_bound_json(auth['matrix_plan'])
    if (hashlib.sha256(canonical({k:matrix.get(k) for k in R32_MATRIX_CONTRACT_KEYS})).hexdigest()!=R33_MATRIX_CONTRACT_SHA256
            or matrix.get('source_manifest_sha256')!=auth['source_manifest_sha256']
            or auth['l1b_sources_and_scores'].get('sha256')!=R32_L1B_EVIDENCE_SHA256):
        raise ValueError('r33 paid matrix/source or B evidence mismatch')
    r3_bound_json(auth['l1b_sources_and_scores'])
    for binding in auth.get('upstream_receipts',[]):
        receipt = r3_bound_json(binding)
        if receipt.get('replay_initial_anchor') != auth.get('replay_initial_anchor'):
            raise ValueError('r33 upstream initial-anchor protocol/manifest mismatch')
        expected_scope = R33_REPLAY_SCOPE if binding.get('stage') in ('L0','L1') else R33_STAGED_SCOPE
        if (receipt.get('scope') != expected_scope or receipt.get('system_revision') != 'C2-r3.3'
                or receipt.get('replay_variant') != R33_REPLAY_VARIANT
                or receipt.get('replay_variant_sha256') != variant_hash):
            raise ValueError('r33 paid upstream revision or variant mismatch')
        if binding.get('stage') in ('L0','L1') and (receipt.get('roster_sha256') != auth['replay_roster_sha256']
                or binding.get('roster_sha256') != auth['replay_roster_sha256']):
            raise ValueError('r33 paid upstream replay roster changed')
        if receipt.get('initialization_observability') != auth.get('initialization_observability'):
            raise ValueError('r33 initialization observability upstream marker mismatch')
        if binding.get('stage') == 'L0':
            if (receipt.get('cpu_only') is not True
                    or any(type(receipt.get(k)) is not int or receipt[k]!=0 for k in ('api_requests','physics_steps'))
                    or receipt.get('matrix_plan_sha256') != auth['matrix_plan']['sha256']
                    or receipt.get('l1b_sources_and_scores_sha256') != R32_L1B_EVIDENCE_SHA256):
                raise ValueError('r33 L0 concrete source/matrix evidence incomplete')
            validate_replay_anchor_receipts(auth,matrix,receipt)
            validate_r32_l0_evidence(receipt,auth)
        required_gates = {
            'L1': ('complete_64_arms','control_reproduction','reproduced_contacts_half','negative_controls','B4'),
            'L2': ('cache_coverage','reaction_three_quarters','cost_projection_verified'),
            'L3': ('infrastructure_complete','media_complete','ledger_complete','per_case_cost_within_30pct')
        }.get(binding.get('stage'), ())
        if any(receipt.get('gates',{}).get(key) is not True for key in required_gates):
            raise ValueError('r33 paid upstream required acceptance gates incomplete')
        if binding.get('stage') == 'L1':
            validate_r33_paid_l1_evidence(auth,binding)
    if stage == 'L2':
        original_profiles = {f'baseline-{i:03d}':C2_R33_FAR.name if i>=10 else C2_R33.name for i in range(20)}
        if any(original_profiles.get(case)!=auth.get('case_profiles',{}).get(case) for case in auth['case_ids']):
            raise ValueError('r33 L2 original20 near/far profiles required')
        for field, cap in (('l2_wall_seconds_cap',3600),('output_bytes_cap',64*1024**3)):
            if type(auth.get(field)) is not int or not 0 < auth[field] <= cap:
                raise ValueError('r33 L2 bounded wall/output scope required')
        probes = auth.get('probe_requests')
        if not isinstance(probes,list) or not 9<=len(probes)<=11 or auth.get('provider_requests_cap')!=11:
            raise ValueError('r33 L2 frozen probe inputs required')
        for probe in probes:
            if not isinstance(probe,dict):
                raise ValueError('r33 L2 probe input binding incomplete')
            for field in ('wire','bundle'):
                path, digest = probe.get(field+'_path'), probe.get(field+'_sha256')
                if (not isinstance(path,str) or not path or not isinstance(digest,str)
                        or not re.fullmatch(r'[0-9a-f]{64}',digest)):
                    raise ValueError('r33 L2 '+field+' binding incomplete')
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                    raise ValueError('r33 L2 '+field+' bytes changed')
        validate_r33_l2_provenance(auth)
    profiles = auth.get('case_profiles')
    if (not isinstance(profiles,dict) or set(profiles)!=set(auth.get('case_ids',[]))
            or any(v not in ('C2-r3.3','C2-r3.3-far') for v in profiles.values())
            or profile.name not in profiles.values()):
        raise ValueError('r33 paid case profiles changed')


def validate_r33_l2_provenance(auth):
    """Bind every API-only probe to audited physical observations and archived history.

    No history synthesis: a request whose genuine archived calls cannot satisfy
    the R33 request contract stays blocked rather than acquiring invented fields.
    """
    binding = next(b for b in auth['upstream_receipts'] if b['stage']=='L1')
    receipt = r3_bound_json(binding)
    root = Path(binding['path']).resolve().parent
    expected = [f"L1-A-{r['case']}-{r['turn']}-candidate" for r in receipt.get('comparisons',[])
                if r.get('group')=='contact' and r.get('control_reproduced') is True]
    probes=auth['probe_requests'];cache=probes[:3];reactions=probes[3:]
    if (not 6<=len(expected)<=8 or len(set(expected))!=len(expected) or len(probes)!=3+len(expected)
            or [p.get('role') for p in cache]!=['cache']*3
            or [p.get('sequence') for p in cache]!=[0,1,2]
            or len({p.get('case_id') for p in cache})!=1
            or len({p.get('source_l1_arm_identity') for p in cache})!=1
            or any(p.get('role')!='contact_reaction' for p in reactions)
            or [p.get('source_l1_arm_identity') for p in reactions]!=expected):
        raise ValueError('r33 L2 requires cache3 and exactly each reproduced contact reaction')
    rows={row['identity']:row for row in receipt.get('arms',[])}
    for probe in probes:
        identity=probe.get('source_l1_arm_identity');row=rows.get(identity)
        if (not isinstance(identity,str) or row is None or row.get('arm')!='candidate'
                or row.get('group')!='L1-A' or row.get('status')!='evaluated'):
            raise ValueError('r33 probe source must be an audited L1-A candidate')
        attempt=(root/'attempts'/identity).resolve()
        if not attempt.is_relative_to(root/'attempts'):
            raise ValueError('r33 probe source path escapes L1')
        result=r3_bound_json(dict(path=str(attempt/'result.json'),sha256=row['result_sha256']))
        end_binding=result['l1_end_observation_binding'];end_path=(attempt/end_binding['path']).resolve()
        source_binding=probe.get('source_end_observation')
        if (not isinstance(source_binding,dict) or Path(source_binding.get('path','')).resolve()!=end_path
                or source_binding.get('sha256')!=end_binding['sha256'] or not end_path.is_relative_to(attempt)):
            raise ValueError('r33 probe end observation binding mismatch')
        end=r3_bound_json(source_binding)
        if (end.get('case_id')!=probe['case_id'] or end.get('replay_arm')!='candidate'
                or result.get('case_id')!=probe['case_id'] or result.get('replay_arm')!='candidate'):
            raise ValueError('r33 probe case/arm differs from physical source')
        profile=PROFILES[auth['case_profiles'][probe['case_id']]]
        current=r3_bound_json(dict(path=probe['bundle_path'],sha256=probe['bundle_sha256']))
        sequence=probe['sequence'];sources=probe.get('source_history')
        if not isinstance(sources,list) or len(sources)!=sequence:
            raise ValueError('r33 probe needs complete hash-bound real history')
        archive=r3_bound_json(probe['archive']).get('archive')
        if not isinstance(archive,list) or len(archive)!=sequence:
            raise ValueError('r33 probe exact request archive missing')
        original_binding=dict(path=str((Path(auth['code_root'])/result['replay_source']['source_path']).resolve()),
                              sha256=result['replay_source']['result_sha256'])
        original=r3_bound_json(original_binding)
        historical_profile=C2_R25_FAR if profile==C2_R33_FAR else C2_R25
        original_key='C2-r2.5-far' if historical_profile==C2_R25_FAR else 'C2-r2.5'
        if (original.get('execution_profile')!=original_key
                or original.get('information_condition')!=historical_profile.name):
            raise ValueError('r33 historical profile must match the hash-bound recorded revision')
        actual_command=next(c for c in result['commands'] if c['turn']==result['replay_turn'])
        original_command=next(c for c in original['commands'] if c['turn']==result['replay_turn'])
        if actual_command['call']!=original_command['call']:
            raise ValueError('r33 mechanical call differs from its original historical-schema source')
        if probe['role']=='contact_reaction' and sequence!=result['replay_turn']+1:
            raise ValueError('r33 reaction must immediately follow the selected L1 command')
        history=[]
        flags={k:getattr(profile,k) for k in ('robot_diagnostics','schedule_precheck','base_progress_monitor',
                   'base_response_monitor','base_response_debt_monitor','execution_lifecycle')}
        for index,source in enumerate(sources):
            latest=probe['role']=='contact_reaction' and index==sequence-1
            expected_result=dict(path=str(attempt/'result.json'),sha256=row['result_sha256']) if latest else original_binding
            if (source.get('turn')!=index or not isinstance(source.get('source_result'),dict)
                    or Path(source['source_result'].get('path','')).resolve()!=Path(expected_result['path']).resolve()
                    or source['source_result'].get('sha256')!=expected_result['sha256']):
                raise ValueError('r33 history result/turn is not the real frozen source')
            raw=r3_bound_json(source['source_result'])
            command=next(c for c in raw['commands'] if c['turn']==index)
            bundle_binding=source.get('source_bundle')
            expected_bundle=Path(original_binding['path']).parent/'provider'/f'turn-{index:03d}'/'request-bundle.json'
            if (not isinstance(bundle_binding,dict)
                    or Path(bundle_binding.get('path','')).resolve()!=expected_bundle.resolve()):
                raise ValueError('r33 history bundle is not the original archived request')
            prior=r3_bound_json(bundle_binding)
            if prior.get('position')!=probe['case_id'] or prior.get('turn')!=index:
                raise ValueError('r33 source history case/turn mismatch')
            prior=deepcopy(prior)
            prior['authorization_sha256']=end['authorization_sha256']
            prior['observation']['calls_remaining']=profile.call_cap-index
            # Earlier history and all sensor/state values remain byte-value exact.
            expected_archive=dict(bundle=prior,call=command['call'],source_profile=historical_profile.name)
            if archive[index]!=expected_archive:
                raise ValueError('r33 request archive differs from genuine source')
            history.append(dict(call=command['call'],result=public_result(command['result'],**flags)))
        if probe['role']=='contact_reaction' and history[-1]!=dict(call=actual_command['call'],result=public_result(actual_command['result'],**flags)):
            raise ValueError('r33 reaction latest feedback differs from executed command')
        observation=deepcopy(end['observation'])
        observation.update(history=history,calls_remaining=profile.call_cap-sequence,
                           previous_state=state_snapshot(archive[-1]['bundle']['observation']) if archive else None)
        expected_bundle=dict(position=probe['case_id'],turn=sequence,kind='tool',
            authorization_sha256=end['authorization_sha256'],observation=observation,sensor_manifest=end['sensor_manifest'])
        if current!=expected_bundle:
            raise ValueError('r33 probe observation/history differs from real command end')
        if Path(probe['wire_path']).read_bytes()!=build_wire(current,archive,profile):
            raise ValueError('r33 probe wire differs from source-bound request')


def validate_final_anchor_attempt(auth, result, case_id, attempt):
    """Read the source-bound evaluator; L4 uses the same reset proof as L1."""
    if 'replay_initial_anchor' not in auth:
        return
    import importlib.util
    import sys
    path=Path(auth['code_root'])/'.runtime/r32_l1_supervisor.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=auth['source_sha256'].get('.runtime/r32_l1_supervisor.py'):
        raise ValueError('actual-anchor evaluator source changed')
    spec=importlib.util.spec_from_file_location('astra_pick_anchor_evaluator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.pick=sys.modules[__name__]
    anchor=replay_initial_anchor(auth,case_id)
    module.validate_initial_anchor_result(auth,result,dict(case=case_id.removeprefix('baseline-'),
        result_sha256=anchor['binding']['result_sha256']),Path(attempt))


def validate_r33_paid_l1_evidence(auth, binding):
    """Audit the bound same-source L1 artifacts without execution or lease renewal."""
    import importlib.util
    import sys
    path = Path(auth['code_root'])/'.runtime/r32_l1_supervisor.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=auth['source_sha256'].get('.runtime/r32_l1_supervisor.py'):
        raise ValueError('r33 L1 audit source changed')
    spec = importlib.util.spec_from_file_location('astra_pick_r33_paid_l1_audit',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pick = sys.modules[__name__]
    module.validate_passed_l1_receipt(auth,binding)



def r3_bound_json(binding):
    """Read the exact evidence bytes named in a stage authorization."""
    if not isinstance(binding, dict) or not isinstance(binding.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', binding['sha256']):
        raise ValueError('r3 evidence binding incomplete')
    raw = Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('r3 evidence hash changed')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('r3 evidence must be an object')
    return value


def validate_r3_scope(auth, profile=C2_R3, *, replay=False, case_id=None):
    if any((profile.schedule_precheck, profile.robot_diagnostics,
            profile.base_heading_hold, profile.continuous_base_speed)):
        raise ValueError('C2-r3.1 is CPU-only; a fresh frozen physical validation scope is required')
    return _validate_staged_scope(auth,profile,replay=replay,case_id=case_id)


def _validate_staged_scope(auth, profile, *, replay=False, case_id=None, scope=R3_SCOPE,
                           batch_prefix='c2-r3-', profiles=(C2_R3,C2_R3_FAR)):
    """Fresh, stage-specific permission; L1 cannot enter any provider path."""
    stage = auth.get('stage')
    if auth.get('scope') != scope or stage not in ('L1', 'L2', 'L3', 'L4'):
        raise ValueError('C2-r3 is CPU-only without fresh staged authorization')
    if (auth.get('status') != 'authorized' or not auth.get('authorization_evidence')
            or type(auth.get('expires_at_epoch')) not in (int, float)
            or not math.isfinite(auth['expires_at_epoch']) or time.time() >= auth['expires_at_epoch']):
        raise ValueError('r3 authorization or lease missing')
    if replay != (stage == 'L1'):
        raise ValueError('L1 is replay-only with zero API; paid stages cannot use replay permission')
    if profile not in profiles:
        raise ValueError('r3 authorization cannot change legacy profiles or independent switches')
    if case_id is not None and (case_id not in auth.get('case_ids', []) or
            ('case_profiles' in auth and auth['case_profiles'].get(case_id) != profile.name)):
        raise ValueError('r3 per-case profile mismatch')
    if any(auth.get(k) is not None for k in ('infrastructure_retry', 'recovery_budget', 'recovery_attempt_id')):
        raise ValueError('r3 stages forbid retries and recovery')
    fixed = dict(currency='USD', campaign_usd_cap='364', prior_budget_transferred=0,
                 condition='V-mobile', cumulative_force_to_model=False,
                 automatic_retries=0, model=MODEL, reasoning='medium')
    if any(auth.get(k) != v for k, v in fixed.items()):
        raise ValueError('r3 frozen campaign contract mismatch')
    if not re.fullmatch(batch_prefix+stage.lower()+r'-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id', ''))):
        raise ValueError('r3 stage needs its own fresh batch namespace')
    source = auth.get('source_sha256')
    package = Path(__file__).parent
    required = {'src/astra_pick/'+str(p.relative_to(package)).replace('\\', '/')
                for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.md', '.json')}
    if not isinstance(source, dict) or not required <= set(source):
        raise ValueError('r3 candidate source manifest incomplete')
    if hashlib.sha256(canonical(source)).hexdigest() != auth.get('source_manifest_sha256'):
        raise ValueError('r3 source manifest binding mismatch')
    code_root = Path(auth['code_root']).resolve()
    for name, digest in source.items():
        path = (code_root/name).resolve()
        if not path.is_relative_to(code_root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('r3 source changed')
    roster = r3_bound_json(dict(path=auth['roster_path'], sha256=auth.get('roster_sha256')))
    cases = auth.get('case_ids')
    if not isinstance(cases, list) or not cases or any(not isinstance(c, str) or not c for c in cases) or len(set(cases)) != len(cases):
        raise ValueError('r3 unique ordered cases required')
    upstream = auth.get('upstream_receipts')
    expected_stages = ['L'+str(i) for i in range(int(stage[1:]))]
    if not isinstance(upstream, list) or [b.get('stage') for b in upstream] != expected_stages:
        raise ValueError('r3 preceding stages must all pass in order')
    for binding in upstream:
        receipt = r3_bound_json(binding)
        if (receipt.get('stage') != binding['stage'] or receipt.get('status') != 'passed'
                or receipt.get('source_manifest_sha256') != auth['source_manifest_sha256']
                or not binding.get('roster_sha256') or receipt.get('roster_sha256') != binding['roster_sha256']
                or not isinstance(receipt.get('gates'), dict) or not receipt['gates']
                or any(value is not True for value in receipt['gates'].values())):
            raise ValueError('r3 preceding-stage receipt incomplete or different source')
    ceilings = {'L1': ('0', 0), 'L2': ('14', 11), 'L3': ('70', 150), 'L4': ('280', 750)}
    money, requests = ceilings[stage]
    if auth.get('usd_cap') != money or type(auth.get('provider_requests_cap')) is not int or not 0 <= auth['provider_requests_cap'] <= requests:
        raise ValueError('r3 stage budget cannot transfer or expand')
    if stage == 'L1':
        limits = dict(resets_max=56, physical_steps_cap=61600, worker_seconds_cap=600,
                      gpu1_process_seconds_cap=7200, cpu_media_seconds_cap=7200,
                      output_bytes_cap=64*1024**3)
        if any(type(auth.get(k)) is not int or not 0 < auth[k] <= cap for k, cap in limits.items()):
            raise ValueError('r3 L1 resource bounds')
        replay_roster = r3_bound_json(dict(path=auth['replay_roster_path'], sha256=auth.get('replay_roster_sha256')))
        if (replay_roster.get('contact_commands') != 8 or replay_roster.get('noncontact_commands') != 20
                or len(replay_roster.get('roster', [])) != 28 or auth.get('recorded_action_replay') is not True):
            raise ValueError('r3 L1 fixed 28-command replay required')
        return
    if auth.get('per_case_usd_cap') != '14' or auth['provider_requests_cap'] <= 0:
        raise ValueError('r3 per-case cost or request ceiling')
    for name in ('provider_reconciliation', 'cache_pricing'):
        receipt = r3_bound_json(auth[name])
        if receipt.get('status') != 'verified' or receipt.get('provider') != 'openai' or receipt.get('currency') != 'USD':
            raise ValueError('r3 provider billing and cache pricing must be verified')
        if name == 'cache_pricing':
            if receipt.get('model') != MODEL:
                raise ValueError('r3 cache price model mismatch')
            for field in ('input_usd_per_million', 'cached_input_usd_per_million', 'output_usd_per_million'):
                amount = Decimal(receipt[field])
                if not amount.is_finite() or amount < 0:
                    raise ValueError('r3 verified provider price invalid')
    if stage == 'L2':
        probes = auth.get('probe_requests')
        if auth.get('physical_steps_cap') != 0 or not isinstance(probes, list) or not 1 <= len(probes) <= auth['provider_requests_cap']:
            raise ValueError('r3 L2 requires frozen API-only probe requests')
        keys = set()
        for probe in probes:
            probe_profile = PROFILES.get(auth.get('case_profiles',{}).get(probe.get('case_id'))) if scope==R33_STAGED_SCOPE else profile
            if probe_profile not in profiles:
                raise ValueError('r33 probe per-case profile mismatch')
            key = (probe.get('case_id'), probe.get('sequence'))
            if (probe.get('case_id') not in cases or probe.get('kind') != 'main' or type(probe.get('sequence')) is not int
                    or not 0 <= probe['sequence'] < probe_profile.call_cap or key in keys
                    or not isinstance(probe.get('wire_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', probe['wire_sha256'])):
                raise ValueError('r3 L2 probe binding invalid')
            keys.add(key)
        return
    expected_cases = [f'arm-dev-{i:03d}' for i in range(5)] if stage == 'L3' else [f'baseline-{i:03d}' for i in (2,3,6,11,12,15,17,18,5,0,1,4,7,8,9,10,13,14,16,19)]
    profiles = {case: (profiles[1].name if stage == 'L4' and int(case.rsplit('-',1)[1]) >= 10 else profiles[0].name) for case in expected_cases}
    if cases != expected_cases or auth.get('case_profiles') != profiles or profile.name not in profiles.values():
        raise ValueError('r3 frozen stage case order or distance profiles')
    if auth.get('physical_steps_cap') != (3000 if stage == 'L3' else 16000) or auth.get('per_case_verification_cap') != 5 or auth.get('detailed_trajectory') is not True:
        raise ValueError('r3 physical or recording scope mismatch')
    if stage == 'L4':
        if auth.get('primary_metric') != 'force_limit_terminations' or auth.get('stage1_force_limit_stop_count') != 4 or auth.get('stage1_cases') != expected_cases[:9]:
            raise ValueError('r3 L4 first-stage stopping rule mismatch')
        accepted = r3_bound_json(auth['accepted_cost'])
        if accepted.get('status') != 'passed' or accepted.get('source_manifest_sha256') != auth['source_manifest_sha256']:
            raise ValueError('r3 measured cost acceptance missing')
        for field in ('near_case_usd', 'far_case_usd'):
            amount = Decimal(accepted[field])
            if not amount.is_finite() or not 0 < amount <= 14:
                raise ValueError('r3 measured per-case cost exceeds approved ceiling')
        if accepted.get('far_decisions') != 40 or accepted.get('near_decisions') != 25:
            raise ValueError('r3 cost estimate cannot reduce decisions')


def validate_scope(auth, profile=C0):
    if profile.response_session or auth.get('scope') == FROZEN['sol_session']['scope']:
        return validate_session_scope(auth, profile)
    if 'initialization_observability' in auth:
        validate_initialization_observability(auth)
    if 'replay_initial_anchor' in auth and auth.get('scope') not in (R33_REPLAY_SCOPE,R33_STAGED_SCOPE):
        raise ValueError('actual-anchor option is forbidden in legacy scope')
    if auth.get('scope') == R33_STAGED_SCOPE:
        return validate_r33_paid_scope(auth,profile)
    if any((profile.target_association, profile.target_evidence_diagnostics,
            profile.base_progress_monitor, profile.base_response_monitor, profile.base_response_debt_monitor,
            profile.base_progress_abort, profile.execution_lifecycle)):
        raise ValueError('C2-r3.2 is CPU-only; new source and physical scope must be frozen before execution')
    if auth.get('scope') == R31_STAGED_SCOPE:
        return validate_r31_paid_scope(auth,profile)
    if any((profile.schedule_precheck, profile.robot_diagnostics,
            profile.base_heading_hold, profile.continuous_base_speed)):
        raise ValueError('C2-r3.1 is CPU-only; a fresh frozen physical validation scope is required')
    if any((profile.staged_metrics, profile.automatic_supervision, profile.history_first, profile.base_motion_abort)):
        return validate_r3_scope(auth, profile)
    if profile.stall_guard_mode == 'reference_progress' and auth.get('scope') == 'baseline20-supervised-20261006':
        expected = dict(status='authorized', execution_profile='C2-baseline20', condition='V-mobile',
            model=MODEL, reasoning='medium', executor='arm-coordinated-v2-600',
            provider_requests_cap=750, physical_steps_cap=16000, usd_cap='280',
            per_case_usd_cap='14', prior_budget_transferred=0, per_case_verification_cap=5,
            camera_resolution=640, texture_resolution=4096, close_uses_endpoint_tolerance=False,
            stall_feedback=True, stall_guard_mode='reference_progress', command_supervision=True,
            detailed_trajectory=True)
        if profile.model_target_boxes:
            expected['execution_profile'] = 'C2-r2.5-baseline20'
            expected['system_revision'] = 'C2-r2.5'
            if not re.fullmatch(r'c2-r25-[a-z0-9][a-z0-9-]{0,63}', str(auth.get('batch_id', ''))):
                raise ValueError('fresh r25 batch namespace required')
            if auth.get('recovery_attempt_id') or auth.get('recovery_budget'):
                raise ValueError('new frozen batch cannot inherit historical recovery namespace')
            retry = auth.get('infrastructure_retry')
            if retry is not None:
                if not isinstance(retry,dict) or retry.get('attempt_index') != 1 or retry.get('case_id') not in auth['case_ids'] or not retry.get('authorization_evidence'):
                    raise ValueError('only one explicitly approved infrastructure retry is allowed')
                source = Path(retry['prior_result_path'])
                if hashlib.sha256(source.read_bytes()).hexdigest() != retry.get('prior_result_sha256'):
                    raise ValueError('infrastructure retry prior result changed')
                result = json.loads(source.read_bytes())
                if result.get('status') != 'infrastructure_censored' or result.get('case_id') != retry['case_id'] or result.get('attempt_index') != 0 or result.get('batch_id') != auth['batch_id']:
                    raise ValueError('only initial infrastructure-censored result permits retry')
        if profile not in (C2_BASELINE_PREP,C2_BASELINE_FAR,C2_R25,C2_R25_FAR) or any(auth.get(k)!=v for k,v in expected.items()) or auth.get('case_ids') != [f'baseline-{i:03d}' for i in range(20)]:
            raise ValueError('baseline20 frozen scope mismatch')
        if not auth.get('authorization_evidence') or not auth.get('asset_manifest_sha256') or not auth.get('asset_root') or time.time() >= auth.get('expires_at_epoch',0):
            raise ValueError('baseline20 authorization assets or lease missing')
        recovery = auth.get('recovery_budget')
        if recovery is not None:
            if (not isinstance(recovery, dict) or recovery.get('status') != 'authorized'
                    or not recovery.get('authorization_evidence')
                    or recovery.get('case_id') not in auth['case_ids']
                    or not re.fullmatch(r'recovery-[0-9]+', str(recovery.get('attempt_id')))
                    or recovery.get('attempt_id') != auth.get('recovery_attempt_id')):
                raise ValueError('explicit recovery budget authorization missing')
            try:
                prior = Decimal(recovery['prior_case_committed_usd'])
                extra = Decimal(recovery['incremental_usd_cap'])
                valid = prior.is_finite() and extra.is_finite() and prior >= 0 and 0 < extra <= 14 and prior+extra <= Decimal(auth['usd_cap'])
            except (KeyError, ArithmeticError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError('invalid recovery spending ceiling')
        return
    if profile.stall_guard_mode == 'reference_progress':
        expected = dict(scope='baseline-single-smoke-20261006', execution_profile='C2-baseline-prep',
            status='authorized', condition='V-mobile', model=MODEL, reasoning='medium',
            executor='arm-coordinated-v2-600', provider_requests_cap=30, per_case_decisions_cap=25,
            per_case_verification_cap=5, per_case_physical_steps_cap=600, physical_steps_cap=600,
            gpu1_process_seconds_cap=1800, worker_seconds_cap=1800, cpu_media_seconds_cap=600,
            media_worker_seconds_cap=600, output_bytes_cap=12*1024**3, prior_budget_transferred=0,
            usd_cap='14', per_case_usd_cap='14', camera_resolution=640, texture_resolution=4096,
            close_uses_endpoint_tolerance=False, stall_feedback=True, stall_guard_mode='reference_progress',
            command_supervision=True, detailed_trajectory=True)
        if profile != C2_BASELINE_PREP or any(auth.get(k) != v for k,v in expected.items()) or auth.get('case_ids') != ['arm-dev-004']:
            raise ValueError('baseline preparation has no frozen live authorization matching this scope')
        if not auth.get('authorization_evidence') or not auth.get('asset_manifest_sha256') or not auth.get('asset_root') or time.time() >= auth.get('expires_at_epoch', 0):
            raise ValueError('baseline smoke authorization, assets or lease missing')
        return
    if profile.identity_revision == 2:
        expected = dict(scope='c2-r2-smoke-20261006-r1', execution_profile='C2-r2',
            status='authorized', condition='V-mobile', model=MODEL, reasoning='medium',
            executor='arm-coordinated-v2-600', provider_requests_cap=30, per_case_decisions_cap=25,
            per_case_verification_cap=5, per_case_physical_steps_cap=600, physical_steps_cap=600,
            gpu1_process_seconds_cap=1800, prior_budget_transferred=0, usd_cap='14', per_case_usd_cap='14',
            camera_resolution=640, texture_resolution=4096, close_uses_endpoint_tolerance=False,
            stall_feedback=False, command_supervision=True, detailed_trajectory=True)
        if any(auth.get(k) != v for k,v in expected.items()) or auth.get('case_ids') != ['c2-expand-000']:
            raise ValueError('C2-r2 requires fresh budget and frozen live authorization')
        if not auth.get('authorization_evidence') or not auth.get('asset_manifest_sha256') or not auth.get('asset_root') or time.time() >= auth.get('expires_at_epoch', 0):
            raise ValueError('C2-r2 frozen assets or lease missing')
        return
    if profile == C2:
        expected = dict(scope='astra-pick-c2-hd-category-verifier-20261005', execution_profile='C2',
            status='authorized', condition='V-mobile', model=MODEL, reasoning='medium',
            executor='arm-coordinated-v2-600', provider_requests_cap=600, per_case_decisions_cap=25,
            per_case_verification_cap=5, per_case_physical_steps_cap=600, physical_steps_cap=12000,
            gpu1_process_seconds_cap=36000, cpu_media_seconds_cap=12000, prior_budget_transferred=0,
            usd_cap='280', per_case_usd_cap='14', camera_resolution=640, texture_resolution=4096,
            close_uses_endpoint_tolerance=False, stall_feedback=False)
        cases = [f'c2-smoke-{i:03d}' for i in range(2)] + [f'c2-base-{i:03d}' for i in range(5)] + [f'c2-expand-{i:03d}' for i in range(13)]
        if any(auth.get(k) != v for k,v in expected.items()) or auth.get('case_ids') != cases:
            raise ValueError('C2 authorization missing or mismatched')
        if not auth.get('authorization_evidence') or not auth.get('asset_manifest_sha256') or not auth.get('asset_root') or time.time() >= auth.get('expires_at_epoch', 0):
            raise ValueError('C2 frozen assets or lease missing')
        return
    if profile.stall_feedback or profile.verification:
        raise ValueError('experimental CPU preparation has no live authorization')
    expected = {'scope': 'wp2-located-semantic-measure-c1' if profile.measure else 'wp2-located-semantic-c0', 'guidance_version': 'guidance-wp2-measure-c1' if profile.measure else 'guidance-wp2-located-c0', 'revision': 3, 'condition': 'V-mobile', 'status': 'authorized', 'model': 'gpt-6-astra', 'reasoning': 'medium', 'executor': 'arm-coordinated-v2-600', 'provider_requests_cap': 125, 'gpu1_process_seconds_cap': 9000, 'prior_budget_transferred': 0}
    if profile.close_uses_endpoint_tolerance:
        expected.update(scope='wp2-located-semantic-c0-close-aligned',
                        execution_profile='C0-close-aligned', close_uses_endpoint_tolerance=True)
    ceiling = 65
    if auth.get('authorization_variant') is not None:
        if auth['authorization_variant'] != 'resume-four-001-004' or profile.measure or profile.close_uses_endpoint_tolerance:
            raise ValueError('unsupported authorization variant')
        expected.update(provider_requests_cap=100, gpu1_process_seconds_cap=7200,
            case_ids=[f'arm-dev-{i:03d}' for i in range(1, 5)], physical_steps_cap=2400,
            cpu_media_seconds_cap=2400, worker_seconds_cap=1800,
            media_worker_seconds_cap=600, per_case_physical_steps_cap=600,
            per_case_decisions_cap=25, output_bytes_cap=12*1024**3)
        ceiling = 52
    if any((auth.get(k) != v for k, v in expected.items())) or not auth.get('authorization_evidence'):
        raise ValueError('arm authorization missing or mismatched')
    if not Decimal(auth['usd_cap']).is_finite() or not 0 < Decimal(auth['usd_cap']) <= ceiling or time.time() >= auth['expires_at_epoch']:
        raise ValueError('arm authorization cap or lease')

def process(bundle, archive, condition, folder, ledger_path, auth, auth_sha, send, profile=C0, *, session=None):
    """Batch holds an exclusive server lock. At most one send, never auto retry."""
    validate_scope(auth, profile)
    if condition != 'V-mobile' or bundle['position'] not in auth['case_ids'] or bundle['authorization_sha256'] != auth_sha:
        raise ValueError('request scope binding')
    if profile.response_session:
        return process_session(bundle, archive, folder, ledger_path, auth, auth_sha, send, profile, session)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if bundle['kind'] != 'tool':
        raise ValueError('mobile tool request only')
    wire = build_wire(bundle, archive, profile)
    if profile.verification:
        return process_c2_wire(wire, 'main', bundle['position'], bundle['turn'], folder, ledger_path, auth, auth_sha, send, bundle=bundle)
    estimate = estimate_wire(wire)
    wire_sha = hashlib.sha256(wire).hexdigest()
    key = f"{bundle['position']}-{condition}-{bundle['kind']}-{bundle['turn']:03d}"
    ledger = json.loads(Path(ledger_path).read_text())
    if ledger['authorization_sha256'] != auth_sha or ledger['status'] != 'running':
        raise ValueError('ledger stopped or identity changed')
    entries = ledger['entries']
    prior = [e for e in entries if e['request_id'] == key]
    if prior:
        if prior[0]['wire_sha256'] != wire_sha:
            raise ValueError('same request changed')
        if (folder / 'delivery.json').exists():
            return json.loads((folder / 'delivery.json').read_text())
        raise ValueError('prior send without durable delivery; reconcile, do not resend')
    if len(entries) >= auth['provider_requests_cap'] or sum((e['case_id'] == bundle['position'] and e['condition'] == condition for e in entries)) >= 25 or committed_cost(entries) + Decimal(estimate['reserve_usd']) > Decimal(auth['usd_cap']):
        ledger['status'] = 'stopped_budget_exhausted'
        save(ledger_path, ledger)
        raise ValueError('arm budget exhausted before network')
    (folder / 'request-wire.json').write_bytes(wire)
    save(folder / 'request-bundle.json', bundle)
    entry = {'request_id': key, 'case_id': bundle['position'], 'condition': condition, 'wire_sha256': wire_sha, 'status': 'reserved', 'reserve_usd': estimate['reserve_usd'], 'estimated_usd': None, 'reserved_at_epoch': time.time()}
    entries.append(entry)
    save(ledger_path, ledger)
    try:
        raw = send(wire, key)
        (folder / 'response-raw.json').write_bytes(raw)
        result = parse_response(raw, estimate, reflection=bundle['kind'] == 'reflection', profile=profile)
        if result['status'] == 'valid' and bundle['kind'] == 'tool':
            try:
                validate_call(result['call'], profile)
            except ValueError:
                raise AssertionError('mobile parser and validator disagree')
        entry.update(result)
        if result['status'] == 'censored':
            ledger['status'] = 'stopped_provider_contract'
    except Exception as exc:
        entry.update(status='unknown', error_type=type(exc).__name__)
        ledger['status'] = 'stopped_unknown_outcome'
        save(ledger_path, ledger)
        raise RuntimeError('provider outcome unknown; reserved charge retained') from None
    save(ledger_path, ledger)
    save(folder / 'delivery.json', result)
    return result



def estimate_c2_wire(wire, kind, profile=C2):
    if kind == 'main':
        return estimate_wire(wire)
    if kind != 'verification':
        raise ValueError('C2 request kind')
    body = json.loads(wire)
    images = [p for m in body['input'] for p in m['content'] if p['type']=='input_image']
    output_cap = 2000 if profile.identity_revision == 2 else 1000
    if len(images)!=1 or body.get('store') is not False or body['max_output_tokens']!=output_cap:
        raise ValueError('C2 verification request envelope')
    bound = non_image_bytes(wire) + 1024 + 480
    if bound > 10000:
        raise ValueError('C2 verification input bound')
    return dict(input_bound=bound, output_cap=output_cap, image_count=1, image_tokens=480,
                reserve_usd=str(cost(bound,output_cap,reserve_write=True)))


def process_c2_wire(wire, kind, case, sequence, folder, ledger_path, auth, auth_sha, send, *, bundle=None):
    """One durable request sharing batch and per-case ceilings; unknown sends stop all work."""
    profile = PROFILES.get(auth.get('execution_profile'), C2)
    if auth.get('scope') in (R3_SCOPE, R31_STAGED_SCOPE, R33_STAGED_SCOPE):
        profile = PROFILES.get(auth.get('case_profiles', {}).get(case, auth.get('execution_profile')), C2)
        allowed_profiles = {R31_STAGED_SCOPE:(C2_R31,C2_R31_FAR),R33_STAGED_SCOPE:(C2_R33,C2_R33_FAR)}.get(auth.get('scope'),(C2_R3,C2_R3_FAR))
        if profile not in allowed_profiles:
            raise ValueError('r3 request requires its frozen per-case profile')
        if auth.get('stage') == 'L2' and not any(p.get('case_id') == case and p.get('sequence') == sequence
                and p.get('kind') == kind and p.get('wire_sha256') == hashlib.sha256(wire).hexdigest()
                for p in auth.get('probe_requests', [])):
            raise ValueError('r3 L2 request differs from frozen probe')
    if auth.get('scope') == 'baseline20-supervised-20261006':
        if case not in auth.get('case_ids', []):
            raise ValueError('baseline case outside scope')
        profile = (C2_R25 if auth['case_ids'].index(case) < 10 else C2_R25_FAR) if auth.get('system_revision') == 'C2-r2.5' else (C2_BASELINE_PREP if auth['case_ids'].index(case) < 10 else C2_BASELINE_FAR)
    if auth.get('scope') == R33_STAGED_SCOPE:
        validate_r33_paid_scope(auth,profile,case_id=case)
        if auth.get('stage')=='L2':
            probe=next(p for p in auth['probe_requests'] if p['case_id']==case and p['sequence']==sequence and p['kind']==kind)
            if bundle is None or canonical(bundle)!=canonical(json.loads(Path(probe['bundle_path']).read_bytes())):
                raise ValueError('r33 L2 bundle differs from frozen probe')
    elif auth.get('scope') == R31_STAGED_SCOPE:
        validate_r31_paid_scope(auth,profile,case_id=case)
    else:
        validate_scope(auth,profile)
    if case not in auth['case_ids'] or kind not in ('main','verification'):
        raise ValueError('C2 request scope')
    estimate=estimate_c2_wire(wire,kind,profile)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(wire).hexdigest();key=f'{case}-{kind}-{sequence:03d}'
    if profile.model_target_boxes:
        retry = auth.get('infrastructure_retry')
        attempt_index = 1 if retry is not None and retry['case_id'] == case else 0
        key = auth['batch_id']+f'-attempt-{attempt_index}-'+key
    attempt=auth.get('recovery_attempt_id')
    if attempt is not None:
        if auth.get('scope')!='baseline20-supervised-20261006' or not isinstance(attempt,str) or not re.fullmatch(r'recovery-[0-9]+',attempt):
            raise ValueError('invalid explicit recovery identity')
        key=f'{attempt}-{key}'
    ledger=json.loads(Path(ledger_path).read_text())
    if ledger['authorization_sha256']!=auth_sha or ledger['status']!='running':
        raise ValueError('C2 ledger stopped or identity changed')
    entries=ledger['entries'];previous=[e for e in entries if e['request_id']==key]
    if profile.model_target_boxes and any(not e['request_id'].startswith(auth['batch_id']+'-') for e in entries):
        raise ValueError('historical ledger entries cannot transfer to fresh r25 batch')
    if previous:
        if previous[0]['wire_sha256']!=digest:raise ValueError('same C2 request changed')
        if (folder/'delivery.json').exists():return json.loads((folder/'delivery.json').read_text())
        raise VerificationOutcomeUnknown('prior C2 send has no durable delivery; never resend')
    case_entries=[e for e in entries if e['case_id']==case]
    attempt_entries = [e for e in case_entries if e['request_id'].startswith(attempt+'-')] if attempt else case_entries
    if profile.model_target_boxes:
        attempt_entries = [e for e in case_entries if e['request_id'].startswith(auth['batch_id']+f'-attempt-{attempt_index}-')]
    kind_limit=profile.call_cap if kind=='main' else profile.verification_call_cap
    reserve=Decimal(estimate['reserve_usd'])
    case_cap = Decimal('14')
    recovery = auth.get('recovery_budget')
    recovery_exhausted = False
    if recovery is not None and case == recovery['case_id']:
        prior_entries = [e for e in case_entries if not e['request_id'].startswith(attempt+'-')]
        prior = committed_cost(prior_entries)
        if prior != Decimal(recovery['prior_case_committed_usd']):
            raise ValueError('recovery prior spending differs from frozen ledger')
        extra = Decimal(recovery['incremental_usd_cap'])
        case_cap = prior+extra
        recovery_exhausted = committed_cost(attempt_entries)+reserve > extra
    global_exhausted = len(entries)>=auth['provider_requests_cap'] or committed_cost(entries)+reserve>Decimal(auth['usd_cap'])
    local_exhausted = sum(e['request_kind']==kind for e in attempt_entries)>=kind_limit or committed_cost(case_entries)+reserve>case_cap or recovery_exhausted
    if global_exhausted or local_exhausted:
        if global_exhausted or not profile.model_target_boxes:
            ledger['status']='stopped_budget_exhausted'
        save(ledger_path,ledger)
        error = C2BudgetExhausted('C2 '+('global' if global_exhausted else 'case')+' budget exhausted before network')
        error.scope = 'global' if global_exhausted else 'case'
        raise error
    (folder/'request-wire.json').write_bytes(wire)
    if bundle is not None:save(folder/'request-bundle.json',bundle)
    entry=dict(request_id=key,case_id=case,request_kind=kind,condition='V-mobile',wire_sha256=digest,
               status='reserved',reserve_usd=estimate['reserve_usd'],estimated_usd=None,reserved_at_epoch=time.time())
    entries.append(entry);save(ledger_path,ledger)
    try:
        send_started = time.monotonic()
        try:
            raw=send(wire,key)
        finally:
            entry['api_wait_seconds'] = time.monotonic()-send_started
    except Exception as error:
        from urllib.error import HTTPError
        if isinstance(error, RequestNotSent):
            entry.update(status='not_sent', estimated_usd='0', error_type=type(error).__name__)
            if not (profile.model_target_boxes and kind=='verification'):
                ledger['status']='stopped_network_before_send'
            save(ledger_path,ledger)
            raise
        if kind=='verification' and isinstance(error,HTTPError) and error.code in (400,401,403,404,413,422,429):
            result=dict(status='invalid_verification',reason='provider_rejected_verification',reply=None,
                        estimated_usd=None,http_status=error.code,error_type=type(error).__name__)
            entry.update(result);save(ledger_path,ledger);save(folder/'delivery.json',result)
            return result
        entry.update(status='unknown',error_type=type(error).__name__)
        if auth.get('scope')==R33_STAGED_SCOPE or not (profile.model_target_boxes and kind=='verification'):
            ledger['status']='stopped_unknown_outcome'
        save(ledger_path,ledger)
        raise VerificationOutcomeUnknown('C2 provider outcome unknown; reserved cost retained') from None
    (folder/'response-raw.json').write_bytes(raw)
    if kind=='main':
        try:result=parse_response(raw,estimate,profile=profile)
        except Exception:result=dict(status='censored',reason='provider_contract',estimated_usd=None)
        if result['status']=='censored':ledger['status']='stopped_provider_contract'
    else:
        result=dict(status='invalid_verification',reason='invalid_verification_reply',estimated_usd=None,reply=None)
        try:
            body=json.loads(raw);usage=body.get('usage') or {};inp=usage.get('input_tokens');out=usage.get('output_tokens')
            if type(inp) is not int or type(out) is not int or not 0<=inp<=estimate['input_bound'] or not 0<=out<=estimate['output_cap']:
                raise ValueError('verification usage')
            result.update(usage=usage,estimated_usd=str(cost(inp,out,reserve_write=True)),provider_id=body.get('id'),reasoning_summaries=reasoning_summaries(body))
            if body.get('status')!='completed' or body.get('model')!=MODEL or body.get('service_tier') not in (None,'default') or not isinstance(body.get('id'),str) or not body['id'].startswith('resp_'):
                raise ValueError('verification provider envelope')
            texts=[p['text'] for item in body.get('output',[]) if item.get('type')=='message' for p in item.get('content',[]) if p.get('type')=='output_text']
            if len(texts)!=1:raise ValueError('verification output')
            reply=json.loads(texts[0])
            if profile.identity_revision == 2:
                validate_evidence_reply(reply)
            else:
                if set(reply)!={'verdict','reason'}:raise ValueError('verification fields')
                validate_identity_feedback(dict(target_id=1,**reply))
            result.update(status='valid',reason=None,reply=reply)
        except Exception as error:
            result['error_type']=type(error).__name__
    entry.update(result);save(ledger_path,ledger);save(folder/'delivery.json',result)
    return result

