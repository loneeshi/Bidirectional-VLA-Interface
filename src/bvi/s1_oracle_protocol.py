"""Causal S1-IA prompt switching that matches the frozen Pick training windows.

STATUS: frozen — historical training and diagnostics (retained)

Uses simulator predicates only for instruction selection; never claims learned
feedback or changes the environment's native success/termination conditions.
"""
from dataclasses import dataclass
import math

from .ia_call_predicates import INSTRUCTIONS

PROTOCOL = 'ia_oracle_train_windows_v1'


@dataclass
class PickOracle:
    family: str = 'reach'
    observation_index: int = 0
    previous_held: bool = False
    local_grasp_streak: int = 0
    near_index: int | None = None
    hold_index: int | None = None
    dropped_after_hold: bool = False

    @property
    def instruction(self):
        return INSTRUCTIONS[self.family]

    def after_action(self, tcp_object_distance, held):
        if not math.isfinite(tcp_object_distance) or tcp_object_distance < 0:
            raise ValueError('Finite nonnegative physical distance required')
        self.observation_index += 1
        before = self.family
        if self.family == 'reach':
            # fetch_segments: near is the first t>=1 with distance[t]<=.08,
            # conditioned on held[t-1], not held[t].
            if tcp_object_distance <= .08 and not self.previous_held:
                self.family = 'grasp'
                self.near_index = self.observation_index
                self.local_grasp_streak = 0
        elif self.family == 'grasp':
            # Exclude held[near] from the first three post-near observations.
            self.local_grasp_streak = self.local_grasp_streak + 1 if held else 0
            if self.local_grasp_streak >= 3:
                self.family = 'move'
                self.hold_index = self.observation_index
        elif self.family == 'move':
            self.dropped_after_hold |= not bool(held)
        else:
            raise ValueError('Unknown oracle phase')
        self.previous_held = bool(held)
        if before == self.family:
            return None
        return dict(observation_index=self.observation_index, from_family=before,
                    to_family=self.family, next_instruction=self.instruction,
                    action_queue_length_before=0, action_queue_length_after=0,
                    queue_policy='only_first_action_used_other_nine_discarded_every_prediction',
                    model_rng_reset=False, feedback='simulator_oracle_not_learned')
