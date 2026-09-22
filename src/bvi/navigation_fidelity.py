"""Opt-in navigation audit controls; historical LightNavSkill stays unchanged.

STATUS: active — evaluation

These are diagnostic factors, not a learned progress/recovery mechanism. The
caller must assign a fresh episode token on every environment reset, including
repeated seeds, and clear any queued actions independently of frame history.
"""
from dataclasses import dataclass
import hashlib
import math

from .protocol import ProtocolError
from .vla_clients import finite_rows


def first_waypoint_velocity(rows, dt, linear_limit=.6, angular_limit=1.2):
    """Official first-row choice; a zero first row must not select a later row."""
    rows = finite_rows(rows, 3)
    if not all(math.isfinite(x) and x > 0 for x in (dt, linear_limit, angular_limit)):
        raise ProtocolError('Finite positive control period and limits required')
    forward, lateral, yaw = rows[0]
    return (max(-linear_limit, min(linear_limit, forward / dt)),
            max(-angular_limit, min(angular_limit, yaw / dt)))


@dataclass(frozen=True)
class HistoryDecision:
    reset: bool
    reason: str
    instruction_changed: bool
    instruction_sha256: str


class NavigationHistoryAudit:
    """Make call-vs-goal reset scope explicit without inventing goal semantics."""
    def __init__(self, client, scope='call'):
        if scope not in ('call', 'goal'):
            raise ProtocolError('History scope must be call or goal')
        self.client, self.scope = client, scope
        self.key = self.instruction = None

    def begin(self, *, episode_token, goal_token, instruction):
        if any(not isinstance(x, str) or not x.strip()
               for x in (episode_token, goal_token, instruction)):
            raise ProtocolError('Explicit episode, goal and instruction required')
        key = (episode_token, goal_token)
        changed = self.key == key and self.instruction != instruction
        reset = self.scope == 'call' or self.key != key
        reason = 'call_boundary' if self.scope == 'call' else 'goal_or_episode_boundary' if reset else 'same_goal_continuation'
        digest = hashlib.sha256(instruction.encode('utf-8')).hexdigest()
        if reset:
            self.client.reset()  # Commit scope only after reset succeeds.
        self.key, self.instruction = key, instruction
        return HistoryDecision(reset, reason, changed, digest)

    def invalidate(self):
        """After interrupted/uncertain transport, force a fresh session reset."""
        self.key = self.instruction = None
