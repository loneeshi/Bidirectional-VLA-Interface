"""Local thresholds, evaluated once per prediction, never per simulator frame.

STATUS: active — feedback legacy facade

Related implementation reference (not verified provenance for these constants):
https://github.com/cxliu0314/openpi/tree/f4eb160ba52b22c1e85fe432de59c24bbbac6187
The paper does not specify these numeric defaults. Treat them as local settings
until an exact upstream file/line mapping is verified; not recovered author defaults.
"""

from dataclasses import dataclass, field
import math

THRESHOLDS = {"reach": 0.9, "grasp": 0.6, "move": 0.9, "release": 0.6}


@dataclass
class ProgressMonitor:
    family: str
    invocation_index: int
    replan_cooldown: int = 0
    history: list = field(default_factory=list)
    above: int = 0

    def update(self, value):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid learned progress")
        self.history.append(value)
        self.above = self.above + 1 if value >= THRESHOLDS[self.family] else 0
        self.replan_cooldown = max(0, self.replan_cooldown - 1)
        if self.above >= 2:
            return "learned_threshold"
        if (
            len(self.history) > 3
            and self.invocation_index > 0
            and self.replan_cooldown == 0
        ):
            if max(self.history) - value > 0.03:
                return "learned_drop"
            if len(self.history) >= 10 and value - self.history[0] < 0.03:
                return "learned_stagnation"
        return None
