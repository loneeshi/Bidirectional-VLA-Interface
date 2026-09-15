"""Author thresholds, evaluated once per policy prediction, never per sim frame."""
from dataclasses import dataclass, field
import math

THRESHOLDS={'reach':.9,'grasp':.6,'move':.9,'release':.6}

@dataclass
class ProgressMonitor:
    family: str
    invocation_index: int
    replan_cooldown: int = 0
    history: list = field(default_factory=list)
    above: int = 0

    def update(self, value):
        if not math.isfinite(value) or not 0 <= value <= 1:raise ValueError('Invalid learned progress')
        self.history.append(value)
        self.above=self.above+1 if value>=THRESHOLDS[self.family] else 0
        self.replan_cooldown=max(0,self.replan_cooldown-1)
        if self.above>=2:return 'learned_threshold'
        if len(self.history)>3 and self.invocation_index>0 and self.replan_cooldown==0:
            if max(self.history)-value>.03:return 'learned_drop'
            if len(self.history)>=10 and value-self.history[0]<.03:return 'learned_stagnation'
        return None
