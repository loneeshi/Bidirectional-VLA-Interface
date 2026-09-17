"""Explicit timing bridge for legacy post-action learned progress.

The author current-observation target is (frame_index + offset)/(episode_len-1).
Legacy Fetch1199checkpoint instead predicts after the next action. This bridge
delays its first value until that action has executed; it does not make the
checkpoint equivalent to the author method or turn progress into a task oracle.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

from .progress_monitor import ProgressMonitor

LEGACY_FETCH_CHECKPOINT_SHA256 = '6c59df4acebca2cfb1a9ef8d37dc8d95834f22f112d93ec953a5e009c3b61c5a'


def require_post_action_contract(checkpoint_report, checkpoint_sha256):
    """Never silently reinterpret a new or unknown checkpoint's progress index."""
    contract = checkpoint_report.get('progress_label_contract')
    if contract is None and checkpoint_sha256 == LEGACY_FETCH_CHECKPOINT_SHA256:
        contract = 'post_action_position_v1'
    if contract != 'post_action_position_v1':
        raise ValueError('Checkpoint does not declare the supported post-action progress contract')
    return contract


def _step(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def current_observation_progress(frame_index, episode_len, offset=0):
    """Author target for a valid observation in an episode of at least 2 frames."""
    frame_index = _step(frame_index, "frame_index")
    offset = _step(offset, "offset")
    episode_len = _step(episode_len, "episode_len")
    if episode_len < 2 or frame_index + offset >= episode_len:
        raise ValueError("Target observation must lie in an episode of >= 2 frames")
    return (frame_index + offset) / (episode_len - 1)


@dataclass(frozen=True)
class PendingProgress:
    call_id: str
    prediction_step: int
    values: tuple[float, ...]

    @property
    def due_step(self):
        return self.prediction_step + 1


class PostActionProgressGate:
    """Consume one legacy prediction only at its corresponding observation.

    Steps count executed actions. Stage at step t, execute the first action,
    then observe at t+1. Remaining predicted values are intentionally unused.
    Callers must cancel before observe when native termination wins. Reset only
    clears gate state; callers own monitor replacement for a new skill family.
    """

    def __init__(self, monitor: ProgressMonitor):
        self.monitor = monitor
        self.reset()

    @property
    def pending(self):
        return self._pending

    @property
    def due_step(self):
        return None if self._pending is None else self._pending.due_step

    @property
    def consumed(self):
        """Whether the most recently staged prediction was consumed."""
        return self._consumed

    def stage(self, call_id, prediction_step, values):
        if self._pending is not None:
            raise ValueError("A progress prediction is already pending")
        prediction_step = _step(prediction_step, "prediction_step")
        if self._last_prediction_step is not None and prediction_step <= self._last_prediction_step:
            raise ValueError("Stale or duplicate prediction step")
        if isinstance(values, (str, bytes)):
            raise ValueError("Progress must be a nonempty numeric vector")
        try:
            vector = tuple(values)
        except TypeError as exc:
            raise ValueError("Progress must be a nonempty numeric vector") from exc
        if not vector or any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1
            for value in vector
        ):
            raise ValueError("Progress must be a finite vector in [0, 1]")
        self._pending = PendingProgress(call_id, prediction_step, tuple(float(v) for v in vector))
        self._last_prediction_step = prediction_step
        self._consumed = False

    def observe(self, call_id, observation_step):
        observation_step = _step(observation_step, "observation_step")
        pending = self._pending
        if pending is None:
            raise ValueError("No progress prediction is pending")
        if call_id != pending.call_id:
            raise ValueError("Observation belongs to a different call")
        if observation_step < pending.prediction_step:
            raise ValueError("Stale observation")
        if observation_step > pending.due_step:
            raise ValueError("Skipped the corresponding post-action observation")
        if observation_step < pending.due_step:
            return None
        reason = self.monitor.update(pending.values[0])
        self._pending = None
        self._consumed = True
        return reason

    def cancel(self):
        """Discard future progress without touching monitor history or counters."""
        self._pending = None
        self._consumed = False

    def reset(self):
        """Begin a new step sequence, leaving the caller-owned monitor intact."""
        self.cancel()
        self._last_prediction_step = None
