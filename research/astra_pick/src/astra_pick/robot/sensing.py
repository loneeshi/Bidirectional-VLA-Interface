"""Static grouping of frozen robot code; algorithms and inheritance preserved."""

# ========================================================================
# Frozen source: eef_compact_config.py
"""Frozen first visual pilot. Changing any constant requires a new version."""
EEF_COMPACT_CONFIG_VERSION = 'eef-compact-640-v1'
RESOLUTION = 640
POSITIONS = (5, 55, 58)
MODEL = 'gpt-6-astra'
OUTPUT_CAP = 4000
TOOL_CAP = 25
STEP_CAP = 200
LOOK_CAP = 40
HEAD_LIMITS = ((-1.57, 1.57), (-.76, 1.45))
HEAD_ERROR = .02
HEAD_SPEED = .03
HEAD_STABLE_READINGS = 2
INPUT_BYTE_CAP = 14336
CONTEXT_LIMIT = 1050000
GPU_UUID = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'

IMAGE_GROUPS = 2
RECALL_GROUPS = 1


# ========================================================================
# Frozen source: eef_compact_tools.py
"""640 depth adapter; motion controller is exactly the validated visual v3 class."""
import numpy as np
from .eef_tools import vector, locate_point
from .executor import CoordinatedVisualToolsV2 as VisualTools


class ObservationDepth:
    """Only the latest sensor frame; root changes also invalidate saved pixels."""
    def __init__(self):
        self.frame = None

    def set(self, observation_id, qpos, cameras):
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError('observation_id required')
        if set(cameras) != {'head', 'hand'}: raise ValueError('two robot cameras required')
        copied = {}
        for name, camera in cameras.items():
            depth = np.asarray(camera['depth_mm'])
            if depth.shape != (RESOLUTION, RESOLUTION) or not np.issubdtype(depth.dtype, np.integer):
                raise ValueError('raw millimeter depth resolution/type')
            copied[name] = {k: np.asarray(v).copy() for k, v in camera.items()}
        self.frame = (observation_id, vector(qpos, 15).copy(), copied)

    def locate(self, camera, pixel, observation_id, measured):
        if self.frame is None or observation_id != self.frame[0]: raise ValueError('use latest observation_id')
        if not np.array_equal(vector(measured, 15), self.frame[1]): raise ValueError('observation is stale after movement')
        if camera not in ('head', 'hand'): raise ValueError('camera must be head or hand')
        if (not isinstance(pixel, list) or len(pixel) != 2 or
                any(type(p) is not int or not 0 <= p < RESOLUTION for p in pixel)):
            raise ValueError('pixel must be integer [x,y] within 0..639')
        frame = self.frame[2][camera]; x, y = pixel
        raw = frame['depth_mm']; depth = float(raw[y, x]) / 1000.
        result = locate_point(pixel, depth, frame['intrinsics'], frame['camera_in_base'], convention='sapien')
        area = raw[max(0, y-1):y+2, max(0, x-1):x+2]
        valid = area[area > 0].astype(float) / 1000.
        return {'accepted': True, 'readonly': True, 'reason': 'ok' if result['valid'] else 'invalid_depth',
                'valid': result['valid'], 'point_base_m': result['point_base'].tolist() if result['valid'] else None,
                'depth_m': depth if depth > 0 else None,
                'depth_spread_m': float(np.ptp(valid)) if len(valid) else None,
                'observation_id': observation_id}

