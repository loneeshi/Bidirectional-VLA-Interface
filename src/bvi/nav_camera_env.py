"""Declared observation augmentation: forward-facing robot-mounted RGB camera.

Original head/wrist cameras and all physics/task checks remain unchanged.
This extra sensor must be disclosed in comparisons with official observations.
"""
import math
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env
from mshab.envs.sequential_task import SequentialTaskEnv


@register_env('BVISequentialNavCamera-v0')
class SequentialNavCameraEnv(SequentialTaskEnv):
    @property
    def _default_sensor_configs(self):
        return [CameraConfig(uid='fetch_nav',
            pose=Pose.create_from_pq(p=[.35,0,1.9],q=[math.cos(.075),0,math.sin(.075),0]),
            width=448,height=256,fov=math.pi/2,near=.01,far=100,
            mount=self.agent.base_link)]
