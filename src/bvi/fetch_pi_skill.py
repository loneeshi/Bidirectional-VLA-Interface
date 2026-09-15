"""Execute only a Fetch-trained pi05 checkpoint, with bounded action chunks."""
from collections import deque
import io

from .mshab_adapter import benchmark_feedback, jsonable
from .protocol import ProtocolError

CONVENTION='Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity'
JOINT_NAMES=['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint',
    'torso_lift_joint','head_pan_joint','shoulder_pan_joint','head_tilt_joint',
    'shoulder_lift_joint','upperarm_roll_joint','elbow_flex_joint','forearm_roll_joint',
    'wrist_flex_joint','wrist_roll_joint','r_gripper_finger_joint','l_gripper_finger_joint']


class FetchPiSkill:
    def __init__(self,name,adapter,client,max_predictions=100,chunk_steps=3):
        if name not in ('pick','place') or not 1<=chunk_steps<=10 or max_predictions<1:
            raise ProtocolError('Invalid Fetch pi skill configuration')
        metadata=client.metadata
        if any(metadata.get(k)!=v for k,v in {
            'robot':'fetch','action_dim':13,'action_convention':CONVENTION,
            'state_conditioning':True}.items()):
            raise ProtocolError('A Fetch-trained model with matching controller metadata is required')
        self.state_dim=metadata.get('state_dim')
        expected=['qpos','qvel'] if self.state_dim==30 else ['qpos']
        if self.state_dim not in (15,30) or metadata.get('state_components',['qpos'])!=expected:
            raise ProtocolError('Unsupported or undeclared Fetch state components')
        names=[j.name for j in adapter.uenv.agent.robot.active_joints]
        if names!=JOINT_NAMES: raise ProtocolError('Fetch state joint ordering differs from training')
        self.name,self.adapter,self.client=name,adapter,client
        self.max_predictions,self.chunk_steps=max_predictions,chunk_steps
        self.total_predictions=0
        self.actions=deque()
        self.index=None

    def start(self,request,observation):
        index=int(observation.metadata['subtask_index'])
        if request.skill!=self.name or self.adapter.original_plan.subtasks[index].type!=self.name:
            raise ProtocolError('Fetch pi skill does not match requested subtask')
        self.index,self.call_id=index,request.call_id
        from .mshab_adapter import describe_target
        self.prompt=describe_target(self.adapter.original_plan,index).description
        self.actions.clear()
        self.adapter.logger.emit('fetch_pi_started',call_id=self.call_id,skill=self.name,
            prompt=self.prompt,model_metadata=self.client.metadata,chunk_steps=self.chunk_steps)

    def act(self,observation):
        if self.index is None: raise ProtocolError('Missing skill start')
        if not self.actions:
            import numpy as np
            from PIL import Image
            if self.total_predictions>=self.max_predictions: raise ProtocolError('Fetch pi prediction cap reached')
            visual=self.adapter.observe()
            if visual.frame_id!=observation.frame_id: raise ProtocolError('Stale Fetch camera')
            def pixels(camera):
                image=next(x for x in visual.images if x.camera==camera)
                return np.asarray(Image.open(io.BytesIO(image.data)).convert('RGB'))
            state=np.asarray(jsonable(self.adapter.uenv.agent.robot.qpos)[0],dtype=np.float32)
            if self.state_dim==30:
                state=np.concatenate([state,np.asarray(jsonable(self.adapter.uenv.agent.robot.qvel)[0],dtype=np.float32)])
            if state.shape!=(self.state_dim,) or not np.isfinite(state).all(): raise ProtocolError('Invalid Fetch state')
            self.total_predictions+=1
            self.adapter.save_observation_images()
            self.adapter.logger.emit('fetch_pi_inference_started',call_id=self.call_id,
                attempt=self.total_predictions,frame_id=observation.frame_id,state=state.tolist())
            rows=self.client.infer({'observation/state':state,'observation/image':pixels('fetch_head'),
                'observation/wrist_image':pixels('fetch_hand'),'prompt':self.prompt},action_dim=13)
            self.adapter.logger.emit('fetch_pi_prediction',call_id=self.call_id,
                frame_id=observation.frame_id,actions=rows)
            self.actions.extend(rows[:self.chunk_steps])
        raw=self.actions.popleft()
        bounded=tuple(max(-1.,min(1.,v)) for v in raw)
        self.adapter.logger.emit('fetch_pi_action',call_id=self.call_id,raw=raw,action=bounded,
                                 clipped=raw!=bounded)
        return self.adapter.action_bounds.validate(bounded)

    def feedback(self,request,transition):
        return benchmark_feedback(request,transition,self.index)
