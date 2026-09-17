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
    def __init__(self,name,adapter,client,max_predictions=100,chunk_steps=3,ensemble_samples=1,instructions=None):
        if name not in ('pick','place') or not 1<=chunk_steps<=10 or max_predictions<1 or not 1<=ensemble_samples<=8:
            raise ProtocolError('Invalid Fetch pi skill configuration')
        metadata=client.metadata
        if any(metadata.get(k)!=v for k,v in {
            'robot':'fetch','action_dim':13,'action_convention':CONVENTION,
            'state_conditioning':True}.items()):
            raise ProtocolError('A Fetch-trained model with matching controller metadata is required')
        self.state_dim=metadata.get('state_dim')
        expected=['native_qpos12','native_qvel12'] if self.state_dim==24 else (['qpos','qvel'] if self.state_dim==30 else ['qpos'])
        if self.state_dim not in (15,24,30) or metadata.get('state_components',['qpos'])!=expected:
            raise ProtocolError('Unsupported or undeclared Fetch state components')
        self.base_reference=metadata.get('base_position_reference','world')
        if self.base_reference not in ('world','skill_start_xy'):
            raise ProtocolError('Unsupported Fetch base position reference')
        self.base_xy_origin=None
        self.base_camera=metadata.get('base_camera','fetch_head')
        self.wrist_camera=metadata.get('wrist_camera','fetch_hand')
        if self.base_camera not in ('fetch_head','fetch_workspace') or self.wrist_camera!='fetch_hand':
            raise ProtocolError('Unsupported Fetch camera contract')
        if self.state_dim==24 and (metadata.get('state_source')!='env_native_agent'
                or self.base_reference!='world' or self.base_camera!='fetch_head'):
            raise ProtocolError('Native24 requires declared env_native_agent/head camera and no relative-base transform')
        names=[j.name for j in adapter.uenv.agent.robot.active_joints]
        if names!=JOINT_NAMES: raise ProtocolError('Fetch state joint ordering differs from training')
        self.name,self.adapter,self.client=name,adapter,client
        self.max_predictions,self.chunk_steps=max_predictions,chunk_steps
        self.ensemble_samples=ensemble_samples
        if instructions is not None and (not isinstance(instructions,dict) or not instructions or not all(
                isinstance(k,str) and k.isdigit() and isinstance(v,str) and v.strip() for k,v in instructions.items())):
            raise ProtocolError('Manipulation instructions must map indices to nonempty strings')
        self.instructions=None if instructions is None else dict(instructions)
        self.total_predictions=0
        self.actions=deque()
        self.index=None

    def start(self,request,observation):
        index=int(observation.metadata['subtask_index'])
        if request.skill!=self.name or self.adapter.original_plan.subtasks[index].type!=self.name:
            raise ProtocolError('Fetch pi skill does not match requested subtask')
        new_subtask = self.index != index
        self.index,self.call_id=index,request.call_id
        from .mshab_adapter import describe_target
        if getattr(request, 'instruction', None) is not None:
            self.prompt=request.instruction
        elif self.instructions is not None:
            if str(index) not in self.instructions:raise ProtocolError('Missing explicit manipulation instruction')
            self.prompt=self.instructions[str(index)]
        else:
            self.prompt=describe_target(self.adapter.original_plan,index).description
        if self.state_dim!=24 and (new_subtask or self.base_xy_origin is None):
            self.base_xy_origin=tuple(float(x) for x in jsonable(self.adapter.uenv.agent.robot.qpos)[0][:2])
        self.actions.clear()
        self.adapter.logger.emit('fetch_pi_started',call_id=self.call_id,skill=self.name,
            prompt=self.prompt,model_metadata=self.client.metadata,chunk_steps=self.chunk_steps,
            base_xy_origin=self.base_xy_origin,ensemble_samples=self.ensemble_samples,
            instruction_source='vlm_invocation' if getattr(request,'instruction',None) is not None else ('explicit_scene_config' if self.instructions is not None else 'task_plan_id_template'))

    def act(self,observation):
        if self.index is None: raise ProtocolError('Missing skill start')
        if not self.actions:
            import numpy as np
            from PIL import Image
            if self.total_predictions+self.ensemble_samples>self.max_predictions: raise ProtocolError('Fetch pi prediction cap reached')
            visual=self.adapter.observe()
            if visual.frame_id!=observation.frame_id: raise ProtocolError('Stale Fetch camera')
            def pixels(camera):
                image=next((x for x in visual.images if x.camera==camera),None)
                if image is None:raise ProtocolError(f'Required Fetch camera is missing: {camera}')
                return np.asarray(Image.open(io.BytesIO(image.data)).convert('RGB'))
            head,hand=pixels(self.base_camera),pixels(self.wrist_camera)
            if self.state_dim==24:
                from .official_fetch_data import native_policy_observation
                agent_obs=self.adapter.uenv._get_obs_agent()
                native=native_policy_observation(
                    np.asarray(jsonable(agent_obs['qpos']))[0],
                    np.asarray(jsonable(agent_obs['qvel']))[0],head,hand,self.prompt)
                state=native['state']
            else:
                state=np.asarray(jsonable(self.adapter.uenv.agent.robot.qpos)[0],dtype=np.float32)
                if self.base_reference=='skill_start_xy':
                    if self.base_xy_origin is None: raise ProtocolError('Missing measured skill-start base origin')
                    state[:2]-=np.asarray(self.base_xy_origin,np.float32)
                if self.state_dim==30:
                    state=np.concatenate([state,np.asarray(jsonable(self.adapter.uenv.agent.robot.qvel)[0],dtype=np.float32)])
            if state.shape!=(self.state_dim,) or not np.isfinite(state).all(): raise ProtocolError('Invalid Fetch state')
            self.adapter.save_observation_images()
            inputs={'observation/state':state,'observation/image':head,
                    'observation/wrist_image':hand,'prompt':self.prompt}
            samples=[]
            for _ in range(self.ensemble_samples):
                self.total_predictions+=1
                self.adapter.logger.emit('fetch_pi_inference_started',call_id=self.call_id,
                    attempt=self.total_predictions,frame_id=observation.frame_id,state=state.tolist())
                rows=self.client.infer(inputs,action_dim=13)
                self.adapter.logger.emit('fetch_pi_prediction',call_id=self.call_id,
                    frame_id=observation.frame_id,actions=rows)
                samples.append(rows)
            if self.ensemble_samples>1:
                if len({len(x) for x in samples})!=1: raise ProtocolError('Ensemble action horizons differ')
                rows=tuple(tuple(float(v) for v in row) for row in np.mean(np.asarray(samples,np.float64),axis=0))
                self.adapter.logger.emit('fetch_pi_ensemble_prediction',call_id=self.call_id,
                    frame_id=observation.frame_id,samples=self.ensemble_samples,actions=rows)
            self.actions.extend(rows[:self.chunk_steps])
        raw=self.actions.popleft()
        bounded=tuple(max(-1.,min(1.,v)) for v in raw)
        self.adapter.logger.emit('fetch_pi_action',call_id=self.call_id,raw=raw,action=bounded,
                                 clipped=raw!=bounded)
        return self.adapter.action_bounds.validate(bounded)

    def feedback(self,request,transition):
        return benchmark_feedback(request,transition,self.index)
