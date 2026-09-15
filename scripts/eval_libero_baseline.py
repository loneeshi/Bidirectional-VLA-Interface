"""Bounded single-task official-policy evaluation. No VLM or fallback controls."""
import argparse, collections, hashlib, json, pathlib, time
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy


def axis_angle(q):
    q = np.array(q, copy=True)
    q[3] = np.clip(q[3], -1., 1.)
    den = np.sqrt(1. - q[3] ** 2)
    return np.zeros(3) if den < 1e-8 else q[:3] * (2. * np.arccos(q[3]) / den)


def element(obs, instruction):
    def rgb(k):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[k][::-1, ::-1]),224,224))
    return {'observation/image':rgb('agentview_image'), 'observation/wrist_image':rgb('robot0_eye_in_hand_image'),
            'observation/state':np.concatenate((obs['robot0_eef_pos'],axis_angle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
            'prompt':instruction}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=8000);p.add_argument('--episodes',type=int,default=5)
    args=p.parse_args();assert 1<=args.episodes<=5
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(1)
    assert 'cream_cheese_box_and_the_butter' in task.name
    env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256)
    env.seed(7);np.random.seed(7);client=websocket_client_policy.WebsocketClientPolicy('127.0.0.1',args.port)
    summaries=[]
    try:
        for ep in range(args.episodes):
            env.reset();state=suite.get_task_init_states(1)[ep];obs=env.set_init_state(state)
            for _ in range(20):obs,_,_,_=env.step([0.]*6+[-1.])
            queue=collections.deque();frames=[];done=False;error=None;start=time.monotonic()
            with (out/f'episode{ep:03d}.jsonl').open('w') as log:
                for step in range(520):
                    inp=element(obs,task.language);frames.append(inp['observation/image'])
                    try:
                        if not queue:
                            pred=client.infer(inp);a=np.asarray(pred['actions'])
                            if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Invalid action chunk')
                            queue.extend(a[:5]);log.write(json.dumps({'event':'prediction','step':step,'actions':a.tolist(),'instruction':task.language})+'\n')
                        action=queue.popleft();obs,_,done,_=env.step(action.tolist())
                        log.write(json.dumps({'event':'action','step':step,'action':action.tolist(),'native_success':bool(done)})+'\n');log.flush()
                        if done:break
                    except Exception as exc:
                        error=repr(exc);break
            video=f'baseline-episode{ep:03d}'+('' if done else '-failed')+'.mp4'
            imageio.mimwrite(out/video,frames,fps=10)
            row={'episode':ep,'task_id':1,'initial_state_sha256':hashlib.sha256(np.asarray(state).tobytes()).hexdigest(),'success':bool(done),'steps':step+1,'error':error,'video':video,'elapsed_seconds':time.monotonic()-start,'vlm_calls':0,'max_steps':520,'wait_steps':20,'seed':7}
            summaries.append(row);(out/'summary.json').write_text(json.dumps(summaries,indent=2));print(row,flush=True)
            if error:raise RuntimeError(error)
    finally:env.close()


if __name__=='__main__':main()
