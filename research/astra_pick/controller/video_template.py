"""Multi-camera execution video renderer. CPU-only; missing signals stay missing.

Input: JSON {fps, frames:[{scene,head,hand,aux,actions,step,condition,case_id,
phase,outcome,cumulative_force,trajectory_world,world_to_camera,K,...}]}.
Image paths are relative to the JSON. Camera convention: OpenCV +Z forward.
Evaluation camera data never belongs in controller/model inputs.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import imageio.v2 as imageio

HERE=Path(__file__).resolve().parent
W,H=1480,800
def font(size):
    for name in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)
FONT=font(18)
SMALL=font(15)
LABELS=['Shoulder pan','Shoulder lift','Upperarm roll','Elbow flex','Forearm roll','Wrist flex','Wrist roll','Gripper','Head pan','Head tilt','Torso lift','Base forward','Base turn']

def full_action_labels(mapping):
    groups=dict(arm=[name+' pos' for name in LABELS[:7]]+[name+' vel' for name in LABELS[:7]],
                gripper=['Gripper'],head=['Head pan','Head tilt'],torso=['Torso velocity'],base=['Base forward','Base turn'])
    if set(mapping)!=set(groups):raise ValueError('Unknown full action groups')
    labels=[None]*20
    for name,(start,end) in mapping.items():
        if type(start) is not int or type(end) is not int or start<0 or end>20 or end-start!=len(groups[name]):raise ValueError('Invalid action label mapping')
        if any(x is not None for x in labels[start:end]):raise ValueError('Overlapping action labels')
        labels[start:end]=groups[name]
    if any(x is None for x in labels):raise ValueError('Missing action labels')
    return labels
def project(points,T,K):
    xyz=np.asarray(points,float);T=np.asarray(T,float);K=np.asarray(K,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or T.shape!=(4,4) or K.shape!=(3,3):raise ValueError('camera/trajectory dimensions')
    if not all(np.isfinite(x).all() for x in [xyz,T,K]):raise ValueError('invalid camera/trajectory')
    camera=(T@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
    valid=camera[:,2]>1e-6
    uv=np.full((len(xyz),2),np.nan)
    projected=(K@camera[valid].T).T
    uv[valid]=projected[:,:2]/projected[:,2,None]
    return uv,valid
def panel(canvas,box,path,label,root,frame=None):
    x,y,w,h=box;draw=ImageDraw.Draw(canvas)
    if path:
        im=Image.open(root/path).convert('RGB')
        if frame and frame.get('trajectory_world') is not None:
            # Reproject ALL past TCP points through this frame's camera.
            if frame.get('world_to_camera') is None or frame.get('K') is None:
                raise ValueError('trajectory requires current camera calibration')
            uv,valid=project(frame['trajectory_world'],frame['world_to_camera'],frame['K'])
            pen=ImageDraw.Draw(im)
            for i in range(1,len(uv)):
                if valid[i-1] and valid[i]:pen.line([tuple(uv[i-1]),tuple(uv[i])],fill='#ffc857',width=4)
        thumb=ImageOps.contain(im,(w,h));canvas.paste(thumb,(x+(w-thumb.width)//2,y+(h-thumb.height)//2))
    else:
        draw.rounded_rectangle((x,y,x+w,y+h),radius=8,fill='#243444')
        draw.text((x+15,y+h//2),label+' / no frame',font=SMALL,fill='#aebdcc')
    draw.rectangle((x,y,x+w,y+28),fill='#172330')
    draw.text((x+8,y+4),label,font=SMALL,fill='white')
def render(frame,root):
    canvas=Image.new('RGB',(W,H),'#f3f5f8');d=ImageDraw.Draw(canvas)
    header=f"{frame.get('case_id','UNBOUND')} | {frame.get('condition','TEMPLATE')} | step {frame.get('step','--')} | {frame.get('outcome','NOT RUN')}"
    d.text((24,15),header,font=FONT,fill='#172330')
    panel(canvas,(24,55,720,710),frame.get('scene'),'SCENE - evaluation only',root,frame)
    for name,y in [('head',55),('hand',295),('aux',535)]:
        panel(canvas,(754,y,220,230),frame.get(name),name.upper(),root)
    d.text((1000,62),'EXECUTION',font=FONT,fill='#172330')
    d.text((1000,100),'Phase: '+str(frame.get('phase','--')),font=SMALL,fill='#172330')
    d.text((1000,132),'Cumulative force: '+str(frame.get('cumulative_force','unavailable')),font=SMALL,fill='#172330')
    d.text((1000,164),'Trajectory: '+('X-RAY / analysis only' if frame.get('trajectory_world') is not None else 'unavailable'),font=SMALL,fill='#785500')
    d.text((1000,209),'Executed normalized actions',font=FONT,fill='#172330')
    actions=frame.get('actions')
    labels=frame.get('action_labels',LABELS)
    if not isinstance(labels,list) or len(labels) not in (13,20) or not all(isinstance(x,str) and x for x in labels):raise ValueError('invalid action labels')
    if actions is not None:
        actions=np.asarray(actions,float)
        if actions.shape!=(len(labels),) or not np.isfinite(actions).all() or np.max(abs(actions))>1+1e-8:raise ValueError('invalid actions')
    for i,label in enumerate(labels):
        y=251+i*(23 if len(labels)==20 else 35);d.text((1000,y),label,font=SMALL,fill='#394858')
        d.line((1300,y,1300,y+22),fill='#9aa7b4')
        if actions is None:d.text((1370,y),'--',font=SMALL,fill='#394858')
        else:
            v=float(actions[i]);end=1300+v*65
            d.rectangle((min(1300,end),y+4,max(1300,end),y+19),fill='#3276b6')
            d.text((1380,y),f'{v:+.2f}',font=SMALL,fill='#394858')
    d.text((1000,730),'Timeline: recorded physical step',font=SMALL,fill='#394858')
    if frame.get('template'):
        d.rectangle((24,355,974,405),fill='#785500')
        d.text((50,370),'TEMPLATE ONLY - NO ROBOT EXECUTION OR SUCCESS CLAIM',font=FONT,fill='white')
    return canvas
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path,default=HERE/'execution-template.mp4')
    args=parser.parse_args()
    if args.input:
        data=json.loads(args.input.read_text());root=args.input.parent
        steps=[f.get('step') for f in data['frames']]
        if not all(type(s) is int for s in steps) or any(b!=a+1 for a,b in zip(steps,steps[1:])):
            raise ValueError('provide consecutive recorded steps; missing images must remain explicit')
    else:
        data=dict(fps=20,frames=[dict(template=True,step='--',phase='awaiting recorded data') for _ in range(40)]);root=HERE
    fps=data['fps']
    if not isinstance(fps,(int,float)) or not 0<fps<=120 or not data['frames']:raise ValueError('invalid video timeline')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(args.output,fps=fps,codec='libx264',macro_block_size=1) as writer:
        for i,frame in enumerate(data['frames']):
            im=render(frame,root);writer.append_data(np.asarray(im))
            if i==0:im.save(args.output.with_suffix('.png'))
    receipt=dict(filename=args.output.name,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        frames=len(data['frames']),fps=fps,template=args.input is None,trajectory='X-ray; no occlusion claims',
        source=str(args.input) if args.input else None)
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
