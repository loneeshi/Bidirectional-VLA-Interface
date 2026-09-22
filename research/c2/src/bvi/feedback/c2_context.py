"""Deterministic, source-bound C2 context compilation; no extra model calls.

STATUS: active — feedback
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
from pathlib import Path

from ..protocol import ImageFrame, ProtocolError

VARIANTS = ('basic', 'trace', 'experience')


def contact_sheet(rows, camera):
    from PIL import Image, ImageDraw
    if not rows or len(rows) > 6: raise ProtocolError('Contact sheet requires 1..6 frames')
    sheet = Image.new('RGB', (256*min(3,len(rows)), 280*math.ceil(len(rows)/3)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        image = Image.open(io.BytesIO(row['image'].data)).convert('RGB')
        image.thumbnail((256,256))
        x,y = (i%3)*256,(i//3)*280
        sheet.paste(image,(x,y))
        draw.text((x+2,y+257), f"step {row['step']} | {row['event'][:20]}",fill='black')
    buffer = io.BytesIO(); sheet.save(buffer,format='PNG')
    return ImageFrame(camera,buffer.getvalue())


class CaseBank:
    def __init__(self, path, excluded_uids):
        path = Path(path); self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cases = json.loads(path.read_text(encoding='utf-8'))['cases']
        if not self.cases: raise ProtocolError('Experience bank is empty')
        seen = set()
        for case in self.cases:
            identity = case['parent_uid'],case['call_id']
            if case['parent_uid'] in excluded_uids or identity in seen:
                raise ProtocolError('Leaked or duplicate parent/call in experience bank')
            seen.add(identity)
            if case['skill'] not in ('pick','place') or not case.get('object_category'):
                raise ProtocolError('Invalid historical skill/category')
            if len(case.get('checkpoint_sha256','')) != 64 or not case.get('source_sha256'):
                raise ProtocolError('Historical case lacks source/checkpoint identity')
            if not isinstance(case.get('start_state'),dict) or not isinstance(case.get('result'),dict):
                raise ProtocolError('Historical case lacks actual state/result')
            if not 1 <= len(case.get('frames',[])) <= 4:
                raise ProtocolError('Historical case requires 1..4 source frames')
            last = -1
            for f in case['frames']:
                p = (path.parent/f['path']).resolve()
                if not p.is_relative_to(path.parent.resolve()):
                    raise ProtocolError('Historical image path escaped bank directory')
                data = p.read_bytes()
                if hashlib.sha256(data).hexdigest() != f['sha256']:
                    raise ProtocolError('Historical frame SHA mismatch')
                if type(f['step']) is not int or f['step'] <= last:
                    raise ProtocolError('Historical frames must have unique chronological sim steps')
                last = f['step']
                frame = ImageFrame(f.get('camera','historical'),data)
                frame.validate(); f['_image'] = frame

    def retrieve(self, skill, category, state, checkpoint_sha256, limit=2):
        # Outcome is deliberately absent from the retrieval key.
        candidates = [c for c in self.cases if c['skill']==skill and
                      c['object_category']==category and c['checkpoint_sha256']==checkpoint_sha256]
        def key(c):
            a,b = state.get('distance'),c['start_state'].get('distance')
            numeric = lambda v: type(v) in (float,int) and math.isfinite(v)
            distance = abs(a-b) if numeric(a) and numeric(b) else float('inf')
            return distance,c['parent_uid'],c['call_id']
        return sorted(candidates,key=key)[:limit]


class InvocationFrames:
    """Retain start/end, first two grasp transitions and closest-distance frame."""
    def __init__(self):
        self.rows = {}; self.first = self.last = self.minimum = None
        self.transitions = []; self.previous_grasp = None
        self.min_distance = float('inf')

    def update(self, observation, state, event=None):
        step = observation.sim_step
        grasp = state.get('grasped')
        changed = self.previous_grasp is not None and grasp != self.previous_grasp
        distance = state.get('distance',float('inf'))
        closest = type(distance) in (int,float) and math.isfinite(distance) and distance < self.min_distance
        self.previous_grasp = grasp
        self.first = step if self.first is None else self.first
        self.last = step
        if changed and len(self.transitions)<2: self.transitions.append(step)
        if closest: self.minimum = step; self.min_distance = distance
        image = next((i for i in observation.images if i.camera=='fetch_workspace'),None)
        if image is None: image = next(iter(observation.images),None)
        if image is None: return
        self.rows[step] = dict(step=step, image=image, state=deepcopy(state),
                              event=event or ('grasp_change' if changed else 'sample'))
        keep = {self.first,self.last,self.minimum,*self.transitions}
        self.rows = {k:v for k,v in self.rows.items() if k in keep}

    def finish(self):
        rows = [deepcopy(v) for k,v in sorted(self.rows.items())]
        if rows:
            rows[0]['event']='start'; rows[-1]['event']='end'
        return rows


class ContextSkill:
    """Observer only: never replaces actions, native feedback or completion."""
    def __init__(self, skill, adapter): self.skill,self.adapter=skill,adapter
    def start(self, request, observation):
        self.skill.start(request,observation)
        self.index=self.adapter.resolve_request_index(request)
        self.trace=InvocationFrames()
        self.trace.update(self.adapter.observe(),self.adapter.progress_snapshot(self.index),'start')
    def act(self, observation): return self.skill.act(observation)
    def feedback(self, request, transition):
        f=self.skill.feedback(request,transition)
        # Capture from the same cached raw observation; never call evaluate/get_obs.
        self.trace.update(self.adapter.observe(),self.adapter.progress_snapshot(self.index))
        return f


class C2ContextCompiler:
    def __init__(self, variant, output, bank=None, categories=None, checkpoint_hashes=None):
        if variant not in VARIANTS: raise ValueError('Unknown C2 feedback variant')
        if (variant=='experience') != (bank is not None): raise ValueError('Only experience consumes a bank')
        self.variant,self.output,self.bank=variant,Path(output),bank
        self.categories=categories or {};self.checkpoints=checkpoint_hashes or {}
        self.latest=None

    def note(self, request, result, frames):
        if request.skill not in ('pick','place'): return
        self.latest={'skill':request.skill,'target_id':request.target_id,'call_id':request.call_id,
                     'reason':result.feedback.reason,'rows':frames}
        # Save independently of variant to keep instrumentation identical.
        directory=self.output/'invocation-frames'/hashlib.sha256(request.call_id.encode()).hexdigest()[:16]
        directory.mkdir(parents=True,exist_ok=False)
        manifest=[]
        for i,r in enumerate(frames):
            filename=f'{i:02d}.png'; (directory/filename).write_bytes(r['image'].data)
            manifest.append({k:v for k,v in r.items() if k!='image'} | {
                'path':filename,'camera':r['image'].camera,
                'sha256':hashlib.sha256(r['image'].data).hexdigest()})
        (directory/'frames.json').write_text(json.dumps(manifest,allow_nan=False),encoding='utf-8')

    def compile(self, context, observation, history):
        context=deepcopy(context)
        # Avoid indirect leakage through trajectory digests or nested recovery snapshots.
        public=('call_id','skill','target_id','instruction','feedback','steps','recovery_goal')
        context['feedback_history']=[{k:(asdict(v) if is_dataclass(v) else deepcopy(v))
                                     for k,v in row.items() if k in public} for row in history]
        images=list(observation.images)
        extra={'protocol':'c2-pose-feedback/3','variant':self.variant,
               'trajectory':{'available':False,'reason':'no_prior_manipulation'},
               'experience':{'available':False,'reason':'not_enabled'}}
        if self.variant!='basic' and self.latest is not None:
            rows=self.latest['rows']
            extra['trajectory']={k:v for k,v in self.latest.items() if k!='rows'}
            extra['trajectory'].update(available=bool(rows), frames=[{
                k:v for k,v in r.items() if k!='image'} for r in rows])
            if rows: images.append(contact_sheet(rows,'current_invocation_trace'))
            if self.bank:
                target=self.latest['target_id']; skill=self.latest['skill']
                start=rows[0]['state'] if rows else {}
                matches=self.bank.retrieve(skill,self.categories.get(target),start,
                                           self.checkpoints.get((skill,target)))
                examples=[]
                for index,c in enumerate(matches):
                    frames=[dict(step=f['step'],event=f.get('event','historical'),image=f['_image'])
                            for f in c['frames']]
                    images.append(contact_sheet(frames,f'historical_case_{index}'))
                    examples.append({k:deepcopy(v) for k,v in c.items() if k!='frames'} |
                                    {'frames':[{k:v for k,v in f.items() if k!='_image'} for f in c['frames']]})
                extra['experience']={'available':bool(examples),'reason':None if examples else 'no_matching_case',
                                     'bank_sha256':self.bank.sha256,'cases':examples}
        context['c2_feedback']=extra
        context['image_order']=[i.camera for i in images]
        directory=self.output/'compiled-context'/observation.frame_id
        directory.mkdir(parents=True,exist_ok=True)
        for i,frame in enumerate(images): (directory/f'{i:02d}.png').write_bytes(frame.data)
        # Full JSON prompt/schema are also saved by VLMCoordinator before dispatch.
        return context,tuple(images)
