"""Explicit learner-prefix / expert-recovery training collection, never an eval policy."""
from .protocol import ProtocolError


class RecoveryCollectionSkill:
    def __init__(self,learner,teacher,adapter,prefix_steps):
        if type(prefix_steps) is not int or not 0<=prefix_steps<=50:
            raise ProtocolError('Recovery collection prefix must be an integer in [0,50]')
        self.learner,self.teacher,self.adapter=learner,teacher,adapter
        self.prefix_steps=prefix_steps
        self.name=learner.name

    def start(self,request,observation):
        self.request=request;self.steps=0;self.teacher_started=False
        self.adapter.record_demonstrations=False
        self.learner.start(request,observation)
        self.adapter.logger.emit('recovery_collection_started',call_id=request.call_id,
            skill=self.name,learner_prefix_steps=self.prefix_steps,evaluation_eligible=False)

    def act(self,observation):
        if self.steps<self.prefix_steps:
            self.adapter.record_demonstrations=False
            action=self.learner.act(observation)
        else:
            if not self.teacher_started:
                self.teacher.start(self.request,observation)
                self.teacher_started=True
                self.adapter.logger.emit('recovery_teacher_takeover',call_id=self.request.call_id,
                    frame_id=observation.frame_id,learner_steps=self.steps,teacher='official_sac',evaluation_eligible=False)
            self.adapter.record_demonstrations=True
            self.adapter.demonstration_source='official_sac_recovery'
            action=self.teacher.act(observation)
        self.steps+=1
        return action

    def feedback(self,request,transition):
        return self.learner.feedback(request,transition)
