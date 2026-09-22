"""Fetch TAPT components, not pretrained skills or a paper-score reproduction.

STATUS: frozen — historical training and diagnostics (retained)

AC-DiT port: explicit four-bank LoRA on selected manipulation DiT projections.
Insertion sites must be supplied and recorded by the training configuration.
Native backbone and mobility expert remain frozen. No automatic family inference.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn

FAMILIES = ('reach', 'grasp', 'move', 'release')


class FamilyLinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError('Expected linear projection and positive rank')
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.ParameterDict({f:nn.Parameter(torch.empty(rank, base.in_features,
                                  device=base.weight.device,dtype=base.weight.dtype)) for f in FAMILIES})
        self.b = nn.ParameterDict({f:nn.Parameter(torch.zeros(base.out_features, rank,
                                  device=base.weight.device,dtype=base.weight.dtype)) for f in FAMILIES})
        for value in self.a.values(): nn.init.kaiming_uniform_(value, a=5**0.5)
        self.family = None

    def forward(self, x):
        if self.family not in FAMILIES:
            raise RuntimeError('Select a tool family for this invocation')
        return self.base(x) + ((x @ self.a[self.family].T) @ self.b[self.family].T) * self.scale


def install_family_residuals(backbone, paths, rank=8, alpha=8):
    """Validate every path before mutation; no mobility-head substitutions."""
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Explicit unique projection paths required')
    layers = [backbone.get_submodule(p) for p in paths]
    if not all(isinstance(x, nn.Linear) for x in layers):
        raise ValueError('Every insertion target must be a native Linear')
    backbone.requires_grad_(False)
    result = []
    for path, base in zip(paths, layers):
        parent, _, name = path.rpartition('.')
        layer = FamilyLinear(base, rank, alpha)
        setattr(backbone.get_submodule(parent) if parent else backbone, name, layer)
        result.append(layer)
    return result


@contextmanager
def family_invocation(layers, family):
    if family not in FAMILIES or not layers or any(x.family is not None for x in layers):
        raise ValueError('Invalid or overlapping invocation')
    for layer in layers: layer.family = family
    try:
        yield
    finally:
        for layer in layers: layer.family = None


class LearnedProgress(nn.Module):
    """Supervised local progress from final action-token hidden features.

Caller must supply real captured DiT hidden features, not simulator predicates.
Training and inference feature extraction must be validated separately.
"""
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,128),
                                  nn.SiLU(),nn.Linear(128,1))

    def forward(self, action_hidden):
        return self.head(action_hidden).squeeze(-1).sigmoid()


def masked_progress_loss(prediction, target, valid):
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Progress shapes differ')
    valid = valid.bool()
    if not valid.any() or not torch.isfinite(target[valid]).all():
        raise ValueError('No finite progress targets')
    if not ((target[valid]>=0)&(target[valid]<=1)).all():
        raise ValueError('Progress outside [0,1]')
    return (prediction[valid]-target[valid]).square().mean()


class ACDiTFamilyTool(nn.Module):
    """Port of tool-family residual/progress design to the native AC-DiT runner.

Uses upstream compute_loss unchanged for the action objective. Progress reads
the manipulation DiT action tokens before its final output projection; during
inference the final denoising invocation supplies these features. This placement
is an explicit AC-DiT port, not an assertion of identical OpenPI architecture.
"""
    def __init__(self, runner, projection_paths, rank=8, alpha=8):
        super().__init__()
        self.runner = runner.requires_grad_(False)
        self._native_predict_action = runner.predict_action
        self.projection_paths = tuple(projection_paths)
        self.layers = install_family_residuals(runner.model,projection_paths,rank,alpha)
        weight=next(runner.model.parameters())
        self.progress=LearnedProgress(runner.model.hidden_size).to(device=weight.device,dtype=weight.dtype)
        self._hidden=None

    @contextmanager
    def _capture(self, family):
        self._hidden=None
        def capture(module, inputs):
            self._hidden=inputs[0][:,-self.runner.pred_horizon:]
        handle=self.runner.model.final_layer.register_forward_pre_hook(capture)
        try:
            with family_invocation(self.layers,family):yield
        finally:
            handle.remove()
            self._hidden=None

    def training_loss(self, family, batch, progress_target, progress_valid, progress_weight=.1):
        if progress_weight<=0:raise ValueError('Progress learning must have positive weight')
        with self._capture(family):
            action=self.runner.compute_loss(**batch)['loss']
            if self._hidden is None:raise RuntimeError('No real DiT features captured')
            prediction=self.progress(self._hidden)
            pl=masked_progress_loss(prediction,progress_target,progress_valid)
            return {'loss':action+progress_weight*pl,'action_loss':action,
                    'progress_loss':pl,'progress':prediction}

    def progress_only_loss(self, family, batch, progress_target, valid, progress_weight=.1):
        """Train only progress on actual native generated action-token features.

        The native inference path supplies the last captured denoising features
        under no_grad; detached features then enter the trainable progress head.
        No action target or native action loss is used. These are generated
        action-token hidden features, not the author's observation-prefix source.
        The action_loss metric is zero because this objective has no action loss.
        """
        if not math.isfinite(progress_weight) or progress_weight <= 0:
            raise ValueError('Progress learning must have finite positive weight')
        with self._capture(family):
            with torch.no_grad():
                self._native_predict_action(**batch)
                if self._hidden is None:
                    raise RuntimeError('No real inference features captured')
                hidden = self._hidden.detach()
            prediction = self.progress(hidden)
            pl = masked_progress_loss(prediction, progress_target, valid)
            return {'loss': progress_weight * pl,
                    'action_loss': prediction.new_zeros(()),
                    'progress_loss': pl, 'progress': prediction}

    @torch.no_grad()
    def predict(self, family, batch):
        with self._capture(family):
            actions=self._native_predict_action(**batch)
            if self._hidden is None:raise RuntimeError('No real inference features captured')
            return actions,self.progress(self._hidden)
