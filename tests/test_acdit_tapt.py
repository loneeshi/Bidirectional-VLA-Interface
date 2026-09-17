import pytest
torch = pytest.importorskip('torch')
from torch import nn
from bvi.acdit_tapt import (FAMILIES, install_family_residuals, family_invocation,
                            LearnedProgress, masked_progress_loss)


def test_bank_isolation_and_frozen_native():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4,5), nn.Tanh(), nn.Linear(5,3))
    x = torch.randn(2,4)
    native = model(x).detach().clone()
    layers = install_family_residuals(model,['0','2'],rank=2)
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    with pytest.raises(RuntimeError): model(x)
    for family in FAMILIES:
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        opt.zero_grad(set_to_none=True)
        with family_invocation(layers,family):
            # This bank starts at zero; earlier banks cannot leak into it.
            torch.testing.assert_close(model(x),native)
            model(x).square().mean().backward()
            with pytest.raises(ValueError):
                with family_invocation(layers,'move'): pass
        opt.step()
        changed = [n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
        assert changed and all(n.endswith((f'.a.{family}',f'.b.{family}')) for n in changed)
        for n,p in model.named_parameters():
            if n in frozen: assert torch.equal(p,frozen[n])
    assert all(layer.family is None for layer in layers)


def test_progress_is_learned_and_masks_invalid_labels():
    head=LearnedProgress(6)
    pred=head(torch.randn(2,2,6)); target=torch.tensor([[0.,1.],[.5,float('nan')]])
    loss=masked_progress_loss(pred,target,torch.tensor([[1,1],[1,0]]))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    with pytest.raises(ValueError):masked_progress_loss(pred,target,torch.zeros(2,2))


def test_inference_dispatch_does_not_recurse_and_returns_real_features():
    from bvi.acdit_tapt import ACDiTFamilyTool
    class DiT(nn.Module):
        def __init__(self):
            super().__init__(); self.hidden_size=4
            self.projection=nn.Linear(4,4);self.final_layer=nn.Linear(4,3)
        def forward(self,x):return self.final_layer(self.projection(x))
    class Runner(nn.Module):
        def __init__(self):
            super().__init__();self.model=DiT();self.pred_horizon=2
        def predict_action(self,x):return self.model(x)
    runner=Runner();tool=ACDiTFamilyTool(runner,['projection'],rank=2)
    runner.predict_action=lambda **batch: tool.predict('reach',batch)[0]
    actions,progress=tool.predict('reach',{'x':torch.ones(1,2,4)})
    assert actions.shape==(1,2,3) and progress.shape==(1,2)
    assert torch.isfinite(progress).all() and tool._hidden is None
    assert all(layer.family is None for layer in tool.layers)
