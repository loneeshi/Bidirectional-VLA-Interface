import pytest

torch = pytest.importorskip('torch')
from torch import nn

from bvi.acdit_tapt import ACDiTFamilyTool


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.projection = nn.Linear(4, 4)
        self.final_layer = nn.Linear(4, 3)

    def forward(self, x):
        return self.final_layer(self.projection(x))


class Runner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiT()
        self.pred_horizon = 2
        self.calls = 0
        self.last_hidden = None

    def predict_action(self, x):
        assert not torch.is_grad_enabled()
        self.calls += 1
        self.model(x)
        # Stand in for the final native denoising invocation; the progress
        # head must see the last invocation, including the horizon slice.
        self.last_hidden = self.model.projection(x + 1)
        return self.model.final_layer(self.last_hidden)

    def compute_loss(self, **batch):
        raise AssertionError('Progress-only examples must never use action loss')


def make_tool():
    torch.manual_seed(19)
    runner = Runner()
    return runner, ACDiTFamilyTool(runner, ['projection'], rank=2)


def test_only_head_has_gradients_from_native_generated_features(monkeypatch):
    runner, tool = make_tool()
    def forbidden(*args, **kwargs):
        raise AssertionError('Must use captured native prediction, not training_loss or patched wrapper')
    monkeypatch.setattr(tool, 'training_loss', forbidden)
    monkeypatch.setattr(runner, 'predict_action', forbidden)
    x = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.tensor([[0., 1.]])
    result = tool.progress_only_loss('reach', {'x': x}, target, torch.ones_like(target).bool())
    assert runner.calls == 1
    assert set(result) == {'loss', 'action_loss', 'progress_loss', 'progress'}
    assert result['action_loss'].item() == 0
    torch.testing.assert_close(result['progress'], tool.progress(runner.last_hidden[:, -2:]))
    torch.testing.assert_close(result['loss'], .1 * result['progress_loss'])
    result['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tool.progress.parameters())
    assert all(p.grad is None for p in runner.parameters())  # Includes every LoRA bank.
    assert x.grad is None
    assert tool._hidden is None and all(layer.family is None for layer in tool.layers)
    assert not runner.model.final_layer._forward_pre_hooks


def test_endpoint_labels_and_invalid_mask():
    runner, tool = make_tool()
    target = torch.tensor([[1., float('nan')]])
    valid = torch.tensor([[True, False]])
    result = tool.progress_only_loss('move', {'x': torch.zeros(1, 2, 4)}, target, valid, .3)
    expected = (result['progress'][0, 0] - 1).square()
    torch.testing.assert_close(result['progress_loss'], expected)
    torch.testing.assert_close(result['loss'], .3 * expected)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert all(p.grad is None for p in runner.parameters())


@pytest.mark.parametrize('target,valid', [
    ([[0., 1.]], [[False, False]]),
    ([[float('nan'), 1.]], [[True, True]]),
    ([[-.1, 1.]], [[True, True]]),
    ([[0., 1.1]], [[True, True]]),
    ([[0.]], [[True]]),
])
def test_invalid_supervision_fails_and_releases_capture(target, valid):
    runner, tool = make_tool()
    with pytest.raises(ValueError):
        tool.progress_only_loss('grasp', {'x': torch.zeros(1, 2, 4)},
                                torch.tensor(target), torch.tensor(valid))
    assert tool._hidden is None and all(layer.family is None for layer in tool.layers)
    assert not runner.model.final_layer._forward_pre_hooks
    assert all(p.grad is None for p in tool.parameters())


@pytest.mark.parametrize('weight', [0, -1, float('nan'), float('inf')])
def test_invalid_weight_rejected_before_inference(weight):
    runner, tool = make_tool()
    with pytest.raises(ValueError):
        tool.progress_only_loss('reach', {}, torch.zeros(1, 2), torch.ones(1, 2), weight)
    assert runner.calls == 0


def test_missing_hidden_is_not_replaced_with_synthetic_features():
    runner, tool = make_tool()
    tool._native_predict_action = lambda **batch: torch.zeros(1, 2, 3)
    with pytest.raises(RuntimeError, match='No real inference features'):
        tool.progress_only_loss('release', {}, torch.zeros(1, 2), torch.ones(1, 2))
    assert tool._hidden is None and all(layer.family is None for layer in tool.layers)
