import hashlib
import json
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from bvi.decision_trace import TorchDecisionRecorder, capture_rng_state, restore_rng_state


def draws():
    return random.random(), np.random.random(3), torch.rand(3)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_exact_values_hashes_and_recording_does_not_consume_rng(tmp_path):
    original_rng = capture_rng_state()
    try:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        before = capture_rng_state()
        expected = draws()
        restore_rng_state(before)
        recorder = TorchDecisionRecorder(tmp_path)
        metadata = {"step": 4, "family": "reach", "nested": [1]}
        tensor = torch.tensor([[1., 2.]], requires_grad=True)
        array = np.array([3, 4], dtype=np.int16)
        decision_id = recorder.begin(metadata, {"history": (array, tensor)})
        recorder.record_batch({"image": tensor, "nested": [array]})
        # Saved values must remain unchanged after mutation of source objects.
        with torch.no_grad():
            tensor.add_(10)
        array[:] = 99
        metadata["nested"].append(2)
        manifest = recorder.finish(torch.tensor([5.]), torch.tensor([.3]))
        assert_draws_equal(draws(), expected)
        raw = load(tmp_path / f"{decision_id}-raw.pt")
        inputs = load(tmp_path / f"{decision_id}-inputs.pt")
        outputs = load(tmp_path / f"{decision_id}-outputs.pt")
        assert isinstance(raw["raw"]["history"], tuple)
        np.testing.assert_array_equal(raw["raw"]["history"][0], [3, 4])
        np.testing.assert_array_equal(inputs["batch"]["nested"][0], [3, 4])
        assert torch.equal(inputs["batch"]["image"], torch.tensor([[1., 2.]]))
        assert not inputs["batch"]["image"].requires_grad
        assert inputs["batch"]["image"].device.type == "cpu"
        assert outputs["actions"].item() == 5
        assert outputs["progress"].item() == pytest.approx(.3)
        assert manifest["metadata"]["nested"] == [1]
        assert json.loads((tmp_path / f"{decision_id}-manifest.json").read_text()) == manifest
        for item in manifest["artifacts"].values():
            data = (tmp_path / item["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"]
            assert len(data) == item["size_bytes"]
        for payload, key in [(raw, "rng_before_preprocessing"),
                             (inputs, "rng_before_model"), (outputs, "rng_after_model")]:
            restore_rng_state(payload[key])
            assert_draws_equal(draws(), expected)
    finally:
        restore_rng_state(original_rng)


def test_ordering_and_no_overwrite(tmp_path):
    recorder = TorchDecisionRecorder(tmp_path)
    with pytest.raises(RuntimeError):
        recorder.record_batch({})
    with pytest.raises(RuntimeError):
        recorder.finish(None, None)
    assert recorder.begin({}, {}) == "000000"
    with pytest.raises(RuntimeError):
        recorder.begin({}, {})
    with pytest.raises(RuntimeError):
        recorder.finish(None, None)
    recorder.record_batch({})
    with pytest.raises(RuntimeError):
        recorder.record_batch({})
    recorder.finish(None, None)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    resumed = TorchDecisionRecorder(tmp_path)
    assert resumed.begin({}, {}) == "000001"
    with pytest.raises(FileExistsError):
        recorder.begin({}, {})  # stale recorder cannot reuse another writer's id
    resumed.record_batch({})
    collision = tmp_path / "000001-outputs.pt"
    collision.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        resumed.finish(None, None)
    assert collision.read_bytes() == b"preserve me"
    for name, content in previous.items():
        assert (tmp_path / name).read_bytes() == content


def test_model_rng_roundtrip(tmp_path):
    original_rng = capture_rng_state()
    try:
        recorder = TorchDecisionRecorder(tmp_path)
        decision_id = recorder.begin({}, {})
        recorder.record_batch({"state": torch.ones(2)})
        prediction = draws()
        recorder.finish(prediction[2], torch.zeros(1))
        next_expected = draws()
        inputs = load(tmp_path / f"{decision_id}-inputs.pt")
        restore_rng_state(inputs["rng_before_model"])
        assert_draws_equal(draws(), prediction)
        outputs = load(tmp_path / f"{decision_id}-outputs.pt")
        restore_rng_state(outputs["rng_after_model"])
        assert_draws_equal(draws(), next_expected)
    finally:
        restore_rng_state(original_rng)
