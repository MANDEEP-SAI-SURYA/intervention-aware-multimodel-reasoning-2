"""Ablation: SCM-VLM without the causal graph (--graph-mode none).
No DETR/BERT download needed: the encoders are replaced with small stubs."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from tests.test_do_intervention import StubText, StubVision


@pytest.fixture()
def stub_encoders(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)


def _inputs(n=5):
    torch.manual_seed(1)
    raw = torch.randn(1, n, 256)
    q = torch.randn(1, 768)
    x0 = torch.rand(1, n, 2) * 0.5
    boxes = torch.cat([x0, x0 + 0.3], dim=-1)
    return raw, q, boxes


def test_no_graph_model_has_no_edges_and_no_message_passing(stub_encoders):
    model = scm_vlm_module.SCMVLM(num_classes=7, graph_mode="none").eval()
    assert model.graph is None and not model.has_graph
    raw, q, boxes = _inputs()

    out = model.forward_features(raw, q, boxes)
    assert out["logits"].shape == (1, 7)
    assert torch.count_nonzero(out["adjacency"]) == 0

    # objects are refined independently: changing object 0 leaves the others untouched
    changed = raw.clone()
    changed[0, 0] += 3.0
    out2 = model.forward_features(changed, q, boxes)
    assert not torch.allclose(out["hidden"][0, 0], out2["hidden"][0, 0])
    assert torch.allclose(out["hidden"][0, 1:], out2["hidden"][0, 1:], atol=1e-6)


def test_full_model_does_pass_messages(stub_encoders):
    """Sanity check of the test above: with the graph, a change does spread."""
    model = scm_vlm_module.SCMVLM(num_classes=7, graph_mode="soft").eval()
    raw, q, boxes = _inputs()
    out = model.forward_features(raw, q, boxes)
    changed = raw.clone()
    changed[0, 0] += 3.0
    out2 = model.forward_features(changed, q, boxes)
    assert not torch.allclose(out["hidden"][0, 1:], out2["hidden"][0, 1:], atol=1e-6)


def test_no_graph_counterfactual_runs(stub_encoders):
    model = scm_vlm_module.SCMVLM(num_classes=7, graph_mode="none").eval()
    raw, q, boxes = _inputs()
    out = model.forward_features(raw, q, boxes)
    replacement = out["objects"][:, 2] + 0.1
    cf = model.counterfactual(out["objects"], out["adjacency"], torch.tensor([2]), replacement, q)
    assert cf["logits"].shape == (1, 7)
    assert cf["descendants"] == [[]]


def test_no_graph_training_step_and_checkpoint_roundtrip(stub_encoders, tmp_path, monkeypatch):
    import train
    import main

    model = scm_vlm_module.SCMVLM(num_classes=7, graph_mode="none")
    raw, q, boxes = _inputs()
    loss, logits, parts = train.sample_losses(model, raw[0], q[0], torch.tensor(3), boxes[0])
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["acyclic"].item() == 0.0 and parts["edge"].item() == 0.0
    assert model.answer_head.classifier[0].weight.grad is not None

    class FakeDataset:
        idx2answer = {i: f"ans{i}" for i in range(7)}
        answer2idx = {v: k for k, v in idx2answer.items()}

        def __getitem__(self, index):
            return f"img{index}.jpg", f"question {index}?", torch.tensor(index % 7)

    class FakeCache:
        def image(self, path):
            return raw[0], boxes[0]

        def question(self, text):
            return q[0]

    factual, cf, shift, relation, predictions = train.evaluate(model, FakeDataset(), [0, 1, 2], FakeCache())
    assert len(predictions) == 3
    assert {"question", "ground_truth", "prediction", "counterfactual_prediction"} <= set(predictions[0])

    path = tmp_path / "nograph.pt"
    train.save_checkpoint(model, FakeDataset(), 1, 0.1, factual, str(path))
    loaded, idx2answer = main.load_trained_model(str(path))
    assert loaded.graph is None and idx2answer[3] == "ans3"
