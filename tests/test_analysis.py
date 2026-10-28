"""relation_analysis.py and explain_reasoning.py with stub encoders (no downloads)."""

import pytest

torch = pytest.importorskip("torch")

import models.scm_vlm as scm_vlm_module
from tests.test_do_intervention import StubText, StubVision

import relation_analysis as ra
import explain_reasoning as er

CLASSES = ["person", "bicycle", "car", "dog"]


@pytest.fixture()
def models(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)
    torch.manual_seed(0)
    return {"full model": scm_vlm_module.SCMVLM(num_classes=6, graph_mode="soft").eval(),
            "no graph": scm_vlm_module.SCMVLM(num_classes=6, graph_mode="none").eval()}


class FakeCache:
    def __init__(self, n_images=30):
        g = torch.Generator().manual_seed(3)
        self.items = {}
        for k in range(n_images):
            n = 2 + k % 5
            x = torch.rand(n, 2, generator=g) * 0.6
            boxes = torch.cat([x, x + 0.1 + torch.rand(n, 2, generator=g) * 0.3], -1)
            labels = [CLASSES[int(i)] for i in torch.randint(0, 4, (n,), generator=g)]
            self.items[f"img{k}.jpg"] = (torch.randn(n, 256, generator=g), boxes, labels)

    def image(self, path):
        raw, boxes, _ = self.items[path]
        return raw, boxes

    def labels(self, path):
        return self.items[path][2]

    def question(self, text):
        return torch.randn(768, generator=torch.Generator().manual_seed(len(text)))


class FakeDataset:
    def __init__(self, cache):
        self.images = sorted(cache.items)

    def __getitem__(self, index):
        return self.images[index % len(self.images)], f"question {index}?", torch.tensor(index % 6)


def test_node_targets():
    boxes = torch.tensor([[0.10, 0.40, 0.30, 0.90],    # person
                          [0.12, 0.10, 0.28, 0.39],    # umbrella, touching, above the person
                          [0.70, 0.70, 0.80, 0.80],    # car, far away
                          [0.85, 0.70, 0.95, 0.80]])   # car, far away
    labels = ["person", "umbrella", "car", "car"]
    t = ra.node_targets(labels, boxes, {"person": 0, "car": 1})
    assert t["touches_person"] == [0, 1, 0, 0]
    assert t["same_class_count"] == [0, 0, 1, 1]
    assert t["object_above"] == [1, 0, 0, 0]
    assert t["own_class"] == [0, 2, 1, 1]


def test_knockout_edges_and_probes(models, tmp_path):
    cache = FakeCache()
    dataset = FakeDataset(cache)
    knock = ra.graph_knockout(models["full model"], dataset, range(40), cache)
    assert {"with learned graph", "message passing off", "scrambled graph"} <= set(knock)
    assert ra.graph_knockout(models["no graph"], dataset, range(5), cache) is None

    question_of = {image: "what?" for image in cache.items}
    edges = ra.edge_alignment(models["full model"], sorted(cache.items), cache, question_of)
    assert 0 <= edges["touching"]["strongest_parent_%"] <= 100

    table, n_train, n_test = ra.relation_probes(models, sorted(cache.items), cache, question_of)
    assert {"raw DETR", "full model", "no graph"} <= set(table.columns)
    assert n_train > 0 and n_test > 0
    ra.plot_probes(table, str(tmp_path / "probes.png"))
    assert (tmp_path / "probes.png").is_file()


def test_scrambled_graph_changes_the_computation(models):
    cache = FakeCache()
    raw, boxes = cache.image("img4.jpg")         # 6 objects
    q = cache.question("x")
    model = models["full model"]
    normal = ra.run(model, raw, q, boxes)["hidden"]
    off = ra.run(model, raw, q, boxes, "no_messages")["hidden"]
    scrambled = ra.run(model, raw, q, boxes, "scrambled", torch.Generator().manual_seed(0))["hidden"]
    assert not torch.allclose(normal, off) and not torch.allclose(normal, scrambled)


def test_explain_and_plot(models, tmp_path):
    from PIL import Image

    cache = FakeCache()
    raw, boxes = cache.image("img3.jpg")
    labels = cache.labels("img3.jpg")
    q = cache.question("what is the man riding?")
    idx2answer = {i: f"ans{i}" for i in range(6)}
    results = {name: er.explain(m, raw, q, boxes, labels, idx2answer) for name, m in models.items()}
    full, nograph = results["full model"], results["no graph"]
    assert len(full["objects"]) == raw.shape[0] and full["edges"] and "chain" in full
    assert "edges" not in nograph
    assert abs(sum(o["attention"] for o in nograph["objects"]) - 1) < 1e-4
    er.print_report("full model", full, labels)

    image = tmp_path / "img.png"
    Image.new("RGB", (120, 80)).save(image)
    er.plot(str(image), boxes, labels, results, str(tmp_path / "r.png"))
    assert (tmp_path / "r.png").is_file()
