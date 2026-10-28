"""The node embedding of each detection must come from the DETR query that produced it.
No DETR download: the DETR network is replaced by a stub with known outputs."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
transformers = pytest.importorskip("transformers")

from models.vision_encoder import VisionEncoder, kept_query_indices

NUM_QUERIES, NUM_CLASSES, DIM = 10, 4, 256


def _outputs():
    logits = torch.full((1, NUM_QUERIES, NUM_CLASSES + 1), -10.0)
    logits[..., -1] = 10.0                      # every query: "no object"
    logits[0, 7, 1] = 10.0; logits[0, 7, -1] = -10.0     # query 7: class 1, very confident
    logits[0, 3, 2] = 3.0;  logits[0, 3, -1] = -10.0     # query 3: class 2, less confident
    boxes = torch.full((1, NUM_QUERIES, 4), 0.2)
    boxes[0, 7] = torch.tensor([0.3, 0.3, 0.2, 0.2])
    boxes[0, 3] = torch.tensor([0.7, 0.6, 0.2, 0.4])
    # embedding of query q is filled with the value q, so we can see which query was used
    hidden = torch.arange(NUM_QUERIES, dtype=torch.float)[None, :, None].expand(1, NUM_QUERIES, DIM)
    return SimpleNamespace(logits=logits, pred_boxes=boxes, decoder_hidden_states=(hidden * 0, hidden))


class StubDetr(nn.Module):
    config = SimpleNamespace(id2label={0: "person", 1: "bicycle", 2: "car", 3: "dog"})

    def forward(self, **inputs):
        return _outputs()


def test_kept_query_indices():
    assert kept_query_indices(_outputs().logits)[0].tolist() == [3, 7]


def test_embeddings_belong_to_the_detected_objects(tmp_path):
    from PIL import Image

    image = tmp_path / "img.png"
    Image.new("RGB", (64, 48)).save(image)
    encoder = VisionEncoder.__new__(VisionEncoder)
    nn.Module.__init__(encoder)
    encoder.freeze = True
    encoder.processor = transformers.DetrImageProcessor()
    encoder.model = StubDetr()

    out = encoder(str(image))
    # ordered by score: bicycle (query 7) first, then car (query 3)
    assert out["labels"][0] == ["bicycle", "car"]
    assert out["embeddings"][0, 0, 0].item() == 7.0
    assert out["embeddings"][0, 1, 0].item() == 3.0
