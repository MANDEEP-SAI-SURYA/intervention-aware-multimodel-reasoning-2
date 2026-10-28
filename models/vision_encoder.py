import torch
import torch.nn as nn

from PIL import Image

from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection
)

from configs.config import (
    DETR_MODEL,
    DEVICE,
    NUM_OBJECTS
)


DETECTION_THRESHOLD = 0.9


def kept_query_indices(logits: torch.Tensor, threshold: float = DETECTION_THRESHOLD) -> list[torch.Tensor]:
    """DETR query index of every detection that ``post_process_object_detection`` keeps,
    in the same order as its output.  logits: [B, Q, C+1] (last class = no object).

    post_process returns only the kept detections, so position r in its output is NOT
    query r.  The decoder embedding of detection r is ``decoder_hidden[b, kept[b][r]]``."""
    scores = torch.softmax(logits, dim=-1)[..., :-1].max(dim=-1).values
    return [torch.nonzero(row > threshold, as_tuple=False).flatten() for row in scores]


class VisionEncoder(nn.Module):
    """
    DETR-based vision encoder.

    Returns:
    - object embeddings
    - object labels
    - bounding boxes
    - confidence scores
    """

    def __init__(
        self,
        model_name: str = DETR_MODEL,
        freeze: bool = True
    ) -> None:

        super().__init__()

        self.freeze = freeze

        self.processor = DetrImageProcessor.from_pretrained(
            model_name
        )

        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            output_hidden_states=True
        )

        self.model.to(DEVICE)

        if freeze:

            self.model.eval()

            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):

        super().train(mode)

        if self.freeze:
            self.model.eval()

        return self

    def _load_images(
        self,
        image_paths: str | list[str]
    ) -> list[Image.Image]:

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        return [
            Image.open(path).convert("RGB")
            for path in image_paths
        ]

    def forward(
        self,
        image_paths: str | list[str]
    ):

        images = self._load_images(image_paths)

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(DEVICE)

        grad_context = (
            torch.no_grad()
            if self.freeze
            else torch.enable_grad()
        )

        with grad_context:

            outputs = self.model(**inputs)

        # =========================
        # OBJECT EMBEDDINGS
        # =========================

        object_embeddings = outputs.decoder_hidden_states[-1]

        # =========================
        # OBJECT DETECTIONS
        # =========================

        target_sizes = torch.tensor(
            [image.size[::-1] for image in images]
        ).to(DEVICE)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=DETECTION_THRESHOLD,
            target_sizes=target_sizes
        )
        # FIX: map each kept detection back to the DETR query that produced it.
        # The earlier version indexed the decoder states with the position in the
        # *filtered* list, so the node embeddings belonged to other queries (mostly
        # "no object" queries) and did not match the detected labels / boxes.
        query_indices = kept_query_indices(outputs.logits, DETECTION_THRESHOLD)

        labels = []
        boxes = []
        scores = []
        filtered_embeddings = []

        for batch_index, result in enumerate(results):

            # =========================
            # KEEP ONLY TOP-K OBJECTS
            # =========================

            keep = min(
                NUM_OBJECTS,
                len(result["scores"])
            )

            top_indices = torch.argsort(
                result["scores"],
                descending=True
            )[:keep]

            filtered_labels = [
                self.model.config.id2label[
                    result["labels"][idx].item()
                ]
                for idx in top_indices
            ]

            filtered_boxes = result["boxes"][top_indices]

            filtered_scores = result["scores"][top_indices]

            filtered_object_embeddings = object_embeddings[
                batch_index,
                query_indices[batch_index].to(top_indices.device)[top_indices],
                :
            ]

            labels.append(filtered_labels)

            boxes.append(filtered_boxes)

            scores.append(filtered_scores)

            filtered_embeddings.append(
                filtered_object_embeddings
            )

        filtered_embeddings = torch.stack(
            filtered_embeddings
        )

        # boxes scaled to 0-1 by the image size (used by the relation graph)
        boxes_norm = [
            box / torch.tensor([image.size[0], image.size[1], image.size[0], image.size[1]],
                               dtype=box.dtype, device=box.device)
            for box, image in zip(boxes, images)
        ]

        return {
            "embeddings": filtered_embeddings,
            "labels": labels,
            "boxes": boxes,
            "boxes_norm": boxes_norm,
            "scores": scores
        }

    def encode(
        self,
        image_paths: str | list[str]
    ):

        return self.forward(image_paths)