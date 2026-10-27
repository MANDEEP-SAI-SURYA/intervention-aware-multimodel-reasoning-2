import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import DEVICE
from models.scm_vlm import SCMVLM

CHECKPOINT_PATH = "checkpoints/scm_vlm_latest.pt"


def load_trained_model(checkpoint_path: str = CHECKPOINT_PATH) -> tuple[SCMVLM, dict[int, str]]:
    """checkpoint_path: e.g. checkpoints/scm_vlm_latest_nograph.pt for the no-graph ablation."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at '{checkpoint_path}'. "
            "Run 'python train.py' first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # checkpoints saved before the answer-head fix have no "question_aware" flag
    question_aware = bool(checkpoint.get("question_aware", False))
    # checkpoints saved before the graph-collapse fix have no "graph_mode"
    graph_mode = checkpoint.get("graph_mode", "legacy")
    model = SCMVLM(
        num_classes=checkpoint["num_classes"], question_aware=question_aware,
        graph_mode=graph_mode, graph_top_k=checkpoint.get("graph_top_k", 3),
        graph_geometry=bool(checkpoint.get("graph_geometry", False)),
    ).to(DEVICE)
    if graph_mode == "none":
        print("NOTE: this is the ablation model WITHOUT the causal graph (no edges, no message passing).")
    if graph_mode == "legacy":
        print(
            "NOTE: this checkpoint uses the old graph (hard 0.5 edge threshold). In models trained "
            "this way the relation graph usually collapsed to no edges. Retrain with "
            "'python train.py' to get the soft graph."
        )
    if not question_aware:
        print(
            "WARNING: this checkpoint uses the old answer head, which never sees the "
            "question, so its answers are the same for every question (mostly yes/no). "
            "Retrain with 'python train.py' to get question-dependent answers."
        )
    incompatible = model.load_state_dict(
        checkpoint["trainable_state_dict"], strict=False
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint has unexpected parameters: {incompatible.unexpected_keys}"
        )

    idx2answer = {
        int(index): answer for index, answer in checkpoint["idx2answer"].items()
    }
    model.eval()
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(training accuracy: {checkpoint['training_accuracy']:.4f})"
    )
    return model, idx2answer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run SCM-VLM on any image + question")
    parser.add_argument("--image", required=True, help="path to any image")
    parser.add_argument("--question", required=True, help="question about the image")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="checkpoint to load (e.g. checkpoints/scm_vlm_latest_nograph.pt)")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")

    model, idx2answer = load_trained_model(args.checkpoint)

    image_path = args.image
    question = args.question

    # =========================
    # VISION ENCODER OUTPUT
    # =========================

    with torch.inference_mode():
        vision_output = model.vision(image_path)

    print("\n=========================")
    print("DETECTED OBJECTS")
    print("=========================\n")

    labels = vision_output["labels"][0]

    for i, label in enumerate(labels):
        print(f"Node {i}: {label}")

    # =========================
    # NORMAL FORWARD PASS
    # =========================

    with torch.inference_mode():
        output = model(image_path, question)

    print("\n=========================")
    print("MODEL OUTPUT (FACTUAL)")
    print("=========================\n")

    print("Logits:\n", output["logits"])

    prediction_index = torch.argmax(output["logits"], dim=-1).item()
    print("\nPredicted answer:", idx2answer.get(prediction_index, "unknown"))

    adjacency = output["adjacency"]

    print("\nAdjacency Matrix Shape:\n", adjacency.shape)
    print("\nAdjacency Matrix:\n", adjacency)

    print("\nFor what-if questions use: python whatif_qa.py --image <path> --whatif \"What if ...\"")

    # =========================
    # COUNTERFACTUAL REASONING  (superseded by whatif_qa.py)
    # =========================

    # print("\n=========================")
    # print("COUNTERFACTUAL ANALYSIS")
    # print("=========================\n")

    # objects = output["objects"]
    # adjacency = output["adjacency"]

    # # -------------------------
    # # STEP 1: Choose node
    # # -------------------------
    # # Example: assume node 1 = bicycle (you printed labels above)
    # node_index = torch.tensor([1]).to(DEVICE)

    # # -------------------------
    # # STEP 2: Create intervention
    # # -------------------------
    # # Option A: remove object (recommended for paper)
    # replacement = torch.zeros(
    #     1,
    #     objects.shape[-1],
    #     device=DEVICE
    # )

    # # -------------------------
    # # STEP 3: Run counterfactual
    # # -------------------------
    # cf_output = model.counterfactual(
    #     objects,
    #     adjacency,
    #     node_index,
    #     replacement
    # )

    # # -------------------------
    # # STEP 4: Compare results
    # # -------------------------
    # print("Factual logits:\n", output["logits"])
    # print("\nCounterfactual logits:\n", cf_output["logits"])

    # print("\nLogit shift:\n",
    #       cf_output["logits"] - output["logits"])


if __name__ == "__main__":
    main()
