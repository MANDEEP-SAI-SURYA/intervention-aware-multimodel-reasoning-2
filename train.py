"""Train SCM-VLM on VQA v2 / COCO.

    python train.py                          # defaults below
    python train.py --samples 20000 --epochs 10
    python train.py --samples 3000 --epochs 20 --lr 1e-4
    python train.py --graph-mode none        # ablation: remove the causal graph

What changed compared with the first version
---------------------------------------------
* The answer head now receives the question (Y = Decoder(h, q)).  The old head
  only saw mean(h), so every question got the same answer (mostly yes/no).
* Answer vocabulary = the 3129 most frequent answers (standard for VQA v2)
  instead of all 22k answers, most of which were never seen in training.
* DETR and BERT are frozen, so their outputs are cached once (cache/) instead
  of being recomputed for every sample in every epoch.
* An image-level train / validation split: accuracies in results/ are measured
  on held-out questions, not on the training samples.
* The acyclicity term is the NOTEARS constraint tr(e^{A o A}) - n from the paper.
* do() interventions cut the parents of the intervened node (row k), and the
  graph is no longer rebuilt after the cut (models/scm_vlm.py).
* Relation graph (second Colab run had NO edges left): edges are soft top-k
  attention over parents (no 0.5 threshold, so the graph cannot vanish), the
  edge scorer sees each pair's box geometry, and edges are weakly supervised:
  objects whose boxes touch should be linked (LAMBDA_EDGE).  The log reports
  the edge probability for touching vs. distant objects on validation images.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from configs.config import DEVICE
from datasets.datasetBuilder import VQACOCO
from losses.losses import acyclicity_loss
from models.scm_vlm import SCMVLM


# =====================================================
# SETTINGS (defaults; override on the command line)
# =====================================================

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "scm_vlm_latest.pt")
CACHE_DIR = Path("cache")
FEATURE_VERSION = 2        # 2 = DETR embeddings taken from the query that made each detection

IMAGE_DIR = "coco/train2014/train2014"
QUESTION_FILE = "coco/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_FILE = "coco/vqa/v2_mscoco_train2014_annotations.json"

EPOCHS = 10
BATCH_SIZE = 16            # gradients are accumulated over the batch
TRAIN_SAMPLES = 20000      # questions used (train + validation)
VAL_FRACTION = 0.1         # held-out fraction, split by image
MAX_ANSWERS = 3129
LEARNING_RATE = 1e-4
SEED = 42

TOP_K = 3
RANDOM_NODE_PROB = 0.3

# Graph-collapse fix: the first Colab model ended with NO edges (all edge scores < 0.5).
# The hard 0.5 threshold gave edges below it no gradient, and sparsity + acyclicity +
# the "predictions must not change" counterfactual terms pushed every score down.
# The soft graph (models/scm_graph.py) removes the threshold; these weights are gentler.
LAMBDA_CF = 0.1          # KL(cf || factual): invariance to small perturbations (no CF labels yet)
LAMBDA_SPARSE = 0.01
LAMBDA_ACYCLIC = 0.05
LAMBDA_SHIFT = 0.0       # penalised ANY change under intervention -> disabled
LAMBDA_ENTROPY = 0.05
LAMBDA_EDGE = 0.5        # weak relation supervision: objects whose boxes touch should be linked
CONTACT_GAP = 0.02       # boxes closer than 2% of the image size count as touching
GRAPH_TOP_K = 3


# =====================================================
# FEATURE CACHE (frozen DETR + BERT)
# =====================================================

def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, model: SCMVLM, root: Path) -> None:
        self.model = model
        self.image_dir = root / "detr"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.questions: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def image(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """DETR object embeddings [N, 256] and boxes [N, 4] (x0, y0, x1, y1 scaled to 0-1).
        Cache files from the earlier version have no boxes and are recomputed once."""
        path = self.path(image_path)
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=False)
            # files written before the DETR query-index fix hold embeddings of the wrong
            # queries; they have no "feature_version" and are recomputed once
            if "boxes" in item and item.get("feature_version", 1) >= FEATURE_VERSION:
                return item["embeddings"], item["boxes"]
        output = self.model.vision(image_path)
        embeddings = output["embeddings"][0].detach().cpu()
        boxes = output["boxes"][0].detach().cpu().float() if output.get("boxes") else torch.zeros(len(embeddings), 4)
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = boxes / torch.tensor([width, height, width, height], dtype=torch.float)
        torch.save({"embeddings": embeddings, "labels": output["labels"][0], "boxes": boxes,
                    "feature_version": FEATURE_VERSION}, path)
        return embeddings, boxes

    def path(self, image_path: str) -> Path:
        return self.image_dir / f"{_key(os.path.abspath(image_path))}.pt"

    def labels(self, image_path: str) -> list[str]:
        """DETR class names of the cached objects (call image() first)."""
        return torch.load(self.path(image_path), map_location="cpu", weights_only=False)["labels"]

    @torch.no_grad()
    def question(self, question: str) -> torch.Tensor:
        if question not in self.questions:
            self.questions[question] = self.model.text(question)[0].detach().cpu()
        return self.questions[question]


# =====================================================
# HELPERS
# =====================================================

def split_by_image(dataset: VQACOCO, num_samples: int, val_fraction: float, seed: int):
    """Pick up to ``num_samples`` questions and split them so that no image
    appears in both train and validation."""
    by_image = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_image[str(record["image_path"])].append(index)
    images = sorted(by_image)
    random.Random(seed).shuffle(images)

    chosen = []
    for image in images:
        chosen.append(image)
        if sum(len(by_image[i]) for i in chosen) >= num_samples:
            break
    num_val = max(1, int(round(len(chosen) * val_fraction)))
    val_images, train_images = chosen[:num_val], chosen[num_val:]
    train = [i for image in train_images for i in by_image[image]]
    val = [i for image in val_images for i in by_image[image]]
    return train, val


def save_checkpoint(model, dataset, epoch, train_accuracy, val_accuracy, path=None):
    """Save learned layers + answer vocabulary. DETR and BERT are frozen
    pretrained backbones, so they are reloaded from Hugging Face instead."""
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }
    torch.save({
        "format_version": 2,
        "question_aware": True,
        "graph_mode": model.graph.mode if model.has_graph else "none",
        "graph_top_k": model.graph.top_k if model.has_graph else 0,
        "graph_geometry": model.graph.geometry if model.has_graph else False,
        "epoch": epoch,
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "num_classes": len(dataset.answer2idx),
        "idx2answer": dataset.idx2answer,
        "trainable_state_dict": trainable_state,
    }, path or CHECKPOINT_PATH)
    print(f"Saved checkpoint: {path or CHECKPOINT_PATH}")


def contact_targets(boxes: torch.Tensor, gap: float = CONTACT_GAP) -> torch.Tensor:
    """[N, N] 1 where two boxes overlap or nearly touch (a physical-interaction prior,
    e.g. person-bicycle, hand-umbrella), 0 otherwise.  Diagonal is 0."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    gap_x = torch.clamp(torch.maximum(x0[:, None], x0[None, :]) - torch.minimum(x1[:, None], x1[None, :]), min=0)
    gap_y = torch.clamp(torch.maximum(y0[:, None], y0[None, :]) - torch.minimum(y1[:, None], y1[None, :]), min=0)
    touching = ((gap_x <= gap) & (gap_y <= gap)).float()
    touching.fill_diagonal_(0.0)
    return touching


def edge_supervision_loss(edge_probs: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """BCE between the (direction-free) edge probability max(p_ij, p_ji) and box contact.
    The direction is left to the model (and the acyclicity term)."""
    n = edge_probs.shape[0]
    if n < 2:
        zero = edge_probs.new_tensor(0.0)
        return zero, {"p_contact": zero, "p_far": zero}
    target = contact_targets(boxes.to(edge_probs.device))
    sym = torch.maximum(edge_probs, edge_probs.T).clamp(1e-6, 1 - 1e-6)
    off = ~torch.eye(n, dtype=torch.bool, device=edge_probs.device)
    positives = target[off].sum()
    weight = torch.where(target[off] > 0, (off.sum() - positives).clamp(min=1) / positives.clamp(min=1),
                         torch.ones_like(target[off]))   # balance contact / non-contact pairs
    loss = F.binary_cross_entropy(sym[off], target[off], weight=weight)
    contact = target[off] > 0
    stats = {
        "p_contact": sym[off][contact].mean().detach() if contact.any() else None,
        "p_far": sym[off][~contact].mean().detach() if (~contact).any() else None,
    }
    return loss, stats


def sample_losses(model, raw_objects, q_emb, label, boxes, supervise_edges=True):
    """Factual forward pass + SCM regularisers + counterfactual terms for one sample."""
    out = model.forward_features(raw_objects.unsqueeze(0).to(DEVICE), q_emb.unsqueeze(0).to(DEVICE),
                                 boxes.unsqueeze(0).to(DEVICE))

    logits = torch.clamp(out["logits"], -30, 30)
    loss_task = F.cross_entropy(logits, label.view(1))

    objects = out["objects"]
    adjacency = torch.clamp(out["adjacency"].float(), 0.0, 1.0)
    A = adjacency[0]

    zero = logits.new_tensor(0.0)
    if model.has_graph:
        loss_sparse = A.mean()
        loss_acyclic = acyclicity_loss(A)
        edge_prob = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        loss_entropy = -torch.mean(edge_prob * torch.log(edge_prob + 1e-6))
    else:
        # no-graph ablation: there is no adjacency, so the DAG / sparsity / entropy
        # regularisers do not exist.  Everything else (task + counterfactual) is unchanged.
        loss_sparse = loss_acyclic = loss_entropy = zero

    # counterfactual: do(X_k = x') on an important or random node
    node_scores = A.sum(dim=0)                      # out-degree (A[i, j]: j -> i)
    k = min(TOP_K, node_scores.shape[0])
    _, topk_idx = torch.topk(node_scores, k=k)
    if not model.has_graph or random.random() < RANDOM_NODE_PROB:   # no graph -> no out-degree
        node_idx = torch.randint(0, objects.shape[1], (1,), device=DEVICE)
    else:
        node_idx = topk_idx[torch.randint(0, k, (1,))].view(1)

    strength = random.random() * 0.2
    replacement = objects[:, node_idx.item()] + torch.randn_like(objects[:, 0]) * strength
    cf_out = model.counterfactual(objects, adjacency, node_idx, replacement, out["question_embedding"])
    cf_logits = torch.clamp(cf_out["logits"], -30, 30)

    loss_cf = F.kl_div(
        F.log_softmax(cf_logits, dim=-1),
        F.softmax(logits.detach(), dim=-1),
        reduction="batchmean",
    ) * (1.0 + strength)
    loss_shift = torch.mean(torch.abs(F.softmax(cf_logits, dim=-1) - F.softmax(logits.detach(), dim=-1)))

    if supervise_edges and model.has_graph:
        loss_edge, edge_stats = edge_supervision_loss(out["edge_probs"][0], boxes)
    else:
        loss_edge, edge_stats = logits.new_tensor(0.0), {"p_contact": None, "p_far": None}

    loss = (
        loss_task
        + LAMBDA_EDGE * loss_edge
        + LAMBDA_SPARSE * loss_sparse
        + LAMBDA_ACYCLIC * loss_acyclic
        + LAMBDA_ENTROPY * loss_entropy
        + LAMBDA_CF * loss_cf
        + LAMBDA_SHIFT * loss_shift
    )
    parts = {"task": loss_task, "cf": loss_cf, "shift": loss_shift, "acyclic": loss_acyclic,
             "edge": loss_edge, **edge_stats,
             # graph health: total parent weight per object and the mean edge probability
             "graph_mass": A.sum(dim=-1).mean().detach(),
             "edge_prob": out["edge_probs"][0][~torch.eye(A.shape[0], dtype=torch.bool,
                                                          device=A.device)].mean().detach()
             if A.shape[0] > 1 else torch.tensor(0.0)}
    return loss, logits, parts


@torch.no_grad()
def evaluate(model, dataset, indices, cache):
    """Factual accuracy, counterfactual accuracy (answer still correct after a
    small do() perturbation of the most influential node) and mean logit shift.

    The most influential node is the one with the largest out-degree in the causal
    graph.  Without a graph (--graph-mode none) it is the node the question-guided
    answer head attends to most.  Also returns every prediction so that the answers
    of different models can be compared question by question."""
    model.eval()
    factual = counterfactual = total = 0
    predictions = []
    shifts = []
    p_contact, p_far = [], []
    for index in indices:
        image_path, question, label = dataset[index]
        try:
            raw, boxes = cache.image(image_path)
            if raw.shape[0] == 0:
                continue
            out = model.forward_features(raw.unsqueeze(0).to(DEVICE),
                                         cache.question(question).unsqueeze(0).to(DEVICE),
                                         boxes.unsqueeze(0).to(DEVICE))
            _, edge_stats = edge_supervision_loss(out["edge_probs"][0], boxes)
            if not model.has_graph:
                edge_stats = {"p_contact": None, "p_far": None}
            if edge_stats["p_contact"] is not None:
                p_contact.append(float(edge_stats["p_contact"]))
            if edge_stats["p_far"] is not None:
                p_far.append(float(edge_stats["p_far"]))
        except Exception as error:  # unreadable image etc.
            print(f"EVAL SKIP {image_path}: {error}")
            continue
        label = label.to(DEVICE)
        logits = out["logits"]
        prediction = logits.argmax(-1).item()
        factual += int(prediction == label.item())

        A = torch.clamp(out["adjacency"], 0.0, 1.0)
        if model.has_graph:
            node_idx = torch.argmax(A[0].sum(dim=0)).view(1)
        else:
            _, attention = model.answer_head(out["hidden"], out["question_embedding"])
            node_idx = torch.argmax(attention[0]).view(1)
        objects = out["objects"]
        replacement = objects[:, node_idx.item()] + torch.randn_like(objects[:, 0]) * 0.05
        cf_out = model.counterfactual(objects, A, node_idx, replacement, out["question_embedding"])
        cf_prediction = cf_out["logits"].argmax(-1).item()
        counterfactual += int(cf_prediction == label.item())
        shifts.append(torch.abs(logits - cf_out["logits"]).mean().item())
        total += 1
        predictions.append({
            "question_index": index,
            "image": os.path.basename(image_path),
            "question": question,
            "ground_truth": dataset.idx2answer[label.item()],
            "prediction": dataset.idx2answer[prediction],
            "confidence": torch.softmax(logits, dim=-1).max().item(),
            "correct": int(prediction == label.item()),
            "counterfactual_prediction": dataset.idx2answer[cf_prediction],
            "num_objects": int(objects.shape[1]),
        })
    model.train()
    total = max(total, 1)
    relation = {
        "val_edge_prob_contact": sum(p_contact) / max(len(p_contact), 1),
        "val_edge_prob_far": sum(p_far) / max(len(p_far), 1),
    }
    return (100.0 * factual / total, 100.0 * counterfactual / total, sum(shifts) / max(len(shifts), 1),
            relation, predictions)


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    global LAMBDA_CF, LAMBDA_EDGE
    parser = argparse.ArgumentParser(description="Train SCM-VLM")
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-answers", type=int, default=MAX_ANSWERS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--no-cf", action="store_true", help="ablation: disable the counterfactual loss")
    parser.add_argument("--no-edge-supervision", action="store_true",
                        help="ablation: disable the box-contact relation supervision")
    parser.add_argument("--graph-mode", choices=["soft", "legacy", "none"], default="soft",
                        help="soft (default): weighted top-k edges, no hard threshold. "
                             "legacy: the original hard-threshold graph (collapsed in practice). "
                             "none: ablation without the causal graph (no edges, no message passing, "
                             "no DAG/sparsity/edge losses)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed for weight init and sample order (the data split is fixed)")
    parser.add_argument("--tag", default=None,
                        help="suffix for checkpoint/results files so runs do not overwrite each other "
                             "(default: none for the full model, 'nograph' for --graph-mode none)")
    args = parser.parse_args()
    tag = args.tag if args.tag is not None else ("nograph" if args.graph_mode == "none" else "")
    if args.tag is None and args.seed != SEED:
        tag = f"{tag}_s{args.seed}" if tag else f"s{args.seed}"
    suffix = f"_{tag}" if tag else ""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"scm_vlm_latest{suffix}.pt") if suffix else CHECKPOINT_PATH

    # --seed changes weight init and sample order only; the train/validation split always
    # uses SEED, so every run (full / no graph / any seed) is scored on the same questions
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if args.no_cf:
        LAMBDA_CF = 0.0
    if args.no_edge_supervision:
        LAMBDA_EDGE = 0.0

    print(f"\nUSING DEVICE: {DEVICE}")
    print(f"GRAPH MODE: {args.graph_mode}" + (" (ABLATION: causal graph removed)" if args.graph_mode == "none" else ""))

    dataset = VQACOCO(IMAGE_DIR, QUESTION_FILE, ANNOTATION_FILE, max_answers=args.max_answers)
    train_idx, val_idx = split_by_image(dataset, args.samples, args.val_fraction, SEED)
    print(f"Answer vocabulary: {len(dataset.answer2idx)} | train questions: {len(train_idx)} "
          f"| validation questions: {len(val_idx)}")

    model = SCMVLM(num_classes=len(dataset.answer2idx), question_aware=True,
                   graph_mode=args.graph_mode, graph_top_k=GRAPH_TOP_K,
                   graph_geometry=args.graph_mode == "soft").to(DEVICE)
    cache = FeatureCache(model, CACHE_DIR)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_val = -1.0
    model.train()

    print("\n========================")
    print("TRAINING STARTED")
    print("========================")

    for epoch in range(args.epochs):
        order = list(train_idx)
        random.shuffle(order)
        total_loss, correct, seen = 0.0, 0, 0
        graph_mass, edge_prob = 0.0, 0.0
        optimizer.zero_grad()
        in_batch = 0

        for step, index in enumerate(order, start=1):
            image_path, question, label = dataset[index]
            try:
                raw, boxes = cache.image(image_path)
                if raw.shape[0] == 0:
                    continue
                loss, logits, parts = sample_losses(model, raw, cache.question(question), label.to(DEVICE),
                                                    boxes, supervise_edges=LAMBDA_EDGE > 0)
            except Exception as error:
                print(f"TRAIN SKIP {image_path}: {error}")
                continue
            if torch.isnan(loss):
                continue

            (loss / args.batch_size).backward()
            in_batch += 1
            correct += int(logits.argmax(-1).item() == label.item())
            total_loss += loss.item()
            graph_mass += float(parts["graph_mass"])
            edge_prob += float(parts["edge_prob"])
            seen += 1

            if in_batch == args.batch_size or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                in_batch = 0

            if seen % 500 == 0:
                print(f"[TRAIN] epoch={epoch + 1} sample={seen} loss={loss.item():.4f} "
                      f"task={parts['task'].item():.4f} cf={parts['cf'].item():.4f} "
                      f"acyc={parts['acyclic'].item():.4f} running_acc={correct / seen:.4f} "
                      f"graph_mass={graph_mass / seen:.3f}")

        scheduler.step()
        train_acc = correct / max(seen, 1)
        val_factual, val_cf, val_shift, relation, predictions = evaluate(model, dataset, val_idx, cache)
        mean_mass = graph_mass / max(seen, 1)
        mean_prob = edge_prob / max(seen, 1)
        history.append({"epoch": epoch + 1, "loss": total_loss, "train_accuracy": train_acc,
                        "val_factual_accuracy": val_factual, "val_counterfactual_accuracy": val_cf,
                        "val_intervention_shift": val_shift, "graph_mass": mean_mass,
                        "mean_edge_prob": mean_prob, **relation})

        print("\n========================")
        print(f"EPOCH {epoch + 1}")
        print("========================")
        print(f"Loss                     : {total_loss:.4f}")
        print(f"Train accuracy           : {train_acc:.4f}")
        print(f"Val factual accuracy     : {val_factual:.2f}%")
        print(f"Val counterfactual acc.  : {val_cf:.2f}%")
        if model.has_graph:
            print(f"Graph: parent weight/obj : {mean_mass:.3f}   mean edge prob: {mean_prob:.3f}")
            print(f"Relations (val)          : edge prob for touching objects "
                  f"{relation['val_edge_prob_contact']:.3f} vs distant {relation['val_edge_prob_far']:.3f}")
        else:
            print("Graph                    : none (ablation - objects exchange no messages)")
        if model.has_graph and mean_mass < 0.05:
            print("WARNING: the relation graph is collapsing (almost no edge weight left).")

        pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"training_history{suffix}.csv"), index=False)
        if val_factual > best_val:
            best_val = val_factual
            save_checkpoint(model, dataset, epoch + 1, train_acc, val_factual, checkpoint_path)
            # every validation answer of the best epoch, for compare_answers.py
            pd.DataFrame(predictions).to_csv(os.path.join(RESULTS_DIR, f"val_predictions{suffix}.csv"),
                                             index=False)

    best = max(history, key=lambda row: row["val_factual_accuracy"]) if history else {}
    results = pd.DataFrame([{
        "Factual Accuracy": best.get("val_factual_accuracy", 0.0),
        "Counterfactual Accuracy": best.get("val_counterfactual_accuracy", 0.0),
        "Avg Intervention Shift": best.get("val_intervention_shift", 0.0),
        "Best Epoch": best.get("epoch", 0),
        "Validation Questions": len(val_idx),
        "Graph Mode": args.graph_mode,
    }])
    results.to_csv(os.path.join(RESULTS_DIR, f"final_metrics{suffix}.csv"), index=False)

    print("\n========================")
    print("FINAL RESULTS (held-out validation, best epoch)")
    print("========================")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
