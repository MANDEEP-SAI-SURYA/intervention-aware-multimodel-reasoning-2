"""Does the causal graph make the model learn relationships between objects?

Three experiments on the held-out validation images (same split as train.py), run after
both models are trained:

    python train.py --samples 3000 --epochs 5
    python train.py --samples 3000 --epochs 5 --graph-mode none
    python relation_analysis.py --samples 3000

1. Graph knock-out (full model only, no retraining).  The trained full model answers every
   validation question three times: with its learned graph, with message passing switched
   off, and with a scrambled graph (every object linked to the wrong objects).  If accuracy
   drops without the graph or with wrong edges, the model's answers depend on the
   relations it learned.

2. What the learned edges connect.  For every object, is its strongest parent an object it
   physically touches / one directly above or below it, compared with a randomly chosen
   object of the same scene (chance)?

3. Relation probes.  A linear classifier is trained to read relational facts about an
   object from that object's final representation alone:
     * touches_person   - the object touches a person (needs the neighbour's class)
     * same_class_count - how many OTHER objects of its class are in the scene (0 / 1 / 2+)
     * object_above     - another object sits directly above it
     * own_class        - control: the object's own class (no relation needed)
   compared for raw DETR features, the full model and the no-graph model.  If the graph
   carries relations, the full model should beat the no-graph model on the relational
   probes while all are similar on the control.  Probes are trained on 70 % of the
   validation images and scored (balanced accuracy, chance = 1 / classes) on the other 30 %.

   Note: raw DETR features already contain some scene context (DETR's decoder lets its
   queries attend to each other), so the interesting number is how much each model adds
   on top of raw DETR.  touches_person partly overlaps with the full model's training
   signal (edges are weakly supervised with box contact); same_class_count and
   object_above do not.

Writes results/relation_analysis.json, results/relation_probes.csv and
results/relation_probes.png.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F

from configs.config import DEVICE

RESULTS_DIR = "results"
CONTACT_GAP = 0.02
PROBE_SPLIT = 0.7


# =====================================================
# relation targets computed from DETR boxes / labels
# =====================================================

def touching(boxes: torch.Tensor, gap: float = CONTACT_GAP) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    gap_x = (torch.maximum(x0[:, None], x0[None, :]) - torch.minimum(x1[:, None], x1[None, :])).clamp(min=0)
    gap_y = (torch.maximum(y0[:, None], y0[None, :]) - torch.minimum(y1[:, None], y1[None, :])).clamp(min=0)
    touch = (gap_x <= gap) & (gap_y <= gap)
    touch.fill_diagonal_(False)
    return touch


def vertical_neighbours(boxes: torch.Tensor) -> torch.Tensor:
    """[N, N] True where object j is directly above object i (horizontal overlap,
    j's centre higher than i's top half)."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    overlap = (torch.minimum(x1[:, None], x1[None, :]) - torch.maximum(x0[:, None], x0[None, :])) > 0
    cy = (y0 + y1) / 2
    above = cy[None, :] < y0[:, None] + 0.25 * (y1 - y0)[:, None]
    result = overlap & above
    result.fill_diagonal_(False)
    return result


def node_targets(labels: list[str], boxes: torch.Tensor, class_index: dict[str, int]) -> dict[str, list[int]]:
    n = len(labels)
    touch = touching(boxes)
    above = vertical_neighbours(boxes)
    targets = defaultdict(list)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        targets["touches_person"].append(int(any(touch[i, j] and labels[j] == "person" for j in others)))
        same = sum(labels[j] == labels[i] for j in others)
        targets["same_class_count"].append(min(same, 2))
        targets["object_above"].append(int(bool(above[i].any())))
        targets["own_class"].append(class_index.get(labels[i], len(class_index)))
    return targets


# =====================================================
# model helpers
# =====================================================

@torch.no_grad()
def run(model, raw, q, boxes, mode="normal", generator=None):
    """mode: normal | no_messages (layers run without the graph) | scrambled (each object's
    parents are swapped for other objects of the same scene)."""
    out = model.forward_features(raw[None].to(DEVICE), q[None].to(DEVICE), boxes[None].to(DEVICE))
    if mode == "normal" or not model.has_graph:
        return out
    objects, adjacency = out["objects"], out["adjacency"]
    if mode == "no_messages":
        hidden = objects
        for layer in model.layers:
            hidden, _ = layer(hidden, None)
    elif mode == "scrambled":
        n = adjacency.shape[-1]
        perm = torch.randperm(n, generator=generator)
        if n > 1:
            while torch.any(perm == torch.arange(n)):     # every object gets different parents
                perm = torch.randperm(n, generator=generator)
        scrambled = adjacency[:, :, perm.to(adjacency.device)]
        eye = torch.eye(n, dtype=torch.bool, device=adjacency.device)
        scrambled = scrambled.masked_fill(eye, 0.0)
        scrambled = scrambled / (scrambled.sum(-1, keepdim=True) + 1e-8)
        hidden, _ = model.run_causal_layers(objects, scrambled)
    else:
        raise ValueError(mode)
    logits = model.decode(hidden, out["question_embedding"])
    return {**out, "logits": logits, "hidden": hidden}


def graph_knockout(model, dataset, indices, cache):
    if not model.has_graph:
        return None
    generator = torch.Generator().manual_seed(0)
    correct = defaultdict(int)
    changed = defaultdict(int)
    total = 0
    for index in indices:
        image_path, question, label = dataset[index]
        raw, boxes = cache.image(image_path)
        if raw.shape[0] < 2:
            continue
        q = cache.question(question)
        normal = run(model, raw, q, boxes).get("logits").argmax(-1).item()
        correct["with learned graph"] += int(normal == label.item())
        for mode, name in (("no_messages", "message passing off"), ("scrambled", "scrambled graph")):
            pred = run(model, raw, q, boxes, mode, generator)["logits"].argmax(-1).item()
            correct[name] += int(pred == label.item())
            changed[name] += int(pred != normal)
        total += 1
    total = max(total, 1)
    return {name: {"accuracy": 100 * c / total,
                   "answers_changed": 100 * changed.get(name, 0) / total}
            for name, c in correct.items()} | {"questions": total}


def edge_alignment(model, images, cache, question_of):
    """How often an object's strongest learned parent touches it / is directly above or
    below it, against the rate for a random other object (chance)."""
    if not model.has_graph:
        return None
    hits = defaultdict(float)
    chance = defaultdict(float)
    nodes = 0
    for image_path in images:
        raw, boxes = cache.image(image_path)
        n = raw.shape[0]
        if n < 2:
            continue
        out = run(model, raw, cache.question(question_of[image_path]), boxes)
        adjacency = out["adjacency"][0].cpu()
        parent = adjacency.argmax(-1)
        touch = touching(boxes)
        above = vertical_neighbours(boxes)
        vertical = above | above.T
        for i in range(n):
            j = parent[i].item()
            hits["touching"] += float(touch[i, j])
            hits["above_or_below"] += float(vertical[i, j])
            chance["touching"] += touch[i].float().sum().item() / (n - 1)
            chance["above_or_below"] += vertical[i].float().sum().item() / (n - 1)
            nodes += 1
    nodes = max(nodes, 1)
    return {key: {"strongest_parent_%": 100 * hits[key] / nodes, "random_object_%": 100 * chance[key] / nodes}
            for key in hits} | {"objects": nodes}


# =====================================================
# linear probes
# =====================================================

def balanced_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    scores = [(pred[target == c] == c).float().mean().item() for c in target.unique()]
    return 100 * sum(scores) / len(scores)


def train_probe(x_train, y_train, x_test, y_test, epochs=300, seed=0) -> float:
    torch.manual_seed(seed)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    classes = int(max(y_train.max(), y_test.max()).item()) + 1
    probe = torch.nn.Linear(x_train.shape[1], classes)
    counts = torch.bincount(y_train, minlength=classes).float()
    weight = torch.where(counts > 0, counts.sum() / (counts * classes).clamp(min=1), torch.zeros_like(counts))
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        F.cross_entropy(probe(x_train), y_train, weight=weight).backward()
        optimizer.step()
    with torch.no_grad():
        return balanced_accuracy(probe(x_test).argmax(-1), y_test)


@torch.no_grad()
def collect_features(models, images, cache, question_of, class_index):
    feats = defaultdict(list)
    targets = defaultdict(list)
    image_of_node = []
    for image_path in images:
        raw, boxes = cache.image(image_path)
        labels = cache.labels(image_path)
        if raw.shape[0] < 2:
            continue
        q = cache.question(question_of[image_path])
        feats["raw DETR"].append(raw)
        for name, model in models.items():
            feats[name].append(run(model, raw, q, boxes)["hidden"][0].cpu())
        for key, values in node_targets(labels, boxes, class_index).items():
            targets[key].extend(values)
        image_of_node.extend([image_path] * raw.shape[0])
    feats = {name: torch.cat(parts) for name, parts in feats.items()}
    targets = {key: torch.tensor(values) for key, values in targets.items()}
    return feats, targets, image_of_node


def relation_probes(models, images, cache, question_of, seed=0):
    # control classes: the 10 most frequent DETR classes, the rest = "other"
    counts = defaultdict(int)
    for image_path in images:
        cache.image(image_path)
        for label in cache.labels(image_path):
            counts[label] += 1
    top = sorted(counts, key=lambda c: -counts[c])[:10]
    class_index = {label: i for i, label in enumerate(top)}

    feats, targets, image_of_node = collect_features(models, images, cache, question_of, class_index)
    unique = sorted(set(image_of_node))
    random.Random(seed).shuffle(unique)
    train_images = set(unique[:int(len(unique) * PROBE_SPLIT)])
    train_mask = torch.tensor([image in train_images for image in image_of_node])

    rows = []
    for task, y in targets.items():
        if len(y[train_mask].unique()) < 2 or len(y[~train_mask].unique()) < 2:
            print(f"probe '{task}': only one class present, skipped")
            continue
        row = {"probe": task, "classes": int(y.max().item()) + 1, "chance": 100 / (int(y.max().item()) + 1)}
        for name, x in feats.items():
            row[name] = train_probe(x[train_mask], y[train_mask], x[~train_mask], y[~train_mask], seed=seed)
        rows.append(row)
    return pd.DataFrame(rows), int(train_mask.sum()), int((~train_mask).sum())


def plot_probes(table: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = [c for c in table.columns if c not in ("probe", "classes", "chance")]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(reps)
    for k, rep in enumerate(reps):
        ax.bar([i + k * width for i in range(len(table))], table[rep], width, label=rep)
    for i, chance in enumerate(table["chance"]):
        ax.hlines(chance, i - width / 2, i + width * (len(reps) - 0.5), colors="black", linestyles="dashed",
                  label="chance" if i == 0 else None)
    ax.set_xticks([i + width * (len(reps) - 1) / 2 for i in range(len(table))])
    ax.set_xticklabels(table["probe"])
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_title("What can be read from one object's representation?")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================
# main
# =====================================================

def main() -> None:
    import train
    from datasets.datasetBuilder import VQACOCO
    from main import load_trained_model

    parser = argparse.ArgumentParser(description="Relation-learning analysis: full model vs no causal graph")
    parser.add_argument("--samples", type=int, default=train.TRAIN_SAMPLES, help="same value as in training")
    parser.add_argument("--val-fraction", type=float, default=train.VAL_FRACTION)
    parser.add_argument("--max-answers", type=int, default=train.MAX_ANSWERS)
    parser.add_argument("--full-checkpoint", default="checkpoints/scm_vlm_latest.pt")
    parser.add_argument("--nograph-checkpoint", default="checkpoints/scm_vlm_latest_nograph.pt")
    args = parser.parse_args()

    dataset = VQACOCO(train.IMAGE_DIR, train.QUESTION_FILE, train.ANNOTATION_FILE, max_answers=args.max_answers)
    _, val_idx = train.split_by_image(dataset, args.samples, args.val_fraction, train.SEED)
    full, _ = load_trained_model(args.full_checkpoint)
    nograph, _ = load_trained_model(args.nograph_checkpoint)
    cache = train.FeatureCache(full, train.CACHE_DIR)

    question_of = {}
    for index in val_idx:
        image_path, question, _ = dataset[index]
        question_of.setdefault(image_path, question)
    images = sorted(question_of)
    print(f"validation: {len(val_idx)} questions on {len(images)} images")

    report = {}
    print("\n=== 1. Graph knock-out (trained full model, no retraining) ===")
    report["knockout"] = graph_knockout(full, dataset, val_idx, cache)
    for name, value in report["knockout"].items():
        if isinstance(value, dict):
            print(f"  {name:22s} accuracy {value['accuracy']:6.2f}%   answers changed {value['answers_changed']:6.2f}%")

    print("\n=== 2. What the learned edges connect (full model) ===")
    report["edges"] = edge_alignment(full, images, cache, question_of)
    for name, value in report["edges"].items():
        if isinstance(value, dict):
            print(f"  strongest parent is {name:15s}: {value['strongest_parent_%']:6.2f}%   "
                  f"(random object: {value['random_object_%']:6.2f}%)")

    print("\n=== 3. Relation probes (balanced accuracy %, held-out images) ===")
    table, n_train, n_test = relation_probes({"full model": full, "no graph": nograph}, images, cache, question_of)
    print(f"  probe training objects: {n_train}, test objects: {n_test}")
    print(table.round(2).to_string(index=False))
    report["probes"] = table.to_dict(orient="records")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "relation_probes.csv"), index=False)
    plot_probes(table, os.path.join(RESULTS_DIR, "relation_probes.png"))
    with open(os.path.join(RESULTS_DIR, "relation_analysis.json"), "w") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved {RESULTS_DIR}/relation_analysis.json, relation_probes.csv, relation_probes.png")


if __name__ == "__main__":
    main()
