"""Show how the model reasons about one image + question, for the full model and the
model without the causal graph side by side.

    python explain_reasoning.py --image images/Man_riding_bicycle.png --question "What is the man riding?"

For each model it reports
  * the answer (top 3 with probabilities)
  * attention   - which objects the answer head looked at for this question
  * evidence    - how much the probability of the answer drops when each object is removed
                  from the scene (the objects the answer actually depends on)
and for the full model additionally
  * the reasoning chain - the relations (graph edges) that fed information into the
                  most-attended object: parent -> object -> answer
  * relation reliance  - the answer when message passing is switched off at test time

A figure (image with boxes, attention, learned edges, and per-object evidence bars) and a
JSON file are written to results/reasoning/.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import re

import torch

from configs.config import DEVICE

OUT_DIR = os.path.join("results", "reasoning")
EDGE_DRAW_MIN = 0.2


@torch.no_grad()
def explain(model, raw, question_embedding, boxes, labels, idx2answer, top=3):
    """raw [N, 256] DETR embeddings, question_embedding [768], boxes [N, 4] scaled 0-1."""
    raw, boxes = raw.to(DEVICE), boxes.to(DEVICE)
    q = question_embedding.to(DEVICE)[None]
    out = model.forward_features(raw[None], q, boxes[None])
    probs = torch.softmax(out["logits"][0], -1)
    answer = int(probs.argmax())
    values, indices = probs.topk(min(top, probs.numel()))
    _, attention = model.answer_head(out["hidden"], q)
    attention = attention[0].cpu()
    n = raw.shape[0]

    # evidence: remove one object at a time (the graph is recomputed without it)
    evidence = []
    for k in range(n):
        keep = [i for i in range(n) if i != k]
        if not keep:
            evidence.append(0.0)
            continue
        removed = model.forward_features(raw[keep][None], q, boxes[keep][None])
        p = torch.softmax(removed["logits"][0], -1)[answer].item()
        evidence.append(probs[answer].item() - p)

    result = {
        "graph": model.has_graph,
        "answer": idx2answer.get(answer, str(answer)),
        "top_answers": [(idx2answer.get(int(i), str(int(i))), float(v)) for v, i in zip(values, indices)],
        "objects": [{"node": i, "label": labels[i], "attention": float(attention[i]),
                     "evidence": float(evidence[i])} for i in range(n)],
    }

    if model.has_graph:
        adjacency = out["adjacency"][0].cpu()        # A[i, j] > 0: j -> i (j is a parent of i)
        result["edges"] = [{"from": j, "to": i, "weight": float(adjacency[i, j])}
                           for i in range(n) for j in range(n) if adjacency[i, j] > 1e-6]
        focus = int(attention.argmax())
        chain = []
        for j in adjacency[focus].argsort(descending=True)[:2].tolist():
            if adjacency[focus, j] <= 1e-6:
                continue
            grand = [(g, float(adjacency[j, g])) for g in adjacency[j].argsort(descending=True)[:1].tolist()
                     if adjacency[j, g] > 1e-6 and g != focus]
            chain.append({"parent": j, "weight": float(adjacency[focus, j]),
                          "grandparents": [{"node": g, "weight": w} for g, w in grand]})
        result["focus"] = focus
        result["chain"] = chain

        hidden = out["objects"]
        for layer in model.layers:
            hidden, _ = layer(hidden, None)
        p_off = torch.softmax(model.decode(hidden, q)[0], -1)
        result["without_messages"] = {"answer": idx2answer.get(int(p_off.argmax()), "?"),
                                      "p_original_answer": float(p_off[answer])}
    return result


def print_report(name, result, labels):
    print(f"\n----- {name} -----")
    print("Answer:", ", ".join(f"{a} ({p:.2f})" for a, p in result["top_answers"]))
    print(f"{'node':>4} {'object':<15}{'attention':>10}{'evidence':>10}")
    for o in sorted(result["objects"], key=lambda o: -o["attention"]):
        print(f"{o['node']:>4} {o['label']:<15}{o['attention']:10.3f}{o['evidence']:+10.3f}")
    print("(evidence = drop in P(answer) when the object is removed; + means the answer relies on it)")
    if result["graph"]:
        f = result["focus"]
        print(f"\nReasoning chain into the most-attended object {f}:{labels[f]}")
        if not result["chain"]:
            print("  (no incoming edges)")
        for link in result["chain"]:
            j = link["parent"]
            start = "".join(f"{g['node']}:{labels[g['node']]} --{g['weight']:.2f}--> " for g in link["grandparents"])
            print(f"  {start}{j}:{labels[j]} --{link['weight']:.2f}--> {f}:{labels[f]} --> answer '{result['answer']}'")
        w = result["without_messages"]
        print(f"\nWith message passing switched off: answer '{w['answer']}', "
              f"P('{result['answer']}') = {w['p_original_answer']:.2f}")


def plot(image_path, boxes, labels, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(7 * len(names), 9),
                             gridspec_kw={"height_ratios": [3, 1.3]}, squeeze=False)
    for col, name in enumerate(names):
        result = results[name]
        ax = axes[0, col]
        ax.imshow(image)
        ax.set_title(f"{name}: '{result['answer']}' ({result['top_answers'][0][1]:.2f})", fontsize=11)
        ax.axis("off")
        attention = torch.tensor([o["attention"] for o in result["objects"]])
        scale = attention / attention.max().clamp(min=1e-8)
        centres = []
        for i, (x0, y0, x1, y1) in enumerate((boxes * torch.tensor([w, h, w, h])).tolist()):
            a = float(scale[i])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1 + 3 * a,
                                   ec=(1.0, 0.25, 0.1, 0.25 + 0.75 * a)))
            ax.text(x0, y0, f"{i}:{labels[i]} {result['objects'][i]['attention']:.2f}", fontsize=7,
                    color="white", bbox=dict(fc="black", alpha=0.5, pad=1, lw=0))
            centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
        for edge in result.get("edges", []):
            if edge["weight"] < EDGE_DRAW_MIN:
                continue
            ax.add_patch(FancyArrowPatch(centres[edge["from"]], centres[edge["to"]], arrowstyle="-|>",
                                         mutation_scale=10, lw=0.5 + 2.5 * edge["weight"],
                                         color=(0.1, 0.6, 1.0, 0.4 + 0.6 * edge["weight"])))
        if not result["graph"]:
            ax.text(0.01, 0.01, "no causal graph: objects are not connected", transform=ax.transAxes,
                    fontsize=8, color="white", bbox=dict(fc="black", alpha=0.6, lw=0))

        bar = axes[1, col]
        evidence = [o["evidence"] for o in result["objects"]]
        bar.bar(range(len(evidence)), evidence, color=["#d1495b" if e > 0 else "#9aa5b1" for e in evidence])
        bar.set_xticks(range(len(evidence)))
        bar.set_xticklabels([f"{i}:{l}" for i, l in enumerate(labels)], rotation=60, fontsize=7, ha="right")
        bar.axhline(0, color="black", lw=0.5)
        bar.set_ylabel("drop in P(answer)\nwhen removed", fontsize=8)
        bar.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    from main import load_trained_model

    parser = argparse.ArgumentParser(description="Explain the reasoning of the full and the no-graph model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--full-checkpoint", default="checkpoints/scm_vlm_latest.pt")
    parser.add_argument("--nograph-checkpoint", default="checkpoints/scm_vlm_latest_nograph.pt")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")

    models = {}
    for name, path in (("full model", args.full_checkpoint), ("no causal graph", args.nograph_checkpoint)):
        if os.path.isfile(path):
            models[name] = load_trained_model(path)
        else:
            print(f"skipping {name}: {path} not found")
    if not models:
        raise SystemExit("no checkpoint found")

    first = next(iter(models.values()))[0]
    with torch.no_grad():
        vision = first.vision(args.image)
        q = first.text(args.question)[0]
    raw = vision["embeddings"][0]
    boxes = vision["boxes_norm"][0].float()
    labels = vision["labels"][0]
    if raw.shape[0] == 0:
        raise SystemExit("DETR found no objects in this image")

    print(f"\nImage: {args.image}\nQuestion: {args.question}")
    print("Objects:", ", ".join(f"{i}:{l}" for i, l in enumerate(labels)))
    results = {}
    for name, (model, idx2answer) in models.items():
        results[name] = explain(model, raw, q, boxes, labels, idx2answer)
        print_report(name, results[name], labels)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0] + "_" + re.sub(r"\W+", "_", args.question.lower()).strip("_")[:40]
    figure = os.path.join(OUT_DIR, stem + ".png")
    plot(args.image, boxes.cpu(), labels, results, figure)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as file:
        json.dump({"image": args.image, "question": args.question, "labels": labels, "models": results}, file, indent=2)
    print(f"\nSaved {figure} and {stem}.json")


if __name__ == "__main__":
    main()
