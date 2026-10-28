# SCM-VLM Research Prototype

Implementation of the core architecture proposed in:

**Intervention Aware Multimodal Reasoning via Structural Causal Graph Integration in Vision Language Transformers**

This version follows the requested research-prototype layout and includes:

- DETR-based object extraction
- BERT-based question encoding
- learnable SCM graph construction
- DAG regularization
- multi-head causal attention
- intervention operators with descendant recomputation
- counterfactual-ready training pipeline
- evaluation utilities
- graph visualization

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

DETR and BERT are frozen; their outputs are cached in `cache/` on the first epoch,
so later epochs are much faster.

```bash
python train.py                               # 20k questions, 10 epochs, top-3129 answers
python train.py --samples 3000 --epochs 20    # smaller run
python train.py --no-cf                       # ablation: no counterfactual loss
```

Accuracies in `results/final_metrics.csv` and `results/training_history.csv` are
measured on a held-out validation split (separate images), not on training data.

**Relation graph.** Each object attends over its 3 most likely parents (soft edges, no hard
threshold, so the graph cannot collapse to empty). The edge scorer combines appearance
(DETR features + question) with a separate box-geometry branch (relative position, gaps,
overlap, size ratio), and edges are weakly supervised: objects whose boxes touch should be
linked. Each epoch logs the parent weight per object and, on validation images, the edge
probability for touching vs. distant objects. Ablation: `--no-edge-supervision`;
original graph: `--graph-mode legacy`.

> Checkpoints from before this change load with the old graph (hard 0.5 threshold), which
> ended with no edges at all. Retrain to get working relations.

> Checkpoints trained before the answer-head fix still load, but that head never
> saw the question, so it answers every question the same way (mostly "yes"/"no").
> Retrain to get question-dependent answers.

## Ablation: without the causal graph

```bash
python train.py --samples 3000 --epochs 5                    # full model
python train.py --samples 3000 --epochs 5 --graph-mode none  # same run, causal graph removed
python compare_answers.py                                    # compare the answers on validation
python compare_answers.py --image images/Man_riding_bicycle.png --question "What is the man riding?"
python main.py --checkpoint checkpoints/scm_vlm_latest_nograph.pt --image photo.jpg --question "..."
```

`--graph-mode none` removes the edge scorer, the adjacency matrix and all message passing between
objects (each object is only refined by its own feed-forward layer), plus the DAG / sparsity / entropy /
edge-supervision losses. The encoders, question-guided answer head, counterfactual loss and data split
are unchanged. Outputs use a `_nograph` suffix (`results/final_metrics_nograph.csv`,
`results/val_predictions_nograph.csv`, `checkpoints/scm_vlm_latest_nograph.pt`).
`compare_answers.py` reports accuracy per answer type, agreement, questions only one model gets right
(McNemar test) and example disagreements. `whatif_qa.py` needs the graph and always uses the full model.

## Does the causal graph learn relationships?

```bash
python train.py --seed 1 ...; python train.py --seed 1 --graph-mode none ...   # extra seeds
python compare_answers.py --seeds 42 1 2       # mean +- std over seeds, relational vs other
python relation_analysis.py --samples 3000     # knock-out, edge alignment, relation probes
python explain_reasoning.py --image images/Man_riding_bicycle.png --question "What is the man riding?"
```

* `compare_answers.py` also splits accuracy into relational questions (spatial / interaction)
  and non-relational ones, with a McNemar test per group.
* `relation_analysis.py` (1) switches off or scrambles the trained full model's graph at test time,
  (2) checks whether each object's strongest learned parent touches it / sits above or below it
  versus chance, and (3) trains linear probes to read relational facts (touches a person, number of
  same-class objects, object above) from a single object's representation: raw DETR vs full model vs
  no-graph model.
* `explain_reasoning.py` prints and draws, for both models, the answer, which objects the answer
  head attended to, how much each object matters (answer change when it is removed), the reasoning
  chain through the graph and the answer with message passing switched off.

> **DETR feature fix:** object embeddings are now taken from the DETR query that produced each
> detection (before, they came from unrelated queries). Cached features are rebuilt automatically;
> retrain both models before comparing.

## Run on any image

```bash
python main.py --image path/to/photo.jpg --question "What is the man holding?"
python evaluate.py --image path/to/photo.jpg --question "Is the light red?"
python scene_graph_qa.py --image path/to/photo.jpg --question "What is he riding?"
```

## What-if (counterfactual) questions with do()

```bash
python whatif_qa.py --image photo.jpg --whatif "What if the bicycle were a motorcycle? What is the man riding?"
python whatif_qa.py --image photo.jpg --whatif "What if there were no umbrella, what is the person holding?"
python whatif_qa.py --image photo.jpg --whatif "What if the traffic light turns green"
python whatif_qa.py --image photo.jpg --whatif "What if there was a dog, is there a dog?"
python whatif_qa.py --image photo.jpg --question "How many cars are there?" --do "remove:car"
python whatif_qa.py --image photo.jpg --question "What is he riding?" --do "replace:#3->motorcycle"
python whatif_qa.py --image photo.jpg --whatif "What if the cars were removed, is the road busy?" --use-llm
```

Two modes:

* `--mode recompute` (default) - edit the scene, then the model **re-calculates all relations**
  between the objects and answers the question with exactly the same pipeline as a normal
  question. The output lists new, lost and changed relations and the changed object's links
  before and after.
* `--mode do` - strict SCM intervention do(X_k = x'): the learned graph is kept, only the edges
  into the changed object are cut, and only its descendants are recomputed.

```bash
python whatif_qa.py --image photo.jpg --whatif "What if the bicycle were a motorcycle? What is the man riding?"
python whatif_qa.py --image photo.jpg --mode do --whatif "What if the bicycle were a motorcycle? What is the man riding?"
```

Supported interventions: **remove** an object, **replace** it with another COCO class,
change an **attribute** (e.g. colour), or **add** an object. Answers are given for the real and the intervened scene, from the
scene graph (counting, existence, colour, relations such as riding/holding/sitting on)
and from the SCM-VLM classifier. A before/after causal-graph figure is saved to
`results/whatif/`.

## Tests

```bash
python -m pytest tests
```

## Google Collab NoteBook
https://colab.research.google.com/drive/1gc8_qWXL_sxYWe8thv1v4v5BfdpCYWNU?usp=sharing
