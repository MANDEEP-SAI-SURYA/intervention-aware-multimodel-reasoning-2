"""Compare the full SCM-VLM with the ablation WITHOUT the causal graph.

1) On the held-out validation questions (needs both training runs):

    python train.py                      # full model  -> results/val_predictions.csv
    python train.py --graph-mode none    # no graph    -> results/val_predictions_nograph.csv
    python compare_answers.py

   Prints accuracy per answer type, how often the two models agree, which questions
   only one of them gets right, and example questions where the answers differ.
   The merged table is written to results/compare_full_vs_nograph.csv.

2) On any image + question (loads both checkpoints):

    python compare_answers.py --image images/Man_riding_bicycle.png \
        --question "What is the man riding?" --question "How many cars are there?"
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import pandas as pd

RESULTS_DIR = "results"
FULL_CSV = os.path.join(RESULTS_DIR, "val_predictions.csv")
NOGRAPH_CSV = os.path.join(RESULTS_DIR, "val_predictions_nograph.csv")
FULL_CKPT = "checkpoints/scm_vlm_latest.pt"
NOGRAPH_CKPT = "checkpoints/scm_vlm_latest_nograph.pt"


def answer_type(answer: str) -> str:
    """VQA v2 style answer types."""
    answer = str(answer).strip().lower()
    if answer in ("yes", "no"):
        return "yes/no"
    if answer.replace(".", "", 1).isdigit():
        return "number"
    return "other"


def compare_csv(full_csv: str, nograph_csv: str, examples: int) -> None:
    for path in (full_csv, nograph_csv):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {path}. Train both models first (see the top of this file).")
    full = pd.read_csv(full_csv)
    nograph = pd.read_csv(nograph_csv)
    keep = ["question_index", "image", "question", "ground_truth", "num_objects"]
    merged = full[keep + ["prediction", "correct", "confidence", "counterfactual_prediction"]].merge(
        nograph[["question_index", "prediction", "correct", "confidence", "counterfactual_prediction"]],
        on="question_index", suffixes=("_full", "_nograph"),
    )
    if merged.empty:
        raise SystemExit("The two prediction files share no questions - were both runs made with the "
                         "same --samples / --val-fraction / --max-answers?")
    if len(merged) < max(len(full), len(nograph)):
        print(f"NOTE: only {len(merged)} questions are in both files "
              f"(full {len(full)}, no-graph {len(nograph)}).")

    merged["answer_type"] = merged["ground_truth"].map(answer_type)
    merged["agree"] = (merged["prediction_full"] == merged["prediction_nograph"]).astype(int)
    for name in ("full", "nograph"):
        merged[f"stable_{name}"] = (merged[f"prediction_{name}"] ==
                                    merged[f"counterfactual_prediction_{name}"]).astype(int)

    n = len(merged)
    print("\n==============================================")
    print(f"FULL SCM-VLM vs. NO CAUSAL GRAPH  ({n} validation questions)")
    print("==============================================")
    print(f"{'':28s}{'full':>10s}{'no graph':>12s}")
    print(f"{'Accuracy (%)':28s}{100 * merged.correct_full.mean():10.2f}{100 * merged.correct_nograph.mean():12.2f}")
    print(f"{'Mean confidence':28s}{merged.confidence_full.mean():10.3f}{merged.confidence_nograph.mean():12.3f}")
    print(f"{'Answer kept under do() (%)':28s}{100 * merged.stable_full.mean():10.2f}"
          f"{100 * merged.stable_nograph.mean():12.2f}")

    print("\nAccuracy by answer type (%):")
    by_type = merged.groupby("answer_type").agg(
        questions=("question_index", "size"),
        full=("correct_full", "mean"),
        no_graph=("correct_nograph", "mean"),
    )
    by_type[["full", "no_graph"]] *= 100
    print(by_type.round(2).to_string())

    bins = pd.cut(merged["num_objects"], bins=[0, 3, 6, 9, 100], labels=["1-3", "4-6", "7-9", "10+"])
    by_objects = merged.groupby(bins, observed=True).agg(
        questions=("question_index", "size"),
        full=("correct_full", "mean"),
        no_graph=("correct_nograph", "mean"),
    )
    by_objects[["full", "no_graph"]] *= 100
    print("\nAccuracy by number of detected objects (%):")
    print(by_objects.round(2).to_string())

    both = int(((merged.correct_full == 1) & (merged.correct_nograph == 1)).sum())
    only_full = int(((merged.correct_full == 1) & (merged.correct_nograph == 0)).sum())
    only_nograph = int(((merged.correct_full == 0) & (merged.correct_nograph == 1)).sum())
    neither = n - both - only_full - only_nograph
    print(f"\nSame answer from both models : {100 * merged.agree.mean():.2f}%")
    print(f"Both correct                 : {both}")
    print(f"Only full model correct      : {only_full}")
    print(f"Only no-graph model correct  : {only_nograph}")
    print(f"Both wrong                   : {neither}")
    # McNemar test on the disagreements (is the accuracy difference more than noise?)
    discordant = only_full + only_nograph
    if discordant:
        from math import erf, sqrt
        z = (abs(only_full - only_nograph) - 1) / sqrt(discordant)
        p = 2 * (1 - 0.5 * (1 + erf(max(z, 0) / sqrt(2))))
        print(f"McNemar test                 : z={z:.2f}, p={p:.4f} "
              f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")

    diff = merged[merged.agree == 0]
    if examples and not diff.empty:
        print(f"\nExamples where the answers differ (first {min(examples, len(diff))}):")
        for _, row in diff.head(examples).iterrows():
            mark_f = "+" if row.correct_full else " "
            mark_n = "+" if row.correct_nograph else " "
            print(f"  Q: {row.question}  [{row.image}]")
            print(f"     truth: {row.ground_truth:<12} full: {mark_f}{row.prediction_full:<12} "
                  f"no graph: {mark_n}{row.prediction_nograph}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "compare_full_vs_nograph.csv")
    merged.to_csv(out, index=False)
    print(f"\nSaved the question-by-question table to {out}")


def compare_live(image: str, questions: list[str], full_ckpt: str, nograph_ckpt: str, top: int) -> None:
    import torch
    from main import load_trained_model

    if not os.path.isfile(image):
        raise SystemExit(f"image not found: {image}")
    answers = {}
    for name, path in (("full", full_ckpt), ("no graph", nograph_ckpt)):
        print(f"\nLoading {name} model from {path}")
        model, idx2answer = load_trained_model(path)
        answers[name] = []
        with torch.inference_mode():
            for question in questions:
                probs = torch.softmax(model(image, question)["logits"][0], dim=-1)
                values, indices = probs.topk(top)
                answers[name].append([(idx2answer.get(i.item(), "?"), v.item()) for v, i in zip(values, indices)])
        del model

    print("\n==============================================")
    print(f"IMAGE: {image}")
    print("==============================================")
    for q_index, question in enumerate(questions):
        print(f"\nQ: {question}")
        for name in answers:
            ranked = ", ".join(f"{a} ({p:.2f})" for a, p in answers[name][q_index])
            print(f"  {name:9s}: {ranked}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full SCM-VLM with the no-causal-graph ablation")
    parser.add_argument("--image", help="compare on this image instead of the validation predictions")
    parser.add_argument("--question", action="append", help="question(s) for --image (repeatable)")
    parser.add_argument("--full-checkpoint", default=FULL_CKPT)
    parser.add_argument("--nograph-checkpoint", default=NOGRAPH_CKPT)
    parser.add_argument("--full-csv", default=FULL_CSV)
    parser.add_argument("--nograph-csv", default=NOGRAPH_CSV)
    parser.add_argument("--examples", type=int, default=25, help="differing examples to print")
    parser.add_argument("--top", type=int, default=3, help="answers to show per model with --image")
    args = parser.parse_args()

    if args.image:
        if not args.question:
            parser.error("give at least one --question with --image")
        compare_live(args.image, args.question, args.full_checkpoint, args.nograph_checkpoint, args.top)
    else:
        compare_csv(args.full_csv, args.nograph_csv, args.examples)


if __name__ == "__main__":
    main()
