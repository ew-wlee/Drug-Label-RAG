"""Summarize paired effects with task-stratified bootstrap intervals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from experiment import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("variant")
    parser.add_argument("--answer-f1", action="store_true",
                        help="Compare answer F1 on paired answerable outputs.")
    args = parser.parse_args()
    systems = defaultdict(dict)
    for row in read_jsonl(args.retrieval):
        has_score = ("f1" in row and row["task"] != "refusal"
                     if args.answer_f1 else bool(row.get("metrics")))
        if has_score:
            systems[row["system"]][row["qid"]] = row
    base, variant = systems[args.baseline], systems[args.variant]
    if not base or base.keys() != variant.keys():
        parser.error("Both systems must contain identical answerable question IDs.")
    rng = random.Random(5018)
    report = {}
    metrics = ("f1",) if args.answer_f1 else ("recall@5", "evidence80@1", "evidence80@5", "mrr@10")
    for metric in metrics:
        by_task = defaultdict(list)
        for qid, row in base.items():
            other = variant[qid]
            original_score = row[metric] if args.answer_f1 else row["metrics"][metric]
            variant_score = other[metric] if args.answer_f1 else other["metrics"][metric]
            by_task[row["task"]].append(variant_score - original_score)
        bootstrap = sorted(
            sum(sum(rng.choices(values, k=len(values))) for values in by_task.values()) / len(base)
            for _ in range(5000)
        )
        report[metric] = {
            "delta": statistics.mean(value for values in by_task.values() for value in values),
            "ci95": [bootstrap[125], bootstrap[4874]],
        }
    print(json.dumps({"baseline": args.baseline, "variant": args.variant,
                      "n": len(base), "effects": report}, indent=2))


if __name__ == "__main__":
    main()
