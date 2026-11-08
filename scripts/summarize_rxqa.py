"""Verify a completed RxQA run and reproduce its paired accuracy intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


SYSTEMS = ("no_rag", "general", "domain")
COMPARISONS = (("no_rag", "general"), ("no_rag", "domain"), ("general", "domain"))


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def row_key(row):
    return row["qid"], row["system"]


def paired_results(predictions):
    rng = random.Random(5018)
    results = []
    for base, variant in COMPARISONS:
        strata = defaultdict(list)
        for qid, row in predictions[base].items():
            other = predictions[variant][qid]
            strata[row["input"]["difficulty"]].append(int(other["correct"]) - int(row["correct"]))
        n = len(predictions[base])
        draws = sorted(
            sum(sum(rng.choices(group, k=len(group))) for group in strata.values()) / n
            for _ in range(5000)
        )
        results.append(dict(baseline=base, variant=variant, n=n,
                            accuracy_delta=sum(map(sum, strata.values())) / n,
                            ci95=[draws[125], draws[4874]]))
    return results


def verify_model(run, size, prompts, gold):
    rows = read_rows(run / f"qwen-{size}.jsonl")
    if len(rows) != len(prompts) or {row_key(row) for row in rows} != prompts.keys():
        raise ValueError(f"{size}: predictions must match the frozen question-system pairs")
    by_system = defaultdict(dict)
    for row in rows:
        match = re.fullmatch(r"([A-D])\.?", row["raw"].strip().upper())
        selected = match[1] if match else None
        qid, system = row_key(row)
        if (row["model"] != f"qwen3.5:{size}" or row["input"] != prompts[qid, system]
                or row["selected"] != selected or row["correct"] != (selected == gold[qid])):
            raise ValueError(f"{size}: input or score mismatch for {qid}/{system}")
        by_system[system][qid] = row
    expected = {}
    for system in SYSTEMS:
        groups = defaultdict(list)
        for row in by_system[system].values():
            groups["all"].append(row)
            groups[row["input"]["difficulty"]].append(row)
        for difficulty, group in groups.items():
            correct = sum(row["correct"] for row in group)
            expected[system, difficulty] = dict(
                system=system, difficulty=difficulty, n=len(group), correct=correct,
                accuracy=correct / len(group), invalid=sum(row["selected"] is None for row in group),
            )
    metrics = json.loads((run / f"qwen-{size}.metrics.json").read_text())
    actual = {(row["system"], row["difficulty"]): row for row in metrics}
    if len(metrics) != len(expected) or actual != expected:
        raise ValueError(f"{size}: stored accuracy summary does not match the predictions")
    return paired_results(by_system)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.frozen / "rxqa_prompts.jsonl")
    prompts = {row_key(row): row for row in rows}
    answers = json.loads((args.frozen / "rxqa_answer_key.json").read_text())["answers"]
    gold = {row["qid"]: row["gold"] for row in answers}
    expected = {(qid, system) for qid in gold for system in SYSTEMS}
    if len(answers) != 300 or len(gold) != 300 or len(rows) != 900 or prompts.keys() != expected:
        raise ValueError("Expected 300 unique questions in each of three conditions")
    if any({"gold", "correct", "correct_idx", "answer"}.intersection(row) for row in rows):
        raise ValueError("Frozen prompts contain answer fields")
    results = {size: verify_model(args.run, size, prompts, gold) for size in ("2b", "4b")}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
