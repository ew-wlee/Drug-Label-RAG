"""Validate the frozen Step 1 corpus, index, samples, and model inputs."""
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment import read_jsonl
from select_subset import atomic_json, sha256


def main():
    config = json.loads((ROOT / "configs" / "step1_main.json").read_text())
    run = ROOT / config["paths"]["run_dir"]
    checks = []

    def check(name, condition, evidence):
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    source = json.loads((run / "source_manifest.json").read_text())
    check("openfda_source", source["status"] in {"passed", "passed_with_same_day_override"}
          and len(source["archives"]) == 14
          and all(row["zip_test"] for row in source["archives"]),
          {"status": source["status"], "archives": len(source["archives"]),
           "same_day_changed_archives": len(source["changed_archives"])})

    selection = read_jsonl(run / "rxqa_selection_input.jsonl")
    answers = json.loads((run / "rxqa_answer_key.json").read_text())["answers"]
    forbidden = {"gold", "correct", "correct_idx", "answer"}
    check("rxqa_source_and_answer_separation", len(selection) == 300 == len(answers)
          and len({row["qid"] for row in selection}) == 300
          and all(not forbidden.intersection(row) for row in selection),
          {"questions": len(selection), "answer_key_entries": len(answers)})

    mapping = json.loads((run / "rxqa_label_map.json").read_text())
    review = json.loads((run / "rxqa_blind_review.json").read_text())
    approved = set(review["approved_qids"])
    pending = [row["qid"] for row in mapping["questions"]
               if row["review_status"] in {"pending", "sample_pending", "rejected"}
               and row["qid"] not in approved]
    check("coverage_mapping_and_blind_review", len(mapping["questions"]) == 300
          and mapping["answer_fields_used"] is False and review["answer_key_accessed"] is False
          and not pending,
          {"coverage_counts": mapping["coverage_counts"], "unreviewed": pending})

    subset = json.loads((run / "subset_manifest.json").read_text())
    subset_ids = subset["label_ids"]
    subset_hash = hashlib.sha256("\n".join(subset_ids).encode()).hexdigest()
    reasons = Counter(subset["reason_counts"])
    check("frozen_subset", len(subset_ids) == len(set(subset_ids)) == 10000
          and subset_hash == subset["label_ids_sha256"]
          and subset["selection_uses_answers"] is False
          and reasons["hard_negative"] > 0 and reasons["background"] > 0,
          {"labels": len(subset_ids), "reasons": dict(reasons)})

    index_dir = run / "index"
    index = json.loads((index_dir / "index.json").read_text())
    db = sqlite3.connect(index_dir / "labels.sqlite")
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    counts = {
        "labels": db.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
        "fixed": db.execute("SELECT COUNT(*) FROM fixed").fetchone()[0],
        "section": db.execute("SELECT COUNT(*) FROM section").fetchone()[0],
        "sparse": db.execute("SELECT COUNT(*) FROM sparse").fetchone()[0],
    }
    vector_sizes = {mode: (index_dir / f"{mode}.f32").stat().st_size for mode in ("fixed", "section")}
    vector_ok = all(vector_sizes[mode] == index["counts"][mode] * index["dimension"] * 4
                    for mode in ("fixed", "section"))
    checkpoints_ok = all(json.loads((index_dir / f"checkpoint-{mode}.json").read_text())["complete"]
                         for mode in ("fixed", "section"))
    fts_hits = db.execute("SELECT COUNT(*) FROM sparse WHERE sparse MATCH 'warfarin'").fetchone()[0]
    check("index_integrity", integrity == "ok" and counts == {
        "labels": 10000, "fixed": index["counts"]["fixed"],
        "section": index["counts"]["section"], "sparse": index["counts"]["section"],
    } and vector_ok and checkpoints_ok and fts_hits > 0,
          {"counts": counts, "vector_sizes": vector_sizes, "fts_warfarin_hits": fts_hits})

    prompts = read_jsonl(run / "rxqa_prompts.jsonl")
    pairs = defaultdict(dict)
    input_hash_ok = True
    forbidden_rows = 0
    for row in prompts:
        pairs[row["qid"]][row["system"]] = row
        forbidden_rows += bool(forbidden.intersection(row))
        copied = {key: value for key, value in row.items() if key != "input_hash"}
        expected_hash = hashlib.sha256(json.dumps(copied, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        input_hash_ok &= expected_hash == row["input_hash"]
    paired = len(pairs) == 300 and all(set(group) == {"no_rag", "general", "domain"}
                                      for group in pairs.values())
    evidence_ok = all(len(row["evidence"]) == (0 if row["system"] == "no_rag" else 5)
                      for row in prompts)
    evidence_resolved = True
    for row in prompts:
        for item in row["evidence"]:
            mode, chunk_id = item["chunk_id"].split(":")
            found = db.execute(f"SELECT data FROM {mode} WHERE id = ?", (int(chunk_id),)).fetchone()
            evidence_resolved &= found is not None
    check("paired_gold_free_prompts", len(prompts) == 900 and paired and input_hash_ok
          and forbidden_rows == 0 and evidence_ok and evidence_resolved,
          {"rows": len(prompts), "questions": len(pairs), "forbidden_rows": forbidden_rows,
           "input_hash_ok": input_hash_ok, "evidence_ok": evidence_ok,
           "evidence_resolved": evidence_resolved})

    extension = json.loads((run / "rxqa_extension_60.json").read_text())
    extension_counts = Counter(extension["difficulty_counts"])
    check("extension_60", len(extension["qids"]) == len(set(extension["qids"])) == 60
          and set(extension_counts.values()) == {15}
          and set(extension["qids"]) <= set(pairs), dict(extension_counts))

    fda = json.loads((run / "fdarxbench_frozen.json").read_text())
    check("fdarxbench_frozen", fda["labels"] == 699 and fda["questions"] == 17207
          and len(fda["retrieval_qids"]) == 90,
          {"labels": fda["labels"], "questions": fda["questions"],
           "retrieval_qids": len(fda["retrieval_qids"])})

    resources = json.loads((run / "resource_usage.json").read_text())
    actual_index_bytes = sum(path.stat().st_size for path in index_dir.iterdir() if path.is_file())
    total_chunks = sum(index["counts"].values())
    check("resource_caps", actual_index_bytes <= config["subset"]["max_index_bytes"]
          and total_chunks <= config["subset"]["max_chunks"]
          and resources["prepare"]["peak_rss_bytes"] <= config["runtime"]["peak_rss_target_bytes"],
          {"index_bytes": actual_index_bytes, "chunks": total_chunks,
           "prepare_peak_rss_bytes": resources["prepare"]["peak_rss_bytes"]})
    db.close()

    report = {
        "run_id": config["run_id"], "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "blocking_items": [row["name"] for row in checks if not row["passed"]], "checks": checks,
    }
    atomic_json(run / "validation_report.json", report)
    artifacts = [
        "configs/step1_main.json", "openfda_rxqa.py", "select_subset.py", "scripts/validate_step1.py",
        "tests/test_step1.py", "README.md",
    ]
    artifacts += [str(path.relative_to(ROOT)) for path in sorted(run.iterdir()) if path.is_file()
                  and path.name != "step1_manifest.json"]
    artifacts += [str(path.relative_to(ROOT)) for path in sorted(index_dir.iterdir()) if path.is_file()]
    model_revisions = {}
    for key, pattern in (("embedder", "models--sentence-transformers--all-MiniLM-L6-v2"),
                         ("reranker", "models--cross-encoder--ms-marco-MiniLM-L6-v2")):
        snapshots = list((ROOT / ".cache" / "huggingface" / "hub" / pattern / "snapshots").glob("*"))
        model_revisions[key] = snapshots[0].name if snapshots else None
    manifest = {
        "run_id": config["run_id"], "status": report["status"], "seed": config["seed"],
        "model_revisions": model_revisions,
        "ollama_runtime_check": "deferred_to_step2",
        "artifacts": {name: sha256(ROOT / name) for name in sorted(set(artifacts))},
    }
    atomic_json(run / "step1_manifest.json", manifest)
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "blocking_items": report["blocking_items"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
