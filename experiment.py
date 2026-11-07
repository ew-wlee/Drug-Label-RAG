"""Reproduce the FDARxBench corpus retrieval and build paired QA prompts."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from math import log
from pathlib import Path
import random
import re
import statistics


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    set_id: str
    drug: str
    text: str
    spans: tuple

    @property
    def display(self):
        return f"{self.drug}\n{self.text}"


def build_chunks(labels, section_aware, size=160, overlap=24):
    chunks = []
    for label in labels:
        tokens = []
        source = []
        for pid, passage in enumerate(label["chunks"]):
            # Keep original words for generation and word offsets for evidence mapping.
            parts = passage.split()
            tokens.extend(parts)
            source.extend((pid, pos) for pos in range(len(parts)))
            if section_aware:
                add_windows(chunks, label, tokens, source, size, overlap)
                tokens, source = [], []
        add_windows(chunks, label, tokens, source, size, overlap)
    return chunks


def add_windows(chunks, label, tokens, source, size, overlap):
    for start in range(0, len(tokens), size - overlap):
        end = min(len(tokens), start + size)
        spans = []
        for pid, offset in source[start:end]:
            if spans and spans[-1][0] == pid:
                spans[-1] = (pid, spans[-1][1], offset + 1)
            else:
                spans.append((pid, offset, offset + 1))
        chunks.append(Chunk(label["set_id"], label["drug_name"],
                            " ".join(tokens[start:end]), tuple(spans)))
        if end == len(tokens):
            break


class BM25:
    def __init__(self, chunks):
        self.postings = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(chunks):
            tokens = words(chunk.display)
            lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term].append((i, freq))
        average = statistics.mean(lengths)
        self.norm = [1.2 * (0.25 + 0.75 * length / average) for length in lengths]
        self.size = len(lengths)

    def rank(self, query, k):
        scores = defaultdict(float)
        for term in sorted(set(words(query))):
            postings = self.postings.get(term, ())
            idf = log(1 + (self.size - len(postings) + .5) / (len(postings) + .5))
            for i, freq in postings:
                scores[i] += idf * freq * 2.2 / (freq + self.norm[i])
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]


def fuse(*rankings):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] += 1 / (60 + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))


def sample_questions(rows, per_task, seed):
    rng = random.Random(seed)
    result = []
    for task in ("factual", "multihop", "refusal"):
        group = sorted((row for row in rows if row["task"] == task), key=lambda r: r["qid"])
        # Keep pilot questions separate if the team later trains or tunes models.
        result.extend(rng.sample(group, min(per_task, len(group))))
    return result


def gold_spans(row, labels):
    label = labels[row["set_id"]]
    gold = {}
    for evidence in row["context"]:
        pid = evidence["doc_chunk_index"]
        passage = " ".join(label["chunks"][pid].split())
        target = " ".join(evidence["text"].split())
        offset = passage.find(target)
        if offset >= 0 and target:
            start = len(passage[:offset].split())
            gold.setdefault(pid, set()).update(range(start, start + len(target.split())))
        else:
            # Preserve the official passage index when text alignment is unavailable.
            gold[pid] = set(range(len(passage.split())))
    return gold


def retrieval_metrics(row, ranked, gold):
    if not gold:
        return {}
    found = defaultdict(set)
    result = {}
    rr = 0.0
    for k in range(1, 11):
        if k <= len(ranked):
            chunk = ranked[k - 1]
            if chunk.set_id == row["set_id"]:
                for pid, start, end in chunk.spans:
                    found[pid].update(range(start, end))
        passage_hits = sum(bool(found[pid] & offsets) for pid, offsets in gold.items())
        evidence_hits = sum(len(found[pid] & offsets) >= .8 * len(offsets)
                            for pid, offsets in gold.items())
        if passage_hits and rr == 0:
            rr = 1 / k
        if k in (1, 5, 10):
            result[f"recall@{k}"] = passage_hits / len(gold)
            result[f"evidence80@{k}"] = evidence_hits / len(gold)
            result[f"all_evidence80@{k}"] = float(evidence_hits == len(gold))
    result["mrr@10"] = rr
    return result


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def retrieve(labels, questions, gold, embedding):
    embedder = None
    if embedding:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(embedding)
        queries = embedder.encode([q["question"] for q in questions], normalize_embeddings=True)
    results = [dict(q, system="no_rag", retrieved=[], metrics={}) for q in questions]
    chunk_counts = {}
    for structure in (False, True):
        chunks = build_chunks(labels.values(), structure)
        chunk_counts["section" if structure else "fixed"] = len(chunks)
        lexical = BM25(chunks)
        if embedder:
            matrix = embedder.encode([c.display for c in chunks], batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=True)
            similarities = queries @ matrix.T
        for qi, q in enumerate(questions):
            sparse = lexical.rank(q["question"], 20)
            if embedder:
                dense = np.argsort(-similarities[qi], kind="stable")[:20].tolist()
                rankings = {"section_dense" if structure else "general": dense}
                if structure:
                    rankings["domain"] = fuse(dense, sparse)[:20]
            else:
                rankings = {"section_bm25" if structure else "fixed_bm25": sparse}
            for name, indices in rankings.items():
                retrieved = [chunks[i] for i in indices[:10]]
                results.append(dict(q, system=name,
                                    retrieved=[vars(c) for c in retrieved],
                                    metrics=retrieval_metrics(q, retrieved, gold[q["qid"]])))
    return results, chunk_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5018)
    parser.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker", default="")
    parser.add_argument("--reranker-device")
    parser.add_argument("--reranker-fp16", action="store_true")
    parser.add_argument("--from-retrieval", type=Path, help="Reuse a previous dense run.")
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()
    if args.per_task < 1 or (args.lexical_only and args.reranker):
        parser.error("Choose a positive sample size and use reranking with dense retrieval.")
    if args.from_retrieval and (not args.reranker or args.lexical_only):
        parser.error("--from-retrieval requires --reranker.")
    args.out.mkdir(parents=True, exist_ok=True)
    label_path = args.data / "labels/labels.jsonl"
    qa_path = args.data / "qa/qa.jsonl"
    raw = read_jsonl(label_path)
    labels = {row["set_id"]: row for row in raw if row["set_id"] and row["chunks"]}
    embedding = None if args.lexical_only else args.embedding
    if args.from_retrieval:
        previous = json.loads((args.from_retrieval / "retrieval_metrics.json").read_text())
        if (previous["label_sha256"], previous["qa_sha256"], previous["seed"]) != (
                sha256(label_path), sha256(qa_path), args.seed):
            parser.error("The reused run must use the same data and seed.")
        if args.per_task > previous["per_task"]:
            parser.error("The reused run must contain at least the requested number of questions.")
        chunk_counts = previous["chunk_counts"]
        embedding = previous["embedding"]
        results = [r for r in read_jsonl(args.from_retrieval / "retrieval.jsonl")
                   if not r["system"].endswith("_rerank")]
        if {r["system"] for r in results} != {"no_rag", "general", "section_dense", "domain"}:
            parser.error("The reused run must contain all dense retrieval conditions.")
        tasks = Counter()
        questions = []
        for row in results:
            if row["system"] == "no_rag" and tasks[row["task"]] < args.per_task:
                questions.append(row)
                tasks[row["task"]] += 1
        gold = {row["qid"]: gold_spans(row, labels) for row in questions}
        results = [row for row in results if row["qid"] in gold]
    else:
        questions = sample_questions(read_jsonl(qa_path), args.per_task, args.seed)
        gold = {row["qid"]: gold_spans(row, labels) for row in questions}
        results, chunk_counts = retrieve(labels, questions, gold, embedding)
    if args.reranker:
        from sentence_transformers import CrossEncoder
        model_kwargs = {"torch_dtype": "float16"} if args.reranker_fp16 else None
        reranker = CrossEncoder(args.reranker, device=args.reranker_device,
                                model_kwargs=model_kwargs)
        candidates = [row for row in results if row["system"] in ("general", "domain")]
        for i, row in enumerate(candidates, 1):
            retrieved = [Chunk(**chunk) for chunk in row["retrieved"]]
            scores = reranker.predict([(row["question"], chunk.display) for chunk in retrieved])
            order = sorted(range(len(retrieved)), key=lambda i: (-float(scores[i]), i))
            ranked = [retrieved[i] for i in order]
            results.append(dict(row, system=row["system"] + "_rerank",
                                retrieved=[vars(c) for c in ranked],
                                metrics=retrieval_metrics(row, ranked, gold[row["qid"]])))
            if i % 10 == 0 or i == len(candidates):
                print(f"Reranked {i}/{len(candidates)} question-system pairs", flush=True)
    write_jsonl(args.out / "retrieval.jsonl", results)
    summaries = {}
    for row in results:
        if not row["metrics"]:
            continue
        for scope in ("answerable", row["task"]):
            summaries.setdefault((row["system"], scope), []).append(row["metrics"])
    report = [{"system": name, "scope": scope, "n": len(rows),
               **{key: round(statistics.mean(r[key] for r in rows), 4) for key in rows[0]}}
              for (name, scope), rows in sorted(summaries.items())]
    manifest = {"seed": args.seed, "per_task": args.per_task, "labels": len(labels),
                "chunk_counts": chunk_counts, "embedding": embedding,
                "reranker": args.reranker, "reranker_device": args.reranker_device,
                "reranker_fp16": args.reranker_fp16, "label_sha256": sha256(label_path),
                "qa_sha256": sha256(qa_path), "retrieval": report}
    (args.out / "retrieval_metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
