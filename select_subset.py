"""Build and freeze an answer-isolated, coverage-oriented openFDA subset."""
import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import ijson

from experiment import read_jsonl, words
from openfda_rxqa import SECTIONS, json_member, normalize, product_names, records

ROOT = Path(__file__).resolve().parent
DRUG_KINDS = {"brand", "generic", "substance", "spl_brand"}
ALIAS_BLOCKLIST = {
    "allergy", "antacid", "antibiotic", "birth control", "control", "cough", "cream",
    "constipation", "creatinine", "diarrhea", "ear", "eye", "gas", "gel", "headache", "headache relief",
    "health", "heart", "injury", "itch", "liquid", "nasal", "night", "nitrogen", "pain",
    "pain relief", "relief", "saline", "saline nasal", "scalp", "sinus", "sleep", "sodium",
    "sun", "treat", "vitamin", "water",
}
SPL_BRAND_ALLOWLIST = {
    "amitriptyline", "adcirca", "agamree", "allergena", "amphotericin", "ampyra", "apokyn", "arcalyst",
    "ascorbic", "aviane", "avonex", "ayvakit", "bosentan", "bronchitol", "calcitriol",
    "caspofungin", "chlorthalidone", "cipro", "clotrimazole", "codeine", "colesevelam",
    "cordarone", "cyproheptadine", "dorzolamide", "doxazosin", "emflaza", "enbrel",
    "enpresse", "epinephrine", "evrysdi", "extavia", "fioricet", "fluconazole",
    "genotropin", "golytely", "hectorol", "hespan", "imiquimod", "jardiance", "joenja",
    "keveyis", "krazati", "lanoxin", "latisse", "linzess", "lisinopril", "lopid",
    "macrilen", "marplan", "memantine", "methergine", "methimazole", "metoprolol",
    "mirena", "nifedipine", "nityr", "olpruva", "opium", "oxervate", "pataday",
    "phenobarbital", "prazosin", "premarin", "promethazine", "purinethol", "questran",
    "relyvrio", "rhofade", "rubraca", "somatuline", "somavert", "talzenna", "targretin",
    "tecfidera", "temodar", "thalitone", "tiazac", "tibsovo", "timolol", "topiramate",
    "travoprost", "trileptal", "uptravi", "vancomycin", "vasotec", "verkazia", "vesanoid",
    "viokace", "votrient", "wegovy", "xenazine", "yonsa", "zerviate", "zilbrysq",
    "zykadia", "zyloprim",
}
SPL_GENERIC_ALLOWLIST = {"amitriptyline"}
CLASS_FIELDS = (
    "pharm_class_cs", "pharm_class_epc", "pharm_class_pe", "pharm_class_moa",
    "pharm_class_cs", "pharm_class_physiologic",
)
STOP_TERMS = set(words("""
about above after again against all also and any are because been before being below best
between both can cannot could did does doing down during each few for from further had has
have having her here hers him his how into its itself just more most other our out over own
same she should some such than that the their theirs them then there these they this those
through too under until very was were what when where which while who whom why with would
you your patient patients medication medications medicine medicines drug drugs treatment
treated therapy dose doses dosing recommended recommendation appropriate important common
following initial daily orally tablet tablets capsule capsules solution injection doctor
pharmacist prescribe prescribed prescription administer administration taking take used use
using clinical history reports report presents presenting diagnosed diagnosis severe mild
moderate chronic currently recently started starting woman women man male female years old
mr mrs ms miss dr most least maximum minimum adult adults pediatric child children pregnancy
pregnant renal hepatic impairment disease disorder syndrome conditions condition effects
effect risk risks adverse reaction reactions side effects including include includes related
associated recommend recommendatio course action situation management manage given giving
receive receiving receives received new now still since every another first second third
"""))


def config_path(value):
    return ROOT / "configs" / value


def load_config(path):
    return json.loads(Path(path).read_text())


def resolve(config, key):
    return ROOT / config["paths"][key]


def run_dir(config):
    return resolve(config, "run_dir")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_hash(seed, *parts):
    return hashlib.sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_name(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)] if str(value).strip() else []


def openfda_meta(path):
    with ZipFile(path) as archive, archive.open(json_member(archive)) as stream:
        return next(ijson.items(stream, "meta"))


def prepare_sources(config):
    destination = run_dir(config)
    destination.mkdir(parents=True, exist_ok=True)
    expected = json.loads(resolve(config, "original_openfda_manifest").read_text())
    expected_by_name = {row["file"]: row for row in expected}
    paths = sorted(resolve(config, "openfda_dir").glob("drug-label-*.json.zip"))
    if len(paths) != config["source"]["openfda_partitions"]:
        raise ValueError(f"Expected 14 openFDA archives, found {len(paths)}")
    actual = []
    for path in paths:
        if path.name not in expected_by_name:
            raise ValueError(f"Unexpected archive: {path.name}")
        with ZipFile(path) as archive:
            bad_member = archive.testzip()
        meta = openfda_meta(path)
        old = expected_by_name[path.name]
        row = {
            "file": path.name,
            "zip_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "zip_test": bad_member is None,
            "meta": meta,
            "previous_zip_bytes": old["zip_bytes"],
            "previous_sha256": old["sha256"],
        }
        row["matches_previous_manifest"] = (
            row["zip_bytes"] == old["zip_bytes"] and row["sha256"] == old["sha256"]
        )
        if not row["zip_test"]:
            raise ValueError(f"Corrupt ZIP member in {path.name}: {bad_member}")
        actual.append(row)
    dates = {row["meta"]["last_updated"] for row in actual}
    totals = {row["meta"]["results"]["total"] for row in actual}
    skips = [row["meta"]["results"]["skip"] for row in actual]
    limits = [row["meta"]["results"]["limit"] for row in actual]
    expected_skips = list(range(0, config["source"]["openfda_total"], 20000))
    if dates != {config["source"]["openfda_last_updated"]}:
        raise ValueError(f"Mixed or unexpected snapshot dates: {dates}")
    if totals != {config["source"]["openfda_total"]} or skips != expected_skips:
        raise ValueError("Inconsistent openFDA totals or partition offsets")
    if sum(limits) != config["source"]["openfda_total"]:
        raise ValueError("Partition limits do not sum to the snapshot total")
    changed = [row["file"] for row in actual if not row["matches_previous_manifest"]]
    if changed and not config["source"]["allow_same_day_official_refresh"]:
        raise ValueError(f"Archives differ from the previous manifest: {changed}")
    source_manifest = {
        "run_id": config["run_id"],
        "status": "passed_with_same_day_override" if changed else "passed",
        "source_override_reason": config["source"]["source_override_reason"] if changed else None,
        "changed_archives": changed,
        "previous_manifest_sha256": sha256(resolve(config, "original_openfda_manifest")),
        "archives": actual,
    }
    atomic_json(destination / "source_manifest.json", source_manifest)

    csv_path = resolve(config, "rxqa_csv")
    if sha256(csv_path) != config["source"]["rxqa_sha256"]:
        raise ValueError("RxQA CSV hash does not match the pinned revision")
    questions = []
    answer_key = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            qid = row["q_id"]
            options = {letter: row[letter] for letter in "ABCD"}
            index = int(row["correct_idx"])
            if not 0 <= index < 4 or options["ABCD"[index]].strip() != row["correct"].strip():
                raise ValueError(f"Inconsistent RxQA answer key at {qid}")
            if row["pharmacist_rated_difficulty"] not in config["evaluation"]["difficulties"]:
                raise ValueError(f"Unexpected difficulty at {qid}")
            questions.append({
                "qid": qid,
                "question": row["question"],
                "options": options,
                "difficulty": row["pharmacist_rated_difficulty"],
            })
            answer_key.append({"qid": qid, "gold": "ABCD"[index]})
    if len(questions) != 300 or len({row["qid"] for row in questions}) != 300:
        raise ValueError("RxQA must contain exactly 300 unique question IDs")
    atomic_jsonl(destination / "rxqa_selection_input.jsonl", questions)
    atomic_json(destination / "rxqa_answer_key.json", {
        "source_sha256": sha256(csv_path),
        "revision": config["source"]["rxqa_revision"],
        "answers": answer_key,
    })

    fda_data = resolve(config, "fdarxbench_data")
    label_path = fda_data / "labels" / "labels.jsonl"
    qa_path = fda_data / "qa" / "qa.jsonl"
    labels = read_jsonl(label_path)
    valid_labels = [row for row in labels if row.get("set_id") and row.get("chunks")]
    questions_fda = read_jsonl(qa_path)
    set_ids = sorted({row["set_id"] for row in valid_labels})
    if len(valid_labels) != 699 or len(questions_fda) != 17207:
        raise ValueError("Unexpected pinned FDARxBench dimensions")
    atomic_json(destination / "fdarxbench_set_ids.json", {"set_ids": set_ids})
    retrieval_rows = list(csv.DictReader((ROOT / "results" / "retrieval_per_question.csv").open()))
    retrieval_qids = sorted({row["qid"] for row in retrieval_rows})
    tasks = defaultdict(set)
    for row in retrieval_rows:
        tasks[row["task"]].add(row["qid"])
    generation = {}
    for path in sorted((ROOT / "results" / "answers").glob("qwen*_predictions.jsonl")):
        rows = read_jsonl(path)
        generation[path.name] = {
            "sha256": sha256(path), "rows": len(rows),
            "qids": sorted({row["qid"] for row in rows}),
            "systems": sorted({row["system"] for row in rows}),
        }
    atomic_json(destination / "fdarxbench_frozen.json", {
        "revision": config["source"]["fdarxbench_revision"],
        "labels_sha256": sha256(label_path), "qa_sha256": sha256(qa_path),
        "labels": len(valid_labels), "questions": len(questions_fda),
        "official_version_fields_available": False,
        "retrieval_qids": retrieval_qids,
        "retrieval_task_counts": {key: len(value) for key, value in sorted(tasks.items())},
        "retrieval_per_question_sha256": sha256(ROOT / "results" / "retrieval_per_question.csv"),
        "generation": generation,
    })
    print(json.dumps({
        "openfda_archives": len(actual), "openfda_records": sum(limits),
        "same_day_changed_archives": len(changed), "rxqa_questions": len(questions),
        "rxqa_difficulty": dict(Counter(row["difficulty"] for row in questions)),
        "fdarxbench_labels": len(valid_labels), "fdarxbench_retrieval_qids": len(retrieval_qids),
    }, indent=2))


def catalog_schema(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS labels (
        label_id TEXT PRIMARY KEY, set_id TEXT NOT NULL, version TEXT, effective_time TEXT,
        brand_names TEXT, generic_names TEXT, substances TEXT, classes TEXT,
        routes TEXT, dosage_forms TEXT, sections TEXT, text_chars INTEGER,
        text_sha256 TEXT, source_archive TEXT
    );
    CREATE INDEX IF NOT EXISTS labels_set_id ON labels(set_id);
    CREATE TABLE IF NOT EXISTS aliases (
        alias TEXT, label_id TEXT, kind TEXT,
        UNIQUE(alias, label_id, kind)
    );
    CREATE INDEX IF NOT EXISTS aliases_alias ON aliases(alias);
    CREATE INDEX IF NOT EXISTS aliases_label ON aliases(label_id);
    """)


def metadata(row, archive):
    fields = row.get("openfda") or {}
    label = normalize(row)
    if label is None:
        return None
    brand_values, generic_values = product_names(row)
    brands = string_list(brand_values)
    generics = string_list(generic_values)
    explicit_brands = string_list(fields.get("brand_name"))
    explicit_generics = string_list(fields.get("generic_name"))
    spl_brands = []
    if not explicit_brands and not explicit_generics:
        product_text = next(iter(row.get("spl_product_data_elements", [])), "")
        parts = product_text.split()
        if parts and re.search(r"[A-Za-z]", parts[0]):
            spl_brands = [parts[0]]
    substances = string_list(fields.get("substance_name"))
    classes = []
    for key in CLASS_FIELDS:
        classes.extend(string_list(fields.get(key)))
    routes = string_list(fields.get("route"))
    forms = string_list(fields.get("dosage_form"))
    text = "\n".join(label["chunks"])
    return {
        "label_id": label["label_id"], "set_id": label["set_id"], "version": label["version"],
        "effective_time": label["effective_time"], "brand_names": brands,
        "generic_names": generics, "substances": substances,
        "classes": list(dict.fromkeys(classes)), "routes": routes, "dosage_forms": forms,
        "sections": label["sections"], "text_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "source_archive": archive,
        "alias_groups": {"brand": explicit_brands, "generic": explicit_generics,
                         "substance": substances, "class": list(dict.fromkeys(classes)),
                         "spl_brand": spl_brands},
    }


def insert_catalog_rows(db, rows):
    inserted = 0
    for row in rows:
        cursor = db.execute("""INSERT OR IGNORE INTO labels VALUES (
            :label_id, :set_id, :version, :effective_time, :brand_names, :generic_names,
            :substances, :classes, :routes, :dosage_forms, :sections, :text_chars,
            :text_sha256, :source_archive)""", {
            **row,
            **{key: json.dumps(row[key], ensure_ascii=False) for key in (
                "brand_names", "generic_names", "substances", "classes", "routes",
                "dosage_forms", "sections")},
        })
        if not cursor.rowcount:
            continue
        inserted += 1
        for kind, values in row["alias_groups"].items():
            for value in values:
                alias = normalize_name(value)
                if len(alias) >= 3:
                    db.execute("INSERT OR IGNORE INTO aliases VALUES (?, ?, ?)",
                               (alias, row["label_id"], kind))
    return inserted


def build_catalog(config):
    destination = run_dir(config)
    db_path = destination / "label_catalog.sqlite"
    progress_path = destination / "catalog_progress.json"
    expected_config_hash = stable_hash(config["seed"], json.dumps(config["source"], sort_keys=True))
    completed = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["config_hash"] != expected_config_hash:
            raise ValueError("Catalog configuration changed; use a new run directory")
        completed = progress["completed_archives"]
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    catalog_schema(db)
    started = time.time()
    for path in sorted(resolve(config, "openfda_dir").glob("drug-label-*.json.zip")):
        if path.name in completed:
            continue
        batch = []
        read = usable = 0
        archive_started = time.time()
        with db:
            for raw in records([path]):
                read += 1
                row = metadata(raw, path.name)
                if row is None:
                    continue
                usable += 1
                batch.append(row)
                if len(batch) >= config["runtime"]["sqlite_batch_size"]:
                    insert_catalog_rows(db, batch)
                    batch.clear()
            insert_catalog_rows(db, batch)
        completed.append(path.name)
        atomic_json(progress_path, {
            "config_hash": expected_config_hash, "completed_archives": completed,
            "last_archive": {"read": read, "usable": usable,
                             "seconds": round(time.time() - archive_started, 3)},
        })
        print(f"catalog {path.name}: read={read} usable={usable}", flush=True)
    counts = {
        "labels": db.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
        "unique_set_ids": db.execute("SELECT COUNT(DISTINCT set_id) FROM labels").fetchone()[0],
        "aliases": db.execute("SELECT COUNT(*) FROM aliases").fetchone()[0],
    }
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    db.close()
    if integrity != "ok":
        raise ValueError(f"Catalog integrity failure: {integrity}")
    atomic_json(destination / "catalog_manifest.json", {
        **counts, "integrity": integrity, "bytes": db_path.stat().st_size,
        "sha256": sha256(db_path), "elapsed_seconds": round(time.time() - started, 3),
        "full_text_vectors": False, "full_text_index": False,
    })
    print(json.dumps(counts, indent=2))


def ngrams(text, maximum=12):
    tokens = normalize_name(text).split()
    return {" ".join(tokens[start:end])
            for start in range(len(tokens))
            for end in range(start + 1, min(len(tokens), start + maximum) + 1)}


def distinctive_spl_mention(text, alias):
    if alias in SPL_GENERIC_ALLOWLIST:
        return True
    for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])",
                             text, flags=re.IGNORECASE):
        raw = match.group()
        if len(raw) >= 4 and raw.isupper():
            return True
        prefix = text[:match.start()].rstrip()
        if len(raw) >= 4 and raw[0].isupper() and prefix and prefix[-1] not in ".?!":
            return True
    return False


def lookup_aliases(db, text):
    candidates = sorted(ngrams(text))
    found = []
    for start in range(0, len(candidates), 500):
        group = candidates[start:start + 500]
        if not group:
            continue
        found.extend(db.execute(
            f"SELECT alias, label_id, kind FROM aliases WHERE alias IN ({','.join('?' * len(group))})",
            group).fetchall())
    grouped = {}
    for alias, label_id, kind in found:
        if (alias in ALIAS_BLOCKLIST or re.fullmatch(r"a \d+", alias)
                or (" " not in alias and alias in STOP_TERMS)
                or (kind == "spl_brand" and (alias not in SPL_BRAND_ALLOWLIST
                    or not distinctive_spl_mention(text, alias)))):
            continue
        entry = grouped.setdefault((alias, kind), {"alias": alias, "kind": kind, "label_ids": set()})
        entry["label_ids"].add(label_id)
    result = []
    for entry in grouped.values():
        entry["label_ids"] = sorted(entry["label_ids"])
        result.append(entry)
    return sorted(result, key=lambda row: (-len(row["alias"].split()), row["alias"], row["kind"]))


def candidate_terms(question):
    terms = set()
    for token in words(question["question"] + " " + " ".join(question["options"].values())):
        if len(token) >= 5 and token not in STOP_TERMS and not token.isdigit():
            terms.add(token)
    return sorted(terms)


def discover_candidates(config, db, pending):
    if not pending:
        return {}
    by_qid = {row["qid"]: row for row in read_jsonl(run_dir(config) / "rxqa_selection_input.jsonl")}
    term_questions = defaultdict(set)
    for qid in pending:
        for term in candidate_terms(by_qid[qid]):
            term_questions[term].add(qid)
    limit = config["subset"]["candidates_per_question"]
    heaps = {qid: [] for qid in pending}
    for path in sorted(resolve(config, "openfda_dir").glob("drug-label-*.json.zip")):
        for raw in records([path]):
            label = normalize(raw)
            if label is None:
                continue
            local = defaultdict(int)
            for token in set(words(" ".join(label["chunks"]))):
                for qid in term_questions.get(token, ()):
                    local[qid] += min(len(token), 12)
            for qid, score in local.items():
                if score < 8:
                    continue
                item = (score, label["label_id"], label["set_id"])
                if len(heaps[qid]) < limit:
                    heapq.heappush(heaps[qid], item)
                elif item > heaps[qid][0]:
                    heapq.heapreplace(heaps[qid], item)
        print(f"candidate discovery {path.name}", flush=True)
    return {qid: [
        {"score": score, "label_id": label_id, "set_id": set_id}
        for score, label_id, set_id in sorted(values, reverse=True)
    ] for qid, values in heaps.items()}


def map_questions(config):
    destination = run_dir(config)
    questions = read_jsonl(destination / "rxqa_selection_input.jsonl")
    db = sqlite3.connect(destination / "label_catalog.sqlite")
    output = []
    for question in questions:
        stem = lookup_aliases(db, question["question"])
        option_text = " ".join(question["options"][letter] for letter in "ABCD")
        options = lookup_aliases(db, option_text)
        stem_drugs = [row for row in stem if row["kind"] in DRUG_KINDS]
        stem_classes = [row for row in stem if row["kind"] == "class"]
        option_drugs = [row for row in options if row["kind"] in DRUG_KINDS]
        option_classes = [row for row in options if row["kind"] == "class"]
        if stem_drugs:
            status = "stem_exact"
        elif option_drugs:
            status = "option_only"
        elif stem_classes or option_classes:
            status = "class_mapped"
        else:
            status = "unresolved"
        output.append({
            **question, "coverage_status": status, "stem_matches": stem,
            "option_matches": options, "candidate_matches": [],
            "review_status": "pending" if status != "stem_exact" else "auto_exact",
            "review_note": "",
        })
    db.close()
    unresolved = [row["qid"] for row in output if row["coverage_status"] == "unresolved"]
    discovered = discover_candidates(config, None, unresolved)
    for row in output:
        candidates = discovered.get(row["qid"], [])
        if candidates:
            row["candidate_matches"] = candidates
            row["coverage_status"] = "candidate_only"
    exact = sorted((row for row in output if row["coverage_status"] == "stem_exact"),
                   key=lambda row: stable_hash(config["seed"], "exact-review", row["qid"]))
    for row in exact[:config["subset"]["exact_match_review_sample"]]:
        row["review_status"] = "sample_pending"
    atomic_json(destination / "rxqa_label_map.json", {
        "answer_fields_used": False,
        "selection_inputs": ["question", "A", "B", "C", "D", "difficulty"],
        "coverage_counts": dict(Counter(row["coverage_status"] for row in output)),
        "review_counts": dict(Counter(row["review_status"] for row in output)),
        "questions": output,
    })
    print(json.dumps({
        "coverage": dict(Counter(row["coverage_status"] for row in output)),
        "review": dict(Counter(row["review_status"] for row in output)),
    }, indent=2))


def fetch_labels(db, label_ids):
    ids = sorted(set(label_ids))
    rows = []
    for start in range(0, len(ids), 500):
        group = ids[start:start + 500]
        if group:
            rows.extend(db.execute(
                f"SELECT * FROM labels WHERE label_id IN ({','.join('?' * len(group))})", group
            ).fetchall())
    columns = [column[0] for column in db.execute("SELECT * FROM labels LIMIT 0").description]
    return [dict(zip(columns, row)) for row in rows]


def choose_labels(db, label_ids, seed, max_set_ids, max_labels):
    rows = fetch_labels(db, label_ids)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["set_id"]].append(row)
    set_ids = sorted(grouped, key=lambda value: stable_hash(seed, value))[:max_set_ids]
    latest = []
    older = []
    for set_id in set_ids:
        versions = sorted(grouped[set_id], key=lambda row: (row["effective_time"], row["label_id"]), reverse=True)
        latest.append(versions[0])
        older.extend(versions[1:])
    selected = latest + sorted(older, key=lambda row: (row["effective_time"], row["label_id"]), reverse=True)
    return selected[:max_labels]


def selection_reasons(config, db, mapping):
    core = defaultdict(set)
    per_question = {}
    limits = config["subset"]
    for question in mapping["questions"]:
        qid = question["qid"]
        selected_ids = set()
        groups = (
            ("stem_matches", "rxqa_stem", limits["stem_set_ids_per_alias"], limits["stem_labels_per_alias"]),
            ("option_matches", "rxqa_option", limits["option_set_ids_per_alias"], limits["option_labels_per_alias"]),
        )
        for field, reason, max_sets, max_labels in groups:
            for match in question[field]:
                rows = choose_labels(db, match["label_ids"], stable_hash(config["seed"], qid, match["alias"]),
                                     max_sets, max_labels)
                for row in rows:
                    core[row["label_id"]].add(reason)
                    selected_ids.add(row["label_id"])
        if question["coverage_status"] in {"class_mapped", "candidate_only"}:
            matches = question["stem_matches"] + question["option_matches"]
            ids = {label_id for match in matches for label_id in match["label_ids"]}
            ids.update(row["label_id"] for row in question["candidate_matches"])
            rows = choose_labels(db, ids, stable_hash(config["seed"], qid, "fallback"), 8, 32)
            for row in rows:
                core[row["label_id"]].add("rxqa_" + question["coverage_status"])
                selected_ids.add(row["label_id"])
        per_question[qid] = sorted(selected_ids)
    fda_ids = set(json.loads((run_dir(config) / "fdarxbench_set_ids.json").read_text())["set_ids"])
    rows = db.execute("SELECT label_id FROM labels WHERE set_id IN (SELECT value FROM json_each(?))",
                      (json.dumps(sorted(fda_ids)),)).fetchall()
    for (label_id,) in rows:
        core[label_id].add("fdarxbench_set_id")
    return core, per_question


def select_subset(config):
    destination = run_dir(config)
    mapping = json.loads((destination / "rxqa_label_map.json").read_text())
    review = json.loads((destination / "rxqa_blind_review.json").read_text())
    if review.get("answer_key_accessed") is not False:
        raise ValueError("Blind review must explicitly confirm answer-key isolation")
    approved = set(review["approved_qids"])
    pending = [row["qid"] for row in mapping["questions"]
               if row["review_status"] in {"pending", "sample_pending", "rejected"}
               and row["qid"] not in approved]
    if pending:
        raise ValueError(f"Blind review incomplete for {len(pending)} questions")
    db = sqlite3.connect(destination / "label_catalog.sqlite")
    core, per_question = selection_reasons(config, db, mapping)
    if len(core) > config["subset"]["max_labels"]:
        raise ValueError(f"Core has {len(core)} labels, above the hard cap; refine deterministic core rules")

    stem_aliases = sorted({match["alias"] for row in mapping["questions"] for match in row["stem_matches"]
                           if match["kind"] in DRUG_KINDS})
    negative_candidates = set()
    for start in range(0, len(stem_aliases), 400):
        group = stem_aliases[start:start + 400]
        if group:
            negative_candidates.update(label_id for (label_id,) in db.execute(
                f"SELECT DISTINCT label_id FROM aliases WHERE alias IN ({','.join('?' * len(group))})",
                group).fetchall())
    negative_candidates.difference_update(core)
    hard_order = sorted(negative_candidates,
                        key=lambda value: stable_hash(config["seed"], "hard-negative", value))
    hard_ids = set(hard_order[:config["subset"]["hard_negative_labels"]])

    remaining = [row[0] for row in db.execute("SELECT label_id FROM labels").fetchall()
                 if row[0] not in core and row[0] not in hard_ids]
    remaining.sort(key=lambda value: stable_hash(config["seed"], "background", value))
    background_ids = set(remaining[:config["subset"]["background_labels"]])

    selected = dict.fromkeys(core)
    for label_id in core:
        selected[label_id] = set(core[label_id])
    for label_id in hard_ids:
        selected.setdefault(label_id, set()).add("hard_negative")
    for label_id in background_ids:
        selected.setdefault(label_id, set()).add("background")
    overflow = len(selected) - config["subset"]["max_labels"]
    if overflow > 0:
        removable = sorted(background_ids, key=lambda value: stable_hash(config["seed"], "background-drop", value),
                           reverse=True)
        for label_id in removable[:overflow]:
            selected.pop(label_id, None)
            background_ids.discard(label_id)
    if len(selected) > config["subset"]["max_labels"]:
        removable = sorted(hard_ids, key=lambda value: stable_hash(config["seed"], "hard-drop", value), reverse=True)
        for label_id in removable[:len(selected) - config["subset"]["max_labels"]]:
            selected.pop(label_id, None)
            hard_ids.discard(label_id)
    if len(selected) > config["subset"]["max_labels"]:
        raise ValueError("Core labels exceed the hard cap")

    metadata = {row["label_id"]: row for row in fetch_labels(db, selected)}
    rows = []
    for label_id in sorted(selected):
        row = metadata[label_id]
        rows.append({
            "label_id": label_id, "set_id": row["set_id"], "version": row["version"],
            "effective_time": row["effective_time"], "text_chars": row["text_chars"],
            "text_sha256": row["text_sha256"], "reasons": sorted(selected[label_id]),
        })
    db.close()
    atomic_jsonl(destination / "subset_labels.jsonl", rows)
    label_ids = sorted(selected)
    atomic_json(destination / "subset_manifest.json", {
        "run_id": config["run_id"], "selection_uses_answers": False,
        "label_count": len(label_ids), "set_id_count": len({row["set_id"] for row in rows}),
        "reason_counts": dict(Counter(reason for row in rows for reason in row["reasons"])),
        "coverage_counts": mapping["coverage_counts"],
        "background_requested": config["subset"]["background_labels"],
        "background_selected": len(background_ids), "hard_negative_selected": len(hard_ids),
        "estimated_chunk_units": sum(max(1, row["text_chars"] // 900) for row in rows),
        "label_ids_sha256": hashlib.sha256("\n".join(label_ids).encode()).hexdigest(),
        "label_ids": label_ids,
    })
    atomic_json(destination / "rxqa_label_map_selected.json", {
        "answer_fields_used": False,
        "questions": [{"qid": qid, "label_ids": ids} for qid, ids in sorted(per_question.items())],
    })
    atomic_json(destination / "scope_decision.json", {
        "decision": "coverage_subset", "full_openfda_vector_index": False,
        "reason": "Build one auditable question-relevant corpus with hard negatives and fixed background under strict disk/RAM limits.",
        "resource_caps": config["subset"], "runtime_controls": config["runtime"],
        "core_labels_before_background": len(core), "selected_labels": len(selected),
        "selection_rules": config["subset"],
        "interpretation_limit": "Paired comparison on a fixed coverage-oriented subset; not full-corpus openFDA retrieval.",
    })
    print(json.dumps({"selected_labels": len(selected), "core": len(core),
                      "hard_negatives": len(hard_ids), "background": len(background_ids)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-sources", "catalog", "map", "select"])
    parser.add_argument("--config", type=Path, default=config_path("step1_main.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.config.resolve() != config_path("step1_main.json").resolve():
        config["run_id"] = json.loads(args.config.read_text())["run_id"]
    {"prepare-sources": prepare_sources, "catalog": build_catalog,
     "map": map_questions, "select": select_subset}[args.command](config)


if __name__ == "__main__":
    main()
