import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import openfda_rxqa
from openfda_rxqa import dense_top_k, import_labels, product_names, questions
from select_subset import normalize_name, stable_hash


class Step1Tests(unittest.TestCase):
    def test_spl_product_name_fallback(self):
        row = {
            "openfda": {},
            "spl_product_data_elements": ["KRAZATI adagrasib ADAGRASIB"],
        }
        self.assertEqual(product_names(row), (["KRAZATI"], ["adagrasib"]))
        explicit = {"openfda": {"brand_name": ["Brand"], "generic_name": ["Generic"]}}
        self.assertEqual(product_names(explicit), (["Brand"], ["Generic"]))

    def test_dense_top_k_matches_exact_sort(self):
        rng = np.random.default_rng(5018)
        matrix = rng.normal(size=(37, 8)).astype(np.float32)
        matrix[4] = matrix[30]
        vector = rng.normal(size=8).astype(np.float32)
        expected = (np.argsort(-(matrix @ vector), kind="stable")[:10] + 1).tolist()
        for block_size in (1, 3, 7, 16, 64):
            self.assertEqual(dense_top_k(matrix, vector, k=10, block_size=block_size), expected)

    def test_questions_do_not_expose_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rxqa.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "q_id", "question", "A", "B", "C", "D", "correct", "correct_idx",
                    "pharmacist_rated_difficulty",
                ])
                writer.writeheader()
                writer.writerow({
                    "q_id": "q1", "question": "Question?", "A": "a", "B": "b", "C": "c", "D": "d",
                    "correct": "b", "correct_idx": "1", "pharmacist_rated_difficulty": "Easy",
                })
            row = questions(path, 0, 5018)[0]
            self.assertNotIn("gold", row)
            self.assertNotIn("correct", json.dumps(row))

    def test_selected_import_skips_unselected_labels(self):
        rows = [{
            "id": f"id-{index}", "set_id": f"set-{index}", "version": "1", "effective_time": "20260101",
            "openfda": {"brand_name": [f"Drug{index}"]},
            "warnings": [f"Warning for drug {index}"],
            "spl_product_data_elements": [f"Drug{index} generic{index}"],
        } for index in range(3)]
        db = sqlite3.connect(":memory:")
        with patch.object(openfda_rxqa, "records", return_value=iter(rows)):
            total, usable = import_labels([], db, {"id-1"})
        self.assertEqual((total, usable), (3, 1))
        self.assertEqual(db.execute("SELECT label_id FROM labels").fetchall(), [("id-1",)])

    def test_stable_hash_and_name_normalization(self):
        self.assertEqual(stable_hash(5018, "a", "b"), stable_hash(5018, "a", "b"))
        self.assertNotEqual(stable_hash(5018, "a"), stable_hash(5018, "b"))
        self.assertEqual(normalize_name("KRAZATI (adagrasib)"), "krazati adagrasib")


if __name__ == "__main__":
    unittest.main()
