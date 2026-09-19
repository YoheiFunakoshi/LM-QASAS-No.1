"""Tests use synthetic sequences only; no research data are required."""

from copy import deepcopy
from itertools import islice, product
import unittest

from lmqasas.selection import TOP_N_PRESETS, select_unique_cdr3, validate_requested_count


def clone(cdr3, score, source_row=1, **extra):
    return {
        "cdr3": cdr3,
        "score": score,
        "source_row": source_row,
        "timepoint": "Peak",
        "metadata": {"v_gene": "synthetic", "tags": ["synthetic"]},
        **extra,
    }


class SelectionTests(unittest.TestCase):
    def test_duplicate_sequence_keeps_all_provenance_and_uses_maximum(self):
        records = [
            clone("CASSF", 2, 1, isotype="synthetic-isotype-A"),
            clone("CARWF", 7, 2),
            clone("CASSF", 9, 3, isotype="synthetic-isotype-B"),
            clone("CAAAF", 5, 4),
        ]
        result = select_unique_cdr3(records, 2)
        self.assertEqual([item.cdr3 for item in result.candidates], ["CASSF", "CARWF"])
        self.assertEqual(result.available, 3)
        self.assertEqual(result.returned, 2)
        self.assertEqual(result.shortfall, 0)
        self.assertEqual(result.candidates[0].score, 9)
        self.assertEqual(result.candidates[0].clone_records, (records[0], records[2]))
        self.assertEqual(
            [record["source_row"] for record in result.candidates[0].clone_records], [1, 3]
        )

    def test_tie_breaking_is_stable_and_reports_cutoff_tie(self):
        records = [clone("CAYYF", 4), clone("CAAAF", 4), clone("CASSF", 9), clone("CARWF", 4)]
        result = select_unique_cdr3(records, 2)
        reversed_result = select_unique_cdr3(reversed(records), 2)
        self.assertEqual([item.cdr3 for item in result.candidates], ["CASSF", "CAAAF"])
        self.assertEqual(
            [item.cdr3 for item in result.candidates],
            [item.cdr3 for item in reversed_result.candidates],
        )
        self.assertEqual(result.boundary_tie.score, 4)
        self.assertEqual(result.boundary_tie.selected_count, 1)
        self.assertEqual(result.boundary_tie.excluded_count, 2)
        self.assertEqual(result.returned, 2)

    def test_tie_with_multiple_selected_and_excluded_candidates(self):
        records = [clone("CAYYF", 4), clone("CAAAF", 4), clone("CARWF", 4)]
        result = select_unique_cdr3(records, 2)
        self.assertEqual(result.boundary_tie.selected_count, 2)
        self.assertEqual(result.boundary_tie.excluded_count, 1)

    def test_no_boundary_tie_when_scores_differ_or_all_tied_records_fit(self):
        records = [clone("CASSF", 2), clone("CARWF", 2), clone("CAAAF", 1)]
        self.assertIsNone(select_unique_cdr3(records, 2).boundary_tie)
        self.assertIsNone(select_unique_cdr3(records, 3).boundary_tie)
        self.assertIsNone(select_unique_cdr3(records, 10).boundary_tie)

    def test_insufficient_candidates_are_not_padded(self):
        result = select_unique_cdr3([clone("CASSF", 2), clone("CASSF", 1)], 300)
        self.assertEqual((result.requested, result.available, result.returned, result.shortfall), (300, 1, 1, 299))
        empty = select_unique_cdr3([], 1000)
        self.assertEqual((empty.requested, empty.available, empty.returned, empty.shortfall), (1000, 0, 0, 1000))
        self.assertIsNone(empty.boundary_tie)

    def test_input_and_nested_metadata_remain_independent(self):
        records = [clone("CASSF", 2), clone("CASSF", 3)]
        before = deepcopy(records)
        result = select_unique_cdr3(records, 1)
        self.assertEqual(records, before)
        result.candidates[0].clone_records[0]["metadata"]["tags"].append("output change")
        self.assertEqual(records, before)
        records[1]["metadata"]["tags"].append("input change")
        self.assertEqual(result.candidates[0].clone_records[1]["metadata"]["tags"], ["synthetic"])

    def test_non_peak_records_are_rejected_even_below_cutoff(self):
        for timepoint in ("Pre", "Post", "peak", "Peak ", None):
            with self.subTest(timepoint=timepoint), self.assertRaises(ValueError):
                select_unique_cdr3([clone("CASSF", 100), clone("CARWF", -1, timepoint=timepoint)], 1)

    def test_noncanonical_or_un_normalized_sequences_are_rejected(self):
        for cdr3 in ("", "cassf", " CASSF", "CASSF ", "CAXSF", "CAS*F", "CA-SF", "CA\nSF", "CAÜSF"):
            with self.subTest(cdr3=cdr3), self.assertRaises(ValueError):
                select_unique_cdr3([clone(cdr3, 1)], 1)
        for cdr3 in (None, 1, b"CASSF"):
            with self.subTest(cdr3=cdr3), self.assertRaises(TypeError):
                select_unique_cdr3([clone(cdr3, 1)], 1)

    def test_invalid_scores_are_rejected_without_disclosing_input(self):
        for score in (float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.subTest(score=score), self.assertRaises(ValueError):
                select_unique_cdr3([clone("CASSF", score)], 1)
        for score in (True, False, "5", None, complex(1, 0)):
            with self.subTest(score=score), self.assertRaises(TypeError):
                select_unique_cdr3([clone("CASSF", score)], 1)
        with self.assertRaises(ValueError) as context:
            select_unique_cdr3([clone("SENSITIVE-INVALID", 1)], 1)
        self.assertNotIn("SENSITIVE-INVALID", str(context.exception))

    def test_valid_zero_and_negative_scores_sort_numerically(self):
        result = select_unique_cdr3([clone("CASSF", -2), clone("CARWF", 0), clone("CAAAF", -1.5)], 3)
        self.assertEqual([item.score for item in result.candidates], [0, -1.5, -2])

    def test_missing_fields_and_non_mapping_records_are_rejected(self):
        for field in ("cdr3", "score", "source_row", "timepoint"):
            record = clone("CASSF", 1)
            del record[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_unique_cdr3([record], 1)
        for record in (None, "CASSF", ["CASSF", 1]):
            with self.subTest(record=record), self.assertRaises(TypeError):
                select_unique_cdr3([record], 1)

    def test_requested_count_must_be_positive_integer(self):
        for requested in (True, False, 1.0, "300", None):
            with self.subTest(requested=requested), self.assertRaises(TypeError):
                select_unique_cdr3([], requested)
        for requested in (0, -1):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                validate_requested_count(requested)
        self.assertEqual(validate_requested_count(17), 17)

    def test_presets_and_arbitrary_count_with_1000_distinct_synthetic_sequences(self):
        sequences = ["C" + "".join(item) + "F" for item in islice(product("ACDEFGHIKLMNPQRSTVWY", repeat=3), 1000)]
        self.assertEqual(len(set(sequences)), 1000)
        records = [clone(cdr3, score=index, source_row=index + 1) for index, cdr3 in enumerate(sequences)]
        self.assertEqual(TOP_N_PRESETS, (300, 500, 1000))
        for requested in (*TOP_N_PRESETS, 17):
            with self.subTest(requested=requested):
                result = select_unique_cdr3(iter(records), requested)
                expected = list(reversed(sequences))[:requested]
                self.assertEqual([item.cdr3 for item in result.candidates], expected)
                self.assertEqual((result.requested, result.available, result.returned, result.shortfall), (requested, 1000, requested, 0))
                self.assertIsNone(result.boundary_tie)


if __name__ == "__main__":
    unittest.main()
