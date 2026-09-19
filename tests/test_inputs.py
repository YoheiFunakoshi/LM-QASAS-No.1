"""Input validation tests use invented sequences and labels, never research data."""

import csv
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lmqasas.inputs import CSV_COLUMNS, InputValidationError, load_three_inputs


def synthetic_row(**changes):
    row = {
        "Vseg": "IGHV1-2*01", "Jseg": "IGHJ4*01", "CDR3": "CASSF",
        "AAlength": "5", "NTlength": "15", "Type": "WithConserved_NoStop",
        "Cseg": "IGHG1*01", "Counts": "10", "Frequency(%)": "25.0",
    }
    row.update(changes)
    return row


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {timepoint: self.root / f"synthetic_{timepoint}.csv" for timepoint in ("Pre", "Peak", "Post")}
        for path in self.paths.values():
            self.write(path, [synthetic_row()])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, rows, headers=CSV_COLUMNS):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def load(self):
        return load_three_inputs(self.paths, subject="synthetic-subject")

    def test_retains_timepoints_metadata_and_original_hashes(self):
        before = {timepoint: path.read_bytes() for timepoint, path in self.paths.items()}
        result = self.load()
        self.assertEqual([c["timepoint"] for c in result.clones], ["Pre", "Peak", "Post"])
        for clone in result.clones:
            self.assertEqual(clone["source_rows"], [2])
            self.assertEqual(clone["raw_records"], [synthetic_row()])
            self.assertEqual(clone["subject"], "synthetic-subject")
            self.assertEqual(clone["isotype"], "IGHG1")
            self.assertTrue(Path(clone["source_file"]).is_absolute())
        self.assertEqual(result.input_hashes, {key: hashlib.sha256(data).hexdigest() for key, data in before.items()})
        self.assertEqual(before, {key: path.read_bytes() for key, path in self.paths.items()})
        self.assertFalse(result.audit["policies"]["full_vdj_functionality_verified"])

    def test_alleles_and_order_collapse_but_keep_all_source_records(self):
        rows = [
            synthetic_row(Vseg="IGHV3-7*01//IGHV1-2*02", Counts="1"),
            synthetic_row(Vseg=" IGHV1-2*01 //IGHV3-7*02//IGHV1-2*03", Jseg="IGHJ4*02", Counts="900"),
            synthetic_row(Vseg="IGHV1-2*01", Counts="7"),
        ]
        self.write(self.paths["Peak"], rows)
        result = self.load()
        peak = [c for c in result.clones if c["timepoint"] == "Peak"]
        self.assertEqual(len(peak), 2)
        self.assertEqual(peak[0]["v_gene"], "IGHV1-2//IGHV3-7")
        self.assertEqual(peak[0]["source_rows"], [2, 3])
        self.assertEqual(peak[0]["raw_records"], rows[:2])
        self.assertEqual(result.audit["samples"]["Peak"]["merged_rows"], 1)
        self.assertEqual(result.audit["samples"]["Peak"]["rows_with_multiple_v_genes"], 2)

    def test_subclasses_and_v_j_remain_distinct_despite_identical_cdr3(self):
        rows = [synthetic_row(), synthetic_row(Cseg="IGHG2*01"), synthetic_row(Vseg="IGHV3-7*01"), synthetic_row(Jseg="IGHJ6*01")]
        self.write(self.paths["Peak"], rows)
        peak = [c for c in self.load().clones if c["timepoint"] == "Peak"]
        self.assertEqual(len(peak), 4)

    def test_counts_do_not_replicate_or_weight_clones(self):
        self.write(self.paths["Peak"], [synthetic_row(Counts="0"), synthetic_row(Counts="1000000")])
        result = self.load()
        self.assertEqual(result.audit["samples"]["Peak"]["clone_count"], 1)
        self.assertFalse(result.audit["policies"]["counts_used_as_weights"])
        self.assertNotIn("weight", result.clones[1])

    def test_row_exclusions_report_reasons_and_physical_source_lines(self):
        rows = [
            synthetic_row(CDR3="CAS*F"),
            synthetic_row(CDR3="CASS", AAlength="4", NTlength="12"),
            synthetic_row(AAlength="6"),
            synthetic_row(NTlength="14"),
            synthetic_row(Type="Other"),
            synthetic_row(Cseg="unknown"),
            synthetic_row(),
        ]
        self.write(self.paths["Peak"], rows)
        result = self.load()
        peak = [c for c in result.clones if c["timepoint"] == "Peak"]
        self.assertEqual(peak[0]["source_row"], 8)
        audit = result.audit["samples"]["Peak"]
        self.assertEqual(audit["accepted_rows"], 1)
        self.assertEqual(audit["excluded_rows"], 6)
        self.assertEqual(audit["exclusions"], [
            {"source_row": 2, "reasons": ["cdr3_noncanonical"]},
            {"source_row": 3, "reasons": ["cdr3_too_short"]},
            {"source_row": 4, "reasons": ["aa_length_mismatch"]},
            {"source_row": 5, "reasons": ["nt_length_mismatch"]},
            {"source_row": 6, "reasons": ["type_not_accepted"]},
            {"source_row": 7, "reasons": ["cseg_unmapped"]},
        ])

    def test_invalid_cells_excluded_without_repair(self):
        for changes, reason in [
            ({"CDR3": "cassf"}, "cdr3_noncanonical"),
            ({"CDR3": " CASSF"}, "cdr3_noncanonical"),
            ({"Vseg": "IGHV1-2*01//"}, "v_annotation_invalid"),
            ({"Jseg": ""}, "j_annotation_invalid"),
            ({"AAlength": "5.0"}, "aa_length_invalid"),
            ({"NTlength": ""}, "nt_length_invalid"),
            ({"Counts": "nan"}, "counts_invalid"),
            ({"Counts": "-1"}, "counts_invalid"),
            ({"Frequency(%)": "inf"}, "frequency_invalid"),
            ({"Frequency(%)": "100.1"}, "frequency_invalid"),
        ]:
            with self.subTest(reason=reason, changes=changes):
                self.write(self.paths["Pre"], [synthetic_row(**changes), synthetic_row()])
                audit = self.load().audit["samples"]["Pre"]
                self.assertIn(reason, audit["exclusions"][0]["reasons"])

    def test_explicit_cseg_alias_and_filename_does_not_force_isotype(self):
        self.paths["Peak"] = self.root / "synthetic_IgG.csv"
        self.write(self.paths["Peak"], [synthetic_row(Cseg="IgA2*01")])
        result = self.load()
        self.assertEqual(result.clones[1]["isotype"], "IGHA2")
        self.assertEqual(result.clones[1]["raw_records"][0]["Cseg"], "IgA2*01")

    def test_reordered_headers_accepted_but_missing_duplicate_extra_abort(self):
        self.write(self.paths["Pre"], [synthetic_row()], headers=list(reversed(CSV_COLUMNS)))
        self.assertEqual(len(self.load().clones), 3)
        for header in [CSV_COLUMNS[:-1], (*CSV_COLUMNS, "extra"), (*CSV_COLUMNS[:-1], "Counts")]:
            with self.subTest(header=header):
                self.paths["Pre"].write_text(",".join(header) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(InputValidationError, "column names"):
                    self.load()

    def test_invalid_row_width_and_csv_syntax_abort(self):
        for text in [",".join(CSV_COLUMNS) + "\na,b\n", ",".join(CSV_COLUMNS) + '\n"unterminated']:
            self.paths["Pre"].write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(InputValidationError, "malformed CSV"):
                self.load()

    def test_no_accepted_rows_aborts(self):
        for rows in [[], [synthetic_row(Type="Unknown")]]:
            self.write(self.paths["Pre"], rows)
            with self.assertRaisesRegex(InputValidationError, "no rows remain"):
                self.load()

    def test_missing_timepoints_subject_and_paths_fail_safely(self):
        with self.assertRaisesRegex(InputValidationError, "Exactly Pre"):
            load_three_inputs({"Pre": self.paths["Pre"]}, "synthetic-subject")
        with self.assertRaisesRegex(InputValidationError, "Subject"):
            load_three_inputs(self.paths, "  ")
        private_path = self.root / "private_missing_filename.csv"
        with self.assertRaises(InputValidationError) as caught:
            load_three_inputs({**self.paths, "Pre": private_path}, "synthetic-subject")
        self.assertNotIn(str(private_path), str(caught.exception))
        self.assertNotIn(private_path.name, str(caught.exception))

    def test_same_path_and_hardlink_cannot_be_distinct_timepoints(self):
        with self.assertRaisesRegex(InputValidationError, "different physical files"):
            load_three_inputs({**self.paths, "Post": self.paths["Pre"]}, "synthetic-subject")
        hardlink = self.root / "synthetic_hardlink.csv"
        try:
            os.link(self.paths["Pre"], hardlink)
        except OSError:
            self.skipTest("Filesystem does not support hard links.")
        with self.assertRaisesRegex(InputValidationError, "different physical files"):
            load_three_inputs({**self.paths, "Post": hardlink}, "synthetic-subject")

    def test_concurrent_change_discards_results(self):
        with patch("lmqasas.inputs._sha256_path", return_value="changed"):
            with self.assertRaisesRegex(InputValidationError, "changed during reading"):
                self.load()

    def test_encoding_failure_does_not_echo_file_content(self):
        self.paths["Pre"].write_bytes(b"private-sequence\xff")
        with self.assertRaises(InputValidationError) as caught:
            self.load()
        self.assertIn("UTF-8", str(caught.exception))
        self.assertNotIn("private-sequence", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
