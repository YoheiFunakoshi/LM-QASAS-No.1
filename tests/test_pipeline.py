"""Synthetic-only integration tests: no checkpoint or research data is used."""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lmqasas.inputs import CSV_COLUMNS
from lmqasas import pipeline
from lmqasas.pipeline import code_provenance, reselect, run_analysis


def artificial_row(cdr3="CASSF", **changes):
    row = {
        "Vseg": "IGHV1-2*01", "Jseg": "IGHJ4*01", "CDR3": cdr3,
        "AAlength": str(len(cdr3)), "NTlength": str(3 * len(cdr3)),
        "Type": "WithConserved_NoStop", "Cseg": "IGHG1*01",
        "Counts": "3", "Frequency(%)": "10",
    }
    row.update(changes)
    return row


class SyntheticEncoder:
    """Deterministic invented vectors; does not approximate AbLang2."""

    def __init__(self, invalid=False):
        self.metadata = {"kind": "synthetic_test"}
        self.calls = []
        self.invalid = invalid

    def encode(self, sequences, progress=None):
        self.calls.append(list(sequences))
        vectors = np.zeros((len(sequences), 480), dtype=np.float64)
        vectors[:, 0] = np.arange(1, len(sequences) + 1)
        vectors[:, 1] = vectors[:, 0] ** 2
        if progress:
            progress("embedding", 1.0)
        return vectors[:, :3] if self.invalid else vectors


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "synthetic_inputs"
        self.input_dir.mkdir()
        self.output_dir = self.root / "synthetic_results"
        self.paths = {phase: self.input_dir / f"{phase}.csv" for phase in ("Pre", "Peak", "Post")}
        self.source_rows = {
            "Pre": [artificial_row(), artificial_row("CAGGF")],
            "Peak": [
                artificial_row(),
                artificial_row(Counts="200"),
                artificial_row(Vseg="IGHV3-7*01"),
                artificial_row("CAGGF"),
                artificial_row("CASWF"),
            ],
            "Post": [artificial_row(), artificial_row("CASYF")],
        }
        for phase, path in self.paths.items():
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self.source_rows[phase])
        self.original_bytes = {phase: path.read_bytes() for phase, path in self.paths.items()}
        self.encoder = SyntheticEncoder()

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, **changes):
        options = {
            "top_n": 3, "n_clusters": 1, "epsilon": 1.0,
            "n_init": 1, "threads": 1, "encoder": self.encoder,
        }
        options.update(changes)
        return run_analysis(self.paths, "synthetic-subject", self.root / "unused_model", self.output_dir, **options)

    def read_json(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def read_csv(self, path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def assert_originals_unchanged(self):
        self.assertEqual(self.original_bytes, {phase: path.read_bytes() for phase, path in self.paths.items()})

    def test_end_to_end_counts_scores_unique_output_and_full_provenance(self):
        progress = []
        run_dir = self.run_pipeline(progress=lambda stage, fraction: progress.append((stage, fraction)))
        metadata = self.read_json(run_dir / "run_metadata.json")
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(all(metadata["inputs_unchanged_after_run"].values()))
        self.assertEqual(metadata["embedding"]["kind"], "synthetic_test")
        self.assertEqual(metadata["scoring"]["observations"], 8)
        self.assertEqual(metadata["scoring"]["unique_embedding_vectors"], 4)
        self.assertEqual(metadata["scoring"]["dimensions"], 480)
        self.assertIn("*", (run_dir / ".gitignore").read_text(encoding="utf-8").splitlines())
        self.assertIn("*", (run_dir / "selection_initial" / ".gitignore").read_text(encoding="utf-8").splitlines())
        provenance_key = "selection_initial/candidate_provenance.json"
        self.assertEqual(metadata["artifact_sha256"][provenance_key],
                         hashlib.sha256((run_dir / provenance_key).read_bytes()).hexdigest())
        self.assertIn(("validated", 1.0), progress)
        self.assertIn(("embedding", 1.0), progress)
        self.assertEqual(progress[-1], ("completed", 1.0))
        self.assertEqual(self.encoder.calls, [["CAGGF", "CASSF", "CASWF", "CASYF"]])
        self.assertEqual(np.load(run_dir / "embeddings.npy", allow_pickle=False).shape, (4, 480))
        self.assertEqual(np.load(run_dir / "cluster_labels.npy", allow_pickle=False).shape, (8,))
        clones = self.read_json(run_dir / "clones.json")
        self.assertEqual(len(clones), 8)
        shared = [clone for clone in clones if clone["cdr3"] == "CASSF"]
        self.assertEqual(len(shared), 4)
        self.assertEqual(len({clone["embedding_index"] for clone in shared}), 1)
        self.assertEqual({clone["timepoint"] for clone in shared}, {"Pre", "Peak", "Post"})

        clusters = self.read_csv(run_dir / "cluster_scores.csv")
        self.assertEqual([(int(c["n_pre"]), int(c["n_peak"]), int(c["n_post"])) for c in clusters], [(2, 4, 2)])
        self.assertAlmostEqual(float(clusters[0]["score"]), 8 / 3)
        candidates = self.read_csv(run_dir / "selection_initial" / "candidates.csv")
        self.assertEqual([c["cdr3"] for c in candidates], ["CAGGF", "CASSF", "CASWF"])
        self.assertEqual(len({c["cdr3"] for c in candidates}), 3)
        self.assertTrue(all(abs(float(c["score"]) - 8 / 3) < 1e-12 for c in candidates))
        provenance = self.read_json(run_dir / "selection_initial" / "candidate_provenance.json")
        selected_shared = next(c for c in provenance if c["cdr3"] == "CASSF")
        self.assertEqual(len(selected_shared["clone_records"]), 2)
        first_clone = selected_shared["clone_records"][0]
        self.assertEqual(first_clone["source_rows"], [2, 3])
        self.assertEqual(first_clone["raw_records"], self.source_rows["Peak"][:2])
        self.assertTrue(all(r["timepoint"] == "Peak" for c in provenance for r in c["clone_records"]))
        self.assert_originals_unchanged()

    def test_reselect_presets_without_encoding_or_overwriting_prior_outputs(self):
        run_dir = self.run_pipeline(top_n=1)
        snapshot = {str(path.relative_to(run_dir)): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
        selected_folders = []
        for requested in (300, 500, 1000, 300):
            folder = reselect(run_dir, requested)
            selected_folders.append(folder)
            summary = self.read_json(folder / "selection_summary.json")
            self.assertEqual(summary["requested"], requested)
            self.assertEqual(summary["returned"], 3)
            self.assertEqual(summary["shortfall"], requested - 3)
            rows = self.read_csv(folder / "candidates.csv")
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["cdr3"] for row in rows}), 3)
            self.assertEqual(self.read_json(folder / "selection_source.json")["source_run"], str(run_dir.resolve()))
            self.assertIn("*", (folder / ".gitignore").read_text(encoding="utf-8").splitlines())
        self.assertEqual(len(set(selected_folders)), 4)
        self.assertEqual(len(self.encoder.calls), 1)
        self.assertEqual(snapshot, {name: (run_dir / name).read_bytes() for name in snapshot})
        self.assert_originals_unchanged()

    def test_output_within_source_directory_is_rejected_before_writing(self):
        for output in (self.input_dir, self.input_dir / "nested_outputs"):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(ValueError, "outside the source CSV directories"):
                    run_analysis(self.paths, "synthetic-subject", self.root / "unused_model", output,
                                 n_clusters=1, encoder=self.encoder)
        self.assertEqual(self.encoder.calls, [])
        self.assertFalse((self.input_dir / "nested_outputs").exists())
        self.assertEqual(set(self.input_dir.iterdir()), set(self.paths.values()))
        self.assert_originals_unchanged()

    def test_invalid_embedding_records_failed_run_and_cannot_reselect(self):
        with self.assertRaisesRegex(ValueError, "Embedding shape"):
            self.run_pipeline(encoder=SyntheticEncoder(invalid=True))
        folders = list(self.output_dir.glob("run_*"))
        self.assertEqual(len(folders), 1)
        metadata = self.read_json(folders[0] / "run_metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["error_type"], "ValueError")
        self.assertTrue(all(metadata["inputs_unchanged_after_run"].values()))
        snapshot = set(folders[0].iterdir())
        with self.assertRaisesRegex(ValueError, "Only completed"):
            reselect(folders[0], 300)
        self.assertEqual(snapshot, set(folders[0].iterdir()))
        self.assert_originals_unchanged()

    def test_modified_scores_reject_reselection_without_new_output(self):
        run_dir = self.run_pipeline()
        scored_file = run_dir / "scored_peak.json"
        # Tamper only with this test's generated artifact, never with an input.
        scores = self.read_json(scored_file)
        scores[0]["score"] += 1
        scored_file.write_text(json.dumps(scores), encoding="utf-8")
        before = set(run_dir.iterdir())
        with self.assertRaisesRegex(ValueError, "integrity check"):
            reselect(run_dir, 500)
        self.assertEqual(before, set(run_dir.iterdir()))
        self.assertEqual(len(self.encoder.calls), 1)
        self.assert_originals_unchanged()

    def test_cluster_count_excess_requires_explicit_change(self):
        with self.assertRaisesRegex(ValueError, "Cluster count"):
            self.run_pipeline(n_clusters=5)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(self.encoder.calls, [])
        self.assert_originals_unchanged()

    def test_invalid_parameters_fail_before_encoding_and_output_creation(self):
        for changes in (
            {"epsilon": float("nan")}, {"epsilon": float("inf")},
            {"epsilon": True}, {"n_init": 0}, {"threads": False},
            {"batch_size": 0}, {"seed": -1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.run_pipeline(**changes)
                self.assertFalse(self.output_dir.exists())
                self.assertEqual(self.encoder.calls, [])
        self.assert_originals_unchanged()

    def test_missing_git_keeps_source_hash_provenance(self):
        with patch("lmqasas.pipeline.subprocess.run", side_effect=FileNotFoundError("synthetic missing git")):
            result = code_provenance()
        self.assertIsNone(result["git_base_commit"])
        self.assertTrue(result["source_sha256"])
        key = "src/lmqasas/pipeline.py"
        self.assertEqual(result["source_sha256"][key],
                         hashlib.sha256((pipeline.ROOT / key).read_bytes()).hexdigest())

    def test_disappearing_synthetic_source_leaves_failed_metadata(self):
        synthetic_path = self.paths["Pre"]

        class DeletingSyntheticEncoder(SyntheticEncoder):
            def encode(self, sequences, progress=None):
                vectors = super().encode(sequences, progress=progress)
                # This path belongs exclusively to this test's temporary fixture.
                synthetic_path.unlink()
                return vectors

        progress = []
        with self.assertRaisesRegex(RuntimeError, "Input changed"):
            self.run_pipeline(encoder=DeletingSyntheticEncoder(),
                              progress=lambda stage, fraction: progress.append((stage, fraction)))
        folders = list(self.output_dir.glob("run_*"))
        self.assertEqual(len(folders), 1)
        metadata = self.read_json(folders[0] / "run_metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["inputs_unchanged_after_run"], {"Pre": False, "Peak": True, "Post": True})
        self.assertNotIn(("completed", 1.0), progress)
        with self.assertRaisesRegex(ValueError, "Only completed"):
            reselect(folders[0], 300)
        for phase in ("Peak", "Post"):
            self.assertEqual(self.original_bytes[phase], self.paths[phase].read_bytes())


if __name__ == "__main__":
    unittest.main()
