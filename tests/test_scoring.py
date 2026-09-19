"""Numeric tests use wholly synthetic vectors, not repertoire data."""

import math
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from lmqasas.scoring import run_kmeans, score_cluster_counts


class ScoreCountsTests(unittest.TestCase):
    def test_score_is_sum_of_ratios(self):
        self.assertEqual(score_cluster_counts(2, 3, 5), 1.5)
        self.assertNotEqual(score_cluster_counts(2, 3, 5), 0.5)
        self.assertEqual(score_cluster_counts(0, 2, 99), 2.02)

    def test_zero_baselines_and_no_peak(self):
        self.assertEqual(score_cluster_counts(0, 3, 0), 6.0)
        self.assertEqual(score_cluster_counts(0, 3, 0, epsilon=0.5), 12.0)
        self.assertEqual(score_cluster_counts(0, 0, 0), 0.0)
        self.assertEqual(score_cluster_counts(3, 0, 9), 0.0)

    def test_invalid_counts_and_epsilon(self):
        for count in (-1, True, np.bool_(False), 1.0, "1", None):
            for position in range(3):
                counts = [1, 1, 1]
                counts[position] = count
                with self.subTest(count=count, position=position), self.assertRaises(ValueError):
                    score_cluster_counts(*counts)
        for epsilon in (0, -1, True, np.nan, np.inf, "1", 1j):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                score_cluster_counts(1, 1, 1, epsilon=epsilon)
        self.assertEqual(score_cluster_counts(np.int64(2), np.int64(3), np.int64(5)), 1.5)

    def test_unrepresentable_score_rejected(self):
        with self.assertRaises(ValueError):
            score_cluster_counts(0, 1, 0, epsilon=np.nextafter(0.0, 1.0))
        with self.assertRaises(ValueError):
            score_cluster_counts(10**1000, 1, 1)


class KMeansTests(unittest.TestCase):
    def setUp(self):
        # Four clone rows per location; repeated vectors must remain repeated.
        self.values = np.array([[-10.0, 0.0]] * 4 + [[10.0, 0.0]] * 4)
        self.timepoints = ["Pre", "Peak", "Peak", "Post", "Pre", "Peak", "Post", "Post"]

    def run_fixture(self, **kwargs):
        params = {"n_clusters": 2, "threads": 1}
        params.update(kwargs)
        return run_kmeans(self.values, self.timepoints, **params)

    def test_duplicate_vectors_keep_each_clone_and_assign_scores(self):
        result = self.run_fixture()
        np.testing.assert_array_equal(result.scores[:4], np.full(4, 2.0))
        np.testing.assert_allclose(result.scores[4:], np.full(4, 5.0 / 6.0))
        self.assertEqual(result.labels.shape, (8,))
        self.assertEqual(len(result.clusters), 2)
        summary = sorted((row["n_pre"], row["n_peak"], row["n_post"]) for row in result.clusters)
        self.assertEqual(summary, [(1, 1, 2), (1, 2, 1)])
        self.assertEqual(sum(sum(row[key] for key in ("n_pre", "n_peak", "n_post")) for row in result.clusters), 8)

    def test_one_cluster_combines_all_three_timepoints(self):
        result = self.run_fixture(n_clusters=1)
        self.assertEqual(result.clusters[0]["n_pre"], 2)
        self.assertEqual(result.clusters[0]["n_peak"], 3)
        self.assertEqual(result.clusters[0]["n_post"], 3)
        np.testing.assert_array_equal(result.scores, np.full(8, 1.75))

    def test_inputs_are_unchanged(self):
        original = self.values.copy()
        points = self.timepoints.copy()
        self.run_fixture()
        np.testing.assert_array_equal(self.values, original)
        self.assertEqual(self.timepoints, points)

    def test_reproducible_for_fixed_configuration(self):
        rng = np.random.default_rng(734)
        vectors = np.concatenate([rng.normal(loc=offset, scale=0.1, size=(12, 3)) for offset in (-3, 0, 3)])
        points = ["Pre", "Peak", "Post"] * 12
        first = run_kmeans(vectors, points, n_clusters=3, seed=0, threads=1)
        second = run_kmeans(vectors, points, n_clusters=3, seed=0, threads=1)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.scores, second.scores)
        self.assertEqual(first.clusters, second.clusters)

    def test_metadata_explains_actual_settings(self):
        result = self.run_fixture(epsilon=0.5, n_init=3, seed=8)
        expected = {"n_clusters": 2, "epsilon": 0.5, "n_init": 3, "seed": 8,
                    "dimensions": 2, "observations": 8, "unique_embedding_vectors": 2,
                    "scaling": "none", "algorithm": "lloyd", "distance": "euclidean",
                    "thread_limit": 1, "fit_dtype": "float64"}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(result.metadata[key], value)
        self.assertIn("epsilon", result.metadata["unconfirmed_paper_settings"])
        self.assertTrue(math.isfinite(result.metadata["inertia"]))

    def test_too_many_clusters_not_silently_reduced(self):
        for clusters in (3, 9):
            with self.subTest(clusters=clusters), self.assertRaises(ValueError):
                self.run_fixture(n_clusters=clusters)

    def test_invalid_parameters(self):
        for name in ("n_clusters", "n_init", "threads"):
            for value in (0, -1, True, 2.5, "2", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.run_fixture(**{name: value})
        for seed in (-1, 2**32, True, 1.0, "1"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.run_fixture(seed=seed)
        for epsilon in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                self.run_fixture(epsilon=epsilon)

    def test_invalid_timepoints_and_missing_timepoint(self):
        malformed = [self.timepoints[:-1], ["Pre"] * 8, ["Pre", "Peak"] * 4,
                     self.timepoints[:-1] + ["post"], self.timepoints[:-1] + ["Post "],
                     self.timepoints[:-1] + [None], "Pre", None]
        for points in malformed:
            with self.subTest(points=points), self.assertRaises(ValueError):
                run_kmeans(self.values, points, n_clusters=2)

    def test_invalid_embeddings(self):
        for values in (self.values.tolist(), np.array([1, 2, 3]), np.empty((0, 2)),
                       np.empty((8, 0)), np.ones((8, 2, 1)), np.ones((8, 2), dtype=bool),
                       np.ones((8, 2), dtype=complex), np.full((8, 2), "not numeric"),
                       np.full((8, 2), np.nan), np.full((8, 2), np.inf)):
            with self.subTest(shape=getattr(values, "shape", None)), self.assertRaises(ValueError):
                run_kmeans(values, self.timepoints, n_clusters=2)

    def test_convergence_warning_stops_scoring(self):
        def warn_on_fit(values):
            warnings.warn("Synthetic convergence warning", ConvergenceWarning)
            return np.zeros(len(values), dtype=np.int64)

        with patch("lmqasas.scoring.KMeans") as model:
            model.return_value.fit_predict.side_effect = warn_on_fit
            with self.assertRaisesRegex(ValueError, "convergence warning"):
                self.run_fixture()

    def test_empty_cluster_stops_scoring(self):
        with patch("lmqasas.scoring.KMeans") as model:
            model.return_value.fit_predict.return_value = np.zeros(8, dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "empty clusters"):
                self.run_fixture()


if __name__ == "__main__":
    unittest.main()
