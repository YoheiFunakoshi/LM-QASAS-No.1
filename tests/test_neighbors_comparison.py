"""Synthetic checks for diagnostic grain, exact search and preservation.

All numbers and sequences here are artificial; no model or research inputs are
read. The end-to-end control test replaces UMAP with a deterministic test double.
"""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import lmqasas
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_neighbors_comparison.py'
if not script.is_file():
    script = Path(__file__).with_name('compute_neighbors_comparison.py')
sys.path.insert(0, str(Path(lmqasas.__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('neighbors_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class CentroidChecks(unittest.TestCase):
    def test_all_clone_rows_have_equal_weight_and_saved_indices_are_respected(self):
        ids = np.array([2, 0, 2, 1, 2, 0])
        coordinates = np.array([[0, 0], [2, 0], [0, 0], [8, 5], [9, 0], [4, 0]], dtype=float)
        before = coordinates.copy()
        centers, counts, rms = comparison.sequence_centroids(coordinates, ids, 3)
        assert_array_equal(counts, [2, 1, 3])
        assert_allclose(centers, [[3, 0], [8, 5], [3, 0]])
        assert_allclose(rms, [1, 0, np.sqrt(18)])
        assert_array_equal(coordinates, before)
        permutation = np.array([4, 1, 5, 2, 0, 3])
        for actual, expected in zip(comparison.sequence_centroids(coordinates[permutation], ids[permutation], 3),
                                    (centers, counts, rms), strict=True):
            assert_allclose(actual, expected)

    def test_rigid_transform_preserves_spread_and_transforms_centers(self):
        coordinates = np.array([[0, 0], [2, 0], [3, 1], [4, 3]], dtype=float)
        ids = np.array([0, 0, 1, 1])
        centers, counts, rms = comparison.sequence_centroids(coordinates, ids, 2)
        transform = np.array([[0.6, 0.8], [0.8, -0.6]])
        moved, moved_counts, moved_rms = comparison.sequence_centroids(coordinates @ transform + [7, -5], ids, 2)
        assert_allclose(moved, centers @ transform + [7, -5])
        assert_array_equal(moved_counts, counts)
        assert_allclose(moved_rms, rms)

    def test_missing_sequence_or_malformed_correspondence_rejected(self):
        coordinates = np.zeros((3, 2))
        for ids, n in (([0, 0, 0], 2), ([-1, 0, 1], 2), ([0, 1, 2], 2),
                       ([0.0, 1.0, 1.0], 2), ([0, 1], 2)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                comparison.sequence_centroids(coordinates, np.asarray(ids), n)
        coordinates[0, 0] = np.nan
        with self.assertRaises(ValueError):
            comparison.sequence_centroids(coordinates, np.array([0, 1, 1]), 2)


class NeighborChecks(unittest.TestCase):
    def test_cosine_excludes_self_but_retains_distinct_parallel_candidate_and_ties(self):
        vectors = np.array([[1, 0], [2, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        ids = np.array([0, 3])
        nearest, boundary, radius = comparison.exact_neighbors(vectors[ids], vectors, ids, 'cosine', 2)
        assert_array_equal(nearest, [[1, 2], [2, 4]])
        assert_allclose(boundary, [[1, 1], [1, 2]])
        assert_allclose(radius, [1, 1])

    def test_euclidean_ties_use_saved_index_and_batch_query_indices_match(self):
        # More than one search batch, with deliberately reversed query order.
        coordinates = np.column_stack((np.arange(41, dtype=float), np.zeros(41)))
        ids = np.arange(40, -1, -1)
        nearest, boundary, radius = comparison.exact_neighbors(coordinates[ids], coordinates, ids, 'euclidean', 3)
        expected = []
        expected_boundary = []
        for query_id in ids:
            order = sorted((index for index in range(len(coordinates)) if index != query_id),
                           key=lambda index: (abs(index-int(query_id)), index))
            expected.append(order[:3])
            expected_boundary.append([abs(order[2]-int(query_id)), abs(order[3]-int(query_id))])
        assert_array_equal(nearest, expected)
        assert_allclose(boundary, expected_boundary)
        assert_allclose(radius, np.asarray(expected_boundary)[:, 0])

    def test_invalid_k_and_undefined_cosine_rejected(self):
        vectors = np.eye(4)
        for k in (0, 3, 4):
            with self.subTest(k=k), self.assertRaises(ValueError):
                comparison.exact_neighbors(vectors[:1], vectors, np.array([0]), 'cosine', k)
        vectors[0] = 0
        with self.assertRaises(ValueError):
            comparison.exact_neighbors(vectors[:1], vectors, np.array([0]), 'cosine', 1)


class DiagnosticChecks(unittest.TestCase):
    def test_saved_diagnostic_uses_all_distinct_candidates_and_reports_duplicate_dispersion(self):
        # Synthetic 480D vectors, with repeated clone indices and no model use.
        rng = np.random.default_rng(31)
        cached = rng.normal(size=(60, 480))
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        sequences = ['A' * (5 + i % 11) + alphabet[i // 20] + alphabet[i % 20] for i in range(60)]
        clone_ids = np.r_[np.arange(60), [4, 0, 4, 17]]
        coordinates = np.c_[cached[clone_ids, 0], cached[clone_ids, 1]]
        coordinates[-4:] += [[3, -1], [0, 5], [-1, 4], [1, 1]]
        with tempfile.TemporaryDirectory(prefix='neighbors_synthetic_', dir=Path(__file__).parent) as tmp:
            output = Path(tmp)
            result = comparison.neighborhood_diagnostics(cached, sequences, clone_ids, {'artificial': coordinates}, output)
            with np.load(output / 'neighborhood_diagnostics.npz', allow_pickle=False) as saved:
                expected_ids = np.array(sorted(range(60), key=lambda i: (len(sequences[i]), i)))
                assert_array_equal(saved['query_ids'], expected_ids)
                self.assertEqual(result['candidate_count'], 60)
                self.assertEqual(result['query_count'], 60)
                manual_centers = np.array([coordinates[clone_ids == i].mean(axis=0) for i in range(60)])
                assert_allclose(saved['artificial_centroids'], manual_centers, atol=1e-14, rtol=0)
                self.assertGreater(saved['artificial_duplicate_rms'][4], 1)
                assert_array_equal(saved['clone_count_by_sequence'], np.bincount(clone_ids, minlength=60))
                vectors = cached.astype(np.float32).astype(np.float64)
                unit_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
                for k in (15, 50):
                    expected_high, expected_low = [], []
                    for query in expected_ids:
                        candidates = [i for i in range(60) if i != query]
                        high_distance = 1 - unit_vectors @ unit_vectors[query]
                        low_distance = np.sqrt(np.sum((manual_centers - manual_centers[query])**2, axis=1))
                        expected_high.append(sorted(candidates, key=lambda i: (high_distance[i], i))[:k])
                        expected_low.append(sorted(candidates, key=lambda i: (low_distance[i], i))[:k])
                    assert_array_equal(saved[f'high_neighbors_k{k}'], expected_high)
                    assert_array_equal(saved[f'artificial_neighbors_k{k}'], expected_low)
                    expected_overlap = [len(set(a).intersection(b)) / k
                                        for a, b in zip(expected_high, expected_low, strict=True)]
                    assert_allclose(saved[f'artificial_overlap_k{k}'], expected_overlap, atol=0, rtol=0)
                    self.assertAlmostEqual(result['fits']['artificial']['neighborhoods'][str(k)]['overlap_fraction']['mean'],
                                           float(np.mean(expected_overlap)))


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='neighbors_synthetic_', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.input_file = self.source / 'input.txt'
        self.input_file.write_text('synthetic input; preserve', encoding='utf-8')
        self.baseline = self.root / 'baseline'
        self.baseline.mkdir()
        self.projection = self.baseline / 'projection'
        self.projection.mkdir()
        self.pooling = self.root / 'prior_comparison'
        self.pooling.mkdir()
        self.cached = np.arange(3 * 480, dtype=np.float64).reshape(3, 480) / 7.0
        self.indices = np.array([2, 0, 1, 1, 2, 0, 0, 2, 1])
        self.clones = [{'embedding_index': int(index), 'timepoint': comparison.PHASES[i // 3]}
                       for i, index in enumerate(self.indices)]
        self.parameters = {'n_neighbors': 15, 'n_components': 2, 'metric': 'cosine',
                           'min_dist': 0.1, 'random_state': 7, 'transform_seed': 7,
                           'n_epochs': 200, 'init': 'random', 'n_jobs': 1}
        self.pm = {'parameters': self.parameters, 'packages': {}, 'kde': {'bandwidth': 0.5}}
        self.run = {'parameters': {'threads': 1}, 'input_hashes': {}, 'input_policies': {}}
        self.coordinates = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 3],
                                     [2, 5], [1, 7], [3, 6], [6, 4]], dtype=np.float32)
        comparison.save_array(self.pooling / 'all_embeddings.npy', self.cached)
        self.hashes = {'embeddings.npy': comparison.digest(self.pooling / 'all_embeddings.npy'),
                       'clones.json': 'synthetic-placeholder'}
        prior_fits = {}
        for seed in (7, 11):
            name = f'all_seed{seed}_raw.npy'
            comparison.save_array(self.pooling / name, self.coordinates + (seed - 7))
            prior_fits[f'all_seed{seed}'] = {
                'parameters': self.parameters | {'random_state': seed, 'transform_seed': seed},
                'source_clone_sha256': self.hashes['clones.json'],
                'embedding_sha256': self.hashes['embeddings.npy'], 'raw_coordinates_file': name}
        self.previous = {'status': 'completed', 'source_sha256': self.hashes,
                         'projection_sha256': {}, 'packages': {}, 'fits': prior_fits,
                         'artifact_sha256': {p.name: comparison.digest(p) for p in self.pooling.iterdir()}}
        comparison.write_json(self.pooling / 'comparison_metadata.json', self.previous)
        self.context = (self.baseline, self.run, self.clones, ['AAAAA', 'CCCCC', 'DDDDD'],
                        self.cached, self.hashes, self.projection, self.pm, {},
                        {phase: self.input_file for phase in comparison.PHASES})
        self.before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_source_existing_and_prior_output_locations_rejected_before_umap(self):
        existing = self.root / 'existing'
        existing.mkdir()
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, '_fit_umap') as fit:
            for output, error in ((existing, FileExistsError), (self.source / 'new', ValueError),
                                  (self.baseline / 'new', ValueError), (self.projection / 'new', ValueError),
                                  (self.pooling / 'new', ValueError)):
                with self.subTest(output=output), self.assertRaises(error):
                    comparison.compute(self.baseline, self.projection, self.pooling, output, 11)
            fit.assert_not_called()
        for path, value in self.before.items():
            self.assertEqual(path.read_bytes(), value)

    def test_invalid_seed_rejected_before_creating_output(self):
        output = self.root / 'new'
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, '_fit_umap') as fit:
            for seed in (7, -1, 2**32, True, np.int64(11)):
                with self.subTest(seed=repr(seed)), self.assertRaises(ValueError):
                    comparison.compute(self.baseline, self.projection, self.pooling, output, seed)
            fit.assert_not_called()
        self.assertFalse(output.exists())

    def test_full_control_flow_uses_all_float32_clone_rows_and_only_declared_changes(self):
        output = self.root / 'new'
        captured = []

        def fake_umap(vectors, settings):
            captured.append((vectors.copy(), dict(settings)))
            return self.coordinates + np.float32(settings['n_neighbors'] / 50 + settings['random_state'] / 30)

        # Deliberately isolate orchestration from the expensive scientific fits;
        # exact-neighbor arithmetic is covered independently above.
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, '_fit_umap', side_effect=fake_umap), \
             patch.object(comparison, 'neighborhood_diagnostics', return_value={'synthetic': True}), \
             patch.object(comparison, 'verify_unchanged'), \
             patch.object(comparison, 'code_provenance', return_value={'synthetic': True}):
            comparison.compute(self.baseline, self.projection, self.pooling, output, 11)
        self.assertEqual(len(captured), 4)
        observed_settings = set()
        for vectors, settings in captured:
            self.assertEqual(vectors.dtype, np.dtype('float32'))
            assert_array_equal(vectors, self.cached[self.indices].astype(np.float32))
            self.assertEqual(len(vectors), len(self.clones))
            observed_settings.add((settings['n_neighbors'], settings['random_state']))
            self.assertEqual(settings, self.parameters | {
                'n_neighbors': settings['n_neighbors'], 'random_state': settings['random_state'],
                'transform_seed': settings['random_state']})
        self.assertEqual(observed_settings, {(30, 7), (30, 11), (50, 7), (50, 11)})
        result = json.loads((output / 'comparison_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(len(result['fits']), 6)
        self.assertTrue(result['all_clones_retained'])
        self.assertFalse(result['downsampling_for_umap'])
        self.assertFalse(result['production_defaults_changed'])
        self.assertEqual(result['common_kde']['bandwidth'], 0.5)
        for seed in (7, 11):
            assert_array_equal(np.load(output / f'nn15_seed{seed}_raw.npy', allow_pickle=False),
                               self.coordinates + (seed - 7))
            self.assertTrue(result['fits'][f'nn15_seed{seed}']['reused_previous_fit'])
        grids = [np.load(output / fit['density_file'], allow_pickle=False) for fit in result['fits'].values()]
        try:
            for grid in grids:
                assert_array_equal(grid['x'], grids[0]['x'])
                assert_array_equal(grid['y'], grids[0]['y'])
                integrals = np.trapezoid(np.trapezoid(grid['densities'], x=grid['x'], axis=2), x=grid['y'], axis=1)
                assert_allclose(integrals, np.ones(3), atol=1e-12, rtol=0)
        finally:
            for grid in grids:
                grid.close()
        for path, value in self.before.items():
            self.assertEqual(path.read_bytes(), value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
