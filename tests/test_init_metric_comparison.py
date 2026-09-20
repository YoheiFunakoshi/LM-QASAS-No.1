"""Synthetic checks for independent initialization and distance-metric comparisons.

These tests use invented vectors and coordinates. UMAP is replaced by a test
double; dual-reference diagnostic arithmetic is also checked against explicit distances. No model, real sequence or research input is opened.
"""
from contextlib import ExitStack
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from types import SimpleNamespace

import lmqasas
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_init_metric_comparison.py'
if not script.is_file():
    script = Path(__file__).with_name('compute_init_metric_comparison.py')
sys.path.insert(0, str(Path(lmqasas.__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('init_metric_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='init_metric_synthetic_', dir=Path(__file__).parent)
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

    def assert_originals_unchanged(self):
        for path, value in self.before.items():
            self.assertEqual(path.read_bytes(), value, str(path))

    def call_compute(self, output, seed=11):
        comparison.compute(self.baseline, self.projection, self.pooling, output, seed)

    def patch_workflow(self, stack, fit=None, diagnostic=None):
        stack.enter_context(patch.object(comparison, 'load_context', return_value=self.context))
        fitter = stack.enter_context(patch.object(comparison, 'audited_fit',
            side_effect=lambda vectors, settings: ((fit(vectors, settings) if fit else self.coordinates + .1),
                {'solver_fallback_to_random': False, 'warnings': [], 'graph_components': {'synthetic': True}})))
        diagnostics = stack.enter_context(patch.object(comparison, 'dual_neighborhood_diagnostics',
            side_effect=diagnostic, return_value={'synthetic': True}))
        source_check = stack.enter_context(patch.object(comparison, 'verify_unchanged'))
        stack.enter_context(patch.object(comparison, 'code_provenance', return_value={'synthetic': True}))
        return fitter, diagnostics, source_check

    def test_existing_and_source_or_prior_subfolders_rejected_before_umap(self):
        existing = self.root / 'existing'
        existing.mkdir()
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, 'audited_fit') as fit:
            for output, error in ((existing, FileExistsError), (self.source / 'new', ValueError),
                                  (self.baseline / 'new', ValueError), (self.projection / 'new', ValueError),
                                  (self.pooling / 'new', ValueError)):
                with self.subTest(output=output), self.assertRaises(error):
                    self.call_compute(output)
            fit.assert_not_called()
        self.assert_originals_unchanged()

    def test_wrong_baseline_and_invalid_seed_rejected_before_creating_output(self):
        output = self.root / 'new'
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, 'audited_fit') as fit:
            for seed in (7, -1, 2**32, True, np.int64(11)):
                with self.subTest(seed=repr(seed)), self.assertRaises(ValueError):
                    self.call_compute(output, seed)
            for key, value in (('n_neighbors', 30), ('min_dist', 0.3), ('init', 'spectral'), ('metric', 'euclidean')):
                with self.subTest(baseline=(key, value)), patch.dict(self.parameters, {key: value}), \
                     self.assertRaises(ValueError):
                    self.call_compute(output)
            fit.assert_not_called()
        self.assertFalse(output.exists())
        self.assert_originals_unchanged()

    def test_previous_metadata_mismatch_rejected_before_creating_output(self):
        output = self.root / 'new'
        for key, value in (('status', 'incomplete'), ('source_sha256', {}),
                           ('projection_sha256', {'synthetic': 'wrong'}),
                           ('packages', {'synthetic': 'different-version'})):
            changed = self.previous | {key: value}
            with self.subTest(field=key), \
                 patch.object(comparison, 'load_context', return_value=self.context), \
                 patch.object(comparison, 'read', return_value=changed), \
                 patch.object(comparison, 'audited_fit') as fit, self.assertRaises(ValueError):
                self.call_compute(output)
            fit.assert_not_called()
            self.assertFalse(output.exists())
        self.assert_originals_unchanged()

    def test_reused_fit_provenance_mismatch_never_reaches_new_umap_or_completion(self):
        for position, (key, value) in enumerate((('parameters', self.parameters | {'metric': 'euclidean'}),
                ('source_clone_sha256', 'wrong-clones'), ('embedding_sha256', 'wrong-embeddings'))):
            output = self.root / f'invalid_reuse_{position}'
            changed = json.loads(json.dumps(self.previous))
            changed['fits']['all_seed7'][key] = value
            with self.subTest(field=key), \
                 patch.object(comparison, 'load_context', return_value=self.context), \
                 patch.object(comparison, 'read', return_value=changed), \
                 patch.object(comparison, 'audited_fit') as fit, self.assertRaises(ValueError):
                self.call_compute(output)
            fit.assert_not_called()
            self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assert_originals_unchanged()

    def test_all_rows_two_reused_four_new_fits_and_fixed_conditions(self):
        output = self.root / 'new'
        captured = []

        def fake_umap(vectors, settings):
            self.assertTrue((output / 'experiment_plan.json').is_file())
            captured.append((vectors.copy(), dict(settings)))
            scale = np.array([1.3 if settings['init'] == 'spectral' else .7, 1 + settings['random_state'] / 30], dtype=np.float32)
            return self.coordinates * scale

        with ExitStack() as stack:
            _, diagnostics, source_check = self.patch_workflow(stack, fit=fake_umap)
            density = stack.enter_context(patch.object(comparison, 'fixed_density', wraps=comparison.fixed_density))
            self.call_compute(output)
        self.assertEqual(len(captured), 4)
        self.assertEqual({(s['init'], s['metric'], s['random_state']) for _, s in captured},
                         {('spectral', 'cosine', 7), ('spectral', 'cosine', 11),
                          ('random', 'euclidean', 7), ('random', 'euclidean', 11)})
        for vectors, settings in captured:
            self.assertEqual(vectors.dtype, np.dtype('float32'))
            assert_array_equal(vectors, self.cached[self.indices].astype(np.float32))
            self.assertEqual(settings, self.parameters | {'init': settings['init'], 'metric': settings['metric'],
                'random_state': settings['random_state'], 'transform_seed': settings['random_state']})
        diagnostics.assert_called_once()
        assert_array_equal(diagnostics.call_args.args[2], self.indices)
        self.assertEqual(len(diagnostics.call_args.args[3]), 6)
        source_check.assert_called_once_with(self.context)
        self.assertEqual(density.call_count, 6)
        self.assertEqual({call.args[-1] for call in density.call_args_list}, {0.5})
        result = comparison.read(output / 'comparison_metadata.json')
        plan = comparison.read(output / 'experiment_plan.json')
        self.assertEqual(plan['variants'], comparison.VARIANTS)
        self.assertEqual(plan['fixed_n_neighbors'], 15)
        self.assertTrue(plan['all_clones_for_fit'])
        self.assertEqual(len(result['fits']), 6)
        self.assertEqual(len(result['coordinate_metrics']), 7)
        self.assertTrue(result['all_clones_retained'])
        self.assertFalse(result['downsampling_for_umap'])
        self.assertFalse(result['production_defaults_changed'])
        self.assertFalse(result['candidate_selection_performed'])
        self.assertEqual(result['common_kde']['bandwidth'], 0.5)
        for seed in (7, 11):
            assert_array_equal(np.load(output / f'baseline_seed{seed}_raw.npy', allow_pickle=False),
                               self.coordinates + (seed - 7))
            self.assertTrue(result['fits'][f'baseline_seed{seed}']['reused_previous_fit'])
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
        comparison._verify_sources(output, result['artifact_sha256'])
        self.assert_originals_unchanged()

    def test_nonfinite_new_coordinates_cannot_mark_run_completed(self):
        output = self.root / 'bad_coordinates'
        with ExitStack() as stack:
            _, diagnostic, source_check = self.patch_workflow(stack,
                fit=lambda *_: np.full(self.coordinates.shape, np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'finite coordinates'):
                self.call_compute(output)
        diagnostic.assert_not_called()
        source_check.assert_not_called()
        self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assert_originals_unchanged()

    def test_prior_metadata_mutation_during_compute_prevents_completed_status(self):
        output = self.root / 'changed_source'

        def change_synthetic_metadata(*_):
            with (self.pooling / 'comparison_metadata.json').open('a', encoding='utf-8') as handle:
                handle.write('\n')
            return {'synthetic': True}

        with ExitStack() as stack:
            self.patch_workflow(stack, diagnostic=change_synthetic_metadata)
            with self.assertRaisesRegex(ValueError, 'Previous comparison changed'):
                self.call_compute(output)
        self.assertFalse((output / 'comparison_metadata.json').exists())
        # This test deliberately changed only a disposable artificial manifest.
        for path, value in self.before.items():
            if path.name != 'comparison_metadata.json':
                self.assertEqual(path.read_bytes(), value)


    def test_spectral_fallback_is_recorded_and_prevents_success(self):
        output = self.root / 'fallback'
        audit = {'solver_fallback_to_random': True,
                 'warnings': [{'message': 'Spectral initialisation failed; falling back to random initialisation!'}]}
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, 'audited_fit', return_value=(self.coordinates.copy(), audit)) as fitter, \
             self.assertRaisesRegex(RuntimeError, 'fell back to random'):
            self.call_compute(output)
        self.assertEqual(fitter.call_count, 1)
        self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assertEqual(comparison.read(output / 'spectral_seed7_initialization_audit.json'), audit)
        self.assert_originals_unchanged()

    def test_current_source_change_prevents_completed_status(self):
        output = self.root / 'source_check_failure'
        with ExitStack() as stack:
            _, _, source_check = self.patch_workflow(stack)
            source_check.side_effect = ValueError('Frozen source changed')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                self.call_compute(output)
        self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assert_originals_unchanged()

    def test_explicit_retention_preserves_fallback_label_and_individual_seed(self):
        output = self.root / 'retained_fallback'
        def fit(vectors, settings):
            fallback = settings['init'] == 'spectral' and settings['random_state'] == 7
            return self.coordinates.copy(), {
                'solver_fallback_to_random': fallback,
                'warnings': [{'category': 'UserWarning', 'message': 'Falling back to random initialisation!'}] if fallback else [],
                'graph_components': {'synthetic': True}}
        with ExitStack() as stack:
            _, _, source_check = self.patch_workflow(stack)
            stack.enter_context(patch.object(comparison, 'audited_fit', side_effect=fit))
            comparison.compute(self.baseline, self.projection, self.pooling, output, 11,
                               retain_spectral_fallback=True)
        result = comparison.read(output / 'comparison_metadata.json')
        self.assertEqual(result['status'], 'completed_with_fallback')
        self.assertEqual(result['fallback_policy'], 'retain_and_label')
        self.assertEqual(result['spectral_fallback_fits'], ['spectral_seed7'])
        self.assertTrue(result['fits']['spectral_seed7']['initialization_audit']['solver_fallback_to_random'])
        self.assertFalse(result['fits']['spectral_seed11']['initialization_audit']['solver_fallback_to_random'])
        self.assertEqual(len(result['fits']), 6)
        self.assertFalse(result['production_defaults_changed'])
        source_check.assert_called_once_with(self.context)
        comparison._verify_sources(output, result['artifact_sha256'])
        self.assert_originals_unchanged()

    def test_retention_option_without_fallback_keeps_normal_completion_status(self):
        output = self.root / 'retention_without_fallback'
        with ExitStack() as stack:
            self.patch_workflow(stack)
            comparison.compute(self.baseline, self.projection, self.pooling, output, 11,
                               retain_spectral_fallback=True)
        result = comparison.read(output / 'comparison_metadata.json')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['fallback_policy'], 'retain_and_label')
        self.assertEqual(result['spectral_fallback_fits'], [])
        self.assert_originals_unchanged()


class DiagnosticChecks(unittest.TestCase):
    def test_audited_fit_captures_warning_and_prunes_weak_component_bridge(self):
        from scipy.sparse import csr_matrix
        graph = np.zeros((7, 7))
        for i, j, weight in ((0, 1, 1.), (1, 2, 1.), (2, 3, 1.),
                              (4, 5, 1.), (5, 6, 1.), (3, 4, .001)):
            graph[i, j] = graph[j, i] = weight
        settings = {'n_epochs': 200, 'n_components': 2, 'init': 'spectral'}
        vectors = np.arange(21, dtype=np.float32).reshape(7, 3)
        class FakeUMAP:
            def __init__(self, **parameters):
                self.graph_ = csr_matrix(graph)
            def fit_transform(self, data):
                warnings.warn('Spectral initialisation failed; falling back to random initialisation!', UserWarning)
                return data[:, :2].astype(np.float64)
        with patch.dict(sys.modules, {'umap': SimpleNamespace(UMAP=FakeUMAP)}):
            coordinates, audit = comparison.audited_fit(vectors, settings)
        self.assertEqual(coordinates.dtype, np.dtype('float32'))
        assert_array_equal(coordinates, vectors[:, :2])
        self.assertTrue(audit['solver_fallback_to_random'])
        self.assertEqual(audit['warnings'][0]['category'], 'UserWarning')
        components = audit['graph_components']
        self.assertEqual(components['count'], 2)
        self.assertEqual(components['sizes'], [4, 3])
        self.assertEqual(components['small_component_count'], 1)
        self.assertEqual(components['small_component_vertices'], 3)
        self.assertEqual(components['prune_threshold'], .005)

    def test_declared_variants_change_exactly_one_factor(self):
        baseline = comparison.VARIANTS['baseline']
        self.assertEqual(baseline, {'init': 'random', 'metric': 'cosine'})
        for variant, expected_factor in (('spectral', 'init'), ('euclidean', 'metric')):
            alternative = comparison.VARIANTS[variant]
            changed = {key for key in baseline if baseline[key] != alternative[key]}
            self.assertEqual(changed, {expected_factor})
            self.assertEqual(set(alternative), set(baseline))

    def test_fallback_warning_is_case_insensitive_and_not_any_warning(self):
        self.assertFalse(comparison.fallback_warning([]))
        self.assertFalse(comparison.fallback_warning([{'message': 'Graph is disconnected.'}]))
        self.assertTrue(comparison.fallback_warning([
            {'message': 'Spectral initialisation failed. Falling BACK TO RANDOM initialisation!'}]))

    def test_dual_references_match_manual_distances_and_keep_all_clone_rows(self):
        # Entirely artificial, nonzero vectors with varying directions and norms.
        rng = np.random.default_rng(1749)
        vectors = (rng.normal(size=(64, 2)) * rng.uniform(.3, 4, (64, 1))).astype(np.float32)
        sequences = [f'SYNTHETIC-{i:02d}' + 'X' * (i % 7) for i in range(64)]
        clone_ids = np.concatenate((np.arange(64), np.arange(64)))
        centers = vectors.astype(np.float64)
        delta = np.array([.125, -.25])
        raw = {'synthetic': np.concatenate((centers + delta, centers - delta))}
        before_vectors, before_ids, before_raw = vectors.copy(), clone_ids.copy(), raw['synthetic'].copy()
        with tempfile.TemporaryDirectory(prefix='dual_reference_synthetic_', dir=Path(__file__).parent) as temp:
            output = Path(temp)
            result = comparison.dual_neighborhood_diagnostics(vectors, sequences, clone_ids, raw, output)
            with np.load(output / 'neighborhood_diagnostics.npz', allow_pickle=False) as saved:
                query_ids = saved['query_ids']
                self.assertEqual(len(query_ids), 64)
                self.assertEqual(result['query_count'], 64)
                assert_array_equal(np.sort(query_ids), np.arange(64))
                assert_allclose(saved['synthetic_centroids'], centers, rtol=0, atol=0)
                assert_array_equal(saved['clone_count_by_sequence'], np.full(64, 2))
                assert_allclose(saved['synthetic_duplicate_rms'], np.sqrt(.125**2+.25**2), rtol=0, atol=1e-15)
                q = centers[query_ids]
                differences = q[:, None, :] - centers[None, :, :]
                euclidean = np.sqrt(np.sum(differences**2, axis=-1))
                cosine = 1 - (q @ centers.T) / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(centers, axis=1)[None, :])
                for metric, distances in (('cosine', cosine), ('euclidean', euclidean)):
                    distances[np.arange(len(query_ids)), query_ids] = np.inf
                    ordering = np.argsort(distances, axis=1, kind='stable')
                    for k in comparison.DIAGNOSTIC_K:
                        expected = ordering[:, :k]
                        assert_array_equal(saved[f'high_{metric}_neighbors_k{k}'], expected)
                        self.assertFalse(np.any(expected == query_ids[:, None]))
                        bounds = np.take_along_axis(distances, ordering[:, [k-1, k]], axis=1)
                        assert_allclose(saved[f'high_{metric}_boundary_k{k}'], bounds, rtol=1e-12, atol=1e-14)
                        low = saved[f'synthetic_neighbors_k{k}']
                        expected_overlap = np.array([len(set(a) & set(b))/k for a, b in zip(expected, low, strict=True)])
                        assert_array_equal(saved[f'synthetic_{metric}_overlap_k{k}'], expected_overlap)
                        summary = result['fits']['synthetic']['neighborhoods'][str(k)]['overlap_fraction_by_reference'][metric]
                        self.assertAlmostEqual(summary['mean'], float(expected_overlap.mean()), places=14)
                for k in comparison.DIAGNOSTIC_K:
                    assert_array_equal(saved[f'synthetic_euclidean_overlap_k{k}'], np.ones(64))
                    expected_cross = np.array([len(set(a) & set(b))/k for a, b in zip(
                        saved[f'high_cosine_neighbors_k{k}'], saved[f'high_euclidean_neighbors_k{k}'], strict=True)])
                    assert_array_equal(saved[f'high_cross_metric_overlap_k{k}'], expected_cross)
                    self.assertLess(float(expected_cross.mean()), .99)
            with self.assertRaises(FileExistsError):
                comparison.dual_neighborhood_diagnostics(vectors, sequences, clone_ids, raw, output)
        assert_array_equal(vectors, before_vectors)
        assert_array_equal(clone_ids, before_ids)
        assert_array_equal(raw['synthetic'], before_raw)

    def test_zero_neighborhood_radius_does_not_hide_duplicate_dispersion(self):
        # All centroids coincide, but repeated observations scatter around them.
        vectors = np.column_stack((np.ones(64), np.arange(64)+1)).astype(np.float32)
        ids = np.tile(np.arange(64), 2)
        coords = np.vstack((np.tile([1., 0.], (64, 1)), np.tile([-1., 0.], (64, 1))))
        with tempfile.TemporaryDirectory(prefix='zero_radius_synthetic_', dir=Path(__file__).parent) as temp:
            with self.assertRaisesRegex(ValueError, 'Zero neighborhood radius'):
                comparison.dual_neighborhood_diagnostics(vectors, ['SYNTHETIC']*64, ids,
                    {'synthetic': coords}, Path(temp))
            self.assertFalse((Path(temp) / 'neighborhood_diagnostics.npz').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
