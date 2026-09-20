"""Synthetic orchestration checks for the n_epochs comparison.

These tests use invented vectors and coordinates. UMAP is replaced by a test
double; diagnostic arithmetic already tested in test_init_metric_comparison is
not duplicated here. No model, real sequence or research input is opened.
"""
from contextlib import ExitStack
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

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_epochs_comparison.py'
if not script.is_file():
    script = Path(__file__).with_name('compute_epochs_comparison.py')
sys.path.insert(0, str(Path(lmqasas.__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('epochs_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='epochs_synthetic_', dir=Path(__file__).parent)
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
            side_effect=lambda vectors, settings: (
                fit(vectors, settings) if fit else self.coordinates + settings['n_epochs'] / 1000,
                {'warnings': [], 'solver_fallback_to_random': False, 'graph_components': {'synthetic': True}})))
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
            for key, value in (('n_neighbors', 30), ('min_dist', 0.3), ('n_epochs', 500),
                               ('init', 'spectral'), ('metric', 'euclidean')):
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
            scale = np.array([1 + settings['n_epochs'] / 1000, 1 + settings['random_state'] / 30], dtype=np.float32)
            return self.coordinates * scale

        with ExitStack() as stack:
            _, diagnostics, source_check = self.patch_workflow(stack, fit=fake_umap)
            density = stack.enter_context(patch.object(comparison, 'fixed_density', wraps=comparison.fixed_density))
            self.call_compute(output)
        self.assertEqual(len(captured), 4)
        self.assertEqual({(s['n_epochs'], s['random_state']) for _, s in captured},
                         {(500, 7), (500, 11), (1000, 7), (1000, 11)})
        for vectors, settings in captured:
            self.assertEqual(vectors.dtype, np.dtype('float32'))
            assert_array_equal(vectors, self.cached[self.indices].astype(np.float32))
            self.assertEqual(settings, self.parameters | {'n_epochs': settings['n_epochs'],
                'random_state': settings['random_state'], 'transform_seed': settings['random_state']})
        diagnostics.assert_called_once()
        assert_array_equal(diagnostics.call_args.args[2], self.indices)
        self.assertEqual(len(diagnostics.call_args.args[3]), 6)
        source_check.assert_called_once_with(self.context)
        self.assertEqual(density.call_count, 6)
        self.assertEqual({call.args[-1] for call in density.call_args_list}, {0.5})
        result = comparison.read(output / 'comparison_metadata.json')
        plan = comparison.read(output / 'experiment_plan.json')
        self.assertEqual(plan['epochs'], [200, 500, 1000])
        self.assertEqual(plan['fixed_n_neighbors'], 15)
        self.assertTrue(plan['all_clones_for_fit'])
        self.assertEqual(len(result['fits']), 6)
        self.assertEqual(len(result['coordinate_metrics']), 7)
        self.assertTrue(result['all_clones_retained'])
        self.assertFalse(result['downsampling_for_umap'])
        self.assertFalse(result['production_defaults_changed'])
        self.assertFalse(result['candidate_selection_performed'])
        self.assertEqual(result['common_kde']['bandwidth'], 0.5)
        self.assertEqual(result['epochs'], [200, 500, 1000])
        self.assertEqual(result['common_kde']['grid_rule'], plan['common_grid_rule'])
        self.assertIn('not continuation', plan['epochs_interpretation'])
        for key, fit in result['fits'].items():
            self.assertEqual(fit['parameters']['n_epochs'], fit['n_epochs'])
            self.assertEqual(fit['reused_previous_fit'], fit['n_epochs'] == 200)
            if fit['n_epochs'] != 200:
                audit_path = output / (key + '_initialization_audit.json')
                self.assertEqual(comparison.read(audit_path), fit['initialization_audit'])
                self.assertFalse(fit['initialization_audit']['solver_fallback_to_random'])
        for seed in (7, 11):
            assert_array_equal(np.load(output / f'ep200_seed{seed}_raw.npy', allow_pickle=False),
                               self.coordinates + (seed - 7))
            self.assertTrue(result['fits'][f'ep200_seed{seed}']['reused_previous_fit'])
        grids = [np.load(output / fit['density_file'], allow_pickle=False) for fit in result['fits'].values()]
        try:
            for grid in grids:
                assert_array_equal(grid['x'], grids[0]['x'])
                assert_array_equal(grid['y'], grids[0]['y'])
                self.assertEqual(len(grid['x']), result['common_kde']['grid_size'])
                self.assertLessEqual(float(np.diff(grid['x']).max()), .25 + 1e-12)
                self.assertLessEqual(float(np.diff(grid['y']).max()), .25 + 1e-12)
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


    def test_current_source_change_prevents_completed_status(self):
        output = self.root / 'source_check_failure'
        with ExitStack() as stack:
            _, _, source_check = self.patch_workflow(stack)
            source_check.side_effect = ValueError('Frozen source changed')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                self.call_compute(output)
        self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assert_originals_unchanged()

    def test_unexpected_fallback_retains_audit_and_prevents_completed_status(self):
        output = self.root / 'unexpected_fallback'
        audit = {'solver_fallback_to_random': True, 'warnings': [
            {'category': 'UserWarning', 'message': 'Falling back to random initialisation!'}]}
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, 'audited_fit', return_value=(self.coordinates.copy(), audit)) as fitter, \
             self.assertRaisesRegex(RuntimeError, 'Unexpected solver fallback'):
            self.call_compute(output)
        self.assertEqual(fitter.call_count, 1)
        self.assertFalse((output / 'comparison_metadata.json').exists())
        self.assertEqual(comparison.read(output / 'ep500_seed7_initialization_audit.json'), audit)
        self.assert_originals_unchanged()


class GridChecks(unittest.TestCase):
    def test_minimum_grid_retains_degenerate_coordinates_with_full_margin(self):
        aligned = {'a': np.array([[2., -3.], [2., -3.]])}
        before = aligned['a'].copy()
        x, y = comparison.common_grid(aligned, 2.)
        self.assertEqual((len(x), len(y)), (96, 96))
        assert_allclose([x[0], x[-1], y[0], y[-1]], [-4., 8., -9., 3.], atol=0, rtol=0)
        self.assertLessEqual(float(np.diff(x).max()), 1.)
        assert_array_equal(aligned['a'], before)

    def test_grid_resolves_small_bandwidth_and_contains_every_fit(self):
        aligned = {'a': np.array([[-50., -5.], [0., 0.]]),
                   'b': np.array([[50., 5.], [20., -3.]])}
        before = {key: value.copy() for key, value in aligned.items()}
        x, y = comparison.common_grid(aligned, .5)
        self.assertEqual((len(x), len(y)), (441, 441))
        self.assertLessEqual(float(np.diff(x).max()), .25 + 1e-12)
        self.assertLessEqual(float(np.diff(y).max()), .25 + 1e-12)
        assert_allclose([x[0], x[-1], y[0], y[-1]], [-55., 55., -6.5, 6.5], atol=0, rtol=0)
        for key, values in aligned.items():
            self.assertGreater(x[-1], values[:, 0].max())
            self.assertLess(x[0], values[:, 0].min())
            self.assertGreater(y[-1], values[:, 1].max())
            self.assertLess(y[0], values[:, 1].min())
            assert_array_equal(values, before[key])

    def test_excessive_extent_stops_instead_of_clipping_or_coarsening(self):
        aligned = {'a': np.array([[-60., -1.], [60., 1.]])}
        before = aligned['a'].copy()
        with self.assertRaisesRegex(ValueError, 'exceed 512'):
            comparison.common_grid(aligned, .5)
        assert_array_equal(aligned['a'], before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
