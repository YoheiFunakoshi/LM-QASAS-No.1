"""Synthetic orchestration checks for the min_dist comparison.

These tests use invented vectors and coordinates. UMAP is replaced by a test
double; diagnostic arithmetic already tested in test_neighbors_comparison is
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

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_mindist_comparison.py'
if not script.is_file():
    script = Path(__file__).with_name('compute_mindist_comparison.py')
sys.path.insert(0, str(Path(lmqasas.__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('mindist_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mindist_synthetic_', dir=Path(__file__).parent)
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
        fitter = stack.enter_context(patch.object(comparison, '_fit_umap',
            side_effect=fit or (lambda vectors, settings: self.coordinates + settings['min_dist'])))
        diagnostics = stack.enter_context(patch.object(comparison, 'neighborhood_diagnostics',
            side_effect=diagnostic, return_value={'synthetic': True}))
        source_check = stack.enter_context(patch.object(comparison, 'verify_unchanged'))
        stack.enter_context(patch.object(comparison, 'code_provenance', return_value={'synthetic': True}))
        return fitter, diagnostics, source_check

    def test_existing_and_source_or_prior_subfolders_rejected_before_umap(self):
        existing = self.root / 'existing'
        existing.mkdir()
        with patch.object(comparison, 'load_context', return_value=self.context), \
             patch.object(comparison, '_fit_umap') as fit:
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
             patch.object(comparison, '_fit_umap') as fit:
            for seed in (7, -1, 2**32, True, np.int64(11)):
                with self.subTest(seed=repr(seed)), self.assertRaises(ValueError):
                    self.call_compute(output, seed)
            for key, value in (('n_neighbors', 30), ('min_dist', 0.3)):
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
                 patch.object(comparison, '_fit_umap') as fit, self.assertRaises(ValueError):
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
                 patch.object(comparison, '_fit_umap') as fit, self.assertRaises(ValueError):
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
            scale = np.array([1 + settings['min_dist'], 1 + settings['random_state'] / 30], dtype=np.float32)
            return self.coordinates * scale

        with ExitStack() as stack:
            _, diagnostics, source_check = self.patch_workflow(stack, fit=fake_umap)
            density = stack.enter_context(patch.object(comparison, 'fixed_density', wraps=comparison.fixed_density))
            self.call_compute(output)
        self.assertEqual(len(captured), 4)
        self.assertEqual({(s['min_dist'], s['random_state']) for _, s in captured},
                         {(0.3, 7), (0.3, 11), (0.5, 7), (0.5, 11)})
        for vectors, settings in captured:
            self.assertEqual(vectors.dtype, np.dtype('float32'))
            assert_array_equal(vectors, self.cached[self.indices].astype(np.float32))
            self.assertEqual(settings, self.parameters | {'min_dist': settings['min_dist'],
                'random_state': settings['random_state'], 'transform_seed': settings['random_state']})
        diagnostics.assert_called_once()
        assert_array_equal(diagnostics.call_args.args[2], self.indices)
        self.assertEqual(len(diagnostics.call_args.args[3]), 6)
        source_check.assert_called_once_with(self.context)
        self.assertEqual(density.call_count, 6)
        self.assertEqual({call.args[-1] for call in density.call_args_list}, {0.5})
        result = comparison.read(output / 'comparison_metadata.json')
        plan = comparison.read(output / 'experiment_plan.json')
        self.assertEqual(plan['min_dists'], [0.1, 0.3, 0.5])
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
            assert_array_equal(np.load(output / f'md0.1_seed{seed}_raw.npy', allow_pickle=False),
                               self.coordinates + (seed - 7))
            self.assertTrue(result['fits'][f'md0.1_seed{seed}']['reused_previous_fit'])
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
