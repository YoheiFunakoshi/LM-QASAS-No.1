"""Synthetic tests only: no real sequence, model or patient input is read."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import lmqasas
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_final_three_comparisons.py'
if not script.is_file():
    script = Path(__file__).with_name('compute_final_three_comparisons.py')
sys.path.insert(0, str(Path(lmqasas.__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(script.parent))
spec = importlib.util.spec_from_file_location('final_three', script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ScientificChecks(unittest.TestCase):
    def test_pca_uses_all_clones_mean_and_minimal_variance_dimension(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(60, 8)) * np.arange(1, 9) + 15
        indices = np.r_[np.arange(60), np.zeros(50, dtype=int)]
        before = raw.copy()
        reps, model, info = m.preprocess(raw, indices)
        fitted = raw.astype(np.float32)[indices].astype(float)
        assert_allclose(model['mean'], fitted.mean(axis=0), atol=1e-12)
        self.assertGreater(np.linalg.norm(model['mean'] - raw.astype(np.float32).mean(axis=0)), .1)
        assert_allclose(model['components'] @ model['components'].T, np.eye(8), atol=1e-12)
        d = info['pca_retained_components']
        self.assertGreaterEqual(info['retained_variance_ratio'], .95)
        self.assertLess(info['previous_cumulative_variance'], .95)
        centered = raw.astype(np.float32).astype(float) - model['mean']
        assert_array_equal(reps['centered'], centered.astype(np.float32))
        assert_array_equal(reps['pca95'], (centered @ model['components'][:d].T).astype(np.float32))
        assert_array_equal(raw, before)

    def test_invalid_cosine_vectors_and_incomplete_sequence_index_rejected(self):
        for values in (np.zeros((4, 3)), np.full((4, 3), np.nan), np.ones(4)):
            with self.assertRaises(ValueError):
                m.validate_vectors(values)
        with self.assertRaises(ValueError):
            m.preprocess(np.arange(12).reshape(4, 3)+1, np.array([0, 1, 2]))

    def test_inverse_mapping_retains_id_distinctness_and_all_observations(self):
        coords = np.array([[2., 3.], [2., 3.], [7., 9.]])
        indices = np.array([2, 0, 2, 1, 0])
        assert_array_equal(m.expand_unique(coords, indices, 3), coords[indices])
        with self.assertRaises(ValueError):
            m.expand_unique(coords, np.array([3]), 3)

    def test_grid_resolves_narrow_rotated_kernel_without_clipping(self):
        angle = .6
        q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        H = q @ np.diag([.2**2, 2.**2]) @ q.T
        xy = np.array([[-3., 2.], [5., 8.]])
        x, y, info = m.shared_grid([xy], [H, np.eye(2)])
        self.assertLessEqual(max(np.diff(x).max(), np.diff(y).max()), .1+1e-12)
        self.assertLessEqual(x[0], -9)
        self.assertGreaterEqual(y[-1], 14)
        self.assertGreaterEqual(info['grid_size'], 96)

    def test_grid_limits_stop_instead_of_clipping(self):
        with self.assertRaises(ValueError):
            m.shared_grid([np.array([[0., 0.], [100., 100.]])], [np.eye(2)*1e-8])
        with self.assertRaises(ValueError):
            m.shared_grid([np.array([[0., 0.], [1., 1.]])], [np.diag([0., 1.])])


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='final_three_synthetic_', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.run_folder, self.pooling = [self.root / n for n in ('inputs', 'run', 'pooling')]
        for folder in (self.source, self.run_folder, self.pooling):
            folder.mkdir()
        self.original = self.source/'synthetic.txt'
        self.original.write_text('Invented data; preserve', encoding='utf-8')
        projection = self.run_folder/'projection'
        projection.mkdir()
        self.values = np.random.default_rng(13).normal(size=(60, 480)) + 5
        self.indices = np.tile(np.arange(60), 3)
        self.clones = [{'embedding_index': int(index), 'timepoint': m.PHASES[i//60]}
                       for i,index in enumerate(self.indices)]
        self.settings = {'n_neighbors': 15, 'min_dist': .1, 'n_epochs': 200, 'init': 'random', 'metric': 'cosine',
                         'unique': False, 'random_state': 7, 'transform_seed': 7, 'n_components': 2}
        self.pm = {'parameters': self.settings, 'packages': {}, 'kde': {'bandwidth': 1.}}
        self.hashes = {'clones.json': 'synthetic', 'embeddings.npy': 'synthetic'}
        m.save_array(self.pooling/'all_embeddings.npy', self.values)
        prior_fits = {}
        for seed in (7, 11):
            coords = np.random.default_rng(seed).normal(size=(180, 2)) * 3
            name = f'all_seed{seed}_raw.npy'
            m.save_array(self.pooling/name, coords)
            prior_fits[f'all_seed{seed}'] = {'parameters': self.settings | {'random_state': seed, 'transform_seed': seed},
                'source_clone_sha256': 'synthetic', 'embedding_sha256': 'synthetic', 'raw_coordinates_file': name}
        prior = {'status': 'completed', 'source_sha256': self.hashes, 'projection_sha256': {}, 'packages': {},
                 'fits': prior_fits, 'artifact_sha256': {p.name: m.digest(p) for p in self.pooling.iterdir()}}
        m.write_json(self.pooling/'comparison_metadata.json', prior)
        run = {'parameters': {'threads': 1}, 'input_hashes': {}, 'input_policies': {}}
        self.context = (self.run_folder, run, self.clones, ['A'*(5+i%10) for i in range(60)], self.values,
                        self.hashes, projection, self.pm, {}, {phase: self.original for phase in m.PHASES})
        self.before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def call(self, output):
        m.compute(self.run_folder, self.context[6], self.pooling, output, 11)

    def test_three_families_six_new_fits_and_exact_unique_inverse_mapping(self):
        captured = []
        def fake_fit(values, settings):
            captured.append((values.copy(), dict(settings)))
            return np.random.default_rng(settings['random_state']).normal(size=(len(values), 2)).astype(np.float32)*3, {
                'warnings': [], 'solver_fallback_to_random': False}
        output = self.root/'comparison'
        with ExitStack() as stack:
            stack.enter_context(patch.object(m, 'load_context', return_value=self.context))
            stack.enter_context(patch.object(m, 'audited_fit', side_effect=fake_fit))
            stack.enter_context(patch.object(m, 'neighborhood_diagnostics', return_value={'synthetic': True}))
            check = stack.enter_context(patch.object(m, 'verify_unchanged'))
            stack.enter_context(patch.object(m, 'code_provenance', return_value={'synthetic': True}))
            self.call(output)
        info = m.read(output/'comparison_metadata.json')
        self.assertEqual(len(captured), 6)
        self.assertEqual([len(v) for v,s in captured], [180,180,180,180,60,60])
        self.assertEqual(len(info['fits']), 8)
        self.assertEqual(len(info['densities']), 14)
        self.assertEqual(set(info['families']), set(m.FAMILIES))
        self.assertTrue(info['stop_after_declared_comparisons'])
        self.assertFalse(info['production_defaults_changed'])
        self.assertFalse(info['candidate_selection_performed'])
        for values, settings in captured:
            self.assertEqual(settings, self.settings | {'random_state': settings['random_state'], 'transform_seed': settings['random_state']})
        for seed in (7, 11):
            fit = info['fits'][f'unique_seed{seed}']
            unique = np.load(output/fit['unique_coordinates_file'])
            assert_array_equal(np.load(output/fit['raw_coordinates_file']), unique[self.indices])
            self.assertEqual(fit['output_observation_count'], 180)
        check.assert_called_once_with(self.context)
        for path, content in self.before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_existing_and_source_output_rejected_before_fit(self):
        with patch.object(m, 'load_context', return_value=self.context), patch.object(m, 'audited_fit') as fit:
            for destination in (self.pooling/'new', self.run_folder/'new', self.source/'new', self.root):
                with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                    self.call(destination)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
