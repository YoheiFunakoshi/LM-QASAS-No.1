"""Independent synthetic checks; no model inference or research-data access."""
from pathlib import Path
import importlib.util
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import pdist

script = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_pooling_comparison.py'
if not script.is_file():
    # Allow independent review of a staged script before installation in the repo.
    script = Path(__file__).with_name('compute_pooling_comparison.py')
spec = importlib.util.spec_from_file_location('pooling_comparison', script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class PoolingChecks(unittest.TestCase):
    def test_specials_excluded_and_float64_used_before_averaging(self):
        # Three residues whose float32 mean would lose the middle +1.
        rows = np.array([12, 1e8, 1, -1e8, 24, 48], dtype=np.float32)
        state = np.repeat(rows[:, None], 480, axis=1)
        before = state.copy()
        all_mean, residues = comparison.pool_states(state, 3)
        self.assertEqual(all_mean.dtype, np.dtype('float64'))
        self.assertEqual(residues.dtype, np.dtype('float64'))
        assert_allclose(residues, 1 / 3, rtol=0, atol=1e-15)
        assert_allclose(all_mean, 85 / 6, rtol=0, atol=1e-15)
        assert_array_equal(state, before)

    def test_incorrect_length_or_nonfinite_state_rejected(self):
        state = np.zeros((7, 480), dtype=np.float32)
        for length in (3, 0, True, np.int64(4)):
            with self.subTest(length=repr(length)), self.assertRaises(ValueError):
                comparison.pool_states(state, length)
        state[1, 0] = np.nan
        with self.assertRaises(ValueError):
            comparison.pool_states(state, 4)


class AlignmentChecks(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([[0, 0], [3, 0], [0, 2], [1, 4], [-2, 1]], dtype=float)

    def test_translation_rotation_and_reflection_preserve_distances(self):
        # Rotation composed with a reflection, plus a translation.
        transform = np.array([[0.6, 0.8], [0.8, -0.6]])
        moved = self.reference @ transform + [8.0, -12.0]
        original = moved.copy()
        aligned, record = comparison.rigid_align(moved, self.reference)
        assert_allclose(aligned, self.reference, rtol=0, atol=1e-12)
        assert_allclose(pdist(aligned), pdist(moved), rtol=0, atol=1e-12)
        self.assertEqual(record['scale'], 1.0)
        self.assertTrue(record['reflection'])
        assert_array_equal(moved, original)

    def test_alignment_does_not_remove_scale_difference(self):
        larger = 2.5 * self.reference + [3.0, 7.0]
        aligned, record = comparison.rigid_align(larger, self.reference)
        assert_allclose(pdist(aligned), 2.5 * pdist(self.reference), atol=1e-12)
        self.assertGreater(record['rms_displacement'], 1.0)
        self.assertEqual(record['scale'], 1.0)


class DensityChecks(unittest.TestCase):
    def test_each_phase_integrates_to_one_and_clone_rows_have_equal_weight(self):
        group = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.75]])
        coordinates = np.vstack([group, group, group])
        phases = np.repeat(comparison.PHASES, len(group))
        x = np.linspace(-4.0, 4.0, 51)
        y = np.linspace(-4.0, 4.0, 49)
        density, integrals = comparison.fixed_density(coordinates, phases, x, y, 0.6)
        self.assertEqual(density.shape, (3, 49, 51))
        self.assertTrue(np.all(density >= 0))
        assert_allclose(np.trapezoid(np.trapezoid(density, x=x, axis=2), x=y, axis=1),
                        np.ones(3), rtol=0, atol=1e-12)
        assert_allclose(density[0], density[1], rtol=0, atol=1e-14)
        assert_allclose(density[1], density[2], rtol=0, atol=1e-14)
        self.assertTrue(all(0.999 < v < 1.001 for v in integrals))

    def test_missing_phase_rejected(self):
        x = np.linspace(-2.0, 2.0, 9)
        with self.assertRaises(ValueError):
            comparison.fixed_density(np.zeros((2, 2)), ['Pre', 'Peak'], x, x, 0.5)


class PreservationChecks(unittest.TestCase):
    def test_array_save_refuses_overwrite_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(prefix='pooling_synthetic_', dir=Path(__file__).parent) as tmp:
            target = Path(tmp) / 'output.npy'
            comparison.save_array(target, np.array([1.0, 2.0]))
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                comparison.save_array(target, np.array([-99.0]))
            self.assertEqual(target.read_bytes(), original)

    def test_existing_and_source_output_directories_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory(prefix='pooling_synthetic_', dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            source_file = source / 'input.csv'
            source_file.write_text('synthetic placeholder', encoding='utf-8')
            existing = root / 'existing'
            existing.mkdir()
            sentinel = existing / 'untouched.txt'
            sentinel.write_bytes(b'preserve me')
            baseline = root / 'baseline'
            baseline.mkdir()
            context = (baseline, {}, [], [], np.empty((0, 480)), {}, baseline / 'projection', {}, {},
                       {phase: source_file for phase in comparison.PHASES})
            # The rejection should precede any model use; skip heavyweight imports.
            fake_torch = types.ModuleType('torch')
            fake_ablang = types.ModuleType('ablang2')
            fake_pretrained = types.ModuleType('ablang2.pretrained')
            fake_pretrained.format_seq_input = lambda *args, **kwargs: self.fail('Tokenizer called')
            with patch.dict('sys.modules', {'torch': fake_torch, 'ablang2': fake_ablang,
                                           'ablang2.pretrained': fake_pretrained}), \
                 patch.object(comparison, 'LocalAbLang2') as model:
                for output, error in ((existing, FileExistsError),
                                      (source / 'result', ValueError),
                                      (baseline / 'result', ValueError)):
                    with self.subTest(output=output.name), self.assertRaises(error):
                        comparison.compute_embeddings(context, root / 'model', output)
                model.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'preserve me')
            self.assertEqual(source_file.read_text(encoding='utf-8'), 'synthetic placeholder')
            self.assertFalse((source / 'result').exists())
            self.assertFalse((baseline / 'result').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
