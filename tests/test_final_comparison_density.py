"""Synthetic-only checks; no research data or models are opened."""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts if scripts.is_dir() else Path(__file__).resolve().parent))
import final_comparison_density as density


class CovarianceChecks(unittest.TestCase):
    def test_scott_covariance_of_rectangle_and_translation(self):
        coords = np.array([[-2., -1.], [2., -1.], [-2., 1.], [2., 1.]])
        before = coords.copy()
        covariance, metadata = density.pooled_scott_covariance(coords)
        expected = np.diag([16/3, 4/3]) * 4**(-1/3)
        assert_allclose(covariance, expected, rtol=1e-14, atol=1e-14)
        shifted, _ = density.pooled_scott_covariance(coords + [23., -10.])
        assert_allclose(shifted, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(metadata['sample_covariance_ddof'], 1)
        self.assertEqual(metadata['pooled_observation_count'], 4)
        self.assertTrue(metadata['shared_across_phases'])
        self.assertFalse(metadata['separate_per_phase_bandwidth_fit'])
        assert_array_equal(coords, before)

    def test_nonfinite_or_singular_pooled_data_rejected_without_regularizing(self):
        for coords in (np.array([[1., 2.], [3., 4.]]), np.zeros((4, 2)),
                       np.array([[0., 0.], [1., 1.], [2., 2.]]), np.full((4, 2), np.nan)):
            with self.subTest(coords=coords.shape), self.assertRaises(ValueError):
                density.pooled_scott_covariance(coords)

    def test_correlated_gaussian_kernel_matches_independent_scipy_formula(self):
        coords = np.array([[-1., 0.], [0., 1.], [2., -1.], [1., 2.], [-2., -1.], [-.5, 1.5]])
        phases = np.repeat(density.PHASES, 2)
        covariance = np.array([[1.4, .45], [.45, .7]])
        x, y = np.linspace(-7, 7, 61), np.linspace(-6, 6, 51)
        values, integrals = density.covariance_density(coords, phases, x, y, covariance)
        xx, yy = np.meshgrid(x, y)
        query = np.stack((xx, yy), axis=-1)
        for i, phase in enumerate(density.PHASES):
            exact = np.mean([multivariate_normal(mean=point, cov=covariance).pdf(query)
                             for point in coords[phases == phase]], axis=0)
            integral = np.trapezoid(np.trapezoid(exact, x=x, axis=1), x=y)
            assert_allclose(values[i], exact/integral, rtol=2e-6, atol=1e-12)
            self.assertAlmostEqual(integrals[i], integral, places=12)
        assert_allclose(np.trapezoid(np.trapezoid(values, x=x, axis=2), x=y, axis=1), 1., atol=1e-14)

    def test_isotropic_covariance_matches_direct_standard_gaussian(self):
        coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
        x = np.linspace(-3, 3, 31)
        y = np.linspace(-3, 3, 31)
        h = .8
        values, integrals = density.covariance_density(coords, density.PHASES, x, y, np.eye(2)*h*h)
        xx, yy = np.meshgrid(x, y)
        for i, point in enumerate(coords):
            exact = np.exp(-((xx-point[0])**2+(yy-point[1])**2)/(2*h*h))/(2*np.pi*h*h)
            assert_allclose(values[i]*integrals[i], exact, rtol=1e-12, atol=1e-14)

    def test_invalid_covariance_labels_and_grid_fail(self):
        coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
        grid = np.linspace(-3, 3, 11)
        for covariance in (np.zeros((2, 2)), np.array([[1., .3], [0., 1.]]),
                           np.diag([-1., 1.]), np.full((2, 2), np.nan)):
            with self.subTest(covariance=covariance.tolist()), self.assertRaises(ValueError):
                density.covariance_density(coords, density.PHASES, grid, grid, covariance)
        for labels in (['Pre']*3, ['Pre', 'Peak', 'Unknown'], ['Pre', 'Peak']):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                density.covariance_density(coords, labels, grid, grid, np.eye(2))
        with self.assertRaises(ValueError):
            density.covariance_density(coords, density.PHASES, grid[::-1], grid, np.eye(2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
