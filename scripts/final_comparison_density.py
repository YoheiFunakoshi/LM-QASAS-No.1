"""Pooled Scott full-covariance Gaussian KDE for a private sensitivity analysis.

Scott's factor and covariance scaling follow scipy.stats.gaussian_kde's documented
rule. A single covariance is estimated from all pooled clone coordinates and is
then shared by the three phases. This is not three separately fitted SciPy KDEs.
Both anisotropy and the bandwidth rule change relative to a fixed isotropic KDE.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.neighbors import KernelDensity

PHASES = ('Pre', 'Peak', 'Post')
ENGINE_SETTINGS = {'kernel': 'gaussian', 'algorithm': 'ball_tree',
                   'bandwidth_in_whitened_space': 1.0, 'atol': 0.0, 'rtol': 1e-6,
                   'counts_weights': False, 'dtype': 'float64'}


def _coordinates(coords):
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite, nonempty two-dimensional coordinates are required.')
    return values


def _cholesky(covariance):
    matrix = np.asarray(covariance, dtype=np.float64)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)):
        raise ValueError('Kernel covariance must be a finite symmetric 2x2 matrix.')
    try:
        lower = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError('Kernel covariance must be positive definite; no automatic regularization.') from error
    return matrix, lower


def pooled_scott_covariance(coords):
    """Return H = N**(-1/3) * sample_covariance(all clone rows), and provenance."""
    values = _coordinates(coords)
    if len(values) < 3:
        raise ValueError('At least three pooled observations are required in two dimensions.')
    sample_covariance = np.cov(values, rowvar=False, ddof=1)
    factor = float(len(values) ** (-1.0 / 6.0))
    covariance, lower = _cholesky(sample_covariance * factor**2)
    metadata = {
        'method': 'pooled_scott_full_covariance', 'pooled_observation_count': len(values),
        'dimensions': 2, 'sample_covariance_ddof': 1, 'scott_factor': factor,
        'factor_formula': 'N**(-1/(d+4))', 'kernel_covariance_formula': 'factor**2 * pooled_sample_covariance',
        'sample_covariance': sample_covariance.tolist(), 'kernel_covariance': covariance.tolist(),
        'kernel_covariance_eigenvalues': np.linalg.eigvalsh(covariance).tolist(),
        'cholesky_lower': lower.tolist(), 'engine': dict(ENGINE_SETTINGS),
        'shared_across_phases': True, 'separate_per_phase_bandwidth_fit': False,
        'normalization': 'per-phase integral one on shared finite grid',
        'interpretation': 'Changes both anisotropy and bandwidth rule; not a bandwidth-matched shape-only comparison.',
        'reference': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html',
    }
    return covariance, metadata


def covariance_density(coords, phases, x, y, covariance):
    """Evaluate an equally weighted per-phase KDE with one shared positive-definite H."""
    values = _coordinates(coords)
    labels = np.asarray(phases)
    if (labels.shape != (len(values),) or set(labels.tolist()) != set(PHASES)):
        raise ValueError('Every coordinate needs one phase label, with all three declared phases present.')
    axes = [np.asarray(axis, dtype=np.float64) for axis in (x, y)]
    if any(axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all()
           or np.any(np.diff(axis) <= 0) for axis in axes):
        raise ValueError('Grid axes must be finite, strictly increasing vectors of at least three points.')
    xx, yy = np.meshgrid(*axes)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    _, lower = _cholesky(covariance)
    whitened = solve_triangular(lower, values.T, lower=True, check_finite=False).T
    whitened_query = solve_triangular(lower, query.T, lower=True, check_finite=False).T
    log_jacobian = float(np.log(np.diag(lower)).sum())
    grids, integrals = [], []
    for phase in PHASES:
        estimator = KernelDensity(kernel='gaussian', bandwidth=1.0, algorithm='ball_tree',
                                  atol=ENGINE_SETTINGS['atol'], rtol=ENGINE_SETTINGS['rtol'])
        estimator.fit(whitened[labels == phase])
        density = np.exp(estimator.score_samples(whitened_query) - log_jacobian).reshape(xx.shape)
        integral = float(np.trapezoid(np.trapezoid(density, x=axes[0], axis=1), x=axes[1]))
        if not np.isfinite(density).all() or not np.isfinite(integral) or integral <= 0:
            raise ValueError('Density and finite-grid integral must be finite and the integral positive.')
        grids.append(density / integral)
        integrals.append(integral)
    return np.asarray(grids), np.asarray(integrals)
