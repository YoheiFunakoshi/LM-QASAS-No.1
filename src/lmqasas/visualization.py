"""Shared UMAP views of saved runs; visualization never changes candidate scores.

Every clone observation is passed to UMAP, including clones sharing one CDR-H3.
The KDE is a per-timepoint probability density, normalized on one finite grid.
All defaults here are provisional display choices, not recovered paper settings.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from time import perf_counter
import uuid

import numpy as np

from .checkpoint import digest
from .pipeline import (PHASES, code_provenance, protect_output, utc_stamp,
                       validate_output_location, write_json)
from .selection import CANONICAL_AMINO_ACIDS, select_unique_cdr3

SOURCE_FILES = ('embeddings.npy', 'embedding_sequences.json', 'clones.json', 'scored_peak.json')
SELECTION_FILES = ('candidates.csv', 'candidate_provenance.json', 'selection_summary.json')
GRID_SIZE = 96


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _checked_hash(folder: Path, name: str, expected: str | None) -> str:
    path = (folder / name).resolve(strict=True)
    if folder not in path.parents or not path.is_file():
        raise ValueError('Artifact must be a file inside its result folder.')
    actual = digest(path)
    if not expected or actual != expected:
        raise ValueError('Stored artifact failed its integrity check.')
    return actual


def _load_run(run_dir: Path, *, embeddings: bool):
    folder = Path(run_dir).resolve(strict=True)
    metadata_path = folder / 'run_metadata.json'
    metadata_hash = digest(metadata_path)
    metadata = _json(metadata_path)
    if metadata.get('status') != 'completed':
        raise ValueError('Only completed runs can be visualized.')
    if not all(metadata.get('inputs_unchanged_after_run', {}).get(p) is True for p in PHASES):
        raise ValueError('The saved run does not confirm unchanged input files.')
    validate_output_location(folder, {key: Path(value) for key, value in metadata['input_paths'].items()})
    hashes = {'run_metadata.json': metadata_hash}
    for name in SOURCE_FILES:
        hashes[name] = _checked_hash(folder, name, metadata.get('artifact_sha256', {}).get(name))
    sequences = _json(folder / 'embedding_sequences.json')
    clones = _json(folder / 'clones.json')
    if (not isinstance(sequences, list) or not sequences or
            any(not isinstance(s, str) or not s or not set(s) <= CANONICAL_AMINO_ACIDS for s in sequences)
            or len(set(sequences)) != len(sequences)):
        raise ValueError('Invalid embedding sequence index.')
    if not isinstance(clones, list) or not clones:
        raise ValueError('The run has no clone observations.')
    for index, clone in enumerate(clones):
        if (not isinstance(clone, dict) or type(clone.get('clone_id')) is not int or
                clone['clone_id'] != index or clone.get('timepoint') not in PHASES or
                type(clone.get('embedding_index')) is not int or
                not 0 <= clone['embedding_index'] < len(sequences) or
                clone.get('cdr3') != sequences[clone['embedding_index']]):
            raise ValueError('Clone order or embedding correspondence is invalid.')
    if {clone['timepoint'] for clone in clones} != set(PHASES):
        raise ValueError('All three timepoints must have clone observations.')
    vectors = None
    if embeddings:
        cached = np.load(folder / 'embeddings.npy', allow_pickle=False)
        if cached.shape != (len(sequences), 480) or not np.issubdtype(cached.dtype, np.floating):
            raise ValueError('Expected floating-point 480-dimensional embeddings.')
        if not np.isfinite(cached).all():
            raise ValueError('Embeddings must be finite.')
        vectors = np.asarray(cached[[c['embedding_index'] for c in clones]], dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError('Embeddings cannot be represented as finite float32 values.')
    _verify_sources(folder, hashes)
    return folder, metadata, clones, vectors, hashes


def _verify_sources(folder: Path, hashes: dict) -> None:
    for name, expected in hashes.items():
        _checked_hash(folder, name, expected)


def _fit_umap(vectors: np.ndarray, settings: dict) -> np.ndarray:
    # Import only when making a new projection. Re-rendering does not load UMAP.
    import umap
    model = umap.UMAP(**settings)
    return np.asarray(model.fit_transform(vectors), dtype=np.float32)


def _density_grid(coordinates: np.ndarray, phases: list[str], progress=None):
    from sklearn.neighbors import KernelDensity
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError('Projection coordinates must be finite two-dimensional points.')
    pooled_std = float(np.sqrt(np.mean(np.var(coordinates, axis=0, ddof=1))))
    bandwidth = pooled_std * len(coordinates) ** (-1 / 6)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('A non-degenerate projection is required for density display.')
    span = np.ptp(coordinates, axis=0)
    margin = np.maximum(3 * bandwidth, .05 * span)
    lower = coordinates.min(axis=0) - margin
    upper = coordinates.max(axis=0) + margin
    x = np.linspace(lower[0], upper[0], GRID_SIZE)
    y = np.linspace(lower[1], upper[1], GRID_SIZE)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    phase_array = np.asarray(phases)
    densities = []
    integrals_before = []
    for index, phase in enumerate(PHASES):
        group = coordinates[phase_array == phase]
        if not len(group):
            raise ValueError('All timepoints need points for density display.')
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian', algorithm='ball_tree',
                            atol=0, rtol=1e-6).fit(group)
        values = np.exp(kde.score_samples(points)).reshape(GRID_SIZE, GRID_SIZE)
        integral = float(np.trapezoid(np.trapezoid(values, x=x, axis=1), x=y))
        if not math.isfinite(integral) or integral <= 0:
            raise ValueError('Could not normalize the density grid.')
        densities.append(values / integral)
        integrals_before.append(integral)
        if progress:
            progress('density', (index + 1) / len(PHASES))
    settings = {'kernel': 'gaussian', 'algorithm': 'ball_tree', 'atol': 0, 'rtol': 1e-6,
                'bandwidth': bandwidth, 'bandwidth_rule': 'sqrt(mean(pooled_axis_sample_variances))*n^(-1/6)',
                'grid_size': GRID_SIZE, 'grid_margin_rule': 'max(3*bandwidth, .05*axis_span)',
                'normalization': 'each_timepoint_trapezoid_integral_on_shared_finite_grid_equals_one',
                'integrals_before_grid_normalization': dict(zip(PHASES, integrals_before)),
                'weighting': 'equal_clone_observations_no_Counts_weights',
                'axis_limits': {'x': [float(x[0]), float(x[-1])], 'y': [float(y[0]), float(y[-1])]}}
    return x, y, np.asarray(densities), settings


def create_projection(run_dir: Path, *, n_neighbors=15, min_dist=.1, metric='cosine',
                      seed=20260919, n_epochs=200, progress=None) -> Path:
    """Make one immutable pooled-clone UMAP projection and common KDE display grid."""
    if type(n_neighbors) is not int or n_neighbors < 2:
        raise ValueError('n_neighbors must be an integer of at least two.')
    if isinstance(min_dist, bool) or not isinstance(min_dist, (int, float)) or not 0 <= min_dist <= 1:
        raise ValueError('min_dist must be a finite number between zero and one.')
    if metric != 'cosine':
        raise ValueError('This display version supports the cosine metric only.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32).')
    if type(n_epochs) is not int or n_epochs < 10:
        raise ValueError('n_epochs must be an integer of at least ten.')
    folder, run, clones, vectors, source_hashes = _load_run(run_dir, embeddings=True)
    if len(clones) < 4 or n_neighbors >= len(clones):
        raise ValueError('Choose n_neighbors explicitly below the observation count (at least four observations).')
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError('Cosine UMAP requires nonzero embedding vectors.')
    settings = {'n_neighbors': n_neighbors, 'n_components': 2, 'metric': metric,
                'min_dist': float(min_dist), 'spread': 1.0, 'n_epochs': n_epochs,
                'random_state': seed, 'transform_seed': seed, 'n_jobs': 1,
                'low_memory': True, 'force_approximation_algorithm': len(clones) > 4096,
                'init': 'random', 'unique': False, 'densmap': False, 'verbose': False}
    target = folder / ('projection_' + utc_stamp() + '_' + uuid.uuid4().hex[:8])
    target.mkdir(exist_ok=False)
    protect_output(target)
    started = perf_counter()
    metadata = {'schema_version': 1, 'status': 'running', 'source_run': str(folder),
                'source_sha256': source_hashes, 'code': code_provenance(),
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': platform.python_version(),
                'packages': {n: importlib.metadata.version(n) for n in
                             ('numpy', 'scipy', 'scikit-learn', 'umap-learn', 'numba', 'pynndescent')},
                'parameters': settings, 'observations': len(clones),
                'input_dimensions': 480, 'input_dtype': 'float32',
                'point_order': 'clones.json array order; clone_id equals row index',
                'observation_unit': 'one_unique_V_J_CDR3_isotype_clone_per_timepoint',
                'timepoint_counts': {p: sum(c['timepoint'] == p for c in clones) for p in PHASES},
                'downsampling': False, 'sequence_deduplication_of_observations': False,
                'purpose': 'visualization_only_not_candidate_scoring',
                'paper_equivalence': 'unconfirmed_provisional_visualization_defaults',
                'outputs_private': True}
    try:
        if progress:
            progress('umap', 0.0)
        coordinates = _fit_umap(vectors, settings)
        if coordinates.shape != (len(clones), 2) or not np.isfinite(coordinates).all():
            raise ValueError('UMAP did not return a finite point for every clone.')
        if progress:
            progress('umap', 1.0)
        with (target / 'coordinates.npy').open('xb') as handle:
            np.save(handle, coordinates, allow_pickle=False)
        x, y, density, kde = _density_grid(coordinates, [c['timepoint'] for c in clones], progress)
        with (target / 'density_grid.npz').open('xb') as handle:
            np.savez_compressed(handle, x=x, y=y, densities=density)
        metadata['kde'] = kde
        metadata['coordinates_content_sha256'] = hashlib.sha256(coordinates.tobytes(order='C')).hexdigest()
        metadata['coordinates_dtype'] = str(coordinates.dtype)
        metadata['artifact_sha256'] = {n: digest(target / n) for n in ('coordinates.npy', 'density_grid.npz')}
        _verify_sources(folder, source_hashes)
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata['status'] = 'failed'
        metadata['error_type'] = type(exc).__name__
        raise
    finally:
        metadata['elapsed_seconds'] = perf_counter() - started
        write_json(target / 'projection_metadata.json', metadata)
    if progress:
        progress('projection_completed', 1.0)
    return target


def _load_selection(folder: Path, run: dict, selection_dir: Path | None):
    selection = (Path(selection_dir) if selection_dir else folder / 'selection_initial').resolve(strict=True)
    if selection.parent != folder:
        raise ValueError('Selection must belong to the same run.')
    files = list(SELECTION_FILES)
    if selection.name == 'selection_initial':
        for name in files:
            _checked_hash(folder, 'selection_initial/' + name,
                          run.get('artifact_sha256', {}).get('selection_initial/' + name))
    else:
        source = _json(selection / 'selection_source.json')
        if (Path(source.get('source_run', '')).resolve() != folder or
                source.get('source_score_sha256') != run['artifact_sha256']['scored_peak.json']):
            raise ValueError('Reselection provenance does not match this run.')
        files.append('selection_source.json')
    hashes = {name: digest(selection / name) for name in files}
    summary = _json(selection / 'selection_summary.json')
    records = _json(folder / 'scored_peak.json')
    result = select_unique_cdr3(records, summary.get('requested'))
    expected_summary = {'requested': result.requested, 'available': result.available,
                        'returned': result.returned, 'shortfall': result.shortfall,
                        'boundary_tie': asdict(result.boundary_tie) if result.boundary_tie else None,
                        'unit': 'distinct_CDR_H3_amino_acid_sequence',
                        'representative_score': 'max', 'tie_break': 'lexical_CDR3'}
    # Recompute the original selection instead of trusting unanchored reselect files.
    expected_provenance = json.loads(json.dumps([asdict(c) for c in result.candidates]))
    if summary != expected_summary or _json(selection / 'candidate_provenance.json') != expected_provenance:
        raise ValueError('Selection summary or provenance differs from the verified scores.')
    with (selection / 'candidates.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['rank', 'cdr3', 'score', 'clone_count', 'cluster_ids']:
            raise ValueError('Candidate CSV schema is invalid.')
        rows = list(reader)
    expected_rows = [{'rank': str(i), 'cdr3': c.cdr3, 'score': str(c.score),
                      'clone_count': str(len(c.clone_records)),
                      'cluster_ids': ';'.join(str(n) for n in sorted({r['cluster_id'] for r in c.clone_records}))}
                     for i, c in enumerate(result.candidates, 1)]
    if rows != expected_rows:
        raise ValueError('Candidate CSV differs from the verified selection.')
    _verify_sources(selection, hashes)
    return selection, summary, {c.cdr3 for c in result.candidates}, hashes


def render_selection(run_dir: Path, projection_dir: Path, selection_dir: Path | None = None) -> Path:
    """Render selected CDR-H3 occurrences at all timepoints without refitting UMAP."""
    folder, run, clones, _, source_hashes = _load_run(run_dir, embeddings=False)
    projection = Path(projection_dir).resolve(strict=True)
    if projection.parent != folder:
        raise ValueError('Projection must belong to the same run.')
    projection_hash = digest(projection / 'projection_metadata.json')
    metadata = _json(projection / 'projection_metadata.json')
    if metadata.get('status') != 'completed' or metadata.get('source_sha256') != source_hashes:
        raise ValueError('Projection is incomplete or belongs to different source artifacts.')
    for name in ('coordinates.npy', 'density_grid.npz'):
        _checked_hash(projection, name, metadata.get('artifact_sha256', {}).get(name))
    coordinates = np.load(projection / 'coordinates.npy', allow_pickle=False)
    if (coordinates.shape != (len(clones), 2) or not np.isfinite(coordinates).all() or
            hashlib.sha256(coordinates.tobytes(order='C')).hexdigest() != metadata.get('coordinates_content_sha256')):
        raise ValueError('Projection point correspondence failed verification.')
    with np.load(projection / 'density_grid.npz', allow_pickle=False) as grid:
        x, y, densities = (grid[name] for name in ('x', 'y', 'densities'))
    if (x.shape != (GRID_SIZE,) or y.shape != (GRID_SIZE,) or densities.shape != (3, GRID_SIZE, GRID_SIZE)
            or not all(np.isfinite(a).all() for a in (x, y, densities)) or np.any(densities < 0)
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError('Density grid is invalid.')
    if not np.allclose(np.trapezoid(np.trapezoid(densities, x=x, axis=2), x=y, axis=1), 1, atol=1e-8):
        raise ValueError('Density grids are not normalized on the shared axes.')
    selection, summary, selected, selection_hashes = _load_selection(folder, run, selection_dir)
    masks = {p: np.asarray([c['timepoint'] == p and c['cdr3'] in selected for c in clones]) for p in PHASES}
    target = projection / ('view_' + utc_stamp() + '_' + uuid.uuid4().hex[:8])
    target.mkdir(exist_ok=False)
    protect_output(target)
    view = {'schema_version': 1, 'status': 'running', 'source_run': str(folder),
            'projection_dir': str(projection), 'selection_dir': str(selection),
            'projection_id': projection.name, 'selection_id': selection.name,
            'candidate_count': summary['returned'],
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'code': code_provenance(),
            'source_sha256': source_hashes, 'projection_metadata_sha256': projection_hash,
            'projection_artifact_sha256': metadata['artifact_sha256'],
            'selection_artifact_sha256': selection_hashes, 'selection': summary,
            'selected_clone_ids': {p: np.flatnonzero(masks[p]).tolist() for p in PHASES},
            'selected_observation_counts': {p: int(masks[p].sum()) for p in PHASES},
            'selected_cdr3_type_counts': {p: len({c['cdr3'] for c in clones if c['timepoint'] == p and c['cdr3'] in selected}) for p in PHASES},
            'overlay_meaning': 'occurrences_at_each_timepoint_of_Peak_selected_CDR_H3_types_not_known_DB_matches',
            'projection_reused_without_refit': True, 'outputs_private': True,
            'display': {'colormap': 'viridis', 'candidate_color': '#ff941f', 'candidate_edge': '#2b1b08',
                        'density_limits': [0.0, float(densities.max())],
                        'x_limits': [float(x[0]), float(x[-1])], 'y_limits': [float(y[0]), float(y[-1])]},
            'packages': {'matplotlib': importlib.metadata.version('matplotlib')}}
    figure = None
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D
        figure = Figure(figsize=(15, 5.8), dpi=150, layout='constrained')
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3, sharex=True, sharey=True)
        levels = np.linspace(0, float(densities.max()), 21)
        for index, (phase, axis) in enumerate(zip(PHASES, axes)):
            contour = axis.contourf(x, y, densities[index], levels=levels, cmap='viridis', vmin=0, vmax=levels[-1])
            axis.contour(x, y, densities[index], levels=levels[1:-1:2], colors='#394a55', alpha=.35, linewidths=.35)
            points = coordinates[masks[phase]]
            axis.scatter(points[:, 0], points[:, 1], s=12, c='#ff941f', edgecolors='#2b1b08', linewidths=.3, alpha=.85)
            count = sum(c['timepoint'] == phase for c in clones)
            axis.set_title(f'{phase} | {count:,} clone observations\n{len(points):,} highlighted occurrences', fontsize=11)
            axis.set(xlabel='Shared UMAP 1', xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
            axis.set_aspect('equal', adjustable='box')
        axes[0].set_ylabel('Shared UMAP 2')
        figure.colorbar(contour, ax=axes, label='Normalized density per timepoint (shared scale)', shrink=.82)
        handle = Line2D([], [], marker='o', linestyle='', markersize=5, markerfacecolor='#ff941f', markeredgecolor='#2b1b08')
        figure.legend([handle], ['Peak-selected CDR-H3 occurrences'], loc='outside lower center', frameon=False)
        figure.suptitle(f'Shared UMAP of Pre / Peak / Post\nDistinct Peak-selected CDR-H3 types: {summary["returned"]:,} (requested {summary["requested"]:,})', fontsize=14)
        figure.savefig(target / 'comparison.png', dpi=150, facecolor='white', bbox_inches='tight', pad_inches=.12)
        _verify_sources(folder, source_hashes)
        _verify_sources(selection, selection_hashes)
        _checked_hash(projection, 'projection_metadata.json', projection_hash)
        for name, expected in metadata['artifact_sha256'].items():
            _checked_hash(projection, name, expected)
        view['artifact_sha256'] = {'comparison.png': digest(target / 'comparison.png')}
        view['status'] = 'completed'
    except BaseException as exc:
        view['status'] = 'failed'
        view['error_type'] = type(exc).__name__
        raise
    finally:
        if figure is not None:
            figure.clear()
        write_json(target / 'view_metadata.json', view)
    return target
