"""Three bounded offline comparisons; stop after the declared conditions.

Preprocessing, shared covariance KDE, and UMAP fit unit are independent families.
The original clone population and candidate scores are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np

from compute_pooling_comparison import (
    read, save_array, summary, rigid_align, fixed_density, load_context, verify_unchanged,
)
from compute_neighbors_comparison import exact_neighbors, sequence_centroids, DIAGNOSTIC_K
from compute_init_metric_comparison import audited_fit
from final_comparison_density import pooled_scott_covariance, covariance_density
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _verify_sources

TARGET_VARIANCE = .95
VARIANTS = ('raw', 'centered', 'pca95', 'unique')
FAMILIES = {
    'preprocessing': ('raw', 'centered', 'pca95'),
    'density': ('isotropic', 'pooled_scott'),
    'fit_unit': ('raw', 'unique'),
}


def validate_vectors(vectors):
    values = np.asarray(vectors)
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] < 1
            or not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0)):
        raise ValueError('Finite, nonzero vectors are required for cosine UMAP.')


def preprocess(cached, indices):
    """Full deterministic SVD once on all clone rows; select >=95% explicitly."""
    from sklearn.decomposition import PCA
    raw = np.asarray(cached, dtype=np.float32)
    validate_vectors(raw)
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or indices.min() < 0 or indices.max() >= len(raw)
            or len(np.unique(indices)) != len(raw)):
        raise ValueError('Every saved sequence must have a valid clone observation.')
    full = raw[indices].astype(np.float64)
    pca = PCA(n_components=None, svd_solver='full', whiten=False, copy=True).fit(full)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    if not np.isfinite(cumulative).all() or cumulative[-1] < TARGET_VARIANCE:
        raise ValueError('Invalid PCA variance spectrum.')
    dimensions = int(np.searchsorted(cumulative, TARGET_VARIANCE, side='left') + 1)
    centered = raw.astype(np.float64) - pca.mean_
    reduced = centered @ pca.components_[:dimensions].T
    representations = {'raw': raw, 'centered': centered.astype(np.float32),
                       'pca95': reduced.astype(np.float32)}
    for values in representations.values():
        validate_vectors(values)
    arrays = {'mean': pca.mean_, 'components': pca.components_,
              'explained_variance': pca.explained_variance_,
              'explained_variance_ratio': pca.explained_variance_ratio_,
              'singular_values': pca.singular_values_}
    info = {'solver': 'full', 'whiten': False, 'target_variance_ratio': TARGET_VARIANCE,
            'selection_rule': 'smallest dimension with cumulative explained variance >= 0.95; searchsorted left',
            'pca_retained_components': dimensions, 'retained_variance_ratio': float(cumulative[dimensions-1]),
            'previous_cumulative_variance': float(cumulative[dimensions-2]) if dimensions > 1 else 0.,
            'fit_unit': 'all clone observations pooled across the three phases, unweighted',
            'fit_observation_count': len(indices), 'fit_dtype': 'float32 source promoted to float64',
            'umap_dtype': 'float32', 'centering_control': 'same fitted clone-weighted mean, no dimension reduction',
            'candidate_embedding_changed': False}
    return representations, arrays, info


def expand_unique(coordinates, indices, n_sequences):
    coordinates, indices = np.asarray(coordinates), np.asarray(indices)
    if (coordinates.shape != (n_sequences, 2) or not np.isfinite(coordinates).all()
            or indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or indices.min() < 0 or indices.max() >= n_sequences):
        raise ValueError('Invalid unique-coordinate inverse map.')
    return coordinates[indices]


def shared_grid(coordinates, matrices):
    """One grid per family across all variants and both seeds; no clipping."""
    values = np.concatenate(coordinates).astype(float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('Invalid grid coordinates.')
    eigenvalues = []
    for matrix in matrices:
        matrix = np.asarray(matrix, float)
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
            raise ValueError('Invalid kernel matrix.')
        eigenvalues.extend(np.linalg.eigvalsh(matrix))
    if not eigenvalues or min(eigenvalues) <= 0:
        raise ValueError('Positive definite kernel matrices are required.')
    minimum, maximum = np.sqrt(min(eigenvalues)), np.sqrt(max(eigenvalues))
    margin = np.maximum(3 * maximum, .05 * np.ptp(values, axis=0))
    lower, upper = values.min(axis=0) - margin, values.max(axis=0) + margin
    size = max(96, 1 + int(np.ceil(max(upper - lower) / (minimum / 2))))
    if size > 512:
        raise ValueError('Grid exceeds 512; stop and inspect, do not clip or downsample.')
    x, y = [np.linspace(lower[i], upper[i], size) for i in (0, 1)]
    return x, y, {'grid_size': size, 'x_limits': x[[0, -1]].tolist(), 'y_limits': y[[0, -1]].tolist(),
                  'minimum_kernel_std': float(minimum), 'maximum_kernel_std': float(maximum),
                  'rule': 'common family grid; min 96; step <= minimum kernel std/2; margin >= 3 maximum std; max 512'}


def neighborhood_diagnostics(representations, sequences, indices, coordinates, output):
    lengths = np.asarray([len(s) for s in sequences])
    order = np.lexsort((np.arange(len(sequences)), lengths))
    query = order[np.unique(np.linspace(0, len(order)-1, min(512, len(order)), dtype=int))]
    arrays = {'query_ids': query, 'sequence_lengths': lengths}
    high = {}
    for name, values in representations.items():
        values = values.astype(np.float64)
        high[name] = {}
        for k in DIAGNOSTIC_K:
            neighbors, boundary, _ = exact_neighbors(values[query], values, query, 'cosine', k)
            high[name][k] = neighbors
            arrays[f'high_{name}_k{k}'] = neighbors
            arrays[f'high_{name}_boundary_k{k}'] = boundary
    high_changes = {}
    for name in ('centered', 'pca95'):
        high_changes[name] = {}
        for k in DIAGNOSTIC_K:
            fractions = np.asarray([len(set(a) & set(b))/k for a,b in zip(high['raw'][k], high[name][k], strict=True)])
            arrays[f'high_raw_vs_{name}_k{k}'] = fractions
            high_changes[name][str(k)] = summary(fractions)
    metrics = {}
    for key, values in coordinates.items():
        centers, counts, rms = sequence_centroids(values, indices, len(sequences))
        arrays[key+'_centroids'], arrays[key+'_duplicate_rms'] = centers, rms
        item = {'neighborhoods': {}, 'within_sequence_rms': summary(rms)}
        for k in DIAGNOSTIC_K:
            low, boundary, _ = exact_neighbors(centers[query], centers, query, 'euclidean', k)
            arrays[f'low_{key}_k{k}'], arrays[f'low_{key}_boundary_k{k}'] = low, boundary
            by_reference = {}
            for name in high:
                fractions = np.asarray([len(set(a) & set(b))/k for a,b in zip(high[name][k], low, strict=True)])
                arrays[f'{key}_vs_{name}_k{k}'] = fractions
                by_reference[name] = summary(fractions)
            item['neighborhoods'][str(k)] = by_reference
        metrics[key] = item
    arrays['clone_count_by_sequence'] = counts
    with (output/'neighborhood_diagnostics.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'query_count': len(query), 'candidate_count': len(sequences), 'k': list(DIAGNOSTIC_K),
            'query_selection': 'fixed length and saved index quantiles, descriptive nonrandom sample',
            'unit': 'distinct CDR-H3 centroids for diagnostic only; KDE keeps all clone observations',
            'search': 'exact cosine in each high dimensional representation; exact Euclidean in 2D; self excluded; saved-index tie break',
            'high_dimensional_changes': high_changes, 'fits': metrics,
            'limitations': 'not paper reproduction, trustworthiness or antigen specificity; unique inverse-map zero dispersion is structural'}


def compute(run_dir, projection_dir, pooling_dir, output_dir, alternate_seed):
    context = load_context(run_dir, projection_dir)
    folder, run, clones, sequences, cached, hashes, projection, pm, ph, inputs = context
    pooling = Path(pooling_dir).resolve(strict=True)
    previous = read(pooling/'comparison_metadata.json')
    previous_hash = digest(pooling/'comparison_metadata.json')
    if (previous.get('status') != 'completed' or previous['source_sha256'] != hashes
            or previous['projection_sha256'] != ph or previous['packages'] != pm['packages']):
        raise ValueError('Prior comparison does not match source.')
    _verify_sources(pooling, previous['artifact_sha256'])
    if not np.array_equal(cached, np.load(pooling/'all_embeddings.npy', allow_pickle=False)):
        raise ValueError('Official embedding cache mismatch.')
    seed = pm['parameters']['random_state']
    required = {'n_neighbors': 15, 'min_dist': .1, 'n_epochs': 200, 'init': 'random', 'metric': 'cosine', 'unique': False}
    if (any(pm['parameters'].get(k) != v for k,v in required.items())
            or type(alternate_seed) is not int or not 0 <= alternate_seed < 2**32 or alternate_seed == seed):
        raise ValueError('Need the fixed baseline and distinct uint32 seeds.')
    seeds = [seed, alternate_seed]
    # Validate both reused fits before any new result directory is created.
    for s in seeds:
        prior = previous['fits'][f'all_seed{s}']
        settings = pm['parameters'] | {'random_state': s, 'transform_seed': s}
        if (prior['parameters'] != settings or prior['source_clone_sha256'] != hashes['clones.json']
                or prior['embedding_sha256'] != hashes['embeddings.npy']):
            raise ValueError('Reused fit provenance mismatch.')
    output = validate_output_location(Path(output_dir), inputs)
    if any(existing == output or existing in output.parents for existing in (folder, pooling)):
        raise ValueError('New output must be outside saved results.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'families': FAMILIES, 'seeds': seeds, 'baseline_parameters': pm['parameters'],
            'pca_target_variance': TARGET_VARIANCE, 'pca_solver': 'full', 'pca_whiten': False,
            'fit_unit_change': 'CDR-H3 ID unique fit in saved order, then inverse-map every original clone',
            'density_change': 'pooled Scott covariance shared across phases within seed; width rule and shape both change',
            'source_sha256': hashes, 'stop_after_declared_comparisons': True,
            'production_defaults_changed': False, 'script_sha256': digest(Path(__file__))}
    write_json(output/'experiment_plan.json', plan)
    indices = np.asarray([c['embedding_index'] for c in clones], dtype=int)
    phases = [c['timepoint'] for c in clones]
    save_array(output/'clone_embedding_indices.npy', indices)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=run['parameters']['threads']):
        representations, pca_arrays, preprocessing = preprocess(cached, indices)
    print('PCA fitted once; retained dimensions:', preprocessing['pca_retained_components'], flush=True)
    with (output/'pca_model.npz').open('xb') as handle:
        np.savez_compressed(handle, **pca_arrays)
    for name, values in representations.items():
        save_array(output/f'{name}_embeddings.npy', values)
    raw, fits = {}, {}
    for variant in VARIANTS:
        for s in seeds:
            key = f'{variant}_seed{s}'
            settings = pm['parameters'] | {'random_state': s, 'transform_seed': s}
            values = representations['raw' if variant == 'unique' else variant]
            fit_values = values if variant == 'unique' else values[indices]
            started = perf_counter()
            if variant == 'raw':
                coords = np.load(pooling/previous['fits'][f'all_seed{s}']['raw_coordinates_file'], allow_pickle=False)
                audit = {'warnings': None, 'solver_fallback_to_random': None,
                         'interpretation': 'Verified baseline reused; prior graph and warning records unavailable.'}
                print('Reuse:', key, flush=True)
            else:
                print('UMAP starting:', key, flush=True)
                coords, audit = audited_fit(fit_values, settings)
                write_json(output/(key+'_initialization_audit.json'), audit)
                if audit['solver_fallback_to_random']:
                    raise RuntimeError('Unexpected initialization fallback; stop, do not substitute a new condition.')
            fit = {'variant': variant, 'seed': s, 'parameters': settings,
                   'reused_previous_fit': variant == 'raw', 'initialization_audit': audit,
                   'fit_observation_count': len(fit_values), 'output_observation_count': len(clones),
                   'elapsed_seconds': perf_counter()-started,
                   'raw_coordinates_file': key+'_raw.npy', 'aligned_coordinates_file': key+'_aligned.npy'}
            if variant == 'unique':
                fit['unique_coordinates_file'] = key+'_unique.npy'
                save_array(output/fit['unique_coordinates_file'], coords)
                coords = expand_unique(coords, indices, len(sequences))
            if coords.shape != (len(clones), 2) or not np.isfinite(coords).all():
                raise ValueError('Every clone needs finite coordinates.')
            raw[key] = coords
            save_array(output/fit['raw_coordinates_file'], coords)
            fits[key] = fit
            write_json(output/(key+'_fit.json'), fit)
    aligned = {}
    reference = raw[f'raw_seed{seed}']
    for key, coords in raw.items():
        aligned[key], fits[key]['alignment'] = rigid_align(coords, reference)
        save_array(output/fits[key]['aligned_coordinates_file'], aligned[key])
    bandwidth = float(pm['kde']['bandwidth'])
    isotropic = np.eye(2) * bandwidth**2
    scott = {s: pooled_scott_covariance(aligned[f'raw_seed{s}']) for s in seeds}
    families, densities = {}, {}
    for family, variants in FAMILIES.items():
        entries = []
        for s in seeds:
            for variant in variants:
                fit_key = f'{"raw" if family == "density" else variant}_seed{s}'
                H = scott[s][0] if variant == 'pooled_scott' else isotropic
                entries.append((s, variant, fit_key, H))
        x, y, grid = shared_grid([aligned[k] for _,_,k,_ in entries], [H for _,_,_,H in entries])
        families[family] = grid | {'variants': list(variants), 'density_keys': {str(s): [] for s in seeds}}
        for s, variant, fit_key, H in entries:
            key = f'{family}__{variant}_seed{s}'
            print('KDE:', key, flush=True)
            if variant == 'pooled_scott':
                values, integrals = covariance_density(aligned[fit_key], phases, x, y, H)
            else:
                values, integrals = fixed_density(aligned[fit_key], phases, x, y, bandwidth)
            name = key+'_density.npz'
            with (output/name).open('xb') as handle:
                np.savez_compressed(handle, x=x, y=y, densities=values)
            densities[key] = {'family': family, 'variant': variant, 'seed': s, 'fit_key': fit_key,
                              'file': name, 'method': 'pooled_scott' if variant == 'pooled_scott' else 'isotropic',
                              'bandwidth_matrix': H.tolist(),
                              'integrals_before_normalization': dict(zip(PHASES, integrals, strict=True))}
            families[family]['density_keys'][str(s)].append(key)
    print('Diagnosing original and preprocessed neighborhoods.', flush=True)
    with threadpool_limits(limits=run['parameters']['threads']):
        diagnostics = neighborhood_diagnostics(representations, sequences, indices, raw, output)
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    ids = np.unique(np.concatenate([np.flatnonzero(np.asarray(phases)==p)[np.unique(np.linspace(
        0, phases.count(p)-1, min(350, phases.count(p)), dtype=int))] for p in PHASES]))
    save_array(output/'coordinate_diagnostic_clone_ids.npy', ids)
    sensitivity = {}
    for variant in VARIANTS:
        a, b = (f'{variant}_seed{s}' for s in seeds)
        _, transform = rigid_align(raw[b], raw[a])
        sensitivity[variant] = {'pair_distance_spearman': float(spearmanr(pdist(raw[a][ids]), pdist(raw[b][ids])).statistic),
                                'rigid_rms_all_clones': transform['rms_displacement']}
    verify_unchanged(context)
    _verify_sources(pooling, previous['artifact_sha256'])
    if digest(pooling/'comparison_metadata.json') != previous_hash:
        raise ValueError('Prior comparison changed.')
    metadata = {'status': 'completed', 'source_run': str(folder), 'source_projection': str(projection),
        'source_pooling_comparison': str(pooling), 'source_sha256': hashes, 'projection_sha256': ph,
        'pooling_metadata_sha256': previous_hash, 'input_hashes': run['input_hashes'], 'input_policies': run['input_policies'],
        'seeds': seeds, 'variants': list(VARIANTS), 'fits': fits, 'densities': densities, 'families': families,
        'observation_counts': dict(Counter(phases)), 'unique_sequence_count': len(sequences),
        'all_clones_retained': True, 'all_clones_used_for_every_fit': False, 'random_downsampling_for_umap': False,
        'preprocessing': preprocessing, 'scott_by_seed': {str(s): scott[s][1] for s in seeds},
        'kde': {'baseline_bandwidth': bandwidth, 'normalization': 'each phase integral one on common family finite grid',
                'kernel': 'Gaussian', 'algorithm': 'ball_tree', 'rtol': 1e-6, 'atol': 0, 'counts_weights': False},
        'neighborhood_diagnostics': diagnostics, 'coordinate_metrics': sensitivity,
        'coordinate_metric_sample_size': len(ids), 'stop_after_declared_comparisons': True,
        'production_defaults_changed': False, 'candidate_selection_performed': False,
        'paper_coordinate_similarity_quantified': False, 'sources_unchanged': True,
        'packages': pm['packages'], 'code': code_provenance(),
        'script_sha256': digest(Path(__file__)), 'density_script_sha256': digest(Path(__file__).with_name('final_comparison_density.py')),
        'limitations': 'Two seeds; no original paper coordinates; no biological validation; unique fit changes multiplicity and graph, not eligibility.'}
    metadata['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.npy', '.npz', '.json')}
    write_json(output/'comparison_metadata.json', metadata)
    print('All three declared comparisons completed. Stop; no further search.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'projection-dir', 'pooling-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--alternate-seed', type=int, default=20260920)
    args = parser.parse_args()
    enable_network_guard()
    compute(args.run_dir, args.projection_dir, args.pooling_dir, args.output_dir, args.alternate_seed)


if __name__ == '__main__':
    main()
