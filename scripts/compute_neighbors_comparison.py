"""Offline UMAP neighborhood sensitivity; original data and production defaults stay fixed.

The predeclared grid is n_neighbors=(15,30,50), crossed with two seeds.
All clone observations are fitted. Distinct-CDR-H3 centroids are used ONLY
for a descriptive neighborhood diagnostic, never for fitting or selection.
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
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _fit_umap, _verify_sources

NEIGHBORS = (15, 30, 50)
DIAGNOSTIC_K = (15, 50)


def sequence_centroids(coordinates, clone_indices, n_sequences):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    clone_indices = np.asarray(clone_indices)
    if (coordinates.shape != (len(clone_indices), 2) or not np.isfinite(coordinates).all()
            or clone_indices.dtype.kind not in 'iu' or len(clone_indices) == 0
            or clone_indices.min() < 0 or clone_indices.max() >= n_sequences):
        raise ValueError('Invalid clone coordinate correspondence.')
    counts = np.bincount(clone_indices, minlength=n_sequences)
    if np.any(counts == 0):
        raise ValueError('Every sequence needs at least one clone observation.')
    centers = np.zeros((n_sequences, 2), dtype=np.float64)
    np.add.at(centers, clone_indices, coordinates)
    centers /= counts[:, None]
    squared = np.sum((coordinates - centers[clone_indices]) ** 2, axis=1)
    rms = np.sqrt(np.bincount(clone_indices, weights=squared, minlength=n_sequences) / counts)
    return centers, counts, rms


def exact_neighbors(query_vectors, all_vectors, query_ids, metric, k):
    """Exact search; deterministic index tie-break; report boundary near-ties."""
    from scipy.spatial.distance import cdist
    if not 0 < k < len(all_vectors) - 1:
        raise ValueError('Need k+1 nonself candidates.')
    groups, boundaries, radii = [], [], []
    for start in range(0, len(query_ids), 32):
        ids = query_ids[start:start+32]
        distances = cdist(query_vectors[start:start+32], all_vectors, metric=metric)
        if not np.isfinite(distances).all():
            raise ValueError('Undefined distances in diagnostic.')
        distances[np.arange(len(ids)), ids] = np.inf
        order = np.argsort(distances, axis=1, kind='stable')[:, :k+1]
        values = np.take_along_axis(distances, order, axis=1)
        groups.append(order[:, :k])
        boundaries.append(values[:, [k-1, k]])
        radii.append(values[:, k-1])
    return np.concatenate(groups), np.concatenate(boundaries), np.concatenate(radii)


def neighborhood_diagnostics(cached, sequences, clone_indices, raw, output):
    # Match the float32 vectors passed to UMAP, promoted to float64 for exact distances.
    vectors = np.asarray(cached, dtype=np.float32).astype(np.float64)
    lengths = np.asarray([len(s) for s in sequences])
    order = np.lexsort((np.arange(len(sequences)), lengths))
    query_ids = order[np.unique(np.linspace(0, len(order)-1, min(512, len(order)), dtype=int))]
    arrays = {'query_ids': query_ids, 'sequence_lengths': lengths}
    high = {}
    for k in DIAGNOSTIC_K:
        high[k], bounds, _ = exact_neighbors(vectors[query_ids], vectors, query_ids, 'cosine', k)
        arrays[f'high_neighbors_k{k}'] = high[k]
        arrays[f'high_boundary_k{k}'] = bounds
    metrics = {}
    for key, coordinates in raw.items():
        centers, counts, rms = sequence_centroids(coordinates, clone_indices, len(sequences))
        arrays[key+'_centroids'] = centers
        arrays[key+'_duplicate_rms'] = rms
        result = {}
        for k in DIAGNOSTIC_K:
            low, bounds, radius = exact_neighbors(centers[query_ids], centers, query_ids, 'euclidean', k)
            overlap = np.asarray([len(set(a) & set(b))/k for a,b in zip(high[k], low, strict=True)])
            arrays[f'{key}_neighbors_k{k}'] = low
            arrays[f'{key}_boundary_k{k}'] = bounds
            arrays[f'{key}_overlap_k{k}'] = overlap
            ratio = np.divide(rms[query_ids], radius, out=np.zeros_like(radius), where=radius>0)
            if np.any((radius == 0) & (rms[query_ids] > 0)):
                raise ValueError('Zero neighborhood radius with nonzero duplicate dispersion.')
            arrays[f'{key}_radius_k{k}'] = radius
            arrays[f'{key}_dispersion_ratio_k{k}'] = ratio
            result[str(k)] = {'overlap_fraction': summary(overlap),
                'query_rms_over_kth_neighbor_radius': summary(ratio),
                'zero_radius_queries': int((radius == 0).sum()),
                'low_boundary_near_tie_queries': int(np.isclose(bounds[:,0], bounds[:,1], rtol=1e-12, atol=1e-14).sum())}
        duplicate = counts > 1
        metrics[key] = {'neighborhoods': result,
            'within_sequence_rms_all': summary(rms),
            'within_sequence_rms_duplicates_only': summary(rms[duplicate]) if duplicate.any() else None}
        arrays['clone_count_by_sequence'] = counts
    with (output/'neighborhood_diagnostics.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'unit': 'distinct CDR-H3, centroid of all its clone observations FOR DIAGNOSTICS ONLY',
        'query_count': len(query_ids), 'candidate_count': len(sequences),
        'query_selection': 'fixed length then saved index quantiles; descriptive, not a random sample',
        'high_dimensional_dtype': 'float32 UMAP input promoted to float64',
        'search': 'exact cosine in 480D vs exact Euclidean in 2D centroids; all distinct candidates; self excluded',
        'k': list(DIAGNOSTIC_K), 'tie_break': 'ascending saved sequence index',
        'high_boundary_near_tie_queries': {str(k): int(np.isclose(arrays[f'high_boundary_k{k}'][:,0],
            arrays[f'high_boundary_k{k}'][:,1], rtol=1e-12, atol=1e-14).sum()) for k in DIAGNOSTIC_K},
        'limitations': 'Centroids can mask scattering of repeated sequences. Dispersion is reported; overlap is not biological validity, trustworthiness, or proof of paper reproduction.',
        'fits': metrics}


def compute(run_dir, projection_dir, pooling_dir, output_dir, alternate_seed):
    context = load_context(run_dir, projection_dir)
    folder, run, clones, sequences, cached, hashes, projection, pm, ph, inputs = context
    pooling = Path(pooling_dir).resolve(strict=True)
    previous = read(pooling/'comparison_metadata.json')
    previous_hash = digest(pooling/'comparison_metadata.json')
    if (previous.get('status') != 'completed' or previous['source_sha256'] != hashes
            or previous['projection_sha256'] != ph or previous['packages'] != pm['packages']):
        raise ValueError('Reuse source does not match the baseline.')
    _verify_sources(pooling, previous['artifact_sha256'])
    if not np.array_equal(cached, np.load(pooling/'all_embeddings.npy', allow_pickle=False)):
        raise ValueError('Reused UMAP embeddings differ from current official cache.')
    seed = pm['parameters']['random_state']
    if (pm['parameters']['n_neighbors'] != NEIGHBORS[0] or type(alternate_seed) is not int
            or not 0 <= alternate_seed < 2**32 or alternate_seed == seed):
        raise ValueError('Need the declared baseline and a distinct uint32 seed.')
    seeds = [seed, alternate_seed]
    output = validate_output_location(Path(output_dir), inputs)
    if folder == output or folder in output.parents or pooling == output or pooling in output.parents:
        raise ValueError('Comparison output must be outside existing result directories.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'neighbors': list(NEIGHBORS), 'seeds': seeds, 'diagnostic_k': list(DIAGNOSTIC_K),
        'baseline_parameters': pm['parameters'], 'source_sha256': hashes,
        'all_clones_for_fit': True, 'no_visual_result_selection': True,
        'production_defaults_changed': False, 'script_sha256': digest(Path(__file__))}
    write_json(output/'experiment_plan.json', plan)
    indices = np.asarray([c['embedding_index'] for c in clones], dtype=int)
    phases = [c['timepoint'] for c in clones]
    vectors = np.asarray(cached[indices], dtype=np.float32)
    raw, fits = {}, {}
    for n in NEIGHBORS:
        for s in seeds:
            key = f'nn{n}_seed{s}'
            settings = pm['parameters'] | {'n_neighbors': n, 'random_state': s, 'transform_seed': s}
            started = perf_counter()
            reuse = n == NEIGHBORS[0] and f'all_seed{s}' in previous['fits']
            if reuse:
                prior = previous['fits'][f'all_seed{s}']
                if (prior['parameters'] != settings or prior['source_clone_sha256'] != hashes['clones.json']
                        or prior['embedding_sha256'] != hashes['embeddings.npy']):
                    raise ValueError('Reused fit settings or source differ.')
                coords = np.load(pooling/prior['raw_coordinates_file'], allow_pickle=False)
                print(f'Reuse verified coordinates: {key}', flush=True)
            else:
                print(f'UMAP starting: {key}', flush=True)
                coords = _fit_umap(vectors, settings)
            if coords.shape != (len(clones), 2) or not np.isfinite(coords).all():
                raise ValueError('Every clone needs finite coordinates.')
            raw[key] = coords
            fit = {'seed': s, 'n_neighbors': n, 'parameters': settings,
                'reused_previous_fit': reuse, 'elapsed_seconds': perf_counter()-started,
                'source_clone_sha256': hashes['clones.json'], 'embedding_sha256': hashes['embeddings.npy'],
                'raw_coordinates_file': key+'_raw.npy', 'aligned_coordinates_file': key+'_aligned.npy',
                'density_file': key+'_density.npz'}
            save_array(output/fit['raw_coordinates_file'], coords)
            fit['raw_sha256'] = digest(output/fit['raw_coordinates_file'])
            write_json(output/(key+'_fit.json'), fit)
            fits[key] = fit
    reference = raw[f'nn{NEIGHBORS[0]}_seed{seed}']
    aligned = {}
    for key, coords in raw.items():
        aligned[key], fits[key]['alignment'] = rigid_align(coords, reference)
        save_array(output/fits[key]['aligned_coordinates_file'], aligned[key])
    bandwidth = float(pm['kde']['bandwidth'])
    stacked = np.concatenate(list(aligned.values()))
    margin = np.maximum(3*bandwidth, .05*np.ptp(stacked, axis=0))
    lower, upper = stacked.min(axis=0)-margin, stacked.max(axis=0)+margin
    x, y = (np.linspace(lower[i], upper[i], 96) for i in (0, 1))
    for key, coords in aligned.items():
        print(f'Fixed-bandwidth KDE: {key}', flush=True)
        density, integrals = fixed_density(coords, phases, x, y, bandwidth)
        with (output/fits[key]['density_file']).open('xb') as handle:
            np.savez_compressed(handle, x=x, y=y, densities=density)
        fits[key]['integrals_before_normalization'] = dict(zip(PHASES, integrals, strict=True))
    print('Measuring sequence-neighborhood preservation and seed sensitivity.', flush=True)
    from threadpoolctl import threadpool_limits
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    with threadpool_limits(limits=run['parameters']['threads']):
        diagnostics = neighborhood_diagnostics(cached, sequences, indices, raw, output)
    ids = np.unique(np.concatenate([np.flatnonzero(np.asarray(phases)==p)[np.unique(np.linspace(
        0, phases.count(p)-1, min(350, phases.count(p)), dtype=int))] for p in PHASES]))
    save_array(output/'coordinate_diagnostic_clone_ids.npy', ids)
    pairs = [(f'nn{n}_seed{seeds[0]}', f'nn{n}_seed{seeds[1]}') for n in NEIGHBORS]
    pairs += [(f'nn15_seed{s}', f'nn{n}_seed{s}') for n in NEIGHBORS[1:] for s in seeds]
    sensitivity = {}
    for a, b in pairs:
        _, transform = rigid_align(raw[b], raw[a])
        sensitivity[a+'__'+b] = {'pair_distance_spearman': float(spearmanr(pdist(raw[a][ids]), pdist(raw[b][ids])).statistic),
            'rigid_rms_all_clones': transform['rms_displacement']}
    verify_unchanged(context)
    _verify_sources(pooling, previous['artifact_sha256'])
    if digest(pooling/'comparison_metadata.json') != previous_hash:
        raise ValueError('Previous comparison changed.')
    metadata = {'status': 'completed', 'purpose': 'UMAP n_neighbors sensitivity, not original paper settings',
        'source_run': str(folder), 'source_projection': str(projection), 'source_pooling_comparison': str(pooling),
        'source_sha256': hashes, 'projection_sha256': ph, 'pooling_metadata_sha256': previous_hash,
        'input_hashes': run['input_hashes'], 'input_policies': run['input_policies'],
        'neighbors': list(NEIGHBORS), 'seeds': seeds, 'fits': fits, 'observation_counts': dict(Counter(phases)),
        'all_clones_retained': True, 'downsampling_for_umap': False,
        'common_kde': {'bandwidth': bandwidth, 'bandwidth_source': 'saved baseline; fixed numeric value for all six fits',
            'normalization': 'per-timepoint integral one on shared finite grid', 'grid_size': 96,
            'kernel': 'gaussian', 'algorithm': 'ball_tree', 'rtol': 1e-6, 'atol': 0, 'counts_weights': False,
            'x_limits': x[[0,-1]].tolist(), 'y_limits': y[[0,-1]].tolist()},
        'neighborhood_diagnostics': diagnostics, 'coordinate_metrics': sensitivity,
        'coordinate_metric_sample_size': len(ids), 'production_defaults_changed': False,
        'candidate_selection_performed': False, 'paper_coordinate_similarity_quantified': False,
        'sources_unchanged': True, 'packages': pm['packages'], 'code': code_provenance(),
        'script_sha256': digest(Path(__file__)),
        'limitations': 'Two seeds, descriptive diagnostics, fixed bandwidth can smooth different extents differently; no paper coordinates.'}
    metadata['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.npy', '.npz', '.json')}
    write_json(output/'comparison_metadata.json', metadata)
    print('UMAP neighborhood comparison completed. All sources unchanged.', flush=True)


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
