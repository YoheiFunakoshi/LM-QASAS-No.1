"""Independently compare initialization and distance metric with two seeds.

All clone rows are fitted; source data and production defaults are read-only.
Diagnostics reuse the validated distinct-CDR-H3 centroid method, not candidate scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
import warnings

from compute_pooling_comparison import (
    read, save_array, summary, rigid_align, fixed_density, load_context, verify_unchanged,
)
from compute_neighbors_comparison import sequence_centroids, exact_neighbors, DIAGNOSTIC_K
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _verify_sources

VARIANTS = {
    'baseline': {'init': 'random', 'metric': 'cosine'},
    'spectral': {'init': 'spectral', 'metric': 'cosine'},
    'euclidean': {'init': 'random', 'metric': 'euclidean'},
}


def fallback_warning(records):
    return any('falling back to random' in item['message'].lower() for item in records)


def audited_fit(vectors, settings):
    from umap import UMAP
    from scipy.sparse.csgraph import connected_components
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = UMAP(**settings)
        coordinates = np.asarray(model.fit_transform(vectors), dtype=np.float32)
    records = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
    # Mirrors the installed UMAP initialization graph pruning for fixed epochs > 10.
    epochs = settings['n_epochs']
    if type(epochs) is not int or epochs <= 10:
        raise ValueError('Graph audit requires an explicit epoch count greater than 10.')
    graph = model.graph_.tocoo(copy=True)
    graph.sum_duplicates()
    threshold = float(graph.data.max()) / epochs
    graph.data[graph.data < threshold] = 0
    graph.eliminate_zeros()
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    # In 2D, components of size <= 3 use a random local layout, without a warning.
    small = sizes < 2 * settings['n_components']
    audit = {'warnings': records, 'solver_fallback_to_random': fallback_warning(records),
        'graph_components': {'count': int(count), 'sizes': sorted(sizes.tolist(), reverse=True),
            'prune_threshold': threshold, 'prune_rule': 'edge weight < max weight / n_epochs',
            'small_component_count': int(small.sum()), 'small_component_vertices': int(sizes[small].sum()),
            'small_component_rule': 'size < 2*n_components; random local layout in spectral initialization'},
        'interpretation': 'Absence of a solver-fallback warning does not imply all disconnected components used a spectral local layout.'}
    return coordinates, audit


def dual_neighborhood_diagnostics(cached, sequences, clone_indices, raw, output):
    vectors = np.asarray(cached, dtype=np.float32).astype(np.float64)
    lengths = np.asarray([len(s) for s in sequences])
    order = np.lexsort((np.arange(len(sequences)), lengths))
    query_ids = order[np.unique(np.linspace(0, len(order)-1, min(512, len(order)), dtype=int))]
    arrays = {'query_ids': query_ids, 'sequence_lengths': lengths}
    high, high_ties, cross = {}, {}, {}
    for metric in ('cosine', 'euclidean'):
        high[metric], high_ties[metric] = {}, {}
        for k in DIAGNOSTIC_K:
            neighbors, bounds, _ = exact_neighbors(vectors[query_ids], vectors, query_ids, metric, k)
            high[metric][k] = neighbors
            arrays[f'high_{metric}_neighbors_k{k}'] = neighbors
            arrays[f'high_{metric}_boundary_k{k}'] = bounds
            high_ties[metric][str(k)] = int(np.isclose(bounds[:,0], bounds[:,1], rtol=1e-12, atol=1e-14).sum())
    for k in DIAGNOSTIC_K:
        overlap = np.asarray([len(set(a) & set(b))/k for a,b in zip(high['cosine'][k], high['euclidean'][k], strict=True)])
        arrays[f'high_cross_metric_overlap_k{k}'] = overlap
        cross[str(k)] = summary(overlap)
    metrics = {}
    for key, coordinates in raw.items():
        centers, counts, rms = sequence_centroids(coordinates, clone_indices, len(sequences))
        arrays[key+'_centroids'], arrays[key+'_duplicate_rms'] = centers, rms
        result = {}
        for k in DIAGNOSTIC_K:
            low, bounds, radius = exact_neighbors(centers[query_ids], centers, query_ids, 'euclidean', k)
            arrays[f'{key}_neighbors_k{k}'], arrays[f'{key}_boundary_k{k}'] = low, bounds
            overlaps = {}
            for metric in high:
                overlap = np.asarray([len(set(a) & set(b))/k for a,b in zip(high[metric][k], low, strict=True)])
                arrays[f'{key}_{metric}_overlap_k{k}'] = overlap
                overlaps[metric] = summary(overlap)
            ratio = np.divide(rms[query_ids], radius, out=np.zeros_like(radius), where=radius>0)
            if np.any((radius == 0) & (rms[query_ids] > 0)):
                raise ValueError('Zero neighborhood radius with nonzero duplicate dispersion.')
            arrays[f'{key}_radius_k{k}'], arrays[f'{key}_dispersion_ratio_k{k}'] = radius, ratio
            result[str(k)] = {'overlap_fraction_by_reference': overlaps,
                'query_rms_over_kth_neighbor_radius': summary(ratio), 'zero_radius_queries': int((radius==0).sum()),
                'low_boundary_near_tie_queries': int(np.isclose(bounds[:,0], bounds[:,1], rtol=1e-12, atol=1e-14).sum())}
        duplicate = counts > 1
        metrics[key] = {'neighborhoods': result, 'within_sequence_rms_all': summary(rms),
            'within_sequence_rms_duplicates_only': summary(rms[duplicate]) if duplicate.any() else None}
        arrays['clone_count_by_sequence'] = counts
    with (output/'neighborhood_diagnostics.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'unit': 'distinct CDR-H3 centroids, evaluation only; all clone rows retained for fitting and KDE',
        'query_count': len(query_ids), 'candidate_count': len(sequences),
        'query_selection': 'fixed length then saved index quantiles; descriptive, not a random sample',
        'high_dimensional_dtype': 'float32 UMAP input promoted to float64; no normalization or scaling added',
        'search': 'both exact cosine and Euclidean references vs exact 2D Euclidean; all distinct candidates; self excluded',
        'k': list(DIAGNOSTIC_K), 'tie_break': 'ascending saved sequence index',
        'high_boundary_near_tie_queries': high_ties, 'high_cross_metric_overlap': cross,
        'limitations': 'Centroids may mask duplicate scattering; descriptive overlap is not biological validity, trustworthiness, or paper reproduction.',
        'fits': metrics}


def compute(run_dir, projection_dir, pooling_dir, output_dir, alternate_seed, retain_spectral_fallback=False):
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
    if (pm['parameters']['n_neighbors'] != 15 or pm['parameters']['min_dist'] != 0.1
            or any(pm['parameters'][name] != value for name, value in VARIANTS['baseline'].items())
            or type(alternate_seed) is not int
            or not 0 <= alternate_seed < 2**32 or alternate_seed == seed):
        raise ValueError('Need the declared baseline and a distinct uint32 seed.')
    seeds = [seed, alternate_seed]
    output = validate_output_location(Path(output_dir), inputs)
    if folder == output or folder in output.parents or pooling == output or pooling in output.parents:
        raise ValueError('Comparison output must be outside existing result directories.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'variants': VARIANTS, 'seeds': seeds, 'diagnostic_k': list(DIAGNOSTIC_K),
        'fallback_policy': 'retain_and_label' if retain_spectral_fallback else 'abort',
        'baseline_parameters': pm['parameters'], 'source_sha256': hashes,
        'fixed_n_neighbors': 15, 'fixed_kde_bandwidth': pm['kde']['bandwidth'],
        'diagnostic_unit': 'distinct CDR-H3 centroids, evaluation only',
        'all_clones_for_fit': True, 'no_visual_result_selection': True,
        'production_defaults_changed': False, 'script_sha256': digest(Path(__file__))}
    write_json(output/'experiment_plan.json', plan)
    indices = np.asarray([c['embedding_index'] for c in clones], dtype=int)
    phases = [c['timepoint'] for c in clones]
    vectors = np.asarray(cached[indices], dtype=np.float32)
    raw, fits = {}, {}
    for variant, changes in VARIANTS.items():
        for s in seeds:
            key = f'{variant}_seed{s}'
            settings = pm['parameters'] | changes | {'random_state': s, 'transform_seed': s}
            started = perf_counter()
            reuse = variant == 'baseline' and f'all_seed{s}' in previous['fits']
            if reuse:
                prior = previous['fits'][f'all_seed{s}']
                if (prior['parameters'] != settings or prior['source_clone_sha256'] != hashes['clones.json']
                        or prior['embedding_sha256'] != hashes['embeddings.npy']):
                    raise ValueError('Reused fit settings or source differ.')
                coords = np.load(pooling/prior['raw_coordinates_file'], allow_pickle=False)
                audit = {'warnings': None, 'solver_fallback_to_random': None, 'graph_components': None,
                    'interpretation': 'Previously verified random baseline reused; graph and warnings were not retained.'}
                print(f'Reuse verified coordinates: {key}', flush=True)
            else:
                print(f'UMAP starting: {key}', flush=True)
                coords, audit = audited_fit(vectors, settings)
                write_json(output/(key+'_initialization_audit.json'), audit)
                if variant == 'spectral' and audit['solver_fallback_to_random'] and not retain_spectral_fallback:
                    raise RuntimeError('Spectral solver fell back to random; warning audit saved, comparison not declared completed.')
                if variant == 'spectral' and audit['solver_fallback_to_random']:
                    print(f'REFERENCE ONLY: solver fallback to random occurred in {key}', flush=True)
            if coords.shape != (len(clones), 2) or not np.isfinite(coords).all():
                raise ValueError('Every clone needs finite coordinates.')
            raw[key] = coords
            fit = {'seed': s, 'variant': variant, 'parameters': settings, 'initialization_audit': audit,
                'reused_previous_fit': reuse, 'elapsed_seconds': perf_counter()-started,
                'source_clone_sha256': hashes['clones.json'], 'embedding_sha256': hashes['embeddings.npy'],
                'raw_coordinates_file': key+'_raw.npy', 'aligned_coordinates_file': key+'_aligned.npy',
                'density_file': key+'_density.npz'}
            save_array(output/fit['raw_coordinates_file'], coords)
            fit['raw_sha256'] = digest(output/fit['raw_coordinates_file'])
            write_json(output/(key+'_fit.json'), fit)
            fits[key] = fit
    reference = raw[f'baseline_seed{seed}']
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
        diagnostics = dual_neighborhood_diagnostics(cached, sequences, indices, raw, output)
    ids = np.unique(np.concatenate([np.flatnonzero(np.asarray(phases)==p)[np.unique(np.linspace(
        0, phases.count(p)-1, min(350, phases.count(p)), dtype=int))] for p in PHASES]))
    save_array(output/'coordinate_diagnostic_clone_ids.npy', ids)
    pairs = [(f'{variant}_seed{seeds[0]}', f'{variant}_seed{seeds[1]}') for variant in VARIANTS]
    pairs += [(f'baseline_seed{s}', f'{variant}_seed{s}') for variant in ('spectral', 'euclidean') for s in seeds]
    sensitivity = {}
    for a, b in pairs:
        _, transform = rigid_align(raw[b], raw[a])
        sensitivity[a+'__'+b] = {'pair_distance_spearman': float(spearmanr(pdist(raw[a][ids]), pdist(raw[b][ids])).statistic),
            'rigid_rms_all_clones': transform['rms_displacement']}
    verify_unchanged(context)
    _verify_sources(pooling, previous['artifact_sha256'])
    if digest(pooling/'comparison_metadata.json') != previous_hash:
        raise ValueError('Previous comparison changed.')
    fallback_fits = [key for key, fit in fits.items() if fit['variant'] == 'spectral'
                     and fit['initialization_audit']['solver_fallback_to_random']]
    metadata = {'status': 'completed_with_fallback' if fallback_fits else 'completed',
        'fallback_policy': plan['fallback_policy'], 'spectral_fallback_fits': fallback_fits,
        'purpose': 'Independent initialization and metric sensitivity, not original paper settings',
        'source_run': str(folder), 'source_projection': str(projection), 'source_pooling_comparison': str(pooling),
        'source_sha256': hashes, 'projection_sha256': ph, 'pooling_metadata_sha256': previous_hash,
        'input_hashes': run['input_hashes'], 'input_policies': run['input_policies'],
        'variants': VARIANTS, 'seeds': seeds, 'fits': fits, 'observation_counts': dict(Counter(phases)),
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
        'limitations': 'Two seed settings, not matched optimizer random streams; no paper coordinates; fixed bandwidth smooths differing extents differently; all high-dimensional references reported.'}
    metadata['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.npy', '.npz', '.json')}
    write_json(output/'comparison_metadata.json', metadata)
    print(f'UMAP comparison status: {metadata["status"]}. All sources unchanged.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'projection-dir', 'pooling-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--alternate-seed', type=int, default=20260920)
    parser.add_argument('--retain-spectral-fallback', action='store_true',
        help='Keep results with spectral solver fallback as explicitly labeled reference-only figures; default is abort.')
    args = parser.parse_args()
    enable_network_guard()
    compute(args.run_dir, args.projection_dir, args.pooling_dir, args.output_dir, args.alternate_seed,
            retain_spectral_fallback=args.retain_spectral_fallback)


if __name__ == '__main__':
    main()
