"""Compare min_dist=(0.1,0.3,0.5) with fixed inputs, n_neighbors=15 and two seeds.

All clone rows are fitted; source data and production defaults are read-only.
Diagnostics reuse the validated distinct-CDR-H3 centroid method, not candidate scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np

from compute_pooling_comparison import (
    read, save_array, rigid_align, fixed_density, load_context, verify_unchanged,
)
from compute_neighbors_comparison import neighborhood_diagnostics, DIAGNOSTIC_K
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _fit_umap, _verify_sources

MIN_DISTS = (0.1, 0.3, 0.5)


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
    if (pm['parameters']['n_neighbors'] != 15 or pm['parameters']['min_dist'] != MIN_DISTS[0] or type(alternate_seed) is not int
            or not 0 <= alternate_seed < 2**32 or alternate_seed == seed):
        raise ValueError('Need the declared baseline and a distinct uint32 seed.')
    seeds = [seed, alternate_seed]
    output = validate_output_location(Path(output_dir), inputs)
    if folder == output or folder in output.parents or pooling == output or pooling in output.parents:
        raise ValueError('Comparison output must be outside existing result directories.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'min_dists': list(MIN_DISTS), 'seeds': seeds, 'diagnostic_k': list(DIAGNOSTIC_K),
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
    for distance in MIN_DISTS:
        for s in seeds:
            key = f'md{distance:g}_seed{s}'
            settings = pm['parameters'] | {'min_dist': distance, 'random_state': s, 'transform_seed': s}
            started = perf_counter()
            reuse = distance == MIN_DISTS[0] and f'all_seed{s}' in previous['fits']
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
            fit = {'seed': s, 'min_dist': distance, 'parameters': settings,
                'reused_previous_fit': reuse, 'elapsed_seconds': perf_counter()-started,
                'source_clone_sha256': hashes['clones.json'], 'embedding_sha256': hashes['embeddings.npy'],
                'raw_coordinates_file': key+'_raw.npy', 'aligned_coordinates_file': key+'_aligned.npy',
                'density_file': key+'_density.npz'}
            save_array(output/fit['raw_coordinates_file'], coords)
            fit['raw_sha256'] = digest(output/fit['raw_coordinates_file'])
            write_json(output/(key+'_fit.json'), fit)
            fits[key] = fit
    reference = raw[f'md{MIN_DISTS[0]:g}_seed{seed}']
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
    pairs = [(f'md{distance:g}_seed{seeds[0]}', f'md{distance:g}_seed{seeds[1]}') for distance in MIN_DISTS]
    pairs += [(f'md{MIN_DISTS[0]:g}_seed{s}', f'md{distance:g}_seed{s}') for distance in MIN_DISTS[1:] for s in seeds]
    sensitivity = {}
    for a, b in pairs:
        _, transform = rigid_align(raw[b], raw[a])
        sensitivity[a+'__'+b] = {'pair_distance_spearman': float(spearmanr(pdist(raw[a][ids]), pdist(raw[b][ids])).statistic),
            'rigid_rms_all_clones': transform['rms_displacement']}
    verify_unchanged(context)
    _verify_sources(pooling, previous['artifact_sha256'])
    if digest(pooling/'comparison_metadata.json') != previous_hash:
        raise ValueError('Previous comparison changed.')
    metadata = {'status': 'completed', 'purpose': 'UMAP min_dist sensitivity, not original paper settings',
        'source_run': str(folder), 'source_projection': str(projection), 'source_pooling_comparison': str(pooling),
        'source_sha256': hashes, 'projection_sha256': ph, 'pooling_metadata_sha256': previous_hash,
        'input_hashes': run['input_hashes'], 'input_policies': run['input_policies'],
        'min_dists': list(MIN_DISTS), 'seeds': seeds, 'fits': fits, 'observation_counts': dict(Counter(phases)),
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
    print('UMAP min_dist comparison completed. All sources unchanged.', flush=True)


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
