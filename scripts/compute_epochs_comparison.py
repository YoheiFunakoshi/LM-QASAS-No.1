"""Compare n_epochs=(200,500,1000) with fixed inputs and two seeds.

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
from compute_neighbors_comparison import DIAGNOSTIC_K
from compute_init_metric_comparison import audited_fit, dual_neighborhood_diagnostics
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, code_provenance, protect_output, validate_output_location, write_json
from lmqasas.visualization import _verify_sources

EPOCHS = (200, 500, 1000)
EPOCHS_CAVEAT = ('Each fit restarts. n_epochs changes weak-edge pruning threshold and learning-rate decay '
                'schedule as well as optimization duration; not continuation of the 200-epoch trajectory '
                'or a convergence guarantee. Other declared UMAP settings remain fixed.')


def common_grid(aligned, bandwidth):
    stacked = np.concatenate(list(aligned.values()))
    margin = np.maximum(3*bandwidth, .05*np.ptp(stacked, axis=0))
    lower, upper = stacked.min(axis=0)-margin, stacked.max(axis=0)+margin
    size = max(96, 1+int(np.ceil(np.max(upper-lower)/(bandwidth/2))))
    if size > 512:
        raise ValueError('Shared grid would exceed 512; investigate extent without downsampling or clipping.')
    x, y = (np.linspace(lower[i], upper[i], size) for i in (0, 1))
    return x, y


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
    if (pm['parameters']['n_neighbors'] != 15 or pm['parameters']['min_dist'] != 0.1
            or pm['parameters']['n_epochs'] != EPOCHS[0] or pm['parameters']['init'] != 'random'
            or pm['parameters']['metric'] != 'cosine' or type(alternate_seed) is not int
            or not 0 <= alternate_seed < 2**32 or alternate_seed == seed):
        raise ValueError('Need the declared baseline and a distinct uint32 seed.')
    seeds = [seed, alternate_seed]
    output = validate_output_location(Path(output_dir), inputs)
    if folder == output or folder in output.parents or pooling == output or pooling in output.parents:
        raise ValueError('Comparison output must be outside existing result directories.')
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'epochs': list(EPOCHS), 'seeds': seeds, 'diagnostic_k': list(DIAGNOSTIC_K),
        'baseline_parameters': pm['parameters'], 'source_sha256': hashes,
        'fixed_n_neighbors': 15, 'fixed_kde_bandwidth': pm['kde']['bandwidth'],
        'epochs_interpretation': EPOCHS_CAVEAT,
        'common_grid_rule': 'at least 96 points per axis; common step <= fixed bandwidth / 2; no clipping; fail above 512',
        'diagnostic_unit': 'distinct CDR-H3 centroids, evaluation only',
        'all_clones_for_fit': True, 'no_visual_result_selection': True,
        'production_defaults_changed': False, 'script_sha256': digest(Path(__file__))}
    write_json(output/'experiment_plan.json', plan)
    indices = np.asarray([c['embedding_index'] for c in clones], dtype=int)
    phases = [c['timepoint'] for c in clones]
    vectors = np.asarray(cached[indices], dtype=np.float32)
    raw, fits = {}, {}
    for epochs in EPOCHS:
        for s in seeds:
            key = f'ep{epochs}_seed{s}'
            settings = pm['parameters'] | {'n_epochs': epochs, 'random_state': s, 'transform_seed': s}
            started = perf_counter()
            reuse = epochs == EPOCHS[0] and f'all_seed{s}' in previous['fits']
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
                if audit['solver_fallback_to_random']:
                    raise RuntimeError('Unexpected solver fallback in random-initialized comparison; investigate saved audit.')
            if coords.shape != (len(clones), 2) or not np.isfinite(coords).all():
                raise ValueError('Every clone needs finite coordinates.')
            raw[key] = coords
            fit = {'seed': s, 'n_epochs': epochs, 'parameters': settings, 'initialization_audit': audit,
                'reused_previous_fit': reuse, 'elapsed_seconds': perf_counter()-started,
                'source_clone_sha256': hashes['clones.json'], 'embedding_sha256': hashes['embeddings.npy'],
                'raw_coordinates_file': key+'_raw.npy', 'aligned_coordinates_file': key+'_aligned.npy',
                'density_file': key+'_density.npz'}
            save_array(output/fit['raw_coordinates_file'], coords)
            fit['raw_sha256'] = digest(output/fit['raw_coordinates_file'])
            write_json(output/(key+'_fit.json'), fit)
            fits[key] = fit
    reference = raw[f'ep{EPOCHS[0]}_seed{seed}']
    aligned = {}
    for key, coords in raw.items():
        aligned[key], fits[key]['alignment'] = rigid_align(coords, reference)
        save_array(output/fits[key]['aligned_coordinates_file'], aligned[key])
    bandwidth = float(pm['kde']['bandwidth'])
    x, y = common_grid(aligned, bandwidth)
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
    pairs = [(f'ep{epochs}_seed{seeds[0]}', f'ep{epochs}_seed{seeds[1]}') for epochs in EPOCHS]
    pairs += [(f'ep{EPOCHS[0]}_seed{s}', f'ep{epochs}_seed{s}') for epochs in EPOCHS[1:] for s in seeds]
    sensitivity = {}
    for a, b in pairs:
        _, transform = rigid_align(raw[b], raw[a])
        sensitivity[a+'__'+b] = {'pair_distance_spearman': float(spearmanr(pdist(raw[a][ids]), pdist(raw[b][ids])).statistic),
            'rigid_rms_all_clones': transform['rms_displacement']}
    verify_unchanged(context)
    _verify_sources(pooling, previous['artifact_sha256'])
    if digest(pooling/'comparison_metadata.json') != previous_hash:
        raise ValueError('Previous comparison changed.')
    metadata = {'status': 'completed', 'purpose': 'UMAP n_epochs sensitivity, not original paper settings',
        'epochs_interpretation': EPOCHS_CAVEAT,
        'source_run': str(folder), 'source_projection': str(projection), 'source_pooling_comparison': str(pooling),
        'source_sha256': hashes, 'projection_sha256': ph, 'pooling_metadata_sha256': previous_hash,
        'input_hashes': run['input_hashes'], 'input_policies': run['input_policies'],
        'epochs': list(EPOCHS), 'seeds': seeds, 'fits': fits, 'observation_counts': dict(Counter(phases)),
        'all_clones_retained': True, 'downsampling_for_umap': False,
        'common_kde': {'bandwidth': bandwidth, 'bandwidth_source': 'saved baseline; fixed numeric value for all six fits',
            'normalization': 'per-timepoint integral one on shared finite grid', 'grid_size': len(x),
            'grid_rule': plan['common_grid_rule'],
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
    print('UMAP n_epochs comparison completed. All sources unchanged.', flush=True)


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
