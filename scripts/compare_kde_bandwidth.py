"""Offline KDE sensitivity on verified saved coordinates; research outputs stay local."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread

import compute_pooling_comparison as compute
import render_pooling_comparison as render
from lmqasas.checkpoint import digest
from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import PHASES, protect_output, validate_output_location, write_json
from lmqasas.visualization import _verify_sources

FACTORS = (0.5, 1.0, 2.0)
FONT = Path('C:/Windows/Fonts/meiryo.ttc')


def load_inputs(pooling_dir):
    folder = Path(pooling_dir).resolve(strict=True)
    metadata_path = folder / 'comparison_metadata.json'
    prior = compute.read(metadata_path)
    if (prior.get('status') != 'completed' or prior.get('all_clones_retained') is not True
            or prior.get('downsampling_for_umap') is not False):
        raise ValueError('A completed full-clone pooling comparison is required.')
    hashes = prior['artifact_sha256'] | {metadata_path.name: digest(metadata_path)}
    _verify_sources(folder, hashes)
    context = compute.load_context(prior['source_run'], prior['source_projection'])
    _, run, clones, _, cached, source_hashes, projection, pm, ph, _ = context
    if (prior['source_sha256'] != source_hashes or prior['projection_sha256'] != ph
            or prior['input_hashes'] != run['input_hashes'] or prior['packages'] != pm['packages']
            or not np.array_equal(cached, np.load(folder / 'all_embeddings.npy', allow_pickle=False))):
        raise ValueError('Pooling provenance or official embeddings differ from the baseline.')
    seeds = prior['seeds']
    if (len(seeds) != 2 or len(set(seeds)) != 2 or seeds[0] != pm['parameters']['random_state']
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('The baseline and exactly one alternate seed are required.')
    if pm['parameters']['n_neighbors'] != 15 or pm['parameters']['min_dist'] != 0.1:
        raise ValueError('This comparison requires the unchanged UMAP baseline.')
    raw = {}
    for seed in seeds:
        fit = prior['fits'][f'all_seed{seed}']
        path = render._contained_file(folder, fit['raw_coordinates_file'])
        expected = pm['parameters'] | {'random_state': seed, 'transform_seed': seed}
        if (fit['parameters'] != expected or fit['pooling'] != 'all_nonpadding'
                or fit['source_clone_sha256'] != source_hashes['clones.json']
                or fit['embedding_sha256'] != digest(folder / 'all_embeddings.npy')
                or path.name not in hashes or digest(path) != fit['raw_sha256']):
            raise ValueError('A saved coordinate fit has inconsistent provenance.')
        raw[seed] = np.load(path, allow_pickle=False)
        if raw[seed].shape != (len(clones), 2) or not np.isfinite(raw[seed]).all():
            raise ValueError('All clone coordinates must be present and finite.')
    if not np.array_equal(raw[seeds[0]], np.load(projection / 'coordinates.npy', allow_pickle=False)):
        raise ValueError('The first coordinate set differs from the saved baseline.')
    return context, folder, prior, hashes, raw


def make_grid(aligned, bandwidth):
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('A positive finite baseline bandwidth is required.')
    stacked = np.concatenate(list(aligned.values()))
    span = np.ptp(stacked, axis=0)
    margin = np.maximum(3 * bandwidth * max(FACTORS), 0.05 * span)
    lower, upper = stacked.min(axis=0) - margin, stacked.max(axis=0) + margin
    size = max(96, int(np.ceil(np.max(upper - lower) / (0.5 * bandwidth * min(FACTORS)))) + 1)
    if size > 2048:
        raise ValueError('Required grid exceeds the predeclared resource limit.')
    return np.linspace(lower[0], upper[0], size), np.linspace(lower[1], upper[1], size)


def density_probes(coords, phases, x, y, density, integrals, bandwidth):
    """Independent explicit Gaussian mixture at four points per phase, no KDE helper."""
    records = []
    for index, phase in enumerate(PHASES):
        group = coords[np.asarray(phases) == phase]
        peak = np.unravel_index(np.argmax(density[index]), density[index].shape)
        positions = [(0, 0), (len(y)-1, len(x)-1), (len(y)//2, len(x)//2), peak]
        peak_raw = float(density[index].max() * integrals[index])
        for iy, ix in positions:
            delta = group - [x[ix], y[iy]]
            exact = float(np.exp(-np.sum(delta * delta, axis=1) / (2 * bandwidth**2)).mean()
                          / (2 * np.pi * bandwidth**2))
            stored = float(density[index, iy, ix] * integrals[index])
            tolerance = 1.05e-6 * exact + 1e-12 * peak_raw
            records.append({'phase': phase, 'iy': int(iy), 'ix': int(ix), 'exact_raw': exact,
                            'stored_raw': stored, 'absolute_error': abs(stored-exact),
                            'tolerance': tolerance, 'passed': bool(abs(stored-exact) <= tolerance)})
    return records


def figure(metadata, grids, pixels, font, seed, vmax):
    fig = render._new_figure((16, 21))
    render._label(fig, font, .045, .977, 'KDEの帯域幅による背景分布の比較', 21, weight='bold')
    render._label(fig, font, .045, .955,
                  f'seed = {seed} ／ 座標を固定・全採用クローン・n_neighbors = 15・min_dist = 0.1', 11)
    render._label(fig, font, .045, .934,
                  '密度表示の感度確認です。帯域幅を変えてもUMAP座標・配列間の関係は変わりません。', 10)
    render._label(fig, font, .045, .905, 'A  論文 Fig. 1c（参照）', 15, weight='bold')
    ax = fig.add_axes([.040, .716, .910, .174])
    ax.imshow(pixels)
    ax.set_axis_off()
    render._label(fig, font, .045, .697,
                  '出典：Masuda et al. (2026), Fig. 1c. DOI: ' + render.DOI + ' ／ 赤点は今回の比較対象外。',
                  9, color='#53606c')
    for index, factor in enumerate(FACTORS):
        bottom = .486 - index * .214
        label = '現行の帯域幅' if factor == 1 else '比較する帯域幅'
        render._label(fig, font, .045, bottom + .183,
                      f'{chr(ord("B")+index)}  KDE帯域幅倍率 = {factor:g} ／ {label}', 15, weight='bold')
        contour = render._density_row(fig, font, grids[f'h{factor:g}_seed{seed}'],
                                      metadata['observation_counts'], bottom, .148, vmax)
    render._colorbar(fig, contour, .058, .576)
    render._label(fig, font, .045, .025,
                  '2枚で軸・グリッド・色尺度を共通化。同じseedの3行は同一座標。各時点の密度積分は1。', 10)
    render._label(fig, font, .045, .010,
                  'seed間は回転・鏡映・平行移動のみで向き合わせ。論文とは座標・色尺度が別です。', 10)
    return fig


def run(pooling_dir, reference_image, output_dir):
    context, folder, prior, hashes, raw = load_inputs(pooling_dir)
    source, run_info, clones, _, _, source_hashes, projection, pm, ph, inputs = context
    reference = Path(reference_image).resolve(strict=True)
    protected = [source, folder, reference.parent]
    output = validate_output_location(Path(output_dir), inputs)
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Output must be separate from prior runs and the reference directory.')
    reference_hash = digest(reference)
    font_hash = digest(FONT)
    scripts = {str(Path(p).resolve()): digest(Path(p))
               for p in (__file__, compute.__file__, render.__file__)}
    bandwidth, seeds = float(pm['kde']['bandwidth']), list(raw)
    output.mkdir(parents=True, exist_ok=False)
    protect_output(output)
    plan = {'factors': list(FACTORS), 'seeds': seeds, 'baseline_bandwidth': bandwidth,
            'coordinate_fitting_performed': False, 'all_clones_retained': True,
            'grid_rule': 'shared; margin max(3*largest_bandwidth,5%span); spacing<=smallest_bandwidth/2; size>=96,<=2048',
            'probe_rule': 'four grid points per phase: opposite corners, center, peak',
            'probe_tolerance': '1.05e-6*exact_raw + 1e-12*panel_peak_raw; sampled display precision only',
            'production_defaults_changed': False}
    write_json(output / 'comparison_plan.json', plan)
    aligned, sets = {}, {}
    for seed, coords in raw.items():
        aligned[seed], alignment = compute.rigid_align(coords, raw[seeds[0]])
        raw_name, aligned_name = f'seed{seed}_raw.npy', f'seed{seed}_aligned.npy'
        compute.save_array(output / raw_name, coords)
        compute.save_array(output / aligned_name, aligned[seed])
        sets[str(seed)] = {'raw_coordinates_file': raw_name, 'aligned_coordinates_file': aligned_name,
                           'alignment': alignment, 'parameters': prior['fits'][f'all_seed{seed}']['parameters'],
                           'pooling_source_file': prior['fits'][f'all_seed{seed}']['raw_coordinates_file']}
    x, y = make_grid(aligned, bandwidth)
    phases = [clone['timepoint'] for clone in clones]
    fits, grids, probes = {}, {}, {}
    for seed in seeds:
        for factor in FACTORS:
            key, h = f'h{factor:g}_seed{seed}', factor * bandwidth
            print(f'KDE comparison: seed={seed}, bandwidth factor={factor:g}', flush=True)
            density, integrals = compute.fixed_density(aligned[seed], phases, x, y, h)
            filename = key + '_density.npz'
            with (output / filename).open('xb') as handle:
                np.savez_compressed(handle, x=x, y=y, densities=density)
            fits[key] = {'seed': seed, 'bandwidth_factor': factor, 'bandwidth': h,
                         'coordinate_set': str(seed), 'density_file': filename,
                         'integrals_before_normalization': dict(zip(PHASES, integrals, strict=True))}
            grids[key] = render._read_grid(output / filename, len(x))
            probes[key] = density_probes(aligned[seed], phases, x, y, density, integrals, h)
    write_json(output / 'gaussian_probes.json', probes)
    if not all(p['passed'] for items in probes.values() for p in items):
        raise ValueError('A direct Gaussian probe failed; retained artifacts need investigation.')
    metadata = {'status': 'completed', 'purpose': 'KDE bandwidth sensitivity only; no adoption',
                **plan, 'source_run': str(source), 'source_projection': str(projection),
                'source_pooling_comparison': str(folder), 'source_pooling_sha256': hashes,
                'source_sha256': source_hashes, 'projection_sha256': ph,
                'input_hashes': run_info['input_hashes'], 'observation_counts': dict(Counter(phases)),
                'coordinate_sets': sets, 'fits': fits, 'packages': pm['packages'],
                'common_kde': {'grid_size': len(x), 'kernel': 'gaussian', 'algorithm': 'ball_tree',
                               'rtol': 1e-6, 'atol': 0, 'counts_weights': False,
                               'normalization': 'per-phase unit integral on common finite grid',
                               'x_limits': x[[0,-1]].tolist(), 'y_limits': y[[0,-1]].tolist()},
                'script_sha256': scripts, 'reference_image': str(reference), 'reference_sha256': reference_hash,
                'font_sha256': font_hash, 'candidate_selection_performed': False,
                'paper_coordinate_similarity_quantified': False, 'outputs_private': True}
    vmax = max(float(grid[2].max()) for grid in grids.values())
    pixels, font = imread(reference), FontProperties(fname=str(FONT))
    for seed in seeds:
        image_path = output / f'kde_bandwidth_seed{seed}_comparison.png'
        with image_path.open('xb') as handle:
            handle.write(render._png_bytes(figure(metadata, grids, pixels, font, seed, vmax)))
    compute.verify_unchanged(context)
    _verify_sources(folder, hashes)
    if digest(reference) != reference_hash or digest(FONT) != font_hash or any(digest(Path(p)) != h for p,h in scripts.items()):
        raise ValueError('A reference, font or script changed during the comparison.')
    metadata['display'] = {'density_limits': [0.0, vmax], 'shared_axes_grid_color_across_all_six_conditions': True,
                           'equal_axis_aspect': True, 'reference_retained_without_editing': True,
                           'no_candidate_points_added': True, 'visual_inspection_required': True}
    metadata['sources_unchanged'] = True
    metadata['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.json','.npy','.npz','.png')}
    write_json(output / 'comparison_metadata.json', metadata)
    print('KDE sensitivity comparison completed; original coordinates and sources preserved.', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pooling-dir', 'reference-image', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    enable_network_guard()
    run(args.pooling_dir, args.reference_image, args.output_dir)


if __name__ == '__main__':
    main()
