"""Render the last three private comparisons from frozen, verified artifacts only.

No model fitting, density estimation, candidate selection or source editing is
performed. Each family shares axes and colors across both seeds; the paper image
retains its own axes and color scale. Research figures must stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread

import render_epochs_comparison as guards
import render_pooling_comparison as shared

PHASES = shared.PHASES
FAMILIES = {
    'preprocessing': ('raw', 'centered', 'pca95'),
    'density': ('isotropic', 'pooled_scott'),
    'fit_unit': ('raw', 'unique'),
}
TITLES = {'preprocessing': 'UMAP前の処理', 'density': '密度の推定方法', 'fit_unit': 'UMAPで計算する点の単位'}
METADATA_NAME = 'render_metadata.json'


def _checked_inputs(folder, reference, font_file):
    metadata_path = folder / 'comparison_metadata.json'
    sources = {str(path): shared._digest(path) for path in (metadata_path, reference, font_file)}
    info = shared._read_json(metadata_path)
    if (info.get('status') != 'completed' or info.get('all_clones_retained') is not True
            or info.get('all_clones_used_for_every_fit') is not False
            or info.get('random_downsampling_for_umap') is not False):
        raise ValueError('Completed, explicitly declared clone-preserving comparisons are required.')
    counts, seeds = info.get('observation_counts'), info.get('seeds')
    if (not isinstance(counts, dict) or set(counts) != set(PHASES)
            or any(type(value) is not int or value <= 0 for value in counts.values())):
        raise ValueError('Positive integer observation counts are required for all three phases.')
    if (not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Two distinct valid integer seeds are required.')
    guards._verify_originals(info, sources)
    projection_path = Path(info['source_projection']).resolve() / 'projection_metadata.json'
    clones_path = Path(info['source_run']).resolve() / 'clones.json'
    if any(str(path) not in sources for path in (projection_path, clones_path)):
        raise ValueError('Frozen projection metadata and clone mapping require source hashes.')
    projection = shared._read_json(projection_path)
    frozen = projection.get('parameters')
    bandwidth = projection.get('kde', {}).get('bandwidth')
    if (not isinstance(frozen, dict) or frozen.get('n_epochs') != 200
            or frozen.get('init') != 'random' or frozen.get('metric') != 'cosine'
            or frozen.get('n_neighbors') != 15 or frozen.get('min_dist') != .1
            or projection.get('timepoint_counts') != counts
            or not isinstance(bandwidth, (int, float)) or isinstance(bandwidth, bool)
            or not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('The frozen baseline must declare the unchanged settings, counts and bandwidth.')
    clones = shared._read_json(clones_path)
    total = sum(counts.values())
    if (not isinstance(clones, list) or len(clones) != total
            or Counter(clone.get('timepoint') for clone in clones) != counts
            or any(type(clone.get('embedding_index')) is not int or clone['embedding_index'] < 0 for clone in clones)):
        raise ValueError('Source clone observations and embedding indices must match the declared counts.')
    indices = np.asarray([clone['embedding_index'] for clone in clones], dtype=int)
    unique_count = len(set(indices.tolist()))
    if not np.array_equal(np.unique(indices), np.arange(unique_count)):
        raise ValueError('All cached sequence rows must be represented by contiguous clone indices.')
    artifacts = info.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Every comparison artifact requires an integrity hash.')
    checked = {name: guards._remember(shared._contained_file(folder, name), value, sources)
               for name, value in artifacts.items()}
    expected_fit_keys = {f'{variant}_seed{seed}' for variant in ('raw', 'centered', 'pca95', 'unique') for seed in seeds}
    fits = info.get('fits')
    if not isinstance(fits, dict) or set(fits) != expected_fit_keys:
        raise ValueError('Four declared fit variants and both seeds are required.')
    aligned_by_key = {}
    for key, fit in fits.items():
        variant, seed = fit.get('variant'), fit.get('seed')
        if key != f'{variant}_seed{seed}' or seed not in seeds:
            raise ValueError('Each fit key must match its variant and seed.')
        if fit.get('parameters') != frozen | {'random_state': seed, 'transform_seed': seed}:
            raise ValueError('UMAP parameters must equal the frozen baseline apart from the paired seed.')
        expected_fit_count = unique_count if variant == 'unique' else total
        if fit.get('fit_observation_count') != expected_fit_count or fit.get('output_observation_count') != total:
            raise ValueError('Fit and output observation counts must identify the inverse mapping explicitly.')
        for field in ('raw_coordinates_file', 'aligned_coordinates_file'):
            if fit.get(field) not in checked:
                raise ValueError('Every coordinate file needs a verified integrity hash.')
        raw = np.load(checked[fit['raw_coordinates_file']], allow_pickle=False)
        aligned = np.load(checked[fit['aligned_coordinates_file']], allow_pickle=False)
        if any(value.shape != (total, 2) or not np.isfinite(value).all() for value in (raw, aligned)):
            raise ValueError('Every displayed coordinate array must contain all clone observations.')
        guards._check_alignment(raw, aligned, fit.get('alignment'))
        if variant == 'unique':
            if fit.get('unique_coordinates_file') not in checked:
                raise ValueError('Unique-fit coordinates require a saved, verified artifact.')
            unique = np.load(checked[fit['unique_coordinates_file']], allow_pickle=False)
            if (unique.shape != (unique_count, 2) or not np.isfinite(unique).all()
                    or not np.array_equal(raw, unique[indices])):
                raise ValueError('Inverse-mapped coordinates must exactly restore every source clone row.')
        aligned_by_key[key] = aligned
    preprocessing = info.get('preprocessing', {})
    if (type(preprocessing.get('pca_retained_components')) is not int
            or preprocessing['pca_retained_components'] < 1
            or preprocessing.get('target_variance_ratio') != .95
            or not isinstance(preprocessing.get('retained_variance_ratio'), (int, float))
            or not .95 <= preprocessing['retained_variance_ratio'] <= 1.0000000001):
        raise ValueError('PCA output requires a retained dimension and the declared 95% variance threshold.')
    families, densities = info.get('families'), info.get('densities')
    expected_density_keys = {f'{family}__{variant}_seed{seed}' for family, variants in FAMILIES.items()
                             for variant in variants for seed in seeds}
    if (not isinstance(families, dict) or set(families) != set(FAMILIES)
            or not isinstance(densities, dict) or set(densities) != expected_density_keys):
        raise ValueError('All three separate comparison families and all declared density grids are required.')
    grids = {}
    for family, variants in FAMILIES.items():
        declaration = families[family]
        if (declaration.get('variants') != list(variants) or type(declaration.get('grid_size')) is not int
                or not 3 <= declaration['grid_size'] <= 512):
            raise ValueError('Family variants and a bounded shared grid must be declared.')
        axes = None
        for seed in seeds:
            keys = [f'{family}__{variant}_seed{seed}' for variant in variants]
            if declaration.get('density_keys', {}).get(str(seed)) != keys:
                raise ValueError('Saved density order must match the declared family order and seed.')
            for variant, key in zip(variants, keys, strict=True):
                record = densities[key]
                fit_key = f'{"raw" if family == "density" else variant}_seed{seed}'
                if (record.get('family') != family or record.get('variant') != variant
                        or record.get('seed') != seed or record.get('fit_key') != fit_key
                        or not isinstance(record.get('method'), str) or not record['method']
                        or record.get('file') not in checked):
                    raise ValueError('Density grids must identify their exact fit, family and method.')
                matrix = np.asarray(record.get('bandwidth_matrix'), dtype=float)
                if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
                        or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
                        or np.linalg.eigvalsh(matrix).min() <= 0):
                    raise ValueError('Every KDE requires a positive-definite covariance matrix.')
                if variant != 'pooled_scott' and not np.allclose(matrix, np.eye(2)*bandwidth**2, rtol=0, atol=1e-12):
                    raise ValueError('The baseline numeric isotropic bandwidth must remain fixed.')
                integrals = record.get('integrals_before_normalization')
                if (not isinstance(integrals, dict) or set(integrals) != set(PHASES)
                        or any(not isinstance(n, (int, float)) or not np.isfinite(n) or n <= 0 for n in integrals.values())):
                    raise ValueError('Every phase requires a finite positive pre-normalization integral.')
                grid = shared._read_grid(checked[record['file']], declaration['grid_size'])
                if axes is None:
                    axes = grid[:2]
                elif not all(np.array_equal(a, b) for a, b in zip(axes, grid[:2], strict=True)):
                    raise ValueError('All conditions and both seeds within a family must share exact grid axes.')
                coordinates = aligned_by_key[fit_key]
                if any(coordinates[:, i].min() < axis[0] or coordinates[:, i].max() > axis[-1]
                       for i, axis in enumerate(grid[:2])):
                    raise ValueError('Family axes may not clip clone coordinates.')
                grids[key] = grid
        if (not np.allclose(declaration.get('x_limits'), [axes[0][0], axes[0][-1]], rtol=0, atol=1e-12)
                or not np.allclose(declaration.get('y_limits'), [axes[1][0], axes[1][-1]], rtol=0, atol=1e-12)):
            raise ValueError('Family axis limits must equal the saved grid limits.')
    return info, grids, sources


def _figure(info, grids, reference_pixels, font, family, seed, vmax):
    three_rows = family == 'preprocessing'
    figure = shared._new_figure((16, 21 if three_rows else 15.5))
    shared._label(figure, font, .045, .977 if three_rows else .973,
                  TITLES[family] + 'による背景分布の比較', 21, weight='bold')
    shared._label(figure, font, .045, .955 if three_rows else .948,
                  f'seed = {seed}  ／  入力条件・候補順位を維持。検討条件を論文の実設定とは扱いません。', 11)
    notes = {
        'preprocessing': '中心化でもcosine距離は変わります。中心化のみとPCAの次元削減を分けて比較します。',
        'density': '異方性と帯域幅規則を同時に変更。pooled Scottでは3時点共通の共分散Hを使います。',
        'fit_unit': 'UMAPの計算単位を変更し、作図時は元の全cloneへ戻します。候補の選び方は変更しません。',
    }
    shared._label(figure, font, .045, .934 if three_rows else .923, notes[family], 10)
    shared._label(figure, font, .045, .905 if three_rows else .890, 'A  論文 Fig. 1c（参照）', 15, weight='bold')
    reference_axis = figure.add_axes([.040, .716, .910, .174] if three_rows else [.043, .654, .902, .222])
    reference_axis.imshow(reference_pixels)
    reference_axis.set_axis_off()
    shared._label(figure, font, .045, .697 if three_rows else .635,
                  '出典：Masuda et al. (2026), Fig. 1c. DOI: ' + shared.DOI + '  ／  赤点は今回の比較対象外。',
                  9, color='#53606c')
    dimension = info['preprocessing']['pca_retained_components']
    variance = info['preprocessing']['retained_variance_ratio'] * 100
    labels = {
        'raw': '現行：元のembedding・全cloneをUMAPへ入力',
        'centered': '比較：全cloneの平均を引く（次元数は維持）',
        'pca95': f'比較：PCAで累積寄与率95%以上を保持（{dimension}次元、{variance:.2f}%）',
        'isotropic': '現行：固定の等方Gaussian KDE',
        'pooled_scott': '比較：全3時点から求めた共分散を使うpooled Scott KDE',
        'unique': '比較：CDR-H3ごとにUMAPを計算 → 元の全cloneへ逆写像',
    }
    for index, variant in enumerate(FAMILIES[family]):
        bottom = .486-index*.214 if three_rows else .365-index*.295
        heading_y, height = (bottom+.183, .148) if three_rows else (bottom+.242, .195)
        shared._label(figure, font, .045, heading_y, f'{chr(ord("B")+index)}  '+labels[variant],
                      14, weight='bold')
        contour = shared._density_row(figure, font, grids[f'{family}__{variant}_seed{seed}'],
                                      info['observation_counts'], bottom, height, vmax)
    shared._colorbar(figure, contour, .058 if three_rows else .070, .576 if three_rows else .490)
    shared._label(figure, font, .045, .025 if three_rows else .032,
                  '同じ比較の2seedで軸・色尺度を共通化。向き合わせは回転・鏡映・平行移動のみ（拡大縮小なし）。', 10)
    shared._label(figure, font, .045, .010 if three_rows else .011,
                  '密度は元の全cloneを等重みで計算し、各時点の積分を1に正規化。論文の座標・色尺度は別です。', 10)
    return figure


def render(comparison_dir, reference_image, font_path):
    folder, reference, font_file = (Path(value).resolve(strict=True)
                                    for value in (comparison_dir, reference_image, font_path))
    if not folder.is_dir() or not reference.is_file() or not font_file.is_file():
        raise ValueError('A comparison directory, original reference image and font are required.')
    if folder == reference.parent or reference.parent in folder.parents:
        raise ValueError('Outputs must be outside the reference image directory.')
    if (folder/METADATA_NAME).exists() or any(list(folder.glob(f'{family}_seed*_comparison.png')) for family in FAMILIES):
        raise FileExistsError('Saved figures or renderer record exist; nothing was overwritten.')
    info, grids, sources = _checked_inputs(folder, reference, font_file)
    dependencies = [Path(shared.__file__).resolve(), Path(guards.__file__).resolve()]
    for path in [Path(__file__).resolve(), *dependencies]:
        sources[str(path)] = shared._digest(path)
    font, reference_pixels = FontProperties(fname=str(font_file)), imread(reference)
    images, targets, displays = {}, {}, {}
    for family in FAMILIES:
        family_grids = {key: grid for key, grid in grids.items() if key.startswith(family+'__')}
        vmax = max(float(grid[2].max()) for grid in family_grids.values())
        first_grid = next(iter(family_grids.values()))
        displays[family] = {
            'density_limits': [0., vmax], 'colormap': 'viridis', 'contour_levels': 21,
            'x_limits': first_grid[0][[0, -1]].tolist(), 'y_limits': first_grid[1][[0, -1]].tolist(),
            'shared_axes_and_color_within_family_and_across_seeds': True,
        }
        for seed in info['seeds']:
            key = f'{family}_seed{seed}'
            targets[key] = folder/(key+'_comparison.png')
            images[key] = shared._png_bytes(_figure(info, grids, reference_pixels, font, family, seed, vmax))
    if any(shared._digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError('A source changed during rendering; no figures were saved.')
    for key, payload in images.items():
        with targets[key].open('xb') as handle:
            handle.write(payload)
    if any(shared._digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError('A source changed while saving; no successful record was written.')
    metadata = {
        'status': 'completed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'output_paths': {key: str(path) for key, path in targets.items()},
        'artifact_sha256': {path.name: shared._digest(path) for path in targets.values()},
        'source_sha256': sources, 'source_files_unchanged': True,
        'renderer_sha256': shared._digest(Path(__file__)),
        'renderer_dependency_sha256': {path.name: shared._digest(path) for path in dependencies},
        'display_by_family': displays, 'seeds': info['seeds'],
        'families': {family: list(variants) for family, variants in FAMILIES.items()},
        'reference_image_retained_without_editing': True, 'reference_axes_and_color_independent': True,
        'all_clone_output_coordinates_validated': True, 'unique_inverse_mapping_validated': True,
        'normalization_verified': 'unit integral for every phase on each shared family grid',
        'rigid_alignment_without_scaling_validated': True, 'no_fitting_or_candidate_selection_performed': True,
        'outputs_private': True, 'visual_inspection_required': True,
    }
    with (folder/METADATA_NAME).open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', required=True, type=Path)
    parser.add_argument('--reference-image', required=True, type=Path)
    parser.add_argument('--font-path', type=Path, default=Path('C:/Windows/Fonts/meiryo.ttc'))
    args = parser.parse_args()
    try:
        render(args.comparison_dir, args.reference_image, args.font_path)
    except Exception as error:
        print(f'Comparison rendering stopped ({type(error).__name__}); source files were not edited.', file=sys.stderr)
        return 1
    print('Six private comparison figures and their rendering record were saved.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
