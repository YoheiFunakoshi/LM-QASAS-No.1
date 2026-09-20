"""Render private D-filter sensitivity figures from frozen analysis artifacts.

No embedding, UMAP, KDE, scoring, matching, or original-data modification occurs
here. The caller prepares the grids using the declared calculation conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.ticker import MaxNLocator

PHASES = ('Pre', 'Peak', 'Post')
MAIN_NAME = 'd_filter_independent_umap_comparison.png'
FIXED_NAME = 'd_filter_fixed_coordinate_comparison.png'
METADATA_NAME = 'render_metadata.json'
VARIANT_FILES = ('variant_coordinates.npy', 'variant_clones.json',
                 'baseline_in_variant_ids.npy', 'variant_density_grid.npz',
                 'fixed_baseline_density_grid.npz')
KEY_FIELDS = ('timepoint', 'cdr3', 'v_gene', 'j_gene', 'isotype')
DOI = '10.3389/fimmu.2026.1844788'


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _read_grid(path):
    with np.load(path, allow_pickle=False) as source:
        x, y, density = (source[name].copy() for name in ('x', 'y', 'densities'))
    if (x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3
            or density.shape != (3, len(y), len(x))
            or not all(np.isfinite(a).all() for a in (x, y, density))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
            or np.any(density < 0) or float(density.max()) <= 0):
        raise ValueError('Malformed saved density grid.')
    integrals = np.trapezoid(np.trapezoid(density, x=x, axis=2), x=y, axis=1)
    if not np.allclose(integrals, 1, atol=1e-6, rtol=1e-6):
        raise ValueError('Each phase must have unit-integral density on its saved grid.')
    return x, y, density


def _validate_counts(clones, declared):
    if not isinstance(declared, dict) or set(declared) != set(PHASES):
        raise ValueError('Counts must explicitly provide Pre, Peak and Post.')
    if any(type(value) is not int or value <= 0 for value in declared.values()):
        raise ValueError('Every phase requires a positive integer clone count.')
    if not isinstance(clones, list) or any(
            not isinstance(clone, dict) or any(field not in clone for field in KEY_FIELDS)
            for clone in clones):
        raise ValueError('Clone correspondence fields are missing.')
    observed = dict(Counter(clone['timepoint'] for clone in clones))
    if observed != declared:
        raise ValueError('Declared counts do not match the saved clone observations.')


def _new_figure(size):
    fig = Figure(figsize=size, dpi=180, facecolor='white')
    FigureCanvasAgg(fig)
    return fig


def _label(fig, font, x, y, value, size=12, **kwargs):
    return fig.text(x, y, value, fontproperties=font, fontsize=size,
                    color=kwargs.pop('color', '#17212b'), **kwargs)


def _density_row(fig, font, grid, counts, bottom, height, *, vmax, colorbar=True,
                 axis_name='UMAP', colorbar_bottom=None, colorbar_height=None):
    x, y, density = grid
    levels = np.linspace(0, vmax, 21)
    contour = None
    for index, phase in enumerate(PHASES):
        ax = fig.add_axes([.06 + index * .285, bottom, .247, height])
        contour = ax.contourf(x, y, density[index], levels=levels,
                             cmap='viridis', vmin=0, vmax=vmax)
        ax.contour(x, y, density[index], levels=levels[1:-1:2],
                   colors='#364957', linewidths=.35, alpha=.35)
        ax.set_title(f'{phase}  |  n = {counts[phase]:,} clones',
                     fontproperties=font, fontsize=11, pad=7)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(float(y[0]), float(y[-1]))
        ax.set_xlabel(axis_name + ' 1', fontsize=10, labelpad=2)
        if index == 0:
            ax.set_ylabel(axis_name + ' 2', fontsize=10, labelpad=2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=9, length=3)
        ax.set_aspect('equal', adjustable='box')
    if colorbar:
        bar_bottom = bottom if colorbar_bottom is None else colorbar_bottom
        bar_height = height if colorbar_height is None else colorbar_height
        cax = fig.add_axes([.93, bar_bottom, .012, bar_height])
        bar = fig.colorbar(contour, cax=cax)
        bar.set_label('Normalized density', fontsize=10, labelpad=8)
        bar.ax.tick_params(labelsize=8)
    return contour


def _png_bytes(figure):
    stream = io.BytesIO()
    figure.savefig(stream, format='png', dpi=180, facecolor='white')
    figure.clear()
    return stream.getvalue()


def render(output_dir, reference_image, font_path):
    """Return paths, source hashes and rendering semantics for two new PNGs.

    output_dir already contains comparison_inputs.json and the prepared variant
    artifacts. Existing PNGs are rejected, even if only one of the two exists.
    D-filter removal is a controlled alternative, not a claim of paper recovery.
    """
    folder = Path(output_dir).resolve(strict=True)
    reference = Path(reference_image).resolve(strict=True)
    font_file = Path(font_path).resolve(strict=True)
    if not folder.is_dir() or not reference.is_file() or not font_file.is_file():
        raise ValueError('Output directory, reference image and font file are required.')
    paths = {'main': folder / MAIN_NAME, 'fixed_coordinate': folder / FIXED_NAME}
    if any(path.exists() for path in (*paths.values(), folder / METADATA_NAME)):
        raise FileExistsError('Comparison output already exists; choose a new output directory.')
    if folder == reference.parent or reference.parent in folder.parents:
        raise ValueError('Comparison outputs must be outside the reference image folder.')
    info_file = folder / 'comparison_inputs.json'
    info = _read_json(info_file)
    if info.get('status') != 'completed':
        raise ValueError('Only completed comparison inputs may be rendered.')
    baseline_run = Path(info['baseline_run']).resolve(strict=True)
    baseline_projection = Path(info['baseline_projection']).resolve(strict=True)
    if baseline_projection.parent != baseline_run:
        raise ValueError('Baseline projection must belong to the declared baseline run.')
    sources = [info_file, reference, font_file,
               baseline_projection / 'density_grid.npz', baseline_run / 'clones.json',
               folder / 'variant_coordinates.npy', folder / 'variant_clones.json',
               folder / 'baseline_in_variant_ids.npy', folder / 'variant_density_grid.npz',
               folder / 'fixed_baseline_density_grid.npz']
    initial_hashes = {str(path): _digest(path) for path in sources}
    artifacts = info.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not set(VARIANT_FILES) <= artifacts.keys():
        raise ValueError('All five variant artifact hashes are required.')
    checks = {folder / name: artifacts[name] for name in VARIANT_FILES}
    source_hashes = info.get('source_hashes')
    projection_hashes = info.get('projection_hashes')
    if not isinstance(source_hashes, dict) or not isinstance(projection_hashes, dict):
        raise ValueError('Saved baseline source and projection hashes are required.')
    checks[baseline_run / 'clones.json'] = source_hashes.get('clones.json')
    checks[baseline_projection / 'density_grid.npz'] = projection_hashes.get('density_grid.npz')
    if any(not isinstance(expected, str) or len(expected) != 64
           or initial_hashes[str(path)] != expected for path, expected in checks.items()):
        raise ValueError('A saved variant or baseline artifact failed its integrity check.')
    baseline_clones = _read_json(baseline_run / 'clones.json')
    variant_clones = _read_json(folder / 'variant_clones.json')
    _validate_counts(baseline_clones, info['baseline_counts'])
    _validate_counts(variant_clones, info['variant_counts'])
    coordinates = np.load(folder / 'variant_coordinates.npy', allow_pickle=False)
    ids = np.load(folder / 'baseline_in_variant_ids.npy', allow_pickle=False)
    if (coordinates.shape != (len(variant_clones), 2)
            or not np.isfinite(coordinates).all()
            or ids.shape != (len(baseline_clones),)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(variant_clones))
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Variant coordinates or baseline subset indices are invalid.')
    if any(tuple(old[field] for field in KEY_FIELDS)
           != tuple(variant_clones[int(index)][field] for field in KEY_FIELDS)
           for old, index in zip(baseline_clones, ids, strict=True)):
        raise ValueError('Baseline subset clone keys do not match their mapped variant rows.')
    baseline_grid = _read_grid(baseline_projection / 'density_grid.npz')
    variant_grid = _read_grid(folder / 'variant_density_grid.npz')
    fixed_grid = _read_grid(folder / 'fixed_baseline_density_grid.npz')
    if not (np.array_equal(variant_grid[0], fixed_grid[0])
            and np.array_equal(variant_grid[1], fixed_grid[1])):
        raise ValueError('Fixed-coordinate comparison grids must have identical axes.')
    reference_pixels = imread(reference)
    font = FontProperties(fname=str(font_file))

    main = _new_figure((16, 16))
    _label(main, font, .05, .977, 'Dの採用条件による背景分布の比較', 21, weight='bold')
    _label(main, font, .05, .951,
           '論文参照と、D以外の入力条件を固定した2つの解析。中段・下段は候補点を描かず背景のみ表示。', 11)
    _label(main, font, .05, .918, 'A  論文 Fig. 1c（参照画像）', 15, weight='bold')
    _label(main, font, .05, .896,
           '参照画像内の赤点はCoV-AbDab関連配列。中段・下段と同じ強調対象ではありません。', 10)
    ref_ax = main.add_axes([.045, .662, .90, .217])
    ref_ax.imshow(reference_pixels)
    ref_ax.set_axis_off()
    _label(main, font, .05, .637,
           '出典：Masuda et al., Frontiers in Immunology (2026), Fig. 1c. DOI: ' + DOI, 9, color='#53606c')
    _label(main, font, .05, .604, 'B  旧条件：Dの注釈・機能ラベルを検査', 15, weight='bold')
    _label(main, font, .05, .583,
           '既存のUMAP座標・KDEをそのまま使用。nは各時点のclone観測数。', 10)
    _density_row(main, font, baseline_grid, info['baseline_counts'], .358, .205,
                 vmax=float(baseline_grid[2].max()))
    _label(main, font, .05, .305, 'C  比較条件：Dの注釈・機能による除外をしない', 15, weight='bold')
    _label(main, font, .05, .284,
           '保存embeddingを再利用し、追加配列をAbLang2処理。全3時点のUMAP・KDEを再計算。', 10)
    _density_row(main, font, variant_grid, info['variant_counts'], .061, .205,
                 vmax=float(variant_grid[2].max()))
    _label(main, font, .05, .018,
           '各段の3時点は共通軸・共通色尺度。段の間は独立したUMAP・密度尺度であり、座標と色値は直接比較できません。', 10)

    fixed = _new_figure((16, 11.5))
    _label(fixed, font, .05, .966, '同じUMAP座標で比べる：旧条件の集合とDを無視した全集合', 19, weight='bold')
    _label(fixed, font, .05, .93,
           '下段の全集合で計算したUMAPへ両方を配置し、同じKDE帯域幅・格子・色尺度で表示。', 11)
    _label(fixed, font, .05, .895,
           '上段は旧条件で採用されたcloneのsubsetを新座標から取り出した再表示です。旧UMAPの計算結果そのものではありません。', 10, weight='bold')
    _label(fixed, font, .05, .851, 'A  旧条件で採用されたcloneのみ（新しい共通座標上）', 14, weight='bold')
    fixed_vmax = max(float(fixed_grid[2].max()), float(variant_grid[2].max()))
    _density_row(fixed, font, fixed_grid, info['baseline_counts'], .555, .255,
                 vmax=fixed_vmax, colorbar=False, axis_name='Shared UMAP')
    _label(fixed, font, .05, .479, 'B  Dの注釈・機能による除外をしない全集合（同じ座標）', 14, weight='bold')
    _density_row(fixed, font, variant_grid, info['variant_counts'], .184, .255,
                 vmax=fixed_vmax, colorbar=True, axis_name='Shared UMAP',
                 colorbar_bottom=.184, colorbar_height=.626)
    _label(fixed, font, .05, .104,
           '6面で座標・格子・帯域幅・色尺度を統一。各面の密度は、その集合・時点のcloneを等重みとして積分1に正規化。', 11)
    _label(fixed, font, .05, .071,
           '色はclone総数の差を表しません。候補点やCoV-AbDabとの照合結果は描いていません。', 11)
    _label(fixed, font, .05, .034,
           '方法の参照：Masuda et al. (2026), DOI: ' + DOI + '。本図は入力条件の比較で、原図の復元を示すものではありません。', 9, color='#53606c')

    images = {'main': _png_bytes(main), 'fixed_coordinate': _png_bytes(fixed)}
    if any(_digest(path) != value for path, value in initial_hashes.items()):
        raise ValueError('An input artifact changed during rendering; comparison images were not saved.')
    # Each write is exclusive. No existing PNG, source, or saved analysis is overwritten.
    for name, payload in images.items():
        with paths[name].open('xb') as handle:
            handle.write(payload)
    return {
        'status': 'completed',
        'main_path': str(paths['main']),
        'fixed_coordinate_path': str(paths['fixed_coordinate']),
        'artifact_sha256': {path.name: _digest(path) for path in paths.values()},
        'source_sha256': initial_hashes,
        'source_files_unchanged': True,
        'candidate_or_db_points_added': False,
        'main_rows_use_independent_umap_scales': True,
        'fixed_comparison_uses_variant_coordinates': True,
        'fixed_comparison_has_shared_axes_and_color_scale': True,
        'fixed_kde_bandwidth_equality': 'declared_by_upstream_calculation_not_reestimated_here',
        'density_normalization': 'each_phase_and_input_set_integral_one_on_saved_grid',
        'renderer_sha256': _digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--reference-image', type=Path, required=True)
    parser.add_argument('--font-path', type=Path, default=Path('C:/Windows/Fonts/meiryo.ttc'))
    args = parser.parse_args()
    try:
        result = render(args.comparison_dir, args.reference_image, args.font_path)
        with (args.comparison_dir / METADATA_NAME).open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    except Exception as exc:
        print(f'Comparison rendering stopped ({type(exc).__name__}); no source-writing operations were performed.',
              file=sys.stderr)
        return 1
    print('Two comparison figures and their metadata were saved locally.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
