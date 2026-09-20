"""Render private pooling comparisons from verified, frozen numerical artifacts.

This renderer never fits a model, aligns coordinates, estimates a density, selects
candidates, or edits source data. Shared grids must already be computed from the
declared rigidly aligned independent UMAP fits. Real figures remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread
from matplotlib.ticker import MaxNLocator


PHASES = ("Pre", "Peak", "Post")
PAPER_NAME = "pooling_paper_comparison.png"
SEED_NAME = "pooling_seed_comparison.png"
METADATA_NAME = "render_metadata.json"
DOI = "10.3389/fimmu.2026.1844788"
POOLING_NAMES = {"all": "all_nonpadding", "residue": "residue_only"}
POOLING_LABELS = {
    "all": "現行：CDR-H3残基と特殊トークンを平均",
    "residue": "比較：CDR-H3残基だけを平均",
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contained_file(folder: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Artifact names must be nonempty relative paths.")
    candidate = (folder / name).resolve(strict=True)
    if folder not in candidate.parents or not candidate.is_file():
        raise ValueError("An artifact must be a file inside its result directory.")
    return candidate


def _read_grid(path: Path, grid_size: int):
    with np.load(path, allow_pickle=False) as saved:
        x, y, densities = (saved[name].copy() for name in ("x", "y", "densities"))
    if (x.shape != (grid_size,) or y.shape != (grid_size,)
            or densities.shape != (len(PHASES), grid_size, grid_size)
            or not all(np.isfinite(a).all() for a in (x, y, densities))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
            or np.any(densities < 0) or float(densities.max()) <= 0):
        raise ValueError("Density grids must be finite, ordered and nonnegative.")
    integrals = np.trapezoid(np.trapezoid(densities, x=x, axis=2), x=y, axis=1)
    if not np.allclose(integrals, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Each phase must have unit-integral density on the shared grid.")
    return x, y, densities


def _checked_inputs(folder: Path, reference: Path, font_file: Path):
    info_path = folder / "comparison_metadata.json"
    initial = {str(path): _digest(path) for path in (info_path, reference, font_file)}
    info = _read_json(info_path)
    if info.get("status") != "completed":
        raise ValueError("Only completed comparison artifacts may be rendered.")
    counts = info.get("observation_counts")
    if (not isinstance(counts, dict) or set(counts) != set(PHASES)
            or any(type(n) is not int or n <= 0 for n in counts.values())):
        raise ValueError("Positive clone observation counts are required for all phases.")
    seeds = info.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Exactly two distinct, valid integer seeds are required.")
    kde = info.get("common_kde")
    if (not isinstance(kde, dict) or type(kde.get("grid_size")) is not int
            or kde["grid_size"] < 3 or isinstance(kde.get("bandwidth"), bool)
            or not isinstance(kde.get("bandwidth"), (int, float))
            or not math.isfinite(kde["bandwidth"]) or kde["bandwidth"] <= 0
            or not isinstance(kde.get("normalization"), str) or not kde["normalization"]):
        raise ValueError("An explicit common bandwidth, grid size and normalization are required.")
    fits = info.get("fits")
    expected_keys = {f"{pool}_seed{seed}" for pool in POOLING_NAMES for seed in seeds}
    if not isinstance(fits, dict) or set(fits) != expected_keys:
        raise ValueError("Four fits, comprising both pooling definitions and seeds, are required.")
    artifacts = info.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Artifact hashes are required.")
    checked = {}
    for name, expected in artifacts.items():
        path = _contained_file(folder, name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Malformed artifact hash.")
        actual = _digest(path)
        if actual != expected:
            raise ValueError("A saved comparison artifact failed its integrity check.")
        initial[str(path)] = actual
        checked[name] = path
    grids = {}
    n_observations = sum(counts.values())
    common_axes = None
    for pool in POOLING_NAMES:
        for seed in seeds:
            key = f"{pool}_seed{seed}"
            fit = fits[key]
            if (not isinstance(fit, dict) or fit.get("seed") != seed
                    or fit.get("pooling") != POOLING_NAMES[pool]):
                raise ValueError("A fit does not match its declared seed and pooling definition.")
            for field in ("raw_coordinates_file", "aligned_coordinates_file", "density_file"):
                if fit.get(field) not in checked:
                    raise ValueError("Every coordinate and density file needs a verified hash.")
            for field in ("raw_coordinates_file", "aligned_coordinates_file"):
                coordinates = np.load(checked[fit[field]], allow_pickle=False)
                if coordinates.shape != (n_observations, 2) or not np.isfinite(coordinates).all():
                    raise ValueError("A coordinate matrix does not contain every clone observation.")
            grid = _read_grid(checked[fit["density_file"]], kde["grid_size"])
            if common_axes is None:
                common_axes = grid[:2]
            elif not all(np.array_equal(a, b) for a, b in zip(common_axes, grid[:2], strict=True)):
                raise ValueError("All four fits must use exactly the same density grid axes.")
            if (coordinates[:, 0].min() < grid[0][0] or coordinates[:, 0].max() > grid[0][-1]
                    or coordinates[:, 1].min() < grid[1][0] or coordinates[:, 1].max() > grid[1][-1]):
                raise ValueError("Shared axes must contain all aligned clone coordinates.")
            grids[key] = grid
    return info, grids, initial


def _new_figure(size):
    figure = Figure(figsize=size, dpi=180, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _label(figure, font, x, y, text, size=11, **kwargs):
    return figure.text(x, y, text, fontproperties=font, fontsize=size,
                       color=kwargs.pop("color", "#17212b"), **kwargs)


def _density_row(figure, font, grid, counts, bottom, height, vmax):
    x, y, densities = grid
    levels = np.linspace(0.0, vmax, 21)
    contour = None
    for index, phase in enumerate(PHASES):
        axis = figure.add_axes([0.062 + index * 0.285, bottom, 0.246, height])
        contour = axis.contourf(x, y, densities[index], levels=levels,
                               cmap="viridis", vmin=0.0, vmax=vmax)
        axis.contour(x, y, densities[index], levels=levels[1:-1:2],
                     colors="#364957", linewidths=0.35, alpha=0.35)
        axis.set_title(f"{phase}  |  {counts[phase]:,} clones",
                       fontproperties=font, fontsize=11, pad=7)
        axis.set(xlim=(float(x[0]), float(x[-1])), ylim=(float(y[0]), float(y[-1])),
                 xlabel="Aligned UMAP 1")
        if index == 0:
            axis.set_ylabel("Aligned UMAP 2", fontsize=10)
        axis.xaxis.label.set_fontsize(10)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.tick_params(labelsize=9, length=3)
        axis.set_aspect("equal", adjustable="box")
    return contour


def _colorbar(figure, contour, bottom, height):
    axis = figure.add_axes([0.915, bottom, 0.012, height])
    bar = figure.colorbar(contour, cax=axis)
    bar.set_label("Normalized density", fontsize=10, labelpad=8)
    bar.ax.tick_params(labelsize=9)


def _png_bytes(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    box = figure.bbox
    # Catch clipped authored notes before writing a deliverable. Visual QA is
    # still required for plot spacing, the reference image and label collisions.
    for text in figure.texts:
        extent = text.get_window_extent(renderer)
        if (extent.x0 < box.x0 - 1 or extent.y0 < box.y0 - 1
                or extent.x1 > box.x1 + 1 or extent.y1 > box.y1 + 1):
            raise ValueError("An authored figure annotation falls outside the canvas.")
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=180, facecolor="white")
    figure.clear()
    return stream.getvalue()


def render(comparison_dir, reference_image, font_path):
    folder = Path(comparison_dir).resolve(strict=True)
    reference = Path(reference_image).resolve(strict=True)
    font_file = Path(font_path).resolve(strict=True)
    if not folder.is_dir() or not reference.is_file() or not font_file.is_file():
        raise ValueError("A result directory, reference image and font file are required.")
    targets = {"paper_comparison": folder / PAPER_NAME, "seed_comparison": folder / SEED_NAME}
    metadata_path = folder / METADATA_NAME
    if any(path.exists() for path in (*targets.values(), metadata_path)):
        raise FileExistsError("Comparison images or render metadata already exist; nothing was overwritten.")
    if folder == reference.parent or reference.parent in folder.parents:
        raise ValueError("Comparison outputs must be outside the reference image directory.")
    info, grids, sources = _checked_inputs(folder, reference, font_file)
    font = FontProperties(fname=str(font_file))
    reference_pixels = imread(reference)
    counts, seeds = info["observation_counts"], info["seeds"]
    vmax = max(float(grid[2].max()) for grid in grids.values())

    paper = _new_figure((16, 15.5))
    _label(paper, font, 0.045, 0.972, "AbLang2の平均範囲による背景分布の比較", 21, weight="bold")
    _label(paper, font, 0.045, 0.948,
           "入力・モデル・UMAP設定を固定。現行の全採用クローンを表示し、候補点は重ねていません。", 11)
    _label(paper, font, 0.045, 0.914, "A  論文 Fig. 1c（参照）", 15, weight="bold")
    reference_axis = paper.add_axes([0.043, 0.665, 0.902, 0.229])
    reference_axis.imshow(reference_pixels)
    reference_axis.set_axis_off()
    _label(paper, font, 0.045, 0.649,
           "出典：Masuda et al. (2026), Fig. 1c. DOI: " + DOI + "  ／  赤点は今回の比較対象外。", 9, color="#53606c")
    _label(paper, font, 0.045, 0.609, "B  " + POOLING_LABELS["all"], 15, weight="bold")
    _label(paper, font, 0.045, 0.588, f"seed = {seeds[0]}  ／  現行条件を基準として表示", 10)
    _density_row(paper, font, grids[f"all_seed{seeds[0]}"], counts, 0.363, 0.199, vmax)
    _label(paper, font, 0.045, 0.304, "C  " + POOLING_LABELS["residue"], 15, weight="bold")
    _label(paper, font, 0.045, 0.283,
           f"seed = {seeds[0]}  ／  独立UMAPを回転・鏡映・平行移動で向き合わせ（拡大縮小なし）", 10)
    contour = _density_row(paper, font, grids[f"residue_seed{seeds[0]}"], counts, 0.068, 0.190, vmax)
    _colorbar(paper, contour, 0.068, 0.494)
    _label(paper, font, 0.045, 0.024,
           "B・Cと別紙4条件は同じ軸・帯域幅・色尺度。各時点の密度積分は1。論文とは座標・色尺度が別です。", 10)

    seed_figure = _new_figure((16, 20))
    _label(seed_figure, font, 0.045, 0.976, "平均範囲と乱数seedの影響を比較する", 21, weight="bold")
    _label(seed_figure, font, 0.045, 0.951,
           "2種類の平均範囲 × 2 seed。全採用クローン・UMAP設定を固定し、背景密度のみ表示。", 11)
    _label(seed_figure, font, 0.045, 0.928,
           "独立UMAPの向きを基準へ合わせています。回転・鏡映・平行移動のみで、拡大縮小はしていません。", 10)
    row_order = [(pool, seed) for pool in POOLING_NAMES for seed in seeds]
    for index, (pool, seed) in enumerate(row_order):
        bottom = 0.704 - index * 0.217
        heading = f"{chr(ord('A') + index)}  {POOLING_LABELS[pool]}  ／  seed = {seed}"
        _label(seed_figure, font, 0.045, bottom + 0.189, heading, 14, weight="bold")
        contour = _density_row(seed_figure, font, grids[f"{pool}_seed{seed}"], counts,
                               bottom, 0.162, vmax)
    _colorbar(seed_figure, contour, 0.053, 0.813)
    _label(seed_figure, font, 0.045, 0.014,
           "12面で軸・KDE帯域幅・色尺度を共通化。各時点の密度積分は1。色はクローン総数を示しません。", 10)

    images = {"paper_comparison": _png_bytes(paper), "seed_comparison": _png_bytes(seed_figure)}
    if any(_digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError("A source file changed during rendering; images were not saved.")
    for name, payload in images.items():
        with targets[name].open("xb") as handle:
            handle.write(payload)
    if any(_digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError("A source changed while saving; no successful render record was written.")
    common_grid = next(iter(grids.values()))
    metadata = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_paths": {name: str(path) for name, path in targets.items()},
        "artifact_sha256": {path.name: _digest(path) for path in targets.values()},
        "source_sha256": sources,
        "source_files_unchanged": True,
        "renderer_sha256": _digest(Path(__file__)),
        "seeds": seeds,
        "fit_order": [f"{pool}_seed{seed}" for pool, seed in row_order],
        "display": {
            "colormap": "viridis", "contour_levels": 21,
            "density_limits": [0.0, vmax],
            "x_limits": [float(common_grid[0][0]), float(common_grid[0][-1])],
            "y_limits": [float(common_grid[1][0]), float(common_grid[1][-1])],
            "shared_axes_and_color_across_four_fits_and_two_figures": True,
            "equal_axis_aspect": True,
            "reference_image_retained_without_editing": True,
            "reference_axes_and_density_scale_independent": True,
            "candidate_or_db_points_added": False,
        },
        "common_kde": info["common_kde"],
        "normalization_verified": "unit_integral_per_phase_and_fit_on_common_finite_grid",
        "alignment": "upstream_rigid_alignment_of_independent_fits_without_scaling",
        "alignment_and_bandwidth_provenance": "declared_by_computation_not_reestimated_by_renderer",
        "all_clone_coordinate_rows_validated": True,
        "no_fitting_or_candidate_selection_performed": True,
        "outputs_private": True,
        "visual_inspection_required": True,
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=Path("C:/Windows/Fonts/meiryo.ttc"))
    args = parser.parse_args()
    try:
        render(args.comparison_dir, args.reference_image, args.font_path)
    except Exception as error:
        print(f"Comparison rendering stopped ({type(error).__name__}); source files were not edited.",
              file=sys.stderr)
        return 1
    print("Two private comparison figures and their rendering record were saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
