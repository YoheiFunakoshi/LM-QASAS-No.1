"""Render private UMAP-neighbor comparisons from frozen numerical artifacts.

Only supplied coordinates and densities are displayed. This script performs no
fitting, density estimation, candidate selection, alignment, or source editing.
The scientific figures and provenance from real data must remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.image import imread

import render_pooling_comparison as shared


PHASES = shared.PHASES
METADATA_NAME = "render_metadata.json"


def _remember(path, expected, sources):
    path = Path(path).resolve(strict=True)
    if (not path.is_file() or not isinstance(expected, str) or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError("An existing file and a valid SHA-256 hash are required.")
    actual = shared._digest(path)
    if actual != expected or (str(path) in sources and sources[str(path)] != actual):
        raise ValueError("A declared source or comparison artifact failed its integrity check.")
    sources[str(path)] = actual
    return path


def _verify_originals(info, sources):
    for directory_key, hashes_key in (("source_run", "source_sha256"),
                                      ("source_projection", "projection_sha256")):
        folder = Path(info[directory_key]).resolve(strict=True)
        hashes = info[hashes_key]
        if not folder.is_dir() or not isinstance(hashes, dict) or not hashes:
            raise ValueError("Source run and projection provenance must be present.")
        for name, expected in hashes.items():
            _remember(shared._contained_file(folder, name), expected, sources)
    run_path = Path(info["source_run"]).resolve() / "run_metadata.json"
    if str(run_path) not in sources:
        raise ValueError("The source run metadata requires a verified hash.")
    run = shared._read_json(run_path)
    expected = info.get("input_hashes")
    if (not isinstance(expected, dict) or set(expected) != set(PHASES)
            or run.get("input_hashes") != expected
            or set(run.get("input_paths", {})) != set(PHASES)):
        raise ValueError("Original inputs must agree with the frozen source run.")
    for phase in PHASES:
        _remember(run["input_paths"][phase], expected[phase], sources)


def _check_alignment(raw, aligned, declaration):
    if not isinstance(declaration, dict) or declaration.get("scale") != 1:
        raise ValueError("Only rigid alignment without scaling is permitted.")
    matrix = np.asarray(declaration.get("matrix"), dtype=np.float64)
    source = np.asarray(declaration.get("source_center"), dtype=np.float64)
    target = np.asarray(declaration.get("target_center"), dtype=np.float64)
    if (matrix.shape != (2, 2) or source.shape != (2,) or target.shape != (2,)
            or not all(np.isfinite(value).all() for value in (matrix, source, target))
            or not np.allclose(matrix.T @ matrix, np.eye(2), rtol=0, atol=1e-10)
            or not np.allclose(source, raw.astype(np.float64).mean(axis=0), rtol=0, atol=1e-9)):
        raise ValueError("The declared transform is not a valid centered rigid alignment.")
    recalculated = (raw.astype(np.float64) - source) @ matrix + target
    if not np.allclose(recalculated, aligned, rtol=1e-10, atol=1e-9):
        raise ValueError("Aligned coordinates differ from their declared rigid transform.")


def _checked_inputs(folder, reference, font_file):
    metadata_path = folder / "comparison_metadata.json"
    sources = {str(path): shared._digest(path) for path in (metadata_path, reference, font_file)}
    info = shared._read_json(metadata_path)
    if (info.get("status") != "completed" or info.get("all_clones_retained") is not True
            or info.get("downsampling_for_umap") is not False):
        raise ValueError("Only completed fits retaining every clone observation may be rendered.")
    counts = info.get("observation_counts")
    if (not isinstance(counts, dict) or set(counts) != set(PHASES)
            or any(type(value) is not int or value <= 0 for value in counts.values())):
        raise ValueError("Positive integer clone counts are required for all phases.")
    seeds, neighbors = info.get("seeds"), info.get("neighbors")
    if (not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or neighbors != [15, 30, 50]):
        raise ValueError("Exactly two valid seeds and the predeclared neighbor series are required.")
    kde = info.get("common_kde")
    if (not isinstance(kde, dict) or type(kde.get("grid_size")) is not int
            or kde["grid_size"] < 3 or isinstance(kde.get("bandwidth"), bool)
            or not isinstance(kde.get("bandwidth"), (int, float))
            or not math.isfinite(kde["bandwidth"]) or kde["bandwidth"] <= 0
            or not isinstance(kde.get("normalization"), str) or not kde["normalization"]
            or kde.get("counts_weights") is not False):
        raise ValueError("A shared unweighted KDE with explicit bandwidth and normalization is required.")
    expected_keys = {f"nn{number}_seed{seed}" for number in neighbors for seed in seeds}
    fits = info.get("fits")
    if not isinstance(fits, dict) or set(fits) != expected_keys:
        raise ValueError("Every combination of the neighbor series and both seeds is required.")
    _verify_originals(info, sources)
    artifacts = info.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("All saved comparison artifacts need integrity hashes.")
    checked = {name: _remember(shared._contained_file(folder, name), expected, sources)
               for name, expected in artifacts.items()}
    grids, common_axes, fixed_parameters = {}, None, None
    total = sum(counts.values())
    for number in neighbors:
        for seed in seeds:
            key = f"nn{number}_seed{seed}"
            fit = fits[key]
            if not isinstance(fit, dict) or fit.get("seed") != seed or fit.get("n_neighbors") != number:
                raise ValueError("A fit does not match its declared neighbor count and seed.")
            parameters = fit.get("parameters")
            if (not isinstance(parameters, dict) or parameters.get("n_neighbors") != number
                    or parameters.get("random_state") != seed or parameters.get("transform_seed") != seed
                    or parameters.get("min_dist") != 0.1 or parameters.get("metric") != "cosine"):
                raise ValueError("UMAP metadata differs from the declared controlled comparison.")
            remaining = {name: value for name, value in parameters.items()
                         if name not in ("n_neighbors", "random_state", "transform_seed")}
            if fixed_parameters is None:
                fixed_parameters = remaining
            elif fixed_parameters != remaining:
                raise ValueError("UMAP settings other than neighbor count and seed must remain fixed.")
            for field in ("raw_coordinates_file", "aligned_coordinates_file", "density_file"):
                if fit.get(field) not in checked:
                    raise ValueError("Every coordinate and density file requires a verified hash.")
            raw = np.load(checked[fit["raw_coordinates_file"]], allow_pickle=False)
            aligned = np.load(checked[fit["aligned_coordinates_file"]], allow_pickle=False)
            for coordinates in (raw, aligned):
                if coordinates.shape != (total, 2) or not np.isfinite(coordinates).all():
                    raise ValueError("Each coordinate matrix must include every clone observation.")
            _check_alignment(raw, aligned, fit.get("alignment"))
            grid = shared._read_grid(checked[fit["density_file"]], kde["grid_size"])
            if common_axes is None:
                common_axes = grid[:2]
            elif not all(np.array_equal(a, b) for a, b in zip(common_axes, grid[:2], strict=True)):
                raise ValueError("All six fits must use the identical density grid axes.")
            if (aligned[:, 0].min() < grid[0][0] or aligned[:, 0].max() > grid[0][-1]
                    or aligned[:, 1].min() < grid[1][0] or aligned[:, 1].max() > grid[1][-1]):
                raise ValueError("The shared axes must contain all aligned clone observations.")
            grids[key] = grid
    if (not np.allclose(kde.get("x_limits"), [common_axes[0][0], common_axes[0][-1]], rtol=0, atol=1e-12)
            or not np.allclose(kde.get("y_limits"), [common_axes[1][0], common_axes[1][-1]], rtol=0, atol=1e-12)):
        raise ValueError("Declared density limits must agree with the saved common grid.")
    return info, grids, sources


def _figure(info, grids, reference_pixels, font, seed, vmax):
    figure = shared._new_figure((16, 21))
    shared._label(figure, font, 0.045, 0.977, "UMAPの近傍数による背景分布の比較", 21, weight="bold")
    shared._label(figure, font, 0.045, 0.955,
                  f"seed = {seed}  ／  現行AbLang2・全採用クローン・cosine・min_dist = 0.1を固定", 11)
    shared._label(figure, font, 0.045, 0.934,
                  "近傍数は検討用の値です。論文の設定値として確認されたものではありません。", 10)
    shared._label(figure, font, 0.045, 0.905, "A  論文 Fig. 1c（参照）", 15, weight="bold")
    reference_axis = figure.add_axes([0.040, 0.716, 0.910, 0.174])
    reference_axis.imshow(reference_pixels)
    reference_axis.set_axis_off()
    shared._label(figure, font, 0.045, 0.697,
                  "出典：Masuda et al. (2026), Fig. 1c. DOI: " + shared.DOI + "  ／  赤点は今回の比較対象外。",
                  9, color="#53606c")
    for index, number in enumerate(info["neighbors"]):
        bottom = 0.486 - index * 0.214
        label = "現行の近傍数" if number == 15 else "比較する近傍数"
        shared._label(figure, font, 0.045, bottom + 0.183,
                      f"{chr(ord('B') + index)}  n_neighbors = {number}  ／  {label}", 15, weight="bold")
        contour = shared._density_row(figure, font, grids[f"nn{number}_seed{seed}"],
                                      info["observation_counts"], bottom, 0.148, vmax)
    shared._colorbar(figure, contour, 0.058, 0.576)
    shared._label(figure, font, 0.045, 0.025,
                  "2枚の計算図で軸・帯域幅・色尺度を共通化。回転・鏡映・平行移動で向き合わせ（拡大縮小なし）。", 10)
    shared._label(figure, font, 0.045, 0.010,
                  "各時点の密度積分は1。論文の座標・色尺度は別です。計算図に候補点を追加していません。", 10)
    return figure


def render(comparison_dir, reference_image, font_path):
    folder = Path(comparison_dir).resolve(strict=True)
    reference = Path(reference_image).resolve(strict=True)
    font_file = Path(font_path).resolve(strict=True)
    if not folder.is_dir() or not reference.is_file() or not font_file.is_file():
        raise ValueError("A result directory, reference image and font file are required.")
    if folder == reference.parent or reference.parent in folder.parents:
        raise ValueError("Comparison outputs must be outside the reference image directory.")
    # Refuse completed output before expensive validation, and refuse every image
    # that this renderer could have left from an interrupted previous attempt.
    metadata_path = folder / METADATA_NAME
    if metadata_path.exists() or list(folder.glob("neighbors_seed*_comparison.png")):
        raise FileExistsError("Comparison images or render metadata already exist; nothing was overwritten.")
    info, grids, sources = _checked_inputs(folder, reference, font_file)
    dependency = Path(shared.__file__).resolve(strict=True)
    for path in (Path(__file__).resolve(), dependency):
        sources[str(path)] = shared._digest(path)
    seeds = info["seeds"]
    targets = {f"seed{seed}": folder / f"neighbors_seed{seed}_comparison.png" for seed in seeds}
    font = FontProperties(fname=str(font_file))
    reference_pixels = imread(reference)
    vmax = max(float(grid[2].max()) for grid in grids.values())
    images = {f"seed{seed}": shared._png_bytes(_figure(info, grids, reference_pixels, font, seed, vmax))
              for seed in seeds}
    if any(shared._digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError("A source changed during rendering; images were not saved.")
    for name, payload in images.items():
        with targets[name].open("xb") as handle:
            handle.write(payload)
    if any(shared._digest(Path(name)) != expected for name, expected in sources.items()):
        raise ValueError("A source changed while saving; no successful render record was written.")
    common_grid = next(iter(grids.values()))
    metadata = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_paths": {name: str(path) for name, path in targets.items()},
        "artifact_sha256": {path.name: shared._digest(path) for path in targets.values()},
        "source_sha256": sources, "source_files_unchanged": True,
        "renderer_sha256": shared._digest(Path(__file__)),
        "renderer_dependency_sha256": {dependency.name: shared._digest(dependency)},
        "neighbors": info["neighbors"], "seeds": seeds,
        "fit_order_by_figure": {f"seed{seed}": [f"nn{n}_seed{seed}" for n in info["neighbors"]]
                                for seed in seeds},
        "display": {
            "colormap": "viridis", "contour_levels": 21, "density_limits": [0.0, vmax],
            "x_limits": [float(common_grid[0][0]), float(common_grid[0][-1])],
            "y_limits": [float(common_grid[1][0]), float(common_grid[1][-1])],
            "shared_axes_and_color_across_six_fits_and_two_figures": True,
            "equal_axis_aspect": True, "reference_image_retained_without_editing": True,
            "reference_axes_and_density_scale_independent": True,
            "candidate_or_db_points_added": False,
        },
        "common_kde": info["common_kde"],
        "normalization_verified": "unit_integral_per_phase_and_fit_on_common_finite_grid",
        "alignment": "verified_rigid_alignment_of_independent_fits_without_scaling",
        "bandwidth_provenance": "declared_by_computation_not_reestimated_by_renderer",
        "all_clone_coordinate_rows_validated": True,
        "no_fitting_or_candidate_selection_performed": True, "outputs_private": True,
        "visual_inspection_required": True,
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
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
    print("Two private neighbor comparison figures and their rendering record were saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
