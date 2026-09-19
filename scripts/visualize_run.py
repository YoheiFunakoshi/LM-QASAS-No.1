"""Create a private shared UMAP figure, or reuse a projection for another Top N."""
import argparse
from pathlib import Path
import sys

from lmqasas.visualization import create_projection, render_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--projection-dir', type=Path, help='Reuse this projection without refitting UMAP')
    parser.add_argument('--selection-dir', type=Path, help='Default: selection_initial in the run')
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--n-epochs', type=int, default=200)
    args = parser.parse_args()

    def progress(stage, fraction):
        print(f'{stage}: {fraction:.0%}', flush=True)

    try:
        projection = args.projection_dir or create_projection(
            args.run_dir, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
            seed=args.seed, n_epochs=args.n_epochs, progress=progress)
        view = render_selection(args.run_dir, projection, args.selection_dir)
    except KeyboardInterrupt:
        print('Visualization interrupted; partial results are not completed views.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Visualization failed ({type(exc).__name__}); check run integrity and settings.', file=sys.stderr)
        return 1
    print(f'Projection saved: {projection}', flush=True)
    print(f'Figure saved: {view / "comparison.png"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
