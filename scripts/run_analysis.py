"""Run local Pre/Peak/Post analysis; originals are read-only and outputs are private."""
import argparse
from pathlib import Path
import sys
import time

from lmqasas.embeddings import enable_network_guard
from lmqasas.pipeline import run_analysis

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ('pre', 'peak', 'post'):
        parser.add_argument('--' + phase, type=Path, required=True)
    parser.add_argument('--subject', required=True, help='Local pseudonymous subject label')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/ABLANG-2-paired')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--top-n', type=int, default=1000)
    parser.add_argument('--clusters', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    last = [0.0, None]

    def progress(stage, fraction):
        now = time.monotonic()
        if stage != last[1] or now - last[0] >= 15 or fraction == 1:
            print(f'{stage}: {fraction:.0%}', flush=True)
            last[:] = [now, stage]

    # Load numerical libraries before guarding, then block Python socket calls.
    import ablang2  # noqa: F401
    import torch  # noqa: F401
    enable_network_guard()
    try:
        folder = run_analysis({'Pre': args.pre, 'Peak': args.peak, 'Post': args.post},
                              args.subject, args.model_dir, args.output_root, top_n=args.top_n,
                              n_clusters=args.clusters, epsilon=args.epsilon, seed=args.seed,
                              n_init=args.n_init, batch_size=args.batch_size, threads=args.threads,
                              progress=progress, network_guard=True)
    except KeyboardInterrupt:
        print('Analysis interrupted. Partial outputs are not completed results.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Our validation errors do not contain sequences; third-party tracebacks may.
        if isinstance(exc, (ValueError, FileExistsError)):
            print(f'Analysis failed: {exc}', file=sys.stderr)
        else:
            print(f'Analysis failed ({type(exc).__name__}). Check local inputs/settings.', file=sys.stderr)
        return 1
    print(f'Results saved: {folder}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
