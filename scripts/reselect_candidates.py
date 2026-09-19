"""Select another number of CDR-H3 types from a completed local run, without inference."""
import argparse
from pathlib import Path
import sys

from lmqasas.pipeline import reselect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--top-n', type=int, required=True)
    args = parser.parse_args()
    try:
        folder = reselect(args.run_dir, args.top_n)
    except Exception as exc:
        print(f'Reselection failed ({type(exc).__name__}); check run status and integrity.', file=sys.stderr)
        return 1
    print(f'Selection saved: {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
