"""Check AbLang2 on 10 artificial CDR-H3-like strings; no real-data input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from prepare_ablang2 import ROOT, verify_model

SEED = 20260919
LENGTHS = (5, 8, 10, 12, 14, 16, 18, 22, 32, 54)
ATOL = 1e-5
RTOL = 1e-5


def synthetic_sequences() -> list[str]:
    rng = random.Random(SEED)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return ["C" + "".join(rng.choices(alphabet, k=n - 2)) + "W" for n in LENGTHS]


def reject_network(event: str, args: tuple) -> None:
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Network is disabled during this local embedding smoke test.")


def run(model_dir: Path, output_dir: Path, device: str, batch_size: int, threads: int) -> dict:
    import numpy as np
    import torch
    import ablang2

    if batch_size < 1 or threads < 1:
        raise ValueError("batch-size and threads must be positive.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this environment has no usable CUDA device.")
    manifest = verify_model(model_dir)
    sequences = synthetic_sequences()
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    # Python socket guard supplements loading a verified local model path.
    sys.addaudithook(reject_network)
    started = time.perf_counter()
    model = ablang2.pretrained(model_to_use=str(model_dir.resolve()), device=device, ncpu=1)
    model.freeze()
    loaded = time.perf_counter()
    pairs = [[seq, ""] for seq in sequences]
    kwargs = dict(mode="seqcoding", align=False, fragmented=False, batch_size=batch_size)
    with torch.inference_mode():
        first = model(pairs, **kwargs)
        repeated = model(pairs, **kwargs)
        single_batch = model(pairs, mode="seqcoding", align=False, fragmented=False, batch_size=1)
    computed = time.perf_counter()
    checks = {
        "shape_10_by_480": list(first.shape) == [10, 480],
        "finite": bool(np.isfinite(first).all()),
        "repeat_equal": bool(np.array_equal(first, repeated)),
        "repeat_close": bool(np.allclose(first, repeated, rtol=RTOL, atol=ATOL)),
        "batch_size_invariance": bool(np.allclose(first, single_batch, rtol=RTOL, atol=ATOL)),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "synthetic_embeddings.npy", first, allow_pickle=False)
    reloaded = np.load(output_dir / "synthetic_embeddings.npy", allow_pickle=False)
    checks["saved_array_round_trip"] = bool(np.array_equal(first, reloaded))
    git_result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "synthetic_only_not_biological_validation",
        "synthetic_seed": SEED, "sequence_count": len(sequences), "sequence_lengths": list(LENGTHS),
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "ablang2", "numpy")},
        "git_base_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": manifest,
        "device": device,
        "model_parameter_dtype": str(next(model.AbLang.parameters()).dtype),
        "embedding_dtype": str(first.dtype),
        "torch_threads": threads, "batch_size": batch_size, "torch_seed": SEED,
        "mode": "official_seqcoding",
        "input_format": "[[heavy_CDR3, empty_light_chain], ...]",
        "formatted_tokens": "<heavy_CDR3>|",
        "pooling": "Mean of all non-padding formatted tokens, including <, > and |.",
        "paper_pooling_equivalence": "unconfirmed",
        "align": False, "fragmented": False,
        "python_socket_network_guard": True,
        "shape": list(first.shape), "checks": checks,
        "repeat_max_abs_difference": float(np.max(np.abs(first - repeated))),
        "batch_size_max_abs_difference": float(np.max(np.abs(first - single_batch))),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "model_load_seconds": loaded - started,
        "three_embedding_passes_seconds": computed - loaded,
        "success": all(checks.values()),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "ABLANG-2-paired")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "local_records" /
                        ("smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if "ABLANG-" not in str(args.model_dir.resolve()):
        parser.error("Local model path must include ABLANG-; run prepare_ablang2.py first.")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory.")
    try:
        result = run(args.model_dir, args.output_dir, args.device, args.batch_size, args.threads)
    except Exception as exc:
        parser.exit(1, f"Smoke test failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({key: result[key] for key in
                      ("success", "shape", "device", "packages", "checks",
                       "model_load_seconds", "three_embedding_passes_seconds")}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
