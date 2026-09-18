"""Fetch the public AbLang2 checkpoint once; never accepts repertoire data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone
import urllib.request
import uuid

URL = "https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz"
EXPECTED_BYTES = 166154117
EXPECTED_MD5 = "6425b35e9b83750fde67a6a6d240d996"
FILES = ("hparams.json", "model.pt")
ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def verify_model(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_url"] != URL or manifest["archive_md5"] != EXPECTED_MD5:
        raise ValueError("Model manifest does not match the selected official checkpoint.")
    if set(manifest["files"]) != set(FILES):
        raise ValueError("Model manifest has unexpected files.")
    for name in FILES:
        if digest(folder / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Model checksum mismatch: {name}")
    return manifest


def prepare(destination: Path) -> dict:
    destination = destination.resolve()
    if "ABLANG-" not in str(destination):
        raise ValueError("AbLang2 0.2.1 local-path loading requires 'ABLANG-' in the path.")
    if destination.exists():
        return verify_model(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep failed attempts for inspection. Never remove or overwrite an existing model.
    staging = destination.parent / f".ablang-download-{uuid.uuid4().hex}"
    staging.mkdir()
    archive = staging / "weights.tar.gz"
    request = urllib.request.Request(URL, headers={"User-Agent": "LM-QASAS-model-setup/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if archive.stat().st_size != EXPECTED_BYTES or digest(archive, "md5") != EXPECTED_MD5:
        raise ValueError("Downloaded checkpoint failed the published size/MD5 check.")
    prepared = staging / "prepared"
    prepared.mkdir()
    # Extract two named regular files, not arbitrary archive paths or links.
    with tarfile.open(archive, "r:gz") as bundle:
        for name in FILES:
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError(f"Unexpected checkpoint member: {name}")
            with bundle.extractfile(member) as source, (prepared / name).open("xb") as output:
                shutil.copyfileobj(source, output)
    hparams = json.loads((prepared / "hparams.json").read_text(encoding="utf-8"))
    if hparams.get("hidden_embed_size") != 480:
        raise ValueError("Unexpected checkpoint embedding dimension.")
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "ablang2-paired",
        "source_url": URL,
        "zenodo_record": "10185169",
        "archive_bytes": EXPECTED_BYTES,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": digest(archive),
        "code_license": "BSD-3-Clause",
        "zenodo_license_id": "bsd-3-clause-clear",
        "files": {name: {"bytes": (prepared / name).stat().st_size,
                         "sha256": digest(prepared / name)} for name in FILES},
        "hparams": hparams,
    }
    (prepared / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.rename(prepared, destination)
    # Retain the official archive locally for reproducibility.
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "ABLANG-2-paired")
    args = parser.parse_args()
    try:
        manifest = prepare(args.destination)
    except Exception as exc:
        parser.exit(1, f"Model setup failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"model": manifest["model_name"], "verified": True,
                      "embedding_dimension": manifest["hparams"]["hidden_embed_size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
