"""Verify the locally prepared public AbLang2 checkpoint without downloading."""
import hashlib
import json
from pathlib import Path

URL = 'https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz'
EXPECTED_BYTES = 166154117
EXPECTED_MD5 = '6425b35e9b83750fde67a6a6d240d996'
FILES = ('hparams.json', 'model.pt')


def digest(path: Path, algorithm: str = 'sha256') -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def verify_model(folder: Path) -> dict:
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_url'] != URL or manifest['archive_md5'] != EXPECTED_MD5:
        raise ValueError('Model manifest does not match the selected official checkpoint.')
    if set(manifest['files']) != set(FILES):
        raise ValueError('Model manifest has unexpected files.')
    for name in FILES:
        if digest(folder / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Model checksum mismatch: {name}')
    return manifest
