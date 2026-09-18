"""Synthetic files only: check corrupted checkpoint rejection and network guard."""
import json
from pathlib import Path
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ablang2 as prepare
import smoke_ablang2 as smoke


class ModelPreparationTests(unittest.TestCase):
    def test_changed_model_is_rejected_without_overwriting(self):
        folder = Path.cwd() / (".test-model-" + uuid.uuid4().hex)
        folder.mkdir()
        try:
            for name in prepare.FILES:
                (folder / name).write_bytes(b"synthetic test artifact")
            manifest = {
                "source_url": prepare.URL, "archive_md5": prepare.EXPECTED_MD5,
                "files": {name: {"sha256": prepare.digest(folder / name)}
                          for name in prepare.FILES},
            }
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(prepare.verify_model(folder), manifest)
            changed = b"synthetic changed artifact"
            (folder / "model.pt").write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare.verify_model(folder)
            self.assertEqual((folder / "model.pt").read_bytes(), changed)
        finally:
            for name in (*prepare.FILES, "manifest.json"):
                (folder / name).unlink(missing_ok=True)
            folder.rmdir()

    def test_smoke_guard_rejects_socket_network_events(self):
        for event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                smoke.reject_network(event, ())
        smoke.reject_network("open", ())
        self.assertEqual([len(seq) for seq in smoke.synthetic_sequences()], list(smoke.LENGTHS))


if __name__ == "__main__":
    unittest.main()
