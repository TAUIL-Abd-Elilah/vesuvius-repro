from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import build_sealed_heldout_scope_manifest as scope


class FakeQuadSurface:
    def __init__(self, centers: list[list[float]]):
        self._centers = np.asarray(centers, dtype=np.float64).reshape((-1, 3))
        self.calls = 0

    def quad_centers(self):
        self.calls += 1
        indices = np.zeros((len(self._centers), 2), dtype=np.int64)
        return self._centers, indices


class SealedHeldoutScopeTests(unittest.TestCase):
    def test_half_open_equivalent_includes_lower_and_excludes_exact_11500(self):
        surface = FakeQuadSurface(
            [
                [10499.999, 0.0, 0.0],
                [10500.0, 0.0, 0.0],
                [math.nextafter(11500.0, -math.inf), 0.0, 0.0],
                [11500.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(scope.count_in_window_points(surface), 2)
        self.assertEqual(surface.calls, 1)
        self.assertLess(scope.SCORE_Z_END, 11500.0)

    def test_scope_hash_is_sorted_and_preserves_zero_point_rows(self):
        rows = [
            {"patch_id": "patch-z", "n_points": 8},
            {"patch_id": "patch-a", "n_points": 0},
        ]
        expected = b"patch-a\t0\npatch-z\t8\n"
        self.assertEqual(scope.canonical_scope_bytes(rows), expected)
        self.assertEqual(
            scope.canonical_scope_sha256(rows), hashlib.sha256(expected).hexdigest()
        )

    def test_heldout_directory_names_must_exactly_match_manifest_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_root = Path(tmp) / "split"
            heldout = split_root / "heldout"
            heldout.mkdir(parents=True)
            (heldout / "patch-a").mkdir()
            (heldout / "unexpected").mkdir()
            manifest = split_root / "split_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(scope.ScopeError, "missing=1.*extra=1"):
                scope.validate_heldout_directory(
                    heldout, manifest, ["patch-a", "patch-b"]
                )

    def test_strict_loader_attempts_every_name_and_reports_any_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            heldout = Path(tmp)
            names = ["a", "b"]
            content_hashes: dict[str, str] = {}
            geometry_hashes: dict[str, str] = {}
            for name in names:
                patch = heldout / name
                patch.mkdir()
                (patch / "meta.json").write_text(
                    json.dumps({"scale": [1.0, 1.0]}), encoding="utf-8"
                )
                for coordinate in "xyz":
                    (patch / f"{coordinate}.tif").write_bytes(
                        f"{name}-{coordinate}".encode()
                    )
                content_hashes[name], geometry_hashes[name] = scope._patch_hashes(patch)

            loaded: list[str] = []

            def loader(path: Path):
                loaded.append(path.name)
                if path.name == "a":
                    raise ValueError("deliberate corrupt patch")
                return FakeQuadSurface([[11000.0, 1.0, 1.0]])

            api = scope.SpiralCheckAPI(loader, FakeQuadSurface, Path("io_tifxyz.py"))
            rows, errors = scope.load_scope_rows(
                heldout=heldout,
                names=names,
                split_document={
                    "content_sha256": content_hashes,
                    "geometry_sha256": geometry_hashes,
                },
                api=api,
            )
            self.assertEqual(loaded, names)
            self.assertEqual(rows, [{"patch_id": "b", "n_points": 1}])
            self.assertIn("deliberate corrupt patch", errors["a"])

    def test_existing_output_is_refused_before_any_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "already.json"
            output.write_text("do not replace\n", encoding="utf-8")
            with self.assertRaisesRegex(scope.ScopeError, "refusing to reuse output"):
                scope.generate(
                    spiralcheck_source=Path(tmp) / "missing-spiralcheck",
                    split_manifest=Path(tmp) / "missing-split.json",
                    heldout_patches=Path(tmp) / "missing-heldout",
                    source_manifest=Path(tmp) / "missing-source.json",
                    output=output,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "do not replace\n")

    def test_checkout_requires_clean_exact_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp)
            source = repository / "src" / "spiralcheck" / "io_tifxyz.py"
            source.parent.mkdir(parents=True)
            source.write_text("# fixture\n", encoding="utf-8")
            with mock.patch.object(
                scope,
                "_git_text",
                side_effect=[scope.EXPECTED_SPIRALCHECK_COMMIT, "?? scratch.txt"],
            ):
                with self.assertRaisesRegex(scope.ScopeError, "not clean"):
                    scope.validate_spiralcheck_checkout(repository)


if __name__ == "__main__":
    unittest.main()
