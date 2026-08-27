from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import materialize_fit_audit_view as view


class MaterializeFitAuditViewTests(unittest.TestCase):
    def _fixture(self, root: Path, *, side: str = "fit") -> dict:
        source = root / "fit"
        patch = source / "patch-a"
        patch.mkdir(parents=True)
        (patch / "meta.json").write_text('{"uuid":"patch-a"}\n', encoding="utf-8")
        for name, payload in (("x.tif", b"x"), ("y.tif", b"y"), ("z.tif", b"z")):
            (patch / name).write_bytes(payload)
        content_hash, geometry_hash = view._patch_hashes(patch)
        artifacts = {}
        for arm in ("baseline", "treatment"):
            final = root / f"{arm}-satisfied.json"
            final.write_text(
                json.dumps({"patches": [{"id": "patch-a"}]}) + "\n",
                encoding="utf-8",
            )
            rejected = root / f"{arm}-rejected.txt"
            rejected.write_text("", encoding="utf-8")
            artifacts[f"{arm}_fit_artifact"] = final
            artifacts[f"{arm}_rejected_artifact"] = rejected
        manifest = root / "split_manifest.json"
        manifest.write_text(
            json.dumps({
                "assignments": {"patch-a": side},
                "content_sha256": {"patch-a": content_hash},
                "geometry_sha256": {"patch-a": geometry_hash},
            }) + "\n",
            encoding="utf-8",
        )
        return {
            **artifacts,
            "source_fit": source,
            "split_manifest": manifest,
            "spiral_fitting": root / "unused-in-injected-replay",
            "expected_count": 1,
            "expected_id_sha256": view._id_sha256({"patch-a"}),
            "replay_document": {
                "retained_ids": {"patch-a"},
                "retained_count": 1,
                "load_errors": 0,
            },
        }

    def test_materializes_exact_hardlink_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            output = root / "audit"
            result = view.materialize(output=output, **kwargs)
            self.assertEqual(result["patch_count"], 1)
            self.assertEqual(result["patch_id_sha256"], kwargs["expected_id_sha256"])
            self.assertTrue(
                os.path.samefile(
                    kwargs["source_fit"] / "patch-a" / "z.tif",
                    output / "patch-a" / "z.tif",
                )
            )
            self.assertTrue((output / ".fit_audit_view.json").is_file())

    def test_rejected_union_is_part_of_consumed_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            kwargs["treatment_fit_artifact"].write_text(
                json.dumps({"patches": []}) + "\n", encoding="utf-8"
            )
            kwargs["treatment_rejected_artifact"].write_text(
                str(kwargs["source_fit"] / "patch-a") + "\n", encoding="utf-8"
            )
            result = view.materialize(output=root / "audit", **kwargs)
            self.assertEqual(result["patch_count"], 1)

    def test_refuses_non_fit_assignment_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root, side="heldout")
            output = root / "audit"
            with self.assertRaisesRegex(view.ViewError, "not assigned to fit"):
                view.materialize(output=output, **kwargs)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
