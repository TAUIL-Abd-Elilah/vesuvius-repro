from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import crossscan_scrollfiesta_obj as O


def write_obj(path: Path, vertices, faces) -> None:
    lines = [f"v {z} {y} {x}" for z, y, x in vertices]
    lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_for(audit: dict) -> dict:
    edges = audit["edge_audit"]
    return {
        "cubes_processed": 8,
        "total_input_verts": audit["vertices"],
        "total_unique_verts": audit["vertices"],
        "total_input_faces": audit["faces"],
        "total_unique_faces": audit["faces"],
        "recoarsen": {"collapses": 0, "faces_in": 0, "faces_out": 0},
        "band_cvt": {
            "patches_accepted": 0,
            "patches_rejected": 0,
            "band_faces_in": 0,
            "band_faces_out": 0,
        },
        "manifold_audit": {
            "unpaired": edges["unpaired"],
            "non_manifold": edges["non_manifold"],
            "same_dir_pairs": edges["same_dir_pairs"],
            "manifold_pairs": edges["manifold_pairs"],
            "pinch_verts": 0,
        },
    }


class ObjAuditTests(unittest.TestCase):
    def test_open_square_has_four_boundary_edges_and_one_manifold_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh.obj"
            write_obj(
                path,
                [(3900, 3800, 1400), (3900, 3810, 1400),
                 (3900, 3810, 1410), (3900, 3800, 1410)],
                [(0, 1, 2), (0, 2, 3)],
            )
            audit = O.audit_obj(path, require_world_span=False)["edge_audit"]
            self.assertEqual(audit["unpaired"], 4)
            self.assertEqual(audit["manifold_pairs"], 1)
            self.assertEqual(audit["same_dir_pairs"], 0)
            self.assertEqual(audit["non_manifold"], 0)

    def test_same_direction_and_nonmanifold_edges_are_recomputed(self):
        vertices = [
            (3900, 3800, 1400), (3900, 3810, 1400),
            (3900, 3805, 1410), (3900, 3805, 1390), (3910, 3805, 1400),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            same = root / "same.obj"
            write_obj(same, vertices, [(0, 1, 2), (0, 1, 3)])
            self.assertEqual(
                O.audit_obj(same, require_world_span=False)["edge_audit"][
                    "same_dir_pairs"
                ],
                1,
            )
            nonmanifold = root / "nonmanifold.obj"
            write_obj(nonmanifold, vertices, [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
            self.assertEqual(
                O.audit_obj(nonmanifold, require_world_span=False)["edge_audit"][
                    "non_manifold"
                ],
                1,
            )

    def test_internal_seam_excludes_outer_face_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner = root / "inner.obj"
            write_obj(
                inner,
                [(3968, 3800, 1400), (3968, 3810, 1400), (3950, 3805, 1410)],
                [(0, 1, 2)],
            )
            edge = O.audit_obj(inner, require_world_span=False)["edge_audit"]
            self.assertEqual(edge["internal_seam_unpaired_edges_union"], 1)
            self.assertEqual(edge["internal_seam_unpaired_edges_by_plane"]["z"], 1)
            outer = root / "outer.obj"
            write_obj(
                outer,
                [(3968, 3712, 1400), (3968, 3712, 1410), (3950, 3712, 1405)],
                [(0, 1, 2)],
            )
            edge = O.audit_obj(outer, require_world_span=False)["edge_audit"]
            self.assertEqual(edge["internal_seam_unpaired_edges_union"], 0)

    def test_malformed_obj_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.obj"
            path.write_text("v nan 0 0\nf 1 1 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                O.parse_obj(path)
            path.write_text("v 1 2 3\nf 1 2 4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                O.parse_obj(path)

    def test_report_must_match_obj_and_locked_cube_count(self):
        vertices = [
            (3850, 3722, 1354), (4086, 3958, 1590),
            (3850, 3958, 1590), (4086, 3722, 1354),
        ]
        faces = [(0, 1, 2), (0, 3, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "welded.obj"
            report = root / "welded.obj.weld_report.json"
            write_obj(obj, vertices, faces)
            audit = O.audit_obj(obj)
            value = report_for(audit)
            report.write_text(json.dumps(value), encoding="utf-8")
            result = O.audit_scrollfiesta_obj(obj, report)
            self.assertEqual(result["status"], "PASS")
            value["manifold_audit"]["unpaired"] += 1
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unpaired differs"):
                O.audit_scrollfiesta_obj(obj, report)


if __name__ == "__main__":
    unittest.main()
