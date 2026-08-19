from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import crossscan_scrollfiesta_adapter as A
import crossscan_scrollfiesta_obj as O
import run_crossscan_scrollfiesta_downstream as D


def write_obj(path: Path) -> dict:
    vertices = [
        (3850, 3722, 1354), (4086, 3958, 1590),
        (3850, 3958, 1590), (4086, 3722, 1354),
    ]
    faces = [(0, 1, 2), (0, 3, 1)]
    lines = [f"v {z} {y} {x}" for z, y, x in vertices]
    lines += ["f " + " ".join(str(i + 1) for i in face) for face in faces]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return O.audit_obj(path)


def write_report(path: Path, audit: dict) -> None:
    edge = audit["edge_audit"]
    value = {
        "cubes_processed": 8,
        "total_input_verts": audit["vertices"],
        "total_unique_verts": audit["vertices"],
        "total_input_faces": audit["faces"],
        "total_unique_faces": audit["faces"],
        "recoarsen": {"collapses": 0, "faces_in": 0, "faces_out": 0},
        "band_cvt": {
            "patches_accepted": 0, "patches_rejected": 0,
            "band_faces_in": 0, "band_faces_out": 0,
        },
        "manifold_audit": {
            "unpaired": edge["unpaired"],
            "non_manifold": edge["non_manifold"],
            "same_dir_pairs": edge["same_dir_pairs"],
            "manifold_pairs": edge["manifold_pairs"],
            "pinch_verts": 0,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def write_png(path: Path, size=(1200, 1000)) -> None:
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    pixels[:, : size[0] // 2] = 255
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")


def failure_receipt(output: Path) -> dict:
    return {
        "schema_version": D.SCHEMA,
        "status": "FAIL",
        "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "metric_lock_content_sha256": D.METRIC_LOCK_CONTENT_SHA256,
        "physical": {"pass": False},
        "scrollfiesta_gate": {"pass": False},
        "visual_evidence": {"pass": False},
        "input_integrity": {"pass": False},
        "artifact_integrity": {
            "invalid_output_entries": [], "pass": True,
        },
        "terminal_gate": {
            "physical_pass": False,
            "scrollfiesta_pass": False,
            "visuals_pass": False,
            "input_integrity_pass": False,
            "artifact_integrity_pass": True,
            "pass": False,
        },
        "claim_boundary": (
            "bounded negative result; no downstream-improvement claim is authorized"
        ),
        "files": D._hash_tree(output),
    }


def emitter_shaped_failure_receipt(output: Path) -> dict:
    receipt = failure_receipt(output)
    receipt.update({
        "created_utc": "2026-08-19T00:00:00+00:00",
        "grid_set_content_sha256": "1" * 64,
        "promotion_content_sha256": "2" * 64,
        "truth": {
            "store": r"C:\sealed\labels0139_L1.zarr",
            "box_local_l1_zyx": [192, 320, 1280, 1408, 192, 320],
            "shape": [128, 128, 128],
            "dtype": "uint8",
            "array_sha256": D.TRUTH_ARRAY_SHA256,
            "counts": D.TRUTH_COUNTS,
            "semantic_audit": {
                "bytes": 1,
                "sha256": "3" * 64,
                "content_sha256": "4" * 64,
                "file_sha256": "3" * 64,
            },
        },
        "truth_snapshot": {},
        "grid_snapshots": {},
        "binaries": {},
        "renderer": {},
        "runtime": {},
        "sanitized_environment_removed_keys": [],
        "provenance": [],
        "arms": {},
    })
    receipt["visual_evidence"] = {
        "cross_sections": [],
        "mesh_fixed_camera_all_arms": False,
        "error": "fixture",
        "pass": False,
    }
    receipt["input_integrity"] = {
        "private_snapshots_revalidated": False,
        "error": "fixture",
        "pass": False,
    }
    receipt["physical"] = {"result": None, "error": "fixture", "pass": False}
    return receipt


class DownstreamRunnerTests(unittest.TestCase):
    def test_metric_lock_and_production_tool_identities_are_pinned(self):
        value, _ = D.validate_metric_lock()
        self.assertEqual(value["content_sha256"], D.METRIC_LOCK_CONTENT_SHA256)
        self.assertEqual(D.PINNED_BINARIES["grid_pipeline.exe"]["bytes"], 414720)
        self.assertEqual(
            D.PINNED_BINARIES["grid_weld.exe"]["sha256"],
            "0ac850e70aaaccdfc932a99b9a9a5b620406702a002df02fc6a785774c98a583",
        )

    def test_exact_command_and_environment_sanitization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = D._pipeline_command(root / "bin", root / "grid", root / "out")
            self.assertEqual(command[3:13], list(D.PIPELINE_TAIL))
            self.assertEqual(command[-4], "--exe")
            self.assertEqual(command[-2], "--weld")
        with mock.patch.dict(
            os.environ,
            {"SEAM_NO_CLEANUP": "1", "VES_CVT_RATIO": "0.1",
             "PATCH_REPAIR_OFF": "1", "SAFE_TEST_KEY": "removed",
             "VESUVIUS_THREADS": "99", "MLS_RADIUS_VOX": "99",
             "PATH": "retained"},
            clear=True,
        ):
            environment, removed = D._sanitized_environment()
        self.assertEqual(environment, {"PATH": "retained"})
        self.assertEqual(
            removed,
            ["MLS_RADIUS_VOX", "PATCH_REPAIR_OFF", "SAFE_TEST_KEY",
             "SEAM_NO_CLEANUP", "VESUVIUS_THREADS", "VES_CVT_RATIO"],
        )

    def test_summary_requires_exact_eight_clean_cubes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline_summary.csv"
            rows = "".join(f"{cube},0,1.25\n" for cube in sorted(D._expected_cube_ids()))
            path.write_text("cube_id,exit_code,wall_seconds\n" + rows, encoding="utf-8")
            self.assertEqual(D._parse_summary(path)["cube_count"], 8)
            bad = rows.replace(",0,1.25", ",7,1.25", 1)
            path.write_text("cube_id,exit_code,wall_seconds\n" + bad, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not complete"):
                D._parse_summary(path)

    def _arm_fixture(self, root: Path, grid: Path) -> Path:
        output = root / "arm"
        output.mkdir()
        ids = sorted(D._expected_cube_ids())
        (output / "pipeline_summary.csv").write_text(
            "cube_id,exit_code,wall_seconds\n"
            + "".join(f"{cube},0,1.0\n" for cube in ids),
            encoding="utf-8",
        )
        (output / "rejected_cubes.txt").write_text("", encoding="utf-8")
        (output / "scrollslice.source.json").write_text(
            json.dumps({"format": "vesuvius-scrollslice-source-v1",
                        "dataset": str(grid.resolve()), "mesh_axes": "zyx",
                        "mesh": "welded.obj"}),
            encoding="utf-8",
        )
        for relative in (
            "grid_weld.log", "welded.obj.bad_edges.obj",
            "mesh_fixed_camera.log", "pipeline_driver.log",
        ):
            (output / relative).write_bytes(b"fixture\n")
        write_png(output / "mesh_fixed_camera.png")
        audit = write_obj(output / "welded.obj")
        write_report(output / "welded.obj.weld_report.json", audit)
        logs = output / "logs"
        logs.mkdir()
        for cube in ids:
            (logs / f"{cube}.log").write_text("clean\n", encoding="utf-8")
            final = output / "dump" / cube / f"{cube}_step12_final" / f"{cube}_step12_final_all.obj"
            final.parent.mkdir(parents=True)
            final.write_bytes((b"# complete fixture\n" * 10))
        return output

    def test_arm_output_requires_all_logs_meshes_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid = root / "grid"
            grid.mkdir()
            output = self._arm_fixture(root, grid)
            result = D._validate_arm_output(
                "baseline-fixed", grid, output,
                {"exit_code": 0, "error": None, "argv": []},
                {"exit_code": 0, "error": None, "argv": []},
            )
            self.assertEqual(result["status"], "PASS")
            (output / "mesh_fixed_camera.png").write_bytes(b"not a png")
            with self.assertRaisesRegex(ValueError, "decodable PNG"):
                D._validate_arm_output(
                    "baseline-fixed", grid, output,
                    {"exit_code": 0, "error": None, "argv": []},
                    {"exit_code": 0, "error": None, "argv": []},
                )
            write_png(output / "mesh_fixed_camera.png")
            (output / "rejected_cubes.txt").write_text("z03840_y03712_x01344\n")
            with self.assertRaisesRegex(ValueError, "rejection"):
                D._validate_arm_output(
                    "baseline-fixed", grid, output,
                    {"exit_code": 0, "error": None, "argv": []},
                    {"exit_code": 0, "error": None, "argv": []},
                )

    def test_fabricated_minimal_fail_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            output.mkdir()
            receipt = failure_receipt(output)
            receipt["content_sha256"] = A.content_hash(receipt)
            D._write_json_exclusive(output / "terminal_receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "receipt schema mismatch"):
                D.verify_result(output)

    def test_emitter_shaped_terminal_rejects_any_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            output.mkdir()
            artifact = output / "artifact.bin"
            artifact.write_bytes(b"original")
            receipt = emitter_shaped_failure_receipt(output)
            receipt["content_sha256"] = A.content_hash(receipt)
            D._write_json_exclusive(output / "terminal_receipt.json", receipt)
            self.assertEqual(D.verify_result(output, deep=False)["status"], "FAIL")
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                D.verify_result(output, deep=False)

    def test_self_consistent_forged_pass_requires_deep_recomputation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            output.mkdir()
            (output / "artifact.bin").write_bytes(b"forged")
            receipt = emitter_shaped_failure_receipt(output)
            receipt.update({
                "status": "PASS",
                "physical": {
                    "result": "physical_metrics.json", "error": None, "pass": True,
                },
                "scrollfiesta_gate": {"pass": True},
                "visual_evidence": {
                    "cross_sections": [],
                    "mesh_fixed_camera_all_arms": True,
                    "error": None,
                    "pass": True,
                },
                "input_integrity": {
                    "private_snapshots_revalidated": True,
                    "error": None,
                    "pass": True,
                },
                "claim_boundary": (
                    "bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement"
                ),
            })
            receipt["terminal_gate"] = {
                "physical_pass": True,
                "scrollfiesta_pass": True,
                "visuals_pass": True,
                "input_integrity_pass": True,
                "artifact_integrity_pass": True,
                "pass": True,
            }
            receipt["content_sha256"] = A.content_hash(receipt)
            D._write_json_exclusive(output / "terminal_receipt.json", receipt)
            with self.assertRaises((FileNotFoundError, ValueError)):
                D.verify_result(output)

    def test_sealed_arm_verification_crosschecks_success_and_failure_without_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result"
            result.mkdir()
            binary_dir = result / "inputs" / "tools"
            renderer_path = binary_dir / "render_mesh.py"
            binaries = {
                name: {"path": str((binary_dir / name).resolve())}
                for name in D.PINNED_BINARIES
            }
            renderer = {"path": str(renderer_path.resolve())}
            runtime = {"python_executable": {"path": os.sys.executable}}

            arm = "baseline-fixed"
            grid = result / "inputs" / "grids" / arm
            grid.mkdir(parents=True)
            fixture = self._arm_fixture(result, grid)
            arm_output = result / "arms" / arm
            arm_output.parent.mkdir()
            fixture.rename(arm_output)
            pipeline_argv = D._pipeline_command(binary_dir, grid, arm_output)
            pipeline = {
                "argv": pipeline_argv, "exit_code": 0,
                "timeout_seconds": D.PIPELINE_TIMEOUT_SECONDS, "error": None,
            }
            renderer_argv = D._renderer_command(
                renderer_path,
                arm_output / "welded.obj",
                arm_output / "mesh_fixed_camera.png",
            )
            renderer_command = {
                "argv": renderer_argv, "exit_code": 0,
                "timeout_seconds": D.RENDERER_TIMEOUT_SECONDS, "error": None,
            }
            successful = D._validate_arm_output(
                arm, grid, arm_output, pipeline, renderer_command
            )

            failed_arm = "candidate-fixed"
            failed_grid = result / "inputs" / "grids" / failed_arm
            failed_grid.mkdir(parents=True)
            failed_output = result / "arms" / failed_arm
            failed_pipeline = {
                "argv": D._pipeline_command(binary_dir, failed_grid, failed_output),
                "exit_code": 7,
                "timeout_seconds": D.PIPELINE_TIMEOUT_SECONDS,
                "error": None,
            }
            failed_renderer = {
                "argv": None, "exit_code": None,
                "timeout_seconds": D.RENDERER_TIMEOUT_SECONDS,
                "error": "welded.obj was not produced",
            }
            failed = {
                "status": "FAIL", "arm": failed_arm,
                "pipeline": failed_pipeline, "renderer": failed_renderer,
                "error": f"ValueError: {failed_arm} direct grid_pipeline command failed",
                "files": [], "invalid_entries": [],
            }
            receipt = {
                "arms": {arm: successful, failed_arm: failed},
                "runtime": runtime,
            }
            with mock.patch.object(
                D.subprocess, "Popen", side_effect=AssertionError("subprocess forbidden")
            ):
                self.assertEqual(
                    D._verify_sealed_arm(result, receipt, arm, binaries, renderer),
                    successful,
                )
                self.assertEqual(
                    D._verify_sealed_arm(
                        result, receipt, failed_arm, binaries, renderer
                    ),
                    failed,
                )

    def test_command_timeout_is_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = D._run_command(
                [os.sys.executable, "-c", "import time; time.sleep(60)"],
                Path(tmp) / "timeout.log",
                dict(os.environ),
                timeout_seconds=0.05,
            )
        self.assertIsNone(result["exit_code"])
        self.assertIn("TimeoutExpired", result["error"])


if __name__ == "__main__":
    unittest.main()
