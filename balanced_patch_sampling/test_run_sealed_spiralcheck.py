from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import run_sealed_spiralcheck as runner


class SealedScoreValidationTests(unittest.TestCase):
    def _scope(self) -> dict:
        return {
            "scope": {
                "patches_with_points": 1,
                "patches_skipped_zero_points": 17815,
                "total_in_window_points": 4,
                "patches": [
                    {"patch_id": "patch-a", "n_points": 4},
                    *[
                        {"patch_id": f"skip-{index}", "n_points": 0}
                        for index in range(17815)
                    ],
                ],
            }
        }

    def _report(self, fit_inputs: Path) -> dict:
        return {
            "meta": {
                "spiralcheck": runner.EXPECTED_SPIRALCHECK_VERSION,
                "variant": "plain",
                "tau": float(runner.TAU),
                "z_range": runner.Z_RANGE,
                "unseen_min_dist": float(runner.UNSEEN_MIN_DIST),
                "fit_inputs_hash_audit": "clean",
                "fit_inputs": str(fit_inputs),
                "patches_dir_listed_in_manifest": 17816,
                "manifest_n_heldout": 17816,
            },
            "intrinsic": {},
            "heldout_aggregate": {
                "z_range": [float(runner.Z_BEGIN), runner.SCORE_Z_END],
                "n_patches": 1,
                "n_patches_skipped": 17815,
                "n_points": 4,
                "evidence_leakage": {"n_input_patches": 8542},
                "unseen": {"n_points": 3},
            },
            "heldout_patches": [
                {"patch_id": "patch-a", "n_points": 4, "n_points_unseen": 3}
            ],
        }

    def test_accepts_literal_meta_and_numeric_aggregate_z_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_inputs = root / "fit"
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(self._report(fit_inputs)) + "\n", encoding="utf-8"
            )
            parsed = runner._validate_score_report(
                report_path, 17816, fit_inputs, self._scope()
            )
            self.assertEqual(parsed["meta"]["z_range"], "10500,11499.999999999998")

    def test_refuses_old_inclusive_or_wrong_meta_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_inputs = root / "fit"
            report = self._report(fit_inputs)
            report["meta"]["z_range"] = [10500.0, 11500.0]
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.RunnerError, "metadata violate"):
                runner._validate_score_report(
                    report_path, 17816, fit_inputs, self._scope()
                )

    def test_paired_scope_rejects_unseen_count_drift(self):
        fit_inputs = Path("fit")
        baseline = self._report(fit_inputs)
        treatment = self._report(fit_inputs)
        treatment["heldout_patches"][0]["n_points_unseen"] = 2
        with self.assertRaisesRegex(runner.RunnerError, "different held-out"):
            runner._validate_paired_score_scope(baseline, treatment)


if __name__ == "__main__":
    unittest.main()
