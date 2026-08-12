from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
import zarr

import crossscan_scrollfiesta_adapter as A
import predict_crossscan_probability_ensemble as P
import score_crossscan_finetune as S
import verify_physical_label_semantics as V


EXECUTION_LOCK_PATH = (
    Path(A.__file__).resolve().parent
    / "results/crossscan_finetune/execution_lock.json"
)


def fixture_runtime_identity() -> dict:
    return {
        "environment": dict(A.PINNED_RUNTIME_ENVIRONMENT),
        "villa": dict(A.PINNED_VILLA),
        "determinism": dict(A.DETERMINISM_CONTRACT),
        "cuda_device": {
            "index": 0,
            "name": "fixture GPU",
            "uuid": "GPU-fixture",
            "driver_version": "fixture-driver",
            "capability": [9, 0],
            "total_memory_bytes": 24 * 1024**3,
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(A._json_bytes(value))


def write_promotion(root: Path, status: str = "POSITIVE_DEPLOYABLE") -> Path:
    primary_deltas = [0.015 + index * 0.001 for index in range(6)]
    safety_deltas = [index * 0.0001 for index in range(6)]
    value = {
        "schema_version": "crossscan-final-result-v1",
        "status": status,
        "plan_content_sha256": A.PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": A.EXECUTION_LOCK_CONTENT_SHA256,
        "pilot_verdict_content_sha256": A.PILOT_VERDICT_CONTENT_SHA256,
        "selected_steps": 4000,
        "seed_rows": [
            {
                "seed": seed,
                "primary_delta": primary_deltas[index],
                "safety_delta": safety_deltas[index],
            }
            for index, seed in enumerate(A.INFERENTIAL_SEEDS)
        ],
        "comparisons": {
            str(seed): {
                "primary": {"overall": {
                    "average_precision_delta": primary_deltas[index]
                }},
                "safety": {"overall": {
                    "average_precision_delta": safety_deltas[index]
                }},
            }
            for index, seed in enumerate(A.INFERENTIAL_SEEDS)
        },
        "gates": {
            "primary_effect": 0.01,
            "minimum_positive_seeds": 5,
            "alpha_two_sided": 0.05,
            "safety_noninferiority_margin": 0.005,
        },
        "primary_summary": S.t_summary(primary_deltas),
        "safety_summary": S.t_summary(safety_deltas),
        "figures": [{} for _ in range(8)],
    }
    value["content_sha256"] = A.content_hash(value)
    path = root / "final_result.json"
    write_json(path, value)
    return path


def write_semantic_audit(root: Path) -> Path:
    records = {}
    for scroll, expected in V.EXPECTED.items():
        records[scroll] = {
            "tar": expected["tar"],
            "extracted_tree": {"root": expected["path"], **expected["tree"]},
            "decoded_zarr": {
                "path": expected["path"],
                "shape": expected["shape"],
                "chunks": expected["chunks"],
                "dtype": "uint8",
                "compressor": "Blosc(zstd)",
                "counts": expected["counts"],
                "containment": expected["containment"],
                "fractions": V.fractions_from_counts(expected["counts"]),
            },
        }
    value = {
        "schema_version": V.SCHEMA,
        "status": "PASS",
        "created_utc": "2026-08-12T00:00:00+00:00",
        "purpose": "test exact semantic receipt",
        "upstream_census_commit": V.UPSTREAM_CENSUS_COMMIT,
        "upstream_census_url": V.UPSTREAM_CENSUS_URL,
        "correction_url": V.CORRECTION_URL,
        "records": records,
    }
    value["content_sha256"] = V.content_hash(value)
    path = root / "semantic.json"
    write_json(path, value)
    V.validate_audit_receipt(path)
    return path


def record(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return A.file_record(path, relative)


def write_baseline_model(root: Path) -> Path:
    model = root / "baseline-model"
    for expected in A.BASE_MODEL_FILES.values():
        payload = (expected["path"] + "-fixture").encode("utf-8")
        path = model / expected["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return model


def write_candidate_release(
    root: Path, promotion: Path, semantic: Path,
) -> Path:
    release = root / "candidate-release"
    models = []
    release_fold = 0
    for seed in A.INFERENTIAL_SEEDS:
        for training_fold in ("even", "odd"):
            checkpoint = record(
                release,
                f"model/fold_{release_fold}/checkpoint_final.pth",
                f"checkpoint-{release_fold}".encode("ascii"),
            )
            models.append({
                "seed": seed,
                "training_fold": training_fold,
                "release_fold": release_fold,
                "checkpoint": checkpoint,
            })
            release_fold += 1
    plans = record(release, "model/plans.json", b"plans")
    dataset = record(release, "model/dataset.json", b"dataset")
    promotion_payload = promotion.read_bytes()
    semantic_payload = semantic.read_bytes()
    promotion_record = record(
        release, "evidence/final_result.json", promotion_payload
    )
    semantic_record = record(
        release, "evidence/physical_label_semantic_audit.json", semantic_payload
    )
    execution_record = record(
        release, "evidence/execution_lock.json", EXECUTION_LOCK_PATH.read_bytes()
    )
    tooling = [
        record(release, name, Path(A.__file__).with_name(name).read_bytes())
        for name in A.INFERENCE_TOOL_NAMES
    ]
    promotion_value = json.loads(promotion_payload)
    semantic_value = json.loads(semantic_payload)
    manifest = {
        "schema_version": "crossscan-model-release-v1",
        "status": "PASS",
        "plan_content_sha256": A.PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": A.EXECUTION_LOCK_CONTENT_SHA256,
        "final_result_content_sha256": promotion_value["content_sha256"],
        "semantic_audit_content_sha256": semantic_value["content_sha256"],
        "semantic_audit_file_sha256": hashlib.sha256(semantic_payload).hexdigest(),
        "outcome": "POSITIVE_DEPLOYABLE",
        "selected_steps": 4000,
        "licenses": P.RELEASE_LICENSES,
        "ensemble": {
            "aggregation": "arithmetic mean of class probabilities",
            "fold_count": 12,
            "mirroring": False,
        },
        "models": models,
        "model_files": {"plans": plans, "dataset": dataset},
        "artifacts": [promotion_record, semantic_record, execution_record],
        "tooling": tooling,
        "reports": [],
    }
    manifest["content_sha256"] = P.content_hash(manifest)
    write_json(release / "release_manifest.json", manifest)
    return release


def write_inference_run(
    root: Path,
    role: str,
    probability: np.ndarray,
    model_source: Path,
    raw_carve: Path,
    promotion: Path,
    semantic: Path,
) -> Path:
    run = root / f"{role}-run"
    run.mkdir()
    probability_path = run / "probability.npy"
    with probability_path.open("wb") as stream:
        np.save(stream, np.asarray(probability, dtype=np.float32), allow_pickle=False)

    def copy_artifact(source: Path, relative: str, validator) -> dict:
        value, payload = validator(source)
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return A._artifact_record(relative, payload, value)

    raw_record = copy_artifact(
        raw_carve / "raw_carve_receipt.json",
        "provenance/raw_carve_receipt.json",
        lambda path: A._load_hashed_json(path, "raw"),
    )
    promotion_record = copy_artifact(
        promotion,
        "provenance/final_result.json",
        A.validate_promotion_receipt,
    )
    execution_record = copy_artifact(
        EXECUTION_LOCK_PATH,
        "provenance/execution_lock.json",
        A.validate_execution_lock,
    )
    semantic_record = copy_artifact(
        semantic,
        "provenance/physical_label_semantic_audit.json",
        A._validate_semantic_audit,
    )
    promotion_value = json.loads(promotion.read_bytes())
    semantic_value = json.loads(semantic.read_bytes())
    model_record = A.validate_model_source(
        role,
        model_source,
        promotion_content_sha256=promotion_value["content_sha256"],
        semantic_audit_content_sha256=semantic_value["content_sha256"],
    )
    release_record = None
    if role == "candidate":
        release_record = copy_artifact(
            model_source / "release_manifest.json",
            "provenance/release_manifest.json",
            lambda path: A._load_hashed_json(path, "release"),
        )
    tooling = []
    for name in A.INFERENCE_TOOL_NAMES:
        source = Path(A.__file__).with_name(name)
        relative = f"provenance/{source.name}"
        target = run / relative
        target.write_bytes(source.read_bytes())
        tooling.append(A.file_record(target, relative))
    receipt = {
        "schema_version": A.INFERENCE_RUN_SCHEMA,
        "status": "PASS",
        "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "plan_content_sha256": A.PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": A.EXECUTION_LOCK_CONTENT_SHA256,
        "runtime_identity": fixture_runtime_identity(),
        "model_role": role,
        "model_source": model_record,
        "context_box_l0_zyx": [3776, 4160, 3648, 4032, 1280, 1664],
        "retained_box_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "context_array_sha256": A.RAW_CONTEXT_ARRAY_SHA256,
        "probability_shape": list(A.SHAPE),
        "probability_dtype": "float32",
        "probability_array_sha256": A.sha256_array(probability),
        "probability_file": A.file_record(probability_path, "probability.npy"),
        "tile_step_size": 0.5,
        "gaussian": True,
        "mirroring": False,
        "normalization": A.NORMALIZATION_CONTRACT,
        "aggregation": (
            "single released m7 fold-0 class probability"
            if role == "baseline"
            else "float64 arithmetic mean of twelve class probabilities"
        ),
        "folds": [0] if role == "baseline" else list(range(12)),
        "raw_carve_receipt": raw_record,
        "promotion_receipt": promotion_record,
        "execution_lock": execution_record,
        "semantic_audit": semantic_record,
        "release_manifest": release_record,
        "tooling": tooling,
    }
    receipt["content_sha256"] = A.content_hash(receipt)
    write_json(run / "inference_receipt.json", receipt)
    A.validate_inference_run(run, model_source, raw_carve)
    return run


class LockedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_context_hash = A.RAW_CONTEXT_ARRAY_SHA256
        cls.original_cube_hashes = A.RAW_CUBE_ARRAY_SHA256
        cls.original_base_files = A.BASE_MODEL_FILES
        context = np.arange(np.prod(A.CONTEXT_SHAPE), dtype=np.uint8).reshape(
            A.CONTEXT_SHAPE
        )
        cls.context = context
        A.RAW_CONTEXT_ARRAY_SHA256 = A.sha256_array(context)
        central = context[64:320, 64:320, 64:320]
        A.RAW_CUBE_ARRAY_SHA256 = {
            name: A.sha256_array(central[z:z + 128, y:y + 128, x:x + 128])
            for (z, y, x), name in A._cube_specs()
        }
        fixture_files = {}
        for name, expected in cls.original_base_files.items():
            payload = (expected["path"] + "-fixture").encode("utf-8")
            fixture_files[name] = {
                "path": expected["path"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        A.BASE_MODEL_FILES = fixture_files

    @classmethod
    def tearDownClass(cls) -> None:
        A.RAW_CONTEXT_ARRAY_SHA256 = cls.original_context_hash
        A.RAW_CUBE_ARRAY_SHA256 = cls.original_cube_hashes
        A.BASE_MODEL_FILES = cls.original_base_files

    def probability(self, offset: float = 0.0) -> np.ndarray:
        z, y, x = np.indices(A.SHAPE, dtype=np.float32)
        return np.ascontiguousarray(
            np.clip((z + 2 * y + 3 * x) / 1530.0 + offset, 0, 1),
            dtype=np.float32,
        )

    def fixture(self, root: Path):
        raw = root / "raw-carve"
        A.carve_locked_raw(raw, _source_array=self.context)
        promotion = write_promotion(root)
        semantic = write_semantic_audit(root)
        baseline_model = write_baseline_model(root)
        candidate_release = write_candidate_release(root, promotion, semantic)
        return raw, promotion, semantic, baseline_model, candidate_release

    def test_production_raw_context_fingerprint_is_pinned(self) -> None:
        self.assertEqual(
            self.original_context_hash,
            "b654ba8428beb5f24378efb2d9e8f5d516cec60410871682bdbff5535b6b665f",
        )
        self.assertEqual(len(self.original_cube_hashes), 8)
        self.assertEqual(
            self.original_cube_hashes["03840_03712_01344.tif"],
            "039d756bff0e084a90b0e1186abfbdfb0abff2195b4156492340a4b7e9accf47",
        )

    def test_raw_carve_is_exact_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            receipt = A.carve_locked_raw(raw, _source_array=self.context)
            self.assertEqual(len(receipt["cubes"]), 8)
            cube = next((raw / "cubes_RAW").glob("*.tif"))
            value = tifffile.imread(cube)
            value.flat[0] ^= np.uint8(1)
            tifffile.imwrite(cube, value, compression=None, rowsperstrip=128)
            with self.assertRaisesRegex(ValueError, "record mismatch"):
                A.verify_raw_carve(raw)

    def test_minimal_or_negative_promotion_cannot_authorize_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            minimal = {
                "schema_version": "crossscan-final-result-v1",
                "status": "POSITIVE_DEPLOYABLE",
                "plan_content_sha256": A.PLAN_CONTENT_SHA256,
                "execution_lock_content_sha256": A.EXECUTION_LOCK_CONTENT_SHA256,
            }
            minimal["content_sha256"] = A.content_hash(minimal)
            write_json(root / "minimal.json", minimal)
            with self.assertRaisesRegex(ValueError, "pilot_verdict"):
                A.validate_promotion_receipt(root / "minimal.json")
            negative = write_promotion(root, "REGRESSION")
            with self.assertRaisesRegex(ValueError, "status"):
                A.validate_promotion_receipt(negative)

    def test_inference_role_is_derived_and_model_bytes_are_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, promotion, semantic, baseline_model, candidate_release = self.fixture(root)
            candidate = write_inference_run(
                root, "candidate", self.probability(), candidate_release,
                raw, promotion, semantic,
            )
            receipt, _ = A.validate_inference_run(candidate, candidate_release, raw)
            self.assertEqual(receipt["model_role"], "candidate")
            self.assertEqual(
                {record["path"] for record in receipt["tooling"]},
                {f"provenance/{name}" for name in A.INFERENCE_TOOL_NAMES},
            )
            checkpoint = candidate_release / "model/fold_0/checkpoint_final.pth"
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                A.validate_inference_run(candidate, candidate_release, raw)

    def test_probability_export_roundtrip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, promotion, semantic, baseline_model, _ = self.fixture(root)
            run = write_inference_run(
                root, "baseline", self.probability(), baseline_model,
                raw, promotion, semantic,
            )
            store = root / "probability.zarr"
            receipt = A.export_probability_zarr(
                run, store, model_source=baseline_model, raw_carve=raw
            )
            recovered, origin = A.read_probability_zarr(store)
            self.assertTrue(np.array_equal(recovered, self.probability()))
            self.assertEqual(origin, A.DEFAULT_ORIGIN)
            self.assertEqual(receipt["model_role"], "baseline")
            self.assertTrue(np.array_equal(zarr.open_array(store / "0", mode="r")[:], recovered))
            with self.assertRaises(FileExistsError):
                A.export_probability_zarr(
                    run, store, model_source=baseline_model, raw_carve=raw, resume=True
                )
            chunk = store / "0/0.0.0"
            payload = bytearray(chunk.read_bytes())
            payload[0] ^= 1
            chunk.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "chunk hash mismatch"):
                A.verify_probability_export(store)

    def test_masks_have_locked_semantics(self) -> None:
        value = np.zeros(A.SHAPE, dtype=np.float32)
        value.flat[:6] = [0.19, 0.20, 0.21, 0.8, 0.8, 0.1]
        fixed = A.fixed_threshold_mask(value, 0.2)
        self.assertEqual(fixed.flat[:6].tolist(), [0, 255, 255, 255, 255, 0])
        matched = A.matched_mass_mask(value, 3)
        self.assertEqual(np.flatnonzero(matched).tolist(), [2, 3, 4])

    def test_inference_probability_file_rejects_dtype_and_two_class_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(A, "SHAPE", (2, 2, 2)):
            path = Path(tmp) / "probability.npy"
            for value in (
                np.zeros((2, 2, 2), dtype=np.float64),
                np.zeros((2, 2, 2, 2), dtype=np.float32),
            ):
                with self.subTest(shape=value.shape, dtype=str(value.dtype)):
                    with path.open("wb") as stream:
                        np.save(stream, value, allow_pickle=False)
                    with self.assertRaisesRegex(ValueError, "must be float32"):
                        A._load_inference_probability(path)

    def test_three_arm_set_recomputes_masks_and_enforces_shared_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, promotion, semantic, baseline_model, candidate_release = self.fixture(root)
            baseline_run = write_inference_run(
                root, "baseline", self.probability(-0.05), baseline_model,
                raw, promotion, semantic,
            )
            candidate_run = write_inference_run(
                root, "candidate", self.probability(0.05), candidate_release,
                raw, promotion, semantic,
            )
            baseline_store = root / "baseline.zarr"
            candidate_store = root / "candidate.zarr"
            A.export_probability_zarr(
                baseline_run, baseline_store,
                model_source=baseline_model, raw_carve=raw,
            )
            A.export_probability_zarr(
                candidate_run, candidate_store,
                model_source=candidate_release, raw_carve=raw,
            )
            baseline_grid = root / "baseline-grid"
            fixed_grid = root / "candidate-fixed-grid"
            matched_grid = root / "candidate-matched-grid"
            A.materialize_scrollfiesta_grid(
                baseline_store, raw, baseline_grid, arm="baseline-fixed"
            )
            A.materialize_scrollfiesta_grid(
                candidate_store, raw, fixed_grid, arm="candidate-fixed"
            )
            A.materialize_scrollfiesta_grid(
                candidate_store, raw, matched_grid,
                arm="candidate-matched-mass",
                baseline_fixed_manifest=baseline_grid / "manifest.json",
            )
            with self.assertRaisesRegex(ValueError, "outside immutable input artifact"):
                A.verify_grid_set(
                    baseline_store, candidate_store, baseline_grid, fixed_grid,
                    matched_grid, fixed_grid / "invalidating-grid-set.json",
                )
            result = A.verify_grid_set(
                baseline_store, candidate_store, baseline_grid, fixed_grid,
                matched_grid, root / "grid_set.json",
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["recomputed_masks_equal_all_pred_tiffs"])
            pred = next((fixed_grid / "cubes_PRED").glob("*.tif"))
            value = tifffile.imread(pred)
            value.flat[0] ^= np.uint8(255)
            tifffile.imwrite(pred, value, compression=None, rowsperstrip=128)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                A.verify_scrollfiesta_grid(fixed_grid)


if __name__ == "__main__":
    unittest.main()
