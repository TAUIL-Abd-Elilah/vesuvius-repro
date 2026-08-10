from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_native_spiral_quality as target


ROOT = Path(__file__).resolve().parent


class FakeSurface:
    def __init__(self, points, valid, meta=None, name=""):
        self.points = np.asarray(points, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.meta = meta or {}
        self.name = name

    @property
    def shape(self):
        return self.points.shape[:2]

    def normals(self):
        normal = np.zeros_like(self.points)
        normal[..., 2] = 1.0
        return normal, self.valid.copy()


def family(windings=(10, 11), shape=(32, 40)):
    result = {}
    normals = {}
    for winding in windings:
        points = np.stack(
            [np.indices(shape)[0], np.indices(shape)[1] + winding, np.indices(shape)[1] + 100],
            axis=-1,
        ).astype(np.float32)
        result[winding] = FakeSurface(points, np.ones(shape, bool), {"scale": [0.05, 0.05]})
        normal = np.zeros_like(points)
        normal[..., 2] = 1.0
        normals[winding] = normal
    return result, normals


def intrinsic_source(violation=0.01, collapsed=0.02, inflated=0.03, validity=0.9, pitch=16.0):
    n_bins = 100_000
    return {
        "winding_ids": [10, 11],
        "median_pitch": pitch,
        "n_bins_checked": n_bins,
        "n_violations": int(round(n_bins * violation)),
        "violated_bin_fraction": violation,
        "n_collapsed": int(round(n_bins * collapsed)),
        "collapsed_bin_fraction": collapsed,
        "n_inflated": int(round(n_bins * inflated)),
        "validity_per_winding": {"10": validity, "11": validity},
        "theta_bins": 48,
        "z_bins": 10,
        "worst": [],
    }


def intrinsic(**kwargs):
    return target.normalize_intrinsic_report(intrinsic_source(**kwargs), label="test")


def test_choose_sites_is_deterministic_shared_and_separated():
    base, base_n = family()
    final, final_n = family()
    first = target.choose_sites(
        base, base_n, final, final_n, seed=17, n_sites=6, rays_per_site=7, radius=2
    )
    second = target.choose_sites(
        base, base_n, final, final_n, seed=17, n_sites=6, rays_per_site=7, radius=2
    )
    assert first == second
    assert all(5 <= len(site["cells"]) <= 7 for site in first)
    centres = [(site["winding"], *site["centre"]) for site in first]
    for index, (winding, row, col) in enumerate(centres):
        for other_w, other_r, other_c in centres[index + 1 :]:
            assert other_w != winding or abs(other_r - row) > 4 or abs(other_c - col) > 4


def test_align_paired_families_uses_normalized_minimum_width_grid():
    rows = 8
    base_points = np.zeros((rows, 9, 3), dtype=np.float32)
    final_points = np.zeros((rows, 6, 3), dtype=np.float32)
    base_points[..., 0] = np.arange(rows)[:, None]
    final_points[..., 0] = np.arange(rows)[:, None]
    base_points[..., 1] = np.linspace(0.0, 1.0, 9)
    final_points[..., 1] = np.linspace(0.0, 1.0, 6)
    base = {10: FakeSurface(base_points, np.ones((rows, 9), bool), {"scale": [0.05, 0.05]})}
    final = {10: FakeSurface(final_points, np.ones((rows, 6), bool), {"scale": [0.05, 0.05]})}
    with pytest.raises(target.GateError, match="incomplete"):
        target.align_paired_families(base, final)
    base[11] = FakeSurface(base_points.copy(), np.ones((rows, 9), bool), {"scale": [0.05, 0.05]})
    final[11] = FakeSurface(final_points.copy(), np.ones((rows, 6), bool), {"scale": [0.05, 0.05]})
    aligned_base, base_normals, aligned_final, final_normals, record = (
        target.align_paired_families(base, final)
    )
    assert aligned_base[10].shape == aligned_final[10].shape == (rows, 6)
    np.testing.assert_allclose(aligned_base[10].points[..., 1], aligned_final[10].points[..., 1])
    assert base_normals[10].shape == final_normals[10].shape == (rows, 6, 3)
    assert record["format"] == target.ALIGNMENT_FORMAT
    assert record["windings"][0]["common_shape"] == [rows, 6]


def test_normalized_resampling_rejects_invalid_brackets():
    points = np.zeros((4, 5, 3), dtype=np.float32)
    points[..., 1] = np.arange(5)
    valid = np.ones((4, 5), bool)
    valid[:, 1] = False
    surface = FakeSurface(points, valid, {"scale": [0.05, 0.05]})
    aligned = target.resample_normalized_columns(surface, 4, label="test")
    assert not aligned.valid[:, 1].any()
    assert np.all(aligned.points[:, 1] == -1.0)


def test_ct_batch_planner_preserves_order_and_cap():
    rays = np.zeros((3, 2, 3), dtype=np.float64)
    rays[:, :, 0] = np.arange(3)[:, None] * 100.0 + np.arange(2)[None, :]
    plans = target.plan_ct_batches(
        {"baseline": rays, "final": rays.copy()}, (1000, 1000, 1000),
        site_id=4, max_voxels=5000,
    )
    assert [(item["cell_start"], item["cell_end"]) for item in plans] == [(0, 1), (1, 2), (2, 3)]
    assert all(item["voxels"] <= 5000 for item in plans)


def test_clustered_interval_detects_favorable_and_insufficient():
    values = {site: [0.2, 0.1, 0.3, 0.2, 0.1, 0.2] for site in range(10)}
    result = target.clustered_interval(values, seed=3)
    assert result["sufficient"] is True
    assert result["ci_lo"] > 0
    insufficient = target.clustered_interval({0: [1.0] * 60}, seed=3)
    assert insufficient["sufficient"] is False
    assert insufficient["estimate"] is None


def rows(delta_support=0.1, offset_reduction=2.0, gap_delta=0):
    output = []
    for site in range(10):
        for ray in range(6):
            base_gap = ray % 2 == 0
            final_gap = bool(int(base_gap) + gap_delta)
            output.append({
                "site_id": site,
                "baseline": {
                    "support": 0.4,
                    "offset_um": 10.0,
                    "profile_usable": True,
                    "gap_structure": base_gap,
                },
                "final": {
                    "support": 0.4 + delta_support,
                    "offset_um": 10.0 - offset_reduction,
                    "profile_usable": True,
                    "gap_structure": final_gap,
                },
            })
    return output


def test_paired_ct_summary_requires_favorable_without_adverse_interval():
    result = target.paired_ct_summary(rows())
    assert result["all_measurements_sufficient"] is True
    assert result["any_favorable_interval"] is True
    assert result["any_adverse_interval"] is False
    adverse = target.paired_ct_summary(rows(delta_support=-0.2, offset_reduction=-2.0))
    assert adverse["any_adverse_interval"] is True


def test_intrinsic_material_regression_thresholds_are_exact():
    safe = target.intrinsic_summary(intrinsic(), intrinsic(violation=0.011, collapsed=0.025))
    assert safe["no_material_regression"] is True
    bad = target.intrinsic_summary(intrinsic(), intrinsic(violation=0.01101))
    assert bad["alerts"]["radial_violation_regression"] is True
    assert bad["no_material_regression"] is False


def test_intrinsic_schema_normalizer_derives_and_cross_checks_fractions():
    source = intrinsic_source(inflated=0.03)
    normalized = target.normalize_intrinsic_report(source, label="test")
    assert "inflated_bin_fraction" not in source
    assert normalized["inflated_bin_fraction"] == 0.03
    inconsistent = intrinsic_source()
    inconsistent["collapsed_bin_fraction"] = 0.5
    with pytest.raises(target.GateError, match="disagrees with counts"):
        target.normalize_intrinsic_report(inconsistent, label="test")
    empty = intrinsic_source()
    empty["n_bins_checked"] = 0
    with pytest.raises(target.GateError, match="positive integer"):
        target.normalize_intrinsic_report(empty, label="test")


def test_intrinsic_mean_validity_is_invariant_to_json_key_order():
    baseline = intrinsic(validity=0.8)
    final = intrinsic(validity=0.9)
    reversed_baseline = dict(baseline)
    reversed_final = dict(final)
    reversed_baseline["validity_per_winding"] = dict(
        reversed(list(baseline["validity_per_winding"].items()))
    )
    reversed_final["validity_per_winding"] = dict(
        reversed(list(final["validity_per_winding"].items()))
    )
    assert target.intrinsic_summary(baseline, final) == target.intrinsic_summary(
        reversed_baseline, reversed_final
    )


def test_intrinsic_rejects_nonpositive_pitch():
    with pytest.raises(target.GateError, match="median pitch"):
        target.intrinsic_summary(intrinsic(), intrinsic(pitch=0.0))


def test_split_surface_enforces_contiguous_winding_ranges():
    points = np.zeros((8, 10, 3), dtype=np.float32)
    valid = np.ones((8, 10), dtype=bool)
    surface = FakeSurface(points, valid, {
        "scale": [0.05, 0.05],
        "component_winding_ids": [10, 11],
        "winding_column_ranges": [[0, 5], [5, 10]],
    })
    split, normals = target.split_surface(surface)
    assert sorted(split) == [10, 11]
    assert split[10].shape == (8, 5)
    assert normals[11].shape == (8, 5, 3)
    surface.meta["winding_column_ranges"] = [[0, 5], [6, 10]]
    with pytest.raises(target.GateError, match="contiguous"):
        target.split_surface(surface)


class FakeVolume:
    voxel_size_um = target.VOXEL_SIZE_UM
    shape = (1000, 1000, 1000)

    def to_level(self, points):
        return np.asarray(points) / 2.0 + 100.0

    def read_box(self, lo, hi):
        clipped_lo = np.maximum(np.floor(lo).astype(np.int64), 0)
        clipped_hi = np.minimum(np.ceil(hi).astype(np.int64) + 1, np.asarray(self.shape))
        return np.ones(tuple(clipped_hi - clipped_lo), dtype=np.uint8), clipped_lo

    @staticmethod
    def sample_box(block, lo, coords):
        return np.ones(coords.shape[:-1], dtype=np.float32) * 100


def test_evaluate_sites_records_every_profile_and_excludes_pitch():
    base, base_n = family(windings=(10,), shape=(10, 10))
    final, final_n = family(windings=(10,), shape=(10, 10))
    sites = [{"site_id": 0, "winding": 10, "centre": [5, 5], "cells": [[4, 4], [5, 5]]}]

    def support(profiles, centre):
        return np.full(len(profiles), 0.75), np.ones(len(profiles), dtype=bool)

    result, batches = target.evaluate_sites(
        sites, base, base_n, final, final_n, FakeVolume(),
        support_scores=support,
        find_sheets=lambda *args, **kwargs: np.array([0.0]),
    )
    assert len(result) == 2
    assert len(result[0]["baseline"]["profile"]) > 100
    assert result[0]["baseline"]["support"] == 0.75
    assert result[0]["final"]["offset_um"] == 0.0
    assert "pitch" not in json.dumps(result).lower()
    assert batches[0]["cell_start"] == 0
    assert batches[-1]["cell_end"] == 2


def test_native_evidence_rehashes_the_runner_tree(tmp_path: Path):
    dataset = tmp_path / "dataset"
    output = dataset / "fit"
    surface = output / "preview" / "surface"
    surface.mkdir(parents=True)
    (surface / "z.tif").write_bytes(b"z")
    manifest = output / "preview" / "manifest.json"
    manifest.write_text(json.dumps({"surface_path": str(surface)}), encoding="utf-8")
    records = []
    for file in sorted(path for path in (output / "preview").rglob("*") if path.is_file()):
        records.append({
            "path": file.relative_to(output).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": target.sha256_file(file),
        })
    preview = {
        "files": records,
        "tree_sha256": target.sha256_bytes(target.runner_canonical_json(records)),
        "manifest_path": str(manifest),
        "manifest_sha256": target.sha256_file(manifest),
        "surface_path": str(surface),
    }
    evidence = {
        "format": target.RUN_FORMAT,
        "complete": True,
        "villa_head": target.EXPECTED_HEAD,
        "request_sha256": "a" * 64,
        "effective_request": {"paths": {"output_directory": str(output)}},
        "milestones": [{"step": 1, "preview": preview}],
    }
    path = dataset / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    _, milestone = target.validate_run_evidence(
        path, dataset_root=dataset, request_sha="a" * 64, required_step=1
    )
    assert milestone["step"] == 1
    (surface / "z.tif").write_bytes(b"tampered")
    with pytest.raises(target.GateError, match="verification failed"):
        target.validate_run_evidence(
            path, dataset_root=dataset, request_sha="a" * 64, required_step=1
        )


def test_source_never_imports_withdrawn_pitch_estimator():
    source = Path(target.__file__).read_text(encoding="utf-8")
    assert "dominant_period" not in source
    assert '"pitch_estimator_used": False' in source


def test_preregistration_hashes_are_bound():
    assert target.EXPECTED_PREREG_SHA == target.sha256_file(
        ROOT / "SPIRAL_PRIZE_SCROLL_NATIVE_FIT_PREREG.md"
    )
    assert target.EXPECTED_QUALITY_PREREG_SHA == target.sha256_file(
        ROOT / "SPIRAL_PRIZE_SCROLL_QUALITY_EVALUATION_PREREG.md"
    )
    assert target.EXPECTED_ALIGNMENT_AMENDMENT_SHA == target.sha256_file(
        ROOT / "PHERC0125_QUALITY_ALIGNMENT_AMENDMENT.md"
    )
    assert target.EXPECTED_BLOCK_BATCH_AMENDMENT_SHA == target.sha256_file(
        ROOT / "PHERC0125_QUALITY_BLOCK_BATCH_AMENDMENT.md"
    )
    assert target.EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA == target.sha256_file(
        ROOT / "PHERC0125_QUALITY_INTRINSIC_SCHEMA_AMENDMENT.md"
    )
    assert target.EXPECTED_FINALIZATION_AMENDMENT_SHA == target.sha256_file(
        ROOT / "PHERC0125_QUALITY_FINALIZATION_AMENDMENT.md"
    )


def test_report_verifier_rejects_over_authorized_claim(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("materialization.json", "native_smoke_step1_300.json", "native_production_step15000.json"):
        (dataset / name).write_text("{}", encoding="utf-8")
    sites = [
        {"site_id": site_id, "cells": [[0, 0], [0, 1], [0, 2]]}
        for site_id in range(target.N_SITES)
    ]
    inputs = {
        "preregistration_sha256": target.EXPECTED_PREREG_SHA,
        "quality_preregistration_sha256": target.EXPECTED_QUALITY_PREREG_SHA,
        "quality_alignment_amendment_sha256": target.EXPECTED_ALIGNMENT_AMENDMENT_SHA,
        "quality_block_batch_amendment_sha256": target.EXPECTED_BLOCK_BATCH_AMENDMENT_SHA,
        "quality_intrinsic_schema_amendment_sha256": target.EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA,
        "quality_finalization_amendment_sha256": target.EXPECTED_FINALIZATION_AMENDMENT_SHA,
        "materialization_sha256": target.sha256_file(dataset / "materialization.json"),
        "baseline_evidence_sha256": target.sha256_file(dataset / "native_smoke_step1_300.json"),
        "final_evidence_sha256": target.sha256_file(dataset / "native_production_step15000.json"),
        "umbilicus_sha256": target.EXPECTED_REFERENCE_SHA,
        "sheetcheck_commit": target.SHEETCHECK_COMMIT,
        "spiralcheck_commit": target.SPIRALCHECK_COMMIT,
        "evaluator_sha256": target.sha256_file(Path(target.__file__)),
    }
    contract = {
        "volume": target.VOLUME, "level": target.VOLUME_LEVEL,
        "voxel_size_um": target.VOXEL_SIZE_UM, "sites": target.N_SITES,
        "rays_per_site_max": target.RAYS_PER_SITE, "site_radius_cells": target.SITE_RADIUS,
        "reach_um": target.REACH_UM, "step_vox": target.STEP_VOX,
        "seed": target.SAMPLE_SEED, "bootstraps": target.BOOTSTRAPS,
        "cluster_unit": "sampled neighbourhood",
        "statistic": "paired median for support/offset; paired mean for gap structure",
        "pitch_estimator_used": False,
        "angular_alignment": target.ALIGNMENT_CONTRACT,
        "ct_block_batching": target.BLOCK_BATCH_CONTRACT,
    }
    decisions = {
        "quantitative_ct_improvement_authorized": False,
        "pherc0211_execution_authorized": False,
        "public_accuracy_wording_authorized": False,
        "letters_or_reading_claim_authorized": True,
        "physical_winding_sense_claim_authorized": False,
        "prize_claim_authorized": False,
    }
    report = {
        "format": target.FORMAT, "complete": True, "scroll": "PHerc0125",
        "inputs": inputs, "ct_contract": contract, "sites": sites,
        "raw_samples": [{} for _ in range(60)], "decisions": decisions,
        "intrinsic_schema_normalization": target.INTRINSIC_SCHEMA_CONTRACT,
        "intrinsic": target.intrinsic_summary(intrinsic(), intrinsic()),
        "angular_alignment": {
            "format": target.ALIGNMENT_FORMAT,
            "contract": target.ALIGNMENT_CONTRACT,
            "windings": [
                {
                    "winding": winding,
                    "baseline_native_shape": [61, winding + 10],
                    "final_native_shape": [61, winding + 8],
                    "common_shape": [61, winding + 8],
                }
                for winding in range(10, 131)
            ],
        },
        "ct_batches": [
            {
                "site_id": site_id,
                "batch_index": 0,
                "cell_start": 0,
                "cell_end": 3,
                "shape": [12, 12, 12],
                "voxels": 12 ** 3,
            }
            for site_id in range(target.N_SITES)
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(target.GateError, match="over-authorizes"):
        target.verify_report(path, dataset)
