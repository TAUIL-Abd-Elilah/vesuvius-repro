#!/usr/bin/env python3
"""Score the frozen pilot and six-seed cross-scan physical fine-tuning experiment."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import crossscan_finetune as C
import run_crossscan_finetune as R


_TRAINING_CHECKPOINT_SHA_CACHE: dict[tuple[str, str, str, int, int, str], str] = {}


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool).ravel()
    p = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != p.shape or y.size == 0:
        raise ValueError(f"AP arrays must be nonempty and shape-matched: {y.shape}, {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("AP scores contain nonfinite values")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("AP requires at least one positive")
    order = np.argsort(-p, kind="stable")
    y = y[order]
    p = p[order]
    cumulative_true = np.cumsum(y, dtype=np.int64)
    cumulative_false = np.cumsum(~y, dtype=np.int64)
    last_at_score = np.r_[np.flatnonzero(p[1:] != p[:-1]), p.size - 1]
    tp = cumulative_true[last_at_score].astype(np.float64)
    fp = cumulative_false[last_at_score].astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def truth_masks(bits: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(bits)
    if value.shape != (64, 64, 64) or value.dtype.kind not in "ui":
        raise ValueError(f"unexpected physical truth array {value.shape} {value.dtype}")
    valid = (value & 1) != 0
    material = (value & 2) != 0
    recto = valid & ((value & 8) != 0)
    negative = valid & ~material & ~recto
    supervised = recto | negative
    return {
        "valid": valid,
        "positive": recto,
        "negative": negative,
        "supervised": supervised,
        "ignored_valid": valid & ~supervised,
    }


def stable_top_n(scores: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).ravel()
    if not 0 <= n <= values.size:
        raise ValueError(f"top-n {n} outside [0,{values.size}]")
    selected = np.zeros(values.size, dtype=bool)
    if n:
        selected[np.argsort(-values, kind="stable")[:n]] = True
    return selected


def binary_metrics(predicted: np.ndarray, positive: np.ndarray,
                   supervised: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(predicted, dtype=bool)
    pos = np.asarray(positive, dtype=bool)
    sup = np.asarray(supervised, dtype=bool)
    if not (pred.shape == pos.shape == sup.shape):
        raise ValueError("binary metric arrays must be shape-matched")
    tp = int((pred & pos).sum())
    fp = int((pred & sup & ~pos).sum())
    fn = int((~pred & pos).sum())
    selected_ignored = int((pred & ~sup).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "true_positive": tp,
        "false_positive_supervised": fp,
        "false_negative": fn,
        "selected_ignored": selected_ignored,
        "recall": float(recall),
        "precision": float(precision),
        "dice": float(dice),
    }


def load_truth(
    data_root: Path, case_id: str, plan: dict[str, Any] | None = None,
    lock: dict[str, Any] | None = None,
) -> np.ndarray:
    if plan is None or lock is None:
        raise ValueError("truth loading requires the frozen plan and execution lock")
    _, value, _ = R.verify_evaluation_case(
        plan, lock, data_root, case_id, verify_ct=False
    )
    assert value is not None
    truth_masks(value)
    return value


def training_checkpoint_sha256(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any],
    seed: int, steps: int, fold: str,
) -> str:
    key = (
        str(data_root.resolve()), plan["content_sha256"], lock["content_sha256"],
        seed, steps, fold,
    )
    if key not in _TRAINING_CHECKPOINT_SHA_CACHE:
        training, _ = R.load_training_receipt(
            plan, lock, data_root, seed, steps, fold
        )
        _TRAINING_CHECKPOINT_SHA_CACHE[key] = training["checkpoint"]["sha256"]
    return _TRAINING_CHECKPOINT_SHA_CACHE[key]


def load_prediction(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any],
    case_id: str, kind: str, scope: str, seed: int | None = None,
    steps: int | None = None, fold: str | None = None,
) -> np.ndarray:
    if kind == "initial":
        checkpoint_sha256 = plan["inputs"]["model"][
            "fold_0/checkpoint_best.pth"
        ]["sha256"]
    elif kind == "finetuned":
        if seed is None or steps is None or fold is None:
            raise ValueError("fine-tuned prediction identity is incomplete")
        checkpoint_sha256 = training_checkpoint_sha256(
            data_root, plan, lock, seed, steps, fold
        )
    else:
        raise ValueError(f"unknown prediction kind: {kind}")
    root = R.prediction_root(data_root, kind, scope, seed, steps, fold)
    array_path = root / f"{case_id}.npz"
    receipt_path = root / f"{case_id}.json"
    expected_metadata = {
        "case_id": case_id,
        "kind": kind,
        "scope": scope,
        "seed": seed,
        "steps": steps,
        "fold": fold,
        "checkpoint_sha256": checkpoint_sha256,
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
    }
    return R.load_verified_prediction(array_path, receipt_path, expected_metadata)


def fold_for_stratum(plan: dict[str, Any], stratum: int) -> str:
    matches = [
        name for name, fold in plan["folds"].items()
        if int(stratum) in set(map(int, fold["held_out_z_strata"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"stratum {stratum} maps to folds {matches}")
    return matches[0]


def _pooled_arrays(
    cases: list[dict[str, Any]], truths: list[np.ndarray],
    initial: list[np.ndarray], candidate: list[np.ndarray],
) -> dict[str, Any]:
    if not (len(cases) == len(truths) == len(initial) == len(candidate)):
        raise ValueError("pooled input lengths differ")
    initial_supervised = []
    candidate_supervised = []
    y_supervised = []
    initial_valid = []
    candidate_valid = []
    positive_valid = []
    supervised_valid = []
    ignored_valid = []
    case_rows = []
    for case, bits, p0, p1 in zip(cases, truths, initial, candidate):
        masks = truth_masks(bits)
        sup = masks["supervised"]
        valid = masks["valid"]
        y = masks["positive"]
        initial_supervised.append(p0[sup])
        candidate_supervised.append(p1[sup])
        y_supervised.append(y[sup])
        initial_valid.append(p0[valid])
        candidate_valid.append(p1[valid])
        positive_valid.append(y[valid])
        supervised_valid.append(sup[valid])
        ignored_valid.append(masks["ignored_valid"][valid])
        case_rows.append({
            "case_id": R.case_identifier(case),
            "z_stratum": int(case["z_stratum"]),
            "supervised": int(sup.sum()),
            "positive": int(y.sum()),
            "valid": int(valid.sum()),
        })
    y_sup = np.concatenate(y_supervised)
    p0_sup = np.concatenate(initial_supervised)
    p1_sup = np.concatenate(candidate_supervised)
    p0_valid = np.concatenate(initial_valid)
    p1_valid = np.concatenate(candidate_valid)
    pos_valid = np.concatenate(positive_valid)
    sup_valid = np.concatenate(supervised_valid)
    ignored = np.concatenate(ignored_valid)
    initial_ap = average_precision(y_sup, p0_sup)
    candidate_ap = average_precision(y_sup, p1_sup)
    budget = int((p0_valid >= 0.2).sum())
    matched_candidate = stable_top_n(p1_valid, budget)
    return {
        "initial_average_precision": initial_ap,
        "candidate_average_precision": candidate_ap,
        "average_precision_delta": candidate_ap - initial_ap,
        "supervised_voxels": int(y_sup.size),
        "positive_voxels": int(y_sup.sum()),
        "valid_voxels": int(p0_valid.size),
        "case_rows": case_rows,
        "fixed_threshold_0_2": {
            "initial": binary_metrics(p0_valid >= 0.2, pos_valid, sup_valid),
            "candidate": binary_metrics(p1_valid >= 0.2, pos_valid, sup_valid),
        },
        "matched_initial_positive_mass": {
            "budget": budget,
            "initial": binary_metrics(p0_valid >= 0.2, pos_valid, sup_valid),
            "candidate": binary_metrics(matched_candidate, pos_valid, sup_valid),
            "candidate_selected_ignored_fraction": float(
                (matched_candidate & ignored).sum() / max(budget, 1)
            ),
        },
    }


def comparison_by_groups(
    cases: list[dict[str, Any]], truths: list[np.ndarray],
    initial: list[np.ndarray], candidate: list[np.ndarray],
) -> dict[str, Any]:
    overall = _pooled_arrays(cases, truths, initial, candidate)
    by_case = {
        R.case_identifier(case): _pooled_arrays(
            [case], [truths[index]], [initial[index]], [candidate[index]]
        )
        for index, case in enumerate(cases)
    }
    by_z = {}
    for stratum in range(C.Z_STRATA):
        idx = [i for i, case in enumerate(cases) if int(case["z_stratum"]) == stratum]
        by_z[str(stratum)] = _pooled_arrays(
            [cases[i] for i in idx], [truths[i] for i in idx],
            [initial[i] for i in idx], [candidate[i] for i in idx],
        )
    by_difficulty = {}
    for bin_id in range(len(C.DIFFICULTY_EDGES) + 1):
        idx = []
        for i, case in enumerate(cases):
            if "difficulty_bin" in case:
                case_bin = int(case["difficulty_bin"])
            else:
                fraction = case["label_stats_score_cube"].get(
                    "boundary_poor_fraction_of_material", 0.0
                )
                case_bin = C.difficulty_bin(float(fraction))
            if case_bin == bin_id:
                idx.append(i)
        if idx:
            by_difficulty[str(bin_id)] = _pooled_arrays(
                [cases[i] for i in idx], [truths[i] for i in idx],
                [initial[i] for i in idx], [candidate[i] for i in idx],
            )
    return {
        "overall": overall,
        "by_case": by_case,
        "by_z_stratum": by_z,
        "by_difficulty_bin": by_difficulty,
    }


def load_comparison(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any], scope: str,
    cases: list[dict[str, Any]], seed: int, steps: int, safety_average: bool = False,
) -> dict[str, Any]:
    truths, initial, candidate = [], [], []
    for case in cases:
        case_id = R.case_identifier(case)
        truths.append(load_truth(data_root, case_id, plan, lock))
        initial.append(load_prediction(
            data_root, plan, lock, case_id, "initial", scope
        ))
        if safety_average:
            arms = [
                load_prediction(
                    data_root, plan, lock, case_id, "finetuned", scope,
                    seed=seed, steps=steps, fold=fold,
                ) for fold in ("even", "odd")
            ]
            candidate.append(np.mean(np.stack(arms), axis=0, dtype=np.float64).astype(np.float32))
        else:
            fold = fold_for_stratum(plan, int(case["z_stratum"]))
            candidate.append(load_prediction(
                data_root, plan, lock, case_id, "finetuned", scope,
                seed=seed, steps=steps, fold=fold,
            ))
    return comparison_by_groups(cases, truths, initial, candidate)


def score_pilot(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any], steps: int,
) -> dict[str, Any]:
    if steps not in (C.PILOT_STEPS, C.PILOT_RETRY_STEPS):
        raise SystemExit("pilot steps are outside the frozen protocol")
    verdict_path = data_root / "pilot_verdict.json"
    if verdict_path.exists():
        raise SystemExit(f"pilot verdict already exists: {verdict_path}")
    attempt_path = data_root / f"pilot_attempt_steps-{steps}.json"
    if attempt_path.exists():
        raise SystemExit(f"pilot attempt already exists: {attempt_path}")
    if steps == C.PILOT_RETRY_STEPS:
        first = data_root / f"pilot_attempt_steps-{C.PILOT_STEPS}.json"
        if not first.is_file():
            raise SystemExit("4,000-step pilot requires the recorded 2,000-step attempt")
        prior = load_content_hashed_any(first)
        if prior.get("decision") != "RETRY_REQUIRED":
            raise SystemExit("4,000-step pilot is allowed only after RETRY_REQUIRED")
        if (prior.get("plan_content_sha256") != plan["content_sha256"]
                or prior.get("execution_lock_content_sha256") != lock["content_sha256"]):
            raise SystemExit("2,000-step pilot attempt belongs to a different lock")
    cases = plan["cases"]["pilot"]
    result = load_comparison(
        data_root, plan, lock, "pilot", cases, C.PILOT_SEED, steps
    )
    delta = float(result["overall"]["average_precision_delta"])
    stratum_deltas = {
        key: float(value["average_precision_delta"])
        for key, value in result["by_z_stratum"].items()
    }
    passed = (
        delta >= C.PILOT_AP_GATE
        and min(stratum_deltas.values()) >= -0.005
    )
    if passed:
        decision = "PASS"
    elif steps == C.PILOT_STEPS:
        decision = "RETRY_REQUIRED"
    else:
        decision = "TARGET_UNLEARNABLE"
    attempt = _seal({
        "schema_version": "crossscan-pilot-attempt-v1",
        "status": decision,
        "decision": decision,
        "created_utc": R.utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "steps": steps,
        "seed": C.PILOT_SEED,
        "gate": {
            "minimum_pooled_ap_delta": C.PILOT_AP_GATE,
            "maximum_allowed_z_stratum_regression": 0.005,
        },
        "result": result,
    })
    R.atomic_write_json(attempt_path, attempt)
    if decision in ("PASS", "TARGET_UNLEARNABLE"):
        verdict = _seal({
            "schema_version": "crossscan-pilot-verdict-v1",
            "status": decision,
            "created_utc": R.utc_now(),
            "selected_steps": steps if decision == "PASS" else None,
            "attempt_content_sha256": attempt["content_sha256"],
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
            "pooled_average_precision_delta": delta,
            "z_stratum_deltas": stratum_deltas,
        })
        R.atomic_write_json(verdict_path, verdict)
    return attempt


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(value)
    out.pop("content_sha256", None)
    out["content_sha256"] = C.sha256_bytes(C.canonical_json(out).encode("ascii"))
    return out


def load_content_hashed_any(path: Path) -> dict[str, Any]:
    value = C.load_json(path)
    if value.get("content_sha256") != C.content_hash_without_field(value):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def t_summary(deltas: Iterable[float]) -> dict[str, Any]:
    from scipy import stats
    values = np.asarray(list(deltas), dtype=np.float64)
    if values.shape != (len(C.INFERENTIAL_SEEDS),):
        raise ValueError(f"expected {len(C.INFERENTIAL_SEEDS)} seed deltas, got {values.shape}")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0:
        p = 0.0 if mean != 0 else 1.0
        ci = [mean, mean]
        statistic = None if mean else 0.0
    else:
        test = stats.ttest_1samp(values, popmean=0.0)
        statistic, p = float(test.statistic), float(test.pvalue)
        half = float(stats.t.ppf(0.975, values.size - 1) * sd / math.sqrt(values.size))
        ci = [mean - half, mean + half]
    return {
        "n_seeds": int(values.size),
        "mean": mean,
        "sd": sd,
        "t": statistic,
        "two_sided_p": p,
        "ci95": ci,
        "positive_seeds": int((values > 0).sum()),
        "negative_seeds": int((values < 0).sum()),
    }


def outcome_bucket(primary: dict[str, Any], safety: dict[str, Any]) -> str:
    positive = (
        primary["mean"] >= C.PRIMARY_AP_EFFECT_GATE
        and primary["positive_seeds"] >= 5
        and primary["two_sided_p"] < 0.05
    )
    if positive:
        if safety["mean"] >= -C.SAFETY_AP_MARGIN:
            return "POSITIVE_DEPLOYABLE"
        return "POSITIVE_WITH_SAFETY_REGRESSION"
    regression = (
        primary["mean"] <= -C.PRIMARY_AP_EFFECT_GATE
        and primary["negative_seeds"] >= 5
        and primary["two_sided_p"] < 0.05
    )
    if regression:
        return "REGRESSION"
    if (
        abs(primary["mean"]) < C.PRIMARY_AP_EFFECT_GATE
        and primary["ci95"][0] > -C.PRIMARY_AP_EFFECT_GATE
        and primary["ci95"][1] < C.PRIMARY_AP_EFFECT_GATE
    ):
        return "NULL"
    return "INCONCLUSIVE_UNDERPOWERED"


def score_final(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any],
) -> dict[str, Any]:
    result_path = data_root / "final_result.json"
    if result_path.exists():
        raise SystemExit(f"final result already exists: {result_path}")
    verdict = R.require_any_pilot_authorization(
        data_root, plan["content_sha256"], lock["content_sha256"]
    )
    steps = int(verdict["selected_steps"])
    primary_cases = plan["cases"]["primary"][C.TRAIN_SCROLL]
    safety_cases = plan["cases"]["primary"][C.SAFETY_SCROLL]
    seed_rows = []
    primary_deltas, safety_deltas = [], []
    comparisons: dict[str, Any] = {}
    for seed in C.INFERENTIAL_SEEDS:
        primary = load_comparison(
            data_root, plan, lock, "primary", primary_cases, seed, steps
        )
        safety = load_comparison(
            data_root, plan, lock, "safety", safety_cases, seed, steps,
            safety_average=True,
        )
        p_delta = float(primary["overall"]["average_precision_delta"])
        s_delta = float(safety["overall"]["average_precision_delta"])
        primary_deltas.append(p_delta)
        safety_deltas.append(s_delta)
        seed_rows.append({
            "seed": seed,
            "primary_initial_ap": primary["overall"]["initial_average_precision"],
            "primary_finetuned_ap": primary["overall"]["candidate_average_precision"],
            "primary_delta": p_delta,
            "safety_initial_ap": safety["overall"]["initial_average_precision"],
            "safety_finetuned_ap": safety["overall"]["candidate_average_precision"],
            "safety_delta": s_delta,
        })
        comparisons[str(seed)] = {"primary": primary, "safety": safety}
    primary_summary = t_summary(primary_deltas)
    safety_summary = t_summary(safety_deltas)
    bucket = outcome_bucket(primary_summary, safety_summary)
    result = {
        "schema_version": "crossscan-final-result-v1",
        "status": bucket,
        "created_utc": R.utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "pilot_verdict_content_sha256": verdict["content_sha256"],
        "selected_steps": steps,
        "seed_rows": seed_rows,
        "primary_summary": primary_summary,
        "safety_summary": safety_summary,
        "gates": {
            "primary_effect": C.PRIMARY_AP_EFFECT_GATE,
            "minimum_positive_seeds": 5,
            "alpha_two_sided": 0.05,
            "safety_noninferiority_margin": C.SAFETY_AP_MARGIN,
        },
        "comparisons": comparisons,
    }
    result["figures"] = make_figures(data_root, plan, lock, steps)
    result = _seal(result)
    R.atomic_write_json(result_path, result)
    return result


def make_figures(
    data_root: Path, plan: dict[str, Any], lock: dict[str, Any], steps: int,
) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = R._case_map(plan)
    out = data_root / "figures"
    staging = data_root / "figures.tmp"
    if out.exists() or staging.exists():
        raise SystemExit(
            f"refusing to overwrite final visual output: {out} or {staging}"
        )
    staging.mkdir(parents=True)
    records = []
    for visual in lock["resolved_protocol"]["visual_cases"]:
        case_id = visual["case_id"]
        scroll = visual["scroll"]
        scope = "primary" if scroll == C.TRAIN_SCROLL else "safety"
        case = cases[case_id]
        ct, truth, _ = R.verify_evaluation_case(plan, lock, data_root, case_id)
        assert ct is not None and truth is not None
        truth_masks(truth)
        initial = load_prediction(data_root, plan, lock, case_id, "initial", scope)
        seed_predictions: list[tuple[int, np.ndarray]] = []
        for seed in C.INFERENTIAL_SEEDS:
            if scroll == C.TRAIN_SCROLL:
                fold = fold_for_stratum(plan, int(case["z_stratum"]))
                pred = load_prediction(
                    data_root, plan, lock, case_id, "finetuned", scope,
                    seed=seed, steps=steps, fold=fold,
                )
            else:
                pred = np.mean(np.stack([
                    load_prediction(
                        data_root, plan, lock, case_id, "finetuned", scope,
                        seed=seed, steps=steps, fold=fold,
                    ) for fold in ("even", "odd")
                ]), axis=0)
            seed_predictions.append((seed, pred))
        fine_mean = np.mean(
            np.stack([prediction for _, prediction in seed_predictions]), axis=0
        )
        k = int(visual["score_slice_l1"])
        ct_slice_l0 = ct[64 + 2*k:64 + 2*k + 2, 64:192, 64:192].mean(axis=0)
        ct_slice = ct_slice_l0.reshape(64, 2, 64, 2).mean(axis=(1, 3))
        bits = truth[k]
        truth_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        valid = (bits & 1) != 0
        truth_rgb[valid] = (0.15, 0.15, 0.15)
        truth_rgb[valid & ((bits & 2) != 0)] = (0.55, 0.42, 0.20)
        truth_rgb[valid & ((bits & 8) != 0)] = (0.10, 0.90, 0.35)
        rows: list[tuple[str, np.ndarray]] = [
            (f"seed {seed}", prediction) for seed, prediction in seed_predictions
        ]
        rows.append(("six-seed mean", fine_mean))
        fig, axes = plt.subplots(
            len(rows), 6, figsize=(18, 19), constrained_layout=True, squeeze=False
        )
        column_titles = (
            "CT", "physical truth", "initial m7", "fine-tuned",
            "additions (Δp > 0)", "removals (Δp < 0)",
        )
        for row_index, (label, fine) in enumerate(rows):
            delta = fine[k] - initial[k]
            axes[row_index, 0].imshow(ct_slice, cmap="gray", vmin=0, vmax=212)
            axes[row_index, 1].imshow(truth_rgb)
            axes[row_index, 2].imshow(initial[k], cmap="magma", vmin=0, vmax=1)
            axes[row_index, 3].imshow(fine[k], cmap="magma", vmin=0, vmax=1)
            axes[row_index, 4].imshow(
                np.maximum(delta, 0), cmap="Greens", vmin=0, vmax=0.5
            )
            axes[row_index, 5].imshow(
                np.maximum(-delta, 0), cmap="Reds", vmin=0, vmax=0.5
            )
            for column, axis in enumerate(axes[row_index]):
                if row_index == 0:
                    axis.set_title(column_titles[column])
                axis.axis("off")
            axes[row_index, 0].text(
                -0.06, 0.5, label, rotation=90, va="center", ha="right",
                transform=axes[row_index, 0].transAxes, fontsize=9,
            )
        fig.suptitle(f"{case_id} | z stratum {visual['z_stratum']} | fixed slice {k}")
        path = staging / f"{case_id}.png"
        try:
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"visual panel was not written: {path}")
        records.append({
            **visual,
            "rows": [seed for seed, _ in seed_predictions] + ["six_seed_mean"],
            "file": {
                "path": f"figures/{path.name}",
                **R.file_record(path),
            },
        })
    staging.replace(out)
    for record in records:
        path = R.resolve_data_path(data_root, record["file"]["path"])
        if R.file_record(path) != {
            "bytes": record["file"]["bytes"],
            "sha256": record["file"]["sha256"],
        }:
            raise RuntimeError(f"moved visual panel changed: {path}")
    return records


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    R.add_runtime_args(pilot)
    pilot.add_argument("--steps", type=int, required=True)
    final = sub.add_parser("final")
    R.add_runtime_args(final)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    repo = args.repo.resolve()
    lock, plan = R.verify_runtime(
        repo, args.lock.resolve(), args.plan.resolve(), args.villa_root.resolve(),
        args.labels_root.resolve(), args.source_manifest.resolve(), args.model_dir.resolve(),
    )
    data_root = args.data_root.resolve()
    if args.command == "pilot":
        result = score_pilot(data_root, plan, lock, args.steps)
    elif args.command == "final":
        result = score_final(data_root, plan, lock)
    else:
        raise AssertionError(args.command)
    print(C.canonical_json({
        "status": result["status"],
        "content_sha256": result["content_sha256"],
    }))


if __name__ == "__main__":
    main()
