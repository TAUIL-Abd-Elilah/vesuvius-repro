"""Conservative multi-threshold splitting of low-confidence surface bridges.

This is deliberately narrower than hole filling.  Given a foreground probability volume,
the baseline is the ordinary low-threshold mask.  Persistent components at a higher threshold
that contain high-confidence seed voxels are protected.  If several protected regions are
joined only by a low-confidence saddle, a marker watershed proposes a one-voxel seam.  The
proposal is committed only when all protected regions become distinct components, no extra
component is created, no protected voxel is removed, and the edit budget is respected.

The operator is removal-only, deterministic, and fail-closed.  It therefore cannot merge two
foreground components.  ``editable_mask`` supports halo-based callers: it prevents an
out-of-core edit, and the remaining proposal must still pass the complete structural contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from bounded_watershed import watershed


@dataclass(frozen=True)
class BridgeSplitConfig:
    """Frozen parameters for one bridge-splitting arm."""

    low_threshold: float = 0.2
    persistence_threshold: float = 0.4
    seed_threshold: float = 0.6
    cut_ceiling: float = 0.35
    min_seed_voxels: int = 32
    min_output_component_voxels: int = 128
    max_removed_fraction: float = 0.02
    connectivity: int = 3

    def validate(self, ndim: int) -> None:
        if not (
            0.0 <= self.low_threshold
            < self.cut_ceiling
            <= self.persistence_threshold
            < self.seed_threshold
            <= 1.0
        ):
            raise ValueError(
                "thresholds must satisfy 0 <= low < cut <= persistence < seed <= 1"
            )
        if self.min_seed_voxels < 1:
            raise ValueError("min_seed_voxels must be positive")
        if self.min_output_component_voxels < 1:
            raise ValueError("min_output_component_voxels must be positive")
        if not (0.0 < self.max_removed_fraction <= 1.0):
            raise ValueError("max_removed_fraction must be in (0, 1]")
        if not (1 <= self.connectivity <= ndim):
            raise ValueError(f"connectivity must be between 1 and {ndim}")


@dataclass
class BridgeSplitResult:
    mask: np.ndarray
    removed: np.ndarray
    audit: dict[str, Any]


def _component_seed_markers(
    probability: np.ndarray,
    component: np.ndarray,
    config: BridgeSplitConfig,
    structure: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return persistent markers that each contain enough high-confidence support."""

    persistent = component & (probability > config.persistence_threshold)
    persistent_labels, n_persistent = ndi.label(persistent, structure=structure)
    if n_persistent < 2:
        return np.zeros(component.shape, dtype=np.int32), 0

    high = component & (probability > config.seed_threshold)
    seed_counts = np.bincount(
        persistent_labels[high], minlength=n_persistent + 1
    )
    eligible = np.flatnonzero(seed_counts >= config.min_seed_voxels)
    eligible = eligible[eligible != 0]
    if eligible.size < 2:
        return np.zeros(component.shape, dtype=np.int32), 0
    remap = np.zeros(n_persistent + 1, dtype=np.int32)
    remap[eligible] = np.arange(1, eligible.size + 1, dtype=np.int32)
    return remap[persistent_labels], int(eligible.size)


def _marker_output_labels(
    output_labels: np.ndarray, markers: np.ndarray, marker_count: int
) -> list[int] | None:
    """Map every protected marker to exactly one nonzero output component."""

    mapped: list[int] = []
    for marker_id in range(1, marker_count + 1):
        ids = np.unique(output_labels[markers == marker_id])
        ids = ids[ids != 0]
        if ids.size != 1:
            return None
        mapped.append(int(ids[0]))
    return mapped


def split_probability_bridges(
    probability: np.ndarray,
    config: BridgeSplitConfig = BridgeSplitConfig(),
    *,
    editable_mask: np.ndarray | None = None,
) -> BridgeSplitResult:
    """Split only validated low-confidence bridges in a probability volume.

    The returned mask is always a subset of ``probability > low_threshold``.  A component edit
    is all-or-nothing.  In particular, filtering a watershed seam by ``cut_ceiling`` or by
    ``editable_mask`` cannot leave a partial, unvalidated cut.
    """

    p = np.asarray(probability)
    if p.ndim not in (2, 3):
        raise ValueError(f"probability must be 2D or 3D, got shape {p.shape}")
    if not np.issubdtype(p.dtype, np.floating):
        p = p.astype(np.float32)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability values must be finite and in [0, 1]")
    config.validate(p.ndim)

    if editable_mask is None:
        editable = np.ones(p.shape, dtype=bool)
    else:
        editable = np.asarray(editable_mask, dtype=bool)
        if editable.shape != p.shape:
            raise ValueError(
                f"editable_mask shape {editable.shape} does not match probability {p.shape}"
            )

    structure = ndi.generate_binary_structure(p.ndim, config.connectivity)
    baseline = p > config.low_threshold
    output = baseline.copy()
    removed = np.zeros(p.shape, dtype=bool)
    base_labels, base_count = ndi.label(baseline, structure=structure)

    audit: dict[str, Any] = {
        "schema_version": 1,
        "config": asdict(config),
        "baseline_components": int(base_count),
        "components_with_multiple_markers": 0,
        "accepted_components": 0,
        "removed_voxels": 0,
        "rejected": {
            "empty_seam": 0,
            "edit_budget": 0,
            "protected_seed": 0,
            "markers_not_separated": 0,
            "extra_or_small_component": 0,
        },
    }

    component_slices = ndi.find_objects(base_labels)
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_labels = base_labels[component_slice]
        component = local_labels == component_id
        component_size = int(np.count_nonzero(component))
        if component_size < 2 * config.min_output_component_voxels:
            continue

        local_p = p[component_slice]
        markers, marker_count = _component_seed_markers(
            local_p, component, config, structure
        )
        if marker_count < 2:
            continue
        audit["components_with_multiple_markers"] += 1

        partition = watershed(
            -local_p,
            markers=markers,
            mask=component,
            connectivity=structure,
            watershed_line=True,
        )
        seam = component & (partition == 0)
        candidate = seam & (local_p <= config.cut_ceiling) & editable[component_slice]
        candidate_count = int(np.count_nonzero(candidate))
        if candidate_count == 0:
            audit["rejected"]["empty_seam"] += 1
            continue
        if candidate_count / component_size > config.max_removed_fraction:
            audit["rejected"]["edit_budget"] += 1
            continue
        if np.any(candidate & (markers > 0)):
            audit["rejected"]["protected_seed"] += 1
            continue

        proposed = component & ~candidate
        proposed_labels, proposed_count = ndi.label(proposed, structure=structure)
        marker_outputs = _marker_output_labels(proposed_labels, markers, marker_count)
        if marker_outputs is None or len(set(marker_outputs)) != marker_count:
            audit["rejected"]["markers_not_separated"] += 1
            continue

        sizes = np.bincount(proposed_labels.ravel(), minlength=proposed_count + 1)[1:]
        if (
            proposed_count != marker_count
            or sizes.size != marker_count
            or np.any(sizes < config.min_output_component_voxels)
        ):
            audit["rejected"]["extra_or_small_component"] += 1
            continue

        local_output = output[component_slice]
        local_removed = removed[component_slice]
        local_output[candidate] = False
        local_removed[candidate] = True
        output[component_slice] = local_output
        removed[component_slice] = local_removed
        audit["accepted_components"] += 1
        audit["removed_voxels"] += candidate_count

    if np.any(output & ~baseline):
        raise AssertionError("bridge splitter must never add foreground")
    if np.any(removed & ~editable):
        raise AssertionError("bridge splitter edited outside editable_mask")
    if np.any(removed & (p > config.persistence_threshold)):
        raise AssertionError("bridge splitter removed a persistent-confidence voxel")
    return BridgeSplitResult(mask=output, removed=removed, audit=audit)
