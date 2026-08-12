#!/usr/bin/env python3
"""Strict, independent audit of the locked ScrollFiesta welded OBJ output."""

from __future__ import annotations

import json
import math
import os
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORLD_BBOX_ZYX = ((3840.0, 4096.0), (3712.0, 3968.0), (1344.0, 1600.0))
INTERNAL_PLANES_ZYX = (
    ("z", 0, 3968.0),
    ("y", 1, 3840.0),
    ("x", 2, 1472.0),
)
GEOMETRY_TOLERANCE = 1.0
EXPECTED_CUBES_PROCESSED = 8

_TOP_LEVEL_REPORT_FIELDS = (
    "cubes_processed",
    "total_input_verts",
    "total_unique_verts",
    "total_input_faces",
    "total_unique_faces",
    "recoarsen",
    "band_cvt",
    "manifold_audit",
)
_RECOARSEN_FIELDS = ("collapses", "faces_in", "faces_out")
_BAND_CVT_FIELDS = (
    "patches_accepted",
    "patches_rejected",
    "band_faces_in",
    "band_faces_out",
)
_MANIFOLD_FIELDS = (
    "unpaired",
    "non_manifold",
    "same_dir_pairs",
    "manifold_pairs",
    "pinch_verts",
)
_RECOMPUTED_EDGE_FIELDS = _MANIFOLD_FIELDS[:-1]
_INT64_MAX = np.iinfo(np.int64).max


class ObjFormatError(ValueError):
    """The OBJ is not in the deliberately narrow accepted format."""


class WeldReportError(ValueError):
    """The weld report is malformed or does not have the pinned schema."""


class AuditMismatchError(ValueError):
    """The independent OBJ audit disagrees with its weld report."""


@dataclass(frozen=True)
class ObjMesh:
    """Parsed triangle mesh; coordinates and zero-based faces are NumPy arrays."""

    vertices_zyx: np.ndarray
    faces: np.ndarray


def _obj_error(path: Path, line_number: int, message: str) -> ObjFormatError:
    return ObjFormatError(f"{path}:{line_number}: {message}")


def _parse_face_vertex(
    token: str, *, vertex_count: int, path: Path, line_number: int
) -> int:
    fields = token.split("/")
    valid_shape = (
        len(fields) == 1
        or (len(fields) == 2 and bool(fields[1]))
        or (
            len(fields) == 3
            and bool(fields[2])
            and (bool(fields[1]) or fields[1] == "")
        )
    )
    if not valid_shape or not fields[0]:
        raise _obj_error(path, line_number, f"malformed face reference {token!r}")

    for field in fields:
        if not field:
            continue
        unsigned = field[1:] if field[:1] in ("+", "-") else field
        if not unsigned or not unsigned.isascii() or not unsigned.isdecimal():
            raise _obj_error(path, line_number, f"malformed face reference {token!r}")
        if int(field, 10) == 0:
            raise _obj_error(path, line_number, "OBJ indices must not be zero")

    raw_index = int(fields[0], 10)
    resolved = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if resolved < 0:
        raise _obj_error(path, line_number, f"face index {raw_index} is out of range")
    if resolved > _INT64_MAX:
        raise _obj_error(path, line_number, f"face index {raw_index} is too large")
    return resolved


def parse_obj(path: os.PathLike[str] | str) -> ObjMesh:
    """Parse only comments, finite ``v z y x [r g b]``, and triangular faces.

    Face references may use positive or relative-negative vertex indices and the
    standard ``v``, ``v/vt``, ``v//vn``, or ``v/vt/vn`` spellings.  Slash
    fields are syntax-checked but otherwise ignored because this auditor does
    not consume texture coordinates or normals.
    """

    obj_path = Path(path)
    vertex_values = array("d")
    face_values = array("q")
    vertex_count = 0
    face_count = 0

    with obj_path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
        for line_number, raw_line in enumerate(stream, 1):
            geometry = raw_line.partition("#")[0]
            if not geometry.strip():
                continue
            try:
                geometry.encode("ascii")
            except UnicodeEncodeError as exc:
                raise _obj_error(
                    obj_path, line_number, "non-ASCII text outside an OBJ comment"
                ) from exc

            tokens = geometry.split()
            record = tokens[0]
            if record == "v":
                if len(tokens) not in (4, 7):
                    raise _obj_error(
                        obj_path,
                        line_number,
                        "vertex must be 'v z y x' with optional RGB",
                    )
                try:
                    values = [float(value) for value in tokens[1:]]
                except ValueError as exc:
                    raise _obj_error(obj_path, line_number, "invalid vertex number") from exc
                if not all(math.isfinite(value) for value in values):
                    raise _obj_error(obj_path, line_number, "non-finite vertex value")
                vertex_values.extend(values[:3])
                vertex_count += 1
            elif record == "f":
                if len(tokens) != 4:
                    raise _obj_error(obj_path, line_number, "faces must be triangles")
                indices = [
                    _parse_face_vertex(
                        token,
                        vertex_count=vertex_count,
                        path=obj_path,
                        line_number=line_number,
                    )
                    for token in tokens[1:]
                ]
                if len(set(indices)) != 3:
                    raise _obj_error(
                        obj_path, line_number, "triangle vertex indices must be distinct"
                    )
                face_values.extend(indices)
                face_count += 1
            else:
                raise _obj_error(
                    obj_path, line_number, f"unsupported OBJ record {record!r}"
                )

    if vertex_count == 0:
        raise ObjFormatError(f"{obj_path}: OBJ contains no vertices")
    if face_count == 0:
        raise ObjFormatError(f"{obj_path}: OBJ contains no faces")

    vertices = np.frombuffer(vertex_values, dtype=np.float64).reshape(vertex_count, 3)
    faces = np.frombuffer(face_values, dtype=np.int64).reshape(face_count, 3)
    invalid = np.argwhere(faces >= vertex_count)
    if invalid.size:
        face_index, corner = (int(value) for value in invalid[0])
        raise ObjFormatError(
            f"{obj_path}: face {face_index + 1} corner {corner + 1} index "
            f"{int(faces[face_index, corner]) + 1} exceeds vertex count {vertex_count}"
        )
    return ObjMesh(vertices_zyx=vertices, faces=faces)


def _edge_incidence(faces: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (n, 3)")
    if faces.shape[0] == 0:
        return {
            "unpaired": 0,
            "non_manifold": 0,
            "same_dir_pairs": 0,
            "manifold_pairs": 0,
        }, np.empty((0, 2), dtype=np.int64)

    src = faces.reshape(-1)
    dst = faces[:, (1, 2, 0)].reshape(-1)
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    order = np.lexsort((high, low))
    low = low[order]
    high = high[order]
    forward = (src[order] < dst[order])

    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero((low[1:] != low[:-1]) | (high[1:] != high[:-1])) + 1,
        )
    )
    lengths = np.diff(np.append(starts, low.size))
    unpaired_mask = lengths == 1
    pair_starts = starts[lengths == 2]
    same_direction = forward[pair_starts] == forward[pair_starts + 1]
    unpaired_starts = starts[unpaired_mask]
    unpaired_edges = np.column_stack(
        (low[unpaired_starts], high[unpaired_starts])
    ).astype(np.int64, copy=False)

    counts = {
        "unpaired": int(np.count_nonzero(unpaired_mask)),
        "non_manifold": int(np.count_nonzero(lengths > 2)),
        "same_dir_pairs": int(np.count_nonzero(same_direction)),
        "manifold_pairs": int(pair_starts.size - np.count_nonzero(same_direction)),
    }
    return counts, unpaired_edges


def recompute_edge_incidence(faces: np.ndarray) -> dict[str, int]:
    """Vectorize the same undirected-edge buckets used by ``grid_weld``."""

    counts, _ = _edge_incidence(np.asarray(faces, dtype=np.int64))
    return counts


def _internal_seam_counts(
    vertices_zyx: np.ndarray, unpaired_edges: np.ndarray
) -> tuple[int, dict[str, int]]:
    per_plane = {name: 0 for name, _, _ in INTERNAL_PLANES_ZYX}
    if unpaired_edges.shape[0] == 0:
        return 0, per_plane

    midpoints = (
        vertices_zyx[unpaired_edges[:, 0]] + vertices_zyx[unpaired_edges[:, 1]]
    ) * 0.5
    clear_of_outer_faces = np.ones(midpoints.shape[0], dtype=bool)
    for axis, (low, high) in enumerate(WORLD_BBOX_ZYX):
        clear_of_outer_faces &= midpoints[:, axis] > low + GEOMETRY_TOLERANCE
        clear_of_outer_faces &= midpoints[:, axis] < high - GEOMETRY_TOLERANCE

    union = np.zeros(midpoints.shape[0], dtype=bool)
    for name, axis, plane in INTERNAL_PLANES_ZYX:
        matches = clear_of_outer_faces & (
            np.abs(midpoints[:, axis] - plane) <= GEOMETRY_TOLERANCE
        )
        per_plane[name] = int(np.count_nonzero(matches))
        union |= matches
    return int(np.count_nonzero(union)), per_plane


def audit_obj(
    path: os.PathLike[str] | str, *, require_world_span: bool = True
) -> dict[str, Any]:
    """Independently audit one OBJ against the locked world-box geometry."""

    mesh = parse_obj(path)
    vertices = mesh.vertices_zyx
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    span = maximum - minimum

    for axis, ((low, high), axis_name) in enumerate(zip(WORLD_BBOX_ZYX, "zyx")):
        if minimum[axis] < low - GEOMETRY_TOLERANCE or maximum[axis] > high + GEOMETRY_TOLERANCE:
            raise AuditMismatchError(
                f"OBJ {axis_name} bounds [{minimum[axis]}, {maximum[axis]}] exceed "
                f"locked [{low}, {high}] +/- {GEOMETRY_TOLERANCE}"
            )

    spans_internal_planes: dict[str, bool] = {}
    for name, axis, plane in INTERNAL_PLANES_ZYX:
        spans = bool(
            minimum[axis] < plane - GEOMETRY_TOLERANCE
            and maximum[axis] > plane + GEOMETRY_TOLERANCE
        )
        if require_world_span and not spans:
            raise AuditMismatchError(
                f"OBJ does not span both sides of internal plane {name} beyond "
                f"{GEOMETRY_TOLERANCE} voxel"
            )
        spans_internal_planes[name] = spans

    edge_audit, unpaired_edges = _edge_incidence(mesh.faces)
    seam_union, seam_by_plane = _internal_seam_counts(vertices, unpaired_edges)
    edge_audit = {
        **edge_audit,
        "internal_seam_unpaired_edges_union": seam_union,
        "internal_seam_unpaired_edges_by_plane": seam_by_plane,
    }
    return {
        "vertices": int(vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "bounds_zyx": {
            "min": [float(value) for value in minimum],
            "max": [float(value) for value in maximum],
            "span": [float(value) for value in span],
        },
        "within_world_bbox_tolerance": True,
        "spans_internal_planes": spans_internal_planes,
        "edge_audit": edge_audit,
    }


def _reject_json_constant(value: str) -> None:
    raise WeldReportError(f"non-standard JSON constant {value!r} is forbidden")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WeldReportError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: Any, fields: tuple[str, ...], location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise WeldReportError(f"{location} must be a JSON object")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        raise WeldReportError(
            f"{location} has wrong fields: missing={missing}, extra={extra}"
        )
    return value


def _require_nonnegative_integer(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        raise WeldReportError(f"{location} must be a nonnegative JSON integer")
    return value


def _integer_section(
    value: Any, fields: tuple[str, ...], location: str
) -> dict[str, int]:
    section = _require_exact_keys(value, fields, location)
    return {
        field: _require_nonnegative_integer(section[field], f"{location}.{field}")
        for field in fields
    }


def parse_weld_report(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Parse the exact pinned ``grid_weld`` JSON schema without coercion."""

    report_path = Path(path)
    try:
        with report_path.open("r", encoding="utf-8", errors="strict") as stream:
            raw = json.load(
                stream,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
    except json.JSONDecodeError as exc:
        raise WeldReportError(f"{report_path}: malformed JSON: {exc}") from exc

    report = _require_exact_keys(raw, _TOP_LEVEL_REPORT_FIELDS, "weld_report")
    return {
        "cubes_processed": _require_nonnegative_integer(
            report["cubes_processed"], "weld_report.cubes_processed"
        ),
        "total_input_verts": _require_nonnegative_integer(
            report["total_input_verts"], "weld_report.total_input_verts"
        ),
        "total_unique_verts": _require_nonnegative_integer(
            report["total_unique_verts"], "weld_report.total_unique_verts"
        ),
        "total_input_faces": _require_nonnegative_integer(
            report["total_input_faces"], "weld_report.total_input_faces"
        ),
        "total_unique_faces": _require_nonnegative_integer(
            report["total_unique_faces"], "weld_report.total_unique_faces"
        ),
        "recoarsen": _integer_section(
            report["recoarsen"], _RECOARSEN_FIELDS, "weld_report.recoarsen"
        ),
        "band_cvt": _integer_section(
            report["band_cvt"], _BAND_CVT_FIELDS, "weld_report.band_cvt"
        ),
        "manifold_audit": _integer_section(
            report["manifold_audit"],
            _MANIFOLD_FIELDS,
            "weld_report.manifold_audit",
        ),
    }


def audit_scrollfiesta_obj(
    obj_path: os.PathLike[str] | str,
    report_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Validate and reconcile a final welded OBJ and its producer report."""

    obj_audit = audit_obj(obj_path)
    report = parse_weld_report(report_path)
    mismatches: list[str] = []
    if report["cubes_processed"] != EXPECTED_CUBES_PROCESSED:
        mismatches.append(
            f"cubes_processed: report={report['cubes_processed']}, "
            f"expected={EXPECTED_CUBES_PROCESSED}"
        )
    for report_field, obj_field in (
        ("total_unique_verts", "vertices"),
        ("total_unique_faces", "faces"),
    ):
        if report[report_field] != obj_audit[obj_field]:
            mismatches.append(
                f"{report_field} differs: report={report[report_field]}, "
                f"OBJ={obj_audit[obj_field]}"
            )
    for field in _RECOMPUTED_EDGE_FIELDS:
        reported = report["manifold_audit"][field]
        recomputed = obj_audit["edge_audit"][field]
        if reported != recomputed:
            mismatches.append(f"{field} differs: report={reported}, OBJ={recomputed}")
    if mismatches:
        raise AuditMismatchError("weld report mismatch: " + "; ".join(mismatches))

    return {
        "status": "PASS",
        "obj": obj_audit,
        "weld_report": report,
        "pinch_verts": report["manifold_audit"]["pinch_verts"],
    }
