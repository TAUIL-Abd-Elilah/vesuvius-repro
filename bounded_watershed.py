"""Pinned compactness-zero watershed with a file-backed fixed-capacity heap."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import mmap
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology._util import _offsets_to_raveled_neighbors, _validate_connectivity
from skimage.segmentation._watershed import _validate_inputs
from skimage.util import crop

HEAP_CAPACITY_ITEMS = 4_000_000_000
EXPECTED_HEAP_ITEM_BYTES = 16
MAX_RAVELED_ELEMENTS = 2_147_483_648
MIN_FREE_AFTER_FULL_HEAP_BYTES = 8 * 1024**3
MIN_FREE_AFTER_SMALL_HEAP_BYTES = 256 * 1024**2
HEAP_FILE_PREFIX = "bounded-watershed-p"
HEAP_FILE_SUFFIX = ".heap"
HEAP_FILENAME_RE = re.compile(
    rf"^{re.escape(HEAP_FILE_PREFIX)}([1-9][0-9]*)-[^.\\/]+"
    rf"{re.escape(HEAP_FILE_SUFFIX)}$"
)
SCRATCH_DIRECTORY = Path(__file__).resolve().parent.parent / "physical_bridge_heap_scratch"
BOUNDED_WATERSHED_BINARY = "_bounded_watershed_cy.pyd"
BOUNDED_WATERSHED_BINARY_SHA256 = (
    "93dfd77cc857cfa1d67e5dd1f2d1865aae8f010a5ff309c999eb2ccf6dd7841a"
)


def _load_verified_core():
    binary = Path(__file__).resolve().parent / BOUNDED_WATERSHED_BINARY
    if not binary.is_file():
        raise ImportError(f"bounded watershed binary is missing: {binary}")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != BOUNDED_WATERSHED_BINARY_SHA256:
        raise ImportError(
            f"bounded watershed binary SHA-256 mismatch: {digest}"
        )
    spec = importlib.util.spec_from_file_location("_bounded_watershed_cy", binary)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create bounded watershed extension spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_verified_core()


def heap_file_bytes(capacity_items: int = HEAP_CAPACITY_ITEMS) -> int:
    if type(capacity_items) is not int or capacity_items < 1:
        raise ValueError("heap capacity must be a positive integer")
    item_bytes = int(_core.heap_item_size())
    if item_bytes != EXPECTED_HEAP_ITEM_BYTES:
        raise RuntimeError(
            f"unexpected bounded watershed heap item size: {item_bytes}"
        )
    return capacity_items * item_bytes


def _validate_compact_index_space(
    shape: tuple[int, ...],
    offset: tuple[int, ...] | np.ndarray,
) -> int:
    """Return padded element count after proving every raveled index fits int32."""

    if len(shape) != len(offset):
        raise ValueError("bounded watershed shape/offset rank mismatch")
    padded_elements = math.prod(
        int(length) + 2 * int(padding)
        for length, padding in zip(shape, offset, strict=True)
    )
    if padded_elements > MAX_RAVELED_ELEMENTS:
        raise ValueError(
            "bounded watershed padded image exceeds the signed-int32 index range"
        )
    return padded_elements


def _heap_pattern(pid: int) -> str:
    if type(pid) is not int or pid <= 0:
        raise ValueError("worker PID must be a positive integer")
    return f"{HEAP_FILE_PREFIX}{pid}-*{HEAP_FILE_SUFFIX}"


def cleanup_heap_files_for_pid(pid: int) -> None:
    """Remove only completed worker heap files from the fixed scratch directory."""

    if not SCRATCH_DIRECTORY.is_dir():
        return
    scratch = SCRATCH_DIRECTORY.resolve()
    for path in SCRATCH_DIRECTORY.glob(_heap_pattern(pid)):
        resolved = path.resolve()
        if resolved.parent != scratch or not resolved.is_file():
            raise RuntimeError("refusing unsafe bounded watershed cleanup target")
        resolved.unlink()


def _process_is_running(pid: int) -> bool:
    """Conservatively report whether a PID still owns a live process."""

    if type(pid) is not int or pid <= 0:
        raise ValueError("worker PID must be a positive integer")
    if sys.platform == "win32":
        import _winapi

        process_query_limited_information = 0x1000
        still_active = 259
        try:
            handle = _winapi.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
        except OSError as exc:
            if exc.winerror == 5:  # Access denied: retain the file conservatively.
                return True
            if exc.winerror in {87, 1168}:  # Invalid PID / process not found.
                return False
            raise
        try:
            return _winapi.GetExitCodeProcess(handle) == still_active
        finally:
            _winapi.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_heap_files(
    scratch_directory: Path = SCRATCH_DIRECTORY,
) -> None:
    """Remove strict-name heap files only after their owning PID has exited."""

    if not scratch_directory.is_dir():
        return
    scratch = scratch_directory.resolve()
    for path in scratch_directory.glob(f"{HEAP_FILE_PREFIX}*{HEAP_FILE_SUFFIX}"):
        resolved = path.resolve()
        if resolved.parent != scratch or not resolved.is_file():
            raise RuntimeError("refusing unsafe bounded watershed cleanup target")
        match = HEAP_FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError("refusing malformed bounded watershed cleanup target")
        if not _process_is_running(int(match.group(1))):
            resolved.unlink()


def _size_sparse_heap_file(
    descriptor: int,
    path: Path,
    storage_bytes: int,
) -> None:
    """Set the logical heap size without physically zero-filling untouched ranges."""

    if sys.platform != "win32":
        os.ftruncate(descriptor, storage_bytes)
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    kernel32.SetEndOfFile.restype = wintypes.BOOL

    handle = msvcrt.get_osfhandle(descriptor)
    bytes_returned = wintypes.DWORD()
    new_position = ctypes.c_longlong()
    if not kernel32.DeviceIoControl(
        handle,
        0x000900C4,  # FSCTL_SET_SPARSE
        None,
        0,
        None,
        0,
        ctypes.byref(bytes_returned),
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.SetFilePointerEx(
        handle,
        storage_bytes,
        ctypes.byref(new_position),
        0,  # FILE_BEGIN
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.SetEndOfFile(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    attributes = path.stat().st_file_attributes
    if not attributes & stat.FILE_ATTRIBUTE_SPARSE_FILE:
        raise OSError("bounded watershed heap file is not sparse")


@contextmanager
def _heap_storage(
    capacity_items: int = HEAP_CAPACITY_ITEMS,
    *,
    scratch_directory: Path = SCRATCH_DIRECTORY,
) -> Iterator[mmap.mmap]:
    storage_bytes = heap_file_bytes(capacity_items)
    scratch_directory.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(scratch_directory).free
    free_reserve = min(
        MIN_FREE_AFTER_FULL_HEAP_BYTES,
        max(MIN_FREE_AFTER_SMALL_HEAP_BYTES, storage_bytes // 4),
    )
    if free_bytes < storage_bytes + free_reserve:
        raise OSError(
            "insufficient scratch space for bounded watershed heap: "
            f"need {storage_bytes + free_reserve}, have {free_bytes}"
        )

    descriptor, raw_path = tempfile.mkstemp(
        prefix=f"{HEAP_FILE_PREFIX}{os.getpid()}-",
        suffix=HEAP_FILE_SUFFIX,
        dir=scratch_directory,
    )
    path = Path(raw_path)
    mapping: mmap.mmap | None = None
    try:
        _size_sparse_heap_file(descriptor, path, storage_bytes)
        if path.stat().st_size != storage_bytes:
            raise OSError("bounded watershed heap file size mismatch")
        mapping = mmap.mmap(descriptor, storage_bytes, access=mmap.ACCESS_WRITE)
        yield mapping
    finally:
        if mapping is not None:
            mapping.close()
        os.close(descriptor)
        if path.exists():
            path.unlink()


def watershed(
    image: np.ndarray,
    markers: np.ndarray | int | None = None,
    connectivity: int | np.ndarray = 1,
    offset: tuple[int, ...] | None = None,
    mask: np.ndarray | None = None,
    compactness: float = 0,
    watershed_line: bool = False,
    *,
    _heap_capacity_items: int | None = None,
    _scratch_directory: Path = SCRATCH_DIRECTORY,
) -> np.ndarray:
    """Match scikit-image 0.26 noncompact watershed with bounded heap storage."""

    image, markers, mask = _validate_inputs(image, markers, mask, connectivity)
    if markers.dtype != np.int32:
        raise TypeError("bounded watershed requires int32 markers")
    if compactness != 0:
        raise ValueError("bounded watershed cost-index heap requires compactness == 0")
    connectivity, offset = _validate_connectivity(image.ndim, connectivity, offset)
    _validate_compact_index_space(image.shape, offset)

    pad_width = [(padding, padding) for padding in offset]
    image = np.pad(image, pad_width, mode="constant")
    mask = np.pad(mask, pad_width, mode="constant").ravel()
    output = np.pad(markers, pad_width, mode="constant")
    flat_neighborhood = _offsets_to_raveled_neighbors(
        image.shape,
        connectivity,
        center=offset,
    )
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(image.strides, dtype=np.intp) // image.itemsize
    if _heap_capacity_items is None:
        heap_capacity_items = HEAP_CAPACITY_ITEMS
    else:
        heap_capacity_items = _heap_capacity_items

    with _heap_storage(
        heap_capacity_items,
        scratch_directory=_scratch_directory,
    ) as heap_storage:
        _core.watershed_raveled_bounded(
            image.ravel(),
            marker_locations,
            flat_neighborhood,
            mask,
            image_strides,
            compactness,
            output.ravel(),
            watershed_line,
            heap_storage,
        )

    return crop(output, pad_width, copy=True)


__all__ = [
    "HEAP_CAPACITY_ITEMS",
    "MAX_RAVELED_ELEMENTS",
    "SCRATCH_DIRECTORY",
    "cleanup_heap_files_for_pid",
    "cleanup_stale_heap_files",
    "heap_file_bytes",
    "watershed",
]
