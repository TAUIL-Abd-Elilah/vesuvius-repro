#!/usr/bin/env python3
"""Download the public PHercParis4 inputs used by the screen (no private data)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import urlopen

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
PATCH_FILES = {"meta.json", "x.tif", "y.tif", "z.tif", "mask.tif", "winding.tif", "overlapping.json"}


def get(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:  # public static data only
        return response.read()


def names(url: str, suffix: str) -> list[str]:
    html = get(url).decode("utf-8")
    values = re.findall(r'href="([^"?#]+)"', html)
    result = sorted(value[:-1] if suffix == "/" else value for value in values if value.endswith(suffix))
    if not result:
        raise RuntimeError(f"empty public index: {url}")
    if any(Path(value).name != value or value in {"", ".", ".."} for value in result):
        raise RuntimeError("unsafe server index entry")
    return result


def write_if_missing(url: str, path: Path) -> int:
    if path.is_file() and path.stat().st_size:
        return path.stat().st_size
    data = get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return len(data)


def fetch_patch(name: str, output: Path) -> tuple[int, int]:
    root = f"{BASE}/verified_patches/{quote(name)}/"
    available = set(names(root, ".json") + names(root, ".tif")) & PATCH_FILES
    required = {"meta.json", "x.tif", "y.tif", "z.tif"}
    if not required <= available:
        raise RuntimeError(f"{name}: missing {sorted(required - available)}")
    count = size = 0
    for filename in sorted(available):
        size += write_if_missing(root + quote(filename), output / "verified_patches" / name / filename)
        count += 1
    return count, size


def write_scroll_spec(output: Path) -> None:
    # Historical PHercParis4 production defaults required by current Villa
    # loaders; this metadata is materialized locally, not downloaded.
    spec = {"schema_version": 1, "name": "s1", "voxel_size_um": 9.6,
            "spiral_outward_sense": "CW", "umbilicus": {"coordinate_scale": 1.0},
            "lasagna_scale": 4,
            "provenance_note": "Historical PHercParis4 production defaults for this public sampler screen."}
    (output / "spiral-scroll.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be in [1, 128]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_if_missing(f"{BASE}/umbilicus.json", output / "umbilicus.json")
    write_scroll_spec(output)
    patches = names(f"{BASE}/verified_patches/", "/")
    fibers = names(f"{BASE}/eval_fibers/", ".json")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        patch_stats = list(pool.map(lambda item: fetch_patch(item, output), patches))
        fiber_sizes = list(pool.map(lambda item: write_if_missing(f"{BASE}/eval_fibers/{quote(item)}", output / "eval_fibers" / item), fibers))
    index = get(f"{BASE}/verified_patches/")
    manifest = {"source": BASE, "patches": {"directories": len(patches), "files": sum(item[0] for item in patch_stats), "bytes": sum(item[1] for item in patch_stats), "index_sha256": hashlib.sha256(index).hexdigest()}, "eval_fibers": {"files": len(fibers), "bytes": sum(fiber_sizes)}}
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
