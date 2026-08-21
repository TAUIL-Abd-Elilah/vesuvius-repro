#!/usr/bin/env python3
"""Build the pinned bounded-watershed Cython extension in this directory."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="vesuvius-bounded-watershed",
    ext_modules=cythonize(
        [
            Extension(
                "_bounded_watershed_cy",
                [str(ROOT / "_bounded_watershed_cy.pyx")],
                include_dirs=[np.get_include()],
            )
        ],
        compiler_directives={"language_level": 3},
    ),
)

target = ROOT / "_bounded_watershed_cy.pyd"
candidates = sorted(
    path
    for path in ROOT.glob("_bounded_watershed_cy*.pyd")
    if path.resolve() != target.resolve()
)
if len(candidates) != 1:
    raise SystemExit(f"expected one ABI-tagged extension, found: {candidates}")
shutil.copyfile(candidates[0], target)
print(
    f"{target.name} "
    f"{hashlib.sha256(target.read_bytes()).hexdigest()}"
)
