"""Shared fixtures for the Python test suite.

The suite mirrors tests/*.cpp. Tests that need the C++ tools (mesh_generator for byte-for-byte
parity, verify_tracking for the exact provenance checks) look for them in VRTEST_BUILD_DIR, or
in ../../build next to this file, and skip when they are absent, so the same suite runs from a
CMake tree (everything), from a wheel's test environment (API only) and from a source checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATA = ROOT / "tests" / "data"
MANIFEST = ROOT / "tests" / "reference_hashes.txt"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_cli: needs the mesh_generator / verify_tracking binaries from a CMake build"
    )


def _find_tool(name: str) -> Path | None:
    exe = name + (".exe" if os.name == "nt" else "")
    env = os.environ.get("VRTEST_BUILD_DIR")
    candidates = []
    if env:
        candidates += [Path(env) / exe, Path(env) / "tests" / exe, Path(env) / "Release" / exe,
                       Path(env) / "tests" / "Release" / exe]
    for b in ("build", "build-py", "build-release"):
        candidates += [ROOT / b / exe, ROOT / b / "tests" / exe, ROOT / b / "Release" / exe,
                       ROOT / b / "tests" / "Release" / exe]
    for c in candidates:
        if c.is_file():
            return c
    found = shutil.which(name)
    return Path(found) if found else None


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def mesh_generator() -> Path:
    exe = _find_tool("mesh_generator")
    if exe is None:
        pytest.skip("mesh_generator not built (set VRTEST_BUILD_DIR)")
    return exe


@pytest.fixture(scope="session")
def verify_tracking() -> Path:
    exe = _find_tool("verify_tracking")
    if exe is None:
        pytest.skip("verify_tracking not built (set VRTEST_BUILD_DIR)")
    return exe


@pytest.fixture(scope="session")
def verify_tracking_2d() -> Path:
    exe = _find_tool("verify_tracking_2d")
    if exe is None:
        pytest.skip("verify_tracking_2d not built (set VRTEST_BUILD_DIR)")
    return exe


def model(name: str) -> Path:
    """A bundled model, or skip when it is not committed."""
    p = MODELS / name
    if not p.is_file():
        pytest.skip(f"model not present: {name}")
    return p


def run_python_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """`python -m volume_remesher <args>` in cwd, importing the same package this process did
    (a relative PYTHONPATH would resolve against cwd)."""
    import volume_remesher

    env = dict(os.environ)
    pkg_parent = str(Path(volume_remesher.__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "volume_remesher", *map(str, args)],
        cwd=cwd, capture_output=True, text=True, env=env,
    )


def run_tool(exe: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), *map(str, args)], cwd=cwd, capture_output=True, text=True)
