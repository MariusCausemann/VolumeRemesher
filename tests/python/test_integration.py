"""tests/integration_tests.cpp, "model outputs are byte-stable across platforms": every entry of
tests/reference_hashes.txt, produced through the Python API and through `python -m
volume_remesher`, must hash to the committed reference. The references were made by the C++ tool,
so this is the bit-identity check against it that needs no binary around."""

from __future__ import annotations

from pathlib import Path

import pytest

import volume_remesher as vr

from conftest import MANIFEST, MODELS, run_python_cli
from helpers import (read_off_faces, read_tet_file, sha256_file, tets_consistently_oriented,
                     tris_consistently_oriented)

FLAGS = {"default": [], "blackfaces": ["-b"], "skin": ["-s"], "tet": ["-t"], "2d": ["-2d"]}
OUTPUT = {"default": "volume.msh", "blackfaces": "black_faces.off", "skin": "skin.off",
          "tet": "volume.tet", "2d": "triangulation.off"}


def load_manifest():
    entries = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f, mode, digest = line.split()
        entries.append((f, mode, digest))
    assert entries
    return entries


ENTRIES = load_manifest()


def produce_with_api(model: Path, mode: str, out: Path) -> None:
    if mode == "2d":
        vr.arrange_2d(model).save(out)
    elif mode == "tet":
        vr.remesh(model, tetrahedralize=True).save(out)
    elif mode == "skin":
        vr.remesh(model, output="skin").save(out)
    elif mode == "blackfaces":
        vr.remesh(model, output="black_faces").save(out)
    else:
        vr.remesh(model).save(out)


def check_valid_complex(mode: str, out: Path) -> None:
    """The .tet must be a consistently oriented, non-overlapping tet complex; the 2D
    triangulation likewise (combinatorial, cheap on the big models too)."""
    if mode == "tet":
        same, over = tets_consistently_oriented(read_tet_file(out))
        assert (same, over) == (0, 0)
    if mode == "2d":
        same, over = tris_consistently_oriented(read_off_faces(out))
        assert (same, over) == (0, 0)


@pytest.mark.parametrize("name,mode,expected", ENTRIES, ids=[f"{f}[{m}]" for f, m, _ in ENTRIES])
def test_api_output_matches_reference_hash(name, mode, expected, tmp_path):
    model = MODELS / name
    if not model.is_file():
        pytest.skip(f"model not present: {name}")
    out = tmp_path / OUTPUT[mode]
    produce_with_api(model, mode, out)
    assert out.is_file()
    assert sha256_file(out) == expected
    check_valid_complex(mode, out)


@pytest.mark.parametrize("name,mode,expected", ENTRIES, ids=[f"{f}[{m}]" for f, m, _ in ENTRIES])
def test_python_cli_output_matches_reference_hash(name, mode, expected, tmp_path):
    model = MODELS / name
    if not model.is_file():
        pytest.skip(f"model not present: {name}")
    r = run_python_cli([model, *FLAGS[mode]], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    out = tmp_path / OUTPUT[mode]
    assert out.is_file()
    assert sha256_file(out) == expected
