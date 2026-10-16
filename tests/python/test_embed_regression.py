"""tests/embed_regression.cpp: embed_tri_in_poly_mesh on the replay capture of tetwild's call for
Thingi10K model 100727 must emit positively oriented tets only. Hidden behind the `slow` marker
(the embedding alone takes minutes); run with `pytest -m slow`."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import volume_remesher as vr

from helpers import orient3d, tets_consistently_oriented

pytestmark = pytest.mark.slow


def read_embed_input(path: Path):
    """Layout: 8-byte magic "VREMBD01", then four length-prefixed arrays (uint64 length,
    native-endian double / uint32 payload)."""
    with open(path, "rb") as f:
        assert f.read(8) == b"VREMBD01"

        def arr(dtype):
            (n,) = struct.unpack("<Q", f.read(8))
            return np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype).copy()

        tri_coords = arr(np.float64).reshape(-1, 3)
        tri_idx = arr(np.uint32).reshape(-1, 3)
        tet_coords = arr(np.float64).reshape(-1, 3)
        tet_idx = arr(np.uint32).reshape(-1, 4)
    return tri_coords, tri_idx, tet_coords, tet_idx


def test_emits_positively_oriented_tets(data_dir):
    path = data_dir / "embed_thingi100727.bin"
    if not path.is_file():
        pytest.skip("replay capture not present")
    tri_coords, tri_idx, tet_coords, tet_idx = read_embed_input(path)

    o = vr.embed_exact((tri_coords, tri_idx), (tet_coords, tet_idx))
    assert len(o.tets) > 0

    # The coplanar-group map is the key for the provenance: one entry per input triangle.
    assert len(o.triangle_group) == len(tri_idx)
    g = o.triangle_group
    assert np.all((g == -1) | (g < len(o.triangle_provenance)))

    # 1. Geometric: every tet has strictly positive exact signed volume. A floating point filter
    # on the double approximations settles the clear cases; the rest are decided in exact
    # rational arithmetic (Shewchuk's orient3d error bound, generously rounded up).
    V = np.array([[float(x) for x in v] for v in o.vertices])
    T = o.tets.astype(np.int64)
    a, b, c, d = (V[T[:, k]] for k in range(4))
    u, v, w = b - a, c - a, d - a
    det = (u[:, 0] * (v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1])
           - u[:, 1] * (v[:, 0] * w[:, 2] - v[:, 2] * w[:, 0])
           + u[:, 2] * (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]))
    permanent = (np.abs(u[:, 0]) * (np.abs(v[:, 1] * w[:, 2]) + np.abs(v[:, 2] * w[:, 1]))
                 + np.abs(u[:, 1]) * (np.abs(v[:, 0] * w[:, 2]) + np.abs(v[:, 2] * w[:, 0]))
                 + np.abs(u[:, 2]) * (np.abs(v[:, 0] * w[:, 1]) + np.abs(v[:, 1] * w[:, 0])))
    bound = 1e-14 * permanent  # ~ 12x Shewchuk's o3derrboundA, plus the Fraction->double rounding
    undecided = np.flatnonzero(np.abs(det) <= bound)
    assert np.all(det[np.abs(det) > bound] > 0), "a tet has clearly negative volume"
    bad = [int(t) for t in undecided
           if orient3d(*(o.vertices[k] for k in T[t])) <= 0]
    assert bad == [], f"tets with non-positive exact volume: {bad[:10]} ({len(bad)} total)"

    # 2. Combinatorial: no face shared by more than two tets, none presented twice with the same
    # winding.
    assert tets_consistently_oriented(o.tets) == (0, 0)
