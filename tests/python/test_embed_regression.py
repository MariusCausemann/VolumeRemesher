"""tests/embed_regression.cpp: embed_tri_in_poly_mesh on the replay capture of tetwild's call for
Thingi10K model 100727 must emit positively oriented tets only. Hidden behind the `slow` marker:
the embedding takes about 4 minutes and the exact check of the ~6M tets the float filter cannot
decide another 10 or so (12M tets, 1.9M vertices; 17 minutes in all on a loaded M3 Max). Run with
`pytest -m slow`."""

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
    # rational arithmetic. The bound must allow for the coordinates themselves being rounded
    # (each by up to eps*|coordinate|): two close vertices then give a difference whose relative
    # error is large, so Shewchuk's permanent-based bound (exact inputs) does not apply here.
    V = np.array([[float(x) for x in v] for v in o.vertices])
    T = o.tets.astype(np.int64)
    a, b, c, d = (V[T[:, k]] for k in range(4))
    u, v, w = b - a, c - a, d - a
    det = (u[:, 0] * (v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1])
           - u[:, 1] * (v[:, 0] * w[:, 2] - v[:, 2] * w[:, 0])
           + u[:, 2] * (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]))
    eps = 2.0 ** -53
    A = np.max(np.abs(np.stack([a, b, c, d], axis=0)), axis=(0, 2))  # per tet, largest |coordinate|
    delta = 3.0 * eps * A  # absolute error of every component of u, v, w
    au, av, aw = np.abs(u), np.abs(v), np.abs(w)
    S = np.zeros(len(T))
    permanent = np.zeros(len(T))
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        S += av[:, j] * aw[:, k] + au[:, i] * aw[:, k] + au[:, i] * av[:, j]
        permanent += au[:, i] * av[:, j] * aw[:, k]
    bound = 2.0 * (delta * S + 8.0 * eps * permanent)
    undecided = np.flatnonzero(np.abs(det) <= bound)
    assert np.all(det[np.abs(det) > bound] > 0), "a tet has clearly negative volume"
    bad = [int(t) for t in undecided
           if orient3d(*(o.vertices[k] for k in T[t])) <= 0]
    assert bad == [], f"tets with non-positive exact volume: {bad[:10]} ({len(bad)} total)"

    # 2. Combinatorial: no face shared by more than two tets, none presented twice with the same
    # winding.
    assert tets_consistently_oriented(o.tets) == (0, 0)
