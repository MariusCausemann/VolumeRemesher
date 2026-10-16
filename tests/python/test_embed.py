"""tests/unit_tests_3d.cpp: the exact embedding API on tiny inputs."""

from __future__ import annotations

import numpy as np
import pytest

import volume_remesher as vr

from helpers import MASK64, has_odd_denominator_factor, in_lowest_terms


def unit_box(lo: float, hi: float):
    """Kuhn's six-tet triangulation of [lo, hi]^3; cube vertex i has bit k set iff coordinate k
    is hi. Positively oriented under ((v1-v0)x(v2-v0)).(v3-v0) > 0."""
    coords = np.array([[hi if i & 1 else lo, hi if i & 2 else lo, hi if i & 4 else lo]
                       for i in range(8)], dtype=np.float64)
    axis = [1, 2, 4]
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def positive(i0, i1, i2, i3):
        a, b, c = coords[i1] - coords[i0], coords[i2] - coords[i0], coords[i3] - coords[i0]
        return np.dot(np.cross(a, b), c) > 0.0

    tets = []
    for p in perms:
        a = axis[p[0]]
        c = axis[p[0]] | axis[p[1]]
        tets.append([0, a, c, 7] if positive(0, a, c, 7) else [0, c, a, 7])
    return coords, np.array(tets, dtype=np.uint32)


def embed(tri_coords, tri_idx, **kw):
    return vr.embed_exact((tri_coords, tri_idx), unit_box(-1.0, 2.0), **kw)


def test_exact_coordinates_in_lowest_terms():
    # Random triangles crossing each other inside the box, so most output vertices are implicit
    # with denominators that need a real GCD, and enough of them for the coordinate pass to run
    # in parallel.
    s = 0x9E3779B97F4A7C15

    def rnd():
        nonlocal s
        s ^= (s << 13) & MASK64
        s ^= s >> 7
        s ^= (s << 17) & MASK64
        return -0.5 + 2.0 * float(s >> 11) * (1.0 / 9007199254740992.0)

    num_tris = 12
    tri_coords = np.array([[rnd() for _ in range(3)] for _ in range(3 * num_tris)])
    tri_idx = np.arange(3 * num_tris, dtype=np.uint32).reshape(num_tris, 3)

    o = embed(tri_coords, tri_idx)

    assert len(o.exact_coords) >= 3 * 512
    assert all(in_lowest_terms(c) for c in o.exact_coords)
    assert any(has_odd_denominator_factor(c) for c in o.exact_coords)
    # The Fraction form agrees with the strings.
    assert o.num_vertices * 3 == len(o.exact_coords) and len(o.vertices) == o.num_vertices


def test_reports_the_coplanar_group_of_every_input_triangle():
    # Two coplanar triangles tiling the square z=0.5 (sharing the diagonal s0-s2), one triangle
    # folded out of that plane across the edge s0-s1, and one degenerate (collinear) triangle
    # that is dropped before the arrangement -- in the MIDDLE on purpose.
    tri_coords = np.array([
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5],
        [0.5, -0.5, 1.0],
        [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6],
    ])
    tri_idx = np.array([[0, 1, 2], [5, 6, 7], [0, 2, 3], [1, 0, 4]], dtype=np.uint32)

    o = embed(tri_coords, tri_idx)

    assert len(o.tets) > 0
    assert len(o.triangle_group) == len(tri_idx)
    for t in (0, 2, 3):
        g = o.triangle_group[t]
        assert 0 <= g < len(o.triangle_provenance)
        assert len(o.triangle_provenance[g]) > 0
    assert o.triangle_group[0] == o.triangle_group[2]  # edge-adjacent and coplanar
    assert o.triangle_group[3] != o.triangle_group[0]  # edge-adjacent, not coplanar
    assert o.triangle_group[1] == -1  # dropped as degenerate

    # The faces attributed to a group are faces of the tets they name.
    faces = o.triangle_provenance.flat
    tets = o.tets[faces[:, 0]]
    for k in range(1, 4):
        assert np.all((faces[:, k][:, None] == tets).any(axis=1))


def test_non_coplanar_triangles_get_distinct_groups():
    tri_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    tri_idx = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]], dtype=np.uint32)
    o = embed(tri_coords, tri_idx)
    assert len(o.triangle_group) == 3
    assert len(set(o.triangle_group.tolist())) == 3


def test_polyhedral_complex_is_consistent():
    tri_coords = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5]])
    o = embed(tri_coords, np.array([[0, 1, 2]], dtype=np.uint32))
    nv, nf, nc = o.num_vertices, len(o.facets), len(o.cells)
    assert nv > 8 and nf > 0 and nc >= 6
    assert o.facets.flat.max() < nv
    assert o.cells.flat.max() < nf
    assert np.all(o.facets.sizes() >= 3)
    assert np.all(o.cells.sizes() >= 4)
    assert o.facets_on_input.max() < nf
    assert len(o.cells_with_faces_on_input) == nc and o.cells_with_faces_on_input.dtype == bool
    assert len(o.tets_parent) == len(o.tets) and o.tets_parent.max() < nc
    assert len(o.tets_parent_faces) == len(o.tets)
    assert o.tets.max() < nv
