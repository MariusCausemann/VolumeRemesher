"""tests/feature_vertex_regression.cpp and tests/feature_vertex_errors.cpp: forced points and
edges on backgrounds that already have the surface's vertices, whose forcing corners would leave
the background, or at a boundary vertex with no inward axis direction; and the inputs that must
be rejected."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import volume_remesher as vr

from helpers import CUBE_TRIS, box_background, cube_coords, grid_background, orient3d


def embed(bg, surface_coords, edge_coords=None, edge_idx=None, point_coords=None):
    edges = None
    if edge_coords is not None:
        edges = (np.asarray(edge_coords, dtype=np.float64).reshape(-1, 3),
                 np.asarray(edge_idx, dtype=np.uint32).reshape(-1, 2))
    points = None if point_coords is None else np.asarray(point_coords, dtype=np.float64).reshape(-1, 3)
    return vr.embed_exact((surface_coords, CUBE_TRIS), bg, edges=edges, points=points)


def F(p):
    return tuple(Fraction(x) for x in p)


def vertex_at(r, v, p) -> bool:
    return v != -1 and v < len(r.vertices) and r.vertices[v] == F(p)


def vertices_outside_tet(r, t) -> int:
    c = [F(t[3 * i: 3 * i + 3]) for i in range(4)]
    s = orient3d(*c)
    assert s != 0
    outside = 0
    for q in r.vertices:
        for i in range(4):
            cq = list(c)
            cq[i] = q
            o = orient3d(*cq)
            if o != 0 and o != s:
                outside += 1
                break
    return outside


def vertices_outside_box(r, lo, hi) -> int:
    lo, hi = Fraction(lo), Fraction(hi)
    return sum(1 for q in r.vertices if any(x < lo or hi < x for x in q))


def tiles_segment(r, chain, A, B) -> bool:
    """The output edges `chain` tile AB: every endpoint lies on the segment and A and B are
    among them."""
    if len(chain) == 0:
        return False
    a, b = F(A), F(B)
    ab = [b[k] - a[k] for k in range(3)]
    ab2 = sum(x * x for x in ab)
    has_a = has_b = False
    for seg in chain.tolist():
        for v in seg[1:3]:
            if v < 0 or v >= len(r.vertices):
                return False
            q = r.vertices[v]
            aq = [q[d] - a[d] for d in range(3)]
            cross = (aq[1] * ab[2] - aq[2] * ab[1], aq[2] * ab[0] - aq[0] * ab[2],
                     aq[0] * ab[1] - aq[1] * ab[0])
            if any(c != 0 for c in cross):
                return False
            t = sum(aq[k] * ab[k] for k in range(3))
            if t < 0 or ab2 < t:
                return False
            has_a |= vertex_at(r, v, A)
            has_b |= vertex_at(r, v, B)
    return has_a and has_b


def check_point(r, p):
    assert len(r.tets) > 0
    assert len(r.point_provenance) == 1
    assert r.point_provenance[0, 1] != -1
    assert vertex_at(r, int(r.point_provenance[0, 1]), p)


def check_edge(r, a, b):
    assert len(r.tets) > 0
    assert len(r.edge_provenance) == 1
    assert tiles_segment(r, r.edge_provenance[0], a, b)


ROTATED_TET = [0, 0, 0, 4, 4, 6, 4, 6, 4, 6, 4, 4]


def rotated_tet_background():
    return (np.array(ROTATED_TET, dtype=np.float64).reshape(4, 3),
            np.array([[0, 1, 2, 3]], dtype=np.uint32))


def test_forced_point_on_a_surface_vertex_the_background_also_has():
    p = [0, 0, 0]
    r = embed(grid_background(), cube_coords(), point_coords=p)
    check_point(r, p)
    assert vertices_outside_box(r, -0.2, 2.2) == 0


def test_forced_edge_starting_at_a_surface_vertex_the_background_also_has():
    e = [0, 0, 0, 0.5, 0.5, 0.5]
    r = embed(grid_background(), cube_coords(), e, [0, 1])
    check_edge(r, e[:3], e[3:])
    assert vertices_outside_box(r, -0.2, 2.2) == 0


def test_forced_point_whose_forcing_corners_would_leave_the_background():
    p = [2, 2, 2]
    r = embed(grid_background(), cube_coords(), point_coords=p)
    check_point(r, p)
    assert vertices_outside_box(r, -0.2, 2.2) == 0


def test_forced_edge_whose_forcing_corners_would_leave_the_background():
    e = [2, 2, 2, 0, 0, 1.5]
    r = embed(grid_background(), cube_coords(), e, [0, 1])
    check_edge(r, e[:3], e[3:])
    assert vertices_outside_box(r, -0.2, 2.2) == 0


def test_forced_point_near_the_slanted_face_of_a_non_box_background():
    tet = [-1, -1, -1, 9, -1, -1, -1, 9, -1, -1, -1, 9]
    bg = (np.array(tet, dtype=np.float64).reshape(4, 3), np.array([[0, 1, 2, 3]], dtype=np.uint32))
    p = [2, 2, 2.9]
    r = embed(bg, cube_coords(), point_coords=p)
    check_point(r, p)
    assert vertices_outside_tet(r, tet) == 0


def test_forced_point_at_a_boundary_vertex_with_no_inward_axis_direction():
    p = [0, 0, 0]
    r = embed(rotated_tet_background(), cube_coords(1.3, 1.5), point_coords=p)
    check_point(r, p)
    assert vertices_outside_tet(r, ROTATED_TET) == 0


def test_forced_edge_from_a_boundary_vertex_with_no_inward_axis_direction():
    e = [0, 0, 0, 1, 1, 1]
    r = embed(rotated_tet_background(), cube_coords(1.3, 1.5), e, [0, 1])
    check_edge(r, e[:3], e[3:])
    assert vertices_outside_tet(r, ROTATED_TET) == 0


def test_forced_point_at_the_position_of_an_unreferenced_background_vertex():
    V, T = grid_background()
    V = np.vstack([V, [[0.0, 0.0, 0.0]]])
    p = [0, 0, 0]
    r = embed((V, T), cube_coords(), point_coords=p)
    check_point(r, p)
    assert vertices_outside_box(r, -0.2, 2.2) == 0


# --- error paths: each ends in ip_error, which reaches Python as VolumeRemesherError ----------
def test_forced_point_outside_the_tet_mesh_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="lies outside the tet mesh"):
        embed(grid_background(), cube_coords(), point_coords=[2.25, 2, 2])


def test_forced_point_on_a_boundary_face_of_the_tet_mesh_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="lies on the boundary of the tet mesh"):
        embed(grid_background(), cube_coords(), point_coords=[1, 1, 2.2])


def test_forced_edge_endpoint_on_a_boundary_edge_of_the_tet_mesh_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="lies on the boundary of the tet mesh"):
        embed(grid_background(), cube_coords(), [1, 2.2, 2.2, 1, 1, 1], [0, 1])


def test_forced_edge_with_coincident_endpoints_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="has coincident endpoints"):
        embed(grid_background(), cube_coords(), [1, 1, 1, 1, 1, 1], [0, 1])


def test_surface_vertex_outside_the_tet_mesh_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="lies outside the tet mesh"):
        embed(box_background(10.0, 11.0), cube_coords())


def test_surface_vertex_on_the_boundary_of_the_tet_mesh_is_an_error():
    with pytest.raises(vr.VolumeRemesherError, match="lies on the boundary of the tet mesh"):
        embed(box_background(0.0, 2.0), cube_coords())


def test_the_interpreter_survives_an_error_and_keeps_working():
    """What the exception machinery is for: after a rejected input, the next call works."""
    with pytest.raises(vr.VolumeRemesherError):
        embed(box_background(0.0, 2.0), cube_coords())
    r = embed(grid_background(), cube_coords(), point_coords=[1, 1, 1])
    check_point(r, [1, 1, 1])
