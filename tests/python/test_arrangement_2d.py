"""tests/unit_tests_2d.cpp, the arrangement and 2D embed cases, through the public API: the
output triangles are strictly positively oriented in exact arithmetic, consistently wound and
non-overlapping, and every input segment is EXACTLY a union of output edges (collinear, both
endpoints reached, no gap, no overlap). The predicate and Delaunay internals stay C++ only."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import volume_remesher as vr

from helpers import Rnd, in_lowest_terms, tris_consistently_oriented


def check_provenance(A: vr.Arrangement, P: np.ndarray, S: np.ndarray) -> str:
    """The exact form of verify_tracking_2d's check, on the API output."""
    mesh_edges = set()
    for a, b, c in A.triangles.tolist():
        for e in ((a, b), (b, c), (c, a)):
            mesh_edges.add((min(e), max(e)))
    X = A.exact_vertices
    assert X is not None
    for s in range(len(S)):
        chain = A.segment_provenance[s]
        p0 = (Fraction(float(P[S[s, 0], 0])), Fraction(float(P[S[s, 0], 1])))
        p1 = (Fraction(float(P[S[s, 1], 0])), Fraction(float(P[S[s, 1], 1])))
        if p0 == p1:
            if len(chain):
                return f"segment {s}: zero length but has output edges"
            continue
        if len(chain) == 0:
            return f"segment {s}: no output edges"
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        len2 = dx * dx + dy * dy
        iv = []
        for tri, v0, v1 in chain.tolist():
            if (min(v0, v1), max(v0, v1)) not in mesh_edges:
                return f"segment {s}: recorded sub-edge is not an edge of the output triangulation"
            if not {v0, v1} <= set(A.triangles[tri].tolist()):
                return f"segment {s}: sub-edge not on its triangle"
            tt = []
            for v in (v0, v1):
                qx, qy = X[v][0] - p0[0], X[v][1] - p0[1]
                if qx * dy - qy * dx != 0:
                    return f"segment {s}: output edge not collinear"
                tt.append((qx * dx + qy * dy) / len2)
            if tt[0] < 0 or tt[1] < 0 or tt[0] > 1 or tt[1] > 1:
                return f"segment {s}: parameter outside [0,1]"
            if not tt[0] < tt[1]:
                return f"segment {s}: output edge reversed or degenerate"
            iv.append((tt[0], tt[1]))
        iv.sort()
        if iv[0][0] != 0 or iv[-1][1] != 1:
            return f"segment {s}: output edges do not reach both endpoints"
        for i in range(1, len(iv)):
            if iv[i][0] != iv[i - 1][1]:
                return f"segment {s}: gap or overlap between output edges"
    return ""


def check_all(coords, idx, what: str) -> vr.Arrangement:
    P = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    S = np.asarray(idx, dtype=np.uint32).reshape(-1, 2)
    A = vr.arrange_2d((P, S), exact=True)
    X = A.exact_vertices
    assert len(A.triangles) > 0, what
    # every output triangle strictly positively oriented, exactly
    for t, (a, b, c) in enumerate(A.triangles.tolist()):
        area2 = (X[b][0] - X[a][0]) * (X[c][1] - X[a][1]) - (X[b][1] - X[a][1]) * (X[c][0] - X[a][0])
        assert area2 > 0, f"{what}: triangle {t} has non-positive exact area"
    # orientable and manifold
    assert tris_consistently_oriented(A.triangles) == (0, 0), what
    # the doubles are the rounded exact values
    assert np.allclose(A.vertices, [[float(x) for x in v] for v in X]), what
    # provenance
    err = check_provenance(A, P, S)
    assert err == "", f"{what}: {err}"
    return A


class TestHandBuiltDegenerateCases:
    def test_two_crossing_segments(self):
        check_all([0, 0, 10, 10, 0, 10, 10, 0], [0, 1, 2, 3], "X")

    def test_three_segments_concurrent_at_one_point(self):
        A = check_all([-10, 0, 10, 0, 0, -10, 0, 10, -10, -10, 10, 10], [0, 1, 2, 3, 4, 5],
                      "three concurrent")
        # One intersection vertex: 6 input points + 4 box corners + 1 crossing.
        assert len(A.vertices) == 6 + 4 + 1

    def test_duplicate_segments_both_orientations(self):
        check_all([0, 0, 10, 0], [0, 1, 0, 1, 1, 0], "duplicates")

    def test_zero_length_segments_are_dropped(self):
        A = check_all([0, 0, 10, 0, 5, 5], [0, 0, 0, 1, 2, 2], "zero-length")
        assert len(A.segment_provenance[0]) == 0
        assert len(A.segment_provenance[2]) == 0
        assert len(A.segment_provenance[1]) > 0

    def test_duplicate_input_points_are_merged(self):
        A = check_all([0, 0, 10, 0, 0, 0, 5, 5], [0, 1, 2, 3], "duplicate points")
        assert A.point_provenance[0, 1] == A.point_provenance[2, 1]
        assert len(A.vertices) == 3 + 4

    def test_partially_overlapping_collinear_chain(self):
        check_all([0, 0, 4, 0, 2, 0, 6, 0], [0, 1, 2, 3], "collinear overlap")

    def test_fully_overlapping_collinear_chain(self):
        check_all([0, 0, 6, 0, 2, 0, 4, 0], [0, 1, 2, 3], "collinear containment")

    def test_segments_sharing_an_endpoint(self):
        check_all([0, 0, 5, 5, 5, -5], [0, 1, 0, 2], "shared endpoint")

    def test_closed_triangle_outline(self):
        check_all([0, 0, 10, 0, 5, 8], [0, 1, 1, 2, 2, 0], "triangle")

    def test_t_junction(self):
        check_all([0, 0, 10, 0, 5, 0, 5, 7], [0, 1, 2, 3], "T-junction")


@pytest.mark.parametrize("k", [3, 5, 8])
def test_grids_of_crossing_segments(k):
    c, s = [], []
    for i in range(k):
        c += [float(i), -1.0, float(i), float(k), -1.0, float(i), float(k), float(i)]
    for i in range(0, 4 * k, 2):
        s += [i, i + 1]
    A = check_all(c, s, "grid")
    # Every vertical meets every horizontal exactly once, and no three are concurrent.
    assert len(A.vertices) == 4 * k + 4 + k * k


@pytest.mark.parametrize("n", [5, 20, 60])
def test_random_segment_soups(n):
    rnd = Rnd(0xA11CE + n)
    c, s = [], []
    for i in range(n):
        c += [rnd.unit(), rnd.unit(), rnd.unit(), rnd.unit()]
        s += [(4 * i) // 2, (4 * i) // 2 + 1]
    check_all(c, s, "random soup")


@pytest.mark.parametrize("n", [10, 40, 100])
def test_random_segments_on_a_small_integer_lattice(n):
    rnd = Rnd(0x1A771CE + n)
    c, s = [], []
    for i in range(n):
        c += [rnd.coord(8), rnd.coord(8), rnd.coord(8), rnd.coord(8)]
        s += [2 * i, 2 * i + 1]
    check_all(c, s, "lattice soup")


def test_heavy_collinear_overlap():
    rnd = Rnd(0xFEED)
    c, idx = [], []
    for i in range(480):
        y = float(i % 20)
        st = rnd.coord(20)
        en = st + 1 + rnd.coord(19)
        c += [st, y, en, y]
        idx += [2 * i, 2 * i + 1]
    check_all(c, idx, "heavy collinear overlap")


def test_dense_lattice_concurrency():
    rnd = Rnd(0x1A77 + 500)
    c, idx = [], []
    for i in range(160):
        c += [rnd.coord(64) for _ in range(4)]
        idx += [2 * i, 2 * i + 1]
    check_all(c, idx, "dense lattice concurrency")


class TestCollinearAndDegenerateDomains:
    def test_a_single_segment(self):
        check_all([0.0, 0.0, 3.0, 1.0], [0, 1], "single segment")

    def test_two_collinear_segments_disjoint(self):
        check_all([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0], [0, 1, 2, 3], "disjoint collinear")

    def test_two_collinear_segments_meeting_at_a_point(self):
        check_all([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0], [0, 1, 2, 3], "touching collinear")

    def test_an_all_collinear_polyline(self):
        c, idx = [], []
        for i in range(12):
            c += [float(i), 2.0 * i]
        for i in range(11):
            idx += [i, i + 1]
        check_all(c, idx, "collinear polyline")

    def test_all_points_on_a_horizontal_line(self):
        c, idx = [], []
        for i in range(8):
            c += [float(i), 5.0]
        for i in range(7):
            idx += [i, i + 1]
        check_all(c, idx, "horizontal, zero y extent")

    def test_all_points_on_a_vertical_line_through_the_origin(self):
        c, idx = [], []
        for i in range(8):
            c += [0.0, -float(i)]
        for i in range(7):
            idx += [i, i + 1]
        check_all(c, idx, "vertical through origin")

    def test_a_single_degenerate_point_no_usable_segment(self):
        A = vr.arrange_2d((np.array([[4.0, 7.0], [4.0, 7.0]]), np.array([[0, 1]], dtype=np.uint32)))
        assert len(A.segment_provenance) == 1 and len(A.segment_provenance[0]) == 0
        assert len(A.vertices) == 1 + 4  # the point and the four box corners
        assert tris_consistently_oriented(A.triangles) == (0, 0)
        assert A.point_provenance[0, 1] == A.point_provenance[1, 1] != -1


def test_2d_embed_exact_coordinates_come_back_in_lowest_terms():
    r = Rnd(7)
    c, idx = [], []
    for i in range(80):
        c += [r.unit() for _ in range(4)]
        idx += [2 * i, 2 * i + 1]
    P = np.asarray(c).reshape(-1, 2)
    S = np.asarray(idx, dtype=np.uint32).reshape(-1, 2)
    e = vr.arrange_2d_exact((P, S))
    assert len(e.exact_coords) >= 2 * 512
    assert all(in_lowest_terms(s) for s in e.exact_coords)
    # and it is the same triangulation arrange_2d gives
    A = vr.arrange_2d((P, S), exact=True)
    assert np.array_equal(A.triangles, e.triangles)
    assert A.exact_vertices == e.vertices
    assert np.array_equal(A.point_provenance, e.point_provenance)
    assert np.array_equal(A.segment_provenance.flat, e.segment_provenance.flat)
