"""Checks shared by the tests: hashing, combinatorial orientation (tet_orientation.h /
tri_orientation.h), exact rationals, the C++ tests' random generator and their tet backgrounds."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

MASK64 = (1 << 64) - 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --- orientation (the port of tets/tri_orientation.h) -------------------------------------
def _canonical_cycles(faces: np.ndarray) -> np.ndarray:
    """Each row rotated so its smallest index comes first; the winding is preserved."""
    n, k = faces.shape
    if k == 2:
        return faces  # a directed edge: rotating it would reverse it
    shift = np.argmin(faces, axis=1)
    cols = (np.arange(k)[None, :] + shift[:, None]) % k
    return faces[np.arange(n)[:, None], cols]


def check_orientation(faces: np.ndarray) -> tuple[int, int]:
    """Given the oriented boundary faces of every simplex (one row each), count the faces two
    simplices present with the same winding (same_winding) and the faces shared by more than two
    (over_shared). Both must be 0 for a consistently oriented, non-overlapping complex."""
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return 0, 0
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    over_shared = int(np.count_nonzero(counts > 2))
    _, counts = np.unique(_canonical_cycles(faces), axis=0, return_counts=True)
    same_winding = int(np.sum(counts[counts > 1] - 1))
    return same_winding, over_shared


def tet_boundary_faces(tets) -> np.ndarray:
    """The 4 outward faces of each tet (a,b,c,d) with (b-a)x(c-a).(d-a) > 0."""
    t = np.asarray(tets, dtype=np.int64)
    a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    return np.vstack([np.column_stack(f) for f in ((b, c, d), (a, d, c), (a, b, d), (a, c, b))])


def tri_boundary_edges(tris) -> np.ndarray:
    t = np.asarray(tris, dtype=np.int64)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return np.vstack([np.column_stack(e) for e in ((a, b), (b, c), (c, a))])


def tets_consistently_oriented(tets) -> tuple[int, int]:
    return check_orientation(tet_boundary_faces(tets))


def tris_consistently_oriented(tris) -> tuple[int, int]:
    return check_orientation(tri_boundary_edges(tris))


def read_tet_file(path: Path) -> np.ndarray:
    """The tets of a .tet file as written by mesh_generator -t."""
    with open(path) as f:
        nv = int(f.readline().split()[0])
        nt = int(f.readline().split()[0])
        for _ in range(nv):
            f.readline()
        body = f.read()
    tets = np.array(body.split(), dtype=np.int64).reshape(nt, 5)[:, 1:]
    return tets


def read_off_faces(path: Path) -> np.ndarray:
    with open(path) as f:
        assert f.readline().strip() == "OFF"
        nv, nf, _ = (int(x) for x in f.readline().split())
        for _ in range(nv):
            f.readline()
        body = f.read()
    rows = body.split("\n")[:nf]
    return np.array([[int(x) for x in r.split()[1:]] for r in rows], dtype=np.int64)


# --- exact rationals ----------------------------------------------------------------------
def parse_rational(s: str) -> tuple[int, int]:
    """'[-]num[/den]' -> (num, den) as written, unreduced."""
    if "/" in s:
        n, d = s.split("/")
        return int(n), int(d)
    return int(s), 1


def in_lowest_terms(s: str) -> bool:
    num, den = parse_rational(s)
    if num == 0:
        return True
    return den > 0 and math.gcd(abs(num), den) == 1


def has_odd_denominator_factor(s: str) -> bool:
    """The denominator is not a power of two, i.e. reducing took a real GCD."""
    num, den = parse_rational(s)
    if num == 0:
        return False
    return (den & (den - 1)) != 0


def orient3d(a, b, c, d) -> int:
    """Sign of det[b-a, c-a, d-a] for points given as Fraction (or int) triples."""
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    w = [d[k] - a[k] for k in range(3)]
    det = (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
           + u[2] * (v[0] * w[1] - v[1] * w[0]))
    return (det > 0) - (det < 0)


# --- the C++ tests' xorshift generator (unit_tests_2d.cpp Rnd) ------------------------------
class Rnd:
    def __init__(self, seed: int):
        self.s = seed if seed else 0x9E3779B97F4A7C15

    def next(self) -> int:
        s = self.s
        s ^= (s << 13) & MASK64
        s ^= s >> 7
        s ^= (s << 17) & MASK64
        self.s = s
        return s

    def coord(self, rng: int) -> float:
        return float((self.next() % (2 * rng)) - rng)

    def unit(self) -> float:
        return float(self.next() >> 11) * (1.0 / 9007199254740992.0)


# --- tet backgrounds (feature_vertex_fixtures.h / unit_tests_3d.cpp) --------------------------
def _kuhn_cell(coords: list[float], tets: list[int], vid, i: int, j: int, k: int) -> None:
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for p in perms:
        c = [i, j, k]
        t = [vid(*c)]
        for s in range(3):
            c[p[s]] += 1
            t.append(vid(*c))
        a = coords[3 * t[0]: 3 * t[0] + 3]
        e = [[coords[3 * t[s + 1] + d] - a[d] for d in range(3)] for s in range(3)]
        vol = (e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
               - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
               + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]))
        if vol < 0:
            t[2], t[3] = t[3], t[2]
        tets.extend(t)


def grid_background():
    """Lines at -0.2, 0, 2, 2.2 on each axis, each cell split into 6 Kuhn tets."""
    xs = [-0.2, 0.0, 2.0, 2.2]
    coords: list[float] = []
    for k in range(4):
        for j in range(4):
            for i in range(4):
                coords += [xs[i], xs[j], xs[k]]
    tets: list[int] = []
    vid = lambda i, j, k: i + 4 * j + 16 * k  # noqa: E731
    for k in range(3):
        for j in range(3):
            for i in range(3):
                _kuhn_cell(coords, tets, vid, i, j, k)
    return np.array(coords).reshape(-1, 3), np.array(tets, dtype=np.uint32).reshape(-1, 4)


def box_background(lo: float, hi: float):
    """The box [lo, hi]^3 as 6 Kuhn tets."""
    coords: list[float] = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                coords += [hi if i else lo, hi if j else lo, hi if k else lo]
    tets: list[int] = []
    _kuhn_cell(coords, tets, lambda i, j, k: i + 2 * j + 4 * k, 0, 0, 0)
    return np.array(coords).reshape(-1, 3), np.array(tets, dtype=np.uint32).reshape(-1, 4)


def cube_coords(lo: float = 0.0, hi: float = 2.0) -> np.ndarray:
    c = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    return np.array([lo if x == 0 else hi for x in c], dtype=np.float64).reshape(-1, 3)


CUBE_TRIS = np.array(
    [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
     2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3],
    dtype=np.uint32,
).reshape(-1, 3)
