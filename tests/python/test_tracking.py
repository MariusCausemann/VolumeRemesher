"""tests/integration_tests.cpp, the "-t ... tracked exactly" cases: the meshes are produced
through the Python API and checked by the independent exact-rational verifiers
(verify_tracking / verify_tracking_2d from the CMake build), which share no code with the
generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import volume_remesher as vr

from conftest import MODELS, model, run_tool

pytestmark = pytest.mark.needs_cli

# (model, strict): see the C++ test for why 112856simplified is lenient.
TRACK_MODELS = [
    ("cube_subdiv.off", True), ("two_cubes.off", True), ("cube_on_cube.off", True),
    ("upsample_box.off", True), ("sphere.off", True), ("double_sphere.off", True),
    ("open_box.off", True), ("112856simplified.off", False),
    ("simwild_double_sphere_notop.off", True),
]


def verify(verify_tracking: Path, cwd: Path, surface, extra: list[str]) -> str:
    r = run_tool(verify_tracking, [surface, "volume.tet", *extra], cwd=cwd)
    return r.stdout + r.stderr


@pytest.mark.parametrize("name,strict", TRACK_MODELS, ids=[m for m, _ in TRACK_MODELS])
def test_face_provenance_is_exact(verify_tracking, name, strict, tmp_path):
    m = model(name)
    vr.remesh(m, tetrahedralize=True, exact=True, keep_all_cells=True).save(tmp_path / "volume.tet")
    out = verify(verify_tracking, tmp_path, m, ["--strict"] if strict else [])
    assert "TRACKING OK" in out, out


@pytest.mark.parametrize("name,edges,points", [
    ("cube_subdiv.off", "cube_subdiv_edges.off", "cube_subdiv_points.off"),
    ("sphere.off", "sphere_edges.off", "sphere_points.off"),
])
def test_inserted_edges_and_points_are_tracked_exactly(verify_tracking, name, edges, points, tmp_path):
    m, e, p = model(name), model(edges), model(points)
    vr.remesh(m, edges=e, points=p, tetrahedralize=True, exact=True, keep_all_cells=True).save(
        tmp_path / "volume.tet")
    out = verify(verify_tracking, tmp_path, m, ["--strict", "--edges", e, "--points", p])
    assert "TRACKING OK" in out, out


# --- the large Release-only cases ------------------------------------------------------------
def _rnd(seed: int):
    """A fixed-seed generator. The C++ test draws from std::mt19937_64, which numpy does not
    provide; these inputs therefore differ from the C++ test's but are reproducible, which is
    what the test needs (the check is exact on whatever the input is)."""
    return np.random.default_rng(seed)


def write_random_points(path: Path, n: int, seed: int):
    g = _rnd(seed)
    P = g.uniform(0.05, 0.95, size=(n, 3))
    with open(path, "w") as f:
        f.write(f"OFF\n{n} 0 0\n")
        for p in P:
            f.write(f"{p[0]!r} {p[1]!r} {p[2]!r}\n")


def write_random_edges(path: Path, n: int, seed: int):
    g = _rnd(seed)
    A = g.uniform(0.05, 0.95, size=(n, 3))
    B = A + g.uniform(-0.03, 0.03, size=(n, 3))
    with open(path, "w") as f:
        f.write(f"OFF\n{2 * n} {n} 0\n")
        for a, b in zip(A, B):
            f.write(f"{a[0]!r} {a[1]!r} {a[2]!r}\n{b[0]!r} {b[1]!r} {b[2]!r}\n")
        for i in range(n):
            f.write(f"2 {2 * i} {2 * i + 1}\n")


def extract_edges_off(model_path: Path, out: Path):
    V, F = vr.read_off(model_path)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    with open(out, "w") as f:
        f.write(f"OFF\n{len(V)} {len(E)} 0\n")
        for v in V:
            f.write(f"{v[0]!r} {v[1]!r} {v[2]!r}\n")
        for a, b in E:
            f.write(f"2 {a} {b}\n")


COMBOS = [
    ("tri", True, False, False), ("edges", False, True, False), ("points", False, False, True),
    ("tri+edges", True, True, False), ("tri+points", True, False, True),
    ("edges+points", False, True, True), ("tri+edges+points", True, True, True),
]


@pytest.mark.large
@pytest.mark.parametrize("name,tri,edge,point", COMBOS, ids=[c[0] for c in COMBOS])
def test_all_input_combinations_tracked_exactly(verify_tracking, name, tri, edge, point, tmp_path):
    N = 1000
    surface = model("cube_subdiv.off")
    edges = tmp_path / "random_edges.off"
    points = tmp_path / "random_points.off"
    write_random_edges(edges, N, 12345)
    write_random_points(points, N, 67890)
    vr.remesh(
        surface if tri else None,
        edges=edges if edge else None,
        points=points if point else None,
        tetrahedralize=True, exact=True, keep_all_cells=True,
    ).save(tmp_path / "volume.tet")
    extra = ["--strict"]
    if edge:
        extra += ["--edges", edges]
    if point:
        extra += ["--points", points]
    out = verify(verify_tracking, tmp_path, surface if tri else "none", extra)
    assert "TRACKING OK" in out, out


@pytest.mark.large
def test_all_edges_of_a_real_mesh_inserted_as_edges(verify_tracking, tmp_path):
    m = model("112856simplified.off")
    edges = tmp_path / "all_edges.off"
    extract_edges_off(m, edges)
    vr.remesh(edges=edges, tetrahedralize=True, exact=True, keep_all_cells=True).save(
        tmp_path / "volume.tet")
    out = verify(verify_tracking, tmp_path, "none", ["--strict", "--edges", edges])
    assert "TRACKING OK" in out, out


# --- 2D ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["2d/england.obj", "2d/siggraph_logo.obj"])
def test_2d_segment_provenance_is_exact(verify_tracking_2d, name, tmp_path):
    m = model(name)
    vr.arrange_2d(m, exact=True).save(tmp_path / "triangulation.off")
    r = run_tool(verify_tracking_2d, [m, "triangulation.off"], cwd=tmp_path)
    out = r.stdout + r.stderr
    assert "TRACKING OK" in out, out
