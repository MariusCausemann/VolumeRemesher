// Regression: forcing points and edges into an embedding
//  - at a surface vertex that the background mesh also has: wildmeshing-toolkit's tetwild builds
//    the background from the input vertices plus box points, so every surface vertex has a
//    coincident background vertex, which inserting the surface copy evicts from every tet;
//  - whose forcing-triangle corners would leave the background: the box is padded by diag/15,
//    less than a point's forcing displacement on small inputs;
//  - at a boundary vertex of the background whose inward cone holds no axis direction (the apex
//    of a rotated tet), so the corners have to come from the tet itself;
//  - at the position of a background vertex that no tet references.
// The grid background is tetwild's at a small scale: lines through the cube's corners, with a
// padding of 0.2 against a point displacement of 0.1 x the mean surface edge (0.228).

#include <catch2/catch_test_macros.hpp>

#include "feature_vertex_fixtures.h"

#include <array>
#include <cstdint>
#include <vector>

namespace {

using namespace feature_vertex_fixtures;
using vol_rem::bigrational;
using Point = std::array<bigrational, 3>;

Point at(const double* p)
{
    return {bigrational(p[0]), bigrational(p[1]), bigrational(p[2])};
}

bool valid_vertex(const Result& r, uint32_t v)
{
    return v != UINT32_MAX && 3 * size_t(v) + 3 <= r.coords.size();
}

Point at(const Result& r, uint32_t v)
{
    return {r.coords[3 * v], r.coords[3 * v + 1], r.coords[3 * v + 2]};
}

// The output vertex v sits exactly at p.
bool vertex_at(const Result& r, uint32_t v, const double* p)
{
    if (!valid_vertex(r, v)) return false;
    const Point q = at(r, v);
    for (int d = 0; d < 3; d++)
        if (!(q[d] == bigrational(p[d]))) return false;
    return true;
}

// The sign of det[b - a, c - a, d - a], exactly.
int orient(const Point& a, const Point& b, const Point& c, const Point& d)
{
    Point u, v, w;
    for (int k = 0; k < 3; k++) {
        u[k] = b[k] - a[k];
        v[k] = c[k] - a[k];
        w[k] = d[k] - a[k];
    }
    const bigrational det = u[0] * (v[1] * w[2] - v[2] * w[1]) -
                            u[1] * (v[0] * w[2] - v[2] * w[0]) +
                            u[2] * (v[0] * w[1] - v[1] * w[0]);
    return det.sgn();
}

// The output vertices outside the closed tet with the 4 corners t (12 doubles).
size_t vertices_outside_tet(const Result& r, const double* t)
{
    std::array<Point, 4> c;
    for (int i = 0; i < 4; i++) c[i] = at(t + 3 * i);
    const int s = orient(c[0], c[1], c[2], c[3]);
    REQUIRE(s != 0);
    size_t outside = 0;
    for (size_t v = 0; 3 * v < r.coords.size(); v++) {
        const Point q = at(r, uint32_t(v));
        for (int i = 0; i < 4; i++) {
            std::array<Point, 4> cq = c;
            cq[i] = q;
            const int o = orient(cq[0], cq[1], cq[2], cq[3]);
            if (o != 0 && o != s) {
                outside++;
                break;
            }
        }
    }
    return outside;
}

// The output vertices outside the closed box [lo, hi]^3.
size_t vertices_outside_box(const Result& r, double lo, double hi)
{
    const bigrational l(lo), h(hi);
    size_t outside = 0;
    for (size_t v = 0; 3 * v < r.coords.size(); v++)
        for (int d = 0; d < 3; d++)
            if (r.coords[3 * v + d] < l || h < r.coords[3 * v + d]) {
                outside++;
                break;
            }
    return outside;
}

// The output edges 'chain' tile the segment AB: each of their vertices lies on the segment, and A
// and B are among them.
bool tiles_segment(
    const Result& r,
    const std::vector<std::array<uint32_t, 3>>& chain,
    const double* A,
    const double* B)
{
    if (chain.empty()) return false;
    const Point a = at(A), b = at(B);
    Point ab;
    for (int k = 0; k < 3; k++) ab[k] = b[k] - a[k];
    const bigrational ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    bool has_a = false, has_b = false;
    for (const auto& seg : chain)
        for (int k = 1; k < 3; k++) {
            const uint32_t v = seg[k];
            if (!valid_vertex(r, v)) return false;
            const Point q = at(r, v);
            Point aq;
            for (int d = 0; d < 3; d++) aq[d] = q[d] - a[d];
            if ((aq[1] * ab[2] - aq[2] * ab[1]).sgn() != 0 ||
                (aq[2] * ab[0] - aq[0] * ab[2]).sgn() != 0 ||
                (aq[0] * ab[1] - aq[1] * ab[0]).sgn() != 0)
                return false; // not on the line AB
            const bigrational t = aq[0] * ab[0] + aq[1] * ab[1] + aq[2] * ab[2];
            if (t.sgn() < 0 || ab2 < t) return false; // on the line, but beyond A or B
            if (vertex_at(r, v, A)) has_a = true;
            if (vertex_at(r, v, B)) has_b = true;
        }
    return has_a && has_b;
}

void check_point(const Result& r, const double* p)
{
    REQUIRE(!r.tets.empty());
    REQUIRE(r.point_provenance.size() == 1);
    REQUIRE(r.point_provenance[0][1] != UINT32_MAX);
    CHECK(vertex_at(r, r.point_provenance[0][1], p));
}

void check_edge(const Result& r, const double* a, const double* b)
{
    REQUIRE(!r.tets.empty());
    REQUIRE(r.edge_provenance.size() == 1);
    CHECK(tiles_segment(r, r.edge_provenance[0], a, b));
}

// A single tet with its apex at the origin and the other corners around the (1,1,1) direction,
// (4,4,6), (4,6,4), (6,4,4): the inward cone at the apex holds none of +-x, +-y, +-z, so a
// feature at the apex can take no axis-aligned forcing corner. A point is strictly inside the
// tet iff 5x > 2(y+z), 5y > 2(x+z), 5z > 2(x+y) and x+y+z < 14, so the cube [1.3,1.5]^3 is
// (a cube [lo,hi]^3 needs lo > 0.8 hi).
const double rotated_tet[12] = {0, 0, 0, 4, 4, 6, 4, 6, 4, 6, 4, 4};
Background rotated_tet_background()
{
    return Background{
        std::vector<double>(rotated_tet, rotated_tet + 12), std::vector<uint32_t>{0, 1, 2, 3}};
}

} // namespace

TEST_CASE("a forced point on a surface vertex that the background also has", "[feature_vertex]")
{
    const double p[3] = {0, 0, 0};
    const Result r =
        embed(grid_background(), cube_coords(), {}, {}, std::vector<double>(p, p + 3));
    check_point(r, p);
    CHECK(vertices_outside_box(r, -0.2, 2.2) == 0);
}

TEST_CASE(
    "a forced edge starting at a surface vertex that the background also has",
    "[feature_vertex]")
{
    const std::vector<double> e = {0, 0, 0, 0.5, 0.5, 0.5};
    const Result r = embed(grid_background(), cube_coords(), e, {0, 1}, {});
    check_edge(r, &e[0], &e[3]);
    CHECK(vertices_outside_box(r, -0.2, 2.2) == 0);
}

// The corner at (2,2,2) is on the background's maximum side, 0.2 inside it. The point's forcing
// triangles reach 0.228 along +x, +y and +z, and the edge's reach 0.287 along +y and +z, so
// both would leave the background mesh unless their corners point inward.
TEST_CASE("a forced point whose forcing corners would leave the background", "[feature_vertex]")
{
    const double p[3] = {2, 2, 2};
    const Result r =
        embed(grid_background(), cube_coords(), {}, {}, std::vector<double>(p, p + 3));
    check_point(r, p);
    CHECK(vertices_outside_box(r, -0.2, 2.2) == 0);
}

TEST_CASE("a forced edge whose forcing corners would leave the background", "[feature_vertex]")
{
    const std::vector<double> e = {2, 2, 2, 0, 0, 1.5};
    const Result r = embed(grid_background(), cube_coords(), e, {0, 1}, {});
    check_edge(r, &e[0], &e[3]);
    CHECK(vertices_outside_box(r, -0.2, 2.2) == 0);
}

// A background that is not a box: one tet with corners (-1,-1,-1), (9,-1,-1), (-1,9,-1),
// (-1,-1,9), whose slanted face is x + y + z = 7. The forced point (2,2,2.9) is inside (sum 6.9);
// its corners at +0.228 along x, y and z have sum 7.128, outside the tet but inside its bounding
// box, so only a test against the tets themselves sends them to -0.228 (sum 6.672). With corners
// left at +0.228 the embedding still finishes, but with vertices outside the background.
TEST_CASE(
    "a forced point near the slanted face of a background that is not a box",
    "[feature_vertex]")
{
    const double tet[12] = {-1, -1, -1, 9, -1, -1, -1, 9, -1, -1, -1, 9};
    const Background bg{std::vector<double>(tet, tet + 12), {0, 1, 2, 3}};
    const double p[3] = {2, 2, 2.9};
    const Result r = embed(bg, cube_coords(), {}, {}, std::vector<double>(p, p + 3));
    check_point(r, p);
    CHECK(vertices_outside_tet(r, tet) == 0);
}

// At the apex of the rotated tet no axis direction points into the mesh at any scale, so the
// forcing corners come from the tet itself. Before, this stopped with an error.
TEST_CASE("a forced point at a boundary vertex with no inward axis direction", "[feature_vertex]")
{
    const double p[3] = {0, 0, 0};
    const Result r = embed(
        rotated_tet_background(), cube_coords(1.3, 1.5), {}, {}, std::vector<double>(p, p + 3));
    check_point(r, p);
    CHECK(vertices_outside_tet(r, rotated_tet) == 0);
}

TEST_CASE("a forced edge from a boundary vertex with no inward axis direction", "[feature_vertex]")
{
    const std::vector<double> e = {0, 0, 0, 1, 1, 1};
    const Result r = embed(rotated_tet_background(), cube_coords(1.3, 1.5), e, {0, 1}, {});
    check_edge(r, &e[0], &e[3]);
    CHECK(vertices_outside_tet(r, rotated_tet) == 0);
}

// The background has a second vertex at (0,0,0) that no tet references (TetMesh::init_tets
// leaves such a vertex in no tet). A forced point there must map to the copy that is in the
// mesh, not to the last one at that position.
TEST_CASE(
    "a forced point at the position of an unreferenced background vertex",
    "[feature_vertex]")
{
    Background bg = grid_background();
    bg.coords.insert(bg.coords.end(), {0.0, 0.0, 0.0});
    const double p[3] = {0, 0, 0};
    const Result r = embed(bg, cube_coords(), {}, {}, std::vector<double>(p, p + 3));
    check_point(r, p);
    CHECK(vertices_outside_box(r, -0.2, 2.2) == 0);
}
