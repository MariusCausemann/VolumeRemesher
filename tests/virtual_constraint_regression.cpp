// Regression: place_virtual_constraints skipped the last half-edge in sorted order.
//
// Every boundary edge of the input (an edge with one incident constraint, as along the rim of
// an open surface) needs a virtual constraint: a triangle through the edge and a vertex off the
// constraint plane, so that the cells along the edge are split there and no facet on the
// constraint plane crosses it. The loop that decides which edges get one ran while
// he < num_half_edges - 1, so the half-edge that sorts last -- the edge whose (min vertex,
// max vertex) pair is the largest -- was never looked at. When that edge was a boundary edge and
// not already an edge of the tet mesh, the cells along it were not split, and the facets on the
// constraint plane straddled it: on the input (BLACK_A) where their barycentre fell inside the
// triangle, not on it (WHITE, a hole) where it fell outside. 0a1d5ca and f12f9b4 had fixed the
// bound; the re-import in 29390c9 brought it back.
//
// The input is one triangle, (0,0,.5) (1,0,.5) (0,1,.5), embedded in two small tet meshes built
// like wildmeshing-toolkit's tetwild background: the triangle's vertices, the corners of its
// bounding box padded by 0.1, and one or two more points placed so that the hypotenuse
// (1,0,.5)-(0,1,.5) is not an edge of the tet mesh. The triangle's vertices are tet-mesh
// vertices 0, 1, 2, so its half-edges sort as (0,1), (0,2), (1,2), and the one skipped was the
// hypotenuse. Measured before the fix (2026-10-20):
//   straddle_mesh: 2 virtual constraints for 3 boundary edges; both on-input facets have a
//                  vertex outside the triangle; on-input area = 1.2 x the triangle's.
//   hole_mesh:     2 virtual constraints; no on-input vertex outside, but on-input area =
//                  0.92 x the triangle's.
// After it: 3 virtual constraints, no vertex outside, and the on-input area equals the
// triangle's exactly, in both.

#include <catch2/catch_test_macros.hpp>

#include <VolumeRemesher/conforming_mesh.h>
#include <VolumeRemesher/embed.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

namespace {

using vol_rem::bigrational;

struct TetMeshData
{
    std::vector<double> coords;
    std::vector<uint32_t> tets;
};

// The triangle's vertices, the 8 corners of [-0.1, 1.1]^2 x [0.4, 0.6], and (0.6, 0.6, 0.5): on
// the triangle's plane, just past the hypotenuse, so that the tet-mesh edge from (0,0,.5) to it
// crosses the hypotenuse.
// clang-format off
const TetMeshData straddle_mesh = {
    {
         0.0,  0.0, 0.5,   1.0,  0.0, 0.5,   0.0,  1.0, 0.5,
        -0.1, -0.1, 0.4,   1.1, -0.1, 0.4,  -0.1,  1.1, 0.4,   1.1,  1.1, 0.4,
        -0.1, -0.1, 0.6,   1.1, -0.1, 0.6,  -0.1,  1.1, 0.6,   1.1,  1.1, 0.6,
         0.6,  0.6, 0.5,
    },
    {
        0, 1, 7, 3,   0, 1, 3, 11,  0, 1, 11, 7,  0, 2, 3, 7,   0, 2, 11, 3,  0, 2, 7, 11,
        1, 3, 4, 8,   1, 3, 11, 4,  1, 3, 8, 7,   1, 4, 6, 10,  1, 4, 11, 6,  1, 4, 10, 8,
        1, 6, 11, 10, 1, 7, 8, 11,  1, 8, 10, 11, 2, 3, 9, 5,   2, 3, 5, 11,  2, 3, 7, 9,
        2, 5, 10, 6,  2, 5, 6, 11,  2, 5, 9, 10,  2, 6, 10, 11, 2, 7, 11, 9,  2, 9, 11, 10,
        3, 4, 6, 11,  3, 5, 11, 6,  7, 8, 11, 10, 7, 9, 10, 11,
    },
};

// The same box with (0.62, 0.62, 0.55) above the plane and (0.3, 0.3, 0.45) below it instead.
const TetMeshData hole_mesh = {
    {
         0.0,  0.0, 0.5,   1.0,  0.0, 0.5,   0.0,  1.0, 0.5,
        -0.1, -0.1, 0.4,   1.1, -0.1, 0.4,  -0.1,  1.1, 0.4,   1.1,  1.1, 0.4,
        -0.1, -0.1, 0.6,   1.1, -0.1, 0.6,  -0.1,  1.1, 0.6,   1.1,  1.1, 0.6,
         0.62, 0.62, 0.55,  0.3,  0.3, 0.45,
    },
    {
        0, 1, 7, 3,   0, 1, 3, 12,  0, 1, 12, 7,  0, 2, 3, 7,   0, 2, 12, 3,  0, 2, 7, 12,
        1, 3, 4, 8,   1, 3, 12, 4,  1, 3, 8, 7,   1, 4, 6, 10,  1, 4, 11, 6,  1, 4, 10, 8,
        1, 4, 12, 11, 1, 6, 11, 10, 1, 7, 8, 12,  1, 8, 10, 11, 1, 8, 11, 12, 2, 3, 9, 5,
        2, 3, 5, 12,  2, 3, 7, 9,   2, 5, 10, 6,  2, 5, 6, 11,  2, 5, 9, 10,  2, 5, 11, 12,
        2, 6, 10, 11, 2, 7, 12, 9,  2, 9, 11, 10, 2, 9, 12, 11, 3, 4, 6, 12,  3, 5, 12, 6,
        4, 6, 12, 11, 5, 6, 11, 12, 7, 8, 11, 10, 7, 8, 12, 11, 7, 9, 10, 11, 7, 9, 11, 12,
    },
};
// clang-format on

const std::vector<double> triangle_coords = {0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5};
const std::vector<uint32_t> triangle_indexes = {0, 1, 2};

// No tet of m has both a and b as corners.
bool not_an_edge(const TetMeshData& m, uint32_t a, uint32_t b)
{
    for (size_t t = 0; t < m.tets.size(); t += 4) {
        const auto first = m.tets.begin() + t, last = first + 4;
        if (std::find(first, last, a) != last && std::find(first, last, b) != last) return false;
    }
    return true;
}

// An edge as (smaller vertex, larger vertex); an array so that Catch2 can print a failure.
using Edge = std::array<uint32_t, 2>;

Edge edge(uint32_t a, uint32_t b)
{
    return {std::min(a, b), std::max(a, b)};
}

// Runs place_virtual_constraints on the triangles 'tris' (vertex ids of m) the way
// remakePolyhedralMesh does, and returns the edges its virtual constraints were placed on.
std::multiset<Edge> virtual_constraint_edges(
    const TetMeshData& m,
    const std::vector<uint32_t>& tris)
{
    vol_rem::TetMesh mesh;
    mesh.init_vertices(m.coords.data(), uint32_t(m.coords.size() / 3));
    mesh.init_tets(m.tets.data(), m.tets.size() / 4);

    // constraints_t frees its arrays with free().
    vol_rem::constraints_t constraints;
    constraints.num_triangles = uint32_t(tris.size() / 3);
    constraints.tri_vertices = (uint32_t*)malloc(tris.size() * sizeof(uint32_t));
    std::copy(tris.begin(), tris.end(), constraints.tri_vertices);
    constraints.constr_group = (uint32_t*)calloc(constraints.num_triangles, sizeof(uint32_t));

    const uint32_t num_half_edges = 3 * constraints.num_triangles;
    vol_rem::half_edge_t* half_edges =
        (vol_rem::half_edge_t*)calloc(num_half_edges, sizeof(vol_rem::half_edge_t));
    vol_rem::fill_half_edges(&constraints, half_edges);
    vol_rem::sort_half_edges(half_edges, num_half_edges);
    const uint32_t nvc = vol_rem::place_virtual_constraints(&mesh, &constraints, half_edges);
    free(half_edges);

    REQUIRE(constraints.num_virtual_triangles == nvc);
    REQUIRE(constraints.num_triangles == tris.size() / 3 + nvc);
    // A virtual constraint is (edge endpoint, edge endpoint, apex).
    std::multiset<Edge> edges;
    for (uint32_t t = constraints.num_triangles - nvc; t < constraints.num_triangles; t++)
        edges.insert(edge(constraints.tri_vertices[3 * t], constraints.tri_vertices[3 * t + 1]));
    return edges;
}

struct EmbedOut
{
    std::vector<bigrational> vertices;
    std::vector<uint32_t> facets, cells, final_tets_parent, facets_on_input;
    std::vector<std::array<uint32_t, 4>> tets;
    std::vector<bool> cells_with_faces_on_input;
    std::vector<std::vector<uint32_t>> final_tets_parent_faces;
    std::vector<std::vector<std::array<uint32_t, 4>>> tri_provenance;
    std::vector<uint32_t> tri_group;
    std::vector<std::vector<std::array<uint32_t, 3>>> edge_provenance;
    std::vector<std::array<uint32_t, 2>> point_provenance;
};

EmbedOut embed(const TetMeshData& m)
{
    EmbedOut o;
    const std::vector<double> no_edge_coords, no_point_coords;
    const std::vector<uint32_t> no_edge_indexes;
    vol_rem::embed_tri_in_poly_mesh(
        triangle_coords,
        triangle_indexes,
        m.coords,
        m.tets,
        o.vertices,
        o.facets,
        o.cells,
        o.tets,
        o.final_tets_parent,
        o.facets_on_input,
        o.cells_with_faces_on_input,
        o.final_tets_parent_faces,
        no_edge_coords,
        no_edge_indexes,
        no_point_coords,
        o.tri_provenance,
        o.tri_group,
        o.edge_provenance,
        o.point_provenance,
        /*verbose=*/false);
    return o;
}

// The facets on the input are exactly the triangle: none has a vertex outside the closed
// triangle (x >= 0, y >= 0, x + y <= 1, z = 1/2), and their areas add up to the triangle's, 1/2.
void check_on_input_facets_tile_the_triangle(const TetMeshData& m)
{
    const EmbedOut o = embed(m);
    REQUIRE(!o.facets_on_input.empty());

    std::vector<size_t> facet_start;
    for (size_t i = 0; i < o.facets.size(); i += o.facets[i] + 1) facet_start.push_back(i);

    const bigrational zero(0.0), half(0.5), one(1.0);
    size_t facets_outside = 0;
    bigrational twice_area(0.0);
    for (const uint32_t f : o.facets_on_input) {
        REQUIRE(f < facet_start.size());
        const uint32_t* poly = &o.facets[facet_start[f] + 1];
        const uint32_t n = o.facets[facet_start[f]];
        bool outside = false;
        bigrational shoelace(0.0);
        for (uint32_t j = 0; j < n; j++) {
            const bigrational* p = &o.vertices[3 * size_t(poly[j])];
            const bigrational* q = &o.vertices[3 * size_t(poly[(j + 1) % n])];
            const bigrational s = one - p[0] - p[1];
            if (sgn(p[0]) < 0 || sgn(p[1]) < 0 || sgn(s) < 0 || !(p[2] == half)) outside = true;
            shoelace = shoelace + p[0] * q[1] - p[1] * q[0];
        }
        if (outside) facets_outside++;
        if (sgn(shoelace) < 0) shoelace = zero - shoelace;
        twice_area = twice_area + shoelace;
    }
    INFO(o.facets_on_input.size() << " on-input facets, twice their area ~ " << twice_area.get_d());
    CHECK(facets_outside == 0);
    const bool area_is_the_triangles = twice_area == one;
    CHECK(area_is_the_triangles);
}

} // namespace

TEST_CASE("every boundary edge gets a virtual constraint", "[virtual_constraint]")
{
    // The precondition for the old bound to bite: the hypotenuse, the last half-edge in sorted
    // order, is not an edge of the tet mesh.
    REQUIRE(not_an_edge(straddle_mesh, 1, 2));

    SECTION("one triangle: its three edges")
    {
        const std::multiset<Edge> expected = {edge(0, 1), edge(0, 2), edge(1, 2)};
        CHECK(virtual_constraint_edges(straddle_mesh, {0, 1, 2}) == expected);
    }
    SECTION("two coplanar triangles: the four rim edges, not the shared diagonal")
    {
        // The quad (0,0) (1,0) (0.6,0.6) (0,1) on z = 0.5, split along 0-11. Its half-edges
        // sort as (0,1), (0,2), (0,11) x2, (1,11), (2,11): the last one is a rim edge.
        const std::multiset<Edge> expected = {edge(0, 1), edge(1, 11), edge(2, 11), edge(0, 2)};
        CHECK(virtual_constraint_edges(straddle_mesh, {0, 1, 11, 0, 11, 2}) == expected);
    }
}

TEST_CASE("no on-input facet crosses the triangle's last boundary edge", "[virtual_constraint]")
{
    REQUIRE(not_an_edge(straddle_mesh, 1, 2));
    check_on_input_facets_tile_the_triangle(straddle_mesh);
}

TEST_CASE("no hole in the input along the triangle's last boundary edge", "[virtual_constraint]")
{
    REQUIRE(not_an_edge(hole_mesh, 1, 2));
    check_on_input_facets_tile_the_triangle(hole_mesh);
}
