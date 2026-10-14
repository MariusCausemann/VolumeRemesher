// Error paths of embedding into a given tet mesh: every vertex of the surface and of the forced
// features must be strictly inside the mesh, and a forced edge must have distinct endpoints.
// Each case ends in ip_error, which prints its message and exits, so this is its own executable:
// tests/CMakeLists.txt registers every case by its name here together with the message it must
// print (PASS_REGULAR_EXPRESSION), and a case that runs to completion fails. Keep the names in
// sync with tests/CMakeLists.txt.

#include <catch2/catch_test_macros.hpp>

#include "feature_vertex_fixtures.h"

#include <vector>

using namespace feature_vertex_fixtures;

TEST_CASE("a forced point outside the tet mesh is an error", "[feature_vertex_errors]")
{
    const std::vector<double> p = {2.25, 2, 2}; // the grid ends at 2.2; the -x corner would fit
    embed(grid_background(), cube_coords(), {}, {}, p);
    FAIL("embedding a forced point outside the tet mesh should have stopped with an error");
}

TEST_CASE("a forced point on a boundary face of the tet mesh is an error", "[feature_vertex_errors]")
{
    const std::vector<double> p = {1, 1, 2.2};
    embed(grid_background(), cube_coords(), {}, {}, p);
    FAIL("embedding a forced point on the tet mesh's boundary should have stopped with an error");
}

TEST_CASE("a forced edge endpoint on a boundary edge of the tet mesh is an error", "[feature_vertex_errors]")
{
    const std::vector<double> e = {1, 2.2, 2.2, 1, 1, 1};
    embed(grid_background(), cube_coords(), e, {0, 1}, {});
    FAIL("embedding a forced edge from the tet mesh's boundary should have stopped with an error");
}

TEST_CASE("a forced edge with coincident endpoints is an error", "[feature_vertex_errors]")
{
    const std::vector<double> e = {1, 1, 1, 1, 1, 1};
    embed(grid_background(), cube_coords(), e, {0, 1}, {});
    FAIL("embedding a zero-length forced edge should have stopped with an error");
}

TEST_CASE("a surface vertex outside the tet mesh is an error", "[feature_vertex_errors]")
{
    embed(box_background(10.0, 11.0), cube_coords(), {}, {}, {});
    FAIL("embedding a surface outside the tet mesh should have stopped with an error");
}

TEST_CASE("a surface vertex on the boundary of the tet mesh is an error", "[feature_vertex_errors]")
{
    // The cube's corners are the box's corners: on its boundary, and coincident with its vertices.
    embed(box_background(0.0, 2.0), cube_coords(), {}, {}, {});
    FAIL("embedding a surface touching the tet mesh's boundary should have stopped with an error");
}
