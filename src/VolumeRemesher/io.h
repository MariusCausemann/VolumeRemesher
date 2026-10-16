// Readers for the input formats mesh_generator accepts. These used to live in main.cpp; they
// are in the library so the Python module parses a file exactly as the command line tool does
// (same fscanf conversions), which keeps the two bit-identical from the first byte read.
//
// All of them call ip_error on a malformed or missing file.

#ifndef VOLUMEREMESHER_IO_H
#define VOLUMEREMESHER_IO_H

#include <cstdint>
#include <vector>

namespace vol_rem {

struct TriMeshData
{
    std::vector<double> coords; // 3 per vertex
    std::vector<uint32_t> tris; // 3 per triangle
    uint32_t num_vertices() const { return (uint32_t)(coords.size() / 3); }
    uint32_t num_triangles() const { return (uint32_t)(tris.size() / 3); }
};

struct EdgeData
{
    std::vector<double> coords; // 3 per vertex
    std::vector<uint32_t> edges; // 2 per edge
    uint32_t num_vertices() const { return (uint32_t)(coords.size() / 3); }
    uint32_t num_edges() const { return (uint32_t)(edges.size() / 2); }
};

struct PointData
{
    std::vector<double> coords; // 3 per point
    uint32_t num_points() const { return (uint32_t)(coords.size() / 3); }
};

struct TetMeshData
{
    std::vector<double> coords; // 3 per vertex
    std::vector<uint32_t> tets; // 4 per tet
    uint32_t num_vertices() const { return (uint32_t)(coords.size() / 3); }
    uint32_t num_tets() const { return (uint32_t)(tets.size() / 4); }
};

// OFF triangle mesh ('#' comment lines allowed after the OFF line, as meshio writes them).
TriMeshData read_OFF(const char* filename, bool verbose);

// OFF-style edge file: "OFF", then "nv ne 0", nv vertex lines "x y z", then ne edge lines
// "2 i j" (like an OFF face record but with two vertex indices).
EdgeData read_edge_OFF(const char* filename, bool verbose);

// OFF-style point file: "OFF", then "np 0 0", np vertex lines "x y z".
PointData read_points_OFF(const char* filename, bool verbose);

// The .tet format mesh_generator -t writes: "<n> vertices", "<m> tets", vertex lines, "4 a b c d".
TetMeshData read_TET(const char* filename, bool verbose);

// MEDIT .mesh/.msh: "Vertices" <n> then "x y z ref", "Tetrahedra" <m> then "a b c d ref"
// (1-based indices).
TetMeshData read_MEDIT(const char* filename, bool verbose);

} // namespace vol_rem

#endif // VOLUMEREMESHER_IO_H
