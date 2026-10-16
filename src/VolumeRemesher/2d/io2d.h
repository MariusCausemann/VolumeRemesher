// I/O for the 2D arrangement pipeline.
//
// Input is an OBJ line soup ("v x y z" + "l i j", 1-based, z ignored). Output mirrors the naming
// and the sidecar conventions of the 3D pipeline's saveMesh (BSP.cpp): an approximate OFF plus an
// exact ".rational" companion and provenance sidecars.

#ifndef VOLUMEREMESHER_2D_IO2D_H
#define VOLUMEREMESHER_2D_IO2D_H

#include "arrangement2d.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vol_rem {
namespace vr2d {

// Reads "v"/"l" records. Returns false if the file cannot be opened or is not a line soup.
// Coordinates are returned as (x,y) pairs; any third coordinate is dropped.
bool read_OBJ_segments(const std::string& path, std::vector<double>& coords,
                       std::vector<uint32_t>& indexes);

// In-memory form of what write_arrangement writes (the 2D analogue of mesh_output.h). Plain
// data only, so it can cross threads; exact coordinates are decimal "[-]num[/den]" strings.
struct ArrangementOutput
{
    std::vector<double> coords; // 2 per vertex (x, y)
    std::vector<std::array<uint32_t, 3>> tris; // counterclockwise
    std::vector<std::string> exact_coords; // 2 per vertex; empty unless requested
    // Per INPUT segment: the output triangle edges tiling it, {tri, v0, v1}, ordered from the
    // segment's own first endpoint; empty for a dropped (zero-length) segment.
    std::vector<std::vector<std::array<uint32_t, 3>>> segment_provenance;
    // Per INPUT point: {tri, vertex}, or {INVALID, INVALID} if it did not survive.
    std::vector<std::array<uint32_t, 2>> point_provenance;
    uint64_t num_vertices() const { return coords.size() / 2; }
};

// Only the finite triangles are emitted, and only the vertices they use; the virtual triangles
// (outside the convex hull) are dropped.
void extract_arrangement(const Arrangement2D& A, bool export_rational, ArrangementOutput& out);

// Writes the triangulation.
//
//   <base>            OFF, z = 0, approximate double coordinates
//   <base>.rational   exact rational coordinates, same vertex order (only if exact_coords is set)
//   <base>.segmentprov   per input segment: <id> <n> then n triples "<tri> <v0> <v1>"
//   <base>.pointprov     per input point:   <id> <tri> <vertex>
bool write_arrangement(const std::string& base, const ArrangementOutput& out);

// = extract_arrangement + write_arrangement.
bool write_arrangement(const std::string& base, const Arrangement2D& A, bool export_rational);

} // namespace vr2d
} // namespace vol_rem

#endif
