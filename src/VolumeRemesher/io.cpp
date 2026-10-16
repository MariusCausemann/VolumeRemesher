#include "io.h"

#include <numerics.h> // ip_error

#include <cctype>
#include <cstdio>
#include <cstring>

using NFG::ip_error;

namespace vol_rem {

namespace {

// Skip whitespace and '#'-comment lines (standard OFF allows them, e.g. files
// exported by meshio) so the parser lands on the next real token.
void off_skip_comments(FILE* file)
{
    int c;
    while ((c = fgetc(file)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(file)) != EOF && c != '\n')
                ;
            continue;
        }
        ungetc(c, file);
        return;
    }
}

// Closes the file on every exit path, including an ip_error that throws.
struct FileCloser
{
    FILE* f;
    ~FileCloser()
    {
        if (f) fclose(f);
    }
};

} // namespace

TriMeshData read_OFF(const char* filename, bool verbose)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL)
        ip_error("read_OFF_file: FATAL ERROR "
                 "cannot open input file.\n");
    FileCloser closer{file};

    // Check OFF mark (1st line).
    char file_ext_read[3];
    char file_ext_target[] = {'O', 'F', 'F'};
    if (fscanf(file, "%3c", file_ext_read) == 0)
        ip_error("read_OFF_file: FATAL ERROR "
                 "cannot read 1st line of input file\n");

    for (uint32_t i = 0; i < 3; i++)
        if (file_ext_read[i] != file_ext_target[i])
            ip_error("read_OFF_file: FATAL ERROR "
                     "1st line of input file is different from OFF\n");

    // Reading number of points and triangles (skipping any comment lines).
    off_skip_comments(file);
    uint32_t npts = 0, ntri = 0;
    if (fscanf(file, " %d %d %*d ", &npts, &ntri) == 0)
        ip_error("read_OFF_file: FATAL ERROR 2st line of "
                 "input file do not contanins point and triangles numbers.\n");

    if (verbose)
        printf("file %s contains %d vertices and %d constraints (triangles)\n", filename, npts, ntri);

    TriMeshData m;
    m.coords.resize(size_t(3) * npts);
    m.tris.resize(size_t(3) * ntri);

    for (uint32_t i = 0; i < npts; i++) {
        if (fscanf(
                file, " %lf %lf %lf ", &m.coords[i * 3], &m.coords[i * 3 + 1],
                &m.coords[i * 3 + 2]) == 0)
            ip_error("error reading input file\n");
    }

    uint32_t nv;
    for (uint32_t i = 0; i < ntri; i++) {
        if (fscanf(
                file, " %u %u %u %u ", &nv, &m.tris[i * 3], &m.tris[i * 3 + 1],
                &m.tris[i * 3 + 2]) == 0)
            ip_error("error reading input file\n");
        if (nv != 3) ip_error("Non-triangular faces not supported\n");
    }
    return m;
}

EdgeData read_edge_OFF(const char* filename, bool verbose)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) ip_error("read_edge_OFF: FATAL ERROR cannot open edge file.\n");
    FileCloser closer{file};

    char ext[3];
    const char target[] = {'O', 'F', 'F'};
    if (fscanf(file, "%3c", ext) == 0) ip_error("read_edge_OFF: cannot read 1st line\n");
    for (uint32_t i = 0; i < 3; i++)
        if (ext[i] != target[i]) ip_error("read_edge_OFF: 1st line is not OFF\n");

    off_skip_comments(file);
    uint32_t npts = 0, nedges = 0, dummy = 0;
    if (fscanf(file, " %u %u %u ", &npts, &nedges, &dummy) < 2)
        ip_error("read_edge_OFF: cannot read vertex/edge counts\n");
    if (verbose) printf("edge file %s: %u vertices, %u edges\n", filename, npts, nedges);

    EdgeData e;
    e.coords.resize(size_t(3) * npts);
    e.edges.resize(size_t(2) * nedges);
    for (uint32_t i = 0; i < npts; i++)
        if (fscanf(
                file, " %lf %lf %lf ", &e.coords[3 * i], &e.coords[3 * i + 1],
                &e.coords[3 * i + 2]) != 3)
            ip_error("read_edge_OFF: error reading vertex\n");
    uint32_t nv;
    for (uint32_t i = 0; i < nedges; i++) {
        if (fscanf(file, " %u %u %u ", &nv, &e.edges[2 * i], &e.edges[2 * i + 1]) != 3)
            ip_error("read_edge_OFF: error reading edge\n");
        if (nv != 2) ip_error("read_edge_OFF: each edge record must list 2 vertices\n");
    }
    return e;
}

PointData read_points_OFF(const char* filename, bool verbose)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) ip_error("read_points_OFF: FATAL ERROR cannot open point file.\n");
    FileCloser closer{file};

    char ext[3];
    const char target[] = {'O', 'F', 'F'};
    if (fscanf(file, "%3c", ext) == 0) ip_error("read_points_OFF: cannot read 1st line\n");
    for (uint32_t i = 0; i < 3; i++)
        if (ext[i] != target[i]) ip_error("read_points_OFF: 1st line is not OFF\n");

    off_skip_comments(file);
    uint32_t npts = 0, nf = 0, dummy = 0;
    if (fscanf(file, " %u %u %u ", &npts, &nf, &dummy) < 1)
        ip_error("read_points_OFF: cannot read vertex count\n");
    if (verbose) printf("point file %s: %u points\n", filename, npts);

    PointData p;
    p.coords.resize(size_t(3) * npts);
    for (uint32_t i = 0; i < npts; i++)
        if (fscanf(
                file, " %lf %lf %lf ", &p.coords[3 * i], &p.coords[3 * i + 1],
                &p.coords[3 * i + 2]) != 3)
            ip_error("read_points_OFF: error reading point\n");
    return p;
}

TetMeshData read_TET(const char* filename, bool verbose)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) ip_error("read_TET_file: FATAL ERROR cannot open input file.\n");
    FileCloser closer{file};

    uint32_t nv, nt;
    if (!fscanf(file, "%d vertices\n", &nv))
        ip_error("read_TET_file: FATAL ERROR\ncannot read 1st line of input file\n");
    if (!fscanf(file, "%d tets\n", &nt))
        ip_error("read_TET_file: FATAL ERROR\ncannot read 2nd line of input file\n");

    if (verbose) printf("file %s contains %u vertices and %u tetrahedra\n", filename, nv, nt);

    TetMeshData m;
    m.coords.resize(size_t(3) * nv);
    for (uint32_t i = 0; i < nv; i++) {
        double* crd = &m.coords[i * 3];
        if (fscanf(file, "%lf %lf %lf\n", crd, crd + 1, crd + 2) != 3)
            ip_error("read_TET_file: FATAL ERROR\ncannot read vertex coordinates\n");
    }

    m.tets.resize(size_t(4) * nt);
    for (uint32_t i = 0; i < nt; i++) {
        uint32_t* nodes = &m.tets[i * 4];
        if (fscanf(file, "4 %u %u %u %u\n", nodes, nodes + 1, nodes + 2, nodes + 3) != 4)
            ip_error("read_TET_file: FATAL ERROR\ncannot read tet indexes\n");
    }
    return m;
}

TetMeshData read_MEDIT(const char* filename, bool verbose)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) ip_error("read_MEDIT_file: FATAL ERROR cannot open input file.\n");
    FileCloser closer{file};

    char line[1024];
    do {
        if (!fscanf(file, "%1023s\n", line))
            ip_error("read_MEDIT_file: Could not find Vertices keyword\n");
    } while (strcmp(line, "Vertices"));

    int dummy, nv = 0;
    if (!fscanf(file, "%d\n", &nv)) ip_error("read_MEDIT_file: Could not read num vertices\n");
    if (verbose) printf("Reading %d vertices\n", nv);

    TetMeshData m;
    m.coords.resize(size_t(3) * nv);
    for (int i = 0; i < nv; i++) {
        double* crd = &m.coords[size_t(i) * 3];
        if (fscanf(file, "%lf %lf %lf %d\n", crd, crd + 1, crd + 2, &dummy) != 4)
            ip_error("read_MEDIT_file: Could not read vertex coords\n");
    }

    int nt;
    if (!fscanf(file, "%1023s\n", line) || strcmp(line, "Tetrahedra"))
        ip_error("read_MEDIT_file: Could not find Tetrahedra keyword\n");
    if (!fscanf(file, "%d\n", &nt)) ip_error("read_MEDIT_file: Could not read num tet\n");
    if (verbose) printf("Reading %d tetrahedra\n", nt);

    m.tets.resize(size_t(4) * nt);
    for (uint32_t i = 0; i < (uint32_t)nt; i++) {
        uint32_t* nodes = &m.tets[i * 4];
        if (fscanf(file, "%d %d %d %d %d\n", nodes, nodes + 1, nodes + 2, nodes + 3, &dummy) != 5)
            ip_error("read_MEDIT_file: Could not read tet indexes\n");
    }
    for (uint32_t i = 0; i < (uint32_t)nt * 4; i++) m.tets[i]--; // Decrease all indexes by one

    if (verbose) printf("file %s contains %u vertices and %u tetrahedra\n", filename, nv, nt);
    return m;
}

} // namespace vol_rem
