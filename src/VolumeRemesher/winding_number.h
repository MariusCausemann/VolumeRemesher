// Generalized winding number (Jacobson, Kavan, Sorkine-Hornung 2013) of a triangle soup,
// evaluated with the hierarchical far-field expansion of Barill, Dickson, Schmidt, Levin and
// Jacobson 2018 ("Fast winding numbers for soups and clouds"). Used by the in/out cell
// classification as the fallback for inputs that are not closed and consistently oriented,
// where the exact combinatorial winding number is undefined (see inOutPartition.cpp).
//
// Determinism: the output of this library is required to be byte-identical across platforms.
// Every operation here is IEEE double +, -, *, /, sqrt (all correctly rounded) or det_atan2,
// a port of fdlibm's atan2 that is itself only IEEE arithmetic, so the same input gives the
// same bits everywhere, provided the compiler does not contract a*b+c into an FMA (the
// project builds with -ffp-contract=off / /fp:strict for this reason).

#ifndef VOLUMEREMESHER_WINDING_NUMBER_H
#define VOLUMEREMESHER_WINDING_NUMBER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vol_rem {

// atan2(y, x) and atan(x) as fdlibm computes them (correct to within 1 ulp), bit-identical
// on every platform, unlike the system libm.
double det_atan2(double y, double x);
double det_atan(double x);

// Signed solid angle subtended at q by the triangle (a, b, c), divided by 4*pi: the
// contribution of one triangle to the winding number. Van Oosterom & Strackee's formula.
double triangle_winding_number(const double* q, const double* a, const double* b, const double* c);

class FastWindingNumber
{
public:
    // coords: 3 doubles per vertex; tris: 3 vertex indices per triangle, ntris of them. The
    // arrays must outlive this object. beta is the far-field acceptance ratio (distance to the
    // node's expansion center over the node's radius): larger is more accurate and slower; 2 is
    // the published default. leaf_size bounds the number of triangles in a leaf.
    FastWindingNumber(
        const double* coords,
        const uint32_t* tris,
        size_t ntris,
        double beta = 2.0,
        uint32_t leaf_size = 16);

    // The generalized winding number at q (3 doubles); in the far field the Taylor expansion of
    // the integrand to second order (dipole, quadrupole and octupole terms, libigl's
    // expansion_order 2).
    double operator()(const double* q) const;

    // The exact sum over every triangle, O(ntris). For tests.
    double exact(const double* q) const;

    size_t num_triangles() const { return ntris_; }

private:
    struct Node
    {
        double center[3]; // area-weighted centroid of the triangles below: the expansion center
        double radius; // max distance from center to a vertex of those triangles
        double N[3]; // sum of area-weighted normals (dipole)
        double M[9]; // sum_t a_t (c_t - center)_i n_t,j at [3i+j] (first moment)
        double T[27]; // sum_t a_t (c_t - center)_i (c_t - center)_j n_t,k at [9i+3j+k] (second)
        uint32_t begin, end; // triangles of this node: order_[begin, end)
        uint32_t child[8]; // UINT32_MAX when absent; a leaf has none
    };

    uint32_t build(uint32_t begin, uint32_t end, int depth);
    void compute_moments(Node& node) const;
    double evaluate(uint32_t node_id, const double* q) const;

    const double* coords_;
    const uint32_t* tris_;
    size_t ntris_;
    double beta_;
    uint32_t leaf_size_;
    std::vector<uint32_t> order_; // triangle indices, permuted so each node owns a range
    std::vector<double> centroid_; // 3 per triangle, in input order
    std::vector<Node> nodes_;
};

} // namespace vol_rem

#endif // VOLUMEREMESHER_WINDING_NUMBER_H
