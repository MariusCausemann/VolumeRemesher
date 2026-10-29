#include "winding_number.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vol_rem {

// ---------------------------------------------------------------------------------------------
// det_atan / det_atan2: ports of fdlibm's s_atan.c and e_atan2.c.
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use, copy, modify, and
// distribute this software is freely granted, provided that this notice is preserved.
//
// Only the "high word / low word" access is rewritten (memcpy instead of a type pun); the
// arithmetic is unchanged, so the results are the ones fdlibm gives on every platform.
// ---------------------------------------------------------------------------------------------
namespace {

inline uint64_t bits_of(double x)
{
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

inline int32_t high_word(double x)
{
    return (int32_t)(bits_of(x) >> 32);
}
inline uint32_t low_word(double x)
{
    return (uint32_t)(bits_of(x) & 0xffffffffu);
}

const double atanhi[] = {
    4.63647609000806093515e-01, // atan(0.5)hi 0x3FDDAC67, 0x0561BB4F
    7.85398163397448278999e-01, // atan(1.0)hi 0x3FE921FB, 0x54442D18
    9.82793723247329054082e-01, // atan(1.5)hi 0x3FEF730B, 0xD281F69B
    1.57079632679489655800e+00, // atan(inf)hi 0x3FF921FB, 0x54442D18
};

const double atanlo[] = {
    2.26987774529616870924e-17, // atan(0.5)lo 0x3C7A2B7F, 0x222F65E2
    3.06161699786838301793e-17, // atan(1.0)lo 0x3C81A626, 0x33145C07
    1.39033110312309984516e-17, // atan(1.5)lo 0x3C700788, 0x7AF0CBBD
    6.12323399573676603587e-17, // atan(inf)lo 0x3C91A626, 0x33145C07
};

const double aT[] = {
    3.33333333333329318027e-01, // 0x3FD55555, 0x5555550D
    -1.99999999998764832476e-01, // 0xBFC99999, 0x9998EBC4
    1.42857142725034663711e-01, // 0x3FC24924, 0x920083FF
    -1.11111104054623557880e-01, // 0xBFBC71C6, 0xFE231671
    9.09088713343650656196e-02, // 0x3FB745CD, 0xC54C206E
    -7.69187620504482999495e-02, // 0xBFB3B0F2, 0xAF749A6D
    6.66107313738753120669e-02, // 0x3FB10D66, 0xA0D03D51
    -5.83357013379057348645e-02, // 0xBFADDE2D, 0x52DEFD9A
    4.97687799461593236017e-02, // 0x3FA97B4B, 0x24760DEB
    -3.65315727442169155270e-02, // 0xBFA2B444, 0x2C6A6C2F
    1.62858201153657823623e-02, // 0x3F90AD3A, 0xE322DA11
};

const double one = 1.0, huge_val = 1.0e300;

} // namespace

double det_atan(double x)
{
    double w, s1, s2, z;
    int32_t ix, hx, id;

    hx = high_word(x);
    ix = hx & 0x7fffffff;
    if (ix >= 0x44100000) { // if |x| >= 2^66
        if (ix > 0x7ff00000 || (ix == 0x7ff00000 && (low_word(x) != 0))) return x + x; // NaN
        if (hx > 0)
            return atanhi[3] + atanlo[3];
        else
            return -atanhi[3] - atanlo[3];
    }
    if (ix < 0x3fdc0000) { // |x| < 0.4375
        if (ix < 0x3e400000) { // |x| < 2^-27
            if (huge_val + x > one) return x; // raise inexact
        }
        id = -1;
    } else {
        x = std::fabs(x);
        if (ix < 0x3ff30000) { // |x| < 1.1875
            if (ix < 0x3fe60000) { // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - one) / (2.0 + x);
            } else { // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - one) / (x + one);
            }
        } else {
            if (ix < 0x40038000) { // |x| < 2.4375
                id = 2;
                x = (x - 1.5) / (one + 1.5 * x);
            } else { // 2.4375 <= |x| < 2^66
                id = 3;
                x = -1.0 / x;
            }
        }
    }
    // end of argument reduction
    z = x * x;
    w = z * z;
    // break sum from i=0 to 10 aT[i]z**(i+1) into odd and even poly
    s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
    s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
    if (id < 0)
        return x - x * (s1 + s2);
    else {
        z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
        return (hx < 0) ? -z : z;
    }
}

namespace {
const double tiny = 1.0e-300;
const double zero = 0.0;
const double pi_o_4 = 7.8539816339744827900e-01; // 0x3FE921FB, 0x54442D18
const double pi_o_2 = 1.5707963267948965580e+00; // 0x3FF921FB, 0x54442D18
const double pi = 3.1415926535897931160e+00; // 0x400921FB, 0x54442D18
const double pi_lo = 1.2246467991473531772e-16; // 0x3CA1A626, 0x33145C07
} // namespace

double det_atan2(double y, double x)
{
    double z;
    int32_t k, m, hx, hy, ix, iy;
    uint32_t lx, ly;

    hx = high_word(x);
    ix = hx & 0x7fffffff;
    lx = low_word(x);
    hy = high_word(y);
    iy = hy & 0x7fffffff;
    ly = low_word(y);
    if (((uint32_t)ix | ((lx | -lx) >> 31)) > 0x7ff00000 ||
        ((uint32_t)iy | ((ly | -ly) >> 31)) > 0x7ff00000) // x or y is NaN
        return x + y;
    if ((hx - 0x3ff00000 | (int32_t)lx) == 0) return det_atan(y); // x=1.0
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2); // 2*sign(x)+sign(y)

    // when y = 0
    if ((iy | (int32_t)ly) == 0) {
        switch (m) {
        case 0:
        case 1: return y; // atan(+-0,+anything)=+-0
        case 2: return pi + tiny; // atan(+0,-anything) = pi
        case 3: return -pi - tiny; // atan(-0,-anything) =-pi
        }
    }
    // when x = 0
    if ((ix | (int32_t)lx) == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;

    // when x is INF
    if (ix == 0x7ff00000) {
        if (iy == 0x7ff00000) {
            switch (m) {
            case 0: return pi_o_4 + tiny; // atan(+INF,+INF)
            case 1: return -pi_o_4 - tiny; // atan(-INF,+INF)
            case 2: return 3.0 * pi_o_4 + tiny; // atan(+INF,-INF)
            case 3: return -3.0 * pi_o_4 - tiny; // atan(-INF,-INF)
            }
        } else {
            switch (m) {
            case 0: return zero; // atan(+...,+INF)
            case 1: return -zero; // atan(-...,+INF)
            case 2: return pi + tiny; // atan(+...,-INF)
            case 3: return -pi - tiny; // atan(-...,-INF)
            }
        }
    }
    // when y is INF
    if (iy == 0x7ff00000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;

    // compute y/x
    k = (iy - ix) >> 20;
    if (k > 60)
        z = pi_o_2 + 0.5 * pi_lo; // |y/x| > 2**60
    else if (hx < 0 && k < -60)
        z = 0.0; // |y|/x < -2**60
    else
        z = det_atan(std::fabs(y / x)); // safe to do y/x
    switch (m) {
    case 0: return z; // atan(+,+)
    case 1: return -z; // atan(-,+)
    case 2: return pi - (z - pi_lo); // atan(+,-)
    default: // case 3
        return (z - pi_lo) - pi; // atan(-,-)
    }
}

// ---------------------------------------------------------------------------------------------
// Solid angles and the fast winding number
// ---------------------------------------------------------------------------------------------
namespace {

const double four_pi = 12.566370614359172953850573533118;

inline double dot(const double* a, const double* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void cross(const double* a, const double* b, double* r)
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}

inline void sub(const double* a, const double* b, double* r)
{
    r[0] = a[0] - b[0];
    r[1] = a[1] - b[1];
    r[2] = a[2] - b[2];
}

} // namespace

double triangle_winding_number(const double* q, const double* a, const double* b, const double* c)
{
    double A[3], B[3], C[3], BxC[3];
    sub(a, q, A);
    sub(b, q, B);
    sub(c, q, C);
    const double la = std::sqrt(dot(A, A)), lb = std::sqrt(dot(B, B)), lc = std::sqrt(dot(C, C));
    cross(B, C, BxC);
    const double num = dot(A, BxC);
    const double den = la * lb * lc + dot(A, B) * lc + dot(A, C) * lb + dot(B, C) * la;
    return 2.0 * det_atan2(num, den) / four_pi;
}

FastWindingNumber::FastWindingNumber(
    const double* coords,
    const uint32_t* tris,
    size_t ntris,
    double beta,
    uint32_t leaf_size)
    : coords_(coords)
    , tris_(tris)
    , ntris_(ntris)
    , beta_(beta)
    , leaf_size_(std::max(1u, leaf_size))
{
    order_.resize(ntris);
    centroid_.resize(3 * ntris);
    for (size_t t = 0; t < ntris; t++) {
        order_[t] = (uint32_t)t;
        const double* a = coords_ + 3 * tris_[3 * t];
        const double* b = coords_ + 3 * tris_[3 * t + 1];
        const double* c = coords_ + 3 * tris_[3 * t + 2];
        for (int k = 0; k < 3; k++) centroid_[3 * t + k] = (a[k] + b[k] + c[k]) / 3.0;
    }
    if (ntris > 0) build(0, (uint32_t)ntris, 0);
}

uint32_t FastWindingNumber::build(uint32_t begin, uint32_t end, int depth)
{
    const uint32_t id = (uint32_t)nodes_.size();
    nodes_.emplace_back();
    {
        Node& n = nodes_.back();
        n.begin = begin;
        n.end = end;
        for (int k = 0; k < 8; k++) n.child[k] = UINT32_MAX;
    }

    // Bounding box of the centroids: the split point is its center.
    double lo[3], hi[3];
    for (int k = 0; k < 3; k++) {
        lo[k] = centroid_[3 * order_[begin] + k];
        hi[k] = lo[k];
    }
    for (uint32_t i = begin + 1; i < end; i++)
        for (int k = 0; k < 3; k++) {
            const double v = centroid_[3 * order_[i] + k];
            lo[k] = std::min(lo[k], v);
            hi[k] = std::max(hi[k], v);
        }
    const bool splittable = lo[0] < hi[0] || lo[1] < hi[1] || lo[2] < hi[2];

    if (end - begin > leaf_size_ && splittable && depth < 48) {
        double mid[3];
        for (int k = 0; k < 3; k++) mid[k] = 0.5 * (lo[k] + hi[k]);
        // Partition order_[begin,end) into the 8 octants, in octant order (stable, so the
        // result does not depend on the sort implementation).
        std::vector<uint32_t> buckets[8];
        for (uint32_t i = begin; i < end; i++) {
            const double* c = &centroid_[3 * order_[i]];
            const int oct =
                (c[0] >= mid[0] ? 1 : 0) | (c[1] >= mid[1] ? 2 : 0) | (c[2] >= mid[2] ? 4 : 0);
            buckets[oct].push_back(order_[i]);
        }
        uint32_t pos = begin;
        uint32_t ranges[8][2];
        for (int o = 0; o < 8; o++) {
            ranges[o][0] = pos;
            for (uint32_t t : buckets[o]) order_[pos++] = t;
            ranges[o][1] = pos;
        }
        for (int o = 0; o < 8; o++)
            if (ranges[o][1] > ranges[o][0]) {
                const uint32_t cid = build(ranges[o][0], ranges[o][1], depth + 1);
                nodes_[id].child[o] = cid; // nodes_ may have reallocated: index, not reference
            }
    }

    compute_moments(nodes_[id]);
    return id;
}

void FastWindingNumber::compute_moments(Node& node) const
{
    // Area-weighted normals and centroid. The expansion center is the area-weighted mean of
    // the triangle centroids (the plain mean when every triangle is degenerate).
    double total_area = 0.0, csum[3] = {0, 0, 0}, psum[3] = {0, 0, 0};
    for (uint32_t i = node.begin; i < node.end; i++) {
        const uint32_t t = order_[i];
        const double* a = coords_ + 3 * tris_[3 * t];
        const double* b = coords_ + 3 * tris_[3 * t + 1];
        const double* c = coords_ + 3 * tris_[3 * t + 2];
        double e1[3], e2[3], n[3];
        sub(b, a, e1);
        sub(c, a, e2);
        cross(e1, e2, n); // 2 * area * unit normal
        const double area = 0.5 * std::sqrt(dot(n, n));
        total_area += area;
        for (int k = 0; k < 3; k++) {
            csum[k] += area * centroid_[3 * t + k];
            psum[k] += centroid_[3 * t + k];
        }
    }
    const double count = (double)(node.end - node.begin);
    for (int k = 0; k < 3; k++)
        node.center[k] = total_area > 0.0 ? csum[k] / total_area : psum[k] / count;

    node.radius = 0.0;
    for (int k = 0; k < 3; k++) node.N[k] = 0.0;
    for (int k = 0; k < 9; k++) node.M[k] = 0.0;
    for (int k = 0; k < 27; k++) node.T[k] = 0.0;
    for (uint32_t i = node.begin; i < node.end; i++) {
        const uint32_t t = order_[i];
        const double* v[3] = {
            coords_ + 3 * tris_[3 * t],
            coords_ + 3 * tris_[3 * t + 1],
            coords_ + 3 * tris_[3 * t + 2]};
        double e1[3], e2[3], an[3]; // an = area-weighted normal
        sub(v[1], v[0], e1);
        sub(v[2], v[0], e2);
        cross(e1, e2, an);
        for (int k = 0; k < 3; k++) an[k] *= 0.5;
        double d[3];
        sub(&centroid_[3 * t], node.center, d);
        for (int i = 0; i < 3; i++) {
            node.N[i] += an[i];
            for (int j = 0; j < 3; j++) {
                node.M[3 * i + j] += d[i] * an[j];
                for (int k = 0; k < 3; k++) node.T[9 * i + 3 * j + k] += d[i] * d[j] * an[k];
            }
        }
        for (int j = 0; j < 3; j++) {
            double dv[3];
            sub(v[j], node.center, dv);
            node.radius = std::max(node.radius, std::sqrt(dot(dv, dv)));
        }
    }
}

double FastWindingNumber::evaluate(uint32_t node_id, const double* q) const
{
    const Node& node = nodes_[node_id];
    double r[3];
    sub(node.center, q, r);
    const double r2 = dot(r, r);
    const double dist = std::sqrt(r2);
    const bool is_leaf = node.child[0] == UINT32_MAX && node.child[1] == UINT32_MAX &&
                         node.child[2] == UINT32_MAX && node.child[3] == UINT32_MAX &&
                         node.child[4] == UINT32_MAX && node.child[5] == UINT32_MAX &&
                         node.child[6] == UINT32_MAX && node.child[7] == UINT32_MAX;

    if (dist > beta_ * node.radius && dist > 0.0) {
        // Far field: the Taylor expansion about p = center of f(p) = (p-q)/|p-q|^3, dotted with
        // the normals and integrated over the node's triangles, with r = center - q:
        //   f_i                = r_i / |r|^3
        //   d_j f_i            = (delta_ij |r|^2 - 3 r_i r_j) / |r|^5
        //   d_j d_k f_i        = (-3 (delta_ij r_k + delta_ik r_j + delta_jk r_i) |r|^2
        //                         + 15 r_i r_j r_k) / |r|^7
        //   w ~ (1/4pi) [ f.N + sum_ij d_j f_i M_ji + 1/2 sum_ijk d_j d_k f_i T_jki ]
        const double r3 = r2 * dist;
        const double r5 = r3 * r2;
        const double r7 = r5 * r2;
        double w = dot(r, node.N) / r3;
        double first = 0.0, second = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                const double dij = (i == j) ? r2 : 0.0;
                first += node.M[3 * j + i] * (dij - 3.0 * r[i] * r[j]);
                for (int k = 0; k < 3; k++) {
                    const double dik = (i == k) ? r2 : 0.0;
                    const double djk = (j == k) ? r2 : 0.0;
                    second +=
                        node.T[9 * j + 3 * k + i] *
                        (-3.0 * (dij * r[k] + dik * r[j] + djk * r[i]) + 15.0 * r[i] * r[j] * r[k]);
                }
            }
        w += first / r5 + 0.5 * second / r7;
        return w / four_pi;
    }

    if (is_leaf) {
        double w = 0.0;
        for (uint32_t i = node.begin; i < node.end; i++) {
            const uint32_t t = order_[i];
            w += triangle_winding_number(
                q,
                coords_ + 3 * tris_[3 * t],
                coords_ + 3 * tris_[3 * t + 1],
                coords_ + 3 * tris_[3 * t + 2]);
        }
        return w;
    }

    double w = 0.0;
    for (int o = 0; o < 8; o++)
        if (node.child[o] != UINT32_MAX) w += evaluate(node.child[o], q);
    return w;
}

double FastWindingNumber::operator()(const double* q) const
{
    if (nodes_.empty()) return 0.0;
    return evaluate(0, q);
}

double FastWindingNumber::exact(const double* q) const
{
    double w = 0.0;
    for (size_t t = 0; t < ntris_; t++)
        w += triangle_winding_number(
            q,
            coords_ + 3 * tris_[3 * t],
            coords_ + 3 * tris_[3 * t + 1],
            coords_ + 3 * tris_[3 * t + 2]);
    return w;
}

} // namespace vol_rem
