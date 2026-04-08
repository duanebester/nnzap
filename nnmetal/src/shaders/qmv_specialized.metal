// ============================================================================
// qmv_specialized.metal — Specialized QMV kernels with constexpr K
// ============================================================================
//
// These kernels are functionally identical to the generic qmv_const_*
// variants in compute.metal, but K and group_size are compile-time
// constants (constexpr).  Loop unrolling is explicitly disabled via
// pragma to keep register pressure low and preserve GPU occupancy
// (these kernels are memory-bound).  The compiler still eliminates
// dead tail code and propagates constants into scheduling decisions.
//
// The three macros SPEC_HIDDEN_K, SPEC_INTER_K, and SPEC_GS are
// prepended by the Zig build system before compilation.  This file
// must NOT define them.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------
// Guard macros — fail loudly if the build system forgot to define them.
// --------------------------------------------------------------------

#ifndef SPEC_HIDDEN_K
    #error "SPEC_HIDDEN_K must be defined by the build system."
#endif

#ifndef SPEC_INTER_K
    #error "SPEC_INTER_K must be defined by the build system."
#endif

#ifndef SPEC_GS
    #error "SPEC_GS must be defined by the build system."
#endif

// Kept for buffer interface compatibility with generic dispatch.
// Only M is used — K and group_size are constexpr in each kernel.
struct QMVDims {
    uint M;
    uint K;
    uint group_size;
};

// ====================================================================
// Kernel 1: qmv_spec_f16io — single-group, half in, half out
// ====================================================================
/// Specialized single-group QMV with K = SPEC_HIDDEN_K and
/// group_size = SPEC_GS.  Half-precision input (constant cache)
/// and half-precision output.  All loop bounds and byte offsets
/// are constexpr, enabling dead-code removal of the tail loop
/// when bytes_per_lane is divisible by 4 and constant
/// propagation into scheduling decisions.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void qmv_spec_f16io(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half*    scales      [[buffer(1)]],
    constant half*        input       [[buffer(2)]],
    device half*          output      [[buffer(3)]],
    constant QMVDims&     dims        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0,
        "K must be SIMD-aligned.");
    static_assert(K % group_size == 0,
        "K must be group-aligned.");
    static_assert(K / 32 <= group_size,
        "Single-group: each lane's columns "
        "must fit in one scale group.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    const uint col_byte_off = col_start / 8;
    const uint grp = col_start / group_size;
    constexpr uint bytes_per_row = K / 8;

    float sig0 = 0.0f;
    float sig1 = 0.0f;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint base0 =
        row0 * bytes_per_row + col_byte_off;
    const uint base1 =
        row1 * bytes_per_row + col_byte_off;

    constexpr uint words_per_lane = bytes_per_lane / 4;
    // Tail is dead code when bytes_per_lane % 4 == 0.
    constexpr uint tail_bytes = bytes_per_lane % 4;
    device const uint32_t* bits32_r0 =
        (device const uint32_t*)(packed_bits + base0);
    device const uint32_t* bits32_r1 =
        (device const uint32_t*)(packed_bits + base1);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t w0 =
            row0_valid ? bits32_r0[w] : 0;
        const uint32_t w1 =
            row1_valid ? bits32_r1[w] : 0;

        // Process 4 bytes from each uint32.
        for (uint bi = 0; bi < 4; bi++) {
            const uint ci =
                col_start + w * 32 + bi * 8;
            // Widen f16 to f32 before accumulation.
            const float x0 = float(input[ci]);
            const float x1 = float(input[ci + 1]);
            const float x2 = float(input[ci + 2]);
            const float x3 = float(input[ci + 3]);
            const float x4 = float(input[ci + 4]);
            const float x5 = float(input[ci + 5]);
            const float x6 = float(input[ci + 6]);
            const float x7 = float(input[ci + 7]);
            const uint bv0 =
                (w0 >> (bi * 8)) & 0xFFu;
            const uint bv1 =
                (w1 >> (bi * 8)) & 0xFFu;
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    // Tail loop for when bytes_per_lane % 4 != 0.
    // Dead code when constexpr tail_bytes == 0 (e.g. K=2048).
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++) {
        const uint ci = col_start + b * 8;
        // Widen f16 to f32 before accumulation.
        const float x0 = float(input[ci]);
        const float x1 = float(input[ci + 1]);
        const float x2 = float(input[ci + 2]);
        const float x3 = float(input[ci + 3]);
        const float x4 = float(input[ci + 4]);
        const float x5 = float(input[ci + 5]);
        const float x6 = float(input[ci + 6]);
        const float x7 = float(input[ci + 7]);
        if (row0_valid) {
            const uint bv0 =
                packed_bits[base0 + b];
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
        }
        if (row1_valid) {
            const uint bv1 =
                packed_bits[base1 + b];
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    const uint scale_off0 =
        row0 * groups_per_row + grp;
    const uint scale_off1 =
        row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc0 =
            float(scales[scale_off0]) * sig0;
        acc0 = simd_sum(acc0);
        if (lane == 0) output[row0] = half(acc0);
    }

    if (row1_valid) {
        float acc1 =
            float(scales[scale_off1]) * sig1;
        acc1 = simd_sum(acc1);
        if (lane == 0) output[row1] = half(acc1);
    }
}

// ====================================================================
// Kernel 2: qmv_spec_f16io_resadd — single-group, fused residual add
// ====================================================================
/// Identical to qmv_spec_f16io but accumulates the f32 result
/// directly into the residual buffer (residual[row] += acc)
/// instead of writing half to an intermediate buffer.  Eliminates
/// one dispatch and one barrier per call site by fusing the
/// residual_add_f16 step into the QMV output write.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void qmv_spec_f16io_resadd(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half*    scales      [[buffer(1)]],
    constant half*        input       [[buffer(2)]],
    device float*         residual    [[buffer(3)]],
    constant QMVDims&     dims        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0,
        "K must be SIMD-aligned.");
    static_assert(K % group_size == 0,
        "K must be group-aligned.");
    static_assert(K / 32 <= group_size,
        "Single-group: each lane's columns "
        "must fit in one scale group.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    const uint col_byte_off = col_start / 8;
    const uint grp = col_start / group_size;
    constexpr uint bytes_per_row = K / 8;

    float sig0 = 0.0f;
    float sig1 = 0.0f;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint base0 =
        row0 * bytes_per_row + col_byte_off;
    const uint base1 =
        row1 * bytes_per_row + col_byte_off;

    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint tail_bytes = bytes_per_lane % 4;
    device const uint32_t* bits32_r0 =
        (device const uint32_t*)(packed_bits + base0);
    device const uint32_t* bits32_r1 =
        (device const uint32_t*)(packed_bits + base1);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t w0 =
            row0_valid ? bits32_r0[w] : 0;
        const uint32_t w1 =
            row1_valid ? bits32_r1[w] : 0;

        // Process 4 bytes from each uint32.
        for (uint bi = 0; bi < 4; bi++) {
            const uint ci =
                col_start + w * 32 + bi * 8;
            // Widen f16 to f32 before accumulation.
            const float x0 = float(input[ci]);
            const float x1 = float(input[ci + 1]);
            const float x2 = float(input[ci + 2]);
            const float x3 = float(input[ci + 3]);
            const float x4 = float(input[ci + 4]);
            const float x5 = float(input[ci + 5]);
            const float x6 = float(input[ci + 6]);
            const float x7 = float(input[ci + 7]);
            const uint bv0 =
                (w0 >> (bi * 8)) & 0xFFu;
            const uint bv1 =
                (w1 >> (bi * 8)) & 0xFFu;
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    // Tail loop — dead code when constexpr tail_bytes == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++) {
        const uint ci = col_start + b * 8;
        const float x0 = float(input[ci]);
        const float x1 = float(input[ci + 1]);
        const float x2 = float(input[ci + 2]);
        const float x3 = float(input[ci + 3]);
        const float x4 = float(input[ci + 4]);
        const float x5 = float(input[ci + 5]);
        const float x6 = float(input[ci + 6]);
        const float x7 = float(input[ci + 7]);
        if (row0_valid) {
            const uint bv0 =
                packed_bits[base0 + b];
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
        }
        if (row1_valid) {
            const uint bv1 =
                packed_bits[base1 + b];
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    const uint scale_off0 =
        row0 * groups_per_row + grp;
    const uint scale_off1 =
        row1 * groups_per_row + grp;

    // Lane 0 accumulates the f32 result into the residual.
    if (row0_valid) {
        float acc0 =
            float(scales[scale_off0]) * sig0;
        acc0 = simd_sum(acc0);
        if (lane == 0) residual[row0] += acc0;
    }

    if (row1_valid) {
        float acc1 =
            float(scales[scale_off1]) * sig1;
        acc1 = simd_sum(acc1);
        if (lane == 0) residual[row1] += acc1;
    }
}

// ====================================================================
// Kernel 3: qmv_spec_fused_pair_f16io — two matrices, one input
// ====================================================================
/// Specialized fused-pair QMV with K = SPEC_HIDDEN_K.  Processes
/// two weight matrices (A and B) against a single shared input
/// vector in one dispatch, halving kernel launch overhead for
/// gate+up projection pairs.  Four accumulators (sig_a0, sig_b0,
/// sig_a1, sig_b1) track two rows from each matrix simultaneously.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void qmv_spec_fused_pair_f16io(
    device const uint8_t* packed_a  [[buffer(0)]],
    device const half*    scales_a  [[buffer(1)]],
    constant half*        input     [[buffer(2)]],
    device half*          output_a  [[buffer(3)]],
    device const uint8_t* packed_b  [[buffer(4)]],
    device const half*    scales_b  [[buffer(5)]],
    device half*          output_b  [[buffer(6)]],
    constant QMVDims&     dims      [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0,
        "K must be SIMD-aligned.");
    static_assert(K % group_size == 0,
        "K must be group-aligned.");
    static_assert(K / 32 <= group_size,
        "Single-group: each lane's columns "
        "must fit in one scale group.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    const uint col_byte_off = col_start / 8;
    const uint grp = col_start / group_size;
    constexpr uint bytes_per_row = K / 8;

    float sig_a0 = 0.0f, sig_b0 = 0.0f;
    float sig_a1 = 0.0f, sig_b1 = 0.0f;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint base_a0 =
        row0 * bytes_per_row + col_byte_off;
    const uint base_b0 =
        row0 * bytes_per_row + col_byte_off;
    const uint base_a1 =
        row1 * bytes_per_row + col_byte_off;
    const uint base_b1 =
        row1 * bytes_per_row + col_byte_off;

    // Load weights as uint32 (4 bytes per load).
    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint tail_bytes = bytes_per_lane % 4;
    device const uint32_t* w32_a0 =
        (device const uint32_t*)(packed_a + base_a0);
    device const uint32_t* w32_b0 =
        (device const uint32_t*)(packed_b + base_b0);
    device const uint32_t* w32_a1 =
        (device const uint32_t*)(packed_a + base_a1);
    device const uint32_t* w32_b1 =
        (device const uint32_t*)(packed_b + base_b1);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t wa0 =
            row0_valid ? w32_a0[w] : 0;
        const uint32_t wb0 =
            row0_valid ? w32_b0[w] : 0;
        const uint32_t wa1 =
            row1_valid ? w32_a1[w] : 0;
        const uint32_t wb1 =
            row1_valid ? w32_b1[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint ci =
                col_start + w * 32 + bi * 8;
            // Widen f16 to f32 before accumulation.
            const float x0 = float(input[ci]);
            const float x1 = float(input[ci + 1]);
            const float x2 = float(input[ci + 2]);
            const float x3 = float(input[ci + 3]);
            const float x4 = float(input[ci + 4]);
            const float x5 = float(input[ci + 5]);
            const float x6 = float(input[ci + 6]);
            const float x7 = float(input[ci + 7]);
            const uint sh = bi * 8;
            const uint ba0 = (wa0 >> sh) & 0xFFu;
            const uint bb0 = (wb0 >> sh) & 0xFFu;
            const uint ba1 = (wa1 >> sh) & 0xFFu;
            const uint bb1 = (wb1 >> sh) & 0xFFu;
            sig_a0 += select(-x0, x0, bool(ba0 & 1));
            sig_a0 += select(-x1, x1, bool(ba0 & 2));
            sig_a0 += select(-x2, x2, bool(ba0 & 4));
            sig_a0 += select(-x3, x3, bool(ba0 & 8));
            sig_a0 += select(-x4, x4, bool(ba0 & 16));
            sig_a0 += select(-x5, x5, bool(ba0 & 32));
            sig_a0 += select(-x6, x6, bool(ba0 & 64));
            sig_a0 += select(-x7, x7, bool(ba0 & 128));
            sig_b0 += select(-x0, x0, bool(bb0 & 1));
            sig_b0 += select(-x1, x1, bool(bb0 & 2));
            sig_b0 += select(-x2, x2, bool(bb0 & 4));
            sig_b0 += select(-x3, x3, bool(bb0 & 8));
            sig_b0 += select(-x4, x4, bool(bb0 & 16));
            sig_b0 += select(-x5, x5, bool(bb0 & 32));
            sig_b0 += select(-x6, x6, bool(bb0 & 64));
            sig_b0 += select(-x7, x7, bool(bb0 & 128));
            sig_a1 += select(-x0, x0, bool(ba1 & 1));
            sig_a1 += select(-x1, x1, bool(ba1 & 2));
            sig_a1 += select(-x2, x2, bool(ba1 & 4));
            sig_a1 += select(-x3, x3, bool(ba1 & 8));
            sig_a1 += select(-x4, x4, bool(ba1 & 16));
            sig_a1 += select(-x5, x5, bool(ba1 & 32));
            sig_a1 += select(-x6, x6, bool(ba1 & 64));
            sig_a1 += select(-x7, x7, bool(ba1 & 128));
            sig_b1 += select(-x0, x0, bool(bb1 & 1));
            sig_b1 += select(-x1, x1, bool(bb1 & 2));
            sig_b1 += select(-x2, x2, bool(bb1 & 4));
            sig_b1 += select(-x3, x3, bool(bb1 & 8));
            sig_b1 += select(-x4, x4, bool(bb1 & 16));
            sig_b1 += select(-x5, x5, bool(bb1 & 32));
            sig_b1 += select(-x6, x6, bool(bb1 & 64));
            sig_b1 += select(-x7, x7, bool(bb1 & 128));
        }
    }

    // Tail bytes — dead code when constexpr tail_bytes == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++) {
        const uint ci = col_start + b * 8;
        const float x0 = float(input[ci]);
        const float x1 = float(input[ci + 1]);
        const float x2 = float(input[ci + 2]);
        const float x3 = float(input[ci + 3]);
        const float x4 = float(input[ci + 4]);
        const float x5 = float(input[ci + 5]);
        const float x6 = float(input[ci + 6]);
        const float x7 = float(input[ci + 7]);
        if (row0_valid) {
            const uint bva0 =
                packed_a[base_a0 + b];
            sig_a0 += select(-x0, x0, bool(bva0 & 1));
            sig_a0 += select(-x1, x1, bool(bva0 & 2));
            sig_a0 += select(-x2, x2, bool(bva0 & 4));
            sig_a0 += select(-x3, x3, bool(bva0 & 8));
            sig_a0 += select(-x4, x4, bool(bva0 & 16));
            sig_a0 += select(-x5, x5, bool(bva0 & 32));
            sig_a0 += select(-x6, x6, bool(bva0 & 64));
            sig_a0 += select(-x7, x7, bool(bva0 & 128));
            const uint bvb0 =
                packed_b[base_b0 + b];
            sig_b0 += select(-x0, x0, bool(bvb0 & 1));
            sig_b0 += select(-x1, x1, bool(bvb0 & 2));
            sig_b0 += select(-x2, x2, bool(bvb0 & 4));
            sig_b0 += select(-x3, x3, bool(bvb0 & 8));
            sig_b0 += select(-x4, x4, bool(bvb0 & 16));
            sig_b0 += select(-x5, x5, bool(bvb0 & 32));
            sig_b0 += select(-x6, x6, bool(bvb0 & 64));
            sig_b0 += select(-x7, x7, bool(bvb0 & 128));
        }
        if (row1_valid) {
            const uint bva1 =
                packed_a[base_a1 + b];
            sig_a1 += select(-x0, x0, bool(bva1 & 1));
            sig_a1 += select(-x1, x1, bool(bva1 & 2));
            sig_a1 += select(-x2, x2, bool(bva1 & 4));
            sig_a1 += select(-x3, x3, bool(bva1 & 8));
            sig_a1 += select(-x4, x4, bool(bva1 & 16));
            sig_a1 += select(-x5, x5, bool(bva1 & 32));
            sig_a1 += select(-x6, x6, bool(bva1 & 64));
            sig_a1 += select(-x7, x7, bool(bva1 & 128));
            const uint bvb1 =
                packed_b[base_b1 + b];
            sig_b1 += select(-x0, x0, bool(bvb1 & 1));
            sig_b1 += select(-x1, x1, bool(bvb1 & 2));
            sig_b1 += select(-x2, x2, bool(bvb1 & 4));
            sig_b1 += select(-x3, x3, bool(bvb1 & 8));
            sig_b1 += select(-x4, x4, bool(bvb1 & 16));
            sig_b1 += select(-x5, x5, bool(bvb1 & 32));
            sig_b1 += select(-x6, x6, bool(bvb1 & 64));
            sig_b1 += select(-x7, x7, bool(bvb1 & 128));
        }
    }

    const uint sc_off0 = row0 * groups_per_row + grp;
    const uint sc_off1 = row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc_a0 =
            float(scales_a[sc_off0]) * sig_a0;
        float acc_b0 =
            float(scales_b[sc_off0]) * sig_b0;
        acc_a0 = simd_sum(acc_a0);
        acc_b0 = simd_sum(acc_b0);
        if (lane == 0) {
            output_a[row0] = half(acc_a0);
            output_b[row0] = half(acc_b0);
        }
    }

    if (row1_valid) {
        float acc_a1 =
            float(scales_a[sc_off1]) * sig_a1;
        float acc_b1 =
            float(scales_b[sc_off1]) * sig_b1;
        acc_a1 = simd_sum(acc_a1);
        acc_b1 = simd_sum(acc_b1);
        if (lane == 0) {
            output_a[row1] = half(acc_a1);
            output_b[row1] = half(acc_b1);
        }
    }
}

// ====================================================================
// Kernel 4: qmv_spec_f16in — single-group, half in, float out
// ====================================================================
/// Specialized single-group QMV with K = SPEC_HIDDEN_K.  Reads
/// the activation vector as half through the constant address
/// space and writes full f32 output.  Used for projections that
/// feed into f32 accumulation (e.g. attention logits).
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void qmv_spec_f16in(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half*    scales      [[buffer(1)]],
    constant half*        input       [[buffer(2)]],
    device float*         output      [[buffer(3)]],
    constant QMVDims&     dims        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0,
        "K must be SIMD-aligned.");
    static_assert(K % group_size == 0,
        "K must be group-aligned.");
    static_assert(K / 32 <= group_size,
        "Single-group: each lane's columns "
        "must fit in one scale group.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    const uint col_byte_off = col_start / 8;
    const uint grp = col_start / group_size;
    constexpr uint bytes_per_row = K / 8;

    float sig0 = 0.0f;
    float sig1 = 0.0f;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint base0 =
        row0 * bytes_per_row + col_byte_off;
    const uint base1 =
        row1 * bytes_per_row + col_byte_off;

    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint tail_bytes = bytes_per_lane % 4;
    device const uint32_t* bits32_r0 =
        (device const uint32_t*)(packed_bits + base0);
    device const uint32_t* bits32_r1 =
        (device const uint32_t*)(packed_bits + base1);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t w0 =
            row0_valid ? bits32_r0[w] : 0;
        const uint32_t w1 =
            row1_valid ? bits32_r1[w] : 0;

        // Process 4 bytes from each uint32.
        for (uint bi = 0; bi < 4; bi++) {
            const uint ci =
                col_start + w * 32 + bi * 8;
            // Widen f16 to f32 before accumulation.
            const float x0 = float(input[ci]);
            const float x1 = float(input[ci + 1]);
            const float x2 = float(input[ci + 2]);
            const float x3 = float(input[ci + 3]);
            const float x4 = float(input[ci + 4]);
            const float x5 = float(input[ci + 5]);
            const float x6 = float(input[ci + 6]);
            const float x7 = float(input[ci + 7]);
            const uint bv0 =
                (w0 >> (bi * 8)) & 0xFFu;
            const uint bv1 =
                (w1 >> (bi * 8)) & 0xFFu;
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    // Tail loop — dead code when constexpr tail_bytes == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++) {
        const uint ci = col_start + b * 8;
        const float x0 = float(input[ci]);
        const float x1 = float(input[ci + 1]);
        const float x2 = float(input[ci + 2]);
        const float x3 = float(input[ci + 3]);
        const float x4 = float(input[ci + 4]);
        const float x5 = float(input[ci + 5]);
        const float x6 = float(input[ci + 6]);
        const float x7 = float(input[ci + 7]);
        if (row0_valid) {
            const uint bv0 =
                packed_bits[base0 + b];
            sig0 += select(-x0, x0, bool(bv0 & 1));
            sig0 += select(-x1, x1, bool(bv0 & 2));
            sig0 += select(-x2, x2, bool(bv0 & 4));
            sig0 += select(-x3, x3, bool(bv0 & 8));
            sig0 += select(-x4, x4, bool(bv0 & 16));
            sig0 += select(-x5, x5, bool(bv0 & 32));
            sig0 += select(-x6, x6, bool(bv0 & 64));
            sig0 += select(-x7, x7, bool(bv0 & 128));
        }
        if (row1_valid) {
            const uint bv1 =
                packed_bits[base1 + b];
            sig1 += select(-x0, x0, bool(bv1 & 1));
            sig1 += select(-x1, x1, bool(bv1 & 2));
            sig1 += select(-x2, x2, bool(bv1 & 4));
            sig1 += select(-x3, x3, bool(bv1 & 8));
            sig1 += select(-x4, x4, bool(bv1 & 16));
            sig1 += select(-x5, x5, bool(bv1 & 32));
            sig1 += select(-x6, x6, bool(bv1 & 64));
            sig1 += select(-x7, x7, bool(bv1 & 128));
        }
    }

    const uint scale_off0 =
        row0 * groups_per_row + grp;
    const uint scale_off1 =
        row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc0 =
            float(scales[scale_off0]) * sig0;
        acc0 = simd_sum(acc0);
        if (lane == 0) output[row0] = acc0;
    }

    if (row1_valid) {
        float acc1 =
            float(scales[scale_off1]) * sig1;
        acc1 = simd_sum(acc1);
        if (lane == 0) output[row1] = acc1;
    }
}

// ====================================================================
// Kernel 5: qmv_spec_mg_f16io_resadd — multi-group, fused residual
// ====================================================================
/// Specialized multi-group QMV with K = SPEC_INTER_K for down
/// projections where K exceeds 32 * group_size.  Uses byte-aligned
/// lane assignment and explicit group-boundary tracking to flush
/// per-group scale multiplications.  Accumulates the f32 result
/// directly into the residual buffer (residual[row] += acc).
///
/// Unlike single-group kernels, each lane may span multiple scale
/// groups, so the inner loop tracks group boundaries and flushes
/// the signed accumulator at each crossing.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void qmv_spec_mg_f16io_resadd(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half*    scales      [[buffer(1)]],
    device const half*    input       [[buffer(2)]],
    device float*         residual    [[buffer(3)]],
    constant QMVDims&     dims        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_INTER_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 8 == 0,
        "K must be byte-aligned.");
    static_assert(group_size % 8 == 0,
        "Group size must be byte-aligned.");
    static_assert(K % group_size == 0,
        "K must be a multiple of group_size.");
    static_assert(K <= 12288,
        "K must fit in 24 KB threadgroup memory.");
    const uint M = dims.M;

    // Cache input vector in threadgroup memory to avoid
    // constant-cache pressure (K=6144 → 12 KB exceeds
    // per-CU constant cache on some Apple Silicon).
    threadgroup half tg_input[K];
    constexpr uint elems_per_thread = K / 512;
    static_assert(K % 512 == 0,
        "K must be divisible by 512 for cooperative load.");
    for (uint j = 0; j < elems_per_thread; j++) {
        tg_input[tid * elems_per_thread + j] =
            input[tid * elems_per_thread + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint total_row_bytes = K / 8;
    constexpr uint base_bytes = total_row_bytes / 32;
    constexpr uint extra_lanes = total_row_bytes % 32;
    // Each lane gets base_bytes, plus one extra if lane < extra.
    const uint my_bytes = base_bytes +
        ((lane < extra_lanes) ? 1u : 0u);
    const uint byte_start = lane * base_bytes +
        min(lane, extra_lanes);
    const uint col_start = byte_start * 8;

    const uint row_byte_base0 =
        row0 * total_row_bytes + byte_start;
    const uint row_byte_base1 =
        row1 * total_row_bytes + byte_start;
    const uint row_scale_off0 =
        row0 * groups_per_row;
    const uint row_scale_off1 =
        row1 * groups_per_row;

    constexpr uint bytes_per_group = group_size / 8;

    // Dual-row signed accumulation with group tracking.
    float accum0 = 0.0f, signed0 = 0.0f;
    float accum1 = 0.0f, signed1 = 0.0f;
    uint cur_group = col_start / group_size;
    uint byte_in_group =
        (col_start % group_size) / 8;

    // Use uint32 reads instead of byte-by-byte.
    // base_bytes is always 4-byte aligned when extra_lanes==0
    // (e.g. K=6144 → base_bytes=24 → 6 words).
    constexpr uint mg_words = base_bytes / 4;
    static_assert(base_bytes % 4 == 0,
        "base_bytes must be 4-byte aligned for uint32.");
    static_assert(extra_lanes == 0,
        "uint32 path requires uniform lane byte counts.");

    device const uint32_t* bits32_r0 =
        (device const uint32_t*)(
            packed_bits + row_byte_base0);
    device const uint32_t* bits32_r1 =
        (device const uint32_t*)(
            packed_bits + row_byte_base1);

    // Prevent full unrolling to preserve GPU occupancy.
    #pragma clang loop unroll(disable)
    for (uint w = 0; w < mg_words; w++) {
        const uint32_t w0 =
            row0_valid ? bits32_r0[w] : 0;
        const uint32_t w1 =
            row1_valid ? bits32_r1[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint base_col =
                col_start + (w * 4 + bi) * 8;

            // Widen f16 to f32 before accumulation.
            const float x0 =
                float(tg_input[base_col + 0]);
            const float x1 =
                float(tg_input[base_col + 1]);
            const float x2 =
                float(tg_input[base_col + 2]);
            const float x3 =
                float(tg_input[base_col + 3]);
            const float x4 =
                float(tg_input[base_col + 4]);
            const float x5 =
                float(tg_input[base_col + 5]);
            const float x6 =
                float(tg_input[base_col + 6]);
            const float x7 =
                float(tg_input[base_col + 7]);

            const uint sh = bi * 8;
            const uint bv0 = (w0 >> sh) & 0xFFu;
            const uint bv1 = (w1 >> sh) & 0xFFu;
            signed0 += select(-x0, x0, bool(bv0 & 1));
            signed0 += select(-x1, x1, bool(bv0 & 2));
            signed0 += select(-x2, x2, bool(bv0 & 4));
            signed0 += select(-x3, x3, bool(bv0 & 8));
            signed0 +=
                select(-x4, x4, bool(bv0 & 16));
            signed0 +=
                select(-x5, x5, bool(bv0 & 32));
            signed0 +=
                select(-x6, x6, bool(bv0 & 64));
            signed0 +=
                select(-x7, x7, bool(bv0 & 128));
            signed1 += select(-x0, x0, bool(bv1 & 1));
            signed1 += select(-x1, x1, bool(bv1 & 2));
            signed1 += select(-x2, x2, bool(bv1 & 4));
            signed1 += select(-x3, x3, bool(bv1 & 8));
            signed1 +=
                select(-x4, x4, bool(bv1 & 16));
            signed1 +=
                select(-x5, x5, bool(bv1 & 32));
            signed1 +=
                select(-x6, x6, bool(bv1 & 64));
            signed1 +=
                select(-x7, x7, bool(bv1 & 128));

            // Flush at group boundary.
            byte_in_group++;
            if (byte_in_group == bytes_per_group) {
                const float s = float(
                    scales[row_scale_off0 + cur_group]);
                accum0 += s * signed0;
                signed0 = 0.0f;
                const float t = float(
                    scales[row_scale_off1 + cur_group]);
                accum1 += t * signed1;
                signed1 = 0.0f;
                cur_group++;
                byte_in_group = 0;
            }
        }
    }

    // Flush remaining partial group.
    if (byte_in_group > 0) {
        accum0 += float(
            scales[row_scale_off0 + cur_group]
        ) * signed0;
        accum1 += float(
            scales[row_scale_off1 + cur_group]
        ) * signed1;
    }

    accum0 = simd_sum(accum0);
    accum1 = simd_sum(accum1);

    if (lane == 0) {
        if (row0_valid) residual[row0] += accum0;
        if (row1_valid) residual[row1] += accum1;
    }
}

// ====================================================================
// Kernel 6: qmv_spec_fused_pair_silu_f16io — gate/up + SiLU fused
// ====================================================================
/// Fused gate+up projection with SiLU activation and elementwise
/// multiply: output[row] = silu(gate_row) * up_row.  Eliminates
/// the separate SiLU+elementwise_mul dispatch and two barriers
/// per block (56 barriers saved for 28-layer Bonsai).
///
/// Uses 1 row per simdgroup (not 2) to keep register pressure low
/// enough for max_threads >= 512 despite the extra exp() in the
/// SiLU output stage.  Two accumulators (sig_gate, sig_up) vs
/// four in the 2-row variant.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void qmv_spec_fused_pair_silu_f16io(
    device const uint8_t* packed_a  [[buffer(0)]],
    device const half*    scales_a  [[buffer(1)]],
    constant half*        input     [[buffer(2)]],
    device half*          output    [[buffer(3)]],
    device const uint8_t* packed_b  [[buffer(4)]],
    device const half*    scales_b  [[buffer(5)]],
    // buffer(6) unused — single output replaces two.
    constant QMVDims&     dims      [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0, "K must be SIMD-aligned.");
    static_assert(K % group_size == 0, "K must be group-aligned.");
    static_assert(K / 32 <= group_size, "Single-group requirement.");
    const uint M = dims.M;

    // 1 row per simdgroup: 16 simdgroups × 1 row = 16 rows/TG.
    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 16 + simdgroup_idx;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    const uint col_byte_off = col_start / 8;
    const uint grp = col_start / group_size;
    constexpr uint bytes_per_row = K / 8;

    float sig_gate = 0.0f;
    float sig_up = 0.0f;
    const bool row_valid = row < M;
    const uint base_a = row * bytes_per_row + col_byte_off;
    const uint base_b = row * bytes_per_row + col_byte_off;

    constexpr uint words_per_lane = bytes_per_lane / 4;
    device const uint32_t* w32_a =
        (device const uint32_t*)(packed_a + base_a);
    device const uint32_t* w32_b =
        (device const uint32_t*)(packed_b + base_b);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t wa = row_valid ? w32_a[w] : 0;
        const uint32_t wb = row_valid ? w32_b[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint ci = col_start + w * 32 + bi * 8;
            const float x0 = float(input[ci]);
            const float x1 = float(input[ci + 1]);
            const float x2 = float(input[ci + 2]);
            const float x3 = float(input[ci + 3]);
            const float x4 = float(input[ci + 4]);
            const float x5 = float(input[ci + 5]);
            const float x6 = float(input[ci + 6]);
            const float x7 = float(input[ci + 7]);
            const uint sh = bi * 8;
            const uint ba = (wa >> sh) & 0xFFu;
            const uint bb = (wb >> sh) & 0xFFu;
            sig_gate += select(-x0, x0, bool(ba & 1));
            sig_gate += select(-x1, x1, bool(ba & 2));
            sig_gate += select(-x2, x2, bool(ba & 4));
            sig_gate += select(-x3, x3, bool(ba & 8));
            sig_gate += select(-x4, x4, bool(ba & 16));
            sig_gate += select(-x5, x5, bool(ba & 32));
            sig_gate += select(-x6, x6, bool(ba & 64));
            sig_gate += select(-x7, x7, bool(ba & 128));
            sig_up += select(-x0, x0, bool(bb & 1));
            sig_up += select(-x1, x1, bool(bb & 2));
            sig_up += select(-x2, x2, bool(bb & 4));
            sig_up += select(-x3, x3, bool(bb & 8));
            sig_up += select(-x4, x4, bool(bb & 16));
            sig_up += select(-x5, x5, bool(bb & 32));
            sig_up += select(-x6, x6, bool(bb & 64));
            sig_up += select(-x7, x7, bool(bb & 128));
        }
    }

    // Tail bytes — dead code when bytes_per_lane % 4 == 0.
    for (uint b = words_per_lane * 4; b < bytes_per_lane; b++) {
        const uint ci = col_start + b * 8;
        const float x0 = float(input[ci]);
        const float x1 = float(input[ci + 1]);
        const float x2 = float(input[ci + 2]);
        const float x3 = float(input[ci + 3]);
        const float x4 = float(input[ci + 4]);
        const float x5 = float(input[ci + 5]);
        const float x6 = float(input[ci + 6]);
        const float x7 = float(input[ci + 7]);
        if (row_valid) {
            const uint bva = packed_a[base_a + b];
            sig_gate += select(-x0, x0, bool(bva & 1));
            sig_gate += select(-x1, x1, bool(bva & 2));
            sig_gate += select(-x2, x2, bool(bva & 4));
            sig_gate += select(-x3, x3, bool(bva & 8));
            sig_gate += select(-x4, x4, bool(bva & 16));
            sig_gate += select(-x5, x5, bool(bva & 32));
            sig_gate += select(-x6, x6, bool(bva & 64));
            sig_gate += select(-x7, x7, bool(bva & 128));
            const uint bvb = packed_b[base_b + b];
            sig_up += select(-x0, x0, bool(bvb & 1));
            sig_up += select(-x1, x1, bool(bvb & 2));
            sig_up += select(-x2, x2, bool(bvb & 4));
            sig_up += select(-x3, x3, bool(bvb & 8));
            sig_up += select(-x4, x4, bool(bvb & 16));
            sig_up += select(-x5, x5, bool(bvb & 32));
            sig_up += select(-x6, x6, bool(bvb & 64));
            sig_up += select(-x7, x7, bool(bvb & 128));
        }
    }

    // Fused SiLU(gate) * up: apply SiLU to gate (matrix A),
    // multiply by up (matrix B), write single f16 output.
    if (row_valid) {
        const uint sc_off = row * groups_per_row + grp;
        float g = float(scales_a[sc_off]) * sig_gate;
        float u = float(scales_b[sc_off]) * sig_up;
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            const float silu = g / (1.0f + exp(-g));
            output[row] = half(silu * u);
        }
    }
}

// ====================================================================
// Kernel 7: qmv_spec_fused_norm_pair_silu_f16io
// ====================================================================
/// Fused FFN-RMSNorm + gate/up projection + SiLU activation.
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs
/// the gate+up QMV with SiLU.  Eliminates 1 dispatch + 1 barrier
/// per block vs the separate RMSNorm → gate+up+SiLU path.
///
/// Uses padded threadgroup stride (cols_per_lane + 2) to avoid
/// bank conflicts when 32 lanes read from TG memory.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void qmv_spec_fused_norm_pair_silu_f16io(
    device const uint8_t* packed_a   [[buffer(0)]],
    device const half*    scales_a   [[buffer(1)]],
    device const float*   residual   [[buffer(2)]],
    device half*          output     [[buffer(3)]],
    device const uint8_t* packed_b   [[buffer(4)]],
    device const half*    scales_b   [[buffer(5)]],
    device const half*    norm_scale [[buffer(6)]],
    constant QMVDims&     dims       [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0, "K must be SIMD-aligned.");
    static_assert(K % group_size == 0, "K must be group-aligned.");
    static_assert(K / 32 <= group_size, "Single-group requirement.");
    static_assert(K <= 4096, "K must fit in threadgroup memory.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    // Pad stride by 2 so adjacent lanes hit different TG memory
    // banks: bank = ((lane * stride) / 2) % 32 = lane when
    // stride is odd after dividing by 2 (66/2 = 33, gcd(33,32)=1).
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    constexpr uint bytes_per_row = K / 8;
    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint elems_per_thread = K / 512;

    static_assert(K % 512 == 0,
        "K must be divisible by 512 for cooperative load.");

    // TG memory: padded normalized input + reduction scratch.
    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────────────
    // Each thread sums squares of its share of the residual.
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v = residual[tid * elems_per_thread + j];
        partial_sos += v * v;
    }
    // Reduce within simdgroup, then across simdgroups.
    partial_sos = simd_sum(partial_sos);
    if (lane == 0) {
        tg_reduce[simdgroup_idx] = partial_sos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces the 16 partial sums.
    if (simdgroup_idx == 0) {
        const float v = (lane < 16)
            ? tg_reduce[lane] : 0.0f;
        const float total = simd_sum(v);
        if (lane == 0) {
            tg_reduce[0] = rsqrt(
                total / float(K) + 1e-6f
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv = tg_reduce[0];

    // ── Phase 2: Normalize and store f16 to TG memory ───────
    // Each thread normalizes its elements and writes them to
    // the padded TG layout for bank-conflict-free QMV reads.
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem = tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane = elem / cols_per_lane;
        const uint lane_off = elem % cols_per_lane;
        tg_input[qmv_lane * padded_stride + lane_off] =
            half(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Gate+Up QMV with SiLU from TG memory ──────
    const uint row = tgid * 16 + simdgroup_idx;
    const uint col_byte_off = lane * bytes_per_lane;
    const uint grp = (lane * cols_per_lane) / group_size;
    const uint tg_base = lane * padded_stride;

    float sig_gate = 0.0f;
    float sig_up = 0.0f;
    const bool row_valid = row < M;
    const uint base_a = row * bytes_per_row + col_byte_off;
    const uint base_b = row * bytes_per_row + col_byte_off;

    device const uint32_t* w32_a =
        (device const uint32_t*)(packed_a + base_a);
    device const uint32_t* w32_b =
        (device const uint32_t*)(packed_b + base_b);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t wa = row_valid ? w32_a[w] : 0;
        const uint32_t wb = row_valid ? w32_b[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint lo = w * 32 + bi * 8;
            const float x0 = float(tg_input[tg_base + lo]);
            const float x1 = float(tg_input[tg_base+lo+1]);
            const float x2 = float(tg_input[tg_base+lo+2]);
            const float x3 = float(tg_input[tg_base+lo+3]);
            const float x4 = float(tg_input[tg_base+lo+4]);
            const float x5 = float(tg_input[tg_base+lo+5]);
            const float x6 = float(tg_input[tg_base+lo+6]);
            const float x7 = float(tg_input[tg_base+lo+7]);
            const uint sh = bi * 8;
            const uint ba = (wa >> sh) & 0xFFu;
            const uint bb = (wb >> sh) & 0xFFu;
            sig_gate += select(-x0, x0, bool(ba & 1));
            sig_gate += select(-x1, x1, bool(ba & 2));
            sig_gate += select(-x2, x2, bool(ba & 4));
            sig_gate += select(-x3, x3, bool(ba & 8));
            sig_gate += select(-x4, x4, bool(ba & 16));
            sig_gate += select(-x5, x5, bool(ba & 32));
            sig_gate += select(-x6, x6, bool(ba & 64));
            sig_gate += select(-x7, x7, bool(ba & 128));
            sig_up += select(-x0, x0, bool(bb & 1));
            sig_up += select(-x1, x1, bool(bb & 2));
            sig_up += select(-x2, x2, bool(bb & 4));
            sig_up += select(-x3, x3, bool(bb & 8));
            sig_up += select(-x4, x4, bool(bb & 16));
            sig_up += select(-x5, x5, bool(bb & 32));
            sig_up += select(-x6, x6, bool(bb & 64));
            sig_up += select(-x7, x7, bool(bb & 128));
        }
    }

    // Tail bytes — dead code when bytes_per_lane % 4 == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++)
    {
        const uint lo = b * 8;
        const float x0 = float(tg_input[tg_base + lo]);
        const float x1 = float(tg_input[tg_base+lo+1]);
        const float x2 = float(tg_input[tg_base+lo+2]);
        const float x3 = float(tg_input[tg_base+lo+3]);
        const float x4 = float(tg_input[tg_base+lo+4]);
        const float x5 = float(tg_input[tg_base+lo+5]);
        const float x6 = float(tg_input[tg_base+lo+6]);
        const float x7 = float(tg_input[tg_base+lo+7]);
        if (row_valid) {
            const uint bva = packed_a[base_a + b];
            sig_gate += select(-x0, x0, bool(bva & 1));
            sig_gate += select(-x1, x1, bool(bva & 2));
            sig_gate += select(-x2, x2, bool(bva & 4));
            sig_gate += select(-x3, x3, bool(bva & 8));
            sig_gate += select(-x4, x4, bool(bva & 16));
            sig_gate += select(-x5, x5, bool(bva & 32));
            sig_gate += select(-x6, x6, bool(bva & 64));
            sig_gate += select(-x7, x7, bool(bva & 128));
            const uint bvb = packed_b[base_b + b];
            sig_up += select(-x0, x0, bool(bvb & 1));
            sig_up += select(-x1, x1, bool(bvb & 2));
            sig_up += select(-x2, x2, bool(bvb & 4));
            sig_up += select(-x3, x3, bool(bvb & 8));
            sig_up += select(-x4, x4, bool(bvb & 16));
            sig_up += select(-x5, x5, bool(bvb & 32));
            sig_up += select(-x6, x6, bool(bvb & 64));
            sig_up += select(-x7, x7, bool(bvb & 128));
        }
    }

    // Fused SiLU(gate) * up output.
    if (row_valid) {
        const uint sc_off = row * groups_per_row + grp;
        float g = float(scales_a[sc_off]) * sig_gate;
        float u = float(scales_b[sc_off]) * sig_up;
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            const float silu = g / (1.0f + exp(-g));
            output[row] = half(silu * u);
        }
    }
}

// ====================================================================
// Kernel 8: qmv_spec_fused_norm_f16io — fused RMSNorm + single QMV
// ====================================================================
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs a
/// single QMV from the TG-cached normalized input.  Used for the
/// Q attention projection to avoid a separate RMSNorm dispatch.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void qmv_spec_fused_norm_f16io(
    device const uint8_t* packed_bits [[buffer(0)]],
    device const half*    scales      [[buffer(1)]],
    device const float*   residual    [[buffer(2)]],
    device half*          output      [[buffer(3)]],
    device const half*    norm_scale  [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0, "K must be SIMD-aligned.");
    static_assert(K % group_size == 0, "K must be group-aligned.");
    static_assert(K / 32 <= group_size, "Single-group requirement.");
    static_assert(K <= 4096, "K must fit in threadgroup memory.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    constexpr uint bytes_per_row = K / 8;
    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint elems_per_thread = K / 512;

    static_assert(K % 512 == 0,
        "K must be divisible by 512 for cooperative load.");

    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────────────
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v = residual[tid * elems_per_thread + j];
        partial_sos += v * v;
    }
    partial_sos = simd_sum(partial_sos);
    if (lane == 0) {
        tg_reduce[simdgroup_idx] = partial_sos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_idx == 0) {
        const float v = (lane < 16)
            ? tg_reduce[lane] : 0.0f;
        const float total = simd_sum(v);
        if (lane == 0) {
            tg_reduce[0] = rsqrt(
                total / float(K) + 1e-6f
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv = tg_reduce[0];

    // ── Phase 2: Normalize and store f16 to TG memory ───────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem = tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane = elem / cols_per_lane;
        const uint lane_off = elem % cols_per_lane;
        tg_input[qmv_lane * padded_stride + lane_off] =
            half(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Single QMV from TG memory ──────────────────
    const uint row = tgid * 16 + simdgroup_idx;
    const uint col_byte_off = lane * bytes_per_lane;
    const uint grp = (lane * cols_per_lane) / group_size;
    const uint tg_base = lane * padded_stride;

    float sig = 0.0f;
    const bool row_valid = row < M;
    const uint base = row * bytes_per_row + col_byte_off;

    device const uint32_t* w32 =
        (device const uint32_t*)(packed_bits + base);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t wv = row_valid ? w32[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint lo = w * 32 + bi * 8;
            const float x0 = float(tg_input[tg_base + lo]);
            const float x1 = float(tg_input[tg_base+lo+1]);
            const float x2 = float(tg_input[tg_base+lo+2]);
            const float x3 = float(tg_input[tg_base+lo+3]);
            const float x4 = float(tg_input[tg_base+lo+4]);
            const float x5 = float(tg_input[tg_base+lo+5]);
            const float x6 = float(tg_input[tg_base+lo+6]);
            const float x7 = float(tg_input[tg_base+lo+7]);
            const uint sh = bi * 8;
            const uint bv = (wv >> sh) & 0xFFu;
            sig += select(-x0, x0, bool(bv & 1));
            sig += select(-x1, x1, bool(bv & 2));
            sig += select(-x2, x2, bool(bv & 4));
            sig += select(-x3, x3, bool(bv & 8));
            sig += select(-x4, x4, bool(bv & 16));
            sig += select(-x5, x5, bool(bv & 32));
            sig += select(-x6, x6, bool(bv & 64));
            sig += select(-x7, x7, bool(bv & 128));
        }
    }

    // Tail bytes — dead code when bytes_per_lane % 4 == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++)
    {
        const uint lo = b * 8;
        const float x0 = float(tg_input[tg_base + lo]);
        const float x1 = float(tg_input[tg_base+lo+1]);
        const float x2 = float(tg_input[tg_base+lo+2]);
        const float x3 = float(tg_input[tg_base+lo+3]);
        const float x4 = float(tg_input[tg_base+lo+4]);
        const float x5 = float(tg_input[tg_base+lo+5]);
        const float x6 = float(tg_input[tg_base+lo+6]);
        const float x7 = float(tg_input[tg_base+lo+7]);
        if (row_valid) {
            const uint bval = packed_bits[base + b];
            sig += select(-x0, x0, bool(bval & 1));
            sig += select(-x1, x1, bool(bval & 2));
            sig += select(-x2, x2, bool(bval & 4));
            sig += select(-x3, x3, bool(bval & 8));
            sig += select(-x4, x4, bool(bval & 16));
            sig += select(-x5, x5, bool(bval & 32));
            sig += select(-x6, x6, bool(bval & 64));
            sig += select(-x7, x7, bool(bval & 128));
        }
    }

    if (row_valid) {
        const uint sc_off = row * groups_per_row + grp;
        float acc = float(scales[sc_off]) * sig;
        acc = simd_sum(acc);
        if (lane == 0) {
            output[row] = half(acc);
        }
    }
}

// ====================================================================
// Kernel 9: qmv_spec_fused_norm_pair_f16io — fused RMSNorm + pair QMV
// ====================================================================
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs a
/// paired QMV (two weight matrices, two outputs) from the
/// TG-cached normalized input.  Used for fused K+V attention
/// projections to avoid a separate RMSNorm dispatch.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void qmv_spec_fused_norm_pair_f16io(
    device const uint8_t* packed_a   [[buffer(0)]],
    device const half*    scales_a   [[buffer(1)]],
    device const float*   residual   [[buffer(2)]],
    device half*          output_a   [[buffer(3)]],
    device const uint8_t* packed_b   [[buffer(4)]],
    device const half*    scales_b   [[buffer(5)]],
    device half*          output_b   [[buffer(6)]],
    device const half*    norm_scale [[buffer(7)]],
    constant QMVDims&     dims       [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_HIDDEN_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0, "K must be SIMD-aligned.");
    static_assert(K % group_size == 0, "K must be group-aligned.");
    static_assert(K / 32 <= group_size, "Single-group requirement.");
    static_assert(K <= 4096, "K must fit in threadgroup memory.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint bytes_per_lane = cols_per_lane / 8;
    constexpr uint bytes_per_row = K / 8;
    constexpr uint words_per_lane = bytes_per_lane / 4;
    constexpr uint elems_per_thread = K / 512;

    static_assert(K % 512 == 0,
        "K must be divisible by 512 for cooperative load.");

    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────────────
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v = residual[tid * elems_per_thread + j];
        partial_sos += v * v;
    }
    partial_sos = simd_sum(partial_sos);
    if (lane == 0) {
        tg_reduce[simdgroup_idx] = partial_sos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_idx == 0) {
        const float v = (lane < 16)
            ? tg_reduce[lane] : 0.0f;
        const float total = simd_sum(v);
        if (lane == 0) {
            tg_reduce[0] = rsqrt(
                total / float(K) + 1e-6f
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv = tg_reduce[0];

    // ── Phase 2: Normalize and store f16 to TG memory ───────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem = tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane = elem / cols_per_lane;
        const uint lane_off = elem % cols_per_lane;
        tg_input[qmv_lane * padded_stride + lane_off] =
            half(val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Paired QMV from TG memory ──────────────────
    const uint row = tgid * 16 + simdgroup_idx;
    const uint col_byte_off = lane * bytes_per_lane;
    const uint grp = (lane * cols_per_lane) / group_size;
    const uint tg_base = lane * padded_stride;

    float sig_a = 0.0f;
    float sig_b = 0.0f;
    const bool row_valid = row < M;
    const uint base_a = row * bytes_per_row + col_byte_off;
    const uint base_b = row * bytes_per_row + col_byte_off;

    device const uint32_t* w32_a =
        (device const uint32_t*)(packed_a + base_a);
    device const uint32_t* w32_b =
        (device const uint32_t*)(packed_b + base_b);

    for (uint w = 0; w < words_per_lane; w++) {
        const uint32_t wa = row_valid ? w32_a[w] : 0;
        const uint32_t wb = row_valid ? w32_b[w] : 0;

        for (uint bi = 0; bi < 4; bi++) {
            const uint lo = w * 32 + bi * 8;
            const float x0 = float(tg_input[tg_base + lo]);
            const float x1 = float(tg_input[tg_base+lo+1]);
            const float x2 = float(tg_input[tg_base+lo+2]);
            const float x3 = float(tg_input[tg_base+lo+3]);
            const float x4 = float(tg_input[tg_base+lo+4]);
            const float x5 = float(tg_input[tg_base+lo+5]);
            const float x6 = float(tg_input[tg_base+lo+6]);
            const float x7 = float(tg_input[tg_base+lo+7]);
            const uint sh = bi * 8;
            const uint ba = (wa >> sh) & 0xFFu;
            const uint bb = (wb >> sh) & 0xFFu;
            sig_a += select(-x0, x0, bool(ba & 1));
            sig_a += select(-x1, x1, bool(ba & 2));
            sig_a += select(-x2, x2, bool(ba & 4));
            sig_a += select(-x3, x3, bool(ba & 8));
            sig_a += select(-x4, x4, bool(ba & 16));
            sig_a += select(-x5, x5, bool(ba & 32));
            sig_a += select(-x6, x6, bool(ba & 64));
            sig_a += select(-x7, x7, bool(ba & 128));
            sig_b += select(-x0, x0, bool(bb & 1));
            sig_b += select(-x1, x1, bool(bb & 2));
            sig_b += select(-x2, x2, bool(bb & 4));
            sig_b += select(-x3, x3, bool(bb & 8));
            sig_b += select(-x4, x4, bool(bb & 16));
            sig_b += select(-x5, x5, bool(bb & 32));
            sig_b += select(-x6, x6, bool(bb & 64));
            sig_b += select(-x7, x7, bool(bb & 128));
        }
    }

    // Tail bytes — dead code when bytes_per_lane % 4 == 0.
    for (uint b = words_per_lane * 4;
         b < bytes_per_lane; b++)
    {
        const uint lo = b * 8;
        const float x0 = float(tg_input[tg_base + lo]);
        const float x1 = float(tg_input[tg_base+lo+1]);
        const float x2 = float(tg_input[tg_base+lo+2]);
        const float x3 = float(tg_input[tg_base+lo+3]);
        const float x4 = float(tg_input[tg_base+lo+4]);
        const float x5 = float(tg_input[tg_base+lo+5]);
        const float x6 = float(tg_input[tg_base+lo+6]);
        const float x7 = float(tg_input[tg_base+lo+7]);
        if (row_valid) {
            const uint bva = packed_a[base_a + b];
            sig_a += select(-x0, x0, bool(bva & 1));
            sig_a += select(-x1, x1, bool(bva & 2));
            sig_a += select(-x2, x2, bool(bva & 4));
            sig_a += select(-x3, x3, bool(bva & 8));
            sig_a += select(-x4, x4, bool(bva & 16));
            sig_a += select(-x5, x5, bool(bva & 32));
            sig_a += select(-x6, x6, bool(bva & 64));
            sig_a += select(-x7, x7, bool(bva & 128));
            const uint bvb = packed_b[base_b + b];
            sig_b += select(-x0, x0, bool(bvb & 1));
            sig_b += select(-x1, x1, bool(bvb & 2));
            sig_b += select(-x2, x2, bool(bvb & 4));
            sig_b += select(-x3, x3, bool(bvb & 8));
            sig_b += select(-x4, x4, bool(bvb & 16));
            sig_b += select(-x5, x5, bool(bvb & 32));
            sig_b += select(-x6, x6, bool(bvb & 64));
            sig_b += select(-x7, x7, bool(bvb & 128));
        }
    }

    if (row_valid) {
        const uint sc_off = row * groups_per_row + grp;
        float acc_a = float(scales_a[sc_off]) * sig_a;
        float acc_b = float(scales_b[sc_off]) * sig_b;
        acc_a = simd_sum(acc_a);
        acc_b = simd_sum(acc_b);
        if (lane == 0) {
            output_a[row] = half(acc_a);
            output_b[row] = half(acc_b);
        }
    }
}
