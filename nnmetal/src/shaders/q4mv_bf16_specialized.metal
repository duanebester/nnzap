// ============================================================================
// q4mv_bf16_specialized.metal — BF16-faithful Q4 QMV kernels
// ============================================================================
//
// 4-bit quantized matrix-vector multiply (Q4 MLX format) specialized
// kernels, parallel to the 1-bit qmv_specialized.metal.  Weights are
// stored as 4-bit unsigned nibbles packed into uint32 (8 nibbles per
// uint32), with per-group BF16 scales and BF16 biases stored as raw uint16
// (affine quant).  Scales and biases are converted inline from the
// raw BF16 bit pattern to f32 for computation.  All output writes
// are BF16-rounded (truncated to 16-bit mantissa) to match MLX's
// BF16 arithmetic precision.
//
// Dequantization: w = scale * float(nibble) + bias
// Dot product:    row = scale * Sum(nibble_i * x_i)
//                     + bias  * Sum(x_i)
//
// Each lane tracks two accumulators per group:
//   nib_dot: the weighted dot product Sum(nibble * x)
//   grp_sum: the input sum Sum(x)  (shared across rows)
//
// Packing layout: each uint32 holds 8 sequential nibbles.
//   nibble i at bits [i*4 .. i*4+3].
//   Extract: (word >> (ni * 4)) & 0xFu
//   nibble_bytes_per_row = K / 2
//
// K and group_size are compile-time constants (constexpr).  Loop
// unrolling is explicitly disabled via pragma to keep register
// pressure low and preserve GPU occupancy (memory-bound kernels).
//
// The three macros SPEC_HIDDEN_K, SPEC_INTER_K, and SPEC_GS are
// prepended by the Zig build system before compilation.  This file
// must NOT define them.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// BF16 helpers — match MLX's bfloat16 arithmetic precision.
// ----------------------------------------------------------------

/// Convert a raw BF16 uint16 bit pattern to f32.
/// BF16 is the upper 16 bits of IEEE 754 f32: zero-extend
/// by shifting left 16.  Lossless.
inline float bf16_to_f32(ushort raw) {
    return as_type<float>(uint(raw) << 16);
}

/// Round an f32 value to BF16 precision by zeroing the lower
/// 16 mantissa bits.  This truncation (not round-to-nearest)
/// matches MLX's BF16 intermediate rounding behavior.
inline float bf16_round(float val) {
    return as_type<float>(as_type<uint>(val) & 0xFFFF0000u);
}

// ----------------------------------------------------------------
// Guard macros — fail loudly if the build system forgot them.
// ----------------------------------------------------------------

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
// Kernel 1: q4mv_spec_f16io — single-group, half in, half out
// ====================================================================
/// Specialized single-group Q4 QMV with K = SPEC_HIDDEN_K and
/// group_size = SPEC_GS.  Half-precision input (constant cache)
/// and half-precision output.  Two accumulators per row (nib_dot
/// and grp_sum) enable the affine dequant identity:
///   row = scale * Sum(nibble * x) + bias * Sum(x)
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_f16io(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    constant half*        input       [[buffer(3)]],
    device half*          output      [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + row0 * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + row1 * nibble_bytes_per_row
        ) + word_off;

    float nib_dot0 = 0.0f, nib_dot1 = 0.0f;
    float grp_sum = 0.0f;

    // Process 8 nibbles (8 columns) per uint32.
    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t word0 =
            row0_valid ? w32_r0[w] : 0;
        const uint32_t word1 =
            row1_valid ? w32_r1[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x = float(input[ci]);
            const uint n0 =
                (word0 >> (ni * 4)) & 0xFu;
            const uint n1 =
                (word1 >> (ni * 4)) & 0xFu;
            nib_dot0 += float(n0) * x;
            nib_dot1 += float(n1) * x;
            grp_sum += x;
        }
    }

    // Affine dequant: row = scale * nib_dot + bias * grp_sum.
    const uint sc_off0 =
        row0 * groups_per_row + grp;
    const uint sc_off1 =
        row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc0 =
            bf16_to_f32(scales[sc_off0]) * nib_dot0
            + bf16_to_f32(biases[sc_off0]) * grp_sum;
        acc0 = simd_sum(acc0);
        if (lane == 0) output[row0] = half(bf16_round(acc0));
    }

    if (row1_valid) {
        float acc1 =
            bf16_to_f32(scales[sc_off1]) * nib_dot1
            + bf16_to_f32(biases[sc_off1]) * grp_sum;
        acc1 = simd_sum(acc1);
        if (lane == 0) output[row1] = half(bf16_round(acc1));
    }
}

// ====================================================================
// Kernel 2: q4mv_spec_f16io_resadd — single-group, fused residual
// ====================================================================
/// Identical to q4mv_spec_f16io but accumulates the f32 result
/// directly into the residual buffer (residual[row] += acc)
/// instead of writing half to an intermediate buffer.  Eliminates
/// one dispatch and one barrier per call site.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_f16io_resadd(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    constant half*        input       [[buffer(3)]],
    device float*         residual    [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + row0 * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + row1 * nibble_bytes_per_row
        ) + word_off;

    float nib_dot0 = 0.0f, nib_dot1 = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t word0 =
            row0_valid ? w32_r0[w] : 0;
        const uint32_t word1 =
            row1_valid ? w32_r1[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x = float(input[ci]);
            const uint n0 =
                (word0 >> (ni * 4)) & 0xFu;
            const uint n1 =
                (word1 >> (ni * 4)) & 0xFu;
            nib_dot0 += float(n0) * x;
            nib_dot1 += float(n1) * x;
            grp_sum += x;
        }
    }

    const uint sc_off0 =
        row0 * groups_per_row + grp;
    const uint sc_off1 =
        row1 * groups_per_row + grp;

    // Lane 0 accumulates the f32 result into the residual.
    if (row0_valid) {
        float acc0 =
            bf16_to_f32(scales[sc_off0]) * nib_dot0
            + bf16_to_f32(biases[sc_off0]) * grp_sum;
        acc0 = simd_sum(acc0);
        if (lane == 0) residual[row0] += bf16_round(acc0);
    }

    if (row1_valid) {
        float acc1 =
            bf16_to_f32(scales[sc_off1]) * nib_dot1
            + bf16_to_f32(biases[sc_off1]) * grp_sum;
        acc1 = simd_sum(acc1);
        if (lane == 0) residual[row1] += bf16_round(acc1);
    }
}

// ====================================================================
// Kernel 3: q4mv_spec_fused_pair_f16io — two matrices, one input
// ====================================================================
/// Specialized fused-pair Q4 QMV with K = SPEC_HIDDEN_K.  Processes
/// two weight matrices (A and B) against a single shared input
/// vector in one dispatch, halving kernel launch overhead for
/// K/V projection and gate/up projection pairs.  Three
/// accumulators per lane: nib_dot_a, nib_dot_b, and grp_sum
/// (shared across both matrices since it depends only on input).
///
/// Uses 1 row per simdgroup (not 2) to manage register pressure
/// from the two concurrent weight streams.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void q4mv_spec_fused_pair_f16io(
    device const uint8_t* packed_a   [[buffer(0)]],
    device const ushort*  scales_a   [[buffer(1)]],
    device const ushort*  biases_a   [[buffer(2)]],
    constant half*        input      [[buffer(3)]],
    device half*          output_a   [[buffer(4)]],
    device const uint8_t* packed_b   [[buffer(5)]],
    device const ushort*  scales_b   [[buffer(6)]],
    device const ushort*  biases_b   [[buffer(7)]],
    device half*          output_b   [[buffer(8)]],
    constant QMVDims&     dims       [[buffer(9)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    const uint M = dims.M;

    // 1 row per simdgroup: 16 simdgroups x 1 = 16 rows/TG.
    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 16 + simdgroup_idx;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;

    const bool row_valid = row < M;
    const uint row_nib_base =
        row * nibble_bytes_per_row;

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + row_nib_base
        ) + word_off;
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + row_nib_base
        ) + word_off;

    float nib_dot_a = 0.0f, nib_dot_b = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t wa =
            row_valid ? w32_a[w] : 0;
        const uint32_t wb =
            row_valid ? w32_b[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x = float(input[ci]);
            const uint na =
                (wa >> (ni * 4)) & 0xFu;
            const uint nb =
                (wb >> (ni * 4)) & 0xFu;
            nib_dot_a += float(na) * x;
            nib_dot_b += float(nb) * x;
            grp_sum += x;
        }
    }

    if (row_valid) {
        const uint sc_off =
            row * groups_per_row + grp;
        float acc_a =
            bf16_to_f32(scales_a[sc_off]) * nib_dot_a
            + bf16_to_f32(biases_a[sc_off]) * grp_sum;
        float acc_b =
            bf16_to_f32(scales_b[sc_off]) * nib_dot_b
            + bf16_to_f32(biases_b[sc_off]) * grp_sum;
        acc_a = simd_sum(acc_a);
        acc_b = simd_sum(acc_b);
        if (lane == 0) {
            output_a[row] = half(bf16_round(acc_a));
            output_b[row] = half(bf16_round(acc_b));
        }
    }
}

// ====================================================================
// Kernel 4: q4mv_spec_f16in — single-group, half in, float out
// ====================================================================
/// Specialized single-group Q4 QMV with K = SPEC_HIDDEN_K.  Reads
/// the activation vector as half through the constant address
/// space and writes full f32 output.  Used for LM head projection
/// that feeds into f32 accumulation (e.g. logits).
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_f16in(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    constant half*        input       [[buffer(3)]],
    device float*         output      [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + row0 * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + row1 * nibble_bytes_per_row
        ) + word_off;

    float nib_dot0 = 0.0f, nib_dot1 = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t word0 =
            row0_valid ? w32_r0[w] : 0;
        const uint32_t word1 =
            row1_valid ? w32_r1[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x = float(input[ci]);
            const uint n0 =
                (word0 >> (ni * 4)) & 0xFu;
            const uint n1 =
                (word1 >> (ni * 4)) & 0xFu;
            nib_dot0 += float(n0) * x;
            nib_dot1 += float(n1) * x;
            grp_sum += x;
        }
    }

    const uint sc_off0 =
        row0 * groups_per_row + grp;
    const uint sc_off1 =
        row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc0 =
            bf16_to_f32(scales[sc_off0]) * nib_dot0
            + bf16_to_f32(biases[sc_off0]) * grp_sum;
        acc0 = simd_sum(acc0);
        if (lane == 0) output[row0] = bf16_round(acc0);
    }

    if (row1_valid) {
        float acc1 =
            bf16_to_f32(scales[sc_off1]) * nib_dot1
            + bf16_to_f32(biases[sc_off1]) * grp_sum;
        acc1 = simd_sum(acc1);
        if (lane == 0) output[row1] = bf16_round(acc1);
    }
}

// ====================================================================
// Kernel 5: q4mv_spec_mg_f16io_resadd — multi-group, fused residual
// ====================================================================
/// Specialized multi-group Q4 QMV with K = SPEC_INTER_K for down
/// projections where K exceeds 32 * group_size.  Uses byte-aligned
/// lane assignment and explicit group-boundary tracking to flush
/// per-group scale+bias multiplications.  Accumulates the f32
/// result directly into the residual buffer (residual[row] += acc).
///
/// Unlike single-group kernels, each lane may span multiple scale
/// groups.  The inner loop tracks group boundaries via a word
/// counter (group_size % 8 == 0 ensures boundaries align with
/// uint32 boundaries) and flushes the nib_dot and grp_sum
/// accumulators at each crossing.
///
/// Input is cached in threadgroup memory to avoid constant-cache
/// pressure (SPEC_INTER_K may exceed per-CU constant cache).
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_mg_f16io_resadd(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    device const half*    input       [[buffer(3)]],
    device float*         residual    [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint K = SPEC_INTER_K;
    constexpr uint group_size = SPEC_GS;
    static_assert(K % 32 == 0,
        "K must be SIMD-aligned.");
    static_assert(K % group_size == 0,
        "K must be a multiple of group_size.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-column-aligned "
        "for uint32 boundary alignment.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 2) % 32 == 0,
        "Nibble bytes per row must divide evenly "
        "across 32 lanes.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative TG load.");
    static_assert(K <= 12288,
        "K must fit in 24 KB threadgroup memory.");
    const uint M = dims.M;

    // Cache input in threadgroup memory to avoid
    // constant-cache pressure (SPEC_INTER_K may
    // exceed per-CU constant cache on Apple Silicon).
    threadgroup half tg_input[K];
    constexpr uint elems_per_thread = K / 512;
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
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint bytes_per_lane =
        nibble_bytes_per_row / 32;
    constexpr uint cols_per_lane = bytes_per_lane * 2;
    constexpr uint uint32s_per_lane =
        bytes_per_lane / 4;
    constexpr uint words_per_group = group_size / 8;

    static_assert(bytes_per_lane % 4 == 0,
        "bytes_per_lane must be 4-byte aligned "
        "for uint32 reads.");

    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;

    // Use safe_row to avoid reading scales/biases at
    // invalid row offsets during the flush loop.  The
    // result for invalid rows is discarded at output.
    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);
    const uint row_sc0 =
        safe_row0 * groups_per_row;
    const uint row_sc1 =
        safe_row1 * groups_per_row;

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs
            + safe_row0 * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs
            + safe_row1 * nibble_bytes_per_row
        ) + word_off;

    // Dual-row accumulation with group tracking.
    float acc0 = 0.0f, acc1 = 0.0f;
    float nib_dot0 = 0.0f, nib_dot1 = 0.0f;
    float grp_sum = 0.0f;
    uint cur_grp = col_start / group_size;
    uint words_in_group =
        (col_start % group_size) / 8;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t word0 =
            row0_valid ? w32_r0[w] : 0;
        const uint32_t word1 =
            row1_valid ? w32_r1[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x =
                float(tg_input[ci]);
            const uint n0 =
                (word0 >> (ni * 4)) & 0xFu;
            const uint n1 =
                (word1 >> (ni * 4)) & 0xFu;
            nib_dot0 += float(n0) * x;
            nib_dot1 += float(n1) * x;
            grp_sum += x;
        }

        // Flush at group boundary (every words_per_group
        // uint32s processed).
        words_in_group++;
        if (words_in_group == words_per_group) {
            const uint s0 = row_sc0 + cur_grp;
            const uint s1 = row_sc1 + cur_grp;
            acc0 +=
                bf16_to_f32(scales[s0]) * nib_dot0
                + bf16_to_f32(biases[s0]) * grp_sum;
            acc1 +=
                bf16_to_f32(scales[s1]) * nib_dot1
                + bf16_to_f32(biases[s1]) * grp_sum;
            nib_dot0 = 0.0f;
            nib_dot1 = 0.0f;
            grp_sum = 0.0f;
            cur_grp++;
            words_in_group = 0;
        }
    }

    // Flush remaining partial group.
    if (words_in_group > 0) {
        const uint s0 = row_sc0 + cur_grp;
        const uint s1 = row_sc1 + cur_grp;
        acc0 +=
            bf16_to_f32(scales[s0]) * nib_dot0
            + bf16_to_f32(biases[s0]) * grp_sum;
        acc1 +=
            bf16_to_f32(scales[s1]) * nib_dot1
            + bf16_to_f32(biases[s1]) * grp_sum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (row0_valid) residual[row0] += bf16_round(acc0);
        if (row1_valid) residual[row1] += bf16_round(acc1);
    }
}

// ====================================================================
// Kernel 6: q4mv_spec_fused_pair_silu_f16io — gate/up + SiLU fused
// ====================================================================
/// Fused gate+up projection with SiLU activation and elementwise
/// multiply: output[row] = silu(gate_row) * up_row.  Eliminates
/// the separate SiLU+elementwise_mul dispatch and two barriers
/// per block.
///
/// Uses 1 row per simdgroup (not 2) to keep register pressure
/// low enough for max_threads >= 512 despite the extra exp()
/// in the SiLU output stage.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void q4mv_spec_fused_pair_silu_f16io(
    device const uint8_t* packed_a  [[buffer(0)]],
    device const ushort*  scales_a  [[buffer(1)]],
    device const ushort*  biases_a  [[buffer(2)]],
    constant half*        input     [[buffer(3)]],
    device half*          output    [[buffer(4)]],
    device const uint8_t* packed_b  [[buffer(5)]],
    device const ushort*  scales_b  [[buffer(6)]],
    device const ushort*  biases_b  [[buffer(7)]],
    constant QMVDims&     dims      [[buffer(8)]],
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
        "Single-group requirement.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    const uint M = dims.M;

    // 1 row per simdgroup: 16 simdgroups x 1 = 16 rows/TG.
    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 16 + simdgroup_idx;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint cols_per_lane = K / 32;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;

    const bool row_valid = row < M;
    const uint row_nib_base =
        row * nibble_bytes_per_row;

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + row_nib_base
        ) + word_off;
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + row_nib_base
        ) + word_off;

    float nib_dot_a = 0.0f, nib_dot_b = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t wa =
            row_valid ? w32_a[w] : 0;
        const uint32_t wb =
            row_valid ? w32_b[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint ci =
                col_start + w * 8 + ni;
            const float x = float(input[ci]);
            const uint na =
                (wa >> (ni * 4)) & 0xFu;
            const uint nb =
                (wb >> (ni * 4)) & 0xFu;
            nib_dot_a += float(na) * x;
            nib_dot_b += float(nb) * x;
            grp_sum += x;
        }
    }

    // Fused SiLU(gate) * up with affine dequant.
    if (row_valid) {
        const uint sc_off =
            row * groups_per_row + grp;
        float g =
            bf16_to_f32(scales_a[sc_off]) * nib_dot_a
            + bf16_to_f32(biases_a[sc_off]) * grp_sum;
        float u =
            bf16_to_f32(scales_b[sc_off]) * nib_dot_b
            + bf16_to_f32(biases_b[sc_off]) * grp_sum;
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            const float silu =
                g / (1.0f + exp(-g));
            output[row] = half(bf16_round(silu * u));
        }
    }
}

// ====================================================================
// Kernel 7: q4mv_spec_fused_norm_pair_silu_f16io
// ====================================================================
/// Fused FFN-RMSNorm + gate/up projection + SiLU activation.
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs
/// the gate+up Q4 QMV with SiLU.  Eliminates 1 dispatch +
/// 1 barrier per block vs the separate RMSNorm -> gate+up+SiLU.
///
/// Uses padded threadgroup stride (cols_per_lane + 2) to avoid
/// bank conflicts when 32 lanes read from TG memory.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void q4mv_spec_fused_norm_pair_silu_f16io(
    device const uint8_t* packed_a   [[buffer(0)]],
    device const ushort*  scales_a   [[buffer(1)]],
    device const ushort*  biases_a   [[buffer(2)]],
    device const float*   residual   [[buffer(3)]],
    device half*          output     [[buffer(4)]],
    device const uint8_t* packed_b   [[buffer(5)]],
    device const ushort*  scales_b   [[buffer(6)]],
    device const ushort*  biases_b   [[buffer(7)]],
    constant QMVDims&     dims       [[buffer(8)]],
    device const float*   norm_scale [[buffer(9)]],
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
        "Single-group requirement.");
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    // Pad stride by 2 so adjacent lanes hit different
    // TG memory banks: gcd((cols_per_lane+2)/2, 32)=1.
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    // TG memory: padded normalized input + reduction.
    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v =
            residual[tid * elems_per_thread + j];
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

    // ── Phase 2: Normalize to padded TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane =
            elem / cols_per_lane;
        const uint lane_off =
            elem % cols_per_lane;
        tg_input[
            qmv_lane * padded_stride + lane_off
        ] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Gate+Up Q4 QMV with SiLU ──────────
    const uint row = tgid * 16 + simdgroup_idx;
    const bool row_valid = row < M;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;
    const uint tg_base = lane * padded_stride;

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + row * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + row * nibble_bytes_per_row
        ) + word_off;

    float nib_dot_a = 0.0f, nib_dot_b = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t wa =
            row_valid ? w32_a[w] : 0;
        const uint32_t wb =
            row_valid ? w32_b[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint lo = w * 8 + ni;
            const float x =
                float(tg_input[tg_base + lo]);
            const uint na =
                (wa >> (ni * 4)) & 0xFu;
            const uint nb =
                (wb >> (ni * 4)) & 0xFu;
            nib_dot_a += float(na) * x;
            nib_dot_b += float(nb) * x;
            grp_sum += x;
        }
    }

    // Fused SiLU(gate) * up with affine dequant.
    if (row_valid) {
        const uint sc_off =
            row * groups_per_row + grp;
        float g =
            bf16_to_f32(scales_a[sc_off]) * nib_dot_a
            + bf16_to_f32(biases_a[sc_off]) * grp_sum;
        float u =
            bf16_to_f32(scales_b[sc_off]) * nib_dot_b
            + bf16_to_f32(biases_b[sc_off]) * grp_sum;
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            const float silu =
                g / (1.0f + exp(-g));
            output[row] = half(bf16_round(silu * u));
        }
    }
}

// ====================================================================
// Kernel 8: q4mv_spec_fused_norm_f16io — fused RMSNorm + single QMV
// ====================================================================
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs a
/// single Q4 QMV from the TG-cached normalized input.  Used for
/// the Q attention projection to avoid a separate RMSNorm
/// dispatch.
///
/// Uses 2 rows per simdgroup (lower register pressure than pair
/// kernels allows it), giving 32 rows per threadgroup.
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_fused_norm_f16io(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    device const float*   residual    [[buffer(3)]],
    device half*          output      [[buffer(4)]],
    constant QMVDims&     dims        [[buffer(5)]],
    device const float*   norm_scale  [[buffer(6)]],
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
        "Single-group requirement.");
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v =
            residual[tid * elems_per_thread + j];
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

    // ── Phase 2: Normalize to padded TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane =
            elem / cols_per_lane;
        const uint lane_off =
            elem % cols_per_lane;
        tg_input[
            qmv_lane * padded_stride + lane_off
        ] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Single Q4 QMV from TG memory ──────
    const uint row0 =
        tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;
    const uint tg_base = lane * padded_stride;

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs
            + row0 * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs
            + row1 * nibble_bytes_per_row
        ) + word_off;

    float nib_dot0 = 0.0f, nib_dot1 = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t word0 =
            row0_valid ? w32_r0[w] : 0;
        const uint32_t word1 =
            row1_valid ? w32_r1[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint lo = w * 8 + ni;
            const float x =
                float(tg_input[tg_base + lo]);
            const uint n0 =
                (word0 >> (ni * 4)) & 0xFu;
            const uint n1 =
                (word1 >> (ni * 4)) & 0xFu;
            nib_dot0 += float(n0) * x;
            nib_dot1 += float(n1) * x;
            grp_sum += x;
        }
    }

    const uint sc_off0 =
        row0 * groups_per_row + grp;
    const uint sc_off1 =
        row1 * groups_per_row + grp;

    if (row0_valid) {
        float acc0 =
            bf16_to_f32(scales[sc_off0]) * nib_dot0
            + bf16_to_f32(biases[sc_off0]) * grp_sum;
        acc0 = simd_sum(acc0);
        if (lane == 0) output[row0] = half(bf16_round(acc0));
    }

    if (row1_valid) {
        float acc1 =
            bf16_to_f32(scales[sc_off1]) * nib_dot1
            + bf16_to_f32(biases[sc_off1]) * grp_sum;
        acc1 = simd_sum(acc1);
        if (lane == 0) output[row1] = half(bf16_round(acc1));
    }
}

// ====================================================================
// Kernel 9: q4mv_spec_fused_norm_pair_f16io
// ====================================================================
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs a
/// paired Q4 QMV (two weight matrices, two outputs) from the
/// TG-cached normalized input.  Used for fused K+V attention
/// projections to avoid a separate RMSNorm dispatch.
///
/// Dispatch: threadgroups = ceil(M / 16), threads = 512.
kernel void q4mv_spec_fused_norm_pair_f16io(
    device const uint8_t* packed_a   [[buffer(0)]],
    device const ushort*  scales_a   [[buffer(1)]],
    device const ushort*  biases_a   [[buffer(2)]],
    device const float*   residual   [[buffer(3)]],
    device half*          out_a      [[buffer(4)]],
    device const uint8_t* packed_b   [[buffer(5)]],
    device const ushort*  scales_b   [[buffer(6)]],
    device const ushort*  biases_b   [[buffer(7)]],
    device half*          out_b      [[buffer(8)]],
    constant QMVDims&     dims       [[buffer(9)]],
    device const float*   norm_scale [[buffer(10)]],
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
        "Single-group requirement.");
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert((K / 32) % 8 == 0,
        "cols_per_lane must be a multiple of 8 "
        "(8 nibbles per uint32).");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint cols_per_lane = K / 32;
    constexpr uint padded_stride = cols_per_lane + 2;
    constexpr uint groups_per_row = K / group_size;
    constexpr uint uint32s_per_lane = cols_per_lane / 8;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    threadgroup half tg_input[32 * padded_stride];
    threadgroup float tg_reduce[16];

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;

    // ── Phase 1: Cooperative RMSNorm ────────────────
    float partial_sos = 0.0f;
    for (uint j = 0; j < elems_per_thread; j++) {
        const float v =
            residual[tid * elems_per_thread + j];
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

    // ── Phase 2: Normalize to padded TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        const uint qmv_lane =
            elem / cols_per_lane;
        const uint lane_off =
            elem % cols_per_lane;
        tg_input[
            qmv_lane * padded_stride + lane_off
        ] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Paired Q4 QMV from TG memory ──────
    const uint row = tgid * 16 + simdgroup_idx;
    const bool row_valid = row < M;
    const uint col_start = lane * cols_per_lane;
    const uint word_off = col_start / 8;
    const uint grp = col_start / group_size;
    const uint tg_base = lane * padded_stride;

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + row * nibble_bytes_per_row
        ) + word_off;
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + row * nibble_bytes_per_row
        ) + word_off;

    float nib_dot_a = 0.0f, nib_dot_b = 0.0f;
    float grp_sum = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint32_t wa =
            row_valid ? w32_a[w] : 0;
        const uint32_t wb =
            row_valid ? w32_b[w] : 0;

        for (uint ni = 0; ni < 8; ni++) {
            const uint lo = w * 8 + ni;
            const float x =
                float(tg_input[tg_base + lo]);
            const uint na =
                (wa >> (ni * 4)) & 0xFu;
            const uint nb =
                (wb >> (ni * 4)) & 0xFu;
            nib_dot_a += float(na) * x;
            nib_dot_b += float(nb) * x;
            grp_sum += x;
        }
    }

    if (row_valid) {
        const uint sc_off =
            row * groups_per_row + grp;
        float acc_a =
            bf16_to_f32(scales_a[sc_off]) * nib_dot_a
            + bf16_to_f32(biases_a[sc_off]) * grp_sum;
        float acc_b =
            bf16_to_f32(scales_b[sc_off]) * nib_dot_b
            + bf16_to_f32(biases_b[sc_off]) * grp_sum;
        acc_a = simd_sum(acc_a);
        acc_b = simd_sum(acc_b);
        if (lane == 0) {
            out_a[row] = half(bf16_round(acc_a));
            out_b[row] = half(bf16_round(acc_b));
        }
    }
}
