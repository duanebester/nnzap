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
// Memory access pattern: weight reads use an interleaved SIMD layout
// where adjacent lanes read adjacent uint32 words for fully coalesced
// 128-byte memory transactions.  Each lane accumulates per-word
// scale/bias contributions since words from different groups are
// visited in interleaved order.
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
// Kernel 1: q4mv_spec_f16io — half in, half out
// ====================================================================
/// Specialized Q4 QMV with K = SPEC_HIDDEN_K and
/// group_size = SPEC_GS.  Half-precision input (constant cache)
/// and half-precision output.  Interleaved SIMD access pattern:
/// adjacent lanes read adjacent uint32 words for coalesced
/// memory transactions.  Per-word scale/bias accumulation.
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row0 = tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + safe_row0 * nibble_bytes_per_row);
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + safe_row1 * nibble_bytes_per_row);

    float acc0 = 0.0f, acc1 = 0.0f;

    // Interleaved access: adjacent lanes read adjacent words.
    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t word0 = w32_r0[word_idx];
        const uint32_t word1 = w32_r1[word_idx];

        float nd0 = 0.0f, nd1 = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(input[col_base + ni]);
            nd0 += float((word0 >> (ni * 4)) & 0xFu) * x;
            nd1 += float((word1 >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc0 = safe_row0 * groups_per_row + grp;
        const uint sc1 = safe_row1 * groups_per_row + grp;
        acc0 += bf16_to_f32(scales[sc0]) * nd0
              + bf16_to_f32(biases[sc0]) * gs;
        acc1 += bf16_to_f32(scales[sc1]) * nd1
              + bf16_to_f32(biases[sc1]) * gs;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (row0_valid && lane == 0) {
        output[row0] = half(bf16_round(acc0));
    }
    if (row1_valid && lane == 0) {
        output[row1] = half(bf16_round(acc1));
    }
}

// ====================================================================
// Kernel 2: q4mv_spec_f16io_resadd — fused residual add
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
    device const half*    input       [[buffer(3)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative TG load.");
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    const uint M = dims.M;

    // Cache input in threadgroup memory — faster than
    // constant cache for non-uniform (interleaved) access.
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

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + safe_row0 * nibble_bytes_per_row);
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + safe_row1 * nibble_bytes_per_row);

    float acc0 = 0.0f, acc1 = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t word0 = w32_r0[word_idx];
        const uint32_t word1 = w32_r1[word_idx];

        float nd0 = 0.0f, nd1 = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(tg_input[col_base + ni]);
            nd0 += float((word0 >> (ni * 4)) & 0xFu) * x;
            nd1 += float((word1 >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc0 = safe_row0 * groups_per_row + grp;
        const uint sc1 = safe_row1 * groups_per_row + grp;
        acc0 += bf16_to_f32(scales[sc0]) * nd0
              + bf16_to_f32(biases[sc0]) * gs;
        acc1 += bf16_to_f32(scales[sc1]) * nd1
              + bf16_to_f32(biases[sc1]) * gs;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (row0_valid) residual[row0] += bf16_round(acc0);
        if (row1_valid) residual[row1] += bf16_round(acc1);
    }
}

// ====================================================================
// Kernel 3: q4mv_spec_fused_pair_f16io — two matrices, one input
// ====================================================================
/// Specialized fused-pair Q4 QMV with K = SPEC_HIDDEN_K.  Processes
/// two weight matrices (A and B) against a single shared input
/// vector in one dispatch, halving kernel launch overhead for
/// K/V projection and gate/up projection pairs.
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 16 + simdgroup_idx;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;

    const bool row_valid = row < M;
    const uint safe_row = min(row, M - 1);

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + safe_row * nibble_bytes_per_row);
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + safe_row * nibble_bytes_per_row);

    float acc_a = 0.0f, acc_b = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t wa = w32_a[word_idx];
        const uint32_t wb = w32_b[word_idx];

        float nda = 0.0f, ndb = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(input[col_base + ni]);
            nda += float((wa >> (ni * 4)) & 0xFu) * x;
            ndb += float((wb >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc = safe_row * groups_per_row + grp;
        acc_a += bf16_to_f32(scales_a[sc]) * nda
               + bf16_to_f32(biases_a[sc]) * gs;
        acc_b += bf16_to_f32(scales_b[sc]) * ndb
               + bf16_to_f32(biases_b[sc]) * gs;
    }

    acc_a = simd_sum(acc_a);
    acc_b = simd_sum(acc_b);

    if (row_valid && lane == 0) {
        output_a[row] = half(bf16_round(acc_a));
        output_b[row] = half(bf16_round(acc_b));
    }
}

// ====================================================================
// Kernel 4: q4mv_spec_f16in — half in, float out
// ====================================================================
/// Specialized Q4 QMV with K = SPEC_HIDDEN_K.  Reads
/// the activation vector as half through the constant address
/// space and writes full f32 output.  Used for LM head projection
/// that feeds into f32 accumulation (e.g. logits).
///
/// Dispatch: threadgroups = ceil(M / 32), threads = 512.
kernel void q4mv_spec_f16in(
    device const uint8_t* packed_nibs [[buffer(0)]],
    device const ushort*  scales      [[buffer(1)]],
    device const ushort*  biases      [[buffer(2)]],
    device const half*    input       [[buffer(3)]],
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative TG load.");
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    const uint M = dims.M;

    // Cache input in threadgroup memory — faster than
    // constant cache for non-uniform (interleaved) access.
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

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;

    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + safe_row0 * nibble_bytes_per_row);
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + safe_row1 * nibble_bytes_per_row);

    float acc0 = 0.0f, acc1 = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t word0 = w32_r0[word_idx];
        const uint32_t word1 = w32_r1[word_idx];

        float nd0 = 0.0f, nd1 = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(tg_input[col_base + ni]);
            nd0 += float((word0 >> (ni * 4)) & 0xFu) * x;
            nd1 += float((word1 >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc0 = safe_row0 * groups_per_row + grp;
        const uint sc1 = safe_row1 * groups_per_row + grp;
        acc0 += bf16_to_f32(scales[sc0]) * nd0
              + bf16_to_f32(biases[sc0]) * gs;
        acc1 += bf16_to_f32(scales[sc1]) * nd1
              + bf16_to_f32(biases[sc1]) * gs;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (row0_valid && lane == 0) {
        output[row0] = bf16_round(acc0);
    }
    if (row1_valid && lane == 0) {
        output[row1] = bf16_round(acc1);
    }
}

// ====================================================================
// Kernel 5: q4mv_spec_mg_f16io_resadd — multi-group, fused residual
// ====================================================================
/// Specialized multi-group Q4 QMV with K = SPEC_INTER_K for down
/// projections where K exceeds 32 * group_size.  Uses interleaved
/// SIMD access with per-word group tracking.  Accumulates the f32
/// result directly into the residual buffer (residual[row] += acc).
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
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative TG load.");
    static_assert(K <= 12288,
        "K must fit in 24 KB threadgroup memory.");
    const uint M = dims.M;

    // Cache input in threadgroup memory.
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
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;

    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs + safe_row0 * nibble_bytes_per_row);
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs + safe_row1 * nibble_bytes_per_row);

    float acc0 = 0.0f, acc1 = 0.0f;

    // Interleaved access with per-word group accumulation.
    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t word0 = w32_r0[word_idx];
        const uint32_t word1 = w32_r1[word_idx];

        float nd0 = 0.0f, nd1 = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(tg_input[col_base + ni]);
            nd0 += float((word0 >> (ni * 4)) & 0xFu) * x;
            nd1 += float((word1 >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc0 =
            safe_row0 * groups_per_row + grp;
        const uint sc1 =
            safe_row1 * groups_per_row + grp;
        acc0 += bf16_to_f32(scales[sc0]) * nd0
              + bf16_to_f32(biases[sc0]) * gs;
        acc1 += bf16_to_f32(scales[sc1]) * nd1
              + bf16_to_f32(biases[sc1]) * gs;
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
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    const uint M = dims.M;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 16 + simdgroup_idx;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;

    const bool row_valid = row < M;
    const uint safe_row = min(row, M - 1);

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + safe_row * nibble_bytes_per_row);
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + safe_row * nibble_bytes_per_row);

    float acc_g = 0.0f, acc_u = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t wa = w32_a[word_idx];
        const uint32_t wb = w32_b[word_idx];

        float nda = 0.0f, ndb = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(input[col_base + ni]);
            nda += float((wa >> (ni * 4)) & 0xFu) * x;
            ndb += float((wb >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc = safe_row * groups_per_row + grp;
        acc_g += bf16_to_f32(scales_a[sc]) * nda
               + bf16_to_f32(biases_a[sc]) * gs;
        acc_u += bf16_to_f32(scales_b[sc]) * ndb
               + bf16_to_f32(biases_b[sc]) * gs;
    }

    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);

    if (row_valid && lane == 0) {
        const float silu = acc_g / (1.0f + exp(-acc_g));
        output[row] = half(bf16_round(silu * acc_u));
    }
}

// ====================================================================
// Kernel 7: q4mv_spec_fused_norm_pair_silu_f16io
// ====================================================================
/// Fused FFN-RMSNorm + gate/up projection + SiLU activation.
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs
/// the gate+up Q4 QMV with SiLU.  Linear TG layout for the
/// normalized input, read with interleaved SIMD access pattern.
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
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    // Linear TG layout — no padding needed with interleaved access.
    threadgroup half tg_input[K];
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

    // ── Phase 2: Normalize to linear TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        tg_input[elem] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Gate+Up Q4 QMV with SiLU ──────────
    const uint row = tgid * 16 + simdgroup_idx;
    const bool row_valid = row < M;
    const uint safe_row = min(row, M - 1);

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + safe_row * nibble_bytes_per_row);
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + safe_row * nibble_bytes_per_row);

    float acc_g = 0.0f, acc_u = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t wa = w32_a[word_idx];
        const uint32_t wb = w32_b[word_idx];

        float nda = 0.0f, ndb = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x = float(tg_input[col_base + ni]);
            nda += float((wa >> (ni * 4)) & 0xFu) * x;
            ndb += float((wb >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc = safe_row * groups_per_row + grp;
        acc_g += bf16_to_f32(scales_a[sc]) * nda
               + bf16_to_f32(biases_a[sc]) * gs;
        acc_u += bf16_to_f32(scales_b[sc]) * ndb
               + bf16_to_f32(biases_b[sc]) * gs;
    }

    acc_g = simd_sum(acc_g);
    acc_u = simd_sum(acc_u);

    if (row_valid && lane == 0) {
        const float silu = acc_g / (1.0f + exp(-acc_g));
        output[row] = half(bf16_round(silu * acc_u));
    }
}

// ====================================================================
// Kernel 8: q4mv_spec_fused_norm_f16io — fused RMSNorm + single QMV
// ====================================================================
/// Reads the f32 residual and f16 norm_scale directly, computes
/// RMSNorm cooperatively in threadgroup memory, then performs a
/// single Q4 QMV from the TG-cached normalized input.  Used for
/// the Q attention projection to avoid a separate RMSNorm
/// dispatch.  Linear TG layout with interleaved SIMD reads.
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
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    threadgroup half tg_input[K];
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

    // ── Phase 2: Normalize to linear TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        tg_input[elem] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Single Q4 QMV from TG memory ──────
    const uint row0 =
        tgid * 32 + simdgroup_idx * 2;
    const uint row1 = row0 + 1;
    const bool row0_valid = row0 < M;
    const bool row1_valid = row1 < M;
    const uint safe_row0 = min(row0, M - 1);
    const uint safe_row1 = min(row1, M - 1);

    device const uint32_t* w32_r0 =
        (device const uint32_t*)(
            packed_nibs
            + safe_row0 * nibble_bytes_per_row);
    device const uint32_t* w32_r1 =
        (device const uint32_t*)(
            packed_nibs
            + safe_row1 * nibble_bytes_per_row);

    float acc0 = 0.0f, acc1 = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t word0 = w32_r0[word_idx];
        const uint32_t word1 = w32_r1[word_idx];

        float nd0 = 0.0f, nd1 = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x =
                float(tg_input[col_base + ni]);
            nd0 += float((word0 >> (ni * 4)) & 0xFu) * x;
            nd1 += float((word1 >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc0 =
            safe_row0 * groups_per_row + grp;
        const uint sc1 =
            safe_row1 * groups_per_row + grp;
        acc0 += bf16_to_f32(scales[sc0]) * nd0
              + bf16_to_f32(biases[sc0]) * gs;
        acc1 += bf16_to_f32(scales[sc1]) * nd1
              + bf16_to_f32(biases[sc1]) * gs;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (row0_valid && lane == 0) {
        output[row0] = half(bf16_round(acc0));
    }
    if (row1_valid && lane == 0) {
        output[row1] = half(bf16_round(acc1));
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
/// Linear TG layout with interleaved SIMD reads.
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
    static_assert(K <= 4096,
        "K must fit in threadgroup memory.");
    static_assert(K >= 256,
        "K must be >= 256 for Q4 uint32 path.");
    static_assert(group_size % 8 == 0,
        "Group size must be 8-aligned.");
    static_assert(K % 512 == 0,
        "K must be divisible by 512 "
        "for cooperative load.");
    const uint M = dims.M;

    constexpr uint groups_per_row = K / group_size;
    constexpr uint words_per_row = K / 8;
    constexpr uint uint32s_per_lane = words_per_row / 32;
    constexpr uint nibble_bytes_per_row = K / 2;
    constexpr uint elems_per_thread = K / 512;

    threadgroup half tg_input[K];
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

    // ── Phase 2: Normalize to linear TG memory ─────
    for (uint j = 0; j < elems_per_thread; j++) {
        const uint elem =
            tid * elems_per_thread + j;
        const float val = residual[elem] * rms_inv
            * float(norm_scale[elem]);
        tg_input[elem] = half(bf16_round(val));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Paired Q4 QMV from TG memory ──────
    const uint row = tgid * 16 + simdgroup_idx;
    const bool row_valid = row < M;
    const uint safe_row = min(row, M - 1);

    device const uint32_t* w32_a =
        (device const uint32_t*)(
            packed_a + safe_row * nibble_bytes_per_row);
    device const uint32_t* w32_b =
        (device const uint32_t*)(
            packed_b + safe_row * nibble_bytes_per_row);

    float acc_a = 0.0f, acc_b = 0.0f;

    #pragma clang loop unroll(disable)
    for (uint w = 0; w < uint32s_per_lane; w++) {
        const uint word_idx = w * 32 + lane;
        const uint col_base = word_idx * 8;
        const uint grp = col_base / group_size;

        const uint32_t wa = w32_a[word_idx];
        const uint32_t wb = w32_b[word_idx];

        float nda = 0.0f, ndb = 0.0f, gs = 0.0f;
        for (uint ni = 0; ni < 8; ni++) {
            const float x =
                float(tg_input[col_base + ni]);
            nda += float((wa >> (ni * 4)) & 0xFu) * x;
            ndb += float((wb >> (ni * 4)) & 0xFu) * x;
            gs += x;
        }

        const uint sc =
            safe_row * groups_per_row + grp;
        acc_a += bf16_to_f32(scales_a[sc]) * nda
               + bf16_to_f32(biases_a[sc]) * gs;
        acc_b += bf16_to_f32(scales_b[sc]) * ndb
               + bf16_to_f32(biases_b[sc]) * gs;
    }

    acc_a = simd_sum(acc_a);
    acc_b = simd_sum(acc_b);

    if (row_valid && lane == 0) {
        out_a[row] = half(bf16_round(acc_a));
        out_b[row] = half(bf16_round(acc_b));
    }
}
