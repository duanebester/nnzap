#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Vector operations — proving the pipeline works (step 1)
// ============================================================================

/// Element-wise vector addition: out[i] = a[i] + b[i]
kernel void vector_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    out[gid] = a[gid] + b[gid];
}

// ============================================================================
// Matrix multiply — the workhorse (step 2)
// ============================================================================

/// General matrix multiply: out = W * x
///   W is [M x K], x is [K x N], out is [M x N]
///
/// Each thread computes one element of the output matrix.
kernel void matmul(
    device const float* W   [[buffer(0)]],  // [M x K]
    device const float* x   [[buffer(1)]],  // [K x N]
    device float*       out [[buffer(2)]],  // [M x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (m)
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0;
    for (uint i = 0; i < K; i++) {
        sum += W[gid.y * K + i] * x[i * N + gid.x];
    }
    out[gid.y * N + gid.x] = sum;
}

// Tiled matmul with threadgroup memory for better memory bandwidth utilization.
// Uses 16x16 tile size to match the dispatch2D threadgroup size.
kernel void matmul_tiled(
    device const float* W   [[buffer(0)]],  // [M x K]
    device const float* x   [[buffer(1)]],  // [K x N]
    device float*       out [[buffer(2)]],  // [M x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]],
    uint2 tid               [[thread_position_in_threadgroup]])
{
    const uint TS = 16;  // Tile size, matches threadgroup 16x16

    // Bounds check before any work.
    if (gid.x >= N || gid.y >= M) return;

    // Threadgroup shared memory for tiles.
    threadgroup float W_tile[TS][TS];
    threadgroup float x_tile[TS][TS];

    float sum = 0.0;

    // Iterate over tiles in the K dimension.
    for (uint k_tile = 0; k_tile < (K + TS - 1) / TS; k_tile++) {
        // Load W tile: each thread loads one element.
        const uint w_row = gid.y;
        const uint w_col = k_tile * TS + tid.x;
        if (w_col < K && w_row < M) {
            W_tile[tid.y][tid.x] = W[w_row * K + w_col];
        } else {
            W_tile[tid.y][tid.x] = 0.0;
        }

        // Load x tile: each thread loads one element.
        const uint x_row = k_tile * TS + tid.y;
        const uint x_col = gid.x;
        if (x_row < K && x_col < N) {
            x_tile[tid.y][tid.x] = x[x_row * N + x_col];
        } else {
            x_tile[tid.y][tid.x] = 0.0;
        }

        // Synchronize to ensure all threads have loaded their data.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product using the loaded tiles.
        for (uint k = 0; k < TS; k++) {
            sum += W_tile[tid.y][k] * x_tile[k][tid.x];
        }

        // Synchronize before loading the next tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write the result.
    out[gid.y * N + gid.x] = sum;
}

/// Fused matmul + bias_add with tiled shared memory:
///   out = W * x + bias
///   W is [M x K], x is [K x N], bias is [N], out is [M x N]
///
/// Uses 16×16 threadgroup tiles to cooperatively load W and x
/// into threadgroup memory, reducing global memory reads by
/// ~16× compared to the naive kernel. Each tile iteration
/// loads a 16×16 block of W and x, computes the partial dot
/// product, then advances to the next tile along K.
constant constexpr uint TS = 16;  // tile size

kernel void matmul_bias(
    device const float* W    [[buffer(0)]],  // [M x K]
    device const float* x    [[buffer(1)]],  // [K x N]
    device float*       out  [[buffer(2)]],  // [M x N]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],  // [N]
    uint2 gid                [[thread_position_in_grid]],
    uint2 lid                [[thread_position_in_threadgroup]])
{
    // Shared memory tiles for cooperative loading.
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS];

    const uint row = gid.y;  // output row (m)
    const uint col = gid.x;  // output col (n)

    float sum = 0.0f;

    // Iterate over tiles along the K dimension.
    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        // Load one element of W tile: W[row, t*TS + lid.x]
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        // Load one element of x tile: x[t*TS + lid.y, col]
        const uint x_row = t * TS + lid.y;
        if (x_row < K && col < N) {
            tileX[lid.y * TS + lid.x] = x[x_row * N + col];
        } else {
            tileX[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate partial dot product from this tile.
        for (uint i = 0; i < TS; i++) {
            sum += tileW[lid.y * TS + i] * tileX[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result with bias, bounds-checked.
    if (row < M && col < N) {
        out[row * N + col] = sum + bias[col];
    }
}

/// Fused matmul + bias_add + ReLU with tiled shared memory:
///   out = max(0, W*x + bias)
///   Also stores pre-activation for the backward pass.
///   Uses the same 16×16 tiling strategy as matmul_bias.
kernel void matmul_bias_relu(
    device const float* W    [[buffer(0)]],  // [M x K]
    device const float* x    [[buffer(1)]],  // [K x N]
    device float*       out  [[buffer(2)]],  // [M x N]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],  // [N]
    device float*    pre_act [[buffer(7)]],  // [M x N]
    uint2 gid                [[thread_position_in_grid]],
    uint2 lid                [[thread_position_in_threadgroup]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS];

    const uint row = gid.y;
    const uint col = gid.x;

    float sum = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        const uint x_row = t * TS + lid.y;
        if (x_row < K && col < N) {
            tileX[lid.y * TS + lid.x] = x[x_row * N + col];
        } else {
            tileX[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileW[lid.y * TS + i]
                 * tileX[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        const uint idx = row * N + col;
        float val = sum + bias[col];
        pre_act[idx] = val;
        out[idx] = max(0.0f, val);
    }
}

// ============================================================================
// Inference-only fused matmul + bias + ReLU (no pre_act store)
// ============================================================================

/// Fused matmul + bias_add + ReLU for inference:
///   out = max(0, W*x + bias)
/// Identical tiling strategy to matmul_bias_relu but omits the
/// pre_act buffer write, since inference has no backward pass.
/// This saves one setBuffer call and one global memory store per
/// output element.
kernel void matmul_bias_relu_infer(
    device const float* W    [[buffer(0)]],  // [M x K]
    device const float* x    [[buffer(1)]],  // [K x N]
    device float*       out  [[buffer(2)]],  // [M x N]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],  // [N]
    uint2 gid                [[thread_position_in_grid]],
    uint2 lid                [[thread_position_in_threadgroup]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS];

    const uint row = gid.y;
    const uint col = gid.x;

    float sum = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        const uint x_row = t * TS + lid.y;
        if (x_row < K && col < N) {
            tileX[lid.y * TS + lid.x] = x[x_row * N + col];
        } else {
            tileX[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileW[lid.y * TS + i]
                 * tileX[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        out[row * N + col] = max(0.0f, sum + bias[col]);
    }
}

// ============================================================================
// Packed-dims matmul variants (fewer setBytes calls)
// ============================================================================

/// Packed matrix dimensions for batched inference.
/// One setBytes call at buffer(3) replaces three separate
/// calls for M, K, N — saving 2 Obj-C message sends per
/// dispatch (~20 us at ~10 us each).
struct MatmulDims {
    uint M;
    uint K;
    uint N;
};

/// matmul_bias with packed MatmulDims struct.
/// Buffer layout matches matmul_bias except buffer(3)
/// is a MatmulDims struct instead of separate M at (3),
/// K at (4), N at (5).  Bias moved to buffer(4).
kernel void matmul_bias_packed(
    device const float* W       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant MatmulDims& dims   [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS];

    const uint row = gid.y;
    const uint col = gid.x;
    const uint M = dims.M;
    const uint K = dims.K;
    const uint N = dims.N;

    float sum = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        const uint x_row = t * TS + lid.y;
        if (x_row < K && col < N) {
            tileX[lid.y * TS + lid.x] = x[x_row * N + col];
        } else {
            tileX[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileW[lid.y * TS + i]
                 * tileX[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        out[row * N + col] = sum + bias[col];
    }
}

/// matmul_bias_relu_infer with packed MatmulDims struct.
/// Inference-only: no pre_act write.  Fused ReLU.
kernel void matmul_bias_relu_infer_packed(
    device const float* W       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant MatmulDims& dims   [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS];

    const uint row = gid.y;
    const uint col = gid.x;
    const uint M = dims.M;
    const uint K = dims.K;
    const uint N = dims.N;

    float sum = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        const uint x_row = t * TS + lid.y;
        if (x_row < K && col < N) {
            tileX[lid.y * TS + lid.x] = x[x_row * N + col];
        } else {
            tileX[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileW[lid.y * TS + i]
                 * tileX[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        out[row * N + col] = max(0.0f, sum + bias[col]);
    }
}

// ============================================================================
// Register-tiled matmul + bias (1×2 output per thread)
// ============================================================================

/// Fused matmul + bias with 1×2 register tiling:
///   out = W * x + bias
/// Each thread computes 2 output columns (col, col+16), doubling
/// arithmetic intensity per shared memory load.  Threadgroup is
/// 16×16; output block per threadgroup is 16 rows × 32 cols.
///
/// Dispatch with width = ceil(N / 2), height = M so that
/// dispatch2D creates ceil(N/32) × ceil(M/16) threadgroups.
kernel void matmul_bias_v2(
    device const float* W    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],
    uint2 lid                [[thread_position_in_threadgroup]],
    uint2 tgid               [[threadgroup_position_in_grid]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS * 2];

    const uint row = tgid.y * TS + lid.y;
    // Two output columns per thread, 32 apart.
    const uint col0 = tgid.x * (TS * 2) + lid.x;
    const uint col1 = col0 + TS;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        // Load W tile [16×16]: one element per thread.
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        // Load X tile [16×32]: two elements per thread.
        const uint x_row = t * TS + lid.y;
        const uint x_stride = TS * 2;
        if (x_row < K && col0 < N) {
            tileX[lid.y * x_stride + lid.x] =
                x[x_row * N + col0];
        } else {
            tileX[lid.y * x_stride + lid.x] = 0.0f;
        }
        if (x_row < K && col1 < N) {
            tileX[lid.y * x_stride + lid.x + TS] =
                x[x_row * N + col1];
        } else {
            tileX[lid.y * x_stride + lid.x + TS] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate dot products from both column halves.
        for (uint i = 0; i < TS; i++) {
            float w = tileW[lid.y * TS + i];
            sum0 += w * tileX[i * x_stride + lid.x];
            sum1 += w * tileX[i * x_stride + lid.x + TS];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col0 < N) {
        out[row * N + col0] = sum0 + bias[col0];
    }
    if (row < M && col1 < N) {
        out[row * N + col1] = sum1 + bias[col1];
    }
}

/// Fused matmul + bias + ReLU with 1×2 register tiling:
///   out = max(0, W*x + bias)
/// Same tiling as matmul_bias_v2; also stores pre-activation.
kernel void matmul_bias_relu_v2(
    device const float* W    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],
    device float*    pre_act [[buffer(7)]],
    uint2 lid                [[thread_position_in_threadgroup]],
    uint2 tgid               [[threadgroup_position_in_grid]])
{
    threadgroup float tileW[TS * TS];
    threadgroup float tileX[TS * TS * 2];

    const uint row = tgid.y * TS + lid.y;
    const uint col0 = tgid.x * (TS * 2) + lid.x;
    const uint col1 = col0 + TS;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        const uint w_col = t * TS + lid.x;
        if (row < M && w_col < K) {
            tileW[lid.y * TS + lid.x] = W[row * K + w_col];
        } else {
            tileW[lid.y * TS + lid.x] = 0.0f;
        }

        const uint x_row = t * TS + lid.y;
        const uint x_stride = TS * 2;
        if (x_row < K && col0 < N) {
            tileX[lid.y * x_stride + lid.x] =
                x[x_row * N + col0];
        } else {
            tileX[lid.y * x_stride + lid.x] = 0.0f;
        }
        if (x_row < K && col1 < N) {
            tileX[lid.y * x_stride + lid.x + TS] =
                x[x_row * N + col1];
        } else {
            tileX[lid.y * x_stride + lid.x + TS] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            float w = tileW[lid.y * TS + i];
            sum0 += w * tileX[i * x_stride + lid.x];
            sum1 += w * tileX[i * x_stride + lid.x + TS];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col0 < N) {
        const uint idx = row * N + col0;
        float val = sum0 + bias[col0];
        pre_act[idx] = val;
        out[idx] = max(0.0f, val);
    }
    if (row < M && col1 < N) {
        const uint idx = row * N + col1;
        float val = sum1 + bias[col1];
        pre_act[idx] = val;
        out[idx] = max(0.0f, val);
    }
}

// ============================================================================
// Transposed matrix multiplies (backward pass)
// ============================================================================

/// Tiled A^T * B where A is [M x K] row-major, B is [M x N].
/// Result is [K x N].  Used for weight gradients: dW = X^T * dY.
///
/// Output element [k, n] = sum_m A[m*K+k] * B[m*N+n].
/// Tiles over the M (summation) dimension.
kernel void matmul_transA(
    device const float* A   [[buffer(0)]],  // [M x K]
    device const float* B   [[buffer(1)]],  // [M x N]
    device float*       out [[buffer(2)]],  // [K x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]],
    uint2 lid               [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TS * TS];
    threadgroup float tileB[TS * TS];

    const uint out_row = gid.y;  // k
    const uint out_col = gid.x;  // n

    float sum = 0.0f;

    // Tile over M (the summation/contraction dimension).
    const uint num_tiles = (M + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        // Load A^T tile: element [out_row, t*TS+lid.x]
        // = A[t*TS+lid.x, out_row] in original layout.
        const uint m_idx = t * TS + lid.x;
        if (out_row < K && m_idx < M) {
            tileA[lid.y * TS + lid.x] = A[m_idx * K + out_row];
        } else {
            tileA[lid.y * TS + lid.x] = 0.0f;
        }

        // Load B tile: element [t*TS+lid.y, out_col].
        const uint m_idy = t * TS + lid.y;
        if (m_idy < M && out_col < N) {
            tileB[lid.y * TS + lid.x] = B[m_idy * N + out_col];
        } else {
            tileB[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileA[lid.y * TS + i]
                 * tileB[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_row < K && out_col < N) {
        out[out_row * N + out_col] = sum;
    }
}

/// Tiled A * B^T where A is [M x K], B is [N x K] row-major.
/// Result is [M x N].  Used for input gradients: dX = dY * W^T.
///
/// Output element [m, n] = sum_k A[m*K+k] * B[n*K+k].
/// Tiles over the K (summation) dimension.
kernel void matmul_transB(
    device const float* A   [[buffer(0)]],  // [M x K]
    device const float* B   [[buffer(1)]],  // [N x K]
    device float*       out [[buffer(2)]],  // [M x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]],
    uint2 lid               [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TS * TS];
    threadgroup float tileB[TS * TS];

    const uint row = gid.y;  // m
    const uint col = gid.x;  // n

    float sum = 0.0f;

    // Tile over K (the summation dimension).
    const uint num_tiles = (K + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        // Load A tile: A[row, t*TS + lid.x].
        const uint k_a = t * TS + lid.x;
        if (row < M && k_a < K) {
            tileA[lid.y * TS + lid.x] = A[row * K + k_a];
        } else {
            tileA[lid.y * TS + lid.x] = 0.0f;
        }

        // Load B^T tile: B^T[t*TS+lid.y, col]
        // = B[col, t*TS+lid.y] in original layout.
        const uint k_b = t * TS + lid.y;
        if (col < N && k_b < K) {
            tileB[lid.y * TS + lid.x] = B[col * K + k_b];
        } else {
            tileB[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileA[lid.y * TS + i]
                 * tileB[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        out[row * N + col] = sum;
    }
}

// ============================================================================
// Activation functions (forward pass)
// ============================================================================

/// ReLU: out[i] = max(0, x[i])
kernel void relu_forward(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint                gid    [[thread_position_in_grid]])
{
    output[gid] = max(0.0f, input[gid]);
}

/// Tanh: out[i] = tanh(x[i])
kernel void tanh_forward(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint                gid    [[thread_position_in_grid]])
{
    output[gid] = tanh(input[gid]);
}

/// Sigmoid: out[i] = 1 / (1 + exp(-x[i]))
kernel void sigmoid_forward(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    uint                gid    [[thread_position_in_grid]])
{
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// ============================================================================
// Activation functions (backward pass)
// ============================================================================

/// ReLU backward: grad_in[i] = grad_out[i] * (input[i] > 0 ? 1 : 0)
kernel void relu_backward(
    device const float* input    [[buffer(0)]],  // forward input (pre-activation)
    device const float* grad_out [[buffer(1)]],  // upstream gradient
    device float*       grad_in  [[buffer(2)]],  // downstream gradient
    uint                gid      [[thread_position_in_grid]])
{
    grad_in[gid] = input[gid] > 0.0f ? grad_out[gid] : 0.0f;
}

/// Tanh backward: grad_in[i] = grad_out[i] * (1 - output[i]^2)
kernel void tanh_backward(
    device const float* output   [[buffer(0)]],  // forward output (post-activation)
    device const float* grad_out [[buffer(1)]],
    device float*       grad_in  [[buffer(2)]],
    uint                gid      [[thread_position_in_grid]])
{
    float t = output[gid];
    grad_in[gid] = grad_out[gid] * (1.0f - t * t);
}

/// Sigmoid backward: grad_in[i] = grad_out[i] * output[i] * (1 - output[i])
kernel void sigmoid_backward(
    device const float* output   [[buffer(0)]],  // forward output (post-activation)
    device const float* grad_out [[buffer(1)]],
    device float*       grad_in  [[buffer(2)]],
    uint                gid      [[thread_position_in_grid]])
{
    float s = output[gid];
    grad_in[gid] = grad_out[gid] * s * (1.0f - s);
}

// ============================================================================
// Bias operations
// ============================================================================

/// Add bias to each row: out[row * N + col] = input[row * N + col] + bias[col]
/// Note: M (row count) is passed to enable bounds checking
/// when dispatched with full threadgroups.
kernel void bias_add(
    device const float* input  [[buffer(0)]],  // [M x N]
    device const float* bias   [[buffer(1)]],  // [N]
    device float*       output [[buffer(2)]],  // [M x N]
    constant uint& N           [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    // gid.x = column, gid.y = row
    // Bounds check needed because dispatch uses full
    // threadgroups (dispatchThreadgroups) which may
    // launch threads beyond the matrix dimensions.
    if (gid.x >= N) return;
    uint idx = gid.y * N + gid.x;
    output[idx] = input[idx] + bias[gid.x];
}

/// Column-wise sum: out[n] = sum_m input[m * N + n].
/// Used for bias gradients — reduces [batch x out] to [out].
/// Each thread computes the sum over all rows for one column.
kernel void bias_grad(
    device const float* input  [[buffer(0)]],  // [M x N]
    device float*       output [[buffer(1)]],  // [N]
    constant uint& M           [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid >= N) return;

    float sum = 0.0;
    for (uint m = 0; m < M; m++) {
        sum += input[m * N + gid];
    }
    output[gid] = sum;
}

// ============================================================================
// SGD parameter update
// ============================================================================

/// params[i] -= lr * grads[i]
kernel void sgd_update(
    device float*       params [[buffer(0)]],
    device const float* grads  [[buffer(1)]],
    constant float& lr         [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]])
{
    params[gid] -= lr * grads[gid];
}

// ============================================================================
// Fused gradient + SGD update kernels
// ============================================================================

/// Fused bias gradient + SGD update:
///   bias[n] -= lr * sum_m( grad_pre_act[m * N + n] )
/// Eliminates the separate bias_grad write to a gradient
/// buffer and the SGD read of that buffer.  One thread per
/// output column sums the batch dimension and directly
/// updates the bias parameter.
kernel void bias_grad_sgd(
    device const float* input    [[buffer(0)]],  // [M x N]
    device float*       params   [[buffer(1)]],  // bias [N]
    constant uint& M             [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant float& lr           [[buffer(4)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= N) return;

    float sum = 0.0f;
    for (uint m = 0; m < M; m++) {
        sum += input[m * N + gid];
    }
    // Fused SGD: apply learning rate and update in-place.
    params[gid] -= lr * sum;
}

/// Fused weight gradient + SGD update (tiled A^T * B):
///   W[k, n] -= lr * sum_m( A[m*K+k] * B[m*N+n] )
/// Same tiled matmul_transA logic but writes the SGD update
/// directly to the param buffer instead of a gradient buffer.
/// Eliminates the weight_grad write + SGD read round-trip.
kernel void weight_grad_sgd(
    device const float* A    [[buffer(0)]],  // [M x K]
    device const float* B    [[buffer(1)]],  // [M x N]
    device float*       W    [[buffer(2)]],  // [K x N] params
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    constant float& lr       [[buffer(6)]],
    uint2 gid                [[thread_position_in_grid]],
    uint2 lid                [[thread_position_in_threadgroup]])
{
    threadgroup float tileA[TS * TS];
    threadgroup float tileB[TS * TS];

    const uint out_row = gid.y;  // k
    const uint out_col = gid.x;  // n

    float sum = 0.0f;

    // Tile over M (summation/contraction dimension).
    const uint num_tiles = (M + TS - 1) / TS;
    for (uint t = 0; t < num_tiles; t++) {
        // Load A^T tile: A[t*TS+lid.x, out_row].
        const uint m_idx = t * TS + lid.x;
        if (out_row < K && m_idx < M) {
            tileA[lid.y * TS + lid.x] =
                A[m_idx * K + out_row];
        } else {
            tileA[lid.y * TS + lid.x] = 0.0f;
        }

        // Load B tile: B[t*TS+lid.y, out_col].
        const uint m_idy = t * TS + lid.y;
        if (m_idy < M && out_col < N) {
            tileB[lid.y * TS + lid.x] =
                B[m_idy * N + out_col];
        } else {
            tileB[lid.y * TS + lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            sum += tileA[lid.y * TS + i]
                 * tileB[i * TS + lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fused SGD: apply gradient directly to weights.
    if (out_row < K && out_col < N) {
        W[out_row * N + out_col] -= lr * sum;
    }
}

// ============================================================================
// Loss functions
// ============================================================================

/// Per-element MSE loss: out[i] = 0.5 * (pred[i] - target[i])^2
kernel void mse_forward(
    device const float* pred   [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]])
{
    float diff = pred[gid] - target[gid];
    out[gid] = 0.5f * diff * diff;
}

/// Per-element MSE loss gradient: grad[i] = (pred[i] - target[i]) / batch_size
kernel void mse_backward(
    device const float* pred       [[buffer(0)]],
    device const float* target     [[buffer(1)]],
    device float*       grad       [[buffer(2)]],
    constant uint& batch_size      [[buffer(3)]],
    uint                gid        [[thread_position_in_grid]])
{
    grad[gid] = (pred[gid] - target[gid]) / float(batch_size);
}

// ============================================================================
// Softmax + Cross-entropy
// ============================================================================

/// Row-wise softmax (numerically stable).
/// One thread per sample; loops over num_classes.
///   1. Find max logit (subtract for stability).
///   2. Exponentiate and sum.
///   3. Divide by sum.
kernel void softmax_forward(
    device const float* logits      [[buffer(0)]],
    device float*       probs       [[buffer(1)]],
    constant uint& num_classes      [[buffer(2)]],
    uint                gid         [[thread_position_in_grid]])
{
    uint base = gid * num_classes;

    // 1. Find max for numerical stability.
    float max_val = logits[base];
    for (uint c = 1; c < num_classes; c++) {
        max_val = max(max_val, logits[base + c]);
    }

    // 2. Exponentiate and sum.
    float sum = 0.0;
    for (uint c = 0; c < num_classes; c++) {
        float e = exp(logits[base + c] - max_val);
        probs[base + c] = e;
        sum += e;
    }

    // 3. Normalize.
    for (uint c = 0; c < num_classes; c++) {
        probs[base + c] /= sum;
    }
}

/// Per-sample cross-entropy loss:
///   loss[s] = -sum_c( target[s*C+c] * log(probs[s*C+c] + eps) )
/// One thread per sample.
kernel void ce_forward(
    device const float* probs       [[buffer(0)]],
    device const float* target      [[buffer(1)]],
    device float*       loss        [[buffer(2)]],
    constant uint& num_classes      [[buffer(3)]],
    uint                gid         [[thread_position_in_grid]])
{
    uint base = gid * num_classes;
    float sum = 0.0;
    for (uint c = 0; c < num_classes; c++) {
        sum += target[base + c] * log(probs[base + c] + 1e-7f);
    }
    loss[gid] = -sum;
}

/// Fused softmax + cross-entropy backward.
/// Computes softmax of logits, then:
///   grad[s*C+c] = (softmax_c - target[s*C+c]) / batch_size
/// One thread per sample; writes num_classes gradient values.
kernel void softmax_ce_backward(
    device const float* logits      [[buffer(0)]],
    device const float* target      [[buffer(1)]],
    device float*       grad        [[buffer(2)]],
    constant uint& num_classes      [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],
    uint                gid         [[thread_position_in_grid]])
{
    uint base = gid * num_classes;

    // Softmax (same stable algorithm as softmax_forward).
    float max_val = logits[base];
    for (uint c = 1; c < num_classes; c++) {
        max_val = max(max_val, logits[base + c]);
    }

    float sum = 0.0;
    for (uint c = 0; c < num_classes; c++) {
        float e = exp(logits[base + c] - max_val);
        grad[base + c] = e;
        sum += e;
    }

    // grad = (softmax - target) / batch_size.
    float inv_batch = 1.0f / float(batch_size);
    for (uint c = 0; c < num_classes; c++) {
        float softmax_c = grad[base + c] / sum;
        grad[base + c] = (softmax_c - target[base + c]) * inv_batch;
    }
}

// ============================================================================
// Argmax — per-sample class prediction for batched evaluation
// ============================================================================

/// Per-sample argmax: finds the index of the largest logit for
/// each sample and writes it as a float to the output buffer.
/// One thread per sample.  Used to batch evaluation forward
/// passes by extracting predictions between forward dispatches
/// within the same command encoder, avoiding per-batch
/// commitAndWait overhead.
///
///   logits: [batch_size x num_classes], row-major.
///   predictions: output buffer, writes at [offset + gid].
///   num_classes: number of classes per sample.
///   offset: write offset in the predictions buffer (in
///           float elements) for this batch group.
kernel void argmax_predictions(
    device const float* logits        [[buffer(0)]],
    device float*       predictions   [[buffer(1)]],
    constant uint& num_classes        [[buffer(2)]],
    constant uint& offset             [[buffer(3)]],
    uint                gid           [[thread_position_in_grid]])
{
    // Each thread processes one sample.
    const uint base = gid * num_classes;

    float best_val = logits[base];
    uint best_idx = 0;
    for (uint c = 1; c < num_classes; c++) {
        float val = logits[base + c];
        if (val > best_val) {
            best_val = val;
            best_idx = c;
        }
    }

    // Store predicted class as float at the offset position.
    predictions[offset + gid] = float(best_idx);
}

// ============================================================================
// Fused 3-layer inference for batch=1 (784→128→64→10)
// ============================================================================

/// Packed weight/bias offsets for the 3-layer fused kernel.
/// Using a single struct at buffer(3) instead of 6 separate
/// constant bindings reduces Obj-C setBytes calls from 6 to 1,
/// cutting ~30% of Metal API overhead per inference dispatch.
struct FusedOffsets {
    uint w0;
    uint b0;
    uint w1;
    uint b1;
    uint w2;
    uint b2;
};

/// Single-dispatch inference for the MNIST architecture.
/// Computes all three layers (784→128 relu, 128→64 relu,
/// 64→10 none) in one kernel dispatch, eliminating two
/// setPipelineState + dispatchThreadgroups round-trips.
///
/// Uses threadgroup memory for inter-layer activations so
/// intermediate results stay on-chip.  128 threads per
/// threadgroup (matching the largest hidden layer).
///
/// Buffer layout:
///   buffer(0): input [784]
///   buffer(1): params [109386] — full param buffer
///   buffer(2): output [10] — final logits
///   buffer(3): FusedOffsets struct (packed weight/bias offsets)
///
/// Thread model: 1 threadgroup of 128 threads.  Each thread
/// computes one output element per layer (threads >= layer
/// output size are idle for that layer).
kernel void forward_fused_infer_3layer(
    device const float* input    [[buffer(0)]],
    device const float* params   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    constant FusedOffsets& offsets [[buffer(3)]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    // Layer dimensions (comptime-known architecture).
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    // Threadgroup memory for intermediate activations.
    // Only OUT0 (128) floats needed — reused between layers.
    threadgroup float act0[OUT0];  // layer 0 output
    threadgroup float act1[OUT1];  // layer 1 output

    // ---- Layer 0: 784 → 128, ReLU ----
    // Each of 128 threads computes one output element.
    {
        device const float* W0 = params + offsets.w0;
        device const float* B0 = params + offsets.b0;

        float sum = B0[tid];
        // W0 is [IN0 x OUT0] row-major: W0[k * OUT0 + tid]
        // gives coalesced reads across threads in a SIMD group.
        for (uint k = 0; k < IN0; k++) {
            sum += input[k] * W0[k * OUT0 + tid];
        }
        act0[tid] = max(0.0f, sum);  // ReLU
    }

    // Sync so all 128 outputs are visible.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    // Only first 64 threads are active.
    if (tid < OUT1) {
        device const float* W1 = params + offsets.w1;
        device const float* B1 = params + offsets.b1;

        float sum = B1[tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * W1[k * OUT1 + tid];
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    // Only first 10 threads are active.
    if (tid < OUT2) {
        device const float* W2 = params + offsets.w2;
        device const float* B2 = params + offsets.b2;

        float sum = B2[tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * W2[k * OUT2 + tid];
        }
        output[tid] = sum;  // No activation.
    }
}

/// Hardcoded-offset variant: eliminates the setBytes call for
/// FusedOffsets, saving 1 Obj-C message send (~15 us) per
/// inference.  Offsets are baked in as constexpr because the
/// kernel is already architecture-specific (fixed dimensions).
///
/// Buffer layout (only 3 bindings, no buffer(3)):
///   buffer(0): input [784]
///   buffer(1): params [109386] — full param buffer
///   buffer(2): output [10] — final logits
///
/// Offsets derived from MNIST layout: each layer's weights are
/// followed immediately by its biases in the flat param buffer.
///   Layer 0: W0[784×128] at 0, B0[128] at 100352
///   Layer 1: W1[128×64]  at 100480, B1[64] at 108672
///   Layer 2: W2[64×10]   at 108736, B2[10] at 109376
kernel void forward_fused_infer_3layer_v2(
    device const float* input    [[buffer(0)]],
    device const float* params   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    // Layer dimensions (comptime-known architecture).
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    // Hardcoded offsets into the flat param buffer.
    // Matches NetworkLayout for 784→128→64→10.
    constexpr uint W0_OFF = 0;
    constexpr uint B0_OFF = 100352;   // 784 * 128
    constexpr uint W1_OFF = 100480;   // 100352 + 128
    constexpr uint B1_OFF = 108672;   // 100480 + 128*64
    constexpr uint W2_OFF = 108736;   // 108672 + 64
    constexpr uint B2_OFF = 109376;   // 108736 + 64*10

    threadgroup float act0[OUT0];
    threadgroup float act1[OUT1];

    // ---- Layer 0: 784 → 128, ReLU ----
    {
        device const float* W0 = params + W0_OFF;
        device const float* B0 = params + B0_OFF;

        float sum = B0[tid];
        for (uint k = 0; k < IN0; k++) {
            sum += input[k] * W0[k * OUT0 + tid];
        }
        act0[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    if (tid < OUT1) {
        device const float* W1 = params + W1_OFF;
        device const float* B1 = params + B1_OFF;

        float sum = B1[tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * W1[k * OUT1 + tid];
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    if (tid < OUT2) {
        device const float* W2 = params + W2_OFF;
        device const float* B2 = params + B2_OFF;

        float sum = B2[tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * W2[k * OUT2 + tid];
        }
        output[tid] = sum;
    }
}

// ============================================================================
// Fused single-sample inference v3: writes completion flag
// ============================================================================

/// Same as forward_fused_infer_3layer_v2 but writes a
/// completion flag to buffer(3) when done.  The CPU can
/// spin-read this flag in shared memory instead of calling
/// waitUntilCompleted, avoiding the Mach kernel trap
/// (~100-150 us overhead for sub-10 us GPU work).
///
/// The flag is a single uint32 at buffer(3)[0].  Thread 0
/// writes 1 after all layers complete and a device memory
/// fence ensures visibility to the CPU on unified memory.
kernel void forward_fused_infer_3layer_v3(
    device const float* input    [[buffer(0)]],
    device const float* params   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    device atomic_uint* flag     [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    // Layer dimensions (comptime-known architecture).
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    // Hardcoded offsets into the flat param buffer.
    constexpr uint W0_OFF = 0;
    constexpr uint B0_OFF = 100352;   // 784 * 128
    constexpr uint W1_OFF = 100480;   // 100352 + 128
    constexpr uint B1_OFF = 108672;   // 100480 + 128*64
    constexpr uint W2_OFF = 108736;   // 108672 + 64
    constexpr uint B2_OFF = 109376;   // 108736 + 64*10

    threadgroup float act0[OUT0];
    threadgroup float act1[OUT1];

    // ---- Layer 0: 784 → 128, ReLU ----
    {
        device const float* W0 = params + W0_OFF;
        device const float* B0 = params + B0_OFF;

        float sum = B0[tid];
        for (uint k = 0; k < IN0; k++) {
            sum += input[k] * W0[k * OUT0 + tid];
        }
        act0[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    if (tid < OUT1) {
        device const float* W1 = params + W1_OFF;
        device const float* B1 = params + B1_OFF;

        float sum = B1[tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * W1[k * OUT1 + tid];
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    if (tid < OUT2) {
        device const float* W2 = params + W2_OFF;
        device const float* B2 = params + B2_OFF;

        float sum = B2[tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * W2[k * OUT2 + tid];
        }
        output[tid] = sum;
    }

    // Thread 0 signals completion after all work.
    // The mem_device barrier ensures all output writes
    // to device memory are visible before the flag.
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(
            flag, 1u, memory_order_relaxed);
    }
}

// ============================================================================
// Fused batched inference: 3-layer forward (784→128→64→10)
// ============================================================================

/// Batched variant of the fused 3-layer inference kernel.
/// Each threadgroup independently processes one sample from
/// the batch.  Dispatch with threadgroups = batch_size,
/// threadsPerThreadgroup = 128.
///
/// This reduces per-forward-pass dispatches from 3 (one per
/// layer) to 1, cutting Obj-C dispatch overhead by ~66%.
/// Input layout:  input[sample * 784 .. (sample+1) * 784]
/// Output layout: output[sample * 10 .. (sample+1) * 10]
kernel void forward_fused_infer_batched(
    device const float* input    [[buffer(0)]],
    device const float* params   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tgid   [[threadgroup_position_in_grid]])
{
    // Layer dimensions (comptime-known architecture).
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    // Hardcoded offsets into the flat param buffer.
    constexpr uint W0_OFF = 0;
    constexpr uint B0_OFF = 100352;
    constexpr uint W1_OFF = 100480;
    constexpr uint B1_OFF = 108672;
    constexpr uint W2_OFF = 108736;
    constexpr uint B2_OFF = 109376;

    // Per-threadgroup intermediates.
    threadgroup float act0[OUT0];
    threadgroup float act1[OUT1];

    // Offset input/output by sample index.
    device const float* sample_in = input + tgid * IN0;
    device float* sample_out = output + tgid * OUT2;

    // ---- Layer 0: 784 → 128, ReLU ----
    {
        device const float* W0 = params + W0_OFF;
        device const float* B0 = params + B0_OFF;

        float sum = B0[tid];
        for (uint k = 0; k < IN0; k++) {
            sum += sample_in[k] * W0[k * OUT0 + tid];
        }
        act0[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    if (tid < OUT1) {
        device const float* W1 = params + W1_OFF;
        device const float* B1 = params + B1_OFF;

        float sum = B1[tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * W1[k * OUT1 + tid];
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    if (tid < OUT2) {
        device const float* W2 = params + W2_OFF;
        device const float* B2 = params + B2_OFF;

        float sum = B2[tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * W2[k * OUT2 + tid];
        }
        sample_out[tid] = sum;
    }
}

// ============================================================================
// Float32 to Float16 parameter conversion
// ============================================================================

/// Convert float32 parameters to float16 for inference.
/// Halves memory bandwidth for weight reads. Accuracy loss
/// is negligible for inference (no gradient accumulation).
/// One thread per parameter element.
kernel void f32_to_f16(
    device const float* src  [[buffer(0)]],
    device half*        dst  [[buffer(1)]],
    uint                gid  [[thread_position_in_grid]])
{
    dst[gid] = half(src[gid]);
}

// ============================================================================
// Fused batched inference with half-precision weights
// ============================================================================

/// Half-precision weight variant of the fused batched kernel.
/// Reads weights as float16 (halving memory bandwidth) and
/// accumulates in float32 for numerical stability. Biases
/// remain float32 (tiny — 202 elements total).
///
/// Weight reads dominate bandwidth: layer 0 alone is 100K
/// weights. In float16, that's 200KB vs 400KB in float32,
/// halving L2 cache pressure for batched inference.
kernel void forward_fused_infer_batched_f16(
    device const float* input     [[buffer(0)]],
    device const half*  params_h  [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    device const float* bias_f32  [[buffer(3)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tgid   [[threadgroup_position_in_grid]])
{
    // Layer dimensions (comptime-known architecture).
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    // Weight offsets in the half-precision param buffer.
    // Only weights are stored as half; biases from bias_f32.
    constexpr uint W0_OFF = 0;
    constexpr uint W1_OFF = 100352;   // 784 * 128
    constexpr uint W2_OFF = 108544;   // W1_OFF + 128*64

    // Bias offsets in the float32 bias buffer.
    // bias_f32 layout: [B0(128), B1(64), B2(10)] = 202 floats.
    constexpr uint B0_OFF = 0;
    constexpr uint B1_OFF = 128;
    constexpr uint B2_OFF = 192;

    // Per-threadgroup intermediates.
    threadgroup float act0[OUT0];
    threadgroup float act1[OUT1];

    // Offset input/output by sample index.
    device const float* sample_in = input + tgid * IN0;
    device float* sample_out = output + tgid * OUT2;

    // ---- Layer 0: 784 → 128, ReLU ----
    {
        device const half* W0 = params_h + W0_OFF;
        float sum = bias_f32[B0_OFF + tid];
        for (uint k = 0; k < IN0; k++) {
            sum += sample_in[k] * float(W0[k * OUT0 + tid]);
        }
        act0[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    if (tid < OUT1) {
        device const half* W1 = params_h + W1_OFF;
        float sum = bias_f32[B1_OFF + tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * float(W1[k * OUT1 + tid]);
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    if (tid < OUT2) {
        device const half* W2 = params_h + W2_OFF;
        float sum = bias_f32[B2_OFF + tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * float(W2[k * OUT2 + tid]);
        }
        sample_out[tid] = sum;
    }
}

// ============================================================================
// Fused single-sample inference with half-precision weights
// ============================================================================

/// Half-precision weight variant for single-sample inference.
/// Same structure as forward_fused_infer_3layer_v2 but reads
/// weights as float16. Reduces memory bandwidth by 2x.
kernel void forward_fused_infer_single_f16(
    device const float* input     [[buffer(0)]],
    device const half*  params_h  [[buffer(1)]],
    device float*       output    [[buffer(2)]],
    device const float* bias_f32  [[buffer(3)]],
    uint tid                      [[thread_index_in_threadgroup]])
{
    constexpr uint IN0  = 784;
    constexpr uint OUT0 = 128;
    constexpr uint OUT1 = 64;
    constexpr uint OUT2 = 10;

    constexpr uint W0_OFF = 0;
    constexpr uint W1_OFF = 100352;
    constexpr uint W2_OFF = 108544;

    constexpr uint B0_OFF = 0;
    constexpr uint B1_OFF = 128;
    constexpr uint B2_OFF = 192;

    threadgroup float act0[OUT0];
    threadgroup float act1[OUT1];

    // ---- Layer 0: 784 → 128, ReLU ----
    {
        device const half* W0 = params_h + W0_OFF;
        float sum = bias_f32[B0_OFF + tid];
        for (uint k = 0; k < IN0; k++) {
            sum += input[k] * float(W0[k * OUT0 + tid]);
        }
        act0[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 1: 128 → 64, ReLU ----
    if (tid < OUT1) {
        device const half* W1 = params_h + W1_OFF;
        float sum = bias_f32[B1_OFF + tid];
        for (uint k = 0; k < OUT0; k++) {
            sum += act0[k] * float(W1[k * OUT1 + tid]);
        }
        act1[tid] = max(0.0f, sum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Layer 2: 64 → 10, none ----
    if (tid < OUT2) {
        device const half* W2 = params_h + W2_OFF;
        float sum = bias_f32[B2_OFF + tid];
        for (uint k = 0; k < OUT1; k++) {
            sum += act1[k] * float(W2[k * OUT2 + tid]);
        }
        output[tid] = sum;
    }
}

// ============================================================================
// Adam parameter update
// ============================================================================

/// Adam optimiser update (Kingma & Ba, 2014).
///
///   m[i]  = beta1 * m[i] + (1 - beta1) * grad[i]
///   v[i]  = beta2 * v[i] + (1 - beta2) * grad[i]^2
///   m_hat = m[i] / (1 - beta1^t)
///   v_hat = v[i] / (1 - beta2^t)
///   params[i] -= lr * m_hat / (sqrt(v_hat) + epsilon)
///
/// Bias correction terms (1 - beta^t) are precomputed on the
/// CPU and passed as `correction1` and `correction2` to avoid
/// a pow() in every thread.
kernel void adam_update(
    device float*       params      [[buffer(0)]],
    device const float* grads       [[buffer(1)]],
    device float*       m           [[buffer(2)]],  // first moment
    device float*       v           [[buffer(3)]],  // second moment
    constant float& lr              [[buffer(4)]],
    constant float& beta1           [[buffer(5)]],
    constant float& beta2           [[buffer(6)]],
    constant float& epsilon         [[buffer(7)]],
    constant float& correction1     [[buffer(8)]],  // 1 - beta1^t
    constant float& correction2     [[buffer(9)]],  // 1 - beta2^t
    uint                gid         [[thread_position_in_grid]])
{
    float g = grads[gid];

    // Update biased first and second moment estimates.
    float mi = beta1 * m[gid] + (1.0f - beta1) * g;
    float vi = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = mi;
    v[gid] = vi;

    // Bias-corrected estimates.
    float m_hat = mi / correction1;
    float v_hat = vi / correction2;

    // Parameter update.
    params[gid] -= lr * m_hat / (sqrt(v_hat) + epsilon);
}

// ============================================================================
// Float32 to 1-bit quantization (Q1_0_g128 format)
// ============================================================================

/// Dimensions for the f32_to_1bit quantization kernel.
struct Q1Dims {
    uint total_weights;  // Total number of f32 weight elements.
    uint group_size;     // Weights per scale group (128).
};

/// Quantize f32 weights to 1-bit Q1_0_g128 format.
/// Each group of 128 weights shares one f16 scale.
/// bit=1 → +scale, bit=0 → −scale.
/// Scale = mean(abs(weights)) for the group.
///
/// Output layout: packed_bits buffer + separate scales buffer.
/// packed_bytes = ceil(total_weights / 8)
///
/// Dispatch: one thread per group (ceil(total_weights / 128) threads).
kernel void f32_to_1bit(
    device const float* weights      [[buffer(0)]],
    device uint8_t*     packed_bits  [[buffer(1)]],
    device half*        scales       [[buffer(2)]],
    constant Q1Dims&    dims         [[buffer(3)]],
    uint                gid          [[thread_position_in_grid]])
{
    const uint group_idx = gid;
    const uint group_start = group_idx * dims.group_size;

    // Bounds check: last group may be partial.
    if (group_start >= dims.total_weights) return;

    const uint group_end = min(
        group_start + dims.group_size,
        dims.total_weights
    );
    const uint count = group_end - group_start;

    // Compute mean absolute value for the scale.
    float abs_sum = 0.0f;
    for (uint i = 0; i < count; i++) {
        abs_sum += abs(weights[group_start + i]);
    }
    float scale = abs_sum / float(count);
    scales[group_idx] = half(scale);

    // Pack weights into bits: bit=1 if weight >= 0, bit=0 if weight < 0.
    // LSB-first within each byte.
    const uint byte_start = group_start / 8;
    const uint full_bytes = count / 8;
    const uint remainder = count % 8;

    for (uint i = 0; i < full_bytes; i++) {
        uint8_t byte_val = 0;
        const uint w_base = group_start + i * 8;
        byte_val |= (weights[w_base + 0] >= 0.0f) ? 0x01 : 0;
        byte_val |= (weights[w_base + 1] >= 0.0f) ? 0x02 : 0;
        byte_val |= (weights[w_base + 2] >= 0.0f) ? 0x04 : 0;
        byte_val |= (weights[w_base + 3] >= 0.0f) ? 0x08 : 0;
        byte_val |= (weights[w_base + 4] >= 0.0f) ? 0x10 : 0;
        byte_val |= (weights[w_base + 5] >= 0.0f) ? 0x20 : 0;
        byte_val |= (weights[w_base + 6] >= 0.0f) ? 0x40 : 0;
        byte_val |= (weights[w_base + 7] >= 0.0f) ? 0x80 : 0;
        packed_bits[byte_start + i] = byte_val;
    }

    // Handle the remainder bits in the last partial byte.
    if (remainder > 0) {
        uint8_t byte_val = 0;
        const uint w_base = group_start + full_bytes * 8;
        for (uint b = 0; b < remainder; b++) {
            if (weights[w_base + b] >= 0.0f) {
                byte_val |= (1u << b);
            }
        }
        packed_bits[byte_start + full_bytes] = byte_val;
    }
}

// ============================================================================
// 1-bit matrix-vector multiply (qmv) — Q1_0_g128 format
// ============================================================================

/// Dimensions for qmv kernel.
struct QMVDims {
    uint M;           // Output rows (weight matrix rows).
    uint K;           // Input dimension (weight matrix columns).
    uint group_size;  // Weights per scale group (128).
};

/// 1-bit matrix-vector multiply: output[row] = dot(W_1bit[row], input).
///
/// W is [M × K] in Q1_0_g128 packed format:
///   - Packed bits: ceil(M * K / 8) bytes, row-major.
///   - Scales: ceil(M * K / group_size) f16 values.
///
/// Each output row is computed by one simdgroup (32 threads).
/// Two simdgroups per threadgroup → two rows per threadgroup.
///
/// Uses the identity: scale * (2 * set_accum - group_sum)
/// where set_accum = sum of input[col] where bit=1,
/// and group_sum = sum of all input[col] in the group.
/// This avoids per-element branching on the sign.
///
/// Dispatch: threadgroups = ceil(M / 2), threads_per_group = 64.
kernel void qmv(
    device const uint8_t* packed_bits  [[buffer(0)]],
    device const half*    scales       [[buffer(1)]],
    device const float*   input        [[buffer(2)]],
    device float*         output       [[buffer(3)]],
    constant QMVDims&     dims         [[buffer(4)]],
    uint                  tgid         [[threadgroup_position_in_grid]],
    uint                  tid_in_tg    [[thread_index_in_threadgroup]])
{
    // Two simdgroups per threadgroup: threads 0–31 handle one row,
    // threads 32–63 handle the next.
    const uint simdgroup_idx = tid_in_tg / 32;
    const uint lane = tid_in_tg % 32;
    const uint row = tgid * 2 + simdgroup_idx;

    // Bounds check: last threadgroup may have only one valid row.
    if (row >= dims.M) return;

    const uint K = dims.K;
    const uint group_size = dims.group_size;
    const uint groups_per_row = (K + group_size - 1) / group_size;

    // Each of 32 lanes processes K/32 columns.
    // K must be a multiple of group_size (128), and 128/32 = 4,
    // so each lane processes complete groups.
    const uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;

    // Byte offset for this row's packed bits.
    const uint row_bit_offset = row * (K / 8);
    // Scale offset for this row.
    const uint row_scale_offset = row * groups_per_row;

    float accum = 0.0f;

    // Track group boundaries for per-group scale application.
    uint cur_group = col_start / group_size;
    float set_accum = 0.0f;   // Sum of input[col] where bit=1.
    float group_sum = 0.0f;   // Sum of all input[col] in group.

    // Iterate over the columns assigned to this lane.
    for (uint c = 0; c < cols_per_lane; c++) {
        const uint col = col_start + c;
        const uint group_idx = col / group_size;

        // When we cross a group boundary, flush the accumulator.
        if (group_idx != cur_group) {
            const float scale = float(
                scales[row_scale_offset + cur_group]
            );
            accum += scale * (2.0f * set_accum - group_sum);
            set_accum = 0.0f;
            group_sum = 0.0f;
            cur_group = group_idx;
        }

        // Which byte and bit within the packed row.
        const uint byte_idx = row_bit_offset + col / 8;
        const uint bit_pos = col % 8;
        const bool bit_set = (packed_bits[byte_idx] >> bit_pos) & 1;

        // Branchless conditional accumulation via select().
        const float x_val = input[col];
        group_sum += x_val;
        set_accum += select(0.0f, x_val, bit_set);
    }

    // Flush the final group.
    {
        const float scale = float(
            scales[row_scale_offset + cur_group]
        );
        accum += scale * (2.0f * set_accum - group_sum);
    }

    // Reduce across the 32 lanes in the simdgroup.
    accum = simd_sum(accum);

    // Lane 0 writes the final result.
    if (lane == 0) {
        output[row] = accum;
    }
}

// ============================================================================
// 1-bit matrix-vector multiply (qmv_fast) — aligned, byte-unrolled
// ============================================================================

/// Fast qmv for aligned dimensions: K % 128 == 0.
///
/// Three key optimisations over qmv:
///   1. Loads the input vector into threadgroup shared memory
///      so all simdgroups share one copy (4 rows per tgroup).
///   2. Processes 8 columns per inner-loop iteration (one
///      packed byte) instead of 1 bit, cutting loop count 8×.
///   3. Removes the per-column group-boundary check — when
///      K % 128 == 0 and cols_per_lane <= 128, no lane ever
///      crosses a group boundary.
///
/// Dispatch: threadgroups = ceil(M / 4), threads = 128.
kernel void qmv_fast(
    device const uint8_t* packed_bits  [[buffer(0)]],
    device const half*    scales       [[buffer(1)]],
    device const float*   input        [[buffer(2)]],
    device float*         output       [[buffer(3)]],
    constant QMVDims&     dims         [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    const uint K = dims.K;
    const uint M = dims.M;
    const uint group_size = dims.group_size;

    // 8 simdgroups of 32 lanes → 8 rows per threadgroup.
    // More rows per threadgroup shares the input vector
    // load across more output rows and increases GPU
    // occupancy (256 vs 128 threads per threadgroup).
    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 8 + simdgroup_idx;

    // Cooperatively load input vector into shared memory.
    // 256 threads load up to K floats (ceil(K/256) iters).
    threadgroup float shared_input[6144];
    for (uint i = tid; i < K; i += 256) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= M) return;

    const uint groups_per_row = K / group_size;
    const uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    const uint bytes_per_lane = cols_per_lane / 8;

    // Byte offset for this row's packed bits.
    const uint row_byte_base = row * (K / 8) + col_start / 8;
    // Scale offset for this row.
    const uint row_scale_offset = row * groups_per_row;
    // Group index for this lane (constant — no crossing).
    const uint grp = col_start / group_size;

    float set_accum = 0.0f;
    float group_sum = 0.0f;

    // Process 8 columns (one packed byte) per iteration.
    for (uint b = 0; b < bytes_per_lane; b++) {
        const uint byte_val = packed_bits[row_byte_base + b];
        const uint base_col = col_start + b * 8;

        // Unrolled: 8 columns from one byte.
        float x0 = shared_input[base_col + 0];
        float x1 = shared_input[base_col + 1];
        float x2 = shared_input[base_col + 2];
        float x3 = shared_input[base_col + 3];
        float x4 = shared_input[base_col + 4];
        float x5 = shared_input[base_col + 5];
        float x6 = shared_input[base_col + 6];
        float x7 = shared_input[base_col + 7];

        group_sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        // Branchless conditional sum via select().
        set_accum += select(0.0f, x0, bool(byte_val & 1));
        set_accum += select(0.0f, x1, bool(byte_val & 2));
        set_accum += select(0.0f, x2, bool(byte_val & 4));
        set_accum += select(0.0f, x3, bool(byte_val & 8));
        set_accum += select(0.0f, x4, bool(byte_val & 16));
        set_accum += select(0.0f, x5, bool(byte_val & 32));
        set_accum += select(0.0f, x6, bool(byte_val & 64));
        set_accum += select(0.0f, x7, bool(byte_val & 128));
    }

    // Apply scale for this lane's group.
    const float scale = float(
        scales[row_scale_offset + grp]
    );
    float accum = scale * (2.0f * set_accum - group_sum);

    // Reduce across the 32 lanes in the simdgroup.
    accum = simd_sum(accum);

    // Lane 0 writes the final result.
    if (lane == 0) {
        output[row] = accum;
    }
}

// ============================================================================
// qmv_fast_sm — smaller shared memory for better occupancy
// ============================================================================

/// qmv_fast variant with 2048-float shared memory (8 KB) for
/// K <= 2048.  The 3× smaller footprint (vs 24 KB in qmv_fast)
/// lets more threadgroups co-execute on one execution unit,
/// hiding memory latency through higher occupancy.  Otherwise
/// identical logic to qmv_fast.
///
/// Dispatch: threadgroups = ceil(M / 4), threads = 128.
kernel void qmv_fast_sm(
    device const uint8_t* packed_bits  [[buffer(0)]],
    device const half*    scales       [[buffer(1)]],
    device const float*   input        [[buffer(2)]],
    device float*         output       [[buffer(3)]],
    constant QMVDims&     dims         [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    const uint K = dims.K;
    const uint M = dims.M;
    const uint group_size = dims.group_size;

    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 4 + simdgroup_idx;

    // 8 KB shared memory — 3× smaller than qmv_fast's 24 KB.
    threadgroup float shared_input[2048];
    for (uint i = tid; i < K; i += 128) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= M) return;

    const uint groups_per_row = K / group_size;
    const uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    const uint bytes_per_lane = cols_per_lane / 8;
    const uint row_byte_base =
        row * (K / 8) + col_start / 8;
    const uint row_scale_offset = row * groups_per_row;
    const uint grp = col_start / group_size;

    float set_accum = 0.0f;
    float group_sum = 0.0f;

    for (uint b = 0; b < bytes_per_lane; b++) {
        const uint byte_val = packed_bits[row_byte_base + b];
        const uint base_col = col_start + b * 8;

        float x0 = shared_input[base_col + 0];
        float x1 = shared_input[base_col + 1];
        float x2 = shared_input[base_col + 2];
        float x3 = shared_input[base_col + 3];
        float x4 = shared_input[base_col + 4];
        float x5 = shared_input[base_col + 5];
        float x6 = shared_input[base_col + 6];
        float x7 = shared_input[base_col + 7];

        group_sum += x0 + x1 + x2 + x3
                   + x4 + x5 + x6 + x7;

        set_accum += select(0.0f, x0, bool(byte_val & 1));
        set_accum += select(0.0f, x1, bool(byte_val & 2));
        set_accum += select(0.0f, x2, bool(byte_val & 4));
        set_accum += select(0.0f, x3, bool(byte_val & 8));
        set_accum += select(
            0.0f, x4, bool(byte_val & 16)
        );
        set_accum += select(
            0.0f, x5, bool(byte_val & 32)
        );
        set_accum += select(
            0.0f, x6, bool(byte_val & 64)
        );
        set_accum += select(
            0.0f, x7, bool(byte_val & 128)
        );
    }

    const float scale = float(
        scales[row_scale_offset + grp]
    );
    float accum = scale * (2.0f * set_accum - group_sum);
    accum = simd_sum(accum);

    if (lane == 0) {
        output[row] = accum;
    }
}

// ============================================================================
// qmv_fast_multigroup — handles lanes spanning multiple groups
// ============================================================================

/// Multi-group variant of qmv_fast for K > 32*group_size where
/// each lane's columns span more than one scale group (e.g.
/// K=6144, group_size=128: cols_per_lane=192 > 128).
///
/// Same structure as qmv_fast but flushes the accumulator at
/// group boundaries within the byte loop.  Group transitions
/// occur every group_size/8 bytes, checked by a counter.
///
/// Dispatch: threadgroups = ceil(M / 4), threads = 128.
kernel void qmv_fast_multigroup(
    device const uint8_t* packed_bits  [[buffer(0)]],
    device const half*    scales       [[buffer(1)]],
    device const float*   input        [[buffer(2)]],
    device float*         output       [[buffer(3)]],
    constant QMVDims&     dims         [[buffer(4)]],
    uint tgid    [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    const uint K = dims.K;
    const uint M = dims.M;
    const uint group_size = dims.group_size;

    // 8 simdgroups of 32 lanes → 8 rows per threadgroup.
    // Matches qmv_fast: more rows per threadgroup shares
    // the input vector load and improves occupancy.
    const uint simdgroup_idx = tid / 32;
    const uint lane = tid % 32;
    const uint row = tgid * 8 + simdgroup_idx;

    // Cooperatively load input vector into shared memory.
    threadgroup float shared_input[6144];
    for (uint i = tid; i < K; i += 256) {
        shared_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= M) return;

    const uint groups_per_row = K / group_size;
    const uint cols_per_lane = K / 32;
    const uint col_start = lane * cols_per_lane;
    const uint bytes_per_lane = cols_per_lane / 8;
    const uint row_byte_base =
        row * (K / 8) + col_start / 8;
    const uint row_scale_offset = row * groups_per_row;

    // Bytes per group (group_size always multiple of 8).
    const uint bytes_per_group = group_size / 8;

    float accum = 0.0f;
    float set_accum = 0.0f;
    float group_sum = 0.0f;
    uint cur_group = col_start / group_size;
    uint byte_in_group = (col_start % group_size) / 8;

    for (uint b = 0; b < bytes_per_lane; b++) {
        const uint byte_val = packed_bits[row_byte_base + b];
        const uint base_col = col_start + b * 8;

        float x0 = shared_input[base_col + 0];
        float x1 = shared_input[base_col + 1];
        float x2 = shared_input[base_col + 2];
        float x3 = shared_input[base_col + 3];
        float x4 = shared_input[base_col + 4];
        float x5 = shared_input[base_col + 5];
        float x6 = shared_input[base_col + 6];
        float x7 = shared_input[base_col + 7];

        group_sum += x0 + x1 + x2 + x3
                   + x4 + x5 + x6 + x7;

        set_accum += select(0.0f, x0, bool(byte_val & 1));
        set_accum += select(0.0f, x1, bool(byte_val & 2));
        set_accum += select(0.0f, x2, bool(byte_val & 4));
        set_accum += select(0.0f, x3, bool(byte_val & 8));
        set_accum += select(0.0f, x4, bool(byte_val & 16));
        set_accum += select(0.0f, x5, bool(byte_val & 32));
        set_accum += select(0.0f, x6, bool(byte_val & 64));
        set_accum += select(0.0f, x7, bool(byte_val & 128));

        // Check group boundary (every group_size/8 bytes).
        byte_in_group++;
        if (byte_in_group == bytes_per_group) {
            const float scale = float(
                scales[row_scale_offset + cur_group]
            );
            accum += scale * (
                2.0f * set_accum - group_sum
            );
            set_accum = 0.0f;
            group_sum = 0.0f;
            cur_group++;
            byte_in_group = 0;
        }
    }

    // Flush remaining partial group.
    if (byte_in_group > 0) {
        const float scale = float(
            scales[row_scale_offset + cur_group]
        );
        accum += scale * (
            2.0f * set_accum - group_sum
        );
    }

    accum = simd_sum(accum);

    if (lane == 0) {
        output[row] = accum;
    }
}

// ============================================================================
// 1-bit matrix-matrix multiply (qmm) — Q1_0_g128 format
// ============================================================================

/// Dimensions for qmm kernel.
struct QMMDims {
    uint M;           // Rows of A / rows of output.
    uint N;           // Cols of output (= cols of W).
    uint K;           // Cols of A / rows of W.
    uint group_size;  // Weights per scale group (128).
};

/// 1-bit tiled matrix multiply: output[m][n] = sum_k(A[m][k] * W_dequant[k][n]).
///
/// A is [M × K] in f32 (activations).
/// W is [K × N] in Q1_0_g128 packed format (row-major, matching the
/// existing matmul convention where weights are [in_size × out_size]).
/// Output is [M × N] in f32.
///
/// Tiling: 16×16 output tiles, K tiles of 128 (= group_size).
/// Dequantizes W tiles into threadgroup memory as f32 (not f16 — preserves
/// accumulation precision since weights are already maximally quantized),
/// then performs a standard tiled matmul.
///
/// Threadgroup memory: 16×128×4 bytes × 2 tiles = 16 KB, within Metal's
/// 32 KB limit.
///
/// Dispatch: threadgroups = (ceil(N/16), ceil(M/16)), threads = 16×16.
kernel void qmm(
    device const float*   A            [[buffer(0)]],
    device const uint8_t* packed_bits  [[buffer(1)]],
    device const half*    scales       [[buffer(2)]],
    device float*         output       [[buffer(3)]],
    constant QMMDims&     dims         [[buffer(4)]],
    uint2                 tgid         [[threadgroup_position_in_grid]],
    uint2                 tid          [[thread_position_in_threadgroup]])
{
    // Tile size matches existing matmul_tiled.
    const uint QMM_TS = 16;
    const uint K_TILE = 128; // = group_size

    // Shared memory tiles.
    threadgroup float tile_A[16][128];
    threadgroup float tile_W[16][128];

    const uint M = dims.M;
    const uint N = dims.N;
    const uint K = dims.K;
    const uint group_size = dims.group_size;

    // Output tile position.
    const uint row_base = tgid.y * QMM_TS; // M dimension
    const uint col_base = tgid.x * QMM_TS; // N dimension

    const uint local_row = tid.y;
    const uint local_col = tid.x;

    const uint global_row = row_base + local_row;
    const uint global_col = col_base + local_col;

    float accum = 0.0f;

    // Step through K in chunks of K_TILE (128).
    const uint num_k_tiles = (K + K_TILE - 1) / K_TILE;
    for (uint kt = 0; kt < num_k_tiles; kt++) {
        const uint k_base = kt * K_TILE;

        // Cooperative load of A tile: each thread loads multiple elements.
        // 16 threads × 16 threads = 256 threads loading 16 × 128 = 2048
        // elements.  Each thread loads 2048 / 256 = 8 elements.
        for (uint e = 0; e < 8; e++) {
            const uint flat_idx =
                (local_row * QMM_TS + local_col) * 8 + e;
            const uint tile_r = flat_idx / K_TILE;
            const uint tile_c = flat_idx % K_TILE;
            const uint a_row = row_base + tile_r;
            const uint a_col = k_base + tile_c;
            if (a_row < M && a_col < K) {
                tile_A[tile_r][tile_c] = A[a_row * K + a_col];
            } else {
                tile_A[tile_r][tile_c] = 0.0f;
            }
        }

        // Cooperative load + dequantize of W tile.
        // W is [K × N] row-major packed (matching the existing matmul
        // convention).  We load a 128×16 (K×N) region, then store it
        // as tile_W[n_local][k_local] so the inner loop can index
        // tile_W[local_col][k] for the dot product.
        for (uint e = 0; e < 8; e++) {
            const uint flat_idx =
                (local_row * QMM_TS + local_col) * 8 + e;
            const uint tile_n = flat_idx / K_TILE; // N offset (0..15)
            const uint tile_k = flat_idx % K_TILE; // K offset (0..127)
            const uint w_k = k_base + tile_k;      // K dimension
            const uint w_n = col_base + tile_n;     // N dimension

            if (w_k < K && w_n < N) {
                // Locate the packed bit in [K × N] storage.
                const uint flat_bit = w_k * N + w_n;
                const uint byte_idx = flat_bit / 8;
                const uint bit_pos = flat_bit % 8;
                const bool bit_set =
                    (packed_bits[byte_idx] >> bit_pos) & 1;

                // Scale group from flat position in packed data.
                const uint g_idx = flat_bit / group_size;
                const float scale = float(scales[g_idx]);

                tile_W[tile_n][tile_k] =
                    select(-scale, +scale, bit_set);
            } else {
                tile_W[tile_n][tile_k] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile matmul: accumulate dot product for this K tile.
        // output[global_row][global_col] +=
        //   sum over k of tile_A[local_row][k] * tile_W[local_col][k]
        // tile_W[n_local][k] holds W[k_base+k][col_base+n_local].
        if (global_row < M && global_col < N) {
            for (uint k = 0; k < K_TILE; k++) {
                accum += tile_A[local_row][k]
                       * tile_W[local_col][k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result.
    if (global_row < M && global_col < N) {
        output[global_row * N + global_col] = accum;
    }
}
