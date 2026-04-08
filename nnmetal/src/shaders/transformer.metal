#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Transformer primitives — Step 1 of the Bonsai implementation plan.
//
// Every kernel here operates on f32 activations.  Weights are either
// f16 (norm scales) or 1-bit packed (embeddings).  The KV cache
// stores f16 values; conversions happen at explicit boundaries
// (kv_cache_update writes f32→f16, gqa_attention reads f16→f32).
//
// Kernel list:
//   rms_norm             — RMSNorm with f16 scale vector
//   silu                 — x * sigmoid(x), elementwise
//   silu_elementwise_mul — fused silu(gate) ⊙ up for SwiGLU
//   rope                 — rotary position embeddings
//   kv_cache_update      — append f32 K/V to f16 cache
//   gqa_attention        — grouped query attention (decode, M=1)
//   embedding_lookup     — gather + dequantize from 1-bit table
//   residual_add         — in-place a[i] += b[i]
// ============================================================================

// ============================================================================
// RMSNorm
// ============================================================================

/// Dimensions for rms_norm kernel.
struct RMSNormDims {
    uint  hidden_size;  // Elements per row.
    uint  num_tokens;   // Number of rows (tokens or heads).
    float eps;          // Epsilon for numerical stability (1e-6).
};

/// RMSNorm: output[i] = input[i] * rsqrt(mean(input²) + eps) * scale[i].
///
/// Input is [num_tokens × hidden_size] in f32.
/// Scale is [hidden_size] in f16 (shared across all tokens).
/// Output is [num_tokens × hidden_size] in f32.
///
/// Dispatch: threadgroups = num_tokens, threads_per_group = 256.
/// Each threadgroup normalizes one row using a cooperative
/// reduction for the sum of squares.
///
/// Also used for QK-norms: set hidden_size = head_dim and
/// num_tokens = num_heads.  The same scale vector is applied
/// to every head.
kernel void rms_norm(
    device const float*    input  [[buffer(0)]],
    device const half*     scale  [[buffer(1)]],
    device float*          output [[buffer(2)]],
    constant RMSNormDims&  dims   [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint token = tgid;
    const uint hidden_size = dims.hidden_size;

    // Out-of-bounds threadgroups exit early.
    if (token >= dims.num_tokens) return;

    device const float* x = input + token * hidden_size;
    device float* out = output + token * hidden_size;

    // Step 1: Each thread accumulates a partial sum of squares.
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_size; i += THREADS) {
        const float val = x[i];
        partial_sum_sq += val * val;
    }

    // Step 2: Threadgroup reduction for total sum of squares.
    threadgroup float shared[256];
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rsqrt(mean(x²) + eps).  Every thread reads the same value.
    const float rms = rsqrt(
        shared[0] / float(hidden_size) + dims.eps
    );

    // Step 3: Normalize and scale.
    for (uint i = tid; i < hidden_size; i += THREADS) {
        out[i] = x[i] * rms * float(scale[i]);
    }
}

// ============================================================================
// SiLU (Sigmoid Linear Unit)
// ============================================================================

/// SiLU: output[i] = input[i] * sigmoid(input[i]).
///
/// Elementwise, one thread per element.
/// Dispatch: 1D, count = total elements.
kernel void silu(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const float x = input[gid];
    const float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sigmoid_x;
}

/// Fused SiLU + elementwise multiply for SwiGLU:
///   output[i] = silu(gate[i]) * up[i]
///             = gate[i] * sigmoid(gate[i]) * up[i]
///
/// Combines the SiLU activation on the gate path with the
/// elementwise multiply against the up path in a single pass,
/// halving memory traffic compared to two separate kernels.
///
/// Dispatch: 1D, count = total elements.
kernel void silu_elementwise_mul(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const float g = gate[gid];
    const float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[gid] = g * sigmoid_g * up[gid];
}

// ============================================================================
// RoPE (Rotary Position Embeddings)
// ============================================================================

/// Dimensions for rope kernel.
struct RoPEDims {
    uint  num_heads;    // Number of heads to rotate.
    uint  head_dim;     // Elements per head (must be even).
    uint  position;     // Sequence position for this token.
    float rope_theta;   // Base frequency (e.g. 1000000.0).
};

/// Apply rotary position embeddings in-place.
///
/// For each head h and pair index p (0 ≤ p < head_dim/2):
///   angle = position / (rope_theta ^ (2p / head_dim))
///   x0' = x0 * cos(angle) − x1 * sin(angle)
///   x1' = x0 * sin(angle) + x1 * cos(angle)
///
/// Input/output is [num_heads × head_dim] in f32.  The kernel
/// modifies the buffer in-place (input == output is allowed and
/// expected).
///
/// Used for both Q and K after projection.  The caller dispatches
/// twice: once with num_heads = num_query_heads, once with
/// num_heads = num_kv_heads.
///
/// Dispatch: 1D, count = num_heads * (head_dim / 2).
kernel void rope(
    device float*        data  [[buffer(0)]],
    constant RoPEDims&   dims  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = dims.head_dim / 2;
    const uint total_pairs = dims.num_heads * half_dim;

    if (gid >= total_pairs) return;

    const uint head = gid / half_dim;
    const uint pair = gid % half_dim;

    // Frequency for this pair index.
    // theta_i = 1 / (rope_theta ^ (2 * pair / head_dim))
    const float exponent =
        float(2 * pair) / float(dims.head_dim);
    // exp2 is a hardware instruction; pow is software-emulated.
    const float theta_i =
        exp2(-exponent * log2(dims.rope_theta));
    const float angle = float(dims.position) * theta_i;

    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    // Element indices within the flat buffer.
    const uint base = head * dims.head_dim + pair * 2;
    const float x0 = data[base];
    const float x1 = data[base + 1];

    data[base]     = x0 * cos_val - x1 * sin_val;
    data[base + 1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================================
// KV Cache Update
// ============================================================================

/// Dimensions for kv_cache_update kernel.
struct KVUpdateDims {
    uint num_kv_heads;         // Number of KV heads.
    uint head_dim;             // Elements per head.
    uint position;             // Sequence position to write at.
    uint max_context_length;   // Cache capacity (time dimension).
};

/// Append one token's K and V projections to the f16 KV cache.
///
/// K/V projections are [num_kv_heads × head_dim] in f32.
/// K/V caches are [num_kv_heads × max_context_length × head_dim]
/// in f16.
///
/// Each thread writes one element to both K and V caches at the
/// given position, performing the f32 → f16 conversion.
///
/// Dispatch: 1D, count = num_kv_heads * head_dim.
kernel void kv_cache_update(
    device const float*    k_proj  [[buffer(0)]],
    device const float*    v_proj  [[buffer(1)]],
    device half*           k_cache [[buffer(2)]],
    device half*           v_cache [[buffer(3)]],
    constant KVUpdateDims& dims    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = dims.num_kv_heads * dims.head_dim;

    if (gid >= total) return;

    const uint head = gid / dims.head_dim;
    const uint d = gid % dims.head_dim;
    const uint pos = dims.position;

    // Cache layout: [head][time][d].
    const uint cache_stride =
        dims.max_context_length * dims.head_dim;
    const uint cache_idx =
        head * cache_stride + pos * dims.head_dim + d;

    // f32 → f16 truncation happens here.
    k_cache[cache_idx] = half(k_proj[gid]);
    v_cache[cache_idx] = half(v_proj[gid]);
}

// ============================================================================
// GQA Attention (decode, M=1)
// ============================================================================

/// Dimensions for gqa_attention kernel.
struct GQADims {
    uint num_query_heads;      // Total query heads.
    uint num_kv_heads;         // Total KV heads.
    uint head_dim;             // Elements per head.
    uint seq_len;              // Valid KV entries (current_pos + 1).
    uint max_context_length;   // Cache capacity (time dimension).
    uint heads_per_kv_group;   // = num_query_heads / num_kv_heads.
};

/// Grouped query attention with causal masking (decode path, M=1).
///
/// For each query head h:
///   kv_head = h / heads_per_kv_group
///   scores[t] = dot(Q[h], K[kv_head][t]) / sqrt(head_dim)
///   attn_weights = softmax(scores[0..seq_len])
///   output[h][d] = sum_t(attn_weights[t] * V[kv_head][t][d])
///
/// Q is [num_query_heads × head_dim] in f32.
/// K cache is [num_kv_heads × max_ctx × head_dim] in f16.
/// V cache is [num_kv_heads × max_ctx × head_dim] in f16.
/// Output is [num_query_heads × head_dim] in f32.
/// Scratch is [num_query_heads × max_ctx] in f32 (attention scores).
///
/// Dispatch: threadgroups = num_query_heads, threads = 256.
/// Each threadgroup computes attention for one query head.
///
/// The scratch buffer stores attention weights in device memory,
/// which allows arbitrary sequence lengths without being limited
/// by threadgroup memory (32 KB).  A threadgroup_barrier with
/// mem_device ensures visibility within the threadgroup.
kernel void gqa_attention(
    device const float*  Q       [[buffer(0)]],
    device const half*   k_cache [[buffer(1)]],
    device const half*   v_cache [[buffer(2)]],
    device float*        output  [[buffer(3)]],
    device float*        scratch [[buffer(4)]],
    constant GQADims&    dims    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint SIMD_SIZE = 32;
    const uint NUM_SIMD_GROUPS = THREADS / SIMD_SIZE;
    const uint head = tgid;

    if (head >= dims.num_query_heads) return;

    const uint kv_head = head / dims.heads_per_kv_group;
    const uint head_dim = dims.head_dim;
    const uint seq_len = dims.seq_len;
    const uint max_ctx = dims.max_context_length;

    const uint simd_group = tid / SIMD_SIZE;
    const uint simd_lane = tid % SIMD_SIZE;

    // Pointers for this head.
    device const float* q =
        Q + head * head_dim;
    const uint kv_stride = max_ctx * head_dim;
    device const half* k =
        k_cache + kv_head * kv_stride;
    device const half* v =
        v_cache + kv_head * kv_stride;
    device float* scores =
        scratch + head * max_ctx;

    // Scaling factor: 1 / sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(head_dim));

    // ── Load Q into threadgroup memory ───────────────────────
    // Q is only head_dim floats (512 bytes for head_dim=128).
    // Cooperative load avoids re-reading from device memory on
    // every timestep in the dot product loop.  We reuse this
    // array for softmax reductions later (Q is no longer needed
    // after Step 1).
    threadgroup float shared_buf[256];

    if (tid < head_dim) {
        shared_buf[tid] = q[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 1: Compute attention scores ─────────────────────
    // Each thread processes a subset of time steps.
    // Read Q from threadgroup memory instead of device memory.
    for (uint t = tid; t < seq_len; t += THREADS) {
        float dot_product = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot_product +=
                shared_buf[d]
                * float(k[t * head_dim + d]);
        }
        scores[t] = dot_product * inv_sqrt_d;
    }

    // Ensure all scores are written before reading them.
    threadgroup_barrier(mem_flags::mem_device);

    // ── Step 2: Softmax — find max (numerical stability) ─────
    // Use simd_max() for intra-SIMD-group reduction (free, no
    // barrier), then shared memory for the 8 inter-group values.
    // This cuts the softmax barrier count from 18 to 3.
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += THREADS) {
        local_max = max(local_max, scores[t]);
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_max = simd_max(local_max);

    // Inter-SIMD reduction: lane 0 of each group writes one
    // value, then all threads reduce 8 values locally.
    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_score = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        max_score = max(max_score, shared_buf[i]);
    }

    // ── Step 3: Compute exp(score − max) and sum ─────────────
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += THREADS) {
        const float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_exp = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        sum_exp += shared_buf[i];
    }

    // ── Step 4: Normalize scores to attention weights ────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        scores[t] /= sum_exp;
    }

    // Ensure all weights are visible before the V summation.
    threadgroup_barrier(mem_flags::mem_device);

    // ── Step 5: Weighted sum of V ────────────────────────────
    // Each thread computes a subset of output dimensions.
    // Every thread reads the full attention weight vector.
    for (uint d = tid; d < head_dim; d += THREADS) {
        float accum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            accum +=
                scores[t] * float(v[t * head_dim + d]);
        }
        output[head * head_dim + d] = accum;
    }
}

// ============================================================================
// Embedding Lookup (1-bit packed)
// ============================================================================

/// Dimensions for embedding_lookup kernel.
struct EmbedDims {
    uint vocab_size;    // Number of rows in the embedding table.
    uint hidden_size;   // Elements per embedding vector.
    uint num_tokens;    // Number of tokens to look up.
    uint group_size;    // Weights per scale group (128).
};

/// Gather + dequantize rows from a 1-bit packed embedding table.
///
/// The embedding table is stored in Q1_0_g128 format:
///   packed_bits: ceil(vocab_size * hidden_size / 8) bytes
///   scales: ceil(vocab_size * hidden_size / group_size) f16 values
///
/// For each token, the kernel dequantizes the corresponding row:
///   flat_bit = token_id * hidden_size + d
///   output = bit ? +scale : −scale
///
/// Output is [num_tokens × hidden_size] in f32.
///
/// Dispatch: 1D, count = num_tokens * hidden_size.
kernel void embedding_lookup(
    device const uint*    token_ids   [[buffer(0)]],
    device const uint8_t* packed_bits [[buffer(1)]],
    device const half*    scales      [[buffer(2)]],
    device float*         output      [[buffer(3)]],
    constant EmbedDims&   dims        [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = dims.num_tokens * dims.hidden_size;

    if (gid >= total) return;

    const uint token_idx = gid / dims.hidden_size;
    const uint d = gid % dims.hidden_size;
    const uint row = token_ids[token_idx];

    // Flat bit position in the packed table.
    const uint flat_bit = row * dims.hidden_size + d;
    const uint byte_idx = flat_bit / 8;
    const uint bit_pos = flat_bit % 8;
    const bool bit_set =
        (packed_bits[byte_idx] >> bit_pos) & 1;

    // Scale group from flat position.
    const uint group_idx = flat_bit / dims.group_size;
    const float scale = float(scales[group_idx]);

    output[gid] = select(-scale, +scale, bit_set);
}

// ============================================================================
// RMSNorm with f16 output
// ============================================================================

/// RMSNorm with f32 input and f16 output.  Internal accumulation
/// stays f32 for precision.  Halves output bandwidth by writing
/// half instead of float.  Used when the downstream kernel
/// (qmv) reads f16 activations via the constant cache.
///
/// Dispatch: 1D threadgroups (one per token), 256 threads.
kernel void rms_norm_f16out(
    device const float*    input  [[buffer(0)]],
    device const half*     scale  [[buffer(1)]],
    device half*           output [[buffer(2)]],
    constant RMSNormDims&  dims   [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint token = tgid;
    const uint hidden_size = dims.hidden_size;

    // Out-of-bounds threadgroups exit early.
    if (token >= dims.num_tokens) return;

    device const float* x = input + token * hidden_size;
    device half* out = output + token * hidden_size;

    // Step 1: Each thread accumulates partial sum of squares.
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_size; i += THREADS) {
        const float val = x[i];
        partial_sum_sq += val * val;
    }

    // Step 2: Threadgroup reduction for total sum of squares.
    threadgroup float shared[256];
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rsqrt(mean(x²) + eps).  Every thread reads the same value.
    const float rms = rsqrt(
        shared[0] / float(hidden_size) + dims.eps
    );

    // Step 3: Normalize, scale, and convert to f16.
    for (uint i = tid; i < hidden_size; i += THREADS) {
        out[i] = half(x[i] * rms * float(scale[i]));
    }
}

// ============================================================================
// Residual Add (in-place)
// ============================================================================

/// In-place residual addition: residual[i] += addition[i].
///
/// Used after attention output projection and after MLP output
/// projection to add the sublayer result back to the residual
/// stream.
///
/// Dispatch: 1D, count = total elements.
kernel void residual_add(
    device float*       residual [[buffer(0)]],
    device const float* addition [[buffer(1)]],
    constant uint&      count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    residual[gid] += addition[gid];
}

// ============================================================================
// f16 Kernel Variants
//
// The kernels below mirror their f32 counterparts but operate on
// f16 inputs and/or outputs.  Internal accumulation stays f32 for
// numerical precision.  Buffer binding indices, dispatch patterns,
// and struct types are identical to the originals.
// ============================================================================

// ============================================================================
// RMSNorm f16 (f16 input, f16 output)
// ============================================================================

/// RMSNorm with f16 input and f16 output for QK head norms.
///
/// Input is [num_tokens × hidden_size] in f16.
/// Scale is [hidden_size] in f16 (shared across all tokens).
/// Output is [num_tokens × hidden_size] in f16.
/// Internal accumulation stays f32 for precision.
///
/// Dispatch: threadgroups = num_tokens, threads_per_group = 256.
kernel void rms_norm_f16(
    device const half*     input  [[buffer(0)]],
    device const half*     scale  [[buffer(1)]],
    device half*           output [[buffer(2)]],
    constant RMSNormDims&  dims   [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint token = tgid;
    const uint hidden_size = dims.hidden_size;

    // Out-of-bounds threadgroups exit early.
    if (token >= dims.num_tokens) return;

    device const half* x = input + token * hidden_size;
    device half* out = output + token * hidden_size;

    // Step 1: Each thread accumulates a partial sum of squares.
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_size; i += THREADS) {
        const float val = float(x[i]);
        partial_sum_sq += val * val;
    }

    // Step 2: Threadgroup reduction for total sum of squares.
    threadgroup float shared[256];
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rsqrt(mean(x²) + eps).  Every thread reads the same value.
    const float rms = rsqrt(
        shared[0] / float(hidden_size) + dims.eps
    );

    // Step 3: Normalize, scale, and convert to f16.
    for (uint i = tid; i < hidden_size; i += THREADS) {
        out[i] = half(
            float(x[i]) * rms * float(scale[i])
        );
    }
}

// ============================================================================
// RoPE f16 (f16 in-place)
// ============================================================================

/// Rotary position embeddings operating in-place on f16 data.
///
/// Identical to rope but reads/writes f16.  Internal trig
/// computation stays f32 for precision.
///
/// Dispatch: 1D, count = num_heads * (head_dim / 2).
kernel void rope_f16(
    device half*         data  [[buffer(0)]],
    constant RoPEDims&   dims  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = dims.head_dim / 2;
    const uint total_pairs = dims.num_heads * half_dim;

    if (gid >= total_pairs) return;

    const uint head = gid / half_dim;
    const uint pair = gid % half_dim;

    // Frequency for this pair index.
    // theta_i = 1 / (rope_theta ^ (2 * pair / head_dim))
    const float exponent =
        float(2 * pair) / float(dims.head_dim);
    // exp2 is a hardware instruction; pow is software-emulated.
    const float theta_i =
        exp2(-exponent * log2(dims.rope_theta));
    const float angle = float(dims.position) * theta_i;

    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    // Element indices within the flat buffer.
    const uint base = head * dims.head_dim + pair * 2;
    const float x0 = float(data[base]);
    const float x1 = float(data[base + 1]);

    data[base]     = half(x0 * cos_val - x1 * sin_val);
    data[base + 1] = half(x0 * sin_val + x1 * cos_val);
}

// ============================================================================
// Fused RMSNorm + RoPE (f16)
// ============================================================================

/// Dimensions for fused per-head RMSNorm + RoPE kernel.
struct FusedNormRoPEDims {
    uint  num_heads;     // Number of heads to process.
    uint  head_dim;      // Elements per head (must be even).
    uint  position;      // Sequence position for this token.
    float eps;           // RMSNorm epsilon (1e-6).
    float rope_theta;    // Base frequency (e.g. 1000000.0).
};

/// Fused per-head RMSNorm + RoPE, in-place on f16 data.
///
/// For each head: compute RMSNorm (sum-of-squares reduction,
/// normalize, scale), then immediately apply RoPE rotation.
/// All internal computation stays f32; loads/stores are f16.
///
/// One threadgroup per head, 256 threads each.  Threads beyond
/// head_dim are idle but participate in the reduction via zero.
///
/// Dispatch: threadgroups = num_heads, threads = 256.
kernel void fused_norm_rope_f16(
    device half*                data   [[buffer(0)]],
    device const half*          scale  [[buffer(1)]],
    constant FusedNormRoPEDims& dims   [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint head = tgid;
    const uint head_dim = dims.head_dim;

    if (head >= dims.num_heads) return;

    device half* x = data + head * head_dim;

    // ── Step 1: Load and compute partial sum of squares ──────
    float val = 0.0f;
    if (tid < head_dim) {
        val = float(x[tid]);
    }
    const float partial_sq = val * val;

    // ── Step 2: Threadgroup reduction for sum of squares ─────
    threadgroup float shared[256];
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms = rsqrt(
        shared[0] / float(head_dim) + dims.eps
    );

    // ── Step 3: Normalize, scale, and apply RoPE ─────────────
    // Each pair of consecutive elements gets rotated.
    // Thread tid handles element tid (if tid < head_dim).
    // We need both elements of a pair to rotate, so we load
    // the normalized pair into threadgroup memory first.
    float normed = 0.0f;
    if (tid < head_dim) {
        normed = val * rms * float(scale[tid]);
    }

    // Store normalized values in shared memory for pair access.
    shared[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply RoPE: each even-indexed thread handles a pair.
    if (tid < head_dim && (tid % 2 == 0)) {
        const uint pair = tid / 2;
        const float exponent =
            float(2 * pair) / float(head_dim);
        const float theta_i =
            exp2(-exponent * log2(dims.rope_theta));
        const float angle =
            float(dims.position) * theta_i;

        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        const float x0 = shared[tid];
        const float x1 = shared[tid + 1];

        x[tid]     = half(x0 * cos_val - x1 * sin_val);
        x[tid + 1] = half(x0 * sin_val + x1 * cos_val);
    }
}

// ============================================================================
// Fused K-Norm + RoPE K + KV Cache Update (f16)
// ============================================================================

/// Dimensions for fused K-norm + RoPE K + KV cache update.
struct FusedKNormRoPEKVDims {
    uint  num_kv_heads;         // Number of KV heads.
    uint  head_dim;             // Elements per head (must be even).
    uint  position;             // Sequence position for this token.
    uint  max_context_length;   // Cache capacity (time dimension).
    float eps;                  // RMSNorm epsilon (1e-6).
    float rope_theta;           // Base frequency (e.g. 1000000.0).
};

/// Fused K RMSNorm + RoPE + KV cache write.
///
/// For each KV head:
///   1. RMSNorm on K projection (cooperative reduction)
///   2. Apply RoPE rotation
///   3. Write result to k_cache
///   4. Copy V projection to v_cache (no norm/rotation)
///
/// K projection is NOT modified — result goes directly to cache.
/// One threadgroup per KV head, 256 threads each.
///
/// Dispatch: threadgroups = num_kv_heads, threads = 256.
kernel void fused_k_norm_rope_kv_cache_f16(
    device const half*            k_proj   [[buffer(0)]],
    device const half*            v_proj   [[buffer(1)]],
    device half*                  k_cache  [[buffer(2)]],
    device half*                  v_cache  [[buffer(3)]],
    device const half*            k_scale  [[buffer(4)]],
    constant FusedKNormRoPEKVDims& dims    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint head = tgid;
    const uint head_dim = dims.head_dim;

    if (head >= dims.num_kv_heads) return;

    device const half* k_in = k_proj + head * head_dim;
    device const half* v_in = v_proj + head * head_dim;

    // Cache layout: [head][time][d].
    const uint cache_stride =
        dims.max_context_length * head_dim;
    const uint cache_base =
        head * cache_stride
        + dims.position * head_dim;

    // ── Step 1: Load K and compute partial sum of squares ────
    float k_val = 0.0f;
    if (tid < head_dim) {
        k_val = float(k_in[tid]);
    }
    const float partial_sq = k_val * k_val;

    // ── Step 2: Threadgroup reduction for sum of squares ─────
    threadgroup float shared[256];
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms = rsqrt(
        shared[0] / float(head_dim) + dims.eps
    );

    // ── Step 3: Normalize K with scale ───────────────────────
    float k_normed = 0.0f;
    if (tid < head_dim) {
        k_normed = k_val * rms * float(k_scale[tid]);
    }

    // Store normalized K in shared memory for RoPE pair access.
    shared[tid] = k_normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 4: Apply RoPE and write K to cache ──────────────
    if (tid < head_dim && (tid % 2 == 0)) {
        const uint pair = tid / 2;
        const float exponent =
            float(2 * pair) / float(head_dim);
        const float theta_i =
            exp2(-exponent * log2(dims.rope_theta));
        const float angle =
            float(dims.position) * theta_i;

        const float cos_val = cos(angle);
        const float sin_val = sin(angle);

        const float x0 = shared[tid];
        const float x1 = shared[tid + 1];

        k_cache[cache_base + tid] =
            half(x0 * cos_val - x1 * sin_val);
        k_cache[cache_base + tid + 1] =
            half(x0 * sin_val + x1 * cos_val);
    }

    // ── Step 5: Copy V to cache (no norm/rotation) ───────────
    if (tid < head_dim) {
        v_cache[cache_base + tid] = v_in[tid];
    }
}

// ============================================================================
// Fused RoPE K + KV Cache Update (f16)
// ============================================================================

/// Combined dimensions for fused rope_k + kv_cache_update.
struct FusedRoPEKVDims {
    uint  num_kv_heads;         // Number of KV heads.
    uint  head_dim;             // Elements per head (must be even).
    uint  position;             // Sequence position for this token.
    uint  max_context_length;   // Cache capacity (time dimension).
    float rope_theta;           // Base frequency (e.g. 1000000.0).
};

/// Fused RoPE K + KV cache update.
///
/// Each thread handles one RoPE pair for K (2 elements) and copies
/// the corresponding 2 V elements to the V cache.  This eliminates
/// the separate rope_k dispatch and the barrier between RoPE and
/// KV cache update, saving 1 dispatch + 1 barrier per block
/// (28 dispatches + 28 barriers for Bonsai's 28 layers).
///
/// K is read from k_proj, RoPE is applied, and the result is
/// written DIRECTLY to k_cache (k_proj buffer is NOT modified).
/// V is copied from v_proj to v_cache unchanged.
///
/// Internal trig computation stays f32 for precision.
///
/// Dispatch: 1D, count = num_kv_heads * (head_dim / 2).
kernel void fused_rope_k_kv_cache_update_f16(
    device const half*        k_proj  [[buffer(0)]],
    device const half*        v_proj  [[buffer(1)]],
    device half*              k_cache [[buffer(2)]],
    device half*              v_cache [[buffer(3)]],
    constant FusedRoPEKVDims& dims    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = dims.head_dim / 2;
    const uint total_pairs = dims.num_kv_heads * half_dim;

    if (gid >= total_pairs) return;

    const uint head = gid / half_dim;
    const uint pair = gid % half_dim;
    const uint pos = dims.position;

    // ── RoPE rotation for K ──────────────────────────────────
    const float exponent =
        float(2 * pair) / float(dims.head_dim);
    const float theta_i =
        exp2(-exponent * log2(dims.rope_theta));
    const float angle = float(pos) * theta_i;

    const float cos_val = cos(angle);
    const float sin_val = sin(angle);

    // Source indices in flat k_proj buffer.
    const uint src_base = head * dims.head_dim + pair * 2;
    const float x0 = float(k_proj[src_base]);
    const float x1 = float(k_proj[src_base + 1]);

    // Cache layout: [head][time][d].
    const uint cache_stride =
        dims.max_context_length * dims.head_dim;
    const uint cache_base =
        head * cache_stride + pos * dims.head_dim
        + pair * 2;

    // Write RoPE-rotated K directly to cache.
    k_cache[cache_base]     =
        half(x0 * cos_val - x1 * sin_val);
    k_cache[cache_base + 1] =
        half(x0 * sin_val + x1 * cos_val);

    // ── V copy (no rotation) ─────────────────────────────────
    v_cache[cache_base]     = v_proj[src_base];
    v_cache[cache_base + 1] = v_proj[src_base + 1];
}

// ============================================================================
// KV Cache Update with f16 Input
// ============================================================================

/// Append one token's f16 K and V projections to the f16 KV cache.
///
/// K/V projections are [num_kv_heads × head_dim] in f16.
/// K/V caches are [num_kv_heads × max_context_length × head_dim]
/// in f16.
///
/// No conversion needed — both source and destination are f16.
///
/// Dispatch: 1D, count = num_kv_heads * head_dim.
kernel void kv_cache_update_f16in(
    device const half*     k_proj  [[buffer(0)]],
    device const half*     v_proj  [[buffer(1)]],
    device half*           k_cache [[buffer(2)]],
    device half*           v_cache [[buffer(3)]],
    constant KVUpdateDims& dims    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = dims.num_kv_heads * dims.head_dim;

    if (gid >= total) return;

    const uint head = gid / dims.head_dim;
    const uint d = gid % dims.head_dim;
    const uint pos = dims.position;

    // Cache layout: [head][time][d].
    const uint cache_stride =
        dims.max_context_length * dims.head_dim;
    const uint cache_idx =
        head * cache_stride + pos * dims.head_dim + d;

    // Both source and cache are f16 — no cast needed.
    k_cache[cache_idx] = k_proj[gid];
    v_cache[cache_idx] = v_proj[gid];
}

// ============================================================================
// GQA Attention with f16 Q Input and f16 Output
// ============================================================================

/// Grouped query attention with f16 Q input and f16 output.
///
/// Q is [num_query_heads × head_dim] in f16.
/// K cache is [num_kv_heads × max_ctx × head_dim] in f16.
/// V cache is [num_kv_heads × max_ctx × head_dim] in f16.
/// Output is [num_query_heads × head_dim] in f16.
/// Scratch is [num_query_heads × max_ctx] in f32.
///
/// Q is loaded into threadgroup memory with f16→f32 conversion.
/// All internal computation (dot products, softmax, V accumulation)
/// stays f32 for precision.  Output is converted f32→f16 on write.
///
/// Dispatch: threadgroups = num_query_heads, threads = 256.
kernel void gqa_attention_f16io(
    device const half*   Q       [[buffer(0)]],
    device const half*   k_cache [[buffer(1)]],
    device const half*   v_cache [[buffer(2)]],
    device half*         output  [[buffer(3)]],
    device float*        scratch [[buffer(4)]],
    constant GQADims&    dims    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint SIMD_SIZE = 32;
    const uint NUM_SIMD_GROUPS = THREADS / SIMD_SIZE;
    const uint head = tgid;

    if (head >= dims.num_query_heads) return;

    const uint kv_head = head / dims.heads_per_kv_group;
    const uint head_dim = dims.head_dim;
    const uint seq_len = dims.seq_len;
    const uint max_ctx = dims.max_context_length;

    const uint simd_group = tid / SIMD_SIZE;
    const uint simd_lane = tid % SIMD_SIZE;

    // Pointers for this head.
    device const half* q =
        Q + head * head_dim;
    const uint kv_stride = max_ctx * head_dim;
    device const half* k =
        k_cache + kv_head * kv_stride;
    device const half* v =
        v_cache + kv_head * kv_stride;
    device float* scores =
        scratch + head * max_ctx;

    // Scaling factor: 1 / sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(head_dim));

    // ── Load Q into threadgroup memory (f16 → f32) ──────────
    threadgroup float shared_buf[256];

    if (tid < head_dim) {
        shared_buf[tid] = float(q[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 1: Compute attention scores ─────────────────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        float dot_product = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot_product +=
                shared_buf[d]
                * float(k[t * head_dim + d]);
        }
        scores[t] = dot_product * inv_sqrt_d;
    }

    // Ensure all scores are written before reading them.
    threadgroup_barrier(mem_flags::mem_device);

    // ── Step 2: Softmax — find max (numerical stability) ─────
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += THREADS) {
        local_max = max(local_max, scores[t]);
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_max = simd_max(local_max);

    // Inter-SIMD reduction.
    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_score = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        max_score = max(max_score, shared_buf[i]);
    }

    // ── Step 3: Compute exp(score − max) and sum ─────────────
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += THREADS) {
        const float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_exp = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        sum_exp += shared_buf[i];
    }

    // ── Step 4: Normalize scores to attention weights ────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        scores[t] /= sum_exp;
    }

    // Ensure all weights are visible before the V summation.
    threadgroup_barrier(mem_flags::mem_device);

    // ── Step 5: Weighted sum of V, write f16 output ──────────
    for (uint d = tid; d < head_dim; d += THREADS) {
        float accum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            accum +=
                scores[t] * float(v[t * head_dim + d]);
        }
        output[head * head_dim + d] = half(accum);
    }
}

// ============================================================================
// GQA Attention f16 I/O — Threadgroup Scores (short sequences)
// ============================================================================

/// Identical to gqa_attention_f16io but stores attention scores in
/// threadgroup memory instead of a device-memory scratch buffer.
/// This eliminates buffer(4) and replaces mem_device barriers with
/// cheaper mem_threadgroup barriers.
///
/// Threadgroup memory budget:
///   shared_buf = 256 × 4 B = 1 024 B
///   tg_scores  = 1024 × 4 B = 4 096 B
///   total      =              5 120 B  (< 32 KB limit)
///
/// Dispatch: 1 threadgroup per query head, 256 threads each.
kernel void gqa_attention_f16io_tg(
    device const half*   Q       [[buffer(0)]],
    device const half*   k_cache [[buffer(1)]],
    device const half*   v_cache [[buffer(2)]],
    device half*         output  [[buffer(3)]],
    constant GQADims&    dims    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint SIMD_SIZE = 32;
    const uint NUM_SIMD_GROUPS = THREADS / SIMD_SIZE;
    const uint TG_SEQ_MAX = 1024;
    const uint head = tgid;

    if (head >= dims.num_query_heads) return;

    const uint kv_head = head / dims.heads_per_kv_group;
    const uint head_dim = dims.head_dim;
    const uint seq_len = dims.seq_len;
    const uint max_ctx = dims.max_context_length;

    // Guard: dispatch side must route long sequences to the
    // device-memory variant.
    if (seq_len > TG_SEQ_MAX) return;

    const uint simd_group = tid / SIMD_SIZE;
    const uint simd_lane = tid % SIMD_SIZE;

    // Pointers for this head.
    device const half* q = Q + head * head_dim;
    const uint kv_stride = max_ctx * head_dim;
    device const half* k = k_cache + kv_head * kv_stride;
    device const half* v = v_cache + kv_head * kv_stride;

    // Scaling factor: 1 / sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(head_dim));

    // Threadgroup allocations.
    threadgroup float shared_buf[256];
    threadgroup float tg_scores[TG_SEQ_MAX];

    // ── Load Q into threadgroup memory (f16 → f32) ──────────
    if (tid < head_dim) {
        shared_buf[tid] = float(q[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 1: Compute attention scores ─────────────────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        float dot_product = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot_product +=
                shared_buf[d]
                * float(k[t * head_dim + d]);
        }
        tg_scores[t] = dot_product * inv_sqrt_d;
    }

    // Ensure all scores are written before reading them.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 2: Softmax — find max (numerical stability) ─────
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += THREADS) {
        local_max = max(local_max, tg_scores[t]);
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_max = simd_max(local_max);

    // Inter-SIMD reduction.
    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_score = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        max_score = max(max_score, shared_buf[i]);
    }

    // ── Step 3: Compute exp(score − max) and sum ─────────────
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += THREADS) {
        const float e = exp(tg_scores[t] - max_score);
        tg_scores[t] = e;
        local_sum += e;
    }

    // Intra-SIMD reduction: 32 → 1, no barrier needed.
    const float warp_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_exp = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        sum_exp += shared_buf[i];
    }

    // ── Step 4: Normalize scores to attention weights ────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        tg_scores[t] /= sum_exp;
    }

    // Ensure all weights are visible before V summation.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 5: Weighted sum of V, write f16 output ──────────
    for (uint d = tid; d < head_dim; d += THREADS) {
        float accum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            accum +=
                tg_scores[t] * float(v[t * head_dim + d]);
        }
        output[head * head_dim + d] = half(accum);
    }
}

// ============================================================================
// Fused GQA Attention + Q-Norm + RoPE + K-Norm + RoPE + KV Cache (f16 TG)
// ============================================================================

/// Extended dims for GQA with fused Q/K norm, RoPE, and KV cache.
struct GQAFusedDims {
    uint  num_query_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  seq_len;
    uint  max_context_length;
    uint  heads_per_kv_group;
    uint  position;
    float eps;
    float rope_theta;
};

/// Fused GQA attention that also applies:
///   - Per-head RMSNorm + RoPE to Q (from raw projection)
///   - Per-head RMSNorm + RoPE to K, then write to k_cache
///   - Copy V to v_cache
///
/// Each TG handles one query head.  The KV cache update is
/// redundantly computed by all query heads sharing a KV head
/// (heads_per_kv_group TGs write identical values).  This
/// eliminates 2 dispatches + 1 barrier per block vs separate
/// norm+RoPE kernels.
///
/// Dispatch: threadgroups = num_query_heads, threads = 256.
kernel void gqa_attention_fused_f16io_tg(
    device const half*    Q          [[buffer(0)]],
    device half*          k_cache    [[buffer(1)]],
    device half*          v_cache    [[buffer(2)]],
    device half*          output     [[buffer(3)]],
    device const half*    K_raw      [[buffer(4)]],
    device const half*    V_raw      [[buffer(5)]],
    device const half*    q_norm_sc  [[buffer(6)]],
    device const half*    k_norm_sc  [[buffer(7)]],
    constant GQAFusedDims& dims      [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    const uint THREADS = 256;
    const uint SIMD_SIZE = 32;
    const uint NUM_SIMD_GROUPS = THREADS / SIMD_SIZE;
    const uint TG_SEQ_MAX = 1024;
    const uint head = tgid;

    if (head >= dims.num_query_heads) return;

    const uint kv_head = head / dims.heads_per_kv_group;
    const uint head_dim = dims.head_dim;
    const uint seq_len = dims.seq_len;
    const uint max_ctx = dims.max_context_length;
    const uint position = dims.position;

    if (seq_len > TG_SEQ_MAX) return;

    const uint simd_group = tid / SIMD_SIZE;
    const uint simd_lane = tid % SIMD_SIZE;

    const uint kv_stride = max_ctx * head_dim;
    device half* kc = k_cache + kv_head * kv_stride;
    device half* vc = v_cache + kv_head * kv_stride;

    const float inv_sqrt_d = rsqrt(float(head_dim));

    threadgroup float shared_buf[256];
    threadgroup float tg_scores[TG_SEQ_MAX];

    // ── Phase 1: Q — load, per-head RMSNorm, RoPE ──────────
    float q_val = (tid < head_dim)
        ? float(Q[head * head_dim + tid]) : 0.0f;

    // Sum of squares reduction for Q RMSNorm.
    float q_sq = q_val * q_val;
    float q_warp_sq = simd_sum(q_sq);
    if (simd_lane == 0) {
        shared_buf[simd_group] = q_warp_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint active_warps = (head_dim + 31) / 32;
    float q_total = 0.0f;
    for (uint i = 0; i < active_warps; i++) {
        q_total += shared_buf[i];
    }
    const float q_rms =
        rsqrt(q_total / float(head_dim) + dims.eps);

    // Normalize and scale Q.
    if (tid < head_dim) {
        q_val = q_val * q_rms * float(q_norm_sc[tid]);
    }
    shared_buf[tid] = q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RoPE on Q: consecutive-pair rotation.
    if (tid < head_dim && (tid % 2 == 0)) {
        const uint pair = tid / 2;
        const float exp2_arg =
            -float(2 * pair) / float(head_dim)
            * log2(dims.rope_theta);
        const float theta_i = exp2(exp2_arg);
        const float angle = float(position) * theta_i;
        const float cv = cos(angle);
        const float sv = sin(angle);
        const float a = shared_buf[tid];
        const float b = shared_buf[tid + 1];
        shared_buf[tid]     = a * cv - b * sv;
        shared_buf[tid + 1] = a * sv + b * cv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // shared_buf[0..head_dim-1] = Q after norm + RoPE.

    // ── Phase 2: K — norm + RoPE → k_cache; V → v_cache ────
    device const half* kr = K_raw + kv_head * head_dim;
    device const half* vr = V_raw + kv_head * head_dim;

    float k_val = (tid < head_dim)
        ? float(kr[tid]) : 0.0f;

    // K RMSNorm reduction — use tg_scores[0..7] as scratch.
    float k_sq = k_val * k_val;
    float k_warp_sq = simd_sum(k_sq);
    if (simd_lane == 0) {
        tg_scores[simd_group] = k_warp_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_total = 0.0f;
    for (uint i = 0; i < active_warps; i++) {
        k_total += tg_scores[i];
    }
    const float k_rms =
        rsqrt(k_total / float(head_dim) + dims.eps);

    // Normalize, scale K, store for RoPE pair access.
    float k_normed = 0.0f;
    if (tid < head_dim) {
        k_normed = k_val * k_rms * float(k_norm_sc[tid]);
    }
    tg_scores[tid] = k_normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // RoPE on K, write to k_cache at current position.
    if (tid < head_dim && (tid % 2 == 0)) {
        const uint pair = tid / 2;
        const float exp2_arg =
            -float(2 * pair) / float(head_dim)
            * log2(dims.rope_theta);
        const float theta_i = exp2(exp2_arg);
        const float angle = float(position) * theta_i;
        const float cv = cos(angle);
        const float sv = sin(angle);
        const float a = tg_scores[tid];
        const float b = tg_scores[tid + 1];
        kc[position * head_dim + tid] =
            half(a * cv - b * sv);
        kc[position * head_dim + tid + 1] =
            half(a * sv + b * cv);
    }

    // Copy V to v_cache at current position.
    if (tid < head_dim) {
        vc[position * head_dim + tid] = vr[tid];
    }

    // Ensure k_cache/v_cache writes are visible within TG.
    threadgroup_barrier(
        mem_flags::mem_device
        | mem_flags::mem_threadgroup
    );

    // ── Phase 3: Attention scores Q·K ───────────────────────
    for (uint t = tid; t < seq_len; t += THREADS) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += shared_buf[d]
                * float(kc[t * head_dim + d]);
        }
        tg_scores[t] = dot * inv_sqrt_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Softmax — max, exp, sum, normalize ─────────
    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += THREADS) {
        local_max = max(local_max, tg_scores[t]);
    }
    const float warp_max = simd_max(local_max);
    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_score = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        max_score = max(max_score, shared_buf[i]);
    }

    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += THREADS) {
        const float e = exp(tg_scores[t] - max_score);
        tg_scores[t] = e;
        local_sum += e;
    }
    const float warp_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        shared_buf[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_exp = shared_buf[0];
    for (uint i = 1; i < NUM_SIMD_GROUPS; i++) {
        sum_exp += shared_buf[i];
    }

    for (uint t = tid; t < seq_len; t += THREADS) {
        tg_scores[t] /= sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Weighted sum of V, write f16 output ────────
    for (uint d = tid; d < head_dim; d += THREADS) {
        float accum = 0.0f;
        for (uint t = 0; t < seq_len; t++) {
            accum +=
                tg_scores[t]
                * float(vc[t * head_dim + d]);
        }
        output[head * head_dim + d] = half(accum);
    }
}

// ============================================================================
// Fused SiLU + Elementwise Mul f16
// ============================================================================

/// Fused SiLU + elementwise multiply with f16 input and f16 output.
///   output[i] = silu(gate[i]) * up[i]
///             = gate[i] * sigmoid(gate[i]) * up[i]
///
/// Internal accumulation stays f32 for precision.
///
/// Dispatch: 1D, count = total elements.
kernel void silu_elementwise_mul_f16(
    device const half* gate   [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const float g = float(gate[gid]);
    const float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[gid] = half(g * sigmoid_g * float(up[gid]));
}

// ============================================================================
// Residual Add f16 (f16 addition into f32 residual)
// ============================================================================

/// Residual add with f16 addition into f32 residual stream.
///
/// residual[i] += float(addition[i])
///
/// The residual stream stays f32 for full-precision accumulation
/// across layers.  The addition (e.g. attention or MLP output) is
/// f16, converted to f32 before accumulation.
///
/// Dispatch: 1D, count = total elements.
kernel void residual_add_f16(
    device float*       residual [[buffer(0)]],
    device const half*  addition [[buffer(1)]],
    constant uint&      count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    residual[gid] += float(addition[gid]);
}

// ============================================================================
// GPU completion flag — single-thread atomic write
// ============================================================================

/// Write 1 to flag[0] via atomic store.  Dispatched as the
/// last kernel in a compute encoder so the CPU can spin-wait
/// on the flag instead of calling waitUntilCompleted (avoids
/// ~100-150 us Mach kernel trap overhead).
///
/// Dispatch: 1 threadgroup, 1 thread.
kernel void set_completion_flag(
    device atomic_uint* flag [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid == 0) {
        atomic_store_explicit(
            flag, 1u, memory_order_relaxed);
    }
}
