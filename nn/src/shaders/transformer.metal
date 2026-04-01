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
    const float theta_i =
        1.0f / pow(dims.rope_theta, exponent);
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
    const uint head = tgid;

    if (head >= dims.num_query_heads) return;

    const uint kv_head = head / dims.heads_per_kv_group;
    const uint head_dim = dims.head_dim;
    const uint seq_len = dims.seq_len;
    const uint max_ctx = dims.max_context_length;

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

    // ── Step 1: Compute attention scores ─────────────────────
    // Each thread processes a subset of time steps.
    for (uint t = tid; t < seq_len; t += THREADS) {
        float dot_product = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot_product +=
                q[d] * float(k[t * head_dim + d]);
        }
        scores[t] = dot_product * inv_sqrt_d;
    }

    // Ensure all scores are written before reading them.
    threadgroup_barrier(mem_flags::mem_device);

    // ── Step 2: Softmax — find max (numerical stability) ─────
    threadgroup float shared_reduce[256];

    float local_max = -INFINITY;
    for (uint t = tid; t < seq_len; t += THREADS) {
        local_max = max(local_max, scores[t]);
    }

    shared_reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared_reduce[tid] = max(
                shared_reduce[tid],
                shared_reduce[tid + stride]
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float max_score = shared_reduce[0];

    // ── Step 3: Compute exp(score − max) and sum ─────────────
    float local_sum = 0.0f;
    for (uint t = tid; t < seq_len; t += THREADS) {
        const float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }

    shared_reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0;
         stride /= 2) {
        if (tid < stride) {
            shared_reduce[tid] +=
                shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float sum_exp = shared_reduce[0];

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
