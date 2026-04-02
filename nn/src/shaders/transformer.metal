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
