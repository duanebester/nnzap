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

/// Fused matmul + bias_add: out = W * x + bias
///   W is [M x K], x is [K x N], bias is [N], out is [M x N]
///
/// Eliminates a separate bias_add dispatch and the memory
/// barrier between matmul and bias_add. Each thread computes
/// one output element including the bias term.
kernel void matmul_bias(
    device const float* W    [[buffer(0)]],  // [M x K]
    device const float* x    [[buffer(1)]],  // [K x N]
    device float*       out  [[buffer(2)]],  // [M x N]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],  // [N]
    uint2 gid                [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (m)
    if (gid.x >= N || gid.y >= M) return;

    float sum = bias[gid.x];
    for (uint i = 0; i < K; i++) {
        sum += W[gid.y * K + i] * x[i * N + gid.x];
    }
    out[gid.y * N + gid.x] = sum;
}

/// Fused matmul + bias_add + ReLU: out = max(0, W*x + bias)
///   Fuses three dispatches into one, saving two kernel
///   launches and two memory barriers per layer.
kernel void matmul_bias_relu(
    device const float* W    [[buffer(0)]],  // [M x K]
    device const float* x    [[buffer(1)]],  // [K x N]
    device float*       out  [[buffer(2)]],  // [M x N]
    constant uint& M         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& N         [[buffer(5)]],
    device const float* bias [[buffer(6)]],  // [N]
    device float*    pre_act [[buffer(7)]],  // [M x N]
    uint2 gid                [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (m)
    if (gid.x >= N || gid.y >= M) return;

    float sum = bias[gid.x];
    for (uint i = 0; i < K; i++) {
        sum += W[gid.y * K + i] * x[i * N + gid.x];
    }
    // Store pre-activation for backward pass
    uint idx = gid.y * N + gid.x;
    pre_act[idx] = sum;
    // Apply ReLU
    out[idx] = max(0.0f, sum);
}

// ============================================================================
// Transposed matrix multiplies (backward pass)
// ============================================================================

/// Compute A^T * B where A is [M x K] row-major, B is [M x N].
/// Result is [K x N].  Used for weight gradients: dW = X^T * dY.
///
/// Each thread computes one element of the output:
///   out[k * N + n] = sum_m  A[m * K + k] * B[m * N + n]
kernel void matmul_transA(
    device const float* A   [[buffer(0)]],  // [M x K]
    device const float* B   [[buffer(1)]],  // [M x N]
    device float*       out [[buffer(2)]],  // [K x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (k) of the output.
    if (gid.x >= N || gid.y >= K) return;

    float sum = 0.0;
    for (uint m = 0; m < M; m++) {
        sum += A[m * K + gid.y] * B[m * N + gid.x];
    }
    out[gid.y * N + gid.x] = sum;
}

/// Compute A * B^T where A is [M x K], B is [N x K] row-major.
/// Result is [M x N].  Used for input gradients: dX = dY * W^T.
///
/// Each thread computes one element of the output:
///   out[m * N + n] = sum_k  A[m * K + k] * B[n * K + k]
kernel void matmul_transB(
    device const float* A   [[buffer(0)]],  // [M x K]
    device const float* B   [[buffer(1)]],  // [N x K]
    device float*       out [[buffer(2)]],  // [M x N]
    constant uint& M        [[buffer(3)]],
    constant uint& K        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    uint2 gid               [[thread_position_in_grid]])
{
    // gid.x = column (n), gid.y = row (m) of the output.
    if (gid.x >= N || gid.y >= M) return;

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[gid.x * K + k];
    }
    out[gid.y * N + gid.x] = sum;
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
kernel void bias_add(
    device const float* input  [[buffer(0)]],  // [M x N]
    device const float* bias   [[buffer(1)]],  // [N]
    device float*       output [[buffer(2)]],  // [M x N]
    constant uint& N           [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    // gid.x = column, gid.y = row
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
