// Fused clamped-softmax CUDA kernel — also outputs per-row λ* for fast backward.
//
// Problem: given scores (N, L), compute clamp(softmax(x)*k − λ*, 0, 1)
//   where λ* is the threshold such that Σ_i output_i = k  (budget constraint).
//
// Algorithm: parallel bisection for each row independently.
//   f(λ) = Σ_i clamp(p_i − λ, 0, 1)  is monotone decreasing in λ
//   f(−1)    = L ≥ k    (p_i − (−1) = p_i + 1 ≥ 1 for all p_i ≥ 0)
//   f(p_max) = 0 ≤ k
//   ⟹ λ* ∈ [−1, p_max]  found in 30 bisection steps (< 4e-6 precision)
//
// One CUDA block per row.  256 threads.  Shared memory: 8 floats only.
// The p array (≤ 64 KB for head_large) stays in L1 cache across all bisection
// iterations — no extra global-memory reads after the initial softmax write.
//
// λ* is written to a separate (N,) output buffer so Python can use it for the
// analytical gradient in the backward pass — avoids re-running topk.

#include <cuda.h>
#include <torch/extension.h>
#include <float.h>


// ─── Warp / block reductions ──────────────────────────────────────────────────

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// smem must have ≥ (blockDim.x / 32) float slots
__device__ float block_max(float v, float* smem) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? smem[threadIdx.x] : -FLT_MAX;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

__device__ float block_sum(float v, float* smem) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}


// ─── Main kernel ──────────────────────────────────────────────────────────────

__global__ void cs_fwd(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    float*       __restrict__ lam_out,   // (N,) — λ* per row for backward
    int N, int L, float k, int iters)
{
    extern __shared__ float smem[];   // 8 floats for block reductions
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = src + (int64_t)row * L;
    float*       y = dst + (int64_t)row * L;

    // ── 1.  p = softmax(x) × k  (written to y for in-place bisection) ────────
    float vmax = -FLT_MAX;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        vmax = fmaxf(vmax, x[i]);
    vmax = block_max(vmax, smem);

    float vsum = 0.f;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        vsum += expf(x[i] - vmax);
    vsum = block_sum(vsum, smem);

    float scale = k / vsum;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        y[i] = expf(x[i] - vmax) * scale;
    __syncthreads();   // ensure all threads see the full p array before bisection

    // ── 2.  upper bound  hi = max(p) ─────────────────────────────────────────
    float pmax = -FLT_MAX;
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        pmax = fmaxf(pmax, y[i]);
    pmax = block_max(pmax, smem);

    // ── 3.  bisect λ* ∈ [−1, p_max] ─────────────────────────────────────────
    // L1 cache: y (≤ 64 KB) stays warm across all iters — no DRAM traffic.
    float lo = -1.f, hi = pmax;
    for (int t = 0; t < iters; t++) {
        float mid = 0.5f * (lo + hi);
        float bud = 0.f;
        for (int i = threadIdx.x; i < L; i += blockDim.x) {
            float v = y[i] - mid;
            bud += fminf(fmaxf(v, 0.f), 1.f);
        }
        bud = block_sum(bud, smem);
        if (bud > k) lo = mid; else hi = mid;
    }
    float lam = 0.5f * (lo + hi);

    // ── 4.  output = clamp(p − λ*, 0, 1) ─────────────────────────────────────
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        y[i] = fminf(fmaxf(y[i] - lam, 0.f), 1.f);

    // ── 5.  write λ* for backward ─────────────────────────────────────────────
    if (threadIdx.x == 0)
        lam_out[row] = lam;
}


// ─── C++ entry called from Python ─────────────────────────────────────────────

// Returns {out, lam}: out=(N,L) clamped-softmax, lam=(N,) per-row λ* threshold.
std::vector<torch::Tensor> clamped_softmax_forward(
    torch::Tensor score,    // (N, L)  fp16 or fp32
    double        k,
    int64_t       n_bisect)
{
    TORCH_CHECK(score.is_cuda(),  "score must be on CUDA");
    TORCH_CHECK(score.dim() == 2, "score must be 2-D (N, L)");

    auto sf  = score.to(torch::kFloat32).contiguous();
    auto out = torch::empty_like(sf);
    int  N   = (int)sf.size(0);
    int  L   = (int)sf.size(1);

    // Per-row λ* in float32 (N floats — negligible memory vs attention matrix)
    auto lam = torch::empty({N}, sf.options());

    // Fixed 256-thread blocks; for L < 256 each thread handles 0 or 1 elements
    // (the loop condition i < L handles this correctly).
    constexpr int BLOCK = 256;
    int smem_bytes = (BLOCK / 32) * sizeof(float);   // 32 bytes — fits in registers

    cs_fwd<<<N, BLOCK, smem_bytes>>>(
        sf.data_ptr<float>(), out.data_ptr<float>(), lam.data_ptr<float>(),
        N, L, (float)k, (int)n_bisect);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cs_fwd CUDA kernel failed");
    return {out.to(score.dtype()), lam};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &clamped_softmax_forward,
          "Clamped-softmax forward (CUDA bisection) → [out, lam]",
          py::arg("score"),
          py::arg("k"),
          py::arg("n_bisect") = 30);
}
