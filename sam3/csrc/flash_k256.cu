// Flash attention with head_dim_qk = 256 and head_dim_v = 64 (SAM 3 video tracker memory attention: the values are the 64-d memory tokens,
// V = Wv x is applied after the attention: softmax(Q K^T) (X Wv^T) = (softmax(Q K^T) X) Wv^T).  fp16 in / out, any Nq, Nk (tails masked).
// FlashAttention-2 schedule: CTA = 4 warps x 16 query rows, K / V tiles of 32 keys double-buffered with cp.async; Q A-fragments held in registers.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) { __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
__device__ __forceinline__ float fast_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// K / Q tile [rows][256] fp16: 32 chunks of 16 B per row, chunk index XOR (row & 7) inside groups of 8 chunks
__device__ __forceinline__ int swz256(int row, int ch) { return row * 256 + (((ch & ~7) | ((ch & 7) ^ (row & 7))) << 3); }
// V tile [rows][64]
__device__ __forceinline__ int swz64(int row, int ch) { return row * 64 + ((ch ^ (row & 7)) << 3); }

constexpr int BR = 64, BC = 32;

template <bool QK16>
__device__ __forceinline__ void flash_k256_impl(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                                                __half* __restrict__ o, int Nq, int Nk, long long qsb, int qsn, long long ksb, int ksn,
                                                long long vsb, int vsn, long long osb, int osn, float scale_log2) {
    // smem: max(Q staging 64 x 256, 2 stages of (K 32 x 256 + V 32 x 64))  = 32 KB / 40 KB
    __shared__ __align__(128) __half smem[2 * (BC * 256 + BC * 64)];
    __half* sQ = smem;
    __half* sK[2] = {smem, smem + BC * 256};
    __half* sV[2] = {smem + 2 * BC * 256, smem + 2 * BC * 256 + BC * 64};

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b = blockIdx.y, q0 = blockIdx.x * BR;
    const __half* qb = q + b * qsb;
    const __half* kb = k + b * ksb;
    const __half* vb = v + b * vsb;

#pragma unroll
    for (int i = 0; i < BR * 32 / 128; ++i) {
        int c = tid + i * 128, row = c >> 5, ch = c & 31;
        int n = q0 + row;
        cp_async16(smem_u32(&sQ[swz256(row, ch)]), qb + (long long)min(n, Nq - 1) * qsn + ch * 8, n < Nq);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    uint32_t qf[16][4];
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        int row = warp * 16 + (lane & 15);
        int ch = kk * 2 + (lane >> 4);
        ldsm_x4(smem_u32(&sQ[swz256(row, ch)]), qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3]);
    }
    __syncthreads();  // everyone has its Q fragments: the staging area becomes K / V stage 0 / 1

    auto load_kv = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < BC * 32 / 128; ++i) {
            int c = tid + i * 128, row = c >> 5, ch = c & 31;
            int n = k0 + row;
            cp_async16(smem_u32(&sK[buf][swz256(row, ch)]), kb + (long long)min(n, Nk - 1) * ksn + ch * 8, n < Nk);
        }
#pragma unroll
        for (int i = 0; i < BC * 8 / 128; ++i) {
            int c = tid + i * 128, row = c >> 3, ch = c & 7;
            int n = k0 + row;
            cp_async16(smem_u32(&sV[buf][swz64(row, ch)]), vb + (long long)min(n, Nk - 1) * vsn + ch * 8, n < Nk);
        }
    };
    load_kv(0, 0);
    cp_async_commit();

    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.f, 0.f};
    float oacc[8][4];
#pragma unroll
    for (int u = 0; u < 8; ++u) oacc[u][0] = oacc[u][1] = oacc[u][2] = oacc[u][3] = 0.f;

    const int nblk = (Nk + BC - 1) / BC;
    for (int j = 0; j < nblk; ++j) {
        cp_async_wait<0>();
        __syncthreads();
        if (j + 1 < nblk) load_kv((j + 1) & 1, (j + 1) * BC);
        cp_async_commit();
        const __half* K_ = sK[j & 1];
        const __half* V_ = sV[j & 1];

        float s[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
        if (QK16) {
            // fp16 accumulation over 64 head dims (4 k-steps), added to the fp32 score
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint32_t sh[4][2];
#pragma unroll
                for (int t = 0; t < 4; ++t) sh[t][0] = sh[t][1] = 0u;
#pragma unroll
                for (int kq = 0; kq < 4; ++kq) {
                    int kk = g * 4 + kq;
#pragma unroll
                    for (int t = 0; t < 4; t += 2) {
                        uint32_t b0, b1, b2, b3;
                        int row = (t + (lane >> 4)) * 8 + (lane & 7);
                        int ch = kk * 2 + ((lane >> 3) & 1);
                        ldsm_x4(smem_u32(&K_[swz256(row, ch)]), b0, b1, b2, b3);
                        mma_f16(sh[t], qf[kk], b0, b1);
                        mma_f16(sh[t + 1], qf[kk], b2, b3);
                    }
                }
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 a = unpack_h2(sh[t][0]), c = unpack_h2(sh[t][1]);
                    s[t][0] += a.x; s[t][1] += a.y; s[t][2] += c.x; s[t][3] += c.y;
                }
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < 16; ++kk)
#pragma unroll
                for (int t = 0; t < 4; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz256(row, ch)]), b0, b1, b2, b3);
                    mma_f32(s[t], qf[kk], b0, b1);
                    mma_f32(s[t + 1], qf[kk], b2, b3);
                }
        }
        if ((j + 1) * BC > Nk) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                int key = j * BC + t * 8 + (lane & 3) * 2;
                if (key >= Nk) s[t][0] = s[t][2] = -INFINITY;
                if (key + 1 >= Nk) s[t][1] = s[t][3] = -INFINITY;
            }
        }
        // online softmax (base 2)
        uint32_t pf[2][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            mx0 = fmaxf(mx0, fmaxf(s[t][0], s[t][1]));
            mx1 = fmaxf(mx1, fmaxf(s[t][2], s[t][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
        float mn0 = fmaxf(m_r[0], mx0 * scale_log2), mn1 = fmaxf(m_r[1], mx1 * scale_log2);
        float c0 = fast_exp2(m_r[0] - mn0), c1 = fast_exp2(m_r[1] - mn1);
        m_r[0] = mn0; m_r[1] = mn1;
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            float p0 = fast_exp2(fmaf(s[t][0], scale_log2, -mn0));
            float p1 = fast_exp2(fmaf(s[t][1], scale_log2, -mn0));
            float p2 = fast_exp2(fmaf(s[t][2], scale_log2, -mn1));
            float p3 = fast_exp2(fmaf(s[t][3], scale_log2, -mn1));
            sum0 += p0 + p1;
            sum1 += p2 + p3;
            pf[t >> 1][(t & 1) * 2 + 0] = pack_h2(p0, p1);
            pf[t >> 1][(t & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        l_r[0] = l_r[0] * c0 + sum0;
        l_r[1] = l_r[1] * c1 + sum1;
#pragma unroll
        for (int u = 0; u < 8; ++u) { oacc[u][0] *= c0; oacc[u][1] *= c0; oacc[u][2] *= c1; oacc[u][3] *= c1; }
        // O += P V   (V = 64-d memory tokens)
#pragma unroll
        for (int kc = 0; kc < 2; ++kc)
#pragma unroll
            for (int u = 0; u < 8; u += 2) {
                uint32_t b0, b1, b2, b3;
                int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                int ch = u + (lane >> 4);
                ldsm_x4_t(smem_u32(&V_[swz64(row, ch)]), b0, b1, b2, b3);
                mma_f32(oacc[u], pf[kc], b0, b1);
                mma_f32(oacc[u + 1], pf[kc], b2, b3);
            }
    }
    float l0 = l_r[0], l1 = l_r[1];
    l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    float i0 = 1.f / l0, i1 = 1.f / l1;
    int r0 = q0 + warp * 16 + (lane >> 2);
    __half* ob = o + b * osb;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        int col = u * 8 + (lane & 3) * 2;
        if (r0 < Nq) *reinterpret_cast<uint32_t*>(ob + (long long)r0 * osn + col) = pack_h2(oacc[u][0] * i0, oacc[u][1] * i0);
        if (r0 + 8 < Nq) *reinterpret_cast<uint32_t*>(ob + (long long)(r0 + 8) * osn + col) = pack_h2(oacc[u][2] * i1, oacc[u][3] * i1);
    }
}

#define ARGS const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o, int Nq, int Nk, \
             long long qsb, int qsn, long long ksb, int ksn, long long vsb, int vsn, long long osb, int osn, float scale_log2
#define CALL q, k, v, o, Nq, Nk, qsb, qsn, ksb, ksn, vsb, vsn, osb, osn, scale_log2
extern "C" __global__ void __launch_bounds__(128) flash_k256v64_ff(ARGS) { flash_k256_impl<false>(CALL); }
extern "C" __global__ void __launch_bounds__(128) flash_k256v64_hf(ARGS) { flash_k256_impl<true>(CALL); }
