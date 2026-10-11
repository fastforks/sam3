"""Triton kernels for FastSam3(kernels=True) (not exportable; the portable path uses plain torch ops)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inplace(qkv_ptr, cos_ptr, sin_ptr, bias_ptr, N, ROW: tl.constexpr, H: tl.constexpr, HALF: tl.constexpr, HAS_BIAS: tl.constexpr):
    # one program per token; rotates the q and k pairs (even, odd) of the packed [3, H, 2*HALF] row in place
    t = tl.program_id(0)
    n = t % N
    j = tl.arange(0, 2 * H * HALF)           # pair index over (q|k, head, freq)
    which = j // (H * HALF)
    rem = j % (H * HALF)
    h = rem // HALF
    f = rem % HALF
    base = qkv_ptr + t.to(tl.int64) * ROW + which * (H * 2 * HALF) + h * (2 * HALF) + 2 * f
    a = tl.load(base).to(tl.float32)
    b = tl.load(base + 1).to(tl.float32)
    if HAS_BIAS:
        boff = which * (H * 2 * HALF) + h * (2 * HALF) + 2 * f
        a = (a + tl.load(bias_ptr + boff).to(tl.float32)).to(tl.float16).to(tl.float32)
        b = (b + tl.load(bias_ptr + boff + 1).to(tl.float32)).to(tl.float16).to(tl.float32)
    c = tl.load(cos_ptr + n * HALF + f)
    s = tl.load(sin_ptr + n * HALF + f)
    tl.store(base, (a * c - b * s).to(tl.float16))
    tl.store(base + 1, (a * s + b * c).to(tl.float16))
    if HAS_BIAS:  # v third of the row: bias only
        jv = tl.arange(0, H * 2 * HALF)
        pv = qkv_ptr + t.to(tl.int64) * ROW + 2 * (H * 2 * HALF) + jv
        tl.store(pv, (tl.load(pv).to(tl.float32) + tl.load(bias_ptr + 2 * (H * 2 * HALF) + jv).to(tl.float32)).to(tl.float16))


def rope_inplace(qkv, cos, sin, N, bias=None):
    """qkv [T, 3*H*64] fp16 contiguous, T = B*N tokens (position n = t % N); cos / sin [N, 32] fp32. Rotates q and k in place."""
    T, row = qkv.shape
    H = row // 3 // 64
    _rope_inplace[(T,)](qkv, cos, sin, bias if bias is not None else qkv, N, ROW=row, H=H, HALF=32, HAS_BIAS=bias is not None, num_warps=8)
    return qkv


@triton.jit
def _gn_stats(x_ptr, st_ptr, HW, C: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    # x NHWC [B, HW, C]; st [B, G, 2] fp32 (sum, sumsq); grid (G, cdiv(HW, BLOCK), B)
    g = tl.program_id(0)
    pb = tl.program_id(1)
    b = tl.program_id(2)
    CG: tl.constexpr = C // G
    rows = pb * BLOCK + tl.arange(0, BLOCK)
    cols = g * CG + tl.arange(0, CG)
    m = rows < HW
    offs = (b * HW + rows)[:, None].to(tl.int64) * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32)
    tl.atomic_add(st_ptr + (b * G + g) * 2, tl.sum(tl.sum(x, 1), 0))
    tl.atomic_add(st_ptr + (b * G + g) * 2 + 1, tl.sum(tl.sum(x * x, 1), 0))


@triton.jit
def _gn_apply(x_ptr, st_ptr, w_ptr, b_ptr, y_ptr, HW, n, eps, C: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr,
              RELU: tl.constexpr):
    pb = tl.program_id(0)
    b = tl.program_id(1)
    CG: tl.constexpr = C // G
    rows = pb * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, C)
    m = rows < HW
    g = cols // CG
    s = tl.load(st_ptr + (b * G + g) * 2)
    ss = tl.load(st_ptr + (b * G + g) * 2 + 1)
    mean = s / n
    rstd = 1.0 / tl.sqrt(ss / n - mean * mean + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    bb = tl.load(b_ptr + cols).to(tl.float32)
    offs = (b * HW + rows)[:, None].to(tl.int64) * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=m[:, None], other=0.0).to(tl.float32)
    y = (x - mean[None, :]) * (rstd * w)[None, :] + bb[None, :]
    if RELU:
        y = tl.maximum(y, 0.0)
    tl.store(y_ptr + offs, y.to(tl.float16), mask=m[:, None])


def group_norm_nhwc(x, weight, bias, groups, eps=1e-5, relu=False):
    """GroupNorm (+ReLU) for a channels_last fp16 tensor [B, C, H, W] (physically NHWC), two passes (stats, apply)."""
    B, C, H, W = x.shape
    x = x.contiguous(memory_format=torch.channels_last)
    HW = H * W
    st = torch.zeros((B, groups, 2), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    cg = C // groups
    BLK = 128
    _gn_stats[(groups, triton.cdiv(HW, BLK), B)](x, st, HW, C=C, G=groups, BLOCK=BLK, num_warps=4)
    _gn_apply[(triton.cdiv(HW, 32), B)](x, st, weight, bias, y, HW, float(HW * cg), eps, C=C, G=groups, BLOCK=32,
                                       RELU=relu, num_warps=4)
    return y


# ------------------------------------------------------------------------------------------------ fp8 (e4m3) activation kernels
@triton.jit
def _add_ln_quant(x_ptr, y_ptr, yb_ptr, w_ptr, b_ptr, m_ptr, q_ptr, inv_sa, eps, C: tl.constexpr, HAS_Y: tl.constexpr, HAS_M: tl.constexpr, HAS_YB: tl.constexpr):
    # x (+= y, in place) -> LayerNorm(x) * mask / sa -> fp8; one row per program
    r = tl.program_id(0)
    cols = tl.arange(0, C)
    off = r.to(tl.int64) * C + cols
    x = tl.load(x_ptr + off).to(tl.float32)
    if HAS_Y:
        yv = tl.load(y_ptr + off).to(tl.float32)
        if HAS_YB:
            yv = (yv + tl.load(yb_ptr + cols).to(tl.float32)).to(tl.float16).to(tl.float32)
        x = (x + yv).to(tl.float16).to(tl.float32)
        tl.store(x_ptr + off, x.to(tl.float16))
    mean = tl.sum(x, 0) / C
    d = x - mean
    var = tl.sum(d * d, 0) / C
    o = d * (1.0 / tl.sqrt(var + eps)) * tl.load(w_ptr + cols).to(tl.float32) + tl.load(b_ptr + cols).to(tl.float32)
    if HAS_M:
        o = o * tl.load(m_ptr + r).to(tl.float32)
    o = tl.minimum(tl.maximum(o * inv_sa, -448.0), 448.0)
    tl.store(q_ptr + off, o.to(tl.float8e4nv))


def add_ln_quant(x, y, w, b, inv_sa, eps, mask=None, out=None, ybias=None):
    """x [M, C] fp16 (updated in place with += y when y is given); returns fp8 LayerNorm(x) / sa (optionally times a per-row mask)"""
    M, C = x.shape
    q = out if out is not None else torch.empty((M, C), device=x.device, dtype=torch.float8_e4m3fn)
    _add_ln_quant[(M,)](x, y if y is not None else x, ybias if ybias is not None else x, w, b, mask if mask is not None else x, q,
                        inv_sa, eps, C=C, HAS_Y=y is not None, HAS_M=mask is not None, HAS_YB=ybias is not None, num_warps=4)
    return q


@triton.jit
def _gelu_quant(h_ptr, bias_ptr, q_ptr, n, NCOL, inv_sa, BLOCK: tl.constexpr, HAS_B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(h_ptr + i, mask=m, other=0.0).to(tl.float32)
    if HAS_B:
        x = (x + tl.load(bias_ptr + i % NCOL, mask=m, other=0.0).to(tl.float32)).to(tl.float16).to(tl.float32)
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e = tl.exp(2.0 * z)
    g = 0.5 * x * (1.0 + (1.0 - 2.0 / (e + 1.0)))   # tanh via exp (cuBLASLt GELU epilogue is the tanh form, as upstream)
    tl.store(q_ptr + i, tl.minimum(tl.maximum(g * inv_sa, -448.0), 448.0).to(tl.float8e4nv), mask=m)


def gelu_quant(h, inv_sa, bias=None):
    q = torch.empty(h.shape, device=h.device, dtype=torch.float8_e4m3fn)
    n = h.numel()
    _gelu_quant[(triton.cdiv(n, 4096),)](h, bias if bias is not None else h, q, n, h.shape[-1], inv_sa, BLOCK=4096,
                                         HAS_B=bias is not None, num_warps=4)
    return q


@triton.jit
def _quant(x_ptr, q_ptr, n, inv_sa, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(q_ptr + i, tl.minimum(tl.maximum(x * inv_sa, -448.0), 448.0).to(tl.float8e4nv), mask=m)


def quant(x, inv_sa):
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    n = x.numel()
    _quant[(triton.cdiv(n, 4096),)](x, q, n, inv_sa, BLOCK=4096, num_warps=4)
    return q


@triton.jit
def _rope_rows(x_ptr, tab_ptr, cos_ptr, sin_ptr, ROWSTRIDE, Ttot, N, NROPE, HAS_TAB: tl.constexpr):
    t = tl.program_id(0)
    j = t % Ttot
    i = tl.arange(0, 128)
    base = x_ptr + t.to(tl.int64) * ROWSTRIDE + 2 * i
    a = tl.load(base).to(tl.float32)
    b = tl.load(base + 1).to(tl.float32)
    if HAS_TAB:
        a += tl.load(tab_ptr + j * 256 + 2 * i).to(tl.float32)
        b += tl.load(tab_ptr + j * 256 + 2 * i + 1).to(tl.float32)
    if j < NROPE:
        n = j % N
        c = tl.load(cos_ptr + n * 128 + i)
        s = tl.load(sin_ptr + n * 128 + i)
        a, b = a * c - b * s, a * s + b * c
    tl.store(base, a.to(tl.float16))
    tl.store(base + 1, b.to(tl.float16))


def rope_rows(x, cos, sin, Ttot, N, nrope, tab=None):
    """x: [B*Ttot, 256] fp16 view (row stride arbitrary, last stride 1); in place: x[t] (+ tab[t % Ttot]) rotated for t % Ttot < nrope (position (t % Ttot) % N)"""
    T = x.shape[0]
    _rope_rows[(T,)](x, tab if tab is not None else x, cos, sin, x.stride(0), Ttot, N, nrope, HAS_TAB=tab is not None, num_warps=4)
