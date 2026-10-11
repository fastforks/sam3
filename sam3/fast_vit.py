"""Fast SAM 3 ViT-L trunk for inference: fp16 weights, the residual stream kept in window order between the global
blocks, real-arithmetic 2D RoPE from cached tables, any token grid (padding windows handled as upstream), nothing
compiled per size. Plain PyTorch ops only (exports to ONNX / TensorRT).

attn="sdpa" (default) or "explicit": MatMul - Softmax - MatMul in fp16 with the softmax scale folded into the q rows of
the qkv weights (call fold_q_scale() first), the pattern TensorRT fuses into one attention kernel. An SDPA export is
decomposed with fp32 casts and runs slower than eager under TensorRT.

kernels=True (cuda-kernels branch): attn="hh" vendored fastbb flash attention with fp16 accumulation (or "fh" / "ff"),
Triton RoPE applied in place on the qkv GEMM output, cuBLASLt bias + GELU epilogue for fc1; calibrate() switches the
trunk GEMMs to fp8 (e4m3, static per-tensor scales) with Triton LayerNorm / GELU / residual / quantisation kernels."""
import torch
import torch.nn.functional as F


def rope_tables(h, w, head_dim, theta, scale_pos, device):
    """cos/sin [h*w, head_dim//2] float32, identical to upstream compute_axial_cis (x fastest-varying, then y)."""
    q = head_dim // 4
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4)[:q].float() / head_dim))
    t = torch.arange(h * w, dtype=torch.float32)
    t_x = (t % w).float() * scale_pos
    t_y = torch.div(t, w, rounding_mode="floor").float() * scale_pos
    f = torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], dim=-1)
    return f.cos().to(device), f.sin().to(device)


class FastViT:
    def __init__(self, vit, ws=24, attn="sdpa", kernels=False):
        self.vit, self.attn, self.kernels = vit, attn, kernels
        vit.half()
        self.blocks = list(vit.blocks)
        self.C = vit.blocks[0].norm1.weight.shape[0]
        a0 = vit.blocks[0].attn
        self.nh, self.hd = a0.num_heads, a0.head_dim
        self.theta = a0.rope_theta
        self.ws = ws
        self.glob = set(vit.full_attn_ids)
        self.eps = vit.blocks[0].norm1.eps
        self._cache = {}
        self.pt = a0.rope_pt_size[0] if a0.rope_pt_size else ws  # RoPE pretraining (window) size = 24
        self.rec = None  # fp8 calibration record
        self.fp8 = None
        if attn in ("hh", "fh", "ff"):
            from .fastbb_vendored.attention import attention as _fa
            self._fa = _fa

    def fold_q_scale(self):
        """explicit-attention mode (TensorRT export): fold the 1/sqrt(64) = 1/8 softmax scale (exact in fp16) into the q rows of
        every qkv projection; RoPE is a rotation, so scaling q before it is equivalent"""
        for blk in self.blocks:
            a = blk.attn.qkv
            a.weight.data[: self.C] *= self.hd ** -0.5
            a.bias.data[: self.C] *= self.hd ** -0.5

    # ---- per-size cached tensors ----
    def _tabs(self, h, w, device):
        k = (h, w)
        if k in self._cache:
            return self._cache[k]
        ws = self.ws
        Hp, Wp = -(-h // ws) * ws, -(-w // ws) * ws
        # window RoPE: 24x24 window, positions 0..23; global RoPE interpolated: positions * (pt / max(h, w))
        cw, sw = rope_tables(ws, ws, self.hd, self.theta, 1.0, device)
        cg, sg = rope_tables(h, w, self.hd, self.theta, self.pt / max(h, w) if max(h, w) != self.pt else 1.0, device)

        def mk(c, s):  # [N, 1, 32, 2]: [cos, cos] / [-sin, +sin], broadcast over (B, q|k, heads)
            cd = torch.stack([c, c], -1).half().unsqueeze(1)
            sd = torch.stack([-s, s], -1).half().unsqueeze(1)
            return cd, sd, c, s

        pos = self.vit.pos_embed.half()[:, 1:].reshape(1, 24, 24, -1).permute(0, 3, 1, 2)
        pos = pos.tile(1, 1, h // 24 + 1, w // 24 + 1)[:, :, :h, :w].permute(0, 2, 3, 1).contiguous()  # tiled abs pos
        mask = None
        if Hp != h or Wp != w:
            m = torch.zeros(1, Hp, Wp, 1, device=device, dtype=torch.float16)
            m[:, :h, :w] = 1
            mask = m.view(1, Hp // ws, ws, Wp // ws, ws, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, 1)
        t = dict(win=mk(cw, sw), glob=mk(cg, sg), pos=pos, mask=mask, Hp=Hp, Wp=Wp)
        t["win"] += (cw.float().contiguous(), sw.float().contiguous())  # fp32 cos / sin for the Triton RoPE kernel
        t["glob"] += (cg.float().contiguous(), sg.float().contiguous())
        self._cache[k] = t
        return t

    # ---- ops ----
    def _lin(self, x, lin, act=None, res=None):
        if act == "gelu" and self.kernels:  # cuBLASLt bias + GELU(tanh) epilogue, as upstream's perflib addmm_act
            y = torch._addmm_activation(lin.bias, x.view(-1, x.shape[-1]), lin.weight.t(), use_gelu=True)
            return y.view(*x.shape[:-1], -1)
        y = F.linear(x, lin.weight, lin.bias)
        if act == "gelu":
            y = F.gelu(y)
        return y if res is None else res + y

    def _rope(self, qkv, tab):
        """rotate q and k of qkv [B, N, 3, H, 64] -> [B, N, 2, H, 64] (pairs (even, odd) as complex numbers)"""
        cd, sd = tab[0], tab[1]
        B, N = qkv.shape[:2]
        qk = qkv[:, :, :2].view(B, N, 2, self.nh, self.hd // 2, 2)
        c = cd.view(1, N, 1, 1, self.hd // 2, 2)
        s = sd.view(1, N, 1, 1, self.hd // 2, 2)
        if self.attn == "explicit":  # export mode: the ONNX addcmul multiplies by an fp32 constant (promotes to fp32)
            out = qk * c + qk.flip(-1) * s
        else:
            out = torch.addcmul(qk * c, qk.flip(-1), s)
        return out.view(B, N, 2, self.nh, self.hd)

    def _attention(self, q, k, v):
        """q, k, v [B, N, H, 64] -> [B, N, H*64]"""
        B, N = q.shape[:2]
        if self.attn == "explicit":  # q scale folded into the qkv weights (fold_q_scale)
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            o = torch.matmul(torch.softmax(torch.matmul(qt, kt.transpose(-1, -2)), dim=-1), vt)
            return o.transpose(1, 2).reshape(B, N, -1)
        # (cast: the ONNX exporter types SDPA as fp32)
        if self.attn != "sdpa":
            return self._fa(q, k, v, variant=self.attn).view(B, N, -1)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).to(q.dtype)
        return o.transpose(1, 2).reshape(B, N, -1)

    def _block(self, blk, x, tab, mask):
        a = blk.attn
        y = F.layer_norm(x, (self.C,), blk.norm1.weight, blk.norm1.bias, self.eps)
        if mask is not None:
            y = y * mask
        B, N, _ = y.shape
        if self.rec is not None:
            self._record("qkv_in", y)
        qkv = self._lin(y, a.qkv).view(B, N, 3, self.nh, self.hd)
        if self.kernels:
            from .fast_kernels import rope_inplace
            rope_inplace(qkv.view(B * N, -1), tab[4], tab[5], N)
            o = self._attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        else:
            qk = self._rope(qkv, tab)
            o = self._attention(qk[:, :, 0], qk[:, :, 1], qkv[:, :, 2])
        if self.rec is not None:
            self._record("proj_in", o)
        x = self._lin(o, a.proj, res=x)
        y = F.layer_norm(x, (self.C,), blk.norm2.weight, blk.norm2.bias, self.eps)
        if self.rec is not None:
            self._record("fc1_in", y)
        h = self._lin(y, blk.mlp.fc1, act="gelu")
        if self.rec is not None:
            self._record("fc2_in", h)
        return self._lin(h, blk.mlp.fc2, res=x)

    # ---- fp8 (e4m3, per-tensor static activation scales from a calibration pass, per-tensor weight scales) ----
    def _record(self, name, t):
        k = (self._cur, name)
        a = t.detach().abs().amax().float()
        self.rec[k] = torch.maximum(self.rec[k], a) if k in self.rec else a

    def calibrate(self, imgs, margin=1.0):
        """imgs: list of normalised fp16 [1,3,R,R] tensors; records the max |input| of every ViT GEMM, then converts the
        weights to fp8 (the fp16 copies are dropped). The fp8 path is used afterwards."""
        self.fp8 = None
        self.rec = {}
        for im in imgs:
            self(im)
        rec = {k: float(v) * margin / 448.0 for k, v in self.rec.items()}
        self.rec = None
        dev = self.blocks[0].norm1.weight.device
        f8 = []
        for i, blk in enumerate(self.blocks):
            d = {}
            for nm, lin in (("qkv", blk.attn.qkv), ("proj", blk.attn.proj), ("fc1", blk.mlp.fc1), ("fc2", blk.mlp.fc2)):
                sw = lin.weight.detach().abs().amax().float() / 448.0
                d[nm] = (lin.weight.detach().float().div(sw).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous(), sw.reshape(()), lin.bias.detach())
            d["sa"] = {nm: rec[(i, k)] for nm, k in (("qkv", "qkv_in"), ("proj", "proj_in"), ("fc1", "fc1_in"), ("fc2", "fc2_in"))}
            d["sa_t"] = {nm: torch.tensor(v, device=dev, dtype=torch.float32) for nm, v in d["sa"].items()}
            f8.append(d)
        self.fp8 = f8
        for blk in self.blocks:  # free the fp16 GEMM weights (memory saving)
            for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
                lin.weight.data = torch.empty(0, device=dev, dtype=torch.float16)

    def _mm8(self, q, i, nm):
        w, sw, bias = self.fp8[i][nm]
        return torch._scaled_mm(q, w.t(), scale_a=self.fp8[i]["sa_t"][nm], scale_b=sw, out_dtype=torch.float16)  # bias is folded into the next Triton kernel (the bias epilogue is a slower kernel)

    def _run8(self, x, t, part, unpart, mask_b, h, w, B, C):
        from .fast_kernels import add_ln_quant, gelu_quant, quant, rope_inplace
        in_win, pend = False, None
        for i, blk in enumerate(self.blocks):
            g = i in self.glob
            if g == in_win:      # layout switch needed
                if pend is not None:
                    x = x + pend + self.fp8[i - 1]["fc2"][2]
                    pend = None
                if in_win:
                    x = unpart(x)
                    in_win = False
                else:
                    x = part(x.view(B, h, w, C))
                    in_win = True
            tab = t["glob"] if g else t["win"]
            mask = mask_b.view(-1) if (not g and mask_b is not None) else None
            Bq, N, _ = x.shape
            x2 = x.view(-1, C)
            sa = self.fp8[i]["sa"]
            q1 = add_ln_quant(x2, pend.view(-1, C) if pend is not None else None, blk.norm1.weight, blk.norm1.bias, 1.0 / sa["qkv"], self.eps, mask,
                              ybias=self.fp8[i - 1]["fc2"][2] if pend is not None else None)
            qkv = self._mm8(q1, i, "qkv")
            rope_inplace(qkv, tab[4], tab[5], N, bias=self.fp8[i]["qkv"][2])
            qkv = qkv.view(Bq, N, 3, self.nh, self.hd)
            o = self._attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
            y = self._mm8(quant(o.view(-1, C), 1.0 / sa["proj"]), i, "proj")
            q2 = add_ln_quant(x2, y, blk.norm2.weight, blk.norm2.bias, 1.0 / sa["fc1"], self.eps, ybias=self.fp8[i]["proj"][2])
            hh = self._mm8(q2, i, "fc1")
            pend = self._mm8(gelu_quant(hh, 1.0 / sa["fc2"], self.fp8[i]["fc1"][2]), i, "fc2").view(Bq, N, C)
        x = x + pend + self.fp8[-1]["fc2"][2]
        return unpart(x) if in_win else x

    # ---- forward ----
    def __call__(self, img):
        """img [B,3,H,W] fp16 normalised -> [B, C, h, w] fp16 (stride 14 grid)"""
        v = self.vit
        x = F.conv2d(img, v.patch_embed.proj.weight, None, stride=14).permute(0, 2, 3, 1)  # [B,h,w,C]
        B, h, w, C = x.shape
        t = self._tabs(h, w, x.device)
        x = x + t["pos"]
        x = F.layer_norm(x, (C,), v.ln_pre.weight, v.ln_pre.bias, self.eps)
        ws, Hp, Wp = self.ws, t["Hp"], t["Wp"]
        nW = (Hp // ws) * (Wp // ws)

        def part(z):  # [B,h,w,C] -> [B*nW, ws*ws, C]
            if Hp != h or Wp != w:
                z = F.pad(z, (0, 0, 0, Wp - w, 0, Hp - h))
            return z.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B * nW, ws * ws, C)

        def unpart(z):  # -> [B, h*w, C]
            z = z.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
            return z[:, :h, :w].reshape(B, h * w, C)

        mask = t["mask"]
        mask_b = mask.repeat(B, 1, 1) if (mask is not None and B > 1) else mask
        if self.fp8 is not None:
            x = self._run8(x.reshape(B, h * w, C), t, part, unpart, mask_b, h, w, B, C)
            return x.view(B, h, w, C).permute(0, 3, 1, 2)
        in_win = False
        for i, blk in enumerate(self.blocks):
            self._cur = i
            if i in self.glob:
                if in_win:
                    x = unpart(x)
                    in_win = False
                elif i == 0:
                    x = x.reshape(B, h * w, C)
                x = self._block(blk, x, t["glob"], None)
            else:
                if not in_win:
                    x = part(x.view(B, h, w, C))
                    in_win = True
                x = self._block(blk, x, t["win"], mask_b)
        if in_win:
            x = unpart(x)
        return x.view(B, h, w, C).permute(0, 3, 1, 2)
