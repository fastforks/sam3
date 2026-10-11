"""Fast SAM 3 ViT-L trunk for inference: fp16 weights, the residual stream kept in window order between the global
blocks, real-arithmetic 2D RoPE from cached tables, any token grid (padding windows handled as upstream), nothing
compiled per size. Plain PyTorch ops only (exports to ONNX / TensorRT)."""
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
    def __init__(self, vit, ws=24):
        self.vit = vit
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
            return cd, sd

        pos = self.vit.pos_embed.half()[:, 1:].reshape(1, 24, 24, -1).permute(0, 3, 1, 2)
        pos = pos.tile(1, 1, h // 24 + 1, w // 24 + 1)[:, :, :h, :w].permute(0, 2, 3, 1).contiguous()  # tiled abs pos
        mask = None
        if Hp != h or Wp != w:
            m = torch.zeros(1, Hp, Wp, 1, device=device, dtype=torch.float16)
            m[:, :h, :w] = 1
            mask = m.view(1, Hp // ws, ws, Wp // ws, ws, 1).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, 1)
        t = dict(win=mk(cw, sw), glob=mk(cg, sg), pos=pos, mask=mask, Hp=Hp, Wp=Wp)
        self._cache[k] = t
        return t

    # ---- ops ----
    def _lin(self, x, lin, act=None, res=None):
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
        out = torch.addcmul(qk * c, qk.flip(-1), s)
        return out.view(B, N, 2, self.nh, self.hd)

    def _attention(self, q, k, v):
        """q, k, v [B, N, H, 64] -> [B, N, H*64]"""
        B, N = q.shape[:2]
        # (cast: the ONNX exporter types SDPA as fp32)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).to(q.dtype)
        return o.transpose(1, 2).reshape(B, N, -1)

    def _block(self, blk, x, tab, mask):
        a = blk.attn
        y = F.layer_norm(x, (self.C,), blk.norm1.weight, blk.norm1.bias, self.eps)
        if mask is not None:
            y = y * mask
        B, N, _ = y.shape
        qkv = self._lin(y, a.qkv).view(B, N, 3, self.nh, self.hd)
        qk = self._rope(qkv, tab)
        o = self._attention(qk[:, :, 0], qk[:, :, 1], qkv[:, :, 2])
        x = self._lin(o, a.proj, res=x)
        y = F.layer_norm(x, (self.C,), blk.norm2.weight, blk.norm2.bias, self.eps)
        h = self._lin(y, blk.mlp.fc1, act="gelu")
        return self._lin(h, blk.mlp.fc2, res=x)

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
        in_win = False
        for i, blk in enumerate(self.blocks):
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
