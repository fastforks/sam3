"""Fast SAM 3 tracker memory attention (the 4-layer RoPE transformer that conditions the current frame on the memory bank):
fp16, fused QKV / KV GEMMs, real-valued RoPE with cached tables, the memory positional term folded into one batch-independent
table, SDPA. Plain PyTorch ops only.

The memory-bank K projection is  Wk (mem + pos) + b = Wk mem + (Wk pos + b): the second term does not depend on the object,
so it is computed once per step (41.5k x 64 -> 256) instead of per object.
"""
import torch
import torch.nn.functional as F


class FastMemAttn:
    def __init__(self, enc, attn=None):
        self.attn_fn = attn or (lambda q, k, v: F.scaled_dot_product_attention(q, k, v))
        h = torch.float16
        self.layers = []
        for L in enc.layers:
            sa, ca = L.self_attn, L.cross_attn_image
            d = dict(
                n1=(L.norm1.weight.data.to(h), L.norm1.bias.data.to(h), L.norm1.eps),
                n2=(L.norm2.weight.data.to(h), L.norm2.bias.data.to(h), L.norm2.eps),
                n3=(L.norm3.weight.data.to(h), L.norm3.bias.data.to(h), L.norm3.eps),
                sa_qkv=(torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]).data.to(h),
                        torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias]).data.to(h)),
                sa_o=(sa.out_proj.weight.data.to(h), sa.out_proj.bias.data.to(h)),
                ca_q=(ca.q_proj.weight.data.to(h), ca.q_proj.bias.data.to(h)),
                ca_kv=(torch.cat([ca.k_proj.weight, ca.v_proj.weight]).data.to(h),
                       torch.cat([torch.zeros_like(ca.k_proj.bias), ca.v_proj.bias]).data.to(h)),
                ca_k=(ca.k_proj.weight.data.to(h), ca.k_proj.bias.data.to(h)),
                ca_o=(ca.out_proj.weight.data.to(h), ca.out_proj.bias.data.to(h)),
                l1=(L.linear1.weight.data.to(h), L.linear1.bias.data.to(h)),
                l2=(L.linear2.weight.data.to(h), L.linear2.bias.data.to(h)),
            )
            self.layers.append(d)
        self.norm = (enc.norm.weight.data.to(h), enc.norm.bias.data.to(h), enc.norm.eps)
        fc = enc.layers[0].self_attn.freqs_cis                      # [N, 128] complex64 (axial, 72 x 72)
        self.N = fc.shape[0]
        cos, sin = fc.real.contiguous().float(), fc.imag.contiguous().float()
        # plain-op tables: x * cos2 + swap(x) * sin2, interleaved pairs (even, odd)
        self.cos2 = cos.repeat_interleave(2, -1).to(h)               # [N, 256]
        sg = torch.stack([-sin, sin], -1).flatten(-2)
        self.sin2 = sg.to(h)

    # ----- rope (plain ops) -----
    def _rope_plain(self, x, nrope):
        """x [B, T, 256]; rotate the first nrope tokens with position t % N; returns new tensor"""
        B, T, C = x.shape
        N = self.N
        xr = x[:, :nrope].reshape(B, nrope // N, N, C)
        sw = xr.view(B, nrope // N, N, C // 2, 2).flip(-1).reshape(B, nrope // N, N, C)
        out = torch.addcmul(xr * self.cos2, sw, self.sin2).reshape(B, nrope, C)
        return out if nrope == T else torch.cat([out, x[:, nrope:]], 1)


    def __call__(self, src, prompt, prompt_pos, num_obj_ptr_tokens, src_pos):
        """src, src_pos: [N, B, 256] (seq first); prompt [M, B, 64]; prompt_pos [M, B, 64] (batch-independent). returns [N, B, 256] fp16"""
        h = torch.float16
        N, B, C = src.shape
        M = prompt.shape[0]
        nrope = M - num_obj_ptr_tokens
        x = (src.to(h) + 0.1 * src_pos.to(h)).transpose(0, 1).contiguous()         # [B, N, 256]
        mem = prompt.to(h).transpose(0, 1).contiguous().view(B * M, -1)              # [B*M, 64]
        pos = prompt_pos[:, 0].to(h)                                                 # [M, 64]
        for d in self.layers:
            # --- self attention
            y = F.layer_norm(x, (C,), *d["n1"][:2], d["n1"][2])
            qkv = F.linear(y, *d["sa_qkv"])                                           # [B, N, 768]
            q, k, v = qkv[..., :C], qkv[..., C:2 * C], qkv[..., 2 * C:]
            q, k = self._rope_plain(q, N), self._rope_plain(k, N)
            o = self.attn_fn(q[:, None], k[:, None], v[:, None])[:, 0]
            x = x + F.linear(o, *d["sa_o"])
            # --- cross attention to the memory bank
            y = F.layer_norm(x, (C,), *d["n2"][:2], d["n2"][2])
            q = F.linear(y, *d["ca_q"])
            q = self._rope_plain(q, N)
            kpos = F.linear(pos, *d["ca_k"])                                          # [M, 256]  Wk pos + bk  (shared by all objects)
            kv = F.linear(mem, *d["ca_kv"])                                           # [B*M, 512]  (k part: no bias; v part: + bv)
            k = kv[:, :C].view(B, M, C) + kpos
            k = self._rope_plain(k, nrope)
            v = kv[:, C:].view(B, M, C)
            o = self.attn_fn(q[:, None], k[:, None], v[:, None])[:, 0]
            x = x + F.linear(o, *d["ca_o"])
            # --- FFN
            y = F.layer_norm(x, (C,), *d["n3"][:2], d["n3"][2])
            x = x + F.linear(F.relu(F.linear(y, *d["l1"])), *d["l2"])
        out = F.layer_norm(x, (C,), *self.norm[:2], self.norm[2])
        return out.transpose(0, 1)                                                    # [N, B, 256]
