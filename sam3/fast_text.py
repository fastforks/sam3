"""Lean SAM 3 text encoder forward: plain PyTorch, fp16, only the real tokens (the causal mask makes the padded
positions irrelevant; they are masked in every consumer), optional per-prompt feature cache."""
import torch
import torch.nn.functional as F


class FastText:
    def __init__(self, vl, cache=False):
        self.vl = vl                       # VETextEncoder (backbone.language_backbone)
        self.enc = vl.encoder
        self.cache = {} if cache else None
        self.tok = vl.tokenizer
        self.ctx = vl.context_length

    def __call__(self, prompt, device):
        if self.cache is not None and prompt in self.cache:
            return self.cache[prompt]
        e = self.enc
        tk = self.tok([prompt], context_length=self.ctx)           # [1, 32] cpu
        T = int(tk[0].nonzero()[-1, 0]) + 1                        # sot + words + eot (causal: later positions never matter)
        tk = tk[:, :T].to(device, non_blocking=True)
        x = F.embedding(tk, e.token_embedding.weight)
        embeds = x
        x = x + e.positional_embedding[:T]
        nh = e.heads
        for blk in e.transformer.resblocks:
            y = F.layer_norm(x, (x.shape[-1],), blk.ln_1.weight, blk.ln_1.bias, blk.ln_1.eps)
            a = blk.attn
            qkv = F.linear(y, a.in_proj_weight, a.in_proj_bias).view(1, T, 3, nh, -1)
            q, k, v = qkv.unbind(2)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + F.linear(o.transpose(1, 2).reshape(1, T, -1), a.out_proj.weight, a.out_proj.bias)
            y = F.layer_norm(x, (x.shape[-1],), blk.ln_2.weight, blk.ln_2.bias, blk.ln_2.eps)
            h = F.gelu(F.linear(y, blk.mlp.c_fc.weight, blk.mlp.c_fc.bias))
            x = x + F.linear(h, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias)
        x = F.layer_norm(x, (x.shape[-1],), e.ln_final.weight, e.ln_final.bias, e.ln_final.eps)
        mem = F.linear(x, self.vl.resizer.weight, self.vl.resizer.bias).transpose(0, 1)       # [T, 1, 256]
        out = {"language_features": mem, "language_mask": tk == 0,
               "language_embeds": embeds.transpose(0, 1)}
        if self.cache is not None:
            self.cache[prompt] = out
        return out
