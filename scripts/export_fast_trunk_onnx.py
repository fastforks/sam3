"""Export the fast SAM 3 ViT-L trunk + FPN neck (sam3.fast, plain PyTorch, fp16) to ONNX at a fixed resolution, for a
TensorRT fp16 engine; the rest of the model runs in PyTorch.

    python scripts/export_fast_trunk_onnx.py --checkpoint sam3.pt --out sam3_trunk_1008.onnx [--res 1008]

Inputs: img [1, 3, res, res] fp16 (normalised as FastSam3._pre); outputs f0, f1, f2 = the three FPN levels used by the
model. Attention is exported as MatMul - Softmax - MatMul (FastViT attn="explicit"), the pattern TensorRT fuses."""
import argparse
import os
import time

import torch
import torch.nn as nn

from sam3.fast import FastSam3


class TrunkNeck(nn.Module):
    def __init__(self, fs):
        super().__init__()
        self.vit, self.convs = fs.vit, fs.neck.convs[:-1]

    def forward(self, x):
        f = self.vit(x)
        return tuple(c(f) for c in self.convs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1008)
    a = ap.parse_args()
    fs = FastSam3.load(a.checkpoint, resolution=a.res)
    fs.m.requires_grad_(False)
    fs.vit.attn = "explicit"
    fs.vit.fold_q_scale()
    net = TrunkNeck(fs).eval()
    x = torch.randn(1, 3, a.res, a.res, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        ref = net(x)
        tabs = fs.vit._cache[(a.res // 14, a.res // 14)]
        fs.vit._tabs = lambda h, w, d: tabs  # RoPE / abs-pos tables as constants: no table maths in the graph
        t = time.time()
        torch.onnx.export(net, (x,), a.out, opset_version=18, input_names=["img"], output_names=["f0", "f1", "f2"], dynamo=False)
    print(f"exported {a.out} ({os.path.getsize(a.out) / 2**20:.0f} MB, {time.time() - t:.1f} s), outputs {[tuple(r.shape) for r in ref]}")


if __name__ == "__main__":
    main()
