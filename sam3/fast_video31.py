"""Fast SAM 3.1 (Object Multiplex) video, detector only: fast ViT-L trunk, fp16 necks and heads; the tracker runs as upstream.

    FastSam3Video31(vm)                         # patches the model in place; use it via the upstream predictor
kernels=True (default here): flash attention / Triton RoPE in the trunk, fp16-accumulate GEMMs and neck convolutions,
Triton GroupNorm in the pixel decoder; kernels=False: plain PyTorch ops only."""
import torch

from .fast import FastSam3, fp16_accumulate
from .fast_vit import FastViT


class FastSam3Video31:
    def __init__(self, vm, kernels=True):
        self.vm, self.kernels = vm, kernels
        det = vm.detector
        neck = det.backbone.vision_backbone           # Sam3TriViTDetNeck: trunk + 3 conv stacks (sam3 / interactive / propagation)
        self.neck = neck
        self.vit = FastViT(neck.trunk, attn="hh" if kernels else "sdpa", kernels=kernels)
        for convs in (neck.convs, neck.interactive_convs, neck.propagation_convs):
            convs.half()
            convs.to(memory_format=torch.channels_last)
            if kernels:
                from .fastbb_vendored.cudnn_conv import patch_convs
                patch_convs(convs)
        neck.trunk.forward = lambda x: [self.vit(x.half())]
        orig = neck.forward

        def forward(*a, **k):
            with torch.autocast("cuda", enabled=False), fp16_accumulate(kernels):
                return orig(*a, **k)
        neck.forward = forward
        # detector heads: fp16 weights (encoder, segmentation head; the decoder stays fp32 as in FastSam3), fp16 autocast around the grounding chunk
        for mod in (det.transformer.encoder, det.segmentation_head):
            mod.half()
        det.segmentation_head.to(memory_format=torch.channels_last)
        if kernels:
            h = FastSam3.__new__(FastSam3)
            h.m = det
            h._fast_pixel_decoder()
        fg = det._process_grounding_chunk_batched

        def chunk(*a, **k):
            with torch.autocast("cuda", dtype=torch.float16), fp16_accumulate(kernels):
                return fg(*a, **k)
        det._process_grounding_chunk_batched = chunk
        torch.cuda.empty_cache()
