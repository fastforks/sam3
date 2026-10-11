"""Fast SAM 3.1 (Object Multiplex) video, detector only: fast ViT-L trunk, fp16 necks and heads; the tracker runs as upstream.
Applied by Sam3MultiplexVideoPredictor.

    FastSam3Video31(vm)                         # patches the model in place
Plain PyTorch ops only."""
import torch

from .fast_vit import FastViT


class FastSam3Video31:
    def __init__(self, vm):
        self.vm = vm
        det = vm.detector
        neck = det.backbone.vision_backbone           # Sam3TriViTDetNeck: trunk + 3 conv stacks (sam3 / interactive / propagation)
        self.neck = neck
        self.vit = FastViT(neck.trunk)
        for convs in (neck.convs, neck.interactive_convs, neck.propagation_convs):
            convs.half()
            convs.to(memory_format=torch.channels_last)
        neck.trunk.forward = lambda x: [self.vit(x.half())]
        orig = neck.forward

        def forward(*a, **k):
            with torch.autocast("cuda", enabled=False):
                return orig(*a, **k)
        neck.forward = forward
        # detector heads: fp16 weights (encoder, segmentation head; the decoder stays fp32 as in sam3.fast), fp16 autocast around the grounding chunk
        for mod in (det.transformer.encoder, det.segmentation_head):
            mod.half()
        det.segmentation_head.to(memory_format=torch.channels_last)
        fg = det._process_grounding_chunk_batched

        def chunk(*a, **k):
            with torch.autocast("cuda", dtype=torch.float16):
                return fg(*a, **k)
        det._process_grounding_chunk_batched = chunk
        torch.cuda.empty_cache()
