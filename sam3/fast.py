"""Fast inference path for the SAM 3 image model, used by Sam3Processor and the video predictors.

Inference only: converts the vision / text backbones, the transformer encoder and the segmentation head to fp16 in
place (the decoder stays fp32). Any model resolution (multiple of 14; 1008 = training size), any input image size,
nothing compiled per size. Plain PyTorch ops (the ONNX exporter covers the fixed-resolution trunk + neck): ViT residual
stream in window order, SDPA, lean text encoder, GPU preprocessing.
"""

import numpy as np
import PIL
import torch
import torch.nn.functional as F

from .fast_text import FastText
from .fast_vit import FastViT


def uses_compile_or_fa3(model):
    """torch.compile or FlashAttention 3 (fp8, Hopper) was requested at build time: the video predictors keep the
    upstream path then (the fast path replaces the compiled / FA3 modules and was not measured against them)"""
    return getattr(model, "compile_model", False) or any(getattr(m, "use_fa3", False) for m in model.modules())


class FastImageModel:
    @classmethod
    def of(cls, model, **kw):
        """the fast path of `model`, converting it on first use (several processors may share one model)"""
        if getattr(model, "_fast", None) is None:
            model._fast = cls(model, **kw)
        return model._fast

    def __init__(self, model, text_trim=True):
        self.m = model.eval()
        bb = model.backbone
        neck = bb.vision_backbone
        for mod in (neck, bb.language_backbone, model.transformer.encoder, model.segmentation_head):
            mod.half()  # the decoder (200 queries) stays fp32 as upstream (its FFN runs outside autocast, residuals fp32)
        self.vit = FastViT(neck.trunk)
        self.neck = neck
        self.pe_cache = {}
        self.ftext = FastText(bb.language_backbone) if text_trim else None
        neck.convs.to(memory_format=torch.channels_last)
        if neck.sam2_convs is not None:
            neck.sam2_convs.to(memory_format=torch.channels_last)
        model.segmentation_head.to(memory_format=torch.channels_last)
        bb.forward_image = self._forward_image
        self.pin = {}
        torch.cuda.empty_cache()  # drop the freed fp32 blocks of the original weights (peak reserved = what the model needs)

    # ---------------- vision backbone ----------------
    def _pos(self, x):
        k = (x.shape[-2], x.shape[-1])
        if k not in self.pe_cache:
            pe = self.neck.position_encoding
            self.pe_cache[k] = pe(x[:1]).to(torch.float16)
        return self.pe_cache[k]

    def _forward_image(self, samples):
        with torch.autocast("cuda", enabled=False):
            f = self.vit(samples.half())
            outs, pos = [], []
            for conv in self.neck.convs[:-1]:  # scalp=1: the lowest-resolution level is computed and dropped upstream
                o = conv(f)
                outs.append(o)
                pos.append(self._pos(o).expand(o.shape[0], -1, -1, -1))
        sam2 = None
        if self.neck.sam2_convs is not None:  # second (tracker) neck on the same trunk output
            with torch.autocast("cuda", enabled=False):
                o2 = [conv(f) for conv in self.neck.sam2_convs[:-1]]
            sam2 = {"vision_features": o2[-1], "vision_pos_enc": pos, "backbone_fpn": o2}
        return {"vision_features": outs[-1], "vision_pos_enc": pos, "backbone_fpn": outs, "sam2_backbone_out": sam2}

    def forward_text(self, prompt, device):
        with torch.autocast("cuda", dtype=torch.float16):
            if self.ftext is not None:
                return self.ftext(prompt, device)
            return self.m.backbone.forward_text([prompt], device=device)

    # ---------------- preprocessing ----------------
    def preprocess(self, image, resolution, device):
        """PIL image, HxWx3 numpy or [3,H,W] tensor (uint8 or floats in [0,1]) -> (normalised fp16 [1,3,R,R], H, W)"""
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
        if isinstance(image, np.ndarray) and np.issubdtype(image.dtype, np.floating):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            key = image.shape
            buf, ev = self.pin.get(key) or (torch.empty(image.shape, dtype=torch.uint8).pin_memory(), None)
            if ev is not None:
                ev.synchronize()  # the previous async upload from this staging buffer has finished
            np.copyto(buf.numpy(), image)
            t = buf.to(device, non_blocking=True).permute(2, 0, 1)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(device))
            self.pin = {key: (buf, ev)}  # only the latest image size keeps a staging buffer
        elif isinstance(image, torch.Tensor):
            t = image.to(device)
            if t.is_floating_point():  # [0, 1] floats, as torchvision's ToDtype(uint8, scale=True)
                t = t.mul(255.999).to(torch.uint8)
            h, w = t.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")
        x = F.interpolate(t[None].float(), size=(resolution, resolution), mode="bilinear", antialias=True,
                          align_corners=False)
        x = x.round_().clamp_(0, 255).mul_(2.0 / 255).sub_(1.0)  # uint8 resize, then /255, (x - .5) / .5
        return x.half(), h, w
