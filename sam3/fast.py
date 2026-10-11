"""Fast inference for the SAM 3 image model (text / exemplar-box concept segmentation).

    from sam3.fast import FastSam3
    fs = FastSam3.load("sam3.pt")                 # or FastSam3(build_sam3_image_model(...), resolution=1008)
    out = fs(image_uint8_HxWx3, "person")         # dict: masks (n,H,W) bool numpy, boxes (n,4) xyxy px, scores (n,)
    state = fs.set_image(image)                   # several prompts on the same image
    out = fs.predict(state, "car")

Any model resolution (multiple of 14; 1008 = training size), any input image size, nothing compiled per size.
Plain PyTorch ops only (exports to ONNX / TensorRT): fp16 weights, ViT residual stream in window order, SDPA, lean text
encoder, GPU preprocessing, no GPU -> host syncs before the outputs.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .fast_text import FastText
from .fast_vit import FastViT
from .model import box_ops


class FastSam3:
    def __init__(self, model, resolution=1008, conf=0.5, text_trim=True, text_cache=False, channels_last=True):
        self.m = model.eval()
        self.res = resolution
        self.conf = conf
        self.text_trim = text_trim
        self.dev = next(model.parameters()).device
        bb = model.backbone
        neck = bb.vision_backbone
        for mod in (neck, bb.language_backbone, model.transformer.encoder, model.segmentation_head):
            mod.half()  # the decoder (200 queries) stays fp32 as upstream (its FFN runs outside autocast, residuals fp32)
        self.vit = FastViT(neck.trunk)
        self.neck = neck
        self.pe_cache = {}
        self.ftext = FastText(bb.language_backbone, cache=text_cache) if text_trim else None
        if channels_last:
            neck.convs.to(memory_format=torch.channels_last)
            if neck.sam2_convs is not None:
                neck.sam2_convs.to(memory_format=torch.channels_last)
            model.segmentation_head.to(memory_format=torch.channels_last)
        self.m.backbone.forward_image = self._forward_image
        self.pin = {}
        torch.cuda.empty_cache()  # drop the freed fp32 blocks of the original weights (peak reserved = what the model needs)

    @classmethod
    def load(cls, ckpt, **kw):
        """build + load the checkpoint with parameters created directly on the GPU (8.4 s -> 2.1 s: the default builder initialises
        848M parameters on the CPU), then wrap"""
        from . import model_builder as mb
        with torch.device("cuda"):
            m = mb.build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False)
        return cls(m, **kw)

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
        if self.neck.sam2_convs is not None:  # video detector: second (tracker) neck on the same trunk output
            with torch.autocast("cuda", enabled=False):
                o2 = [conv(f) for conv in self.neck.sam2_convs[:-1]]
            sam2 = {"vision_features": o2[-1], "vision_pos_enc": pos, "backbone_fpn": o2}
        return {"vision_features": outs[-1], "vision_pos_enc": pos, "backbone_fpn": outs, "sam2_backbone_out": sam2}

    # ---------------- pre / post ----------------
    def _pre(self, im):
        """im: uint8 HxWx3 numpy (or torch [3,H,W] uint8 on the GPU) -> normalised fp16 [1,3,R,R]"""
        if isinstance(im, np.ndarray):
            h, w = im.shape[:2]
            key = im.shape
            buf, ev = self.pin.get(key) or (torch.empty(im.shape, dtype=torch.uint8).pin_memory(), None)
            if ev is not None:
                ev.synchronize()          # the previous async upload from this staging buffer has finished
            buf.copy_(torch.from_numpy(im))
            t = buf.to(self.dev, non_blocking=True).permute(2, 0, 1)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.dev))
            self.pin = {key: (buf, ev)}  # only the latest image size keeps a staging buffer
        else:
            t = im
            h, w = t.shape[-2:]
        x = F.interpolate(t[None].float(), size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        x = x.round_().clamp_(0, 255).mul_(2.0 / 255).sub_(1.0)  # uint8 resize, then /255, (x - .5) / .5
        return x.half(), h, w

    @torch.inference_mode()
    def set_image(self, im):
        """image (uint8 HxWx3 numpy, or [3,H,W] uint8 GPU tensor) -> state (vision features); reuse it for several prompts"""
        x, H, W = self._pre(im)
        with torch.autocast("cuda", dtype=torch.float16):
            bo = self.m.backbone.forward_image(x)
        return dict(bo=bo, H=H, W=W)

    def __call__(self, im, prompt, boxes=None, labels=None, to_host=True):
        return self.predict(self.set_image(im), prompt, boxes, labels, to_host)

    @torch.inference_mode()
    def predict(self, state, prompt, boxes=None, labels=None, to_host=True):
        """text prompt (and / or exemplar boxes [[cx, cy, w, h] normalised], labels [bool], default
        all positive; prompt "visual" for boxes only)"""
        m, H, W = self.m, state["H"], state["W"]
        bo = dict(state["bo"])
        with torch.autocast("cuda", dtype=torch.float16):
            txt = self.ftext(prompt, self.dev) if self.ftext else m.backbone.forward_text([prompt], device=self.dev)
            bo.update(txt)
            gp = m._get_dummy_prompt()
            if boxes is not None:
                gp.append_boxes(torch.as_tensor(boxes, device=self.dev, dtype=torch.float32).view(-1, 1, 4),
                                torch.as_tensor(True if labels is None else labels, device=self.dev,
                                                dtype=torch.bool).expand(len(boxes)).reshape(-1, 1))
            from .model.data_misc import FindStage
            fs = FindStage(img_ids=torch.zeros(1, device=self.dev, dtype=torch.long),
                           text_ids=torch.zeros(1, device=self.dev, dtype=torch.long), input_boxes=None,
                           input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
            o = m.forward_grounding(backbone_out=bo, find_input=fs, geometric_prompt=gp, find_target=None)
        logits = o["pred_logits"].float()
        probs = (logits.sigmoid() * o["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)
        keep = probs > self.conf
        probs, masks, bb = probs[keep], o["pred_masks"][keep], o["pred_boxes"][keep].float()
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(bb) * torch.tensor([W, H, W, H], device=self.dev)
        up = F.interpolate(masks.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0
        if not to_host:
            return dict(masks=up, boxes=boxes_xyxy, scores=probs)
        n = up.shape[0]
        mh = torch.empty((n, H, W), dtype=torch.uint8, pin_memory=True)
        mh.copy_(up.view(torch.uint8), non_blocking=True)
        bs = torch.stack([probs, *boxes_xyxy.unbind(-1)], -1).cpu()
        torch.cuda.current_stream().synchronize()
        return dict(masks=mh.numpy().view(bool), boxes=bs[:, 1:].numpy(), scores=bs[:, 0].numpy())
