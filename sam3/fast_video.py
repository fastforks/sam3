"""Fast SAM 3 video (dense tracking: text-prompted detection + SAM 2-style tracker).

    vm = build_sam3_video_model(...)            # Sam3VideoInferenceWithInstanceInteractivity
    FastSam3Video(vm)                           # patches the model in place; use it via Sam3VideoPredictor as upstream
Detector: the FastSam3 image path (trunk, both necks, encoder, heads); tracker: FastMemAttn memory attention and an fp16
memory encoder. Plain PyTorch ops only.
"""
import torch
import torch.nn.functional as F

from .fast import FastSam3
from .fast_memattn import FastMemAttn
from .fast_text import FastText


class FastSam3Video:
    def __init__(self, vm):
        self.vm = vm
        det = vm.detector
        self.fd = FastSam3(det, text_trim=False)   # trunk / neck / sam2 neck / encoder / heads
        self._patch_detector(det)
        self._patch_tracker(vm.tracker)

    def _patch_tracker(self, trk):
        enc = trk.transformer.encoder
        fma = FastMemAttn(enc)
        self.fma = fma

        def forward(src, prompt, src_mask=None, prompt_mask=None, src_key_padding_mask=None, prompt_key_padding_mask=None, src_pos=None,
                    prompt_pos=None, feat_sizes=None, num_obj_ptr_tokens=0):
            with torch.autocast("cuda", enabled=False):
                mem = fma(src[0] if isinstance(src, list) else src, prompt, prompt_pos, num_obj_ptr_tokens,
                          src_pos[0] if isinstance(src_pos, list) else src_pos)
            return {"memory": mem, "pos_embed": src_pos[0] if isinstance(src_pos, list) else src_pos, "padding_mask": None}
        enc.forward = forward
        # memory encoder (mask downsampler + fuser on 1008 x 1008 masks, once per object and frame): fp16 weights / activations
        mb = trk.maskmem_backbone
        mb.half()
        md = mb.mask_downsampler

        def md_fwd(x):
            if md.interpol_size is not None and md.interpol_size != list(x.shape[-2:]):
                x = F.interpolate(x.float(), size=md.interpol_size, align_corners=False, mode="bilinear", antialias=True).half()
            return md.encoder(x)
        md.forward = md_fwd
        mb_forward = mb.forward

        def mb_fwd(pix_feat, masks, skip_mask_sigmoid=False):
            with torch.autocast("cuda", enabled=False):
                return mb_forward(pix_feat.half(), masks.half(), skip_mask_sigmoid=skip_mask_sigmoid)
        mb.forward = mb_fwd

    def _patch_detector(self, det):
        ft = FastText(det.backbone.language_backbone)

        def forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                outs = [ft(c, device) for c in captions]
            T = max(o["language_features"].shape[0] for o in outs)
            feats, masks, embs = [], [], []
            for o in outs:
                t = o["language_features"].shape[0]
                pad = T - t
                feats.append(F.pad(o["language_features"], (0, 0, 0, 0, 0, pad)))
                embs.append(F.pad(o["language_embeds"], (0, 0, 0, 0, 0, pad)))
                masks.append(F.pad(o["language_mask"], (0, pad), value=True))
            return {"language_features": torch.cat(feats, 1), "language_mask": torch.cat(masks, 0),
                    "language_embeds": torch.cat(embs, 1)}
        det.backbone.forward_text = forward_text

        fg = det.forward_grounding

        def forward_grounding(*a, **k):
            with torch.autocast("cuda", dtype=torch.float16):
                return fg(*a, **k)
        det.forward_grounding = forward_grounding
