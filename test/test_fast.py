"""Tests for the fast inference path (sam3/fast*.py). Needs a CUDA GPU.

The trunk tests use random weights. Set SAM3_CHECKPOINT=/path/to/sam3.pt to also compare the full image model with
the upstream Sam3Processor flow (bf16 autocast) on the bundled example images.
"""

import copy
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)
CKPT = os.environ.get("SAM3_CHECKPOINT")
IMAGES = os.path.join(os.path.dirname(__file__), "..", "assets", "images")


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


@pytest.fixture(scope="module")
def vit():
    from sam3.model_builder import _create_vit_backbone

    torch.manual_seed(0)
    return _create_vit_backbone().cuda().eval()


@torch.inference_mode()
def test_fast_vit_matches_trunk(vit):
    from sam3.fast_vit import FastViT

    x = torch.randn(1, 3, 1008, 1008, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = vit(x)[-1]
    out = FastViT(copy.deepcopy(vit))(x.half())
    assert out.shape == ref.shape
    # bf16 reference: 1e-3 input noise alone moves it by ~0.8 %
    assert rel_err(out, ref) < 0.02


@torch.inference_mode()
def test_explicit_attention_matches_sdpa(vit):
    from sam3.fast_vit import FastViT

    x = torch.randn(
        1, 3, 532, 812, device="cuda"
    ).half()  # 38 x 58 tokens: padded windows
    ref = FastViT(copy.deepcopy(vit))(x)
    fv = FastViT(copy.deepcopy(vit), attn="explicit")
    fv.fold_q_scale()
    assert rel_err(fv(x), ref) < 5e-3


def _iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def _upstream_masks(model, im, prompt):
    """the upstream Sam3Processor flow (set_image + set_text_prompt), run under bf16 autocast as in the notebooks"""
    from torchvision.transforms import v2

    from sam3.model.data_misc import FindStage, interpolate

    tf = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    zero = torch.zeros(1, device="cuda", dtype=torch.long)
    find = FindStage(img_ids=zero, text_ids=zero, input_boxes=None, input_boxes_mask=None,
                     input_boxes_label=None, input_points=None, input_points_mask=None)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        bo = model.backbone.forward_image(tf(v2.functional.to_image(im).cuda()).unsqueeze(0))
        bo.update(model.backbone.forward_text([prompt], device="cuda"))
        o = model.forward_grounding(backbone_out=bo, find_input=find, geometric_prompt=model._get_dummy_prompt(),
                                    find_target=None)
        probs = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        masks = interpolate(o["pred_masks"][probs > 0.5].unsqueeze(1), (im.height, im.width), mode="bilinear",
                            align_corners=False)
    return (masks[:, 0] > 0).cpu().numpy()


@pytest.mark.skipif(not CKPT, reason="set SAM3_CHECKPOINT")
@pytest.mark.parametrize(
    "image,prompt", [("truck.jpg", "wheel"), ("groceries.jpg", "bag")]
)
def test_processor_matches_upstream(image, prompt):
    from PIL import Image

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    im = Image.open(os.path.join(IMAGES, image)).convert("RGB")
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False)
    ref_masks = _upstream_masks(model, im, prompt)
    proc = Sam3Processor(model)  # converts the model to the fast path
    out = proc.set_text_prompt(state=proc.set_image(im), prompt=prompt)
    masks = out["masks"][:, 0].cpu().numpy()
    assert out["masks"].dtype == torch.bool and out["masks_logits"].dtype == torch.float32
    assert len(out["boxes"]) == len(out["scores"]) == len(masks)
    assert len(ref_masks) > 0
    assert abs(len(masks) - len(ref_masks)) <= 1  # a score at the 0.5 threshold may flip
    assert _iou(masks.any(0), ref_masks.any(0)) > 0.97
