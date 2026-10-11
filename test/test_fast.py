"""Tests for the fast inference path (sam3/fast*.py). Needs a CUDA GPU.

The trunk tests use random weights. Set SAM3_CHECKPOINT=/path/to/sam3.pt to also compare the full image model with
the upstream Sam3Processor flow (bf16 autocast) on the bundled example images.
"""

import copy
import os

import numpy as np
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


@pytest.mark.skipif(not CKPT, reason="set SAM3_CHECKPOINT")
@pytest.mark.parametrize(
    "image,prompt", [("truck.jpg", "wheel"), ("groceries.jpg", "bag")]
)
def test_fast_sam3_matches_upstream(image, prompt):
    from PIL import Image

    from sam3.fast import FastSam3
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    im = Image.open(os.path.join(IMAGES, image)).convert("RGB")
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proc = Sam3Processor(model)
        ref = proc.set_text_prompt(state=proc.set_image(im), prompt=prompt)
    ref_masks = ref["masks"][:, 0].cpu().numpy()
    out = FastSam3(model)(np.asarray(im), prompt)
    assert len(ref_masks) > 0
    assert (
        abs(len(out["masks"]) - len(ref_masks)) <= 1
    )  # a score at the 0.5 threshold may flip
    assert _iou(out["masks"].any(0), ref_masks.any(0)) > 0.97
