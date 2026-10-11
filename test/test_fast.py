"""Tests for the fast inference path (sam3/fast*.py). Needs a CUDA GPU.

The trunk and kernel tests use random weights / inputs. Set SAM3_CHECKPOINT=/path/to/sam3.pt to also compare the full image model with
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


@torch.inference_mode()
def test_kernel_vit_matches_portable(vit):
    from sam3.fast_vit import FastViT

    x = torch.randn(1, 3, 532, 812, device="cuda").half()
    ref = FastViT(copy.deepcopy(vit))(x)
    out = FastViT(copy.deepcopy(vit), attn="hh", kernels=True)(x)
    assert rel_err(out, ref) < 5e-3


@pytest.mark.parametrize("variant", ["hh", "fh", "ff"])
def test_flash_attention_matches_sdpa(variant):
    import torch.nn.functional as F

    from sam3.fastbb_vendored.attention import attention

    q, k, v = (torch.randn(2, 1000, 16, 64, device="cuda").half() for _ in range(3))
    ref = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
    out = attention(q, k, v, variant=variant)
    assert rel_err(out, ref.transpose(1, 2)) < 2e-3


@pytest.mark.parametrize("variant", ["ff", "hf"])
def test_flash_k256_matches_reference(variant):
    from sam3.fast_cuda import flash_k256v64

    q = torch.randn(2, 300, 256, device="cuda").half()
    k = torch.randn(2, 777, 256, device="cuda").half()
    v = torch.randn(2, 777, 64, device="cuda").half()
    p = torch.softmax(q.float() @ k.float().transpose(1, 2) / 16, -1)
    assert rel_err(flash_k256v64(q, k, v, variant=variant), p @ v.float()) < 2e-3


def test_rope_rows_matches_plain():
    from sam3.fast_kernels import rope_rows

    N, nrope, T, C = (
        64,
        128,
        140,
        256,
    )  # two rotated copies of the grid + 12 unrotated tokens
    ang = torch.rand(N, C // 2, device="cuda") * 6.3
    cos, sin = ang.cos(), ang.sin()
    x = torch.randn(3 * T, C, device="cuda").half()
    tab = torch.randn(T, C, device="cuda").half()
    y = (x.view(3, T, C) + tab).float()
    a, b = y[..., 0::2].clone(), y[..., 1::2].clone()
    c, s = cos.repeat(nrope // N, 1), sin.repeat(nrope // N, 1)
    y[:, :nrope, 0::2] = a[:, :nrope] * c - b[:, :nrope] * s
    y[:, :nrope, 1::2] = a[:, :nrope] * s + b[:, :nrope] * c
    rope_rows(x, cos, sin, T, N, nrope, tab=tab)
    assert rel_err(x.view(3, T, C), y) < 1e-3


def _iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


@pytest.mark.skipif(not CKPT, reason="set SAM3_CHECKPOINT")
@pytest.mark.parametrize("kernels", [False, True])
@pytest.mark.parametrize(
    "image,prompt", [("truck.jpg", "wheel"), ("groceries.jpg", "bag")]
)
def test_fast_sam3_matches_upstream(image, prompt, kernels):
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
    out = FastSam3(model, kernels=kernels)(np.asarray(im), prompt)
    assert len(ref_masks) > 0
    assert (
        abs(len(out["masks"]) - len(ref_masks)) <= 1
    )  # a score at the 0.5 threshold may flip
    assert _iou(out["masks"].any(0), ref_masks.any(0)) > 0.97
