"""Checkpoint-free tests: text trimming, start_session, staging buffer, CPU processor."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sam3
from sam3 import fast
from sam3.fast_text import FastText
from sam3.model import sam3_image_processor
from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer

ROOT = Path(sam3.__file__).resolve().parent.parent


@torch.inference_mode()
@pytest.mark.parametrize("prompt", ["a sign !$ on a wall", "a car"])
def test_text_token_zero_retains_eot_and_mask(prompt):
    tok = SimpleTokenizer(ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    torch.manual_seed(7)
    vl = VETextEncoder(
        d_model=4, tokenizer=tok, width=8, heads=2, layers=1, use_act_checkpoint=False
    ).eval()
    vl.encoder.positional_embedding.normal_(std=0.01)
    vl.encoder.text_projection.normal_(std=0.01)
    mask, feat, emb = vl([prompt], device="cpu")
    tokens = tok([prompt], context_length=32)
    n = int(tokens[0].nonzero()[-1, 0]) + 1
    out = FastText(vl)(prompt, "cpu")
    assert out["language_features"].shape[0] == n
    assert torch.equal(out["language_mask"], mask[:, :n])
    torch.testing.assert_close(out["language_features"], feat[:n], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["language_embeds"], emb[:n])


def test_start_session_multiplex_default():
    cls = Sam3BasePredictor
    p = cls()
    p.async_loading_frames = False

    def init_state(
        resource_path, offload_video_to_cpu=False, async_loading_frames=False
    ):
        return {
            "resource": resource_path,
            "offload": offload_video_to_cpu,
            "async": async_loading_frames,
        }

    p.model = SimpleNamespace(init_state=init_state)
    assert p.start_session("clip", session_id="review") == {"session_id": "review"}
    assert p._all_inference_states["review"]["state"] == {
        "resource": "clip",
        "offload": False,
        "async": False,
    }


def test_explicit_state_offload_is_forwarded():
    p = Sam3BasePredictor()
    p.model = SimpleNamespace(init_state=lambda **kw: kw)
    p.start_session("clip", session_id="review", offload_state_to_cpu=True)
    assert p._all_inference_states["review"]["state"]["offload_state_to_cpu"] is True


def test_pinned_staging_cache_is_bounded(monkeypatch):
    import numpy as np

    class Event:
        def record(self, stream=None):
            assert stream == "upload-stream"

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(torch.cuda, "Event", Event)

    def current_stream(device):
        assert device == "cpu"
        return "upload-stream"

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    obj = SimpleNamespace(pin={})
    preprocess = fast.FastImageModel.preprocess
    for h in (12, 13, 14, 14):
        x, H, W = preprocess(obj, np.zeros((h, 15, 3), dtype=np.uint8), 14, "cpu")
        assert (H, W) == (h, 15)
        assert torch.equal(x, torch.full_like(x, -1))
        assert len(obj.pin) == 1


@pytest.mark.parametrize("kind", ["pil", "numpy", "tensor"])
def test_cpu_processor_keeps_upstream_path(monkeypatch, kind):
    import numpy as np
    from PIL import Image

    mod = sam3_image_processor

    def no_cuda_patch(*a, **kw):
        raise AssertionError(
            "CPU processor must not convert the model to the CUDA path"
        )

    monkeypatch.setattr(mod.FastImageModel, "of", no_cuda_patch)
    images = []
    prompts = []

    def image_forward(x):
        assert x.dtype == torch.float32 and x.device.type == "cpu"
        images.append(x)
        return {}

    def text_forward(texts, device):
        prompts.extend(texts)
        return {"language_features": torch.zeros(1, 1, 4)}

    def grounding(**kw):
        return {
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
            "pred_logits": torch.ones(1, 1, 1) * 9,
            "presence_logit_dec": torch.ones(1, 1) * 9,
            "pred_masks": torch.ones(1, 1, 2, 2),
        }

    model = SimpleNamespace(
        backbone=SimpleNamespace(
            forward_image=image_forward, forward_text=text_forward
        ),
        inst_interactive_predictor=None,
        _get_dummy_prompt=lambda: None,
        forward_grounding=grounding,
    )
    p = mod.Sam3Processor(model, resolution=14, device="cpu")
    pixels = np.zeros((12, 15, 3), dtype=np.uint8)
    im = (
        Image.fromarray(pixels)
        if kind == "pil"
        else pixels
        if kind == "numpy"
        else torch.zeros(3, 12, 15)
    )
    state = p.set_image(im)
    out = p.set_text_prompt("wheel", state)
    assert out["masks"].shape == (1, 1, 12, 15)
    assert prompts == ["wheel"]
    assert images[0].shape == (1, 3, 14, 14)
    p.set_image_batch([Image.fromarray(pixels), Image.fromarray(pixels)])
    assert images[-1].shape == (2, 3, 14, 14)


def test_float_numpy_preprocessing_matches_float_tensor():
    import numpy as np

    from sam3.fast import FastImageModel

    image = np.linspace(0, 1, 12 * 16 * 3, dtype=np.float32).reshape(12, 16, 3)[:, ::2]
    obj = SimpleNamespace(pin={})
    out, h, w = FastImageModel.preprocess(obj, image, 14, "cpu")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    expected, _, _ = FastImageModel.preprocess(obj, tensor, 14, "cpu")
    assert (h, w) == (12, 8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
