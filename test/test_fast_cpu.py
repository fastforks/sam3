"""Checkpoint-free tests: text trimming, start_session, staging buffer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sam3
from sam3 import fast
from sam3.fast_text import FastText
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
    obj = SimpleNamespace(pin={}, dev="cpu", res=14)
    preprocess = fast.FastSam3._pre
    for h in (12, 13, 14, 14):
        x, H, W = preprocess(obj, np.zeros((h, 15, 3), dtype=np.uint8))
        assert (H, W) == (h, 15)
        assert torch.equal(x, torch.full_like(x, -1))
        assert len(obj.pin) == 1
