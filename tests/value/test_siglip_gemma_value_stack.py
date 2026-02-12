#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

import lerobot.value.modeling as value_modeling
import lerobot.value.preprocess as value_preprocess
from lerobot.value.configuration import SiglipGemmaValueConfig
from lerobot.value.modeling import make_value_model
from lerobot.value.preprocess import SiglipGemmaValuePreprocessor


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, prompts, max_length, truncation, padding, return_tensors):
        del truncation, padding, return_tensors
        bsize = len(prompts)
        input_ids = torch.zeros((bsize, max_length), dtype=torch.long)
        attention_mask = torch.zeros((bsize, max_length), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            token_count = min(max_length, max(1, len(prompt.split())))
            input_ids[i, :token_count] = torch.arange(1, token_count + 1, dtype=torch.long)
            attention_mask[i, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyImageProcessor:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, images, return_tensors, do_rescale, size=None):
        del return_tensors, do_rescale
        if size is None:
            height, width = 32, 32
        else:
            height, width = int(size["height"]), int(size["width"])

        out = []
        for image in images:
            img = image
            if img.ndim != 3:
                raise ValueError(f"Expected rank-3 image tensor, got {tuple(img.shape)}.")
            if img.shape[0] not in {1, 3}:
                img = img.permute(2, 0, 1)
            img = img.to(dtype=torch.float32)
            img = F.interpolate(img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)
            out.append(img.squeeze(0))
        return {"pixel_values": torch.stack(out, dim=0)}


class _DummyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(3, 16)
        self.config = SimpleNamespace(hidden_size=16)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        return self.image_proj(pooled)


class _DummyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)
        self.embed = nn.Embedding(2048, 32)
        self.proj = nn.Linear(32, 32)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        hidden = self.proj(self.embed(input_ids))
        return SimpleNamespace(last_hidden_state=hidden)


class _DummyAutoModel:
    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        del kwargs
        if "siglip" in repo_id.lower():
            return _DummyVisionModel()
        return _DummyLanguageModel()


@pytest.fixture
def hf_stubs(monkeypatch):
    monkeypatch.setattr(value_preprocess, "AutoTokenizer", _DummyTokenizer)
    monkeypatch.setattr(value_preprocess, "AutoImageProcessor", _DummyImageProcessor)
    monkeypatch.setattr(value_modeling, "AutoModel", _DummyAutoModel)


def test_siglip_gemma_preprocessor_pads_missing_cameras_and_tokenizes(hf_stubs):
    del hf_stubs
    cfg = SiglipGemmaValueConfig(
        camera_features=["observation.images.front", "observation.images.wrist"],
        tokenizer_max_length=16,
    )
    preprocessor = SiglipGemmaValuePreprocessor(
        cfg=cfg,
        dataset_camera_features=["observation.images.front", "observation.images.wrist"],
    )

    raw_batch = {
        "task": ["pick bottle", "place bottle"],
        "observation.images.front": torch.rand(2, 3, 48, 40),
    }
    processed = preprocessor(raw_batch, device=torch.device("cpu"))

    assert processed["input_ids"].shape == (2, 16)
    assert processed["attention_mask"].dtype == torch.bool
    assert processed["images"].shape == (2, 2, 3, 32, 32)
    assert torch.equal(processed["image_attention_mask"][:, 0], torch.ones(2, dtype=torch.bool))
    assert torch.equal(processed["image_attention_mask"][:, 1], torch.zeros(2, dtype=torch.bool))


def test_siglip_gemma_value_model_forward(hf_stubs):
    del hf_stubs
    cfg = SiglipGemmaValueConfig(
        camera_features=["observation.images.front"],
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        state_proj_dim=16,
        num_bins=17,
    )
    model = make_value_model(cfg=cfg, state_dim=6, num_tasks=4)

    outputs = model(
        input_ids=torch.randint(0, 20, (3, 12), dtype=torch.long),
        attention_mask=torch.ones(3, 12, dtype=torch.bool),
        images=torch.rand(3, 1, 3, 32, 32),
        image_attention_mask=torch.ones(3, 1, dtype=torch.bool),
    )
    assert outputs.shape == (3, 17)


def test_siglip_gemma_value_model_requires_valid_camera_mask(hf_stubs):
    del hf_stubs
    cfg = SiglipGemmaValueConfig(
        camera_features=["observation.images.front"],
        fusion_hidden_dim=32,
        fusion_num_heads=8,
        state_proj_dim=16,
    )
    model = make_value_model(cfg=cfg, state_dim=4, num_tasks=2)

    with pytest.raises(ValueError, match="at least one valid camera"):
        model(
            input_ids=torch.randint(0, 10, (2, 8), dtype=torch.long),
            attention_mask=torch.ones(2, 8, dtype=torch.bool),
            images=torch.rand(2, 1, 3, 16, 16),
            image_attention_mask=torch.zeros(2, 1, dtype=torch.bool),
        )
