#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available
from lerobot.value.configuration import SiglipGemmaValueConfig, ValueModelConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
else:
    AutoConfig = None
    AutoModel = None
    AutoModelForCausalLM = None


def _resolve_load_dtype(dtype_name: str) -> torch.dtype:
    requested_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    if requested_dtype == torch.bfloat16 and not torch.cuda.is_available():
        logging.warning("value.dtype=bfloat16 requested but CUDA unavailable. Falling back to float32.")
        return torch.float32
    return requested_dtype


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = True


def _extract_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Cannot infer hidden size for model type {type(model)}: missing `.config`.")

    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError(f"Cannot infer hidden size for model config type {type(config)}.")


def _extract_vision_feature_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Cannot infer vision feature size for model type {type(model)}: missing `.config`.")

    if hasattr(config, "projection_dim"):
        return int(config.projection_dim)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "projection_dim"):
        return int(config.vision_config.projection_dim)
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
        return int(config.vision_config.hidden_size)
    raise ValueError(f"Cannot infer vision feature size for model config type {type(config)}.")


def _validate_loading_info(repo_id: str, model_label: str, loading_info: dict[str, list] | None) -> None:
    if loading_info is None:
        return
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    if not missing and not unexpected and not mismatched:
        return
    raise RuntimeError(
        f"Pretrained weights for {model_label} from '{repo_id}' did not load cleanly: "
        f"missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}. "
        "This usually indicates a model class/checkpoint mismatch."
    )


def _load_language_model(
    repo_id: str,
    revision: str | None,
    dtype: torch.dtype,
) -> nn.Module:
    if AutoConfig is None or AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError("transformers is not installed. Install with `pip install 'lerobot[pi0]'`.")

    model_config = AutoConfig.from_pretrained(repo_id, revision=revision)
    architectures = getattr(model_config, "architectures", None) or []
    prefer_causal_lm = any(
        isinstance(arch, str) and arch.endswith("ForCausalLM") for arch in architectures
    )

    if prefer_causal_lm:
        lm_with_head, loading_info = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        _validate_loading_info(repo_id, "language_model(causal_lm)", loading_info)
        if not hasattr(lm_with_head, "model"):
            raise RuntimeError(
                f"AutoModelForCausalLM loaded from '{repo_id}' does not expose `.model` text backbone."
            )
        return lm_with_head.model

    language_model, loading_info = AutoModel.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        output_loading_info=True,
    )
    _validate_loading_info(repo_id, "language_model(auto_model)", loading_info)
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"AutoModel loaded from '{repo_id}' returned unexpected type: {type(language_model)}."
        )
    return language_model


class SiglipGemmaValueModel(nn.Module):
    def __init__(self, cfg: SiglipGemmaValueConfig, state_dim: int, num_tasks: int):  # noqa: ARG002
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is not installed. Install with `pip install 'lerobot[pi0]'`.")

        self.cfg = cfg
        self.state_dim = state_dim
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            torch_dtype=self.model_dtype,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
        )
        vision_feature_size = _extract_vision_feature_size(self.vision_encoder)
        language_hidden_size = _extract_hidden_size(self.language_model)

        self.image_projector = nn.Sequential(
            nn.Linear(vision_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_hidden_dim * 2)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_bins),
        )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)

        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)

    def _encode_images(self, flat_images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)

        vision_outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError("Unsupported vision encoder output. Expected pooler_output or last_hidden_state.")

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Language model output does not contain `last_hidden_state`.")

        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"'input_ids' must have shape [B, T], got {tuple(input_ids.shape)}.")
        if attention_mask.ndim != 2:
            raise ValueError(f"'attention_mask' must have shape [B, T], got {tuple(attention_mask.shape)}.")
        if images.ndim != 5:
            raise ValueError(f"'images' must have shape [B, N, C, H, W], got {tuple(images.shape)}.")
        if image_attention_mask.ndim != 2:
            raise ValueError(
                f"'image_attention_mask' must have shape [B, N], got {tuple(image_attention_mask.shape)}."
            )

        bsize = input_ids.shape[0]
        if attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch between input_ids and attention_mask.")
        if images.shape[0] != bsize or image_attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch between language and image inputs.")
        if images.shape[1] == 0:
            raise ValueError("At least one camera is required for SiglipGemmaValueModel.")

        image_attention_mask = image_attention_mask.to(dtype=torch.bool, device=images.device)
        if not torch.all(image_attention_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera input.")
        language_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if not torch.all(language_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token.")

        num_cameras = images.shape[1]
        flat_images = images.reshape(bsize * num_cameras, *images.shape[2:])

        image_context = torch.no_grad() if self.cfg.freeze_vision_encoder else nullcontext()
        with image_context:
            image_features = self._encode_images(flat_images)

        language_context = torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        with language_context:
            language_features = self._encode_language(input_ids=input_ids, attention_mask=language_mask.long())

        feature_dtype = torch.float32
        image_features = image_features.to(dtype=feature_dtype)
        language_features = language_features.to(dtype=feature_dtype)

        image_tokens = self.image_projector(image_features).view(bsize, num_cameras, -1)
        camera_token_mask = image_attention_mask.unsqueeze(-1).to(dtype=image_tokens.dtype)
        image_tokens = image_tokens * camera_token_mask

        camera_denominator = image_attention_mask.sum(dim=1, keepdim=True).to(dtype=image_tokens.dtype).clamp_min(1.0)
        image_pooled = image_tokens.sum(dim=1) / camera_denominator
        language_token = self.language_projector(language_features)

        joint_features = torch.cat([image_pooled, language_token], dim=-1)
        return self.value_head(self.final_norm(joint_features))


def make_value_model(cfg: ValueModelConfig, state_dim: int, num_tasks: int) -> nn.Module:
    if isinstance(cfg, SiglipGemmaValueConfig):
        return SiglipGemmaValueModel(cfg=cfg, state_dim=state_dim, num_tasks=num_tasks)
    raise ValueError(f"Unsupported value model type '{cfg.type}'.")
