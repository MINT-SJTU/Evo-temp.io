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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


@dataclass(frozen=True)
class EpisodeValueInfo:
    length: int
    task: str
    is_success: bool


def _coerce_success_label(value: Any, default_success: str = "failure") -> bool:
    if value is None:
        value = default_success
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"success", "s", "true", "1", "yes"}:
            return True
        if normalized in {"failure", "fail", "f", "false", "0", "no"}:
            return False
    raise ValueError(f"Unsupported success label: {value!r}")


def _coerce_task_name(task_entry: Any) -> str:
    if isinstance(task_entry, str):
        return task_entry
    if isinstance(task_entry, (list, tuple)):
        if len(task_entry) == 0:
            return "__default_task__"
        return str(task_entry[0])
    return "__default_task__"


def build_episode_value_info(
    episodes_table: Any,
    success_field: str = "episode_success",
    default_success: str = "failure",
) -> tuple[dict[int, EpisodeValueInfo], dict[str, int]]:
    column_names = set(getattr(episodes_table, "column_names", []))

    episode_indices = [int(x) for x in episodes_table["episode_index"]]
    lengths = [int(x) for x in episodes_table["length"]]
    tasks = episodes_table["tasks"] if "tasks" in column_names else ["__default_task__"] * len(episode_indices)
    raw_success = (
        episodes_table[success_field] if success_field in column_names else [default_success] * len(episode_indices)
    )

    episode_info: dict[int, EpisodeValueInfo] = {}
    task_lmax: dict[str, int] = {}
    for ep_idx, length, task_entry, success in zip(episode_indices, lengths, tasks, raw_success, strict=True):
        task_name = _coerce_task_name(task_entry)
        is_success = _coerce_success_label(success, default_success=default_success)
        episode_info[ep_idx] = EpisodeValueInfo(length=length, task=task_name, is_success=is_success)
        task_lmax[task_name] = max(task_lmax.get(task_name, 0), length)

    return episode_info, task_lmax


def build_normalized_returns(
    frame_indices: Tensor,
    episode_indices: Tensor,
    episode_info: dict[int, EpisodeValueInfo],
    task_lmax: dict[str, int],
    c_fail: float,
    clamp_to_range: bool = True,
) -> Tensor:
    frame_indices = frame_indices.view(-1).to(dtype=torch.long)
    episode_indices = episode_indices.view(-1).to(dtype=torch.long)
    values = torch.empty_like(frame_indices, dtype=torch.float32)

    for i in range(frame_indices.shape[0]):
        ep_idx = int(episode_indices[i].item())
        t = int(frame_indices[i].item())
        if ep_idx not in episode_info:
            raise KeyError(f"episode_index {ep_idx} is missing from meta/episodes")

        info = episode_info[ep_idx]
        remaining_steps = max(info.length - t - 1, 0)
        if info.is_success:
            g_t = -float(remaining_steps)
        else:
            g_t = -float(remaining_steps) - float(c_fail)

        denom = float(task_lmax.get(info.task, info.length) + c_fail)
        if denom <= 0:
            raise ValueError(f"Normalization denominator must be positive, got {denom} for task {info.task!r}")
        g_norm = g_t / denom
        if clamp_to_range:
            g_norm = max(-1.0, min(0.0, g_norm))
        values[i] = g_norm

    return values


def project_values_to_bins(values: Tensor, num_bins: int) -> tuple[Tensor, Tensor]:
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    values = values.to(dtype=torch.float32).view(-1)
    bins = torch.linspace(-1.0, 0.0, steps=num_bins, device=values.device, dtype=values.dtype)
    clamped = values.clamp(min=float(bins[0]), max=float(bins[-1]))
    upper_idx = torch.searchsorted(bins, clamped, right=True).clamp(min=1, max=num_bins - 1)
    lower_idx = upper_idx - 1

    lower_values = bins[lower_idx]
    upper_values = bins[upper_idx]
    denom = (upper_values - lower_values).clamp_min(1e-8)
    alpha = (clamped - lower_values) / denom

    targets = torch.zeros((values.shape[0], num_bins), device=values.device, dtype=values.dtype)
    targets.scatter_add_(1, lower_idx.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
    targets.scatter_add_(1, upper_idx.unsqueeze(1), alpha.unsqueeze(1))
    return targets, bins


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"logits and targets must be rank-2, got {logits.shape} and {targets.shape}")
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {logits.shape} must match targets shape {targets.shape}")
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def expected_value_from_logits(logits: Tensor, bins: Tensor) -> Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * bins.view(1, -1), dim=-1)


class MLPValueModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        hidden_dim: int,
        num_bins: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        task_embed_dim = max(16, hidden_dim // 4)
        self.task_embedding = nn.Embedding(max(1, num_tasks), task_embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + task_embed_dim),
            nn.Linear(state_dim + task_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state.view(state.shape[0], -1)
        state = state.to(dtype=torch.float32)

        task_index = batch["task_index"].view(-1).to(dtype=torch.long)
        task_emb = self.task_embedding(task_index)
        features = torch.cat([state, task_emb], dim=-1)
        return self.net(features)


class PI05ValueModel(nn.Module):
    def __init__(self, pi05_policy: nn.Module, num_bins: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = pi05_policy
        hidden_size = self.backbone.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
        mid_size = max(128, hidden_size // 2)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mid_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, num_bins),
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        images, img_masks = self.backbone._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.backbone.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_4d = self.backbone.model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), _ = self.backbone.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        if prefix_out is None:
            raise RuntimeError("PI05 backbone returned no prefix output for value head input.")

        pad_mask = prefix_pad_masks.to(dtype=prefix_out.dtype).unsqueeze(-1)
        pooled = (prefix_out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp_min(1e-6)
        return self.value_head(pooled.to(dtype=torch.float32))


def configure_pi05_trainable_params(
    model: PI05ValueModel,
    freeze_backbone: bool = True,
    finetune_last_n_layers: int = 0,
) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone

    if freeze_backbone and finetune_last_n_layers > 0:
        layers = model.backbone.model.paligemma_with_expert.paligemma.language_model.layers
        for layer in layers[-finetune_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.backbone.model.paligemma_with_expert.paligemma.language_model.norm.parameters():
            param.requires_grad = True

    for param in model.value_head.parameters():
        param.requires_grad = True
