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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EpisodeTargetInfo:
    episode_index: int
    task_index: int
    length: int
    success: bool


def build_bin_centers(
    num_bins: int,
    bin_min: float,
    bin_max: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.linspace(bin_min, bin_max, num_bins, dtype=torch.float32, device=device)


def project_values_to_bins(values: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if bin_centers.ndim != 1:
        raise ValueError(f"'bin_centers' must be rank-1, got shape={tuple(bin_centers.shape)}.")
    if bin_centers.shape[0] < 2:
        raise ValueError("At least 2 bins are required.")

    values = values.clamp(min=bin_centers[0], max=bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = torch.floor(scaled).long()
    high = torch.clamp(low + 1, max=bin_centers.shape[0] - 1)
    high_weight = (scaled - low.float()).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    target = torch.zeros(values.shape[0], bin_centers.shape[0], device=values.device, dtype=torch.float32)
    target.scatter_add_(1, low.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != soft_targets.shape:
        raise ValueError(
            "'logits' and 'soft_targets' must have same shape, "
            f"got {tuple(logits.shape)} vs {tuple(soft_targets.shape)}."
        )
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def expected_value_from_logits(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def compute_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: dict[int, EpisodeTargetInfo],
    task_max_lengths: dict[int, int],
    c_fail_coef: float,
    *,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    if episode_indices.shape != frame_indices.shape:
        raise ValueError("episode_indices and frame_indices must have the same shape.")
    if c_fail_coef < 0:
        raise ValueError("'c_fail_coef' must be non-negative.")

    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)
    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        if ep_idx not in episode_info:
            raise KeyError(f"Missing episode metadata for episode_index={ep_idx}.")
        ep = episode_info[ep_idx]
        task_max = task_max_lengths.get(ep.task_index)
        if task_max is None:
            raise KeyError(f"Missing task max length for task_index={ep.task_index}.")
        if task_max <= 0:
            raise ValueError(f"Invalid task max length {task_max} for task_index={ep.task_index}.")

        remaining_steps = ep.length - int(frame_indices[i]) - 1
        c_fail = float(task_max) * c_fail_coef
        g = -float(remaining_steps)
        if not ep.success:
            g -= c_fail

        denom = float(task_max) + c_fail
        g_norm = g / denom
        targets[i] = np.clip(g_norm, clip_min, clip_max)

    return targets


def compute_normalized_rewards_from_targets(
    targets: np.ndarray, episode_indices: np.ndarray, frame_indices: np.ndarray
) -> np.ndarray:
    rewards = np.zeros_like(targets, dtype=np.float32)
    unique_episodes = np.unique(episode_indices)
    for ep_idx in unique_episodes:
        mask = episode_indices == ep_idx
        local_indices = np.nonzero(mask)[0]
        ordered_local = local_indices[np.argsort(frame_indices[local_indices], kind="stable")]
        ordered_targets = targets[ordered_local]

        ep_rewards = np.empty_like(ordered_targets)
        if ordered_targets.shape[0] > 1:
            ep_rewards[:-1] = ordered_targets[:-1] - ordered_targets[1:]
        ep_rewards[-1] = ordered_targets[-1]
        rewards[ordered_local] = ep_rewards
    return rewards


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")

    advantages = np.zeros_like(values, dtype=np.float32)
    unique_episodes = np.unique(episode_indices)
    for ep_idx in unique_episodes:
        mask = episode_indices == ep_idx
        local_indices = np.nonzero(mask)[0]
        ordered_local = local_indices[np.argsort(frame_indices[local_indices], kind="stable")]
        ep_rewards = rewards[ordered_local]
        ep_values = values[ordered_local]
        ep_len = ep_rewards.shape[0]

        reward_prefix = np.zeros(ep_len + 1, dtype=np.float32)
        reward_prefix[1:] = np.cumsum(ep_rewards, dtype=np.float32)

        ep_adv = np.empty(ep_len, dtype=np.float32)
        for t in range(ep_len):
            end = min(t + n_step, ep_len)
            reward_sum = reward_prefix[end] - reward_prefix[t]
            bootstrap = float(ep_values[t + n_step]) if (t + n_step) < ep_len else 0.0
            ep_adv[t] = reward_sum + bootstrap - float(ep_values[t])

        advantages[ordered_local] = ep_adv

    return advantages


def compute_task_thresholds(
    task_indices: np.ndarray, advantages: np.ndarray, positive_ratio: float
) -> dict[int, float]:
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")

    thresholds: dict[int, float] = {}
    unique_tasks = np.unique(task_indices)
    for task_idx in unique_tasks:
        task_mask = task_indices == task_idx
        task_adv = advantages[task_mask]
        if task_adv.size == 0:
            continue

        if positive_ratio <= 0.0:
            thresholds[int(task_idx)] = float(np.max(task_adv))
        elif positive_ratio >= 1.0:
            thresholds[int(task_idx)] = float(np.min(task_adv) - 1e-6)
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, 1.0 - positive_ratio))

    return thresholds


def binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    interventions: np.ndarray | None = None,
    *,
    force_intervention_positive: bool = False,
) -> np.ndarray:
    indicator = np.zeros_like(task_indices, dtype=np.int64)
    for i in range(task_indices.shape[0]):
        task_idx = int(task_indices[i])
        eps = thresholds.get(task_idx)
        if eps is None:
            raise KeyError(f"Missing task threshold for task_index={task_idx}.")
        indicator[i] = int(float(advantages[i]) > eps)

    if force_intervention_positive:
        if interventions is None:
            raise ValueError(
                "'interventions' must be provided when 'force_intervention_positive' is enabled."
            )
        indicator = np.where(interventions > 0, 1, indicator).astype(np.int64)

    return indicator
