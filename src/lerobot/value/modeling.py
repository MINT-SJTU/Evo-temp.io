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

import torch
from torch import nn

from lerobot.value.configuration import MLPValueConfig, ValueModelConfig


class MLPValueModel(nn.Module):
    def __init__(self, cfg: MLPValueConfig, state_dim: int, num_tasks: int):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"Expected positive state_dim, got {state_dim}.")
        if num_tasks <= 0:
            raise ValueError(f"Expected positive num_tasks, got {num_tasks}.")

        self.cfg = cfg
        self.state_dim = state_dim
        self.num_tasks = num_tasks

        self.task_embedding = nn.Embedding(num_tasks, cfg.task_embedding_dim)

        layers: list[nn.Module] = []
        in_dim = state_dim + cfg.task_embedding_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_dim, cfg.num_bins)

    def forward(self, state: torch.Tensor, task_index: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"'state' must have shape [B, D], got {tuple(state.shape)}")
        if task_index.ndim != 1:
            raise ValueError(f"'task_index' must have shape [B], got {tuple(task_index.shape)}")
        if state.shape[0] != task_index.shape[0]:
            raise ValueError(
                f"Batch mismatch between state ({state.shape[0]}) and task_index ({task_index.shape[0]})."
            )

        task_embed = self.task_embedding(task_index.long())
        x = torch.cat([state, task_embed], dim=-1)
        x = self.backbone(x)
        return self.output_head(x)


def make_value_model(cfg: ValueModelConfig, state_dim: int, num_tasks: int) -> nn.Module:
    if isinstance(cfg, MLPValueConfig):
        return MLPValueModel(cfg=cfg, state_dim=state_dim, num_tasks=num_tasks)
    raise ValueError(f"Unsupported value model type '{cfg.type}'.")
